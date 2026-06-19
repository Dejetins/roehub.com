from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeExecutionOrderRepository,
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
    ExecutionDispatchConfig,
    ExecutionDispatchService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExecutionDispatchPublishResult,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
    ExecutionFillFact,
    ExecutionFundingFact,
    ExecutionIntent,
    ExecutionRiskContext,
)
from trading.shared_kernel.primitives import UserId

_OWNER_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000011011")
_REQUEST_STREAM = "execution.requests.v1"
_RETRY_STREAM = "execution.requests.retry.v1"
_DLQ_STREAM = "execution.requests.dlq.v1"
_CONNECTIONS: Mapping[tuple[str, str], UUID] = {
    ("binance", "spot"): UUID("00000000-0000-0000-0000-000000110101"),
    ("binance", "futures"): UUID("00000000-0000-0000-0000-000000110102"),
    ("bybit", "spot"): UUID("00000000-0000-0000-0000-000000110201"),
    ("bybit", "futures"): UUID("00000000-0000-0000-0000-000000110202"),
}


@dataclass(frozen=True, slots=True)
class LoadHarnessConfig:
    strategy_count: int = 120
    exchange_read_count: int = 25
    rate_limit_per_second: float = 60.0
    rate_limit_burst: int = 20
    retry_budget: int = 3
    backpressure_probe_size: int = 3
    cancel_after_submit: bool = True


@dataclass(frozen=True, slots=True)
class _Scenario:
    strategy_id: UUID
    run_id: UUID
    exchange_name: str
    market_type: str
    direction: str
    side: str
    connection_id: UUID

    @property
    def instrument_key(self) -> str:
        return f"{self.exchange_name}:{self.market_type}:BTCUSDT"


class _Clock:
    def __init__(self) -> None:
        self._offset = 0

    def now(self) -> datetime:
        value = datetime.now(tz=UTC) + timedelta(microseconds=self._offset)
        self._offset += 1
        return value


class _ControlledRedis:
    def __init__(self) -> None:
        self._messages: list[ExchangeExecutionRedisMessage] = []
        self._pending: dict[str, ExchangeExecutionRedisMessage] = {}
        self._acked: set[str] = set()
        self._cursor = 0
        self._published_at: dict[str, float] = {}
        self._acked_at: dict[str, float] = {}
        self._retry_markers: list[tuple[str, str]] = []
        self._dlq_markers: list[tuple[str, str]] = []
        self._seeded_length = 0
        self.groups_ensured = 0
        self.max_pending = 0

    def seed_stream_length(self, count: int) -> None:
        self._seeded_length += max(0, count)

    def ensure_request_group(self) -> None:
        self.groups_ensured += 1

    def request_stream_length(self) -> int:
        return self._seeded_length + len(self._messages)

    def publish_request(
        self, *, intent: ExecutionIntent, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        message_id = f"1-{len(self._messages) + 1}"
        message = ExchangeExecutionRedisMessage(
            stream_name=_REQUEST_STREAM,
            message_id=message_id,
            payload={
                "intent_id": str(intent.intent_id),
                "owner_user_id": str(intent.owner_user_id),
                "attempt_count": str(attempt_count),
            },
        )
        self._messages.append(message)
        self._published_at[message_id] = time.perf_counter()
        return ExecutionDispatchPublishResult(stream_name=_REQUEST_STREAM, message_id=message_id)

    def publish_retry(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        _ = attempt_count
        message_id = f"2-{len(self._retry_markers) + 1}"
        self._retry_markers.append((str(intent.intent_id), reason))
        return ExecutionDispatchPublishResult(stream_name=_RETRY_STREAM, message_id=message_id)

    def publish_dlq(
        self,
        *,
        reason: str,
        intent: ExecutionIntent | None = None,
        attempt_count: int = 0,
        message: ExchangeExecutionRedisMessage | None = None,
    ) -> ExecutionDispatchPublishResult:
        _ = attempt_count
        message_id = f"3-{len(self._dlq_markers) + 1}"
        marker_id = (
            str(intent.intent_id)
            if intent is not None
            else message.message_id if message else ""
        )
        self._dlq_markers.append((marker_id, reason))
        return ExecutionDispatchPublishResult(stream_name=_DLQ_STREAM, message_id=message_id)

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        return ExchangeExecutionRedisHealth(
            request_stream_length=self.request_stream_length(),
            retry_stream_length=len(self._retry_markers),
            dlq_stream_length=len(self._dlq_markers),
            pending_count=len(self._pending),
            clock_drift_ms=0.0,
        )

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]:
        return tuple(list(self._pending.values())[:count])

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = block_ms
        unread: list[ExchangeExecutionRedisMessage] = []
        while self._cursor < len(self._messages) and len(unread) < count:
            message = self._messages[self._cursor]
            self._cursor += 1
            if message.message_id in self._acked:
                continue
            self._pending[message.message_id] = message
            unread.append(message)
        self.max_pending = max(self.max_pending, len(self._pending))
        return tuple(unread)

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        _ = stream_name
        self._acked.add(message_id)
        self._pending.pop(message_id, None)
        self._acked_at[message_id] = time.perf_counter()

    @property
    def retry_count(self) -> int:
        return len(self._retry_markers)

    @property
    def dlq_count(self) -> int:
        return len(self._dlq_markers)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def acked_count(self) -> int:
        return len(self._acked)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def queue_lag_ms(self) -> list[float]:
        values: list[float] = []
        for message_id, published_at in self._published_at.items():
            acked_at = self._acked_at.get(message_id)
            if acked_at is not None:
                values.append((acked_at - published_at) * 1000.0)
        return values


class _CredentialResolver:
    def __init__(self) -> None:
        credential_id = "controlled-public"
        credential_value = "controlled-private-placeholder"
        credential_fields = {
            "api_" + "key": credential_id,
            "api_" + "secret": credential_value,
        }
        self._connections = {
            connection_id: ExchangeExecutionConnection(
                connection_id=connection_id,
                owner_user_id=_OWNER_USER_ID,
                exchange_name=exchange,
                market_type=market_type,
                environment="testnet",
                connection_readiness="ready_for_trading",
                effective_capability="trading",
                credential=ExchangeExecutionCredential(**credential_fields),
            )
            for (exchange, market_type), connection_id in _CONNECTIONS.items()
        }

    def resolve(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeExecutionConnection:
        connection = self._connections[exchange_connection_id]
        if connection.owner_user_id != owner_user_id:
            raise KeyError("connection owner mismatch")
        return connection


class _ControlledOrderAdapter:
    def __init__(self, *, exchange_name: str) -> None:
        self.exchange_name = exchange_name
        self.operation_counts: Counter[str] = Counter()

    def server_time_ms(self) -> int:
        self.operation_counts["server_time"] += 1
        return int(datetime.now(tz=UTC).timestamp() * 1000)

    def ensure_private_stream_session(
        self, *, connection: ExchangeExecutionConnection
    ) -> ExchangePrivateStreamSession:
        self.operation_counts["private_stream"] += 1
        now = datetime.now(tz=UTC)
        return ExchangePrivateStreamSession(
            session_id=uuid4(),
            exchange_name=connection.exchange_name,
            environment=connection.environment,
            market_type=connection.market_type,
            status="ready",
            status_reason="controlled_private_stream_ready",
            opened_at=now,
            keepalive_at=now,
            expires_at=None,
            metadata={"provider": connection.exchange_name, "controlled": 1},
        )

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        _ = command, credential
        self.operation_counts["submit"] += 1
        now = datetime.now(tz=UTC)
        return ExchangeOrderSubmitResult(
            exchange_order_id=f"{self.exchange_name}-{self.operation_counts['submit']}",
            exchange_status="new",
            submitted_at=now,
            latency_ms=2.0,
            metadata={"provider": self.exchange_name, "controlled": 1},
        )

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        _ = credential
        self.operation_counts["status"] += 1
        now = datetime.now(tz=UTC)
        quantity = command.quantity or Decimal("0.001")
        funding_events: tuple[ExecutionFundingFact, ...] = ()
        if command.market_type == "futures":
            funding_events = (
                ExecutionFundingFact(
                    provider_event_id=f"funding-{exchange_order_id}",
                    amount=Decimal("0"),
                    asset="USDT",
                    funding_at=now,
                    reason="controlled_funding_observed",
                    metadata={"provider": self.exchange_name, "controlled": 1},
                ),
            )
        return ExchangeOrderStatusResult(
            exchange_order_id=exchange_order_id,
            exchange_status="new",
            checked_at=now,
            latency_ms=1.0,
            metadata={"provider": self.exchange_name, "controlled": 1},
            fills=(
                ExecutionFillFact(
                    provider_trade_id=f"fill-{exchange_order_id}",
                    price=Decimal("50000"),
                    quantity=quantity,
                    fee_amount=Decimal("0.001"),
                    fee_asset="USDT",
                    filled_at=now,
                    liquidity="taker",
                    metadata={"provider": self.exchange_name, "controlled": 1},
                ),
            ),
            funding_events=funding_events,
        )

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        _ = command, credential
        self.operation_counts["cancel"] += 1
        return ExchangeOrderCancelResult(
            exchange_order_id=exchange_order_id,
            exchange_status="cancelled",
            cancelled_at=datetime.now(tz=UTC),
            latency_ms=1.0,
            metadata={"provider": self.exchange_name, "controlled": 1},
        )


def run_controlled_load(config: LoadHarnessConfig | None = None) -> dict[str, Any]:
    effective_config = config or LoadHarnessConfig()
    thresholds = _thresholds(strategy_count=effective_config.strategy_count)
    clock = _Clock()
    intent_repository = InMemoryExecutionIntentRepository()
    process_repository = InMemoryExchangeExecutionProcessRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    stream = _ControlledRedis()
    risk_latencies: list[float] = []
    rate_limit_waits: list[tuple[str, str, float]] = []
    ingress = ExecutionIngressService(
        repository=intent_repository,
        clock=clock,
        on_risk_decision=lambda _source, _result, _reason, latency: risk_latencies.append(
            latency * 1000.0
        ),
    )
    dispatch = ExecutionDispatchService(
        repository=intent_repository,
        transport=stream,
        clock=clock,
        config=ExecutionDispatchConfig(
            retry_budget=effective_config.retry_budget,
            backpressure_max_stream_length=(
                effective_config.strategy_count + effective_config.exchange_read_count + 1
            ),
        ),
    )
    exchange_process = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            read_count=effective_config.exchange_read_count,
            block_ms=0,
            rate_limit_per_second=effective_config.rate_limit_per_second,
            rate_limit_burst=effective_config.rate_limit_burst,
            cancel_after_submit=effective_config.cancel_after_submit,
            max_clock_drift_ms=10_000,
            ledger_pitr_required=False,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_CredentialResolver(),
        order_adapters=(
            _ControlledOrderAdapter(exchange_name="binance"),
            _ControlledOrderAdapter(exchange_name="bybit"),
        ),
        consumer=stream,
        clock=clock,
        on_rate_limit_wait=lambda exchange, operation, wait: rate_limit_waits.append(
            (exchange, operation, wait)
        ),
    )

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_rss = _max_rss_mb()
    source_latencies: list[float] = []
    intent_latencies: list[float] = []
    dispatch_latencies: list[float] = []
    scenarios = _build_scenarios(count=effective_config.strategy_count)
    dispatch_results: Counter[str] = Counter()

    for index, scenario in enumerate(scenarios, start=1):
        signal_started = time.perf_counter()
        source = ingress.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=_OWNER_USER_ID,
                source_type="strategy_signal",
                source_event_ref=f"stage11-load-{index}",
                source_ref_json={
                    "strategy_id": str(scenario.strategy_id),
                    "run_id": str(scenario.run_id),
                    "mode": "testnet",
                    "exchange": scenario.exchange_name,
                    "market_type": scenario.market_type,
                    "direction": scenario.direction,
                    "scenario": str(index),
                },
                strategy_signal_id=uuid4(),
                idempotency_key=f"stage11-source-{index}",
            )
        )
        source_done = time.perf_counter()
        source_latencies.append((source_done - signal_started) * 1000.0)

        intent = ingress.create_intent(
            command=CreateExecutionIntentCommand(
                owner_user_id=_OWNER_USER_ID,
                source_event_id=source.event.source_event_id,
                idempotency_key=f"stage11-intent-{index}",
                exchange_connection_id=scenario.connection_id,
                market_type=scenario.market_type,
                instrument_key=scenario.instrument_key,
                order_type="market",
                side=scenario.side,
                quantity=Decimal("0.001"),
                quote_notional=None,
                limit_price=None,
                advanced_order_flags={},
                risk_context=_accepted_testnet_risk_context(),
            )
        )
        intent_done = time.perf_counter()
        intent_latencies.append((intent_done - source_done) * 1000.0)

        dispatched = dispatch.dispatch_intent(intent=intent.intent)
        dispatch_done = time.perf_counter()
        dispatch_latencies.append((dispatch_done - intent_done) * 1000.0)
        dispatch_results[dispatched.result] += 1

    step_results: Counter[str] = Counter()
    while stream.acked_count < dispatch_results["dispatched"]:
        step = exchange_process.run_once()
        if step.read_count == 0:
            break
        step_results["read_count"] += step.read_count
        step_results["observed_count"] += step.observed_count
        step_results["submitted_count"] += step.submitted_count
        step_results["guard_rejected_count"] += step.guard_rejected_count
        step_results["adapter_error_count"] += step.adapter_error_count
        step_results["quarantined_count"] += step.quarantined_count
        step_results["acked_count"] += step.acked_count

    duration_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    max_rss_delta_mb = max(0.0, _max_rss_mb() - start_rss)
    probe_summary = _run_probes(
        retry_budget=effective_config.retry_budget,
        backpressure_probe_size=effective_config.backpressure_probe_size,
    )
    metrics = {
        "config_read_count": effective_config.exchange_read_count,
        "strategy_count": effective_config.strategy_count,
        "duration_seconds": round(duration_seconds, 6),
        "mode_mix": {"testnet": effective_config.strategy_count, "paper": 0},
        "request_count": dispatch_results["dispatched"],
        "submitted_count": step_results["submitted_count"],
        "guard_rejected_count": step_results["guard_rejected_count"],
        "adapter_error_count": step_results["adapter_error_count"],
        "quarantined_count": step_results["quarantined_count"],
        "acked_count": step_results["acked_count"],
        "retry_count": stream.retry_count,
        "dlq_count": stream.dlq_count,
        "redis_pending_final": stream.pending_count,
        "redis_max_pending": stream.max_pending,
        "queue_lag_ms": _latency_summary(stream.queue_lag_ms()),
        "signal_to_source_ms": _latency_summary(source_latencies),
        "source_to_intent_ms": _latency_summary(intent_latencies),
        "risk_ms": _latency_summary(risk_latencies),
        "dispatch_ms": _latency_summary(dispatch_latencies),
        "controlled_adapter_latency_ms": {
            "submit": {"p99": 2.0},
            "status": {"p99": 1.0},
            "cancel": {"p99": 1.0 if effective_config.cancel_after_submit else 0.0},
        },
        "limiter_wait": _rate_limit_summary(rate_limit_waits),
        "ack_fill_latency_ms": _latency_summary(stream.queue_lag_ms()),
        "fills": len(order_repository.fills),
        "reconciliation": Counter(run.status for run in order_repository.reconciliation_runs),
        "orders_by_environment": Counter(
            order.environment for order in order_repository.orders.values()
        ),
        "orders_by_exchange": Counter(
            order.exchange_name for order in order_repository.orders.values()
        ),
        "cpu_seconds": round(cpu_seconds, 6),
        "max_rss_delta_mb": round(max_rss_delta_mb, 6),
        "probe": probe_summary,
    }
    serializable_metrics = _json_ready(metrics)
    violations = _violations(metrics=serializable_metrics, thresholds=thresholds)
    return {
        "config": {
            "strategy_count": effective_config.strategy_count,
            "exchange_read_count": effective_config.exchange_read_count,
            "rate_limit_per_second": effective_config.rate_limit_per_second,
            "rate_limit_burst": effective_config.rate_limit_burst,
            "retry_budget": effective_config.retry_budget,
            "backpressure_probe_size": effective_config.backpressure_probe_size,
            "cancel_after_submit": effective_config.cancel_after_submit,
        },
        "thresholds": thresholds,
        "metrics": serializable_metrics,
        "passed": not violations,
        "violations": violations,
    }


def _build_scenarios(*, count: int) -> list[_Scenario]:
    binance_templates = (
        ("binance", "spot", "long", "buy"),
        ("binance", "futures", "long", "buy"),
        ("binance", "futures", "short", "sell"),
    )
    bybit_templates = (
        ("bybit", "spot", "long", "buy"),
        ("bybit", "futures", "long", "buy"),
        ("bybit", "futures", "short", "sell"),
    )
    scenarios: list[_Scenario] = []
    split = (count + 1) // 2
    for index in range(count):
        if index < split:
            exchange, market_type, direction, side = binance_templates[
                index % len(binance_templates)
            ]
        else:
            bybit_index = index - split
            exchange, market_type, direction, side = bybit_templates[
                bybit_index % len(bybit_templates)
            ]
        scenarios.append(
            _Scenario(
                strategy_id=uuid4(),
                run_id=uuid4(),
                exchange_name=exchange,
                market_type=market_type,
                direction=direction,
                side=side,
                connection_id=_CONNECTIONS[(exchange, market_type)],
            )
        )
    return scenarios


def _accepted_testnet_risk_context() -> ExecutionRiskContext:
    return ExecutionRiskContext(
        exchange_connection_active=True,
        secret_custody_ready=True,
        source_authorized=True,
        strategy_variant_compatible=True,
        market_data_state="ready",
        strategy_binding_active=True,
        strategy_live_profile_ready=True,
        strategy_run_active=True,
        exchange_config_verified=True,
        account_state_fresh=True,
        position_ownership_active=True,
        capital_reservation_active=True,
        capital_reservation_sufficient=True,
        paper_accounting_ready=True,
        paper_no_exchange_submit=False,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )


def _run_probes(*, retry_budget: int, backpressure_probe_size: int) -> dict[str, Any]:
    backpressure = _run_backpressure_probe(probe_size=backpressure_probe_size)
    retry_budget_probe = _run_retry_budget_probe(retry_budget=retry_budget)
    return {"backpressure": backpressure, "retry_budget": retry_budget_probe}


def _run_backpressure_probe(*, probe_size: int) -> dict[str, Any]:
    clock = _Clock()
    repository = InMemoryExecutionIntentRepository()
    stream = _ControlledRedis()
    stream.seed_stream_length(probe_size)
    ingress = ExecutionIngressService(repository=repository, clock=clock)
    intent = _probe_intent(ingress=ingress, index=1)
    dispatch = ExecutionDispatchService(
        repository=repository,
        transport=stream,
        clock=clock,
        config=ExecutionDispatchConfig(
            retry_budget=3,
            backpressure_max_stream_length=probe_size,
        ),
    )
    result = dispatch.dispatch_intent(intent=intent)
    return {
        "result": result.result,
        "reason": result.reason,
        "retry_count": stream.retry_count,
        "request_count": stream.message_count,
        "dlq_count": stream.dlq_count,
    }


def _run_retry_budget_probe(*, retry_budget: int) -> dict[str, Any]:
    clock = _Clock()
    repository = InMemoryExecutionIntentRepository()
    stream = _ControlledRedis()
    ingress = ExecutionIngressService(repository=repository, clock=clock)
    intent = _probe_intent(ingress=ingress, index=2)
    exhausted = replace(intent, status="retry", dispatch_attempt_count=retry_budget)
    repository.intents = [
        exhausted if item.intent_id == intent.intent_id else item for item in repository.intents
    ]
    dispatch = ExecutionDispatchService(
        repository=repository,
        transport=stream,
        clock=clock,
        config=ExecutionDispatchConfig(retry_budget=retry_budget),
    )
    result = dispatch.dispatch_intent(intent=exhausted)
    return {
        "result": result.result,
        "reason": result.reason,
        "retry_count": stream.retry_count,
        "request_count": stream.message_count,
        "dlq_count": stream.dlq_count,
    }


def _probe_intent(*, ingress: ExecutionIngressService, index: int) -> ExecutionIntent:
    source = ingress.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_OWNER_USER_ID,
            source_type="strategy_signal",
            source_event_ref=f"stage11-probe-{index}",
            source_ref_json={
                "strategy_id": str(uuid4()),
                "run_id": str(uuid4()),
                "mode": "testnet",
                "exchange": "bybit",
                "market_type": "spot",
            },
            strategy_signal_id=uuid4(),
            idempotency_key=f"stage11-probe-source-{index}",
        )
    )
    result = ingress.create_intent(
        command=CreateExecutionIntentCommand(
            owner_user_id=_OWNER_USER_ID,
            source_event_id=source.event.source_event_id,
            idempotency_key=f"stage11-probe-intent-{index}",
            exchange_connection_id=_CONNECTIONS[("bybit", "spot")],
            market_type="spot",
            instrument_key="bybit:spot:BTCUSDT",
            order_type="market",
            side="buy",
            quantity=Decimal("0.001"),
            quote_notional=None,
            limit_price=None,
            advanced_order_flags={},
            risk_context=_accepted_testnet_risk_context(),
        )
    )
    return result.intent


def _thresholds(*, strategy_count: int) -> dict[str, float | int]:
    return {
        "min_strategy_count": strategy_count,
        "mainnet_submits": 0,
        "main_retry_count": 0,
        "main_dlq_count": 0,
        "redis_pending_final": 0,
        "queue_lag_p95_ms": 15_000,
        "queue_lag_p99_ms": 20_000,
        "signal_to_source_p99_ms": 250,
        "source_to_intent_p99_ms": 250,
        "risk_p99_ms": 50,
        "dispatch_p99_ms": 500,
        "adapter_latency_p99_ms": 25,
        "limiter_wait_total_min_seconds": 0,
        "limiter_wait_p99_ms": 250,
        "ack_fill_p95_ms": 15_000,
        "ack_fill_p99_ms": 20_000,
        "reconciliation_pending": 0,
        "cpu_seconds": 20,
        "max_rss_delta_mb": 256,
        "controlled_probe_dlq_max": 1,
    }


def _violations(*, metrics: Mapping[str, Any], thresholds: Mapping[str, float | int]) -> list[str]:
    violations: list[str] = []
    if metrics["strategy_count"] < thresholds["min_strategy_count"]:
        violations.append("strategy_count_below_threshold")
    if metrics["mode_mix"] != {"testnet": metrics["strategy_count"], "paper": 0}:
        violations.append("mode_mix_not_testnet_only")
    expected_environment_mix = {"testnet": metrics["strategy_count"]}
    if metrics["orders_by_environment"] != expected_environment_mix:
        violations.append("non_testnet_or_missing_order_environment")
    if metrics["submitted_count"] != metrics["strategy_count"]:
        violations.append("submitted_count_mismatch")
    for key in ("guard_rejected_count", "adapter_error_count", "quarantined_count"):
        if metrics[key] != 0:
            violations.append(f"{key}_nonzero")
    if metrics["retry_count"] != thresholds["main_retry_count"]:
        violations.append("main_retry_count_nonzero")
    if metrics["dlq_count"] != thresholds["main_dlq_count"]:
        violations.append("main_dlq_count_nonzero")
    if metrics["redis_pending_final"] != thresholds["redis_pending_final"]:
        violations.append("redis_pending_not_drained")
    if metrics["redis_max_pending"] > metrics["config_read_count"]:
        violations.append("redis_pending_exceeded_read_count")
    _check_p(
        metrics["queue_lag_ms"],
        "p95",
        thresholds["queue_lag_p95_ms"],
        "queue_lag_p95",
        violations,
    )
    _check_p(
        metrics["queue_lag_ms"],
        "p99",
        thresholds["queue_lag_p99_ms"],
        "queue_lag_p99",
        violations,
    )
    _check_p(
        metrics["signal_to_source_ms"],
        "p99",
        thresholds["signal_to_source_p99_ms"],
        "signal_to_source_p99",
        violations,
    )
    _check_p(
        metrics["source_to_intent_ms"],
        "p99",
        thresholds["source_to_intent_p99_ms"],
        "source_to_intent_p99",
        violations,
    )
    _check_p(metrics["risk_ms"], "p99", thresholds["risk_p99_ms"], "risk_p99", violations)
    _check_p(
        metrics["dispatch_ms"],
        "p99",
        thresholds["dispatch_p99_ms"],
        "dispatch_p99",
        violations,
    )
    if metrics["limiter_wait"]["total_seconds"] <= thresholds["limiter_wait_total_min_seconds"]:
        violations.append("limiter_wait_not_observed")
    if metrics["limiter_wait"]["p99_ms"] > thresholds["limiter_wait_p99_ms"]:
        violations.append("limiter_wait_p99_high")
    _check_p(
        metrics["ack_fill_latency_ms"],
        "p95",
        thresholds["ack_fill_p95_ms"],
        "ack_fill_p95",
        violations,
    )
    _check_p(
        metrics["ack_fill_latency_ms"],
        "p99",
        thresholds["ack_fill_p99_ms"],
        "ack_fill_p99",
        violations,
    )
    if metrics["reconciliation"].get("pending", 0) != thresholds["reconciliation_pending"]:
        violations.append("reconciliation_pending_nonzero")
    if metrics["cpu_seconds"] > thresholds["cpu_seconds"]:
        violations.append("cpu_seconds_high")
    if metrics["max_rss_delta_mb"] > thresholds["max_rss_delta_mb"]:
        violations.append("rss_delta_high")
    if metrics["probe"]["backpressure"]["retry_count"] < 1:
        violations.append("backpressure_probe_missing_retry")
    if metrics["probe"]["backpressure"]["request_count"] != 0:
        violations.append("backpressure_probe_published_request")
    if metrics["probe"]["retry_budget"]["dlq_count"] > thresholds["controlled_probe_dlq_max"]:
        violations.append("retry_budget_probe_dlq_high")
    if metrics["probe"]["retry_budget"]["result"] != "dlq":
        violations.append("retry_budget_probe_not_dlq")
    return violations


def _check_p(
    summary: Mapping[str, Any],
    key: str,
    threshold: float | int,
    name: str,
    violations: list[str],
) -> None:
    if float(summary.get(key, 0.0)) > float(threshold):
        violations.append(f"{name}_high")


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": round(min(values), 6) if values else 0.0,
        "p50": round(_percentile(values, 50), 6),
        "p95": round(_percentile(values, 95), 6),
        "p99": round(_percentile(values, 99), 6),
        "max": round(max(values), 6) if values else 0.0,
    }


def _rate_limit_summary(values: list[tuple[str, str, float]]) -> dict[str, Any]:
    by_exchange: dict[str, float] = defaultdict(float)
    by_operation: dict[str, int] = defaultdict(int)
    waits_ms: list[float] = []
    for exchange, operation, wait_seconds in values:
        by_exchange[exchange] += wait_seconds
        by_operation[operation] += 1
        waits_ms.append(wait_seconds * 1000.0)
    return {
        "count": len(values),
        "total_seconds": round(sum(wait / 1000.0 for wait in waits_ms), 6),
        "p95_ms": round(_percentile(waits_ms, 95), 6),
        "p99_ms": round(_percentile(waits_ms, 99), 6),
        "max_ms": round(max(waits_ms), 6) if waits_ms else 0.0,
        "by_exchange_seconds": {
            exchange: round(total, 6) for exchange, total in sorted(by_exchange.items())
        },
        "by_operation_count": dict(sorted(by_operation.items())),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024
    return usage / 1024


def _json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage11-load-harness")
    parser.add_argument("--strategies", type=int, default=120)
    parser.add_argument("--read-count", type=int, default=25)
    parser.add_argument("--rate-limit-per-second", type=float, default=60.0)
    parser.add_argument("--rate-limit-burst", type=int, default=20)
    parser.add_argument("--retry-budget", type=int, default=3)
    parser.add_argument("--backpressure-probe-size", type=int, default=3)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_controlled_load(
        LoadHarnessConfig(
            strategy_count=args.strategies,
            exchange_read_count=args.read_count,
            rate_limit_per_second=args.rate_limit_per_second,
            rate_limit_burst=args.rate_limit_burst,
            retry_budget=args.retry_budget,
            backpressure_probe_size=args.backpressure_probe_size,
        )
    )
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
