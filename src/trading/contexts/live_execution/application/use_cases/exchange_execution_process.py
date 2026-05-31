from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionConsumer,
    ExchangeExecutionCredentialResolver,
    ExchangeExecutionCredentialUnavailable,
    ExchangeExecutionOrderRepository,
    ExchangeExecutionProcessRepository,
    ExchangeOrderAdapter,
    ExchangeOrderAdapterError,
    ExecutionDispatchUnavailableError,
    ExecutionIntentRepository,
    LiveExecutionClock,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionDependencyHealth,
    ExchangeExecutionHealthSnapshot,
    ExchangeExecutionObservationStatus,
    ExchangeExecutionProcessHeartbeat,
    ExchangeExecutionProcessStatus,
    ExchangeExecutionRequestObservation,
    ExchangeOrderCommand,
    ExecutionIntent,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class ExchangeExecutionProcessConfig:
    service_id: str = "exchange-execution"
    adapter_mode: str = "disabled"
    request_stream: str = "execution.requests.v1"
    retry_stream: str = "execution.requests.retry.v1"
    dlq_stream: str = "execution.requests.dlq.v1"
    consumer_group: str = "exchange-execution.v1"
    consumer_name: str = "exchange-execution-local"
    consumer_enabled: bool = False
    read_count: int = 10
    block_ms: int = 100
    backpressure_max_stream_length: int = 10_000
    max_clock_drift_ms: float = 1_000.0
    rate_limit_per_second: float = 5.0
    rate_limit_burst: int = 10
    enabled_exchanges: tuple[str, ...] = ("binance", "bybit")
    cancel_after_submit: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.service_id.strip() == "":
            raise ValueError("ExchangeExecutionProcessConfig.service_id must be non-empty")
        if self.adapter_mode not in {"disabled", "testnet"}:
            raise ValueError("exchange-execution adapter_mode must be disabled or testnet")
        if self.consumer_name.strip() == "":
            raise ValueError("ExchangeExecutionProcessConfig.consumer_name must be non-empty")
        if self.read_count <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.read_count must be > 0")
        if self.block_ms < 0:
            raise ValueError("ExchangeExecutionProcessConfig.block_ms must be >= 0")
        if self.backpressure_max_stream_length <= 0:
            raise ValueError(
                "ExchangeExecutionProcessConfig.backpressure_max_stream_length must be > 0"
            )
        if self.max_clock_drift_ms <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.max_clock_drift_ms must be > 0")
        if self.rate_limit_per_second <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.rate_limit_per_second must be > 0")
        if self.rate_limit_burst <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.rate_limit_burst must be > 0")


@dataclass(frozen=True, slots=True)
class ExchangeExecutionProcessStepResult:
    read_count: int
    observed_count: int
    submitted_count: int
    guard_rejected_count: int
    adapter_error_count: int
    quarantined_count: int
    acked_count: int
    reason: str


class ExchangeExecutionProcessService:
    def __init__(
        self,
        *,
        config: ExchangeExecutionProcessConfig,
        repository: ExchangeExecutionProcessRepository,
        intent_repository: ExecutionIntentRepository,
        consumer: ExchangeExecutionConsumer | None,
        clock: LiveExecutionClock,
        order_repository: ExchangeExecutionOrderRepository | None = None,
        credential_resolver: ExchangeExecutionCredentialResolver | None = None,
        order_adapters: tuple[ExchangeOrderAdapter, ...] = (),
        started_at: datetime | None = None,
        on_observation: Callable[[str, str], None] | None = None,
        on_dlq: Callable[[str], None] | None = None,
        on_ack: Callable[[str], None] | None = None,
        on_order_submit: Callable[[str, str], None] | None = None,
        on_private_stream: Callable[[str, str], None] | None = None,
        on_order_latency: Callable[[str, float], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires repository")
        if intent_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires intent_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires clock")
        self._config = config
        self._repository = repository
        self._intent_repository = intent_repository
        self._order_repository = order_repository
        self._credential_resolver = credential_resolver
        self._order_adapters = {adapter.exchange_name: adapter for adapter in order_adapters}
        self._consumer = consumer
        self._clock = clock
        self._started_at = started_at or clock.now()
        self._on_observation = on_observation
        self._on_dlq = on_dlq
        self._on_ack = on_ack
        self._on_order_submit = on_order_submit
        self._on_private_stream = on_private_stream
        self._on_order_latency = on_order_latency

    def readiness(self) -> ExchangeExecutionHealthSnapshot:
        now = self._clock.now()
        dependencies: list[ExchangeExecutionDependencyHealth] = [
            ExchangeExecutionDependencyHealth(
                name="config",
                status="ready",
                reason="config_loaded",
                metadata={
                    "consumer_enabled": int(self._config.consumer_enabled),
                    "fail_fast": int(self._config.fail_fast),
                },
            ),
            self._adapter_dependency(),
            ExchangeExecutionDependencyHealth(
                name="rate_limit",
                status="ready",
                reason="rate_limit_guard_configured",
                metadata={
                    "per_second": self._config.rate_limit_per_second,
                    "burst": self._config.rate_limit_burst,
                },
            ),
        ]
        dependencies.extend(self._redis_dependencies())
        status, reason = _rollup_status(dependencies=dependencies)
        heartbeat = ExchangeExecutionProcessHeartbeat(
            service_id=self._config.service_id,
            status=status,
            status_reason=reason,
            adapter_mode=self._config.adapter_mode,  # type: ignore[arg-type]
            started_at=self._started_at,
            heartbeat_at=now,
            request_stream=self._config.request_stream,
            consumer_group=self._config.consumer_group,
            consumer_name=self._config.consumer_name,
            metadata={"dependency_count": len(dependencies)},
        )
        try:
            self._repository.record_heartbeat(heartbeat=heartbeat)
            dependencies.append(
                ExchangeExecutionDependencyHealth(
                    name="postgres",
                    status="ready",
                    reason="heartbeat_recorded",
                    metadata={},
                )
            )
        except Exception:  # noqa: BLE001
            dependencies.append(
                ExchangeExecutionDependencyHealth(
                    name="postgres",
                    status="not_ready" if self._config.fail_fast else "degraded",
                    reason="heartbeat_record_failed",
                    metadata={},
                )
            )
        status, reason = _rollup_status(dependencies=dependencies)
        return ExchangeExecutionHealthSnapshot(
            service_id=self._config.service_id,
            status=status,
            status_reason=reason,
            adapter_mode=self._config.adapter_mode,  # type: ignore[arg-type]
            checked_at=now,
            dependencies=tuple(dependencies),
        )

    def run_once(self) -> ExchangeExecutionProcessStepResult:
        if self._consumer is None or not self._config.consumer_enabled:
            return ExchangeExecutionProcessStepResult(
                read_count=0,
                observed_count=0,
                submitted_count=0,
                guard_rejected_count=0,
                adapter_error_count=0,
                quarantined_count=0,
                acked_count=0,
                reason="consumer_disabled",
            )
        self._consumer.ensure_request_group()
        messages = self._consumer.read_new_requests(
            count=self._config.read_count,
            block_ms=self._config.block_ms,
        )
        observed_count = 0
        submitted_count = 0
        guard_rejected_count = 0
        adapter_error_count = 0
        quarantined_count = 0
        acked_count = 0
        for message in messages:
            status, reason, intent = self._classify_message(payload=dict(message.payload))
            intent_id = (
                intent.intent_id if intent is not None else _intent_id_or_none(message.payload)
            )
            if status == "adapter_disabled":
                pass
            elif status == "skipped":
                status, reason = self._execute_testnet_intent(intent=intent)
            observation = ExchangeExecutionRequestObservation(
                observation_id=uuid4(),
                service_id=self._config.service_id,
                intent_id=intent_id,
                stream_name=message.stream_name,
                redis_message_id=message.message_id,
                status=status,
                status_reason=reason,
                adapter_mode=self._config.adapter_mode,  # type: ignore[arg-type]
                observed_at=self._clock.now(),
                metadata={"payload_fields": len(message.payload)},
            )
            self._repository.record_request_observation(observation=observation)
            self._record_observation(status=status, reason=reason)
            observed_count += 1
            if status == "quarantined":
                self._consumer.publish_dlq(message=message, reason=reason)
                self._record_dlq(reason=reason)
                self._consumer.ack_after_durable_state_change(
                    stream_name=message.stream_name,
                    message_id=message.message_id,
                )
                self._record_ack(reason=reason)
                quarantined_count += 1
                acked_count += 1
            elif status in {"testnet_submitted", "guard_rejected", "adapter_error"}:
                self._consumer.ack_after_durable_state_change(
                    stream_name=message.stream_name,
                    message_id=message.message_id,
                )
                self._record_ack(reason=reason)
                acked_count += 1
                if status == "testnet_submitted":
                    submitted_count += 1
                elif status == "guard_rejected":
                    guard_rejected_count += 1
                else:
                    adapter_error_count += 1
        return ExchangeExecutionProcessStepResult(
            read_count=len(messages),
            observed_count=observed_count,
            submitted_count=submitted_count,
            guard_rejected_count=guard_rejected_count,
            adapter_error_count=adapter_error_count,
            quarantined_count=quarantined_count,
            acked_count=acked_count,
            reason=(
                "adapter_disabled_no_submit"
                if self._config.adapter_mode == "disabled"
                else "testnet_adapter_processed"
            ),
        )

    def _adapter_dependency(self) -> ExchangeExecutionDependencyHealth:
        if self._config.adapter_mode == "disabled":
            return ExchangeExecutionDependencyHealth(
                name="adapter",
                status="degraded",
                reason="adapter_disabled_stage13",
                metadata={"submit_enabled": 0},
            )
        missing = []
        if self._order_repository is None:
            missing.append("order_repository")
        if self._credential_resolver is None:
            missing.append("credential_resolver")
        for exchange in self._config.enabled_exchanges:
            if exchange not in self._order_adapters:
                missing.append(f"{exchange}_adapter")
        if missing:
            return ExchangeExecutionDependencyHealth(
                name="adapter",
                status="not_ready" if self._config.fail_fast else "degraded",
                reason="testnet_adapter_dependency_missing",
                metadata={"missing_count": len(missing), "submit_enabled": 0},
            )
        return ExchangeExecutionDependencyHealth(
            name="adapter",
            status="ready",
            reason="testnet_adapters_ready",
            metadata={
                "submit_enabled": 1,
                "enabled_exchange_count": len(self._config.enabled_exchanges),
            },
        )

    def _redis_dependencies(self) -> tuple[ExchangeExecutionDependencyHealth, ...]:
        if self._consumer is None:
            return (
                ExchangeExecutionDependencyHealth(
                    name="redis",
                    status="degraded",
                    reason="redis_consumer_disabled",
                    metadata={"consumer_enabled": 0},
                ),
                ExchangeExecutionDependencyHealth(
                    name="backpressure",
                    status="degraded",
                    reason="backpressure_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="dlq",
                    status="degraded",
                    reason="dlq_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="clock_drift",
                    status="degraded",
                    reason="clock_drift_state_unavailable",
                    metadata={},
                ),
            )
        try:
            self._consumer.ensure_request_group()
            snapshot = self._consumer.health_snapshot()
        except ExecutionDispatchUnavailableError as error:
            return (
                ExchangeExecutionDependencyHealth(
                    name="redis",
                    status="not_ready" if self._config.fail_fast else "degraded",
                    reason=error.reason,
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="backpressure",
                    status="degraded",
                    reason="backpressure_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="dlq",
                    status="degraded",
                    reason="dlq_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="clock_drift",
                    status="degraded",
                    reason="clock_drift_state_unavailable",
                    metadata={},
                ),
            )
        backpressure_status = (
            "degraded"
            if snapshot.request_stream_length >= self._config.backpressure_max_stream_length
            else "ready"
        )
        clock_status = (
            "degraded"
            if abs(snapshot.clock_drift_ms) > self._config.max_clock_drift_ms
            else "ready"
        )
        return (
            ExchangeExecutionDependencyHealth(
                name="redis",
                status="ready",
                reason="redis_streams_observed",
                metadata={
                    "request_stream_length": snapshot.request_stream_length,
                    "pending_count": snapshot.pending_count,
                },
            ),
            ExchangeExecutionDependencyHealth(
                name="backpressure",
                status=backpressure_status,
                reason=(
                    "dispatch_backpressure"
                    if backpressure_status == "degraded"
                    else "backpressure_within_limit"
                ),
                metadata={
                    "request_stream_length": snapshot.request_stream_length,
                    "max_stream_length": self._config.backpressure_max_stream_length,
                },
            ),
            ExchangeExecutionDependencyHealth(
                name="dlq",
                status="ready",
                reason="dlq_stream_observed",
                metadata={"dlq_stream_length": snapshot.dlq_stream_length},
            ),
            ExchangeExecutionDependencyHealth(
                name="clock_drift",
                status=clock_status,
                reason=(
                    "clock_drift_exceeds_limit"
                    if clock_status == "degraded"
                    else "clock_drift_within_limit"
                ),
                metadata={
                    "clock_drift_ms": round(snapshot.clock_drift_ms, 3),
                    "max_clock_drift_ms": self._config.max_clock_drift_ms,
                },
            ),
        )

    def _classify_message(
        self, *, payload: dict[str, str]
    ) -> tuple[ExchangeExecutionObservationStatus, str, ExecutionIntent | None]:
        try:
            intent_id = UUID(payload["intent_id"])
            owner_user_id = UserId.from_string(payload["owner_user_id"])
        except (KeyError, ValueError):
            return "quarantined", "dispatch_payload_invalid_identity", None
        intent = self._intent_repository.get_intent_by_id(
            owner_user_id=owner_user_id,
            intent_id=intent_id,
        )
        if intent is None:
            return "quarantined", "intent_not_found", None
        if intent.status != "dispatched" or intent.risk_status != "accepted":
            return "quarantined", "intent_not_dispatchable", intent
        if self._config.adapter_mode == "disabled":
            return "adapter_disabled", "adapter_disabled_stage13", intent
        return "skipped", "testnet_execution_pending", intent

    def _execute_testnet_intent(
        self, *, intent: ExecutionIntent | None
    ) -> tuple[ExchangeExecutionObservationStatus, str]:
        if intent is None:
            return "quarantined", "intent_not_found"
        if self._order_repository is None or self._credential_resolver is None:
            return "guard_rejected", "testnet_adapter_dependency_missing"
        exchange_name = _exchange_from_instrument_key(intent.instrument_key)
        adapter = self._order_adapters.get(exchange_name)
        if exchange_name not in self._config.enabled_exchanges or adapter is None:
            command = _command_from_intent(intent=intent, exchange_name=exchange_name)
            self._order_repository.record_guard_rejection(
                command=command,
                reason="exchange_adapter_not_enabled",
            )
            return "guard_rejected", "exchange_adapter_not_enabled"
        command = _command_from_intent(intent=intent, exchange_name=exchange_name)
        existing = self._order_repository.get_by_intent(intent_id=intent.intent_id)
        if existing is not None and existing.exchange_order_id is not None:
            return "testnet_submitted", "order_already_processed"
        try:
            connection = self._credential_resolver.resolve(
                owner_user_id=intent.owner_user_id,
                exchange_connection_id=intent.exchange_connection_id,
            )
        except ExchangeExecutionCredentialUnavailable as error:
            self._order_repository.record_guard_rejection(command=command, reason=error.reason)
            return "guard_rejected", error.reason
        guard_reason = _connection_guard_reason(intent=intent, connection=connection)
        if guard_reason is not None:
            guarded_command = _command_from_intent(
                intent=intent,
                exchange_name=connection.exchange_name,
                environment=connection.environment,
            )
            self._order_repository.record_guard_rejection(
                command=guarded_command,
                reason=guard_reason,
            )
            return "guard_rejected", guard_reason
        command = _command_from_intent(
            intent=intent,
            exchange_name=connection.exchange_name,
            environment=connection.environment,
        )
        clock_reason = self._exchange_clock_guard_reason(adapter=adapter)
        if clock_reason is not None:
            self._order_repository.record_guard_rejection(command=command, reason=clock_reason)
            return "guard_rejected", clock_reason
        self._order_repository.record_submit_pending(command=command)
        try:
            session = adapter.ensure_private_stream_session(connection=connection)
            self._order_repository.record_private_stream_session(
                connection_id=connection.connection_id,
                session=session,
            )
            self._record_private_stream(
                exchange=connection.exchange_name,
                reason=session.status_reason,
            )
            submitted = adapter.submit_order(
                command=command,
                credential=connection.credential,
            )
            self._order_repository.record_submit_result(
                intent_id=intent.intent_id,
                result=submitted,
            )
            self._record_order_submit(
                exchange=connection.exchange_name,
                reason=submitted.exchange_status,
            )
            self._record_order_latency(
                exchange=connection.exchange_name,
                latency_ms=submitted.latency_ms,
            )
            status = adapter.get_order_status(
                command=command,
                exchange_order_id=submitted.exchange_order_id,
                credential=connection.credential,
            )
            self._order_repository.record_status_result(intent_id=intent.intent_id, result=status)
            if self._config.cancel_after_submit:
                cancelled = adapter.cancel_order(
                    command=command,
                    exchange_order_id=submitted.exchange_order_id,
                    credential=connection.credential,
                )
                self._order_repository.record_cancel_result(
                    intent_id=intent.intent_id,
                    result=cancelled,
                )
                self._record_order_latency(
                    exchange=connection.exchange_name,
                    latency_ms=cancelled.latency_ms,
                )
        except ExchangeOrderAdapterError as error:
            reason = (
                "adapter_unknown_state_reconciliation_required"
                if error.unknown_state
                else error.reason
            )
            self._order_repository.record_adapter_error(intent_id=intent.intent_id, reason=reason)
            return "adapter_error", reason
        return "testnet_submitted", "testnet_submit_status_cancel_recorded"

    def _exchange_clock_guard_reason(self, *, adapter: ExchangeOrderAdapter) -> str | None:
        try:
            server_time_ms = adapter.server_time_ms()
        except ExchangeOrderAdapterError as error:
            return error.reason
        local_ms = int(self._clock.now().timestamp() * 1000)
        drift_ms = abs(local_ms - server_time_ms)
        if drift_ms > self._config.max_clock_drift_ms:
            return "clock_drift_exceeds_limit"
        return None

    def _record_observation(self, *, status: str, reason: str) -> None:
        if self._on_observation is not None:
            self._on_observation(status, reason)

    def _record_dlq(self, *, reason: str) -> None:
        if self._on_dlq is not None:
            self._on_dlq(reason)

    def _record_ack(self, *, reason: str) -> None:
        if self._on_ack is not None:
            self._on_ack(reason)

    def _record_order_submit(self, *, exchange: str, reason: str) -> None:
        if self._on_order_submit is not None:
            self._on_order_submit(exchange, reason)

    def _record_private_stream(self, *, exchange: str, reason: str) -> None:
        if self._on_private_stream is not None:
            self._on_private_stream(exchange, reason)

    def _record_order_latency(self, *, exchange: str, latency_ms: float) -> None:
        if self._on_order_latency is not None:
            self._on_order_latency(exchange, latency_ms)


def _rollup_status(
    *, dependencies: list[ExchangeExecutionDependencyHealth]
) -> tuple[ExchangeExecutionProcessStatus, str]:
    if any(item.status == "not_ready" for item in dependencies):
        return "not_ready", "dependency_not_ready"
    if any(item.status == "degraded" for item in dependencies):
        reasons = [item.reason for item in dependencies if item.status == "degraded"]
        return "degraded", reasons[0] if reasons else "dependency_degraded"
    return "ready", "all_dependencies_ready"


def _intent_id_or_none(payload: object) -> UUID | None:
    try:
        value = payload["intent_id"]  # type: ignore[index]
        return UUID(str(value))
    except (KeyError, TypeError, ValueError):
        return None


def _exchange_from_instrument_key(instrument_key: str) -> str:
    parts = instrument_key.split(":")
    return parts[0].strip().lower() if len(parts) >= 3 else "unknown"


def _command_from_intent(
    *,
    intent: ExecutionIntent,
    exchange_name: str,
    environment: str = "testnet",
) -> ExchangeOrderCommand:
    return ExchangeOrderCommand.from_intent(
        intent=intent,
        exchange_name=exchange_name,
        environment=environment,
        client_order_id=f"rh_{intent.idempotency_key_hash[:32]}",
    )


def _connection_guard_reason(
    *,
    intent: ExecutionIntent,
    connection: object,
) -> str | None:
    exchange_connection = connection
    exchange_name = getattr(exchange_connection, "exchange_name")
    market_type = getattr(exchange_connection, "market_type")
    environment = getattr(exchange_connection, "environment")
    readiness = getattr(exchange_connection, "connection_readiness")
    capability = getattr(exchange_connection, "effective_capability")
    if environment != "testnet":
        return "mainnet_hard_block"
    if readiness != "ready_for_trading" or capability != "trading":
        return "exchange_connection_not_ready_for_trading"
    if exchange_name != _exchange_from_instrument_key(intent.instrument_key):
        return "exchange_config_mismatch"
    if market_type != intent.market_type:
        return "exchange_config_mismatch"
    return None
