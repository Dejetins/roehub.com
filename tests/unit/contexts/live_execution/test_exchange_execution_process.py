from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExecutionDispatchPublishResult,
)
from trading.contexts.live_execution.domain import ExecutionIntent
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000013001")
_NOW = datetime(2026, 5, 31, 18, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = _NOW + timedelta(seconds=self._index)
        self._index += 1
        return value


class _Consumer:
    def __init__(self, messages: tuple[ExchangeExecutionRedisMessage, ...]) -> None:
        self.messages = messages
        self.groups_ensured = 0
        self.dlq: list[tuple[str, str]] = []
        self.acked: list[tuple[str, str]] = []

    def ensure_request_group(self) -> None:
        self.groups_ensured += 1

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        return ExchangeExecutionRedisHealth(
            request_stream_length=len(self.messages),
            retry_stream_length=0,
            dlq_stream_length=len(self.dlq),
            pending_count=0,
            clock_drift_ms=8.0,
        )

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = block_ms
        return self.messages[:count]

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult:
        self.dlq.append((message.message_id, reason))
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.dlq.v1",
            message_id=f"dlq-{len(self.dlq)}",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        self.acked.append((stream_name, message_id))


def test_readiness_records_heartbeat_and_reports_adapter_disabled_degraded() -> None:
    repository = InMemoryExchangeExecutionProcessRepository()
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=repository,
        intent_repository=InMemoryExecutionIntentRepository(),
        consumer=_Consumer(messages=()),
        clock=_Clock(),
    )

    snapshot = service.readiness()

    assert snapshot.status == "degraded"
    assert snapshot.status_reason == "adapter_disabled_stage13"
    assert repository.heartbeats["exchange-execution"].status == "degraded"
    assert {item.name for item in snapshot.dependencies} >= {
        "postgres",
        "redis",
        "backpressure",
        "dlq",
        "clock_drift",
        "adapter",
        "rate_limit",
    }


def test_valid_dispatched_intent_is_observed_but_not_acked_with_adapter_disabled() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    message = _message(
        payload={
            "intent_id": str(intent.intent_id),
            "owner_user_id": str(intent.owner_user_id),
        }
    )
    consumer = _Consumer(messages=(message,))
    observations: list[tuple[str, str]] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=process_repository,
        intent_repository=intent_repository,
        consumer=consumer,
        clock=_Clock(),
        on_observation=lambda status, reason: observations.append((status, reason)),
    )

    result = service.run_once()

    assert result.read_count == 1
    assert result.observed_count == 1
    assert result.quarantined_count == 0
    assert result.acked_count == 0
    assert consumer.acked == []
    assert process_repository.observations[0].status == "adapter_disabled"
    assert process_repository.observations[0].status_reason == "adapter_disabled_stage13"
    assert observations == [("adapter_disabled", "adapter_disabled_stage13")]


def test_invalid_or_non_dispatchable_message_is_quarantined_to_dlq_and_acked() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(uuid4()),
                    "owner_user_id": str(_USER_ID),
                }
            ),
        )
    )
    dlq_reasons: list[str] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=process_repository,
        intent_repository=InMemoryExecutionIntentRepository(),
        consumer=consumer,
        clock=_Clock(),
        on_dlq=dlq_reasons.append,
    )

    result = service.run_once()

    assert result.quarantined_count == 1
    assert result.acked_count == 1
    assert process_repository.observations[0].status == "quarantined"
    assert process_repository.observations[0].status_reason == "intent_not_found"
    assert consumer.dlq == [("1-0", "intent_not_found")]
    assert consumer.acked == [("execution.requests.v1", "1-0")]
    assert dlq_reasons == ["intent_not_found"]


def _message(*, payload: dict[str, str]) -> ExchangeExecutionRedisMessage:
    return ExchangeExecutionRedisMessage(
        stream_name="execution.requests.v1",
        message_id="1-0",
        payload=payload,
    )


def _intent(*, status: str, risk_status: str) -> ExecutionIntent:
    return ExecutionIntent(
        intent_id=uuid4(),
        source_event_id=uuid4(),
        owner_user_id=_USER_ID,
        source_type="ops_test",
        strategy_signal_id=None,
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000013101"),
        market_type="spot",
        instrument_key="bybit:spot:BTCUSDT",
        side="buy",
        order_type="market",
        quantity=Decimal("0.01"),
        quote_notional=None,
        limit_price=None,
        status=status,  # type: ignore[arg-type]
        status_reason="redis_xadd_ok",
        risk_status=risk_status,
        risk_reason="risk_gate_accepted",
        idempotency_key_hash="b" * 64,
        created_at=_NOW,
        dispatch_attempt_count=1,
        dispatch_stream_name="execution.requests.v1",
        dispatch_redis_message_id="1-0",
        dispatch_updated_at=_NOW,
    )
