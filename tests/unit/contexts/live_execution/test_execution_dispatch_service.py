from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    ExecutionDispatchConfig,
    ExecutionDispatchService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPoisonMessageError,
    ExecutionDispatchPublishResult,
    ExecutionDispatchUnavailableError,
)
from trading.contexts.live_execution.domain import ExecutionIntent, ExecutionRiskContext
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000012000"))
_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000012001")
_NOW = datetime(2026, 5, 31, 16, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = _NOW + timedelta(seconds=self._index)
        self._index += 1
        return value


class _Transport:
    def __init__(
        self,
        *,
        stream_length: int = 0,
        request_error: Exception | None = None,
    ) -> None:
        self.stream_length = stream_length
        self.request_error = request_error
        self.requests: list[ExecutionIntent] = []
        self.retries: list[tuple[ExecutionIntent, str]] = []
        self.dlq: list[tuple[ExecutionIntent, str]] = []
        self.groups_ensured = 0
        self.acked: list[tuple[str, str]] = []

    def ensure_request_group(self) -> None:
        self.groups_ensured += 1

    def request_stream_length(self) -> int:
        return self.stream_length

    def publish_request(
        self, *, intent: ExecutionIntent, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        if self.request_error is not None:
            raise self.request_error
        self.requests.append(intent)
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.v1",
            message_id=f"1-{attempt_count}",
        )

    def publish_retry(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        self.retries.append((intent, reason))
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.retry.v1",
            message_id=f"2-{attempt_count}",
        )

    def publish_dlq(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        self.dlq.append((intent, reason))
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.dlq.v1",
            message_id=f"3-{attempt_count}",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        self.acked.append((stream_name, message_id))


def test_dispatches_only_accepted_intent_and_marks_dispatched_once() -> None:
    repository = InMemoryExecutionIntentRepository()
    intent = _accepted_intent(repository=repository)
    transport = _Transport()
    metrics: list[tuple[str, str]] = []
    service = ExecutionDispatchService(
        repository=repository,
        transport=transport,
        clock=_Clock(),
        on_dispatch=lambda result, reason: metrics.append((result, reason)),
    )

    result = service.dispatch_intent(intent=intent)
    replay = service.dispatch_intent(intent=result.intent)

    assert result.result == "dispatched"
    assert result.intent.status == "dispatched"
    assert result.intent.dispatch_attempt_count == 1
    assert result.intent.dispatch_stream_name == "execution.requests.v1"
    assert result.intent.dispatch_redis_message_id == "1-1"
    assert replay.result == "duplicate"
    assert len(transport.requests) == 1
    assert metrics == [("dispatched", "redis_xadd_ok")]


def test_rejected_intent_is_not_dispatched() -> None:
    repository = InMemoryExecutionIntentRepository()
    intent = _rejected_intent(repository=repository)
    transport = _Transport()
    service = ExecutionDispatchService(
        repository=repository,
        transport=transport,
        clock=_Clock(),
    )

    result = service.dispatch_intent(intent=intent)

    assert result.result == "skipped"
    assert result.reason == "risk_not_accepted"
    assert transport.requests == []
    assert result.intent.status == "rejected"


def test_redis_outage_marks_retry_without_dlq_or_order_side_effect() -> None:
    repository = InMemoryExecutionIntentRepository()
    intent = _accepted_intent(repository=repository)
    transport = _Transport(
        request_error=ExecutionDispatchUnavailableError(reason="ConnectionError")
    )
    retry_reasons: list[str] = []
    service = ExecutionDispatchService(
        repository=repository,
        transport=transport,
        clock=_Clock(),
        on_retry=retry_reasons.append,
    )

    result = service.dispatch_intent(intent=intent)

    assert result.result == "retry"
    assert result.intent.status == "retry"
    assert result.intent.dispatch_last_error == "ConnectionError"
    assert transport.requests == []
    assert transport.dlq == []
    assert retry_reasons == ["ConnectionError"]


def test_backpressure_marks_retry_and_writes_retry_stream_marker() -> None:
    repository = InMemoryExecutionIntentRepository()
    intent = _accepted_intent(repository=repository)
    transport = _Transport(stream_length=10)
    backpressure_reasons: list[str] = []
    service = ExecutionDispatchService(
        repository=repository,
        transport=transport,
        clock=_Clock(),
        config=ExecutionDispatchConfig(backpressure_max_stream_length=10),
        on_backpressure=backpressure_reasons.append,
    )

    result = service.dispatch_intent(intent=intent)

    assert result.result == "retry"
    assert result.reason == "dispatch_backpressure"
    assert result.intent.status == "retry"
    assert transport.requests == []
    assert [(item.intent_id, reason) for item, reason in transport.retries] == [
        (intent.intent_id, "dispatch_backpressure")
    ]
    assert backpressure_reasons == ["dispatch_backpressure"]


def test_poison_payload_marks_quarantined_and_writes_dlq_marker() -> None:
    repository = InMemoryExecutionIntentRepository()
    intent = _accepted_intent(repository=repository)
    transport = _Transport(
        request_error=ExecutionDispatchPoisonMessageError(reason="dispatch_payload_invalid")
    )
    dlq_reasons: list[str] = []
    service = ExecutionDispatchService(
        repository=repository,
        transport=transport,
        clock=_Clock(),
        on_dlq=dlq_reasons.append,
    )

    result = service.dispatch_intent(intent=intent)

    assert result.result == "dlq"
    assert result.intent.status == "quarantined"
    assert result.intent.dispatch_last_error == "dispatch_payload_invalid"
    assert [(item.intent_id, reason) for item, reason in transport.dlq] == [
        (intent.intent_id, "dispatch_payload_invalid")
    ]
    assert dlq_reasons == ["dispatch_payload_invalid"]


def test_ack_after_durable_state_change_is_exposed_on_transport_port() -> None:
    transport = _Transport()

    transport.ack_after_durable_state_change(
        stream_name="execution.requests.v1",
        message_id="1-0",
    )

    assert transport.acked == [("execution.requests.v1", "1-0")]


def _accepted_intent(*, repository: InMemoryExecutionIntentRepository) -> ExecutionIntent:
    return _create_intent(repository=repository, risk_context=_accepted_context())


def _rejected_intent(*, repository: InMemoryExecutionIntentRepository) -> ExecutionIntent:
    return _create_intent(repository=repository, risk_context=None)


def _create_intent(
    *, repository: InMemoryExecutionIntentRepository, risk_context: ExecutionRiskContext | None
) -> ExecutionIntent:
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            source_type="ops_test",
            source_event_ref=str(uuid4()),
            source_ref_json={"ops_test_id": "stage12"},
            strategy_signal_id=None,
            idempotency_key=str(uuid4()),
        )
    )
    return service.create_intent(
        command=CreateExecutionIntentCommand(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            source_event_id=source.event.source_event_id,
            idempotency_key=str(uuid4()),
            exchange_connection_id=UUID("00000000-0000-0000-0000-000000012101"),
            market_type="spot",
            instrument_key="binance:spot:BTCUSDT",
            order_type="market",
            side="buy",
            quantity=Decimal("0.01"),
            quote_notional=None,
            limit_price=None,
            advanced_order_flags={},
            risk_context=risk_context,
        )
    ).intent


def _accepted_context() -> ExecutionRiskContext:
    return ExecutionRiskContext(
        organization_ownership_verified=True,
        account_ownership_verified=True,
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
        manual_recent_auth=True,
        ml_agent_policy_active=True,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )
