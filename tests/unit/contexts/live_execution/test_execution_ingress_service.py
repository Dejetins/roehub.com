from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import (
    ExecutionOrderModelRejectedError,
    ExecutionSourceValidationError,
)
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000010001")
_NOW = datetime(2026, 5, 31, 13, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = _NOW + timedelta(seconds=self._index)
        self._index += 1
        return value


def test_records_source_event_and_intent_for_strategy_signal_idempotently() -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    signal_id = UUID("00000000-0000-0000-0000-000000010101")

    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type="strategy_signal",
            source_event_ref="strategy-signal-1",
            source_ref_json={"strategy_id": str(uuid4()), "signal_id": str(signal_id)},
            strategy_signal_id=signal_id,
            idempotency_key="source-key",
        )
    )
    replay = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type="strategy_signal",
            source_event_ref="strategy-signal-1",
            source_ref_json={"strategy_id": str(uuid4()), "signal_id": str(signal_id)},
            strategy_signal_id=signal_id,
            idempotency_key="source-key",
        )
    )
    intent = service.create_intent(command=_intent_command(source.event.source_event_id))
    intent_replay = service.create_intent(command=_intent_command(source.event.source_event_id))

    assert source.duplicate is False
    assert replay.duplicate is True
    assert replay.event.source_event_id == source.event.source_event_id
    assert intent.duplicate is False
    assert intent_replay.duplicate is True
    assert intent_replay.intent.intent_id == intent.intent.intent_id
    assert len(repository.source_events) == 1
    assert len(repository.intents) == 1
    assert repository.source_events[0].outcome == "intent_created"
    assert repository.source_events[0].intent_id == intent.intent.intent_id
    assert intent.intent.status == "recorded"
    assert intent.intent.status_reason == "stage10_recorded_no_dispatch"
    assert intent.intent.risk_status == "not_evaluated"


@pytest.mark.parametrize(
    "source_type",
    ("strategy_signal", "manual_request", "ml_agent_decision", "ops_test"),
)
def test_supported_source_types_share_the_same_ingress(source_type: str) -> None:
    service = ExecutionIngressService(
        repository=InMemoryExecutionIntentRepository(),
        clock=_Clock(),
    )
    signal_id = uuid4() if source_type == "strategy_signal" else None

    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type=source_type,
            source_event_ref=f"{source_type}-ref",
            source_ref_json={f"{source_type}_id": "producer-ref"},
            strategy_signal_id=signal_id,
            idempotency_key=f"{source_type}-source-key",
        )
    )
    intent = service.create_intent(
        command=_intent_command(
            source.event.source_event_id,
            idempotency_key=f"{source_type}-intent-key",
        )
    )

    assert source.event.source_type == source_type
    assert intent.intent.source_type == source_type
    assert intent.event.outcome == "intent_created"


def test_rejects_unsupported_order_model_and_links_source_event_outcome() -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type="manual_request",
            source_event_ref="manual-1",
            source_ref_json={"manual_request_id": "manual-1"},
            strategy_signal_id=None,
            idempotency_key="manual-source-key",
        )
    )

    with pytest.raises(ExecutionOrderModelRejectedError) as error_info:
        service.create_intent(
            command=_intent_command(
                source.event.source_event_id,
                order_type="oco",
                advanced_order_flags={"oco": {"kind": "bracket"}},
            )
        )

    assert error_info.value.reason == "oco_not_supported"
    assert repository.source_events[0].outcome == "order_model_rejected"
    assert repository.source_events[0].outcome_reason == "oco_not_supported"
    assert repository.intents == []


def test_rejects_invalid_source_policy() -> None:
    service = ExecutionIngressService(
        repository=InMemoryExecutionIntentRepository(),
        clock=_Clock(),
    )

    with pytest.raises(ExecutionSourceValidationError) as error_info:
        service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=_USER_ID,
                source_type="strategy_signal",
                source_event_ref="missing-signal-id",
                source_ref_json={"strategy_id": str(uuid4())},
                strategy_signal_id=None,
                idempotency_key="invalid-source-key",
            )
        )

    assert error_info.value.reason == "strategy_signal_id_required"


def _intent_command(
    source_event_id: UUID,
    *,
    idempotency_key: str = "intent-key",
    order_type: str = "market",
    advanced_order_flags: dict[str, object] | None = None,
) -> CreateExecutionIntentCommand:
    return CreateExecutionIntentCommand(
        owner_user_id=_USER_ID,
        source_event_id=source_event_id,
        idempotency_key=idempotency_key,
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000010201"),
        market_type="spot",
        instrument_key="binance:spot:BTCUSDT",
        order_type=order_type,
        side="buy",
        quantity=Decimal("0.01"),
        quote_notional=None,
        limit_price=None,
        advanced_order_flags=advanced_order_flags or {},
    )
