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
    EmitExecutionNotificationCommand,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import (
    ExecutionNotificationValidationError,
    ExecutionOrderModelRejectedError,
    ExecutionRiskContext,
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
    assert intent.intent.status == "accepted"
    assert intent.intent.status_reason == "risk_gate_accepted"
    assert intent.intent.risk_status == "accepted"
    assert repository.risk_audit_events[0].event_type == "risk_gate_accepted"


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


def test_risk_gate_rejects_missing_context_and_records_audit() -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type="ops_test",
            source_event_ref="ops-missing-risk",
            source_ref_json={"ops_test_id": "missing-risk"},
            strategy_signal_id=None,
            idempotency_key="missing-risk-source-key",
        )
    )

    intent = service.create_intent(
        command=CreateExecutionIntentCommand(
            owner_user_id=_USER_ID,
            source_event_id=source.event.source_event_id,
            idempotency_key="missing-risk-intent-key",
            exchange_connection_id=UUID("00000000-0000-0000-0000-000000010201"),
            market_type="spot",
            instrument_key="binance:spot:BTCUSDT",
            order_type="market",
            side="buy",
            quantity=Decimal("0.01"),
            quote_notional=None,
            limit_price=None,
            advanced_order_flags={},
            risk_context=None,
        )
    )

    assert intent.intent.status == "rejected"
    assert intent.intent.risk_status == "rejected"
    assert intent.intent.risk_reason == "risk_state_unavailable"
    assert repository.risk_audit_events[0].event_type == "risk_gate_rejected"
    assert repository.risk_audit_events[0].metadata_json == {"dispatch": "no-dispatch"}
    assert repository.source_events[0].outcome == "risk_rejected"
    assert repository.notifications[0].event_type == "producer_rejected"
    assert repository.notifications[0].reason == "risk_state_unavailable"


@pytest.mark.parametrize(
    ("source_type", "context_overrides", "reason"),
    (
        ("ops_test", {"exchange_connection_active": False}, "exchange_connection_inactive"),
        ("ops_test", {"exchange_config_verified": False}, "exchange_config_mismatch"),
        ("ops_test", {"account_state_fresh": False}, "account_projection_stale"),
        (
            "strategy_signal",
            {"strategy_variant_compatible": False},
            "strategy_variant_incompatible",
        ),
        ("strategy_signal", {"market_data_state": "missing"}, "market_data_missing"),
        ("strategy_signal", {"market_data_state": "stale"}, "market_data_stale"),
        ("strategy_signal", {"strategy_binding_active": False}, "strategy_binding_missing"),
        (
            "strategy_signal",
            {"strategy_live_profile_ready": False},
            "strategy_live_profile_blocked",
        ),
        ("strategy_signal", {"strategy_run_active": False}, "strategy_run_inactive"),
        (
            "strategy_signal",
            {"position_ownership_active": False},
            "position_ownership_conflict",
        ),
        (
            "strategy_signal",
            {"capital_reservation_sufficient": False},
            "capital_reservation_insufficient",
        ),
        ("manual_request", {"manual_recent_auth": False}, "manual_recent_auth_required"),
        ("ml_agent_decision", {"ml_agent_policy_active": False}, "ml_agent_policy_missing"),
        ("ops_test", {"kill_switch_open": False}, "kill_switch_closed"),
        ("ops_test", {"environment_policy_allows": False}, "mainnet_canary_not_approved"),
    ),
)
def test_risk_gate_rejects_source_aware_safety_cases(
    source_type: str,
    context_overrides: dict[str, object],
    reason: str,
) -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    signal_id = uuid4() if source_type == "strategy_signal" else None
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type=source_type,
            source_event_ref=f"{source_type}-{reason}",
            source_ref_json={f"{source_type}_id": reason},
            strategy_signal_id=signal_id,
            idempotency_key=f"{source_type}-{reason}-source-key",
        )
    )

    intent = service.create_intent(
        command=_intent_command(
            source.event.source_event_id,
            idempotency_key=f"{source_type}-{reason}-intent-key",
            risk_context=_accepted_context(**context_overrides),
        )
    )

    assert intent.intent.status == "rejected"
    assert intent.intent.risk_reason == reason
    assert repository.source_events[0].outcome_reason == reason
    if reason == "kill_switch_closed":
        assert repository.notifications[0].event_type == "producer_kill_switch"


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


def test_emits_redacted_notification_outbox_event_idempotently() -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            owner_user_id=_USER_ID,
            source_type="ops_test",
            source_event_ref="ops-terminal",
            source_ref_json={"ops_test_id": "terminal"},
            strategy_signal_id=None,
            idempotency_key="terminal-source-key",
        )
    )

    first = service.emit_notification(
        command=EmitExecutionNotificationCommand(
            owner_user_id=_USER_ID,
            source_type="ops_test",
            event_type="producer_terminal",
            severity="info",
            reason="cancelled",
            source_event_id=source.event.source_event_id,
            labels={"exchange": "bybit", "status": "cancelled"},
        )
    )
    replay = service.emit_notification(
        command=EmitExecutionNotificationCommand(
            owner_user_id=_USER_ID,
            source_type="ops_test",
            event_type="producer_terminal",
            severity="info",
            reason="cancelled",
            source_event_id=source.event.source_event_id,
            labels={"exchange": "bybit", "status": "cancelled"},
        )
    )

    assert first.duplicate is False
    assert replay.duplicate is True
    assert replay.notification.notification_id == first.notification.notification_id
    assert repository.notifications[0].labels_json == {
        "exchange": "bybit",
        "status": "cancelled",
    }


def test_rejects_sensitive_notification_labels() -> None:
    service = ExecutionIngressService(
        repository=InMemoryExecutionIntentRepository(),
        clock=_Clock(),
    )

    with pytest.raises(ExecutionNotificationValidationError) as error_info:
        service.emit_notification(
            command=EmitExecutionNotificationCommand(
                owner_user_id=_USER_ID,
                source_type="ops_test",
                event_type="producer_unknown",
                severity="critical",
                reason="adapter_unknown_state_reconciliation_required",
                labels={"api_key": "secret"},
            )
        )

    assert error_info.value.reason == "sensitive_notification_label_rejected"


def _intent_command(
    source_event_id: UUID,
    *,
    idempotency_key: str = "intent-key",
    order_type: str = "market",
    advanced_order_flags: dict[str, object] | None = None,
    risk_context: ExecutionRiskContext | None = None,
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
        risk_context=_accepted_context() if risk_context is None else risk_context,
    )


def _accepted_context(**overrides: object) -> ExecutionRiskContext:
    values = {
        "exchange_connection_active": True,
        "secret_custody_ready": True,
        "source_authorized": True,
        "strategy_variant_compatible": True,
        "market_data_state": "ready",
        "strategy_binding_active": True,
        "strategy_live_profile_ready": True,
        "strategy_run_active": True,
        "exchange_config_verified": True,
        "account_state_fresh": True,
        "position_ownership_active": True,
        "capital_reservation_active": True,
        "capital_reservation_sufficient": True,
        "paper_accounting_ready": True,
        "manual_recent_auth": True,
        "ml_agent_policy_active": True,
        "kill_switch_open": True,
        "environment_policy_allows": True,
        "max_order_size_ok": True,
        "daily_limit_ok": True,
    }
    values.update(overrides)
    return ExecutionRiskContext(**values)
