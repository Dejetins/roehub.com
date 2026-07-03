from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    InMemoryPaperAccountingRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    ExecutionIngressService,
)
from trading.contexts.live_execution.domain import ExecutionRiskContext
from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.domain import (
    Stage13DecisionContext,
    Stage13InferenceDecision,
)
from trading.shared_kernel.primitives import UserId


def test_monitor_only_rl_inference_records_no_intent_source_event_only() -> None:
    repository = InMemoryExecutionIntentRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=repository,
    )
    context = _context()
    decision = _decision()

    event = producer.record_monitor_only_decision(context=context, decision=decision)
    replayed = producer.record_monitor_only_decision(context=context, decision=decision)

    assert event.source_event_id == replayed.source_event_id
    assert event.owner_user_id == UserId.from_string(context.owner_user_id)
    assert event.source_type == "ml_agent_decision"
    assert event.outcome == "no_intent"
    assert event.outcome_reason == "monitor_only_no_intent"
    assert event.intent_id is None
    assert event.strategy_signal_id is None
    assert event.source_ref_json["mode"] == "monitor_only"
    assert event.source_ref_json["action"] == "hold"
    assert event.source_ref_json["strategy_id"] == context.strategy_id
    assert len(repository.source_events) == 1
    assert repository.intents == []


def test_paper_rl_inference_creates_no_dispatch_intent_and_accounting_idempotently() -> None:
    execution_repository = InMemoryExecutionIntentRepository()
    paper_repository = InMemoryPaperAccountingRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=execution_repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=execution_repository,
    )
    accounting_service = CapitalReservationPaperAccountingService(
        repository=paper_repository,
        account_projection_repository=None,
        clock=SystemLiveExecutionClock(),
    )
    context = _context()
    decision = _decision(action_id=1, action_name="open_long")

    first = producer.record_paper_decision(
        context=context,
        decision=decision,
        risk_context=_paper_risk_context(),
        paper_accounting_service=accounting_service,
        quote_notional=Decimal("50"),
        reference_price=Decimal("10000"),
    )
    replay = producer.record_paper_decision(
        context=context,
        decision=decision,
        risk_context=_paper_risk_context(),
        paper_accounting_service=accounting_service,
        quote_notional=Decimal("50"),
        reference_price=Decimal("10000"),
    )

    assert first.intent is not None
    assert first.accounting is not None
    assert replay.intent is not None
    assert replay.accounting is not None
    assert replay.duplicate is True
    assert replay.event.source_event_id == first.event.source_event_id
    assert replay.intent.intent_id == first.intent.intent_id
    assert replay.accounting.accounting_id == first.accounting.accounting_id
    assert first.event.source_type == "ml_agent_decision"
    assert first.event.outcome == "risk_rejected"
    assert first.event.outcome_reason == "paper_no_exchange_submit"
    assert first.event.intent_id == first.intent.intent_id
    assert first.event.source_ref_json["mode"] == "paper"
    assert first.event.source_ref_json["action"] == "open_long"
    assert first.intent.status == "rejected"
    assert first.intent.risk_reason == "paper_no_exchange_submit"
    assert first.intent.dispatch_stream_name is None
    assert first.accounting.position_quantity == Decimal("0.00500000")
    assert first.accounting.equity == Decimal("49.95000000")
    assert first.accounting.fee_total == Decimal("0.05000000")
    assert len(execution_repository.source_events) == 1
    assert len(execution_repository.intents) == 1
    assert len(execution_repository.risk_audit_events) == 1
    assert len(paper_repository.orders) == 1
    assert paper_repository.orders[0].source_event_id == first.event.source_event_id
    assert paper_repository.orders[0].reason == "paper_market_fill_from_ml_agent_decision"
    assert len(paper_repository.fills) == 1
    assert len(paper_repository.accounting) == 1


def test_paper_rl_hold_stays_source_event_only() -> None:
    execution_repository = InMemoryExecutionIntentRepository()
    paper_repository = InMemoryPaperAccountingRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=execution_repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=execution_repository,
    )
    accounting_service = CapitalReservationPaperAccountingService(
        repository=paper_repository,
        account_projection_repository=None,
        clock=SystemLiveExecutionClock(),
    )

    result = producer.record_paper_decision(
        context=_context(),
        decision=_decision(),
        risk_context=_paper_risk_context(),
        paper_accounting_service=accounting_service,
        quote_notional=Decimal("50"),
        reference_price=Decimal("10000"),
    )

    assert result.intent is None
    assert result.accounting is None
    assert result.event.outcome == "no_intent"
    assert result.event.outcome_reason == "paper_hold_no_intent"
    assert len(execution_repository.source_events) == 1
    assert execution_repository.intents == []
    assert paper_repository.orders == []


def test_testnet_rl_inference_creates_accepted_intent_idempotently() -> None:
    execution_repository = InMemoryExecutionIntentRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=execution_repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=execution_repository,
    )
    connection_id = UUID("00000000-0000-0000-0000-000000014001")
    context = _context()
    decision = _decision(action_id=1, action_name="open_long")

    first = producer.record_testnet_decision(
        context=context,
        decision=decision,
        risk_context=_testnet_risk_context(),
        exchange_connection_id=connection_id,
        quote_notional=Decimal("50"),
    )
    replay = producer.record_testnet_decision(
        context=context,
        decision=decision,
        risk_context=_testnet_risk_context(),
        exchange_connection_id=connection_id,
        quote_notional=Decimal("50"),
    )

    assert first.intent is not None
    assert replay.intent is not None
    assert replay.duplicate is True
    assert replay.event.source_event_id == first.event.source_event_id
    assert replay.intent.intent_id == first.intent.intent_id
    assert first.event.source_type == "ml_agent_decision"
    assert first.event.outcome == "intent_created"
    assert first.event.outcome_reason == "risk_gate_accepted"
    assert first.event.source_ref_json["mode"] == "testnet"
    assert first.event.source_ref_json["action"] == "open_long"
    assert first.intent.status == "accepted"
    assert first.intent.risk_reason == "risk_gate_accepted"
    assert first.intent.side == "buy"
    assert first.intent.quote_notional == Decimal("50")
    assert len(execution_repository.source_events) == 1
    assert len(execution_repository.intents) == 1


def test_testnet_rl_spot_short_stays_source_event_only() -> None:
    execution_repository = InMemoryExecutionIntentRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=execution_repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=execution_repository,
    )
    context = Stage13DecisionContext(
        owner_user_id="00000000-0000-0000-0000-000000013001",
        strategy_id="00000000-0000-0000-0000-000000013101",
        strategy_run_id="00000000-0000-0000-0000-000000013201",
        exchange="bybit",
        market_type="spot",
        symbol="BTCUSDT",
        instrument_key="bybit:spot:BTCUSDT",
    )

    result = producer.record_testnet_decision(
        context=context,
        decision=_decision(action_id=2, action_name="open_short"),
        risk_context=_testnet_risk_context(),
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000014002"),
        quote_notional=Decimal("50"),
    )

    assert result.intent is None
    assert result.event.outcome == "no_intent"
    assert result.event.outcome_reason == "testnet_spot_short_not_supported"
    assert execution_repository.intents == []


def _context() -> Stage13DecisionContext:
    return Stage13DecisionContext(
        owner_user_id="00000000-0000-0000-0000-000000013001",
        strategy_id="00000000-0000-0000-0000-000000013101",
        strategy_run_id="00000000-0000-0000-0000-000000013201",
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )


def _decision(*, action_id: int = 0, action_name: str = "hold") -> Stage13InferenceDecision:
    return Stage13InferenceDecision(
        decision_id="d" * 64,
        model_version_id="stage08m_a3823cbd01143878_fd7c614b",
        action_id=action_id,
        action_name=action_name,
        confidence=0.75,
        feature_hash="e" * 64,
        feature_contract_hash="f" * 64,
        window_ts_close_utc=datetime(2026, 7, 3, 12, 2, tzinfo=UTC),
    )


def _paper_risk_context() -> ExecutionRiskContext:
    return ExecutionRiskContext(
        exchange_connection_active=True,
        secret_custody_ready=True,
        source_authorized=True,
        strategy_live_profile_ready=True,
        strategy_run_active=True,
        market_data_state="ready",
        position_ownership_active=True,
        capital_reservation_active=True,
        capital_reservation_sufficient=True,
        paper_accounting_ready=True,
        paper_no_exchange_submit=True,
        ml_agent_policy_active=True,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )


def _testnet_risk_context() -> ExecutionRiskContext:
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
        ml_agent_policy_active=True,
        kill_switch_open=True,
        environment_policy_allows=True,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )
