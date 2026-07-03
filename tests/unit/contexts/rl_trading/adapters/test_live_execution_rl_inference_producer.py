from __future__ import annotations

from datetime import UTC, datetime

from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExecutionIngressService
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


def _decision() -> Stage13InferenceDecision:
    return Stage13InferenceDecision(
        decision_id="d" * 64,
        model_version_id="stage08m_a3823cbd01143878_fd7c614b",
        action_id=0,
        action_name="hold",
        confidence=0.75,
        feature_hash="e" * 64,
        feature_contract_hash="f" * 64,
        window_ts_close_utc=datetime(2026, 7, 3, 12, 2, tzinfo=UTC),
    )
