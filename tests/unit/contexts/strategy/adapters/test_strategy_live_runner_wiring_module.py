from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from apps.worker.strategy_live_runner.wiring.modules.strategy_live_runner import (
    GuardedStrategyExecutionProducer,
    StrategyLiveRunnerMetrics,
    _require_non_empty_env_value,
)
from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.strategy.adapters.outbound.acl.live_execution_producer import (
    LiveExecutionStrategySignalProducer,
)
from trading.contexts.strategy.adapters.outbound.config import StrategyProducerRuntimeConfig
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import UserId


class _ExecutionProducerSpy:
    def __init__(self) -> None:
        self.signals: list[StrategySignal] = []

    def record_signal(self, *, signal: StrategySignal) -> None:
        self.signals.append(signal)


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 6, 17, 10, 2, tzinfo=timezone.utc)


def test_require_non_empty_env_value_returns_secret() -> None:
    """
    Ensure helper returns configured non-empty environment variable value.
    """
    value = _require_non_empty_env_value(
        environ={"TELEGRAM_BOT_TOKEN": "token-value"},
        key="TELEGRAM_BOT_TOKEN",
        setting_name="strategy_live_runner.telegram.bot_token_env",
    )

    assert value == "token-value"


def test_require_non_empty_env_value_rejects_missing_or_blank_value() -> None:
    """
    Ensure helper fails fast when required environment variable is missing or blank.
    """
    with pytest.raises(ValueError):
        _require_non_empty_env_value(
            environ={},
            key="TELEGRAM_BOT_TOKEN",
            setting_name="strategy_live_runner.telegram.bot_token_env",
        )

    with pytest.raises(ValueError):
        _require_non_empty_env_value(
            environ={"TELEGRAM_BOT_TOKEN": "   "},
            key="TELEGRAM_BOT_TOKEN",
            setting_name="strategy_live_runner.telegram.bot_token_env",
        )


def test_guarded_strategy_execution_producer_blocks_disabled_switch() -> None:
    """
    Ensure admin switch disabled blocks source-event production.
    """
    delegate = _ExecutionProducerSpy()
    blocked_reasons: list[str] = []
    producer = GuardedStrategyExecutionProducer(
        delegate=delegate,
        producer_config=StrategyProducerRuntimeConfig(
            enabled=False,
            allow_all=False,
            allowed_modes=("paper", "testnet"),
            allowed_user_ids=("00000000-0000-0000-0000-000000000901",),
            allowed_strategy_ids=(),
        ),
        on_source_event_blocked=lambda *, reason: blocked_reasons.append(reason),
    )

    producer.record_signal(signal=_signal(mode="paper"))

    assert delegate.signals == []
    assert blocked_reasons == ["producer_disabled"]


def test_guarded_strategy_execution_producer_blocks_missing_allowlist() -> None:
    """
    Ensure enabled producer still blocks when neither user nor strategy is allowlisted.
    """
    delegate = _ExecutionProducerSpy()
    producer = GuardedStrategyExecutionProducer(
        delegate=delegate,
        producer_config=StrategyProducerRuntimeConfig(
            enabled=True,
            allow_all=False,
            allowed_modes=("paper", "testnet"),
            allowed_user_ids=(),
            allowed_strategy_ids=(),
        ),
    )

    decision = producer.evaluate(signal=_signal(mode="testnet"))
    producer.record_signal(signal=_signal(mode="testnet"))

    assert decision.allowed is False
    assert decision.reason == "producer_allowlist_missing"
    assert delegate.signals == []


def test_guarded_strategy_execution_producer_allows_paper_user_allowlist() -> None:
    """
    Ensure user allowlist permits paper source-event production through existing port.
    """
    delegate = _ExecutionProducerSpy()
    created: list[StrategySignal] = []
    signal = _signal(mode="paper")
    producer = GuardedStrategyExecutionProducer(
        delegate=delegate,
        producer_config=StrategyProducerRuntimeConfig(
            enabled=True,
            allow_all=False,
            allowed_modes=("paper", "testnet"),
            allowed_user_ids=(str(signal.owner_user_id),),
            allowed_strategy_ids=(),
        ),
        on_source_event_created=created.append,
    )

    producer.record_signal(signal=signal)

    assert delegate.signals == [signal]
    assert created == [signal]


def test_live_execution_producer_records_paper_no_dispatch_intent() -> None:
    repository = InMemoryExecutionIntentRepository()
    signal = _signal(
        mode="paper",
        expected_order_json={
            "schema": "strategy_signal_expected_order_v1",
            "mode": "paper",
            "quote_notional": "50",
            "paper_no_exchange_submit": True,
        },
    )
    producer = LiveExecutionStrategySignalProducer(
        ingress_service=ExecutionIngressService(repository=repository, clock=_Clock()),
        repository=repository,
    )

    producer.record_signal(signal=signal)

    assert len(repository.source_events) == 1
    assert len(repository.intents) == 1
    assert repository.source_events[0].outcome == "risk_rejected"
    assert repository.source_events[0].outcome_reason == "paper_no_exchange_submit"
    assert repository.source_events[0].intent_id == repository.intents[0].intent_id
    assert repository.intents[0].status == "rejected"
    assert repository.intents[0].risk_reason == "paper_no_exchange_submit"
    assert repository.intents[0].dispatch_stream_name is None


def test_guarded_strategy_execution_producer_blocks_live_mode_even_when_allow_all() -> None:
    """
    Ensure Stage 06 producer has no live/mainnet mode path.
    """
    delegate = _ExecutionProducerSpy()
    producer = GuardedStrategyExecutionProducer(
        delegate=delegate,
        producer_config=StrategyProducerRuntimeConfig(
            enabled=True,
            allow_all=True,
            allowed_modes=("paper", "testnet"),
            allowed_user_ids=(),
            allowed_strategy_ids=(),
        ),
    )

    decision = producer.evaluate(signal=_signal(mode="live"))
    producer.record_signal(signal=_signal(mode="live"))

    assert decision.allowed is False
    assert decision.reason == "producer_mode_not_allowed"
    assert delegate.signals == []


def test_strategy_producer_metrics_do_not_use_user_or_strategy_labels() -> None:
    """
    Ensure producer metrics expose bounded labels only.
    """
    registry = CollectorRegistry()
    metrics = StrategyLiveRunnerMetrics(registry=registry)
    signal = _signal(mode="paper")

    metrics.observe_source_event_created(signal)
    metrics.observe_source_event_blocked(reason="producer_allowlist_missing")
    payload = generate_latest(registry).decode("utf-8")

    assert "strategy_producer_source_events_total" in payload
    assert 'mode="paper",outcome="signal"' in payload
    assert 'reason="producer_allowlist_missing"' in payload
    assert str(signal.owner_user_id) not in payload
    assert str(signal.strategy_id) not in payload


def _signal(
    *, mode: str, expected_order_json: dict[str, object] | None = None
) -> StrategySignal:
    bar_ts_open = datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc)
    return StrategySignal(
        signal_id=UUID("00000000-0000-0000-0000-000000000903"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000901"),
        strategy_id=UUID("00000000-0000-0000-0000-000000000902"),
        strategy_run_id=UUID("00000000-0000-0000-0000-000000000904"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000000905"),
        mode=mode,  # type: ignore[arg-type]
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        timeframe="1m",
        bar_ts_open=bar_ts_open,
        bar_ts_close=bar_ts_open + timedelta(minutes=1),
        signal_action="open",
        side="buy",
        outcome="signal",
        reason_code="ma_fast_crossed_above_slow_paper_no_exchange_submit",
        reference_price=Decimal("50000"),
        confidence=Decimal("1"),
        source_message_id="m-1",
        evaluator_version="test",
        expected_order_json=expected_order_json or {},
        created_at=bar_ts_open + timedelta(minutes=1),
    )
