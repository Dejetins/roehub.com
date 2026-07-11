from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from prometheus_client import CollectorRegistry

from apps.worker.rl_trading_inference.wiring.modules.rl_trading_inference import (
    RlTradingInferenceInstrumentConfig,
    RlTradingInferenceMetrics,
    RlTradingInferenceOperatorContextConfig,
    RlTradingRedisCandleMessage,
)
from apps.worker.rl_trading_inference.wiring.modules.stage08k_monitor_worker import (
    Stage08kMonitorWorker,
)
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.rl_trading.adapters.outbound import LiveExecutionRlInferenceProducer
from trading.contexts.rl_trading.adapters.outbound.persistence.file_monitor_state import (
    FileStage08kMonitorStateStore,
)
from trading.contexts.rl_trading.domain import (
    RlFeatureCandle,
    Stage13FeatureWindow,
    Stage13MonitorOnlyInferenceError,
)
from trading.contexts.rl_trading.domain.stage08k_monitor_policy import (
    Stage08kArticleSignal,
    Stage08kMonitorDecision,
    Stage08kMonitorPolicyConfig,
)


def test_worker_records_open_and_one_minute_virtual_close_without_intents(
    tmp_path: Path,
) -> None:
    instrument = RlTradingInferenceInstrumentConfig(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    first_close = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    windows = {
        "1-0": _window(close_time=first_close, close_price=100.0),
        "2-0": _window(
            close_time=first_close + timedelta(minutes=1),
            close_price=101.0,
        ),
    }
    stream = _FakeStream()
    repository = InMemoryExecutionIntentRepository()
    producer = LiveExecutionRlInferenceProducer(
        ingress_service=ExecutionIngressService(
            repository=repository,
            clock=SystemLiveExecutionClock(),
        ),
        repository=repository,
    )
    store = FileStage08kMonitorStateStore(path=(tmp_path / "state.json").resolve())
    metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
    worker = Stage08kMonitorWorker(
        instruments=(instrument,),
        operator_context=RlTradingInferenceOperatorContextConfig(
            owner_user_id="00000000-0000-0000-0000-000000013001",
            strategy_id="00000000-0000-0000-0000-000000013101",
            strategy_run_id="00000000-0000-0000-0000-000000013201",
        ),
        stream=cast(Any, stream),
        window_reader=cast(Any, _FakeWindowReader(windows)),
        policy=cast(Any, _FakePolicy()),
        producer=producer,
        state_store=store,
        metrics=metrics,
    )

    worker.process(
        message=RlTradingRedisCandleMessage(
            instrument_key=instrument.instrument_key,
            message_id="1-0",
        )
    )
    assert store.get(instrument_key=instrument.instrument_key) is not None
    worker.process(
        message=RlTradingRedisCandleMessage(
            instrument_key=instrument.instrument_key,
            message_id="2-0",
        )
    )

    assert store.get(instrument_key=instrument.instrument_key) is None
    assert stream.acked == ["1-0", "2-0"]
    assert len(repository.source_events) == 2
    assert len(repository.intents) == 0
    assert [event.outcome for event in repository.source_events] == ["no_intent", "no_intent"]
    assert [event.source_ref_json["action"] for event in repository.source_events] == [
        "open_long",
        "close",
    ]
    rendered = metrics.render_latest().decode("utf-8")
    assert "rl_trading_inference_virtual_exits_total" in rendered
    assert "rl_trading_inference_virtual_realized_pnl_quote 42.5" in rendered


def test_worker_replay_after_ack_failure_is_idempotent(tmp_path: Path) -> None:
    instrument = RlTradingInferenceInstrumentConfig(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    close_time = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    stream = _FakeStream(fail_first_ack=True)
    repository = InMemoryExecutionIntentRepository()
    policy = _FakePolicy()
    store = FileStage08kMonitorStateStore(path=(tmp_path / "state.json").resolve())
    worker = Stage08kMonitorWorker(
        instruments=(instrument,),
        operator_context=RlTradingInferenceOperatorContextConfig(
            owner_user_id="00000000-0000-0000-0000-000000013001",
            strategy_id="00000000-0000-0000-0000-000000013101",
            strategy_run_id="00000000-0000-0000-0000-000000013201",
        ),
        stream=cast(Any, stream),
        window_reader=cast(
            Any,
            _FakeWindowReader({"1-0": _window(close_time=close_time, close_price=100.0)}),
        ),
        policy=cast(Any, policy),
        producer=LiveExecutionRlInferenceProducer(
            ingress_service=ExecutionIngressService(
                repository=repository,
                clock=SystemLiveExecutionClock(),
            ),
            repository=repository,
        ),
        state_store=store,
        metrics=RlTradingInferenceMetrics(registry=CollectorRegistry()),
    )
    message = RlTradingRedisCandleMessage(
        instrument_key=instrument.instrument_key,
        message_id="1-0",
    )

    with pytest.raises(RuntimeError, match="simulated ack failure"):
        worker.process(message=message)
    worker.process(message=message)

    assert stream.acked == ["1-0"]
    assert len(repository.source_events) == 1
    assert repository.intents == []
    assert policy.calls == 1
    assert store.get(instrument_key=instrument.instrument_key) is not None
    assert store.last_processed_close_utc(instrument_key=instrument.instrument_key) == close_time


def test_worker_retries_early_stream_message_without_accepting_unclosed_candle(
    tmp_path: Path,
) -> None:
    instrument = RlTradingInferenceInstrumentConfig(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    close_time = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    stream = _FakeStream()
    repository = InMemoryExecutionIntentRepository()
    reader = _FakeCloseBoundaryReader(
        window=_window(close_time=close_time, close_price=100.0),
        failures=2,
    )
    sleeps: list[float] = []
    metrics = RlTradingInferenceMetrics(registry=CollectorRegistry())
    worker = Stage08kMonitorWorker(
        instruments=(instrument,),
        operator_context=RlTradingInferenceOperatorContextConfig(
            owner_user_id="00000000-0000-0000-0000-000000013001",
            strategy_id="00000000-0000-0000-0000-000000013101",
            strategy_run_id="00000000-0000-0000-0000-000000013201",
        ),
        stream=cast(Any, stream),
        window_reader=cast(Any, reader),
        policy=cast(Any, _FakePolicy()),
        producer=LiveExecutionRlInferenceProducer(
            ingress_service=ExecutionIngressService(
                repository=repository,
                clock=SystemLiveExecutionClock(),
            ),
            repository=repository,
        ),
        state_store=FileStage08kMonitorStateStore(path=(tmp_path / "state.json").resolve()),
        metrics=metrics,
        close_boundary_sleep=sleeps.append,
    )
    message = RlTradingRedisCandleMessage(
        instrument_key=instrument.instrument_key,
        message_id="1-0",
    )

    worker.process_with_close_boundary_retry(message=message)

    assert reader.calls == 3
    assert sleeps == [0.05, 0.05]
    assert stream.acked == ["1-0"]
    assert len(repository.source_events) == 1
    rendered = metrics.render_latest().decode("utf-8")
    assert "rl_trading_inference_close_boundary_retries_total 2.0" in rendered
    assert "rl_trading_inference_errors_total{" not in rendered


class _FakeStream:
    def __init__(self, *, fail_first_ack: bool = False) -> None:
        self.acked: list[str] = []
        self._fail_first_ack = fail_first_ack

    def ack(self, *, message: RlTradingRedisCandleMessage) -> None:
        if self._fail_first_ack:
            self._fail_first_ack = False
            raise RuntimeError("simulated ack failure")
        self.acked.append(message.message_id)


class _FakeWindowReader:
    def __init__(self, windows: dict[str, Stage13FeatureWindow]) -> None:
        self._windows = windows

    def read_window_at_message(self, **kwargs: object) -> Stage13FeatureWindow:
        return self._windows[str(kwargs["message_id"])]


class _FakeCloseBoundaryReader:
    def __init__(self, *, window: Stage13FeatureWindow, failures: int) -> None:
        self._window = window
        self._failures = failures
        self.calls = 0

    def read_window_at_message(self, **kwargs: object) -> Stage13FeatureWindow:
        del kwargs
        self.calls += 1
        if self.calls <= self._failures:
            raise Stage13MonitorOnlyInferenceError(
                reason="redis_window_contains_unclosed_candle"
            )
        return self._window


class _FakePolicy:
    model_version_id = "stage08k_roehub_native_best_3e033951"

    def __init__(self) -> None:
        self.policy_config = Stage08kMonitorPolicyConfig()
        self.calls = 0

    def decide(self, candles: object) -> Stage08kMonitorDecision:
        del candles
        self.calls += 1
        eligible = self.calls == 1
        signal = Stage08kArticleSignal(
            eligible=eligible,
            event_return=0.06 if eligible else 0.0,
            volatility_score=0.06 if eligible else 0.0,
            contrast_max_abs_return=0.0,
            reason="article_signal_eligible" if eligible else "event_move_below_threshold",
        )
        return Stage08kMonitorDecision(
            requested_action_id=1 if eligible else 0,
            requested_action_name="open_long" if eligible else "hold",
            action_id=1 if eligible else 0,
            action_name="open_long" if eligible else "hold",
            confidence=0.2 if eligible else 0.0,
            q_values=(0.0, 0.2, 0.0, 0.0),
            feature_hash=("f" if eligible else "e") * 64,
            policy_reason="model_action_allowed" if eligible else signal.reason,
            signal=signal,
        )


def _window(*, close_time: datetime, close_price: float) -> Stage13FeatureWindow:
    candles = tuple(
        RlFeatureCandle(
            open=100.0,
            high=max(100.0, close_price),
            low=min(100.0, close_price),
            close=close_price if index == 89 else 100.0,
            volume_base=10.0,
            volume_quote=1_000.0,
            trades_count=5,
        )
        for index in range(90)
    )
    return Stage13FeatureWindow(
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
        ts_open_utc=close_time - timedelta(minutes=90),
        ts_close_utc=close_time,
        candles=candles,
    )
