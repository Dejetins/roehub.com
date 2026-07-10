from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading.contexts.rl_trading.adapters.outbound.persistence.file_monitor_state import (
    FileStage08kMonitorStateStore,
)
from trading.contexts.rl_trading.domain.stage08k_monitor_policy import (
    Stage08kMonitorPolicyConfig,
)
from trading.contexts.rl_trading.domain.stage08k_monitor_runtime import (
    open_stage08k_virtual_trade_v1,
)


def test_file_monitor_state_survives_store_recreation(tmp_path) -> None:
    path = (tmp_path / "monitor-state.json").resolve()
    store = FileStage08kMonitorStateStore(path=path)
    trade = open_stage08k_virtual_trade_v1(
        instrument_key="binance:futures:BTCUSDT",
        symbol="BTCUSDT",
        entry_decision_id="d" * 64,
        entry_time_utc=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        entry_price=100.0,
        policy=Stage08kMonitorPolicyConfig(),
    )

    store.upsert(trade=trade)
    store.commit_processed(
        instrument_key=trade.instrument_key,
        candle_close_utc=trade.entry_time_utc,
        pending_trade=trade,
    )
    restored = FileStage08kMonitorStateStore(path=path)

    assert restored.get(instrument_key=trade.instrument_key) == trade
    assert (
        restored.last_processed_close_utc(instrument_key=trade.instrument_key)
        == trade.entry_time_utc
    )
    with pytest.raises(ValueError, match="time regression"):
        restored.commit_processed(
            instrument_key=trade.instrument_key,
            candle_close_utc=trade.entry_time_utc - timedelta(minutes=1),
            pending_trade=trade,
        )
    restored.remove(instrument_key=trade.instrument_key)
    assert FileStage08kMonitorStateStore(path=path).all_pending() == ()


def test_file_monitor_state_rejects_overwriting_an_open_trade(tmp_path) -> None:
    path = (tmp_path / "monitor-state.json").resolve()
    store = FileStage08kMonitorStateStore(path=path)
    first = open_stage08k_virtual_trade_v1(
        instrument_key="binance:futures:BTCUSDT",
        symbol="BTCUSDT",
        entry_decision_id="a" * 64,
        entry_time_utc=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        entry_price=100.0,
        policy=Stage08kMonitorPolicyConfig(),
    )
    second = open_stage08k_virtual_trade_v1(
        instrument_key=first.instrument_key,
        symbol=first.symbol,
        entry_decision_id="b" * 64,
        entry_time_utc=first.entry_time_utc + timedelta(seconds=30),
        entry_price=101.0,
        policy=Stage08kMonitorPolicyConfig(),
    )

    store.upsert(trade=first)

    with pytest.raises(ValueError, match="already exists"):
        store.upsert(trade=second)
