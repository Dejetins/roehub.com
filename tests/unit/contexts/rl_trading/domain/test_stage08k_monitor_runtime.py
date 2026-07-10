from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading.contexts.rl_trading.domain.stage08k_monitor_policy import (
    Stage08kMonitorPolicyConfig,
)
from trading.contexts.rl_trading.domain.stage08k_monitor_runtime import (
    Stage08kPendingVirtualTrade,
    close_stage08k_virtual_trade_v1,
    open_stage08k_virtual_trade_v1,
)


def test_virtual_long_closes_after_one_minute_with_taker_and_slippage_costs() -> None:
    policy = Stage08kMonitorPolicyConfig()
    entry_time = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    trade = open_stage08k_virtual_trade_v1(
        instrument_key="binance:futures:BTCUSDT",
        symbol="BTCUSDT",
        entry_decision_id="d" * 64,
        entry_time_utc=entry_time,
        entry_price=100.0,
        policy=policy,
    )

    result = close_stage08k_virtual_trade_v1(
        trade=trade,
        exit_time_utc=entry_time + timedelta(minutes=1),
        exit_price=101.0,
        policy=policy,
    )

    assert result.gross_return == pytest.approx(0.01)
    assert result.net_return == pytest.approx(0.0085)
    assert result.pnl_quote == pytest.approx(42.5)
    assert result.hold_seconds == 60.0
    assert result.valid_for_policy_evaluation is True
    assert result.reason == "virtual_close_after_1m"


def test_late_virtual_close_is_recorded_but_excluded_from_policy_evaluation() -> None:
    policy = Stage08kMonitorPolicyConfig()
    entry_time = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    trade = open_stage08k_virtual_trade_v1(
        instrument_key="binance:futures:BTCUSDT",
        symbol="BTCUSDT",
        entry_decision_id="d" * 64,
        entry_time_utc=entry_time,
        entry_price=100.0,
        policy=policy,
    )

    result = close_stage08k_virtual_trade_v1(
        trade=trade,
        exit_time_utc=entry_time + timedelta(minutes=2),
        exit_price=101.0,
        policy=policy,
    )

    assert result.valid_for_policy_evaluation is False
    assert result.reason == "late_virtual_close_excluded"


def test_pending_virtual_trade_round_trips_through_payload() -> None:
    policy = Stage08kMonitorPolicyConfig()
    trade = open_stage08k_virtual_trade_v1(
        instrument_key="binance:futures:BTCUSDT",
        symbol="BTCUSDT",
        entry_decision_id="d" * 64,
        entry_time_utc=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        entry_price=100.0,
        policy=policy,
    )

    assert Stage08kPendingVirtualTrade.from_payload(trade.as_payload()) == trade
