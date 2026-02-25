from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.contexts.backtest.application.services import BacktestEquityCurveBuilderV1
from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


def test_equity_curve_builder_allows_reversal_on_same_bar() -> None:
    """
    Verify reporting supports engine-style reversal on one close bar.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Reversal is represented as one trade exiting on bar t and another entering on bar t.
    Raises:
        AssertionError: If builder rejects reversal trade schedule.
    Side Effects:
        None.
    """
    candles = _candles(bars=4)
    execution_params = _execution_params()
    builder = BacktestEquityCurveBuilderV1()

    # Trade 1: entry 0 -> exit 2
    # Trade 2: entry 2 -> exit 3 (reversal on bar 2)
    trades = (
        _trade(
            trade_id=1,
            direction="long",
            entry_bar_index=0,
            exit_bar_index=2,
            entry_quote_amount=100.0,
            gross_pnl_quote=10.0,
        ),
        _trade(
            trade_id=2,
            direction="short",
            entry_bar_index=2,
            exit_bar_index=3,
            entry_quote_amount=110.0,
            gross_pnl_quote=-5.0,
        ),
    )

    curve = builder.build(
        candles=candles,
        target_slice=slice(0, 4),
        trades=trades,
        execution_params=execution_params,
    )

    assert curve.close_ts_ms.shape == (4,)
    assert curve.equity_close_quote.shape == (4,)
    assert curve.have_position.shape == (4,)
    assert curve.exposure_frac.shape == (4,)
    assert np.isfinite(curve.equity_close_quote).all()


def test_equity_curve_builder_allows_one_bar_trade_entry_equals_exit() -> None:
    """
    Verify builder supports entry and exit on the same bar for one trade id.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        One-bar trades can occur in forced-close scenarios on single-bar target slices.
    Raises:
        AssertionError: If builder rejects one-bar trade schedule.
    Side Effects:
        None.
    """
    candles = _candles(bars=2)
    execution_params = _execution_params()
    builder = BacktestEquityCurveBuilderV1()

    trades = (
        _trade(
            trade_id=1,
            direction="long",
            entry_bar_index=1,
            exit_bar_index=1,
            entry_quote_amount=100.0,
            gross_pnl_quote=0.0,
        ),
    )

    curve = builder.build(
        candles=candles,
        target_slice=slice(0, 2),
        trades=trades,
        execution_params=execution_params,
    )

    assert curve.close_ts_ms.shape == (2,)
    assert np.isfinite(curve.equity_close_quote).all()


def test_equity_curve_builder_rejects_true_overlap() -> None:
    """
    Verify builder still fails fast for true overlapping trades.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        True overlap means the second trade enters strictly before the first exits.
    Raises:
        AssertionError: If true overlap is not rejected.
    Side Effects:
        None.
    """
    candles = _candles(bars=4)
    execution_params = _execution_params()
    builder = BacktestEquityCurveBuilderV1()

    trades = (
        _trade(
            trade_id=1,
            direction="long",
            entry_bar_index=0,
            exit_bar_index=3,
            entry_quote_amount=100.0,
            gross_pnl_quote=0.0,
        ),
        _trade(
            trade_id=2,
            direction="short",
            entry_bar_index=2,
            exit_bar_index=2,
            entry_quote_amount=100.0,
            gross_pnl_quote=0.0,
        ),
    )

    with pytest.raises(ValueError, match="active"):
        builder.build(
            candles=candles,
            target_slice=slice(0, 4),
            trades=trades,
            execution_params=execution_params,
        )


def _execution_params() -> ExecutionParamsV1:
    """Build deterministic execution params fixture."""
    return ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="fixed_quote",
        init_cash_quote=1000.0,
        fixed_quote=100.0,
        safe_profit_percent=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )


def _trade(
    *,
    trade_id: int,
    direction: str,
    entry_bar_index: int,
    exit_bar_index: int,
    entry_quote_amount: float,
    gross_pnl_quote: float,
) -> TradeV1:
    """Build deterministic TradeV1 fixture with minimal required fields."""
    return TradeV1(
        trade_id=trade_id,
        direction=direction,
        entry_bar_index=entry_bar_index,
        exit_bar_index=exit_bar_index,
        entry_fill_price=100.0,
        exit_fill_price=100.0,
        qty_base=1.0,
        entry_quote_amount=entry_quote_amount,
        exit_quote_amount=entry_quote_amount + gross_pnl_quote,
        entry_fee_quote=0.0,
        exit_fee_quote=0.0,
        gross_pnl_quote=gross_pnl_quote,
        net_pnl_quote=gross_pnl_quote,
        locked_profit_quote=0.0,
        exit_reason="signal_exit",
    )


def _candles(*, bars: int) -> CandleArrays:
    """Build deterministic candle arrays fixture."""
    if bars <= 0:
        raise ValueError("bars must be > 0")
    ts_open = np.asarray(
        [int((_EPOCH_UTC + (index * _ONE_MINUTE)).timestamp() * 1000) for index in range(bars)],
        dtype=np.int64,
    )
    close = np.linspace(100.0, 100.0 + float(bars - 1), num=bars, dtype=np.float32)
    time_range = TimeRange(
        UtcTimestamp(_EPOCH_UTC),
        UtcTimestamp(_EPOCH_UTC + (bars * _ONE_MINUTE)),
    )
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=time_range,
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full(bars, 1.0, dtype=np.float32),
    )
