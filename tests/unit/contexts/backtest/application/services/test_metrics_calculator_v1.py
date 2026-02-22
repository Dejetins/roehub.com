from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    BacktestEquityCurveV1,
    BacktestMetricsCalculatorV1,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_backtest_metrics_calculator_v1_computes_benchmark_and_ratios() -> None:
    """
    Verify deterministic benchmark and Sharpe/Sortino/Calmar formulas on synthetic daily equity.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Equity curve already matches target-slice bars and daily resample points.
    Raises:
        AssertionError: If benchmark/ratio values differ from EPIC-06 formulas.
    Side Effects:
        None.
    """
    closes = (100.0, 101.0, 100.495)
    candles = _build_daily_candles(closes=closes)
    close_ts = _close_ts_ms(candles=candles)
    equity = np.asarray(closes, dtype=np.float64)
    calculator = BacktestMetricsCalculatorV1()

    metrics = calculator.calculate(
        requested_time_range=candles.time_range,
        candles=candles,
        target_slice=slice(0, len(closes)),
        execution_params=_execution_params(init_cash_quote=100.0),
        trades=(),
        equity_curve=BacktestEquityCurveV1(
            close_ts_ms=close_ts,
            equity_close_quote=equity,
            have_position=np.zeros(len(closes), dtype=np.float64),
            exposure_frac=np.zeros(len(closes), dtype=np.float64),
        ),
    )

    returns = np.asarray((0.01, -0.005), dtype=np.float64)
    geometric_mean = float(np.exp(np.mean(np.log(1.0 + returns))) - 1.0)
    annual_return = float((1.0 + geometric_mean) ** 365 - 1.0)
    expected_sharpe = annual_return / (float(np.std(returns, ddof=1)) * np.sqrt(365.0))
    downside = np.minimum(returns, 0.0)
    downside_ann = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(365.0))
    expected_sortino = annual_return / downside_ann
    expected_calmar = annual_return / 0.005

    expected_benchmark_return = ((float(candles.close[-1]) / float(candles.close[0])) - 1.0) * 100.0
    assert metrics["Benchmark Return [%]"] == pytest.approx(
        expected_benchmark_return,
        rel=1e-12,
        abs=1e-12,
    )
    assert metrics["Sharpe Ratio"] == pytest.approx(expected_sharpe, rel=1e-12, abs=1e-12)
    assert metrics["Sortino Ratio"] == pytest.approx(expected_sortino, rel=1e-12, abs=1e-12)
    assert metrics["Calmar Ratio"] == pytest.approx(expected_calmar, rel=1e-12, abs=1e-12)


def test_backtest_metrics_calculator_v1_detects_drawdown_episode_durations() -> None:
    """
    Verify drawdown duration metrics include recovered and unrecovered episodes deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Drawdown episodes are closed at recovery bar or at the last bar when unrecovered.
    Raises:
        AssertionError: If max/avg drawdown duration metrics differ from expected values.
    Side Effects:
        None.
    """
    closes = (100.0, 120.0, 110.0, 120.0, 90.0)
    candles = _build_daily_candles(closes=closes)
    calculator = BacktestMetricsCalculatorV1()
    metrics = calculator.calculate(
        requested_time_range=candles.time_range,
        candles=candles,
        target_slice=slice(0, len(closes)),
        execution_params=_execution_params(init_cash_quote=100.0),
        trades=(),
        equity_curve=BacktestEquityCurveV1(
            close_ts_ms=_close_ts_ms(candles=candles),
            equity_close_quote=np.asarray(closes, dtype=np.float64),
            have_position=np.zeros(len(closes), dtype=np.float64),
            exposure_frac=np.zeros(len(closes), dtype=np.float64),
        ),
    )

    assert metrics["Max. Drawdown [%]"] == pytest.approx(25.0)
    assert metrics["Max. Drawdown Duration"] == timedelta(days=1)
    assert metrics["Avg. Drawdown Duration"] == timedelta(hours=12)


def test_backtest_metrics_calculator_v1_handles_no_trades_and_short_ratio_period() -> None:
    """
    Verify no-trade and too-short-period edge cases return deterministic undefined metric values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ratios require at least two daily returns after 1d resample.
    Raises:
        AssertionError: If undefined metrics are not represented by `None`.
    Side Effects:
        None.
    """
    candles = _build_daily_candles(closes=(100.0,))
    calculator = BacktestMetricsCalculatorV1()
    metrics = calculator.calculate(
        requested_time_range=candles.time_range,
        candles=candles,
        target_slice=slice(0, 1),
        execution_params=_execution_params(init_cash_quote=100.0),
        trades=(),
        equity_curve=BacktestEquityCurveV1(
            close_ts_ms=_close_ts_ms(candles=candles),
            equity_close_quote=np.asarray((100.0,), dtype=np.float64),
            have_position=np.asarray((0.0,), dtype=np.float64),
            exposure_frac=np.asarray((0.0,), dtype=np.float64),
        ),
    )

    assert metrics["Num. Trades"] == 0
    assert metrics["Win Rate [%]"] is None
    assert metrics["Best Trade [%]"] is None
    assert metrics["Worst Trade [%]"] is None
    assert metrics["Avg. Trade [%]"] is None
    assert metrics["Expectancy"] is None
    assert metrics["SQN"] is None
    assert metrics["Sharpe Ratio"] is None
    assert metrics["Sortino Ratio"] is None
    assert metrics["Calmar Ratio"] is None


def _execution_params(*, init_cash_quote: float) -> ExecutionParamsV1:
    """
    Build deterministic execution-parameter fixture with disabled fee/slippage impact.

    Args:
        init_cash_quote: Initial quote balance.
    Returns:
        ExecutionParamsV1: Immutable execution settings fixture.
    Assumptions:
        Direction/sizing modes are irrelevant for metrics-only tests.
    Raises:
        ValueError: If execution params violate domain invariants.
    Side Effects:
        None.
    """
    return ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="all_in",
        init_cash_quote=init_cash_quote,
        fixed_quote=100.0,
        safe_profit_percent=30.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )


def _build_daily_candles(*, closes: tuple[float, ...]) -> CandleArrays:
    """
    Build deterministic `1d` candle arrays fixture with one close value per UTC day.

    Args:
        closes: Daily close sequence.
    Returns:
        CandleArrays: Dense daily candles fixture.
    Assumptions:
        Open/high/low prices equal close for simplicity in reporting metric tests.
    Raises:
        ValueError: If primitive constructors reject generated timeline values.
    Side Effects:
        None.
    """
    bars = len(closes)
    day_ms = int(timedelta(days=1) // timedelta(milliseconds=1))
    ts_open = np.arange(bars, dtype=np.int64) * day_ms
    close_array = np.asarray(closes, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(_EPOCH_UTC),
            end=UtcTimestamp(_EPOCH_UTC + timedelta(days=bars)),
        ),
        timeframe=Timeframe("1d"),
        ts_open=ts_open,
        open=close_array.copy(),
        high=close_array.copy(),
        low=close_array.copy(),
        close=close_array,
        volume=np.ones(bars, dtype=np.float32),
    )


def _close_ts_ms(*, candles: CandleArrays) -> np.ndarray:
    """
    Convert daily candles open timestamps to close timestamps in epoch milliseconds.

    Args:
        candles: Daily candle arrays fixture.
    Returns:
        np.ndarray: Close timestamps in epoch milliseconds.
    Assumptions:
        Candle timeframe is `1d` for this test helper.
    Raises:
        None.
    Side Effects:
        None.
    """
    day_ms = int(timedelta(days=1) // timedelta(milliseconds=1))
    return np.ascontiguousarray(
        candles.ts_open.astype(np.int64, copy=False) + np.int64(day_ms),
        dtype=np.int64,
    )
