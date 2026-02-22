from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Mapping

import numpy as np

from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import Timeframe, TimeRange

from .equity_curve_builder_v1 import BacktestEquityCurveV1

BacktestMetricValueV1 = datetime | timedelta | float | int | str | None
_DAY_MS = 24 * 60 * 60 * 1000
_ANNUALIZATION_DAYS = 365.0
_DRAWNDOWN_ZERO_EPS = 1e-12

BACKTEST_METRIC_ORDER_V1 = (
    "Start",
    "End",
    "Duration",
    "Init. Cash",
    "Total Profit",
    "Total Return [%]",
    "Benchmark Return [%]",
    "Position Coverage [%]",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Max. Drawdown Duration",
    "Avg. Drawdown Duration",
    "Num. Trades",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg. Trade [%]",
    "Max. Trade Duration",
    "Avg. Trade Duration",
    "Expectancy",
    "SQN",
    "Gross Exposure",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
)


@dataclass(frozen=True, slots=True)
class _DrawdownStatsV1:
    """
    Internal drawdown summary used to populate reporting metrics table values.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/services/table_formatter_v1.py
    """

    max_drawdown_pct: float | None
    avg_drawdown_pct: float | None
    max_drawdown_duration: timedelta | None
    avg_drawdown_duration: timedelta | None
    max_drawdown_frac: float | None


@dataclass(frozen=True, slots=True)
class _TradeStatsV1:
    """
    Internal trade-statistics payload for deterministic table metric assembly.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
    """

    num_trades: int
    win_rate_pct: float | None
    best_trade_pct: float | None
    worst_trade_pct: float | None
    avg_trade_pct: float | None
    max_trade_duration: timedelta | None
    avg_trade_duration: timedelta | None
    expectancy: float | None
    sqn: float | None


@dataclass(frozen=True, slots=True)
class _RatioStatsV1:
    """
    Internal annualized ratio payload computed from 1d-resampled equity returns.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py
    """

    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None


class BacktestMetricsCalculatorV1:
    """
    Calculate deterministic EPIC-06 raw metrics for one Stage-B top-k variant report.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py
      - src/trading/contexts/backtest/application/services/table_formatter_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """

    def calculate(
        self,
        *,
        requested_time_range: TimeRange,
        candles: CandleArrays,
        target_slice: slice,
        execution_params: ExecutionParamsV1,
        trades: tuple[TradeV1, ...],
        equity_curve: BacktestEquityCurveV1,
    ) -> Mapping[str, BacktestMetricValueV1]:
        """
        Calculate deterministic raw metric values in fixed EPIC-06 metric order.

        Args:
            requested_time_range: User request range used for Start/End/Duration rows.
            candles: Warmup-inclusive candle arrays.
            target_slice: Stage-B target bars slice.
            execution_params: Execution settings used by close-fill engine.
            trades: Closed trade snapshots for this variant.
            equity_curve: Pre-built target-slice equity and exposure vectors.
        Returns:
            Mapping[str, BacktestMetricValueV1]: Raw metric values keyed by fixed metric names.
        Assumptions:
            `equity_curve` was built from the same candles/target_slice/trades payload.
        Raises:
            ValueError: If equity-curve size does not match target slice bars count.
        Side Effects:
            None.
        """
        start_index, stop_index = _normalized_slice_bounds(
            target_slice=target_slice,
            total_bars=int(candles.close.shape[0]),
        )
        target_bars = stop_index - start_index
        if target_bars != int(equity_curve.equity_close_quote.shape[0]):
            raise ValueError("equity_curve size must match target_slice bars count")

        final_equity_quote = float(execution_params.init_cash_quote)
        if target_bars > 0:
            final_equity_quote = float(equity_curve.equity_close_quote[-1])
        total_profit_quote = final_equity_quote - float(execution_params.init_cash_quote)
        total_return_pct = None
        if execution_params.init_cash_quote != 0.0:
            total_return_pct = (
                (final_equity_quote / float(execution_params.init_cash_quote)) - 1.0
            ) * 100.0

        drawdown_stats = _drawdown_stats(equity_curve=equity_curve)
        trade_stats = _trade_stats(
            candles=candles,
            trades=trades,
        )
        ratios = _ratio_stats(
            equity_curve=equity_curve,
            max_drawdown_frac=drawdown_stats.max_drawdown_frac,
        )

        coverage_pct = None
        if target_bars > 0:
            coverage_pct = float(np.mean(equity_curve.have_position, dtype=np.float64) * 100.0)

        gross_exposure = None
        if target_bars > 0:
            if np.all(np.isfinite(equity_curve.exposure_frac)):
                gross_exposure = float(np.mean(equity_curve.exposure_frac, dtype=np.float64))

        metrics: dict[str, BacktestMetricValueV1] = {
            "Start": requested_time_range.start.value,
            "End": requested_time_range.end.value,
            "Duration": requested_time_range.duration(),
            "Init. Cash": float(execution_params.init_cash_quote),
            "Total Profit": total_profit_quote,
            "Total Return [%]": total_return_pct,
            "Benchmark Return [%]": _benchmark_return_pct(
                candles=candles,
                start_index=start_index,
                stop_index=stop_index,
            ),
            "Position Coverage [%]": coverage_pct,
            "Max. Drawdown [%]": drawdown_stats.max_drawdown_pct,
            "Avg. Drawdown [%]": drawdown_stats.avg_drawdown_pct,
            "Max. Drawdown Duration": drawdown_stats.max_drawdown_duration,
            "Avg. Drawdown Duration": drawdown_stats.avg_drawdown_duration,
            "Num. Trades": trade_stats.num_trades,
            "Win Rate [%]": trade_stats.win_rate_pct,
            "Best Trade [%]": trade_stats.best_trade_pct,
            "Worst Trade [%]": trade_stats.worst_trade_pct,
            "Avg. Trade [%]": trade_stats.avg_trade_pct,
            "Max. Trade Duration": trade_stats.max_trade_duration,
            "Avg. Trade Duration": trade_stats.avg_trade_duration,
            "Expectancy": trade_stats.expectancy,
            "SQN": trade_stats.sqn,
            "Gross Exposure": gross_exposure,
            "Sharpe Ratio": ratios.sharpe_ratio,
            "Sortino Ratio": ratios.sortino_ratio,
            "Calmar Ratio": ratios.calmar_ratio,
        }
        return MappingProxyType(metrics)


def _benchmark_return_pct(
    *,
    candles: CandleArrays,
    start_index: int,
    stop_index: int,
) -> float | None:
    """
    Compute buy-and-hold benchmark return on `target_slice` closes without fees/slippage.

    Args:
        candles: Warmup-inclusive candle arrays.
        start_index: Inclusive start index for target slice.
        stop_index: Exclusive stop index for target slice.
    Returns:
        float | None: Benchmark return in percent, or `None` when target slice is empty.
    Assumptions:
        Close prices are positive finite values for valid bars.
    Raises:
        None.
    Side Effects:
        None.
    """
    if stop_index <= start_index:
        return None
    close_first = float(candles.close[start_index])
    close_last = float(candles.close[stop_index - 1])
    if close_first <= 0.0:
        return None
    return ((close_last / close_first) - 1.0) * 100.0


def _drawdown_stats(*, equity_curve: BacktestEquityCurveV1) -> _DrawdownStatsV1:
    """
    Compute drawdown percentages and drawdown episode durations from equity series.

    Args:
        equity_curve: Equity-curve payload aligned to target bars.
    Returns:
        _DrawdownStatsV1: Drawdown metrics payload.
    Assumptions:
        Equity values are finite for deterministic metric output.
    Raises:
        None.
    Side Effects:
        None.
    """
    if equity_curve.equity_close_quote.shape[0] == 0:
        return _DrawdownStatsV1(
            max_drawdown_pct=None,
            avg_drawdown_pct=None,
            max_drawdown_duration=None,
            avg_drawdown_duration=None,
            max_drawdown_frac=None,
        )

    peaks = np.maximum.accumulate(equity_curve.equity_close_quote)
    if np.any(peaks <= 0.0):
        return _DrawdownStatsV1(
            max_drawdown_pct=None,
            avg_drawdown_pct=None,
            max_drawdown_duration=None,
            avg_drawdown_duration=None,
            max_drawdown_frac=None,
        )

    dd_frac = 1.0 - np.divide(equity_curve.equity_close_quote, peaks)
    if not np.all(np.isfinite(dd_frac)):
        return _DrawdownStatsV1(
            max_drawdown_pct=None,
            avg_drawdown_pct=None,
            max_drawdown_duration=None,
            avg_drawdown_duration=None,
            max_drawdown_frac=None,
        )

    durations = _drawdown_episode_durations(
        close_ts_ms=equity_curve.close_ts_ms,
        dd_frac=dd_frac,
    )
    max_duration = None
    avg_duration = None
    if len(durations) > 0:
        max_duration = max(durations)
        total_ms = sum(int(duration // timedelta(milliseconds=1)) for duration in durations)
        avg_duration = timedelta(milliseconds=(total_ms // len(durations)))

    max_drawdown_frac = float(np.max(dd_frac))
    avg_drawdown_frac = float(np.mean(dd_frac, dtype=np.float64))
    return _DrawdownStatsV1(
        max_drawdown_pct=max_drawdown_frac * 100.0,
        avg_drawdown_pct=avg_drawdown_frac * 100.0,
        max_drawdown_duration=max_duration,
        avg_drawdown_duration=avg_duration,
        max_drawdown_frac=max_drawdown_frac,
    )


def _drawdown_episode_durations(
    *,
    close_ts_ms: np.ndarray,
    dd_frac: np.ndarray,
) -> tuple[timedelta, ...]:
    """
    Detect drawdown episodes by `0 -> >0 -> 0` transitions and compute durations.

    Args:
        close_ts_ms: Bar close timestamps in epoch milliseconds.
        dd_frac: Drawdown fraction series for aligned bars.
    Returns:
        tuple[timedelta, ...]: Deterministic drawdown-episode durations.
    Assumptions:
        Arrays are aligned one-dimensional vectors of equal length.
    Raises:
        None.
    Side Effects:
        None.
    """
    if close_ts_ms.shape[0] == 0:
        return ()

    in_drawdown = False
    episode_start_index = 0
    durations: list[timedelta] = []

    for index, dd_value in enumerate(dd_frac):
        is_drawdown = bool(dd_value > _DRAWNDOWN_ZERO_EPS)
        if not in_drawdown and is_drawdown:
            in_drawdown = True
            episode_start_index = index
            continue
        if in_drawdown and not is_drawdown:
            duration_ms = int(close_ts_ms[index] - close_ts_ms[episode_start_index])
            durations.append(timedelta(milliseconds=duration_ms))
            in_drawdown = False

    if in_drawdown:
        last_index = int(close_ts_ms.shape[0] - 1)
        duration_ms = int(close_ts_ms[last_index] - close_ts_ms[episode_start_index])
        durations.append(timedelta(milliseconds=duration_ms))
    return tuple(durations)


def _trade_stats(*, candles: CandleArrays, trades: tuple[TradeV1, ...]) -> _TradeStatsV1:
    """
    Compute deterministic trade-count, return, duration, expectancy, and SQN metrics.

    Args:
        candles: Warmup-inclusive candle arrays.
        trades: Closed trade snapshots emitted by execution engine.
    Returns:
        _TradeStatsV1: Trade metrics payload.
    Assumptions:
        Trades are chronologically ordered and satisfy domain invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    trade_count = len(trades)
    if trade_count == 0:
        return _TradeStatsV1(
            num_trades=0,
            win_rate_pct=None,
            best_trade_pct=None,
            worst_trade_pct=None,
            avg_trade_pct=None,
            max_trade_duration=None,
            avg_trade_duration=None,
            expectancy=None,
            sqn=None,
        )

    bar_close_ts = _bar_close_ts_ms(candles=candles)
    trade_returns_pct = np.empty(trade_count, dtype=np.float64)
    trade_net_pnl = np.empty(trade_count, dtype=np.float64)
    trade_duration_ms = np.empty(trade_count, dtype=np.int64)

    for index, trade in enumerate(trades):
        trade_returns_pct[index] = (trade.net_pnl_quote / trade.entry_quote_amount) * 100.0
        trade_net_pnl[index] = float(trade.net_pnl_quote)
        duration_ms = int(bar_close_ts[trade.exit_bar_index] - bar_close_ts[trade.entry_bar_index])
        trade_duration_ms[index] = duration_ms

    win_rate_pct = float(np.mean(trade_net_pnl > 0.0, dtype=np.float64) * 100.0)
    max_trade_duration = timedelta(milliseconds=int(np.max(trade_duration_ms)))
    avg_trade_duration = timedelta(
        milliseconds=int(np.sum(trade_duration_ms, dtype=np.int64) // trade_count)
    )
    expectancy = float(np.mean(trade_returns_pct, dtype=np.float64))

    sqn = None
    if trade_count > 1:
        pnl_std = float(np.std(trade_net_pnl, ddof=1))
        if pnl_std > 0.0 and np.isfinite(pnl_std):
            sqn = float(np.sqrt(float(trade_count)) * np.mean(trade_net_pnl) / pnl_std)

    return _TradeStatsV1(
        num_trades=trade_count,
        win_rate_pct=win_rate_pct,
        best_trade_pct=float(np.max(trade_returns_pct)),
        worst_trade_pct=float(np.min(trade_returns_pct)),
        avg_trade_pct=float(np.mean(trade_returns_pct, dtype=np.float64)),
        max_trade_duration=max_trade_duration,
        avg_trade_duration=avg_trade_duration,
        expectancy=expectancy,
        sqn=sqn,
    )


def _ratio_stats(
    *,
    equity_curve: BacktestEquityCurveV1,
    max_drawdown_frac: float | None,
) -> _RatioStatsV1:
    """
    Compute Sharpe/Sortino/Calmar from 1d-resampled equity and annualization=365.

    Args:
        equity_curve: Equity-curve payload aligned to target bars.
        max_drawdown_frac: Max drawdown fraction used for Calmar denominator.
    Returns:
        _RatioStatsV1: Ratios payload with `None` for undefined values.
    Assumptions:
        Ratios are undefined for too-short or non-finite daily returns sequences.
    Raises:
        None.
    Side Effects:
        None.
    """
    daily_returns = _daily_returns_from_equity(equity_curve=equity_curve)
    if daily_returns is None or daily_returns.shape[0] < 2:
        return _RatioStatsV1(sharpe_ratio=None, sortino_ratio=None, calmar_ratio=None)

    one_plus_returns = 1.0 + daily_returns
    if np.any(one_plus_returns <= 0.0):
        return _RatioStatsV1(sharpe_ratio=None, sortino_ratio=None, calmar_ratio=None)

    log_growth = np.log(one_plus_returns)
    if not np.all(np.isfinite(log_growth)):
        return _RatioStatsV1(sharpe_ratio=None, sortino_ratio=None, calmar_ratio=None)

    geometric_mean = float(np.exp(np.mean(log_growth, dtype=np.float64)) - 1.0)
    annual_return = float((1.0 + geometric_mean) ** _ANNUALIZATION_DAYS - 1.0)
    if not np.isfinite(annual_return):
        return _RatioStatsV1(sharpe_ratio=None, sortino_ratio=None, calmar_ratio=None)

    sharpe_ratio = None
    volatility = float(np.std(daily_returns, ddof=1))
    vol_ann = volatility * float(np.sqrt(_ANNUALIZATION_DAYS))
    if vol_ann > 0.0 and np.isfinite(vol_ann):
        sharpe_ratio = annual_return / vol_ann

    sortino_ratio = None
    downside_returns = np.minimum(daily_returns, 0.0)
    downside_rms = float(np.sqrt(np.mean(np.square(downside_returns), dtype=np.float64)))
    downside_ann = downside_rms * float(np.sqrt(_ANNUALIZATION_DAYS))
    if downside_ann > 0.0 and np.isfinite(downside_ann):
        sortino_ratio = annual_return / downside_ann

    calmar_ratio = None
    if max_drawdown_frac is not None and max_drawdown_frac > 0.0:
        calmar_ratio = annual_return / max_drawdown_frac

    return _RatioStatsV1(
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
    )


def _daily_returns_from_equity(*, equity_curve: BacktestEquityCurveV1) -> np.ndarray | None:
    """
    Resample equity to UTC 1d by `last` and return daily pct-change vector.

    Args:
        equity_curve: Equity curve payload aligned to bar closes.
    Returns:
        np.ndarray | None: Daily returns vector or `None` when sequence is undefined.
    Assumptions:
        Daily resample uses UTC epoch day boundaries.
    Raises:
        None.
    Side Effects:
        None.
    """
    if equity_curve.equity_close_quote.shape[0] < 2:
        return None
    equity_1d = _resample_equity_last_1d(
        close_ts_ms=equity_curve.close_ts_ms,
        equity_close_quote=equity_curve.equity_close_quote,
    )
    if equity_1d.shape[0] < 2:
        return None

    previous = equity_1d[:-1]
    current = equity_1d[1:]
    if np.any(previous <= 0.0):
        return None
    returns = np.divide(current, previous) - 1.0
    if not np.all(np.isfinite(returns)):
        return None
    return np.ascontiguousarray(returns, dtype=np.float64)


def _resample_equity_last_1d(
    *,
    close_ts_ms: np.ndarray,
    equity_close_quote: np.ndarray,
) -> np.ndarray:
    """
    Resample close-equity sequence into one value per UTC day using `last` aggregator.

    Args:
        close_ts_ms: Bar close timestamps in epoch milliseconds.
        equity_close_quote: Close-equity series aligned to timestamps.
    Returns:
        np.ndarray: One-dimensional daily last-equity vector.
    Assumptions:
        Input arrays are aligned and sorted by timestamp.
    Raises:
        None.
    Side Effects:
        None.
    """
    if close_ts_ms.shape[0] == 0:
        return np.asarray((), dtype=np.float64)

    day_ids = np.floor_divide(close_ts_ms, np.int64(_DAY_MS))
    values: list[float] = []
    active_day = int(day_ids[0])
    last_equity = float(equity_close_quote[0])

    for index in range(1, close_ts_ms.shape[0]):
        current_day = int(day_ids[index])
        if current_day != active_day:
            values.append(last_equity)
            active_day = current_day
        last_equity = float(equity_close_quote[index])

    values.append(last_equity)
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)


def _bar_close_ts_ms(*, candles: CandleArrays) -> np.ndarray:
    """
    Build full-timeline bar close timestamps from candle opens and timeframe duration.

    Args:
        candles: Candle arrays for one timeframe.
    Returns:
        np.ndarray: Full timeline close timestamps in epoch milliseconds.
    Assumptions:
        Candle timeline uses regular timeframe spacing.
    Raises:
        ValueError: If timeframe duration is not positive.
    Side Effects:
        None.
    """
    timeframe_ms = _timeframe_millis(timeframe=candles.timeframe)
    return np.ascontiguousarray(
        candles.ts_open.astype(np.int64, copy=False) + np.int64(timeframe_ms),
        dtype=np.int64,
    )


def _normalized_slice_bounds(*, target_slice: slice, total_bars: int) -> tuple[int, int]:
    """
    Normalize optional target-slice bounds and validate half-open range invariants.

    Args:
        target_slice: Requested half-open index slice.
        total_bars: Total bars in candles timeline.
    Returns:
        tuple[int, int]: Resolved `(start, stop)` bounds.
    Assumptions:
        `target_slice` semantics follow `[start, stop)` contract.
    Raises:
        ValueError: If bounds violate deterministic range invariants.
    Side Effects:
        None.
    """
    start_index = 0 if target_slice.start is None else int(target_slice.start)
    stop_index = total_bars if target_slice.stop is None else int(target_slice.stop)
    if start_index < 0:
        raise ValueError("target_slice.start must be >= 0")
    if stop_index < start_index:
        raise ValueError("target_slice.stop must be >= target_slice.start")
    if stop_index > total_bars:
        raise ValueError("target_slice.stop must be <= candles bars count")
    return start_index, stop_index


def _timeframe_millis(*, timeframe: Timeframe) -> int:
    """
    Convert timeframe duration to positive integer milliseconds for timestamp math.

    Args:
        timeframe: Shared-kernel timeframe primitive.
    Returns:
        int: Positive timeframe duration in milliseconds.
    Assumptions:
        Timeframe primitive has already validated supported code.
    Raises:
        ValueError: If computed duration is non-positive.
    Side Effects:
        None.
    """
    timeframe_ms = int(timeframe.duration() // timedelta(milliseconds=1))
    if timeframe_ms <= 0:
        raise ValueError("timeframe duration must be > 0")
    return timeframe_ms


__all__ = [
    "BACKTEST_METRIC_ORDER_V1",
    "BacktestMetricValueV1",
    "BacktestMetricsCalculatorV1",
]
