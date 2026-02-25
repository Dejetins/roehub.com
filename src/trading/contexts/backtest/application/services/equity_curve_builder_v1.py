from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import Timeframe


@dataclass(frozen=True, slots=True)
class BacktestEquityCurveV1:
    """
    Deterministic equity-curve payload aligned to Stage-B `target_slice` bar closes.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """

    close_ts_ms: np.ndarray
    equity_close_quote: np.ndarray
    have_position: np.ndarray
    exposure_frac: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate aligned array shapes and deterministic ordering invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Arrays are pre-aligned to the same target-slice index boundaries.
        Raises:
            ValueError: If one array has unexpected dtype, shape, or ordering.
        Side Effects:
            None.
        """
        expected_size = _validate_array(
            name="close_ts_ms",
            values=self.close_ts_ms,
            expected_dtype=np.int64,
            expected_size=None,
        )
        _validate_array(
            name="equity_close_quote",
            values=self.equity_close_quote,
            expected_dtype=np.float64,
            expected_size=expected_size,
        )
        _validate_array(
            name="have_position",
            values=self.have_position,
            expected_dtype=np.float64,
            expected_size=expected_size,
        )
        _validate_array(
            name="exposure_frac",
            values=self.exposure_frac,
            expected_dtype=np.float64,
            expected_size=expected_size,
        )
        if expected_size > 1 and not np.all(self.close_ts_ms[1:] >= self.close_ts_ms[:-1]):
            raise ValueError("BacktestEquityCurveV1.close_ts_ms must be sorted")


class BacktestEquityCurveBuilderV1:
    """
    Build deterministic equity curve and coverage/exposure vectors from engine trades.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - tests/unit/contexts/backtest/application/services/test_reporting_service_v1.py
    """

    def build(
        self,
        *,
        candles: CandleArrays,
        target_slice: slice,
        trades: tuple[TradeV1, ...],
        execution_params: ExecutionParamsV1,
    ) -> BacktestEquityCurveV1:
        """
        Build deterministic close-equity series and exposure vectors on `target_slice`.

        Args:
            candles: Warmup-inclusive candle arrays.
            target_slice: Half-open trading/reporting slice on candles timeline.
            trades: Closed trades emitted by execution engine.
            execution_params: Execution defaults used by the engine.
        Returns:
            BacktestEquityCurveV1: Equity and exposure payload aligned to target bars.
        Assumptions:
            Trades are produced by close-fill engine with one active position at most.
        Raises:
            ValueError: If slice boundaries are invalid or trades violate one-position invariants.
        Side Effects:
            None.
        """
        start_index, stop_index = _normalized_slice_bounds(
            target_slice=target_slice,
            total_bars=int(candles.close.shape[0]),
        )
        target_bars = stop_index - start_index
        if target_bars == 0:
            return BacktestEquityCurveV1(
                close_ts_ms=np.asarray((), dtype=np.int64),
                equity_close_quote=np.asarray((), dtype=np.float64),
                have_position=np.asarray((), dtype=np.float64),
                exposure_frac=np.asarray((), dtype=np.float64),
            )

        timeframe_ms = _timeframe_millis(timeframe=candles.timeframe)
        close_ts_ms = np.ascontiguousarray(
            candles.ts_open[start_index:stop_index].astype(np.int64, copy=False)
            + np.int64(timeframe_ms),
            dtype=np.int64,
        )
        equity_close_quote = np.empty(target_bars, dtype=np.float64)
        have_position = np.zeros(target_bars, dtype=np.float64)
        exposure_frac = np.zeros(target_bars, dtype=np.float64)
        entry_quote_by_bar = np.zeros(target_bars, dtype=np.float64)

        ordered_trades = tuple(
            sorted(trades, key=lambda item: (item.entry_bar_index, item.trade_id))
        )
        entries_by_bar: dict[int, TradeV1] = {}
        exits_by_bar: dict[int, TradeV1] = {}
        for trade in ordered_trades:
            if trade.entry_bar_index < start_index or trade.exit_bar_index >= stop_index:
                raise ValueError("trade bar indexes must be inside target_slice")
            if trade.entry_bar_index in entries_by_bar:
                raise ValueError("multiple entries on one bar are not supported")
            if trade.exit_bar_index in exits_by_bar:
                raise ValueError("multiple exits on one bar are not supported")
            entries_by_bar[trade.entry_bar_index] = trade
            exits_by_bar[trade.exit_bar_index] = trade
            local_start = trade.entry_bar_index - start_index
            local_stop = trade.exit_bar_index - start_index + 1
            have_position[local_start:local_stop] = 1.0
            entry_quote_by_bar[local_start:local_stop] = float(trade.entry_quote_amount)

        available_quote = float(execution_params.init_cash_quote)
        safe_quote = 0.0
        active_trade: TradeV1 | None = None

        # HOT PATH: bar loop must match engine event ordering.
        # Engine ordering (v1): risk exit -> signal exit -> signal entry on the same close.
        for local_index, bar_index in enumerate(range(start_index, stop_index)):
            entry_trade = entries_by_bar.get(bar_index)
            exit_trade = exits_by_bar.get(bar_index)

            if (
                entry_trade is not None
                and exit_trade is not None
                and entry_trade.trade_id == exit_trade.trade_id
            ):
                # One-bar trade: open then close on the same bar.
                if active_trade is not None:
                    raise ValueError("cannot open a new trade while another trade is active")
                available_quote = (
                    available_quote - entry_trade.entry_quote_amount - entry_trade.entry_fee_quote
                )
                active_trade = entry_trade

                available_quote = (
                    available_quote
                    + exit_trade.entry_quote_amount
                    + exit_trade.gross_pnl_quote
                    - exit_trade.exit_fee_quote
                )
                if exit_trade.locked_profit_quote > 0.0:
                    available_quote = available_quote - exit_trade.locked_profit_quote
                    safe_quote = safe_quote + exit_trade.locked_profit_quote
                active_trade = None

            else:
                # Reversal-compatible ordering: close first, then open.
                if exit_trade is not None:
                    if active_trade is None or active_trade.trade_id != exit_trade.trade_id:
                        raise ValueError("exit trade does not match active trade")
                    available_quote = (
                        available_quote
                        + exit_trade.entry_quote_amount
                        + exit_trade.gross_pnl_quote
                        - exit_trade.exit_fee_quote
                    )
                    if exit_trade.locked_profit_quote > 0.0:
                        available_quote = available_quote - exit_trade.locked_profit_quote
                        safe_quote = safe_quote + exit_trade.locked_profit_quote
                    active_trade = None

                if entry_trade is not None:
                    if active_trade is not None:
                        raise ValueError("cannot open a new trade while another trade is active")
                    available_quote = (
                        available_quote
                        - entry_trade.entry_quote_amount
                        - entry_trade.entry_fee_quote
                    )
                    active_trade = entry_trade

            if active_trade is None:
                equity_close_quote[local_index] = available_quote + safe_quote
                continue

            close_price = float(candles.close[bar_index])
            position_value_quote = _position_value_quote(
                trade=active_trade,
                close_price=close_price,
            )
            equity_close_quote[local_index] = available_quote + safe_quote + position_value_quote

        if active_trade is not None:
            raise ValueError("trade must be closed by end of target_slice")

        for local_index in range(target_bars):
            if have_position[local_index] <= 0.0:
                continue
            equity_value = float(equity_close_quote[local_index])
            if equity_value == 0.0:
                exposure_frac[local_index] = np.nan
                continue
            exposure_frac[local_index] = entry_quote_by_bar[local_index] / equity_value

        return BacktestEquityCurveV1(
            close_ts_ms=close_ts_ms,
            equity_close_quote=np.ascontiguousarray(equity_close_quote, dtype=np.float64),
            have_position=np.ascontiguousarray(have_position, dtype=np.float64),
            exposure_frac=np.ascontiguousarray(exposure_frac, dtype=np.float64),
        )


def _position_value_quote(*, trade: TradeV1, close_price: float) -> float:
    """
    Compute deterministic mark-to-market position value for long/short close-fill trade.

    Args:
        trade: Active trade snapshot.
        close_price: Raw close price used for mark-to-market valuation.
    Returns:
        float: Position value in quote currency.
    Assumptions:
        `trade` was produced by execution engine and satisfies invariants.
    Raises:
        ValueError: If close price is non-positive.
    Side Effects:
        None.
    """
    if close_price <= 0.0:
        raise ValueError("close_price must be > 0")
    if trade.direction == "long":
        return trade.qty_base * close_price

    current_notional = trade.qty_base * close_price
    unrealized_gross_pnl_quote = trade.entry_quote_amount - current_notional
    return trade.entry_quote_amount + unrealized_gross_pnl_quote


def _normalized_slice_bounds(*, target_slice: slice, total_bars: int) -> tuple[int, int]:
    """
    Normalize optional slice bounds against candles size and validate half-open contract.

    Args:
        target_slice: Requested half-open slice.
        total_bars: Total bars in candles timeline.
    Returns:
        tuple[int, int]: Resolved `(start_index, stop_index)` bounds.
    Assumptions:
        Target slice semantics follow `[start, stop)` contract.
    Raises:
        ValueError: If bounds are invalid or out of candles range.
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
    Convert shared-kernel timeframe duration into positive integer milliseconds.

    Args:
        timeframe: Timeframe primitive.
    Returns:
        int: Positive timeframe duration in milliseconds.
    Assumptions:
        Timeframe primitive validates supported codes during construction.
    Raises:
        ValueError: If timeframe duration is non-positive.
    Side Effects:
        None.
    """
    timeframe_ms = int(timeframe.duration() // timedelta(milliseconds=1))
    if timeframe_ms <= 0:
        raise ValueError("timeframe duration must be > 0")
    return timeframe_ms


def _validate_array(
    *,
    name: str,
    values: np.ndarray,
    expected_dtype: object,
    expected_size: int | None,
) -> int:
    """
    Validate one reporting array for dtype, one-dimensional shape, and optional size.

    Args:
        name: Field name for deterministic validation error messages.
        values: Candidate numpy array.
        expected_dtype: Required array dtype.
        expected_size: Required array size or `None` for baseline field.
    Returns:
        int: Validated size of the array.
    Assumptions:
        Arrays are already materialized as numpy ndarrays.
    Raises:
        ValueError: If one validation check fails.
    Side Effects:
        None.
    """
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D numpy array")
    if values.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {values.dtype}")
    size = int(values.shape[0])
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{name} size must be {expected_size}, got {size}")
    return size


__all__ = [
    "BacktestEquityCurveBuilderV1",
    "BacktestEquityCurveV1",
]
