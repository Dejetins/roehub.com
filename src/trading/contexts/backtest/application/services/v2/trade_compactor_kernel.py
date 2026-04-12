"""Pure Stage A compact-trade, internal exact-payload, and no-risk metric kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numba as nb
import numpy as np

from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1

from .contracts import (
    SIGNAL_CODE_LONG_V2,
    SIGNAL_CODE_NEUTRAL_V2,
    SIGNAL_CODE_SHORT_V2,
    StageACompactTradeV2,
    StageADirectionModeLiteralV2,
    StageANoRiskMetricsV2,
)

_BARS_PER_YEAR_EXEC_1M_V2 = 365.0 * 24.0 * 60.0
_DIRECTION_MODE_LONG_SHORT_CODE_V2 = 0
_DIRECTION_MODE_LONG_ONLY_CODE_V2 = 1
_DIRECTION_MODE_SHORT_ONLY_CODE_V2 = 2


def _readonly_compact_trade_array_v2(
    *,
    field_name: str,
    values: np.ndarray,
) -> np.ndarray:
    """
    Normalize one retained compact-trade array into a readonly one-dimensional NumPy vector.

    Args:
        field_name: Payload field name used in fail-fast validation errors.
        values: Candidate dense compact-trade vector for one retained finalist payload.
    Returns:
        np.ndarray: Contiguous readonly vector preserving the authored trade order.
    Assumptions:
        Stage A exact payloads trim compact-trade arrays down to one finalist row before this
        helper runs, so the returned vector must not retain references to a larger batch matrix.
    Raises:
        ValueError: If `values` is not one-dimensional.
    Side Effects:
        Marks the returned array as readonly.
    """
    normalized = np.ascontiguousarray(values)
    if normalized.ndim != 1:
        raise ValueError(f"StageACompactExactPayloadV2.{field_name} must be 1D")
    normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True, slots=True, eq=False)
class StageACompactExactPayloadV2:
    """
    Internal compact exact payload for one retained Stage A candidate.

    Args:
        entry_signal_idx: Dense compact-trade array of entry signal indexes.
        entry_exec_idx: Dense compact-trade array of entry execution indexes.
        direction: Dense compact-trade array of trade directions.
        sig_exit_signal_idx: Dense compact-trade array of signal-exit indexes with `-1` sentinel.
        sig_exit_exec_idx: Dense compact-trade array of signal-exit execution indexes.
        memory_shape_bucket: Additive contract marker describing the retained payload memory shape.
    Returns:
        None.
    Assumptions:
        This payload remains internal-only, keeps the risk path on compact trade arrays only, and
        summary-only launch surfaces must not materialize it as user-facing trades by default.
    Raises:
        ValueError: If array lengths drift or the payload widens beyond compact-trade arrays.
    Side Effects:
        Normalizes compact-trade arrays into readonly contiguous vectors.
    """

    entry_signal_idx: np.ndarray
    entry_exec_idx: np.ndarray
    direction: np.ndarray
    sig_exit_signal_idx: np.ndarray
    sig_exit_exec_idx: np.ndarray
    memory_shape_bucket: str = "compact_trade_arrays"
    _compact_trades_cache: tuple[StageACompactTradeV2, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """
        Validate one compact exact payload used behind the retained-candidate frontier.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Retained exact payloads hold only compact-trade arrays and must stay detached from any
            default user-visible materialization path.
        Raises:
            ValueError: If array lengths drift, values are invalid, or the memory-shape marker is
                not the compact-trade-only literal.
        Side Effects:
            Normalizes compact-trade arrays into readonly contiguous vectors.
        """
        entry_signal_idx = _readonly_compact_trade_array_v2(
            field_name="entry_signal_idx",
            values=np.asarray(self.entry_signal_idx, dtype=np.int64),
        )
        trade_count = int(entry_signal_idx.shape[0])
        entry_exec_idx = _readonly_compact_trade_array_v2(
            field_name="entry_exec_idx",
            values=np.asarray(self.entry_exec_idx, dtype=np.int64),
        )
        direction = _readonly_compact_trade_array_v2(
            field_name="direction",
            values=np.asarray(self.direction, dtype=np.int8),
        )
        sig_exit_signal_idx = _readonly_compact_trade_array_v2(
            field_name="sig_exit_signal_idx",
            values=np.asarray(self.sig_exit_signal_idx, dtype=np.int64),
        )
        sig_exit_exec_idx = _readonly_compact_trade_array_v2(
            field_name="sig_exit_exec_idx",
            values=np.asarray(self.sig_exit_exec_idx, dtype=np.int64),
        )
        if self.memory_shape_bucket != "compact_trade_arrays":
            raise ValueError(
                "StageACompactExactPayloadV2.memory_shape_bucket must be "
                "'compact_trade_arrays'"
            )
        if int(entry_exec_idx.shape[0]) != trade_count:
            raise ValueError(
                "StageACompactExactPayloadV2.entry_exec_idx must align with entry_signal_idx"
            )
        if int(direction.shape[0]) != trade_count:
            raise ValueError(
                "StageACompactExactPayloadV2.direction must align with entry_signal_idx"
            )
        if int(sig_exit_signal_idx.shape[0]) != trade_count:
            raise ValueError(
                "StageACompactExactPayloadV2.sig_exit_signal_idx must align with "
                "entry_signal_idx"
            )
        if int(sig_exit_exec_idx.shape[0]) != trade_count:
            raise ValueError(
                "StageACompactExactPayloadV2.sig_exit_exec_idx must align with entry_signal_idx"
            )
        if np.any(entry_signal_idx < 0):
            raise ValueError("StageACompactExactPayloadV2.entry_signal_idx must be >= 0")
        if np.any(entry_exec_idx < 0):
            raise ValueError("StageACompactExactPayloadV2.entry_exec_idx must be >= 0")
        if np.any(np.logical_and(direction != -1, direction != 1)):
            raise ValueError("StageACompactExactPayloadV2.direction must contain only -1 or 1")
        if np.any(sig_exit_signal_idx < -1):
            raise ValueError("StageACompactExactPayloadV2.sig_exit_signal_idx must be >= -1")
        if np.any(sig_exit_exec_idx < entry_exec_idx):
            raise ValueError(
                "StageACompactExactPayloadV2.sig_exit_exec_idx must stay >= entry_exec_idx"
            )
        object.__setattr__(self, "entry_signal_idx", entry_signal_idx)
        object.__setattr__(self, "entry_exec_idx", entry_exec_idx)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "sig_exit_signal_idx", sig_exit_signal_idx)
        object.__setattr__(self, "sig_exit_exec_idx", sig_exit_exec_idx)
        object.__setattr__(self, "memory_shape_bucket", "compact_trade_arrays")

    @property
    def trade_count(self) -> int:
        """
        Return the deterministic retained compact-trade count for this exact payload.

        Args:
            None.
        Returns:
            int: Count of compact trades stored in the retained payload.
        Assumptions:
            The retained payload keeps one trimmed row of compact-trade arrays only.
        Raises:
            None.
        Side Effects:
            None.
        """
        return int(self.entry_signal_idx.shape[0])

    @property
    def compact_trades(self) -> tuple[StageACompactTradeV2, ...]:
        """
        Materialize the compatibility compact-trade tuple view from retained compact arrays.

        Args:
            None.
        Returns:
            tuple[StageACompactTradeV2, ...]: Ordered compact trades for this retained payload.
        Assumptions:
            Stage B exact replay and tests may still consume compact trades as immutable objects,
            while the retained payload itself stays compact-trade-array-first for memory shape.
        Raises:
            None.
        Side Effects:
            Memoizes one immutable compatibility tuple on first access.
        """
        cached = self._compact_trades_cache
        if cached is not None:
            return cached
        materialized = tuple(
            StageACompactTradeV2(
                entry_signal_idx=int(self.entry_signal_idx[trade_index]),
                entry_exec_idx=int(self.entry_exec_idx[trade_index]),
                direction=int(self.direction[trade_index]),
                sig_exit_signal_idx=(
                    None
                    if int(self.sig_exit_signal_idx[trade_index]) < 0
                    else int(self.sig_exit_signal_idx[trade_index])
                ),
                sig_exit_exec_idx=int(self.sig_exit_exec_idx[trade_index]),
            )
            for trade_index in range(self.trade_count)
        )
        object.__setattr__(self, "_compact_trades_cache", materialized)
        return materialized


@dataclass(frozen=True, slots=True)
class _CompactTradeBatchV2:
    """
    Dense internal trade-list-first batch state for retained exact-candidate evaluation.

    Args:
        entry_signal_idx: Dense `[V, max_trades]` entry-signal indexes with `-1` padding.
        entry_exec_idx: Dense `[V, max_trades]` entry execution indexes.
        direction: Dense `[V, max_trades]` trade directions in `{-1, 0, 1}`.
        sig_exit_signal_idx: Dense `[V, max_trades]` signal-exit indexes with `-1` for sentinel.
        sig_exit_exec_idx: Dense `[V, max_trades]` exit execution indexes.
        trade_count: Per-row retained compact-trade counts.
    Returns:
        None.
    Assumptions:
        This batch state remains internal-only so trade-list-first extraction can stay dense and
        batch-friendly until shortlisted rows need an internal exact payload object.
    Raises:
        None.
    Side Effects:
        Holds dense NumPy arrays for later metric scoring and selective payload materialization.
    """

    entry_signal_idx: np.ndarray
    entry_exec_idx: np.ndarray
    direction: np.ndarray
    sig_exit_signal_idx: np.ndarray
    sig_exit_exec_idx: np.ndarray
    trade_count: np.ndarray

    def trade_row_at(self, *, row_index: int) -> tuple[StageACompactTradeV2, ...]:
        """
        Materialize one retained candidate's compact trade row from dense batch state.

        Args:
            row_index: Batch-local retained candidate index.
        Returns:
            tuple[StageACompactTradeV2, ...]: Ordered compact trades for the requested row.
        Assumptions:
            Dense arrays are already aligned by retained-candidate row and padded beyond
            `trade_count[row_index]`.
        Raises:
            ValueError: If `row_index` is outside the retained batch bounds.
        Side Effects:
            Materializes immutable compact-trade objects for one row only.
        """
        row_count = int(self.trade_count.shape[0])
        if row_index < 0 or row_index >= row_count:
            raise ValueError(
                f"row_index must stay within [0, {row_count}), got {row_index}"
            )
        retained_trade_count = int(self.trade_count[row_index])
        return tuple(
            StageACompactTradeV2(
                entry_signal_idx=int(self.entry_signal_idx[row_index, trade_index]),
                entry_exec_idx=int(self.entry_exec_idx[row_index, trade_index]),
                direction=int(self.direction[row_index, trade_index]),
                sig_exit_signal_idx=(
                    None
                    if int(self.sig_exit_signal_idx[row_index, trade_index]) < 0
                    else int(self.sig_exit_signal_idx[row_index, trade_index])
                ),
                sig_exit_exec_idx=int(self.sig_exit_exec_idx[row_index, trade_index]),
            )
            for trade_index in range(retained_trade_count)
        )

    def exact_payload_at(self, *, row_index: int) -> StageACompactExactPayloadV2:
        """
        Materialize one internal exact payload from dense trade-list-first batch state.

        Args:
            row_index: Batch-local retained candidate index.
        Returns:
            StageACompactExactPayloadV2: Internal exact payload for the requested retained row.
        Assumptions:
            Payload materialization should happen only for rows that survive shortlist ranking and
            must trim compact trade arrays so the retained risk path does not keep batch-shaped
            baggage.
        Raises:
            ValueError: Propagated if `row_index` is outside batch bounds.
        Side Effects:
            Allocates one immutable payload wrapper around trimmed compact-trade arrays for the
            selected row.
        """
        row_count = int(self.trade_count.shape[0])
        if row_index < 0 or row_index >= row_count:
            raise ValueError(
                f"row_index must stay within [0, {row_count}), got {row_index}"
            )
        retained_trade_count = int(self.trade_count[row_index])
        return StageACompactExactPayloadV2(
            entry_signal_idx=np.array(
                self.entry_signal_idx[row_index, :retained_trade_count],
                dtype=np.int64,
                copy=True,
            ),
            entry_exec_idx=np.array(
                self.entry_exec_idx[row_index, :retained_trade_count],
                dtype=np.int64,
                copy=True,
            ),
            direction=np.array(
                self.direction[row_index, :retained_trade_count],
                dtype=np.int8,
                copy=True,
            ),
            sig_exit_signal_idx=np.array(
                self.sig_exit_signal_idx[row_index, :retained_trade_count],
                dtype=np.int64,
                copy=True,
            ),
            sig_exit_exec_idx=np.array(
                self.sig_exit_exec_idx[row_index, :retained_trade_count],
                dtype=np.int64,
                copy=True,
            ),
        )

    def materialize_trade_list(self) -> tuple[tuple[StageACompactTradeV2, ...], ...]:
        """
        Materialize all compact trade rows from dense batch state.

        Args:
            None.
        Returns:
            tuple[tuple[StageACompactTradeV2, ...], ...]: One ordered trade row per retained
                candidate.
        Assumptions:
            This helper preserves the public compact-trade API while dense kernels own the hot
            extraction work internally.
        Raises:
            None.
        Side Effects:
            Allocates immutable compact-trade tuples for every retained candidate row.
        """
        return tuple(
            self.trade_row_at(row_index=row_index)
            for row_index in range(int(self.trade_count.shape[0]))
        )

    def materialize_exact_payloads(self) -> tuple[StageACompactExactPayloadV2, ...]:
        """
        Materialize all internal exact payloads from dense batch state.

        Args:
            None.
        Returns:
            tuple[StageACompactExactPayloadV2, ...]: One internal exact payload per retained
                candidate row.
        Assumptions:
            Public callers that still need all payloads can reuse the same dense trade-list-first
            extraction without re-running the hot path.
        Raises:
            None.
        Side Effects:
            Allocates immutable payload wrappers for every retained candidate row.
        """
        return tuple(
            self.exact_payload_at(row_index=row_index)
            for row_index in range(int(self.trade_count.shape[0]))
        )


@nb.njit(cache=True)
def _direction_allowed_code_v2(direction: int, direction_mode_code: int) -> bool:
    """
    Resolve whether one signal direction may open a trade for the encoded direction mode.

    Args:
        direction: Raw signal direction (`-1` or `1`).
        direction_mode_code: Encoded direction-mode literal used inside batch kernels.
    Returns:
        bool: `True` when the signal may open a new trade.
    Assumptions:
        The encoded mode preserves the same semantics as the public direction-mode literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    if direction_mode_code == _DIRECTION_MODE_LONG_ONLY_CODE_V2:
        return direction == 1
    if direction_mode_code == _DIRECTION_MODE_SHORT_ONLY_CODE_V2:
        return direction == -1
    return True


@nb.njit(cache=True)
def _trade_sharpe_kernel_v2(
    trade_count: int,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: int,
) -> float:
    """
    Compute Stage A trade-level Sharpe inside Numba batch metric kernels.

    Args:
        trade_count: Number of closed trades in one retained candidate row.
        sum_trade_return: Sum of per-trade returns after fees.
        sum_trade_return_squared: Sum of squared per-trade returns after fees.
        bars_per_year_exec: Annualization denominator in execution bars.
        sentinel_index: Total execution bars in the retained replay window.
    Returns:
        float: Deterministic trade-level Sharpe ratio.
    Assumptions:
        Batch no-risk scoring must match the scalar Stage A metric semantics exactly.
    Raises:
        None.
    Side Effects:
        None.
    """
    if trade_count <= 1:
        return 0.0
    mean_trade_return = sum_trade_return / float(trade_count)
    variance = (sum_trade_return_squared / float(trade_count)) - (
        mean_trade_return * mean_trade_return
    )
    if variance <= 0.0:
        return 0.0
    years = float(sentinel_index) / float(bars_per_year_exec)
    if years <= 0.0:
        years = 1.0
    trades_per_year = float(trade_count) / years
    return (mean_trade_return / math.sqrt(variance)) * math.sqrt(trades_per_year)


@nb.njit(parallel=True, cache=True)
def _build_compact_trade_batch_kernel_v2(
    *,
    final_signal_i8: np.ndarray,
    entry_exec_idx_i64: np.ndarray,
    sentinel_index: int,
    direction_mode_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build dense trade-list-first rows for one retained exact-candidate matrix in parallel.

    Args:
        final_signal_i8: Canonical `int8` retained `final_signal[V, T_signal]` matrix.
        entry_exec_idx_i64: Canonical local entry indexes aligned to `T_signal`.
        sentinel_index: Local execution timeline length.
        direction_mode_code: Encoded direction-mode literal.
    Returns:
        tuple[np.ndarray, ...]: Dense trade fields plus per-row trade counts.
    Assumptions:
        Each retained row is independent, so dense trade extraction can parallelize across rows
        while preserving the authored signal timeline order inside each row.
    Raises:
        None.
    Side Effects:
        Allocates dense trade-field arrays for the retained batch.
    """
    row_count = int(final_signal_i8.shape[0])
    signal_count = int(final_signal_i8.shape[1])
    entry_signal_idx = np.full((row_count, signal_count), -1, dtype=np.int64)
    entry_exec_idx = np.zeros((row_count, signal_count), dtype=np.int64)
    direction = np.zeros((row_count, signal_count), dtype=np.int8)
    sig_exit_signal_idx = np.full((row_count, signal_count), -1, dtype=np.int64)
    sig_exit_exec_idx = np.zeros((row_count, signal_count), dtype=np.int64)
    trade_count = np.zeros(row_count, dtype=np.int64)

    for row_index in nb.prange(row_count):
        current_direction = 0
        current_entry_signal_idx = 0
        current_entry_exec_idx = 0
        row_trade_count = 0
        for signal_bar_idx in range(signal_count):
            raw_direction = int(final_signal_i8[row_index, signal_bar_idx])
            if raw_direction == SIGNAL_CODE_NEUTRAL_V2:
                continue
            mapped_entry_exec_idx = int(entry_exec_idx_i64[signal_bar_idx])
            if mapped_entry_exec_idx >= sentinel_index:
                break
            if current_direction == 0:
                if _direction_allowed_code_v2(raw_direction, direction_mode_code):
                    current_direction = raw_direction
                    current_entry_signal_idx = signal_bar_idx
                    current_entry_exec_idx = mapped_entry_exec_idx
                continue
            if raw_direction == current_direction:
                continue
            entry_signal_idx[row_index, row_trade_count] = current_entry_signal_idx
            entry_exec_idx[row_index, row_trade_count] = current_entry_exec_idx
            direction[row_index, row_trade_count] = current_direction
            sig_exit_signal_idx[row_index, row_trade_count] = signal_bar_idx
            sig_exit_exec_idx[row_index, row_trade_count] = mapped_entry_exec_idx
            row_trade_count += 1
            if _direction_allowed_code_v2(raw_direction, direction_mode_code):
                current_direction = raw_direction
                current_entry_signal_idx = signal_bar_idx
                current_entry_exec_idx = mapped_entry_exec_idx
            else:
                current_direction = 0
        if current_direction != 0:
            entry_signal_idx[row_index, row_trade_count] = current_entry_signal_idx
            entry_exec_idx[row_index, row_trade_count] = current_entry_exec_idx
            direction[row_index, row_trade_count] = current_direction
            sig_exit_signal_idx[row_index, row_trade_count] = -1
            sig_exit_exec_idx[row_index, row_trade_count] = sentinel_index
            row_trade_count += 1
        trade_count[row_index] = row_trade_count

    return (
        entry_signal_idx,
        entry_exec_idx,
        direction,
        sig_exit_signal_idx,
        sig_exit_exec_idx,
        trade_count,
    )


@nb.njit(parallel=True, cache=True)
def _batch_no_risk_metrics_kernel_v2(
    *,
    trade_count_i64: np.ndarray,
    entry_exec_idx_i64: np.ndarray,
    direction_i8: np.ndarray,
    sig_exit_exec_idx_i64: np.ndarray,
    exec_open_f64: np.ndarray,
    exec_close_f64: np.ndarray,
    sentinel_index: int,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: bool,
    use_profit_lock: bool,
    bars_per_year_exec: float,
    close_on_end: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute deterministic Stage A no-risk metrics for one retained trade batch in parallel.

    Args:
        trade_count_i64: Per-row compact-trade counts.
        entry_exec_idx_i64: Dense entry execution indexes.
        direction_i8: Dense trade directions.
        sig_exit_exec_idx_i64: Dense signal-exit execution indexes.
        exec_open_f64: Local execution open prices.
        exec_close_f64: Local execution close prices.
        sentinel_index: Local execution timeline length.
        init_cash_quote: Initial quote balance from execution settings.
        fixed_quote: Fixed quote allocation when fixed sizing is active.
        fee_rate: Decimal per-side fee rate.
        slippage_rate: Decimal slippage rate.
        safe_profit_percent: Locked-profit percentage for profit-lock sizing.
        use_fixed_quote: Whether entry sizing uses `fixed_quote`.
        use_profit_lock: Whether profitable trades lock safe quote.
        bars_per_year_exec: Annualization denominator in execution bars.
        close_on_end: Whether sentinel trades close on the last execution close.
    Returns:
        tuple[np.ndarray, ...]: Dense metric columns aligned to retained candidate row order.
    Assumptions:
        Each retained candidate is independent, so no-risk scoring can parallelize across rows
        after dense trade-list-first extraction.
    Raises:
        None.
    Side Effects:
        Allocates dense metric arrays for the retained batch.
    """
    row_count = int(trade_count_i64.shape[0])
    total_return_pct = np.zeros(row_count, dtype=np.float64)
    max_drawdown_pct = np.zeros(row_count, dtype=np.float64)
    return_over_max_drawdown = np.zeros(row_count, dtype=np.float64)
    profit_factor = np.zeros(row_count, dtype=np.float64)
    trade_count = np.zeros(row_count, dtype=np.int64)
    sharpe_trades = np.zeros(row_count, dtype=np.float64)
    win_rate_pct = np.zeros(row_count, dtype=np.float64)
    avg_trade_ret_pct = np.zeros(row_count, dtype=np.float64)
    avg_trade_exec_bars = np.zeros(row_count, dtype=np.float64)
    exposure_pct = np.zeros(row_count, dtype=np.float64)

    for row_index in nb.prange(row_count):
        available_quote = init_cash_quote
        safe_quote = 0.0
        equity = init_cash_quote
        peak_equity = equity
        row_max_drawdown_pct = 0.0
        gross_profit_quote = 0.0
        gross_loss_quote = 0.0
        closed_trade_count = 0
        win_count = 0
        sum_trade_return = 0.0
        sum_trade_return_squared = 0.0
        total_trade_return_pct = 0.0
        total_trade_exec_bars = 0.0
        exposure_bars = 0.0

        for trade_index in range(int(trade_count_i64[row_index])):
            entry_exec_idx = int(entry_exec_idx_i64[row_index, trade_index])
            if entry_exec_idx >= sentinel_index:
                continue

            signal_exit_exec_idx = int(sig_exit_exec_idx_i64[row_index, trade_index])
            if signal_exit_exec_idx < sentinel_index:
                exit_exec_idx = signal_exit_exec_idx
                exit_price_raw = exec_open_f64[exit_exec_idx]
            elif close_on_end and sentinel_index > 0:
                exit_exec_idx = sentinel_index - 1
                exit_price_raw = exec_close_f64[exit_exec_idx]
            else:
                continue

            if available_quote <= 0.0:
                continue
            quote_amount = available_quote
            if use_fixed_quote and fixed_quote < quote_amount:
                quote_amount = fixed_quote
            if quote_amount <= 0.0:
                continue

            trade_direction = int(direction_i8[row_index, trade_index])
            entry_price_raw = exec_open_f64[entry_exec_idx]
            entry_fill_price = entry_price_raw
            if trade_direction == 1:
                entry_fill_price *= 1.0 + slippage_rate
            else:
                entry_fill_price *= 1.0 - slippage_rate
            exit_fill_price = exit_price_raw
            if trade_direction == -1:
                exit_fill_price *= 1.0 + slippage_rate
            else:
                exit_fill_price *= 1.0 - slippage_rate

            qty_base = quote_amount / entry_fill_price
            entry_fee_quote = quote_amount * fee_rate
            available_quote -= quote_amount + entry_fee_quote

            exit_quote_amount = qty_base * exit_fill_price
            exit_fee_quote = exit_quote_amount * fee_rate
            if trade_direction == 1:
                gross_pnl_quote = exit_quote_amount - quote_amount
            else:
                gross_pnl_quote = quote_amount - exit_quote_amount
            available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
            net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote

            if use_profit_lock and net_pnl_quote > 0.0:
                locked_profit_quote = net_pnl_quote * (safe_profit_percent / 100.0)
                available_quote -= locked_profit_quote
                safe_quote += locked_profit_quote

            equity = available_quote + safe_quote
            if equity > peak_equity:
                peak_equity = equity
            elif peak_equity > 0.0:
                drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
                if drawdown_pct > row_max_drawdown_pct:
                    row_max_drawdown_pct = drawdown_pct

            trade_return_pct = (net_pnl_quote / quote_amount) * 100.0
            trade_return = net_pnl_quote / quote_amount
            bars_held = float(exit_exec_idx - entry_exec_idx)
            if bars_held < 0.0:
                bars_held = 0.0
            closed_trade_count += 1
            if net_pnl_quote > 0.0:
                win_count += 1
                gross_profit_quote += net_pnl_quote
            elif net_pnl_quote < 0.0:
                gross_loss_quote += abs(net_pnl_quote)
            sum_trade_return += trade_return
            sum_trade_return_squared += trade_return * trade_return
            total_trade_return_pct += trade_return_pct
            total_trade_exec_bars += bars_held
            exposure_bars += bars_held

        row_total_return_pct = ((equity / init_cash_quote) - 1.0) * 100.0
        total_return_pct[row_index] = row_total_return_pct
        max_drawdown_pct[row_index] = row_max_drawdown_pct
        trade_count[row_index] = closed_trade_count
        if closed_trade_count > 0:
            win_rate_pct[row_index] = (float(win_count) / float(closed_trade_count)) * 100.0
            avg_trade_ret_pct[row_index] = total_trade_return_pct / float(closed_trade_count)
            avg_trade_exec_bars[row_index] = total_trade_exec_bars / float(
                closed_trade_count
            )
        if gross_loss_quote > 0.0:
            profit_factor[row_index] = gross_profit_quote / gross_loss_quote
        elif gross_profit_quote > 0.0:
            profit_factor[row_index] = np.inf
        else:
            profit_factor[row_index] = 0.0
        if row_max_drawdown_pct > 0.0:
            return_over_max_drawdown[row_index] = (
                row_total_return_pct / row_max_drawdown_pct
            )
        elif row_total_return_pct > 0.0:
            return_over_max_drawdown[row_index] = np.inf
        else:
            return_over_max_drawdown[row_index] = 0.0
        exposure_pct[row_index] = (
            (exposure_bars / float(sentinel_index)) * 100.0 if sentinel_index > 0 else 0.0
        )
        sharpe_trades[row_index] = _trade_sharpe_kernel_v2(
            closed_trade_count,
            sum_trade_return,
            sum_trade_return_squared,
            bars_per_year_exec,
            sentinel_index,
        )

    return (
        total_return_pct,
        max_drawdown_pct,
        return_over_max_drawdown,
        profit_factor,
        trade_count,
        sharpe_trades,
        win_rate_pct,
        avg_trade_ret_pct,
        avg_trade_exec_bars,
        exposure_pct,
    )


def _direction_mode_code_v2(*, direction_mode: StageADirectionModeLiteralV2) -> int:
    """
    Encode one validated direction-mode literal for dense internal batch kernels.

    Args:
        direction_mode: Canonical lower-case direction-mode literal.
    Returns:
        int: Dense kernel code representing the same direction-mode semantics.
    Assumptions:
        Encoded direction modes stay internal-only and never replace the public literal surface.
    Raises:
        ValueError: If the validated direction mode is unsupported.
    Side Effects:
        None.
    """
    if direction_mode == "long-short":
        return _DIRECTION_MODE_LONG_SHORT_CODE_V2
    if direction_mode == "long-only":
        return _DIRECTION_MODE_LONG_ONLY_CODE_V2
    if direction_mode == "short-only":
        return _DIRECTION_MODE_SHORT_ONLY_CODE_V2
    raise ValueError(
        "direction_mode must be one of ('long-only', 'short-only', 'long-short')"
    )


def build_compact_trade_batch_v2(
    *,
    final_signal: np.ndarray,
    bar_close_1m_idx: np.ndarray,
    sentinel_index: int,
    direction_mode: str = "long-short",
) -> _CompactTradeBatchV2:
    """
    Build dense internal trade-list-first batch state for retained exact candidates.

    Args:
        final_signal: Aggregated retained-candidate signal matrix shaped `[V, T_signal]`.
        bar_close_1m_idx: Local execution-timeline close mapping for the same `T_signal` bars.
        sentinel_index: Local execution timeline length used as the sentinel fallback.
        direction_mode: Strategy direction policy (`long-only`, `short-only`, `long-short`).
    Returns:
        _CompactTradeBatchV2: Dense internal batch state aligned to retained candidate row order.
    Assumptions:
        Trade-list-first stays internal-only and exists to make retained exact payload work batch-
        friendly before any shortlisted rows need payload objects.
    Raises:
        ValueError: Propagated when signal or mapping inputs drift from compact-trade contracts.
    Side Effects:
        May trigger Numba compilation on first use.
    """
    normalized_signal = _normalize_final_signal_matrix_v2(values=final_signal)
    normalized_mapping = _normalize_bar_close_1m_idx_v2(
        values=bar_close_1m_idx,
        expected_length=normalized_signal.shape[1],
        sentinel_index=sentinel_index,
    )
    resolved_direction_mode = _validate_direction_mode_v2(direction_mode=direction_mode)
    direction_mode_code = _direction_mode_code_v2(direction_mode=resolved_direction_mode)
    entry_exec_idx = np.minimum(normalized_mapping + 1, sentinel_index).astype(
        np.int64,
        copy=False,
    )
    (
        entry_signal_idx,
        entry_exec_idx_by_trade,
        direction,
        sig_exit_signal_idx,
        sig_exit_exec_idx,
        trade_count,
    ) = _build_compact_trade_batch_kernel_v2(
        final_signal_i8=np.ascontiguousarray(normalized_signal, dtype=np.int8),
        entry_exec_idx_i64=np.ascontiguousarray(entry_exec_idx, dtype=np.int64),
        sentinel_index=sentinel_index,
        direction_mode_code=direction_mode_code,
    )
    return _CompactTradeBatchV2(
        entry_signal_idx=entry_signal_idx,
        entry_exec_idx=entry_exec_idx_by_trade,
        direction=direction,
        sig_exit_signal_idx=sig_exit_signal_idx,
        sig_exit_exec_idx=sig_exit_exec_idx,
        trade_count=trade_count,
    )


def compute_no_risk_metrics_for_trade_batch_v2(
    *,
    compact_trade_batch: _CompactTradeBatchV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    sentinel_index: int,
    execution_params: ExecutionParamsV1,
    bars_per_year_exec: float = _BARS_PER_YEAR_EXEC_1M_V2,
    close_on_end: bool = True,
) -> tuple[StageANoRiskMetricsV2, ...]:
    """
    Compute deterministic Stage A no-risk metrics for one dense retained trade batch.

    Args:
        compact_trade_batch: Dense internal trade-list-first batch state.
        exec_open: Local execution-timeline open prices.
        exec_close: Local execution-timeline close prices.
        sentinel_index: Local execution timeline length (`T_exec`).
        execution_params: Resolved execution defaults for sizing, fees, and slippage.
        bars_per_year_exec: Annualization denominator in execution bars for `sharpe_trades`.
        close_on_end: Whether open trades close on the last execution close when no signal exit
            exists.
    Returns:
        tuple[StageANoRiskMetricsV2, ...]: One no-risk metric payload per retained candidate row.
    Assumptions:
        Dense trade-list-first extraction already validated the retained frontier, so this helper
        can batch no-risk scoring across candidates without materializing user-facing trades.
    Raises:
        ValueError:
            If execution arrays drift from `sentinel_index` or annualization denominator is
            invalid.
    Side Effects:
        May trigger Numba compilation on first use.
    """
    normalized_exec_open = _normalize_execution_prices_v2(
        field_name="exec_open",
        values=exec_open,
        sentinel_index=sentinel_index,
    )
    normalized_exec_close = _normalize_execution_prices_v2(
        field_name="exec_close",
        values=exec_close,
        sentinel_index=sentinel_index,
    )
    if bars_per_year_exec <= 0.0:
        raise ValueError("bars_per_year_exec must be > 0")
    use_fixed_quote = execution_params.sizing_mode == "fixed_quote"
    use_profit_lock = execution_params.sizing_mode == "strategy_compound_profit_lock"
    metric_columns = _batch_no_risk_metrics_kernel_v2(
        trade_count_i64=np.ascontiguousarray(compact_trade_batch.trade_count, dtype=np.int64),
        entry_exec_idx_i64=np.ascontiguousarray(
            compact_trade_batch.entry_exec_idx,
            dtype=np.int64,
        ),
        direction_i8=np.ascontiguousarray(compact_trade_batch.direction, dtype=np.int8),
        sig_exit_exec_idx_i64=np.ascontiguousarray(
            compact_trade_batch.sig_exit_exec_idx,
            dtype=np.int64,
        ),
        exec_open_f64=np.ascontiguousarray(normalized_exec_open, dtype=np.float64),
        exec_close_f64=np.ascontiguousarray(normalized_exec_close, dtype=np.float64),
        sentinel_index=sentinel_index,
        init_cash_quote=float(execution_params.init_cash_quote),
        fixed_quote=float(execution_params.fixed_quote),
        fee_rate=float(execution_params.fee_rate),
        slippage_rate=float(execution_params.slippage_rate),
        safe_profit_percent=float(execution_params.safe_profit_percent),
        use_fixed_quote=use_fixed_quote,
        use_profit_lock=use_profit_lock,
        bars_per_year_exec=bars_per_year_exec,
        close_on_end=close_on_end,
    )
    (
        total_return_pct,
        max_drawdown_pct,
        return_over_max_drawdown,
        profit_factor,
        trade_count,
        sharpe_trades,
        win_rate_pct,
        avg_trade_ret_pct,
        avg_trade_exec_bars,
        exposure_pct,
    ) = metric_columns
    row_count = int(compact_trade_batch.trade_count.shape[0])
    return tuple(
        StageANoRiskMetricsV2(
            total_return_pct=float(total_return_pct[row_index]),
            max_drawdown_pct=float(max_drawdown_pct[row_index]),
            return_over_max_drawdown=float(return_over_max_drawdown[row_index]),
            profit_factor=float(profit_factor[row_index]),
            trade_count=int(trade_count[row_index]),
            sharpe_trades=float(sharpe_trades[row_index]),
            win_rate_pct=float(win_rate_pct[row_index]),
            avg_trade_ret_pct=float(avg_trade_ret_pct[row_index]),
            avg_trade_exec_bars=float(avg_trade_exec_bars[row_index]),
            exposure_pct=float(exposure_pct[row_index]),
        )
        for row_index in range(row_count)
    )


def build_compact_trade_list_v2(
    *,
    final_signal: np.ndarray,
    bar_close_1m_idx: np.ndarray,
    sentinel_index: int,
    direction_mode: str = "long-short",
) -> tuple[tuple[StageACompactTradeV2, ...], ...]:
    """
    Build deterministic Stage A compact trades from `final_signal` and local `bar_close_1m_idx`.

    Args:
        final_signal: Aggregated signal matrix shaped `[V, T_signal]` or one row `[T_signal]`
            with value set `{-1, 0, 1}`.
        bar_close_1m_idx: Local execution-timeline close mapping for the same `T_signal` bars.
        sentinel_index: Local execution timeline length used as the sentinel fallback.
        direction_mode: Strategy direction policy (`long-only`, `short-only`, `long-short`).
    Returns:
        tuple[tuple[StageACompactTradeV2, ...], ...]: One ordered compact-trade tuple per
            variant row.
    Assumptions:
        `neutral` bars do not close positions; only opposite confirmations create signal exits.
        One-side modes treat forbidden opposite signals as exit-only events.
    Raises:
        ValueError: If shapes drift, indexes are invalid, or direction mode is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    return build_compact_trade_batch_v2(
        final_signal=final_signal,
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        direction_mode=direction_mode,
    ).materialize_trade_list()


def build_compact_exact_payloads_v2(
    *,
    final_signal: np.ndarray,
    bar_close_1m_idx: np.ndarray,
    sentinel_index: int,
    direction_mode: str = "long-short",
) -> tuple[StageACompactExactPayloadV2, ...]:
    """
    Build internal compact exact payloads for retained candidates only.

    Args:
        final_signal: Aggregated retained-candidate signal matrix shaped `[V, T_signal]`.
        bar_close_1m_idx: Local execution-timeline close mapping for the same `T_signal` bars.
        sentinel_index: Local execution timeline length used as the sentinel fallback.
        direction_mode: Strategy direction policy (`long-only`, `short-only`, `long-short`).
    Returns:
        tuple[StageACompactExactPayloadV2, ...]: One internal compact payload per retained
            candidate row.
    Assumptions:
        The payload stays internal and exists only for retained exact candidates after prefilter;
        summary-only launch outputs remain unchanged.
    Raises:
        ValueError: Propagated when compact-trade construction input shapes or indexes drift.
    Side Effects:
        None.
    """
    return build_compact_trade_batch_v2(
        final_signal=final_signal,
        bar_close_1m_idx=bar_close_1m_idx,
        sentinel_index=sentinel_index,
        direction_mode=direction_mode,
    ).materialize_exact_payloads()


def compute_no_risk_metrics_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    sentinel_index: int,
    execution_params: ExecutionParamsV1,
    bars_per_year_exec: float = _BARS_PER_YEAR_EXEC_1M_V2,
    close_on_end: bool = True,
) -> StageANoRiskMetricsV2:
    """
    Compute deterministic Stage A no-risk metrics over compact trades without TP/SL replay.

    Args:
        compact_trades: Ordered compact trades for one variant.
        exec_open: Local execution-timeline open prices.
        exec_close: Local execution-timeline close prices.
        sentinel_index: Local execution timeline length (`T_exec`).
        execution_params: Resolved execution defaults for sizing, fees, and slippage.
        bars_per_year_exec: Annualization denominator in execution bars for `sharpe_trades`.
        close_on_end: Whether open trades close on the last execution close when no signal exit
            exists.
    Returns:
        StageANoRiskMetricsV2: Deterministic shortlist-ready no-risk metrics.
    Assumptions:
        The metric kernel works over compact trades only, uses `entry_exec_idx`,
        `sig_exit_exec_idx`, and never depends on Stage B risk artifacts.
    Raises:
        ValueError:
            If execution arrays drift from `sentinel_index`, annualization denominator is
            invalid, or one trade index is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    normalized_exec_open = _normalize_execution_prices_v2(
        field_name="exec_open",
        values=exec_open,
        sentinel_index=sentinel_index,
    )
    normalized_exec_close = _normalize_execution_prices_v2(
        field_name="exec_close",
        values=exec_close,
        sentinel_index=sentinel_index,
    )
    if bars_per_year_exec <= 0.0:
        raise ValueError("bars_per_year_exec must be > 0")

    available_quote = float(execution_params.init_cash_quote)
    safe_quote = 0.0
    equity = float(execution_params.init_cash_quote)
    peak_equity = equity
    max_drawdown_pct = 0.0
    gross_profit_quote = 0.0
    gross_loss_quote = 0.0
    trade_count = 0
    win_count = 0
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    total_trade_return_pct = 0.0
    total_trade_exec_bars = 0.0
    exposure_bars = 0.0

    for trade in compact_trades:
        _validate_trade_indexes_v2(trade=trade, sentinel_index=sentinel_index)
        if trade.entry_exec_idx >= sentinel_index:
            continue
        exit_exec_idx, exit_price_raw = _resolve_trade_exit_v2(
            trade=trade,
            exec_open=normalized_exec_open,
            exec_close=normalized_exec_close,
            sentinel_index=sentinel_index,
            close_on_end=close_on_end,
        )
        if exit_exec_idx is None or exit_price_raw is None:
            continue

        quote_amount = _entry_quote_amount_v2(
            available_quote=available_quote,
            execution_params=execution_params,
        )
        if quote_amount <= 0.0:
            continue

        entry_price_raw = float(normalized_exec_open[trade.entry_exec_idx])
        entry_fill_price = _fill_price_from_mark_v2(
            price=entry_price_raw,
            slippage_rate=execution_params.slippage_rate,
            is_buy=trade.direction == 1,
        )
        exit_fill_price = _fill_price_from_mark_v2(
            price=exit_price_raw,
            slippage_rate=execution_params.slippage_rate,
            is_buy=trade.direction == -1,
        )
        qty_base = quote_amount / entry_fill_price
        entry_fee_quote = quote_amount * execution_params.fee_rate
        available_quote -= quote_amount + entry_fee_quote

        exit_quote_amount = qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * execution_params.fee_rate
        if trade.direction == 1:
            gross_pnl_quote = exit_quote_amount - quote_amount
        else:
            gross_pnl_quote = quote_amount - exit_quote_amount
        available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote

        if (
            execution_params.sizing_mode == "strategy_compound_profit_lock"
            and net_pnl_quote > 0.0
        ):
            locked_profit_quote = net_pnl_quote * (execution_params.safe_profit_percent / 100.0)
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote

        equity = available_quote + safe_quote
        if equity > peak_equity:
            peak_equity = equity
        elif peak_equity > 0.0:
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        trade_return_pct = (net_pnl_quote / quote_amount) * 100.0
        trade_return = net_pnl_quote / quote_amount
        bars_held = float(max(exit_exec_idx - trade.entry_exec_idx, 0))
        trade_count += 1
        if net_pnl_quote > 0.0:
            win_count += 1
            gross_profit_quote += net_pnl_quote
        elif net_pnl_quote < 0.0:
            gross_loss_quote += abs(net_pnl_quote)
        sum_trade_return += trade_return
        sum_trade_return_squared += trade_return * trade_return
        total_trade_return_pct += trade_return_pct
        total_trade_exec_bars += bars_held
        exposure_bars += bars_held

    total_return_pct = ((equity / float(execution_params.init_cash_quote)) - 1.0) * 100.0
    if trade_count > 0:
        win_rate_pct = (float(win_count) / float(trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(trade_count)
        avg_trade_exec_bars = total_trade_exec_bars / float(trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0

    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = float("inf")
    else:
        return_over_max_drawdown = 0.0

    exposure_pct = (
        (exposure_bars / float(sentinel_index)) * 100.0 if sentinel_index > 0 else 0.0
    )
    sharpe_trades = _trade_sharpe_v2(
        trade_count=trade_count,
        sum_trade_return=sum_trade_return,
        sum_trade_return_squared=sum_trade_return_squared,
        bars_per_year_exec=bars_per_year_exec,
        sentinel_index=sentinel_index,
    )
    return StageANoRiskMetricsV2(
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        return_over_max_drawdown=return_over_max_drawdown,
        profit_factor=profit_factor,
        trade_count=trade_count,
        sharpe_trades=sharpe_trades,
        win_rate_pct=win_rate_pct,
        avg_trade_ret_pct=avg_trade_ret_pct,
        avg_trade_exec_bars=avg_trade_exec_bars,
        exposure_pct=exposure_pct,
    )


def no_risk_metrics_to_ranking_payload_v2(
    *,
    metrics: StageANoRiskMetricsV2,
) -> Mapping[str, float]:
    """
    Convert Stage A no-risk metrics into deterministic ranking payload keys.

    Args:
        metrics: No-risk metric payload for one variant.
    Returns:
        Mapping[str, float]: Immutable mapping keyed by stable ranking/summary literals.
    Assumptions:
        Ranking keys remain explicit and additive until R6-04 top-N materialization lands.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    return MappingProxyType(
        {
            "total_return_pct": metrics.total_return_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "return_over_max_drawdown": metrics.return_over_max_drawdown,
            "profit_factor": metrics.profit_factor,
            "sharpe_trades": metrics.sharpe_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "trade_count": float(metrics.trade_count),
            "avg_trade_ret_pct": metrics.avg_trade_ret_pct,
            "avg_trade_exec_bars": metrics.avg_trade_exec_bars,
            "exposure_pct": metrics.exposure_pct,
        }
    )


def _trade_sharpe_v2(
    *,
    trade_count: int,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: int,
) -> float:
    """
    Compute notebook-style Sharpe over trade returns with execution-bar annualization.

    Args:
        trade_count: Number of closed trades in the replay.
        sum_trade_return: Sum of per-trade returns after fees.
        sum_trade_return_squared: Sum of squared per-trade returns after fees.
        bars_per_year_exec: Annualization denominator in execution bars.
        sentinel_index: Total execution bars in the replay window.
    Returns:
        float: Deterministic trade-level Sharpe ratio.
    Assumptions:
        Sharpe uses `trades_per_year`, not bar returns, to match notebook semantics.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if trade_count <= 1:
        return 0.0
    mean_trade_return = sum_trade_return / float(trade_count)
    variance = (sum_trade_return_squared / float(trade_count)) - (
        mean_trade_return * mean_trade_return
    )
    if variance <= 0.0:
        return 0.0
    years = float(sentinel_index) / float(bars_per_year_exec)
    if years <= 0.0:
        years = 1.0
    trades_per_year = float(trade_count) / years
    return (mean_trade_return / math.sqrt(variance)) * math.sqrt(trades_per_year)


def _build_compact_trade_row_v2(
    *,
    final_signal_row: np.ndarray,
    entry_exec_idx: np.ndarray,
    sentinel_index: int,
    direction_mode: StageADirectionModeLiteralV2,
) -> tuple[StageACompactTradeV2, ...]:
    """
    Build one compact-trade row for a single Stage A variant.

    Args:
        final_signal_row: One-dimensional aggregated signal row.
        entry_exec_idx: Local execution entry indexes derived from `bar_close_1m_idx + 1`.
        sentinel_index: Local execution timeline length.
        direction_mode: Strategy direction policy.
    Returns:
        tuple[StageACompactTradeV2, ...]: Ordered compact trades for the variant.
    Assumptions:
        `entry_exec_idx` is monotone because `bar_close_1m_idx` follows deterministic mappings.
    Raises:
        ValueError: If input arrays drift in length.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if final_signal_row.shape[0] != entry_exec_idx.shape[0]:
        raise ValueError("final_signal_row length must match entry_exec_idx length")

    compact_trades: list[StageACompactTradeV2] = []
    current_direction = 0
    current_entry_signal_idx = 0
    current_entry_exec_idx = 0
    for signal_bar_idx, raw_direction in enumerate(final_signal_row):
        if raw_direction == SIGNAL_CODE_NEUTRAL_V2:
            continue
        mapped_entry_exec_idx = int(entry_exec_idx[signal_bar_idx])
        if mapped_entry_exec_idx >= sentinel_index:
            break
        if current_direction == 0:
            if _direction_allowed_v2(direction=int(raw_direction), direction_mode=direction_mode):
                current_direction = int(raw_direction)
                current_entry_signal_idx = signal_bar_idx
                current_entry_exec_idx = mapped_entry_exec_idx
            continue
        if int(raw_direction) == current_direction:
            continue
        compact_trades.append(
            StageACompactTradeV2(
                entry_signal_idx=current_entry_signal_idx,
                entry_exec_idx=current_entry_exec_idx,
                direction=current_direction,
                sig_exit_signal_idx=signal_bar_idx,
                sig_exit_exec_idx=mapped_entry_exec_idx,
            )
        )
        if _direction_allowed_v2(direction=int(raw_direction), direction_mode=direction_mode):
            current_direction = int(raw_direction)
            current_entry_signal_idx = signal_bar_idx
            current_entry_exec_idx = mapped_entry_exec_idx
        else:
            current_direction = 0
    if current_direction != 0:
        compact_trades.append(
            StageACompactTradeV2(
                entry_signal_idx=current_entry_signal_idx,
                entry_exec_idx=current_entry_exec_idx,
                direction=current_direction,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=sentinel_index,
            )
        )
    return tuple(compact_trades)


def _normalize_final_signal_matrix_v2(*, values: np.ndarray) -> np.ndarray:
    """
    Normalize one `final_signal` payload into a deterministic two-dimensional `np.int8` matrix.

    Args:
        values: Candidate `final_signal` array.
    Returns:
        np.ndarray: Canonical `[V, T_signal]` compact signal matrix.
    Assumptions:
        One-dimensional inputs represent exactly one variant row.
    Raises:
        ValueError: If shape is invalid or values leave the `{-1, 0, 1}` contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    normalized = np.asarray(values, dtype=np.int8)
    if normalized.ndim == 1:
        normalized = normalized[np.newaxis, :]
    if normalized.ndim != 2:
        raise ValueError("final_signal must be a 1D or 2D array")
    invalid_mask = ~np.isin(
        normalized,
        (SIGNAL_CODE_SHORT_V2, SIGNAL_CODE_NEUTRAL_V2, SIGNAL_CODE_LONG_V2),
    )
    if bool(np.any(invalid_mask)):
        invalid_values = tuple(int(value) for value in np.unique(normalized[invalid_mask]))
        raise ValueError(f"final_signal must contain only {{-1, 0, 1}}, got {invalid_values!r}")
    return normalized


def _normalize_bar_close_1m_idx_v2(
    *,
    values: np.ndarray,
    expected_length: int,
    sentinel_index: int,
) -> np.ndarray:
    """
    Normalize one local `bar_close_1m_idx` vector for compact trade construction.

    Args:
        values: Candidate local mapping vector.
        expected_length: Expected `T_signal` length.
        sentinel_index: Local execution timeline length.
    Returns:
        np.ndarray: Canonical `int64` local `bar_close_1m_idx` vector.
    Assumptions:
        Mapping indexes are already rebased to the local execution timeline of the current run.
    Raises:
        ValueError: If shape/length is invalid or one mapped close leaves `[0, sentinel_index)`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    normalized = np.asarray(values, dtype=np.int64)
    if normalized.ndim != 1:
        raise ValueError("bar_close_1m_idx must be a 1D array")
    if normalized.shape[0] != expected_length:
        raise ValueError(
            "bar_close_1m_idx length must match final_signal timeline; "
            f"got {normalized.shape[0]} vs {expected_length}"
        )
    if sentinel_index < 0:
        raise ValueError("sentinel_index must be >= 0")
    if sentinel_index == 0 and normalized.shape[0] > 0:
        raise ValueError("sentinel_index must be > 0 when bar_close_1m_idx is non-empty")
    if bool(np.any((normalized < 0) | (normalized >= sentinel_index))):
        raise ValueError(
            "bar_close_1m_idx values must stay within "
            f"[0, {sentinel_index}); got {tuple(int(value) for value in normalized)!r}"
        )
    return normalized


def _normalize_execution_prices_v2(
    *,
    field_name: str,
    values: np.ndarray,
    sentinel_index: int,
) -> np.ndarray:
    """
    Normalize one execution price vector used by Stage A no-risk metrics.

    Args:
        field_name: Human-readable field name for diagnostics.
        values: Candidate price vector.
        sentinel_index: Local execution timeline length.
    Returns:
        np.ndarray: Canonical `float64` price vector.
    Assumptions:
        Artifact-backed price arrays are already finite and strictly positive on valid bars.
    Raises:
        ValueError: If shape drifts from `sentinel_index` or one price is non-positive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    normalized = np.asarray(values, dtype=np.float64)
    if normalized.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D array")
    if normalized.shape[0] != sentinel_index:
        raise ValueError(
            f"{field_name} length must equal sentinel_index; got {normalized.shape[0]} "
            f"vs {sentinel_index}"
        )
    if bool(np.any(normalized <= 0.0)):
        raise ValueError(f"{field_name} must contain only positive prices")
    return normalized


def _resolve_trade_exit_v2(
    *,
    trade: StageACompactTradeV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    sentinel_index: int,
    close_on_end: bool,
) -> tuple[int | None, float | None]:
    """
    Resolve one Stage A no-risk trade exit price without Stage B risk artifacts.

    Args:
        trade: Compact trade payload.
        exec_open: Local execution open prices.
        exec_close: Local execution close prices.
        sentinel_index: Local execution timeline length.
        close_on_end: Whether open trades close on the final execution close.
    Returns:
        tuple[int | None, float | None]: Exit execution index and raw price, or `(None, None)`
            when the trade stays unclosed and `close_on_end` is disabled.
    Assumptions:
        Signal exits always use execution-bar open prices and sentinel exits use last close.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if trade.sig_exit_exec_idx < sentinel_index:
        exit_exec_idx = int(trade.sig_exit_exec_idx)
        return (exit_exec_idx, float(exec_open[exit_exec_idx]))
    if close_on_end and sentinel_index > 0:
        exit_exec_idx = sentinel_index - 1
        return (exit_exec_idx, float(exec_close[exit_exec_idx]))
    return (None, None)


def _entry_quote_amount_v2(
    *,
    available_quote: float,
    execution_params: ExecutionParamsV1,
) -> float:
    """
    Resolve entry quote budget for one no-risk trade using v1 sizing semantics.

    Args:
        available_quote: Current available strategy quote balance.
        execution_params: Resolved execution settings.
    Returns:
        float: Entry quote amount for the next trade.
    Assumptions:
        `all_in`, `strategy_compound`, and `strategy_compound_profit_lock` all consume the
        currently available quote balance.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if available_quote <= 0.0:
        return 0.0
    if execution_params.sizing_mode == "fixed_quote":
        return min(available_quote, execution_params.fixed_quote)
    return available_quote


def _fill_price_from_mark_v2(*, price: float, slippage_rate: float, is_buy: bool) -> float:
    """
    Build deterministic Stage A fill price from raw execution mark and slippage rate.

    Args:
        price: Raw execution price (`exec_open` or final `exec_close`).
        slippage_rate: Decimal slippage rate.
        is_buy: Whether the fill side is buy.
    Returns:
        float: Slippage-adjusted fill price.
    Assumptions:
        Buy fills pay `+slippage`, sell fills receive `-slippage`.
    Raises:
        ValueError: If the raw price is non-positive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if price <= 0.0:
        raise ValueError("price must be > 0")
    if is_buy:
        return price * (1.0 + slippage_rate)
    return price * (1.0 - slippage_rate)


def _direction_allowed_v2(
    *,
    direction: int,
    direction_mode: StageADirectionModeLiteralV2,
) -> bool:
    """
    Check whether one raw signal direction may open a new trade under current direction mode.

    Args:
        direction: Raw signal direction (`-1` or `1`).
        direction_mode: Strategy direction policy.
    Returns:
        bool: `True` when the direction may open a new trade.
    Assumptions:
        Exit-only handling for forbidden opposite signals is resolved by the caller.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if direction_mode == "long-only":
        return direction == 1
    if direction_mode == "short-only":
        return direction == -1
    return True


def _validate_trade_indexes_v2(*, trade: StageACompactTradeV2, sentinel_index: int) -> None:
    """
    Validate compact trade indexes against the local execution sentinel.

    Args:
        trade: Compact trade payload.
        sentinel_index: Local execution timeline length.
    Returns:
        None.
    Assumptions:
        `sig_exit_exec_idx == sentinel_index` denotes an open trade carried to end-of-run.
    Raises:
        ValueError: If entry or exit indexes leave the bounded-by-sentinel contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    if trade.entry_exec_idx < 0 or trade.entry_exec_idx >= sentinel_index:
        raise ValueError(
            f"entry_exec_idx must stay within [0, {sentinel_index}), got {trade.entry_exec_idx}"
        )
    if trade.sig_exit_exec_idx < trade.entry_exec_idx or trade.sig_exit_exec_idx > sentinel_index:
        raise ValueError(
            "sig_exit_exec_idx must stay within "
            f"[entry_exec_idx, {sentinel_index}], got {trade.sig_exit_exec_idx}"
        )


def _validate_direction_mode_v2(*, direction_mode: str) -> StageADirectionModeLiteralV2:
    """
    Validate one Stage A direction-mode literal.

    Args:
        direction_mode: Raw direction-mode literal.
    Returns:
        StageADirectionModeLiteralV2: Canonical lower-case direction mode.
    Assumptions:
        Stage A compact trade construction must preserve the existing v1 direction-mode surface.
    Raises:
        ValueError: If the literal is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """
    normalized = direction_mode.strip().lower()
    if normalized not in {"long-only", "short-only", "long-short"}:
        raise ValueError(
            "direction_mode must be one of ('long-only', 'short-only', 'long-short')"
        )
    return normalized  # type: ignore[return-value]


__all__ = [
    "build_compact_trade_list_v2",
    "compute_no_risk_metrics_v2",
    "no_risk_metrics_to_ranking_payload_v2",
]
