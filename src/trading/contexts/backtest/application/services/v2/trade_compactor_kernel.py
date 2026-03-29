"""Pure Stage A compact-trade and no-risk metric kernels for artifacts-only inputs."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

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
    normalized_signal = _normalize_final_signal_matrix_v2(values=final_signal)
    normalized_mapping = _normalize_bar_close_1m_idx_v2(
        values=bar_close_1m_idx,
        expected_length=normalized_signal.shape[1],
        sentinel_index=sentinel_index,
    )
    resolved_direction_mode = _validate_direction_mode_v2(direction_mode=direction_mode)
    entry_exec_idx = np.minimum(normalized_mapping + 1, sentinel_index).astype(np.int64, copy=False)
    return tuple(
        _build_compact_trade_row_v2(
            final_signal_row=final_signal_row,
            entry_exec_idx=entry_exec_idx,
            sentinel_index=sentinel_index,
            direction_mode=resolved_direction_mode,
        )
        for final_signal_row in normalized_signal
    )


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
