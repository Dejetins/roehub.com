"""Deterministic `metrics_kernel.py` helpers for Stage B exact replay payloads."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from trading.contexts.backtest.application.ports import RankingMetricsV1
from trading.contexts.backtest.domain.entities import ExecutionOutcomeV1, TradeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1

from .contracts import StageBMetricsV2, StageBReplayPayloadV2, StageBTradeExitV2
from .trade_compactor_kernel import (
    _entry_quote_amount_v2,
    _fill_price_from_mark_v2,
    _trade_sharpe_v2,
)

_BARS_PER_YEAR_EXEC_1M_V2 = 365.0 * 24.0 * 60.0


def compute_stage_b_metrics_v2(
    *,
    replay: StageBReplayPayloadV2,
    fee_rate: float,
    bars_per_year_exec: float = _BARS_PER_YEAR_EXEC_1M_V2,
) -> StageBMetricsV2:
    """
    Compute deterministic Stage B ranking/summary metrics from one exact replay payload.

    Args:
        replay: Exact replay payload for one selected TP/SL cell.
        fee_rate: Per-side fee rate expressed as decimal fraction.
        bars_per_year_exec: Annualization denominator in execution bars for Sharpe.
    Returns:
        StageBMetricsV2: Deterministic Stage B metrics over closed compact trades only.
    Assumptions:
        Metrics follow notebook `metrics over compact trades` semantics and use
        `fee_two_sides = (1 - fee_rate)^2`.
    Raises:
        ValueError: If fee rate or annualization denominator is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    if fee_rate < 0.0 or fee_rate >= 1.0:
        raise ValueError("fee_rate must be in [0.0, 1.0)")
    if bars_per_year_exec <= 0.0:
        raise ValueError("bars_per_year_exec must be > 0")

    fee_two_sides = (1.0 - fee_rate) * (1.0 - fee_rate)
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_count = 0
    win_count = 0
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    sum_trade_bars = 0.0
    exposure_bars = 0.0

    for trade_exit in replay.trade_exits:
        if not trade_exit.closed:
            continue
        factor_after_fees = fee_two_sides * float(trade_exit.gross_factor)
        trade_return = factor_after_fees - 1.0
        equity *= factor_after_fees
        trade_count += 1
        if trade_return > 0.0:
            win_count += 1
            gross_profit += trade_return
        elif trade_return < 0.0:
            gross_loss += abs(trade_return)
        sum_trade_return += trade_return
        sum_trade_return_squared += trade_return * trade_return
        bars_held = float(max(trade_exit.exit_exec_idx - trade_exit.entry_exec_idx, 0))
        sum_trade_bars += bars_held
        exposure_bars += bars_held
        if equity > peak:
            peak = equity
        elif peak > 0.0:
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    total_return_pct = (equity - 1.0) * 100.0
    max_drawdown_pct = max_drawdown * 100.0
    if trade_count > 0:
        win_rate_pct = (float(win_count) / float(trade_count)) * 100.0
        avg_trade_ret_pct = (sum_trade_return / float(trade_count)) * 100.0
        avg_trade_exec_bars = sum_trade_bars / float(trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
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
        (exposure_bars / float(replay.sentinel_index)) * 100.0 if replay.sentinel_index > 0 else 0.0
    )
    sharpe_trades = _trade_sharpe_v2(
        trade_count=trade_count,
        sum_trade_return=sum_trade_return,
        sum_trade_return_squared=sum_trade_return_squared,
        bars_per_year_exec=bars_per_year_exec,
        sentinel_index=replay.sentinel_index,
    )
    return StageBMetricsV2(
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        return_over_max_drawdown=return_over_max_drawdown,
        profit_factor=profit_factor,
        trade_count=trade_count,
        win_rate_pct=win_rate_pct,
        avg_trade_ret_pct=avg_trade_ret_pct,
        avg_trade_exec_bars=avg_trade_exec_bars,
        exposure_pct=exposure_pct,
        sharpe_trades=sharpe_trades,
    )


def stage_b_metrics_to_ranking_payload_v2(
    *,
    metrics: StageBMetricsV2,
) -> RankingMetricsV1:
    """
    Convert Stage B metrics into stable ranking/summary keys consumed by staged runners.

    Args:
        metrics: Deterministic Stage B metrics for one TP/SL cell.
    Returns:
        RankingMetricsV1: Immutable mapping with additive Stage B ranking aliases.
    Assumptions:
        Public ranking/task contracts keep v1 literals stable while Stage B v2 lands additively.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    return MappingProxyType(
        {
            "total_return_pct": float(metrics.total_return_pct),
            "Total Return [%]": float(metrics.total_return_pct),
            "max_drawdown_pct": float(metrics.max_drawdown_pct),
            "Max. Drawdown [%]": float(metrics.max_drawdown_pct),
            "return_over_max_drawdown": float(metrics.return_over_max_drawdown),
            "profit_factor": float(metrics.profit_factor),
            "trade_count": float(metrics.trade_count),
            "win_rate_pct": float(metrics.win_rate_pct),
            "avg_trade_ret_pct": float(metrics.avg_trade_ret_pct),
            "avg_trade_exec_bars": float(metrics.avg_trade_exec_bars),
            "exposure_pct": float(metrics.exposure_pct),
            "sharpe_trades": float(metrics.sharpe_trades),
        }
    )


def build_execution_outcome_from_replay_v2(
    *,
    replay: StageBReplayPayloadV2,
    metrics: StageBMetricsV2,
    execution_params: ExecutionParamsV1,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
) -> ExecutionOutcomeV1:
    """
    Build a deterministic `ExecutionOutcomeV1` from exact Stage B replay facts.

    Args:
        replay: Exact replay payload for one selected TP/SL cell.
        metrics: Deterministic Stage B metrics already computed from `replay`.
        execution_params: Runtime execution settings used for report/trade payloads.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        tp_values: Local TP grid values as decimal rates.
        sl_values: Local SL grid values as decimal rates.
    Returns:
        ExecutionOutcomeV1: Deterministic trade/body payload compatible with existing reports.
    Assumptions:
        Ranking metrics come from compact-trade replay; this helper only materializes compatible
        trade bodies for details/reporting flows and does not redefine exit semantics.
    Raises:
        ValueError: If execution arrays drift from `replay.sentinel_index`.
    Side Effects:
        Materializes deterministic `TradeV1` objects in replay order.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """
    normalized_exec_open = _normalize_execution_prices_v2(
        field_name="exec_open",
        values=exec_open,
        sentinel_index=replay.sentinel_index,
    )
    normalized_exec_close = _normalize_execution_prices_v2(
        field_name="exec_close",
        values=exec_close,
        sentinel_index=replay.sentinel_index,
    )
    normalized_tp_values = np.asarray(tp_values, dtype=np.float64)
    normalized_sl_values = np.asarray(sl_values, dtype=np.float64)
    available_quote = float(execution_params.init_cash_quote)
    safe_quote = 0.0
    trades: list[TradeV1] = []

    for trade_id, trade_exit in enumerate(replay.trade_exits, start=1):
        if not trade_exit.closed:
            continue
        entry_mark_price = float(normalized_exec_open[trade_exit.entry_exec_idx])
        exit_mark_price = _resolve_exit_mark_price_v2(
            replay=replay,
            trade_exit=trade_exit,
            exec_open=normalized_exec_open,
            exec_close=normalized_exec_close,
            tp_values=normalized_tp_values,
            sl_values=normalized_sl_values,
        )
        if entry_mark_price <= 0.0 or exit_mark_price <= 0.0:
            continue
        quote_amount = _entry_quote_amount_v2(
            available_quote=available_quote,
            execution_params=execution_params,
        )
        if quote_amount <= 0.0:
            continue
        entry_fill_price = _fill_price_from_mark_v2(
            price=entry_mark_price,
            slippage_rate=execution_params.slippage_rate,
            is_buy=trade_exit.direction == 1,
        )
        exit_fill_price = _fill_price_from_mark_v2(
            price=exit_mark_price,
            slippage_rate=execution_params.slippage_rate,
            is_buy=trade_exit.direction == -1,
        )
        qty_base = quote_amount / entry_fill_price
        entry_fee_quote = quote_amount * execution_params.fee_rate
        available_quote -= quote_amount + entry_fee_quote
        exit_quote_amount = qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * execution_params.fee_rate
        if trade_exit.direction == 1:
            gross_pnl_quote = exit_quote_amount - quote_amount
        else:
            gross_pnl_quote = quote_amount - exit_quote_amount
        available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote
        locked_profit_quote = 0.0
        if execution_params.sizing_mode == "strategy_compound_profit_lock" and net_pnl_quote > 0.0:
            locked_profit_quote = net_pnl_quote * (execution_params.safe_profit_percent / 100.0)
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        trades.append(
            TradeV1(
                trade_id=trade_id,
                direction="long" if trade_exit.direction == 1 else "short",
                entry_bar_index=trade_exit.entry_exec_idx,
                exit_bar_index=trade_exit.exit_exec_idx,
                entry_fill_price=entry_fill_price,
                exit_fill_price=exit_fill_price,
                qty_base=qty_base,
                entry_quote_amount=quote_amount,
                exit_quote_amount=exit_quote_amount,
                entry_fee_quote=entry_fee_quote,
                exit_fee_quote=exit_fee_quote,
                gross_pnl_quote=gross_pnl_quote,
                net_pnl_quote=net_pnl_quote,
                locked_profit_quote=locked_profit_quote,
                exit_reason=trade_exit.exit_reason,
            )
        )
    equity_end_quote = available_quote + safe_quote
    return ExecutionOutcomeV1(
        trades=tuple(trades),
        equity_end_quote=equity_end_quote,
        available_quote=available_quote,
        safe_quote=safe_quote,
        total_return_pct=float(metrics.total_return_pct),
    )
def _normalize_execution_prices_v2(
    *,
    field_name: str,
    values: np.ndarray,
    sentinel_index: int,
) -> np.ndarray:
    """
    Normalize one execution price vector to deterministic `np.float64`.

    Args:
        field_name: Deterministic diagnostics field label.
        values: Candidate execution price vector.
        sentinel_index: Expected execution timeline length.
    Returns:
        np.ndarray: Canonical one-dimensional `np.float64` vector.
    Assumptions:
        Details/report helpers operate on the same local execution window as replay payloads.
    Raises:
        ValueError: If the array shape drifts from `sentinel_index`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    normalized = np.asarray(values, dtype=np.float64)
    if normalized.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D array")
    if normalized.shape[0] != sentinel_index:
        raise ValueError(
            f"{field_name} length must match sentinel_index; "
            f"got {normalized.shape[0]} vs {sentinel_index}"
        )
    return normalized


def _resolve_exit_mark_price_v2(
    *,
    replay: StageBReplayPayloadV2,
    trade_exit: StageBTradeExitV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
) -> float:
    """
    Resolve synthetic mark exit price for one exact replay trade exit.

    Args:
        replay: Exact replay payload owning selected TP/SL indexes.
        trade_exit: Exact closed trade exit fact.
        exec_open: Local execution opens.
        exec_close: Local execution closes.
        tp_values: TP grid values in decimal-rate form.
        sl_values: SL grid values in decimal-rate form.
    Returns:
        float: Deterministic mark price used for details/report trade materialization.
    Assumptions:
        TP/SL exits use the selected cell rates, while signal/end exits use actual execution
        prices from the local `1m` timeline.
    Raises:
        ValueError: If replay payload lacks selected TP/SL index required by the exit reason.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """
    if trade_exit.exit_reason == "signal_exit":
        return float(exec_open[trade_exit.exit_exec_idx])
    if trade_exit.exit_reason == "close_on_end":
        return float(exec_close[trade_exit.exit_exec_idx])
    entry_open = float(exec_open[trade_exit.entry_exec_idx])
    if trade_exit.exit_reason == "tp":
        if replay.tp_index is None:
            raise ValueError("replay.tp_index must be set for tp exit materialization")
        tp_rate = float(tp_values[replay.tp_index])
        if trade_exit.direction == 1:
            return entry_open * (1.0 + tp_rate)
        return entry_open * max(0.0, 1.0 - tp_rate)
    if trade_exit.exit_reason == "sl":
        if replay.sl_index is None:
            raise ValueError("replay.sl_index must be set for sl exit materialization")
        sl_rate = float(sl_values[replay.sl_index])
        if trade_exit.direction == 1:
            return entry_open * max(0.0, 1.0 - sl_rate)
        return entry_open * (1.0 + sl_rate)
    raise ValueError(
        "unsupported exit_reason for materialized trade body: " f"{trade_exit.exit_reason}"
    )


__all__ = [
    "build_execution_outcome_from_replay_v2",
    "compute_stage_b_metrics_v2",
    "stage_b_metrics_to_ranking_payload_v2",
]
