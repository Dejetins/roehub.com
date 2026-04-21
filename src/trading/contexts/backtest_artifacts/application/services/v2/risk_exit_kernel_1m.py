"""Stage B `risk_exit_kernel_1m.py` for compact trades over shipped `1m hit-times`."""

from __future__ import annotations

import math

import numpy as np

from .contracts import (
    ArtifactHitTimesArraysV2,
    StageACompactTradeV2,
    StageBFastSearchResultV2,
    StageBHitTimesSliceV2,
    StageBReferenceVsFastSelfCheckResultV2,
    StageBReplayPayloadV2,
    StageBTradeExitV2,
)
from .metrics_kernel import compute_stage_b_metrics_v2


def slice_hit_times_to_execution_window_v2(
    *,
    hit_times_arrays: ArtifactHitTimesArraysV2,
    exec_target_slice: slice,
) -> StageBHitTimesSliceV2:
    """
    Rebase strict `1m hit-times` tables to the local execution window of the current run.

    Args:
        hit_times_arrays: Full-slot strict `1m hit-times` arrays loaded from artifacts.
        exec_target_slice: Explicit local execution slice `[start:stop)` in global `1m` indexes.
    Returns:
        StageBHitTimesSliceV2: Local rebased tables with `sentinel_index == stop - start`.
    Assumptions:
        Runtime consumes shipped artifacts only; hits outside the local execution window become
        the local `sentinel_index`.
    Raises:
        ValueError: If the slice is implicit, inverted, or outside the artifact execution range.
    Side Effects:
        Allocates local rebased table copies for deterministic Stage B kernels.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    start, stop = _normalize_explicit_slice_v2(
        target_slice=exec_target_slice,
        upper_bound=int(hit_times_arrays.manifest.sentinel_index),
        field_name="exec_target_slice",
    )
    sentinel_index = stop - start
    tp_values = _normalize_hit_times_level_grid_v2(
        field_name="tp_values",
        values=hit_times_arrays.tp_values,
    )
    sl_values = _normalize_hit_times_level_grid_v2(
        field_name="sl_values",
        values=hit_times_arrays.sl_values,
    )
    return StageBHitTimesSliceV2(
        tp_values=tp_values,
        sl_values=sl_values,
        long_tp=_rebase_hit_times_table_v2(
            values=hit_times_arrays.long_tp,
            start=start,
            stop=stop,
            local_sentinel_index=sentinel_index,
        ),
        long_sl=_rebase_hit_times_table_v2(
            values=hit_times_arrays.long_sl,
            start=start,
            stop=stop,
            local_sentinel_index=sentinel_index,
        ),
        short_tp=_rebase_hit_times_table_v2(
            values=hit_times_arrays.short_tp,
            start=start,
            stop=stop,
            local_sentinel_index=sentinel_index,
        ),
        short_sl=_rebase_hit_times_table_v2(
            values=hit_times_arrays.short_sl,
            start=start,
            stop=stop,
            local_sentinel_index=sentinel_index,
        ),
        sentinel_index=sentinel_index,
    )


def resolve_risk_trade_exit_1m_v2(
    *,
    trade_index: int,
    trade: StageACompactTradeV2,
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    tp_index: int | None,
    sl_index: int | None,
    close_on_end: bool = True,
) -> StageBTradeExitV2:
    """
    Resolve one exact Stage B trade exit from compact-trade fields and local `1m hit-times`.

    Args:
        trade_index: Stable ordinal index in the compact trade list.
        trade: Compact trade entry with `entry_exec_idx` and `sig_exit_exec_idx`.
        hit_times: Local `1m hit-times` slice for the current execution window.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        tp_index: Selected TP level index, or `None` when TP is disabled.
        sl_index: Selected SL level index, or `None` when SL is disabled.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
    Returns:
        StageBTradeExitV2: Exact deterministic exit fact for this compact trade.
    Assumptions:
        `TP/SL lookup starts at entry_exec + 1`, `signal exit wins on equal bar`, and
        `SL wins TP tie`.
    Raises:
        ValueError: If execution arrays drift from `sentinel_index` or one level index is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    sentinel_index = int(hit_times.sentinel_index)
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
    _validate_compact_trade_for_stage_b_v2(
        trade=trade,
        sentinel_index=sentinel_index,
        trade_index=trade_index,
    )
    _validate_level_indexes_v2(
        hit_times=hit_times,
        tp_index=tp_index,
        sl_index=sl_index,
    )

    if trade.entry_exec_idx >= sentinel_index:
        return StageBTradeExitV2(
            trade_index=trade_index,
            entry_exec_idx=trade.entry_exec_idx,
            direction=trade.direction,
            sig_exit_exec_idx=trade.sig_exit_exec_idx,
            exit_exec_idx=trade.entry_exec_idx,
            exit_reason="unclosed",
            gross_factor=1.0,
            closed=False,
        )

    entry_open = float(normalized_exec_open[trade.entry_exec_idx])
    if entry_open <= 0.0:
        return StageBTradeExitV2(
            trade_index=trade_index,
            entry_exec_idx=trade.entry_exec_idx,
            direction=trade.direction,
            sig_exit_exec_idx=trade.sig_exit_exec_idx,
            exit_exec_idx=trade.entry_exec_idx,
            exit_reason="unclosed",
            gross_factor=1.0,
            closed=False,
        )

    lookup_exec = trade.entry_exec_idx + 1
    tp_exec = sentinel_index
    sl_exec = sentinel_index
    tp_factor = 1.0
    sl_factor = 1.0
    if lookup_exec < sentinel_index:
        if tp_index is not None:
            tp_exec = _tp_table_for_trade_direction_v2(
                hit_times=hit_times,
                direction=trade.direction,
            )[tp_index, lookup_exec]
            tp_factor = _level_factor_from_tp_rate_v2(
                tp_rate=float(hit_times.tp_values[tp_index]),
            )
        if sl_index is not None:
            sl_exec = _sl_table_for_trade_direction_v2(
                hit_times=hit_times,
                direction=trade.direction,
            )[sl_index, lookup_exec]
            sl_factor = _level_factor_from_sl_rate_v2(
                sl_rate=float(hit_times.sl_values[sl_index]),
            )

    tp_sl_exec = sentinel_index
    tp_sl_reason = "tp"
    tp_sl_factor = 1.0
    if sl_index is not None and sl_exec <= tp_sl_exec:
        tp_sl_exec = int(sl_exec)
        tp_sl_reason = "sl"
        tp_sl_factor = sl_factor
    if tp_index is not None and tp_exec < tp_sl_exec:
        tp_sl_exec = int(tp_exec)
        tp_sl_reason = "tp"
        tp_sl_factor = tp_factor

    if trade.sig_exit_exec_idx < sentinel_index and trade.sig_exit_exec_idx <= tp_sl_exec:
        exit_open = float(normalized_exec_open[trade.sig_exit_exec_idx])
        gross_factor = (
            _signal_or_end_factor_v2(
                direction=trade.direction,
                entry_open=entry_open,
                exit_price=exit_open,
            )
            if exit_open > 0.0
            else 1.0
        )
        return StageBTradeExitV2(
            trade_index=trade_index,
            entry_exec_idx=trade.entry_exec_idx,
            direction=trade.direction,
            sig_exit_exec_idx=trade.sig_exit_exec_idx,
            exit_exec_idx=trade.sig_exit_exec_idx,
            exit_reason="signal_exit",
            gross_factor=gross_factor,
            closed=True,
        )

    if tp_sl_exec < sentinel_index:
        return StageBTradeExitV2(
            trade_index=trade_index,
            entry_exec_idx=trade.entry_exec_idx,
            direction=trade.direction,
            sig_exit_exec_idx=trade.sig_exit_exec_idx,
            exit_exec_idx=tp_sl_exec,
            exit_reason=tp_sl_reason,
            gross_factor=tp_sl_factor,
            closed=True,
        )

    if close_on_end and sentinel_index > 0:
        last_close = float(normalized_exec_close[sentinel_index - 1])
        gross_factor = (
            _signal_or_end_factor_v2(
                direction=trade.direction,
                entry_open=entry_open,
                exit_price=last_close,
            )
            if last_close > 0.0
            else 1.0
        )
        return StageBTradeExitV2(
            trade_index=trade_index,
            entry_exec_idx=trade.entry_exec_idx,
            direction=trade.direction,
            sig_exit_exec_idx=trade.sig_exit_exec_idx,
            exit_exec_idx=sentinel_index - 1,
            exit_reason="close_on_end",
            gross_factor=gross_factor,
            closed=True,
        )

    return StageBTradeExitV2(
        trade_index=trade_index,
        entry_exec_idx=trade.entry_exec_idx,
        direction=trade.direction,
        sig_exit_exec_idx=trade.sig_exit_exec_idx,
        exit_exec_idx=trade.entry_exec_idx,
        exit_reason="unclosed",
        gross_factor=1.0,
        closed=False,
    )


def search_risk_cells_total_return_fast_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    fee_rate: float,
    close_on_end: bool = True,
) -> StageBFastSearchResultV2:
    """
    Run fast TP/SL grid search over compact trades using monotone diff-buffer decomposition.

    Args:
        compact_trades: Ordered compact trades from Stage A artifact-backed output.
        hit_times: Local `1m hit-times` slice for the execution window.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        fee_rate: Per-side fee rate expressed as decimal fraction.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
    Returns:
        StageBFastSearchResultV2: Total-return matrix plus deterministic best TP/SL coordinates.
    Assumptions:
        This kernel avoids naive full replay over all TP/SL cells and uses exact replay only
        after winner selection.
    Raises:
        ValueError: If execution arrays, fee rate, or hit-times grids are invalid.
    Side Effects:
        Allocates deterministic diff buffers shaped by `tp_values x sl_values`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    sentinel_index = int(hit_times.sentinel_index)
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
    if fee_rate < 0.0 or fee_rate >= 1.0:
        raise ValueError("fee_rate must be in [0.0, 1.0)")
    n_tp = int(hit_times.tp_values.shape[0])
    n_sl = int(hit_times.sl_values.shape[0])
    if n_tp <= 0 or n_sl <= 0:
        raise ValueError("fast search requires at least one TP level and one SL level")

    fee_two_sides = (1.0 - fee_rate) * (1.0 - fee_rate)
    tp_factors = 1.0 + np.asarray(hit_times.tp_values, dtype=np.float64)
    sl_factors = np.maximum(0.0, 1.0 - np.asarray(hit_times.sl_values, dtype=np.float64))
    row_diff = np.zeros((n_tp, n_sl + 1), dtype=np.float64)
    col_diff = np.zeros((n_tp + 1, n_sl), dtype=np.float64)
    rect_diff = np.zeros((n_tp + 1, n_sl + 1), dtype=np.float64)
    row_zero_diff = np.zeros((n_tp, n_sl + 1), dtype=np.int64)
    col_zero_diff = np.zeros((n_tp + 1, n_sl), dtype=np.int64)
    rect_zero_diff = np.zeros((n_tp + 1, n_sl + 1), dtype=np.int64)

    for trade_index, trade in enumerate(compact_trades):
        _validate_compact_trade_for_stage_b_v2(
            trade=trade,
            sentinel_index=sentinel_index,
            trade_index=trade_index,
        )
        if trade.entry_exec_idx >= sentinel_index:
            continue
        entry_open = float(normalized_exec_open[trade.entry_exec_idx])
        if entry_open <= 0.0:
            continue

        start = trade.entry_exec_idx + 1
        signal_exit_exec_idx = min(trade.sig_exit_exec_idx, sentinel_index)
        if start >= sentinel_index:
            _add_terminal_region_contribution_v2(
                direction=trade.direction,
                entry_open=entry_open,
                signal_exit_exec_idx=signal_exit_exec_idx,
                exec_open=normalized_exec_open,
                exec_close=normalized_exec_close,
                sentinel_index=sentinel_index,
                close_on_end=close_on_end,
                fee_two_sides=fee_two_sides,
                row_stop=n_tp,
                col_stop=n_sl,
                rect_diff=rect_diff,
                rect_zero_diff=rect_zero_diff,
            )
            continue

        hit_tp = _tp_table_for_trade_direction_v2(hit_times=hit_times, direction=trade.direction)[
            :, start
        ]
        hit_sl = _sl_table_for_trade_direction_v2(hit_times=hit_times, direction=trade.direction)[
            :, start
        ]

        if signal_exit_exec_idx < sentinel_index:
            i_sig = _lower_bound_ge_hit_v2(
                hit_values=hit_tp,
                threshold=signal_exit_exec_idx,
            )
            j_sig = _lower_bound_ge_hit_v2(
                hit_values=hit_sl,
                threshold=signal_exit_exec_idx,
            )
            signal_exit_open = float(normalized_exec_open[signal_exit_exec_idx])
            signal_factor = (
                _signal_or_end_factor_v2(
                    direction=trade.direction,
                    entry_open=entry_open,
                    exit_price=signal_exit_open,
                )
                if signal_exit_open > 0.0
                else 1.0
            )
            _add_weighted_rect_v2(
                rect_diff=rect_diff,
                rect_zero_diff=rect_zero_diff,
                row_start=i_sig,
                col_start=j_sig,
                row_stop=n_tp,
                col_stop=n_sl,
                factor=fee_two_sides * signal_factor,
            )
            j_ptr = 0
            for tp_index in range(i_sig):
                tp_exec = int(hit_tp[tp_index])
                while j_ptr < j_sig and int(hit_sl[j_ptr]) <= tp_exec:
                    j_ptr += 1
                _add_weighted_row_range_v2(
                    row_diff=row_diff,
                    row_zero_diff=row_zero_diff,
                    row_i=tp_index,
                    col_start=j_ptr,
                    col_stop=n_sl,
                    factor=fee_two_sides * tp_factors[tp_index],
                )
            i_ptr = 0
            for sl_index in range(j_sig):
                sl_exec = int(hit_sl[sl_index])
                while i_ptr < i_sig and int(hit_tp[i_ptr]) < sl_exec:
                    i_ptr += 1
                _add_weighted_col_range_v2(
                    col_diff=col_diff,
                    col_zero_diff=col_zero_diff,
                    row_start=i_ptr,
                    row_stop=n_tp,
                    col_j=sl_index,
                    factor=fee_two_sides * sl_factors[sl_index],
                )
            continue

        i_never = _first_equal_hit_v2(
            hit_values=hit_tp,
            sentinel_index=sentinel_index,
        )
        j_never = _first_equal_hit_v2(
            hit_values=hit_sl,
            sentinel_index=sentinel_index,
        )
        if close_on_end and sentinel_index > 0:
            last_close = float(normalized_exec_close[sentinel_index - 1])
            end_factor = (
                _signal_or_end_factor_v2(
                    direction=trade.direction,
                    entry_open=entry_open,
                    exit_price=last_close,
                )
                if last_close > 0.0
                else 1.0
            )
            _add_weighted_rect_v2(
                rect_diff=rect_diff,
                rect_zero_diff=rect_zero_diff,
                row_start=i_never,
                col_start=j_never,
                row_stop=n_tp,
                col_stop=n_sl,
                factor=fee_two_sides * end_factor,
            )
        j_ptr = 0
        for tp_index in range(i_never):
            tp_exec = int(hit_tp[tp_index])
            while j_ptr < j_never and int(hit_sl[j_ptr]) <= tp_exec:
                j_ptr += 1
            _add_weighted_row_range_v2(
                row_diff=row_diff,
                row_zero_diff=row_zero_diff,
                row_i=tp_index,
                col_start=j_ptr,
                col_stop=n_sl,
                factor=fee_two_sides * tp_factors[tp_index],
            )
        i_ptr = 0
        for sl_index in range(j_never):
            sl_exec = int(hit_sl[sl_index])
            while i_ptr < i_never and int(hit_tp[i_ptr]) < sl_exec:
                i_ptr += 1
            _add_weighted_col_range_v2(
                col_diff=col_diff,
                col_zero_diff=col_zero_diff,
                row_start=i_ptr,
                row_stop=n_tp,
                col_j=sl_index,
                factor=fee_two_sides * sl_factors[sl_index],
            )

    total_log = (
        _integrate_row_diff_v2(values=row_diff)
        + _integrate_col_diff_v2(values=col_diff)
        + _integrate_rect_diff_v2(values=rect_diff)
    )
    zero_hits = (
        _integrate_row_diff_v2(values=row_zero_diff.astype(np.float64))
        + _integrate_col_diff_v2(values=col_zero_diff.astype(np.float64))
        + _integrate_rect_diff_v2(values=rect_zero_diff.astype(np.float64))
    )
    total_return_pct = np.where(
        zero_hits > 0.0,
        -100.0,
        (np.exp(total_log) - 1.0) * 100.0,
    )
    best_flat_index = int(np.argmax(total_return_pct))
    best_tp_index, best_sl_index = np.unravel_index(best_flat_index, total_return_pct.shape)
    return StageBFastSearchResultV2(
        total_return_pct=total_return_pct,
        best_tp_index=int(best_tp_index),
        best_sl_index=int(best_sl_index),
        best_total_return_pct=float(total_return_pct[best_tp_index, best_sl_index]),
    )


def replay_risk_cell_exact_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    tp_index: int | None,
    sl_index: int | None,
    close_on_end: bool = True,
) -> StageBReplayPayloadV2:
    """
    Execute exact replay for one explicit TP/SL cell over the compact trade list only.

    Args:
        compact_trades: Ordered compact trades from Stage A artifact-backed output.
        hit_times: Local `1m hit-times` slice for the execution window.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        tp_index: Selected TP level index, or `None` when TP is disabled.
        sl_index: Selected SL level index, or `None` when SL is disabled.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
    Returns:
        StageBReplayPayloadV2: Exact replay payload for the selected TP/SL cell.
    Assumptions:
        This function performs the `exact replay of best TP/SL cell` or of an explicit runtime
        cell, but never recomputes `1m hit-times`.
    Raises:
        ValueError: If execution arrays or selected level indexes are invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    trade_exits = tuple(
        resolve_risk_trade_exit_1m_v2(
            trade_index=trade_index,
            trade=trade,
            hit_times=hit_times,
            exec_open=exec_open,
            exec_close=exec_close,
            tp_index=tp_index,
            sl_index=sl_index,
            close_on_end=close_on_end,
        )
        for trade_index, trade in enumerate(compact_trades)
    )
    return StageBReplayPayloadV2(
        tp_index=tp_index,
        sl_index=sl_index,
        sentinel_index=hit_times.sentinel_index,
        close_on_end=close_on_end,
        trade_exits=trade_exits,
    )


def replay_best_risk_cell_exact_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    fee_rate: float,
    close_on_end: bool = True,
) -> tuple[StageBFastSearchResultV2, StageBReplayPayloadV2]:
    """
    Search the best TP/SL cell fast, then replay that winning cell exactly once.

    Args:
        compact_trades: Ordered compact trades from Stage A artifact-backed output.
        hit_times: Local `1m hit-times` slice for the execution window.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        fee_rate: Per-side fee rate expressed as decimal fraction.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
    Returns:
        tuple[StageBFastSearchResultV2, StageBReplayPayloadV2]: Fast winner matrix and exact
            replay payload for the selected best cell.
    Assumptions:
        Runtime keeps exact replay limited to the chosen best TP/SL cell only.
    Raises:
        ValueError: Propagated from fast search or exact replay on invalid inputs.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    fast_result = search_risk_cells_total_return_fast_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=fee_rate,
        close_on_end=close_on_end,
    )
    return (
        fast_result,
        replay_risk_cell_exact_v2(
            compact_trades=compact_trades,
            hit_times=hit_times,
            exec_open=exec_open,
            exec_close=exec_close,
            tp_index=fast_result.best_tp_index,
            sl_index=fast_result.best_sl_index,
            close_on_end=close_on_end,
        ),
    )


def run_reference_vs_fast_self_check_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    fee_rate: float,
    max_trade_count: int | None = None,
    max_tp_level_count: int | None = None,
    max_sl_level_count: int | None = None,
    close_on_end: bool = True,
    return_tolerance: float = 1e-9,
) -> StageBReferenceVsFastSelfCheckResultV2:
    """
    Execute the Stage B reference-vs-fast self-check on one deterministic bounded subset.

    Args:
        compact_trades: Ordered compact trades from the exact-risk pipeline.
        hit_times: Local `1m hit-times` slice for the execution window.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        fee_rate: Per-side fee rate expressed as decimal fraction.
        max_trade_count: Optional leading-trade cap for the bounded subset.
        max_tp_level_count: Optional leading-TP cap for the bounded subset.
        max_sl_level_count: Optional leading-SL cap for the bounded subset.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
        return_tolerance: Allowed absolute total-return drift for deterministic parity checks.
    Returns:
        StageBReferenceVsFastSelfCheckResultV2: Diagnostics for the bounded subset and both
            fast/reference winners.
    Assumptions:
        This helper is test/debug-oriented, uses exact replay as the slow reference, and must stay
        off the default production hot path.
    Raises:
        ValueError: If one requested bound or the return tolerance is invalid.
        AssertionError: If the reference-vs-fast self-check fails deterministic parity.
    Side Effects:
        Allocates bounded matrices and exact replay payloads for the bounded subset only.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    total_trade_count = len(compact_trades)
    total_tp_level_count = int(hit_times.tp_values.shape[0])
    total_sl_level_count = int(hit_times.sl_values.shape[0])
    bounded_trade_count = _normalize_optional_trade_count_v2(
        total_count=total_trade_count,
        requested_count=max_trade_count,
        field_name="max_trade_count",
    )
    bounded_tp_level_count = _normalize_bounded_level_count_v2(
        total_count=total_tp_level_count,
        requested_count=max_tp_level_count,
        field_name="max_tp_level_count",
    )
    bounded_sl_level_count = _normalize_bounded_level_count_v2(
        total_count=total_sl_level_count,
        requested_count=max_sl_level_count,
        field_name="max_sl_level_count",
    )
    bounded_compact_trades = compact_trades[:bounded_trade_count]
    bounded_hit_times = _slice_hit_times_level_subset_v2(
        hit_times=hit_times,
        tp_level_count=bounded_tp_level_count,
        sl_level_count=bounded_sl_level_count,
    )
    fast_result = search_risk_cells_total_return_fast_v2(
        compact_trades=bounded_compact_trades,
        hit_times=bounded_hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=fee_rate,
        close_on_end=close_on_end,
    )
    reference_total_return_pct = _build_reference_total_return_matrix_v2(
        compact_trades=bounded_compact_trades,
        hit_times=bounded_hit_times,
        exec_open=exec_open,
        exec_close=exec_close,
        fee_rate=fee_rate,
        close_on_end=close_on_end,
    )
    reference_flat_index = int(np.argmax(reference_total_return_pct))
    reference_best_tp_index, reference_best_sl_index = np.unravel_index(
        reference_flat_index,
        reference_total_return_pct.shape,
    )
    reference_best_total_return_pct = float(
        reference_total_return_pct[reference_best_tp_index, reference_best_sl_index]
    )
    max_abs_total_return_diff = _assert_reference_vs_fast_self_check_parity_v2(
        fast_result=fast_result,
        reference_total_return_pct=reference_total_return_pct,
        reference_best_tp_index=int(reference_best_tp_index),
        reference_best_sl_index=int(reference_best_sl_index),
        reference_best_total_return_pct=reference_best_total_return_pct,
        return_tolerance=return_tolerance,
    )
    return StageBReferenceVsFastSelfCheckResultV2(
        total_trade_count=total_trade_count,
        bounded_trade_count=bounded_trade_count,
        total_tp_level_count=total_tp_level_count,
        bounded_tp_level_count=bounded_tp_level_count,
        total_sl_level_count=total_sl_level_count,
        bounded_sl_level_count=bounded_sl_level_count,
        fast_result=fast_result,
        reference_total_return_pct=reference_total_return_pct,
        reference_best_tp_index=int(reference_best_tp_index),
        reference_best_sl_index=int(reference_best_sl_index),
        reference_best_total_return_pct=reference_best_total_return_pct,
        max_abs_total_return_diff=max_abs_total_return_diff,
    )


def _normalize_explicit_slice_v2(
    *,
    target_slice: slice,
    upper_bound: int,
    field_name: str,
) -> tuple[int, int]:
    """
    Normalize one explicit half-open slice against an upper bound.

    Args:
        target_slice: Candidate Python slice.
        upper_bound: Exclusive upper bound for the indexed timeline.
        field_name: Deterministic diagnostics field label.
    Returns:
        tuple[int, int]: Explicit `(start, stop)` coordinates.
    Assumptions:
        Runtime slice contracts are always explicit and use `step is None`.
    Raises:
        ValueError: If the slice is implicit, stepped, inverted, or outside bounds.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if target_slice.start is None or target_slice.stop is None:
        raise ValueError(f"{field_name} must be explicit")
    if target_slice.step not in (None, 1):
        raise ValueError(f"{field_name} step must be None or 1")
    start = int(target_slice.start)
    stop = int(target_slice.stop)
    if start < 0 or stop < start or stop > upper_bound:
        raise ValueError(
            f"{field_name} must satisfy 0 <= start <= stop <= {upper_bound}; "
            f"got slice({start}, {stop}, {target_slice.step})"
        )
    return (start, stop)


def _normalize_optional_trade_count_v2(
    *,
    total_count: int,
    requested_count: int | None,
    field_name: str,
) -> int:
    """
    Normalize one optional compact-trade bound for the self-check bounded subset.

    Args:
        total_count: Total number of compact trades available to the caller.
        requested_count: Optional requested leading-trade count.
        field_name: Deterministic diagnostics field label.
    Returns:
        int: Normalized leading-trade count in `[0, total_count]`.
    Assumptions:
        The reference-vs-fast self-check may intentionally validate an empty trade subset.
    Raises:
        ValueError: If the requested count is non-numeric, negative, or larger than `total_count`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    if total_count < 0:
        raise ValueError("total_count must be >= 0")
    if requested_count is None:
        return total_count
    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError(f"{field_name} must be an integer when set")
    if requested_count < 0 or requested_count > total_count:
        raise ValueError(f"{field_name} must satisfy 0 <= value <= {total_count}")
    return requested_count


def _normalize_bounded_level_count_v2(
    *,
    total_count: int,
    requested_count: int | None,
    field_name: str,
) -> int:
    """
    Normalize one TP/SL grid bound for the self-check bounded subset.

    Args:
        total_count: Total number of TP or SL levels available to the caller.
        requested_count: Optional requested leading-level count.
        field_name: Deterministic diagnostics field label.
    Returns:
        int: Normalized leading-level count in `[1, total_count]`.
    Assumptions:
        The bounded subset must keep at least one TP level and one SL level for exact replay.
    Raises:
        ValueError: If the requested count is non-numeric, non-positive, or larger than total.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if total_count <= 0:
        raise ValueError(f"{field_name} total_count must be > 0")
    if requested_count is None:
        return total_count
    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError(f"{field_name} must be an integer when set")
    if requested_count <= 0 or requested_count > total_count:
        raise ValueError(f"{field_name} must satisfy 1 <= value <= {total_count}")
    return requested_count


def _slice_hit_times_level_subset_v2(
    *,
    hit_times: StageBHitTimesSliceV2,
    tp_level_count: int,
    sl_level_count: int,
) -> StageBHitTimesSliceV2:
    """
    Slice the Stage B hit-times grid to a leading bounded subset of TP and SL levels.

    Args:
        hit_times: Full local `1m hit-times` slice for one execution window.
        tp_level_count: Number of leading TP levels to retain.
        sl_level_count: Number of leading SL levels to retain.
    Returns:
        StageBHitTimesSliceV2: Bounded subset with the original execution sentinel preserved.
    Assumptions:
        Level ordering is deterministic, so prefix slicing keeps reviewable parity semantics.
    Raises:
        ValueError: If one requested level count is out of bounds for the source grid.
    Side Effects:
        Allocates a small Stage B slice wrapper over the bounded subset.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
    """
    normalized_tp_level_count = _normalize_bounded_level_count_v2(
        total_count=int(hit_times.tp_values.shape[0]),
        requested_count=tp_level_count,
        field_name="tp_level_count",
    )
    normalized_sl_level_count = _normalize_bounded_level_count_v2(
        total_count=int(hit_times.sl_values.shape[0]),
        requested_count=sl_level_count,
        field_name="sl_level_count",
    )
    return StageBHitTimesSliceV2(
        tp_values=hit_times.tp_values[:normalized_tp_level_count],
        sl_values=hit_times.sl_values[:normalized_sl_level_count],
        long_tp=hit_times.long_tp[:normalized_tp_level_count, :],
        long_sl=hit_times.long_sl[:normalized_sl_level_count, :],
        short_tp=hit_times.short_tp[:normalized_tp_level_count, :],
        short_sl=hit_times.short_sl[:normalized_sl_level_count, :],
        sentinel_index=hit_times.sentinel_index,
    )


def _rebase_hit_times_table_v2(
    *,
    values: np.ndarray,
    start: int,
    stop: int,
    local_sentinel_index: int,
) -> np.ndarray:
    """
    Rebase one strict hit-times table from slot-global to local execution coordinates.

    Args:
        values: Global strict hit-times table shaped `[level, T_exec_global]`.
        start: Inclusive global execution offset of the local window.
        stop: Exclusive global execution offset of the local window.
        local_sentinel_index: Local execution sentinel index (`stop - start`).
    Returns:
        np.ndarray: Rebased local hit-times table shaped `[level, local_sentinel_index]`.
    Assumptions:
        Hits outside `[start, stop)` must map to the local sentinel.
    Raises:
        ValueError: If the source table is not 2D or does not cover the requested local window.
    Side Effects:
        Allocates a rebased integer array.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if values.ndim != 2:
        raise ValueError("hit-times table must be 2D")
    if start < 0 or stop < start or stop > values.shape[1]:
        raise ValueError(
            "hit-times table width must cover exec_target_slice; "
            f"got slice({start}, {stop}), width={values.shape[1]}"
        )
    window = np.asarray(values[:, start:stop], dtype=np.int64)
    rebased = np.where(
        (window >= start) & (window < stop),
        window - start,
        local_sentinel_index,
    )
    return np.asarray(rebased, dtype=np.int64)


def _normalize_hit_times_level_grid_v2(
    *,
    field_name: str,
    values: np.ndarray,
) -> np.ndarray:
    """
    Normalize one artifact-backed TP/SL level grid for grid-agnostic Stage B runtime use.

    Args:
        field_name: Deterministic diagnostics field label.
        values: Candidate artifact `tp_values` or `sl_values` vector.
    Returns:
        np.ndarray: Canonical one-dimensional `np.float32` level grid.
    Assumptions:
        Shipped `hit_times/1m` artifacts publish TP/SL levels as explicit 1D arrays and Stage B
        reads grid shape from those artifact arrays instead of fixed-size literals.
    Raises:
        ValueError: If the grid is not one-dimensional or has no levels.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    normalized = np.asarray(values, dtype=np.float32)
    if normalized.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D array")
    if normalized.shape[0] <= 0:
        raise ValueError(f"{field_name} must contain at least one level")
    return normalized


def _build_reference_total_return_matrix_v2(
    *,
    compact_trades: tuple[StageACompactTradeV2, ...],
    hit_times: StageBHitTimesSliceV2,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    fee_rate: float,
    close_on_end: bool,
) -> np.ndarray:
    """
    Build the slow exact-replay total-return matrix for a bounded Stage B subset.

    Args:
        compact_trades: Ordered compact trades already reduced to the bounded subset.
        hit_times: Bounded local `1m hit-times` slice for the self-check.
        exec_open: Local `1m` execution opens.
        exec_close: Local `1m` execution closes.
        fee_rate: Per-side fee rate expressed as decimal fraction.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
    Returns:
        np.ndarray: Exact total-return matrix shaped `[n_tp_subset, n_sl_subset]`.
    Assumptions:
        This helper exists only for the reference-vs-fast self-check and therefore favors exact
        replay over production-path speed.
    Raises:
        ValueError: Propagated from exact replay or metric computation on invalid inputs.
    Side Effects:
        Allocates one dense matrix by running exact replay for every bounded TP/SL cell.
    Docs:
      - docs/architecture/backtest/README.md
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """
    total_return_pct = np.empty(
        (hit_times.tp_values.shape[0], hit_times.sl_values.shape[0]),
        dtype=np.float64,
    )
    for tp_index in range(hit_times.tp_values.shape[0]):
        for sl_index in range(hit_times.sl_values.shape[0]):
            replay = replay_risk_cell_exact_v2(
                compact_trades=compact_trades,
                hit_times=hit_times,
                exec_open=exec_open,
                exec_close=exec_close,
                tp_index=tp_index,
                sl_index=sl_index,
                close_on_end=close_on_end,
            )
            total_return_pct[tp_index, sl_index] = compute_stage_b_metrics_v2(
                replay=replay,
                fee_rate=fee_rate,
            ).total_return_pct
    return total_return_pct


def _assert_reference_vs_fast_self_check_parity_v2(
    *,
    fast_result: StageBFastSearchResultV2,
    reference_total_return_pct: np.ndarray,
    reference_best_tp_index: int,
    reference_best_sl_index: int,
    reference_best_total_return_pct: float,
    return_tolerance: float,
) -> float:
    """
    Assert deterministic parity between fast search and the slow exact-replay reference matrix.

    Args:
        fast_result: Fast-search result over the bounded subset.
        reference_total_return_pct: Slow exact-replay matrix over the same bounded subset.
        reference_best_tp_index: Reference best TP index from the bounded subset.
        reference_best_sl_index: Reference best SL index from the bounded subset.
        reference_best_total_return_pct: Reference best total return for the bounded subset.
        return_tolerance: Allowed absolute total-return drift between fast and reference outputs.
    Returns:
        float: Maximum absolute total-return difference across the bounded subset.
    Assumptions:
        Both matrices already cover the same bounded subset and use the same deterministic order.
    Raises:
        ValueError: If `return_tolerance` is invalid.
        AssertionError: If best-cell coordinates or matrix values drift beyond tolerance.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
    Related:
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
    """
    if isinstance(return_tolerance, bool) or not isinstance(return_tolerance, int | float):
        raise ValueError("return_tolerance must be numeric")
    normalized_tolerance = float(return_tolerance)
    if normalized_tolerance < 0.0:
        raise ValueError("return_tolerance must be >= 0")
    if fast_result.total_return_pct.shape != reference_total_return_pct.shape:
        raise AssertionError(
            "reference-vs-fast self-check matrix shape drifted on the bounded subset"
        )
    if fast_result.best_tp_index != reference_best_tp_index:
        raise AssertionError(
            "reference-vs-fast self-check best_tp_index drifted on the bounded subset"
        )
    if fast_result.best_sl_index != reference_best_sl_index:
        raise AssertionError(
            "reference-vs-fast self-check best_sl_index drifted on the bounded subset"
        )
    best_total_return_diff = abs(
        fast_result.best_total_return_pct - reference_best_total_return_pct
    )
    if best_total_return_diff > normalized_tolerance:
        raise AssertionError(
            "reference-vs-fast self-check best_total_return_pct drifted on the bounded subset"
        )
    max_abs_total_return_diff = float(
        np.max(np.abs(fast_result.total_return_pct - reference_total_return_pct))
    )
    if max_abs_total_return_diff > normalized_tolerance:
        raise AssertionError(
            "reference-vs-fast self-check total_return_pct matrix drifted on the bounded subset"
        )
    return max_abs_total_return_diff


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
        Stage B runtime kernels operate on local `1m` execution arrays only.
    Raises:
        ValueError: If the array shape drifts from `sentinel_index`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
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


def _validate_compact_trade_for_stage_b_v2(
    *,
    trade: StageACompactTradeV2,
    sentinel_index: int,
    trade_index: int,
) -> None:
    """
    Validate one compact trade against the local execution sentinel contract.

    Args:
        trade: Compact trade entry.
        sentinel_index: Local execution sentinel index.
        trade_index: Stable ordinal index for diagnostics.
    Returns:
        None.
    Assumptions:
        `sig_exit_exec_idx == sentinel_index` denotes missing signal exit in the local window.
    Raises:
        ValueError: If one entry/exit index violates the local sentinel bounds.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if trade.entry_exec_idx < 0 or trade.entry_exec_idx > sentinel_index:
        raise ValueError(
            f"compact trade[{trade_index}] entry_exec_idx must be in [0, sentinel_index]"
        )
    if trade.sig_exit_exec_idx < trade.entry_exec_idx or trade.sig_exit_exec_idx > sentinel_index:
        raise ValueError(
            f"compact trade[{trade_index}] sig_exit_exec_idx must be in "
            "[entry_exec_idx, sentinel_index]"
        )


def _validate_level_indexes_v2(
    *,
    hit_times: StageBHitTimesSliceV2,
    tp_index: int | None,
    sl_index: int | None,
) -> None:
    """
    Validate optional TP/SL indexes against the local hit-times grid dimensions.

    Args:
        hit_times: Local `1m hit-times` slice for the execution window.
        tp_index: Optional TP level index.
        sl_index: Optional SL level index.
    Returns:
        None.
    Assumptions:
        Disabled TP/SL axes are represented as `None`.
    Raises:
        ValueError: If one enabled level index is out of range.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if tp_index is not None and (tp_index < 0 or tp_index >= hit_times.tp_values.shape[0]):
        raise ValueError("tp_index is out of range for local tp_values grid")
    if sl_index is not None and (sl_index < 0 or sl_index >= hit_times.sl_values.shape[0]):
        raise ValueError("sl_index is out of range for local sl_values grid")


def _tp_table_for_trade_direction_v2(
    *,
    hit_times: StageBHitTimesSliceV2,
    direction: int,
) -> np.ndarray:
    """
    Resolve the TP lookup table matching one trade direction.

    Args:
        hit_times: Local `1m hit-times` slice.
        direction: Trade direction (`+1` long, `-1` short).
    Returns:
        np.ndarray: Direction-specific TP table.
    Assumptions:
        Equity-space TP factors are identical for long and short; the table picks the direction.
    Raises:
        ValueError: If direction is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if direction == 1:
        return hit_times.long_tp
    if direction == -1:
        return hit_times.short_tp
    raise ValueError(f"trade direction must be -1 or 1; got {direction!r}")


def _sl_table_for_trade_direction_v2(
    *,
    hit_times: StageBHitTimesSliceV2,
    direction: int,
) -> np.ndarray:
    """
    Resolve the SL lookup table matching one trade direction.

    Args:
        hit_times: Local `1m hit-times` slice.
        direction: Trade direction (`+1` long, `-1` short).
    Returns:
        np.ndarray: Direction-specific SL table.
    Assumptions:
        Equity-space SL factors are identical for long and short; the table picks the direction.
    Raises:
        ValueError: If direction is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if direction == 1:
        return hit_times.long_sl
    if direction == -1:
        return hit_times.short_sl
    raise ValueError(f"trade direction must be -1 or 1; got {direction!r}")


def _level_factor_from_tp_rate_v2(*, tp_rate: float) -> float:
    """
    Convert one TP rate into notebook-equivalent gross equity factor.

    Args:
        tp_rate: Decimal TP rate (`0.01 == 1%`).
    Returns:
        float: Gross factor before fees.
    Assumptions:
        Production runtime keeps equity-space TP factors aligned with notebook `1 + tp`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    return 1.0 + float(tp_rate)


def _level_factor_from_sl_rate_v2(*, sl_rate: float) -> float:
    """
    Convert one SL rate into notebook-equivalent gross equity factor.

    Args:
        sl_rate: Decimal SL rate (`0.01 == 1%`).
    Returns:
        float: Gross factor before fees, clipped at zero for extreme short-model cases.
    Assumptions:
        Production runtime keeps equity-space SL factors aligned with notebook `max(0, 1 - sl)`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    return max(0.0, 1.0 - float(sl_rate))


def _signal_or_end_factor_v2(
    *,
    direction: int,
    entry_open: float,
    exit_price: float,
) -> float:
    """
    Compute notebook-derived gross factor for signal-exit or end-of-series close.

    Args:
        direction: Trade direction (`+1` long, `-1` short).
        entry_open: Entry execution-bar open price.
        exit_price: Signal-exit open price or final close price.
    Returns:
        float: Gross factor before fees.
    Assumptions:
        Short trades use the x1 USDT ROI model `max(0, 2 - exit / entry)`.
    Raises:
        ValueError: If direction is unsupported or entry price is non-positive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if entry_open <= 0.0:
        raise ValueError("entry_open must be > 0")
    if direction == 1:
        return exit_price / entry_open
    if direction == -1:
        return max(0.0, 2.0 - (exit_price / entry_open))
    raise ValueError(f"trade direction must be -1 or 1; got {direction!r}")


def _lower_bound_ge_hit_v2(
    *,
    hit_values: np.ndarray,
    threshold: int,
) -> int:
    """
    Return the first level index whose hit time is greater than or equal to `threshold`.

    Args:
        hit_values: Monotone hit-time vector for one lookup start.
        threshold: Execution index threshold.
    Returns:
        int: Leftmost level index with `hit >= threshold`, or `len(hit_values)` if absent.
    Assumptions:
        Hit-time tables are non-decreasing by level as enforced by artifact manifests.
    Raises:
        ValueError: If the hit vector is not one-dimensional.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    normalized = np.asarray(hit_values, dtype=np.int64)
    if normalized.ndim != 1:
        raise ValueError("hit_values must be 1D")
    return int(np.searchsorted(normalized, threshold, side="left"))


def _first_equal_hit_v2(
    *,
    hit_values: np.ndarray,
    sentinel_index: int,
) -> int:
    """
    Return the first level index whose hit time equals the local `sentinel_index`.

    Args:
        hit_values: Monotone hit-time vector for one lookup start.
        sentinel_index: Local execution sentinel index.
    Returns:
        int: Leftmost level index whose hit equals `sentinel_index`, or `len(hit_values)`.
    Assumptions:
        Sentinel values are grouped at the tail because hit-time vectors are monotone by level.
    Raises:
        ValueError: If the hit vector is not one-dimensional.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    return _lower_bound_ge_hit_v2(hit_values=hit_values, threshold=sentinel_index)


def _add_row_range_v2(
    *,
    diff: np.ndarray,
    row_i: int,
    col_start: int,
    col_stop: int,
    value: float,
) -> None:
    """
    Add one scalar to a half-open row segment by row-diff update.

    Args:
        diff: Row-diff buffer shaped `[n_tp, n_sl + 1]`.
        row_i: Row index.
        col_start: Inclusive column start.
        col_stop: Exclusive column stop.
        value: Scalar value to add.
    Returns:
        None.
    Assumptions:
        Caller already validated half-open bounds against buffer shape.
    Raises:
        None.
    Side Effects:
        Mutates `diff` in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if col_start >= col_stop:
        return
    diff[row_i, col_start] += value
    diff[row_i, col_stop] -= value


def _add_col_range_v2(
    *,
    diff: np.ndarray,
    row_start: int,
    row_stop: int,
    col_j: int,
    value: float,
) -> None:
    """
    Add one scalar to a half-open column segment by column-diff update.

    Args:
        diff: Column-diff buffer shaped `[n_tp + 1, n_sl]`.
        row_start: Inclusive row start.
        row_stop: Exclusive row stop.
        col_j: Column index.
        value: Scalar value to add.
    Returns:
        None.
    Assumptions:
        Caller already validated half-open bounds against buffer shape.
    Raises:
        None.
    Side Effects:
        Mutates `diff` in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if row_start >= row_stop:
        return
    diff[row_start, col_j] += value
    diff[row_stop, col_j] -= value


def _add_rect_v2(
    *,
    diff: np.ndarray,
    row_start: int,
    col_start: int,
    row_stop: int,
    col_stop: int,
    value: float,
) -> None:
    """
    Add one scalar to a half-open rectangle by 2D diff update.

    Args:
        diff: Rectangle-diff buffer shaped `[n_tp + 1, n_sl + 1]`.
        row_start: Inclusive row start.
        col_start: Inclusive column start.
        row_stop: Exclusive row stop.
        col_stop: Exclusive column stop.
        value: Scalar value to add.
    Returns:
        None.
    Assumptions:
        Caller already validated half-open bounds against buffer shape.
    Raises:
        None.
    Side Effects:
        Mutates `diff` in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if row_start >= row_stop or col_start >= col_stop:
        return
    diff[row_start, col_start] += value
    diff[row_stop, col_start] -= value
    diff[row_start, col_stop] -= value
    diff[row_stop, col_stop] += value


def _add_weighted_row_range_v2(
    *,
    row_diff: np.ndarray,
    row_zero_diff: np.ndarray,
    row_i: int,
    col_start: int,
    col_stop: int,
    factor: float,
) -> None:
    """
    Add one row-segment contribution in factor space, keeping exact zero-factor handling.

    Args:
        row_diff: Floating row-diff buffer for positive-factor log contributions.
        row_zero_diff: Integer row-diff buffer counting zero-factor contributions.
        row_i: Target row index.
        col_start: Inclusive column start.
        col_stop: Exclusive column stop.
        factor: Positive factor or exact zero.
    Returns:
        None.
    Assumptions:
        Zero factors must win over any positive log contributions after integration.
    Raises:
        ValueError: If factor is negative.
    Side Effects:
        Mutates one of the diff buffers in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if factor < 0.0:
        raise ValueError("factor must be >= 0")
    if factor == 0.0:
        _add_row_range_v2(
            diff=row_zero_diff,
            row_i=row_i,
            col_start=col_start,
            col_stop=col_stop,
            value=1,
        )
        return
    _add_row_range_v2(
        diff=row_diff,
        row_i=row_i,
        col_start=col_start,
        col_stop=col_stop,
        value=math.log(factor),
    )


def _add_weighted_col_range_v2(
    *,
    col_diff: np.ndarray,
    col_zero_diff: np.ndarray,
    row_start: int,
    row_stop: int,
    col_j: int,
    factor: float,
) -> None:
    """
    Add one column-segment contribution in factor space with exact zero-factor handling.

    Args:
        col_diff: Floating column-diff buffer for positive-factor log contributions.
        col_zero_diff: Integer column-diff buffer counting zero-factor contributions.
        row_start: Inclusive row start.
        row_stop: Exclusive row stop.
        col_j: Target column index.
        factor: Positive factor or exact zero.
    Returns:
        None.
    Assumptions:
        Zero factors must win over any positive log contributions after integration.
    Raises:
        ValueError: If factor is negative.
    Side Effects:
        Mutates one of the diff buffers in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if factor < 0.0:
        raise ValueError("factor must be >= 0")
    if factor == 0.0:
        _add_col_range_v2(
            diff=col_zero_diff,
            row_start=row_start,
            row_stop=row_stop,
            col_j=col_j,
            value=1,
        )
        return
    _add_col_range_v2(
        diff=col_diff,
        row_start=row_start,
        row_stop=row_stop,
        col_j=col_j,
        value=math.log(factor),
    )


def _add_weighted_rect_v2(
    *,
    rect_diff: np.ndarray,
    rect_zero_diff: np.ndarray,
    row_start: int,
    col_start: int,
    row_stop: int,
    col_stop: int,
    factor: float,
) -> None:
    """
    Add one rectangle contribution in factor space with exact zero-factor handling.

    Args:
        rect_diff: Floating rectangle-diff buffer for positive-factor log contributions.
        rect_zero_diff: Integer rectangle-diff buffer counting zero-factor contributions.
        row_start: Inclusive row start.
        col_start: Inclusive column start.
        row_stop: Exclusive row stop.
        col_stop: Exclusive column stop.
        factor: Positive factor or exact zero.
    Returns:
        None.
    Assumptions:
        Zero factors must win over any positive log contributions after integration.
    Raises:
        ValueError: If factor is negative.
    Side Effects:
        Mutates one of the diff buffers in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if factor < 0.0:
        raise ValueError("factor must be >= 0")
    if factor == 0.0:
        _add_rect_v2(
            diff=rect_zero_diff,
            row_start=row_start,
            col_start=col_start,
            row_stop=row_stop,
            col_stop=col_stop,
            value=1,
        )
        return
    _add_rect_v2(
        diff=rect_diff,
        row_start=row_start,
        col_start=col_start,
        row_stop=row_stop,
        col_stop=col_stop,
        value=math.log(factor),
    )


def _add_terminal_region_contribution_v2(
    *,
    direction: int,
    entry_open: float,
    signal_exit_exec_idx: int,
    exec_open: np.ndarray,
    exec_close: np.ndarray,
    sentinel_index: int,
    close_on_end: bool,
    fee_two_sides: float,
    row_stop: int,
    col_stop: int,
    rect_diff: np.ndarray,
    rect_zero_diff: np.ndarray,
) -> None:
    """
    Add uniform full-grid contribution when TP/SL lookup cannot start after entry.

    Args:
        direction: Trade direction (`+1` long, `-1` short).
        entry_open: Entry execution-bar open price.
        signal_exit_exec_idx: Opposite-signal execution index or local sentinel.
        exec_open: Local execution opens.
        exec_close: Local execution closes.
        sentinel_index: Local execution sentinel index.
        close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
        fee_two_sides: Two-sided fee multiplier.
        row_stop: Exclusive TP dimension upper bound.
        col_stop: Exclusive SL dimension upper bound.
        rect_diff: Floating rectangle-diff buffer.
        rect_zero_diff: Integer rectangle-diff buffer counting zero-factor contributions.
    Returns:
        None.
    Assumptions:
        When `entry_exec == sentinel_index - 1`, TP/SL lookup is impossible and exit is uniform.
    Raises:
        None.
    Side Effects:
        Mutates rectangle diff buffers in place.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if signal_exit_exec_idx < sentinel_index:
        exit_open = float(exec_open[signal_exit_exec_idx])
        signal_factor = (
            _signal_or_end_factor_v2(
                direction=direction,
                entry_open=entry_open,
                exit_price=exit_open,
            )
            if exit_open > 0.0
            else 1.0
        )
        _add_weighted_rect_v2(
            rect_diff=rect_diff,
            rect_zero_diff=rect_zero_diff,
            row_start=0,
            col_start=0,
            row_stop=row_stop,
            col_stop=col_stop,
            factor=fee_two_sides * signal_factor,
        )
        return
    if close_on_end and sentinel_index > 0:
        last_close = float(exec_close[sentinel_index - 1])
        end_factor = (
            _signal_or_end_factor_v2(
                direction=direction,
                entry_open=entry_open,
                exit_price=last_close,
            )
            if last_close > 0.0
            else 1.0
        )
        _add_weighted_rect_v2(
            rect_diff=rect_diff,
            rect_zero_diff=rect_zero_diff,
            row_start=0,
            col_start=0,
            row_stop=row_stop,
            col_stop=col_stop,
            factor=fee_two_sides * end_factor,
        )


def _integrate_row_diff_v2(*, values: np.ndarray) -> np.ndarray:
    """
    Integrate one row-diff buffer into its dense `[n_tp, n_sl]` contribution matrix.

    Args:
        values: Row-diff buffer shaped `[n_tp, n_sl + 1]`.
    Returns:
        np.ndarray: Dense row-contribution matrix shaped `[n_tp, n_sl]`.
    Assumptions:
        Row-diff buffers integrate along the SL axis only.
    Raises:
        ValueError: If the diff buffer is not 2D.
    Side Effects:
        Allocates one dense integration result.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if values.ndim != 2:
        raise ValueError("row diff buffer must be 2D")
    return np.cumsum(values[:, :-1], axis=1)


def _integrate_col_diff_v2(*, values: np.ndarray) -> np.ndarray:
    """
    Integrate one column-diff buffer into its dense `[n_tp, n_sl]` contribution matrix.

    Args:
        values: Column-diff buffer shaped `[n_tp + 1, n_sl]`.
    Returns:
        np.ndarray: Dense column-contribution matrix shaped `[n_tp, n_sl]`.
    Assumptions:
        Column-diff buffers integrate along the TP axis only.
    Raises:
        ValueError: If the diff buffer is not 2D.
    Side Effects:
        Allocates one dense integration result.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if values.ndim != 2:
        raise ValueError("column diff buffer must be 2D")
    return np.cumsum(values[:-1, :], axis=0)


def _integrate_rect_diff_v2(*, values: np.ndarray) -> np.ndarray:
    """
    Integrate one rectangle-diff buffer into its dense `[n_tp, n_sl]` contribution matrix.

    Args:
        values: Rectangle-diff buffer shaped `[n_tp + 1, n_sl + 1]`.
    Returns:
        np.ndarray: Dense rectangle-contribution matrix shaped `[n_tp, n_sl]`.
    Assumptions:
        Rectangle-diff buffers integrate by 2D prefix sum over both TP and SL axes.
    Raises:
        ValueError: If the diff buffer is not 2D.
    Side Effects:
        Allocates one dense integration result.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    """
    if values.ndim != 2:
        raise ValueError("rectangle diff buffer must be 2D")
    return np.cumsum(np.cumsum(values[:-1, :-1], axis=0), axis=1)


__all__ = [
    "replay_best_risk_cell_exact_v2",
    "replay_risk_cell_exact_v2",
    "resolve_risk_trade_exit_1m_v2",
    "run_reference_vs_fast_self_check_v2",
    "search_risk_cells_total_return_fast_v2",
    "slice_hit_times_to_execution_window_v2",
]
