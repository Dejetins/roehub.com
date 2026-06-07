from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import trading.contexts.backtest.application.services.v2.tp_sl_exact as tp_sl_exact
from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningResult,
    BacktestPreparePoolsResult,
    BacktestTpSlExactConfig,
    BacktestTpSlHitTimesResult,
    BacktestTpSlHitTimesSubset,
)
from trading.contexts.backtest.application.services.v2.matrix_backend.trade_tape import (
    SparseTradeTapeExtraction,
    extract_selected_trade_tapes,
)

TP_SL_SELECTED_CELL_SHADOW_ENV_KEY = "ROEHUB_BACKTEST_TP_SL_SELECTED_CELL_SHADOW"
TP_SL_SELECTED_CELL_MAX_TP_COUNT = 8
TP_SL_SELECTED_CELL_MAX_SL_COUNT = 8
TP_SL_SELECTED_CELL_DEFAULT_MAX_CANDIDATES = 8
TP_SL_SELECTED_CELL_RETURN_TOLERANCE_PCT = 1.0e-4
SL_WINS_TIE_RULE_LITERAL = "SL wins"


@dataclass(frozen=True, slots=True)
class ByEntryHitTimesLayout:
    selected_entry_count: int
    tp_count: int
    sl_count: int
    selected_arrays: Mapping[str, np.ndarray]
    materialize_ms: float

    @property
    def selected_arrays_bytes(self) -> int:
        return sum(int(array.nbytes) for array in self.selected_arrays.values())

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_tp_sl_by_entry_hit_times_layout_v1",
            "selected_entry_count": self.selected_entry_count,
            "tp_count": self.tp_count,
            "sl_count": self.sl_count,
            "materialize_ms": self.materialize_ms,
            "selected_arrays_bytes": self.selected_arrays_bytes,
            "arrays": {
                name: {
                    "dtype": "uint32",
                    "shape": list(array.shape),
                    "c_contiguous": bool(array.flags.c_contiguous),
                }
                for name, array in self.selected_arrays.items()
            },
            "sidecar_policy": "job-local selected arrays only; no publisher or manifest write",
            "required_sidecar_names": [
                "long_tp_by_entry.u32.npy",
                "long_sl_by_entry.u32.npy",
                "short_tp_by_entry.u32.npy",
                "short_sl_by_entry.u32.npy",
            ],
        }


@dataclass(frozen=True, slots=True)
class TpSlSelectedCellValidation:
    status: str
    enabled: bool
    tp_count: int
    sl_count: int
    tp_sl_cells: int
    candidate_count: int
    selected_cell_scores: int
    parity_pass: bool
    max_abs_return_diff_pct: float
    best_cell_equal: bool
    trade_count_equal: bool
    sl_wins_tie_rule: str
    elapsed_ms: float
    tape: SparseTradeTapeExtraction | None = None
    by_entry_layout: ByEntryHitTimesLayout | None = None
    first_mismatch: Mapping[str, Any] | None = None
    skip_reason: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_tp_sl_selected_cell_shadow_v1",
            "status": self.status,
            "enabled": self.enabled,
            "tp_count": self.tp_count,
            "sl_count": self.sl_count,
            "tp_sl_cells": self.tp_sl_cells,
            "limit": {
                "tp_count <= 8": self.tp_count <= TP_SL_SELECTED_CELL_MAX_TP_COUNT,
                "sl_count <= 8": self.sl_count <= TP_SL_SELECTED_CELL_MAX_SL_COUNT,
            },
            "candidate_count": self.candidate_count,
            "selected_cell_scores": self.selected_cell_scores,
            "parity_pass": self.parity_pass,
            "max_abs_return_diff_pct": self.max_abs_return_diff_pct,
            "best_cell_equal": self.best_cell_equal,
            "trade_count_equal": self.trade_count_equal,
            "sl_wins_tie_rule": self.sl_wins_tie_rule,
            "elapsed_ms": self.elapsed_ms,
            "skip_reason": self.skip_reason,
            "trade_tape": None if self.tape is None else self.tape.as_mapping(),
            "by_entry_layout": None
            if self.by_entry_layout is None
            else self.by_entry_layout.as_mapping(),
            "first_mismatch": None
            if self.first_mismatch is None
            else dict(self.first_mismatch),
            "production_topn_feed": "current_path_only",
        }


def tp_sl_selected_cell_shadow_enabled() -> bool:
    raw = os.environ.get(TP_SL_SELECTED_CELL_SHADOW_ENV_KEY, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_tp_sl_selected_cell_shadow(
    *,
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    hit_times_result: BacktestTpSlHitTimesResult,
    normalized_request: Mapping[str, Any],
    max_candidates: int = TP_SL_SELECTED_CELL_DEFAULT_MAX_CANDIDATES,
) -> TpSlSelectedCellValidation:
    start = time.perf_counter()
    hit_times = hit_times_result.hit_times
    tp_count = int(hit_times.tp_values.shape[0])
    sl_count = int(hit_times.sl_values.shape[0])
    tp_sl_cells = tp_count * sl_count
    if tp_count > TP_SL_SELECTED_CELL_MAX_TP_COUNT or sl_count > TP_SL_SELECTED_CELL_MAX_SL_COUNT:
        return _skipped(
            enabled=True,
            start=start,
            tp_count=tp_count,
            sl_count=sl_count,
            reason="grid_larger_than_stage_08_selected_cell_limit",
        )
    selected_batch = tp_sl_exact._next_selected_candidate_batch(  # noqa: SLF001
        tp_sl_exact._iter_selected_candidate_batches(  # noqa: SLF001
            prepared_result=prepared_result,
            combo_planning_result=combo_planning_result,
        )
    )
    if selected_batch is None:
        return _skipped(
            enabled=True,
            start=start,
            tp_count=tp_count,
            sl_count=sl_count,
            reason="no_selected_candidates",
        )

    selected_rows_by_indicator = {
        indicator_id: np.ascontiguousarray(
            np.asarray(rows, dtype=np.int32)[:max_candidates],
            dtype=np.int32,
        )
        for indicator_id, rows in selected_batch.rows_by_indicator.items()
    }
    candidate_count = tp_sl_exact._selected_size(selected_rows_by_indicator)  # noqa: SLF001
    execution_settings = tp_sl_exact._execution_settings_from_normalized(  # noqa: SLF001
        normalized_request,
        expected_direction_mode=combo_planning_result.backend.direction_mode,
        config=BacktestTpSlExactConfig(),
    )
    runtime = tp_sl_exact._tp_sl_runtime_context_from_prepared(  # noqa: SLF001
        prepared_result=prepared_result,
        hit_times=hit_times,
        execution_settings=execution_settings,
    )
    tape = extract_selected_trade_tapes(
        prepared_result=prepared_result,
        selected_rows_by_indicator=selected_rows_by_indicator,
        direction_mode=combo_planning_result.backend.direction_mode,
        max_candidates=max_candidates,
    )
    by_entry_layout = build_selected_by_entry_hit_times_layout(
        hit_times=hit_times,
        selected_entries=tape.selected_entries,
    )

    buffers = tp_sl_exact._allocate_score_buffers(candidate_count)  # noqa: SLF001
    tp_sl_exact.evaluate_tp_sl_exact_chunk(
        selected_rows_by_indicator=selected_rows_by_indicator,
        prepared_result=prepared_result,
        combo_planning_result=combo_planning_result,
        hit_times=hit_times,
        runtime=runtime,
        buffers=buffers,
    )

    first_mismatch: dict[str, Any] | None = None
    max_abs_return_diff = 0.0
    best_cell_equal = True
    trade_count_equal = True
    for candidate_idx, candidate_tape in enumerate(tape.tapes):
        selected = _score_selected_cells_for_tape(
            tape=candidate_tape,
            hit_times=hit_times,
            runtime=runtime,
        )
        fast_tp = int(buffers.best_tp_idx[candidate_idx])
        fast_sl = int(buffers.best_sl_idx[candidate_idx])
        fast_return = float(buffers.total_return_pct[candidate_idx])
        fast_trades = int(buffers.trade_count[candidate_idx])
        abs_diff = abs(float(selected["total_return_pct"]) - fast_return)
        max_abs_return_diff = max(max_abs_return_diff, abs_diff)
        same_cell = int(selected["best_tp_idx"]) == fast_tp and int(
            selected["best_sl_idx"]
        ) == fast_sl
        same_trades = int(selected["trade_count"]) == fast_trades
        if not same_cell and abs_diff > TP_SL_SELECTED_CELL_RETURN_TOLERANCE_PCT:
            best_cell_equal = False
        if not same_trades:
            trade_count_equal = False
        if (
            first_mismatch is None
            and (
                (not same_cell and abs_diff > TP_SL_SELECTED_CELL_RETURN_TOLERANCE_PCT)
                or not same_trades
                or abs_diff > TP_SL_SELECTED_CELL_RETURN_TOLERANCE_PCT
            )
        ):
            first_mismatch = {
                "candidate_idx": candidate_idx,
                "selected_best_tp_idx": int(selected["best_tp_idx"]),
                "selected_best_sl_idx": int(selected["best_sl_idx"]),
                "fast_best_tp_idx": fast_tp,
                "fast_best_sl_idx": fast_sl,
                "selected_trade_count": int(selected["trade_count"]),
                "fast_trade_count": fast_trades,
                "selected_total_return_pct": float(selected["total_return_pct"]),
                "fast_total_return_pct": fast_return,
                "abs_return_diff_pct": abs_diff,
            }

    parity_pass = (
        first_mismatch is None
        and best_cell_equal
        and trade_count_equal
        and max_abs_return_diff <= TP_SL_SELECTED_CELL_RETURN_TOLERANCE_PCT
    )
    return TpSlSelectedCellValidation(
        status="passed" if parity_pass else "failed",
        enabled=True,
        tp_count=tp_count,
        sl_count=sl_count,
        tp_sl_cells=tp_sl_cells,
        candidate_count=candidate_count,
        selected_cell_scores=candidate_count * tp_sl_cells,
        parity_pass=parity_pass,
        max_abs_return_diff_pct=max_abs_return_diff,
        best_cell_equal=best_cell_equal,
        trade_count_equal=trade_count_equal,
        sl_wins_tie_rule=SL_WINS_TIE_RULE_LITERAL,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        tape=tape,
        by_entry_layout=by_entry_layout,
        first_mismatch=first_mismatch,
    )


def build_selected_by_entry_hit_times_layout(
    *,
    hit_times: BacktestTpSlHitTimesSubset,
    selected_entries: np.ndarray,
) -> ByEntryHitTimesLayout:
    layout_start = time.perf_counter()
    entries = np.ascontiguousarray(np.asarray(selected_entries, dtype=np.int32))
    if int(entries.shape[0]) == 0:
        arrays = {
            "long_tp_by_entry.u32.npy": np.empty(
                (0, int(hit_times.tp_values.shape[0])), dtype=np.uint32
            ),
            "long_sl_by_entry.u32.npy": np.empty(
                (0, int(hit_times.sl_values.shape[0])), dtype=np.uint32
            ),
            "short_tp_by_entry.u32.npy": np.empty(
                (0, int(hit_times.tp_values.shape[0])), dtype=np.uint32
            ),
            "short_sl_by_entry.u32.npy": np.empty(
                (0, int(hit_times.sl_values.shape[0])), dtype=np.uint32
            ),
        }
    else:
        arrays = {
            "long_tp_by_entry.u32.npy": np.ascontiguousarray(hit_times.long_tp[:, entries].T),
            "long_sl_by_entry.u32.npy": np.ascontiguousarray(hit_times.long_sl[:, entries].T),
            "short_tp_by_entry.u32.npy": np.ascontiguousarray(hit_times.short_tp[:, entries].T),
            "short_sl_by_entry.u32.npy": np.ascontiguousarray(hit_times.short_sl[:, entries].T),
        }
    return ByEntryHitTimesLayout(
        selected_entry_count=int(entries.shape[0]),
        tp_count=int(hit_times.tp_values.shape[0]),
        sl_count=int(hit_times.sl_values.shape[0]),
        selected_arrays=arrays,
        materialize_ms=(time.perf_counter() - layout_start) * 1000.0,
    )


def _score_selected_cells_for_tape(
    *,
    tape: Any,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: Any,
) -> dict[str, float | int]:
    best_return = -math.inf
    best_tp = 0
    best_sl = 0
    best_trade_count = 0
    for tp_idx in range(int(hit_times.tp_values.shape[0])):
        for sl_idx in range(int(hit_times.sl_values.shape[0])):
            _one_tp, _one_sl, total_return, trade_count = (
                tp_sl_exact.evaluate_tp_sl_reference_trade_list_direct(
                    tape.entry_abs,
                    tape.direction,
                    tape.signal_exit_abs,
                    runtime.price_open_15m,
                    runtime.last_close_15m,
                    np.ascontiguousarray(hit_times.long_tp[tp_idx : tp_idx + 1]),
                    np.ascontiguousarray(hit_times.long_sl[sl_idx : sl_idx + 1]),
                    np.ascontiguousarray(hit_times.short_tp[tp_idx : tp_idx + 1]),
                    np.ascontiguousarray(hit_times.short_sl[sl_idx : sl_idx + 1]),
                    np.ascontiguousarray(runtime.log_fac_tp_long[tp_idx : tp_idx + 1]),
                    np.ascontiguousarray(runtime.log_fac_sl_long[sl_idx : sl_idx + 1]),
                    np.ascontiguousarray(runtime.log_fac_tp_short[tp_idx : tp_idx + 1]),
                    np.ascontiguousarray(runtime.log_fac_sl_short[sl_idx : sl_idx + 1]),
                    runtime.log_fee_two_sides,
                    runtime.close_on_end,
                    runtime.initial_cash_quote,
                    runtime.sizing_mode_code,
                    runtime.quote_amount,
                    runtime.equity_pct,
                    runtime.min_quote,
                    runtime.max_quote,
                    runtime.safe_profit_percent,
                    runtime.use_profit_lock,
                    runtime.t_exec_abs_15m,
                )
            )
            if float(total_return) > best_return:
                best_return = float(total_return)
                best_tp = tp_idx
                best_sl = sl_idx
                best_trade_count = int(trade_count)
    if not math.isfinite(best_return):
        best_return = 0.0
    return {
        "best_tp_idx": best_tp,
        "best_sl_idx": best_sl,
        "total_return_pct": best_return * 100.0,
        "trade_count": best_trade_count,
    }


def _skipped(
    *,
    enabled: bool,
    start: float,
    tp_count: int,
    sl_count: int,
    reason: str,
) -> TpSlSelectedCellValidation:
    return TpSlSelectedCellValidation(
        status="skipped",
        enabled=enabled,
        tp_count=tp_count,
        sl_count=sl_count,
        tp_sl_cells=tp_count * sl_count,
        candidate_count=0,
        selected_cell_scores=0,
        parity_pass=False,
        max_abs_return_diff_pct=0.0,
        best_cell_equal=False,
        trade_count_equal=False,
        sl_wins_tie_rule=SL_WINS_TIE_RULE_LITERAL,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        skip_reason=reason,
    )


__all__ = [
    "SL_WINS_TIE_RULE_LITERAL",
    "TP_SL_SELECTED_CELL_SHADOW_ENV_KEY",
    "ByEntryHitTimesLayout",
    "TpSlSelectedCellValidation",
    "build_selected_by_entry_hit_times_layout",
    "build_tp_sl_selected_cell_shadow",
    "tp_sl_selected_cell_shadow_enabled",
]
