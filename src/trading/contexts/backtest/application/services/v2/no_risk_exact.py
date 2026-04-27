from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence
from uuid import UUID

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    NO_RISK_SUMMARY_METRIC_NAMES,
    BacktestComboChunk,
    BacktestComboPlanningResult,
    BacktestNoRiskChunkScores,
    BacktestNoRiskExactScoringConfig,
    BacktestNoRiskExecutionConfig,
    BacktestNoRiskExecutionPrices,
    BacktestNoRiskScoringResult,
    BacktestNoRiskScoringTelemetry,
    BacktestNoRiskSelfCheckResult,
    BacktestNoRiskSummaryMetrics,
    BacktestNoRiskTopRow,
    BacktestPreparePoolsResult,
    BacktestProxyContext,
    BacktestProxyFilterResult,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.application.services.v2.combo_planning import (
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestComboPlanningService,
    build_local_row_pools,
    cartesian_combo_count,
    iter_combo_chunks,
    make_combo_idx_matrix,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant

SERVICE_WARMUP_STAGE_NAME = "service_warmup"
NUMBA_WARMUP_STAGE_NAME = "numba_warmup"
SAMPLE_WARMUP_STAGE_NAME = "sample_warmup"
SELF_CHECK_STAGE_NAME = "self_check"
EXACT_SCORING_STAGE_NAME = "exact_scoring"
HEAP_UPDATE_STAGE_NAME = "heap_update"
TOP_RESULT_PROXY_FILL_STAGE_NAME = "top_result_proxy_fill"
TOP_RESULT_ASSEMBLY_STAGE_NAME = "top_result_assembly"
TOTAL_WITHOUT_WARMUP_STAGE_NAME = "total_without_warmup"
PERSIST_TOP_N_IO_STAGE_NAME = "persist_top_n_io"

DIRECTION_MODE_LONG_ONLY = "long_only"
DIRECTION_MODE_LONG_SHORT_REVERSAL = "long_short_reversal"
_DIRECTION_MODE_LONG_ONLY_CODE = np.int8(1)
_DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE = np.int8(2)
_NEG_INF = np.float32(-1e30)
_SUPPORTED_SIZING_MODES = ("all_in", "fixed_quote")


class BacktestNoRiskExactScoringRejected(ValueError):
    """
    Deterministic internal rejection for unsupported Iteration 4 no-risk inputs.
    """


@dataclass(frozen=True, slots=True)
class _TopCandidate:
    variant_index: int
    local_rows: tuple[int, ...]
    row_ids: tuple[int, ...]
    metrics: BacktestNoRiskSummaryMetrics
    confirm_count: int | None
    proxy_score: float | None


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactScoringService:
    """
    Iteration 4 no-risk exact scoring over prepared pools and combo-planning contexts.
    """

    config: BacktestNoRiskExactScoringConfig = BacktestNoRiskExactScoringConfig()
    combo_planning_service: BacktestComboPlanningService = BacktestComboPlanningService()

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        normalized_request: Mapping[str, Any],
        execution_prices: BacktestNoRiskExecutionPrices,
    ) -> BacktestNoRiskScoringResult:
        if combo_planning_result.backend.risk_mode != "none":
            raise BacktestNoRiskExactScoringRejected(
                "BacktestNoRiskExactScoringService supports risk.mode='none' only"
            )
        execution_config = no_risk_execution_config_from_normalized(
            normalized_request=normalized_request,
            bars_per_year_exec=self.config.bars_per_year_exec,
        )
        ranking_metric, ranking_direction, top_n = _ranking_config(
            normalized_request=normalized_request,
            fallback_config=self.config,
        )
        _validate_execution_prices(
            prepared_result=prepared_result,
            execution_prices=execution_prices,
        )

        stage_timings = _zero_stage_timings()
        local_row_pools = build_local_row_pools(prepared_result=prepared_result)
        cartesian_combinations = cartesian_combo_count(
            indicator_ids=prepared_result.indicator_ids,
            local_row_pools=local_row_pools,
        )
        combo_iter = iter_combo_chunks(
            indicator_ids=prepared_result.indicator_ids,
            local_row_pools=local_row_pools,
            chunk_size=self.config.combo_chunk_size,
        )

        top_candidates: list[_TopCandidate] = []
        combo_chunks_processed = 0
        exact_candidates_evaluated = 0
        heap_candidates_seen = 0
        combo_global_start = 0
        self_check_result: BacktestNoRiskSelfCheckResult | None = None

        while True:
            try:
                combo_chunk = next(combo_iter)
            except StopIteration:
                break
            combo_chunks_processed += 1

            filter_result = self.combo_planning_service.proxy_filter(
                combo_chunk=combo_chunk,
                proxy_context=combo_planning_result.proxy_context,
            )
            if filter_result.selected_candidate_count <= 0:
                combo_global_start += combo_chunk.size
                continue

            if self_check_result is None:
                stage_start = time.perf_counter()
                self_check_result = run_fast_vs_reference_self_check_two(
                    selected_rows_by_indicator=filter_result.selected_rows_by_indicator,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    execution_config=execution_config,
                    execution_prices=execution_prices,
                    check_n=self.config.self_check_n,
                    ret_tol=self.config.self_check_return_tolerance,
                )
                stage_timings[SELF_CHECK_STAGE_NAME] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            chunk_scores = evaluate_no_risk_exact_chunk(
                selected_rows_by_indicator=filter_result.selected_rows_by_indicator,
                prepared_result=prepared_result,
                combo_planning_result=combo_planning_result,
                execution_config=execution_config,
                execution_prices=execution_prices,
            )
            stage_timings[EXACT_SCORING_STAGE_NAME] += time.perf_counter() - stage_start
            exact_candidates_evaluated += chunk_scores.size

            stage_start = time.perf_counter()
            new_candidates = _top_candidates_from_chunk(
                prepared_result=prepared_result,
                filter_result=filter_result,
                chunk_scores=chunk_scores,
                combo_global_start=combo_global_start,
                ranking_metric=ranking_metric,
                ranking_direction=ranking_direction,
                top_n=top_n,
            )
            heap_candidates_seen += len(new_candidates)
            top_candidates = heapq.nsmallest(
                top_n,
                [*top_candidates, *new_candidates],
                key=lambda candidate: _candidate_sort_key(
                    candidate=candidate,
                    ranking_metric=ranking_metric,
                    ranking_direction=ranking_direction,
                ),
            )
            stage_timings[HEAP_UPDATE_STAGE_NAME] += time.perf_counter() - stage_start
            combo_global_start += combo_chunk.size

        if self_check_result is None:
            self_check_result = BacktestNoRiskSelfCheckResult(
                checked=0,
                passed=True,
                exact_backend=combo_planning_result.backend.backend_id,
                direction_mode=combo_planning_result.backend.direction_mode,
                return_tolerance=self.config.self_check_return_tolerance,
                max_abs_exact_backend_ret_diff=0.0,
                trade_count_equal=True,
            )

        stage_start = time.perf_counter()
        ordered_candidates = heapq.nsmallest(
            top_n,
            top_candidates,
            key=lambda candidate: _candidate_sort_key(
                candidate=candidate,
                ranking_metric=ranking_metric,
                ranking_direction=ranking_direction,
            ),
        )
        filled_proxy_by_candidate_pos, proxy_filled = _proxy_fill_missing_candidates(
            candidates=ordered_candidates,
            prepared_result=prepared_result,
            proxy_context=combo_planning_result.proxy_context,
        )
        stage_timings[TOP_RESULT_PROXY_FILL_STAGE_NAME] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        top_rows = _build_top_rows_with_proxy_fill(
            candidates=ordered_candidates,
            filled_proxy_by_candidate_pos=filled_proxy_by_candidate_pos,
            prepared_result=prepared_result,
            execution_config=execution_config,
            normalized_request=normalized_request,
            ranking_metric=ranking_metric,
        )
        stage_timings[TOP_RESULT_ASSEMBLY_STAGE_NAME] += time.perf_counter() - stage_start
        stage_timings[TOTAL_WITHOUT_WARMUP_STAGE_NAME] = (
            stage_timings[SELF_CHECK_STAGE_NAME]
            + stage_timings[EXACT_SCORING_STAGE_NAME]
            + stage_timings[HEAP_UPDATE_STAGE_NAME]
            + stage_timings[TOP_RESULT_PROXY_FILL_STAGE_NAME]
            + stage_timings[TOP_RESULT_ASSEMBLY_STAGE_NAME]
        )

        return BacktestNoRiskScoringResult(
            top_rows=top_rows,
            telemetry=BacktestNoRiskScoringTelemetry(
                stage_timings=stage_timings,
                cartesian_combinations=cartesian_combinations,
                combo_chunks_processed=combo_chunks_processed,
                exact_candidates_evaluated=exact_candidates_evaluated,
                heap_candidates_seen=heap_candidates_seen,
                top_result_proxy_filled=proxy_filled,
                self_check=self_check_result,
            ),
        )


def no_risk_execution_config_from_normalized(
    *,
    normalized_request: Mapping[str, Any],
    bars_per_year_exec: float = 365.0 * 24.0 * 60.0,
) -> BacktestNoRiskExecutionConfig:
    risk = _required_mapping(normalized_request, "risk", path="normalized_request.risk")
    if str(risk.get("mode")) != "none":
        raise BacktestNoRiskExactScoringRejected("no-risk exact scoring requires risk.mode='none'")
    execution = _required_mapping(
        normalized_request,
        "execution",
        path="normalized_request.execution",
    )
    direction_mode = str(execution.get("direction_mode", "")).strip().lower()
    if direction_mode not in (DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL):
        raise BacktestNoRiskExactScoringRejected(
            f"Unsupported direction_mode={direction_mode!r}"
        )
    sizing = _required_mapping(execution, "sizing", path="normalized_request.execution.sizing")
    sizing_mode = str(sizing.get("mode", "")).strip().lower()
    if sizing_mode not in _SUPPORTED_SIZING_MODES:
        raise BacktestNoRiskExactScoringRejected(
            "Iteration 4 no-risk exact scoring supports sizing modes "
            f"{_SUPPORTED_SIZING_MODES!r}; got {sizing_mode!r}"
        )
    initial_cash_quote = _positive_float(
        execution.get("initial_cash_quote"),
        path="normalized_request.execution.initial_cash_quote",
    )
    fixed_quote = 0.0
    use_fixed_quote = sizing_mode == "fixed_quote"
    if use_fixed_quote:
        fixed_quote = _positive_float(
            sizing.get("quote_amount", sizing.get("fixed_quote")),
            path="normalized_request.execution.sizing.quote_amount",
        )
    profit_lock = _required_mapping(
        execution,
        "profit_lock",
        path="normalized_request.execution.profit_lock",
    )
    use_profit_lock = bool(profit_lock.get("enabled", False))
    safe_profit_percent = (
        float(profit_lock.get("safe_profit_percent", 0.0)) if use_profit_lock else 0.0
    )
    return BacktestNoRiskExecutionConfig(
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        initial_cash_quote=initial_cash_quote,
        fixed_quote=fixed_quote,
        fee_rate=_non_negative_float(
            execution.get("fee_rate", 0.0),
            path="normalized_request.execution.fee_rate",
        ),
        slippage_rate=_non_negative_float(
            execution.get("slippage_rate", 0.0),
            path="normalized_request.execution.slippage_rate",
        ),
        safe_profit_percent=safe_profit_percent,
        use_fixed_quote=use_fixed_quote,
        use_profit_lock=use_profit_lock,
        bars_per_year_exec=bars_per_year_exec,
        close_on_end=bool(execution.get("close_on_end", True)),
    )


def evaluate_no_risk_exact_chunk(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    execution_config: BacktestNoRiskExecutionConfig,
    execution_prices: BacktestNoRiskExecutionPrices,
) -> BacktestNoRiskChunkScores:
    indicator_ids = tuple(prepared_result.indicator_ids)
    backend_id = combo_planning_result.backend.backend_id
    size = _selected_size(
        selected_rows_by_indicator=selected_rows_by_indicator,
        indicator_ids=indicator_ids,
    )
    scores = _empty_chunk_scores(size)
    if size == 0:
        return scores

    pools = _ordered_pools(prepared_result=prepared_result)
    direction_code = _direction_mode_code(execution_config.direction_mode)
    use_fixed_quote = np.int8(1 if execution_config.use_fixed_quote else 0)
    use_profit_lock = np.int8(1 if execution_config.use_profit_lock else 0)
    close_on_end = np.int8(1 if execution_config.close_on_end else 0)
    t_exec = np.int32(prepared_result.execution_mapping.t_exec_limit_1m)
    signal_entry_exec_idx = np.ascontiguousarray(
        np.asarray(prepared_result.execution_mapping.signal_entry_exec_idx_15m, dtype=np.int32)
    )

    if backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND:
        if len(indicator_ids) != 2:
            raise BacktestNoRiskExactScoringRejected(
                "event_segments_2_no_risk requires arity=2"
            )
        left_rows = np.ascontiguousarray(
            np.asarray(selected_rows_by_indicator[indicator_ids[0]], dtype=np.int32)
        )
        right_rows = np.ascontiguousarray(
            np.asarray(selected_rows_by_indicator[indicator_ids[1]], dtype=np.int32)
        )
        evaluate_no_risk_event_segments_two(
            left_rows,
            right_rows,
            pools[0].segments.starts,
            pools[0].segments.ends,
            pools[0].segments.values,
            pools[0].segments.counts,
            pools[1].segments.starts,
            pools[1].segments.ends,
            pools[1].segments.values,
            pools[1].segments.counts,
            signal_entry_exec_idx,
            execution_prices.open_1m,
            execution_prices.close_1m,
            t_exec,
            execution_config.initial_cash_quote,
            execution_config.fixed_quote,
            execution_config.fee_rate,
            execution_config.slippage_rate,
            execution_config.safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            execution_config.bars_per_year_exec,
            close_on_end,
            direction_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return scores

    if backend_id == STREAMING_2_NO_RISK_BACKEND:
        if len(indicator_ids) != 2:
            raise BacktestNoRiskExactScoringRejected("streaming_2_no_risk requires arity=2")
        left_rows = np.ascontiguousarray(
            np.asarray(selected_rows_by_indicator[indicator_ids[0]], dtype=np.int32)
        )
        right_rows = np.ascontiguousarray(
            np.asarray(selected_rows_by_indicator[indicator_ids[1]], dtype=np.int32)
        )
        evaluate_no_risk_streaming_two(
            left_rows,
            right_rows,
            pools[0].trade_T,
            pools[1].trade_T,
            signal_entry_exec_idx,
            execution_prices.open_1m,
            execution_prices.close_1m,
            t_exec,
            execution_config.initial_cash_quote,
            execution_config.fixed_quote,
            execution_config.fee_rate,
            execution_config.slippage_rate,
            execution_config.safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            execution_config.bars_per_year_exec,
            close_on_end,
            direction_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return scores

    if backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND:
        exact_context = combo_planning_result.exact_context
        if (
            exact_context.starts is None
            or exact_context.ends is None
            or exact_context.values is None
            or exact_context.counts is None
        ):
            raise BacktestNoRiskExactScoringRejected(
                "event_segments_n_no_risk requires materialized exact context"
            )
        combo_idx_by_indicator = make_combo_idx_matrix(
            combo_chunk=BacktestComboChunk(
                indicator_ids=indicator_ids,
                rows_by_indicator=selected_rows_by_indicator,
            ),
            indicator_ids=indicator_ids,
        )
        segment_pos_workspace = np.empty(
            (combo_idx_by_indicator.shape[1], combo_idx_by_indicator.shape[0]),
            dtype=np.int32,
        )
        evaluate_no_risk_event_segments_n(
            combo_idx_by_indicator,
            exact_context.starts,
            exact_context.ends,
            exact_context.values,
            exact_context.counts,
            segment_pos_workspace,
            signal_entry_exec_idx,
            execution_prices.open_1m,
            execution_prices.close_1m,
            t_exec,
            execution_config.initial_cash_quote,
            execution_config.fixed_quote,
            execution_config.fee_rate,
            execution_config.slippage_rate,
            execution_config.safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            execution_config.bars_per_year_exec,
            close_on_end,
            direction_code,
            scores.total_return_pct,
            scores.max_drawdown_pct,
            scores.return_over_max_drawdown,
            scores.profit_factor,
            scores.trade_count,
            scores.sharpe_trades,
            scores.win_rate_pct,
            scores.avg_trade_ret_pct,
            scores.avg_trade_exec_bars,
            scores.exposure_pct,
        )
        return scores

    raise BacktestNoRiskExactScoringRejected(
        f"Unsupported no-risk exact backend={backend_id!r}"
    )


def run_fast_vs_reference_self_check_two(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    execution_config: BacktestNoRiskExecutionConfig,
    execution_prices: BacktestNoRiskExecutionPrices,
    check_n: int = 2,
    ret_tol: float = 1e-4,
) -> BacktestNoRiskSelfCheckResult:
    indicator_ids = tuple(prepared_result.indicator_ids)
    n_check = min(
        max(0, int(check_n)),
        _selected_size(
            selected_rows_by_indicator=selected_rows_by_indicator,
            indicator_ids=indicator_ids,
        ),
    )
    if n_check <= 0:
        return BacktestNoRiskSelfCheckResult(
            checked=0,
            passed=True,
            exact_backend=combo_planning_result.backend.backend_id,
            direction_mode=execution_config.direction_mode,
            return_tolerance=ret_tol,
            max_abs_exact_backend_ret_diff=0.0,
            trade_count_equal=True,
        )

    subset = {
        indicator_id: np.ascontiguousarray(
            np.asarray(selected_rows_by_indicator[indicator_id][:n_check], dtype=np.int32)
        )
        for indicator_id in indicator_ids
    }
    fast_scores = evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=subset,
        prepared_result=prepared_result,
        combo_planning_result=combo_planning_result,
        execution_config=execution_config,
        execution_prices=execution_prices,
    )
    reference_total_return_pct = np.empty(n_check, dtype=np.float64)
    reference_trade_count = np.empty(n_check, dtype=np.int32)
    for row_idx in range(n_check):
        local_indices = tuple(int(subset[indicator_id][row_idx]) for indicator_id in indicator_ids)
        metrics = evaluate_no_risk_reference_rows_slow(
            indicator_ids=indicator_ids,
            prepared_result=prepared_result,
            local_indices=local_indices,
            execution_config=execution_config,
            execution_prices=execution_prices,
        )
        reference_total_return_pct[row_idx] = metrics.total_return_pct
        reference_trade_count[row_idx] = metrics.trade_count

    trade_count_equal = bool(np.array_equal(reference_trade_count, fast_scores.trade_count))
    if not trade_count_equal:
        raise AssertionError(
            f"Exact backend {combo_planning_result.backend.backend_id!r} trade counts "
            "differ from generic slow reference"
        )
    max_abs_exact_backend_ret_diff = float(
        np.max(np.abs(reference_total_return_pct - fast_scores.total_return_pct))
    )
    if max_abs_exact_backend_ret_diff > ret_tol:
        raise AssertionError(
            f"Exact backend {combo_planning_result.backend.backend_id!r} total return "
            f"differs from generic slow reference by {max_abs_exact_backend_ret_diff}, "
            f"tolerance {ret_tol}"
        )
    return BacktestNoRiskSelfCheckResult(
        checked=n_check,
        passed=True,
        exact_backend=combo_planning_result.backend.backend_id,
        direction_mode=execution_config.direction_mode,
        return_tolerance=ret_tol,
        max_abs_exact_backend_ret_diff=max_abs_exact_backend_ret_diff,
        trade_count_equal=True,
    )


def build_trade_list_for_indicator_rows_slow(
    *,
    indicator_ids: tuple[str, ...],
    prepared_result: BacktestPreparePoolsResult,
    local_indices: tuple[int, ...],
    execution_config: BacktestNoRiskExecutionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pools_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    rows = [
        pools_by_id[indicator_id].trade_T[local_indices[pos]]
        for pos, indicator_id in enumerate(indicator_ids)
    ]
    n_sig = int(rows[0].shape[0])
    entry_exec: list[int] = []
    directions: list[int] = []
    sig_exit_exec: list[int] = []
    current_dir = 0
    current_entry = 0
    signal_entry_exec_idx = prepared_result.execution_mapping.signal_entry_exec_idx_15m
    t_exec_limit = int(prepared_result.execution_mapping.t_exec_limit_1m)

    for signal_idx in range(n_sig):
        raw_dir = int(rows[0][signal_idx])
        if raw_dir != 0:
            for row in rows[1:]:
                if int(row[signal_idx]) != raw_dir:
                    raw_dir = 0
                    break
        dirn = _apply_direction_mode_py(raw_dir, execution_config.direction_mode)
        if dirn == 0 and not (
            execution_config.direction_mode == DIRECTION_MODE_LONG_ONLY
            and current_dir != 0
        ):
            continue
        entry_idx = int(signal_entry_exec_idx[signal_idx])
        if entry_idx >= t_exec_limit:
            break
        if dirn == 0:
            entry_exec.append(current_entry)
            directions.append(current_dir)
            sig_exit_exec.append(entry_idx)
            current_dir = 0
            current_entry = 0
            continue
        if current_dir == 0:
            current_dir = dirn
            current_entry = entry_idx
            continue
        if dirn == current_dir:
            continue
        entry_exec.append(current_entry)
        directions.append(current_dir)
        sig_exit_exec.append(entry_idx)
        current_dir = dirn
        current_entry = entry_idx

    if current_dir != 0:
        entry_exec.append(current_entry)
        directions.append(current_dir)
        sig_exit_exec.append(t_exec_limit)

    return (
        np.asarray(entry_exec, dtype=np.int32),
        np.asarray(directions, dtype=np.int8),
        np.asarray(sig_exit_exec, dtype=np.int32),
    )


def evaluate_no_risk_reference_rows_slow(
    *,
    indicator_ids: tuple[str, ...],
    prepared_result: BacktestPreparePoolsResult,
    local_indices: tuple[int, ...],
    execution_config: BacktestNoRiskExecutionConfig,
    execution_prices: BacktestNoRiskExecutionPrices,
) -> BacktestNoRiskSummaryMetrics:
    entry_arr, dir_arr, exit_arr = build_trade_list_for_indicator_rows_slow(
        indicator_ids=indicator_ids,
        prepared_result=prepared_result,
        local_indices=local_indices,
        execution_config=execution_config,
    )
    if int(entry_arr.size) == 0:
        return BacktestNoRiskSummaryMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    metrics = score_trade_list_no_risk(
        entry_arr,
        dir_arr,
        exit_arr,
        np.int32(entry_arr.size),
        execution_prices.open_1m,
        execution_prices.close_1m,
        np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
        execution_config.initial_cash_quote,
        execution_config.fixed_quote,
        execution_config.fee_rate,
        execution_config.slippage_rate,
        execution_config.safe_profit_percent,
        np.int8(1 if execution_config.use_fixed_quote else 0),
        np.int8(1 if execution_config.use_profit_lock else 0),
        execution_config.bars_per_year_exec,
        np.int8(1 if execution_config.close_on_end else 0),
    )
    return BacktestNoRiskSummaryMetrics(
        total_return_pct=float(metrics[0]),
        max_drawdown_pct=float(metrics[1]),
        return_over_max_drawdown=float(metrics[2]),
        profit_factor=float(metrics[3]),
        trade_count=int(metrics[4]),
        sharpe_trades=float(metrics[5]),
        win_rate_pct=float(metrics[6]),
        avg_trade_ret_pct=float(metrics[7]),
        avg_trade_exec_bars=float(metrics[8]),
        exposure_pct=float(metrics[9]),
    )


def proxy_for_indicator_rows(
    *,
    eval_rows: tuple[np.ndarray, ...],
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
) -> tuple[int, float]:
    if len(eval_rows) == 0:
        raise ValueError("eval_rows must not be empty")
    returns = np.asarray(ret_15m, dtype=np.float32)
    confirms = 0
    proxy = np.float32(0.0)
    n_intervals = int(returns.shape[0])
    for interval_idx in range(n_intervals):
        dirn = int(eval_rows[0][interval_idx])
        if dirn == 0:
            continue
        for row in eval_rows[1:]:
            if int(row[interval_idx]) != dirn:
                dirn = 0
                break
        if dirn == 1:
            confirms += 1
            proxy += returns[interval_idx]
        elif dirn == -1:
            confirms += 1
            proxy -= returns[interval_idx]
    if confirms < int(min_confirm):
        return confirms, float(_NEG_INF)
    proxy -= fee_penalty_per_confirm * np.float32(confirms)
    return confirms, float(proxy)


@nb.njit(parallel=True, cache=True)
def proxy_for_indicator_rows_batch(
    local_rows_by_candidate: np.ndarray,
    eval_stack: np.ndarray,
    ret_15m: np.ndarray,
    min_confirm: np.int32,
    fee_penalty_per_confirm: np.float32,
    out_confirm: np.ndarray,
    out_proxy: np.ndarray,
) -> None:
    n_candidates = local_rows_by_candidate.shape[0]
    arity = local_rows_by_candidate.shape[1]
    n_intervals = ret_15m.shape[0]
    for candidate_pos in nb.prange(n_candidates):
        confirms = np.int32(0)
        proxy = np.float32(0.0)
        for interval_idx in range(n_intervals):
            direction = eval_stack[0, local_rows_by_candidate[candidate_pos, 0], interval_idx]
            if direction == 0:
                continue
            for indicator_pos in range(1, arity):
                local_row = local_rows_by_candidate[candidate_pos, indicator_pos]
                if eval_stack[indicator_pos, local_row, interval_idx] != direction:
                    direction = np.int8(0)
                    break
            if direction == 1:
                confirms += 1
                proxy += ret_15m[interval_idx]
            elif direction == -1:
                confirms += 1
                proxy -= ret_15m[interval_idx]
        out_confirm[candidate_pos] = confirms
        if confirms < min_confirm:
            out_proxy[candidate_pos] = _NEG_INF
        else:
            out_proxy[candidate_pos] = proxy - (fee_penalty_per_confirm * np.float32(confirms))


def build_persisted_top_n_summary_rows(
    *,
    job_id: UUID,
    top_rows: Sequence[BacktestNoRiskTopRow],
    updated_at: datetime,
) -> tuple[BacktestJobTopVariant, ...]:
    rows: list[BacktestJobTopVariant] = []
    for row in top_rows:
        payload = row.as_mapping()
        payload["storage_variant_key"] = row.variant_hash
        rows.append(
            BacktestJobTopVariant(
                job_id=job_id,
                rank=row.rank,
                variant_key=row.variant_hash,
                indicator_variant_key=row.indicator_variant_hash,
                variant_index=row.variant_index,
                total_return_pct=row.summary_metrics.total_return_pct,
                payload_json=payload,
                updated_at=updated_at,
                summary_metrics_json=row.summary_metrics.as_mapping(),
                best_tp_pct=None,
                best_sl_pct=None,
                report_table_md=None,
                trades_json=None,
            )
        )
    return tuple(rows)


@nb.njit(inline="always", cache=True)
def _consensus_dir2(left_value: np.int8, right_value: np.int8) -> np.int8:
    if left_value == 1 and right_value == 1:
        return np.int8(1)
    if left_value == -1 and right_value == -1:
        return np.int8(-1)
    return np.int8(0)


@nb.njit(inline="always", cache=True)
def apply_direction_mode(raw_dir: np.int8 | int, direction_mode: np.int8 | int) -> np.int8:
    if direction_mode == 1:
        if raw_dir == 1:
            return np.int8(1)
        return np.int8(0)
    return np.int8(raw_dir)


@nb.njit(inline="always", cache=True)
def trade_sharpe_kernel(
    trade_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: np.int32,
) -> float:
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


@nb.njit(inline="always", cache=True)
def _finalize_no_risk_metrics(
    equity: float,
    init_cash_quote: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    bars_per_year_exec: float,
    t_exec: np.int32,
) -> tuple[float, float, float, float, np.int32, float, float, float, float, float]:
    if closed_trade_count <= 0:
        return (0.0, 0.0, 0.0, 0.0, np.int32(0), 0.0, 0.0, 0.0, 0.0, 0.0)
    total_return_pct = ((equity / init_cash_quote) - 1.0) * 100.0
    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = np.inf
    else:
        return_over_max_drawdown = 0.0

    if closed_trade_count > 0:
        win_rate_pct = (float(win_count) / float(closed_trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(closed_trade_count)
        avg_trade_exec_bars = total_trade_exec_bars / float(closed_trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0

    exposure_pct = (exposure_bars / float(t_exec)) * 100.0 if t_exec > 0 else 0.0
    sharpe_trades = trade_sharpe_kernel(
        closed_trade_count,
        sum_trade_return,
        sum_trade_return_squared,
        bars_per_year_exec,
        t_exec,
    )
    return (
        total_return_pct,
        max_drawdown_pct,
        return_over_max_drawdown,
        profit_factor,
        closed_trade_count,
        sharpe_trades,
        win_rate_pct,
        avg_trade_ret_pct,
        avg_trade_exec_bars,
        exposure_pct,
    )


@nb.njit(inline="always", cache=True)
def apply_no_risk_trade_to_state(
    entry_idx: np.int32,
    trade_direction: np.int8,
    exit_exec_idx: np.int32,
    exit_price_raw: float,
    exec_open_1m: np.ndarray,
    available_quote: float,
    safe_quote: float,
    equity: float,
    peak_equity: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
):
    if available_quote <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )
    quote_amount = available_quote
    if use_fixed_quote == 1 and fixed_quote < quote_amount:
        quote_amount = fixed_quote
    if quote_amount <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )

    entry_price_raw = float(exec_open_1m[entry_idx])
    if trade_direction == 1:
        entry_fill_price = entry_price_raw * (1.0 + slippage_rate)
        exit_fill_price = exit_price_raw * (1.0 - slippage_rate)
    else:
        entry_fill_price = entry_price_raw * (1.0 - slippage_rate)
        exit_fill_price = exit_price_raw * (1.0 + slippage_rate)

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

    if use_profit_lock == 1 and net_pnl_quote > 0.0:
        locked_profit_quote = net_pnl_quote * (safe_profit_percent / 100.0)
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
    bars_held = float(exit_exec_idx - entry_idx)
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

    return (
        available_quote,
        safe_quote,
        equity,
        peak_equity,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
    )


@nb.njit(cache=True)
def score_trade_list_no_risk(
    entry_exec_idx: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_exec_idx: np.ndarray,
    n_trades: np.int32,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
) -> tuple[float, float, float, float, np.int32, float, float, float, float, float]:
    available_quote = init_cash_quote
    safe_quote = 0.0
    equity = init_cash_quote
    peak_equity = equity
    max_drawdown_pct = 0.0
    gross_profit_quote = 0.0
    gross_loss_quote = 0.0
    closed_trade_count = np.int32(0)
    win_count = np.int32(0)
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    total_trade_return_pct = 0.0
    total_trade_exec_bars = 0.0
    exposure_bars = 0.0

    for trade_index in range(n_trades):
        entry_idx = np.int32(entry_exec_idx[trade_index])
        if entry_idx >= t_exec:
            continue
        exit_idx = np.int32(sig_exit_exec_idx[trade_index])
        if exit_idx < t_exec:
            exit_exec_idx = exit_idx
            exit_price_raw = float(exec_open_1m[exit_exec_idx])
        elif close_on_end == 1 and t_exec > 0:
            exit_exec_idx = np.int32(t_exec - 1)
            exit_price_raw = float(exec_close_1m[exit_exec_idx])
        else:
            continue
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        ) = apply_no_risk_trade_to_state(
            entry_idx,
            np.int8(dir_arr[trade_index]),
            exit_exec_idx,
            exit_price_raw,
            exec_open_1m,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
        )

    return _finalize_no_risk_metrics(
        equity,
        init_cash_quote,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
        bars_per_year_exec,
        t_exec,
    )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_streaming_two(
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_trade_t: np.ndarray,
    right_trade_t: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    combo_count = combo_left_idx.shape[0]
    n_sig = left_trade_t.shape[1]
    for combo_pos in nb.prange(combo_count):
        left_row = combo_left_idx[combo_pos]
        right_row = combo_right_idx[combo_pos]
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            current_dir,
            current_entry,
        ) = _initial_no_risk_state(init_cash_quote)

        for signal_idx in range(n_sig):
            raw_dir = _consensus_dir2(
                left_trade_t[left_row, signal_idx],
                right_trade_t[right_row, signal_idx],
            )
            dirn = apply_direction_mode(raw_dir, direction_mode)
            if dirn == 0 and not (direction_mode == 1 and current_dir != 0):
                continue
            entry_exec = sig_entry_exec_idx[signal_idx]
            if entry_exec >= t_exec:
                break
            if dirn == 0:
                (
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                ) = apply_no_risk_trade_to_state(
                    current_entry,
                    current_dir,
                    np.int32(entry_exec),
                    float(exec_open_1m[entry_exec]),
                    exec_open_1m,
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                    fixed_quote,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    use_fixed_quote,
                    use_profit_lock,
                )
                current_dir = np.int8(0)
                current_entry = np.int32(0)
            elif current_dir == 0:
                current_dir = dirn
                current_entry = np.int32(entry_exec)
            elif dirn != current_dir:
                (
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                ) = apply_no_risk_trade_to_state(
                    current_entry,
                    current_dir,
                    np.int32(entry_exec),
                    float(exec_open_1m[entry_exec]),
                    exec_open_1m,
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                    fixed_quote,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    use_fixed_quote,
                    use_profit_lock,
                )
                current_dir = dirn
                current_entry = np.int32(entry_exec)

        _close_and_write_no_risk_metrics(
            current_dir,
            current_entry,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            exec_open_1m,
            exec_close_1m,
            t_exec,
            init_cash_quote,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            bars_per_year_exec,
            close_on_end,
            combo_pos,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_event_segments_two(
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_segment_starts: np.ndarray,
    left_segment_ends: np.ndarray,
    left_segment_values: np.ndarray,
    left_segment_counts: np.ndarray,
    right_segment_starts: np.ndarray,
    right_segment_ends: np.ndarray,
    right_segment_values: np.ndarray,
    right_segment_counts: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    combo_count = combo_left_idx.shape[0]
    for combo_pos in nb.prange(combo_count):
        left_row = combo_left_idx[combo_pos]
        right_row = combo_right_idx[combo_pos]
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            current_dir,
            current_entry,
        ) = _initial_no_risk_state(init_cash_quote)
        left_segment_idx = 0
        right_segment_idx = 0

        while (
            left_segment_idx < left_segment_counts[left_row]
            and right_segment_idx < right_segment_counts[right_row]
        ):
            left_start = left_segment_starts[left_row, left_segment_idx]
            left_end = left_segment_ends[left_row, left_segment_idx]
            right_start = right_segment_starts[right_row, right_segment_idx]
            right_end = right_segment_ends[right_row, right_segment_idx]
            segment_start = left_start if left_start >= right_start else right_start
            segment_end = left_end if left_end <= right_end else right_end

            if segment_start < segment_end:
                raw_dir = _consensus_dir2(
                    left_segment_values[left_row, left_segment_idx],
                    right_segment_values[right_row, right_segment_idx],
                )
                dirn = apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
                    if entry_exec >= t_exec:
                        break
                    (
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                        current_dir,
                        current_entry,
                    ) = _apply_direction_transition(
                        dirn,
                        np.int32(entry_exec),
                        current_dir,
                        current_entry,
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                        exec_open_1m,
                        fixed_quote,
                        fee_rate,
                        slippage_rate,
                        safe_profit_percent,
                        use_fixed_quote,
                        use_profit_lock,
                    )

            if left_end == segment_end:
                left_segment_idx += 1
            if right_end == segment_end:
                right_segment_idx += 1

        _close_and_write_no_risk_metrics(
            current_dir,
            current_entry,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            exec_open_1m,
            exec_close_1m,
            t_exec,
            init_cash_quote,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            bars_per_year_exec,
            close_on_end,
            combo_pos,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(parallel=True, cache=True, fastmath=True)
def evaluate_no_risk_event_segments_n(
    combo_idx_by_indicator: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_values: np.ndarray,
    segment_counts: np.ndarray,
    segment_pos_workspace: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    arity = combo_idx_by_indicator.shape[0]
    combo_count = combo_idx_by_indicator.shape[1]
    for combo_pos in nb.prange(combo_count):
        for indicator_pos in range(arity):
            segment_pos_workspace[combo_pos, indicator_pos] = np.int32(0)
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            current_dir,
            current_entry,
        ) = _initial_no_risk_state(init_cash_quote)

        while True:
            active = True
            segment_start = np.int32(0)
            segment_end = np.int32(2147483647)
            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                if segment_idx >= segment_counts[indicator_pos, row_idx]:
                    active = False
                    break
                start_value = segment_starts[indicator_pos, row_idx, segment_idx]
                end_value = segment_ends[indicator_pos, row_idx, segment_idx]
                if start_value > segment_start:
                    segment_start = start_value
                if end_value < segment_end:
                    segment_end = end_value
            if not active:
                break

            if segment_start < segment_end:
                first_row_idx = combo_idx_by_indicator[0, combo_pos]
                first_segment_idx = segment_pos_workspace[combo_pos, 0]
                raw_dir = segment_values[0, first_row_idx, first_segment_idx]
                if raw_dir != 0:
                    for indicator_pos in range(1, arity):
                        row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                        segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                        if segment_values[indicator_pos, row_idx, segment_idx] != raw_dir:
                            raw_dir = np.int8(0)
                            break
                dirn = apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
                    if entry_exec >= t_exec:
                        break
                    (
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                        current_dir,
                        current_entry,
                    ) = _apply_direction_transition(
                        dirn,
                        np.int32(entry_exec),
                        current_dir,
                        current_entry,
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                        exec_open_1m,
                        fixed_quote,
                        fee_rate,
                        slippage_rate,
                        safe_profit_percent,
                        use_fixed_quote,
                        use_profit_lock,
                    )

            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, combo_pos]
                segment_idx = segment_pos_workspace[combo_pos, indicator_pos]
                if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                    segment_pos_workspace[combo_pos, indicator_pos] = np.int32(segment_idx + 1)

        _close_and_write_no_risk_metrics(
            current_dir,
            current_entry,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            exec_open_1m,
            exec_close_1m,
            t_exec,
            init_cash_quote,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
            bars_per_year_exec,
            close_on_end,
            combo_pos,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(inline="always", cache=True)
def _initial_no_risk_state(
    init_cash_quote: float,
):
    return (
        init_cash_quote,
        0.0,
        init_cash_quote,
        init_cash_quote,
        0.0,
        0.0,
        0.0,
        np.int32(0),
        np.int32(0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.int8(0),
        np.int32(0),
    )


@nb.njit(inline="always", cache=True)
def _apply_direction_transition(
    dirn: np.int8,
    entry_exec: np.int32,
    current_dir: np.int8,
    current_entry: np.int32,
    available_quote: float,
    safe_quote: float,
    equity: float,
    peak_equity: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    exec_open_1m: np.ndarray,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
):
    if dirn == 0:
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        ) = apply_no_risk_trade_to_state(
            current_entry,
            current_dir,
            entry_exec,
            float(exec_open_1m[entry_exec]),
            exec_open_1m,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
        )
        current_dir = np.int8(0)
        current_entry = np.int32(0)
    elif current_dir == 0:
        current_dir = dirn
        current_entry = entry_exec
    elif dirn != current_dir:
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        ) = apply_no_risk_trade_to_state(
            current_entry,
            current_dir,
            entry_exec,
            float(exec_open_1m[entry_exec]),
            exec_open_1m,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
        )
        current_dir = dirn
        current_entry = entry_exec
    return (
        available_quote,
        safe_quote,
        equity,
        peak_equity,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
        current_dir,
        current_entry,
    )


@nb.njit(cache=True)
def _close_and_write_no_risk_metrics(
    current_dir: np.int8,
    current_entry: np.int32,
    available_quote: float,
    safe_quote: float,
    equity: float,
    peak_equity: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    fixed_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_fixed_quote: np.int8,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    combo_pos: int,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    if current_dir != 0 and close_on_end == 1 and t_exec > 0:
        exit_exec_idx = np.int32(t_exec - 1)
        (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        ) = apply_no_risk_trade_to_state(
            current_entry,
            current_dir,
            exit_exec_idx,
            float(exec_close_1m[exit_exec_idx]),
            exec_open_1m,
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            fixed_quote,
            fee_rate,
            slippage_rate,
            safe_profit_percent,
            use_fixed_quote,
            use_profit_lock,
        )

    metrics = _finalize_no_risk_metrics(
        equity,
        init_cash_quote,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
        bars_per_year_exec,
        t_exec,
    )
    out_total_return_pct[combo_pos] = metrics[0]
    out_max_drawdown_pct[combo_pos] = metrics[1]
    out_return_over_max_drawdown[combo_pos] = metrics[2]
    out_profit_factor[combo_pos] = metrics[3]
    out_trade_count[combo_pos] = metrics[4]
    out_sharpe_trades[combo_pos] = metrics[5]
    out_win_rate_pct[combo_pos] = metrics[6]
    out_avg_trade_ret_pct[combo_pos] = metrics[7]
    out_avg_trade_exec_bars[combo_pos] = metrics[8]
    out_exposure_pct[combo_pos] = metrics[9]


def _top_candidates_from_chunk(
    *,
    prepared_result: BacktestPreparePoolsResult,
    filter_result: BacktestProxyFilterResult,
    chunk_scores: BacktestNoRiskChunkScores,
    combo_global_start: int,
    ranking_metric: str,
    ranking_direction: str,
    top_n: int,
) -> list[_TopCandidate]:
    indicator_ids = tuple(prepared_result.indicator_ids)
    pools_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    selected_score_indexes = _selected_top_score_indexes_from_chunk(
        prepared_result=prepared_result,
        filter_result=filter_result,
        chunk_scores=chunk_scores,
        ranking_metric=ranking_metric,
        ranking_direction=ranking_direction,
        top_n=top_n,
    )
    candidates: list[_TopCandidate] = []
    for score_index in selected_score_indexes:
        local_rows = tuple(
            int(filter_result.selected_rows_by_indicator[indicator_id][score_index])
            for indicator_id in indicator_ids
        )
        row_ids = tuple(
            int(pools_by_id[indicator_id].row_ids[local_rows[pos]])
            for pos, indicator_id in enumerate(indicator_ids)
        )
        confirm_count = (
            None if filter_result.confirm is None else int(filter_result.confirm[score_index])
        )
        proxy_score = (
            None if filter_result.proxy is None else float(filter_result.proxy[score_index])
        )
        candidates.append(
            _TopCandidate(
                variant_index=combo_global_start + int(filter_result.selected_indexes[score_index]),
                local_rows=local_rows,
                row_ids=row_ids,
                metrics=chunk_scores.metrics_at(score_index),
                confirm_count=confirm_count,
                proxy_score=proxy_score,
            )
        )
    return candidates


def _selected_top_score_indexes_from_chunk(
    *,
    prepared_result: BacktestPreparePoolsResult,
    filter_result: BacktestProxyFilterResult,
    chunk_scores: BacktestNoRiskChunkScores,
    ranking_metric: str,
    ranking_direction: str,
    top_n: int,
) -> np.ndarray:
    size = chunk_scores.size
    if size <= 0:
        return np.empty(0, dtype=np.int32)

    metric_values = np.asarray(
        getattr(chunk_scores, ranking_metric),
        dtype=np.float64,
    )
    metric_key = np.where(np.isnan(metric_values), np.inf, metric_values)
    if ranking_direction == "desc":
        metric_key = -metric_key

    pools_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    row_id_keys: list[np.ndarray] = []
    for indicator_id in prepared_result.indicator_ids:
        local_rows = np.asarray(
            filter_result.selected_rows_by_indicator[indicator_id],
            dtype=np.int32,
        )
        row_id_keys.append(
            np.asarray(pools_by_id[indicator_id].row_ids[local_rows], dtype=np.int32)
        )
    variant_key = np.asarray(filter_result.selected_indexes, dtype=np.int64)

    order = np.lexsort(
        tuple([variant_key, *reversed(row_id_keys), metric_key])
    )
    if size > top_n:
        order = order[:top_n]
    return np.ascontiguousarray(order.astype(np.int32))


def _candidate_sort_key(
    *,
    candidate: _TopCandidate,
    ranking_metric: str,
    ranking_direction: str,
) -> tuple[float, tuple[int, ...], int]:
    score = candidate.metrics.metric_value(ranking_metric)
    if math.isnan(score):
        metric_key = math.inf
    else:
        metric_key = -score if ranking_direction == "desc" else score
    return metric_key, candidate.row_ids, candidate.variant_index


def _build_top_rows_with_proxy_fill(
    *,
    candidates: Sequence[_TopCandidate],
    filled_proxy_by_candidate_pos: Mapping[int, tuple[int, float]],
    prepared_result: BacktestPreparePoolsResult,
    execution_config: BacktestNoRiskExecutionConfig,
    normalized_request: Mapping[str, Any],
    ranking_metric: str,
) -> tuple[BacktestNoRiskTopRow, ...]:
    rows: list[BacktestNoRiskTopRow] = []
    for candidate_pos, candidate in enumerate(candidates):
        rank = candidate_pos + 1
        confirm_count = candidate.confirm_count
        proxy_score = candidate.proxy_score
        if confirm_count is None or proxy_score is None:
            confirm_count, proxy_score = filled_proxy_by_candidate_pos[candidate_pos]

        variant_params = _variant_params(
            prepared_result=prepared_result,
            candidate=candidate,
            execution_config=execution_config,
            normalized_request=normalized_request,
        )
        indicator_variant_hash = _canonical_sha256({"indicators": variant_params["indicators"]})
        variant_hash = _canonical_sha256(variant_params)
        public_variant_key = _public_variant_key(
            variant_params=variant_params,
            execution_config=execution_config,
        )
        rows.append(
            BacktestNoRiskTopRow(
                rank=rank,
                variant_index=candidate.variant_index,
                public_variant_key=public_variant_key,
                variant_hash=variant_hash,
                indicator_variant_hash=indicator_variant_hash,
                row_ids_by_indicator={
                    indicator_id: candidate.row_ids[pos]
                    for pos, indicator_id in enumerate(prepared_result.indicator_ids)
                },
                local_rows_by_indicator={
                    indicator_id: candidate.local_rows[pos]
                    for pos, indicator_id in enumerate(prepared_result.indicator_ids)
                },
                summary_metrics=candidate.metrics,
                ranking_metric=ranking_metric,
                ranking_score=candidate.metrics.metric_value(ranking_metric),
                confirm_count=int(confirm_count),
                proxy_score=float(proxy_score),
                variant_params=variant_params,
            )
        )
    return tuple(rows)


def _proxy_fill_missing_candidates(
    *,
    candidates: Sequence[_TopCandidate],
    prepared_result: BacktestPreparePoolsResult,
    proxy_context: BacktestProxyContext,
) -> tuple[dict[int, tuple[int, float]], int]:
    missing_positions = [
        candidate_pos
        for candidate_pos, candidate in enumerate(candidates)
        if candidate.confirm_count is None or candidate.proxy_score is None
    ]
    if len(missing_positions) == 0:
        return {}, 0

    local_rows = np.ascontiguousarray(
        np.asarray(
            [candidates[candidate_pos].local_rows for candidate_pos in missing_positions],
            dtype=np.int32,
        )
    )
    eval_stack = _proxy_fill_eval_stack(prepared_result=prepared_result)
    confirm_out = np.empty(len(missing_positions), dtype=np.int32)
    proxy_out = np.empty(len(missing_positions), dtype=np.float32)
    proxy_for_indicator_rows_batch(
        local_rows,
        eval_stack,
        prepared_result.signal_returns_15m,
        np.int32(proxy_context.combo_min_confirm),
        np.float32(proxy_context.fee_penalty_per_confirm),
        confirm_out,
        proxy_out,
    )
    return (
        {
            candidate_pos: (int(confirm_out[out_pos]), float(proxy_out[out_pos]))
            for out_pos, candidate_pos in enumerate(missing_positions)
        },
        len(missing_positions),
    )


def _proxy_fill_eval_stack(*, prepared_result: BacktestPreparePoolsResult) -> np.ndarray:
    pools = tuple(prepared_result.indicator_pools)
    arity = len(pools)
    max_rows = max(int(pool.eval_T.shape[0]) for pool in pools)
    eval_t_length = int(prepared_result.eval_T_length)
    eval_stack = np.zeros((arity, max_rows, eval_t_length), dtype=np.int8)
    for indicator_pos, pool in enumerate(pools):
        row_count = int(pool.eval_T.shape[0])
        eval_stack[indicator_pos, :row_count, :] = pool.eval_T[:, :eval_t_length]
    return np.ascontiguousarray(eval_stack)


def _variant_params(
    *,
    prepared_result: BacktestPreparePoolsResult,
    candidate: _TopCandidate,
    execution_config: BacktestNoRiskExecutionConfig,
    normalized_request: Mapping[str, Any],
) -> dict[str, Any]:
    pools_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    indicators: list[dict[str, Any]] = []
    for pos, indicator_id in enumerate(prepared_result.indicator_ids):
        pool = pools_by_id[indicator_id]
        local_row = candidate.local_rows[pos]
        metadata = pool.metadata[local_row]
        indicators.append(
            {
                "indicator_id": indicator_id,
                "local_row": local_row,
                "row_id": int(pool.row_ids[local_row]),
                "source": metadata.source,
                "window": metadata.window,
            }
        )
    return {
        "schema_version": 1,
        "risk": {"mode": "none"},
        "execution": execution_config.as_mapping(),
        "ranking": dict(normalized_request.get("ranking", {})),
        "indicators": indicators,
    }


def _public_variant_key(
    *,
    variant_params: Mapping[str, Any],
    execution_config: BacktestNoRiskExecutionConfig,
) -> str:
    indicator_parts = []
    for indicator in variant_params["indicators"]:
        source = "none" if indicator["source"] is None else str(indicator["source"])
        indicator_parts.append(
            f"{indicator['indicator_id']}:{source}:w{indicator['window']}:r{indicator['row_id']}"
        )
    return (
        "no-risk/v1|"
        f"{execution_config.direction_mode}|"
        f"{execution_config.sizing_mode}|"
        + "|".join(indicator_parts)
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _zero_stage_timings() -> dict[str, float]:
    return {
        SELF_CHECK_STAGE_NAME: 0.0,
        EXACT_SCORING_STAGE_NAME: 0.0,
        HEAP_UPDATE_STAGE_NAME: 0.0,
        TOP_RESULT_PROXY_FILL_STAGE_NAME: 0.0,
        TOP_RESULT_ASSEMBLY_STAGE_NAME: 0.0,
        TOTAL_WITHOUT_WARMUP_STAGE_NAME: 0.0,
    }


def _empty_chunk_scores(size: int) -> BacktestNoRiskChunkScores:
    return BacktestNoRiskChunkScores(
        total_return_pct=np.zeros(size, dtype=np.float64),
        max_drawdown_pct=np.zeros(size, dtype=np.float64),
        return_over_max_drawdown=np.zeros(size, dtype=np.float64),
        profit_factor=np.zeros(size, dtype=np.float64),
        trade_count=np.zeros(size, dtype=np.int32),
        sharpe_trades=np.zeros(size, dtype=np.float64),
        win_rate_pct=np.zeros(size, dtype=np.float64),
        avg_trade_ret_pct=np.zeros(size, dtype=np.float64),
        avg_trade_exec_bars=np.zeros(size, dtype=np.float64),
        exposure_pct=np.zeros(size, dtype=np.float64),
    )


def _ranking_config(
    *,
    normalized_request: Mapping[str, Any],
    fallback_config: BacktestNoRiskExactScoringConfig,
) -> tuple[str, str, int]:
    ranking = normalized_request.get("ranking")
    metric = fallback_config.ranking_metric
    direction = fallback_config.ranking_direction
    if isinstance(ranking, Mapping):
        metric = str(ranking.get("primary_metric", metric)).strip()
        direction = str(ranking.get("direction", direction)).strip().lower()
    top_n = int(normalized_request.get("top_n", fallback_config.top_n))
    if metric not in NO_RISK_SUMMARY_METRIC_NAMES:
        raise BacktestNoRiskExactScoringRejected(f"Unsupported ranking metric={metric!r}")
    if direction not in ("asc", "desc"):
        raise BacktestNoRiskExactScoringRejected(f"Unsupported ranking direction={direction!r}")
    if top_n <= 0:
        raise BacktestNoRiskExactScoringRejected("top_n must be > 0")
    return metric, direction, top_n


def _selected_size(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    indicator_ids: Sequence[str],
) -> int:
    sizes = {
        int(np.asarray(selected_rows_by_indicator[indicator_id]).shape[0])
        for indicator_id in indicator_ids
    }
    if len(sizes) != 1:
        raise BacktestNoRiskExactScoringRejected("selected row arrays must share length")
    return sizes.pop()


def _ordered_pools(
    *,
    prepared_result: BacktestPreparePoolsResult,
) -> tuple[PreparedIndicatorPool, ...]:
    pools_by_id = {pool.indicator_id: pool for pool in prepared_result.indicator_pools}
    return tuple(pools_by_id[indicator_id] for indicator_id in prepared_result.indicator_ids)


def _validate_execution_prices(
    *,
    prepared_result: BacktestPreparePoolsResult,
    execution_prices: BacktestNoRiskExecutionPrices,
) -> None:
    t_exec = int(prepared_result.execution_mapping.t_exec_limit_1m)
    if t_exec <= 0:
        raise BacktestNoRiskExactScoringRejected("t_exec_limit_1m must be > 0")
    if t_exec > int(execution_prices.open_1m.shape[0]):
        raise BacktestNoRiskExactScoringRejected(
            "t_exec_limit_1m exceeds execution price array length"
        )


def _direction_mode_code(direction_mode: str) -> np.int8:
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        return _DIRECTION_MODE_LONG_ONLY_CODE
    if direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        return _DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE
    raise BacktestNoRiskExactScoringRejected(f"Unsupported direction_mode={direction_mode!r}")


def _apply_direction_mode_py(raw_dir: int, direction_mode: str) -> int:
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        return 1 if raw_dir == 1 else 0
    if direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        return int(raw_dir)
    raise BacktestNoRiskExactScoringRejected(f"Unsupported direction_mode={direction_mode!r}")


def _required_mapping(payload: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BacktestNoRiskExactScoringRejected(f"{path} must be a mapping")
    return value


def _positive_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BacktestNoRiskExactScoringRejected(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise BacktestNoRiskExactScoringRejected(f"{path} must be > 0")
    return numeric


def _non_negative_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BacktestNoRiskExactScoringRejected(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise BacktestNoRiskExactScoringRejected(f"{path} must be >= 0")
    return numeric


__all__ = [
    "EXACT_SCORING_STAGE_NAME",
    "HEAP_UPDATE_STAGE_NAME",
    "NUMBA_WARMUP_STAGE_NAME",
    "PERSIST_TOP_N_IO_STAGE_NAME",
    "SAMPLE_WARMUP_STAGE_NAME",
    "SELF_CHECK_STAGE_NAME",
    "SERVICE_WARMUP_STAGE_NAME",
    "TOP_RESULT_ASSEMBLY_STAGE_NAME",
    "TOP_RESULT_PROXY_FILL_STAGE_NAME",
    "TOTAL_WITHOUT_WARMUP_STAGE_NAME",
    "BacktestNoRiskExactScoringRejected",
    "BacktestNoRiskExactScoringService",
    "apply_direction_mode",
    "apply_no_risk_trade_to_state",
    "build_persisted_top_n_summary_rows",
    "build_trade_list_for_indicator_rows_slow",
    "evaluate_no_risk_event_segments_n",
    "evaluate_no_risk_event_segments_two",
    "evaluate_no_risk_exact_chunk",
    "evaluate_no_risk_reference_rows_slow",
    "evaluate_no_risk_streaming_two",
    "no_risk_execution_config_from_normalized",
    "proxy_for_indicator_rows",
    "run_fast_vs_reference_self_check_two",
    "score_trade_list_no_risk",
    "trade_sharpe_kernel",
]
