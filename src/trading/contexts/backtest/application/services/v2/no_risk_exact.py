from __future__ import annotations

import heapq
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, NamedTuple

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningResult,
    BacktestNoRiskExactConfig,
    BacktestNoRiskExactResult,
    BacktestNoRiskExactTelemetry,
    BacktestNoRiskExecutionContext,
    BacktestNoRiskMemoryCleanupEvidence,
    BacktestNoRiskSelfCheckSummary,
    BacktestNoRiskTopResult,
    BacktestPreparePoolsResult,
)
from trading.contexts.backtest.application.services.v2.combo_planning import (
    COMBO_CHUNK_SIZE,
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    NEG_INF,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestComboPlanningService,
    build_local_row_pools,
    iter_ordinal_combo_chunks,
    make_combo_idx_matrix,
)
from trading.contexts.backtest.application.services.v2.execution_sizing import (
    DIRECTION_MODE_LONG_ONLY,
    DIRECTION_MODE_LONG_SHORT_REVERSAL,
    execution_quote_amount,
    execution_settings_from_normalized,
)
from trading.contexts.backtest.application.services.v2.execution_sizing import (
    ExecutionSettings as _ExecutionSettings,
)
from trading.contexts.backtest.application.services.v2.numba_runtime import (
    current_backtest_numba_telemetry,
)

NO_RISK_EXACT_BOUNDARY_STAGE_NAME = "no_risk_exact_boundary"
NO_RISK_EXACT_SCORING_STAGE_NAME = "exact_scoring"
NO_RISK_HEAP_UPDATE_STAGE_NAME = "heap_update"
NO_RISK_TOP_RESULT_ASSEMBLY_STAGE_NAME = "top_result_assembly"
NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME = "top_result_proxy_fill"
NO_RISK_SELF_CHECK_STAGE_NAME = "self_check"
NO_RISK_COMBO_CHUNK_DECODE_STAGE_NAME = "combo_chunk_decode"
NO_RISK_PROXY_FILTER_STAGE_NAME = "proxy_filter"
NO_RISK_METRIC_BUFFER_ALLOCATION_STAGE_NAME = "metric_buffer_allocation"
NO_RISK_TELEMETRY_BUILD_STAGE_NAME = "telemetry_build"
NO_RISK_EXACT_BOUNDARY_STATUS = "boundary_ready"
NO_RISK_EXACT_SCORED_STATUS = "scored"
NO_RISK_SELF_CHECK_NOT_RUN_STATUS = "not_run"
NO_RISK_SELF_CHECK_PASSED_STATUS = "passed"
CANONICAL_EXECUTION_TIMEFRAME_V1 = "1m"
BARS_PER_YEAR_EXEC_1M = 365.0 * 24.0 * 60.0
NO_RISK_METRIC_NAMES = (
    "total_return_pct",
    "max_drawdown_pct",
    "return_over_max_drawdown",
    "profit_factor",
    "trade_count",
    "sharpe_trades",
    "win_rate_pct",
    "avg_trade_ret_pct",
    "avg_trade_exec_bars",
    "exposure_pct",
)


class BacktestNoRiskExactRejected(ValueError):
    """
    Deterministic internal rejection for unsupported no-risk exact boundary inputs.
    """


class BacktestNoRiskSelfCheckFailed(AssertionError):
    """
    Raised when fast exact scoring diverges from the bounded slow reference.
    """


@dataclass(frozen=True, slots=True)
class _MetricBuffers:
    total_return_pct: np.ndarray
    max_drawdown_pct: np.ndarray
    return_over_max_drawdown: np.ndarray
    profit_factor: np.ndarray
    trade_count: np.ndarray
    sharpe_trades: np.ndarray
    win_rate_pct: np.ndarray
    avg_trade_ret_pct: np.ndarray
    avg_trade_exec_bars: np.ndarray
    exposure_pct: np.ndarray

    @property
    def size(self) -> int:
        return int(self.total_return_pct.shape[0])


@dataclass(frozen=True, slots=True)
class _SelectedCandidateBatch:
    rows_by_indicator: Mapping[str, np.ndarray]
    confirm: np.ndarray | None
    proxy: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _RankingSpec:
    metric_name: str
    direction: str

    @property
    def is_default_total_return_desc(self) -> bool:
        return self.metric_name == "total_return_pct" and self.direction == "desc"


@dataclass(frozen=True, slots=True)
class _TopKContext:
    indicator_ids: tuple[str, ...]
    row_ids_by_pos: tuple[np.ndarray, ...]
    metadata_by_pos: tuple[tuple[Any, ...], ...]


class _NoRiskHeapEntry(NamedTuple):
    score: float
    original_rows: tuple[int, ...]
    local_indices: tuple[int, ...]
    metric_values: tuple[float, ...]
    metric_buffers: _MetricBuffers | None
    metric_index: int
    metadata_by_pos: tuple[Any, ...]
    confirm_count: int
    proxy_score: float
    proxy_pending: bool


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactScoringService:
    """
    Internal service for no-risk exact scoring and notebook-compatible top-K heap work.
    """

    config: BacktestNoRiskExactConfig = BacktestNoRiskExactConfig()

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        normalized_request: Mapping[str, Any],
    ) -> BacktestNoRiskExactResult:
        """
        Validate the no-risk boundary, run optional self-check, score candidates,
        and keep the canonical benchmark top-K heap.
        """

        boundary_start = time.perf_counter()
        risk_mode = _risk_mode_from_normalized(normalized_request)
        if risk_mode != "none":
            raise BacktestNoRiskExactRejected(
                f"no-risk exact boundary requires risk.mode='none'; got {risk_mode!r}"
            )

        backend = combo_planning_result.backend
        if backend.risk_mode != "none":
            raise BacktestNoRiskExactRejected(
                f"combo planning backend risk_mode must be 'none'; got {backend.risk_mode!r}"
            )
        arity = len(prepared_result.indicator_ids)
        if backend.arity != arity:
            raise BacktestNoRiskExactRejected(
                f"combo planning arity {backend.arity} does not match prepared arity {arity}"
            )
        _validate_backend_for_exact_scoring(backend_id=backend.backend_id, arity=arity)
        execution_settings = _execution_settings_from_normalized(
            normalized_request,
            expected_direction_mode=backend.direction_mode,
            config=self.config,
        )
        execution_open_1m, execution_close_1m = _execution_price_arrays_from_prepared(
            prepared_result
        )

        request_top_n = _request_top_n_from_normalized(
            normalized_request,
            default_request_top_n=self.config.default_request_top_n,
        )
        ranking = _ranking_from_normalized(normalized_request)
        top_k_context = _top_k_context_from_prepared(prepared_result)
        backend_logical_name = _backend_logical_name(backend_id=backend.backend_id, arity=arity)
        stage_timings = {
            NO_RISK_EXACT_BOUNDARY_STAGE_NAME: time.perf_counter() - boundary_start,
        }
        execution_context = _execution_context_from_prepared(prepared_result)

        self_check_summary = BacktestNoRiskSelfCheckSummary(
            enabled=self.config.run_self_check,
            status=NO_RISK_SELF_CHECK_NOT_RUN_STATUS,
            backend_logical_name=backend_logical_name,
            backend_implementation_id=backend.backend_id,
            direction_mode=backend.direction_mode,
        )

        top_results: tuple[BacktestNoRiskTopResult, ...] | None = None
        telemetry: BacktestNoRiskExactTelemetry | None = None
        cleanup_duration_s = 0.0
        selected_batches_iter: Iterator[_SelectedCandidateBatch] | None = None
        first_selected_batch: _SelectedCandidateBatch | None = None
        selected_batch: _SelectedCandidateBatch | None = None
        heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]] | None = None
        try:
            profile_stage_timings = stage_timings if _exact_profile_enabled() else None
            selected_batches_iter = _iter_selected_candidate_batches(
                prepared_result=prepared_result,
                combo_planning_result=combo_planning_result,
                profile_stage_timings=profile_stage_timings,
            )
            first_selected_batch = _next_selected_candidate_batch(selected_batches_iter)
            if self.config.run_self_check:
                check_start = time.perf_counter()
                self_check_summary = self._run_self_check(
                    selected_rows_by_indicator=None
                    if first_selected_batch is None
                    else first_selected_batch.rows_by_indicator,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    execution_settings=execution_settings,
                    execution_open_1m=execution_open_1m,
                    execution_close_1m=execution_close_1m,
                    backend_logical_name=backend_logical_name,
                )
                stage_timings[NO_RISK_SELF_CHECK_STAGE_NAME] = (
                    time.perf_counter() - check_start
                )

            stage_timings[NO_RISK_EXACT_SCORING_STAGE_NAME] = 0.0
            stage_timings[NO_RISK_HEAP_UPDATE_STAGE_NAME] = 0.0
            heap = []
            scored_count = 0
            sample_metrics: Mapping[str, float] | None = None
            if first_selected_batch is not None:
                scored, sample_metrics = self._score_selected_rows(
                    selected_batch=first_selected_batch,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    execution_settings=execution_settings,
                    execution_open_1m=execution_open_1m,
                    execution_close_1m=execution_close_1m,
                    stage_timings=stage_timings,
                    sample_metrics=sample_metrics,
                    heap=heap,
                    top_k_context=top_k_context,
                    top_k=request_top_n,
                    ranking=ranking,
                )
                scored_count += scored
            assert selected_batches_iter is not None
            for selected_batch in selected_batches_iter:
                scored, sample_metrics = self._score_selected_rows(
                    selected_batch=selected_batch,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    execution_settings=execution_settings,
                    execution_open_1m=execution_open_1m,
                    execution_close_1m=execution_close_1m,
                    stage_timings=stage_timings,
                    sample_metrics=sample_metrics,
                    heap=heap,
                    top_k_context=top_k_context,
                    top_k=request_top_n,
                    ranking=ranking,
                )
                scored_count += scored

            top_result_start = time.perf_counter()
            top_results = _top_results_from_heap(
                heap,
                prepared_result=prepared_result,
                combo_planning_result=combo_planning_result,
                top_k_context=top_k_context,
            )
            stage_timings[NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME] = (
                time.perf_counter() - top_result_start
            )
            telemetry_start = time.perf_counter()
            numba_telemetry = current_backtest_numba_telemetry()
            _add_timing(
                stage_timings=profile_stage_timings,
                key=NO_RISK_TELEMETRY_BUILD_STAGE_NAME,
                elapsed=time.perf_counter() - telemetry_start,
            )
            telemetry = BacktestNoRiskExactTelemetry(
                stage_timings=stage_timings,
                request_top_n=request_top_n,
                benchmark_top_k=self.config.benchmark_top_k,
                heap_capacity=request_top_n,
                top_results_count=len(top_results),
                exact_candidates_evaluated=scored_count,
                risk_mode=risk_mode,
                direction_mode=backend.direction_mode,
                backend_id=backend.backend_id,
                arity=arity,
                status=NO_RISK_EXACT_SCORED_STATUS,
                backend_logical_name=backend_logical_name,
                backend_implementation_id=backend.backend_id,
                metric_names=NO_RISK_METRIC_NAMES,
                sample_metrics=sample_metrics,
                numba_num_threads=int(numba_telemetry["numba_num_threads"]),
                numba_thread_source=str(numba_telemetry["numba_thread_source"]),
            )
        finally:
            cleanup_start = time.perf_counter()
            if heap is not None:
                heap.clear()
            selected_batch = None
            first_selected_batch = None
            selected_batches_iter = None
            del selected_batch
            del first_selected_batch
            del selected_batches_iter
            del top_k_context
            del execution_open_1m
            del execution_close_1m
            del prepared_result
            del combo_planning_result
            del normalized_request
            cleanup_duration_s = time.perf_counter() - cleanup_start

        if top_results is None or telemetry is None:
            raise RuntimeError("no-risk exact scoring did not produce a compact result")
        return BacktestNoRiskExactResult(
            execution_context=execution_context,
            top_results=top_results,
            telemetry=telemetry,
            self_check=self_check_summary,
            memory_cleanup_evidence=BacktestNoRiskMemoryCleanupEvidence(
                checked_reference_names=(
                    "prepared_result",
                    "combo_planning_result",
                    "prepared_pools",
                    "exact_context",
                    "proxy_context",
                    "execution_open_1m",
                    "execution_close_1m",
                    "selected_batches_iter",
                    "metric_buffers",
                    "heap",
                ),
                retained_heavy_reference_names=(),
                result_contains_heavy_references=False,
                cleanup_duration_s=cleanup_duration_s,
            ),
        )

    def _score_selected_rows(
        self,
        *,
        selected_batch: _SelectedCandidateBatch,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        execution_settings: _ExecutionSettings,
        execution_open_1m: np.ndarray,
        execution_close_1m: np.ndarray,
        stage_timings: dict[str, float],
        sample_metrics: Mapping[str, float] | None,
        heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]],
        top_k_context: _TopKContext,
        top_k: int,
        ranking: _RankingSpec,
    ) -> tuple[int, Mapping[str, float] | None]:
        selected_rows_by_indicator = selected_batch.rows_by_indicator
        selected_size = _selected_size(selected_rows_by_indicator)
        if selected_size <= 0:
            return 0, sample_metrics
        allocation_start = time.perf_counter()
        buffers = _allocate_metric_buffers(selected_size)
        _add_timing(
            stage_timings=stage_timings if _exact_profile_enabled() else None,
            key=NO_RISK_METRIC_BUFFER_ALLOCATION_STAGE_NAME,
            elapsed=time.perf_counter() - allocation_start,
        )
        exact_start = time.perf_counter()
        evaluate_no_risk_exact_chunk(
            selected_rows_by_indicator=selected_rows_by_indicator,
            prepared_result=prepared_result,
            combo_planning_result=combo_planning_result,
            execution_settings=execution_settings,
            execution_open_1m=execution_open_1m,
            execution_close_1m=execution_close_1m,
            buffers=buffers,
        )
        stage_timings[NO_RISK_EXACT_SCORING_STAGE_NAME] += (
            time.perf_counter() - exact_start
        )
        if sample_metrics is None:
            sample_metrics = _metrics_at(buffers=buffers, index=0)
        heap_start = time.perf_counter()
        if ranking.is_default_total_return_desc:
            _update_heap_total_return_desc(
                heap=heap,
                top_k_context=top_k_context,
                selected_rows_by_indicator=selected_rows_by_indicator,
                buffers=buffers,
                confirm=selected_batch.confirm,
                proxy=selected_batch.proxy,
                top_k=top_k,
            )
        else:
            _update_heap_generic_ranking(
                heap=heap,
                top_k_context=top_k_context,
                selected_rows_by_indicator=selected_rows_by_indicator,
                buffers=buffers,
                confirm=selected_batch.confirm,
                proxy=selected_batch.proxy,
                top_k=top_k,
                ranking=ranking,
            )
        stage_timings[NO_RISK_HEAP_UPDATE_STAGE_NAME] += time.perf_counter() - heap_start
        return buffers.size, sample_metrics

    def _run_self_check(
        self,
        *,
        selected_rows_by_indicator: Mapping[str, np.ndarray] | None,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        execution_settings: _ExecutionSettings,
        execution_open_1m: np.ndarray,
        execution_close_1m: np.ndarray,
        backend_logical_name: str,
    ) -> BacktestNoRiskSelfCheckSummary:
        if selected_rows_by_indicator is not None:
            return run_fast_vs_reference_self_check(
                selected_rows_by_indicator=selected_rows_by_indicator,
                prepared_result=prepared_result,
                combo_planning_result=combo_planning_result,
                execution_settings=execution_settings,
                execution_open_1m=execution_open_1m,
                execution_close_1m=execution_close_1m,
                backend_logical_name=backend_logical_name,
                check_n=self.config.self_check_sample_size,
                return_tolerance=self.config.self_check_return_tolerance,
            )
        return BacktestNoRiskSelfCheckSummary(
            enabled=True,
            status=NO_RISK_SELF_CHECK_PASSED_STATUS,
            sample_size=0,
            mismatches=0,
            max_abs_diff=0.0,
            backend_logical_name=backend_logical_name,
            backend_implementation_id=combo_planning_result.backend.backend_id,
            direction_mode=combo_planning_result.backend.direction_mode,
            trade_count_equal=True,
            return_tolerance=self.config.self_check_return_tolerance,
        )


def run_fast_vs_reference_self_check(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    execution_settings: _ExecutionSettings,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    backend_logical_name: str,
    check_n: int,
    return_tolerance: float,
) -> BacktestNoRiskSelfCheckSummary:
    """
    Compare the selected fast backend against a slow generic reference.
    """

    if check_n < 0:
        raise BacktestNoRiskExactRejected("self_check_sample_size must be >= 0")
    n_check = min(int(check_n), _selected_size(selected_rows_by_indicator))
    if n_check <= 0:
        return BacktestNoRiskSelfCheckSummary(
            enabled=True,
            status=NO_RISK_SELF_CHECK_PASSED_STATUS,
            sample_size=0,
            mismatches=0,
            max_abs_diff=0.0,
            backend_logical_name=backend_logical_name,
            backend_implementation_id=combo_planning_result.backend.backend_id,
            direction_mode=combo_planning_result.backend.direction_mode,
            trade_count_equal=True,
            return_tolerance=return_tolerance,
        )

    subset = {
        indicator_id: np.ascontiguousarray(rows[:n_check])
        for indicator_id, rows in selected_rows_by_indicator.items()
    }
    buffers = _allocate_metric_buffers(n_check)
    evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=subset,
        prepared_result=prepared_result,
        combo_planning_result=combo_planning_result,
        execution_settings=execution_settings,
        execution_open_1m=execution_open_1m,
        execution_close_1m=execution_close_1m,
        buffers=buffers,
    )

    max_abs_diff = 0.0
    trade_count_equal = True
    mismatches = 0
    for row_idx in range(n_check):
        local_indices = tuple(
            int(subset[indicator_id][row_idx])
            for indicator_id in prepared_result.indicator_ids
        )
        reference_metrics = evaluate_no_risk_reference_rows_slow(
            prepared_result=prepared_result,
            local_indices=local_indices,
            execution_settings=execution_settings,
            execution_open_1m=execution_open_1m,
            execution_close_1m=execution_close_1m,
        )
        if int(reference_metrics["trade_count"]) != int(buffers.trade_count[row_idx]):
            trade_count_equal = False
            mismatches += 1
            continue
        abs_diff = abs(
            float(reference_metrics["total_return_pct"])
            - float(buffers.total_return_pct[row_idx])
        )
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
        if abs_diff > return_tolerance:
            mismatches += 1

    if mismatches > 0 or not trade_count_equal:
        raise BacktestNoRiskSelfCheckFailed(
            "no-risk exact self-check failed: "
            f"mismatches={mismatches}, trade_count_equal={trade_count_equal}, "
            f"max_abs_diff={max_abs_diff}, tolerance={return_tolerance}"
        )
    return BacktestNoRiskSelfCheckSummary(
        enabled=True,
        status=NO_RISK_SELF_CHECK_PASSED_STATUS,
        sample_size=n_check,
        mismatches=0,
        max_abs_diff=max_abs_diff,
        backend_logical_name=backend_logical_name,
        backend_implementation_id=combo_planning_result.backend.backend_id,
        direction_mode=combo_planning_result.backend.direction_mode,
        trade_count_equal=True,
        return_tolerance=return_tolerance,
    )


def evaluate_no_risk_exact_chunk(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    execution_settings: _ExecutionSettings,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    buffers: _MetricBuffers,
) -> None:
    """
    Dispatch one selected no-risk chunk through the configured exact backend.
    """

    backend = combo_planning_result.backend
    indicator_ids = tuple(prepared_result.indicator_ids)
    if backend.backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND:
        left_id, right_id = indicator_ids
        left_pool = _pool_by_id(prepared_result)[left_id]
        right_pool = _pool_by_id(prepared_result)[right_id]
        event_segments_2_no_risk(
            np.asarray(selected_rows_by_indicator[left_id], dtype=np.int32),
            np.asarray(selected_rows_by_indicator[right_id], dtype=np.int32),
            left_pool.segments.starts,
            left_pool.segments.ends,
            left_pool.segments.values,
            left_pool.segments.counts,
            right_pool.segments.starts,
            right_pool.segments.ends,
            right_pool.segments.values,
            right_pool.segments.counts,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            execution_open_1m,
            execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.quote_amount,
            execution_settings.equity_pct,
            execution_settings.min_quote,
            execution_settings.max_quote,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.use_profit_lock,
            BARS_PER_YEAR_EXEC_1M,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            buffers.total_return_pct,
            buffers.max_drawdown_pct,
            buffers.return_over_max_drawdown,
            buffers.profit_factor,
            buffers.trade_count,
            buffers.sharpe_trades,
            buffers.win_rate_pct,
            buffers.avg_trade_ret_pct,
            buffers.avg_trade_exec_bars,
            buffers.exposure_pct,
        )
        return

    if backend.backend_id == STREAMING_2_NO_RISK_BACKEND:
        left_id, right_id = indicator_ids
        pool_by_id = _pool_by_id(prepared_result)
        streaming_2_no_risk(
            np.asarray(selected_rows_by_indicator[left_id], dtype=np.int32),
            np.asarray(selected_rows_by_indicator[right_id], dtype=np.int32),
            pool_by_id[left_id].trade_T,
            pool_by_id[right_id].trade_T,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            execution_open_1m,
            execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.quote_amount,
            execution_settings.equity_pct,
            execution_settings.min_quote,
            execution_settings.max_quote,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.use_profit_lock,
            BARS_PER_YEAR_EXEC_1M,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            buffers.total_return_pct,
            buffers.max_drawdown_pct,
            buffers.return_over_max_drawdown,
            buffers.profit_factor,
            buffers.trade_count,
            buffers.sharpe_trades,
            buffers.win_rate_pct,
            buffers.avg_trade_ret_pct,
            buffers.avg_trade_exec_bars,
            buffers.exposure_pct,
        )
        return

    if backend.backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND:
        exact_context = combo_planning_result.exact_context
        if (
            exact_context.starts is None
            or exact_context.ends is None
            or exact_context.values is None
            or exact_context.counts is None
        ):
            raise BacktestNoRiskExactRejected(
                "event_segments_n_no_risk requires a materialized exact context"
            )
        combo_idx_by_indicator = make_combo_idx_matrix(
            combo_chunk=_synthetic_combo_chunk(
                indicator_ids=indicator_ids,
                selected_rows_by_indicator=selected_rows_by_indicator,
            ),
            indicator_ids=indicator_ids,
        )
        segment_pos_workspace = np.empty(
            (combo_idx_by_indicator.shape[1], combo_idx_by_indicator.shape[0]),
            dtype=np.int32,
        )
        event_segments_n_no_risk(
            combo_idx_by_indicator,
            exact_context.starts,
            exact_context.ends,
            exact_context.values,
            exact_context.counts,
            segment_pos_workspace,
            prepared_result.execution_mapping.signal_entry_exec_idx_15m,
            execution_open_1m,
            execution_close_1m,
            np.int32(prepared_result.execution_mapping.t_exec_limit_1m),
            execution_settings.initial_cash_quote,
            execution_settings.sizing_mode_code,
            execution_settings.quote_amount,
            execution_settings.equity_pct,
            execution_settings.min_quote,
            execution_settings.max_quote,
            execution_settings.fee_rate,
            execution_settings.slippage_rate,
            execution_settings.safe_profit_percent,
            execution_settings.use_profit_lock,
            BARS_PER_YEAR_EXEC_1M,
            execution_settings.close_on_end,
            execution_settings.direction_mode_code,
            buffers.total_return_pct,
            buffers.max_drawdown_pct,
            buffers.return_over_max_drawdown,
            buffers.profit_factor,
            buffers.trade_count,
            buffers.sharpe_trades,
            buffers.win_rate_pct,
            buffers.avg_trade_ret_pct,
            buffers.avg_trade_exec_bars,
            buffers.exposure_pct,
        )
        return

    raise BacktestNoRiskExactRejected(f"Unsupported no-risk backend {backend.backend_id!r}")


@nb.njit(cache=True, inline="always")
def _consensus_dir2(left_value: np.int8, right_value: np.int8) -> np.int8:
    if left_value == 1 and right_value == 1:
        return np.int8(1)
    if left_value == -1 and right_value == -1:
        return np.int8(-1)
    return np.int8(0)


@nb.njit(cache=True, inline="always")
def proxy_for_two_rows(
    left_eval_row: np.ndarray,
    right_eval_row: np.ndarray,
    ret_15m: np.ndarray,
    min_confirm: np.int32,
    fee_penalty_per_confirm: np.float32,
) -> tuple[np.int32, np.float32]:
    confirms = np.int32(0)
    proxy = np.float32(0.0)
    for interval_idx in range(ret_15m.shape[0]):
        dirn = _consensus_dir2(left_eval_row[interval_idx], right_eval_row[interval_idx])
        if dirn == 1:
            confirms += 1
            proxy += ret_15m[interval_idx]
        elif dirn == -1:
            confirms += 1
            proxy -= ret_15m[interval_idx]
    if confirms >= min_confirm:
        return confirms, proxy - fee_penalty_per_confirm * np.float32(confirms)
    return confirms, NEG_INF


def proxy_for_indicator_rows(
    *,
    eval_rows: tuple[np.ndarray, ...],
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
) -> tuple[int, float]:
    if len(eval_rows) == 2:
        confirms, proxy = proxy_for_two_rows(
            eval_rows[0],
            eval_rows[1],
            np.asarray(ret_15m, dtype=np.float32),
            np.int32(min_confirm),
            fee_penalty_per_confirm,
        )
        return int(confirms), float(proxy)

    consensus = np.asarray(eval_rows[0], dtype=np.int8).copy()
    for eval_row in eval_rows[1:]:
        consensus[consensus != eval_row] = np.int8(0)
    confirms = int(np.count_nonzero(consensus))
    if confirms < int(min_confirm):
        return confirms, float(NEG_INF)

    returns = np.asarray(ret_15m, dtype=np.float32)
    proxy = np.float32(0.0)
    if confirms:
        proxy += np.sum(returns[consensus == 1], dtype=np.float32)
        proxy -= np.sum(returns[consensus == -1], dtype=np.float32)
    proxy -= fee_penalty_per_confirm * np.float32(confirms)
    return confirms, float(proxy)


@nb.njit(cache=True, inline="always")
def _apply_direction_mode(raw_dir: np.int8 | int, direction_mode: np.int8 | int) -> np.int8:
    if direction_mode == 1:
        if raw_dir == 1:
            return np.int8(1)
        return np.int8(0)
    return np.int8(raw_dir)


@nb.njit(cache=True, inline="always")
def _trade_sharpe(
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


@nb.njit(cache=True, inline="always")
def _apply_no_risk_trade_to_state(
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
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    np.int32,
    np.int32,
    float,
    float,
    float,
    float,
    float,
]:
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

    quote_amount = execution_quote_amount(
        available_quote,
        equity,
        sizing_mode_code,
        configured_quote_amount,
        equity_pct,
        min_quote,
        max_quote,
    )
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


@nb.njit(cache=True, inline="always")
def _write_final_metrics(
    k: int,
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
    t_exec: np.int32,
    bars_per_year_exec: float,
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
    sharpe_trades = _trade_sharpe(
        closed_trade_count,
        sum_trade_return,
        sum_trade_return_squared,
        bars_per_year_exec,
        t_exec,
    )
    out_total_return_pct[k] = total_return_pct
    out_max_drawdown_pct[k] = max_drawdown_pct
    out_return_over_max_drawdown[k] = return_over_max_drawdown
    out_profit_factor[k] = profit_factor
    out_trade_count[k] = closed_trade_count
    out_sharpe_trades[k] = sharpe_trades
    out_win_rate_pct[k] = win_rate_pct
    out_avg_trade_ret_pct[k] = avg_trade_ret_pct
    out_avg_trade_exec_bars[k] = avg_trade_exec_bars
    out_exposure_pct[k] = exposure_pct


@nb.njit(cache=True, parallel=True, fastmath=True)
def event_segments_2_no_risk(
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
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
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
    for k in nb.prange(combo_left_idx.shape[0]):
        left_row = combo_left_idx[k]
        right_row = combo_right_idx[k]
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
        current_dir = np.int8(0)
        current_entry = np.int32(0)
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
                dirn = _apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
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
                        ) = _apply_no_risk_trade_to_state(
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
                            init_cash_quote,
                            sizing_mode_code,
                            configured_quote_amount,
                            equity_pct,
                            min_quote,
                            max_quote,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
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
                        ) = _apply_no_risk_trade_to_state(
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
                            init_cash_quote,
                            sizing_mode_code,
                            configured_quote_amount,
                            equity_pct,
                            min_quote,
                            max_quote,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            use_profit_lock,
                        )
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)
            if left_end == segment_end:
                left_segment_idx += 1
            if right_end == segment_end:
                right_segment_idx += 1

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
            ) = _apply_no_risk_trade_to_state(
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
                init_cash_quote,
                sizing_mode_code,
                configured_quote_amount,
                equity_pct,
                min_quote,
                max_quote,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                use_profit_lock,
            )
        _write_final_metrics(
            k,
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
            t_exec,
            bars_per_year_exec,
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


@nb.njit(cache=True, parallel=True, fastmath=True)
def streaming_2_no_risk(
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_trade_t: np.ndarray,
    right_trade_t: np.ndarray,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
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
    for k in nb.prange(combo_left_idx.shape[0]):
        left_row = combo_left_idx[k]
        right_row = combo_right_idx[k]
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
        current_dir = np.int8(0)
        current_entry = np.int32(0)

        for signal_idx in range(left_trade_t.shape[1]):
            raw_dir = _consensus_dir2(
                left_trade_t[left_row, signal_idx],
                right_trade_t[right_row, signal_idx],
            )
            dirn = _apply_direction_mode(raw_dir, direction_mode)
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
                ) = _apply_no_risk_trade_to_state(
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
                    init_cash_quote,
                    sizing_mode_code,
                    configured_quote_amount,
                    equity_pct,
                    min_quote,
                    max_quote,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
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
                ) = _apply_no_risk_trade_to_state(
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
                    init_cash_quote,
                    sizing_mode_code,
                    configured_quote_amount,
                    equity_pct,
                    min_quote,
                    max_quote,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    use_profit_lock,
                )
                current_dir = dirn
                current_entry = np.int32(entry_exec)

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
            ) = _apply_no_risk_trade_to_state(
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
                init_cash_quote,
                sizing_mode_code,
                configured_quote_amount,
                equity_pct,
                min_quote,
                max_quote,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                use_profit_lock,
            )
        _write_final_metrics(
            k,
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
            t_exec,
            bars_per_year_exec,
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


@nb.njit(cache=True, parallel=True, fastmath=True)
def event_segments_n_no_risk(
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
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
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
    for k in nb.prange(combo_count):
        for indicator_pos in range(arity):
            segment_pos_workspace[k, indicator_pos] = np.int32(0)

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
        current_dir = np.int8(0)
        current_entry = np.int32(0)

        while True:
            active = True
            segment_start = np.int32(0)
            segment_end = np.int32(2147483647)
            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, k]
                segment_idx = segment_pos_workspace[k, indicator_pos]
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
                first_row_idx = combo_idx_by_indicator[0, k]
                first_segment_idx = segment_pos_workspace[k, 0]
                raw_dir = segment_values[0, first_row_idx, first_segment_idx]
                if raw_dir != 0:
                    for indicator_pos in range(1, arity):
                        row_idx = combo_idx_by_indicator[indicator_pos, k]
                        segment_idx = segment_pos_workspace[k, indicator_pos]
                        if segment_values[indicator_pos, row_idx, segment_idx] != raw_dir:
                            raw_dir = np.int8(0)
                            break
                dirn = _apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_exec = sig_entry_exec_idx[segment_start]
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
                        ) = _apply_no_risk_trade_to_state(
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
                            init_cash_quote,
                            sizing_mode_code,
                            configured_quote_amount,
                            equity_pct,
                            min_quote,
                            max_quote,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
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
                        ) = _apply_no_risk_trade_to_state(
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
                            init_cash_quote,
                            sizing_mode_code,
                            configured_quote_amount,
                            equity_pct,
                            min_quote,
                            max_quote,
                            fee_rate,
                            slippage_rate,
                            safe_profit_percent,
                            use_profit_lock,
                        )
                        current_dir = dirn
                        current_entry = np.int32(entry_exec)
            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, k]
                segment_idx = segment_pos_workspace[k, indicator_pos]
                if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                    segment_pos_workspace[k, indicator_pos] += np.int32(1)

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
            ) = _apply_no_risk_trade_to_state(
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
                init_cash_quote,
                sizing_mode_code,
                configured_quote_amount,
                equity_pct,
                min_quote,
                max_quote,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                use_profit_lock,
            )
        _write_final_metrics(
            k,
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
            t_exec,
            bars_per_year_exec,
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


def evaluate_no_risk_reference_rows_slow(
    *,
    prepared_result: BacktestPreparePoolsResult,
    local_indices: tuple[int, ...],
    execution_settings: _ExecutionSettings,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
) -> dict[str, float | int]:
    entry_arr, dir_arr, exit_arr = build_trade_list_for_indicator_rows_slow(
        prepared_result=prepared_result,
        local_indices=local_indices,
        direction_mode=execution_settings.direction_mode,
    )
    return _score_trade_list_no_risk_reference(
        entry_exec_idx=entry_arr,
        dir_arr=dir_arr,
        sig_exit_exec_idx=exit_arr,
        execution_open_1m=execution_open_1m,
        execution_close_1m=execution_close_1m,
        execution_settings=execution_settings,
        t_exec=prepared_result.execution_mapping.t_exec_limit_1m,
    )


def build_trade_list_for_indicator_rows_slow(
    *,
    prepared_result: BacktestPreparePoolsResult,
    local_indices: tuple[int, ...],
    direction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pools_by_id = _pool_by_id(prepared_result)
    rows = tuple(
        pools_by_id[indicator_id].trade_T[local_indices[pos]]
        for pos, indicator_id in enumerate(prepared_result.indicator_ids)
    )
    raw_signal = np.asarray(rows[0], dtype=np.int8).copy()
    for row in rows[1:]:
        raw_signal = np.where(row == raw_signal, raw_signal, np.int8(0)).astype(
            np.int8,
            copy=False,
        )
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        direction_signal = (raw_signal == np.int8(1)).astype(np.int8)
    elif direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        direction_signal = raw_signal
    else:
        raise BacktestNoRiskExactRejected(
            f"Unsupported direction_mode={direction_mode!r}; expected "
            f"{(DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL)!r}"
        )

    entry_exec: list[int] = []
    directions: list[int] = []
    sig_exit_exec: list[int] = []
    current_dir = 0
    current_entry = 0
    t_exec_limit = int(prepared_result.execution_mapping.t_exec_limit_1m)
    if int(direction_signal[0]) != 0:
        entry_idx = int(prepared_result.execution_mapping.signal_entry_exec_idx_15m[0])
        if entry_idx < t_exec_limit:
            current_dir = int(direction_signal[0])
            current_entry = entry_idx
    change_indices = np.flatnonzero(direction_signal[1:] != direction_signal[:-1]) + 1
    for signal_idx_raw in change_indices:
        signal_idx = int(signal_idx_raw)
        dirn = int(direction_signal[signal_idx])
        entry_idx = int(prepared_result.execution_mapping.signal_entry_exec_idx_15m[signal_idx])
        if entry_idx >= t_exec_limit:
            break
        if dirn == 0:
            if direction_mode != DIRECTION_MODE_LONG_ONLY or current_dir == 0:
                continue
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


def _score_trade_list_no_risk_reference(
    *,
    entry_exec_idx: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_exec_idx: np.ndarray,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    execution_settings: _ExecutionSettings,
    t_exec: int,
) -> dict[str, float | int]:
    available_quote = execution_settings.initial_cash_quote
    safe_quote = 0.0
    equity = execution_settings.initial_cash_quote
    peak_equity = equity
    max_drawdown_pct = 0.0
    gross_profit_quote = 0.0
    gross_loss_quote = 0.0
    closed_trade_count = 0
    win_count = 0
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    total_trade_return_pct = 0.0
    total_trade_exec_bars = 0.0
    exposure_bars = 0.0
    for trade_index in range(int(entry_exec_idx.size)):
        entry_idx = int(entry_exec_idx[trade_index])
        if entry_idx >= t_exec:
            continue
        exit_idx = int(sig_exit_exec_idx[trade_index])
        if exit_idx < t_exec:
            exit_exec_idx = exit_idx
            exit_price_raw = float(execution_open_1m[exit_exec_idx])
        elif execution_settings.close_on_end == 1 and t_exec > 0:
            exit_exec_idx = t_exec - 1
            exit_price_raw = float(execution_close_1m[exit_exec_idx])
        else:
            continue
        if available_quote <= 0.0:
            continue
        quote_amount = execution_quote_amount(
            available_quote,
            equity,
            execution_settings.sizing_mode_code,
            execution_settings.quote_amount,
            execution_settings.equity_pct,
            execution_settings.min_quote,
            execution_settings.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        trade_direction = int(dir_arr[trade_index])
        entry_price_raw = float(execution_open_1m[entry_idx])
        if trade_direction == 1:
            entry_fill_price = entry_price_raw * (1.0 + execution_settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 - execution_settings.slippage_rate)
        else:
            entry_fill_price = entry_price_raw * (1.0 - execution_settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 + execution_settings.slippage_rate)
        qty_base = quote_amount / entry_fill_price
        entry_fee_quote = quote_amount * execution_settings.fee_rate
        available_quote -= quote_amount + entry_fee_quote
        exit_quote_amount = qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * execution_settings.fee_rate
        if trade_direction == 1:
            gross_pnl_quote = exit_quote_amount - quote_amount
        else:
            gross_pnl_quote = quote_amount - exit_quote_amount
        available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote
        if execution_settings.use_profit_lock == 1 and net_pnl_quote > 0.0:
            locked_profit_quote = net_pnl_quote * (
                execution_settings.safe_profit_percent / 100.0
            )
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        equity = available_quote + safe_quote
        if equity > peak_equity:
            peak_equity = equity
        elif peak_equity > 0.0:
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
        trade_return = net_pnl_quote / quote_amount
        trade_return_pct = trade_return * 100.0
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

    total_return_pct = ((equity / execution_settings.initial_cash_quote) - 1.0) * 100.0
    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = math.inf
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
    sharpe_trades = _trade_sharpe_py(
        trade_count=closed_trade_count,
        sum_trade_return=sum_trade_return,
        sum_trade_return_squared=sum_trade_return_squared,
        bars_per_year_exec=BARS_PER_YEAR_EXEC_1M,
        sentinel_index=t_exec,
    )
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "return_over_max_drawdown": return_over_max_drawdown,
        "profit_factor": profit_factor,
        "trade_count": closed_trade_count,
        "sharpe_trades": sharpe_trades,
        "win_rate_pct": win_rate_pct,
        "avg_trade_ret_pct": avg_trade_ret_pct,
        "avg_trade_exec_bars": avg_trade_exec_bars,
        "exposure_pct": exposure_pct,
    }


def _trade_sharpe_py(
    *,
    trade_count: int,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: int,
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


def _apply_direction_mode_py(*, raw_dir: int, direction_mode: str) -> int:
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        return 1 if raw_dir == 1 else 0
    if direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        return int(raw_dir)
    raise BacktestNoRiskExactRejected(
        f"Unsupported direction_mode={direction_mode!r}; expected "
        f"{(DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL)!r}"
    )


def _execution_context_from_prepared(
    prepared_result: BacktestPreparePoolsResult,
) -> BacktestNoRiskExecutionContext:
    return BacktestNoRiskExecutionContext(
        timeframe=prepared_result.timeframe,
        execution_timeframe=CANONICAL_EXECUTION_TIMEFRAME_V1,
        time_slice_start_15m=prepared_result.time_slice_start_15m,
        time_slice_stop_15m=prepared_result.time_slice_stop_15m,
        trade_T_length=prepared_result.trade_T_length,
        eval_T_length=prepared_result.eval_T_length,
        t_exec_limit_1m=prepared_result.execution_mapping.t_exec_limit_1m,
    )


def _risk_mode_from_normalized(normalized_request: Mapping[str, Any]) -> str:
    risk = normalized_request.get("risk")
    if not isinstance(risk, Mapping):
        raise BacktestNoRiskExactRejected("normalized_request.risk must be a mapping")
    return str(risk.get("mode"))


def _request_top_n_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    default_request_top_n: int,
) -> int:
    raw_top_n = normalized_request.get("top_n", default_request_top_n)
    if isinstance(raw_top_n, bool) or not isinstance(raw_top_n, int):
        raise BacktestNoRiskExactRejected("normalized_request.top_n must be an integer")
    if raw_top_n <= 0:
        raise BacktestNoRiskExactRejected("normalized_request.top_n must be > 0")
    return raw_top_n


def _execution_settings_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    expected_direction_mode: str,
    config: BacktestNoRiskExactConfig,
) -> _ExecutionSettings:
    return execution_settings_from_normalized(
        normalized_request,
        expected_direction_mode=expected_direction_mode,
        config=config,
        rejection_cls=BacktestNoRiskExactRejected,
    )


def _execution_price_arrays_from_prepared(
    prepared_result: BacktestPreparePoolsResult,
) -> tuple[np.ndarray, np.ndarray]:
    if prepared_result.execution_open_1m is None or prepared_result.execution_close_1m is None:
        raise BacktestNoRiskExactRejected(
            "no-risk exact scoring requires execution_open_1m and execution_close_1m "
            "from prepare_pools"
        )
    open_1m = np.ascontiguousarray(
        np.asarray(prepared_result.execution_open_1m, dtype=np.float32)
    )
    close_1m = np.ascontiguousarray(
        np.asarray(prepared_result.execution_close_1m, dtype=np.float32)
    )
    if open_1m.ndim != 1 or close_1m.ndim != 1:
        raise BacktestNoRiskExactRejected("execution price arrays must be one-dimensional")
    t_exec = int(prepared_result.execution_mapping.t_exec_limit_1m)
    if int(open_1m.shape[0]) < t_exec or int(close_1m.shape[0]) < t_exec:
        raise BacktestNoRiskExactRejected("execution price arrays are shorter than t_exec_limit_1m")
    return open_1m, close_1m


def _validate_backend_for_exact_scoring(*, backend_id: str, arity: int) -> None:
    if arity == 2 and backend_id in (
        EVENT_SEGMENTS_2_NO_RISK_BACKEND,
        STREAMING_2_NO_RISK_BACKEND,
    ):
        return
    if backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND and 1 <= arity <= 10:
        return
    raise BacktestNoRiskExactRejected(
        f"backend {backend_id!r} does not support no-risk exact arity {arity}"
    )


def _backend_logical_name(*, backend_id: str, arity: int) -> str:
    if backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND:
        return f"event_segments_{arity}_no_risk"
    return backend_id


def _iter_selected_candidate_batches(
    *,
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    profile_stage_timings: dict[str, float] | None = None,
) -> Iterator[_SelectedCandidateBatch]:
    local_row_pools = build_local_row_pools(prepared_result=prepared_result)
    filter_service = BacktestComboPlanningService()
    combo_chunks = iter_ordinal_combo_chunks(
        indicator_ids=prepared_result.indicator_ids,
        local_row_pools=local_row_pools,
        chunk_size=COMBO_CHUNK_SIZE,
    )
    while True:
        decode_start = time.perf_counter()
        try:
            combo_chunk = next(combo_chunks)
        except StopIteration:
            _add_timing(
                stage_timings=profile_stage_timings,
                key=NO_RISK_COMBO_CHUNK_DECODE_STAGE_NAME,
                elapsed=time.perf_counter() - decode_start,
            )
            break
        _add_timing(
            stage_timings=profile_stage_timings,
            key=NO_RISK_COMBO_CHUNK_DECODE_STAGE_NAME,
            elapsed=time.perf_counter() - decode_start,
        )
        if not combo_planning_result.proxy_context.active:
            yield _SelectedCandidateBatch(
                rows_by_indicator=combo_chunk.rows_by_indicator,
                confirm=None,
                proxy=None,
            )
            continue
        proxy_start = time.perf_counter()
        filter_result = filter_service.proxy_filter(
            combo_chunk=combo_chunk,
            proxy_context=combo_planning_result.proxy_context,
        )
        _add_timing(
            stage_timings=profile_stage_timings,
            key=NO_RISK_PROXY_FILTER_STAGE_NAME,
            elapsed=time.perf_counter() - proxy_start,
        )
        if filter_result.selected_candidate_count > 0:
            yield _SelectedCandidateBatch(
                rows_by_indicator=filter_result.selected_rows_by_indicator,
                confirm=filter_result.confirm,
                proxy=filter_result.proxy,
            )


def _exact_profile_enabled() -> bool:
    raw = os.environ.get("ROEHUB_BACKTEST_EXACT_PROFILE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _add_timing(
    *,
    stage_timings: dict[str, float] | None,
    key: str,
    elapsed: float,
) -> None:
    if stage_timings is None:
        return
    stage_timings[key] = stage_timings.get(key, 0.0) + elapsed


def _next_selected_candidate_batch(
    selected_batches_iter: Iterator[_SelectedCandidateBatch],
) -> _SelectedCandidateBatch | None:
    for selected_batch in selected_batches_iter:
        if _selected_size(selected_batch.rows_by_indicator) > 0:
            return selected_batch
    return None


def _selected_size(selected_rows_by_indicator: Mapping[str, np.ndarray]) -> int:
    if not selected_rows_by_indicator:
        return 0
    first = next(iter(selected_rows_by_indicator.values()))
    return int(first.shape[0])


def _allocate_metric_buffers(size: int) -> _MetricBuffers:
    return _MetricBuffers(
        total_return_pct=np.empty(size, dtype=np.float64),
        max_drawdown_pct=np.empty(size, dtype=np.float64),
        return_over_max_drawdown=np.empty(size, dtype=np.float64),
        profit_factor=np.empty(size, dtype=np.float64),
        trade_count=np.empty(size, dtype=np.int32),
        sharpe_trades=np.empty(size, dtype=np.float64),
        win_rate_pct=np.empty(size, dtype=np.float64),
        avg_trade_ret_pct=np.empty(size, dtype=np.float64),
        avg_trade_exec_bars=np.empty(size, dtype=np.float64),
        exposure_pct=np.empty(size, dtype=np.float64),
    )


def _metrics_at(*, buffers: _MetricBuffers, index: int) -> Mapping[str, float]:
    return {
        "total_return_pct": float(buffers.total_return_pct[index]),
        "max_drawdown_pct": float(buffers.max_drawdown_pct[index]),
        "return_over_max_drawdown": float(buffers.return_over_max_drawdown[index]),
        "profit_factor": float(buffers.profit_factor[index]),
        "trade_count": float(buffers.trade_count[index]),
        "sharpe_trades": float(buffers.sharpe_trades[index]),
        "win_rate_pct": float(buffers.win_rate_pct[index]),
        "avg_trade_ret_pct": float(buffers.avg_trade_ret_pct[index]),
        "avg_trade_exec_bars": float(buffers.avg_trade_exec_bars[index]),
        "exposure_pct": float(buffers.exposure_pct[index]),
    }


def _metric_values_at(*, buffers: _MetricBuffers, index: int) -> tuple[float, ...]:
    return (
        float(buffers.total_return_pct[index]),
        float(buffers.max_drawdown_pct[index]),
        float(buffers.return_over_max_drawdown[index]),
        float(buffers.profit_factor[index]),
        float(buffers.trade_count[index]),
        float(buffers.sharpe_trades[index]),
        float(buffers.win_rate_pct[index]),
        float(buffers.avg_trade_ret_pct[index]),
        float(buffers.avg_trade_exec_bars[index]),
        float(buffers.exposure_pct[index]),
    )


def _metrics_from_values(metric_values: tuple[float, ...]) -> dict[str, float]:
    return {
        metric_name: metric_values[pos]
        for pos, metric_name in enumerate(NO_RISK_METRIC_NAMES)
    }


def _top_k_context_from_prepared(
    prepared_result: BacktestPreparePoolsResult,
) -> _TopKContext:
    pools_by_id = _pool_by_id(prepared_result)
    indicator_ids = tuple(prepared_result.indicator_ids)
    return _TopKContext(
        indicator_ids=indicator_ids,
        row_ids_by_pos=tuple(pools_by_id[indicator_id].row_ids for indicator_id in indicator_ids),
        metadata_by_pos=tuple(
            pools_by_id[indicator_id].metadata for indicator_id in indicator_ids
        ),
    )


def _update_heap_total_return_desc(
    *,
    heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]],
    top_k_context: _TopKContext,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    buffers: _MetricBuffers,
    confirm: np.ndarray | None,
    proxy: np.ndarray | None,
    top_k: int,
) -> None:
    _update_heap_from_score_values(
        heap=heap,
        top_k_context=top_k_context,
        selected_rows_by_indicator=selected_rows_by_indicator,
        buffers=buffers,
        score_values=buffers.total_return_pct,
        score_multiplier=1.0,
        confirm=confirm,
        proxy=proxy,
        top_k=top_k,
    )


def _update_heap_generic_ranking(
    *,
    heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]],
    top_k_context: _TopKContext,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    buffers: _MetricBuffers,
    confirm: np.ndarray | None,
    proxy: np.ndarray | None,
    top_k: int,
    ranking: _RankingSpec,
) -> None:
    score_multiplier = 1.0 if ranking.direction == "desc" else -1.0
    _update_heap_from_score_values(
        heap=heap,
        top_k_context=top_k_context,
        selected_rows_by_indicator=selected_rows_by_indicator,
        buffers=buffers,
        score_values=_score_array_for_metric(buffers=buffers, metric_name=ranking.metric_name),
        score_multiplier=score_multiplier,
        confirm=confirm,
        proxy=proxy,
        top_k=top_k,
    )


def _update_heap_from_score_values(
    *,
    heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]],
    top_k_context: _TopKContext,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    buffers: _MetricBuffers,
    score_values: np.ndarray,
    score_multiplier: float,
    confirm: np.ndarray | None,
    proxy: np.ndarray | None,
    top_k: int,
) -> None:
    selected_rows_by_pos = tuple(
        selected_rows_by_indicator[indicator_id]
        for indicator_id in top_k_context.indicator_ids
    )
    row_ids_by_pos = top_k_context.row_ids_by_pos
    arity = len(selected_rows_by_pos)
    if arity == 1:
        selected_0 = selected_rows_by_pos[0]
        row_ids_0 = row_ids_by_pos[0]
        for result_index in range(buffers.size):
            local_0 = int(selected_0[result_index])
            local_indices = (local_0,)
            original_rows = (int(row_ids_0[local_0]),)
            score = float(score_values[result_index])
            heap_key = (score * score_multiplier, original_rows)
            if len(heap) < top_k:
                heapq.heappush(
                    heap,
                    (
                        heap_key,
                        _materialize_heap_entry_arity1(
                            top_k_context=top_k_context,
                            local_index=local_0,
                            original_row=original_rows[0],
                            score=score,
                            buffers=buffers,
                            result_index=result_index,
                            confirm=confirm,
                            proxy=proxy,
                        ),
                    ),
                )
            elif heap_key > heap[0][0]:
                heapq.heapreplace(
                    heap,
                    (
                        heap_key,
                        _materialize_heap_entry_arity1(
                            top_k_context=top_k_context,
                            local_index=local_0,
                            original_row=original_rows[0],
                            score=score,
                            buffers=buffers,
                            result_index=result_index,
                            confirm=confirm,
                            proxy=proxy,
                        ),
                    ),
                )
        return

    if arity == 2:
        selected_0 = selected_rows_by_pos[0]
        selected_1 = selected_rows_by_pos[1]
        row_ids_0 = row_ids_by_pos[0]
        row_ids_1 = row_ids_by_pos[1]
        for result_index in range(buffers.size):
            local_0 = int(selected_0[result_index])
            local_1 = int(selected_1[result_index])
            local_indices = (local_0, local_1)
            original_rows = (int(row_ids_0[local_0]), int(row_ids_1[local_1]))
            score = float(score_values[result_index])
            heap_key = (score * score_multiplier, original_rows)
            if len(heap) < top_k:
                heapq.heappush(
                    heap,
                    (
                        heap_key,
                        _materialize_heap_entry(
                            top_k_context=top_k_context,
                            local_indices=local_indices,
                            original_rows=original_rows,
                            score=score,
                            buffers=buffers,
                            result_index=result_index,
                            confirm=confirm,
                            proxy=proxy,
                        ),
                    ),
                )
            elif heap_key > heap[0][0]:
                heapq.heapreplace(
                    heap,
                    (
                        heap_key,
                        _materialize_heap_entry(
                            top_k_context=top_k_context,
                            local_indices=local_indices,
                            original_rows=original_rows,
                            score=score,
                            buffers=buffers,
                            result_index=result_index,
                            confirm=confirm,
                            proxy=proxy,
                        ),
                    ),
                )
        return

    for result_index in range(buffers.size):
        local_values = []
        original_values = []
        for pos, selected_rows in enumerate(selected_rows_by_pos):
            local_row = int(selected_rows[result_index])
            local_values.append(local_row)
            original_values.append(int(row_ids_by_pos[pos][local_row]))
        local_indices = tuple(local_values)
        original_rows = tuple(original_values)
        score = float(score_values[result_index])
        heap_key = (score * score_multiplier, original_rows)
        if len(heap) < top_k:
            heapq.heappush(
                heap,
                (
                    heap_key,
                    _materialize_heap_entry(
                        top_k_context=top_k_context,
                        local_indices=local_indices,
                        original_rows=original_rows,
                        score=score,
                        buffers=buffers,
                        result_index=result_index,
                        confirm=confirm,
                        proxy=proxy,
                    ),
                ),
            )
        elif heap_key > heap[0][0]:
            heapq.heapreplace(
                heap,
                (
                    heap_key,
                    _materialize_heap_entry(
                        top_k_context=top_k_context,
                        local_indices=local_indices,
                        original_rows=original_rows,
                        score=score,
                        buffers=buffers,
                        result_index=result_index,
                        confirm=confirm,
                        proxy=proxy,
                    ),
                ),
            )


def _materialize_heap_entry(
    *,
    top_k_context: _TopKContext,
    local_indices: tuple[int, ...],
    original_rows: tuple[int, ...],
    score: float,
    buffers: _MetricBuffers,
    result_index: int,
    confirm: np.ndarray | None,
    proxy: np.ndarray | None,
) -> _NoRiskHeapEntry:
    proxy_pending = confirm is None or proxy is None
    if proxy_pending:
        confirm_count = 0
        proxy_score = 0.0
    else:
        assert confirm is not None
        assert proxy is not None
        confirm_count = int(confirm[result_index])
        proxy_score = float(proxy[result_index])

    return _NoRiskHeapEntry(
        score=score,
        original_rows=original_rows,
        local_indices=local_indices,
        metric_values=_metric_values_at(buffers=buffers, index=result_index),
        metric_buffers=None,
        metric_index=-1,
        metadata_by_pos=tuple(
            top_k_context.metadata_by_pos[pos][local_indices[pos]]
            for pos in range(len(local_indices))
        ),
        confirm_count=confirm_count,
        proxy_score=proxy_score,
        proxy_pending=proxy_pending,
    )


def _materialize_heap_entry_arity1(
    *,
    top_k_context: _TopKContext,
    local_index: int,
    original_row: int,
    score: float,
    buffers: _MetricBuffers,
    result_index: int,
    confirm: np.ndarray | None,
    proxy: np.ndarray | None,
) -> _NoRiskHeapEntry:
    proxy_pending = confirm is None or proxy is None
    if proxy_pending:
        confirm_count = 0
        proxy_score = 0.0
    else:
        assert confirm is not None
        assert proxy is not None
        confirm_count = int(confirm[result_index])
        proxy_score = float(proxy[result_index])

    return _NoRiskHeapEntry(
        score=score,
        original_rows=(original_row,),
        local_indices=(local_index,),
        metric_values=_metric_values_at(buffers=buffers, index=result_index),
        metric_buffers=None,
        metric_index=-1,
        metadata_by_pos=(top_k_context.metadata_by_pos[0][local_index],),
        confirm_count=confirm_count,
        proxy_score=proxy_score,
        proxy_pending=proxy_pending,
    )


def _top_results_from_heap(
    heap: list[tuple[tuple[float, tuple[int, ...]], _NoRiskHeapEntry]],
    *,
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    top_k_context: _TopKContext,
) -> tuple[BacktestNoRiskTopResult, ...]:
    pools_by_id = _pool_by_id(prepared_result)
    eval_rows_by_pos = tuple(
        np.ascontiguousarray(pools_by_id[indicator_id].eval_T)
        for indicator_id in top_k_context.indicator_ids
    )
    proxy_context = combo_planning_result.proxy_context
    ret_15m = np.asarray(prepared_result.signal_returns_15m, dtype=np.float32)

    return tuple(
        BacktestNoRiskTopResult(
            rank=rank,
            score=entry.score,
            indicator_rows={
                indicator_id: entry.original_rows[pos]
                for pos, indicator_id in enumerate(top_k_context.indicator_ids)
            },
            metrics=_metrics_from_values(_metric_values_from_heap_entry(entry)),
            metadata=_top_result_metadata(
                entry,
                top_k_context=top_k_context,
                proxy_fill=_proxy_fill_for_heap_entry(
                    entry=entry,
                    eval_rows_by_pos=eval_rows_by_pos,
                    ret_15m=ret_15m,
                    min_confirm=proxy_context.combo_min_confirm,
                    fee_penalty_per_confirm=proxy_context.fee_penalty_per_confirm,
                ),
            ),
        )
        for rank, (_, entry) in enumerate(
            sorted(heap, key=lambda pair: pair[0], reverse=True),
            start=1,
        )
    )


def _metric_values_from_heap_entry(entry: _NoRiskHeapEntry) -> tuple[float, ...]:
    if entry.metric_values:
        return entry.metric_values
    if entry.metric_buffers is None:
        raise RuntimeError("deferred no-risk heap metrics require metric buffers")
    return _metric_values_at(buffers=entry.metric_buffers, index=entry.metric_index)


def _top_result_metadata(
    entry: _NoRiskHeapEntry,
    *,
    top_k_context: _TopKContext,
    proxy_fill: tuple[int, float],
) -> dict[str, Any]:
    confirm_count, proxy_score = proxy_fill
    metadata: dict[str, Any] = {
        "confirm_count": confirm_count,
        "proxy_score": proxy_score,
    }
    for pos, indicator_id in enumerate(top_k_context.indicator_ids):
        row_metadata = entry.metadata_by_pos[pos].as_mapping()
        for key, value in row_metadata.items():
            metadata[f"{indicator_id}.{key}"] = value
    return metadata


def _proxy_fill_for_heap_entry(
    *,
    entry: _NoRiskHeapEntry,
    eval_rows_by_pos: tuple[np.ndarray, ...],
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
) -> tuple[int, float]:
    if not entry.proxy_pending:
        return entry.confirm_count, entry.proxy_score

    eval_rows = tuple(
        eval_rows_by_pos[pos][entry.local_indices[pos]]
        for pos in range(len(entry.local_indices))
    )
    return proxy_for_indicator_rows(
        eval_rows=eval_rows,
        ret_15m=ret_15m,
        min_confirm=min_confirm,
        fee_penalty_per_confirm=fee_penalty_per_confirm,
    )


def _ranking_from_normalized(normalized_request: Mapping[str, Any]) -> _RankingSpec:
    ranking = normalized_request.get("ranking")
    if ranking is None:
        return _RankingSpec(metric_name="total_return_pct", direction="desc")
    if not isinstance(ranking, Mapping):
        raise BacktestNoRiskExactRejected("normalized_request.ranking must be a mapping")
    raw_metric = ranking.get("primary_metric", ranking.get("metric", "total_return_pct"))
    metric_name = str(raw_metric)
    if metric_name not in NO_RISK_METRIC_NAMES:
        raise BacktestNoRiskExactRejected(
            f"unsupported no-risk ranking metric {metric_name!r}; expected one of "
            f"{NO_RISK_METRIC_NAMES!r}"
        )
    direction = str(ranking.get("direction", "desc"))
    if direction not in ("asc", "desc"):
        raise BacktestNoRiskExactRejected("ranking.direction must be 'asc' or 'desc'")
    return _RankingSpec(metric_name=metric_name, direction=direction)


def _score_array_for_metric(*, buffers: _MetricBuffers, metric_name: str) -> np.ndarray:
    if metric_name == "total_return_pct":
        return buffers.total_return_pct
    if metric_name == "max_drawdown_pct":
        return buffers.max_drawdown_pct
    if metric_name == "return_over_max_drawdown":
        return buffers.return_over_max_drawdown
    if metric_name == "profit_factor":
        return buffers.profit_factor
    if metric_name == "trade_count":
        return buffers.trade_count
    if metric_name == "sharpe_trades":
        return buffers.sharpe_trades
    if metric_name == "win_rate_pct":
        return buffers.win_rate_pct
    if metric_name == "avg_trade_ret_pct":
        return buffers.avg_trade_ret_pct
    if metric_name == "avg_trade_exec_bars":
        return buffers.avg_trade_exec_bars
    if metric_name == "exposure_pct":
        return buffers.exposure_pct
    raise BacktestNoRiskExactRejected(f"unsupported no-risk ranking metric {metric_name!r}")


def _pool_by_id(prepared_result: BacktestPreparePoolsResult):
    return {pool.indicator_id: pool for pool in prepared_result.indicator_pools}


def _synthetic_combo_chunk(
    *,
    indicator_ids: tuple[str, ...],
    selected_rows_by_indicator: Mapping[str, np.ndarray],
):
    from trading.contexts.backtest.application.dto import BacktestComboChunk

    return BacktestComboChunk(
        indicator_ids=indicator_ids,
        rows_by_indicator=selected_rows_by_indicator,
    )


__all__ = [
    "BARS_PER_YEAR_EXEC_1M",
    "CANONICAL_EXECUTION_TIMEFRAME_V1",
    "DIRECTION_MODE_LONG_ONLY",
    "DIRECTION_MODE_LONG_SHORT_REVERSAL",
    "NO_RISK_EXACT_BOUNDARY_STAGE_NAME",
    "NO_RISK_EXACT_BOUNDARY_STATUS",
    "NO_RISK_EXACT_SCORED_STATUS",
    "NO_RISK_EXACT_SCORING_STAGE_NAME",
    "NO_RISK_HEAP_UPDATE_STAGE_NAME",
    "NO_RISK_METRIC_NAMES",
    "NO_RISK_SELF_CHECK_NOT_RUN_STATUS",
    "NO_RISK_SELF_CHECK_PASSED_STATUS",
    "NO_RISK_SELF_CHECK_STAGE_NAME",
    "NO_RISK_TOP_RESULT_ASSEMBLY_STAGE_NAME",
    "NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME",
    "BacktestNoRiskExactRejected",
    "BacktestNoRiskExactScoringService",
    "BacktestNoRiskSelfCheckFailed",
    "build_trade_list_for_indicator_rows_slow",
    "evaluate_no_risk_exact_chunk",
    "evaluate_no_risk_reference_rows_slow",
    "event_segments_2_no_risk",
    "event_segments_n_no_risk",
    "proxy_for_indicator_rows",
    "proxy_for_two_rows",
    "run_fast_vs_reference_self_check",
    "streaming_2_no_risk",
]
