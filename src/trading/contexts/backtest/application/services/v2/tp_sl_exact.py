from __future__ import annotations

import heapq
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, NamedTuple, cast

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    TP_SL_EXACT_METRIC_NAMES,
    BacktestComboChunk,
    BacktestComboPlanningResult,
    BacktestPreparePoolsResult,
    BacktestTpSlExactConfig,
    BacktestTpSlExactResult,
    BacktestTpSlExactTelemetry,
    BacktestTpSlExecutionContext,
    BacktestTpSlHitTimesResult,
    BacktestTpSlHitTimesSubset,
    BacktestTpSlMemoryCleanupEvidence,
    BacktestTpSlSelfCheckSummary,
    BacktestTpSlTopResult,
)
from trading.contexts.backtest.application.services.v2.combo_planning import (
    COMBO_CHUNK_SIZE,
    EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
    MATRIX_CELL_TP_SL_V1_BACKEND,
    BacktestComboPlanningService,
    build_local_row_pools,
    iter_ordinal_combo_chunks,
    make_combo_idx_matrix,
)
from trading.contexts.backtest.application.services.v2.execution_sizing import (
    DIRECTION_MODE_LONG_ONLY,
    DIRECTION_MODE_LONG_ONLY_CODE,
    DIRECTION_MODE_LONG_SHORT_REVERSAL,
    DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE,
    SIZING_MODE_ALL_IN_CODE,
    execution_quote_amount,
    execution_quote_amount_py,
    execution_settings_from_normalized,
)
from trading.contexts.backtest.application.services.v2.execution_sizing import (
    ExecutionSettings as _ExecutionSettings,
)
from trading.contexts.backtest.application.services.v2.no_risk_exact import (
    BARS_PER_YEAR_EXEC_1M,
    _pool_by_id,
    _request_top_n_from_normalized,
    _risk_mode_from_normalized,
)
from trading.contexts.backtest.application.services.v2.numba_runtime import (
    current_backtest_numba_telemetry,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    HIT_TIMES_ARTIFACT_PATH_V2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactFundingArraysV2,
)

from .no_risk_funding import (
    FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING,
    FUNDING_ADJUSTMENT_SCOPE,
    FUNDING_ADJUSTMENT_SCOPE_CANDIDATE_POOL,
    FUNDING_ADJUSTMENT_SCOPE_UNAVAILABLE,
    FUNDING_DATA_QUALITY,
    FUNDING_INCLUDED,
    FUNDING_WARNING_CODES,
    TOTAL_RETURN_PCT_NET_OF_FUNDING,
)
from .tp_sl_funding import (
    TP_SL_FUNDING_ADJUSTMENT_STAGE_NAME,
    TP_SL_FUNDING_METRIC_NAMES,
    calculate_tp_sl_funding_adjustment,
    resolve_tp_sl_selected_exit,
)

TP_SL_EXACT_BOUNDARY_STAGE_NAME = "tp_sl_exact_boundary"
TP_SL_EXACT_SCORING_STAGE_NAME = "exact_scoring"
TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME = "tp_sl_exact_scoring"
TP_SL_HEAP_UPDATE_STAGE_NAME = "heap_update"
TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME = "tp_sl_full_metrics_second_pass"
TP_SL_SELF_CHECK_STAGE_NAME = "self_check"
TP_SL_COMBO_CHUNK_DECODE_STAGE_NAME = "combo_chunk_decode"
TP_SL_PROXY_FILTER_STAGE_NAME = "proxy_filter"
TP_SL_SCORE_BUFFER_ALLOCATION_STAGE_NAME = "score_buffer_allocation"
TP_SL_TELEMETRY_BUILD_STAGE_NAME = "telemetry_build"
TP_SL_CELL_BLOCK_SHAPE_LITERAL = "16 x 16"
TP_SL_CELL_BLOCK_TP_COUNT_ENV_KEY = "ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_TP_COUNT"
TP_SL_CELL_BLOCK_SL_COUNT_ENV_KEY = "ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_SL_COUNT"
TP_SL_EXACT_SCORED_STATUS = "scored"
TP_SL_SELF_CHECK_NOT_RUN_STATUS = "not_run"
TP_SL_SELF_CHECK_PASSED_STATUS = "passed"
NEG_LARGE = -1.0e300
TP_SL_BEST_CELL_TIE_EPS = 1.0e-12
TP_SL_SELF_CHECK_BEST_CELL_TIE_TOLERANCE_PCT = 1.0e-9


class BacktestTpSlExactRejected(ValueError):
    """
    Deterministic internal rejection for unsupported TP/SL exact boundary inputs.
    """


class BacktestTpSlSelfCheckFailed(AssertionError):
    """
    Raised when fast TP/SL exact scoring diverges from the bounded slow reference.
    """


def _execution_settings_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    expected_direction_mode: str,
    config: BacktestTpSlExactConfig,
) -> _ExecutionSettings:
    return execution_settings_from_normalized(
        normalized_request,
        expected_direction_mode=expected_direction_mode,
        config=config,
        rejection_cls=BacktestTpSlExactRejected,
    )


def _min_closed_trades_from_normalized(normalized_request: Mapping[str, Any]) -> int:
    quality_constraints = normalized_request.get("quality_constraints")
    if quality_constraints is None:
        return 0
    if not isinstance(quality_constraints, Mapping):
        raise BacktestTpSlExactRejected(
            "normalized_request.quality_constraints must be a mapping when provided"
        )
    raw_min_closed_trades = quality_constraints.get("min_closed_trades")
    if raw_min_closed_trades is None:
        return 0
    if (
        isinstance(raw_min_closed_trades, bool)
        or not isinstance(raw_min_closed_trades, int)
        or raw_min_closed_trades <= 0
    ):
        raise BacktestTpSlExactRejected(
            "normalized_request.quality_constraints.min_closed_trades must be a positive integer"
        )
    return int(raw_min_closed_trades)


def _ranking_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    effective: bool,
) -> _RankingSpec:
    ranking = _mapping_payload(normalized_request.get("ranking"))
    requested_metric = str(
        ranking.get("requested_primary_metric", ranking.get("primary_metric", "total_return_pct"))
    )
    metric_name = requested_metric
    if effective:
        metric_name = str(ranking.get("effective_primary_metric", requested_metric))
        if (
            metric_name == "total_return_pct"
            and _funding_adjustment_enabled(normalized_request=normalized_request)
        ):
            metric_name = TOTAL_RETURN_PCT_NET_OF_FUNDING
    direction = str(ranking.get("direction", "desc")).lower()
    if direction not in {"asc", "desc"}:
        direction = "desc"
    return _RankingSpec(metric_name=metric_name, direction=direction)


def _funding_adjustment_enabled(*, normalized_request: Mapping[str, Any]) -> bool:
    coordinates = _mapping_payload(normalized_request.get("coordinates"))
    risk = _mapping_payload(normalized_request.get("risk"))
    execution = _mapping_payload(normalized_request.get("execution"))
    funding = _mapping_payload(execution.get("funding"))
    return (
        str(coordinates.get("market_type")) == "futures"
        and str(risk.get("mode")) == "tp_sl_grid"
        and str(funding.get("mode")) == "include_when_futures"
    )


def _funding_candidate_pool_size(requested_top_n: int) -> int:
    return max(int(requested_top_n) * 5, int(requested_top_n) + 100)


def _mapping_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(frozen=True, slots=True)
class _TpSlRuntimeContext:
    run_abs_start_15m: np.int32
    t_exec_abs_15m: np.int32
    price_open_15m: np.ndarray
    log_open_15m: np.ndarray
    last_close_15m: float
    log_last_close_15m: float
    log_fac_tp_long: np.ndarray
    log_fac_sl_long: np.ndarray
    log_fac_tp_short: np.ndarray
    log_fac_sl_short: np.ndarray
    log_fee_two_sides: float
    close_on_end: np.int8
    initial_cash_quote: float
    sizing_mode_code: np.int8
    quote_amount: float
    equity_pct: float
    min_quote: float
    max_quote: float
    safe_profit_percent: float
    use_profit_lock: np.int8


@dataclass(frozen=True, slots=True)
class _TpSlScoreBuffers:
    total_return_pct: np.ndarray
    trade_count: np.ndarray
    best_tp_idx: np.ndarray
    best_sl_idx: np.ndarray

    @property
    def size(self) -> int:
        return int(self.total_return_pct.shape[0])


@dataclass(frozen=True, slots=True)
class _SelectedCandidateBatch:
    rows_by_indicator: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class _RankingSpec:
    metric_name: str
    direction: str


@dataclass(frozen=True, slots=True)
class _TopKContext:
    indicator_ids: tuple[str, ...]
    row_ids_by_pos: tuple[np.ndarray, ...]
    metadata_by_pos: tuple[tuple[Any, ...], ...]


class _TpSlHeapEntry(NamedTuple):
    score: float
    original_rows: tuple[int, ...]
    local_indices: tuple[int, ...]
    best_tp_idx: int
    best_sl_idx: int
    best_tp_pct: float
    best_sl_pct: float
    total_return_pct: float
    trade_count: int
    candidate_ordinal: int
    metadata_by_pos: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class BacktestTpSlExactScoringService:
    """
    Internal service for artifact-backed TP/SL exact scoring and canonical top-K heap work.
    """

    config: BacktestTpSlExactConfig = BacktestTpSlExactConfig()

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        hit_times_result: BacktestTpSlHitTimesResult,
        normalized_request: Mapping[str, Any],
        funding_arrays: ArtifactFundingArraysV2 | None = None,
    ) -> BacktestTpSlExactResult:
        boundary_start = time.perf_counter()
        risk_mode = _risk_mode_from_normalized(normalized_request)
        if risk_mode != "tp_sl_grid":
            raise BacktestTpSlExactRejected(
                f"TP/SL exact boundary requires risk.mode='tp_sl_grid'; got {risk_mode!r}"
            )
        backend = combo_planning_result.backend
        if backend.backend_id not in {
            EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
            MATRIX_CELL_TP_SL_V1_BACKEND,
        }:
            raise BacktestTpSlExactRejected(
                f"TP/SL exact boundary requires backend "
                f"{EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND!r} or "
                f"{MATRIX_CELL_TP_SL_V1_BACKEND!r}; got {backend.backend_id!r}"
            )
        if backend.risk_mode != "tp_sl_grid":
            raise BacktestTpSlExactRejected(
                f"combo planning backend risk_mode must be 'tp_sl_grid'; got "
                f"{backend.risk_mode!r}"
            )
        arity = len(prepared_result.indicator_ids)
        if backend.arity != arity:
            raise BacktestTpSlExactRejected(
                f"combo planning arity {backend.arity} does not match prepared arity {arity}"
            )
        if not 1 <= arity <= 10:
            raise BacktestTpSlExactRejected(f"TP/SL exact arity must be 1..10; got {arity}")
        execution_settings = _execution_settings_from_normalized(
            normalized_request,
            expected_direction_mode=backend.direction_mode,
            config=cast(Any, self.config),
        )
        hit_times = hit_times_result.hit_times
        _validate_hit_times_for_prepared(hit_times=hit_times, prepared_result=prepared_result)
        runtime = _tp_sl_runtime_context_from_prepared(
            prepared_result=prepared_result,
            hit_times=hit_times,
            execution_settings=execution_settings,
        )
        request_top_n = _request_top_n_from_normalized(
            normalized_request,
            default_request_top_n=self.config.default_request_top_n,
        )
        requested_ranking = _ranking_from_normalized(normalized_request, effective=False)
        effective_ranking = _ranking_from_normalized(normalized_request, effective=True)
        funding_adjustment_enabled = _funding_adjustment_enabled(
            normalized_request=normalized_request
        )
        candidate_pool_top_k = (
            _funding_candidate_pool_size(request_top_n)
            if funding_adjustment_enabled
            else request_top_n
        )
        min_closed_trades = _min_closed_trades_from_normalized(normalized_request)
        top_k_context = _top_k_context_from_prepared(prepared_result)
        backend_logical_name = _backend_logical_name(arity=arity)
        cell_block_tp_count, cell_block_sl_count = _tp_sl_cell_block_counts_from_env(
            self.config
        )
        stage_timings = {TP_SL_EXACT_BOUNDARY_STAGE_NAME: time.perf_counter() - boundary_start}
        execution_context = BacktestTpSlExecutionContext(
            timeframe=prepared_result.timeframe,
            hit_times_path=HIT_TIMES_ARTIFACT_PATH_V2,
            time_slice_start_15m=prepared_result.time_slice_start_15m,
            time_slice_stop_15m=prepared_result.time_slice_stop_15m,
            trade_T_length=prepared_result.trade_T_length,
            eval_T_length=prepared_result.eval_T_length,
            sentinel_index=hit_times.sentinel_index,
        )
        self_check_summary = BacktestTpSlSelfCheckSummary(
            enabled=self.config.run_self_check,
            status=TP_SL_SELF_CHECK_NOT_RUN_STATUS,
            backend_logical_name=backend_logical_name,
            backend_implementation_id=backend.backend_id,
            direction_mode=backend.direction_mode,
        )

        top_results: tuple[BacktestTpSlTopResult, ...] | None = None
        telemetry: BacktestTpSlExactTelemetry | None = None
        selected_batches_iter: Iterator[_SelectedCandidateBatch] | None = None
        first_selected_batch: _SelectedCandidateBatch | None = None
        selected_batch: _SelectedCandidateBatch | None = None
        heap: list[tuple[tuple[float, float, float, int, tuple[int, ...]], _TpSlHeapEntry]]
        heap = []
        cleanup_duration_s = 0.0
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
                    hit_times=hit_times,
                    runtime=runtime,
                    backend_logical_name=backend_logical_name,
                )
                stage_timings[TP_SL_SELF_CHECK_STAGE_NAME] = time.perf_counter() - check_start

            stage_timings[TP_SL_EXACT_SCORING_STAGE_NAME] = 0.0
            stage_timings[TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME] = 0.0
            stage_timings[TP_SL_HEAP_UPDATE_STAGE_NAME] = 0.0
            scored_count = 0
            below_min_trades_count = 0
            heap_eligible_count = 0
            sample_metrics: Mapping[str, float] | None = None
            candidate_ordinal_start = 0
            if first_selected_batch is not None:
                (
                    scored,
                    sample_metrics,
                    below_min_trades,
                    heap_eligible,
                ) = self._score_selected_rows(
                    selected_batch=first_selected_batch,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    hit_times=hit_times,
                    runtime=runtime,
                    stage_timings=stage_timings,
                    sample_metrics=sample_metrics,
                    heap=heap,
                    top_k_context=top_k_context,
                    top_k=candidate_pool_top_k,
                    candidate_ordinal_start=candidate_ordinal_start,
                    min_closed_trades=min_closed_trades,
                    cell_block_tp_count=cell_block_tp_count,
                    cell_block_sl_count=cell_block_sl_count,
                )
                scored_count += scored
                below_min_trades_count += below_min_trades
                heap_eligible_count += heap_eligible
                candidate_ordinal_start += scored
            assert selected_batches_iter is not None
            for selected_batch in selected_batches_iter:
                (
                    scored,
                    sample_metrics,
                    below_min_trades,
                    heap_eligible,
                ) = self._score_selected_rows(
                    selected_batch=selected_batch,
                    prepared_result=prepared_result,
                    combo_planning_result=combo_planning_result,
                    hit_times=hit_times,
                    runtime=runtime,
                    stage_timings=stage_timings,
                    sample_metrics=sample_metrics,
                    heap=heap,
                    top_k_context=top_k_context,
                    top_k=candidate_pool_top_k,
                    candidate_ordinal_start=candidate_ordinal_start,
                    min_closed_trades=min_closed_trades,
                    cell_block_tp_count=cell_block_tp_count,
                    cell_block_sl_count=cell_block_sl_count,
                )
                scored_count += scored
                below_min_trades_count += below_min_trades
                heap_eligible_count += heap_eligible
                candidate_ordinal_start += scored

            metrics_start = time.perf_counter()
            top_results = _top_results_from_heap(
                heap,
                prepared_result=prepared_result,
                hit_times=hit_times,
                runtime=runtime,
                top_k_context=top_k_context,
                direction_mode=backend.direction_mode,
            )
            if funding_adjustment_enabled:
                funding_start = time.perf_counter()
                top_results = _apply_funding_adjustment_to_top_results(
                    top_results=top_results,
                    prepared_result=prepared_result,
                    hit_times=hit_times,
                    runtime=runtime,
                    direction_mode=backend.direction_mode,
                    funding_arrays=funding_arrays,
                    requested_top_n=request_top_n,
                    requested_ranking=requested_ranking,
                    effective_ranking=effective_ranking,
                )
                stage_timings[TP_SL_FUNDING_ADJUSTMENT_STAGE_NAME] = (
                    time.perf_counter() - funding_start
                )
            stage_timings[TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME] = (
                time.perf_counter() - metrics_start
            )
            telemetry_start = time.perf_counter()
            numba_telemetry = current_backtest_numba_telemetry()
            _add_timing(
                stage_timings=profile_stage_timings,
                key=TP_SL_TELEMETRY_BUILD_STAGE_NAME,
                elapsed=time.perf_counter() - telemetry_start,
            )
            telemetry = BacktestTpSlExactTelemetry(
                stage_timings=stage_timings,
                request_top_n=request_top_n,
                benchmark_top_k=self.config.benchmark_top_k,
                heap_capacity=candidate_pool_top_k,
                top_results_count=len(top_results),
                exact_candidates_evaluated=scored_count,
                risk_mode=risk_mode,
                direction_mode=backend.direction_mode,
                backend_id=backend.backend_id,
                arity=arity,
                status=TP_SL_EXACT_SCORED_STATUS,
                backend_logical_name=backend_logical_name,
                backend_implementation_id=backend.backend_id,
                metric_names=TP_SL_EXACT_METRIC_NAMES
                + (TP_SL_FUNDING_METRIC_NAMES if funding_adjustment_enabled else ()),
                sample_metrics=sample_metrics,
                numba_num_threads=int(numba_telemetry["numba_num_threads"]),
                numba_thread_source=str(numba_telemetry["numba_thread_source"]),
                min_closed_trades=min_closed_trades,
                quality_candidates_below_min_trades=below_min_trades_count,
                quality_candidates_heap_eligible=heap_eligible_count,
                cell_backend=_cell_backend_telemetry(
                    backend_id=backend.backend_id,
                    scored_count=scored_count,
                    tp_count=int(hit_times.tp_values.shape[0]),
                    sl_count=int(hit_times.sl_values.shape[0]),
                    tp_block_count=cell_block_tp_count,
                    sl_block_count=cell_block_sl_count,
                    exact_scoring_s=stage_timings[TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME],
                ),
            )
        finally:
            cleanup_start = time.perf_counter()
            heap.clear()
            selected_batch = None
            first_selected_batch = None
            selected_batches_iter = None
            del selected_batch
            del first_selected_batch
            del selected_batches_iter
            del runtime
            del top_k_context
            del prepared_result
            del combo_planning_result
            del hit_times_result
            del normalized_request
            cleanup_duration_s = time.perf_counter() - cleanup_start

        if top_results is None or telemetry is None:
            raise RuntimeError("TP/SL exact scoring did not produce a compact result")
        return BacktestTpSlExactResult(
            execution_context=execution_context,
            top_results=top_results,
            telemetry=telemetry,
            self_check=self_check_summary,
            memory_cleanup_evidence=BacktestTpSlMemoryCleanupEvidence(
                checked_reference_names=(
                    "prepared_result",
                    "combo_planning_result",
                    "hit_times_subset",
                    "tp_sl_diff_buffers",
                    "score_arrays",
                    "selected_top_rows",
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
        hit_times: BacktestTpSlHitTimesSubset,
        runtime: _TpSlRuntimeContext,
        stage_timings: dict[str, float],
        sample_metrics: Mapping[str, float] | None,
        heap: list[tuple[tuple[float, float, float, int, tuple[int, ...]], _TpSlHeapEntry]],
        top_k_context: _TopKContext,
        top_k: int,
        candidate_ordinal_start: int,
        min_closed_trades: int,
        cell_block_tp_count: int,
        cell_block_sl_count: int,
    ) -> tuple[int, Mapping[str, float] | None, int, int]:
        selected_rows_by_indicator = selected_batch.rows_by_indicator
        selected_size = _selected_size(selected_rows_by_indicator)
        if selected_size <= 0:
            return 0, sample_metrics, 0, 0
        allocation_start = time.perf_counter()
        buffers = _allocate_score_buffers(selected_size)
        _add_timing(
            stage_timings=stage_timings if _exact_profile_enabled() else None,
            key=TP_SL_SCORE_BUFFER_ALLOCATION_STAGE_NAME,
            elapsed=time.perf_counter() - allocation_start,
        )
        exact_start = time.perf_counter()
        evaluate_tp_sl_exact_chunk(
            selected_rows_by_indicator=selected_rows_by_indicator,
            prepared_result=prepared_result,
            combo_planning_result=combo_planning_result,
            hit_times=hit_times,
            runtime=runtime,
            buffers=buffers,
            cell_block_tp_count=cell_block_tp_count,
            cell_block_sl_count=cell_block_sl_count,
        )
        elapsed = time.perf_counter() - exact_start
        stage_timings[TP_SL_EXACT_SCORING_STAGE_NAME] += elapsed
        stage_timings[TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME] += elapsed
        below_min_trades = int(np.count_nonzero(buffers.trade_count < min_closed_trades))
        heap_eligible = buffers.size - below_min_trades
        if sample_metrics is None:
            sample_index = _first_quality_eligible_index(
                trade_count=buffers.trade_count,
                min_closed_trades=min_closed_trades,
            )
        else:
            sample_index = None
        if sample_index is not None:
            sample_metrics = _sample_metrics_at(
                buffers=buffers,
                hit_times=hit_times,
                index=sample_index,
            )
        heap_start = time.perf_counter()
        _update_heap_total_return_desc(
            heap=heap,
            top_k_context=top_k_context,
            selected_rows_by_indicator=selected_rows_by_indicator,
            hit_times=hit_times,
            buffers=buffers,
            top_k=top_k,
            candidate_ordinal_start=candidate_ordinal_start,
            min_closed_trades=min_closed_trades,
        )
        stage_timings[TP_SL_HEAP_UPDATE_STAGE_NAME] += time.perf_counter() - heap_start
        return buffers.size, sample_metrics, below_min_trades, heap_eligible

    def _run_self_check(
        self,
        *,
        selected_rows_by_indicator: Mapping[str, np.ndarray] | None,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        hit_times: BacktestTpSlHitTimesSubset,
        runtime: _TpSlRuntimeContext,
        backend_logical_name: str,
    ) -> BacktestTpSlSelfCheckSummary:
        if selected_rows_by_indicator is None:
            return BacktestTpSlSelfCheckSummary(
                enabled=True,
                status=TP_SL_SELF_CHECK_PASSED_STATUS,
                sample_size=0,
                mismatches=0,
                max_abs_return_diff=0.0,
                backend_logical_name=backend_logical_name,
                backend_implementation_id=combo_planning_result.backend.backend_id,
                direction_mode=combo_planning_result.backend.direction_mode,
                trade_count_equal=True,
                best_cell_equal=True,
                valid_tp_sl_indexes=True,
                return_tolerance=self.config.self_check_return_tolerance,
            )
        return run_tp_sl_fast_vs_reference_self_check(
            selected_rows_by_indicator=selected_rows_by_indicator,
            prepared_result=prepared_result,
            combo_planning_result=combo_planning_result,
            hit_times=hit_times,
            runtime=runtime,
            backend_logical_name=backend_logical_name,
            check_n=self.config.self_check_sample_size,
            return_tolerance=self.config.self_check_return_tolerance,
        )


def evaluate_tp_sl_exact_chunk(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    buffers: _TpSlScoreBuffers,
    cell_block_tp_count: int = 16,
    cell_block_sl_count: int = 16,
) -> None:
    exact_context = combo_planning_result.exact_context
    if (
        exact_context.starts is None
        or exact_context.ends is None
        or exact_context.values is None
        or exact_context.counts is None
    ):
        raise BacktestTpSlExactRejected(
            "event_segments_n_tp_sl_15m_grid requires a materialized exact context"
        )
    combo_idx_by_indicator = make_combo_idx_matrix(
        combo_chunk=BacktestComboChunk(
            indicator_ids=tuple(prepared_result.indicator_ids),
            rows_by_indicator=selected_rows_by_indicator,
        ),
        indicator_ids=tuple(prepared_result.indicator_ids),
    )
    segment_pos_workspace = np.empty(
        (combo_idx_by_indicator.shape[1], combo_idx_by_indicator.shape[0]),
        dtype=np.int32,
    )
    best_ret = np.empty(buffers.size, dtype=np.float32)
    if (
        runtime.sizing_mode_code == SIZING_MODE_ALL_IN_CODE
        and runtime.use_profit_lock == 0
        and runtime.close_on_end == 1
    ):
        if combo_planning_result.backend.backend_id == MATRIX_CELL_TP_SL_V1_BACKEND:
            event_segments_n_tp_sl_15m_grid_cell_blocks(
                combo_idx_by_indicator,
                exact_context.starts,
                exact_context.ends,
                exact_context.values,
                exact_context.counts,
                segment_pos_workspace,
                runtime.run_abs_start_15m,
                runtime.t_exec_abs_15m,
                runtime.price_open_15m,
                runtime.log_open_15m,
                runtime.last_close_15m,
                runtime.log_last_close_15m,
                hit_times.long_tp,
                hit_times.long_sl,
                hit_times.short_tp,
                hit_times.short_sl,
                runtime.log_fac_tp_long,
                runtime.log_fac_sl_long,
                runtime.log_fac_tp_short,
                runtime.log_fac_sl_short,
                runtime.log_fee_two_sides,
                runtime.close_on_end,
                _direction_mode_code(combo_planning_result.backend.direction_mode),
                np.int32(cell_block_tp_count),
                np.int32(cell_block_sl_count),
                buffers.best_tp_idx,
                buffers.best_sl_idx,
                best_ret,
                buffers.trade_count,
            )
            buffers.total_return_pct[:] = best_ret.astype(np.float64) * 100.0
            return
        if combo_planning_result.backend.backend_id not in {
            EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
        }:
            raise BacktestTpSlExactRejected(
                f"unsupported TP/SL exact backend: "
                f"{combo_planning_result.backend.backend_id!r}"
            )
        event_segments_n_tp_sl_15m_grid(
            combo_idx_by_indicator,
            exact_context.starts,
            exact_context.ends,
            exact_context.values,
            exact_context.counts,
            segment_pos_workspace,
            runtime.run_abs_start_15m,
            runtime.t_exec_abs_15m,
            runtime.price_open_15m,
            runtime.log_open_15m,
            runtime.last_close_15m,
            runtime.log_last_close_15m,
            hit_times.long_tp,
            hit_times.long_sl,
            hit_times.short_tp,
            hit_times.short_sl,
            runtime.log_fac_tp_long,
            runtime.log_fac_sl_long,
            runtime.log_fac_tp_short,
            runtime.log_fac_sl_short,
            runtime.log_fee_two_sides,
            runtime.close_on_end,
            _direction_mode_code(combo_planning_result.backend.direction_mode),
            buffers.best_tp_idx,
            buffers.best_sl_idx,
            best_ret,
            buffers.trade_count,
        )
    else:
        if combo_planning_result.backend.backend_id == MATRIX_CELL_TP_SL_V1_BACKEND:
            raise BacktestTpSlExactRejected(
                "matrix_cell_tp_sl_v1 currently supports all_in sizing, "
                "profit_lock=false, and close_on_end=true only"
            )
        event_segments_n_tp_sl_15m_grid_execution_sizing(
            combo_idx_by_indicator,
            exact_context.starts,
            exact_context.ends,
            exact_context.values,
            exact_context.counts,
            segment_pos_workspace,
            runtime.run_abs_start_15m,
            runtime.t_exec_abs_15m,
            runtime.price_open_15m,
            runtime.last_close_15m,
            hit_times.long_tp,
            hit_times.long_sl,
            hit_times.short_tp,
            hit_times.short_sl,
            runtime.log_fac_tp_long,
            runtime.log_fac_sl_long,
            runtime.log_fac_tp_short,
            runtime.log_fac_sl_short,
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
            _direction_mode_code(combo_planning_result.backend.direction_mode),
            buffers.best_tp_idx,
            buffers.best_sl_idx,
            best_ret,
            buffers.trade_count,
        )
    buffers.total_return_pct[:] = best_ret.astype(np.float64) * 100.0


def run_tp_sl_fast_vs_reference_self_check(
    *,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    prepared_result: BacktestPreparePoolsResult,
    combo_planning_result: BacktestComboPlanningResult,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    backend_logical_name: str,
    check_n: int,
    return_tolerance: float,
) -> BacktestTpSlSelfCheckSummary:
    if check_n < 0:
        raise BacktestTpSlExactRejected("self_check_sample_size must be >= 0")
    n_check = min(int(check_n), _selected_size(selected_rows_by_indicator))
    if n_check <= 0:
        return BacktestTpSlSelfCheckSummary(
            enabled=True,
            status=TP_SL_SELF_CHECK_PASSED_STATUS,
            sample_size=0,
            mismatches=0,
            max_abs_return_diff=0.0,
            backend_logical_name=backend_logical_name,
            backend_implementation_id=combo_planning_result.backend.backend_id,
            direction_mode=combo_planning_result.backend.direction_mode,
            trade_count_equal=True,
            best_cell_equal=True,
            valid_tp_sl_indexes=True,
            return_tolerance=return_tolerance,
        )
    subset = {
        indicator_id: np.ascontiguousarray(rows[:n_check])
        for indicator_id, rows in selected_rows_by_indicator.items()
    }
    buffers = _allocate_score_buffers(n_check)
    evaluate_tp_sl_exact_chunk(
        selected_rows_by_indicator=subset,
        prepared_result=prepared_result,
        combo_planning_result=combo_planning_result,
        hit_times=hit_times,
        runtime=runtime,
        buffers=buffers,
    )
    max_abs_diff = 0.0
    trade_count_equal = True
    best_cell_equal = True
    valid_tp_sl_indexes = True
    mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for row_idx in range(n_check):
        fast_tp = int(buffers.best_tp_idx[row_idx])
        fast_sl = int(buffers.best_sl_idx[row_idx])
        if (
            fast_tp < 0
            or fast_tp >= int(hit_times.tp_values.shape[0])
            or fast_sl < 0
            or fast_sl >= int(hit_times.sl_values.shape[0])
        ):
            valid_tp_sl_indexes = False
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row_idx": row_idx,
                    "reason": "invalid_best_cell",
                    "fast_best_tp_idx": fast_tp,
                    "fast_best_sl_idx": fast_sl,
                }
            continue
        local_indices = tuple(
            int(subset[indicator_id][row_idx])
            for indicator_id in prepared_result.indicator_ids
        )
        reference = evaluate_tp_sl_reference_rows_slow(
            prepared_result=prepared_result,
            local_indices=local_indices,
            hit_times=hit_times,
            runtime=runtime,
            direction_mode=combo_planning_result.backend.direction_mode,
        )
        if int(reference["trade_count"]) != int(buffers.trade_count[row_idx]):
            trade_count_equal = False
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row_idx": row_idx,
                    "reason": "trade_count",
                    "reference": int(reference["trade_count"]),
                    "fast": int(buffers.trade_count[row_idx]),
                }
            continue
        abs_diff = abs(
            float(reference["total_return_pct"]) - float(buffers.total_return_pct[row_idx])
        )
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
        same_cell = int(reference["best_tp_idx"]) == fast_tp and int(
            reference["best_sl_idx"]
        ) == fast_sl
        if (
            not same_cell
            and abs_diff > TP_SL_SELF_CHECK_BEST_CELL_TIE_TOLERANCE_PCT
        ):
            best_cell_equal = False
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row_idx": row_idx,
                    "reason": "best_cell",
                    "reference_best_tp_idx": int(reference["best_tp_idx"]),
                    "reference_best_sl_idx": int(reference["best_sl_idx"]),
                    "fast_best_tp_idx": fast_tp,
                    "fast_best_sl_idx": fast_sl,
                }
        if abs_diff > return_tolerance:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row_idx": row_idx,
                    "reason": "return",
                    "reference": float(reference["total_return_pct"]),
                    "fast": float(buffers.total_return_pct[row_idx]),
                    "abs_diff": abs_diff,
                }

    if (
        mismatches > 0
        or not trade_count_equal
        or not best_cell_equal
        or not valid_tp_sl_indexes
    ):
        raise BacktestTpSlSelfCheckFailed(
            "TP/SL exact self-check failed: "
            f"mismatches={mismatches}, trade_count_equal={trade_count_equal}, "
            f"best_cell_equal={best_cell_equal}, "
            f"valid_tp_sl_indexes={valid_tp_sl_indexes}, max_abs_diff={max_abs_diff}, "
            f"tolerance={return_tolerance}, first_mismatch={first_mismatch}"
        )
    return BacktestTpSlSelfCheckSummary(
        enabled=True,
        status=TP_SL_SELF_CHECK_PASSED_STATUS,
        sample_size=n_check,
        mismatches=0,
        max_abs_return_diff=max_abs_diff,
        backend_logical_name=backend_logical_name,
        backend_implementation_id=combo_planning_result.backend.backend_id,
        direction_mode=combo_planning_result.backend.direction_mode,
        trade_count_equal=True,
        best_cell_equal=best_cell_equal,
        valid_tp_sl_indexes=True,
        return_tolerance=return_tolerance,
        first_mismatch=None,
    )


@nb.njit(cache=True, inline="always")
def _tp_sl_add_row_range(
    row_diff: np.ndarray,
    row_i: np.int32,
    col_start: np.int32,
    col_stop: np.int32,
    value: float,
) -> None:
    if col_start < col_stop:
        row_diff[row_i, col_start] += value
        row_diff[row_i, col_stop] -= value


@nb.njit(cache=True, inline="always")
def _tp_sl_add_col_range(
    col_diff: np.ndarray,
    row_start: np.int32,
    row_stop: np.int32,
    col_i: np.int32,
    value: float,
) -> None:
    if row_start < row_stop:
        col_diff[row_start, col_i] += value
        col_diff[row_stop, col_i] -= value


@nb.njit(cache=True, inline="always")
def _tp_sl_add_rect(
    rect_diff: np.ndarray,
    row_start: np.int32,
    col_start: np.int32,
    row_stop: np.int32,
    col_stop: np.int32,
    value: float,
) -> None:
    if row_start < row_stop and col_start < col_stop:
        rect_diff[row_start, col_start] += value
        rect_diff[row_stop, col_start] -= value
        rect_diff[row_start, col_stop] -= value
        rect_diff[row_stop, col_stop] += value


@nb.njit(cache=True, inline="always")
def _tp_sl_lower_bound_ge_hit(
    hit_table: np.ndarray,
    start_exec: np.int32,
    n_levels: np.int32,
    target: np.int32,
) -> np.int32:
    lo = np.int32(0)
    hi = np.int32(n_levels)
    while lo < hi:
        mid = np.int32((lo + hi) // 2)
        if np.int32(hit_table[mid, start_exec]) >= target:
            hi = mid
        else:
            lo = np.int32(mid + 1)
    return lo


@nb.njit(cache=True, inline="always")
def _tp_sl_signal_exit_log_contrib(
    dirn: np.int8,
    entry_abs: np.int32,
    exit_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    log_fee: float,
) -> float:
    entry_open = float(price_open[entry_abs])
    exit_open = float(price_open[exit_abs])
    if entry_open <= 0.0 or exit_open <= 0.0:
        return NEG_LARGE
    entry_log = float(log_open[entry_abs])
    contrib = log_fee
    if dirn == 1:
        contrib += float(log_open[exit_abs]) - entry_log
    else:
        ratio = exit_open / entry_open
        if ratio >= 2.0:
            return NEG_LARGE
        contrib += math.log(2.0 - ratio)
    return contrib


@nb.njit(cache=True, inline="always")
def _tp_sl_final_close_log_contrib(
    dirn: np.int8,
    entry_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    last_close: float,
    log_last_close: float,
    log_fee: float,
) -> float:
    entry_open = float(price_open[entry_abs])
    if entry_open <= 0.0 or last_close <= 0.0:
        return NEG_LARGE
    entry_log = float(log_open[entry_abs])
    contrib = log_fee
    if dirn == 1:
        contrib += log_last_close - entry_log
    else:
        ratio = last_close / entry_open
        if ratio >= 2.0:
            return NEG_LARGE
        contrib += math.log(2.0 - ratio)
    return contrib


@nb.njit(cache=True, inline="always")
def _tp_sl_apply_trade_to_diff(
    dirn: np.int8,
    entry_abs: np.int32,
    sig_exit_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    last_close: float,
    log_last_close: float,
    t_exec_abs: np.int32,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    row_diff: np.ndarray,
    col_diff: np.ndarray,
    rect_diff: np.ndarray,
) -> None:
    n_tp = np.int32(hit_long_tp.shape[0])
    n_sl = np.int32(hit_long_sl.shape[0])
    if float(price_open[entry_abs]) <= 0.0:
        return
    start = np.int32(entry_abs + 1)
    if dirn == 1:
        hit_tp = hit_long_tp
        hit_sl = hit_long_sl
        log_tp_arr = log_fac_tp_long
        log_sl_arr = log_fac_sl_long
    else:
        hit_tp = hit_short_tp
        hit_sl = hit_short_sl
        log_tp_arr = log_fac_tp_short
        log_sl_arr = log_fac_sl_short

    if start >= t_exec_abs:
        if sig_exit_abs < t_exec_abs:
            contrib = _tp_sl_signal_exit_log_contrib(
                dirn,
                entry_abs,
                sig_exit_abs,
                price_open,
                log_open,
                log_fee_two_sides,
            )
            _tp_sl_add_rect(rect_diff, np.int32(0), np.int32(0), n_tp, n_sl, contrib)
        elif close_on_end == 1 and t_exec_abs > 0:
            contrib = _tp_sl_final_close_log_contrib(
                dirn,
                entry_abs,
                price_open,
                log_open,
                last_close,
                log_last_close,
                log_fee_two_sides,
            )
            _tp_sl_add_rect(rect_diff, np.int32(0), np.int32(0), n_tp, n_sl, contrib)
        return

    if sig_exit_abs < t_exec_abs:
        i_sig = _tp_sl_lower_bound_ge_hit(hit_tp, start, n_tp, sig_exit_abs)
        j_sig = _tp_sl_lower_bound_ge_hit(hit_sl, start, n_sl, sig_exit_abs)
        contrib = _tp_sl_signal_exit_log_contrib(
            dirn,
            entry_abs,
            sig_exit_abs,
            price_open,
            log_open,
            log_fee_two_sides,
        )
        _tp_sl_add_rect(rect_diff, i_sig, j_sig, n_tp, n_sl, contrib)

        j_ptr = np.int32(0)
        for i in range(i_sig):
            t_tp = np.int32(hit_tp[i, start])
            while j_ptr < j_sig and np.int32(hit_sl[j_ptr, start]) <= t_tp:
                j_ptr = np.int32(j_ptr + 1)
            _tp_sl_add_row_range(row_diff, np.int32(i), j_ptr, n_sl, float(log_tp_arr[i]))

        i_ptr = np.int32(0)
        for j in range(j_sig):
            t_sl = np.int32(hit_sl[j, start])
            while i_ptr < i_sig and np.int32(hit_tp[i_ptr, start]) < t_sl:
                i_ptr = np.int32(i_ptr + 1)
            _tp_sl_add_col_range(col_diff, i_ptr, n_tp, np.int32(j), float(log_sl_arr[j]))
    else:
        i_end = _tp_sl_lower_bound_ge_hit(hit_tp, start, n_tp, t_exec_abs)
        j_end = _tp_sl_lower_bound_ge_hit(hit_sl, start, n_sl, t_exec_abs)
        if close_on_end == 1 and t_exec_abs > 0:
            contrib = _tp_sl_final_close_log_contrib(
                dirn,
                entry_abs,
                price_open,
                log_open,
                last_close,
                log_last_close,
                log_fee_two_sides,
            )
            _tp_sl_add_rect(rect_diff, i_end, j_end, n_tp, n_sl, contrib)

        j_ptr = np.int32(0)
        for i in range(i_end):
            t_tp = np.int32(hit_tp[i, start])
            while j_ptr < n_sl and np.int32(hit_sl[j_ptr, start]) <= t_tp:
                j_ptr = np.int32(j_ptr + 1)
            _tp_sl_add_row_range(row_diff, np.int32(i), j_ptr, n_sl, float(log_tp_arr[i]))

        i_ptr = np.int32(0)
        for j in range(j_end):
            t_sl = np.int32(hit_sl[j, start])
            while i_ptr < i_end and np.int32(hit_tp[i_ptr, start]) < t_sl:
                i_ptr = np.int32(i_ptr + 1)
            _tp_sl_add_col_range(col_diff, i_ptr, n_tp, np.int32(j), float(log_sl_arr[j]))


@nb.njit(cache=True, inline="always")
def _tp_sl_add_row_range_block(
    row_diff: np.ndarray,
    row_i: np.int32,
    col_start: np.int32,
    col_stop: np.int32,
    value: float,
    tp_block_start: np.int32,
    sl_block_start: np.int32,
) -> None:
    local_i = np.int32(row_i - tp_block_start)
    if local_i < 0 or local_i >= row_diff.shape[0]:
        return
    block_sl_stop = np.int32(sl_block_start + row_diff.shape[1] - 1)
    clipped_start = col_start if col_start > sl_block_start else sl_block_start
    clipped_stop = col_stop if col_stop < block_sl_stop else block_sl_stop
    if clipped_start < clipped_stop:
        local_start = np.int32(clipped_start - sl_block_start)
        local_stop = np.int32(clipped_stop - sl_block_start)
        row_diff[local_i, local_start] += value
        row_diff[local_i, local_stop] -= value


@nb.njit(cache=True, inline="always")
def _tp_sl_add_col_range_block(
    col_diff: np.ndarray,
    row_start: np.int32,
    row_stop: np.int32,
    col_i: np.int32,
    value: float,
    tp_block_start: np.int32,
    sl_block_start: np.int32,
) -> None:
    local_j = np.int32(col_i - sl_block_start)
    if local_j < 0 or local_j >= col_diff.shape[1]:
        return
    block_tp_stop = np.int32(tp_block_start + col_diff.shape[0] - 1)
    clipped_start = row_start if row_start > tp_block_start else tp_block_start
    clipped_stop = row_stop if row_stop < block_tp_stop else block_tp_stop
    if clipped_start < clipped_stop:
        local_start = np.int32(clipped_start - tp_block_start)
        local_stop = np.int32(clipped_stop - tp_block_start)
        col_diff[local_start, local_j] += value
        col_diff[local_stop, local_j] -= value


@nb.njit(cache=True, inline="always")
def _tp_sl_add_rect_block(
    rect_diff: np.ndarray,
    row_start: np.int32,
    col_start: np.int32,
    row_stop: np.int32,
    col_stop: np.int32,
    value: float,
    tp_block_start: np.int32,
    sl_block_start: np.int32,
) -> None:
    block_tp_stop = np.int32(tp_block_start + rect_diff.shape[0] - 1)
    block_sl_stop = np.int32(sl_block_start + rect_diff.shape[1] - 1)
    clipped_row_start = row_start if row_start > tp_block_start else tp_block_start
    clipped_col_start = col_start if col_start > sl_block_start else sl_block_start
    clipped_row_stop = row_stop if row_stop < block_tp_stop else block_tp_stop
    clipped_col_stop = col_stop if col_stop < block_sl_stop else block_sl_stop
    if clipped_row_start < clipped_row_stop and clipped_col_start < clipped_col_stop:
        local_row_start = np.int32(clipped_row_start - tp_block_start)
        local_col_start = np.int32(clipped_col_start - sl_block_start)
        local_row_stop = np.int32(clipped_row_stop - tp_block_start)
        local_col_stop = np.int32(clipped_col_stop - sl_block_start)
        rect_diff[local_row_start, local_col_start] += value
        rect_diff[local_row_stop, local_col_start] -= value
        rect_diff[local_row_start, local_col_stop] -= value
        rect_diff[local_row_stop, local_col_stop] += value


@nb.njit(cache=True, inline="always")
def _tp_sl_apply_trade_to_block_diff(
    dirn: np.int8,
    entry_abs: np.int32,
    sig_exit_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    last_close: float,
    log_last_close: float,
    t_exec_abs: np.int32,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    row_diff: np.ndarray,
    col_diff: np.ndarray,
    rect_diff: np.ndarray,
    tp_block_start: np.int32,
    sl_block_start: np.int32,
) -> None:
    n_tp = np.int32(hit_long_tp.shape[0])
    n_sl = np.int32(hit_long_sl.shape[0])
    if float(price_open[entry_abs]) <= 0.0:
        return
    start = np.int32(entry_abs + 1)
    if dirn == 1:
        hit_tp = hit_long_tp
        hit_sl = hit_long_sl
        log_tp_arr = log_fac_tp_long
        log_sl_arr = log_fac_sl_long
    else:
        hit_tp = hit_short_tp
        hit_sl = hit_short_sl
        log_tp_arr = log_fac_tp_short
        log_sl_arr = log_fac_sl_short

    if start >= t_exec_abs:
        if sig_exit_abs < t_exec_abs:
            contrib = _tp_sl_signal_exit_log_contrib(
                dirn,
                entry_abs,
                sig_exit_abs,
                price_open,
                log_open,
                log_fee_two_sides,
            )
            _tp_sl_add_rect_block(
                rect_diff,
                np.int32(0),
                np.int32(0),
                n_tp,
                n_sl,
                contrib,
                tp_block_start,
                sl_block_start,
            )
        elif close_on_end == 1 and t_exec_abs > 0:
            contrib = _tp_sl_final_close_log_contrib(
                dirn,
                entry_abs,
                price_open,
                log_open,
                last_close,
                log_last_close,
                log_fee_two_sides,
            )
            _tp_sl_add_rect_block(
                rect_diff,
                np.int32(0),
                np.int32(0),
                n_tp,
                n_sl,
                contrib,
                tp_block_start,
                sl_block_start,
            )
        return

    if sig_exit_abs < t_exec_abs:
        i_sig = _tp_sl_lower_bound_ge_hit(hit_tp, start, n_tp, sig_exit_abs)
        j_sig = _tp_sl_lower_bound_ge_hit(hit_sl, start, n_sl, sig_exit_abs)
        contrib = _tp_sl_signal_exit_log_contrib(
            dirn,
            entry_abs,
            sig_exit_abs,
            price_open,
            log_open,
            log_fee_two_sides,
        )
        _tp_sl_add_rect_block(
            rect_diff,
            i_sig,
            j_sig,
            n_tp,
            n_sl,
            contrib,
            tp_block_start,
            sl_block_start,
        )

        j_ptr = np.int32(0)
        for i in range(i_sig):
            t_tp = np.int32(hit_tp[i, start])
            while j_ptr < j_sig and np.int32(hit_sl[j_ptr, start]) <= t_tp:
                j_ptr = np.int32(j_ptr + 1)
            _tp_sl_add_row_range_block(
                row_diff,
                np.int32(i),
                j_ptr,
                n_sl,
                float(log_tp_arr[i]),
                tp_block_start,
                sl_block_start,
            )

        i_ptr = np.int32(0)
        for j in range(j_sig):
            t_sl = np.int32(hit_sl[j, start])
            while i_ptr < i_sig and np.int32(hit_tp[i_ptr, start]) < t_sl:
                i_ptr = np.int32(i_ptr + 1)
            _tp_sl_add_col_range_block(
                col_diff,
                i_ptr,
                n_tp,
                np.int32(j),
                float(log_sl_arr[j]),
                tp_block_start,
                sl_block_start,
            )
    else:
        i_end = _tp_sl_lower_bound_ge_hit(hit_tp, start, n_tp, t_exec_abs)
        j_end = _tp_sl_lower_bound_ge_hit(hit_sl, start, n_sl, t_exec_abs)
        if close_on_end == 1 and t_exec_abs > 0:
            contrib = _tp_sl_final_close_log_contrib(
                dirn,
                entry_abs,
                price_open,
                log_open,
                last_close,
                log_last_close,
                log_fee_two_sides,
            )
            _tp_sl_add_rect_block(
                rect_diff,
                i_end,
                j_end,
                n_tp,
                n_sl,
                contrib,
                tp_block_start,
                sl_block_start,
            )

        j_ptr = np.int32(0)
        for i in range(i_end):
            t_tp = np.int32(hit_tp[i, start])
            while j_ptr < n_sl and np.int32(hit_sl[j_ptr, start]) <= t_tp:
                j_ptr = np.int32(j_ptr + 1)
            _tp_sl_add_row_range_block(
                row_diff,
                np.int32(i),
                j_ptr,
                n_sl,
                float(log_tp_arr[i]),
                tp_block_start,
                sl_block_start,
            )

        i_ptr = np.int32(0)
        for j in range(j_end):
            t_sl = np.int32(hit_sl[j, start])
            while i_ptr < i_end and np.int32(hit_tp[i_ptr, start]) < t_sl:
                i_ptr = np.int32(i_ptr + 1)
            _tp_sl_add_col_range_block(
                col_diff,
                i_ptr,
                n_tp,
                np.int32(j),
                float(log_sl_arr[j]),
                tp_block_start,
                sl_block_start,
            )


@nb.njit(cache=True, parallel=True, fastmath=True)
def event_segments_n_tp_sl_15m_grid_cell_blocks(
    combo_idx_by_indicator: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_values: np.ndarray,
    segment_counts: np.ndarray,
    segment_pos_workspace: np.ndarray,
    run_abs_start: np.int32,
    t_exec_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    last_close: float,
    log_last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    tp_block_count: np.int32,
    sl_block_count: np.int32,
    out_best_tp_idx: np.ndarray,
    out_best_sl_idx: np.ndarray,
    out_best_ret: np.ndarray,
    out_trade_count: np.ndarray,
) -> None:
    arity = combo_idx_by_indicator.shape[0]
    combo_count = combo_idx_by_indicator.shape[1]
    n_tp = hit_long_tp.shape[0]
    n_sl = hit_long_sl.shape[0]
    max_trade_count = max(np.int32(1), np.int32(t_exec_abs - run_abs_start + 1))

    for k in nb.prange(combo_count):
        for indicator_pos in range(arity):
            segment_pos_workspace[k, indicator_pos] = np.int32(0)
        entry_abs_arr = np.empty(max_trade_count, dtype=np.int32)
        direction_arr = np.empty(max_trade_count, dtype=np.int8)
        signal_exit_arr = np.empty(max_trade_count, dtype=np.int32)
        current_dir = np.int8(0)
        current_entry_abs = np.int32(0)
        trade_count = np.int32(0)

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
                dirn = _tp_sl_apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_abs = np.int32(run_abs_start + segment_start + 1)
                    if entry_abs >= t_exec_abs:
                        break
                    if dirn == 0:
                        entry_abs_arr[trade_count] = current_entry_abs
                        direction_arr[trade_count] = current_dir
                        signal_exit_arr[trade_count] = entry_abs
                        trade_count += np.int32(1)
                        current_dir = np.int8(0)
                        current_entry_abs = np.int32(0)
                    elif current_dir == 0:
                        current_dir = dirn
                        current_entry_abs = entry_abs
                    elif dirn != current_dir:
                        entry_abs_arr[trade_count] = current_entry_abs
                        direction_arr[trade_count] = current_dir
                        signal_exit_arr[trade_count] = entry_abs
                        trade_count += np.int32(1)
                        current_dir = dirn
                        current_entry_abs = entry_abs

            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, k]
                segment_idx = segment_pos_workspace[k, indicator_pos]
                if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                    segment_pos_workspace[k, indicator_pos] = np.int32(segment_idx + 1)

        if current_dir != 0 and close_on_end == 1 and t_exec_abs > 0:
            entry_abs_arr[trade_count] = current_entry_abs
            direction_arr[trade_count] = current_dir
            signal_exit_arr[trade_count] = t_exec_abs
            trade_count += np.int32(1)

        best_log = -1.0e300
        best_tp = np.int32(0)
        best_sl = np.int32(0)
        for tp_block_start in range(0, n_tp, tp_block_count):
            tp_stop = min(tp_block_start + tp_block_count, n_tp)
            block_tp = np.int32(tp_stop - tp_block_start)
            for sl_block_start in range(0, n_sl, sl_block_count):
                sl_stop = min(sl_block_start + sl_block_count, n_sl)
                block_sl = np.int32(sl_stop - sl_block_start)
                row_diff = np.zeros((block_tp, block_sl + 1), dtype=np.float64)
                col_diff = np.zeros((block_tp + 1, block_sl), dtype=np.float64)
                rect_diff = np.zeros((block_tp + 1, block_sl + 1), dtype=np.float64)

                for trade_idx in range(trade_count):
                    _tp_sl_apply_trade_to_block_diff(
                        direction_arr[trade_idx],
                        entry_abs_arr[trade_idx],
                        signal_exit_arr[trade_idx],
                        price_open,
                        log_open,
                        last_close,
                        log_last_close,
                        t_exec_abs,
                        hit_long_tp,
                        hit_long_sl,
                        hit_short_tp,
                        hit_short_sl,
                        log_fac_tp_long,
                        log_fac_sl_long,
                        log_fac_tp_short,
                        log_fac_sl_short,
                        log_fee_two_sides,
                        close_on_end,
                        row_diff,
                        col_diff,
                        rect_diff,
                        np.int32(tp_block_start),
                        np.int32(sl_block_start),
                    )

                for i in range(block_tp):
                    run = 0.0
                    for j in range(block_sl):
                        run += row_diff[i, j]
                        row_diff[i, j] = run
                for j in range(block_sl):
                    run = 0.0
                    for i in range(block_tp):
                        run += col_diff[i, j]
                        col_diff[i, j] = run
                for i in range(block_tp):
                    row_run = 0.0
                    for j in range(block_sl):
                        row_run += rect_diff[i, j]
                        if i == 0:
                            rect_diff[i, j] = row_run
                        else:
                            rect_diff[i, j] = row_run + rect_diff[i - 1, j]

                for i in range(block_tp):
                    for j in range(block_sl):
                        value = row_diff[i, j] + col_diff[i, j] + rect_diff[i, j]
                        if value > best_log:
                            best_log = value
                            best_tp = np.int32(tp_block_start + i)
                            best_sl = np.int32(sl_block_start + j)

        out_best_tp_idx[k] = best_tp
        out_best_sl_idx[k] = best_sl
        out_trade_count[k] = trade_count
        if trade_count <= 0:
            out_best_ret[k] = np.float32(0.0)
        elif best_log <= -1.0e200:
            out_best_ret[k] = np.float32(-1.0)
        else:
            out_best_ret[k] = np.float32(math.exp(best_log) - 1.0)


@nb.njit(cache=True, parallel=True, fastmath=True)
def event_segments_n_tp_sl_15m_grid(
    combo_idx_by_indicator: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_values: np.ndarray,
    segment_counts: np.ndarray,
    segment_pos_workspace: np.ndarray,
    run_abs_start: np.int32,
    t_exec_abs: np.int32,
    price_open: np.ndarray,
    log_open: np.ndarray,
    last_close: float,
    log_last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    direction_mode: np.int8,
    out_best_tp_idx: np.ndarray,
    out_best_sl_idx: np.ndarray,
    out_best_ret: np.ndarray,
    out_trade_count: np.ndarray,
) -> None:
    arity = combo_idx_by_indicator.shape[0]
    combo_count = combo_idx_by_indicator.shape[1]
    n_tp = hit_long_tp.shape[0]
    n_sl = hit_long_sl.shape[0]

    for k in nb.prange(combo_count):
        for indicator_pos in range(arity):
            segment_pos_workspace[k, indicator_pos] = np.int32(0)
        row_diff = np.zeros((n_tp, n_sl + 1), dtype=np.float64)
        col_diff = np.zeros((n_tp + 1, n_sl), dtype=np.float64)
        rect_diff = np.zeros((n_tp + 1, n_sl + 1), dtype=np.float64)
        current_dir = np.int8(0)
        current_entry_abs = np.int32(0)
        trade_count = np.int32(0)

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
                dirn = _tp_sl_apply_direction_mode(raw_dir, direction_mode)
                if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                    entry_abs = np.int32(run_abs_start + segment_start + 1)
                    if entry_abs >= t_exec_abs:
                        break
                    if dirn == 0:
                        _tp_sl_apply_trade_to_diff(
                            current_dir,
                            current_entry_abs,
                            entry_abs,
                            price_open,
                            log_open,
                            last_close,
                            log_last_close,
                            t_exec_abs,
                            hit_long_tp,
                            hit_long_sl,
                            hit_short_tp,
                            hit_short_sl,
                            log_fac_tp_long,
                            log_fac_sl_long,
                            log_fac_tp_short,
                            log_fac_sl_short,
                            log_fee_two_sides,
                            close_on_end,
                            row_diff,
                            col_diff,
                            rect_diff,
                        )
                        trade_count += np.int32(1)
                        current_dir = np.int8(0)
                        current_entry_abs = np.int32(0)
                    elif current_dir == 0:
                        current_dir = dirn
                        current_entry_abs = entry_abs
                    elif dirn != current_dir:
                        _tp_sl_apply_trade_to_diff(
                            current_dir,
                            current_entry_abs,
                            entry_abs,
                            price_open,
                            log_open,
                            last_close,
                            log_last_close,
                            t_exec_abs,
                            hit_long_tp,
                            hit_long_sl,
                            hit_short_tp,
                            hit_short_sl,
                            log_fac_tp_long,
                            log_fac_sl_long,
                            log_fac_tp_short,
                            log_fac_sl_short,
                            log_fee_two_sides,
                            close_on_end,
                            row_diff,
                            col_diff,
                            rect_diff,
                        )
                        trade_count += np.int32(1)
                        current_dir = dirn
                        current_entry_abs = entry_abs

            for indicator_pos in range(arity):
                row_idx = combo_idx_by_indicator[indicator_pos, k]
                segment_idx = segment_pos_workspace[k, indicator_pos]
                if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                    segment_pos_workspace[k, indicator_pos] = np.int32(segment_idx + 1)

        if current_dir != 0 and close_on_end == 1 and t_exec_abs > 0:
            _tp_sl_apply_trade_to_diff(
                current_dir,
                current_entry_abs,
                t_exec_abs,
                price_open,
                log_open,
                last_close,
                log_last_close,
                t_exec_abs,
                hit_long_tp,
                hit_long_sl,
                hit_short_tp,
                hit_short_sl,
                log_fac_tp_long,
                log_fac_sl_long,
                log_fac_tp_short,
                log_fac_sl_short,
                log_fee_two_sides,
                close_on_end,
                row_diff,
                col_diff,
                rect_diff,
            )
            trade_count += np.int32(1)

        for i in range(n_tp):
            run = 0.0
            for j in range(n_sl):
                run += row_diff[i, j]
                row_diff[i, j] = run
        for j in range(n_sl):
            run = 0.0
            for i in range(n_tp):
                run += col_diff[i, j]
                col_diff[i, j] = run
        for i in range(n_tp):
            row_run = 0.0
            for j in range(n_sl):
                row_run += rect_diff[i, j]
                if i == 0:
                    rect_diff[i, j] = row_run
                else:
                    rect_diff[i, j] = row_run + rect_diff[i - 1, j]

        best_log = -1.0e300
        best_tp = np.int32(0)
        best_sl = np.int32(0)
        for i in range(n_tp):
            for j in range(n_sl):
                value = row_diff[i, j] + col_diff[i, j] + rect_diff[i, j]
                if value > best_log:
                    best_log = value
                    best_tp = np.int32(i)
                    best_sl = np.int32(j)
        out_best_tp_idx[k] = best_tp
        out_best_sl_idx[k] = best_sl
        out_trade_count[k] = trade_count
        if trade_count <= 0:
            out_best_ret[k] = np.float32(0.0)
        elif best_log <= -1.0e200:
            out_best_ret[k] = np.float32(-1.0)
        else:
            out_best_ret[k] = np.float32(math.exp(best_log) - 1.0)


@nb.njit(cache=True, parallel=True, fastmath=False)
def event_segments_n_tp_sl_15m_grid_execution_sizing(
    combo_idx_by_indicator: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_values: np.ndarray,
    segment_counts: np.ndarray,
    segment_pos_workspace: np.ndarray,
    run_abs_start: np.int32,
    t_exec_abs: np.int32,
    price_open: np.ndarray,
    last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    initial_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
    direction_mode: np.int8,
    out_best_tp_idx: np.ndarray,
    out_best_sl_idx: np.ndarray,
    out_best_ret: np.ndarray,
    out_trade_count: np.ndarray,
) -> None:
    arity = combo_idx_by_indicator.shape[0]
    combo_count = combo_idx_by_indicator.shape[1]
    n_tp = hit_long_tp.shape[0]
    n_sl = hit_long_sl.shape[0]

    for k in nb.prange(combo_count):
        best_return = -1.0e300
        best_tp = np.int32(0)
        best_sl = np.int32(0)
        best_trade_count = np.int32(0)

        for tp_i in range(n_tp):
            for sl_i in range(n_sl):
                for indicator_pos in range(arity):
                    segment_pos_workspace[k, indicator_pos] = np.int32(0)

                available_quote = initial_cash_quote
                safe_quote = 0.0
                equity = initial_cash_quote
                current_dir = np.int8(0)
                current_entry_abs = np.int32(0)
                closed_trade_count = np.int32(0)

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
                                if (
                                    segment_values[indicator_pos, row_idx, segment_idx]
                                    != raw_dir
                                ):
                                    raw_dir = np.int8(0)
                                    break
                        dirn = _tp_sl_apply_direction_mode(raw_dir, direction_mode)
                        if dirn != 0 or (direction_mode == 1 and current_dir != 0):
                            entry_abs = np.int32(run_abs_start + segment_start + 1)
                            if entry_abs >= t_exec_abs:
                                break
                            if dirn == 0:
                                (
                                    available_quote,
                                    safe_quote,
                                    equity,
                                    closed_trade_count,
                                ) = _tp_sl_apply_trade_to_account_for_cell(
                                    current_dir,
                                    current_entry_abs,
                                    entry_abs,
                                    np.int32(tp_i),
                                    np.int32(sl_i),
                                    price_open,
                                    last_close,
                                    hit_long_tp,
                                    hit_long_sl,
                                    hit_short_tp,
                                    hit_short_sl,
                                    log_fac_tp_long,
                                    log_fac_sl_long,
                                    log_fac_tp_short,
                                    log_fac_sl_short,
                                    log_fee_two_sides,
                                    close_on_end,
                                    t_exec_abs,
                                    available_quote,
                                    safe_quote,
                                    equity,
                                    sizing_mode_code,
                                    configured_quote_amount,
                                    equity_pct,
                                    min_quote,
                                    max_quote,
                                    safe_profit_percent,
                                    use_profit_lock,
                                    closed_trade_count,
                                )
                                current_dir = np.int8(0)
                                current_entry_abs = np.int32(0)
                            elif current_dir == 0:
                                current_dir = dirn
                                current_entry_abs = entry_abs
                            elif dirn != current_dir:
                                (
                                    available_quote,
                                    safe_quote,
                                    equity,
                                    closed_trade_count,
                                ) = _tp_sl_apply_trade_to_account_for_cell(
                                    current_dir,
                                    current_entry_abs,
                                    entry_abs,
                                    np.int32(tp_i),
                                    np.int32(sl_i),
                                    price_open,
                                    last_close,
                                    hit_long_tp,
                                    hit_long_sl,
                                    hit_short_tp,
                                    hit_short_sl,
                                    log_fac_tp_long,
                                    log_fac_sl_long,
                                    log_fac_tp_short,
                                    log_fac_sl_short,
                                    log_fee_two_sides,
                                    close_on_end,
                                    t_exec_abs,
                                    available_quote,
                                    safe_quote,
                                    equity,
                                    sizing_mode_code,
                                    configured_quote_amount,
                                    equity_pct,
                                    min_quote,
                                    max_quote,
                                    safe_profit_percent,
                                    use_profit_lock,
                                    closed_trade_count,
                                )
                                current_dir = dirn
                                current_entry_abs = entry_abs

                    for indicator_pos in range(arity):
                        row_idx = combo_idx_by_indicator[indicator_pos, k]
                        segment_idx = segment_pos_workspace[k, indicator_pos]
                        if segment_ends[indicator_pos, row_idx, segment_idx] == segment_end:
                            segment_pos_workspace[k, indicator_pos] = np.int32(
                                segment_idx + 1
                            )

                if current_dir != 0:
                    (
                        available_quote,
                        safe_quote,
                        equity,
                        closed_trade_count,
                    ) = _tp_sl_apply_trade_to_account_for_cell(
                        current_dir,
                        current_entry_abs,
                        t_exec_abs,
                        np.int32(tp_i),
                        np.int32(sl_i),
                        price_open,
                        last_close,
                        hit_long_tp,
                        hit_long_sl,
                        hit_short_tp,
                        hit_short_sl,
                        log_fac_tp_long,
                        log_fac_sl_long,
                        log_fac_tp_short,
                        log_fac_sl_short,
                        log_fee_two_sides,
                        close_on_end,
                        t_exec_abs,
                        available_quote,
                        safe_quote,
                        equity,
                        sizing_mode_code,
                        configured_quote_amount,
                        equity_pct,
                        min_quote,
                        max_quote,
                        safe_profit_percent,
                        use_profit_lock,
                        closed_trade_count,
                    )

                total_return = (equity / initial_cash_quote) - 1.0
                if total_return > best_return + TP_SL_BEST_CELL_TIE_EPS:
                    best_return = total_return
                    best_tp = np.int32(tp_i)
                    best_sl = np.int32(sl_i)
                    best_trade_count = closed_trade_count

        out_best_tp_idx[k] = best_tp
        out_best_sl_idx[k] = best_sl
        out_trade_count[k] = best_trade_count
        out_best_ret[k] = np.float32(best_return)


@nb.njit(cache=True, inline="always")
def _tp_sl_apply_trade_to_account_for_cell(
    dirn: np.int8,
    entry_abs: np.int32,
    sig_exit_abs: np.int32,
    tp_i: np.int32,
    sl_i: np.int32,
    price_open: np.ndarray,
    last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    t_exec_abs: np.int32,
    available_quote: float,
    safe_quote: float,
    equity: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
    closed_trade_count: np.int32,
) -> tuple[float, float, float, np.int32]:
    log_value, closed = _tp_sl_trade_log_contrib_and_closed(
        dirn,
        entry_abs,
        sig_exit_abs,
        tp_i,
        sl_i,
        price_open,
        last_close,
        hit_long_tp,
        hit_long_sl,
        hit_short_tp,
        hit_short_sl,
        log_fac_tp_long,
        log_fac_sl_long,
        log_fac_tp_short,
        log_fac_sl_short,
        log_fee_two_sides,
        close_on_end,
        t_exec_abs,
    )
    if closed == 0:
        return available_quote, safe_quote, equity, closed_trade_count
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
        return available_quote, safe_quote, equity, closed_trade_count
    trade_return = -1.0 if log_value <= -1.0e200 else math.exp(log_value) - 1.0
    net_pnl_quote = quote_amount * trade_return
    available_quote += net_pnl_quote
    if use_profit_lock == 1 and net_pnl_quote > 0.0:
        locked_profit_quote = net_pnl_quote * (safe_profit_percent / 100.0)
        available_quote -= locked_profit_quote
        safe_quote += locked_profit_quote
    equity = available_quote + safe_quote
    closed_trade_count += np.int32(1)
    return available_quote, safe_quote, equity, closed_trade_count


@nb.njit(cache=True, inline="always")
def _tp_sl_trade_log_contrib_and_closed(
    dirn: np.int8,
    entry_abs: np.int32,
    sig_exit_abs: np.int32,
    tp_i: np.int32,
    sl_i: np.int32,
    price_open: np.ndarray,
    last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    t_exec_abs: np.int32,
) -> tuple[float, np.int8]:
    entry_open = float(price_open[entry_abs])
    if entry_open <= 0.0:
        return 0.0, np.int8(0)
    start = np.int32(entry_abs + 1)
    if dirn == 1:
        hit_tp_value = (
            np.int32(hit_long_tp[tp_i, start])
            if start < t_exec_abs
            else np.int32(2147483647)
        )
        hit_sl_value = (
            np.int32(hit_long_sl[sl_i, start])
            if start < t_exec_abs
            else np.int32(2147483647)
        )
        log_tp = float(log_fac_tp_long[tp_i])
        log_sl = float(log_fac_sl_long[sl_i])
    else:
        hit_tp_value = (
            np.int32(hit_short_tp[tp_i, start])
            if start < t_exec_abs
            else np.int32(2147483647)
        )
        hit_sl_value = (
            np.int32(hit_short_sl[sl_i, start])
            if start < t_exec_abs
            else np.int32(2147483647)
        )
        log_tp = float(log_fac_tp_short[tp_i])
        log_sl = float(log_fac_sl_short[sl_i])

    stop_abs = sig_exit_abs if sig_exit_abs < t_exec_abs else t_exec_abs
    if start < t_exec_abs:
        if hit_tp_value < stop_abs and hit_tp_value < hit_sl_value:
            return log_tp, np.int8(1)
        if hit_sl_value < stop_abs and hit_sl_value <= hit_tp_value:
            return log_sl, np.int8(1)

    if sig_exit_abs < t_exec_abs:
        exit_open = float(price_open[sig_exit_abs])
        if exit_open <= 0.0:
            return NEG_LARGE, np.int8(1)
        if dirn == 1:
            return log_fee_two_sides + math.log(exit_open / entry_open), np.int8(1)
        ratio = exit_open / entry_open
        if ratio >= 2.0:
            return NEG_LARGE, np.int8(1)
        return log_fee_two_sides + math.log(2.0 - ratio), np.int8(1)

    if close_on_end == 1 and t_exec_abs > 0:
        if last_close <= 0.0:
            return NEG_LARGE, np.int8(1)
        if dirn == 1:
            return log_fee_two_sides + math.log(last_close / entry_open), np.int8(1)
        ratio = last_close / entry_open
        if ratio >= 2.0:
            return NEG_LARGE, np.int8(1)
        return log_fee_two_sides + math.log(2.0 - ratio), np.int8(1)
    return 0.0, np.int8(0)


@nb.njit(cache=True, inline="always")
def _tp_sl_apply_direction_mode(raw_dir: np.int8 | int, direction_mode: np.int8 | int) -> np.int8:
    if direction_mode == 1:
        if raw_dir == 1:
            return np.int8(1)
        return np.int8(0)
    return np.int8(raw_dir)


def evaluate_tp_sl_reference_rows_slow(
    *,
    prepared_result: BacktestPreparePoolsResult,
    local_indices: tuple[int, ...],
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    direction_mode: str,
) -> dict[str, float | int]:
    entry_abs, dir_arr, sig_exit_abs = build_trade_list_15m_for_indicator_rows_slow(
        prepared_result=prepared_result,
        local_indices=local_indices,
        direction_mode=direction_mode,
    )
    best_tp, best_sl, best_ret, trade_count = evaluate_tp_sl_reference_trade_list_direct(
        entry_abs,
        dir_arr,
        sig_exit_abs,
        runtime.price_open_15m,
        runtime.last_close_15m,
        hit_times.long_tp,
        hit_times.long_sl,
        hit_times.short_tp,
        hit_times.short_sl,
        runtime.log_fac_tp_long,
        runtime.log_fac_sl_long,
        runtime.log_fac_tp_short,
        runtime.log_fac_sl_short,
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
    return {
        "best_tp_idx": int(best_tp),
        "best_sl_idx": int(best_sl),
        "total_return_pct": float(best_ret) * 100.0,
        "trade_count": int(trade_count),
    }


def build_trade_list_15m_for_indicator_rows_slow(
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
        raise BacktestTpSlExactRejected(
            f"Unsupported direction_mode={direction_mode!r}; expected "
            f"{(DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL)!r}"
        )
    entry_abs: list[int] = []
    directions: list[int] = []
    sig_exit_abs: list[int] = []
    current_dir = 0
    current_entry = 0
    start_15m = int(prepared_result.time_slice_start_15m)
    stop_15m = int(prepared_result.time_slice_stop_15m)
    for signal_idx in range(int(direction_signal.shape[0])):
        dirn = int(direction_signal[signal_idx])
        if dirn == 0 and not (
            direction_mode == DIRECTION_MODE_LONG_ONLY and current_dir != 0
        ):
            continue
        entry_idx = start_15m + signal_idx + 1
        if entry_idx >= stop_15m:
            break
        if dirn == 0:
            entry_abs.append(current_entry)
            directions.append(current_dir)
            sig_exit_abs.append(entry_idx)
            current_dir = 0
            current_entry = 0
            continue
        if current_dir == 0:
            current_dir = dirn
            current_entry = entry_idx
            continue
        if dirn == current_dir:
            continue
        entry_abs.append(current_entry)
        directions.append(current_dir)
        sig_exit_abs.append(entry_idx)
        current_dir = dirn
        current_entry = entry_idx
    if current_dir != 0:
        entry_abs.append(current_entry)
        directions.append(current_dir)
        sig_exit_abs.append(stop_15m)
    return (
        np.asarray(entry_abs, dtype=np.int32),
        np.asarray(directions, dtype=np.int8),
        np.asarray(sig_exit_abs, dtype=np.int32),
    )


@nb.njit(cache=True, fastmath=False)
def evaluate_tp_sl_reference_trade_list_direct(
    entry_abs: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_abs: np.ndarray,
    price_open: np.ndarray,
    last_close: float,
    hit_long_tp: np.ndarray,
    hit_long_sl: np.ndarray,
    hit_short_tp: np.ndarray,
    hit_short_sl: np.ndarray,
    log_fac_tp_long: np.ndarray,
    log_fac_sl_long: np.ndarray,
    log_fac_tp_short: np.ndarray,
    log_fac_sl_short: np.ndarray,
    log_fee_two_sides: float,
    close_on_end: np.int8,
    initial_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
    t_exec_abs: np.int32,
) -> tuple[np.int32, np.int32, float, np.int32]:
    n_trades = np.int32(entry_abs.shape[0])
    if n_trades == 0:
        return np.int32(0), np.int32(0), 0.0, np.int32(0)
    n_tp = np.int32(hit_long_tp.shape[0])
    n_sl = np.int32(hit_long_sl.shape[0])
    best_log = -1.0e300
    best_tp = np.int32(0)
    best_sl = np.int32(0)
    best_trade_count = np.int32(0)
    for tp_i in range(n_tp):
        for sl_i in range(n_sl):
            available_quote = initial_cash_quote
            safe_quote = 0.0
            equity = initial_cash_quote
            closed_trade_count = np.int32(0)
            for trade_idx in range(n_trades):
                (
                    available_quote,
                    safe_quote,
                    equity,
                    closed_trade_count,
                ) = _tp_sl_apply_trade_to_account_for_cell(
                    dir_arr[trade_idx],
                    entry_abs[trade_idx],
                    sig_exit_abs[trade_idx],
                    np.int32(tp_i),
                    np.int32(sl_i),
                    price_open,
                    last_close,
                    hit_long_tp,
                    hit_long_sl,
                    hit_short_tp,
                    hit_short_sl,
                    log_fac_tp_long,
                    log_fac_sl_long,
                    log_fac_tp_short,
                    log_fac_sl_short,
                    log_fee_two_sides,
                    close_on_end,
                    t_exec_abs,
                    available_quote,
                    safe_quote,
                    equity,
                    sizing_mode_code,
                    configured_quote_amount,
                    equity_pct,
                    min_quote,
                    max_quote,
                    safe_profit_percent,
                    use_profit_lock,
                    closed_trade_count,
                )
            total_return = (equity / initial_cash_quote) - 1.0
            if total_return > best_log:
                best_log = total_return
                best_tp = np.int32(tp_i)
                best_sl = np.int32(sl_i)
                best_trade_count = closed_trade_count
    best_ret = best_log
    return best_tp, best_sl, best_ret, best_trade_count


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
                key=TP_SL_COMBO_CHUNK_DECODE_STAGE_NAME,
                elapsed=time.perf_counter() - decode_start,
            )
            break
        _add_timing(
            stage_timings=profile_stage_timings,
            key=TP_SL_COMBO_CHUNK_DECODE_STAGE_NAME,
            elapsed=time.perf_counter() - decode_start,
        )
        if not combo_planning_result.proxy_context.active:
            yield _SelectedCandidateBatch(
                rows_by_indicator=combo_chunk.rows_by_indicator
            )
            continue
        proxy_start = time.perf_counter()
        filter_result = filter_service.proxy_filter(
            combo_chunk=combo_chunk,
            proxy_context=combo_planning_result.proxy_context,
        )
        _add_timing(
            stage_timings=profile_stage_timings,
            key=TP_SL_PROXY_FILTER_STAGE_NAME,
            elapsed=time.perf_counter() - proxy_start,
        )
        if filter_result.selected_candidate_count > 0:
            yield _SelectedCandidateBatch(
                rows_by_indicator=filter_result.selected_rows_by_indicator
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


def _allocate_score_buffers(size: int) -> _TpSlScoreBuffers:
    return _TpSlScoreBuffers(
        total_return_pct=np.empty(size, dtype=np.float64),
        trade_count=np.empty(size, dtype=np.int32),
        best_tp_idx=np.empty(size, dtype=np.int32),
        best_sl_idx=np.empty(size, dtype=np.int32),
    )


def _sample_metrics_at(
    *,
    buffers: _TpSlScoreBuffers,
    hit_times: BacktestTpSlHitTimesSubset,
    index: int,
) -> Mapping[str, float]:
    best_tp_idx = int(buffers.best_tp_idx[index])
    best_sl_idx = int(buffers.best_sl_idx[index])
    return {
        "total_return_pct": float(buffers.total_return_pct[index]),
        "trade_count": float(buffers.trade_count[index]),
        "best_tp_pct": float(hit_times.tp_values[best_tp_idx] * np.float32(100.0)),
        "best_sl_pct": float(hit_times.sl_values[best_sl_idx] * np.float32(100.0)),
    }


def _cell_backend_telemetry(
    *,
    backend_id: str,
    scored_count: int,
    tp_count: int,
    sl_count: int,
    tp_block_count: int,
    sl_block_count: int,
    exact_scoring_s: float,
) -> Mapping[str, Any] | None:
    if backend_id != MATRIX_CELL_TP_SL_V1_BACKEND:
        return None
    tp_blocks = math.ceil(tp_count / tp_block_count) if tp_count > 0 else 0
    sl_blocks = math.ceil(sl_count / sl_block_count) if sl_count > 0 else 0
    tp_sl_cells = tp_count * sl_count
    trade_cell_evals = scored_count * tp_sl_cells
    cell_block_bytes = (
        tp_block_count * (sl_block_count + 1)
        + (tp_block_count + 1) * sl_block_count
        + (tp_block_count + 1) * (sl_block_count + 1)
    ) * np.dtype(np.float64).itemsize
    return {
        "schema": "backtest_tp_sl_full_grid_cell_backend_v1",
        "backend_id": MATRIX_CELL_TP_SL_V1_BACKEND,
        "reference_backend_id": EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
        "full_grid_parity_required": True,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "tp_sl_cells": tp_sl_cells,
        "cell_block_tp_count": tp_block_count,
        "cell_block_sl_count": sl_block_count,
        "cell_block_shape": f"{tp_block_count} x {sl_block_count}",
        "required_literal_block_shape": TP_SL_CELL_BLOCK_SHAPE_LITERAL,
        "cell_blocks_per_candidate": tp_blocks * sl_blocks,
        "cell_block_estimated_peak_bytes": cell_block_bytes,
        "trade_cell_evals": trade_cell_evals,
        "trade_cell_evals_per_sec": None
        if exact_scoring_s <= 0.0
        else trade_cell_evals / exact_scoring_s,
        "sl_wins_tie_rule": "SL wins",
    }


def _tp_sl_cell_block_counts_from_env(
    config: BacktestTpSlExactConfig,
) -> tuple[int, int]:
    return (
        _positive_int_env(
            TP_SL_CELL_BLOCK_TP_COUNT_ENV_KEY,
            default=config.cell_block_tp_count,
        ),
        _positive_int_env(
            TP_SL_CELL_BLOCK_SL_COUNT_ENV_KEY,
            default=config.cell_block_sl_count,
        ),
    )


def _positive_int_env(key: str, *, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise BacktestTpSlExactRejected(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise BacktestTpSlExactRejected(f"{key} must be a positive integer")
    return value


def _first_quality_eligible_index(
    *,
    trade_count: np.ndarray,
    min_closed_trades: int,
) -> int | None:
    for idx in range(int(trade_count.shape[0])):
        if int(trade_count[idx]) >= min_closed_trades:
            return idx
    return None


def _top_k_context_from_prepared(prepared_result: BacktestPreparePoolsResult) -> _TopKContext:
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
    heap: list[tuple[tuple[float, float, float, int, tuple[int, ...]], _TpSlHeapEntry]],
    top_k_context: _TopKContext,
    selected_rows_by_indicator: Mapping[str, np.ndarray],
    hit_times: BacktestTpSlHitTimesSubset,
    buffers: _TpSlScoreBuffers,
    top_k: int,
    candidate_ordinal_start: int,
    min_closed_trades: int,
) -> None:
    selected_rows_by_pos = tuple(
        selected_rows_by_indicator[indicator_id]
        for indicator_id in top_k_context.indicator_ids
    )
    row_ids_by_pos = top_k_context.row_ids_by_pos
    selected_indices, selected_count = _tp_sl_select_top_k_indices(
        buffers.total_return_pct,
        buffers.trade_count,
        buffers.best_tp_idx,
        buffers.best_sl_idx,
        hit_times.tp_values,
        hit_times.sl_values,
        np.int64(candidate_ordinal_start),
        np.int32(top_k),
        np.int32(min_closed_trades),
    )
    for selected_pos in range(selected_count):
        result_index = int(selected_indices[selected_pos])
        local_values = []
        original_values = []
        for pos, selected_rows in enumerate(selected_rows_by_pos):
            local_row = int(selected_rows[result_index])
            local_values.append(local_row)
            original_values.append(int(row_ids_by_pos[pos][local_row]))
        local_indices = tuple(local_values)
        original_rows = tuple(original_values)
        score = float(buffers.total_return_pct[result_index])
        best_tp_idx = int(buffers.best_tp_idx[result_index])
        best_sl_idx = int(buffers.best_sl_idx[result_index])
        best_tp_pct = float(hit_times.tp_values[best_tp_idx] * np.float32(100.0))
        best_sl_pct = float(hit_times.sl_values[best_sl_idx] * np.float32(100.0))
        candidate_ordinal = candidate_ordinal_start + result_index
        heap_key = (score, best_tp_pct, best_sl_pct, -candidate_ordinal, original_rows)
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
                        best_tp_idx=best_tp_idx,
                        best_sl_idx=best_sl_idx,
                        best_tp_pct=best_tp_pct,
                        best_sl_pct=best_sl_pct,
                        trade_count=int(buffers.trade_count[result_index]),
                        candidate_ordinal=candidate_ordinal,
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
                        best_tp_idx=best_tp_idx,
                        best_sl_idx=best_sl_idx,
                        best_tp_pct=best_tp_pct,
                        best_sl_pct=best_sl_pct,
                        trade_count=int(buffers.trade_count[result_index]),
                        candidate_ordinal=candidate_ordinal,
                    ),
                ),
            )


@nb.njit(cache=True, inline="always")
def _tp_sl_top_key_greater(
    score: float,
    tp_pct: float,
    sl_pct: float,
    neg_ordinal: np.int64,
    other_score: float,
    other_tp_pct: float,
    other_sl_pct: float,
    other_neg_ordinal: np.int64,
) -> bool:
    if score > other_score:
        return True
    if score < other_score:
        return False
    if tp_pct > other_tp_pct:
        return True
    if tp_pct < other_tp_pct:
        return False
    if sl_pct > other_sl_pct:
        return True
    if sl_pct < other_sl_pct:
        return False
    return bool(neg_ordinal > other_neg_ordinal)


@nb.njit(cache=True, inline="always")
def _tp_sl_top_key_less(
    score: float,
    tp_pct: float,
    sl_pct: float,
    neg_ordinal: np.int64,
    other_score: float,
    other_tp_pct: float,
    other_sl_pct: float,
    other_neg_ordinal: np.int64,
) -> bool:
    if score < other_score:
        return True
    if score > other_score:
        return False
    if tp_pct < other_tp_pct:
        return True
    if tp_pct > other_tp_pct:
        return False
    if sl_pct < other_sl_pct:
        return True
    if sl_pct > other_sl_pct:
        return False
    return bool(neg_ordinal < other_neg_ordinal)


@nb.njit(cache=True)
def _tp_sl_select_top_k_indices(
    scores: np.ndarray,
    trade_count: np.ndarray,
    best_tp_idx: np.ndarray,
    best_sl_idx: np.ndarray,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
    candidate_ordinal_start: np.int64,
    top_k: np.int32,
    min_closed_trades: np.int32,
) -> tuple[np.ndarray, np.int32]:
    selected_indices = np.empty(top_k, dtype=np.int32)
    selected_scores = np.empty(top_k, dtype=np.float64)
    selected_tp_pct = np.empty(top_k, dtype=np.float64)
    selected_sl_pct = np.empty(top_k, dtype=np.float64)
    selected_neg_ord = np.empty(top_k, dtype=np.int64)
    count = np.int32(0)
    for result_index in range(scores.shape[0]):
        if trade_count[result_index] < min_closed_trades:
            continue
        score = float(scores[result_index])
        tp_pct = float(tp_values[best_tp_idx[result_index]]) * 100.0
        sl_pct = float(sl_values[best_sl_idx[result_index]]) * 100.0
        neg_ordinal = -(candidate_ordinal_start + np.int64(result_index))
        if count < top_k:
            selected_indices[count] = np.int32(result_index)
            selected_scores[count] = score
            selected_tp_pct[count] = tp_pct
            selected_sl_pct[count] = sl_pct
            selected_neg_ord[count] = neg_ordinal
            count += np.int32(1)
            continue

        worst = np.int32(0)
        for selected_pos in range(np.int32(1), count):
            if _tp_sl_top_key_less(
                selected_scores[selected_pos],
                selected_tp_pct[selected_pos],
                selected_sl_pct[selected_pos],
                selected_neg_ord[selected_pos],
                selected_scores[worst],
                selected_tp_pct[worst],
                selected_sl_pct[worst],
                selected_neg_ord[worst],
            ):
                worst = selected_pos

        if _tp_sl_top_key_greater(
            score,
            tp_pct,
            sl_pct,
            neg_ordinal,
            selected_scores[worst],
            selected_tp_pct[worst],
            selected_sl_pct[worst],
            selected_neg_ord[worst],
        ):
            selected_indices[worst] = np.int32(result_index)
            selected_scores[worst] = score
            selected_tp_pct[worst] = tp_pct
            selected_sl_pct[worst] = sl_pct
            selected_neg_ord[worst] = neg_ordinal

    for left in range(count):
        best = left
        for right in range(left + 1, count):
            if _tp_sl_top_key_greater(
                selected_scores[right],
                selected_tp_pct[right],
                selected_sl_pct[right],
                selected_neg_ord[right],
                selected_scores[best],
                selected_tp_pct[best],
                selected_sl_pct[best],
                selected_neg_ord[best],
            ):
                best = right
        if best != left:
            tmp_idx = selected_indices[left]
            tmp_score = selected_scores[left]
            tmp_tp = selected_tp_pct[left]
            tmp_sl = selected_sl_pct[left]
            tmp_ord = selected_neg_ord[left]
            selected_indices[left] = selected_indices[best]
            selected_scores[left] = selected_scores[best]
            selected_tp_pct[left] = selected_tp_pct[best]
            selected_sl_pct[left] = selected_sl_pct[best]
            selected_neg_ord[left] = selected_neg_ord[best]
            selected_indices[best] = tmp_idx
            selected_scores[best] = tmp_score
            selected_tp_pct[best] = tmp_tp
            selected_sl_pct[best] = tmp_sl
            selected_neg_ord[best] = tmp_ord

    return selected_indices, count


def _materialize_heap_entry(
    *,
    top_k_context: _TopKContext,
    local_indices: tuple[int, ...],
    original_rows: tuple[int, ...],
    score: float,
    best_tp_idx: int,
    best_sl_idx: int,
    best_tp_pct: float,
    best_sl_pct: float,
    trade_count: int,
    candidate_ordinal: int,
) -> _TpSlHeapEntry:
    return _TpSlHeapEntry(
        score=score,
        original_rows=original_rows,
        local_indices=local_indices,
        best_tp_idx=best_tp_idx,
        best_sl_idx=best_sl_idx,
        best_tp_pct=best_tp_pct,
        best_sl_pct=best_sl_pct,
        total_return_pct=score,
        trade_count=trade_count,
        candidate_ordinal=candidate_ordinal,
        metadata_by_pos=tuple(
            top_k_context.metadata_by_pos[pos][local_indices[pos]]
            for pos in range(len(local_indices))
        ),
    )


def _top_results_from_heap(
    heap: list[tuple[tuple[float, float, float, int, tuple[int, ...]], _TpSlHeapEntry]],
    *,
    prepared_result: BacktestPreparePoolsResult,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    top_k_context: _TopKContext,
    direction_mode: str,
) -> tuple[BacktestTpSlTopResult, ...]:
    out: list[BacktestTpSlTopResult] = []
    for rank, (_, entry) in enumerate(
        sorted(heap, key=lambda pair: pair[0], reverse=True),
        start=1,
    ):
        full_metrics = _full_metrics_for_heap_entry(
            entry=entry,
            prepared_result=prepared_result,
            hit_times=hit_times,
            runtime=runtime,
            direction_mode=direction_mode,
        )
        metrics = {
            **full_metrics,
            "total_return_pct": entry.total_return_pct,
            "trade_count": float(entry.trade_count),
            "best_tp_pct": entry.best_tp_pct,
            "best_sl_pct": entry.best_sl_pct,
        }
        out.append(
            BacktestTpSlTopResult(
                rank=rank,
                score=entry.score,
                indicator_rows={
                    indicator_id: entry.original_rows[pos]
                    for pos, indicator_id in enumerate(top_k_context.indicator_ids)
                },
                best_tp_idx=entry.best_tp_idx,
                best_sl_idx=entry.best_sl_idx,
                metrics=metrics,
                metadata=_top_result_metadata(entry, top_k_context=top_k_context),
            )
        )
    return tuple(out)


def _top_result_metadata(
    entry: _TpSlHeapEntry,
    *,
    top_k_context: _TopKContext,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "candidate_ordinal": entry.candidate_ordinal,
    }
    for pos, indicator_id in enumerate(top_k_context.indicator_ids):
        row_metadata = entry.metadata_by_pos[pos].as_mapping()
        for key, value in row_metadata.items():
            metadata[f"{indicator_id}.{key}"] = value
    return metadata


def _apply_funding_adjustment_to_top_results(
    *,
    top_results: tuple[BacktestTpSlTopResult, ...],
    prepared_result: BacktestPreparePoolsResult,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    direction_mode: str,
    funding_arrays: ArtifactFundingArraysV2 | None,
    requested_top_n: int,
    requested_ranking: _RankingSpec,
    effective_ranking: _RankingSpec,
) -> tuple[BacktestTpSlTopResult, ...]:
    time_arrays = _tp_sl_time_arrays_from_prepared(prepared_result)
    if funding_arrays is None or time_arrays is None:
        return _rerank_funding_adjusted_top_results(
            tuple(
                _annotate_no_funding_available(
                    top_result=top_result,
                    requested_ranking=requested_ranking,
                    effective_ranking=effective_ranking,
                    requested_top_n=requested_top_n,
                    candidate_pool_size=len(top_results),
                )
                for top_result in top_results
            ),
            requested_top_n=requested_top_n,
            effective_ranking=effective_ranking,
        )

    open_time_15m, close_time_15m, execution_close_time_1m = time_arrays
    execution_close_1m = np.ascontiguousarray(
        np.asarray(prepared_result.execution_close_1m, dtype=np.float32)
    )
    adjusted: list[BacktestTpSlTopResult] = []
    for top_result in top_results:
        local_indices = _local_indices_from_top_result(
            prepared_result=prepared_result,
            top_result=top_result,
        )
        entry_abs, dir_arr, sig_exit_abs = build_trade_list_15m_for_indicator_rows_slow(
            prepared_result=prepared_result,
            local_indices=local_indices,
            direction_mode=direction_mode,
        )
        trade_returns, _bars_held = _selected_cell_trade_returns(
            entry_abs=entry_abs,
            dir_arr=dir_arr,
            sig_exit_abs=sig_exit_abs,
            best_tp_idx=top_result.best_tp_idx,
            best_sl_idx=top_result.best_sl_idx,
            hit_times=hit_times,
            runtime=runtime,
        )
        funding_summary = calculate_tp_sl_funding_adjustment(
            entry_abs=entry_abs,
            dir_arr=dir_arr,
            sig_exit_abs=sig_exit_abs,
            trade_returns=trade_returns,
            best_tp_idx=top_result.best_tp_idx,
            best_sl_idx=top_result.best_sl_idx,
            hit_times=hit_times,
            runtime=runtime,
            open_time_15m=open_time_15m,
            close_time_15m=close_time_15m,
            execution_close_1m=execution_close_1m,
            execution_close_time_1m=execution_close_time_1m,
            funding_time=funding_arrays.funding_time,
            funding_rate=funding_arrays.funding_rate,
            mark_price=funding_arrays.mark_price,
            data_quality=funding_arrays.data_quality,
            funding_data_quality=str(funding_arrays.coverage_status),
            warning_codes=funding_arrays.manifest.reason_codes,
        )
        funding_metrics = funding_summary.metric_payload(
            gross_total_return_pct=float(top_result.metrics["total_return_pct"]),
            initial_cash_quote=float(runtime.initial_cash_quote),
        )
        adjusted.append(
            BacktestTpSlTopResult(
                rank=top_result.rank,
                score=top_result.score,
                indicator_rows=top_result.indicator_rows,
                best_tp_idx=top_result.best_tp_idx,
                best_sl_idx=top_result.best_sl_idx,
                metrics={**dict(top_result.metrics), **funding_metrics},
                metadata={
                    **dict(top_result.metadata),
                    FUNDING_INCLUDED: True,
                    FUNDING_DATA_QUALITY: funding_summary.funding_data_quality,
                    FUNDING_WARNING_CODES: funding_summary.funding_warning_codes,
                    FUNDING_ADJUSTMENT_SCOPE: FUNDING_ADJUSTMENT_SCOPE_CANDIDATE_POOL,
                    FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING: False,
                    "requested_ranking_metric": requested_ranking.metric_name,
                    "effective_ranking_metric": effective_ranking.metric_name,
                    "funding_candidate_pool_size": len(top_results),
                    "requested_top_n": requested_top_n,
                    "funding_manifest_hash": funding_arrays.funding_manifest_hash,
                },
            )
        )
    return _rerank_funding_adjusted_top_results(
        tuple(adjusted),
        requested_top_n=requested_top_n,
        effective_ranking=effective_ranking,
    )


def _annotate_no_funding_available(
    *,
    top_result: BacktestTpSlTopResult,
    requested_ranking: _RankingSpec,
    effective_ranking: _RankingSpec,
    requested_top_n: int,
    candidate_pool_size: int,
) -> BacktestTpSlTopResult:
    gross_return = float(top_result.metrics["total_return_pct"])
    return BacktestTpSlTopResult(
        rank=top_result.rank,
        score=top_result.score,
        indicator_rows=top_result.indicator_rows,
        best_tp_idx=top_result.best_tp_idx,
        best_sl_idx=top_result.best_sl_idx,
        metrics={
            **dict(top_result.metrics),
            TOTAL_RETURN_PCT_NET_OF_FUNDING: gross_return,
            "funding_return_pct": 0.0,
            "funding_pnl_quote": 0.0,
            "funding_events_count": 0.0,
        },
        metadata={
            **dict(top_result.metadata),
            FUNDING_INCLUDED: False,
            FUNDING_DATA_QUALITY: "unavailable",
            FUNDING_WARNING_CODES: ("funding_artifacts_unavailable",),
            FUNDING_ADJUSTMENT_SCOPE: FUNDING_ADJUSTMENT_SCOPE_UNAVAILABLE,
            FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING: False,
            "requested_ranking_metric": requested_ranking.metric_name,
            "effective_ranking_metric": effective_ranking.metric_name,
            "funding_candidate_pool_size": candidate_pool_size,
            "requested_top_n": requested_top_n,
            "funding_manifest_hash": None,
        },
    )


def _rerank_funding_adjusted_top_results(
    top_results: tuple[BacktestTpSlTopResult, ...],
    *,
    requested_top_n: int,
    effective_ranking: _RankingSpec,
) -> tuple[BacktestTpSlTopResult, ...]:
    def sort_key(item: BacktestTpSlTopResult) -> tuple[float, int]:
        score = float(item.metrics[effective_ranking.metric_name])
        comparable_score = -score if effective_ranking.direction == "desc" else score
        return comparable_score, item.rank

    ranked = sorted(top_results, key=sort_key)[:requested_top_n]
    return tuple(
        BacktestTpSlTopResult(
            rank=rank,
            score=float(item.metrics[effective_ranking.metric_name]),
            indicator_rows=item.indicator_rows,
            best_tp_idx=item.best_tp_idx,
            best_sl_idx=item.best_sl_idx,
            metrics=item.metrics,
            metadata=item.metadata,
        )
        for rank, item in enumerate(ranked, start=1)
    )


def _tp_sl_time_arrays_from_prepared(
    prepared_result: BacktestPreparePoolsResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if (
        prepared_result.execution_open_time_1m is None
        or prepared_result.execution_close_time_1m is None
    ):
        return None
    open_time_1m = np.ascontiguousarray(
        np.asarray(prepared_result.execution_open_time_1m, dtype=np.int64)
    )
    close_time_1m = np.ascontiguousarray(
        np.asarray(prepared_result.execution_close_time_1m, dtype=np.int64)
    )
    mapping = prepared_result.execution_mapping
    open_idx = np.asarray(mapping.run_bar_open_1m_idx_15m, dtype=np.int32)
    close_idx = np.asarray(mapping.run_bar_close_1m_idx_15m, dtype=np.int32)
    start = int(prepared_result.time_slice_start_15m)
    stop = int(prepared_result.time_slice_stop_15m)
    if stop <= start:
        return None
    if int(open_idx.shape[0]) != stop - start or int(close_idx.shape[0]) != stop - start:
        return None
    if int(np.max(open_idx)) >= int(open_time_1m.shape[0]) or int(np.max(close_idx)) >= int(
        close_time_1m.shape[0]
    ):
        return None
    open_time_15m = np.zeros(stop, dtype=np.int64)
    close_time_15m = np.zeros(stop, dtype=np.int64)
    open_time_15m[start:stop] = open_time_1m[open_idx]
    close_time_15m[start:stop] = close_time_1m[close_idx]
    return (
        np.ascontiguousarray(open_time_15m),
        np.ascontiguousarray(close_time_15m),
        close_time_1m,
    )


def _local_indices_from_top_result(
    *,
    prepared_result: BacktestPreparePoolsResult,
    top_result: BacktestTpSlTopResult,
) -> tuple[int, ...]:
    pools_by_id = _pool_by_id(prepared_result)
    local_indices: list[int] = []
    for indicator_id in prepared_result.indicator_ids:
        row_id = int(top_result.indicator_rows[indicator_id])
        row_ids = np.asarray(pools_by_id[indicator_id].row_ids, dtype=np.int64)
        matches = np.flatnonzero(row_ids == row_id)
        if int(matches.shape[0]) == 0:
            raise BacktestTpSlExactRejected(
                f"funding adjustment cannot find original row {row_id}"
            )
        local_indices.append(int(matches[0]))
    return tuple(local_indices)


def _full_metrics_for_heap_entry(
    *,
    entry: _TpSlHeapEntry,
    prepared_result: BacktestPreparePoolsResult,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
    direction_mode: str,
) -> dict[str, float]:
    entry_abs, dir_arr, sig_exit_abs = build_trade_list_15m_for_indicator_rows_slow(
        prepared_result=prepared_result,
        local_indices=entry.local_indices,
        direction_mode=direction_mode,
    )
    trade_returns, bars_held = _selected_cell_trade_returns(
        entry_abs=entry_abs,
        dir_arr=dir_arr,
        sig_exit_abs=sig_exit_abs,
        best_tp_idx=entry.best_tp_idx,
        best_sl_idx=entry.best_sl_idx,
        hit_times=hit_times,
        runtime=runtime,
    )
    return _summary_metrics_from_trade_returns(
        trade_returns=trade_returns,
        bars_held=bars_held,
        t_exec_abs=int(runtime.t_exec_abs_15m),
        runtime=runtime,
    )


def _selected_cell_trade_returns(
    *,
    entry_abs: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_abs: np.ndarray,
    best_tp_idx: int,
    best_sl_idx: int,
    hit_times: BacktestTpSlHitTimesSubset,
    runtime: _TpSlRuntimeContext,
) -> tuple[list[float], list[float]]:
    trade_returns: list[float] = []
    bars_held: list[float] = []
    for trade_idx in range(int(entry_abs.shape[0])):
        log_value_raw, closed = _tp_sl_trade_log_contrib_and_closed(
            np.int8(dir_arr[trade_idx]),
            np.int32(entry_abs[trade_idx]),
            np.int32(sig_exit_abs[trade_idx]),
            np.int32(best_tp_idx),
            np.int32(best_sl_idx),
            runtime.price_open_15m,
            runtime.last_close_15m,
            hit_times.long_tp,
            hit_times.long_sl,
            hit_times.short_tp,
            hit_times.short_sl,
            runtime.log_fac_tp_long,
            runtime.log_fac_sl_long,
            runtime.log_fac_tp_short,
            runtime.log_fac_sl_short,
            runtime.log_fee_two_sides,
            runtime.close_on_end,
            runtime.t_exec_abs_15m,
        )
        if int(closed) == 0:
            continue
        log_value = float(log_value_raw)
        if log_value <= -1.0e200:
            trade_return = -1.0
        else:
            trade_return = math.exp(log_value) - 1.0
        selected_exit = resolve_tp_sl_selected_exit(
            direction=int(dir_arr[trade_idx]),
            entry_abs=int(entry_abs[trade_idx]),
            signal_exit_abs=int(sig_exit_abs[trade_idx]),
            best_tp_idx=best_tp_idx,
            best_sl_idx=best_sl_idx,
            hit_times=hit_times,
            runtime=runtime,
        )
        trade_returns.append(trade_return)
        bars_held.append(max(0.0, float(selected_exit.exit_abs - int(entry_abs[trade_idx]))))
    return trade_returns, bars_held


def _summary_metrics_from_trade_returns(
    *,
    trade_returns: list[float],
    bars_held: list[float],
    t_exec_abs: int,
    runtime: _TpSlRuntimeContext,
) -> dict[str, float]:
    available_quote = runtime.initial_cash_quote
    safe_quote = 0.0
    equity = runtime.initial_cash_quote
    peak_equity = equity
    max_drawdown_pct = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    win_count = 0
    sum_trade_return = 0.0
    sum_trade_return_squared = 0.0
    total_trade_return_pct = 0.0
    total_bars = 0.0
    trade_count = 0
    for idx, trade_return in enumerate(trade_returns):
        quote_amount = execution_quote_amount_py(
            available_quote=available_quote,
            equity=equity,
            sizing_mode_code=runtime.sizing_mode_code,
            quote_amount=runtime.quote_amount,
            equity_pct=runtime.equity_pct,
            min_quote=runtime.min_quote,
            max_quote=runtime.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        pnl = quote_amount * trade_return
        available_quote += pnl
        if runtime.use_profit_lock == 1 and pnl > 0.0:
            locked_profit_quote = pnl * (runtime.safe_profit_percent / 100.0)
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        equity = available_quote + safe_quote
        if equity > peak_equity:
            peak_equity = equity
        elif peak_equity > 0.0:
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
        if pnl > 0.0:
            gross_profit += pnl
            win_count += 1
        elif pnl < 0.0:
            gross_loss += abs(pnl)
        sum_trade_return += trade_return
        sum_trade_return_squared += trade_return * trade_return
        total_trade_return_pct += trade_return * 100.0
        total_bars += bars_held[idx]
        trade_count += 1
    total_return_pct = ((equity / runtime.initial_cash_quote) - 1.0) * 100.0
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = math.inf
    else:
        return_over_max_drawdown = 0.0
    if trade_count > 0:
        win_rate_pct = (float(win_count) / float(trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(trade_count)
        avg_trade_exec_bars = total_bars / float(trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0
    exposure_pct = (total_bars / float(t_exec_abs)) * 100.0 if t_exec_abs > 0 else 0.0
    sharpe_trades = _trade_sharpe(
        trade_count=trade_count,
        sum_trade_return=sum_trade_return,
        sum_trade_return_squared=sum_trade_return_squared,
        bars_per_year_exec=BARS_PER_YEAR_EXEC_1M / 15.0,
        sentinel_index=t_exec_abs,
    )
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "return_over_max_drawdown": return_over_max_drawdown,
        "profit_factor": profit_factor,
        "trade_count": float(trade_count),
        "sharpe_trades": sharpe_trades,
        "win_rate_pct": win_rate_pct,
        "avg_trade_ret_pct": avg_trade_ret_pct,
        "avg_trade_exec_bars": avg_trade_exec_bars,
        "exposure_pct": exposure_pct,
    }


def _trade_sharpe(
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


def _validate_hit_times_for_prepared(
    *,
    hit_times: BacktestTpSlHitTimesSubset,
    prepared_result: BacktestPreparePoolsResult,
) -> None:
    if hit_times.sentinel_index < int(prepared_result.time_slice_stop_15m):
        raise BacktestTpSlExactRejected(
            "hit-times sentinel_index must cover prepared time_slice_stop_15m"
        )


def _tp_sl_runtime_context_from_prepared(
    *,
    prepared_result: BacktestPreparePoolsResult,
    hit_times: BacktestTpSlHitTimesSubset,
    execution_settings: _ExecutionSettings,
) -> _TpSlRuntimeContext:
    if prepared_result.execution_open_1m is None or prepared_result.execution_close_1m is None:
        raise BacktestTpSlExactRejected(
            "TP/SL exact scoring requires execution_open_1m and execution_close_1m"
        )
    open_1m = np.asarray(prepared_result.execution_open_1m, dtype=np.float32)
    close_1m = np.asarray(prepared_result.execution_close_1m, dtype=np.float32)
    mapping = prepared_result.execution_mapping
    start = int(prepared_result.time_slice_start_15m)
    stop = int(prepared_result.time_slice_stop_15m)
    if stop <= start:
        raise BacktestTpSlExactRejected("prepared time slice must be non-empty")
    if stop > hit_times.sentinel_index:
        raise BacktestTpSlExactRejected("prepared time slice exceeds hit-times sentinel")
    price_open_15m = np.zeros(stop, dtype=np.float32)
    open_idx = np.asarray(mapping.run_bar_open_1m_idx_15m, dtype=np.int32)
    close_idx = np.asarray(mapping.run_bar_close_1m_idx_15m, dtype=np.int32)
    if int(open_idx.shape[0]) != stop - start or int(close_idx.shape[0]) != stop - start:
        raise BacktestTpSlExactRejected("prepared mapping length must match time slice")
    if int(np.max(open_idx)) >= int(open_1m.shape[0]) or int(np.max(close_idx)) >= int(
        close_1m.shape[0]
    ):
        raise BacktestTpSlExactRejected("prepared mapping indexes exceed execution prices")
    price_open_15m[start:stop] = open_1m[open_idx]
    log_open_15m = np.zeros(price_open_15m.shape[0], dtype=np.float64)
    positive = price_open_15m > np.float32(0.0)
    log_open_15m[positive] = np.log(price_open_15m[positive].astype(np.float64, copy=False))
    last_close = float(close_1m[int(close_idx[-1])])
    log_last_close = float(math.log(last_close)) if last_close > 0.0 else 0.0
    fee_two_sides = float((1.0 - execution_settings.fee_rate) * (1.0 - execution_settings.fee_rate))
    if fee_two_sides <= 0.0:
        raise BacktestTpSlExactRejected("fee_rate leaves non-positive two-sided fee factor")
    return _TpSlRuntimeContext(
        run_abs_start_15m=np.int32(start),
        t_exec_abs_15m=np.int32(stop),
        price_open_15m=np.ascontiguousarray(price_open_15m),
        log_open_15m=np.ascontiguousarray(log_open_15m),
        last_close_15m=last_close,
        log_last_close_15m=log_last_close,
        log_fac_tp_long=np.ascontiguousarray(
            np.log((1.0 + hit_times.tp_values).astype(np.float64) * fee_two_sides)
        ),
        log_fac_sl_long=np.ascontiguousarray(
            np.log((1.0 - hit_times.sl_values).astype(np.float64) * fee_two_sides)
        ),
        log_fac_tp_short=np.ascontiguousarray(
            np.log((1.0 + hit_times.tp_values).astype(np.float64) * fee_two_sides)
        ),
        log_fac_sl_short=np.ascontiguousarray(
            np.log((1.0 - hit_times.sl_values).astype(np.float64) * fee_two_sides)
        ),
        log_fee_two_sides=float(math.log(fee_two_sides)),
        close_on_end=execution_settings.close_on_end,
        initial_cash_quote=execution_settings.initial_cash_quote,
        sizing_mode_code=execution_settings.sizing_mode_code,
        quote_amount=execution_settings.quote_amount,
        equity_pct=execution_settings.equity_pct,
        min_quote=execution_settings.min_quote,
        max_quote=execution_settings.max_quote,
        safe_profit_percent=execution_settings.safe_profit_percent,
        use_profit_lock=execution_settings.use_profit_lock,
    )


def _direction_mode_code(direction_mode: str) -> np.int8:
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        return DIRECTION_MODE_LONG_ONLY_CODE
    if direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        return DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE
    raise BacktestTpSlExactRejected(
        f"Unsupported direction_mode={direction_mode!r}; expected "
        f"{(DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL)!r}"
    )


def _backend_logical_name(*, arity: int) -> str:
    return f"event_segments_{arity}_tp_sl_15m_grid"


__all__ = [
    "TP_SL_EXACT_BOUNDARY_STAGE_NAME",
    "TP_SL_EXACT_SCORED_STATUS",
    "TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME",
    "TP_SL_EXACT_SCORING_STAGE_NAME",
    "TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME",
    "TP_SL_HEAP_UPDATE_STAGE_NAME",
    "TP_SL_SELF_CHECK_NOT_RUN_STATUS",
    "TP_SL_SELF_CHECK_PASSED_STATUS",
    "TP_SL_SELF_CHECK_STAGE_NAME",
    "BacktestTpSlExactRejected",
    "BacktestTpSlExactScoringService",
    "BacktestTpSlSelfCheckFailed",
    "build_trade_list_15m_for_indicator_rows_slow",
    "evaluate_tp_sl_exact_chunk",
    "evaluate_tp_sl_reference_rows_slow",
    "evaluate_tp_sl_reference_trade_list_direct",
    "event_segments_n_tp_sl_15m_grid",
    "run_tp_sl_fast_vs_reference_self_check",
]
