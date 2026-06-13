from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestComboChunk,
    BacktestComboPlanningConfig,
    BacktestComboPlanningResult,
    BacktestComboPlanningTelemetry,
    BacktestExactContext,
    BacktestPreparePoolsResult,
    BacktestProxyContext,
    BacktestProxyFilterResult,
    BacktestSelectedBackend,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import topk_fraction_idx

EVENT_SEGMENTS_2_NO_RISK_BACKEND = "event_segments_2_no_risk"
STREAMING_2_NO_RISK_BACKEND = "streaming_2_no_risk"
EVENT_SEGMENTS_N_NO_RISK_BACKEND = "event_segments_n_no_risk"
MATRIX_BITSET_NO_RISK_V1_BACKEND = "matrix_bitset_no_risk_v1"
COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND = (
    "compiled_prefix_product_traversal_v1"
)
MATRIX_CELL_TP_SL_V1_BACKEND = "matrix_cell_tp_sl_v1"
EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND = "event_segments_n_tp_sl_15m_grid"
BUILD_EXACT_CONTEXT_STAGE_NAME = "build_exact_context"
BUILD_PROXY_CONTEXT_STAGE_NAME = "build_proxy_context"
COMBO_ITERATION_STAGE_NAME = "combo_iteration"
PROXY_FILTER_STAGE_NAME = "proxy_filter"
COMBO_CHUNK_SIZE = 4096
LEGACY_PRODUCT_HELPER_MAX_COMBINATIONS = 1_000_000
NEG_INF = np.float32(-1e30)

_SUPPORTED_DIRECTIONS = ("long_only", "long_short_reversal")
_SUPPORTED_ARITIES = tuple(range(1, 11))


class BacktestComboPlanningRejected(ValueError):
    """
    Deterministic internal rejection for unsupported combo-planning inputs.
    """


@dataclass(frozen=True, slots=True)
class _BackendDescriptor:
    backend_id: str
    risk_mode: str
    arities: tuple[int, ...]
    role: str
    requires_exact_context: bool

    def supports(self, *, risk_mode: str, arity: int) -> bool:
        return self.risk_mode == risk_mode and arity in self.arities


@dataclass(frozen=True, slots=True)
class BacktestBackendRegistry:
    """
    Internal v1 backend registry for combo planning and later exact dispatch.
    """

    descriptors: tuple[_BackendDescriptor, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> BacktestBackendRegistry:
        return cls(
            descriptors=(
                _BackendDescriptor(
                    backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
                    risk_mode="none",
                    arities=(2,),
                    role="default",
                    requires_exact_context=False,
                ),
                _BackendDescriptor(
                    backend_id=STREAMING_2_NO_RISK_BACKEND,
                    risk_mode="none",
                    arities=(2,),
                    role="fallback",
                    requires_exact_context=False,
                ),
                _BackendDescriptor(
                    backend_id=EVENT_SEGMENTS_N_NO_RISK_BACKEND,
                    risk_mode="none",
                    arities=(1, 3, 4, 5, 6, 7, 8, 9, 10),
                    role="generic",
                    requires_exact_context=True,
                ),
                _BackendDescriptor(
                    backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
                    risk_mode="none",
                    arities=(2, 3, 6),
                    role="matrix_mvp",
                    requires_exact_context=False,
                ),
                _BackendDescriptor(
                    backend_id=COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND,
                    risk_mode="none",
                    arities=(6, 7),
                    role="compiled_prefix_traversal",
                    requires_exact_context=False,
                ),
                _BackendDescriptor(
                    backend_id=EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
                    risk_mode="tp_sl_grid",
                    arities=_SUPPORTED_ARITIES,
                    role="generic",
                    requires_exact_context=True,
                ),
                _BackendDescriptor(
                    backend_id=MATRIX_CELL_TP_SL_V1_BACKEND,
                    risk_mode="tp_sl_grid",
                    arities=_SUPPORTED_ARITIES,
                    role="matrix_full_grid",
                    requires_exact_context=True,
                ),
            )
        )

    def select(
        self,
        *,
        risk_mode: str,
        arity: int,
        direction_mode: str,
        requested_backend_id: str | None = None,
    ) -> BacktestSelectedBackend:
        if direction_mode not in _SUPPORTED_DIRECTIONS:
            raise BacktestComboPlanningRejected(
                f"Unsupported direction_mode={direction_mode!r}; expected one of "
                f"{_SUPPORTED_DIRECTIONS!r}"
            )
        if arity not in _SUPPORTED_ARITIES:
            raise BacktestComboPlanningRejected(
                f"Unsupported indicator arity={arity}; expected 1..10"
            )

        descriptor = (
            self._select_requested(
                requested_backend_id=requested_backend_id,
                risk_mode=risk_mode,
                arity=arity,
            )
            if requested_backend_id is not None
            else self._select_default(risk_mode=risk_mode, arity=arity)
        )
        return BacktestSelectedBackend(
            backend_id=descriptor.backend_id,
            risk_mode=risk_mode,
            arity=arity,
            direction_mode=direction_mode,
            requires_exact_context=descriptor.requires_exact_context,
            role=descriptor.role,
        )

    def _select_default(self, *, risk_mode: str, arity: int) -> _BackendDescriptor:
        if risk_mode == "none" and arity == 2:
            return self._descriptor(EVENT_SEGMENTS_2_NO_RISK_BACKEND)
        if risk_mode == "none":
            return self._descriptor(EVENT_SEGMENTS_N_NO_RISK_BACKEND)
        if risk_mode == "tp_sl_grid":
            return self._descriptor(EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND)
        raise BacktestComboPlanningRejected(
            f"Unsupported risk_mode={risk_mode!r}; expected 'none' or 'tp_sl_grid'"
        )

    def _select_requested(
        self,
        *,
        requested_backend_id: str,
        risk_mode: str,
        arity: int,
    ) -> _BackendDescriptor:
        descriptor = self._descriptor(requested_backend_id)
        if not descriptor.supports(risk_mode=risk_mode, arity=arity):
            raise BacktestComboPlanningRejected(
                f"Backend {requested_backend_id!r} does not support "
                f"risk_mode={risk_mode!r}, arity={arity}"
            )
        return descriptor

    def _descriptor(self, backend_id: str) -> _BackendDescriptor:
        for descriptor in self.descriptors:
            if descriptor.backend_id == backend_id:
                return descriptor
        raise BacktestComboPlanningRejected(f"Unsupported backend_id={backend_id!r}")


@dataclass(frozen=True, slots=True)
class BacktestComboPlanningService:
    """
    Internal application service for Iteration 3 combo planning contexts.
    """

    config: BacktestComboPlanningConfig = BacktestComboPlanningConfig()
    backend_registry: BacktestBackendRegistry = field(
        default_factory=BacktestBackendRegistry.default
    )

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        normalized_request: Mapping[str, Any],
        requested_backend_id: str | None = None,
    ) -> BacktestComboPlanningResult:
        indicator_ids = tuple(prepared_result.indicator_ids)
        backend = self.backend_registry.select(
            risk_mode=_risk_mode_from_normalized(normalized_request),
            arity=len(indicator_ids),
            direction_mode=_direction_mode_from_normalized(normalized_request),
            requested_backend_id=requested_backend_id,
        )
        stage_timings = _zero_stage_timings()

        stage_start = time.perf_counter()
        exact_context = self.build_exact_context(
            prepared_result=prepared_result,
            backend=backend,
        )
        stage_timings[BUILD_EXACT_CONTEXT_STAGE_NAME] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        proxy_context = self.build_proxy_context(
            prepared_result=prepared_result,
            fee_rate=_fee_rate_from_normalized(normalized_request),
        )
        stage_timings[BUILD_PROXY_CONTEXT_STAGE_NAME] = time.perf_counter() - stage_start

        local_row_pools = build_local_row_pools(prepared_result=prepared_result)
        cartesian_combinations = cartesian_combo_count(
            indicator_ids=indicator_ids,
            local_row_pools=local_row_pools,
        )
        if not proxy_context.active:
            return BacktestComboPlanningResult(
                backend=backend,
                exact_context=exact_context,
                proxy_context=proxy_context,
                telemetry=BacktestComboPlanningTelemetry(
                    stage_timings=stage_timings,
                    cartesian_combinations=cartesian_combinations,
                    combo_chunks_processed=0,
                    exact_candidates_evaluated=cartesian_combinations,
                    proxy_candidates_seen=cartesian_combinations,
                    proxy_candidates_valid=cartesian_combinations,
                    proxy_candidates_selected=cartesian_combinations,
                    combo_iteration_mode="ordinal_streaming_pass_through",
                    streamed_candidate_count=cartesian_combinations,
                    materialized_candidate_count=0,
                ),
            )

        combo_iter = iter_ordinal_combo_chunks(
            indicator_ids=indicator_ids,
            local_row_pools=local_row_pools,
            chunk_size=self.config.combo_chunk_size,
        )

        combo_chunks_processed = 0
        exact_candidates_evaluated = 0
        proxy_candidates_seen = 0
        proxy_candidates_valid = 0
        proxy_candidates_selected = 0
        while True:
            stage_start = time.perf_counter()
            try:
                combo_chunk = next(combo_iter)
            except StopIteration:
                stage_timings[COMBO_ITERATION_STAGE_NAME] += time.perf_counter() - stage_start
                break
            stage_timings[COMBO_ITERATION_STAGE_NAME] += time.perf_counter() - stage_start
            combo_chunks_processed += 1

            stage_start = time.perf_counter()
            filter_result = self.proxy_filter(
                combo_chunk=combo_chunk,
                proxy_context=proxy_context,
            )
            stage_timings[PROXY_FILTER_STAGE_NAME] += time.perf_counter() - stage_start
            proxy_candidates_seen += filter_result.input_candidate_count
            proxy_candidates_valid += filter_result.valid_candidate_count
            proxy_candidates_selected += filter_result.selected_candidate_count
            exact_candidates_evaluated += filter_result.selected_candidate_count

        return BacktestComboPlanningResult(
            backend=backend,
            exact_context=exact_context,
            proxy_context=proxy_context,
            telemetry=BacktestComboPlanningTelemetry(
                stage_timings=stage_timings,
                cartesian_combinations=cartesian_combinations,
                combo_chunks_processed=combo_chunks_processed,
                exact_candidates_evaluated=exact_candidates_evaluated,
                proxy_candidates_seen=proxy_candidates_seen,
                proxy_candidates_valid=proxy_candidates_valid,
                proxy_candidates_selected=proxy_candidates_selected,
                combo_iteration_mode="ordinal_proxy_filter",
                streamed_candidate_count=cartesian_combinations,
                materialized_candidate_count=proxy_candidates_seen,
            ),
        )

    def build_exact_context(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        backend: BacktestSelectedBackend,
    ) -> BacktestExactContext:
        pools = _ordered_pools(prepared_result=prepared_result)
        row_counts = tuple(int(pool.trade_T.shape[0]) for pool in pools)
        segment_widths = tuple(int(pool.segments.starts.shape[1]) for pool in pools)
        max_rows = max(row_counts)
        max_segments = max(segment_widths)
        if not backend.requires_exact_context:
            return BacktestExactContext(
                indicator_ids=tuple(prepared_result.indicator_ids),
                required=False,
                starts=None,
                ends=None,
                values=None,
                counts=None,
                row_counts=row_counts,
                max_rows=max_rows,
                max_segments=max_segments,
            )
        return build_segment_stack(
            indicator_ids=prepared_result.indicator_ids,
            indicator_pools=pools,
        )

    def build_proxy_context(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        fee_rate: float,
    ) -> BacktestProxyContext:
        indicator_ids = tuple(prepared_result.indicator_ids)
        fee_penalty_per_confirm = np.float32(self.config.fee_penalty_multiplier * fee_rate)
        if not self.config.proxy_filter_active:
            return BacktestProxyContext(
                indicator_ids=indicator_ids,
                active=False,
                context_type="pass_through",
                combo_top_frac=self.config.combo_top_frac,
                combo_min_confirm=self.config.combo_min_confirm,
                fee_penalty_per_confirm=fee_penalty_per_confirm,
            )

        pools = _ordered_pools(prepared_result=prepared_result)
        if len(indicator_ids) == 2:
            confirm_matrix, proxy_matrix = build_combo_proxy_cache_two(
                left_eval_T=pools[0].eval_T,
                right_eval_T=pools[1].eval_T,
                ret_15m=prepared_result.signal_returns_15m,
                min_confirm=self.config.combo_min_confirm,
                fee_penalty_per_confirm=fee_penalty_per_confirm,
            )
            return BacktestProxyContext(
                indicator_ids=indicator_ids,
                active=True,
                context_type="matrix_two",
                combo_top_frac=self.config.combo_top_frac,
                combo_min_confirm=self.config.combo_min_confirm,
                fee_penalty_per_confirm=fee_penalty_per_confirm,
                confirm_matrix=confirm_matrix,
                proxy_matrix=proxy_matrix,
            )

        return BacktestProxyContext(
            indicator_ids=indicator_ids,
            active=True,
            context_type="generic_n",
            combo_top_frac=self.config.combo_top_frac,
            combo_min_confirm=self.config.combo_min_confirm,
            fee_penalty_per_confirm=fee_penalty_per_confirm,
            eval_stack=build_eval_stack(
                indicator_ids=indicator_ids,
                indicator_pools=pools,
            ),
            ret_15m=np.ascontiguousarray(
                np.asarray(prepared_result.signal_returns_15m, dtype=np.float32)
            ),
        )

    def proxy_filter(
        self,
        *,
        combo_chunk: BacktestComboChunk,
        proxy_context: BacktestProxyContext,
    ) -> BacktestProxyFilterResult:
        chunk_len = combo_chunk.size
        if not proxy_context.active:
            selected_indexes = np.arange(chunk_len, dtype=np.int32)
            return BacktestProxyFilterResult(
                indicator_ids=combo_chunk.indicator_ids,
                selected_indexes=selected_indexes,
                selected_rows_by_indicator={
                    indicator_id: combo_chunk.rows_by_indicator[indicator_id]
                    for indicator_id in combo_chunk.indicator_ids
                },
                input_candidate_count=chunk_len,
                valid_candidate_count=chunk_len,
                selected_candidate_count=chunk_len,
            )

        out_confirm = np.empty(chunk_len, dtype=np.int32)
        out_proxy = np.empty(chunk_len, dtype=np.float32)
        if proxy_context.context_type == "matrix_two":
            if proxy_context.confirm_matrix is None or proxy_context.proxy_matrix is None:
                raise BacktestComboPlanningRejected("matrix_two proxy context is incomplete")
            out_confirm, out_proxy = gather_combo_proxy_cache_two(
                combo_chunk=combo_chunk,
                combo_proxy_cache=(proxy_context.confirm_matrix, proxy_context.proxy_matrix),
                indicator_ids=combo_chunk.indicator_ids,
            )
        elif proxy_context.context_type == "generic_n":
            if proxy_context.eval_stack is None:
                raise BacktestComboPlanningRejected("generic_n proxy context is incomplete")
            combo_idx_by_indicator = make_combo_idx_matrix(
                combo_chunk=combo_chunk,
                indicator_ids=combo_chunk.indicator_ids,
            )
            proxy_prefilter_combos_chunk_n(
                combo_idx_by_indicator=combo_idx_by_indicator,
                eval_stack=proxy_context.eval_stack,
                ret_15m=_proxy_context_returns(proxy_context),
                min_confirm=proxy_context.combo_min_confirm,
                fee_penalty_per_confirm=proxy_context.fee_penalty_per_confirm,
                out_confirm=out_confirm,
                out_proxy=out_proxy,
            )
        else:
            raise BacktestComboPlanningRejected(
                f"Unsupported proxy context type={proxy_context.context_type!r}"
            )

        valid_idx = np.flatnonzero(out_proxy > NEG_INF / np.float32(2.0))
        if int(valid_idx.size) == 0:
            selected_indexes = np.empty(0, dtype=np.int32)
        else:
            keep_local = topk_fraction_idx(out_proxy[valid_idx], proxy_context.combo_top_frac)
            selected_indexes = np.sort(valid_idx[keep_local].astype(np.int32))
        return BacktestProxyFilterResult(
            indicator_ids=combo_chunk.indicator_ids,
            selected_indexes=selected_indexes,
            selected_rows_by_indicator={
                indicator_id: combo_chunk.rows_by_indicator[indicator_id][selected_indexes]
                for indicator_id in combo_chunk.indicator_ids
            },
            input_candidate_count=chunk_len,
            valid_candidate_count=int(valid_idx.size),
            selected_candidate_count=int(selected_indexes.size),
            confirm=out_confirm[selected_indexes],
            proxy=out_proxy[selected_indexes],
        )


def build_segment_stack(
    *,
    indicator_ids: Sequence[str],
    indicator_pools: Sequence[PreparedIndicatorPool],
) -> BacktestExactContext:
    """
    Notebook-compatible `build_segment_stack` with arity-first arrays.
    """

    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    pools = _ordered_pools_from_sequence(indicator_ids=ids, indicator_pools=indicator_pools)
    row_counts = [int(pool.trade_T.shape[0]) for pool in pools]
    segment_widths = [int(pool.segments.starts.shape[1]) for pool in pools]
    max_rows = max(row_counts)
    max_segments = max(segment_widths)
    arity = len(ids)

    starts = np.zeros((arity, max_rows, max_segments), dtype=np.int32)
    ends = np.zeros((arity, max_rows, max_segments), dtype=np.int32)
    values = np.zeros((arity, max_rows, max_segments), dtype=np.int8)
    counts = np.zeros((arity, max_rows), dtype=np.int32)

    for indicator_pos, pool in enumerate(pools):
        row_count = row_counts[indicator_pos]
        width = segment_widths[indicator_pos]
        starts[indicator_pos, :row_count, :width] = pool.segments.starts
        ends[indicator_pos, :row_count, :width] = pool.segments.ends
        values[indicator_pos, :row_count, :width] = pool.segments.values
        counts[indicator_pos, :row_count] = pool.segments.counts

    return BacktestExactContext(
        indicator_ids=ids,
        required=True,
        starts=np.ascontiguousarray(starts),
        ends=np.ascontiguousarray(ends),
        values=np.ascontiguousarray(values),
        counts=np.ascontiguousarray(counts),
        row_counts=tuple(row_counts),
        max_rows=max_rows,
        max_segments=max_segments,
    )


def build_eval_stack(
    *,
    indicator_ids: Sequence[str],
    indicator_pools: Sequence[PreparedIndicatorPool],
) -> np.ndarray:
    """
    Notebook-compatible `build_eval_stack` with `eval_stack[arity, max_rows, n_intervals]`.
    """

    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    pools = _ordered_pools_from_sequence(indicator_ids=ids, indicator_pools=indicator_pools)
    row_counts = [int(pool.eval_T.shape[0]) for pool in pools]
    interval_counts = {int(pool.eval_T.shape[1]) for pool in pools}
    if len(interval_counts) != 1:
        raise BacktestComboPlanningRejected("all eval_T arrays must share interval length")
    max_rows = max(row_counts)
    n_intervals = interval_counts.pop()
    eval_stack = np.zeros((len(ids), max_rows, n_intervals), dtype=np.int8)
    for indicator_pos, pool in enumerate(pools):
        rows = row_counts[indicator_pos]
        eval_stack[indicator_pos, :rows, :] = pool.eval_T
    return np.ascontiguousarray(eval_stack)


def build_combo_proxy_cache_two(
    *,
    left_eval_T: np.ndarray,
    right_eval_T: np.ndarray,
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matrix-backed confirm/proxy lookup tables for active two-indicator pruning.
    """

    ret = np.asarray(ret_15m, dtype=np.float32)
    left_pos = (left_eval_T == 1).astype(np.float32)
    left_neg = (left_eval_T == -1).astype(np.float32)
    right_pos = (right_eval_T == 1).astype(np.float32)
    right_neg = (right_eval_T == -1).astype(np.float32)

    proxy_matrix = left_pos @ np.ascontiguousarray((right_pos * ret).T)
    proxy_matrix -= left_neg @ np.ascontiguousarray((right_neg * ret).T)
    confirm_matrix = left_pos @ np.ascontiguousarray(right_pos.T)
    confirm_matrix += left_neg @ np.ascontiguousarray(right_neg.T)
    confirm_matrix = np.rint(confirm_matrix).astype(np.int32)
    proxy_matrix = proxy_matrix.astype(np.float32, copy=False)
    proxy_matrix -= fee_penalty_per_confirm * confirm_matrix.astype(np.float32)
    proxy_matrix[confirm_matrix < int(min_confirm)] = NEG_INF
    return np.ascontiguousarray(confirm_matrix), np.ascontiguousarray(proxy_matrix)


def gather_combo_proxy_cache_two(
    *,
    combo_chunk: BacktestComboChunk,
    combo_proxy_cache: tuple[np.ndarray, np.ndarray],
    indicator_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather confirm/proxy vectors for one two-indicator chunk.
    """

    confirm_matrix, proxy_matrix = combo_proxy_cache
    left_id, right_id = tuple(indicator_ids)
    left_idx = combo_chunk.rows_by_indicator[left_id]
    right_idx = combo_chunk.rows_by_indicator[right_id]
    return (
        np.ascontiguousarray(confirm_matrix[left_idx, right_idx]),
        np.ascontiguousarray(proxy_matrix[left_idx, right_idx]),
    )


def proxy_prefilter_combos_chunk_two(
    *,
    combo_left_idx: np.ndarray,
    combo_right_idx: np.ndarray,
    left_eval_T: np.ndarray,
    right_eval_T: np.ndarray,
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
    out_confirm: np.ndarray,
    out_proxy: np.ndarray,
) -> None:
    """
    Direct two-indicator proxy scorer matching notebook semantics.
    """

    returns = np.asarray(ret_15m, dtype=np.float32)
    for combo_pos in range(int(combo_left_idx.shape[0])):
        left_row = int(combo_left_idx[combo_pos])
        right_row = int(combo_right_idx[combo_pos])
        confirms = 0
        proxy = np.float32(0.0)
        for interval_idx in range(int(returns.shape[0])):
            dirn = _consensus_dir2(
                left_eval_T[left_row, interval_idx],
                right_eval_T[right_row, interval_idx],
            )
            if dirn == 1:
                confirms += 1
                proxy += returns[interval_idx]
            elif dirn == -1:
                confirms += 1
                proxy -= returns[interval_idx]
        out_confirm[combo_pos] = np.int32(confirms)
        out_proxy[combo_pos] = (
            proxy - fee_penalty_per_confirm * np.float32(confirms)
            if confirms >= int(min_confirm)
            else NEG_INF
        )


def proxy_prefilter_combos_chunk_n(
    *,
    combo_idx_by_indicator: np.ndarray,
    eval_stack: np.ndarray,
    ret_15m: np.ndarray,
    min_confirm: int,
    fee_penalty_per_confirm: np.float32,
    out_confirm: np.ndarray,
    out_proxy: np.ndarray,
) -> None:
    """
    Generic N-indicator proxy scorer over `eval_stack`.
    """

    returns = np.asarray(ret_15m, dtype=np.float32)
    arity = int(combo_idx_by_indicator.shape[0])
    chunk_len = int(combo_idx_by_indicator.shape[1])
    for combo_pos in range(chunk_len):
        confirms = 0
        proxy = np.float32(0.0)
        for interval_idx in range(int(returns.shape[0])):
            first_row = int(combo_idx_by_indicator[0, combo_pos])
            dirn = eval_stack[0, first_row, interval_idx]
            if dirn == 0:
                continue
            for indicator_pos in range(1, arity):
                row_idx = int(combo_idx_by_indicator[indicator_pos, combo_pos])
                if eval_stack[indicator_pos, row_idx, interval_idx] != dirn:
                    dirn = np.int8(0)
                    break
            if dirn == 1:
                confirms += 1
                proxy += returns[interval_idx]
            elif dirn == -1:
                confirms += 1
                proxy -= returns[interval_idx]
        out_confirm[combo_pos] = np.int32(confirms)
        out_proxy[combo_pos] = (
            proxy - fee_penalty_per_confirm * np.float32(confirms)
            if confirms >= int(min_confirm)
            else NEG_INF
        )


def iter_combo_chunks(
    *,
    indicator_ids: Sequence[str],
    local_row_pools: Mapping[str, np.ndarray],
    chunk_size: int,
) -> Iterator[BacktestComboChunk]:
    """
    Yield small bounded Cartesian chunks in `itertools.product` order.

    This is retained as a reference/test helper. Production full-job scoring must
    use ordinal streaming so large grids cannot enter Python object Cartesian
    generation.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    if not ids:
        raise ValueError("indicator_ids must not be empty")
    total_count = cartesian_combo_count(
        indicator_ids=ids,
        local_row_pools=local_row_pools,
    )
    if total_count > LEGACY_PRODUCT_HELPER_MAX_COMBINATIONS:
        raise ValueError(
            "iter_combo_chunks is limited to small reference grids; use "
            "iter_ordinal_combo_chunks for production-sized Cartesian spaces"
        )
    buffers: dict[str, list[int]] = {indicator_id: [] for indicator_id in ids}
    ordered_pools = tuple(
        np.asarray(local_row_pools[indicator_id], dtype=np.int32)
        for indicator_id in ids
    )
    for combo in itertools.product(*ordered_pools):
        for indicator_id, value in zip(ids, combo):
            buffers[indicator_id].append(int(value))
        if len(buffers[ids[0]]) >= chunk_size:
            yield BacktestComboChunk(
                indicator_ids=ids,
                rows_by_indicator={
                    indicator_id: np.asarray(buffers[indicator_id], dtype=np.int32)
                    for indicator_id in ids
                },
            )
            buffers = {indicator_id: [] for indicator_id in ids}

    if buffers[ids[0]]:
        yield BacktestComboChunk(
            indicator_ids=ids,
            rows_by_indicator={
                indicator_id: np.asarray(buffers[indicator_id], dtype=np.int32)
                for indicator_id in ids
            },
        )


def iter_ordinal_combo_chunks(
    *,
    indicator_ids: Sequence[str],
    local_row_pools: Mapping[str, np.ndarray],
    chunk_size: int,
) -> Iterator[BacktestComboChunk]:
    """
    Yield bounded Cartesian chunks by decoding ordinal ranges.

    The decoded order matches `itertools.product(*ordered_pools)`: the last
    indicator varies fastest, the first varies slowest.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    if not ids:
        raise ValueError("indicator_ids must not be empty")
    pool_matrix, pool_sizes = _pool_decode_matrix(
        indicator_ids=ids,
        local_row_pools=local_row_pools,
    )
    total_count = cartesian_combo_count(
        indicator_ids=ids,
        local_row_pools=local_row_pools,
    )
    for start_ordinal in range(0, total_count, chunk_size):
        current_size = min(chunk_size, total_count - start_ordinal)
        decoded = np.empty((len(ids), current_size), dtype=np.int32)
        decode_ordinal_combo_chunk(
            pool_matrix,
            pool_sizes,
            np.int64(start_ordinal),
            decoded,
        )
        yield BacktestComboChunk(
            indicator_ids=ids,
            rows_by_indicator={
                indicator_id: decoded[pos]
                for pos, indicator_id in enumerate(ids)
            },
        )


def _pool_decode_matrix(
    *,
    indicator_ids: Sequence[str],
    local_row_pools: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    ordered_pools = tuple(
        np.ascontiguousarray(np.asarray(local_row_pools[indicator_id], dtype=np.int32))
        for indicator_id in ids
    )
    if any(int(pool.shape[0]) <= 0 for pool in ordered_pools):
        raise ValueError("all local row pools must be non-empty")
    max_rows = max(int(pool.shape[0]) for pool in ordered_pools)
    pool_matrix = np.zeros((len(ids), max_rows), dtype=np.int32)
    pool_sizes = np.empty(len(ids), dtype=np.int64)
    for pos, pool in enumerate(ordered_pools):
        pool_size = int(pool.shape[0])
        pool_matrix[pos, :pool_size] = pool
        pool_sizes[pos] = np.int64(pool_size)
    return np.ascontiguousarray(pool_matrix), pool_sizes


@nb.njit(cache=True)
def decode_ordinal_combo_chunk(
    pool_matrix: np.ndarray,
    pool_sizes: np.ndarray,
    start_ordinal: np.int64,
    out_rows_by_pos: np.ndarray,
) -> None:
    arity = int(pool_sizes.shape[0])
    chunk_len = int(out_rows_by_pos.shape[1])
    for chunk_pos in range(chunk_len):
        remaining = np.int64(start_ordinal + np.int64(chunk_pos))
        for indicator_pos in range(arity - 1, -1, -1):
            pool_size = np.int64(pool_sizes[indicator_pos])
            local_index = remaining % pool_size
            remaining = remaining // pool_size
            out_rows_by_pos[indicator_pos, chunk_pos] = pool_matrix[
                indicator_pos,
                local_index,
            ]


def make_combo_idx_matrix(
    *,
    combo_chunk: BacktestComboChunk,
    indicator_ids: Sequence[str],
) -> np.ndarray:
    """
    Convert a chunk dict into an arity x K int32 matrix.
    """

    return np.ascontiguousarray(
        np.vstack(
            [
                np.asarray(combo_chunk.rows_by_indicator[indicator_id], dtype=np.int32)
                for indicator_id in indicator_ids
            ]
        )
    )


def build_local_row_pools(
    *,
    prepared_result: BacktestPreparePoolsResult,
) -> dict[str, np.ndarray]:
    pools = _ordered_pools(prepared_result=prepared_result)
    return {
        indicator_id: np.arange(pool.trade_T.shape[0], dtype=np.int32)
        for indicator_id, pool in zip(prepared_result.indicator_ids, pools)
    }


def cartesian_combo_count(
    *,
    indicator_ids: Sequence[str],
    local_row_pools: Mapping[str, np.ndarray],
) -> int:
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    return int(math.prod(int(local_row_pools[indicator_id].shape[0]) for indicator_id in ids))


def _zero_stage_timings() -> dict[str, float]:
    return {
        BUILD_EXACT_CONTEXT_STAGE_NAME: 0.0,
        BUILD_PROXY_CONTEXT_STAGE_NAME: 0.0,
        COMBO_ITERATION_STAGE_NAME: 0.0,
        PROXY_FILTER_STAGE_NAME: 0.0,
    }


def _ordered_pools(
    *,
    prepared_result: BacktestPreparePoolsResult,
) -> tuple[PreparedIndicatorPool, ...]:
    return _ordered_pools_from_sequence(
        indicator_ids=prepared_result.indicator_ids,
        indicator_pools=prepared_result.indicator_pools,
    )


def _ordered_pools_from_sequence(
    *,
    indicator_ids: Sequence[str],
    indicator_pools: Sequence[PreparedIndicatorPool],
) -> tuple[PreparedIndicatorPool, ...]:
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    if not ids:
        raise BacktestComboPlanningRejected("at least one indicator is required")
    pool_by_id: dict[str, PreparedIndicatorPool] = {}
    for pool in indicator_pools:
        if pool.indicator_id in pool_by_id:
            raise BacktestComboPlanningRejected(f"duplicate pool for {pool.indicator_id!r}")
        pool_by_id[pool.indicator_id] = pool
    missing = [indicator_id for indicator_id in ids if indicator_id not in pool_by_id]
    if missing:
        raise BacktestComboPlanningRejected(f"prepared pools missing indicator ids: {missing!r}")
    pools = tuple(pool_by_id[indicator_id] for indicator_id in ids)
    for pool in pools:
        if int(pool.trade_T.shape[0]) == 0:
            raise BacktestComboPlanningRejected(f"pool {pool.indicator_id!r} has no rows")
        if int(pool.eval_T.shape[0]) != int(pool.trade_T.shape[0]):
            raise BacktestComboPlanningRejected(
                f"pool {pool.indicator_id!r} eval_T row count differs from trade_T"
            )
    return pools


def _risk_mode_from_normalized(normalized_request: Mapping[str, Any]) -> str:
    risk = normalized_request.get("risk")
    if not isinstance(risk, Mapping):
        raise BacktestComboPlanningRejected("normalized_request.risk must be a mapping")
    return str(risk.get("mode"))


def _direction_mode_from_normalized(normalized_request: Mapping[str, Any]) -> str:
    execution = normalized_request.get("execution")
    if not isinstance(execution, Mapping):
        raise BacktestComboPlanningRejected("normalized_request.execution must be a mapping")
    return str(execution.get("direction_mode"))


def _fee_rate_from_normalized(normalized_request: Mapping[str, Any]) -> float:
    execution = normalized_request.get("execution")
    if not isinstance(execution, Mapping):
        raise BacktestComboPlanningRejected("normalized_request.execution must be a mapping")
    return float(execution.get("fee_rate", 0.0))


def _proxy_context_returns(proxy_context: BacktestProxyContext) -> np.ndarray:
    returns = proxy_context.ret_15m
    if returns is None:
        raise BacktestComboPlanningRejected("generic_n proxy context missing returns")
    return np.asarray(returns, dtype=np.float32)


def _consensus_dir2(left_value: np.int8, right_value: np.int8) -> np.int8:
    if left_value == 1 and right_value == 1:
        return np.int8(1)
    if left_value == -1 and right_value == -1:
        return np.int8(-1)
    return np.int8(0)


__all__ = [
    "BUILD_EXACT_CONTEXT_STAGE_NAME",
    "BUILD_PROXY_CONTEXT_STAGE_NAME",
    "COMBO_CHUNK_SIZE",
    "COMBO_ITERATION_STAGE_NAME",
    "COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND",
    "EVENT_SEGMENTS_2_NO_RISK_BACKEND",
    "EVENT_SEGMENTS_N_NO_RISK_BACKEND",
    "EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND",
    "LEGACY_PRODUCT_HELPER_MAX_COMBINATIONS",
    "MATRIX_CELL_TP_SL_V1_BACKEND",
    "NEG_INF",
    "PROXY_FILTER_STAGE_NAME",
    "STREAMING_2_NO_RISK_BACKEND",
    "BacktestBackendRegistry",
    "BacktestComboPlanningRejected",
    "BacktestComboPlanningService",
    "build_combo_proxy_cache_two",
    "build_eval_stack",
    "build_local_row_pools",
    "build_segment_stack",
    "cartesian_combo_count",
    "decode_ordinal_combo_chunk",
    "gather_combo_proxy_cache_two",
    "iter_combo_chunks",
    "iter_ordinal_combo_chunks",
    "make_combo_idx_matrix",
    "proxy_prefilter_combos_chunk_n",
    "proxy_prefilter_combos_chunk_two",
]
