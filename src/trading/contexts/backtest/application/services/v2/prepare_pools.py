from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparedSignalSegments,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.application.ports.artifact_arrays import (
    BacktestArtifactArrayLoader,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactMappingArraysV2,
    ArtifactPriceArraysV2,
    ArtifactSignalMatrixV2,
    ArtifactSlotPinnedRuntimeContextV2,
)
from trading.contexts.indicators.domain.specifications.grid_param_spec import GridValue

PREPARE_POOLS_CORE_STAGE_NAME = "prepare_pools_core"
PREPARE_POOLS_TOTAL_STAGE_NAME = "prepare_pools_total"
PREPARE_POOLS_STAGE_NAME = PREPARE_POOLS_TOTAL_STAGE_NAME
ARTIFACT_CONTEXT_RESOLVE_SEGMENT = "artifact_context_resolve"
ARTIFACT_ARRAY_OPEN_SEGMENT = "artifact_array_open"
REQUEST_SLICE_PREPARE_SEGMENT = "request_slice_prepare"
ARTIFACT_MANIFEST_LOAD_SEGMENT = "artifact_manifest_load"
ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT = "artifact_array_mmap_load"
TIME_RANGE_SLICE_SEGMENT = "time_range_slice"
SIGNAL_ROW_SELECTION_SEGMENT = "signal_row_selection"
ROW_PREFILTER_SEGMENT = "row_prefilter"
SEGMENT_BUILD_SEGMENT = "segment_build"
CANONICAL_BACKTEST_TIMEFRAME_V1 = "15m"
CANONICAL_EXECUTION_TIMEFRAME_V1 = "1m"
WINDOW_AXIS_NAME = "window"


class BacktestPreparePoolsRejected(ValueError):
    """
    Deterministic internal rejection for invalid artifact/runtime preparation inputs.
    """


@dataclass(frozen=True, slots=True)
class BacktestPreparePoolsRuntimeArrays:
    """
    Resolved artifact context and opened mmap handles for one normalized request.
    """

    context: ArtifactSlotPinnedRuntimeContextV2
    timeframe: str
    price_arrays_15m: ArtifactPriceArraysV2
    price_arrays_1m: ArtifactPriceArraysV2
    mapping_arrays: ArtifactMappingArraysV2
    signal_matrices: Mapping[str, ArtifactSignalMatrixV2]


@dataclass(frozen=True, slots=True)
class BacktestPreparePoolsRequestSlice:
    """
    Prepared request-time slice inputs consumed by notebook-compatible pool compute.
    """

    time_slice: slice
    signal_returns_15m: np.ndarray
    execution_mapping: PreparedExecutionMapping


@dataclass(frozen=True, slots=True)
class BacktestPreparePoolsService:
    """
    Internal application service for Iteration 2 `prepare_pools`.
    """

    artifact_array_loader: BacktestArtifactArrayLoader
    defaults_provider: BacktestGridDefaultsProvider
    config: BacktestPreparePoolsConfig = BacktestPreparePoolsConfig()

    def execute(
        self,
        *,
        normalized_request: Mapping[str, Any],
        artifact_metadata: BacktestArtifactMetadata,
    ) -> BacktestPreparePoolsResult:
        """
        Compatibility facade that measures aggregate service telemetry.

        The notebook-comparable scope is exposed by `prepare_pools_core`; this
        facade keeps resolve/open/slice overhead visible without folding it into
        the core benchmark target.
        """

        total_start = time.perf_counter()
        coordinates = _coordinates_from_normalized(normalized_request)
        timeframe = _timeframe_from_normalized(normalized_request)
        total_subsegments: dict[str, float] = {}

        segment_start = time.perf_counter()
        context = self.resolve_artifact_context(
            coordinates=coordinates,
            artifact_metadata=artifact_metadata,
        )
        _record_elapsed_aliases(
            total_subsegments,
            segment_start,
            ARTIFACT_CONTEXT_RESOLVE_SEGMENT,
            ARTIFACT_MANIFEST_LOAD_SEGMENT,
        )

        segment_start = time.perf_counter()
        runtime_arrays = self.open_artifact_arrays(
            normalized_request=normalized_request,
            context=context,
            timeframe=timeframe,
        )
        _record_elapsed_aliases(
            total_subsegments,
            segment_start,
            ARTIFACT_ARRAY_OPEN_SEGMENT,
            ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT,
        )

        segment_start = time.perf_counter()
        request_slice = self.prepare_request_slice(
            normalized_request=normalized_request,
            runtime_arrays=runtime_arrays,
        )
        _record_elapsed_aliases(
            total_subsegments,
            segment_start,
            REQUEST_SLICE_PREPARE_SEGMENT,
            TIME_RANGE_SLICE_SEGMENT,
        )

        core_result = self.prepare_pools_core(
            normalized_request=normalized_request,
            runtime_arrays=runtime_arrays,
            request_slice=request_slice,
        )
        total_subsegments.update(core_result.timing.subsegments)
        total_subsegments[PREPARE_POOLS_TOTAL_STAGE_NAME] = time.perf_counter() - total_start
        return replace(
            core_result,
            timing=PreparePoolsTiming(
                stage_name=PREPARE_POOLS_TOTAL_STAGE_NAME,
                wall_time_s=total_subsegments[PREPARE_POOLS_TOTAL_STAGE_NAME],
                subsegments=total_subsegments,
            ),
        )

    def resolve_artifact_context(
        self,
        *,
        coordinates: BacktestCoordinates,
        artifact_metadata: BacktestArtifactMetadata,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        return self.artifact_array_loader.resolve_context(
            coordinates=coordinates,
            artifact_metadata=artifact_metadata,
        )

    def open_artifact_arrays(
        self,
        *,
        normalized_request: Mapping[str, Any],
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str | None = None,
    ) -> BacktestPreparePoolsRuntimeArrays:
        opened_timeframe = (
            _timeframe_from_normalized(normalized_request)
            if timeframe is None
            else timeframe
        )
        indicator_requests = _indicator_requests_from_normalized(normalized_request)
        price_arrays_15m = self.artifact_array_loader.load_price_arrays(
            context=context,
            timeframe=opened_timeframe,
        )
        price_arrays_1m = self.artifact_array_loader.load_price_arrays(
            context=context,
            timeframe=CANONICAL_EXECUTION_TIMEFRAME_V1,
        )
        mapping_arrays = self.artifact_array_loader.load_mapping_arrays(
            context=context,
            timeframe=opened_timeframe,
        )
        signal_matrices = _load_signal_matrices(
            artifact_array_loader=self.artifact_array_loader,
            context=context,
            timeframe=opened_timeframe,
            indicator_requests=indicator_requests,
        )
        return BacktestPreparePoolsRuntimeArrays(
            context=context,
            timeframe=opened_timeframe,
            price_arrays_15m=price_arrays_15m,
            price_arrays_1m=price_arrays_1m,
            mapping_arrays=mapping_arrays,
            signal_matrices=signal_matrices,
        )

    def prepare_request_slice(
        self,
        *,
        normalized_request: Mapping[str, Any],
        runtime_arrays: BacktestPreparePoolsRuntimeArrays,
    ) -> BacktestPreparePoolsRequestSlice:
        time_slice = time_range_slice(
            open_time_15m=runtime_arrays.price_arrays_15m.open_time,
            close_time_15m=runtime_arrays.price_arrays_15m.close_time,
            time_range=_required_mapping(
                normalized_request,
                "time_range",
                path="normalized_request.time_range",
            ),
        )
        signal_returns_15m = _signal_returns_15m(
            ohlcv_15m=runtime_arrays.price_arrays_15m.ohlcv,
            time_slice=time_slice,
        )
        execution_mapping = _execution_mapping_no_risk_15m_to_1m(
            mapping_open_1m_idx_15m=runtime_arrays.mapping_arrays.bar_open_1m_idx,
            mapping_close_1m_idx_15m=runtime_arrays.mapping_arrays.bar_close_1m_idx,
            price_1m_length=int(runtime_arrays.price_arrays_1m.ohlcv.shape[0]),
            time_slice=time_slice,
        )
        return BacktestPreparePoolsRequestSlice(
            time_slice=time_slice,
            signal_returns_15m=signal_returns_15m,
            execution_mapping=execution_mapping,
        )

    def prepare_pools_core(
        self,
        *,
        normalized_request: Mapping[str, Any],
        runtime_arrays: BacktestPreparePoolsRuntimeArrays,
        request_slice: BacktestPreparePoolsRequestSlice,
    ) -> BacktestPreparePoolsResult:
        """
        Notebook-compatible `prepare_indicator_pools(...)` scope.
        """

        core_start = time.perf_counter()
        subsegments: dict[str, float] = {}
        timeframe = _timeframe_from_normalized(normalized_request)
        if runtime_arrays.timeframe != timeframe:
            raise BacktestPreparePoolsRejected(
                f"runtime arrays timeframe {runtime_arrays.timeframe!r} does not match "
                f"request timeframe {timeframe!r}"
        )
        indicator_requests = _indicator_requests_from_normalized(normalized_request)
        fee_rate = _fee_rate_from_normalized(normalized_request)

        segment_start = time.perf_counter()
        selected_rows = tuple(
            _select_indicator_rows(
                defaults_provider=self.defaults_provider,
                signal_matrix=runtime_arrays.signal_matrices[
                    indicator_request["indicator_id"]
                ],
                indicator_request=indicator_request,
                time_slice=request_slice.time_slice,
            )
            for indicator_request in indicator_requests
        )
        _record_elapsed(subsegments, SIGNAL_ROW_SELECTION_SEGMENT, segment_start)

        segment_start = time.perf_counter()
        filtered_rows = tuple(
            prefilter_indicator_rows(
                trade_T=selection.trade_T,
                indicator_id=selection.indicator_id,
                row_ids=selection.row_ids,
                metadata=selection.metadata,
                signal_returns_15m=request_slice.signal_returns_15m,
                top_frac=self.config.row_prefilter_top_fraction,
                min_nonzero=self.config.row_prefilter_min_nonzero,
                fee_rate=fee_rate,
                time_chunk=self.config.time_chunk,
            )
            for selection in selected_rows
        )
        _record_elapsed(subsegments, ROW_PREFILTER_SEGMENT, segment_start)

        segment_start = time.perf_counter()
        indicator_pools = tuple(
            prepare_indicator_pool(
                filtered_rows=filtered,
                segments=build_signal_segments(
                    filtered.trade_T,
                    change_count=filtered.change_count,
                ),
            )
            for filtered in filtered_rows
        )
        _record_elapsed(subsegments, SEGMENT_BUILD_SEGMENT, segment_start)
        subsegments[PREPARE_POOLS_CORE_STAGE_NAME] = time.perf_counter() - core_start

        result = BacktestPreparePoolsResult(
            timeframe=timeframe,
            indicator_ids=tuple(request["indicator_id"] for request in indicator_requests),
            indicator_pools=indicator_pools,
            signal_returns_15m=request_slice.signal_returns_15m,
            execution_mapping=request_slice.execution_mapping,
            time_slice_start_15m=int(request_slice.time_slice.start or 0),
            time_slice_stop_15m=int(request_slice.time_slice.stop or 0),
            trade_T_length=int(indicator_pools[0].trade_T_length),
            eval_T_length=int(indicator_pools[0].eval_T_length),
            row_metadata_order_hash=row_metadata_order_hash(indicator_pools),
            timing=PreparePoolsTiming(
                stage_name=PREPARE_POOLS_CORE_STAGE_NAME,
                wall_time_s=subsegments[PREPARE_POOLS_CORE_STAGE_NAME],
                subsegments=subsegments,
            ),
            execution_open_1m=np.ascontiguousarray(
                np.asarray(runtime_arrays.price_arrays_1m.ohlcv[:, 0], dtype=np.float32)
            ),
            execution_close_1m=np.ascontiguousarray(
                np.asarray(runtime_arrays.price_arrays_1m.ohlcv[:, 3], dtype=np.float32)
            ),
        )
        return result


@dataclass(frozen=True, slots=True)
class _SelectedSignalRows:
    indicator_id: str
    row_ids: np.ndarray
    trade_T: np.ndarray
    metadata: tuple[PreparedIndicatorRowMetadata, ...]


@dataclass(frozen=True, slots=True)
class _PrefilteredIndicatorRows:
    indicator_id: str
    row_ids: np.ndarray
    filtered_row_ids: np.ndarray
    trade_T: np.ndarray
    eval_T: np.ndarray
    row_score: np.ndarray
    score_adj: np.ndarray
    nonzero: np.ndarray
    proxy: np.ndarray
    change_count: np.ndarray
    metadata: tuple[PreparedIndicatorRowMetadata, ...]


def prepare_indicator_pools(
    *,
    selected_rows: Sequence[_SelectedSignalRows],
    signal_returns_15m: np.ndarray,
    config: BacktestPreparePoolsConfig,
    fee_rate: float,
) -> tuple[PreparedIndicatorPool, ...]:
    """
    Prepare multiple indicators into standard pools.
    """

    pools: list[PreparedIndicatorPool] = []
    for selection in selected_rows:
        filtered = prefilter_indicator_rows(
            trade_T=selection.trade_T,
            indicator_id=selection.indicator_id,
            row_ids=selection.row_ids,
            metadata=selection.metadata,
            signal_returns_15m=signal_returns_15m,
            top_frac=config.row_prefilter_top_fraction,
            min_nonzero=config.row_prefilter_min_nonzero,
            fee_rate=fee_rate,
            time_chunk=config.time_chunk,
        )
        pools.append(
            prepare_indicator_pool(
                filtered_rows=filtered,
                segments=build_signal_segments(
                    filtered.trade_T,
                    change_count=filtered.change_count,
                ),
            )
        )
    return tuple(pools)


def prepare_indicator_pool(
    *,
    filtered_rows: _PrefilteredIndicatorRows,
    segments: PreparedSignalSegments,
) -> PreparedIndicatorPool:
    """
    Build one prepared indicator pool from filtered rows and compressed segments.
    """

    if not np.array_equal(segments.change_count, filtered_rows.change_count):
        raise AssertionError(
            f"Segment change counts differ from fused row stats for "
            f"{filtered_rows.indicator_id!r}"
        )
    return PreparedIndicatorPool(
        indicator_id=filtered_rows.indicator_id,
        row_ids=filtered_rows.row_ids,
        filtered_row_ids=filtered_rows.filtered_row_ids,
        trade_T=filtered_rows.trade_T,
        eval_T=filtered_rows.eval_T,
        segments=segments,
        row_score=filtered_rows.row_score,
        score_adj=filtered_rows.score_adj,
        nonzero=filtered_rows.nonzero,
        proxy=filtered_rows.proxy,
        change_count=filtered_rows.change_count,
        metadata=filtered_rows.metadata,
    )


def prefilter_indicator_rows(
    *,
    trade_T: np.ndarray,
    indicator_id: str,
    row_ids: np.ndarray,
    metadata: Sequence[PreparedIndicatorRowMetadata],
    signal_returns_15m: np.ndarray,
    top_frac: float,
    min_nonzero: int,
    fee_rate: float,
    time_chunk: int,
) -> _PrefilteredIndicatorRows:
    """
    Notebook-equivalent `fused_row_prefilter_stats` plus `topk_fraction_idx`.
    """

    del time_chunk
    row_ids_i32 = np.asarray(row_ids, dtype=np.int32)
    if trade_T.ndim != 2 or int(trade_T.shape[0]) == 0:
        raise ValueError(f"trade_T for {indicator_id!r} must be a non-empty 2D matrix")
    if int(trade_T.shape[0]) != int(row_ids_i32.shape[0]):
        raise ValueError(f"Row id alignment mismatch for {indicator_id!r}")
    if len(metadata) != int(row_ids_i32.shape[0]):
        raise ValueError(f"Metadata alignment mismatch for {indicator_id!r}")
    if int(signal_returns_15m.shape[0]) > int(trade_T.shape[1]):
        raise ValueError("signal return intervals cannot exceed trade_T length")

    nonzero = np.empty(trade_T.shape[0], dtype=np.int32)
    proxy = np.empty(trade_T.shape[0], dtype=np.float32)
    change_count = np.empty(trade_T.shape[0], dtype=np.int32)
    fused_row_prefilter_stats(trade_T, signal_returns_15m, nonzero, proxy, change_count)

    adjusted = proxy - (np.float32(fee_rate) * nonzero.astype(np.float32))
    valid = nonzero >= int(min_nonzero)
    if not np.any(valid):
        raise BacktestPreparePoolsRejected(
            f"No rows survive min_nonzero={min_nonzero} for {indicator_id!r}"
        )

    valid_idx = np.flatnonzero(valid)
    keep_from_valid = topk_fraction_idx(adjusted[valid_idx], top_frac)
    keep_idx = np.sort(valid_idx[keep_from_valid].astype(np.int32))

    filtered_row_ids = np.ascontiguousarray(row_ids_i32[keep_idx])
    filtered_trade_T = np.ascontiguousarray(trade_T[keep_idx])
    filtered_eval_T = filtered_trade_T[:, : int(signal_returns_15m.shape[0])]
    filtered_metadata = tuple(metadata[int(index)] for index in keep_idx)
    row_score = np.ascontiguousarray(adjusted[keep_idx])
    return _PrefilteredIndicatorRows(
        indicator_id=indicator_id,
        row_ids=filtered_row_ids,
        filtered_row_ids=filtered_row_ids,
        trade_T=filtered_trade_T,
        eval_T=filtered_eval_T,
        row_score=row_score,
        score_adj=row_score,
        nonzero=np.ascontiguousarray(nonzero[keep_idx]),
        proxy=np.ascontiguousarray(proxy[keep_idx]),
        change_count=np.ascontiguousarray(change_count[keep_idx]),
        metadata=filtered_metadata,
    )


def topk_fraction_idx(score: np.ndarray, frac: float) -> np.ndarray:
    """
    Select deterministic original-order top indices by score fraction.
    """

    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac!r}")
    n = int(score.shape[0])
    if n <= 0:
        raise ValueError("score must contain at least one element")
    k = max(1, int(math.ceil(n * frac)))
    if k >= n:
        return np.arange(n, dtype=np.int32)
    idx = np.argpartition(score, n - k)[n - k :]
    return np.sort(idx.astype(np.int32))


@nb.njit(parallel=True, cache=True, fastmath=False)
def fused_row_prefilter_stats(
    trade_T: np.ndarray,
    ret_15m: np.ndarray,
    out_nonzero: np.ndarray,
    out_proxy: np.ndarray,
    out_change_count: np.ndarray,
) -> None:
    """
    Compute row nonzero count, directional return proxy, and signal change count.
    """

    n_rows = trade_T.shape[0]
    n_sig = trade_T.shape[1]
    n_intervals = ret_15m.shape[0]

    for row_idx in nb.prange(n_rows):
        nonzero = np.int32(0)
        proxy = np.float32(0.0)
        change_count = np.int32(0)

        for t in range(n_intervals):
            value = trade_T[row_idx, t]
            if value != 0:
                nonzero += 1
                proxy += np.float32(value) * ret_15m[t]
            next_t = t + 1
            if next_t < n_sig and trade_T[row_idx, next_t] != value:
                change_count += 1

        for t in range(n_intervals + 1, n_sig):
            if trade_T[row_idx, t] != trade_T[row_idx, t - 1]:
                change_count += 1

        out_nonzero[row_idx] = nonzero
        out_proxy[row_idx] = proxy
        out_change_count[row_idx] = change_count


@nb.njit(parallel=True, cache=True)
def fill_signal_segments_i8(
    trade_T: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
) -> None:
    """
    Fill padded compressed int8 signal segments.
    """

    n_rows = trade_T.shape[0]
    n_sig = trade_T.shape[1]

    for row_idx in nb.prange(n_rows):
        segment_idx = 0
        segment_start = np.int32(0)
        current_value = trade_T[row_idx, 0]
        for t in range(1, n_sig):
            value = trade_T[row_idx, t]
            if value != current_value:
                starts[row_idx, segment_idx] = segment_start
                ends[row_idx, segment_idx] = np.int32(t)
                values[row_idx, segment_idx] = current_value
                segment_idx += 1
                segment_start = np.int32(t)
                current_value = value

        starts[row_idx, segment_idx] = segment_start
        ends[row_idx, segment_idx] = np.int32(n_sig)
        values[row_idx, segment_idx] = current_value
        counts[row_idx] = np.int32(segment_idx + 1)


def build_signal_segments(
    trade_T: np.ndarray,
    *,
    change_count: np.ndarray | None = None,
) -> PreparedSignalSegments:
    """
    Build padded compressed signal segments for one filtered int8 matrix.
    """

    if trade_T.ndim != 2 or int(trade_T.shape[0]) == 0 or int(trade_T.shape[1]) == 0:
        raise ValueError("trade_T must be a non-empty 2D signal matrix")
    if change_count is None:
        resolved_change_count = (trade_T[:, 1:] != trade_T[:, :-1]).sum(axis=1).astype(
            np.int32
        )
    else:
        resolved_change_count = np.asarray(change_count, dtype=np.int32)
        if resolved_change_count.ndim != 1 or int(resolved_change_count.shape[0]) != int(
            trade_T.shape[0]
        ):
            raise ValueError("change_count must align to trade_T rows")
        resolved_change_count = np.ascontiguousarray(resolved_change_count)
    counts_expected = resolved_change_count + np.int32(1)
    max_segments = int(counts_expected.max())
    starts = np.zeros((trade_T.shape[0], max_segments), dtype=np.int32)
    ends = np.zeros((trade_T.shape[0], max_segments), dtype=np.int32)
    values = np.zeros((trade_T.shape[0], max_segments), dtype=np.int8)
    counts = np.zeros(trade_T.shape[0], dtype=np.int32)
    fill_signal_segments_i8(trade_T, starts, ends, values, counts)
    if not np.array_equal(counts, counts_expected):
        raise AssertionError("Segment count mismatch while compressing signals")
    return PreparedSignalSegments(
        starts=starts,
        ends=ends,
        values=values,
        counts=counts,
        change_count=resolved_change_count,
    )


def time_range_slice(
    *,
    open_time_15m: np.ndarray,
    close_time_15m: np.ndarray,
    time_range: Mapping[str, Any],
) -> slice:
    """
    Deterministic `[start, end)` slicing by 15m `open_time`.
    """

    start_ms = _utc_timestamp_ms(time_range.get("start"), path="time_range.start")
    end_ms = _utc_timestamp_ms(time_range.get("end"), path="time_range.end")
    if start_ms >= end_ms:
        raise BacktestPreparePoolsRejected("time_range must use non-empty [start, end)")
    if int(open_time_15m.shape[0]) == 0:
        raise BacktestPreparePoolsRejected("prices/15m open_time is empty")
    if start_ms < int(open_time_15m[0]) or end_ms > int(close_time_15m[-1]) + 1:
        raise BacktestPreparePoolsRejected("time_range is outside prices/15m coverage")

    start_idx = int(np.searchsorted(open_time_15m, start_ms, side="left"))
    stop_idx = int(np.searchsorted(open_time_15m, end_ms, side="left"))
    if start_idx >= stop_idx:
        raise BacktestPreparePoolsRejected("time_range selects no 15m bars")
    if stop_idx - start_idx < 2:
        raise BacktestPreparePoolsRejected("time_range must select at least two 15m bars")
    return slice(start_idx, stop_idx)


def extract_signal_rows(
    matrix: np.ndarray,
    *,
    row_ids: np.ndarray,
    time_slice: slice,
) -> np.ndarray:
    """
    Copy only requested signal rows and sliced bars into a contiguous int8 matrix.
    """

    row_ids_i32 = np.asarray(row_ids, dtype=np.int32)
    if row_ids_i32.ndim != 1 or int(row_ids_i32.size) == 0:
        raise ValueError("row_ids must be a non-empty one-dimensional array")
    if int(row_ids_i32.min()) < 0 or int(row_ids_i32.max()) >= int(matrix.shape[0]):
        raise ValueError("row_ids are outside signal matrix bounds")
    row_selector = _contiguous_row_selector(row_ids_i32)
    selected = matrix[row_selector, time_slice]
    return np.ascontiguousarray(np.asarray(selected, dtype=np.int8))


def row_metadata_order_hash(indicator_pools: Sequence[PreparedIndicatorPool]) -> str:
    """
    Stable hash of prepared per-indicator row metadata in pool/order position.
    """

    payload = [
        [metadata.as_mapping() for metadata in pool.metadata]
        for pool in indicator_pools
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def notebook_compatible_prepare_pools_core_s(timing: PreparePoolsTiming) -> float:
    """
    Return the canonical notebook-comparable prepare-pools duration.
    """

    value = timing.subsegments.get(PREPARE_POOLS_CORE_STAGE_NAME)
    if value is not None:
        return float(value)
    return float(
        timing.subsegments.get(SIGNAL_ROW_SELECTION_SEGMENT, 0.0)
        + timing.subsegments.get(ROW_PREFILTER_SEGMENT, 0.0)
        + timing.subsegments.get(SEGMENT_BUILD_SEGMENT, 0.0)
    )


def _load_signal_matrices(
    *,
    artifact_array_loader: BacktestArtifactArrayLoader,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
    indicator_requests: Sequence[Mapping[str, Any]],
) -> dict[str, ArtifactSignalMatrixV2]:
    matrices: dict[str, ArtifactSignalMatrixV2] = {}
    for indicator_request in indicator_requests:
        indicator_id = str(indicator_request["indicator_id"])
        if indicator_id in matrices:
            continue
        matrices[indicator_id] = artifact_array_loader.load_signal_matrix(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
    return matrices


def _select_indicator_rows(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
    signal_matrix: ArtifactSignalMatrixV2,
    indicator_request: Mapping[str, Any],
    time_slice: slice,
) -> _SelectedSignalRows:
    indicator_id = str(indicator_request["indicator_id"])
    row_ids, metadata = _resolve_signal_row_ids(
        defaults_provider=defaults_provider,
        indicator_request=indicator_request,
        artifact_rows_count=int(signal_matrix.manifest.rows_count),
    )
    return _SelectedSignalRows(
        indicator_id=indicator_id,
        row_ids=row_ids,
        trade_T=extract_signal_rows(
            signal_matrix.matrix,
            row_ids=row_ids,
            time_slice=time_slice,
        ),
        metadata=metadata,
    )


def _resolve_signal_row_ids(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
    indicator_request: Mapping[str, Any],
    artifact_rows_count: int,
) -> tuple[np.ndarray, tuple[PreparedIndicatorRowMetadata, ...]]:
    indicator_id = str(indicator_request["indicator_id"])
    source_axis = _artifact_source_values(
        defaults_provider=defaults_provider,
        indicator_id=indicator_id,
    )
    full_window_values = _artifact_window_values(
        defaults_provider=defaults_provider,
        indicator_id=indicator_id,
    )
    requested_sources = _requested_sources(
        indicator_request=indicator_request,
        source_axis=source_axis,
    )
    requested_windows = _requested_window_values(indicator_request)

    source_to_index = {
        source: index
        for index, source in enumerate(source_axis)
    }
    window_to_index = {
        window: index
        for index, window in enumerate(full_window_values)
    }
    row_ids: list[int] = []
    metadata: list[PreparedIndicatorRowMetadata] = []
    for source in requested_sources:
        if source not in source_to_index:
            raise BacktestPreparePoolsRejected(
                f"source {source!r} is not present in artifact row axis for {indicator_id!r}"
            )
        for window in requested_windows:
            if window not in window_to_index:
                raise BacktestPreparePoolsRejected(
                    f"window {window!r} is not present in artifact row axis for {indicator_id!r}"
                )
            row_id = source_to_index[source] * len(full_window_values) + window_to_index[window]
            if row_id >= artifact_rows_count:
                raise BacktestPreparePoolsRejected(
                    f"resolved row_id {row_id} exceeds artifact rows_count "
                    f"{artifact_rows_count} for {indicator_id!r}"
                )
            row_ids.append(row_id)
            metadata.append(
                PreparedIndicatorRowMetadata(
                    indicator_id=indicator_id,
                    row_id=row_id,
                    source=source,
                    window=window,
                )
            )
    return np.asarray(row_ids, dtype=np.int32), tuple(metadata)


def _artifact_source_values(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
    indicator_id: str,
) -> tuple[str | None, ...]:
    defaults = defaults_provider.compute_defaults(indicator_id=indicator_id)
    if defaults is not None and defaults.source is not None:
        return tuple(str(value).strip().lower() for value in defaults.source.materialize())
    allowed_sources = defaults_provider.allowed_source_values(indicator_id=indicator_id)
    if len(allowed_sources) > 0:
        return tuple(str(value).strip().lower() for value in allowed_sources)
    return (None,)


def _artifact_window_values(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
    indicator_id: str,
) -> tuple[int, ...]:
    defaults = defaults_provider.compute_defaults(indicator_id=indicator_id)
    if defaults is None or WINDOW_AXIS_NAME not in defaults.params:
        raise BacktestPreparePoolsRejected(
            f"indicator_id {indicator_id!r} does not expose a window axis"
        )
    return tuple(
        _grid_value_to_int(value)
        for value in defaults.params[WINDOW_AXIS_NAME].materialize()
    )


def _requested_sources(
    *,
    indicator_request: Mapping[str, Any],
    source_axis: Sequence[str | None],
) -> tuple[str | None, ...]:
    if source_axis == (None,):
        return (None,)
    raw_sources = indicator_request.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes, bytearray)):
        raise BacktestPreparePoolsRejected("indicator sources must be a sequence")
    return tuple(str(source).strip().lower() for source in raw_sources)


def _requested_window_values(indicator_request: Mapping[str, Any]) -> tuple[int, ...]:
    window = _required_mapping(indicator_request, "window", path="indicator.window")
    start = _positive_int(window.get("start"), path="indicator.window.start")
    stop = _positive_int(window.get("stop"), path="indicator.window.stop")
    step = _positive_int(window.get("step"), path="indicator.window.step")
    if start > stop:
        raise BacktestPreparePoolsRejected("indicator.window.start must be <= stop")
    return tuple(range(start, stop + 1, step))


def _signal_returns_15m(*, ohlcv_15m: np.ndarray, time_slice: slice) -> np.ndarray:
    signal_close_15m = np.asarray(ohlcv_15m[time_slice, 3], dtype=np.float32)
    if int(signal_close_15m.shape[0]) < 2:
        raise BacktestPreparePoolsRejected("time_range must select at least two close prices")
    return np.ascontiguousarray(
        ((signal_close_15m[1:] / signal_close_15m[:-1]) - 1.0).astype(np.float32)
    )


def _execution_mapping_no_risk_15m_to_1m(
    *,
    mapping_open_1m_idx_15m: np.ndarray,
    mapping_close_1m_idx_15m: np.ndarray,
    price_1m_length: int,
    time_slice: slice,
) -> PreparedExecutionMapping:
    run_bar_open_1m_idx_15m = np.asarray(mapping_open_1m_idx_15m[time_slice], dtype=np.int32)
    run_bar_close_1m_idx_15m = np.asarray(mapping_close_1m_idx_15m[time_slice], dtype=np.int32)
    if int(run_bar_open_1m_idx_15m.shape[0]) == 0:
        raise BacktestPreparePoolsRejected("time_range selects no mapping rows")
    t_exec_limit_1m = int(run_bar_close_1m_idx_15m[-1]) + 1
    if t_exec_limit_1m > price_1m_length:
        raise BacktestPreparePoolsRejected(
            "mappings/15m close index exceeds prices/1m coverage"
        )
    signal_entry_exec_idx_15m = np.empty(run_bar_open_1m_idx_15m.shape[0], dtype=np.int32)
    if int(signal_entry_exec_idx_15m.shape[0]) > 1:
        signal_entry_exec_idx_15m[:-1] = run_bar_open_1m_idx_15m[1:]
    signal_entry_exec_idx_15m[-1] = np.int32(t_exec_limit_1m)
    return PreparedExecutionMapping(
        signal_entry_exec_idx_15m=signal_entry_exec_idx_15m,
        run_bar_open_1m_idx_15m=np.ascontiguousarray(run_bar_open_1m_idx_15m),
        run_bar_close_1m_idx_15m=np.ascontiguousarray(run_bar_close_1m_idx_15m),
        t_exec_limit_1m=t_exec_limit_1m,
    )


def _coordinates_from_normalized(normalized_request: Mapping[str, Any]) -> BacktestCoordinates:
    coordinates = _required_mapping(
        normalized_request,
        "coordinates",
        path="normalized_request.coordinates",
    )
    return BacktestCoordinates(
        exchange=str(coordinates["exchange"]),
        market_type=str(coordinates["market_type"]),
        symbol=str(coordinates["symbol"]),
    )


def _timeframe_from_normalized(normalized_request: Mapping[str, Any]) -> str:
    timeframe = str(normalized_request.get("timeframe", "")).strip().lower()
    if timeframe != CANONICAL_BACKTEST_TIMEFRAME_V1:
        raise BacktestPreparePoolsRejected("prepare_pools supports timeframe '15m' only")
    return timeframe


def _indicator_requests_from_normalized(
    normalized_request: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_indicators = normalized_request.get("indicators")
    if not isinstance(raw_indicators, Sequence) or isinstance(
        raw_indicators,
        (str, bytes, bytearray),
    ):
        raise BacktestPreparePoolsRejected("normalized indicators must be a sequence")
    if len(raw_indicators) == 0:
        raise BacktestPreparePoolsRejected("normalized indicators must not be empty")
    indicators: list[Mapping[str, Any]] = []
    for raw_indicator in raw_indicators:
        if not isinstance(raw_indicator, Mapping):
            raise BacktestPreparePoolsRejected("normalized indicator must be a mapping")
        indicators.append(raw_indicator)
    return tuple(indicators)


def _fee_rate_from_normalized(normalized_request: Mapping[str, Any]) -> float:
    execution = _required_mapping(
        normalized_request,
        "execution",
        path="normalized_request.execution",
    )
    return float(execution.get("fee_rate", 0.0))


def _required_mapping(payload: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BacktestPreparePoolsRejected(f"{path} must be a mapping")
    return value


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BacktestPreparePoolsRejected(f"{path} must be a positive integer")
    return value


def _grid_value_to_int(value: GridValue) -> int:
    if isinstance(value, bool):
        raise BacktestPreparePoolsRejected("window grid values must not be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise BacktestPreparePoolsRejected(f"window grid value must be integer-like; got {value!r}")


def _utc_timestamp_ms(value: Any, *, path: str) -> int:
    if not isinstance(value, str):
        raise BacktestPreparePoolsRejected(f"{path} must be an UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BacktestPreparePoolsRejected(f"{path} must be a valid UTC timestamp") from error
    if parsed.tzinfo is None:
        raise BacktestPreparePoolsRejected(f"{path} must include timezone")
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def _contiguous_row_selector(row_ids: np.ndarray) -> slice | np.ndarray:
    if int(row_ids.size) == 1:
        start = int(row_ids[0])
        return slice(start, start + 1)
    expected = np.arange(int(row_ids[0]), int(row_ids[-1]) + 1, dtype=np.int32)
    if int(expected.size) == int(row_ids.size) and np.array_equal(row_ids, expected):
        return slice(int(row_ids[0]), int(row_ids[-1]) + 1)
    return row_ids


def _record_elapsed(
    subsegments: dict[str, float],
    segment_name: str,
    segment_start: float,
) -> float:
    elapsed = time.perf_counter() - segment_start
    subsegments[segment_name] = elapsed
    return elapsed


def _record_elapsed_aliases(
    subsegments: dict[str, float],
    segment_start: float,
    segment_name: str,
    legacy_segment_name: str,
) -> None:
    elapsed = _record_elapsed(subsegments, segment_name, segment_start)
    subsegments[legacy_segment_name] = elapsed


__all__ = [
    "ARTIFACT_ARRAY_OPEN_SEGMENT",
    "ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT",
    "ARTIFACT_CONTEXT_RESOLVE_SEGMENT",
    "ARTIFACT_MANIFEST_LOAD_SEGMENT",
    "BacktestPreparePoolsRequestSlice",
    "BacktestPreparePoolsRejected",
    "BacktestPreparePoolsRuntimeArrays",
    "BacktestPreparePoolsService",
    "PREPARE_POOLS_CORE_STAGE_NAME",
    "PREPARE_POOLS_TOTAL_STAGE_NAME",
    "REQUEST_SLICE_PREPARE_SEGMENT",
    "ROW_PREFILTER_SEGMENT",
    "SEGMENT_BUILD_SEGMENT",
    "SIGNAL_ROW_SELECTION_SEGMENT",
    "TIME_RANGE_SLICE_SEGMENT",
    "build_signal_segments",
    "extract_signal_rows",
    "fill_signal_segments_i8",
    "fused_row_prefilter_stats",
    "notebook_compatible_prepare_pools_core_s",
    "prepare_indicator_pool",
    "prepare_indicator_pools",
    "prefilter_indicator_rows",
    "row_metadata_order_hash",
    "time_range_slice",
    "topk_fraction_idx",
]
