"""Deterministic `hit_times/15m` compute kernels for the backtest artifact pipeline v2."""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .contracts import (
    ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
    ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
)


@dataclass(frozen=True, slots=True)
class HitTimesArraysV2:
    """
    Immutable in-memory `hit_times/15m` arrays ready for strict artifact serialization.

    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    tp_values: np.ndarray
    sl_values: np.ndarray
    long_tp: np.ndarray
    long_sl: np.ndarray
    short_tp: np.ndarray
    short_sl: np.ndarray
    sentinel_index: int


def hit_times_table_cell_count_v2(
    *,
    timeline_bar_count: int,
    tp_level_count: int,
    sl_level_count: int,
) -> int:
    """
    Count total strict hit-times table cells across all four TP/SL table families.

    Args:
        timeline_bar_count: Canonical `1m` timeline length.
        tp_level_count: Number of TP levels carried by `tp_values`.
        sl_level_count: Number of SL levels carried by `sl_values`.
    Returns:
        int: Total cell count across `long_tp`, `long_sl`, `short_tp`, and `short_sl`.
    Assumptions:
        The strict artifact family always materializes two TP tables and two SL tables with
        shape `[level, time]`.
    Raises:
        ValueError: If one input count is negative.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if timeline_bar_count < 0:
        raise ValueError(f"timeline_bar_count must be >= 0; got {timeline_bar_count!r}")
    if tp_level_count < 0:
        raise ValueError(f"tp_level_count must be >= 0; got {tp_level_count!r}")
    if sl_level_count < 0:
        raise ValueError(f"sl_level_count must be >= 0; got {sl_level_count!r}")
    return timeline_bar_count * (2 * tp_level_count + 2 * sl_level_count)


def materialize_hit_times_from_ohlcv_v2(
    *,
    ohlcv: np.ndarray,
    tp_levels_pct: tuple[float, ...],
    sl_levels_pct: tuple[float, ...],
    max_hit_times_cells: int,
) -> HitTimesArraysV2:
    """
    Build strict `hit_times/15m` arrays from canonical `prices/1m.ohlcv`.

    Args:
        ohlcv: Canonical `prices/1m` OHLCV matrix with shape `[T, 5]`.
        tp_levels_pct: Positive ascending TP percentage levels in human-percent units.
        sl_levels_pct: Positive ascending SL percentage levels in human-percent units.
        max_hit_times_cells: Fail-fast upper bound for all emitted table cells.
    Returns:
        HitTimesArraysV2: Fresh strict grids and lookup tables with `sentinel_index == T`.
    Assumptions:
        Entry semantics match the notebook baseline: entry occurs at `open[t]` and the same bar
        may immediately hit TP/SL via `high[t]` or `low[t]`.
    Raises:
        ValueError: If inputs are empty, non-finite, exceed budgets, or produce invalid tables.
    Side Effects:
        Allocates fresh numpy arrays and triggers Numba compilation on first use.
    Docs:
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    normalized_ohlcv = _normalize_hit_times_ohlcv_v2(ohlcv=ohlcv)
    tp_values = _normalize_hit_times_level_grid_v2(
        levels_pct=tp_levels_pct,
        field_name="tp_levels_pct",
    )
    sl_values = _normalize_hit_times_level_grid_v2(
        levels_pct=sl_levels_pct,
        field_name="sl_levels_pct",
    )
    timeline_bar_count = int(normalized_ohlcv.shape[0])
    table_cell_count = hit_times_table_cell_count_v2(
        timeline_bar_count=timeline_bar_count,
        tp_level_count=int(tp_values.shape[0]),
        sl_level_count=int(sl_values.shape[0]),
    )
    if table_cell_count > max_hit_times_cells:
        raise ValueError(
            "hit-times table cells exceed max_hit_times_cells: "
            f"cells={table_cell_count}, max_hit_times_cells={max_hit_times_cells}"
        )

    open_values = np.ascontiguousarray(normalized_ohlcv[:, 0], dtype=np.float32)
    high_values = np.ascontiguousarray(normalized_ohlcv[:, 1], dtype=np.float32)
    low_values = np.ascontiguousarray(normalized_ohlcv[:, 2], dtype=np.float32)
    long_tp, long_sl, short_tp, short_sl = _compute_hit_times_tables_v2(
        open_f32=open_values,
        high_f32=high_values,
        low_f32=low_values,
        tp_values=tp_values,
        sl_values=sl_values,
    )
    result = HitTimesArraysV2(
        tp_values=tp_values,
        sl_values=sl_values,
        long_tp=long_tp,
        long_sl=long_sl,
        short_tp=short_tp,
        short_sl=short_sl,
        sentinel_index=timeline_bar_count,
    )
    _validate_materialized_hit_times_v2(hit_times=result)
    return result


def merge_hit_times_prefix_with_rebuilt_tail_v2(
    *,
    prefix: HitTimesArraysV2 | None,
    rebuilt_tail: HitTimesArraysV2,
    prefix_bars: int,
    total_timeline_bars: int,
) -> HitTimesArraysV2:
    """
    Merge a reused hit-times prefix with a freshly rebuilt tail in global timeline space.

    Args:
        prefix: Existing unchanged prefix slice, or `None` for full rebuild.
        rebuilt_tail: Fresh tail arrays computed on the local tail-only `ohlcv` slice.
        prefix_bars: Number of unchanged leading timeline bars preserved from the existing slot.
        total_timeline_bars: Final global timeline length after merge.
    Returns:
        HitTimesArraysV2: Strict merged arrays with global indexes and sentinel bounds.
    Assumptions:
        `rebuilt_tail` indexes are local to its tail slice and must be rebased by `prefix_bars`
        before concatenation.
    Raises:
        ValueError: If prefix/tail grids drift, prefix sizes are inconsistent, or the final merged
            arrays violate the strict contract.
    Side Effects:
        Allocates fresh merged numpy arrays in memory.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if prefix_bars < 0:
        raise ValueError(f"prefix_bars must be >= 0; got {prefix_bars!r}")
    if total_timeline_bars <= 0:
        raise ValueError(f"total_timeline_bars must be > 0; got {total_timeline_bars!r}")
    expected_total_timeline_bars = prefix_bars + rebuilt_tail.sentinel_index
    if expected_total_timeline_bars != total_timeline_bars:
        raise ValueError(
            "total_timeline_bars must equal prefix_bars + rebuilt_tail.sentinel_index; got "
            f"{total_timeline_bars!r}, expected {expected_total_timeline_bars!r}"
        )
    if prefix is not None and prefix.sentinel_index != prefix_bars:
        raise ValueError(
            "prefix.sentinel_index must equal prefix_bars; got "
            f"{prefix.sentinel_index!r}, expected {prefix_bars!r}"
        )

    rebased_tail = HitTimesArraysV2(
        tp_values=np.ascontiguousarray(rebuilt_tail.tp_values, dtype=np.float32),
        sl_values=np.ascontiguousarray(rebuilt_tail.sl_values, dtype=np.float32),
        long_tp=_rebase_hit_times_table_indexes_v2(
            values=rebuilt_tail.long_tp,
            prefix_bars=prefix_bars,
            tail_sentinel_index=rebuilt_tail.sentinel_index,
            total_timeline_bars=total_timeline_bars,
        ),
        long_sl=_rebase_hit_times_table_indexes_v2(
            values=rebuilt_tail.long_sl,
            prefix_bars=prefix_bars,
            tail_sentinel_index=rebuilt_tail.sentinel_index,
            total_timeline_bars=total_timeline_bars,
        ),
        short_tp=_rebase_hit_times_table_indexes_v2(
            values=rebuilt_tail.short_tp,
            prefix_bars=prefix_bars,
            tail_sentinel_index=rebuilt_tail.sentinel_index,
            total_timeline_bars=total_timeline_bars,
        ),
        short_sl=_rebase_hit_times_table_indexes_v2(
            values=rebuilt_tail.short_sl,
            prefix_bars=prefix_bars,
            tail_sentinel_index=rebuilt_tail.sentinel_index,
            total_timeline_bars=total_timeline_bars,
        ),
        sentinel_index=total_timeline_bars,
    )
    if prefix is None or prefix_bars == 0:
        _validate_materialized_hit_times_v2(hit_times=rebased_tail)
        return rebased_tail

    _require_matching_hit_times_grids_v2(prefix=prefix, rebuilt_tail=rebuilt_tail)
    merged = HitTimesArraysV2(
        tp_values=np.ascontiguousarray(prefix.tp_values, dtype=np.float32),
        sl_values=np.ascontiguousarray(prefix.sl_values, dtype=np.float32),
        long_tp=np.ascontiguousarray(
            np.concatenate((prefix.long_tp, rebased_tail.long_tp), axis=1),
            dtype=np.uint32,
        ),
        long_sl=np.ascontiguousarray(
            np.concatenate((prefix.long_sl, rebased_tail.long_sl), axis=1),
            dtype=np.uint32,
        ),
        short_tp=np.ascontiguousarray(
            np.concatenate((prefix.short_tp, rebased_tail.short_tp), axis=1),
            dtype=np.uint32,
        ),
        short_sl=np.ascontiguousarray(
            np.concatenate((prefix.short_sl, rebased_tail.short_sl), axis=1),
            dtype=np.uint32,
        ),
        sentinel_index=total_timeline_bars,
    )
    _validate_materialized_hit_times_v2(hit_times=merged)
    return merged


def _normalize_hit_times_ohlcv_v2(*, ohlcv: np.ndarray) -> np.ndarray:
    """
    Normalize the canonical `prices/1m.ohlcv` matrix for hit-times kernels.

    Args:
        ohlcv: Candidate OHLCV matrix.
    Returns:
        np.ndarray: Contiguous `float32` matrix with shape `[T, 5]`.
    Assumptions:
        Hit-times kernels only depend on the `open/high/low` columns but preserve the strict
        canonical OHLCV layout contract.
    Raises:
        ValueError: If the matrix is not two-dimensional, empty, non-finite, or not `[T, 5]`.
    Side Effects:
        May allocate one contiguous `float32` copy.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    normalized = np.ascontiguousarray(np.asarray(ohlcv, dtype=np.float32))
    if normalized.ndim != 2:
        raise ValueError(f"hit-times ohlcv must be 2D; got ndim={normalized.ndim}")
    if normalized.shape[0] <= 0:
        raise ValueError("hit-times ohlcv must contain at least one bar")
    if normalized.shape[1] != 5:
        raise ValueError(f"hit-times ohlcv must have shape [T, 5]; got {normalized.shape!r}")
    if not np.isfinite(normalized[:, :3]).all():
        raise ValueError("hit-times ohlcv open/high/low values must be finite")
    return normalized


def _normalize_hit_times_level_grid_v2(
    *,
    levels_pct: tuple[float, ...],
    field_name: str,
) -> np.ndarray:
    """
    Normalize one deterministic TP/SL level grid into canonical `float32` fractions.

    Args:
        levels_pct: Candidate human-percent levels.
        field_name: Stable field label used in fail-fast diagnostics.
    Returns:
        np.ndarray: Strictly increasing contiguous `float32` fractions in `[0, 1)`.
    Assumptions:
        The same level grid is reused across long/short tables and therefore must stay below
        `100%` to keep `(1 - level)` positive for downside thresholds.
    Raises:
        ValueError: If the grid is empty, non-positive, duplicates, or contains `>= 100`.
    Side Effects:
        Allocates one small contiguous `float32` array.
    Docs:
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if len(levels_pct) == 0:
        raise ValueError(f"{field_name} must contain at least one level")
    normalized_pct = tuple(sorted(float(value) for value in levels_pct))
    if normalized_pct[0] <= 0.0:
        raise ValueError(f"{field_name} values must be > 0; got {normalized_pct!r}")
    if normalized_pct[-1] >= 100.0:
        raise ValueError(f"{field_name} values must be < 100; got {normalized_pct!r}")
    if any(left == right for left, right in zip(normalized_pct, normalized_pct[1:])):
        raise ValueError(f"{field_name} must not contain duplicate values; got {normalized_pct!r}")
    return np.ascontiguousarray(
        np.asarray([value / 100.0 for value in normalized_pct], dtype=np.float32)
    )


@nb.njit(cache=True)
def _heap_push_min_v2(
    heap_prices: np.ndarray,
    heap_indexes: np.ndarray,
    size: int,
    value: np.float32,
    index: int,
) -> int:
    """
    Push one `(value, index)` pair into the in-place min-heap used by hit-times kernels.

    Args:
        heap_prices: Heap key storage buffer.
        heap_indexes: Heap index storage buffer.
        size: Current heap size.
        value: Heap key to insert.
        index: Timeline index associated with the key.
    Returns:
        int: New heap size.
    Assumptions:
        Buffers are preallocated to timeline length and only the leading `size` cells are live.
    Raises:
        None.
    Side Effects:
        Mutates the heap buffers in place.
    Docs:
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
    """
    position = size
    size += 1
    while position > 0:
        parent = (position - 1) // 2
        if heap_prices[parent] <= value:
            break
        heap_prices[position] = heap_prices[parent]
        heap_indexes[position] = heap_indexes[parent]
        position = parent
    heap_prices[position] = value
    heap_indexes[position] = index
    return size


@nb.njit(cache=True)
def _heap_pop_min_v2(
    heap_prices: np.ndarray,
    heap_indexes: np.ndarray,
    size: int,
) -> tuple[np.float32, int, int]:
    """
    Pop the smallest `(value, index)` pair from the in-place min-heap.

    Args:
        heap_prices: Heap key storage buffer.
        heap_indexes: Heap index storage buffer.
        size: Current heap size.
    Returns:
        tuple[np.float32, int, int]: Popped value, popped index, and the new heap size.
    Assumptions:
        Callers ensure `size > 0`.
    Raises:
        None.
    Side Effects:
        Mutates the heap buffers in place.
    Docs:
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
    """
    value = heap_prices[0]
    index = heap_indexes[0]
    size -= 1
    if size > 0:
        last_value = heap_prices[size]
        last_index = heap_indexes[size]
        position = 0
        while True:
            left = 2 * position + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if right < size and heap_prices[right] < heap_prices[left]:
                child = right
            if heap_prices[child] >= last_value:
                break
            heap_prices[position] = heap_prices[child]
            heap_indexes[position] = heap_indexes[child]
            position = child
        heap_prices[position] = last_value
        heap_indexes[position] = last_index
    return value, index, size


@nb.njit(cache=True)
def _hit_times_1level_v2(
    open_f32: np.ndarray,
    trigger_f32: np.ndarray,
    multiplier: np.float32,
    negate_open: bool,
    out_u32: np.ndarray,
    heap_prices: np.ndarray,
    heap_indexes: np.ndarray,
) -> None:
    """
    Compute the first hit index for one TP/SL level across the entire `1m` timeline.

    Args:
        open_f32: Entry-price series interpreted as `open[t]`.
        trigger_f32: `high` or `low` trigger series depending on table family.
        multiplier: Level multiplier (`1 + pct` or `1 - pct`) for the chosen family.
        negate_open: Whether the kernel should emulate a max-heap via negated opens.
        out_u32: Output buffer of length `T`.
        heap_prices: Reusable heap key buffer of length `T`.
        heap_indexes: Reusable heap index buffer of length `T`.
    Returns:
        None.
    Assumptions:
        Entry at `open[t]` is inclusive for the same candle, so bar `t` may immediately resolve.
    Raises:
        None.
    Side Effects:
        Mutates the provided output and heap buffers in place.
    Docs:
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
    """
    timeline_bar_count = open_f32.shape[0]
    sentinel_index = np.uint32(timeline_bar_count)
    for position in range(timeline_bar_count):
        out_u32[position] = sentinel_index

    size = 0
    for bar_index in range(timeline_bar_count):
        open_value = open_f32[bar_index]
        if negate_open:
            open_value = np.float32(-open_value)
        size = _heap_push_min_v2(
            heap_prices,
            heap_indexes,
            size,
            np.float32(open_value),
            bar_index,
        )
        threshold = np.float32(trigger_f32[bar_index] / multiplier)
        if negate_open:
            threshold = np.float32(-threshold)
        while size > 0 and heap_prices[0] <= threshold:
            _, start_index, size = _heap_pop_min_v2(heap_prices, heap_indexes, size)
            out_u32[start_index] = np.uint32(bar_index)


@nb.njit(parallel=True, cache=True)
def _compute_hit_times_tables_v2(
    *,
    open_f32: np.ndarray,
    high_f32: np.ndarray,
    low_f32: np.ndarray,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Materialize all four strict hit-times table families for the configured TP/SL grids.

    Args:
        open_f32: Canonical `open` series.
        high_f32: Canonical `high` series.
        low_f32: Canonical `low` series.
        tp_values: TP levels as fractions.
        sl_values: SL levels as fractions.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: `long_tp`, `long_sl`,
        `short_tp`, and `short_sl` tables with shape `[level, time]`.
    Assumptions:
        Grids are already validated, strictly increasing, and below `1.0`.
    Raises:
        None.
    Side Effects:
        Allocates the final tables plus per-level scratch buffers.
    Docs:
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
    """
    timeline_bar_count = open_f32.shape[0]
    tp_level_count = tp_values.shape[0]
    sl_level_count = sl_values.shape[0]
    long_tp = np.empty((tp_level_count, timeline_bar_count), dtype=np.uint32)
    long_sl = np.empty((sl_level_count, timeline_bar_count), dtype=np.uint32)
    short_tp = np.empty((tp_level_count, timeline_bar_count), dtype=np.uint32)
    short_sl = np.empty((sl_level_count, timeline_bar_count), dtype=np.uint32)
    for level_index in nb.prange(tp_level_count):
        heap_prices = np.empty(timeline_bar_count, dtype=np.float32)
        heap_indexes = np.empty(timeline_bar_count, dtype=np.int32)
        out_u32 = np.empty(timeline_bar_count, dtype=np.uint32)
        _hit_times_1level_v2(
            open_f32=open_f32,
            trigger_f32=high_f32,
            multiplier=np.float32(1.0 + tp_values[level_index]),
            negate_open=False,
            out_u32=out_u32,
            heap_prices=heap_prices,
            heap_indexes=heap_indexes,
        )
        long_tp[level_index, :] = out_u32
        out_u32 = np.empty(timeline_bar_count, dtype=np.uint32)
        _hit_times_1level_v2(
            open_f32=open_f32,
            trigger_f32=low_f32,
            multiplier=np.float32(1.0 - tp_values[level_index]),
            negate_open=True,
            out_u32=out_u32,
            heap_prices=heap_prices,
            heap_indexes=heap_indexes,
        )
        short_tp[level_index, :] = out_u32

    for level_index in nb.prange(sl_level_count):
        heap_prices = np.empty(timeline_bar_count, dtype=np.float32)
        heap_indexes = np.empty(timeline_bar_count, dtype=np.int32)
        out_u32 = np.empty(timeline_bar_count, dtype=np.uint32)
        _hit_times_1level_v2(
            open_f32=open_f32,
            trigger_f32=low_f32,
            multiplier=np.float32(1.0 - sl_values[level_index]),
            negate_open=True,
            out_u32=out_u32,
            heap_prices=heap_prices,
            heap_indexes=heap_indexes,
        )
        long_sl[level_index, :] = out_u32
        out_u32 = np.empty(timeline_bar_count, dtype=np.uint32)
        _hit_times_1level_v2(
            open_f32=open_f32,
            trigger_f32=high_f32,
            multiplier=np.float32(1.0 + sl_values[level_index]),
            negate_open=False,
            out_u32=out_u32,
            heap_prices=heap_prices,
            heap_indexes=heap_indexes,
        )
        short_sl[level_index, :] = out_u32

    return long_tp, long_sl, short_tp, short_sl


def _validate_materialized_hit_times_v2(*, hit_times: HitTimesArraysV2) -> None:
    """
    Validate deterministic dtype, shape, bounds, and monotonicity invariants in memory.

    Args:
        hit_times: Fresh hit-times arrays returned by the compute kernels.
    Returns:
        None.
    Assumptions:
        Local validation should fail before filesystem writes when kernels or config drift.
    Raises:
        ValueError: If one grid/table violates the strict artifact contract.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if hit_times.sentinel_index <= 0:
        raise ValueError(f"hit_times sentinel_index must be > 0; got {hit_times.sentinel_index}")
    _validate_hit_times_level_grid_v2(
        values=hit_times.tp_values,
        field_name="tp_values",
    )
    _validate_hit_times_level_grid_v2(
        values=hit_times.sl_values,
        field_name="sl_values",
    )
    _validate_hit_times_table_v2(
        values=hit_times.long_tp,
        level_count=int(hit_times.tp_values.shape[0]),
        sentinel_index=hit_times.sentinel_index,
        field_name="long_tp",
    )
    _validate_hit_times_table_v2(
        values=hit_times.long_sl,
        level_count=int(hit_times.sl_values.shape[0]),
        sentinel_index=hit_times.sentinel_index,
        field_name="long_sl",
    )
    _validate_hit_times_table_v2(
        values=hit_times.short_tp,
        level_count=int(hit_times.tp_values.shape[0]),
        sentinel_index=hit_times.sentinel_index,
        field_name="short_tp",
    )
    _validate_hit_times_table_v2(
        values=hit_times.short_sl,
        level_count=int(hit_times.sl_values.shape[0]),
        sentinel_index=hit_times.sentinel_index,
        field_name="short_sl",
    )


def _rebase_hit_times_table_indexes_v2(
    *,
    values: np.ndarray,
    prefix_bars: int,
    tail_sentinel_index: int,
    total_timeline_bars: int,
) -> np.ndarray:
    """
    Translate one tail-local hit-times table into global timeline indexes.

    Args:
        values: Tail-local `uint32[level, time]` table.
        prefix_bars: Number of unchanged leading timeline bars kept before the tail.
        tail_sentinel_index: Tail-local sentinel equal to the local tail timeline length.
        total_timeline_bars: Final global sentinel equal to the merged timeline length.
    Returns:
        np.ndarray: Rebasing result with indexes in the global timeline space.
    Assumptions:
        Non-sentinel cells are local bar indexes in `[0, tail_sentinel_index)`, while sentinel
        cells must become the final global sentinel.
    Raises:
        ValueError: If the resulting table would overflow the global sentinel bound.
    Side Effects:
        Allocates one rebased contiguous `uint32` array.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    sentinel_mask = values == np.uint32(tail_sentinel_index)
    rebased = np.asarray(values, dtype=np.uint64) + np.uint64(prefix_bars)
    rebased[sentinel_mask] = np.uint64(total_timeline_bars)
    if np.any(rebased > np.uint64(total_timeline_bars)):
        raise ValueError(
            "rebased hit-times values must stay within the merged sentinel bound; got "
            f"max={int(np.max(rebased))}, total_timeline_bars={total_timeline_bars}"
        )
    return np.ascontiguousarray(rebased, dtype=np.uint32)


def _require_matching_hit_times_grids_v2(
    *,
    prefix: HitTimesArraysV2,
    rebuilt_tail: HitTimesArraysV2,
) -> None:
    """
    Require reused prefix grids to match the freshly rebuilt tail grids exactly.

    Args:
        prefix: Existing prefix slice selected for reuse.
        rebuilt_tail: Freshly rebuilt tail arrays.
    Returns:
        None.
    Assumptions:
        Prefix reuse is safe only when TP/SL level grids are byte-identical across both slices.
    Raises:
        ValueError: If TP or SL grids drift across the merge boundary.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """
    if not np.array_equal(prefix.tp_values, rebuilt_tail.tp_values):
        raise ValueError("hit-times prefix tp_values must match rebuilt tail tp_values")
    if not np.array_equal(prefix.sl_values, rebuilt_tail.sl_values):
        raise ValueError("hit-times prefix sl_values must match rebuilt tail sl_values")


def _validate_hit_times_level_grid_v2(*, values: np.ndarray, field_name: str) -> None:
    """
    Validate one in-memory strict TP/SL level grid.

    Args:
        values: Candidate level grid as fractions.
        field_name: Stable field label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Level grids are serialized as `float32` and must stay strictly increasing.
    Raises:
        ValueError: If dtype, dimensionality, emptiness, or ordering drift.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if values.dtype.name != ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{field_name} dtype must be {ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2}; "
            f"got {values.dtype.name!r}"
        )
    if values.ndim != 1:
        raise ValueError(f"{field_name} must have shape [N_levels]; got {values.shape!r}")
    if values.shape[0] <= 0:
        raise ValueError(f"{field_name} must contain at least one level")
    if not np.all(np.diff(values) > 0):
        raise ValueError(f"{field_name} must be strictly increasing")


def _validate_hit_times_table_v2(
    *,
    values: np.ndarray,
    level_count: int,
    sentinel_index: int,
    field_name: str,
) -> None:
    """
    Validate one in-memory strict hit-times lookup table.

    Args:
        values: Candidate table with shape `[level, time]`.
        level_count: Expected number of level rows.
        sentinel_index: Expected inclusive upper bound and timeline length.
        field_name: Stable field label used in fail-fast diagnostics.
    Returns:
        None.
    Assumptions:
        Every table uses `uint32`, the `level` major layout, and the
        `non_decreasing_by_level` invariant.
    Raises:
        ValueError: If dtype, shape, bounds, or monotonicity drift.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if values.dtype.name != ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2:
        raise ValueError(
            f"{field_name} dtype must be {ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2}; "
            f"got {values.dtype.name!r}"
        )
    if values.ndim != 2:
        raise ValueError(f"{field_name} must have shape [level, time]; got {values.shape!r}")
    expected_shape = (level_count, sentinel_index)
    if values.shape != expected_shape:
        raise ValueError(f"{field_name} shape must be {expected_shape!r}; got {values.shape!r}")
    if np.any(values > sentinel_index):
        raise ValueError(
            f"{field_name} values must stay within [0, {sentinel_index}]; "
            f"got max={int(values.max())}"
        )
    if values.shape[0] > 1 and not np.all(values[1:, :] >= values[:-1, :]):
        raise ValueError(f"{field_name} must satisfy non-decreasing-by-level monotonicity")
