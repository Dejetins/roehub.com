"""Pure Stage A `final_signal` aggregation kernels for artifacts-only inputs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numba as nb
import numpy as np

from .contracts import SIGNAL_CODE_LONG_V2, SIGNAL_CODE_NEUTRAL_V2, SIGNAL_CODE_SHORT_V2


@nb.njit(cache=True)
def _resolve_pair_consensus_signal_v2(
    left_signal_value: int,
    right_signal_value: int,
) -> int:
    """
    Resolve one two-indicator consensus signal without building a dense indicator cube.

    Args:
        left_signal_value: Left indicator signal value inside `{-1, 0, 1}`.
        right_signal_value: Right indicator signal value inside `{-1, 0, 1}`.
    Returns:
        int: Consensus signal code preserving the Stage A long/short/neutral contract.
    Assumptions:
        The pair-first parity path evaluates exactly two indicators, so consensus reduces to
        checking whether both rows confirm the same non-neutral direction.
    Raises:
        None.
    Side Effects:
        None.
    """
    if left_signal_value == right_signal_value:
        if left_signal_value == SIGNAL_CODE_LONG_V2:
            return SIGNAL_CODE_LONG_V2
        if left_signal_value == SIGNAL_CODE_SHORT_V2:
            return SIGNAL_CODE_SHORT_V2
    return SIGNAL_CODE_NEUTRAL_V2


@nb.njit(parallel=True, cache=True)
def _aggregate_signal_pair_rows_kernel_v2(
    *,
    left_signal_rows_i8: np.ndarray,
    right_signal_rows_i8: np.ndarray,
) -> np.ndarray:
    """
    Aggregate one two-indicator Stage A signal pair directly in pair-first row order.

    Args:
        left_signal_rows_i8: Left indicator rows shaped `[variant, time]`.
        right_signal_rows_i8: Right indicator rows shaped `[variant, time]`.
    Returns:
        np.ndarray: Aggregated `final_signal[V, T_signal]` matrix for the same variant order.
    Assumptions:
        The parity-only pair path must avoid dense `[indicator, variant, time]` cube allocation,
        so every `[variant, time]` cell is resolved directly from the aligned row pair.
    Raises:
        None.
    Side Effects:
        Allocates one aggregated `np.int8` matrix.
    """
    variant_count = int(left_signal_rows_i8.shape[0])
    timeline_length = int(left_signal_rows_i8.shape[1])
    aggregated = np.full(
        (variant_count, timeline_length),
        SIGNAL_CODE_NEUTRAL_V2,
        dtype=np.int8,
    )
    for row_index in nb.prange(variant_count):
        for time_index in range(timeline_length):
            aggregated[row_index, time_index] = _resolve_pair_consensus_signal_v2(
                int(left_signal_rows_i8[row_index, time_index]),
                int(right_signal_rows_i8[row_index, time_index]),
            )
    return aggregated


@nb.njit(parallel=True, cache=True)
def _aggregate_final_signal_row_cube_kernel_v2(
    *,
    signal_row_cube_i8: np.ndarray,
) -> np.ndarray:
    """
    Aggregate one dense `[indicator, variant, time]` Stage A signal cube in parallel.

    Args:
        signal_row_cube_i8: Canonical `np.int8` signal cube whose values stay inside
            `{-1, 0, 1}`.
    Returns:
        np.ndarray: Aggregated `final_signal[V, T_signal]` matrix preserving variant/time order.
    Assumptions:
        Each `[variant, time]` cell is independent once the dense indicator cube is prepared, so
        the dominant Stage A frontier aggregation can stay single-process and parallel-capable via
        Numba row-level scheduling.
    Raises:
        None.
    Side Effects:
        Allocates one aggregated `np.int8` output matrix.
    """
    indicator_count = int(signal_row_cube_i8.shape[0])
    variant_count = int(signal_row_cube_i8.shape[1])
    timeline_length = int(signal_row_cube_i8.shape[2])
    aggregated = np.full(
        (variant_count, timeline_length),
        SIGNAL_CODE_NEUTRAL_V2,
        dtype=np.int8,
    )
    for row_index in nb.prange(variant_count):
        for time_index in range(timeline_length):
            all_long = True
            all_short = True
            for indicator_index in range(indicator_count):
                signal_value = int(signal_row_cube_i8[indicator_index, row_index, time_index])
                if signal_value != SIGNAL_CODE_LONG_V2:
                    all_long = False
                if signal_value != SIGNAL_CODE_SHORT_V2:
                    all_short = False
                if not all_long and not all_short:
                    break
            if all_long:
                aggregated[row_index, time_index] = SIGNAL_CODE_LONG_V2
            elif all_short:
                aggregated[row_index, time_index] = SIGNAL_CODE_SHORT_V2
    return aggregated


def aggregate_final_signal_rows_v2(
    *,
    selected_signal_rows: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Aggregate per-indicator signal rows into deterministic Stage A `final_signal[V, T_signal]`.

    Args:
        selected_signal_rows: Mapping `indicator_id -> signal rows` where every array has shape
            `[V, T_signal]` and contains only `{-1, 0, 1}`.
    Returns:
        np.ndarray: Compact `np.int8` matrix `final_signal[V, T_signal]` with value set
            `{-1, 0, 1}`.
    Assumptions:
        Aggregation policy is explicit consensus AND:
        every indicator must confirm `+1` for long, every indicator must confirm `-1` for short,
        and every other combination is neutral `0`.
    Raises:
        ValueError: If inputs are empty, shapes drift, or one row contains unsupported values.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_aggregator_kernel_v2.py
    """
    ordered_items = tuple(sorted(selected_signal_rows.items(), key=lambda item: item[0]))
    return aggregate_ordered_final_signal_rows_v2(
        ordered_signal_rows=tuple(values for _, values in ordered_items),
        indicator_ids=tuple(indicator_id for indicator_id, _ in ordered_items),
    )


def aggregate_ordered_final_signal_rows_v2(
    *,
    ordered_signal_rows: Sequence[np.ndarray],
    indicator_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Aggregate already ordered per-indicator Stage A signal rows via a batched kernel path.

    Args:
        ordered_signal_rows: Deterministic per-indicator signal-row matrices aligned to one Stage
            A chunk order.
        indicator_ids: Optional indicator ids aligned to `ordered_signal_rows` for diagnostics.
    Returns:
        np.ndarray: Compact `np.int8` matrix `final_signal[V, T_signal]` with value set
            `{-1, 0, 1}`.
    Assumptions:
        Callers that already preserve indicator ordering should avoid rebuilding temporary
        mappings so the dominant Stage A frontier path stays kernel-driven instead of spending
        extra Python work on dict construction and sorting.
    Raises:
        ValueError: If inputs are empty, indicator-id alignment drifts, shapes drift, or one row
            contains unsupported values.
    Side Effects:
        Allocates one dense indicator cube and one aggregated output matrix.
    """
    normalized_rows = _normalize_signal_row_matrices_v2(
        ordered_signal_rows=ordered_signal_rows,
        indicator_ids=indicator_ids,
    )
    return _aggregate_normalized_signal_rows_v2(normalized_rows=normalized_rows)


def aggregate_signal_pairs_v2(
    *,
    left_signal_rows: np.ndarray,
    right_signal_rows: np.ndarray,
    indicator_ids: tuple[str, str] | None = None,
) -> np.ndarray:
    """
    Aggregate exactly two ordered indicator signal matrices through the pair-first kernel path.

    Args:
        left_signal_rows: Left indicator signal rows shaped `[V, T_signal]`.
        right_signal_rows: Right indicator signal rows shaped `[V, T_signal]`.
        indicator_ids: Optional `(left_id, right_id)` pair used in fail-fast diagnostics.
    Returns:
        np.ndarray: Aggregated `final_signal[V, T_signal]` matrix with value set `{-1, 0, 1}`.
    Assumptions:
        Canonical no-risk parity currently narrows to a two-indicator class, so the exact path can
        stay pair-first and avoid the broader dense indicator cube used by generic kernels.
    Raises:
        ValueError: If indicator-id alignment drifts, shapes drift, or one matrix is invalid.
    Side Effects:
        Allocates one aggregated `np.int8` matrix and may trigger Numba compilation on first use.
    """
    resolved_indicator_ids = (
        ("indicator[0]", "indicator[1]")
        if indicator_ids is None
        else tuple(str(indicator_id) for indicator_id in indicator_ids)
    )
    if len(resolved_indicator_ids) != 2:
        raise ValueError(
            "indicator_ids must provide exactly two ids for pair aggregation; got "
            f"{len(resolved_indicator_ids)}"
        )
    left_normalized, right_normalized = _normalize_signal_pair_matrices_v2(
        left_indicator_id=resolved_indicator_ids[0],
        left_signal_rows=left_signal_rows,
        right_indicator_id=resolved_indicator_ids[1],
        right_signal_rows=right_signal_rows,
    )
    return _aggregate_signal_pair_rows_kernel_v2(
        left_signal_rows_i8=left_normalized,
        right_signal_rows_i8=right_normalized,
    )


def _aggregate_normalized_signal_rows_v2(
    *,
    normalized_rows: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Aggregate normalized Stage A signal-row matrices through the batched kernel path.

    Args:
        normalized_rows: Shape-validated `np.int8` matrices aligned by indicator order.
    Returns:
        np.ndarray: Aggregated `final_signal[V, T_signal]` matrix.
    Assumptions:
        All matrices already share one deterministic shape and value contract.
    Raises:
        ValueError: If no normalized rows are provided.
    Side Effects:
        Allocates a dense indicator cube when more than one indicator participates.
    """
    if len(normalized_rows) == 0:
        raise ValueError("selected_signal_rows must contain at least one indicator matrix")
    if len(normalized_rows) == 1:
        return np.array(normalized_rows[0], dtype=np.int8, copy=True, order="C")
    signal_row_cube = _stack_normalized_signal_rows_v2(normalized_rows=normalized_rows)
    return _aggregate_final_signal_row_cube_kernel_v2(signal_row_cube_i8=signal_row_cube)


def _normalize_signal_row_matrices_v2(
    *,
    ordered_signal_rows: Sequence[np.ndarray],
    indicator_ids: Sequence[str] | None,
) -> tuple[np.ndarray, ...]:
    """
    Normalize and shape-check one ordered Stage A signal-row batch.

    Args:
        ordered_signal_rows: Deterministic per-indicator signal-row matrices.
        indicator_ids: Optional indicator ids aligned to `ordered_signal_rows`.
    Returns:
        tuple[np.ndarray, ...]: Canonical `np.int8` matrices ready for batched aggregation.
    Assumptions:
        Ordered callers preserve indicator sequencing explicitly, and diagnostics should still
        name the offending indicator when shape or value contracts drift.
    Raises:
        ValueError: If the batch is empty, indicator-id count drifts, or shapes are inconsistent.
    Side Effects:
        None.
    """
    signal_row_count = len(ordered_signal_rows)
    if signal_row_count == 0:
        raise ValueError("selected_signal_rows must contain at least one indicator matrix")
    resolved_indicator_ids = _resolve_indicator_ids_v2(
        signal_row_count=signal_row_count,
        indicator_ids=indicator_ids,
    )
    normalized_rows = tuple(
        _normalize_signal_row_matrix_v2(indicator_id=indicator_id, values=values)
        for indicator_id, values in zip(
            resolved_indicator_ids,
            ordered_signal_rows,
            strict=True,
        )
    )
    variant_count, timeline_length = normalized_rows[0].shape
    for indicator_id, values in zip(resolved_indicator_ids, normalized_rows, strict=True):
        if values.shape != (variant_count, timeline_length):
            raise ValueError(
                "all selected signal rows must share one deterministic shape; "
                f"{indicator_id!r} has {values.shape!r}, expected "
                f"{(variant_count, timeline_length)!r}"
            )
    return normalized_rows


def _normalize_signal_pair_matrices_v2(
    *,
    left_indicator_id: str,
    left_signal_rows: np.ndarray,
    right_indicator_id: str,
    right_signal_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize and shape-check one ordered two-indicator Stage A signal-row pair.

    Args:
        left_indicator_id: Left indicator id used in diagnostics.
        left_signal_rows: Left indicator rows shaped `[V, T_signal]`.
        right_indicator_id: Right indicator id used in diagnostics.
        right_signal_rows: Right indicator rows shaped `[V, T_signal]`.
    Returns:
        tuple[np.ndarray, np.ndarray]: Canonical `np.int8` matrices aligned by pair row order.
    Assumptions:
        Pair-first parity scoring keeps row order explicit and must fail fast if either side
        drifts from the shared deterministic `[variant, time]` contract.
    Raises:
        ValueError: If either matrix is invalid or their shapes drift.
    Side Effects:
        None.
    """
    left_normalized = _normalize_signal_row_matrix_v2(
        indicator_id=left_indicator_id,
        values=left_signal_rows,
    )
    right_normalized = _normalize_signal_row_matrix_v2(
        indicator_id=right_indicator_id,
        values=right_signal_rows,
    )
    if left_normalized.shape != right_normalized.shape:
        raise ValueError(
            "pair signal rows must share one deterministic shape; "
            f"{left_indicator_id!r} has {left_normalized.shape!r}, "
            f"{right_indicator_id!r} has {right_normalized.shape!r}"
        )
    return left_normalized, right_normalized


def _resolve_indicator_ids_v2(
    *,
    signal_row_count: int,
    indicator_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """
    Resolve deterministic indicator ids used in Stage A aggregation diagnostics.

    Args:
        signal_row_count: Number of per-indicator matrices supplied for aggregation.
        indicator_ids: Optional caller-provided ids aligned to the same order.
    Returns:
        tuple[str, ...]: Deterministic indicator ids for validation and error messages.
    Assumptions:
        Ordered Stage A callers may omit ids, in which case positional synthetic ids keep error
        handling deterministic without rebuilding a mapping.
    Raises:
        ValueError: If provided `indicator_ids` length drifts from `signal_row_count`.
    Side Effects:
        None.
    """
    if indicator_ids is None:
        return tuple(f"indicator[{index}]" for index in range(signal_row_count))
    resolved_indicator_ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    if len(resolved_indicator_ids) != signal_row_count:
        raise ValueError(
            "indicator_ids must stay 1:1 aligned with ordered_signal_rows; got "
            f"{len(resolved_indicator_ids)} ids for {signal_row_count} signal matrices"
        )
    return resolved_indicator_ids


def _stack_normalized_signal_rows_v2(
    *,
    normalized_rows: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Materialize one dense Stage A indicator cube for kernel-driven aggregation.

    Args:
        normalized_rows: Shape-validated per-indicator `np.int8` matrices.
    Returns:
        np.ndarray: Contiguous `np.int8` cube shaped `[indicator, variant, time]`.
    Assumptions:
        The dense cube is internal-only and exists to remove Python-heavy per-indicator boolean
        reductions from the dominant Stage A frontier path.
    Raises:
        ValueError: If `normalized_rows` is empty.
    Side Effects:
        Allocates one dense cube spanning all indicators for the current chunk.
    """
    if len(normalized_rows) == 0:
        raise ValueError("normalized_rows must contain at least one signal matrix")
    indicator_count = len(normalized_rows)
    variant_count, timeline_length = normalized_rows[0].shape
    signal_row_cube = np.empty(
        (indicator_count, variant_count, timeline_length),
        dtype=np.int8,
    )
    for indicator_index, values in enumerate(normalized_rows):
        signal_row_cube[indicator_index, :, :] = values
    return signal_row_cube


def _normalize_signal_row_matrix_v2(*, indicator_id: str, values: np.ndarray) -> np.ndarray:
    """
    Normalize one per-indicator Stage A signal-row matrix to compact deterministic `np.int8`.

    Args:
        indicator_id: Indicator identifier used in fail-fast diagnostics.
        values: Candidate `[V, T_signal]` matrix.
    Returns:
        np.ndarray: Canonical `np.int8` matrix preserving caller ordering.
    Assumptions:
        Stage A kernel accepts only already-selected subset rows and therefore requires explicit
        two-dimensional `[variant, time]` inputs.
    Raises:
        ValueError: If the matrix is not 2D or contains values outside `{-1, 0, 1}`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_aggregator_kernel_v2.py
    """
    if values.ndim != 2:
        raise ValueError(f"{indicator_id}: selected signal rows must be a 2D matrix")
    normalized = np.asarray(values, dtype=np.int8)
    invalid_mask = ~np.isin(
        normalized,
        (SIGNAL_CODE_SHORT_V2, SIGNAL_CODE_NEUTRAL_V2, SIGNAL_CODE_LONG_V2),
    )
    if bool(np.any(invalid_mask)):
        invalid_values = tuple(int(value) for value in np.unique(normalized[invalid_mask]))
        raise ValueError(
            f"{indicator_id}: selected signal rows must contain only "
            f"{{-1, 0, 1}}, got {invalid_values!r}"
        )
    return normalized


__all__ = ["aggregate_final_signal_rows_v2", "aggregate_signal_pairs_v2"]
