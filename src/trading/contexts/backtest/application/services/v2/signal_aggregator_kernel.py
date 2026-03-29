"""Pure Stage A `final_signal` aggregation kernels for artifacts-only inputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import SIGNAL_CODE_LONG_V2, SIGNAL_CODE_NEUTRAL_V2, SIGNAL_CODE_SHORT_V2


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
    if len(ordered_items) == 0:
        raise ValueError("selected_signal_rows must contain at least one indicator matrix")

    normalized_rows = tuple(
        _normalize_signal_row_matrix_v2(indicator_id=indicator_id, values=values)
        for indicator_id, values in ordered_items
    )
    variant_count, timeline_length = normalized_rows[0].shape
    for indicator_id, values in zip((item[0] for item in ordered_items), normalized_rows):
        if values.shape != (variant_count, timeline_length):
            raise ValueError(
                "all selected signal rows must share one deterministic shape; "
                f"{indicator_id!r} has {values.shape!r}, expected "
                f"{(variant_count, timeline_length)!r}"
            )

    final_long = np.ones((variant_count, timeline_length), dtype=np.bool_)
    final_short = np.ones((variant_count, timeline_length), dtype=np.bool_)
    for values in normalized_rows:
        final_long &= values == SIGNAL_CODE_LONG_V2
        final_short &= values == SIGNAL_CODE_SHORT_V2

    conflict_mask = final_long & final_short
    aggregated = np.full(
        (variant_count, timeline_length),
        SIGNAL_CODE_NEUTRAL_V2,
        dtype=np.int8,
    )
    aggregated[final_long & ~conflict_mask] = SIGNAL_CODE_LONG_V2
    aggregated[final_short & ~conflict_mask] = SIGNAL_CODE_SHORT_V2
    return aggregated


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


__all__ = ["aggregate_final_signal_rows_v2"]
