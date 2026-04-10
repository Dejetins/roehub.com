from __future__ import annotations

import numpy as np
import pytest

from trading.contexts.backtest.application.services.v2.hit_times_compute_v2 import (
    materialize_hit_times_from_ohlcv_v2,
)


def test_materialize_hit_times_from_ohlcv_v2_matches_same_bar_and_sentinel_semantics() -> None:
    """
    Verify hit-times kernels follow same-bar-inclusive entry semantics and sentinel fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Entry at `open[t]` may resolve on bar `t` itself through `high[t]`/`low[t]`, while
        unresolved levels keep the sentinel index equal to timeline length.
    Raises:
        AssertionError: If grids, tables, or sentinel semantics drift from the notebook contract.
    Side Effects:
        Triggers one small in-memory materialization and Numba compilation on first use.
    Docs:
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - tests/notebook_tests/05_hit_time_grid.ipynb
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    ohlcv = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 1.0],
            [100.0, 101.0, 99.0, 100.0, 1.0],
            [100.0, 100.0, 100.0, 100.0, 1.0],
            [100.0, 102.0, 98.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = materialize_hit_times_from_ohlcv_v2(
        ohlcv=ohlcv,
        tp_levels_pct=(1.0, 2.0, 3.0),
        sl_levels_pct=(1.0, 2.0, 3.0),
        max_hit_times_cells=256,
    )

    np.testing.assert_allclose(result.tp_values, np.array([0.01, 0.02, 0.03], dtype=np.float32))
    np.testing.assert_allclose(result.sl_values, np.array([0.01, 0.02, 0.03], dtype=np.float32))
    np.testing.assert_array_equal(
        result.long_tp,
        np.array(
            [
                [1, 1, 3, 3],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(
        result.long_sl,
        np.array(
            [
                [1, 1, 3, 3],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(result.short_tp, result.long_sl)
    np.testing.assert_array_equal(result.short_sl, result.long_tp)
    assert result.sentinel_index == 4


def test_materialize_hit_times_from_ohlcv_v2_shapes_follow_level_grid_counts() -> None:
    """
    Verify strict hit-times table shapes expand exactly with configured TP/SL level counts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The kernels remain fully dynamic over asymmetric TP/SL grids and keep the canonical
        `[level, time]` table layout for every family.
    Raises:
        AssertionError: If emitted array shapes drift from the configured level counts.
    Side Effects:
        Triggers one small in-memory materialization and Numba compilation on first use.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    ohlcv = np.array(
        [
            [100.0, 100.5, 99.5, 100.0, 1.0],
            [100.0, 101.0, 99.0, 100.0, 1.0],
            [100.0, 102.0, 98.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = materialize_hit_times_from_ohlcv_v2(
        ohlcv=ohlcv,
        tp_levels_pct=(0.5, 1.0, 1.5, 2.0),
        sl_levels_pct=(0.5, 1.0),
        max_hit_times_cells=256,
    )

    assert result.tp_values.shape == (4,)
    assert result.sl_values.shape == (2,)
    assert result.long_tp.shape == (4, 3)
    assert result.long_sl.shape == (2, 3)
    assert result.short_tp.shape == (4, 3)
    assert result.short_sl.shape == (2, 3)


def test_materialize_hit_times_from_ohlcv_v2_rejects_cells_over_budget() -> None:
    """
    Verify hit-times materialization fails fast when configured table cells exceed the budget.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Budget checks should trigger before any large allocations or filesystem writes happen.
    Raises:
        AssertionError: If oversized hit-times requests do not fail with the documented error.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py
    """
    ohlcv = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 1.0],
            [100.0, 101.0, 99.0, 100.0, 1.0],
            [100.0, 100.0, 100.0, 100.0, 1.0],
            [100.0, 102.0, 98.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="max_hit_times_cells"):
        materialize_hit_times_from_ohlcv_v2(
            ohlcv=ohlcv,
            tp_levels_pct=(1.0, 2.0),
            sl_levels_pct=(1.0, 2.0),
            max_hit_times_cells=31,
        )
