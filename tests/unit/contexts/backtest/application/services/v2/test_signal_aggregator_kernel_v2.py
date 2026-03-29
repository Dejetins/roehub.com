from __future__ import annotations

import numpy as np
import pytest

from trading.contexts.backtest.application.services import aggregate_final_signal_rows_v2


def test_aggregate_final_signal_rows_v2_applies_consensus_and_is_order_independent() -> None:
    """
    Verify Stage A aggregation keeps exact consensus semantics independent of mapping order.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Long requires every indicator to confirm `+1`, short requires every indicator to confirm
        `-1`, and every mixed combination must become neutral `0`.
    Raises:
        AssertionError: If output values, dtype, or deterministic ordering drift.
    Side Effects:
        None.
    """
    indicator_rows = {
        "z.indicator": np.array([[1, 1, 0, -1], [1, -1, -1, -1]], dtype=np.int8),
        "a.indicator": np.array([[1, 0, 0, -1], [1, -1, -1, 0]], dtype=np.int8),
    }

    aggregated = aggregate_final_signal_rows_v2(selected_signal_rows=indicator_rows)

    assert aggregated.dtype == np.int8
    np.testing.assert_array_equal(
        aggregated,
        np.array([[1, 0, 0, -1], [1, -1, -1, 0]], dtype=np.int8),
    )


def test_aggregate_final_signal_rows_v2_rejects_invalid_signal_values() -> None:
    """
    Verify Stage A aggregation fails fast when one subset row leaves the `{-1, 0, 1}` contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Artifact-backed kernels must reject malformed subset rows before downstream compaction.
    Raises:
        AssertionError: If invalid values are not rejected.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match=r"\{\-1, 0, 1\}"):
        aggregate_final_signal_rows_v2(
            selected_signal_rows={
                "ema": np.array([[1, 2, 0]], dtype=np.int8),
            }
        )


def test_aggregate_final_signal_rows_v2_rejects_shape_drift() -> None:
    """
    Verify Stage A aggregation rejects subset matrices whose variant/time shapes drift.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Every selected indicator matrix must share one deterministic `[V, T_signal]` shape.
    Raises:
        AssertionError: If shape drift is not rejected.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="deterministic shape"):
        aggregate_final_signal_rows_v2(
            selected_signal_rows={
                "ema": np.array([[1, 0, -1]], dtype=np.int8),
                "rsi": np.array([[1, 0], [0, -1]], dtype=np.int8),
            }
        )
