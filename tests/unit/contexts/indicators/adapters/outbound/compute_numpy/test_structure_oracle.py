from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_structure_grid_f32,
)


def test_numpy_oracle_zscore_returns_nan_when_sd_is_zero() -> None:
    """
    Verify z-score oracle returns NaN when rolling standard deviation equals zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Div-by-zero policy for `structure.zscore` is `sd==0 -> NaN`.
    Raises:
        AssertionError: If zero-variance windows produce finite values.
    Side Effects:
        None.
    """
    source = np.ones(10, dtype=np.float32)
    out = compute_structure_grid_f32(
        indicator_id="structure.zscore",
        source_variants=np.ascontiguousarray(source.reshape(1, source.shape[0])),
        windows=np.asarray([3], dtype=np.int64),
    )

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.all(np.isnan(out[0, :]))


def test_numpy_oracle_candle_pct_returns_nan_when_range_is_zero() -> None:
    """
    Verify candle percent outputs become NaN when candle range equals zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Div-by-zero policy for candle `%` outputs is `range==0 -> NaN`.
    Raises:
        AssertionError: If range-zero bars produce finite percent values.
    Side Effects:
        None.
    """
    open_series = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    high_series = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    low_series = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    close_series = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)

    body_pct = compute_structure_grid_f32(
        indicator_id="structure.candle_body_pct",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
    )
    stats_primary = compute_structure_grid_f32(
        indicator_id="structure.candle_stats",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
    )

    assert np.all(np.isnan(body_pct[0, :]))
    assert np.all(np.isnan(stats_primary[0, :]))


def test_numpy_oracle_atr_normalized_outputs_nan_when_atr_is_zero() -> None:
    """
    Verify ATR-normalized candle outputs become NaN when ATR is exactly zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Div-by-zero policy for `*_atr` outputs is `atr==0 -> NaN`.
    Raises:
        AssertionError: If ATR-zero bars produce finite normalized values.
    Side Effects:
        None.
    """
    open_series = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    high_series = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    low_series = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    close_series = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

    out = compute_structure_grid_f32(
        indicator_id="structure.candle_stats_atr_norm",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        atr_windows=np.asarray([3], dtype=np.int64),
    )

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.all(np.isnan(out[0, :]))


def test_numpy_oracle_distance_to_ma_norm_returns_nan_when_atr_is_zero() -> None:
    """
    Verify distance-to-MA normalization returns NaN when ATR denominator is zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Div-by-zero policy for `structure.distance_to_ma_norm` is `atr==0 -> NaN`.
    Raises:
        AssertionError: If ATR-zero bars produce finite normalized distance values.
    Side Effects:
        None.
    """
    source = np.asarray([25.0, 25.0, 25.0, 25.0, 25.0], dtype=np.float32)
    high = np.asarray([25.0, 25.0, 25.0, 25.0, 25.0], dtype=np.float32)
    low = np.asarray([25.0, 25.0, 25.0, 25.0, 25.0], dtype=np.float32)
    close = np.asarray([25.0, 25.0, 25.0, 25.0, 25.0], dtype=np.float32)

    out = compute_structure_grid_f32(
        indicator_id="structure.distance_to_ma_norm",
        source_variants=np.ascontiguousarray(source.reshape(1, source.shape[0])),
        high=high,
        low=low,
        close=close,
        windows=np.asarray([3], dtype=np.int64),
    )

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.all(np.isnan(out[0, :]))


def test_numpy_oracle_pivots_use_shift_confirm_true_semantics() -> None:
    """
    Verify pivots are emitted only on confirmation index (`shift_confirm=true`).

    Args:
        None.
    Returns:
        None.
    Assumptions:
        For left=right=2, pivot at center index `c` is emitted at `c+2`.
    Raises:
        AssertionError: If pivot confirmation shift semantics regress.
    Side Effects:
        None.
    """
    high = np.asarray([1.0, 2.0, 5.0, 2.0, 1.0, 1.5, 1.2], dtype=np.float32)
    low = np.asarray([5.0, 4.0, 1.0, 4.0, 5.0, 4.5, 4.8], dtype=np.float32)

    pivot_high = compute_structure_grid_f32(
        indicator_id="structure.pivots",
        high=high,
        low=low,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )
    pivot_low = compute_structure_grid_f32(
        indicator_id="structure.pivot_low",
        high=high,
        low=low,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )

    expected_high = np.asarray([np.nan, np.nan, np.nan, np.nan, 5.0, np.nan, np.nan])
    expected_low = np.asarray([np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan])

    np.testing.assert_allclose(
        pivot_high[0, :],
        expected_high,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        pivot_low[0, :],
        expected_low,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
