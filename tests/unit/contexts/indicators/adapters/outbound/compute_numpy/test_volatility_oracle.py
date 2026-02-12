from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_volatility_grid_f32,
)


def test_numpy_oracle_stddev_applies_warmup_and_nan_window_policy() -> None:
    """
    Verify stddev oracle applies warmup NaNs and NaN-in-window propagation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Rolling stddev outputs NaN for `t < window - 1` and any window containing NaN.
    Raises:
        AssertionError: If stddev NaN policy differs from EPIC-07 contract.
    Side Effects:
        None.
    """
    source = np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))
    windows = np.asarray([3], dtype=np.int64)

    out = compute_volatility_grid_f32(
        indicator_id="volatility.stddev",
        source_variants=source_variants,
        windows=windows,
    )

    expected = np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, 0.8164966], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out[0, :], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_numpy_oracle_atr_resets_state_on_nan_holes() -> None:
    """
    Verify ATR oracle resets internal RMA state when TR contains NaN values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        ATR uses RMA(alpha=1/window) and resets state on NaN inputs.
    Raises:
        AssertionError: If reset-on-NaN policy regresses.
    Side Effects:
        None.
    """
    high = np.asarray([11.0, 12.0, np.nan, 14.0, 15.0], dtype=np.float32)
    low = np.asarray([9.0, 10.0, np.nan, 12.0, 13.0], dtype=np.float32)
    close = np.asarray([10.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float32)
    windows = np.asarray([3], dtype=np.int64)

    out = compute_volatility_grid_f32(
        indicator_id="volatility.atr",
        high=high,
        low=low,
        close=close,
        windows=windows,
    )

    expected = np.asarray([2.0, 2.0, np.nan, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(out[0, :], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_numpy_oracle_bbands_bandwidth_returns_nan_when_middle_is_zero() -> None:
    """
    Verify Bollinger bandwidth oracle returns NaN when middle line equals zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Bandwidth is `(upper - lower) / middle` and must be NaN when middle is zero.
    Raises:
        AssertionError: If zero-denominator policy regresses.
    Side Effects:
        None.
    """
    source = np.zeros(8, dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))
    windows = np.asarray([3], dtype=np.int64)
    mults = np.asarray([2.0], dtype=np.float64)

    out = compute_volatility_grid_f32(
        indicator_id="volatility.bbands_bandwidth",
        source_variants=source_variants,
        windows=windows,
        mults=mults,
    )

    expected = np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(out[0, :], expected, rtol=0.0, atol=0.0, equal_nan=True)
