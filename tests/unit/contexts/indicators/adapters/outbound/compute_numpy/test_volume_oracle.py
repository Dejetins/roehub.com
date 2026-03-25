from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import compute_volume_grid_f32


def test_numpy_oracle_ad_line_resets_cumsum_on_nan_holes() -> None:
    """
    Verify AD-line oracle resets cumulative state when encountering NaN holes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        NaN on OHLCV inputs emits NaN output and resets AD cumulative state.
    Raises:
        AssertionError: If reset-on-NaN policy regresses.
    Side Effects:
        None.
    """
    out = compute_volume_grid_f32(
        indicator_id="volume.ad_line",
        high=np.asarray([2.0, 3.0, np.nan, 4.0], dtype=np.float32),
        low=np.asarray([1.0, 2.0, np.nan, 3.0], dtype=np.float32),
        close=np.asarray([1.5, 2.5, np.nan, 3.5], dtype=np.float32),
        volume=np.asarray([10.0, 10.0, np.nan, 10.0], dtype=np.float32),
    )

    expected = np.asarray([0.0, 0.0, np.nan, 0.0], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out[0, :], expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_numpy_oracle_vwap_returns_nan_when_window_volume_sum_is_zero() -> None:
    """
    Verify VWAP oracle returns NaN when rolling volume denominator equals zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        VWAP denominator is `rolling_sum(volume)` and must be non-zero.
    Raises:
        AssertionError: If denominator-zero policy regresses.
    Side Effects:
        None.
    """
    out = compute_volume_grid_f32(
        indicator_id="volume.vwap",
        high=np.asarray([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        low=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        close=np.asarray([1.5, 1.5, 1.5, 1.5], dtype=np.float32),
        volume=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        windows=np.asarray([2], dtype=np.int64),
    )

    expected = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    np.testing.assert_allclose(out[0, :], expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_numpy_oracle_volume_sma_applies_window_warmup() -> None:
    """
    Verify volume SMA output follows rolling-window warmup policy.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Volume SMA is undefined until the first full rolling window is available.
    Raises:
        AssertionError: If multiplier axis does not affect output values.
    Side Effects:
        None.
    """
    out = compute_volume_grid_f32(
        indicator_id="volume.volume_sma",
        volume=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        windows=np.asarray([3], dtype=np.int64),
    )

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    expected = np.asarray([np.nan, np.nan, 20.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(out[0, :], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
