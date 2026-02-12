from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import compute_trend_grid_f32


def test_numpy_oracle_linreg_slope_applies_warmup_and_nan_window_policy() -> None:
    """
    Verify linreg slope oracle applies warmup NaNs and NaN-in-window propagation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Rolling linreg output is NaN for `t < window - 1` and any window containing NaN.
    Raises:
        AssertionError: If linreg warmup or NaN-window policy regresses.
    Side Effects:
        None.
    """
    source = np.asarray([1.0, 2.0, 3.0, np.nan, 5.0, 6.0], dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))

    out = compute_trend_grid_f32(
        indicator_id="trend.linreg_slope",
        source_variants=source_variants,
        windows=np.asarray([3], dtype=np.int64),
    )

    expected = np.asarray([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out[0, :], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_numpy_oracle_directional_trend_indicators_reset_on_nan_holes() -> None:
    """
    Verify PSAR/SuperTrend outputs do not carry state across NaN holes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Directional trend kernels emit NaN at hole index and restart at next valid index.
    Raises:
        AssertionError: If reset-on-NaN behavior regresses.
    Side Effects:
        None.
    """
    high = np.asarray([10.0, 11.0, 12.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float32)
    low = np.asarray([9.0, 10.0, 11.0, 10.0, np.nan, 12.0, 13.0], dtype=np.float32)
    close = np.asarray([9.5, 10.5, 11.5, 10.5, np.nan, 12.5, 13.5], dtype=np.float32)

    psar = compute_trend_grid_f32(
        indicator_id="trend.psar",
        high=high,
        low=low,
        accel_starts=np.asarray([0.02], dtype=np.float64),
        accel_steps=np.asarray([0.02], dtype=np.float64),
        accel_maxes=np.asarray([0.2], dtype=np.float64),
    )
    supertrend = compute_trend_grid_f32(
        indicator_id="trend.supertrend",
        high=high,
        low=low,
        close=close,
        windows=np.asarray([3], dtype=np.int64),
        mults=np.asarray([2.0], dtype=np.float64),
    )

    for output in (psar, supertrend):
        assert output.dtype == np.float32
        assert output.flags["C_CONTIGUOUS"]
        assert np.isnan(output[0, 4])
        assert not np.isnan(output[0, 5])


def test_numpy_oracle_ichimoku_span_a_uses_negative_displacement_shift() -> None:
    """
    Verify Ichimoku primary output applies shift with `periods = -displacement`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Forward shift introduces trailing NaN values at the end of series.
    Raises:
        AssertionError: If shift semantics for span_a regress.
    Side Effects:
        None.
    """
    high = np.asarray([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    low = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    close = np.asarray([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dtype=np.float32)

    out = compute_trend_grid_f32(
        indicator_id="trend.ichimoku",
        high=high,
        low=low,
        close=close,
        conversion_windows=np.asarray([2], dtype=np.int64),
        base_windows=np.asarray([2], dtype=np.int64),
        span_b_windows=np.asarray([2], dtype=np.int64),
        displacements=np.asarray([2], dtype=np.int64),
    )

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.isnan(out[0, -1])
    assert np.isnan(out[0, -2])
    assert np.isfinite(out[0, 2])
