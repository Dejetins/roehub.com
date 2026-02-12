from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import compute_ma_grid_f32


def test_numpy_oracle_sma_applies_warmup_and_nan_window_policy() -> None:
    """
    Verify SMA oracle applies warmup NaNs and NaN-in-window propagation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Windowed MA outputs NaN for `t < window - 1` and any window containing NaN.
    Raises:
        AssertionError: If SMA policy differs from architecture contract.
    Side Effects:
        None.
    """
    source = np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
    windows = np.asarray([3], dtype=np.int64)

    out = compute_ma_grid_f32(
        indicator_id="ma.sma",
        source=source,
        windows=windows,
    )

    expected = np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, 5.0], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out[:, 0], expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_numpy_oracle_ema_and_rma_reset_state_on_nan_holes() -> None:
    """
    Verify EMA/RMA reset state when input contains NaN and reseed on next valid value.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        EMA alpha is `2/(w+1)` and RMA alpha is `1/w`.
    Raises:
        AssertionError: If state-reset policy regresses.
    Side Effects:
        None.
    """
    source = np.asarray([1.0, 2.0, 3.0, np.nan, 4.0, 5.0], dtype=np.float32)
    windows = np.asarray([3], dtype=np.int64)

    ema = compute_ma_grid_f32(
        indicator_id="ma.ema",
        source=source,
        windows=windows,
    )
    rma = compute_ma_grid_f32(
        indicator_id="ma.rma",
        source=source,
        windows=windows,
    )

    expected_ema = np.asarray([1.0, 1.5, 2.25, np.nan, 4.0, 4.5], dtype=np.float32)
    expected_rma = np.asarray(
        [1.0, 1.3333334, 1.8888888, np.nan, 4.0, 4.3333335],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ema[:, 0], expected_ema, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(rma[:, 0], expected_rma, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_numpy_oracle_vwma_returns_nan_when_window_volume_sum_is_zero() -> None:
    """
    Verify VWMA oracle returns NaN when rolling volume denominator is zero.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        VWMA denominator equals rolling sum of volume and must be non-zero.
    Raises:
        AssertionError: If denominator-zero policy regresses.
    Side Effects:
        None.
    """
    source = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    volume = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    windows = np.asarray([2], dtype=np.int64)

    out = compute_ma_grid_f32(
        indicator_id="ma.vwma",
        source=source,
        windows=windows,
        volume=volume,
    )

    expected = np.asarray([np.nan, 1.0, np.nan, np.nan], dtype=np.float32)
    np.testing.assert_allclose(out[:, 0], expected, rtol=0.0, atol=0.0, equal_nan=True)
