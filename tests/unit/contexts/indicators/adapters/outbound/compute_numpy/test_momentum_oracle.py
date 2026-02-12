from __future__ import annotations

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numpy import compute_momentum_grid_f32


def test_numpy_oracle_roc_applies_warmup_and_zero_denominator_policy() -> None:
    """
    Verify ROC oracle applies fixed warmup and NaN for zero denominator.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        ROC is undefined for `t < window` and when lagged source equals zero.
    Raises:
        AssertionError: If ROC warmup or denominator policy regresses.
    Side Effects:
        None.
    """
    source = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))
    windows = np.asarray([2], dtype=np.int64)

    out = compute_momentum_grid_f32(
        indicator_id="momentum.roc",
        source_variants=source_variants,
        windows=windows,
    )

    expected = np.asarray([np.nan, np.nan, np.nan, 200.0, 100.0], dtype=np.float32)
    np.testing.assert_allclose(out[0, :], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_numpy_oracle_rsi_resets_state_on_nan_holes() -> None:
    """
    Verify RSI oracle resets state when source contains NaN and reseeds afterward.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        RSI RMA states reset on NaN and first valid value after reset starts a new segment.
    Raises:
        AssertionError: If reset-on-NaN policy regresses.
    Side Effects:
        None.
    """
    source = np.asarray([10.0, 11.0, 12.0, np.nan, 13.0, 14.0], dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))
    windows = np.asarray([3], dtype=np.int64)

    out = compute_momentum_grid_f32(
        indicator_id="momentum.rsi",
        source_variants=source_variants,
        windows=windows,
    )

    assert np.isnan(out[0, 0])
    assert out[0, 1] == 100.0
    assert out[0, 2] == 100.0
    assert np.isnan(out[0, 3])
    assert np.isnan(out[0, 4])
    assert out[0, 5] == 100.0


def test_numpy_oracle_stateful_momentum_indicators_reset_on_nan_holes() -> None:
    """
    Verify TRIX, MACD, and PPO outputs do not carry state across NaN holes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        EMA-based chains emit NaN at hole indices and restart from next valid segment.
    Raises:
        AssertionError: If reset-on-NaN behavior regresses.
    Side Effects:
        None.
    """
    source = np.asarray([100.0, 101.0, 102.0, np.nan, 103.0, 104.0, 105.0], dtype=np.float32)
    source_variants = np.ascontiguousarray(source.reshape(1, source.shape[0]))

    trix = compute_momentum_grid_f32(
        indicator_id="momentum.trix",
        source_variants=source_variants,
        windows=np.asarray([5], dtype=np.int64),
        signal_windows=np.asarray([3], dtype=np.int64),
    )
    macd = compute_momentum_grid_f32(
        indicator_id="momentum.macd",
        source_variants=source_variants,
        fast_windows=np.asarray([3], dtype=np.int64),
        slow_windows=np.asarray([5], dtype=np.int64),
        signal_windows=np.asarray([2], dtype=np.int64),
    )
    ppo = compute_momentum_grid_f32(
        indicator_id="momentum.ppo",
        source_variants=source_variants,
        fast_windows=np.asarray([3], dtype=np.int64),
        slow_windows=np.asarray([5], dtype=np.int64),
        signal_windows=np.asarray([2], dtype=np.int64),
    )

    for out in (trix, macd, ppo):
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        assert np.isnan(out[0, 3])

    assert np.isnan(trix[0, 4])
    assert macd[0, 4] == 0.0
    assert ppo[0, 4] == 0.0
