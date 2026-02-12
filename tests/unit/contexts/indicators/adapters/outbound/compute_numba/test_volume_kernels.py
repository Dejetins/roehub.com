from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_volume_grid_f32 as compute_volume_grid_numba_f32,
)
from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_volume_grid_f32 as compute_volume_grid_numpy_f32,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    ExplicitValuesSpec,
)
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.platform.config import IndicatorsComputeNumbaConfig
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)


def _time_range() -> TimeRange:
    """
    Build deterministic UTC half-open time range for volume kernel tests.

    Args:
        None.
    Returns:
        TimeRange: Stable test range.
    Assumptions:
        Range metadata is used only for `CandleArrays` invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = UtcTimestamp(datetime(2026, 2, 13, 16, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 13, 22, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles_with_holes(*, t_size: int) -> CandleArrays:
    """
    Build deterministic OHLCV candles with NaN holes for volume policy tests.

    Args:
        t_size: Number of timeline rows.
    Returns:
        CandleArrays: Dense candles payload with deterministic NaN holes.
    Assumptions:
        Generated rows satisfy CandleArrays dtype/shape invariants.
    Raises:
        ValueError: If generated payload violates DTO invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    rng = np.random.default_rng(20260215)
    base = np.linspace(100.0, 170.0, t_size, dtype=np.float32)
    open_series = np.ascontiguousarray(base + rng.normal(0.0, 0.45, t_size).astype(np.float32))
    high_delta = np.abs(rng.normal(1.1, 0.3, t_size).astype(np.float32))
    low_delta = np.abs(rng.normal(1.1, 0.3, t_size).astype(np.float32))
    high_series = np.ascontiguousarray(open_series + high_delta)
    low_series = np.ascontiguousarray(open_series - low_delta)
    close_series = np.ascontiguousarray(
        open_series + rng.normal(0.0, 0.35, t_size).astype(np.float32)
    )
    volume_series = np.ascontiguousarray(rng.uniform(0.0, 600.0, t_size).astype(np.float32))

    hole_indices = np.asarray([1, 11, 35, 97, 201, 333, 479], dtype=np.int64)
    for idx in hole_indices:
        if idx < t_size:
            open_series[idx] = np.nan
            high_series[idx] = np.nan
            low_series[idx] = np.nan
            close_series[idx] = np.nan
            volume_series[idx] = np.nan

    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60_000)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=_time_range(),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        volume=volume_series,
    )


def _compute_engine(*, cache_dir: Path) -> NumbaIndicatorCompute:
    """
    Build deterministic Numba compute adapter instance for volume tests.

    Args:
        cache_dir: Numba cache directory path.
    Returns:
        NumbaIndicatorCompute: Ready compute adapter.
    Assumptions:
        Hard definitions are deterministic from `all_defs()`.
    Raises:
        ValueError: If config values are invalid.
    Side Effects:
        None.
    """
    config = IndicatorsComputeNumbaConfig(
        numba_num_threads=1,
        numba_cache_dir=cache_dir,
        max_compute_bytes_total=5 * 1024**3,
    )
    return NumbaIndicatorCompute(defs=all_defs(), config=config)


def test_numba_volume_kernels_match_numpy_oracle_with_nan_holes() -> None:
    """
    Verify Numba volume kernels match NumPy oracle on deterministic NaN-hole inputs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Both implementations use identical NaN policy and float32 output contract.
    Raises:
        AssertionError: If kernel math diverges from oracle beyond float32 tolerance.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=640)

    windows_i64 = np.asarray([5, 14, 28, 42], dtype=np.int64)
    mults_f64 = np.asarray([1.5, 2.0, 2.5, 3.0], dtype=np.float64)

    cases: tuple[tuple[str, dict[str, np.ndarray]], ...] = (
        (
            "volume.ad_line",
            {
                "high": candles.high,
                "low": candles.low,
                "close": candles.close,
                "volume": candles.volume,
            },
        ),
        (
            "volume.cmf",
            {
                "high": candles.high,
                "low": candles.low,
                "close": candles.close,
                "volume": candles.volume,
                "windows": windows_i64,
            },
        ),
        (
            "volume.mfi",
            {
                "high": candles.high,
                "low": candles.low,
                "close": candles.close,
                "volume": candles.volume,
                "windows": windows_i64,
            },
        ),
        (
            "volume.obv",
            {
                "close": candles.close,
                "volume": candles.volume,
            },
        ),
        (
            "volume.volume_sma",
            {
                "volume": candles.volume,
                "windows": windows_i64,
            },
        ),
        (
            "volume.vwap",
            {
                "high": candles.high,
                "low": candles.low,
                "close": candles.close,
                "volume": candles.volume,
                "windows": windows_i64,
            },
        ),
        (
            "volume.vwap_deviation",
            {
                "high": candles.high,
                "low": candles.low,
                "close": candles.close,
                "volume": candles.volume,
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
    )

    for indicator_id, kwargs in cases:
        numba_out = compute_volume_grid_numba_f32(indicator_id=indicator_id, **kwargs)
        numpy_out = compute_volume_grid_numpy_f32(indicator_id=indicator_id, **kwargs)

        assert numba_out.dtype == np.float32
        assert numba_out.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(
            numba_out,
            numpy_out,
            rtol=3e-5,
            atol=3e-5,
            equal_nan=True,
            err_msg=f"indicator_id={indicator_id}",
        )


def test_volume_vwap_deviation_uses_mult_axis_in_primary_output() -> None:
    """
    Verify vwap_deviation primary output changes with mult axis in v1 mapping.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Primary output for `volume.vwap_deviation` is `vwap_upper` and must depend on `mult`.
    Raises:
        AssertionError: If mult axis does not affect computed primary output.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=220)

    out = compute_volume_grid_numba_f32(
        indicator_id="volume.vwap_deviation",
        high=candles.high,
        low=candles.low,
        close=candles.close,
        volume=candles.volume,
        windows=np.asarray([20, 20], dtype=np.int64),
        mults=np.asarray([1.0, 3.0], dtype=np.float64),
    )

    assert out.shape[0] == 2
    valid = np.where(np.isfinite(out[0, :]) & np.isfinite(out[1, :]))[0]
    assert valid.size > 0
    assert np.any(np.abs(out[0, valid] - out[1, valid]) > 1e-6)


def test_numba_engine_preserves_volume_axis_order_and_supports_all_ids(
    tmp_path: Path,
) -> None:
    """
    Verify volume engine dispatch supports all ids and preserves explicit axis order.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Axis order for `volume.vwap_deviation` follows definition axes: mult, window.
    Raises:
        AssertionError: If tensor contracts or axis ordering are violated.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=320)
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")

    grids = (
        GridSpec(
            indicator_id=IndicatorId("volume.ad_line"),
            params={},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.cmf"),
            params={"window": ExplicitValuesSpec(name="window", values=(10, 20))},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.mfi"),
            params={"window": ExplicitValuesSpec(name="window", values=(10, 20))},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.obv"),
            params={},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.volume_sma"),
            params={"window": ExplicitValuesSpec(name="window", values=(10, 20))},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.vwap"),
            params={"window": ExplicitValuesSpec(name="window", values=(10, 20))},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.vwap_deviation"),
            params={
                "mult": ExplicitValuesSpec(name="mult", values=(2.0, 1.5)),
                "window": ExplicitValuesSpec(name="window", values=(20, 10)),
            },
            layout_preference=Layout.TIME_MAJOR,
        ),
    )

    for grid in grids:
        tensor = engine.compute(
            ComputeRequest(
                candles=candles,
                grid=grid,
                max_variants_guard=100_000,
            )
        )

        assert tensor.values.dtype == np.float32
        assert tensor.values.flags["C_CONTIGUOUS"]
        assert tensor.values.shape[0] == candles.ts_open.shape[0]
        assert tensor.values.shape[1] == tensor.meta.variants

        if grid.indicator_id.value == "volume.vwap_deviation":
            assert tensor.axes[0].name == "mult"
            assert tensor.axes[0].values_float == (2.0, 1.5)
            assert tensor.axes[1].name == "window"
            assert tensor.axes[1].values_int == (20, 10)
