from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_trend_grid_f32 as compute_trend_grid_numba_f32,
)
from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_trend_grid_f32 as compute_trend_grid_numpy_f32,
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
    Build deterministic UTC half-open time range for trend kernel tests.

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
    start = UtcTimestamp(datetime(2026, 2, 13, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 13, 16, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles_with_holes(*, t_size: int) -> CandleArrays:
    """
    Build deterministic OHLCV candles with NaN holes for trend policy tests.

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
    rng = np.random.default_rng(20260214)
    base = np.linspace(90.0, 210.0, t_size, dtype=np.float32)
    open_series = np.ascontiguousarray(base + rng.normal(0.0, 0.7, t_size).astype(np.float32))
    high_delta = np.abs(rng.normal(1.4, 0.4, t_size).astype(np.float32))
    low_delta = np.abs(rng.normal(1.4, 0.4, t_size).astype(np.float32))
    high_series = np.ascontiguousarray(open_series + high_delta)
    low_series = np.ascontiguousarray(open_series - low_delta)
    close_series = np.ascontiguousarray(
        open_series + rng.normal(0.0, 0.55, t_size).astype(np.float32)
    )
    volume_series = np.ascontiguousarray(rng.uniform(20.0, 900.0, t_size).astype(np.float32))

    hole_indices = np.asarray([0, 5, 31, 72, 128, 256, 377, 511], dtype=np.int64)
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
    Build deterministic Numba compute adapter instance for trend tests.

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


def _source_map(*, candles: CandleArrays) -> dict[str, np.ndarray]:
    """
    Build deterministic source map matching compute engine source semantics.

    Args:
        candles: Dense candles payload.
    Returns:
        dict[str, np.ndarray]: Base + derived source vectors.
    Assumptions:
        Derived sources propagate NaNs naturally through vector arithmetic.
    Raises:
        None.
    Side Effects:
        Allocates derived source arrays.
    """
    open_series = np.ascontiguousarray(candles.open, dtype=np.float32)
    high_series = np.ascontiguousarray(candles.high, dtype=np.float32)
    low_series = np.ascontiguousarray(candles.low, dtype=np.float32)
    close_series = np.ascontiguousarray(candles.close, dtype=np.float32)

    return {
        "open": open_series,
        "high": high_series,
        "low": low_series,
        "close": close_series,
        "hlc3": np.ascontiguousarray((high_series + low_series + close_series) / np.float32(3.0)),
        "ohlc4": np.ascontiguousarray(
            (open_series + high_series + low_series + close_series) / np.float32(4.0)
        ),
    }


def test_numba_trend_kernels_match_numpy_oracle_with_nan_holes() -> None:
    """
    Verify Numba trend kernels match NumPy oracle on deterministic NaN-hole inputs.

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
    source_map = _source_map(candles=candles)

    source_variants = np.ascontiguousarray(
        np.stack(
            [
                source_map["close"],
                source_map["hlc3"],
                source_map["ohlc4"],
                source_map["open"],
            ],
            axis=0,
        )
    )

    windows_i64 = np.asarray([7, 14, 21, 34], dtype=np.int64)
    smoothings_i64 = np.asarray([5, 7, 9, 11], dtype=np.int64)
    mults_f64 = np.asarray([1.5, 2.0, 2.5, 3.0], dtype=np.float64)

    cases: tuple[tuple[str, dict[str, np.ndarray]], ...] = (
        (
            "trend.adx",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
                "smoothings": smoothings_i64,
            },
        ),
        (
            "trend.aroon",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
            },
        ),
        (
            "trend.chandelier_exit",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
        (
            "trend.donchian",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
            },
        ),
        (
            "trend.ichimoku",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "conversion_windows": np.asarray([9, 12, 15], dtype=np.int64),
                "base_windows": np.asarray([26, 30, 34], dtype=np.int64),
                "span_b_windows": np.asarray([52, 60, 68], dtype=np.int64),
                "displacements": np.asarray([26, 30, 34], dtype=np.int64),
            },
        ),
        (
            "trend.keltner",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
        (
            "trend.linreg_slope",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
            },
        ),
        (
            "trend.psar",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "accel_starts": np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
                "accel_steps": np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
                "accel_maxes": np.asarray([0.2, 0.3, 0.4], dtype=np.float64),
            },
        ),
        (
            "trend.supertrend",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
        (
            "trend.vortex",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
            },
        ),
    )

    for indicator_id, kwargs in cases:
        numba_out = compute_trend_grid_numba_f32(indicator_id=indicator_id, **kwargs)
        numpy_out = compute_trend_grid_numpy_f32(indicator_id=indicator_id, **kwargs)

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


def test_trend_directional_kernels_reset_state_on_nan_holes() -> None:
    """
    Verify PSAR/SuperTrend do not carry direction state across NaN holes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Directional kernels emit NaN on hole bars and restart on next valid segment.
    Raises:
        AssertionError: If reset-on-NaN behavior regresses.
    Side Effects:
        None.
    """
    high = np.asarray([10.0, 11.0, 12.0, 11.0, np.nan, 13.0, 14.0], dtype=np.float32)
    low = np.asarray([9.0, 10.0, 11.0, 10.0, np.nan, 12.0, 13.0], dtype=np.float32)
    close = np.asarray([9.5, 10.5, 11.5, 10.5, np.nan, 12.5, 13.5], dtype=np.float32)

    psar = compute_trend_grid_numba_f32(
        indicator_id="trend.psar",
        high=high,
        low=low,
        accel_starts=np.asarray([0.02], dtype=np.float64),
        accel_steps=np.asarray([0.02], dtype=np.float64),
        accel_maxes=np.asarray([0.2], dtype=np.float64),
    )
    supertrend = compute_trend_grid_numba_f32(
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


def test_numba_engine_preserves_trend_axis_order_and_matches_oracle(tmp_path: Path) -> None:
    """
    Verify trend engine preserves explicit axis ordering and matches numpy oracle.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Variant order for `trend.linreg_slope` follows definition axes: source, window.
    Raises:
        AssertionError: If axis ordering, dtype, or numeric output differ.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=320)
    source_map = _source_map(candles=candles)
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")

    sources = ("open", "close")
    windows = (20, 10)
    grid = GridSpec(
        indicator_id=IndicatorId("trend.linreg_slope"),
        params={"window": ExplicitValuesSpec(name="window", values=windows)},
        source=ExplicitValuesSpec(name="source", values=sources),
        layout_preference=Layout.TIME_MAJOR,
    )

    tensor = engine.compute(
        ComputeRequest(
            candles=candles,
            grid=grid,
            max_variants_guard=100_000,
        )
    )

    assert tensor.layout is Layout.TIME_MAJOR
    assert tensor.values.dtype == np.float32
    assert tensor.values.flags["C_CONTIGUOUS"]
    assert tensor.axes[0].name == "source"
    assert tensor.axes[0].values_enum == sources
    assert tensor.axes[1].name == "window"
    assert tensor.axes[1].values_int == windows

    expected = np.empty_like(tensor.values)
    variant_index = 0
    for source_name in sources:
        per_source_variants = np.ascontiguousarray(
            np.repeat(
                source_map[source_name].reshape(1, -1),
                repeats=len(windows),
                axis=0,
            )
        )
        oracle = compute_trend_grid_numpy_f32(
            indicator_id="trend.linreg_slope",
            source_variants=per_source_variants,
            windows=np.asarray(windows, dtype=np.int64),
        )
        for window_index in range(len(windows)):
            expected[:, variant_index] = oracle[window_index, :]
            variant_index += 1

    np.testing.assert_allclose(
        tensor.values,
        expected,
        rtol=3e-5,
        atol=3e-5,
        equal_nan=True,
    )
