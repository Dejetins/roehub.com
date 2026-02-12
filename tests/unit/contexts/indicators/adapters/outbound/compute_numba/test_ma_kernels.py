from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_ma_grid_f32 as compute_ma_grid_numba_f32,
)
from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_ma_grid_f32 as compute_ma_grid_numpy_f32,
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
    Build deterministic UTC half-open time range for test candles.

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
    start = UtcTimestamp(datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles_with_holes(*, t_size: int) -> CandleArrays:
    """
    Build deterministic OHLCV candles with NaN holes for MA policy tests.

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
    rng = np.random.default_rng(20260211)
    base = np.linspace(100.0, 140.0, t_size, dtype=np.float32)
    open_series = np.ascontiguousarray(base + rng.normal(0.0, 0.25, t_size).astype(np.float32))
    high_series = np.ascontiguousarray(open_series + np.float32(1.5))
    low_series = np.ascontiguousarray(open_series - np.float32(1.5))
    close_series = np.ascontiguousarray(open_series + np.float32(0.3))
    volume_series = np.ascontiguousarray(
        np.linspace(10.0, 100.0, t_size, dtype=np.float32)
    )

    hole_indices = np.asarray([3, 17, 64, 129, 233], dtype=np.int64)
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
    Build deterministic Numba compute adapter instance for MA tests.

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
        "hl2": np.ascontiguousarray((high_series + low_series) / np.float32(2.0)),
        "hlc3": np.ascontiguousarray(
            (high_series + low_series + close_series) / np.float32(3.0)
        ),
        "ohlc4": np.ascontiguousarray(
            (open_series + high_series + low_series + close_series) / np.float32(4.0)
        ),
        "volume": np.ascontiguousarray(candles.volume, dtype=np.float32),
    }


def test_numba_ma_kernels_match_numpy_oracle_with_nan_holes() -> None:
    """
    Verify Numba MA kernels match NumPy oracle on deterministic NaN-hole inputs.

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
    rng = np.random.default_rng(20260212)
    t_size = 512
    source = np.ascontiguousarray(rng.normal(100.0, 5.0, t_size).astype(np.float32))
    volume = np.ascontiguousarray(rng.uniform(1.0, 100.0, t_size).astype(np.float32))
    windows = np.ascontiguousarray(np.asarray([2, 5, 11, 23], dtype=np.int64))

    hole_indices = np.asarray([0, 7, 64, 129, 311], dtype=np.int64)
    source[hole_indices] = np.nan
    volume[[3, 64, 200]] = np.nan
    volume[[9, 10, 11]] = 0.0

    indicator_ids = (
        "ma.sma",
        "ma.ema",
        "ma.wma",
        "ma.lwma",
        "ma.rma",
        "ma.smma",
        "ma.vwma",
        "ma.dema",
        "ma.tema",
        "ma.zlema",
        "ma.hma",
    )

    for indicator_id in indicator_ids:
        numba_out = compute_ma_grid_numba_f32(
            indicator_id=indicator_id,
            source=source,
            windows=windows,
            volume=volume,
        )
        numpy_out = compute_ma_grid_numpy_f32(
            indicator_id=indicator_id,
            source=source,
            windows=windows,
            volume=volume,
        )

        assert numba_out.dtype == np.float32
        assert numba_out.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(
            numba_out,
            numpy_out,
            rtol=2e-5,
            atol=2e-5,
            equal_nan=True,
            err_msg=f"indicator_id={indicator_id}",
        )


def test_numba_engine_matches_numpy_oracle_for_all_derived_sources(tmp_path: Path) -> None:
    """
    Verify engine output matches NumPy oracle for base + derived MA sources.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Variant order follows definition axes order: source, then window.
    Raises:
        AssertionError: If axis ordering, dtype, contiguity, or numeric output differ.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=320)
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")

    sources = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")
    windows = (3, 5)
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
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
    assert tensor.axes[0].values_enum == sources
    assert tensor.axes[1].values_int == windows

    source_map = _source_map(candles=candles)
    expected = np.empty_like(tensor.values)
    variant_index = 0
    for source_name in sources:
        oracle = compute_ma_grid_numpy_f32(
            indicator_id="ma.sma",
            source=source_map[source_name],
            windows=np.asarray(windows, dtype=np.int64),
            volume=source_map["volume"],
        )
        for window_index in range(len(windows)):
            expected[:, variant_index] = oracle[:, window_index]
            variant_index += 1

    np.testing.assert_allclose(
        tensor.values,
        expected,
        rtol=2e-5,
        atol=2e-5,
        equal_nan=True,
    )
