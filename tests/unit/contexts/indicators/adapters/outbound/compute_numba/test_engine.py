from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.engine import (
    _build_series_map,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    ExplicitValuesSpec,
)
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorId, InputSeries, Layout
from trading.contexts.indicators.domain.errors import ComputeBudgetExceeded
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
        Range only provides metadata for `CandleArrays` contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = UtcTimestamp(datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 11, 11, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles(*, t_size: int) -> CandleArrays:
    """
    Build deterministic dense candle arrays payload.

    Args:
        t_size: Number of rows in the timeline.
    Returns:
        CandleArrays: Valid dense OHLCV payload.
    Assumptions:
        Generated OHLCV values are monotonic and float32.
    Raises:
        ValueError: If generated payload violates `CandleArrays` invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60)
    base = np.linspace(100.0, 130.0, t_size, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=_time_range(),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=base,
        high=base + np.float32(1.0),
        low=base - np.float32(1.0),
        close=base + np.float32(0.5),
        volume=np.linspace(10.0, 20.0, t_size, dtype=np.float32),
    )


def _compute_engine(
    *,
    cache_dir: Path,
    max_compute_bytes_total: int = 5 * 1024**3,
) -> NumbaIndicatorCompute:
    """
    Build `NumbaIndicatorCompute` instance for tests.

    Args:
        cache_dir: Numba cache directory path.
        max_compute_bytes_total: Total memory budget.
    Returns:
        NumbaIndicatorCompute: Ready compute adapter.
    Assumptions:
        Hard definitions from `all_defs()` are deterministic.
    Raises:
        ValueError: If config values are invalid.
    Side Effects:
        None.
    """
    config = IndicatorsComputeNumbaConfig(
        numba_num_threads=1,
        numba_cache_dir=cache_dir,
        max_compute_bytes_total=max_compute_bytes_total,
    )
    return NumbaIndicatorCompute(defs=all_defs(), config=config)


def _compute_request(*, candles: CandleArrays, layout: Layout) -> ComputeRequest:
    """
    Build deterministic compute request for `ma.sma`.

    Args:
        candles: Dense candles payload.
        layout: Requested output layout.
    Returns:
        ComputeRequest: Valid request for engine compute.
    Assumptions:
        `ma.sma` uses axes (`source`, `window`) in hard definitions.
    Raises:
        ValueError: If DTO invariants fail.
    Side Effects:
        None.
    """
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={
            "window": ExplicitValuesSpec(name="window", values=(20, 10)),
        },
        source=ExplicitValuesSpec(name="source", values=("open", "close")),
        layout_preference=layout,
    )
    return ComputeRequest(candles=candles, grid=grid, max_variants_guard=100_000)


def test_compute_supports_time_major_layout_and_preserves_explicit_axis_order(
    tmp_path: Path,
) -> None:
    """
    Verify TIME_MAJOR compute output shape, dtype, and explicit axis ordering.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Engine keeps explicit axis values ordering from request materialization.
    Raises:
        AssertionError: If shape/dtype/axes contract is violated.
    Side Effects:
        None.
    """
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")
    candles = _candles(t_size=128)
    req = _compute_request(candles=candles, layout=Layout.TIME_MAJOR)

    tensor = engine.compute(req)

    assert tensor.layout is Layout.TIME_MAJOR
    assert tensor.values.dtype == np.float32
    assert tensor.values.shape == (128, 4)
    assert tensor.meta.t == 128
    assert tensor.meta.variants == 4
    assert tensor.axes[0].name == "source"
    assert tensor.axes[0].values_enum == ("open", "close")
    assert tensor.axes[1].name == "window"
    assert tensor.axes[1].values_int == (20, 10)


def test_compute_supports_variant_major_layout(tmp_path: Path) -> None:
    """
    Verify VARIANT_MAJOR compute output shape and dtype contract.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Variant count is cartesian product of source and window axes.
    Raises:
        AssertionError: If output tensor contract is violated.
    Side Effects:
        None.
    """
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")
    candles = _candles(t_size=96)
    req = _compute_request(candles=candles, layout=Layout.VARIANT_MAJOR)

    tensor = engine.compute(req)

    assert tensor.layout is Layout.VARIANT_MAJOR
    assert tensor.values.dtype == np.float32
    assert tensor.values.shape == (4, 96)
    assert tensor.meta.t == 96
    assert tensor.meta.variants == 4


def test_compute_raises_budget_exceeded_for_total_memory_guard(tmp_path: Path) -> None:
    """
    Verify compute enforces `max_compute_bytes_total` and returns detailed error.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Guard uses total bytes estimate (`bytes_out + workspace reserve`).
    Raises:
        AssertionError: If expected error is not raised.
    Side Effects:
        None.
    """
    engine = _compute_engine(
        cache_dir=tmp_path / "numba-cache",
        max_compute_bytes_total=8_192,
    )
    candles = _candles(t_size=300)
    req = _compute_request(candles=candles, layout=Layout.TIME_MAJOR)

    with pytest.raises(ComputeBudgetExceeded) as exc_info:
        engine.compute(req)

    details = exc_info.value.details
    assert list(details.keys()) == [
        "T",
        "V",
        "bytes_out",
        "bytes_total_est",
        "max_compute_bytes_total",
    ]
    assert details["T"] == 300
    assert details["V"] == 4
    assert details["max_compute_bytes_total"] == 8_192


def test_build_series_map_allocates_only_requested_close_series() -> None:
    """
    Verify lazy series-map path allocates only `close` when request requires only close source.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Source set is resolved before matrix kernels and can exclude derived OHLC aggregates.
    Raises:
        AssertionError: If unnecessary series are materialized.
    Side Effects:
        None.
    """
    candles = _candles(t_size=32)

    series_map = _build_series_map(
        candles=candles,
        required_sources=(InputSeries.CLOSE.value,),
    )

    assert tuple(series_map.keys()) == (InputSeries.CLOSE.value,)
    assert series_map[InputSeries.CLOSE.value].dtype == np.float32
    assert series_map[InputSeries.CLOSE.value].flags.c_contiguous


def test_build_series_map_lazily_allocates_derived_source_only_when_requested() -> None:
    """
    Verify lazy series-map computes derived `ohlc4` only when request explicitly asks for it.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Derived source values are computed from float32 contiguous OHLC arrays.
    Raises:
        AssertionError: If derived series is missing or incorrectly computed.
    Side Effects:
        None.
    """
    candles = _candles(t_size=24)

    series_map = _build_series_map(
        candles=candles,
        required_sources=(InputSeries.OHLC4.value,),
    )
    expected = np.ascontiguousarray(
        (candles.open + candles.high + candles.low + candles.close) / np.float32(4.0),
        dtype=np.float32,
    )

    assert tuple(series_map.keys()) == (InputSeries.OHLC4.value,)
    np.testing.assert_allclose(series_map[InputSeries.OHLC4.value], expected)
