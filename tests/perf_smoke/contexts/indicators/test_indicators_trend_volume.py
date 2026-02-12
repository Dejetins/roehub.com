from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    ExplicitValuesSpec,
    RangeValuesSpec,
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


def _candles(*, t_size: int) -> CandleArrays:
    """
    Build deterministic dense candles payload for trend/volume perf-smoke.

    Args:
        t_size: Number of one-minute rows.
    Returns:
        CandleArrays: Valid dense OHLCV payload.
    Assumptions:
        Payload is deterministic and includes sparse NaN holes.
    Raises:
        ValueError: If generated arrays violate CandleArrays invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    start = datetime(2026, 2, 13, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=t_size)

    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60_000)
    base = np.linspace(100.0, 260.0, t_size, dtype=np.float32)
    drift = np.sin(np.linspace(0.0, 60.0, t_size, dtype=np.float32)) * np.float32(0.9)
    open_series = np.ascontiguousarray(base + drift)
    high_series = np.ascontiguousarray(open_series + np.float32(1.6))
    low_series = np.ascontiguousarray(open_series - np.float32(1.6))
    close_series = np.ascontiguousarray(open_series + np.float32(0.25))
    volume_series = np.ascontiguousarray(np.linspace(40.0, 800.0, t_size, dtype=np.float32))

    open_series[::257] = np.nan
    high_series[::257] = np.nan
    low_series[::257] = np.nan
    close_series[::257] = np.nan
    volume_series[::257] = np.nan

    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(start=UtcTimestamp(start), end=UtcTimestamp(end)),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        volume=volume_series,
    )


def test_indicators_trend_volume_perf_smoke(tmp_path: Path) -> None:
    """
    Run perf-smoke for trend+volume grid compute and assert guard-safe completion.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Perf-smoke validates successful runtime execution, not strict latency SLA.
    Raises:
        AssertionError: If output tensor contracts are violated.
    Side Effects:
        Triggers Numba warmup and kernel JIT compilation.
    """
    config = IndicatorsComputeNumbaConfig(
        numba_num_threads=1,
        numba_cache_dir=tmp_path / "numba-cache",
        max_compute_bytes_total=5 * 1024**3,
        max_variants_per_compute=600_000,
    )
    compute = NumbaIndicatorCompute(defs=all_defs(), config=config)
    compute.warmup()

    candles = _candles(t_size=40_000)

    grids = (
        GridSpec(
            indicator_id=IndicatorId("trend.adx"),
            params={
                "window": RangeValuesSpec(
                    name="window",
                    start=5,
                    stop_inclusive=50,
                    step=1,
                ),
                "smoothing": RangeValuesSpec(
                    name="smoothing",
                    start=5,
                    stop_inclusive=20,
                    step=1,
                ),
            },
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volume.vwap_deviation"),
            params={
                "mult": ExplicitValuesSpec(name="mult", values=(1.0, 2.0, 3.0)),
                "window": RangeValuesSpec(
                    name="window",
                    start=5,
                    stop_inclusive=80,
                    step=5,
                ),
            },
            layout_preference=Layout.TIME_MAJOR,
        ),
    )

    elapsed_by_id: dict[str, float] = {}
    for grid in grids:
        started = time.perf_counter()
        tensor = compute.compute(
            ComputeRequest(
                candles=candles,
                grid=grid,
                max_variants_guard=600_000,
            )
        )
        elapsed = time.perf_counter() - started
        elapsed_by_id[grid.indicator_id.value] = elapsed

        assert tensor.meta.variants <= 600_000
        assert tensor.values.shape[0] == candles.ts_open.shape[0]
        assert tensor.values.shape[1] == tensor.meta.variants
        assert tensor.values.dtype == np.float32
        assert tensor.values.flags["C_CONTIGUOUS"]
        assert elapsed > 0.0

    assert elapsed_by_id["trend.adx"] > 0.0
    assert elapsed_by_id["volume.vwap_deviation"] > 0.0
