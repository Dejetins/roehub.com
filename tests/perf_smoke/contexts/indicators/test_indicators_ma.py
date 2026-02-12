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
    Build deterministic dense candles payload for MA perf-smoke.

    Args:
        t_size: Number of one-minute rows.
    Returns:
        CandleArrays: Valid dense OHLCV payload.
    Assumptions:
        Payload is deterministic and contains sparse NaN holes.
    Raises:
        ValueError: If generated arrays violate CandleArrays invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    start = datetime(2026, 2, 10, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=t_size)

    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60_000)
    base = np.linspace(100.0, 140.0, t_size, dtype=np.float32)
    base[::97] = np.nan

    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(start),
            end=UtcTimestamp(end),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=base,
        high=base + np.float32(1.0),
        low=base - np.float32(1.0),
        close=base + np.float32(0.2),
        volume=np.linspace(10.0, 100.0, t_size, dtype=np.float32),
    )


def test_indicators_ma_grid_perf_smoke(tmp_path: Path) -> None:
    """
    Run perf-smoke for MA grid compute and assert guard-safe successful completion.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Perf-smoke validates successful runtime execution, not strict latency SLA.
    Raises:
        AssertionError: If output tensor shape/contracts are violated.
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

    candles = _candles(t_size=4_096)
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={
            "window": RangeValuesSpec(
                name="window",
                start=5,
                stop_inclusive=200,
                step=1,
            ),
        },
        source=ExplicitValuesSpec(
            name="source",
            values=("close", "hlc3", "ohlc4", "low", "high", "open"),
        ),
        layout_preference=Layout.TIME_MAJOR,
    )

    started = time.perf_counter()
    tensor = compute.compute(
        ComputeRequest(
            candles=candles,
            grid=grid,
            max_variants_guard=600_000,
        )
    )
    elapsed = time.perf_counter() - started

    assert tensor.meta.variants == 1_176
    assert tensor.values.shape == (4_096, 1_176)
    assert tensor.values.dtype == np.float32
    assert tensor.values.flags["C_CONTIGUOUS"]
    assert elapsed > 0.0
