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
    Build deterministic dense candles payload for volatility/momentum perf-smoke.

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
    start = datetime(2026, 2, 12, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=t_size)

    ts_open = np.arange(t_size, dtype=np.int64) * np.int64(60_000)
    base = np.linspace(100.0, 200.0, t_size, dtype=np.float32)
    noise = np.sin(np.linspace(0.0, 24.0, t_size, dtype=np.float32)) * np.float32(0.8)
    open_series = np.ascontiguousarray(base + noise)
    high_series = np.ascontiguousarray(open_series + np.float32(1.4))
    low_series = np.ascontiguousarray(open_series - np.float32(1.4))
    close_series = np.ascontiguousarray(open_series + np.float32(0.2))
    volume_series = np.ascontiguousarray(np.linspace(50.0, 500.0, t_size, dtype=np.float32))

    open_series[::211] = np.nan
    high_series[::211] = np.nan
    low_series[::211] = np.nan
    close_series[::211] = np.nan
    volume_series[::211] = np.nan

    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(start),
            end=UtcTimestamp(end),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        volume=volume_series,
    )


def test_indicators_volatility_momentum_perf_smoke(tmp_path: Path) -> None:
    """
    Run perf-smoke for ATR/RSI/MACD and assert guard-safe successful completion.

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

    candles = _candles(t_size=50_000)

    grids = (
        GridSpec(
            indicator_id=IndicatorId("volatility.atr"),
            params={
                "window": RangeValuesSpec(
                    name="window",
                    start=5,
                    stop_inclusive=120,
                    step=1,
                ),
            },
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("momentum.rsi"),
            params={
                "window": RangeValuesSpec(
                    name="window",
                    start=3,
                    stop_inclusive=60,
                    step=1,
                ),
            },
            source=ExplicitValuesSpec(
                name="source",
                values=("close", "hlc3", "ohlc4", "low", "high", "open"),
            ),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("momentum.macd"),
            params={
                "fast_window": ExplicitValuesSpec(name="fast_window", values=(8, 12, 16)),
                "slow_window": ExplicitValuesSpec(name="slow_window", values=(20, 26, 34)),
                "signal_window": ExplicitValuesSpec(name="signal_window", values=(5, 9)),
            },
            source=ExplicitValuesSpec(
                name="source",
                values=("close", "hlc3", "ohlc4"),
            ),
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

    assert elapsed_by_id["volatility.atr"] > 0.0
    assert elapsed_by_id["momentum.rsi"] > 0.0
    assert elapsed_by_id["momentum.macd"] > 0.0
