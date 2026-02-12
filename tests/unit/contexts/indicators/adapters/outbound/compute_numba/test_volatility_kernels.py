from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_volatility_grid_f32 as compute_volatility_grid_numba_f32,
)
from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_volatility_grid_f32 as compute_volatility_grid_numpy_f32,
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
    start = UtcTimestamp(datetime(2026, 2, 12, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 12, 14, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles_with_holes(*, t_size: int) -> CandleArrays:
    """
    Build deterministic OHLCV candles with NaN holes for volatility policy tests.

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
    rng = np.random.default_rng(20260212)
    base = np.linspace(100.0, 160.0, t_size, dtype=np.float32)
    open_series = np.ascontiguousarray(base + rng.normal(0.0, 0.8, t_size).astype(np.float32))
    high_delta = np.abs(rng.normal(1.2, 0.4, t_size).astype(np.float32))
    low_delta = np.abs(rng.normal(1.2, 0.4, t_size).astype(np.float32))
    high_series = np.ascontiguousarray(open_series + high_delta)
    low_series = np.ascontiguousarray(open_series - low_delta)
    close_series = np.ascontiguousarray(
        open_series + rng.normal(0.0, 0.5, t_size).astype(np.float32)
    )
    volume_series = np.ascontiguousarray(rng.uniform(10.0, 500.0, t_size).astype(np.float32))

    hole_indices = np.asarray([0, 7, 33, 89, 144, 233, 377], dtype=np.int64)
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
    Build deterministic Numba compute adapter instance for volatility tests.

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


def test_numba_volatility_kernels_match_numpy_oracle_with_nan_holes() -> None:
    """
    Verify Numba volatility kernels match NumPy oracle on deterministic NaN-hole inputs.

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
    candles = _candles_with_holes(t_size=512)
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
    windows_i64 = np.asarray([5, 14, 31, 55], dtype=np.int64)
    mults_f64 = np.asarray([1.5, 2.0, 2.5, 3.0], dtype=np.float64)
    annualizations_i64 = np.asarray([252, 365, 252, 365], dtype=np.int64)

    cases: tuple[tuple[str, dict[str, np.ndarray]], ...] = (
        (
            "volatility.tr",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
            },
        ),
        (
            "volatility.atr",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64[:3],
            },
        ),
        (
            "volatility.stddev",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
            },
        ),
        (
            "volatility.variance",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
            },
        ),
        (
            "volatility.hv",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
                "annualizations": annualizations_i64,
            },
        ),
        (
            "volatility.bbands",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
        (
            "volatility.bbands_bandwidth",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
        (
            "volatility.bbands_percent_b",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
                "mults": mults_f64,
            },
        ),
    )

    for indicator_id, kwargs in cases:
        numba_out = compute_volatility_grid_numba_f32(
            indicator_id=indicator_id,
            **kwargs,
        )
        numpy_out = compute_volatility_grid_numpy_f32(
            indicator_id=indicator_id,
            **kwargs,
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


def test_numba_engine_supports_all_volatility_indicators(tmp_path: Path) -> None:
    """
    Verify compute engine dispatch executes all volatility indicator ids in EPIC-07.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Engine keeps explicit axis values order from request materialization.
    Raises:
        AssertionError: If any volatility indicator fails compute tensor contracts.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=320)
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")

    grids = (
        GridSpec(
            indicator_id=IndicatorId("volatility.tr"),
            params={},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.atr"),
            params={"window": ExplicitValuesSpec(name="window", values=(5, 14))},
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.stddev"),
            params={"window": ExplicitValuesSpec(name="window", values=(5, 14))},
            source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.variance"),
            params={"window": ExplicitValuesSpec(name="window", values=(5, 14))},
            source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.hv"),
            params={
                "window": ExplicitValuesSpec(name="window", values=(10,)),
                "annualization": ExplicitValuesSpec(name="annualization", values=(252, 365)),
            },
            source=ExplicitValuesSpec(name="source", values=("close",)),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.bbands"),
            params={
                "window": ExplicitValuesSpec(name="window", values=(20,)),
                "mult": ExplicitValuesSpec(name="mult", values=(1.5, 2.0)),
            },
            source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.bbands_bandwidth"),
            params={
                "window": ExplicitValuesSpec(name="window", values=(20,)),
                "mult": ExplicitValuesSpec(name="mult", values=(1.5, 2.0)),
            },
            source=ExplicitValuesSpec(name="source", values=("close",)),
            layout_preference=Layout.TIME_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("volatility.bbands_percent_b"),
            params={
                "window": ExplicitValuesSpec(name="window", values=(20,)),
                "mult": ExplicitValuesSpec(name="mult", values=(1.5, 2.0)),
            },
            source=ExplicitValuesSpec(name="source", values=("close",)),
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
        assert tensor.layout is Layout.TIME_MAJOR
        assert tensor.values.dtype == np.float32
        assert tensor.values.flags["C_CONTIGUOUS"]
        assert tensor.values.shape[0] == candles.ts_open.shape[0]
        assert tensor.values.shape[1] == tensor.meta.variants
