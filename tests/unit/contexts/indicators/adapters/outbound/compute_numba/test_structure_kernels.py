from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_structure_grid_f32 as compute_structure_grid_numba_f32,
)
from trading.contexts.indicators.adapters.outbound.compute_numpy import (
    compute_structure_grid_f32 as compute_structure_grid_numpy_f32,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    ExplicitValuesSpec,
)
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.definitions.structure import defs as structure_defs
from trading.contexts.indicators.domain.entities import Layout, ParamKind
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
    Build deterministic UTC half-open time range for structure kernel tests.

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
    start = UtcTimestamp(datetime(2026, 2, 13, 9, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 13, 17, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def _candles_with_holes(*, t_size: int) -> CandleArrays:
    """
    Build deterministic OHLCV candles with NaN holes for structure policy tests.

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
    base = np.linspace(80.0, 220.0, t_size, dtype=np.float32)
    open_series = np.ascontiguousarray(base + rng.normal(0.0, 0.6, t_size).astype(np.float32))
    high_delta = np.abs(rng.normal(1.1, 0.35, t_size).astype(np.float32))
    low_delta = np.abs(rng.normal(1.0, 0.35, t_size).astype(np.float32))
    high_series = np.ascontiguousarray(open_series + high_delta)
    low_series = np.ascontiguousarray(open_series - low_delta)
    close_series = np.ascontiguousarray(
        open_series + rng.normal(0.0, 0.45, t_size).astype(np.float32)
    )
    volume_series = np.ascontiguousarray(rng.uniform(50.0, 1500.0, t_size).astype(np.float32))

    hole_indices = np.asarray([0, 8, 19, 47, 95, 144, 233, 377, 511], dtype=np.int64)
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
    Build deterministic Numba compute adapter instance for structure tests.

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


def test_numba_structure_kernels_match_numpy_oracle_with_nan_holes() -> None:
    """
    Verify Numba structure kernels match NumPy oracle on deterministic NaN-hole inputs.

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
    windows_i64 = np.asarray([5, 10, 20, 30], dtype=np.int64)

    cases: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "structure.zscore",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
            },
        ),
        (
            "structure.percent_rank",
            {
                "source_variants": source_variants,
                "windows": windows_i64,
            },
        ),
        (
            "structure.candle_stats",
            {
                "open": source_map["open"],
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
            },
        ),
        (
            "structure.candle_upper_wick_pct",
            {
                "open": source_map["open"],
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
            },
        ),
        (
            "structure.candle_stats_atr_norm",
            {
                "open": source_map["open"],
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "atr_windows": windows_i64,
            },
        ),
        (
            "structure.candle_range_atr",
            {
                "open": source_map["open"],
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "atr_windows": windows_i64,
            },
        ),
        (
            "structure.pivots",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "lefts": np.asarray([2, 3, 4], dtype=np.int64),
                "rights": np.asarray([2, 3, 4], dtype=np.int64),
            },
        ),
        (
            "structure.pivot_low",
            {
                "high": source_map["high"],
                "low": source_map["low"],
                "lefts": np.asarray([2, 3, 4], dtype=np.int64),
                "rights": np.asarray([2, 3, 4], dtype=np.int64),
            },
        ),
        (
            "structure.distance_to_ma_norm",
            {
                "source_variants": source_variants,
                "high": source_map["high"],
                "low": source_map["low"],
                "close": source_map["close"],
                "windows": windows_i64,
            },
        ),
    )

    for indicator_id, kwargs in cases:
        numba_out = compute_structure_grid_numba_f32(indicator_id=indicator_id, **kwargs)
        numpy_out = compute_structure_grid_numpy_f32(indicator_id=indicator_id, **kwargs)

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


def test_structure_division_policies_match_expected_nan_behavior() -> None:
    """
    Verify `sd==0`, `range==0`, and `atr==0` policies produce deterministic NaNs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Div-by-zero policy is fixed by EPIC-09 for structure indicators.
    Raises:
        AssertionError: If any policy emits finite values where NaN is required.
    Side Effects:
        None.
    """
    flat = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

    zscore = compute_structure_grid_numba_f32(
        indicator_id="structure.zscore",
        source_variants=np.ascontiguousarray(flat.reshape(1, flat.shape[0])),
        windows=np.asarray([3], dtype=np.int64),
    )
    body_pct = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_body_pct",
        open=flat,
        high=flat,
        low=flat,
        close=flat,
    )
    body_atr = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_stats_atr_norm",
        open=flat,
        high=flat,
        low=flat,
        close=flat,
        atr_windows=np.asarray([3], dtype=np.int64),
    )
    distance = compute_structure_grid_numba_f32(
        indicator_id="structure.distance_to_ma_norm",
        source_variants=np.ascontiguousarray(flat.reshape(1, flat.shape[0])),
        high=flat,
        low=flat,
        close=flat,
        windows=np.asarray([3], dtype=np.int64),
    )

    assert np.all(np.isnan(zscore[0, :]))
    assert np.all(np.isnan(body_pct[0, :]))
    assert np.all(np.isnan(body_atr[0, :]))
    assert np.all(np.isnan(distance[0, :]))


def test_structure_pivots_use_shift_confirm_true_semantics() -> None:
    """
    Verify pivots are emitted only on confirmation index (`shift_confirm=true`).

    Args:
        None.
    Returns:
        None.
    Assumptions:
        For left=right=2, pivot at center index `c` is emitted at `c+2`.
    Raises:
        AssertionError: If pivot confirmation shift semantics regress.
    Side Effects:
        None.
    """
    high = np.asarray([1.0, 2.0, 5.0, 2.0, 1.0, 1.5, 1.2], dtype=np.float32)
    low = np.asarray([5.0, 4.0, 1.0, 4.0, 5.0, 4.5, 4.8], dtype=np.float32)

    pivot_high = compute_structure_grid_numba_f32(
        indicator_id="structure.pivots",
        high=high,
        low=low,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )
    pivot_low = compute_structure_grid_numba_f32(
        indicator_id="structure.pivot_low",
        high=high,
        low=low,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )

    expected_high = np.asarray([np.nan, np.nan, np.nan, np.nan, 5.0, np.nan, np.nan])
    expected_low = np.asarray([np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan])

    np.testing.assert_allclose(
        pivot_high[0, :],
        expected_high,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        pivot_low[0, :],
        expected_low,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_structure_wrappers_match_primary_outputs_of_base_indicators() -> None:
    """
    Verify wrapper outputs equal corresponding primary outputs from base indicators.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        v1 primary mapping is fixed: candle_stats->body_pct, candle_stats_atr_norm->body_atr,
        pivots->pivot_high.
    Raises:
        AssertionError: If wrapper delegation semantics diverge.
    Side Effects:
        None.
    """
    open_series = np.asarray([10.0, 11.0, 12.0, 12.5, 12.2, 12.7], dtype=np.float32)
    high_series = np.asarray([11.0, 12.0, 13.0, 13.0, 12.9, 13.1], dtype=np.float32)
    low_series = np.asarray([9.0, 10.0, 11.0, 11.5, 11.8, 12.2], dtype=np.float32)
    close_series = np.asarray([10.5, 11.2, 12.4, 12.2, 12.6, 12.9], dtype=np.float32)

    candle_stats = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_stats",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
    )
    candle_body_pct = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_body_pct",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
    )

    candle_stats_atr = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_stats_atr_norm",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        atr_windows=np.asarray([3], dtype=np.int64),
    )
    candle_body_atr = compute_structure_grid_numba_f32(
        indicator_id="structure.candle_body_atr",
        open=open_series,
        high=high_series,
        low=low_series,
        close=close_series,
        atr_windows=np.asarray([3], dtype=np.int64),
    )

    pivots = compute_structure_grid_numba_f32(
        indicator_id="structure.pivots",
        high=high_series,
        low=low_series,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )
    pivot_high = compute_structure_grid_numba_f32(
        indicator_id="structure.pivot_high",
        high=high_series,
        low=low_series,
        lefts=np.asarray([2], dtype=np.int64),
        rights=np.asarray([2], dtype=np.int64),
    )

    np.testing.assert_allclose(candle_stats, candle_body_pct, equal_nan=True)
    np.testing.assert_allclose(candle_stats_atr, candle_body_atr, equal_nan=True)
    np.testing.assert_allclose(pivots, pivot_high, equal_nan=True)


def test_numba_engine_supports_all_structure_indicator_ids(tmp_path: Path) -> None:
    """
    Verify compute engine dispatch executes every hard-defined structure indicator id.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Hard definitions in `structure.py` represent registry-computable ids.
    Raises:
        AssertionError: If any structure indicator fails compute tensor contracts.
    Side Effects:
        None.
    """
    candles = _candles_with_holes(t_size=320)
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")

    for definition in structure_defs():
        params: dict[str, ExplicitValuesSpec] = {}
        for param in definition.params:
            value = param.default
            if value is None:
                if param.kind is ParamKind.FLOAT:
                    value = float(param.hard_min if param.hard_min is not None else 1.0)
                elif param.kind is ParamKind.INT:
                    value = int(param.hard_min if param.hard_min is not None else 2)
                else:
                    if not param.enum_values:
                        raise AssertionError(f"enum parameter has no values: {param.name}")
                    value = param.enum_values[0]
            params[param.name] = ExplicitValuesSpec(name=param.name, values=(value,))

        source_spec = None
        if "source" in definition.axes:
            source_spec = ExplicitValuesSpec(name="source", values=("close",))

        grid = GridSpec(
            indicator_id=definition.indicator_id,
            params=params,
            source=source_spec,
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
        assert tensor.values.shape[0] == candles.ts_open.shape[0]
        assert tensor.values.shape[1] == tensor.meta.variants
