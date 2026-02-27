from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.indicators.adapters.outbound.compute_numba import NumbaIndicatorCompute
from trading.contexts.indicators.adapters.outbound.compute_numba import (
    engine as numba_engine,
)
from trading.contexts.indicators.adapters.outbound.compute_numba.engine import (
    _build_series_map,
)
from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    PRECISION_MODE_FLOAT32,
    PRECISION_MODE_FLOAT64,
    PRECISION_MODE_MIXED,
    compute_momentum_grid_f32,
    compute_structure_grid_f32,
    compute_trend_grid_f32,
    compute_volatility_grid_f32,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    ExplicitValuesSpec,
)
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import (
    AxisDef,
    IndicatorId,
    InputSeries,
    Layout,
)
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


def _axes_for_variant_expansion() -> tuple[AxisDef, ...]:
    """
    Build representative deterministic axes for repeat/tile variant expansion tests.

    Args:
        None.
    Returns:
        tuple[AxisDef, ...]: Three-axis tuple with enum/int/float values.
    Assumptions:
        Axis order (`source`, `window`, `mult`) defines cartesian variant ordering.
    Raises:
        ValueError: If axis definitions violate `AxisDef` invariants.
    Side Effects:
        None.
    """
    return (
        AxisDef(name="source", values_enum=("open", "close")),
        AxisDef(name="window", values_int=(5, 10, 20)),
        AxisDef(name="mult", values_float=(1.25, 2.5)),
    )


def _expand_axis_values_ndindex_reference(
    *,
    axis_values: tuple[int, ...] | tuple[float, ...] | tuple[str, ...],
    axis_lengths: tuple[int, ...],
    axis_index: int,
) -> tuple[int | float | str, ...]:
    """
    Expand axis values with legacy `np.ndindex` cartesian traversal for equivalence checks.

    Args:
        axis_values: Values of one axis in materialized order.
        axis_lengths: Length of each axis in definition order.
        axis_index: Target axis index to project from each coordinate.
    Returns:
        tuple[int | float | str, ...]: Expanded value per variant coordinate.
    Assumptions:
        `np.ndindex(axis_lengths)` follows the same deterministic variant traversal as engine v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    values: list[int | float | str] = []
    for coordinate in np.ndindex(axis_lengths):
        values.append(axis_values[coordinate[axis_index]])
    return tuple(values)


def _grouped_source_grids() -> tuple[GridSpec, ...]:
    """
    Build representative source-parameterized grids for grouped compute regression tests.

    Args:
        None.
    Returns:
        tuple[GridSpec, ...]: Deterministic grids covering volatility/momentum/trend/structure.
    Assumptions:
        Grid order matches expected grouped-source indicator coverage for phase-3 pipeline.
    Raises:
        ValueError: If grid specs violate domain DTO invariants.
    Side Effects:
        None.
    """
    return (
        GridSpec(
            indicator_id=IndicatorId("volatility.stddev"),
            params={"window": ExplicitValuesSpec(name="window", values=(5, 10))},
            source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
            layout_preference=Layout.VARIANT_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("momentum.rsi"),
            params={"window": ExplicitValuesSpec(name="window", values=(7, 14))},
            source=ExplicitValuesSpec(name="source", values=("open", "close")),
            layout_preference=Layout.VARIANT_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("trend.linreg_slope"),
            params={"window": ExplicitValuesSpec(name="window", values=(8, 16))},
            source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
            layout_preference=Layout.VARIANT_MAJOR,
        ),
        GridSpec(
            indicator_id=IndicatorId("structure.distance_to_ma_norm"),
            params={"window": ExplicitValuesSpec(name="window", values=(5, 10))},
            source=ExplicitValuesSpec(name="source", values=("close", "ohlc4")),
            layout_preference=Layout.VARIANT_MAJOR,
        ),
    )


def _definition_by_id(*, indicator_id: str):
    """
    Resolve hard indicator definition by identifier for test reference computations.

    Args:
        indicator_id: Target indicator id.
    Returns:
        IndicatorDef: Matching hard definition.
    Assumptions:
        Hard definitions from `all_defs()` are deterministic and include the requested id.
    Raises:
        StopIteration: If indicator id is missing in hard definitions.
    Side Effects:
        None.
    """
    return next(
        definition
        for definition in all_defs()
        if definition.indicator_id.value == indicator_id
    )


def _legacy_full_source_matrix_reference(
    *,
    grid: GridSpec,
    candles: CandleArrays,
    axes: tuple[AxisDef, ...],
) -> np.ndarray:
    """
    Compute legacy full-`source_variants` reference matrix for grouped-source indicators.

    Args:
        grid: Grid spec under verification.
        candles: Input candle arrays.
        axes: Materialized axes from engine compute result.
    Returns:
        np.ndarray: Expected variant-major matrix `(V, T)` with legacy full-matrix semantics.
    Assumptions:
        Legacy semantics used one full duplicated `(V, T)` `source_variants` matrix.
    Raises:
        AssertionError: If indicator id is outside covered grouped-source regression set.
    Side Effects:
        Allocates legacy full-source matrix and kernel outputs.
    """
    definition = _definition_by_id(indicator_id=grid.indicator_id.value)
    t_size = int(candles.ts_open.shape[0])
    required_sources = numba_engine._required_sources_for_request(
        definition=definition,
        axes=axes,
    )
    series_map = _build_series_map(candles=candles, required_sources=required_sources)
    variant_source_labels = numba_engine._variant_source_labels(
        definition=definition,
        axes=axes,
    )
    numba_engine._validate_required_series_available(
        variant_source_labels=variant_source_labels,
        available_series=series_map,
    )
    source_variants = numba_engine._build_variant_source_matrix(
        variant_source_labels=variant_source_labels,
        available_series=series_map,
        t_size=t_size,
    )

    indicator_id = grid.indicator_id.value
    if indicator_id == "volatility.stddev":
        windows_i64 = np.ascontiguousarray(
            np.asarray(
                numba_engine._variant_int_values(axes=axes, axis_name="window"),
                dtype=np.int64,
            )
        )
        return compute_volatility_grid_f32(
            indicator_id=indicator_id,
            source_variants=source_variants,
            windows=windows_i64,
        )

    if indicator_id == "momentum.rsi":
        windows_i64 = np.ascontiguousarray(
            np.asarray(
                numba_engine._variant_int_values(axes=axes, axis_name="window"),
                dtype=np.int64,
            )
        )
        return compute_momentum_grid_f32(
            indicator_id=indicator_id,
            source_variants=source_variants,
            windows=windows_i64,
        )

    if indicator_id == "trend.linreg_slope":
        windows_i64 = np.ascontiguousarray(
            np.asarray(
                numba_engine._variant_int_values(axes=axes, axis_name="window"),
                dtype=np.int64,
            )
        )
        return compute_trend_grid_f32(
            indicator_id=indicator_id,
            source_variants=source_variants,
            windows=windows_i64,
        )

    if indicator_id == "structure.distance_to_ma_norm":
        windows_i64 = np.ascontiguousarray(
            np.asarray(
                numba_engine._variant_int_values(axes=axes, axis_name="window"),
                dtype=np.int64,
            )
        )
        return compute_structure_grid_f32(
            indicator_id=indicator_id,
            source_variants=source_variants,
            high=series_map[InputSeries.HIGH.value],
            low=series_map[InputSeries.LOW.value],
            close=series_map[InputSeries.CLOSE.value],
            windows=windows_i64,
        )

    raise AssertionError(f"unexpected indicator_id for grouped-source reference: {indicator_id}")


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
    assert tensor.values.flags.c_contiguous
    assert tensor.meta.t == 96
    assert tensor.meta.variants == 4


def test_compute_variant_major_reuses_valid_matrix_without_extra_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify VARIANT_MAJOR path reuses valid `(V, T)` matrix without extra allocation.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        If variant-major matrix is already float32 and C-contiguous, adapter returns it as-is.
    Raises:
        AssertionError: If compute path allocates an unnecessary copy.
    Side Effects:
        Temporarily overrides MA matrix builder in compute engine module.
    """
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")
    candles = _candles(t_size=64)
    req = _compute_request(candles=candles, layout=Layout.VARIANT_MAJOR)
    expected = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)

    def _matrix_builder_stub(**_: object) -> np.ndarray:
        """
        Return prepared contiguous float32 matrix for fast-path test.

        Args:
            **_: Ignored keyword arguments from compute adapter call site.
        Returns:
            np.ndarray: Prepared contiguous `(V, T)` matrix.
        Assumptions:
            Test controls matrix shape to match request variants and timeline size.
        Raises:
            None.
        Side Effects:
            None.
        """
        return expected

    monkeypatch.setattr(
        numba_engine,
        "_compute_ma_variant_source_matrix",
        _matrix_builder_stub,
    )

    tensor = engine.compute(req)

    assert tensor.layout is Layout.VARIANT_MAJOR
    assert tensor.values is expected
    assert tensor.values.flags.c_contiguous


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


def test_variant_int_values_match_ndindex_reference_order() -> None:
    """
    Verify integer axis variant expansion preserves legacy ndindex ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Axis `window` index is stable in representative test tuple.
    Raises:
        AssertionError: If repeat/tile result differs from ndindex reference.
    Side Effects:
        None.
    """
    axes = _axes_for_variant_expansion()
    axis_lengths = tuple(axis.length() for axis in axes)
    window_index = 1
    window_values = axes[window_index].values_int
    assert window_values is not None

    expected = _expand_axis_values_ndindex_reference(
        axis_values=window_values,
        axis_lengths=axis_lengths,
        axis_index=window_index,
    )
    actual = numba_engine._variant_int_values(axes=axes, axis_name="window")

    assert actual == expected


def test_variant_float_values_match_ndindex_reference_order() -> None:
    """
    Verify float axis variant expansion preserves legacy ndindex ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Axis `mult` index is stable in representative test tuple.
    Raises:
        AssertionError: If repeat/tile result differs from ndindex reference.
    Side Effects:
        None.
    """
    axes = _axes_for_variant_expansion()
    axis_lengths = tuple(axis.length() for axis in axes)
    mult_index = 2
    mult_values = axes[mult_index].values_float
    assert mult_values is not None

    expected = _expand_axis_values_ndindex_reference(
        axis_values=mult_values,
        axis_lengths=axis_lengths,
        axis_index=mult_index,
    )
    actual = numba_engine._variant_float_values(axes=axes, axis_name="mult")

    assert actual == expected


def test_variant_window_values_match_ndindex_reference_order() -> None:
    """
    Verify window-axis helper preserves legacy ndindex ordering for MA-style axes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Window axis is present and integer-valued in representative test tuple.
    Raises:
        AssertionError: If helper result differs from ndindex reference.
    Side Effects:
        None.
    """
    axes = _axes_for_variant_expansion()
    axis_lengths = tuple(axis.length() for axis in axes)
    window_index = 1
    window_values = axes[window_index].values_int
    assert window_values is not None

    expected = _expand_axis_values_ndindex_reference(
        axis_values=window_values,
        axis_lengths=axis_lengths,
        axis_index=window_index,
    )
    actual = numba_engine._variant_window_values(axes=axes)

    assert actual == expected


def test_variant_source_labels_match_ndindex_reference_order() -> None:
    """
    Verify source-label expansion preserves legacy ndindex ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.sma` hard definition is available in deterministic hard definitions list.
    Raises:
        AssertionError: If expanded labels differ from ndindex reference.
    Side Effects:
        None.
    """
    axes = _axes_for_variant_expansion()
    axis_lengths = tuple(axis.length() for axis in axes)
    source_index = 0
    source_values = axes[source_index].values_enum
    assert source_values is not None
    definition = next(
        definition
        for definition in all_defs()
        if definition.indicator_id.value == "ma.sma"
    )

    expected = _expand_axis_values_ndindex_reference(
        axis_values=source_values,
        axis_lengths=axis_lengths,
        axis_index=source_index,
    )
    actual = numba_engine._variant_source_labels(definition=definition, axes=axes)

    assert actual == expected


def test_variant_axis_expansion_functions_use_vectorized_repeat_tile_path() -> None:
    """
    Verify axis expansion implementation uses repeat/tile helper and avoids ndindex loops.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Source inspection is stable for module-level functions in tests.
    Raises:
        AssertionError: If helper/functions violate vectorized-path expectations.
    Side Effects:
        None.
    """
    helper_source = inspect.getsource(numba_engine._expand_axis_values_repeat_tile)
    assert "np.repeat" in helper_source
    assert "np.tile" in helper_source

    for function in (
        numba_engine._variant_int_values,
        numba_engine._variant_float_values,
        numba_engine._variant_window_values,
        numba_engine._variant_source_labels,
    ):
        assert "np.ndindex" not in inspect.getsource(function)


def test_group_variant_indexes_by_source_preserves_first_seen_source_order() -> None:
    """
    Verify source grouping keeps deterministic first-seen source order and index mapping.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Grouping helper is used for grouped source compute scatter path.
    Raises:
        AssertionError: If source order or grouped indexes drift.
    Side Effects:
        None.
    """
    grouped = numba_engine._group_variant_indexes_by_source(
        variant_source_labels=("close", "hlc3", "close", "open", "hlc3", "close")
    )

    assert tuple(source for source, _ in grouped) == ("close", "hlc3", "open")
    assert tuple(grouped[0][1].tolist()) == (0, 2, 5)
    assert tuple(grouped[1][1].tolist()) == (1, 4)
    assert tuple(grouped[2][1].tolist()) == (3,)


def test_grouped_source_paths_match_legacy_full_matrix_semantics(tmp_path: Path) -> None:
    """
    Verify grouped-source pipeline matches legacy full-source-matrix values and ordering.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Legacy semantics are represented by explicit `_build_variant_source_matrix` reference path.
    Raises:
        AssertionError: If grouped compute diverges from legacy values/layout/order.
    Side Effects:
        None.
    """
    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")
    candles = _candles(t_size=192)

    for grid in _grouped_source_grids():
        request = ComputeRequest(candles=candles, grid=grid, max_variants_guard=100_000)
        tensor = engine.compute(request)
        expected = _legacy_full_source_matrix_reference(
            grid=grid,
            candles=candles,
            axes=tensor.axes,
        )

        assert tensor.layout is Layout.VARIANT_MAJOR
        assert tensor.values.dtype == np.float32
        assert tensor.values.flags.c_contiguous
        assert tensor.values.shape == expected.shape
        np.testing.assert_allclose(
            tensor.values,
            expected,
            rtol=2e-5,
            atol=2e-5,
            equal_nan=True,
            err_msg=f"indicator_id={grid.indicator_id.value}",
        )


def test_grouped_source_paths_do_not_use_full_variant_source_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify grouped source indicators bypass global `_build_variant_source_matrix` allocation path.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Source-parameterized volatility/momentum/trend/structure paths use grouped pipeline.
    Raises:
        AssertionError: If grouped path falls back to global full source matrix builder.
    Side Effects:
        Temporarily patches `_build_variant_source_matrix` helper.
    """

    def _forbidden_full_source_matrix(**_: object) -> np.ndarray:
        """
        Fail test when grouped path unexpectedly requests global full source matrix.

        Args:
            **_: Ignored keyword args from compute engine call site.
        Returns:
            np.ndarray: Never returns.
        Assumptions:
            Grouped source pipeline must not use this helper for covered indicators.
        Raises:
            AssertionError: Always.
        Side Effects:
            None.
        """
        raise AssertionError("unexpected call to _build_variant_source_matrix")

    monkeypatch.setattr(
        numba_engine,
        "_build_variant_source_matrix",
        _forbidden_full_source_matrix,
    )

    engine = _compute_engine(cache_dir=tmp_path / "numba-cache")
    candles = _candles(t_size=160)

    for grid in _grouped_source_grids():
        tensor = engine.compute(
            ComputeRequest(
                candles=candles,
                grid=grid,
                max_variants_guard=100_000,
            )
        )
        assert tensor.layout is Layout.VARIANT_MAJOR
        assert tensor.values.dtype == np.float32
        assert tensor.values.flags.c_contiguous


def test_precision_policy_dispatch_maps_tier_a_b_c_and_fallback() -> None:
    """
    Verify deterministic precision policy dispatch for Tier A/B/C and unknown fallback ids.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Indicator ids are normalized to lowercase in policy helper.
    Raises:
        AssertionError: If policy mapping returns unexpected precision mode.
    Side Effects:
        None.
    """
    assert (
        numba_engine._precision_mode_for_indicator(indicator_id="ma.sma")
        == PRECISION_MODE_FLOAT32
    )
    assert (
        numba_engine._precision_mode_for_indicator(indicator_id="momentum.macd")
        == PRECISION_MODE_MIXED
    )
    assert (
        numba_engine._precision_mode_for_indicator(indicator_id="volatility.stddev")
        == PRECISION_MODE_FLOAT64
    )
    assert (
        numba_engine._precision_mode_for_indicator(indicator_id="custom.unknown")
        == PRECISION_MODE_FLOAT64
    )


def test_precision_tier_label_matches_policy_constants() -> None:
    """
    Verify precision mode to tier label mapping is explicit and deterministic.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mapping function covers all shared precision constants.
    Raises:
        AssertionError: If tier labels drift from migration-plan semantics.
    Side Effects:
        None.
    """
    assert (
        numba_engine._precision_tier_label(precision=PRECISION_MODE_FLOAT32)
        == "Tier A"
    )
    assert (
        numba_engine._precision_tier_label(precision=PRECISION_MODE_MIXED)
        == "Tier B"
    )
    assert (
        numba_engine._precision_tier_label(precision=PRECISION_MODE_FLOAT64)
        == "Tier C"
    )
