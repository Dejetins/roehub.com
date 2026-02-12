"""
CPU/Numba implementation of indicators compute application port.

Docs: docs/architecture/indicators/indicators-compute-engine-core.md
Related: trading.contexts.indicators.application.ports.compute.indicator_compute,
  trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
  trading.contexts.indicators.adapters.outbound.compute_numba.kernels.ma,
  trading.contexts.indicators.adapters.outbound.compute_numba.warmup
"""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    WORKSPACE_FACTOR_DEFAULT,
    WORKSPACE_FIXED_BYTES_DEFAULT,
    check_total_budget_or_raise,
    compute_ma_grid_f32,
    compute_momentum_grid_f32,
    compute_trend_grid_f32,
    compute_volatility_grid_f32,
    compute_volume_grid_f32,
    estimate_tensor_bytes,
    estimate_total_bytes,
    is_supported_ma_indicator,
    is_supported_momentum_indicator,
    is_supported_trend_indicator,
    is_supported_volatility_indicator,
    is_supported_volume_indicator,
    write_series_grid_time_major,
    write_series_grid_variant_major,
)
from trading.contexts.indicators.adapters.outbound.compute_numba.warmup import (
    ComputeNumbaWarmupRunner,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    TensorMeta,
)
from trading.contexts.indicators.application.dto.registry_view import MergedIndicatorView
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.application.services import GridBuilder, MaterializedAxis
from trading.contexts.indicators.domain.entities import (
    AxisDef,
    IndicatorDef,
    IndicatorId,
    InputSeries,
    Layout,
)
from trading.contexts.indicators.domain.errors import (
    GridValidationError,
    MissingRequiredSeries,
    UnknownIndicatorError,
)
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.platform.config import IndicatorsComputeNumbaConfig


class NumbaIndicatorCompute(IndicatorCompute):
    """
    Indicator compute engine backed by CPU+Numba kernels.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.application.dto.indicator_tensor,
      trading.contexts.indicators.application.services.grid_builder,
      trading.contexts.indicators.adapters.outbound.compute_numba.kernels.ma
    """

    def __init__(
        self,
        *,
        defs: tuple[IndicatorDef, ...],
        config: IndicatorsComputeNumbaConfig,
        workspace_factor: float = WORKSPACE_FACTOR_DEFAULT,
        workspace_fixed_bytes: int = WORKSPACE_FIXED_BYTES_DEFAULT,
        warmup_runner: ComputeNumbaWarmupRunner | None = None,
    ) -> None:
        """
        Build deterministic definition index and runtime dependencies.

        Docs: docs/architecture/indicators/indicators-compute-engine-core.md
        Related:
          src/trading/contexts/indicators/domain/definitions/__init__.py,
          src/trading/contexts/indicators/application/services/grid_builder.py

        Args:
            defs: Hard indicator definitions.
            config: Runtime config for Numba threads/cache and memory budget.
            workspace_factor: Internal proportional workspace reserve factor.
            workspace_fixed_bytes: Internal fixed workspace reserve.
            warmup_runner: Optional warmup runner override for testing/composition.
        Returns:
            None.
        Assumptions:
            Definition tuple is immutable after adapter construction.
        Raises:
            ValueError: If duplicate indicator ids or invalid workspace settings are provided.
        Side Effects:
            None.
        """
        if workspace_factor < 0:
            raise ValueError(f"workspace_factor must be >= 0, got {workspace_factor}")
        if workspace_fixed_bytes < 0:
            raise ValueError(
                f"workspace_fixed_bytes must be >= 0, got {workspace_fixed_bytes}"
            )

        defs_by_id: dict[str, IndicatorDef] = {}
        ordered_defs: list[IndicatorDef] = []
        for definition in sorted(defs, key=lambda item: item.indicator_id.value):
            key = definition.indicator_id.value
            if key in defs_by_id:
                raise ValueError(f"duplicate indicator_id in defs: {key}")
            defs_by_id[key] = definition
            ordered_defs.append(definition)

        self._defs_by_id: Mapping[str, IndicatorDef] = MappingProxyType(defs_by_id)
        self._defs: tuple[IndicatorDef, ...] = tuple(ordered_defs)
        self._grid_builder = GridBuilder(
            registry=_DefinitionsRegistry(defs=self._defs, defs_by_id=self._defs_by_id)
        )
        self._config = config
        self._workspace_factor = workspace_factor
        self._workspace_fixed_bytes = workspace_fixed_bytes
        self._warmup_runner = warmup_runner or ComputeNumbaWarmupRunner(config=config)

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Estimate axis cardinality and variant count without kernel execution.

        Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
        Related:
          src/trading/contexts/indicators/application/services/grid_builder.py,
          src/trading/contexts/indicators/application/dto/estimate_result.py

        Args:
            grid: Grid specification for one indicator.
            max_variants_guard: Upper bound for allowed variant count.
        Returns:
            EstimateResult: Deterministic estimate snapshot.
        Assumptions:
            Grid materialization is deterministic for each axis.
        Raises:
            UnknownIndicatorError: If indicator id is not defined.
            GridValidationError: If axis materialization is invalid or exceeds guard.
        Side Effects:
            None.
        """
        materialized = self._grid_builder.materialize_indicator(grid=grid)
        axes = _axis_defs_from_materialized_axes(axes=materialized.axes)
        variants = materialized.variants

        if max_variants_guard <= 0:
            raise GridValidationError("max_variants_guard must be > 0")
        if variants > max_variants_guard:
            raise GridValidationError(
                "variants exceed guard: "
                f"variants={variants}, max_variants_guard={max_variants_guard}"
            )

        return EstimateResult(
            indicator_id=grid.indicator_id,
            axes=axes,
            variants=variants,
            max_variants_guard=max_variants_guard,
        )

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Compute indicator tensor for one request.

        Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
        Related:
          src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
          src/trading/contexts/indicators/application/dto/indicator_tensor.py,
          src/trading/contexts/indicators/application/services/grid_builder.py

        Args:
            req: Compute request with candles and grid specification.
        Returns:
            IndicatorTensor: Float32 tensor in selected layout.
        Assumptions:
            Guards are checked before large output allocations.
        Raises:
            UnknownIndicatorError: If indicator id is not defined.
            GridValidationError: If grid/layout/guard invariants are violated.
            MissingRequiredSeries: If required source series cannot be resolved.
            ComputeBudgetExceeded: If estimated total bytes exceed budget.
        Side Effects:
            Allocates output tensor and intermediate source matrices.
        """
        started = time.perf_counter()
        definition = self._get_definition(indicator_id=req.grid.indicator_id)

        materialized = self._grid_builder.materialize_indicator(grid=req.grid)
        axes = _axis_defs_from_materialized_axes(axes=materialized.axes)
        variants = materialized.variants
        if variants > req.max_variants_guard:
            raise GridValidationError(
                "variants exceed guard: "
                f"variants={variants}, max_variants_guard={req.max_variants_guard}"
            )

        layout = req.grid.layout_preference or Layout.TIME_MAJOR
        if layout not in (Layout.TIME_MAJOR, Layout.VARIANT_MAJOR):
            raise GridValidationError(f"invalid layout: {layout!r}")

        t_size = int(req.candles.ts_open.shape[0])
        if t_size <= 0:
            raise GridValidationError("compute requires candles with t > 0")

        bytes_out = estimate_tensor_bytes(t=t_size, variants=variants)
        bytes_total_est = estimate_total_bytes(
            bytes_out=bytes_out,
            workspace_factor=self._workspace_factor,
            workspace_fixed_bytes=self._workspace_fixed_bytes,
        )
        check_total_budget_or_raise(
            t=t_size,
            variants=variants,
            bytes_out=bytes_out,
            bytes_total_est=bytes_total_est,
            max_compute_bytes_total=self._config.max_compute_bytes_total,
        )

        series_map = _build_series_map(candles=req.candles)
        if is_supported_ma_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_ma_variant_source_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
            )
        elif is_supported_volatility_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_volatility_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
            )
        elif is_supported_momentum_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_momentum_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
            )
        elif is_supported_trend_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_trend_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
            )
        elif is_supported_volume_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_volume_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
            )
        else:
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=series_map,
            )
            variant_series_matrix = _build_variant_source_matrix(
                variant_source_labels=variant_source_labels,
                available_series=series_map,
                t_size=t_size,
            )

        if layout is Layout.TIME_MAJOR:
            values = np.empty((t_size, variants), dtype=np.float32, order="C")
            write_series_grid_time_major(values, variant_series_matrix)
        else:
            values = np.empty((variants, t_size), dtype=np.float32, order="C")
            write_series_grid_variant_major(values, variant_series_matrix)

        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        meta = TensorMeta(
            t=t_size,
            variants=variants,
            nan_policy="propagate",
            compute_ms=elapsed_ms,
        )
        return IndicatorTensor(
            indicator_id=req.grid.indicator_id,
            layout=layout,
            axes=axes,
            values=values,
            meta=meta,
        )

    def warmup(self) -> None:
        """
        Warm up Numba runtime and core kernels.

        Docs: docs/architecture/indicators/indicators-compute-engine-core.md
        Related:
          src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py,
          src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup runner is idempotent.
        Raises:
            ValueError: If runtime config is invalid or cache dir is not writable.
        Side Effects:
            Triggers JIT compilation and logs warmup summary.
        """
        self._warmup_runner.warmup()

    def _get_definition(self, *, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one indicator definition by id.

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/domain/definitions/__init__.py,
          src/trading/contexts/indicators/domain/errors/unknown_indicator_error.py

        Args:
            indicator_id: Indicator identifier.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            `_defs_by_id` map is immutable and deterministic.
        Raises:
            UnknownIndicatorError: If indicator id is not found.
        Side Effects:
            None.
        """
        definition = self._defs_by_id.get(indicator_id.value)
        if definition is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return definition


class _DefinitionsRegistry(IndicatorRegistry):
    """
    Minimal immutable registry adapter used by `GridBuilder` inside compute engine.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/application/ports/registry/indicator_registry.py
    """

    def __init__(
        self,
        *,
        defs: tuple[IndicatorDef, ...],
        defs_by_id: Mapping[str, IndicatorDef],
    ) -> None:
        """
        Store immutable definitions snapshot.

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/domain/definitions/__init__.py,
          src/trading/contexts/indicators/application/services/grid_builder.py

        Args:
            defs: Stable ordered indicator definitions.
            defs_by_id: Immutable indicator lookup mapping.
        Returns:
            None.
        Assumptions:
            Input collections are deterministic and immutable for runtime.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._defs = defs
        self._defs_by_id = defs_by_id

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return hard definitions tuple.

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/application/ports/registry/indicator_registry.py,
          src/trading/contexts/indicators/domain/definitions/__init__.py

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Stable ordered definitions.
        Assumptions:
            Definitions are immutable.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._defs

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one hard definition by id.

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/application/ports/registry/indicator_registry.py,
          src/trading/contexts/indicators/domain/errors/unknown_indicator_error.py

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            Identifier normalization is handled by `IndicatorId` value-object.
        Raises:
            UnknownIndicatorError: If id is absent.
        Side Effects:
            None.
        """
        definition = self._defs_by_id.get(indicator_id.value)
        if definition is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return definition

    def list_merged(self) -> tuple[MergedIndicatorView, ...]:
        """
        Return empty merged-view tuple (not used by compute engine internals).

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/application/ports/registry/indicator_registry.py,
          src/trading/contexts/indicators/application/services/grid_builder.py

        Args:
            None.
        Returns:
            tuple[MergedIndicatorView, ...]: Empty tuple.
        Assumptions:
            Compute engine needs only `get_def`/`list_defs` for grid materialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ()

    def get_merged(self, indicator_id: IndicatorId) -> MergedIndicatorView:
        """
        Reject merged-view lookups for internal compute registry.

        Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
        Related:
          src/trading/contexts/indicators/application/ports/registry/indicator_registry.py,
          src/trading/contexts/indicators/domain/errors/unknown_indicator_error.py

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            MergedIndicatorView: Never returns.
        Assumptions:
            Method exists only for protocol completeness.
        Raises:
            UnknownIndicatorError: Always for this minimal registry.
        Side Effects:
            None.
        """
        raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")


def _axis_defs_from_materialized_axes(*, axes: tuple[MaterializedAxis, ...]) -> tuple[AxisDef, ...]:
    """
    Convert `GridBuilder` materialized axes into domain `AxisDef` tuple.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py

    Args:
        axes: Materialized axes preserving request order semantics.
    Returns:
        tuple[AxisDef, ...]: Domain axis objects in deterministic order.
    Assumptions:
        Materialized axes are non-empty and validated by `GridBuilder`.
    Raises:
        GridValidationError: If values family cannot be represented as `AxisDef`.
    Side Effects:
        None.
    """
    return tuple(_axis_def_from_materialized_axis(axis=axis) for axis in axes)


def _axis_def_from_materialized_axis(*, axis: MaterializedAxis) -> AxisDef:
    """
    Convert one materialized axis into `AxisDef` while preserving value order.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py

    Args:
        axis: Materialized axis from grid builder.
    Returns:
        AxisDef: Domain axis object with one active value family.
    Assumptions:
        `axis.values` are homogeneous by grid validation contracts.
    Raises:
        GridValidationError: If values contain unsupported or mixed scalar families.
    Side Effects:
        None.
    """
    values = axis.values
    if len(values) == 0:
        raise GridValidationError(f"axis '{axis.name}' materialized to empty values")

    if all(isinstance(value, str) for value in values):
        return AxisDef(name=axis.name, values_enum=tuple(str(value) for value in values))

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return AxisDef(name=axis.name, values_int=tuple(int(value) for value in values))

    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return AxisDef(name=axis.name, values_float=tuple(float(value) for value in values))

    raise GridValidationError(
        f"axis '{axis.name}' contains unsupported values family: {values!r}"
    )


def _build_series_map(*, candles: CandleArrays) -> Mapping[str, np.ndarray]:
    """
    Build deterministic source-series map including derived OHLC aggregates.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/dto/candle_arrays.py,
      src/trading/contexts/indicators/domain/entities/input_series.py

    Args:
        candles: Dense candle arrays payload.
    Returns:
        Mapping[str, np.ndarray]: Source name to contiguous float32 array.
    Assumptions:
        Candle arrays are aligned and share the same length.
    Raises:
        None.
    Side Effects:
        Allocates derived arrays (`hl2`, `hlc3`, `ohlc4`).
    """
    open_series = np.ascontiguousarray(candles.open, dtype=np.float32)
    high_series = np.ascontiguousarray(candles.high, dtype=np.float32)
    low_series = np.ascontiguousarray(candles.low, dtype=np.float32)
    close_series = np.ascontiguousarray(candles.close, dtype=np.float32)
    volume_series = np.ascontiguousarray(candles.volume, dtype=np.float32)

    hl2_series = np.ascontiguousarray((high_series + low_series) / np.float32(2.0))
    hlc3_series = np.ascontiguousarray(
        (high_series + low_series + close_series) / np.float32(3.0)
    )
    ohlc4_series = np.ascontiguousarray(
        (open_series + high_series + low_series + close_series) / np.float32(4.0)
    )

    return {
        InputSeries.CLOSE.value: close_series,
        InputSeries.HIGH.value: high_series,
        InputSeries.HL2.value: hl2_series,
        InputSeries.HLC3.value: hlc3_series,
        InputSeries.LOW.value: low_series,
        InputSeries.OHLC4.value: ohlc4_series,
        InputSeries.OPEN.value: open_series,
        InputSeries.VOLUME.value: volume_series,
    }


def _compute_ma_variant_source_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Compute variant-major MA matrix `(V, T)` via Numba MA kernels.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py,
      src/trading/contexts/indicators/application/dto/indicator_tensor.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)`.
    Assumptions:
        MA indicators always expose integer `window` axis in hard definitions.
    Raises:
        GridValidationError: If required `window` axis is missing or malformed.
        MissingRequiredSeries: If required source/volume series is missing.
        ValueError: If MA kernel inputs are invalid.
    Side Effects:
        Allocates per-source MA grid matrices and one variant-major output matrix.
    """
    window_values = _window_values_from_axes(axes=axes)
    windows_i64 = np.ascontiguousarray(np.asarray(window_values, dtype=np.int64))

    variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
    variant_window_values = _variant_window_values(axes=axes)
    _validate_required_series_available(
        variant_source_labels=variant_source_labels,
        available_series=available_series,
    )

    volume_series = available_series.get(InputSeries.VOLUME.value)
    if volume_series is None:
        raise MissingRequiredSeries(
            "MissingRequiredSeries: missing=('volume',); "
            f"required={tuple(sorted(set(variant_source_labels)))}"
        )

    unique_sources_in_order = tuple(dict.fromkeys(variant_source_labels))
    per_source_grid: dict[str, np.ndarray] = {}
    for source_name in unique_sources_in_order:
        source_series = available_series.get(source_name)
        if source_series is None:
            raise MissingRequiredSeries(f"MissingRequiredSeries: missing={source_name!r}")
        try:
            per_source_grid[source_name] = compute_ma_grid_f32(
                indicator_id=definition.indicator_id.value,
                source=source_series,
                windows=windows_i64,
                volume=volume_series,
            )
        except ValueError as error:
            raise GridValidationError(str(error)) from error

    window_to_index = {window: idx for idx, window in enumerate(window_values)}
    variants = len(variant_source_labels)
    t_size = int(next(iter(available_series.values())).shape[0])
    out = np.empty((variants, t_size), dtype=np.float32, order="C")

    for variant_index in range(variants):
        source_name = variant_source_labels[variant_index]
        window_value = variant_window_values[variant_index]
        window_index = window_to_index[window_value]
        out[variant_index, :] = per_source_grid[source_name][:, window_index]

    return out


def _compute_volatility_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
) -> np.ndarray:
    """
    Compute variant-major matrix `(V, T)` for volatility-family indicators.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py,
      src/trading/contexts/indicators/domain/definitions/volatility.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
        t_size: Time dimension length.
    Returns:
        np.ndarray: Float32 C-contiguous variant-major matrix `(V, T)`.
    Assumptions:
        Volatility kernels return one primary output line per indicator id in v1.
    Raises:
        GridValidationError: If required axes are missing or malformed.
        MissingRequiredSeries: If required OHLC/source series are unavailable.
    Side Effects:
        Allocates source/parameter vectors and one output matrix.
    """
    indicator_id = definition.indicator_id.value

    try:
        if indicator_id == "volatility.tr":
            return compute_volatility_grid_f32(
                indicator_id=indicator_id,
                high=_require_series(
                    available_series=available_series,
                    name=InputSeries.HIGH.value,
                ),
                low=_require_series(
                    available_series=available_series,
                    name=InputSeries.LOW.value,
                ),
                close=_require_series(
                    available_series=available_series,
                    name=InputSeries.CLOSE.value,
                ),
            )

        if indicator_id == "volatility.atr":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_volatility_grid_f32(
                indicator_id=indicator_id,
                high=_require_series(
                    available_series=available_series,
                    name=InputSeries.HIGH.value,
                ),
                low=_require_series(
                    available_series=available_series,
                    name=InputSeries.LOW.value,
                ),
                close=_require_series(
                    available_series=available_series,
                    name=InputSeries.CLOSE.value,
                ),
                windows=windows_i64,
            )

        variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
        _validate_required_series_available(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
        )
        source_variants = _build_variant_source_matrix(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
            t_size=t_size,
        )

        if indicator_id in {"volatility.stddev", "volatility.variance"}:
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_volatility_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
            )

        if indicator_id == "volatility.hv":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            annualizations_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="annualization"),
                    dtype=np.int64,
                )
            )
            return compute_volatility_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
                annualizations=annualizations_i64,
            )

        if indicator_id in {
            "volatility.bbands",
            "volatility.bbands_bandwidth",
            "volatility.bbands_percent_b",
        }:
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            mults_f64 = np.ascontiguousarray(
                np.asarray(_variant_float_values(axes=axes, axis_name="mult"), dtype=np.float64)
            )
            return compute_volatility_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
                mults=mults_f64,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported volatility indicator_id: {indicator_id}")


def _compute_momentum_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
) -> np.ndarray:
    """
    Compute variant-major matrix `(V, T)` for momentum-family indicators.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py,
      src/trading/contexts/indicators/domain/definitions/momentum.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
        t_size: Time dimension length.
    Returns:
        np.ndarray: Float32 C-contiguous variant-major matrix `(V, T)`.
    Assumptions:
        Momentum kernels return one primary output line per indicator id in v1.
    Raises:
        GridValidationError: If required axes are missing or malformed.
        MissingRequiredSeries: If required OHLC/source series are unavailable.
    Side Effects:
        Allocates source/parameter vectors and one output matrix.
    """
    indicator_id = definition.indicator_id.value

    try:
        if indicator_id in {"momentum.rsi", "momentum.roc"}:
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
            )
            source_variants = _build_variant_source_matrix(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
                t_size=t_size,
            )
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
            )

        if indicator_id in {"momentum.cci", "momentum.williams_r"}:
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                high=_require_series(
                    available_series=available_series,
                    name=InputSeries.HIGH.value,
                ),
                low=_require_series(
                    available_series=available_series,
                    name=InputSeries.LOW.value,
                ),
                close=_require_series(
                    available_series=available_series,
                    name=InputSeries.CLOSE.value,
                ),
                windows=windows_i64,
            )

        if indicator_id == "momentum.fisher":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                high=_require_series(
                    available_series=available_series,
                    name=InputSeries.HIGH.value,
                ),
                low=_require_series(
                    available_series=available_series,
                    name=InputSeries.LOW.value,
                ),
                windows=windows_i64,
            )

        if indicator_id == "momentum.stoch":
            k_windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="k_window"), dtype=np.int64)
            )
            smoothings_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="smoothing"), dtype=np.int64)
            )
            d_windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="d_window"), dtype=np.int64)
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                high=_require_series(
                    available_series=available_series,
                    name=InputSeries.HIGH.value,
                ),
                low=_require_series(
                    available_series=available_series,
                    name=InputSeries.LOW.value,
                ),
                close=_require_series(
                    available_series=available_series,
                    name=InputSeries.CLOSE.value,
                ),
                k_windows=k_windows_i64,
                smoothings=smoothings_i64,
                d_windows=d_windows_i64,
            )

        variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
        _validate_required_series_available(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
        )
        source_variants = _build_variant_source_matrix(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
            t_size=t_size,
        )

        if indicator_id == "momentum.stoch_rsi":
            rsi_windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="rsi_window"), dtype=np.int64)
            )
            k_windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="k_window"), dtype=np.int64)
            )
            smoothings_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="smoothing"), dtype=np.int64)
            )
            d_windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="d_window"), dtype=np.int64)
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                rsi_windows=rsi_windows_i64,
                k_windows=k_windows_i64,
                smoothings=smoothings_i64,
                d_windows=d_windows_i64,
            )

        if indicator_id == "momentum.trix":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            signal_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="signal_window"),
                    dtype=np.int64,
                )
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
                signal_windows=signal_windows_i64,
            )

        if indicator_id in {"momentum.ppo", "momentum.macd"}:
            fast_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="fast_window"),
                    dtype=np.int64,
                )
            )
            slow_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="slow_window"),
                    dtype=np.int64,
                )
            )
            signal_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="signal_window"),
                    dtype=np.int64,
                )
            )
            return compute_momentum_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                fast_windows=fast_windows_i64,
                slow_windows=slow_windows_i64,
                signal_windows=signal_windows_i64,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported momentum indicator_id: {indicator_id}")


def _compute_trend_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
) -> np.ndarray:
    """
    Compute variant-major matrix `(V, T)` for trend-family indicators.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py,
      src/trading/contexts/indicators/domain/definitions/trend.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
        t_size: Time dimension length.
    Returns:
        np.ndarray: Float32 C-contiguous variant-major matrix `(V, T)`.
    Assumptions:
        Trend kernels return one primary output line per indicator id in v1.
    Raises:
        GridValidationError: If required axes are missing or malformed.
        MissingRequiredSeries: If required OHLC/source series are unavailable.
    Side Effects:
        Allocates source/parameter vectors and one output matrix.
    """
    indicator_id = definition.indicator_id.value

    try:
        if indicator_id == "trend.linreg_slope":
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
            )
            source_variants = _build_variant_source_matrix(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
                t_size=t_size,
            )
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                source_variants=source_variants,
                windows=windows_i64,
            )

        high = _require_series(available_series=available_series, name=InputSeries.HIGH.value)
        low = _require_series(available_series=available_series, name=InputSeries.LOW.value)

        if indicator_id == "trend.psar":
            accel_starts_f64 = np.ascontiguousarray(
                np.asarray(
                    _variant_float_values(axes=axes, axis_name="accel_start"),
                    dtype=np.float64,
                )
            )
            accel_steps_f64 = np.ascontiguousarray(
                np.asarray(
                    _variant_float_values(axes=axes, axis_name="accel_step"),
                    dtype=np.float64,
                )
            )
            accel_maxes_f64 = np.ascontiguousarray(
                np.asarray(
                    _variant_float_values(axes=axes, axis_name="accel_max"),
                    dtype=np.float64,
                )
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                accel_starts=accel_starts_f64,
                accel_steps=accel_steps_f64,
                accel_maxes=accel_maxes_f64,
            )

        close = _require_series(
            available_series=available_series,
            name=InputSeries.CLOSE.value,
        )

        if indicator_id == "trend.adx":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            smoothings_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="smoothing"), dtype=np.int64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
                smoothings=smoothings_i64,
            )

        if indicator_id == "trend.aroon":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
            )

        if indicator_id == "trend.chandelier_exit":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            mults_f64 = np.ascontiguousarray(
                np.asarray(_variant_float_values(axes=axes, axis_name="mult"), dtype=np.float64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
                mults=mults_f64,
            )

        if indicator_id == "trend.donchian":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
            )

        if indicator_id == "trend.ichimoku":
            conversion_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="conversion_window"),
                    dtype=np.int64,
                )
            )
            base_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="base_window"),
                    dtype=np.int64,
                )
            )
            span_b_windows_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="span_b_window"),
                    dtype=np.int64,
                )
            )
            displacements_i64 = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="displacement"),
                    dtype=np.int64,
                )
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                conversion_windows=conversion_windows_i64,
                base_windows=base_windows_i64,
                span_b_windows=span_b_windows_i64,
                displacements=displacements_i64,
            )

        if indicator_id in {"trend.keltner", "trend.supertrend"}:
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            mults_f64 = np.ascontiguousarray(
                np.asarray(_variant_float_values(axes=axes, axis_name="mult"), dtype=np.float64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
                mults=mults_f64,
            )

        if indicator_id == "trend.vortex":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_trend_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                windows=windows_i64,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported trend indicator_id: {indicator_id}")


def _compute_volume_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Compute variant-major matrix `(V, T)` for volume-family indicators.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py,
      src/trading/contexts/indicators/domain/definitions/volume.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
    Returns:
        np.ndarray: Float32 C-contiguous variant-major matrix `(V, T)`.
    Assumptions:
        Volume kernels return one primary output line per indicator id in v1.
    Raises:
        GridValidationError: If required axes are missing or malformed.
        MissingRequiredSeries: If required OHLCV series are unavailable.
    Side Effects:
        Allocates source/parameter vectors and one output matrix.
    """
    indicator_id = definition.indicator_id.value

    try:
        volume = _require_series(
            available_series=available_series,
            name=InputSeries.VOLUME.value,
        )

        if indicator_id == "volume.obv":
            close = _require_series(
                available_series=available_series,
                name=InputSeries.CLOSE.value,
            )
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                close=close,
                volume=volume,
            )

        if indicator_id == "volume.volume_sma":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                volume=volume,
                windows=windows_i64,
            )

        high = _require_series(
            available_series=available_series,
            name=InputSeries.HIGH.value,
        )
        low = _require_series(
            available_series=available_series,
            name=InputSeries.LOW.value,
        )
        close = _require_series(
            available_series=available_series,
            name=InputSeries.CLOSE.value,
        )

        if indicator_id == "volume.ad_line":
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )

        windows_i64 = np.ascontiguousarray(
            np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
        )

        if indicator_id in {"volume.cmf", "volume.mfi", "volume.vwap"}:
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                volume=volume,
                windows=windows_i64,
            )

        if indicator_id == "volume.vwap_deviation":
            mults_f64 = np.ascontiguousarray(
                np.asarray(_variant_float_values(axes=axes, axis_name="mult"), dtype=np.float64)
            )
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                high=high,
                low=low,
                close=close,
                volume=volume,
                windows=windows_i64,
                mults=mults_f64,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported volume indicator_id: {indicator_id}")


def _variant_int_values(*, axes: tuple[AxisDef, ...], axis_name: str) -> tuple[int, ...]:
    """
    Resolve one integer axis value per variant in deterministic variant order.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py

    Args:
        axes: Materialized axes in definition order.
        axis_name: Target integer axis name.
    Returns:
        tuple[int, ...]: One axis value per variant.
    Assumptions:
        Variant order follows cartesian product of axes in definition order.
    Raises:
        GridValidationError: If axis is missing or not integer-valued.
    Side Effects:
        None.
    """
    axis_index = _axis_index(axes=axes, axis_name=axis_name)
    axis = axes[axis_index]
    if axis.values_int is None:
        raise GridValidationError(f"axis '{axis_name}' must have values_int")
    axis_lengths = tuple(axis_item.length() for axis_item in axes)
    items: list[int] = []
    for coordinate in np.ndindex(axis_lengths):
        items.append(axis.values_int[coordinate[axis_index]])
    return tuple(items)


def _variant_float_values(*, axes: tuple[AxisDef, ...], axis_name: str) -> tuple[float, ...]:
    """
    Resolve one float axis value per variant in deterministic variant order.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py

    Args:
        axes: Materialized axes in definition order.
        axis_name: Target float axis name.
    Returns:
        tuple[float, ...]: One axis value per variant.
    Assumptions:
        Variant order follows cartesian product of axes in definition order.
    Raises:
        GridValidationError: If axis is missing or not float-valued.
    Side Effects:
        None.
    """
    axis_index = _axis_index(axes=axes, axis_name=axis_name)
    axis = axes[axis_index]
    if axis.values_float is None:
        raise GridValidationError(f"axis '{axis_name}' must have values_float")
    axis_lengths = tuple(axis_item.length() for axis_item in axes)
    items: list[float] = []
    for coordinate in np.ndindex(axis_lengths):
        items.append(axis.values_float[coordinate[axis_index]])
    return tuple(items)


def _axis_index(*, axes: tuple[AxisDef, ...], axis_name: str) -> int:
    """
    Resolve axis index by name in deterministic definition order.

    Args:
        axes: Materialized axes in definition order.
        axis_name: Target axis name.
    Returns:
        int: Axis index.
    Assumptions:
        Axis names in tuple are unique by `AxisDef` invariants.
    Raises:
        GridValidationError: If axis is absent.
    Side Effects:
        None.
    """
    axis_names = [axis.name for axis in axes]
    if axis_name not in axis_names:
        raise GridValidationError(f"axis '{axis_name}' is required")
    return axis_names.index(axis_name)


def _require_series(*, available_series: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """
    Resolve one required series from available map or raise deterministic domain error.

    Args:
        available_series: Mapping of available source arrays.
        name: Required source name.
    Returns:
        np.ndarray: Resolved source series.
    Assumptions:
        Series map keys are normalized source names.
    Raises:
        MissingRequiredSeries: If required series is not available.
    Side Effects:
        None.
    """
    series = available_series.get(name)
    if series is None:
        raise MissingRequiredSeries(f"MissingRequiredSeries: missing={name!r}")
    return series


def _window_values_from_axes(*, axes: tuple[AxisDef, ...]) -> tuple[int, ...]:
    """
    Extract window axis integer values from materialized axes.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/axis_def.py,
      src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        axes: Materialized axes in definition order.
    Returns:
        tuple[int, ...]: Window values preserving request materialization order.
    Assumptions:
        MA indicators define `window` axis in hard contracts.
    Raises:
        GridValidationError: If `window` axis is missing or not integer-valued.
    Side Effects:
        None.
    """
    for axis in axes:
        if axis.name != "window":
            continue
        if axis.values_int is None:
            raise GridValidationError("window axis must be integer-valued")
        return axis.values_int
    raise GridValidationError("window axis is required for MA compute")


def _variant_window_values(*, axes: tuple[AxisDef, ...]) -> tuple[int, ...]:
    """
    Resolve one window value for each variant in deterministic variant order.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/axis_def.py,
      src/trading/contexts/indicators/application/services/grid_builder.py

    Args:
        axes: Materialized axes in definition order.
    Returns:
        tuple[int, ...]: Window value per variant index.
    Assumptions:
        Variant order follows cartesian product of axis values in axis order.
    Raises:
        GridValidationError: If window axis metadata is missing.
    Side Effects:
        None.
    """
    axis_names = [axis.name for axis in axes]
    if "window" not in axis_names:
        raise GridValidationError("window axis is required for MA compute")

    window_axis_index = axis_names.index("window")
    window_axis = axes[window_axis_index]
    if window_axis.values_int is None:
        raise GridValidationError("window axis must have values_int")

    axis_lengths = tuple(axis.length() for axis in axes)
    labels: list[int] = []
    for coordinate in np.ndindex(axis_lengths):
        labels.append(window_axis.values_int[coordinate[window_axis_index]])
    return tuple(labels)


def _variant_source_labels(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
) -> tuple[str, ...]:
    """
    Resolve one source label for each variant in deterministic variant order.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/axis_def.py,
      src/trading/contexts/indicators/domain/entities/indicator_def.py

    Args:
        definition: Hard indicator definition.
        axes: Materialized axes in definition order.
    Returns:
        tuple[str, ...]: Source name per variant.
    Assumptions:
        Variant order follows cartesian product of axes in definition order.
    Raises:
        GridValidationError: If source axis metadata is invalid.
    Side Effects:
        None.
    """
    variants = _variants_from_axes(axes=axes)
    axis_names = [axis.name for axis in axes]
    if "source" not in axis_names:
        fallback = _fallback_source(definition=definition)
        return tuple(fallback for _ in range(variants))

    source_axis_index = axis_names.index("source")
    source_axis = axes[source_axis_index]
    source_values = source_axis.values_enum
    if source_values is None:
        raise GridValidationError("source axis must have values_enum")

    axis_lengths = tuple(axis.length() for axis in axes)
    labels: list[str] = []
    for coordinate in np.ndindex(axis_lengths):
        labels.append(source_values[coordinate[source_axis_index]])
    return tuple(labels)


def _fallback_source(*, definition: IndicatorDef) -> str:
    """
    Pick deterministic fallback source when indicator has no source axis.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/indicator_def.py,
      src/trading/contexts/indicators/domain/entities/input_series.py

    Args:
        definition: Hard indicator definition.
    Returns:
        str: Deterministic source label.
    Assumptions:
        At least one input series exists by `IndicatorDef` invariants.
    Raises:
        GridValidationError: If definition contains no inputs.
    Side Effects:
        None.
    """
    input_names = sorted(series.value for series in definition.inputs)
    if not input_names:
        raise GridValidationError("indicator definition requires at least one input series")
    if InputSeries.CLOSE.value in input_names:
        return InputSeries.CLOSE.value
    return input_names[0]


def _validate_required_series_available(
    *,
    variant_source_labels: tuple[str, ...],
    available_series: Mapping[str, np.ndarray],
) -> None:
    """
    Validate that all source labels used by variants exist in available series map.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/errors/missing_required_series.py,
      src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        variant_source_labels: Source labels selected for each variant.
        available_series: Mapping of available source arrays.
    Returns:
        None.
    Assumptions:
        `available_series` keys are deterministic source identifiers.
    Raises:
        MissingRequiredSeries: If one or more source labels are unavailable.
    Side Effects:
        None.
    """
    required = tuple(sorted(set(variant_source_labels)))
    available = tuple(sorted(available_series.keys()))
    missing = tuple(name for name in required if name not in available_series)
    if missing:
        raise MissingRequiredSeries(
            "MissingRequiredSeries: "
            f"missing={missing}; required={required}; available={available}"
        )


def _build_variant_source_matrix(
    *,
    variant_source_labels: tuple[str, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
) -> np.ndarray:
    """
    Build contiguous variant-major matrix `(V, T)` from source label assignments.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related:
      src/trading/contexts/indicators/application/dto/indicator_tensor.py,
      src/trading/contexts/indicators/domain/errors/missing_required_series.py

    Args:
        variant_source_labels: Source label for each variant.
        available_series: Source arrays mapping.
        t_size: Time dimension length.
    Returns:
        np.ndarray: Variant-major matrix in float32.
    Assumptions:
        Labels have been validated against available series map.
    Raises:
        MissingRequiredSeries: If source label unexpectedly cannot be resolved.
    Side Effects:
        Allocates one `(V, T)` float32 array.
    """
    variants = len(variant_source_labels)
    matrix = np.empty((variants, t_size), dtype=np.float32, order="C")
    for variant_index, source_name in enumerate(variant_source_labels):
        source = available_series.get(source_name)
        if source is None:
            raise MissingRequiredSeries(f"MissingRequiredSeries: missing={source_name!r}")
        matrix[variant_index, :] = source
    return matrix


def _variants_from_axes(*, axes: tuple[AxisDef, ...]) -> int:
    """
    Calculate deterministic variant count as product of axis lengths.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/axis_def.py,
      src/trading/contexts/indicators/application/services/grid_builder.py

    Args:
        axes: Materialized axis tuple.
    Returns:
        int: Total variant count.
    Assumptions:
        Axis invariants guarantee each axis has positive length.
    Raises:
        None.
    Side Effects:
        None.
    """
    if len(axes) == 0:
        return 1
    variants = 1
    for axis in axes:
        variants = variants * axis.length()
    return variants


__all__ = ["NumbaIndicatorCompute"]
