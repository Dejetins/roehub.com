"""
CPU/Numba implementation of indicators compute application port.

Docs: docs/architecture/indicators/indicators-compute-engine-core.md
Related: trading.contexts.indicators.application.ports.compute.indicator_compute,
  trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
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
    estimate_tensor_bytes,
    estimate_total_bytes,
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
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.domain.entities import (
    AxisDef,
    IndicatorDef,
    IndicatorId,
    InputSeries,
    Layout,
    ParamDef,
    ParamKind,
)
from trading.contexts.indicators.domain.errors import (
    GridValidationError,
    MissingRequiredSeries,
    UnknownIndicatorError,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.platform.config import IndicatorsComputeNumbaConfig


class NumbaIndicatorCompute(IndicatorCompute):
    """
    Indicator compute engine skeleton based on CPU+Numba common kernels.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.application.dto.indicator_tensor,
      trading.contexts.indicators.domain.entities.layout,
      trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common
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
        for definition in sorted(defs, key=lambda item: item.indicator_id.value):
            key = definition.indicator_id.value
            if key in defs_by_id:
                raise ValueError(f"duplicate indicator_id in defs: {key}")
            defs_by_id[key] = definition

        self._defs_by_id: Mapping[str, IndicatorDef] = MappingProxyType(defs_by_id)
        self._config = config
        self._workspace_factor = workspace_factor
        self._workspace_fixed_bytes = workspace_fixed_bytes
        self._warmup_runner = warmup_runner or ComputeNumbaWarmupRunner(config=config)

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Estimate axis cardinality and variant count without kernel execution.

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
        definition = self._get_definition(indicator_id=grid.indicator_id)
        axes = self._resolve_axes(definition=definition, grid=grid)
        variants = _variants_from_axes(axes=axes)
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
        Compute indicator tensor skeleton with deterministic layout and guards.

        Args:
            req: Compute request with candles and grid specification.
        Returns:
            IndicatorTensor: Float32 tensor in selected layout.
        Assumptions:
            Indicator-specific math kernels are introduced in later epics; v1 skeleton
            writes selected source series per variant.
        Raises:
            UnknownIndicatorError: If indicator id is not defined.
            GridValidationError: If grid/layout/guard invariants are violated.
            MissingRequiredSeries: If required source series cannot be resolved.
            ComputeBudgetExceeded: If estimated total bytes exceed budget.
        Side Effects:
            Allocates output tensor and intermediate source matrix.
        """
        started = time.perf_counter()
        definition = self._get_definition(indicator_id=req.grid.indicator_id)
        axes = self._resolve_axes(definition=definition, grid=req.grid)
        variants = _variants_from_axes(axes=axes)
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

    def _resolve_axes(self, *, definition: IndicatorDef, grid: GridSpec) -> tuple[AxisDef, ...]:
        """
        Materialize axis metadata from grid and hard definition contracts.

        Args:
            definition: Hard indicator definition.
            grid: Grid specification from request.
        Returns:
            tuple[AxisDef, ...]: Deterministic axis descriptors in definition order.
        Assumptions:
            `definition.axes` describes all variant dimensions.
        Raises:
            GridValidationError: If required axis spec is missing or invalid.
        Side Effects:
            None.
        """
        param_by_name = {param.name: param for param in definition.params}
        materialized_axes: list[AxisDef] = []

        for axis_name in definition.axes:
            if axis_name == "source":
                if grid.source is None:
                    raise GridValidationError("source axis is required by indicator definition")
                materialized_axes.append(_build_source_axis(spec=grid.source))
                continue

            spec = grid.params.get(axis_name)
            if spec is None:
                raise GridValidationError(f"missing grid param axis: {axis_name}")
            param_def = param_by_name.get(axis_name)
            if param_def is None:
                raise GridValidationError(f"unknown axis in definition: {axis_name}")
            materialized_axes.append(
                _build_param_axis(
                    axis_name=axis_name,
                    spec=spec,
                    param_def=param_def,
                )
            )

        if not materialized_axes:
            raise GridValidationError("indicator definition must expose at least one axis")
        return tuple(materialized_axes)


def _build_source_axis(*, spec: GridParamSpec) -> AxisDef:
    """
    Materialize deterministic source axis from grid specification.

    Args:
        spec: Grid axis specification for `source`.
    Returns:
        AxisDef: Source axis definition with sorted enum values.
    Assumptions:
        Source axis values are string identifiers of `InputSeries`.
    Raises:
        GridValidationError: If source values are empty or invalid.
    Side Effects:
        None.
    """
    values = spec.materialize()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise GridValidationError("source axis values must be strings")
        candidate = value.strip().lower()
        if not candidate:
            raise GridValidationError("source axis values must be non-empty")
        normalized.append(candidate)

    unique_sorted = tuple(sorted(set(normalized)))
    if not unique_sorted:
        raise GridValidationError("source axis values must be non-empty")
    for source_name in unique_sorted:
        try:
            InputSeries(source_name)
        except ValueError as error:
            raise GridValidationError(f"unsupported source axis value: {source_name!r}") from error
    return AxisDef(name="source", values_enum=unique_sorted)


def _build_param_axis(*, axis_name: str, spec: GridParamSpec, param_def: ParamDef) -> AxisDef:
    """
    Materialize deterministic parameter axis according to declared parameter kind.

    Args:
        axis_name: Axis name for diagnostics.
        spec: Grid axis specification.
        param_def: Hard parameter definition.
    Returns:
        AxisDef: Materialized axis descriptor with sorted values.
    Assumptions:
        Spec materialization follows `GridParamSpec` contract.
    Raises:
        GridValidationError: If axis values are incompatible with parameter kind.
    Side Effects:
        None.
    """
    values = spec.materialize()
    if len(values) == 0:
        raise GridValidationError(f"axis '{axis_name}' materialized to empty values")

    if param_def.kind is ParamKind.ENUM:
        normalized = _normalize_enum_values(values=values, axis_name=axis_name)
        return AxisDef(name=axis_name, values_enum=normalized)

    if param_def.kind is ParamKind.INT:
        normalized_int = _normalize_int_values(values=values, axis_name=axis_name)
        return AxisDef(name=axis_name, values_int=normalized_int)

    normalized_float = _normalize_float_values(values=values, axis_name=axis_name)
    return AxisDef(name=axis_name, values_float=normalized_float)


def _normalize_enum_values(*, values: tuple[object, ...], axis_name: str) -> tuple[str, ...]:
    """
    Normalize enum axis values into deterministic sorted tuple.

    Args:
        values: Raw axis values.
        axis_name: Axis name for diagnostics.
    Returns:
        tuple[str, ...]: Sorted unique enum values.
    Assumptions:
        Enum values are represented as strings.
    Raises:
        GridValidationError: If values are not non-empty strings.
    Side Effects:
        None.
    """
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise GridValidationError(f"axis '{axis_name}' expects string values")
        candidate = value.strip()
        if not candidate:
            raise GridValidationError(f"axis '{axis_name}' contains blank enum value")
        normalized.add(candidate)
    return tuple(sorted(normalized))


def _normalize_int_values(*, values: tuple[object, ...], axis_name: str) -> tuple[int, ...]:
    """
    Normalize integer axis values into deterministic sorted tuple.

    Args:
        values: Raw axis values.
        axis_name: Axis name for diagnostics.
    Returns:
        tuple[int, ...]: Sorted unique integer values.
    Assumptions:
        Integer axes do not accept booleans.
    Raises:
        GridValidationError: If values are not integers.
    Side Effects:
        None.
    """
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise GridValidationError(f"axis '{axis_name}' expects integer values")
        normalized.add(int(value))
    return tuple(sorted(normalized))


def _normalize_float_values(*, values: tuple[object, ...], axis_name: str) -> tuple[float, ...]:
    """
    Normalize float axis values into deterministic sorted tuple.

    Args:
        values: Raw axis values.
        axis_name: Axis name for diagnostics.
    Returns:
        tuple[float, ...]: Sorted unique float values.
    Assumptions:
        Numeric float axes accept ints/floats but reject booleans.
    Raises:
        GridValidationError: If values are non-numeric.
    Side Effects:
        None.
    """
    normalized: set[float] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GridValidationError(f"axis '{axis_name}' expects numeric values")
        normalized.add(float(value))
    return tuple(sorted(normalized))


def _variants_from_axes(*, axes: tuple[AxisDef, ...]) -> int:
    """
    Calculate deterministic variant count as product of axis lengths.

    Args:
        axes: Materialized axis tuple.
    Returns:
        int: Total variant count.
    Assumptions:
        Axis invariants guarantee each axis has positive length.
    Raises:
        GridValidationError: If axis tuple is empty.
    Side Effects:
        None.
    """
    if len(axes) == 0:
        raise GridValidationError("axes must not be empty")
    variants = 1
    for axis in axes:
        variants = variants * axis.length()
    return variants


def _build_series_map(*, candles: CandleArrays) -> Mapping[str, np.ndarray]:
    """
    Build deterministic source-series map including derived OHLC aggregates.

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


def _variant_source_labels(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
) -> tuple[str, ...]:
    """
    Resolve one source label for each variant in deterministic variant order.

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


__all__ = ["NumbaIndicatorCompute"]
