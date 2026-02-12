"""
Grid materialization and batch estimate services for indicators preflight endpoint.

Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
Related: trading.contexts.indicators.application.dto.grid,
  trading.contexts.indicators.application.dto.estimate_result,
  apps.api.routes.indicators
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_FLOOR, Decimal

from trading.contexts.indicators.application.dto import GridSpec
from trading.contexts.indicators.application.errors import (
    EstimateMemoryGuardExceeded,
    EstimateVariantsGuardExceeded,
)
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.domain.entities import IndicatorDef, ParamDef, ParamKind
from trading.contexts.indicators.domain.errors import GridValidationError
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    RangeValuesSpec,
)
from trading.shared_kernel.primitives import Timeframe, TimeRange

GridScalar = int | float | str

MAX_VARIANTS_PER_COMPUTE_DEFAULT = 600_000  # 600000
MAX_COMPUTE_BYTES_TOTAL_DEFAULT = 5 * 1024**3

_FLOAT32_BYTES = 4
_CANDLES_BYTES_PER_STEP = (5 * _FLOAT32_BYTES) + 8
_RESERVE_FACTOR = 0.20
_RESERVE_FIXED_BYTES = 64 * 1024**2
_MAX_AXIS_VALUES = 1_000_000
_FLOAT_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class MaterializedAxis:
    """
    Deterministically materialized axis preserving request value order.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.domain.specifications.grid_param_spec,
      trading.contexts.indicators.domain.entities.param_def
    """

    name: str
    values: tuple[GridScalar, ...]

    def __post_init__(self) -> None:
        """
        Validate axis shape invariants for materialized values.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Values are already validated against hard indicator contracts by caller.
        Raises:
            ValueError: If name is blank or values are empty.
        Side Effects:
            Normalizes axis name by stripping spaces.
        """
        normalized_name = self.name.strip()
        object.__setattr__(self, "name", normalized_name)
        if not normalized_name:
            raise ValueError("MaterializedAxis requires non-empty name")
        if len(self.values) == 0:
            raise ValueError("MaterializedAxis requires non-empty values")

    def length(self) -> int:
        """
        Return deterministic axis cardinality.

        Args:
            None.
        Returns:
            int: Number of materialized values.
        Assumptions:
            `values` passed post-init checks and is non-empty.
        Raises:
            None.
        Side Effects:
            None.
        """
        return len(self.values)


@dataclass(frozen=True, slots=True)
class MaterializedIndicatorGrid:
    """
    Materialized and validated grid for one indicator request block.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.dto.grid,
      trading.contexts.indicators.domain.entities.indicator_def
    """

    indicator_id: str
    axes: tuple[MaterializedAxis, ...]
    variants: int

    def __post_init__(self) -> None:
        """
        Validate computed indicator-grid snapshot invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants are computed as product of axis lengths or one for empty-axis indicators.
        Raises:
            ValueError: If identifier is blank or variants are not positive.
        Side Effects:
            Normalizes indicator id to stripped lowercase form.
        """
        normalized_id = self.indicator_id.strip().lower()
        object.__setattr__(self, "indicator_id", normalized_id)
        if not normalized_id:
            raise ValueError("MaterializedIndicatorGrid requires non-empty indicator_id")
        if self.variants <= 0:
            raise ValueError("MaterializedIndicatorGrid requires variants > 0")


@dataclass(frozen=True, slots=True)
class BatchEstimateSnapshot:
    """
    Deterministic batch estimate totals returned by preflight service.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.dto.estimate_result,
      apps.api.dto.indicators
    """

    total_variants: int
    estimated_memory_bytes: int

    def __post_init__(self) -> None:
        """
        Validate batch totals are positive deterministic integers.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Totals are produced by `BatchEstimator` formulas.
        Raises:
            ValueError: If totals are non-positive.
        Side Effects:
            None.
        """
        if self.total_variants <= 0:
            raise ValueError("BatchEstimateSnapshot requires total_variants > 0")
        if self.estimated_memory_bytes <= 0:
            raise ValueError("BatchEstimateSnapshot requires estimated_memory_bytes > 0")


class GridBuilder:
    """
    Validate and materialize indicator axes from request grid specs.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.dto.grid,
      trading.contexts.indicators.application.ports.registry.indicator_registry,
      trading.contexts.indicators.domain.entities.param_def
    """

    def __init__(self, *, registry: IndicatorRegistry) -> None:
        """
        Construct service with registry dependency.

        Args:
            registry: Indicator registry application port.
        Returns:
            None.
        Assumptions:
            Registry is initialized and fail-fast validated during application startup.
        Raises:
            ValueError: If registry dependency is missing.
        Side Effects:
            None.
        """
        if registry is None:  # type: ignore[truthy-bool]
            raise ValueError("GridBuilder requires registry")
        self._registry = registry

    def materialize_indicator(self, *, grid: GridSpec) -> MaterializedIndicatorGrid:
        """
        Materialize one indicator grid with deterministic axis ordering and validation.

        Args:
            grid: Per-indicator request grid specification.
        Returns:
            MaterializedIndicatorGrid: Validated materialized axes and variants count.
        Assumptions:
            Indicator axis order is defined by `IndicatorDef.axes`.
        Raises:
            UnknownIndicatorError: If indicator id does not exist in registry.
            GridValidationError: If source/params specs violate hard contracts.
        Side Effects:
            None.
        """
        definition = self._registry.get_def(grid.indicator_id)
        expected_param_axes = {axis_name for axis_name in definition.axes if axis_name != "source"}
        unexpected_param_axes = tuple(sorted(set(grid.params.keys()) - expected_param_axes))
        if unexpected_param_axes:
            raise GridValidationError(
                "unexpected grid param axes: "
                f"indicator_id={definition.indicator_id.value}, axes={unexpected_param_axes}"
            )

        has_source_axis = "source" in definition.axes
        if has_source_axis and grid.source is None:
            raise GridValidationError("source axis is required by indicator definition")
        if not has_source_axis and grid.source is not None:
            raise GridValidationError("source axis is not supported by indicator definition")

        params_by_name = {param.name: param for param in definition.params}
        axes: list[MaterializedAxis] = []

        for axis_name in definition.axes:
            if axis_name == "source":
                source_spec = grid.source
                if source_spec is None:  # pragma: no cover - guarded above
                    raise GridValidationError("source axis is required by indicator definition")
                source_values = _materialize_source_values(
                    spec=source_spec,
                    definition=definition,
                )
                axes.append(MaterializedAxis(name="source", values=source_values))
                continue

            spec = grid.params.get(axis_name)
            if spec is None:
                raise GridValidationError(f"missing grid param axis: {axis_name}")
            param_def = params_by_name.get(axis_name)
            if param_def is None:
                raise GridValidationError(f"unknown axis in definition: {axis_name}")
            axis_values = _materialize_param_values(
                axis_name=axis_name,
                spec=spec,
                param_def=param_def,
            )
            axes.append(MaterializedAxis(name=axis_name, values=axis_values))

        variants = _variants_from_axes(axes=tuple(axes))
        return MaterializedIndicatorGrid(
            indicator_id=definition.indicator_id.value,
            axes=tuple(axes),
            variants=variants,
        )


class BatchEstimator:
    """
    Estimate batch totals (variants + memory) without loading candles.

    Docs: docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related: trading.contexts.indicators.application.services.grid_builder.GridBuilder,
      trading.shared_kernel.primitives.time_range,
      trading.shared_kernel.primitives.timeframe
    """

    def __init__(self, *, grid_builder: GridBuilder) -> None:
        """
        Construct estimator with grid-builder dependency.

        Args:
            grid_builder: Grid materialization service.
        Returns:
            None.
        Assumptions:
            Grid builder performs hard contract checks for each indicator block.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if grid_builder is None:  # type: ignore[truthy-bool]
            raise ValueError("BatchEstimator requires grid_builder")
        self._grid_builder = grid_builder

    def estimate_batch(
        self,
        *,
        indicator_grids: tuple[GridSpec, ...],
        sl_spec: GridParamSpec,
        tp_spec: GridParamSpec,
        time_range: TimeRange,
        timeframe: Timeframe,
    ) -> BatchEstimateSnapshot:
        """
        Calculate deterministic batch totals for `/indicators/estimate`.

        Args:
            indicator_grids: Batch indicator blocks with per-indicator grids.
            sl_spec: Risk SL axis specification.
            tp_spec: Risk TP axis specification.
            time_range: Half-open time range `[start, end)` for T estimation.
            timeframe: Timeframe primitive used to derive dense timeline size.
        Returns:
            BatchEstimateSnapshot: Totals for variants and estimated memory bytes.
        Assumptions:
            Memory estimate uses float32-only policy for indicator tensors.
        Raises:
            GridValidationError: If batch is empty or any axis specification is invalid.
            UnknownIndicatorError: If indicator definition is not found in registry.
        Side Effects:
            None.
        """
        if len(indicator_grids) == 0:
            raise GridValidationError("estimate batch requires at least one indicator block")

        materialized = tuple(
            self._grid_builder.materialize_indicator(grid=grid)
            for grid in indicator_grids
        )
        indicator_variants_product = 1
        for item in materialized:
            indicator_variants_product = indicator_variants_product * item.variants

        sl_values = _materialize_risk_axis_values(spec=sl_spec, axis_name="sl")
        tp_values = _materialize_risk_axis_values(spec=tp_spec, axis_name="tp")
        total_variants = indicator_variants_product * len(sl_values) * len(tp_values)

        timeline_size = _time_steps_from_time_range_and_timeframe(
            time_range=time_range,
            timeframe=timeframe,
        )
        bytes_candles = timeline_size * _CANDLES_BYTES_PER_STEP
        bytes_indicators = 0
        for item in materialized:
            bytes_indicators = bytes_indicators + (timeline_size * item.variants * _FLOAT32_BYTES)

        reserve_base = bytes_candles + bytes_indicators
        reserve = max(_RESERVE_FIXED_BYTES, int(math.ceil(reserve_base * _RESERVE_FACTOR)))
        estimated_memory_bytes = reserve_base + reserve

        return BatchEstimateSnapshot(
            total_variants=total_variants,
            estimated_memory_bytes=estimated_memory_bytes,
        )


def enforce_batch_guards(
    *,
    estimate: BatchEstimateSnapshot,
    max_variants_per_compute: int,
    max_compute_bytes_total: int,
) -> None:
    """
    Enforce public preflight limits for variants and memory totals.

    Args:
        estimate: Batch estimate totals from `BatchEstimator`.
        max_variants_per_compute: Allowed variants limit for one request.
        max_compute_bytes_total: Allowed total memory estimate budget in bytes.
    Returns:
        None.
    Assumptions:
        Limits are loaded from runtime config with fail-fast startup checks.
    Raises:
        ValueError: If configured limits are non-positive.
        EstimateVariantsGuardExceeded: If variants exceed configured limit.
        EstimateMemoryGuardExceeded: If memory estimate exceeds configured limit.
    Side Effects:
        None.
    """
    if max_variants_per_compute <= 0:
        raise ValueError(
            "max_variants_per_compute must be > 0, "
            f"got {max_variants_per_compute}"
        )
    if max_compute_bytes_total <= 0:
        raise ValueError(
            "max_compute_bytes_total must be > 0, "
            f"got {max_compute_bytes_total}"
        )

    if estimate.total_variants > max_variants_per_compute:
        raise EstimateVariantsGuardExceeded(
            total_variants=estimate.total_variants,
            max_variants_per_compute=max_variants_per_compute,
        )
    if estimate.estimated_memory_bytes > max_compute_bytes_total:
        raise EstimateMemoryGuardExceeded(
            estimated_memory_bytes=estimate.estimated_memory_bytes,
            max_compute_bytes_total=max_compute_bytes_total,
        )


def _materialize_source_values(
    *,
    spec: GridParamSpec,
    definition: IndicatorDef,
) -> tuple[str, ...]:
    """
    Materialize and validate `source` axis values in request order.

    Args:
        spec: Source axis specification.
        definition: Hard indicator definition for allowed input series.
    Returns:
        tuple[str, ...]: Normalized source names preserving request ordering.
    Assumptions:
        Source values are validated against `definition.inputs`.
    Raises:
        GridValidationError: If values are blank, duplicated, or unsupported.
    Side Effects:
        None.
    """
    raw_values = _materialize_axis_values(spec=spec)
    if len(raw_values) == 0:
        raise GridValidationError("source axis values must be non-empty")

    allowed = {series.value for series in definition.inputs}
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            raise GridValidationError("source axis values must be strings")
        candidate = value.strip().lower()
        if not candidate:
            raise GridValidationError("source axis values must be non-empty")
        if candidate not in allowed:
            raise GridValidationError(f"unsupported source axis value: {candidate!r}")
        if candidate in seen:
            raise GridValidationError(f"duplicate source axis value: {candidate!r}")
        normalized.append(candidate)
        seen.add(candidate)
    return tuple(normalized)


def _materialize_param_values(
    *,
    axis_name: str,
    spec: GridParamSpec,
    param_def: ParamDef,
) -> tuple[GridScalar, ...]:
    """
    Materialize one parameter axis with hard kind/bounds validation.

    Args:
        axis_name: Axis name for deterministic diagnostics.
        spec: Axis spec (`explicit` or `range`).
        param_def: Hard parameter definition from indicator contract.
    Returns:
        tuple[GridScalar, ...]: Materialized values preserving explicit order.
    Assumptions:
        Hard bounds are inclusive and `ParamDef` invariants already hold.
    Raises:
        GridValidationError: If values violate kind, bounds, or hard step constraints.
    Side Effects:
        None.
    """
    values = _materialize_axis_values(spec=spec)
    if len(values) == 0:
        raise GridValidationError(f"axis '{axis_name}' materialized to empty values")

    if param_def.kind is ParamKind.ENUM:
        return _validate_enum_axis_values(
            axis_name=axis_name,
            values=values,
            param_def=param_def,
        )

    if param_def.kind is ParamKind.INT:
        return _validate_int_axis_values(
            axis_name=axis_name,
            values=values,
            spec=spec,
            param_def=param_def,
        )

    return _validate_float_axis_values(
        axis_name=axis_name,
        values=values,
        spec=spec,
        param_def=param_def,
    )


def _validate_enum_axis_values(
    *,
    axis_name: str,
    values: tuple[GridScalar, ...],
    param_def: ParamDef,
) -> tuple[str, ...]:
    """
    Validate enum-axis values against hard enum contract.

    Args:
        axis_name: Axis name for deterministic error messages.
        values: Materialized raw values.
        param_def: Enum hard definition.
    Returns:
        tuple[str, ...]: Normalized enum values preserving request order.
    Assumptions:
        Enum values are expected as non-empty strings.
    Raises:
        GridValidationError: If values are invalid or outside allowed enum set.
    Side Effects:
        None.
    """
    allowed = set(param_def.enum_values or ())
    if not allowed:
        raise GridValidationError(f"axis '{axis_name}' enum definition is empty")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise GridValidationError(f"axis '{axis_name}' expects string values")
        candidate = value.strip()
        if not candidate:
            raise GridValidationError(f"axis '{axis_name}' contains blank enum value")
        if candidate not in allowed:
            raise GridValidationError(
                f"axis '{axis_name}' value {candidate!r} is outside allowed enum set"
            )
        if candidate in seen:
            raise GridValidationError(f"axis '{axis_name}' contains duplicate value {candidate!r}")
        normalized.append(candidate)
        seen.add(candidate)

    return tuple(normalized)


def _validate_int_axis_values(
    *,
    axis_name: str,
    values: tuple[GridScalar, ...],
    spec: GridParamSpec,
    param_def: ParamDef,
) -> tuple[int, ...]:
    """
    Validate integer-axis values against hard int bounds and range-step rules.

    Args:
        axis_name: Axis name for deterministic error messages.
        values: Materialized raw values.
        spec: Source axis spec to detect range-specific validations.
        param_def: Integer hard parameter definition.
    Returns:
        tuple[int, ...]: Validated integer values preserving request order.
    Assumptions:
        Integer axes reject booleans and float-like values.
    Raises:
        GridValidationError: If type, bounds, duplicates, or step rules are violated.
    Side Effects:
        None.
    """
    normalized: list[int] = []
    seen: set[int] = set()

    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise GridValidationError(f"axis '{axis_name}' expects integer values")
        candidate = int(value)
        _validate_numeric_bounds(axis_name=axis_name, value=float(candidate), param_def=param_def)
        if candidate in seen:
            raise GridValidationError(f"axis '{axis_name}' contains duplicate value {candidate}")
        normalized.append(candidate)
        seen.add(candidate)

    if isinstance(spec, RangeValuesSpec):
        if isinstance(spec.step, bool) or not isinstance(spec.step, int):
            raise GridValidationError(f"axis '{axis_name}' range step must be integer")
        _validate_range_step_against_hard_step(
            axis_name=axis_name,
            range_step=float(spec.step),
            param_def=param_def,
        )
        _validate_values_on_step_grid(
            axis_name=axis_name,
            values=tuple(float(item) for item in normalized),
            param_def=param_def,
        )

    return tuple(normalized)


def _validate_float_axis_values(
    *,
    axis_name: str,
    values: tuple[GridScalar, ...],
    spec: GridParamSpec,
    param_def: ParamDef,
) -> tuple[float, ...]:
    """
    Validate float-axis values against hard float bounds and range-step rules.

    Args:
        axis_name: Axis name for deterministic error messages.
        values: Materialized raw values.
        spec: Source axis spec to detect range-specific validations.
        param_def: Float hard parameter definition.
    Returns:
        tuple[float, ...]: Validated float values preserving request order.
    Assumptions:
        Float axes accept int/float numeric values but reject booleans.
    Raises:
        GridValidationError: If type, bounds, duplicates, or step rules are violated.
    Side Effects:
        None.
    """
    normalized: list[float] = []
    seen: set[float] = set()

    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GridValidationError(f"axis '{axis_name}' expects numeric values")
        candidate = float(value)
        _validate_numeric_bounds(axis_name=axis_name, value=candidate, param_def=param_def)
        if candidate in seen:
            raise GridValidationError(f"axis '{axis_name}' contains duplicate value {candidate}")
        normalized.append(candidate)
        seen.add(candidate)

    if isinstance(spec, RangeValuesSpec):
        if isinstance(spec.step, bool) or not isinstance(spec.step, (int, float)):
            raise GridValidationError(f"axis '{axis_name}' range step must be numeric")
        _validate_range_step_against_hard_step(
            axis_name=axis_name,
            range_step=float(spec.step),
            param_def=param_def,
        )
        _validate_values_on_step_grid(
            axis_name=axis_name,
            values=tuple(normalized),
            param_def=param_def,
        )

    return tuple(normalized)


def _validate_numeric_bounds(*, axis_name: str, value: float, param_def: ParamDef) -> None:
    """
    Validate one numeric value against inclusive hard min/max.

    Args:
        axis_name: Axis name for deterministic diagnostics.
        value: Numeric value to validate.
        param_def: Hard numeric parameter definition.
    Returns:
        None.
    Assumptions:
        Hard bounds are inclusive when present.
    Raises:
        GridValidationError: If value falls outside hard bounds.
    Side Effects:
        None.
    """
    if param_def.hard_min is not None and value < float(param_def.hard_min):
        raise GridValidationError(
            f"axis '{axis_name}' value {value} is below hard_min={param_def.hard_min}"
        )
    if param_def.hard_max is not None and value > float(param_def.hard_max):
        raise GridValidationError(
            f"axis '{axis_name}' value {value} is above hard_max={param_def.hard_max}"
        )


def _validate_range_step_against_hard_step(
    *,
    axis_name: str,
    range_step: float,
    param_def: ParamDef,
) -> None:
    """
    Validate provided range step against hard parameter step.

    Args:
        axis_name: Axis name for deterministic diagnostics.
        range_step: Step from request `range` specification.
        param_def: Hard numeric parameter definition.
    Returns:
        None.
    Assumptions:
        Hard numeric step is optional but positive when present.
    Raises:
        GridValidationError: If range step is not aligned with hard step.
    Side Effects:
        None.
    """
    if param_def.step is None:
        return
    hard_step = float(param_def.step)
    if hard_step <= 0:
        raise GridValidationError(f"axis '{axis_name}' has invalid hard step={hard_step}")
    quotient = range_step / hard_step
    if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=_FLOAT_EPS):
        raise GridValidationError(
            "range step must be a multiple of hard step: "
            f"axis='{axis_name}', step={range_step}, hard_step={hard_step}"
        )


def _validate_values_on_step_grid(
    *,
    axis_name: str,
    values: tuple[float, ...],
    param_def: ParamDef,
) -> None:
    """
    Validate numeric values lie on hard-step grid when hard step is defined.

    Args:
        axis_name: Axis name for deterministic diagnostics.
        values: Numeric values to validate.
        param_def: Hard numeric parameter definition.
    Returns:
        None.
    Assumptions:
        Step-grid origin is `hard_min` when present, otherwise zero.
    Raises:
        GridValidationError: If any value is not aligned with hard step grid.
    Side Effects:
        None.
    """
    if param_def.step is None:
        return
    hard_step = float(param_def.step)
    origin = float(param_def.hard_min) if param_def.hard_min is not None else 0.0
    for value in values:
        ratio = (value - origin) / hard_step
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=_FLOAT_EPS):
            raise GridValidationError(
                "value is not aligned with hard step grid: "
                f"axis='{axis_name}', value={value}, hard_step={hard_step}, origin={origin}"
            )


def _materialize_risk_axis_values(*, spec: GridParamSpec, axis_name: str) -> tuple[float, ...]:
    """
    Materialize SL/TP axis values for batch variants multiplication.

    Args:
        spec: Risk-axis specification (`sl` or `tp`).
        axis_name: Axis name for deterministic diagnostics.
    Returns:
        tuple[float, ...]: Numeric values preserving explicit request ordering.
    Assumptions:
        Risk axes in v1 are numeric and do not have hard bounds in indicator defs.
    Raises:
        GridValidationError: If values are non-numeric, duplicated, or empty.
    Side Effects:
        None.
    """
    raw_values = _materialize_axis_values(spec=spec)
    if len(raw_values) == 0:
        raise GridValidationError(f"risk axis '{axis_name}' must be non-empty")

    normalized: list[float] = []
    seen: set[float] = set()
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GridValidationError(f"risk axis '{axis_name}' expects numeric values")
        candidate = float(value)
        if candidate in seen:
            raise GridValidationError(
                f"risk axis '{axis_name}' contains duplicate value {candidate}"
            )
        normalized.append(candidate)
        seen.add(candidate)
    return tuple(normalized)


def _materialize_axis_values(*, spec: GridParamSpec) -> tuple[GridScalar, ...]:
    """
    Materialize one axis spec with deterministic `explicit`/`range` semantics.

    Args:
        spec: Axis specification contract.
    Returns:
        tuple[GridScalar, ...]: Materialized values in deterministic order.
    Assumptions:
        v1 supports `ExplicitValuesSpec` and `RangeValuesSpec`.
    Raises:
        GridValidationError: If materialization is unsupported or invalid.
    Side Effects:
        None.
    """
    if isinstance(spec, ExplicitValuesSpec):
        values = tuple(spec.values)
        if len(values) == 0:
            raise GridValidationError("explicit axis values must be non-empty")
        return values

    if isinstance(spec, RangeValuesSpec):
        return _materialize_range_values(spec=spec)

    try:
        values = tuple(spec.materialize())
    except ValueError as error:
        raise GridValidationError(str(error)) from error
    if len(values) == 0:
        raise GridValidationError("axis values must be non-empty")
    return values


def _materialize_range_values(*, spec: RangeValuesSpec) -> tuple[int, ...] | tuple[float, ...]:
    """
    Materialize inclusive range using index-based deterministic formula.

    Args:
        spec: Range specification with `start`, `stop_inclusive`, and positive `step`.
    Returns:
        tuple[int, ...] | tuple[float, ...]: Inclusive deterministic values.
    Assumptions:
        Range values are computed with `n=floor((stop-start)/step)+1` and `start+i*step`.
    Raises:
        GridValidationError: If range contains invalid numeric types or is too large.
    Side Effects:
        None.
    """
    start = spec.start
    stop_inclusive = spec.stop_inclusive
    step = spec.step

    if isinstance(start, bool) or isinstance(stop_inclusive, bool) or isinstance(step, bool):
        raise GridValidationError("range values must be numeric and must not be booleans")
    if not isinstance(start, (int, float)):
        raise GridValidationError("range start must be numeric")
    if not isinstance(stop_inclusive, (int, float)):
        raise GridValidationError("range stop_inclusive must be numeric")
    if not isinstance(step, (int, float)):
        raise GridValidationError("range step must be numeric")
    if step <= 0:
        raise GridValidationError("range step must be > 0")
    if start > stop_inclusive:
        raise GridValidationError("range requires start <= stop_inclusive")

    is_int_range = (
        isinstance(start, int)
        and isinstance(stop_inclusive, int)
        and isinstance(step, int)
    )
    if is_int_range:
        span = int(stop_inclusive) - int(start)
        count = (span // int(step)) + 1
        if count <= 0:
            raise GridValidationError("range materialized an empty sequence")
        if count > _MAX_AXIS_VALUES:
            raise GridValidationError(
                f"range generated too many values: count={count}, max={_MAX_AXIS_VALUES}"
            )
        values_int = tuple(int(start) + int(step) * index for index in range(count))
        return values_int

    start_dec = Decimal(str(start))
    stop_dec = Decimal(str(stop_inclusive))
    step_dec = Decimal(str(step))
    if step_dec <= 0:
        raise GridValidationError("range step must be > 0")
    if start_dec > stop_dec:
        raise GridValidationError("range requires start <= stop_inclusive")

    count_dec = ((stop_dec - start_dec) / step_dec).to_integral_value(rounding=ROUND_FLOOR)
    count = int(count_dec) + 1
    if count <= 0:
        raise GridValidationError("range materialized an empty sequence")
    if count > _MAX_AXIS_VALUES:
        raise GridValidationError(
            f"range generated too many values: count={count}, max={_MAX_AXIS_VALUES}"
        )

    values_float = tuple(float(start_dec + (step_dec * index)) for index in range(count))
    return values_float


def _variants_from_axes(*, axes: tuple[MaterializedAxis, ...]) -> int:
    """
    Compute indicator variants as product of materialized axis lengths.

    Args:
        axes: Materialized axes in deterministic definition order.
    Returns:
        int: Variant count (`1` for empty-axis indicator definitions).
    Assumptions:
        Each axis has at least one value by `MaterializedAxis` invariants.
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


def _time_steps_from_time_range_and_timeframe(
    *,
    time_range: TimeRange,
    timeframe: Timeframe,
) -> int:
    """
    Derive dense timeline size `T` from `time_range` and `timeframe` only.

    Args:
        time_range: Half-open UTC interval `[start, end)`.
        timeframe: Timeframe primitive describing one bucket duration.
    Returns:
        int: Number of timeline steps (`T`) without loading candles.
    Assumptions:
        Duration must align exactly to timeframe bucket duration.
    Raises:
        GridValidationError: If timeframe duration is invalid or range is misaligned.
    Side Effects:
        None.
    """
    frame_micros = timeframe.duration() // timedelta(microseconds=1)
    if frame_micros <= 0:
        raise GridValidationError(f"timeframe duration must be > 0, got {frame_micros}")

    duration_micros = time_range.duration() // timedelta(microseconds=1)
    if duration_micros <= 0:
        raise GridValidationError(
            f"time_range duration must be > 0, got {duration_micros}"
        )
    if duration_micros % frame_micros != 0:
        raise GridValidationError(
            "time_range duration must align with timeframe: "
            f"duration_us={duration_micros}, timeframe_us={frame_micros}"
        )

    timeline_size = int(duration_micros // frame_micros)
    if timeline_size <= 0:
        raise GridValidationError(f"time steps must be > 0, got {timeline_size}")
    return timeline_size


__all__ = [
    "BatchEstimateSnapshot",
    "BatchEstimator",
    "GridBuilder",
    "MAX_COMPUTE_BYTES_TOTAL_DEFAULT",
    "MAX_VARIANTS_PER_COMPUTE_DEFAULT",
    "MaterializedAxis",
    "MaterializedIndicatorGrid",
    "enforce_batch_guards",
]
