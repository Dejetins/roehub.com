from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    IndicatorVariantSelection,
    build_variant_key_v1,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.platform.errors import RoehubError

STAGE_A_LITERAL = "stage_a"
STAGE_B_LITERAL = "stage_b"

_FLOAT32_BYTES = 4
_CANDLES_BYTES_PER_STEP = (5 * _FLOAT32_BYTES) + 8
_RESERVE_FACTOR = 0.20
_RESERVE_FIXED_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class _IndicatorAxisPlan:
    """
    Internal deterministic axis plan for one compute-axis of an indicator.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/indicators/application/dto/estimate_result.py
      - src/trading/contexts/indicators/domain/entities/axis_def.py
    """

    name: str
    values: tuple[int | float | str, ...]


@dataclass(frozen=True, slots=True)
class _IndicatorPlan:
    """
    Internal deterministic compute plan for one indicator in Stage A grid.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
    """

    indicator_id: str
    axes: tuple[_IndicatorAxisPlan, ...]
    variants: int


@dataclass(frozen=True, slots=True)
class _SignalAxisPlan:
    """
    Internal deterministic signal-parameter axis plan used in Stage A grid.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - configs/prod/indicators.yaml
    """

    indicator_id: str
    param_name: str
    values: tuple[BacktestVariantScalar, ...]


@dataclass(frozen=True, slots=True)
class BacktestRiskVariantV1:
    """
    One deterministic Stage B risk variant (`sl_enabled/sl_pct/tp_enabled/tp_pct`).

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_grid_builder_v1.py
    """

    risk_index: int
    risk_params: Mapping[str, BacktestVariantScalar]

    def __post_init__(self) -> None:
        """
        Validate and freeze deterministic risk payload mapping.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Risk payload contains stable scalar keys owned by backtest context.
        Raises:
            ValueError: If index is negative or one mapping key is blank.
        Side Effects:
            Replaces mutable mapping with immutable mapping proxy.
        """
        if self.risk_index < 0:
            raise ValueError("BacktestRiskVariantV1.risk_index must be >= 0")

        normalized: dict[str, BacktestVariantScalar] = {}
        for raw_key in sorted(self.risk_params.keys()):
            key = str(raw_key).strip()
            if not key:
                raise ValueError("BacktestRiskVariantV1 risk_params keys must be non-empty")
            normalized[key] = self.risk_params[raw_key]
        object.__setattr__(self, "risk_params", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class BacktestStageABaseVariant:
    """
    Deterministic Stage A base variant identity before Stage B risk expansion.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
    """

    stage_a_index: int
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]]
    indicator_variant_key: str
    base_variant_key: str

    def __post_init__(self) -> None:
        """
        Validate base variant identity payload invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variant keys are canonical lowercase SHA-256 strings.
        Raises:
            ValueError: If index is negative or one key has invalid shape.
        Side Effects:
            None.
        """
        if self.stage_a_index < 0:
            raise ValueError("BacktestStageABaseVariant.stage_a_index must be >= 0")
        if len(self.indicator_variant_key) != 64:
            raise ValueError(
                "BacktestStageABaseVariant.indicator_variant_key must be 64 hex chars"
            )
        if len(self.base_variant_key) != 64:
            raise ValueError("BacktestStageABaseVariant.base_variant_key must be 64 hex chars")


@dataclass(frozen=True, slots=True)
class BacktestGridBuildContextV1:
    """
    Deterministic build context for Stage A enumeration and Stage B expansion math.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    indicator_plans: tuple[_IndicatorPlan, ...]
    signal_axes: tuple[_SignalAxisPlan, ...]
    risk_variants: tuple[BacktestRiskVariantV1, ...]
    instrument_id_literal: str
    timeframe_code: str
    direction_mode: str
    sizing_mode: str
    execution_params: Mapping[str, BacktestVariantScalar]
    stage_a_variants_total: int
    stage_b_variants_total: int
    estimated_memory_bytes: int
    indicator_estimate_calls: int

    def __post_init__(self) -> None:
        """
        Validate and freeze deterministic context-level scalar invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stage totals and memory totals are guarded by builder before construction.
        Raises:
            ValueError: If totals are invalid or identity literals are blank.
        Side Effects:
            Replaces execution parameters mapping with immutable mapping proxy.
        """
        if not self.instrument_id_literal.strip():
            raise ValueError("BacktestGridBuildContextV1.instrument_id_literal must be non-empty")
        if not self.timeframe_code.strip():
            raise ValueError("BacktestGridBuildContextV1.timeframe_code must be non-empty")
        if self.stage_a_variants_total <= 0:
            raise ValueError("BacktestGridBuildContextV1.stage_a_variants_total must be > 0")
        if self.stage_b_variants_total <= 0:
            raise ValueError("BacktestGridBuildContextV1.stage_b_variants_total must be > 0")
        if self.estimated_memory_bytes <= 0:
            raise ValueError("BacktestGridBuildContextV1.estimated_memory_bytes must be > 0")
        if self.indicator_estimate_calls < 0:
            raise ValueError("BacktestGridBuildContextV1.indicator_estimate_calls must be >= 0")
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )

    def iter_stage_a_variants(self) -> Iterator[BacktestStageABaseVariant]:
        """
        Iterate deterministic Stage A base variants using mixed-radix variant indexes.

        Args:
            None.
        Returns:
            Iterator[BacktestStageABaseVariant]: Deterministic Stage A variants.
        Assumptions:
            Indicator and signal plans are already normalized and sorted by builder.
        Raises:
            ValueError: If one mixed-radix coordinate cannot be decoded.
        Side Effects:
            None.
        """
        signal_variants_total = _product(
            values=tuple(len(axis.values) for axis in self.signal_axes)
        )
        indicator_radices = tuple(plan.variants for plan in self.indicator_plans)
        for stage_a_index in range(self.stage_a_variants_total):
            compute_index = stage_a_index // signal_variants_total
            signal_index = stage_a_index % signal_variants_total

            indicator_variant_indexes = _decode_mixed_radix(
                flat_index=compute_index,
                radices=indicator_radices,
            )
            indicator_selections = tuple(
                _indicator_selection_from_variant_index(
                    plan=plan,
                    variant_index=indicator_variant_indexes[position],
                )
                for position, plan in enumerate(self.indicator_plans)
            )

            indicator_variant_key = build_variant_key_v1(
                instrument_id=self.instrument_id_literal,
                timeframe=self.timeframe_code,
                indicators=indicator_selections,
            )
            signal_params = _signal_params_from_variant_index(
                signal_axes=self.signal_axes,
                variant_index=signal_index,
            )
            base_variant_key = build_backtest_variant_key_v1(
                indicator_variant_key=indicator_variant_key,
                direction_mode=self.direction_mode,
                sizing_mode=self.sizing_mode,
                signals=signal_params,
                risk_params={
                    "sl_enabled": False,
                    "sl_pct": None,
                    "tp_enabled": False,
                    "tp_pct": None,
                },
                execution_params=self.execution_params,
            )
            yield BacktestStageABaseVariant(
                stage_a_index=stage_a_index,
                indicator_selections=indicator_selections,
                signal_params=signal_params,
                indicator_variant_key=indicator_variant_key,
                base_variant_key=base_variant_key,
            )


class BacktestGridBuilderV1:
    """
    Build deterministic backtest Stage A/Stage B grids and enforce sync guards.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - src/trading/platform/errors/roehub_error.py
    """

    def build(
        self,
        *,
        template: RunBacktestTemplate,
        candles: CandleArrays,
        indicator_compute: IndicatorCompute,
        preselect: int,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    ) -> BacktestGridBuildContextV1:
        """
        Build deterministic staged grid context with Stage A/Stage B guard checks.

        Args:
            template: Resolved backtest template payload.
            candles: Dense candles including warmup bars.
            indicator_compute: Indicator estimate port used for compute-axis materialization.
            preselect: Stage A shortlist size before Stage B expansion.
            defaults_provider: Optional defaults provider for compute/signal axis fallback.
            max_variants_per_compute: Stage guard for variants count.
            max_compute_bytes_total: Guard for estimated compute memory.
        Returns:
            BacktestGridBuildContextV1: Prepared deterministic staged grid context.
        Assumptions:
            `len(candles.ts_open)` already reflects warmup-aware internal load range.
        Raises:
            RoehubError: If stage variants or memory exceed configured sync guards.
            ValueError: If request payload contains invalid deterministic axis settings.
        Side Effects:
            Calls `indicator_compute.estimate(...)` once per indicator block.
        """
        if preselect <= 0:
            raise ValueError("BacktestGridBuilderV1 preselect must be > 0")
        if max_variants_per_compute <= 0:
            raise ValueError("max_variants_per_compute must be > 0")
        if max_compute_bytes_total <= 0:
            raise ValueError("max_compute_bytes_total must be > 0")

        indicator_plans = self._build_indicator_plans(
            template=template,
            indicator_compute=indicator_compute,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
        )
        signal_axes = self._build_signal_axes(
            template=template,
            defaults_provider=defaults_provider,
            indicator_plans=indicator_plans,
        )

        stage_a_variants_total = _product(
            values=tuple(plan.variants for plan in indicator_plans)
        ) * _product(values=tuple(len(axis.values) for axis in signal_axes))
        if stage_a_variants_total > max_variants_per_compute:
            raise _variants_guard_error(
                stage=STAGE_A_LITERAL,
                total_variants=stage_a_variants_total,
                max_variants_per_compute=max_variants_per_compute,
            )

        estimated_memory_bytes = _estimate_memory_bytes(
            bars=len(candles.ts_open),
            indicator_plans=indicator_plans,
        )
        if estimated_memory_bytes > max_compute_bytes_total:
            raise _memory_guard_error(
                stage=STAGE_A_LITERAL,
                estimated_memory_bytes=estimated_memory_bytes,
                max_compute_bytes_total=max_compute_bytes_total,
            )

        risk_variants = _risk_variants_from_template(template=template)
        shortlist_len = min(preselect, stage_a_variants_total)
        stage_b_variants_total = shortlist_len * len(risk_variants)
        if stage_b_variants_total > max_variants_per_compute:
            raise _variants_guard_error(
                stage=STAGE_B_LITERAL,
                total_variants=stage_b_variants_total,
                max_variants_per_compute=max_variants_per_compute,
            )

        return BacktestGridBuildContextV1(
            indicator_plans=indicator_plans,
            signal_axes=signal_axes,
            risk_variants=risk_variants,
            instrument_id_literal=_instrument_id_literal(template=template),
            timeframe_code=template.timeframe.code,
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            execution_params=template.execution_params or {},
            stage_a_variants_total=stage_a_variants_total,
            stage_b_variants_total=stage_b_variants_total,
            estimated_memory_bytes=estimated_memory_bytes,
            indicator_estimate_calls=len(indicator_plans),
        )

    def _build_indicator_plans(
        self,
        *,
        template: RunBacktestTemplate,
        indicator_compute: IndicatorCompute,
        defaults_provider: BacktestGridDefaultsProvider | None,
        max_variants_per_compute: int,
    ) -> tuple[_IndicatorPlan, ...]:
        """
        Build deterministic compute plans by estimating materialized axes per indicator.

        Args:
            template: Backtest template with request grids.
            indicator_compute: Indicator estimate port.
            defaults_provider: Optional provider for missing compute axes.
            max_variants_per_compute: Per-indicator estimate guard.
        Returns:
            tuple[_IndicatorPlan, ...]: Sorted indicator compute plans.
        Assumptions:
            Indicator ids in template are unique.
        Raises:
            ValueError: If template has duplicate indicator ids.
        Side Effects:
            Calls `indicator_compute.estimate(...)` for every indicator plan.
        """
        grids_by_id: dict[str, GridSpec] = {}
        for request_grid in sorted(
            template.indicator_grids,
            key=lambda item: item.indicator_id.value,
        ):
            indicator_id = request_grid.indicator_id.value
            if indicator_id in grids_by_id:
                raise ValueError(f"duplicate indicator_id in indicator_grids: {indicator_id}")
            defaults_grid: GridSpec | None = None
            if defaults_provider is not None:
                defaults_grid = defaults_provider.compute_defaults(indicator_id=indicator_id)
            grids_by_id[indicator_id] = _merge_grid_with_defaults(
                request_grid=request_grid,
                defaults_grid=defaults_grid,
            )

        plans: list[_IndicatorPlan] = []
        for indicator_id in sorted(grids_by_id.keys()):
            grid = grids_by_id[indicator_id]
            estimate = indicator_compute.estimate(
                grid,
                max_variants_guard=max_variants_per_compute,
            )
            plans.append(
                _indicator_plan_from_estimate(
                    indicator_id=indicator_id,
                    axes=estimate.axes,
                )
            )
        return tuple(plans)

    def _build_signal_axes(
        self,
        *,
        template: RunBacktestTemplate,
        defaults_provider: BacktestGridDefaultsProvider | None,
        indicator_plans: tuple[_IndicatorPlan, ...],
    ) -> tuple[_SignalAxisPlan, ...]:
        """
        Build deterministic signal-axis plans from request and optional defaults.

        Args:
            template: Backtest template with optional signal grids.
            defaults_provider: Optional provider for fallback signal defaults.
            indicator_plans: Indicator compute plans selected for Stage A.
        Returns:
            tuple[_SignalAxisPlan, ...]: Sorted materialized signal axes.
        Assumptions:
            Signal grids are stored under `indicator_id -> param_name -> GridParamSpec`.
        Raises:
            ValueError: If one signal axis materializes to an empty sequence.
        Side Effects:
            Reads optional defaults via `defaults_provider`.
        """
        request_signal_grids = template.signal_grids or {}
        axes: list[_SignalAxisPlan] = []
        for indicator_plan in indicator_plans:
            indicator_id = indicator_plan.indicator_id
            defaults_signal_map: Mapping[str, GridParamSpec] = {}
            if defaults_provider is not None:
                defaults_signal_map = defaults_provider.signal_param_defaults(
                    indicator_id=indicator_id
                )
            request_signal_map = request_signal_grids.get(indicator_id, {})
            merged_signal_map = dict(defaults_signal_map)
            merged_signal_map.update(request_signal_map)
            for param_name in sorted(merged_signal_map.keys(), key=lambda name: name.lower()):
                spec = merged_signal_map[param_name]
                values = _materialize_signal_values(spec=spec, axis_name=param_name)
                axes.append(
                    _SignalAxisPlan(
                        indicator_id=indicator_id,
                        param_name=param_name.strip().lower(),
                        values=values,
                    )
                )
        return tuple(sorted(axes, key=lambda axis: (axis.indicator_id, axis.param_name)))


def _indicator_plan_from_estimate(
    *,
    indicator_id: str,
    axes: tuple[AxisDef, ...],
) -> _IndicatorPlan:
    """
    Build internal indicator plan from `IndicatorCompute.estimate(...)` axis payload.

    Args:
        indicator_id: Indicator identifier.
        axes: Materialized axis definitions from estimate result.
    Returns:
        _IndicatorPlan: Deterministic plan with axis values and variant count.
    Assumptions:
        Axis value families are validated by `AxisDef` invariants.
    Raises:
        ValueError: If one axis has unsupported value family.
    Side Effects:
        None.
    """
    axis_plans: list[_IndicatorAxisPlan] = []
    variants = 1
    for axis in axes:
        values = _axis_values(axis=axis)
        axis_plans.append(_IndicatorAxisPlan(name=axis.name, values=values))
        variants = variants * len(values)
    return _IndicatorPlan(
        indicator_id=indicator_id,
        axes=tuple(axis_plans),
        variants=variants,
    )


def _axis_values(*, axis: AxisDef) -> tuple[int | float | str, ...]:
    """
    Convert one `AxisDef` into deterministic scalar tuple preserving axis ordering.

    Args:
        axis: Domain axis definition from indicators estimate.
    Returns:
        tuple[int | float | str, ...]: Deterministic scalar values.
    Assumptions:
        Exactly one value family is set in `AxisDef`.
    Raises:
        ValueError: If axis does not contain a supported value family.
    Side Effects:
        None.
    """
    if axis.values_enum is not None:
        return tuple(axis.values_enum)
    if axis.values_int is not None:
        return tuple(int(value) for value in axis.values_int)
    if axis.values_float is not None:
        return tuple(float(value) for value in axis.values_float)
    raise ValueError(f"AxisDef contains no values: {axis.name}")


def _signal_params_from_variant_index(
    *,
    signal_axes: tuple[_SignalAxisPlan, ...],
    variant_index: int,
) -> Mapping[str, Mapping[str, BacktestVariantScalar]]:
    """
    Build deterministic nested signal-params mapping for one signal mixed-radix index.

    Args:
        signal_axes: Sorted materialized signal axes.
        variant_index: Signal-space flat index.
    Returns:
        Mapping[str, Mapping[str, BacktestVariantScalar]]: Nested signal values map.
    Assumptions:
        Signal axes are sorted by `(indicator_id, param_name)`.
    Raises:
        ValueError: If index is outside valid mixed-radix bounds.
    Side Effects:
        None.
    """
    if len(signal_axes) == 0:
        return {}

    radices = tuple(len(axis.values) for axis in signal_axes)
    coordinates = _decode_mixed_radix(flat_index=variant_index, radices=radices)
    values_by_indicator: dict[str, dict[str, BacktestVariantScalar]] = {}
    for position, axis in enumerate(signal_axes):
        indicator_payload = values_by_indicator.setdefault(axis.indicator_id, {})
        indicator_payload[axis.param_name] = axis.values[coordinates[position]]

    normalized: dict[str, Mapping[str, BacktestVariantScalar]] = {}
    for indicator_id in sorted(values_by_indicator.keys()):
        payload = values_by_indicator[indicator_id]
        normalized[indicator_id] = MappingProxyType(
            {name: payload[name] for name in sorted(payload.keys())}
        )
    return MappingProxyType(normalized)


def _indicator_selection_from_variant_index(
    *,
    plan: _IndicatorPlan,
    variant_index: int,
) -> IndicatorVariantSelection:
    """
    Build explicit `IndicatorVariantSelection` for one indicator mixed-radix index.

    Args:
        plan: Indicator compute plan.
        variant_index: Flat variant index in indicator-local grid space.
    Returns:
        IndicatorVariantSelection: Explicit deterministic indicator selection.
    Assumptions:
        `plan.axes` ordering matches estimate axis materialization order.
    Raises:
        ValueError: If index is outside plan variant range.
    Side Effects:
        None.
    """
    coordinates = _decode_mixed_radix(
        flat_index=variant_index,
        radices=tuple(len(axis.values) for axis in plan.axes),
    )
    inputs: dict[str, int | float | str] = {}
    params: dict[str, int | float | str] = {}
    for position, axis in enumerate(plan.axes):
        value = axis.values[coordinates[position]]
        if axis.name == "source":
            inputs[axis.name] = value
            continue
        params[axis.name] = value
    return IndicatorVariantSelection(
        indicator_id=plan.indicator_id,
        inputs=inputs,
        params=params,
    )


def _materialize_signal_values(
    *,
    spec: GridParamSpec,
    axis_name: str,
) -> tuple[BacktestVariantScalar, ...]:
    """
    Materialize one signal-axis specification into deterministic scalar tuple.

    Args:
        spec: Signal axis specification.
        axis_name: Axis name for deterministic diagnostics.
    Returns:
        tuple[BacktestVariantScalar, ...]: Materialized signal values.
    Assumptions:
        Signal axis materialization uses the same explicit/range semantics as indicators.
    Raises:
        ValueError: If axis materialization fails or yields an empty sequence.
    Side Effects:
        None.
    """
    values = tuple(spec.materialize())
    if len(values) == 0:
        raise ValueError(f"signal axis '{axis_name}' materialized to empty values")
    return values


def _risk_variants_from_template(
    *,
    template: RunBacktestTemplate,
) -> tuple[BacktestRiskVariantV1, ...]:
    """
    Build deterministic Stage B risk variants from request risk grid or scalar fallback.

    Args:
        template: Backtest template payload.
    Returns:
        tuple[BacktestRiskVariantV1, ...]: Deterministic Stage B risk variants.
    Assumptions:
        Risk values represent percentages where `3.0 == 3%`.
    Raises:
        ValueError: If enabled risk axis is missing or contains non-numeric values.
    Side Effects:
        None.
    """
    risk_grid = template.risk_grid or BacktestRiskGridSpec()
    risk_params = template.risk_params or {}
    sl_enabled = risk_grid.sl_enabled
    tp_enabled = risk_grid.tp_enabled
    if "sl_enabled" in risk_params:
        sl_enabled = _bool_scalar(value=risk_params["sl_enabled"], field_name="sl_enabled")
    if "tp_enabled" in risk_params:
        tp_enabled = _bool_scalar(value=risk_params["tp_enabled"], field_name="tp_enabled")

    if sl_enabled:
        sl_values = _materialize_risk_axis(
            spec=risk_grid.sl,
            axis_name="sl",
            fallback_value=risk_params.get("sl_pct"),
        )
    else:
        sl_values = (None,)

    if tp_enabled:
        tp_values = _materialize_risk_axis(
            spec=risk_grid.tp,
            axis_name="tp",
            fallback_value=risk_params.get("tp_pct"),
        )
    else:
        tp_values = (None,)

    variants: list[BacktestRiskVariantV1] = []
    variant_index = 0
    for sl_pct in sl_values:
        for tp_pct in tp_values:
            variants.append(
                BacktestRiskVariantV1(
                    risk_index=variant_index,
                    risk_params={
                        "sl_enabled": sl_enabled,
                        "sl_pct": sl_pct,
                        "tp_enabled": tp_enabled,
                        "tp_pct": tp_pct,
                    },
                )
            )
            variant_index += 1
    return tuple(variants)


def _bool_scalar(*, value: BacktestVariantScalar, field_name: str) -> bool:
    """
    Validate one boolean scalar value from risk payload mappings.

    Args:
        value: Raw scalar value from risk payload.
        field_name: Field name for deterministic diagnostics.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Risk enable flags must be explicit booleans.
    Raises:
        ValueError: If value is not boolean.
    Side Effects:
        None.
    """
    if not isinstance(value, bool):
        raise ValueError(f"risk field '{field_name}' must be boolean")
    return value


def _materialize_risk_axis(
    *,
    spec: GridParamSpec | None,
    axis_name: str,
    fallback_value: BacktestVariantScalar,
) -> tuple[float, ...]:
    """
    Materialize enabled Stage B risk axis from grid spec or fallback scalar value.

    Args:
        spec: Optional risk grid spec.
        axis_name: Axis name (`sl` or `tp`) for diagnostics.
        fallback_value: Optional scalar fallback from `template.risk_params`.
    Returns:
        tuple[float, ...]: Materialized numeric percentages.
    Assumptions:
        Enabled risk axis requires at least one numeric value.
    Raises:
        ValueError: If axis is missing or contains non-numeric/empty payload.
    Side Effects:
        None.
    """
    if spec is None:
        if fallback_value is None:
            raise ValueError(f"risk axis '{axis_name}' must be provided when enabled")
        if isinstance(fallback_value, bool) or not isinstance(fallback_value, int | float):
            raise ValueError(f"risk axis '{axis_name}' fallback value must be numeric")
        return (float(fallback_value),)

    values = tuple(spec.materialize())
    if len(values) == 0:
        raise ValueError(f"risk axis '{axis_name}' materialized to empty values")

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"risk axis '{axis_name}' values must be numeric")
        normalized.append(float(value))
    return tuple(normalized)


def _estimate_memory_bytes(*, bars: int, indicator_plans: tuple[_IndicatorPlan, ...]) -> int:
    """
    Estimate total memory bytes for staged sync guard using indicators policy.

    Args:
        bars: Number of bars in dense warmup-inclusive timeline.
        indicator_plans: Materialized per-indicator compute plans.
    Returns:
        int: Estimated total memory bytes with reserve.
    Assumptions:
        Policy matches indicators estimator (`candles + tensors + reserve`).
    Raises:
        ValueError: If bars count is non-positive.
    Side Effects:
        None.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0 for memory estimate")

    bytes_candles = bars * _CANDLES_BYTES_PER_STEP
    bytes_indicators = 0
    for plan in indicator_plans:
        bytes_indicators += bars * plan.variants * _FLOAT32_BYTES

    reserve_base = bytes_candles + bytes_indicators
    reserve = max(_RESERVE_FIXED_BYTES, int(math.ceil(reserve_base * _RESERVE_FACTOR)))
    return reserve_base + reserve


def _decode_mixed_radix(*, flat_index: int, radices: tuple[int, ...]) -> tuple[int, ...]:
    """
    Decode mixed-radix coordinates from flat index without full cartesian materialization.

    Args:
        flat_index: Flat zero-based index in mixed-radix space.
        radices: Axis radices in deterministic order.
    Returns:
        tuple[int, ...]: Coordinate for each axis.
    Assumptions:
        Every radix is positive integer.
    Raises:
        ValueError: If index/radices are invalid or index is out of bounds.
    Side Effects:
        None.
    """
    if len(radices) == 0:
        if flat_index != 0:
            raise ValueError("flat_index must be 0 when radices are empty")
        return ()

    total = _product(values=radices)
    if flat_index < 0 or flat_index >= total:
        raise ValueError(f"mixed-radix index out of bounds: index={flat_index}, total={total}")

    remainder = flat_index
    coords_reversed: list[int] = []
    for radix in reversed(radices):
        if radix <= 0:
            raise ValueError("mixed-radix radices must be > 0")
        coords_reversed.append(remainder % radix)
        remainder = remainder // radix
    return tuple(reversed(coords_reversed))


def _product(*, values: tuple[int, ...]) -> int:
    """
    Compute deterministic product for integer tuple values.

    Args:
        values: Integer tuple values.
    Returns:
        int: Product value (`1` for empty tuple).
    Assumptions:
        Values are non-negative integers.
    Raises:
        ValueError: If one value is negative.
    Side Effects:
        None.
    """
    product = 1
    for value in values:
        if value < 0:
            raise ValueError("product values must be >= 0")
        product *= value
    return product


def _instrument_id_literal(*, template: RunBacktestTemplate) -> str:
    """
    Build canonical `<market_id>:<symbol>` instrument literal for variant-key builder.

    Args:
        template: Resolved backtest template.
    Returns:
        str: Canonical instrument literal.
    Assumptions:
        Instrument value-object fields are already validated.
    Raises:
        None.
    Side Effects:
        None.
    """
    return f"{template.instrument_id.market_id.value}:{template.instrument_id.symbol.value}"


def _merge_grid_with_defaults(
    *,
    request_grid: GridSpec,
    defaults_grid: GridSpec | None,
) -> GridSpec:
    """
    Merge request compute grid with defaults (`request overrides defaults`) deterministically.

    Args:
        request_grid: Request grid payload.
        defaults_grid: Optional defaults grid for the same indicator id.
    Returns:
        GridSpec: Merged grid specification.
    Assumptions:
        Both request and defaults grids target the same indicator id.
    Raises:
        ValueError: If defaults grid uses mismatched indicator id.
    Side Effects:
        None.
    """
    if defaults_grid is None:
        return request_grid
    if defaults_grid.indicator_id != request_grid.indicator_id:
        raise ValueError(
            "defaults indicator_id mismatch: "
            f"{defaults_grid.indicator_id.value} != {request_grid.indicator_id.value}"
        )

    merged_params = dict(defaults_grid.params)
    merged_params.update(request_grid.params)
    merged_source = request_grid.source if request_grid.source is not None else defaults_grid.source
    merged_layout = (
        request_grid.layout_preference
        if request_grid.layout_preference is not None
        else defaults_grid.layout_preference
    )
    return GridSpec(
        indicator_id=IndicatorId(request_grid.indicator_id.value),
        params=merged_params,
        source=merged_source,
        layout_preference=merged_layout,
    )


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> dict[str, BacktestVariantScalar]:
    """
    Normalize scalar mapping into deterministic key-sorted plain dictionary.

    Args:
        values: Scalar mapping payload.
    Returns:
        dict[str, BacktestVariantScalar]: Deterministic normalized mapping.
    Assumptions:
        Values are JSON-compatible scalars.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    normalized: dict[str, BacktestVariantScalar] = {}
    for raw_key in sorted(values.keys()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized


def _variants_guard_error(
    *,
    stage: str,
    total_variants: int,
    max_variants_per_compute: int,
) -> RoehubError:
    """
    Build canonical deterministic variants-guard `validation_error` payload.

    Args:
        stage: Stage literal (`stage_a` or `stage_b`).
        total_variants: Actual variants count at stage boundary.
        max_variants_per_compute: Configured variants limit.
    Returns:
        RoehubError: Canonical machine-readable 422 payload.
    Assumptions:
        Stage literal is deterministic and stable.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="validation_error",
        message="Backtest variants guard exceeded",
        details={
            "error": "max_variants_per_compute_exceeded",
            "stage": stage,
            "total_variants": total_variants,
            "max_variants_per_compute": max_variants_per_compute,
        },
    )


def _memory_guard_error(
    *,
    stage: str,
    estimated_memory_bytes: int,
    max_compute_bytes_total: int,
) -> RoehubError:
    """
    Build canonical deterministic memory-guard `validation_error` payload.

    Args:
        stage: Stage literal (`stage_a` for staged preflight in sync flow).
        estimated_memory_bytes: Estimated memory bytes by indicators policy.
        max_compute_bytes_total: Configured memory limit.
    Returns:
        RoehubError: Canonical machine-readable 422 payload.
    Assumptions:
        Memory estimate uses warmup-inclusive timeline length.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="validation_error",
        message="Backtest memory guard exceeded",
        details={
            "error": "max_compute_bytes_total_exceeded",
            "stage": stage,
            "estimated_memory_bytes": estimated_memory_bytes,
            "max_compute_bytes_total": max_compute_bytes_total,
        },
    )


__all__ = [
    "BacktestGridBuildContextV1",
    "BacktestGridBuilderV1",
    "BacktestRiskVariantV1",
    "BacktestStageABaseVariant",
    "STAGE_A_LITERAL",
    "STAGE_B_LITERAL",
]
