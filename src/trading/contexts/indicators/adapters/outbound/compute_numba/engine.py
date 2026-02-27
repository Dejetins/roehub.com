"""
CPU/Numba implementation of indicators compute application port.

Docs:
  - docs/architecture/indicators/indicators-overview.md
  - docs/architecture/indicators/indicators-compute-engine-core.md
  - docs/architecture/indicators/README.md
Related:
  - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
  - src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py
  - src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py
"""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Mapping, TypeVar, cast

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    PRECISION_MODE_FLOAT32,
    PRECISION_MODE_FLOAT64,
    PRECISION_MODE_MIXED,
    SUPPORTED_PRECISION_MODES,
    WORKSPACE_FACTOR_DEFAULT,
    WORKSPACE_FIXED_BYTES_DEFAULT,
    check_total_budget_or_raise,
    compute_ma_grid_f32,
    compute_momentum_grid_f32,
    compute_structure_grid_f32,
    compute_trend_grid_f32,
    compute_volatility_grid_f32,
    compute_volume_grid_f32,
    estimate_tensor_bytes,
    estimate_total_bytes,
    is_supported_ma_indicator,
    is_supported_momentum_indicator,
    is_supported_structure_indicator,
    is_supported_trend_indicator,
    is_supported_volatility_indicator,
    is_supported_volume_indicator,
    write_series_grid_time_major,
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

_STRUCTURE_IDS_REQUIRING_OHLC = {
    "structure.candle_body",
    "structure.candle_body_atr",
    "structure.candle_body_pct",
    "structure.candle_lower_wick",
    "structure.candle_lower_wick_atr",
    "structure.candle_lower_wick_pct",
    "structure.candle_range",
    "structure.candle_range_atr",
    "structure.candle_stats",
    "structure.candle_stats_atr_norm",
    "structure.candle_upper_wick",
    "structure.candle_upper_wick_atr",
    "structure.candle_upper_wick_pct",
}
_STRUCTURE_IDS_REQUIRING_HL = {
    "structure.pivot_high",
    "structure.pivot_low",
    "structure.pivots",
}
_STRUCTURE_IDS_REQUIRING_HLC = {
    "structure.distance_to_ma_norm",
}

_AxisValue = TypeVar("_AxisValue", int, float, str)
_TIER_A_LABEL = "Tier A"
_TIER_B_LABEL = "Tier B"
_TIER_C_LABEL = "Tier C"
_TIER_A_FLOAT32_IDS = frozenset(
    (
        "ma.dema",
        "ma.ema",
        "ma.hma",
        "ma.lwma",
        "ma.rma",
        "ma.sma",
        "ma.smma",
        "ma.tema",
        "ma.wma",
        "ma.zlema",
        "structure.candle_body",
        "structure.candle_body_atr",
        "structure.candle_body_pct",
        "structure.candle_lower_wick",
        "structure.candle_lower_wick_atr",
        "structure.candle_lower_wick_pct",
        "structure.candle_range",
        "structure.candle_range_atr",
        "structure.candle_stats",
        "structure.candle_stats_atr_norm",
        "structure.candle_upper_wick",
        "structure.candle_upper_wick_atr",
        "structure.candle_upper_wick_pct",
        "structure.pivot_high",
        "structure.pivot_low",
        "structure.pivots",
        "trend.aroon",
        "trend.donchian",
    )
)
_TIER_B_MIXED_IDS = frozenset(
    (
        "ma.vwma",
        "momentum.cci",
        "momentum.macd",
        "momentum.ppo",
        "momentum.roc",
        "momentum.rsi",
        "momentum.stoch",
        "momentum.stoch_rsi",
        "momentum.trix",
        "momentum.williams_r",
        "structure.distance_to_ma_norm",
        "structure.percent_rank",
        "structure.zscore",
        "trend.adx",
        "trend.keltner",
        "trend.psar",
        "trend.supertrend",
        "trend.vortex",
        "volume.vwap",
        "volume.vwap_deviation",
    )
)
_TIER_C_FLOAT64_IDS = frozenset(
    (
        "trend.linreg_slope",
        "volatility.bbands",
        "volatility.bbands_bandwidth",
        "volatility.bbands_percent_b",
        "volatility.hv",
        "volatility.stddev",
        "volatility.variance",
        "volume.ad_line",
        "volume.cmf",
        "volume.mfi",
        "volume.obv",
    )
)


class NumbaIndicatorCompute(IndicatorCompute):
    """
    Indicator compute engine backed by CPU+Numba kernels.

    Docs:
      - docs/architecture/indicators/indicators-overview.md
      - docs/architecture/indicators/indicators-compute-engine-core.md
      - docs/architecture/indicators/README.md
    Related:
      - src/trading/contexts/indicators/application/dto/indicator_tensor.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py
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

        Docs:
          - docs/architecture/indicators/indicators-overview.md
          - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
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

        Docs:
          - docs/architecture/indicators/indicators-overview.md
          - docs/architecture/indicators/indicators-ma.md
          - docs/architecture/indicators/indicators-volatility.md
          - docs/architecture/indicators/indicators-momentum.md
          - docs/architecture/indicators/indicators-trend.md
          - docs/architecture/indicators/indicators-volume.md
          - docs/architecture/indicators/indicators-structure.md
          - docs/runbooks/indicators-why-nan.md
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

        required_sources = _required_sources_for_request(definition=definition, axes=axes)
        series_map = _build_series_map(
            candles=req.candles,
            required_sources=required_sources,
        )
        precision = _precision_mode_for_indicator(indicator_id=definition.indicator_id.value)
        if is_supported_ma_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_ma_variant_source_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                precision=precision,
            )
        elif is_supported_volatility_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_volatility_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
                precision=precision,
            )
        elif is_supported_momentum_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_momentum_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
                precision=precision,
            )
        elif is_supported_trend_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_trend_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
                precision=precision,
            )
        elif is_supported_volume_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_volume_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                precision=precision,
            )
        elif is_supported_structure_indicator(indicator_id=definition.indicator_id.value):
            variant_series_matrix = _compute_structure_variant_matrix(
                definition=definition,
                axes=axes,
                available_series=series_map,
                t_size=t_size,
                precision=precision,
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
            values = _prepare_variant_major_values(
                variant_series_matrix=variant_series_matrix,
                variants=variants,
                t_size=t_size,
            )

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

        Docs:
          - docs/architecture/indicators/indicators-overview.md
          - docs/architecture/indicators/indicators-compute-engine-core.md
          - docs/runbooks/indicators-numba-warmup-jit.md
          - docs/runbooks/indicators-numba-cache-and-threads.md
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


def _prepare_variant_major_values(
    *,
    variant_series_matrix: np.ndarray,
    variants: int,
    t_size: int,
) -> np.ndarray:
    """
    Validate and normalize variant-major output matrix for tensor contract.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related:
      src/trading/contexts/indicators/application/dto/indicator_tensor.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      apps/api/routes/indicators.py

    Args:
        variant_series_matrix: Candidate variant-major matrix from compute path.
        variants: Expected variant count `V`.
        t_size: Expected timeline size `T`.
    Returns:
        np.ndarray: Float32 C-contiguous matrix with shape `(V, T)`.
    Assumptions:
        Matrix shape is deterministic after grid materialization.
    Raises:
        GridValidationError: If matrix shape is incompatible with `(V, T)`.
    Side Effects:
        May allocate one normalized copy when dtype/order contract is not met.
    """
    expected_shape = (variants, t_size)
    if variant_series_matrix.shape != expected_shape:
        raise GridValidationError(
            "variant-major matrix shape mismatch: "
            f"expected={expected_shape}, got={variant_series_matrix.shape}"
        )

    if (
        variant_series_matrix.dtype == np.float32
        and variant_series_matrix.flags.c_contiguous
    ):
        return variant_series_matrix

    return np.ascontiguousarray(variant_series_matrix, dtype=np.float32)


def _precision_mode_for_indicator(*, indicator_id: str) -> str:
    """
    Resolve deterministic kernel precision mode by indicator id.

    Docs: docs/architecture/indicators/indicators-kernels-f32-migration-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Normalized indicator identifier.
    Returns:
        str: Precision mode (`float32`, `mixed`, or `float64`).
    Assumptions:
        Unknown ids must deterministically fallback to `float64` core path.
    Raises:
        GridValidationError: If computed mode is outside supported precision constants.
    Side Effects:
        None.
    """
    normalized_id = indicator_id.strip().lower()
    if normalized_id in _TIER_A_FLOAT32_IDS:
        precision = PRECISION_MODE_FLOAT32
    elif normalized_id in _TIER_B_MIXED_IDS:
        precision = PRECISION_MODE_MIXED
    elif normalized_id in _TIER_C_FLOAT64_IDS:
        precision = PRECISION_MODE_FLOAT64
    else:
        precision = PRECISION_MODE_FLOAT64
    if precision not in SUPPORTED_PRECISION_MODES:
        raise GridValidationError(
            "unsupported precision mode: "
            f"indicator_id={normalized_id!r}, precision={precision!r}"
        )
    return precision


def _precision_tier_label(*, precision: str) -> str:
    """
    Convert precision mode into explicit migration tier label.

    Docs: docs/architecture/indicators/indicators-kernels-f32-migration-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      docs/architecture/indicators/indicators-compute-engine-core.md

    Args:
        precision: Kernel precision mode (`float32`, `mixed`, `float64`).
    Returns:
        str: Human-readable tier label (`Tier A`, `Tier B`, `Tier C`).
    Assumptions:
        Mode comes from `_precision_mode_for_indicator`.
    Raises:
        GridValidationError: If precision mode is unknown.
    Side Effects:
        None.
    """
    if precision == PRECISION_MODE_FLOAT32:
        return _TIER_A_LABEL
    if precision == PRECISION_MODE_MIXED:
        return _TIER_B_LABEL
    if precision == PRECISION_MODE_FLOAT64:
        return _TIER_C_LABEL
    raise GridValidationError(f"unsupported precision mode: {precision!r}")


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


def _required_sources_for_request(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
) -> tuple[str, ...]:
    """
    Resolve minimal source set required for one compute request.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/indicator_def.py,
      src/trading/contexts/indicators/domain/entities/input_series.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized axes preserving request value order.
    Returns:
        tuple[str, ...]: Sorted unique source names required by this request.
    Assumptions:
        Source axis values define dynamic source requirements when axis exists.
    Raises:
        GridValidationError: If source axis exists but is not enum-based.
    Side Effects:
        None.
    """
    axis_names = [axis.name for axis in axes]
    required: set[str] = set()
    dynamic_price_sources = {
        InputSeries.OPEN.value,
        InputSeries.HIGH.value,
        InputSeries.LOW.value,
        InputSeries.CLOSE.value,
        InputSeries.HL2.value,
        InputSeries.HLC3.value,
        InputSeries.OHLC4.value,
    }

    if "source" in axis_names:
        source_axis = axes[axis_names.index("source")]
        source_values = source_axis.values_enum
        if source_values is None:
            raise GridValidationError("source axis must have values_enum")
        for value in source_values:
            normalized_value = str(value).strip().lower()
            if not normalized_value:
                raise GridValidationError("source axis values must be non-empty")
            required.add(normalized_value)
        for input_series in definition.inputs:
            source_name = input_series.value
            if source_name not in dynamic_price_sources:
                required.add(source_name)
        required.update(
            _required_fixed_series_for_indicator_id(
                indicator_id=definition.indicator_id.value
            )
        )
    else:
        required.update(series.value for series in definition.inputs)
        if len(required) == 0:
            required.add(_fallback_source(definition=definition))

    return tuple(sorted(required))


def _expanded_required_sources_for_derivatives(
    *,
    required_sources: set[str],
) -> set[str]:
    """
    Expand required source set with base OHLC dependencies for derived source requests.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/entities/input_series.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      docs/architecture/indicators/indicators_formula.yaml

    Args:
        required_sources: Request-level required sources.
    Returns:
        set[str]: Expanded source set including derived-series dependencies.
    Assumptions:
        Derived series are `hl2`, `hlc3`, and `ohlc4`.
    Raises:
        None.
    Side Effects:
        None.
    """
    expanded = set(required_sources)
    if InputSeries.HL2.value in required_sources:
        expanded.update((InputSeries.HIGH.value, InputSeries.LOW.value))
    if InputSeries.HLC3.value in required_sources:
        expanded.update(
            (
                InputSeries.HIGH.value,
                InputSeries.LOW.value,
                InputSeries.CLOSE.value,
            )
        )
    if InputSeries.OHLC4.value in required_sources:
        expanded.update(
            (
                InputSeries.OPEN.value,
                InputSeries.HIGH.value,
                InputSeries.LOW.value,
                InputSeries.CLOSE.value,
            )
        )
    return expanded


def _required_fixed_series_for_indicator_id(*, indicator_id: str) -> tuple[str, ...]:
    """
    Resolve fixed OHLC requirements that remain mandatory even when source axis is present.

    Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py,
      src/trading/contexts/indicators/domain/definitions/structure.py

    Args:
        indicator_id: Normalized indicator identifier.
    Returns:
        tuple[str, ...]: Fixed mandatory source names for the indicator.
    Assumptions:
        Only structure wrappers currently require fixed OHLC in addition to source axis.
    Raises:
        None.
    Side Effects:
        None.
    """
    normalized_id = indicator_id.strip().lower()
    if normalized_id in _STRUCTURE_IDS_REQUIRING_OHLC:
        return (
            InputSeries.OPEN.value,
            InputSeries.HIGH.value,
            InputSeries.LOW.value,
            InputSeries.CLOSE.value,
        )
    if normalized_id in _STRUCTURE_IDS_REQUIRING_HLC:
        return (
            InputSeries.HIGH.value,
            InputSeries.LOW.value,
            InputSeries.CLOSE.value,
        )
    if normalized_id in _STRUCTURE_IDS_REQUIRING_HL:
        return (InputSeries.HIGH.value, InputSeries.LOW.value)
    return ()


def _build_series_map(
    *,
    candles: CandleArrays,
    required_sources: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    """
    Build deterministic source-series map and lazily allocate only request-required series.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/application/dto/candle_arrays.py,
      src/trading/contexts/indicators/domain/entities/input_series.py

    Args:
        candles: Dense candle arrays payload.
        required_sources: Source labels required for current compute request.
    Returns:
        Mapping[str, np.ndarray]: Source name to contiguous float32 array.
    Assumptions:
        `required_sources` already reflects request-level source axis and fixed indicator inputs.
    Raises:
        None.
    Side Effects:
        Allocates only required base/derived arrays for current request.
    """
    required = set(required_sources)
    expanded_required = _expanded_required_sources_for_derivatives(required_sources=required)

    open_series: np.ndarray | None = None
    high_series: np.ndarray | None = None
    low_series: np.ndarray | None = None
    close_series: np.ndarray | None = None
    volume_series: np.ndarray | None = None

    if InputSeries.OPEN.value in expanded_required:
        open_series = np.ascontiguousarray(candles.open, dtype=np.float32)
    if InputSeries.HIGH.value in expanded_required:
        high_series = np.ascontiguousarray(candles.high, dtype=np.float32)
    if InputSeries.LOW.value in expanded_required:
        low_series = np.ascontiguousarray(candles.low, dtype=np.float32)
    if InputSeries.CLOSE.value in expanded_required:
        close_series = np.ascontiguousarray(candles.close, dtype=np.float32)
    if InputSeries.VOLUME.value in expanded_required:
        volume_series = np.ascontiguousarray(candles.volume, dtype=np.float32)

    series_map: dict[str, np.ndarray] = {}
    if InputSeries.OPEN.value in required and open_series is not None:
        series_map[InputSeries.OPEN.value] = open_series
    if InputSeries.HIGH.value in required and high_series is not None:
        series_map[InputSeries.HIGH.value] = high_series
    if InputSeries.LOW.value in required and low_series is not None:
        series_map[InputSeries.LOW.value] = low_series
    if InputSeries.CLOSE.value in required and close_series is not None:
        series_map[InputSeries.CLOSE.value] = close_series
    if InputSeries.VOLUME.value in required and volume_series is not None:
        series_map[InputSeries.VOLUME.value] = volume_series

    if InputSeries.HL2.value in required and high_series is not None and low_series is not None:
        series_map[InputSeries.HL2.value] = np.ascontiguousarray(
            (high_series + low_series) / np.float32(2.0)
        )
    if (
        InputSeries.HLC3.value in required
        and high_series is not None
        and low_series is not None
        and close_series is not None
    ):
        series_map[InputSeries.HLC3.value] = np.ascontiguousarray(
            (high_series + low_series + close_series) / np.float32(3.0)
        )
    if (
        InputSeries.OHLC4.value in required
        and open_series is not None
        and high_series is not None
        and low_series is not None
        and close_series is not None
    ):
        series_map[InputSeries.OHLC4.value] = np.ascontiguousarray(
            (open_series + high_series + low_series + close_series) / np.float32(4.0)
        )
    return series_map


def _compute_ma_variant_source_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    precision: str,
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
        precision: Precision mode resolved by engine dispatch policy.
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
    if definition.indicator_id.value == "ma.vwma" and volume_series is None:
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
                precision=precision,
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
    precision: str,
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
        precision: Precision mode resolved by engine dispatch policy.
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
                precision=precision,
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
                precision=precision,
            )

        variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
        _validate_required_series_available(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
        )
        source_groups = _group_variant_indexes_by_source(
            variant_source_labels=variant_source_labels
        )
        variants = len(variant_source_labels)
        out = np.empty((variants, t_size), dtype=np.float32, order="C")

        if indicator_id in {"volatility.stddev", "volatility.variance"}:
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_volatility_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_volatility_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    annualizations=_slice_variant_vector(
                        values=annualizations_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_volatility_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    mults=_slice_variant_vector(
                        values=mults_f64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported volatility indicator_id: {indicator_id}")


def _compute_momentum_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
    precision: str,
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
        precision: Precision mode resolved by engine dispatch policy.
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
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
            )
            source_groups = _group_variant_indexes_by_source(
                variant_source_labels=variant_source_labels
            )
            variants = len(variant_source_labels)
            out = np.empty((variants, t_size), dtype=np.float32, order="C")
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_momentum_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
            )

        variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
        _validate_required_series_available(
            variant_source_labels=variant_source_labels,
            available_series=available_series,
        )
        source_groups = _group_variant_indexes_by_source(
            variant_source_labels=variant_source_labels
        )
        variants = len(variant_source_labels)
        out = np.empty((variants, t_size), dtype=np.float32, order="C")

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
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_momentum_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    rsi_windows=_slice_variant_vector(
                        values=rsi_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    k_windows=_slice_variant_vector(
                        values=k_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    smoothings=_slice_variant_vector(
                        values=smoothings_i64,
                        variant_indexes=variant_indexes,
                    ),
                    d_windows=_slice_variant_vector(
                        values=d_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_momentum_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    signal_windows=_slice_variant_vector(
                        values=signal_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_momentum_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    fast_windows=_slice_variant_vector(
                        values=fast_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    slow_windows=_slice_variant_vector(
                        values=slow_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    signal_windows=_slice_variant_vector(
                        values=signal_windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported momentum indicator_id: {indicator_id}")


def _compute_trend_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
    precision: str,
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
        precision: Precision mode resolved by engine dispatch policy.
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
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
            )
            source_groups = _group_variant_indexes_by_source(
                variant_source_labels=variant_source_labels
            )
            variants = len(variant_source_labels)
            out = np.empty((variants, t_size), dtype=np.float32, order="C")
            for source_name, variant_indexes in source_groups:
                source_variants = _build_group_source_matrix(
                    source=_require_series(
                        available_series=available_series,
                        name=source_name,
                    ),
                    variants=int(variant_indexes.shape[0]),
                    t_size=t_size,
                )
                group_values = compute_trend_grid_f32(
                    indicator_id=indicator_id,
                    source_variants=source_variants,
                    windows=_slice_variant_vector(
                        values=windows_i64,
                        variant_indexes=variant_indexes,
                    ),
                    precision=precision,
                )
                _scatter_group_values(
                    destination=out,
                    variant_indexes=variant_indexes,
                    group_values=group_values,
                    t_size=t_size,
                )
            return out

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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported trend indicator_id: {indicator_id}")


def _compute_volume_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    precision: str,
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
        precision: Precision mode resolved by engine dispatch policy.
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
                precision=precision,
            )

        if indicator_id == "volume.volume_sma":
            windows_i64 = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
            return compute_volume_grid_f32(
                indicator_id=indicator_id,
                volume=volume,
                windows=windows_i64,
                precision=precision,
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
                precision=precision,
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
                precision=precision,
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
                precision=precision,
            )
    except ValueError as error:
        raise GridValidationError(str(error)) from error

    raise GridValidationError(f"unsupported volume indicator_id: {indicator_id}")


def _compute_structure_variant_matrix(
    *,
    definition: IndicatorDef,
    axes: tuple[AxisDef, ...],
    available_series: Mapping[str, np.ndarray],
    t_size: int,
    precision: str,
) -> np.ndarray:
    """
    Compute variant-major matrix `(V, T)` for structure-family indicators.

    Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py,
      src/trading/contexts/indicators/domain/definitions/structure.py

    Args:
        definition: Hard indicator definition for the request.
        axes: Materialized domain axes preserving request order.
        available_series: Mapping of available source arrays.
        t_size: Time dimension length.
        precision: Precision mode resolved by engine dispatch policy.
    Returns:
        np.ndarray: Float32 C-contiguous variant-major matrix `(V, T)`.
    Assumptions:
        Structure wrappers expose one v1 output and keep deterministic variant indexing.
    Raises:
        GridValidationError: If required axes are missing or malformed.
        MissingRequiredSeries: If required OHLC/source series are unavailable.
    Side Effects:
        Allocates source/parameter vectors and one output matrix.
    """
    indicator_id = definition.indicator_id.value

    try:
        axis_names = [axis.name for axis in axes]
        kernel_kwargs: dict[str, np.ndarray] = {}

        variant_source_labels: tuple[str, ...] | None = None
        if "source" in axis_names:
            variant_source_labels = _variant_source_labels(definition=definition, axes=axes)
            _validate_required_series_available(
                variant_source_labels=variant_source_labels,
                available_series=available_series,
            )

        if "window" in axis_names:
            kernel_kwargs["windows"] = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="window"), dtype=np.int64)
            )
        if "atr_window" in axis_names:
            kernel_kwargs["atr_windows"] = np.ascontiguousarray(
                np.asarray(
                    _variant_int_values(axes=axes, axis_name="atr_window"),
                    dtype=np.int64,
                )
            )
        if "left" in axis_names:
            kernel_kwargs["lefts"] = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="left"), dtype=np.int64)
            )
        if "right" in axis_names:
            kernel_kwargs["rights"] = np.ascontiguousarray(
                np.asarray(_variant_int_values(axes=axes, axis_name="right"), dtype=np.int64)
            )

        requires_ohlc = indicator_id in _STRUCTURE_IDS_REQUIRING_OHLC
        requires_hlc = indicator_id in _STRUCTURE_IDS_REQUIRING_HLC
        requires_hl = indicator_id in _STRUCTURE_IDS_REQUIRING_HL

        if requires_ohlc:
            kernel_kwargs["open"] = _require_series(
                available_series=available_series,
                name=InputSeries.OPEN.value,
            )
        if requires_ohlc or requires_hlc or requires_hl:
            kernel_kwargs["high"] = _require_series(
                available_series=available_series,
                name=InputSeries.HIGH.value,
            )
        if requires_ohlc or requires_hlc or requires_hl:
            kernel_kwargs["low"] = _require_series(
                available_series=available_series,
                name=InputSeries.LOW.value,
            )
        if requires_ohlc or requires_hlc:
            kernel_kwargs["close"] = _require_series(
                available_series=available_series,
                name=InputSeries.CLOSE.value,
            )

        if variant_source_labels is None:
            return compute_structure_grid_f32(
                indicator_id=indicator_id,
                precision=precision,
                **kernel_kwargs,
            )

        source_groups = _group_variant_indexes_by_source(
            variant_source_labels=variant_source_labels
        )
        variants = len(variant_source_labels)
        out = np.empty((variants, t_size), dtype=np.float32, order="C")
        group_slice_keys = frozenset(("windows", "atr_windows", "lefts", "rights"))

        for source_name, variant_indexes in source_groups:
            group_kwargs: dict[str, np.ndarray] = {}
            for key, values in kernel_kwargs.items():
                if key in group_slice_keys:
                    group_kwargs[key] = _slice_variant_vector(
                        values=values,
                        variant_indexes=variant_indexes,
                    )
                else:
                    group_kwargs[key] = values

            group_kwargs["source_variants"] = _build_group_source_matrix(
                source=_require_series(
                    available_series=available_series,
                    name=source_name,
                ),
                variants=int(variant_indexes.shape[0]),
                t_size=t_size,
            )
            group_values = compute_structure_grid_f32(
                indicator_id=indicator_id,
                precision=precision,
                **group_kwargs,
            )
            _scatter_group_values(
                destination=out,
                variant_indexes=variant_indexes,
                group_values=group_values,
                t_size=t_size,
            )

        return out
    except ValueError as error:
        raise GridValidationError(str(error)) from error


def _expand_axis_values_repeat_tile(
    *,
    axis_values: tuple[_AxisValue, ...],
    axis_lengths: tuple[int, ...],
    axis_index: int,
) -> tuple[_AxisValue, ...]:
    """
    Expand one materialized axis into per-variant values using repeat/tile cartesian layout.

    Docs: docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      src/trading/contexts/indicators/application/services/grid_builder.py,
      src/trading/contexts/indicators/domain/entities/axis_def.py

    Args:
        axis_values: Materialized values of one axis preserving request order.
        axis_lengths: Length of each axis in definition order.
        axis_index: Index of axis being expanded.
    Returns:
        tuple[_AxisValue, ...]: Expanded values aligned with variant index order.
    Assumptions:
        Cartesian variant order follows axis definition order with the last axis changing fastest.
    Raises:
        None.
    Side Effects:
        None.
    """
    repeat_factor = 1
    for length in axis_lengths[axis_index + 1 :]:
        repeat_factor = repeat_factor * length

    tile_factor = 1
    for length in axis_lengths[:axis_index]:
        tile_factor = tile_factor * length

    expanded = np.tile(
        np.repeat(np.asarray(axis_values), repeat_factor),
        tile_factor,
    )
    return cast(tuple[_AxisValue, ...], tuple(expanded))


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
    return tuple(
        _expand_axis_values_repeat_tile(
            axis_values=axis.values_int,
            axis_lengths=axis_lengths,
            axis_index=axis_index,
        )
    )


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
    return tuple(
        _expand_axis_values_repeat_tile(
            axis_values=axis.values_float,
            axis_lengths=axis_lengths,
            axis_index=axis_index,
        )
    )


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
    return tuple(
        _expand_axis_values_repeat_tile(
            axis_values=window_axis.values_int,
            axis_lengths=axis_lengths,
            axis_index=window_axis_index,
        )
    )


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
    return tuple(
        _expand_axis_values_repeat_tile(
            axis_values=source_values,
            axis_lengths=axis_lengths,
            axis_index=source_axis_index,
        )
    )


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


def _group_variant_indexes_by_source(
    *,
    variant_source_labels: tuple[str, ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    """
    Build deterministic source groups with global variant indexes.

    Docs: docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py

    Args:
        variant_source_labels: Source label assigned to each global variant index.
    Returns:
        tuple[tuple[str, np.ndarray], ...]: Source groups preserving first-seen source order.
    Assumptions:
        Variant labels already reflect deterministic cartesian axis ordering.
    Raises:
        None.
    Side Effects:
        Allocates per-source int64 index vectors.
    """
    grouped: dict[str, list[int]] = {}
    for variant_index, source_name in enumerate(variant_source_labels):
        grouped.setdefault(source_name, []).append(variant_index)

    groups: list[tuple[str, np.ndarray]] = []
    for source_name, indexes in grouped.items():
        groups.append(
            (
                source_name,
                np.ascontiguousarray(np.asarray(indexes, dtype=np.int64)),
            )
        )
    return tuple(groups)


def _build_group_source_matrix(
    *,
    source: np.ndarray,
    variants: int,
    t_size: int,
) -> np.ndarray:
    """
    Build contiguous `(group_variants, T)` matrix from one source series for grouped compute.

    Docs: docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py

    Args:
        source: Source series for one source label.
        variants: Number of variants in the current source group.
        t_size: Expected time dimension.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(group_variants, T)`.
    Assumptions:
        Source series is aligned to request candles timeline.
    Raises:
        ValueError: If source length does not match `t_size`.
    Side Effects:
        Allocates one grouped source matrix.
    """
    source_row = np.ascontiguousarray(source, dtype=np.float32).reshape(1, t_size)
    return np.ascontiguousarray(
        np.repeat(source_row, repeats=variants, axis=0),
        dtype=np.float32,
    )


def _slice_variant_vector(
    *,
    values: np.ndarray,
    variant_indexes: np.ndarray,
) -> np.ndarray:
    """
    Slice one per-variant parameter vector by grouped global indexes.

    Docs: docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py

    Args:
        values: Per-variant parameter vector in global deterministic order.
        variant_indexes: Group-local selection of global variant indexes.
    Returns:
        np.ndarray: Contiguous parameter vector aligned with grouped source matrix rows.
    Assumptions:
        Index vector is int64 and deterministic for the source group.
    Raises:
        IndexError: If group indexes are outside vector bounds.
    Side Effects:
        Allocates one contiguous slice vector.
    """
    return np.ascontiguousarray(values[variant_indexes])


def _scatter_group_values(
    *,
    destination: np.ndarray,
    variant_indexes: np.ndarray,
    group_values: np.ndarray,
    t_size: int,
) -> None:
    """
    Scatter grouped output rows back into global variant-major output matrix.

    Docs: docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/application/dto/indicator_tensor.py,
      docs/architecture/indicators/indicators-compute-engine-core.md

    Args:
        destination: Global output matrix `(V, T)` to fill.
        variant_indexes: Global variant indexes for grouped rows.
        group_values: Group output matrix `(group_variants, T)`.
        t_size: Expected time dimension.
    Returns:
        None.
    Assumptions:
        `variant_indexes` and grouped compute inputs are aligned by source group.
    Raises:
        GridValidationError: If grouped result shape is incompatible with target scatter.
    Side Effects:
        Mutates destination matrix in-place.
    """
    expected_shape = (int(variant_indexes.shape[0]), t_size)
    if group_values.shape != expected_shape:
        raise GridValidationError(
            "grouped matrix shape mismatch: "
            f"expected={expected_shape}, got={group_values.shape}"
        )
    destination[variant_indexes, :] = group_values


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
