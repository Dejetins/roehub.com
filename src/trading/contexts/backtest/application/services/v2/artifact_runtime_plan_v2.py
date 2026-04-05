"""Artifact-backed runtime planning for sync, worker, and lazy-detail cutover."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator, Literal, Mapping

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

from .adaptive_selector_v2 import (
    AdaptiveExecutionSelectorV2,
    AdaptiveSelectorDecisionV2,
    AdaptiveSelectorPlanningEvidenceV2,
    AdaptiveSelectorPolicyV2,
    CostModelAdaptiveExecutionSelectorV2,
    default_adaptive_selector_policy_v2,
)
from .execution_profile_v2 import (
    ExecutionProfileModeLiteralV2,
    ExecutionProfilesCatalogV2,
    ExecutionProfileV2,
    default_execution_profiles_catalog_v2,
    execution_profile_supports_requested_runtime_v2,
    execution_profile_uses_hierarchical_shortlist_runtime_v2,
)

STAGE_A_LITERAL_V2 = "stage_a"
STAGE_B_LITERAL_V2 = "stage_b"
PlannerLaunchBudgetModeV2 = Literal["ignore", "sync_inline"]

_FLOAT32_BYTES = 4
_CANDLES_BYTES_PER_STEP = (5 * _FLOAT32_BYTES) + 8
_RESERVE_FACTOR = 0.20
_RESERVE_FIXED_BYTES = 64 * 1024**2
_STAGE_A_DISABLED_RISK_PARAMS_V2: Mapping[str, BacktestVariantScalar] = MappingProxyType(
    {
        "sl_enabled": False,
        "sl_pct": None,
        "tp_enabled": False,
        "tp_pct": None,
    }
)


@dataclass(frozen=True, slots=True)
class BacktestIndicatorAxisPlanV2:
    """
    Deterministic compute-axis plan for one artifact-backed indicator.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    name: str
    values: tuple[int | float | str, ...]


@dataclass(frozen=True, slots=True)
class BacktestIndicatorPlanV2:
    """
    Deterministic indicator plan used by artifact-backed runtime row addressing.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    indicator_id: str
    axes: tuple[BacktestIndicatorAxisPlanV2, ...]
    variants: int


@dataclass(frozen=True, slots=True)
class BacktestSignalAxisPlanV2:
    """
    Deterministic signal-axis plan for artifact-backed Stage A enumeration.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - configs/prod/indicators.yaml
    """

    indicator_id: str
    param_name: str
    values: tuple[BacktestVariantScalar, ...]


@dataclass(frozen=True, slots=True)
class BacktestRiskVariantV2:
    """
    One deterministic Stage B risk variant for artifact-backed runtime.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
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
            Risk payload contains only stable backtest-owned scalar keys.
        Raises:
            ValueError: If the index is negative or one key is blank.
        Side Effects:
            Replaces the mapping with an immutable key-sorted proxy.
        """
        if self.risk_index < 0:
            raise ValueError("BacktestRiskVariantV2.risk_index must be >= 0")

        normalized: dict[str, BacktestVariantScalar] = {}
        for raw_key in sorted(self.risk_params.keys()):
            key = str(raw_key).strip()
            if not key:
                raise ValueError("BacktestRiskVariantV2 risk_params keys must be non-empty")
            normalized[key] = self.risk_params[raw_key]
        object.__setattr__(self, "risk_params", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class BacktestStageABaseVariantV2:
    """
    Deterministic Stage A base variant before Stage B risk expansion.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
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
            Variant keys preserve existing v1 SHA-256 semantics.
        Raises:
            ValueError: If indexes or key lengths are invalid.
        Side Effects:
            None.
        """
        if self.stage_a_index < 0:
            raise ValueError("BacktestStageABaseVariantV2.stage_a_index must be >= 0")
        if len(self.indicator_variant_key) != 64:
            raise ValueError(
                "BacktestStageABaseVariantV2.indicator_variant_key must be 64 hex chars"
            )
        if len(self.base_variant_key) != 64:
            raise ValueError(
                "BacktestStageABaseVariantV2.base_variant_key must be 64 hex chars"
            )


@dataclass(frozen=True, slots=True)
class BacktestSignalFeaturesAccessPlanV2:
    """
    Additive per-indicator warm-cache access plan for optional `signal_features` runtime reads.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """

    indicator_id: str
    timeframe: str
    optional: bool = True

    def __post_init__(self) -> None:
        """
        Validate one deterministic warm-cache access entry for runtime feature loading.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Milestone C keeps signal-feature access additive and optional for exact profiles.
        Raises:
            ValueError: If one identifying literal is blank or `optional` is not boolean.
        Side Effects:
            None.
        """
        if not self.indicator_id.strip():
            raise ValueError(
                "BacktestSignalFeaturesAccessPlanV2.indicator_id must be non-empty"
            )
        if not self.timeframe.strip():
            raise ValueError("BacktestSignalFeaturesAccessPlanV2.timeframe must be non-empty")
        if not isinstance(self.optional, bool):
            raise ValueError("BacktestSignalFeaturesAccessPlanV2.optional must be bool")


@dataclass(frozen=True, slots=True)
class BacktestArtifactRuntimePlanV2:
    """
    Deterministic artifact-backed runtime plan for Stage A enumeration and Stage B expansion.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """

    indicator_plans: tuple[BacktestIndicatorPlanV2, ...]
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...]
    risk_variants: tuple[BacktestRiskVariantV2, ...]
    execution_profile: ExecutionProfileV2
    instrument_id_literal: str
    timeframe_code: str
    direction_mode: str
    sizing_mode: str
    execution_params: Mapping[str, BacktestVariantScalar]
    stage_a_variants_total: int
    stage_b_variants_total: int
    estimated_memory_bytes: int
    indicator_estimate_calls: int
    signal_features_access: tuple[BacktestSignalFeaturesAccessPlanV2, ...] = field(
        default=(),
    )
    adaptive_selector_decision: AdaptiveSelectorDecisionV2 | None = None

    def __post_init__(self) -> None:
        """
        Validate and freeze deterministic plan invariants.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stage totals and memory totals were guard-validated by the planner; when selector
            debug metadata is attached, it must agree with the effective execution profile.
        Raises:
            ValueError: If one scalar invariant is invalid or selector debug metadata drifts from
                the effective execution profile.
        Side Effects:
            Replaces execution params with an immutable key-sorted proxy.
        """
        if not self.instrument_id_literal.strip():
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.instrument_id_literal must be non-empty"
            )
        if self.execution_profile is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactRuntimePlanV2.execution_profile is required")
        if (
            self.adaptive_selector_decision is not None
            and self.adaptive_selector_decision.effective_profile.mode
            != self.execution_profile.mode
        ):
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.adaptive_selector_decision must match the "
                "effective execution profile"
            )
        if not self.timeframe_code.strip():
            raise ValueError("BacktestArtifactRuntimePlanV2.timeframe_code must be non-empty")
        if self.stage_a_variants_total <= 0:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_a_variants_total must be > 0"
            )
        if self.stage_b_variants_total <= 0:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_b_variants_total must be > 0"
            )
        if self.estimated_memory_bytes <= 0:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.estimated_memory_bytes must be > 0"
            )
        if self.indicator_estimate_calls < 0:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.indicator_estimate_calls must be >= 0"
            )
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping_v2(values=self.execution_params)),
        )
        object.__setattr__(
            self,
            "signal_features_access",
            tuple(
                sorted(
                    self.signal_features_access,
                    key=lambda plan: (plan.timeframe, plan.indicator_id),
                )
            ),
        )

    def iter_stage_a_variants(self) -> Iterator[BacktestStageABaseVariantV2]:
        """
        Iterate deterministic Stage A base variants using mixed-radix indexes.

        Args:
            None.
        Returns:
            Iterator[BacktestStageABaseVariantV2]: Deterministic Stage A variants.
        Assumptions:
            Indicator and signal plans are normalized and sorted by the planner.
        Raises:
            ValueError: If mixed-radix coordinates drift outside valid bounds.
        Side Effects:
            Reuses indicator-selection and `signal_params` payloads across repeated exact-path
            groups that share the same `compute_index` or `signal_index`.
        """
        signal_variants_total = _product_v2(
            values=tuple(len(axis.values) for axis in self.signal_axes)
        )
        signal_params_cache = tuple(
            _signal_params_from_variant_index_v2(
                signal_axes=self.signal_axes,
                variant_index=signal_index,
            )
            for signal_index in range(signal_variants_total)
        )
        indicator_radices = tuple(plan.variants for plan in self.indicator_plans)
        compute_variants_total = self.stage_a_variants_total // signal_variants_total
        if compute_variants_total * signal_variants_total != self.stage_a_variants_total:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_a_variants_total must stay aligned with "
                "indicator and signal mixed-radix products"
            )
        for compute_index in range(compute_variants_total):
            indicator_variant_indexes = _decode_mixed_radix_v2(
                flat_index=compute_index,
                radices=indicator_radices,
            )
            indicator_selections = tuple(
                _indicator_selection_from_variant_index_v2(
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
            stage_a_index_base = compute_index * signal_variants_total
            for signal_index, signal_params in enumerate(signal_params_cache):
                stage_a_index = stage_a_index_base + signal_index
                base_variant_key = build_backtest_variant_key_v1(
                    indicator_variant_key=indicator_variant_key,
                    direction_mode=self.direction_mode,
                    sizing_mode=self.sizing_mode,
                    signals=signal_params,
                    risk_params=_STAGE_A_DISABLED_RISK_PARAMS_V2,
                    execution_params=self.execution_params,
                )
                yield BacktestStageABaseVariantV2(
                    stage_a_index=stage_a_index,
                    indicator_selections=indicator_selections,
                    signal_params=signal_params,
                    indicator_variant_key=indicator_variant_key,
                    base_variant_key=base_variant_key,
                )

    def signal_features_access_for_indicator(
        self,
        *,
        indicator_id: str,
    ) -> BacktestSignalFeaturesAccessPlanV2 | None:
        """
        Resolve the additive warm-cache access entry for one indicator when present.

        Args:
            indicator_id: Canonical indicator identifier from Stage A plans.
        Returns:
            BacktestSignalFeaturesAccessPlanV2 | None: Access entry for the indicator or `None`
                when the current runtime plan does not request optional feature access.
        Assumptions:
            Runtime plan contains at most one signal-feature access entry per indicator/timeframe.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
        """
        for access_plan in self.signal_features_access:
            if access_plan.indicator_id == indicator_id:
                return access_plan
        return None

    def signal_variants_total(self) -> int:
        """
        Return the deterministic signal-space variants total owned by this runtime plan.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            None.
        Returns:
            int: Signal mixed-radix variants total (`1` when `signal_axes` is empty).
        Assumptions:
            Empty signal-axis sets still expand to one deterministic default-only signal payload.
        Raises:
            ValueError: If one signal axis materializes to zero values.
        Side Effects:
            None.
        """
        return _product_v2(values=tuple(len(axis.values) for axis in self.signal_axes))

    def stage_a_variant_for_index(
        self,
        *,
        stage_a_index: int,
    ) -> BacktestStageABaseVariantV2:
        """
        Materialize one exact Stage A base variant by stable mixed-radix index.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            stage_a_index: Zero-based Stage A index in the exact runtime enumeration order.
        Returns:
            BacktestStageABaseVariantV2: Canonical exact Stage A variant payload for the index.
        Assumptions:
            Mixed-radix ordering must stay identical to `iter_stage_a_variants()` so proposal
            layer runtimes may retain exact survivors without enumerating the full cartesian
            product.
        Raises:
            ValueError: If `stage_a_index` falls outside the exact Stage A range or if mixed-radix
                totals drift from plan invariants.
        Side Effects:
            None.
        """
        if stage_a_index < 0 or stage_a_index >= self.stage_a_variants_total:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_a_variant_for_index requires "
                f"stage_a_index in [0, {self.stage_a_variants_total}), got {stage_a_index}"
            )
        signal_variants_total = self.signal_variants_total()
        if signal_variants_total <= 0:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.signal_variants_total must be > 0"
            )
        compute_variants_total = self.stage_a_variants_total // signal_variants_total
        if compute_variants_total * signal_variants_total != self.stage_a_variants_total:
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_a_variants_total must stay aligned with "
                "indicator and signal mixed-radix products"
            )
        compute_index = stage_a_index // signal_variants_total
        signal_index = stage_a_index % signal_variants_total
        indicator_variant_indexes = _decode_mixed_radix_v2(
            flat_index=compute_index,
            radices=tuple(plan.variants for plan in self.indicator_plans),
        )
        indicator_selections = tuple(
            _indicator_selection_from_variant_index_v2(
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
        signal_params = _signal_params_from_variant_index_v2(
            signal_axes=self.signal_axes,
            variant_index=signal_index,
        )
        return BacktestStageABaseVariantV2(
            stage_a_index=stage_a_index,
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            indicator_variant_key=indicator_variant_key,
            base_variant_key=build_backtest_variant_key_v1(
                indicator_variant_key=indicator_variant_key,
                direction_mode=self.direction_mode,
                sizing_mode=self.sizing_mode,
                signals=signal_params,
                risk_params=_STAGE_A_DISABLED_RISK_PARAMS_V2,
                execution_params=self.execution_params,
            ),
        )


class BacktestArtifactRuntimePlannerV2:
    """
    Build deterministic artifact-backed runtime plans and enforce guard budgets.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """

    def __init__(
        self,
        *,
        execution_profiles: ExecutionProfilesCatalogV2 | None = None,
        launch_budget_mode: PlannerLaunchBudgetModeV2 = "ignore",
        adaptive_selector_policy: AdaptiveSelectorPolicyV2 | None = None,
        adaptive_selector: AdaptiveExecutionSelectorV2 | None = None,
    ) -> None:
        """
        Store typed execution-profile catalog used by exact profile selection and launch routing.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            execution_profiles: Optional startup-validated execution-profile catalog.
            launch_budget_mode:
                Whether to ignore heavy-request launch budgets (`ignore`) or to raise a
                deterministic `background_auto` routing signal when sync launch budgets are
                exceeded (`sync_inline`).
            adaptive_selector_policy:
                Optional startup-validated adaptive-selector rollout policy controlling
                `disabled`, `shadow`, and `active` automatic profile selection.
            adaptive_selector:
                Optional selector implementation. When omitted, the default deterministic cost
                model is used.
        Returns:
            None.
        Assumptions:
            Sync launch paths may enforce stricter profile launch budgets than background/worker
            paths, but both still reuse the same execution-profile catalog and adaptive-selector
            policy surface.
        Raises:
            ValueError: If resolved catalog or launch-budget mode is invalid.
        Side Effects:
            None.
        """
        resolved_execution_profiles = (
            execution_profiles or default_execution_profiles_catalog_v2()
        )
        if resolved_execution_profiles is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactRuntimePlannerV2 requires execution_profiles")
        if launch_budget_mode not in {"ignore", "sync_inline"}:
            raise ValueError(
                "BacktestArtifactRuntimePlannerV2.launch_budget_mode must be 'ignore' or "
                "'sync_inline'"
            )
        self._execution_profiles = resolved_execution_profiles
        self._launch_budget_mode = launch_budget_mode
        self._adaptive_selector_policy = (
            adaptive_selector_policy or default_adaptive_selector_policy_v2()
        )
        self._adaptive_selector = (
            adaptive_selector or CostModelAdaptiveExecutionSelectorV2()
        )

    def resolve_execution_profile(
        self,
        *,
        stage_a_variants_total: int | None = None,
        stage_b_variants_total: int | None = None,
        estimated_memory_bytes: int | None = None,
        requested_execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
        indicator_ids: tuple[str, ...] | None = None,
    ) -> ExecutionProfileV2:
        """
        Resolve the effective execution profile from deterministic planner cost evidence.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - apps/api/wiring/modules/backtest.py

        Args:
            stage_a_variants_total: Optional prepared Stage A variants count for future policy.
            stage_b_variants_total: Optional prepared Stage B variants count.
            estimated_memory_bytes: Optional prepared deterministic memory estimate.
            requested_execution_profile_mode:
                Optional internal-only requested execution profile mode. When present, automatic
                exact-profile selection is bypassed and the requested profile is validated against
                live runtime gating plus sync launch budgets.
            indicator_ids:
                Optional deterministic indicator ids from the prepared plan. These stay internal
                and are used only to validate `hybrid_family` plugin availability.
        Returns:
            ExecutionProfileV2: Selected execution profile for the prepared request.
        Assumptions:
            Requested exact-profile overrides keep precedence, while requested hybrid overrides are
            allowed only when selector rollout has reached explicit `opt_in` or `active`
            semantics; automatic selection uses the typed adaptive selector only when planning
            evidence is available.
        Raises:
            RoehubError: If sync launch budgets are exceeded and background routing is required.
            ValueError: If configured profiles cannot be resolved from the catalog.
        Side Effects:
            None.
        """
        if requested_execution_profile_mode is not None:
            requested_profile = self._execution_profiles.profile_for_mode(
                mode=requested_execution_profile_mode
            )
            if (
                execution_profile_uses_hierarchical_shortlist_runtime_v2(
                    profile=requested_profile
                )
                and not _adaptive_selector_policy_supports_requested_hybrid_profile_v2(
                    policy_mode=self._adaptive_selector_policy.mode
                )
            ):
                raise _requested_execution_profile_not_enabled_error_v2(
                    execution_profile_mode=requested_profile.mode,
                    policy_mode=self._adaptive_selector_policy.mode,
                )
            if not execution_profile_supports_requested_runtime_v2(
                profile=requested_profile
            ):
                raise _requested_execution_profile_not_enabled_error_v2(
                    execution_profile_mode=requested_profile.mode,
                    policy_mode=self._adaptive_selector_policy.mode,
                )
            if (
                self._launch_budget_mode == "sync_inline"
                and stage_a_variants_total is not None
                and stage_b_variants_total is not None
                and estimated_memory_bytes is not None
                and not requested_profile.launch_budget.allows(
                    stage_a_variants_total=stage_a_variants_total,
                    stage_b_variants_total=stage_b_variants_total,
                    estimated_memory_bytes=estimated_memory_bytes,
                )
            ):
                raise _background_auto_required_error_v2(
                    execution_profile_mode=requested_profile.mode,
                    stage_a_variants_total=stage_a_variants_total,
                    stage_b_variants_total=stage_b_variants_total,
                    estimated_memory_bytes=estimated_memory_bytes,
                )
            return requested_profile

        if (
            stage_a_variants_total is None
            or stage_b_variants_total is None
            or estimated_memory_bytes is None
        ):
            return self._execution_profiles.default_profile()

        return self._resolve_execution_profile_selection(
            stage_a_variants_total=stage_a_variants_total,
            stage_b_variants_total=stage_b_variants_total,
            estimated_memory_bytes=estimated_memory_bytes,
            indicator_ids=indicator_ids,
        )[0]

    def _resolve_execution_profile_selection(
        self,
        *,
        stage_a_variants_total: int,
        stage_b_variants_total: int,
        estimated_memory_bytes: int,
        indicator_ids: tuple[str, ...] | None = None,
    ) -> tuple[ExecutionProfileV2, AdaptiveSelectorDecisionV2]:
        """
        Resolve both the effective execution profile and the internal selector decision payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            stage_a_variants_total: Prepared Stage A variants count.
            stage_b_variants_total: Prepared Stage B variants count.
            estimated_memory_bytes: Deterministic memory estimate.
            indicator_ids: Optional deterministic indicator ids from the prepared plan.
        Returns:
            tuple[ExecutionProfileV2, AdaptiveSelectorDecisionV2]: Effective execution profile
                plus the full selector decision payload for internal inspection.
        Assumptions:
            This helper is used only after guard math is available, so the selector can stay
            deterministic and free of runtime IO.
        Raises:
            RoehubError: If sync launch budgets are exceeded and background routing is required.
        Side Effects:
            None.
        """
        selector_decision = self._adaptive_selector.select(
            evidence=AdaptiveSelectorPlanningEvidenceV2(
                grid_cardinality=stage_a_variants_total,
                stage_a_variants_total=stage_a_variants_total,
                stage_b_variants_total=stage_b_variants_total,
                estimated_memory_bytes=estimated_memory_bytes,
                runtime_mode=(
                    "sync_inline"
                    if self._launch_budget_mode == "sync_inline"
                    else "background_capable"
                ),
                indicator_ids=indicator_ids or (),
            ),
            execution_profiles=self._execution_profiles,
            policy=self._adaptive_selector_policy,
        )
        if selector_decision.requires_background_auto:
            raise _background_auto_required_error_v2(
                execution_profile_mode=selector_decision.exact_fallback_profile.mode,
                stage_a_variants_total=stage_a_variants_total,
                stage_b_variants_total=stage_b_variants_total,
                estimated_memory_bytes=estimated_memory_bytes,
            )
        return selector_decision.effective_profile, selector_decision

    def build(
        self,
        *,
        template: RunBacktestTemplate,
        candles: CandleArrays,
        indicator_compute: IndicatorCompute,
        preselect: int,
        requested_execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    ) -> BacktestArtifactRuntimePlanV2:
        """
        Build deterministic artifact-backed runtime plan with guard checks.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py

        Args:
            template: Resolved backtest template payload.
            candles: Warmup-inclusive request-timeframe candles from pinned artifacts.
            indicator_compute: Indicator estimate port used for compute-axis materialization.
            preselect: Stage A shortlist size before Stage B expansion.
            requested_execution_profile_mode:
                Optional internal-only execution profile mode overriding automatic exact profile
                selection for explicit rollout/test/manual wiring.
            defaults_provider: Optional defaults provider for compute/signal fallback.
            max_variants_per_compute: Variants guard budget.
            max_compute_bytes_total: Memory guard budget.
        Returns:
            BacktestArtifactRuntimePlanV2: Prepared artifact-backed runtime plan.
        Assumptions:
            Artifact runtime still uses `IndicatorCompute.estimate(...)` only for guard math and
            mixed-radix plan materialization, not for per-variant hot-path compute.
        Raises:
            RoehubError: If guard limits are exceeded.
            ValueError: If request axes are invalid.
        Side Effects:
            Calls `indicator_compute.estimate(...)` once per indicator block.
        """
        if preselect <= 0:
            raise ValueError("BacktestArtifactRuntimePlannerV2 preselect must be > 0")
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
        stage_a_variants_total = _product_v2(
            values=tuple(plan.variants for plan in indicator_plans)
        ) * _product_v2(values=tuple(len(axis.values) for axis in signal_axes))
        if stage_a_variants_total > max_variants_per_compute:
            raise _variants_guard_error_v2(
                stage=STAGE_A_LITERAL_V2,
                total_variants=stage_a_variants_total,
                max_variants_per_compute=max_variants_per_compute,
                execution_profile_mode=self._execution_profiles.background_exact_profile().mode,
            )

        estimated_memory_bytes = _estimate_memory_bytes_v2(
            bars=len(candles.ts_open),
            indicator_plans=indicator_plans,
        )
        if estimated_memory_bytes > max_compute_bytes_total:
            raise _memory_guard_error_v2(
                stage=STAGE_A_LITERAL_V2,
                estimated_memory_bytes=estimated_memory_bytes,
                max_compute_bytes_total=max_compute_bytes_total,
                execution_profile_mode=self._execution_profiles.background_exact_profile().mode,
            )

        risk_variants = _risk_variants_from_template_v2(template=template)
        shortlist_len = min(preselect, stage_a_variants_total)
        stage_b_variants_total = shortlist_len * len(risk_variants)
        if stage_b_variants_total > max_variants_per_compute:
            raise _variants_guard_error_v2(
                stage=STAGE_B_LITERAL_V2,
                total_variants=stage_b_variants_total,
                max_variants_per_compute=max_variants_per_compute,
                execution_profile_mode=self._execution_profiles.background_exact_profile().mode,
            )
        adaptive_selector_decision: AdaptiveSelectorDecisionV2 | None = None
        if requested_execution_profile_mode is None:
            execution_profile, adaptive_selector_decision = (
                self._resolve_execution_profile_selection(
                    stage_a_variants_total=stage_a_variants_total,
                    stage_b_variants_total=stage_b_variants_total,
                    estimated_memory_bytes=estimated_memory_bytes,
                    indicator_ids=tuple(plan.indicator_id for plan in indicator_plans),
                )
            )
        else:
            execution_profile = self.resolve_execution_profile(
                stage_a_variants_total=stage_a_variants_total,
                stage_b_variants_total=stage_b_variants_total,
                estimated_memory_bytes=estimated_memory_bytes,
                requested_execution_profile_mode=requested_execution_profile_mode,
                indicator_ids=tuple(plan.indicator_id for plan in indicator_plans),
            )

        return BacktestArtifactRuntimePlanV2(
            indicator_plans=indicator_plans,
            signal_axes=signal_axes,
            risk_variants=risk_variants,
            execution_profile=execution_profile,
            instrument_id_literal=_instrument_id_literal_v2(template=template),
            timeframe_code=template.timeframe.code,
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            execution_params=template.execution_params or {},
            signal_features_access=tuple(
                BacktestSignalFeaturesAccessPlanV2(
                    indicator_id=plan.indicator_id,
                    timeframe=template.timeframe.code,
                    optional=True,
                )
                for plan in indicator_plans
            ),
            stage_a_variants_total=stage_a_variants_total,
            stage_b_variants_total=stage_b_variants_total,
            estimated_memory_bytes=estimated_memory_bytes,
            indicator_estimate_calls=len(indicator_plans),
            adaptive_selector_decision=adaptive_selector_decision,
        )

    def _build_indicator_plans(
        self,
        *,
        template: RunBacktestTemplate,
        indicator_compute: IndicatorCompute,
        defaults_provider: BacktestGridDefaultsProvider | None,
        max_variants_per_compute: int,
    ) -> tuple[BacktestIndicatorPlanV2, ...]:
        """
        Build deterministic compute plans by estimating materialized axes per indicator.

        Args:
            template: Backtest template with request grids.
            indicator_compute: Indicator estimate port.
            defaults_provider: Optional provider for missing compute axes.
            max_variants_per_compute: Per-indicator estimate guard.
        Returns:
            tuple[BacktestIndicatorPlanV2, ...]: Sorted indicator compute plans.
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
            grids_by_id[indicator_id] = _merge_grid_with_defaults_v2(
                request_grid=request_grid,
                defaults_grid=defaults_grid,
            )

        plans: list[BacktestIndicatorPlanV2] = []
        for indicator_id in sorted(grids_by_id.keys()):
            grid = grids_by_id[indicator_id]
            estimate = indicator_compute.estimate(
                grid,
                max_variants_guard=max_variants_per_compute,
            )
            plans.append(
                _indicator_plan_from_estimate_v2(
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
        indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
    ) -> tuple[BacktestSignalAxisPlanV2, ...]:
        """
        Build deterministic signal-axis plans from request payload and defaults.

        Args:
            template: Backtest template with optional signal grids.
            defaults_provider: Optional provider for fallback signal defaults.
            indicator_plans: Indicator compute plans selected for Stage A.
        Returns:
            tuple[BacktestSignalAxisPlanV2, ...]: Sorted materialized signal axes.
        Assumptions:
            Signal grids are stored under `indicator_id -> param_name -> GridParamSpec`.
        Raises:
            ValueError: If one signal axis materializes to an empty sequence.
        Side Effects:
            Reads optional defaults via `defaults_provider`.
        """
        request_signal_grids = template.signal_grids or {}
        axes: list[BacktestSignalAxisPlanV2] = []
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
                values = _materialize_signal_values_v2(spec=spec, axis_name=param_name)
                axes.append(
                    BacktestSignalAxisPlanV2(
                        indicator_id=indicator_id,
                        param_name=param_name.strip().lower(),
                        values=values,
                    )
                )
        return tuple(sorted(axes, key=lambda axis: (axis.indicator_id, axis.param_name)))


def _indicator_plan_from_estimate_v2(
    *,
    indicator_id: str,
    axes: tuple[AxisDef, ...],
) -> BacktestIndicatorPlanV2:
    """
    Build deterministic indicator plan from `IndicatorCompute.estimate(...)` axis payload.

    Args:
        indicator_id: Indicator identifier.
        axes: Materialized axis definitions from estimate result.
    Returns:
        BacktestIndicatorPlanV2: Deterministic plan with axis values and variant count.
    Assumptions:
        Axis value families are validated by `AxisDef` invariants.
    Raises:
        ValueError: If one axis has no supported value family.
    Side Effects:
        None.
    """
    axis_plans: list[BacktestIndicatorAxisPlanV2] = []
    variants = 1
    for axis in axes:
        values = _axis_values_v2(axis=axis)
        axis_plans.append(BacktestIndicatorAxisPlanV2(name=axis.name, values=values))
        variants = variants * len(values)
    return BacktestIndicatorPlanV2(
        indicator_id=indicator_id,
        axes=tuple(axis_plans),
        variants=variants,
    )


def _axis_values_v2(*, axis: AxisDef) -> tuple[int | float | str, ...]:
    """
    Convert one `AxisDef` into deterministic scalar tuple preserving axis ordering.

    Args:
        axis: Domain axis definition from indicators estimate.
    Returns:
        tuple[int | float | str, ...]: Deterministic scalar values.
    Assumptions:
        Exactly one value family is set in `AxisDef`.
    Raises:
        ValueError: If axis contains no supported values.
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


def build_signal_params_for_variant_index_v2(
    *,
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
    variant_index: int,
) -> Mapping[str, Mapping[str, BacktestVariantScalar]]:
    """
    Build canonical signal params for one signal-space mixed-radix index.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        signal_axes: Sorted materialized signal axes from the runtime plan.
        variant_index: Flat signal-space index.
    Returns:
        Mapping[str, Mapping[str, BacktestVariantScalar]]: Canonical nested signal params map.
    Assumptions:
        Hybrid rollout must reuse the same signal expansion semantics as the exact runtime.
    Raises:
        ValueError: If `variant_index` is outside the mixed-radix signal bounds.
    Side Effects:
        None.
    """
    return _signal_params_from_variant_index_v2(
        signal_axes=signal_axes,
        variant_index=variant_index,
    )


def build_indicator_selection_for_variant_index_v2(
    *,
    plan: BacktestIndicatorPlanV2,
    variant_index: int,
) -> IndicatorVariantSelection:
    """
    Build canonical indicator selection for one indicator-local mixed-radix index.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        plan: One materialized indicator plan from the runtime planner.
        variant_index: Flat indicator-local variant index.
    Returns:
        IndicatorVariantSelection: Canonical explicit indicator selection.
    Assumptions:
        Hybrid rollout must preserve the same variant-key inputs as the exact runtime.
    Raises:
        ValueError: If `variant_index` is outside the indicator-local mixed-radix bounds.
    Side Effects:
        None.
    """
    return _indicator_selection_from_variant_index_v2(
        plan=plan,
        variant_index=variant_index,
    )


def _signal_params_from_variant_index_v2(
    *,
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
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
        ValueError: If the flat index is outside valid mixed-radix bounds.
    Side Effects:
        None.
    """
    if len(signal_axes) == 0:
        return {}

    radices = tuple(len(axis.values) for axis in signal_axes)
    coordinates = _decode_mixed_radix_v2(flat_index=variant_index, radices=radices)
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


def _indicator_selection_from_variant_index_v2(
    *,
    plan: BacktestIndicatorPlanV2,
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
        ValueError: If the index is outside plan variant range.
    Side Effects:
        None.
    """
    coordinates = _decode_mixed_radix_v2(
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


def _materialize_signal_values_v2(
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
        Signal-axis materialization uses the same explicit/range semantics as indicators.
    Raises:
        ValueError: If materialization yields an empty sequence.
    Side Effects:
        None.
    """
    values = tuple(spec.materialize())
    if len(values) == 0:
        raise ValueError(f"signal axis '{axis_name}' materialized to empty values")
    return values


def _risk_variants_from_template_v2(
    *,
    template: RunBacktestTemplate,
) -> tuple[BacktestRiskVariantV2, ...]:
    """
    Build deterministic Stage B risk variants from request risk grid or scalar fallback.

    Args:
        template: Backtest template payload.
    Returns:
        tuple[BacktestRiskVariantV2, ...]: Deterministic Stage B risk variants.
    Assumptions:
        Risk values represent human percentages where `3.0 == 3%`.
    Raises:
        ValueError: If enabled risk axes are missing or non-numeric.
    Side Effects:
        None.
    """
    risk_grid = template.risk_grid or BacktestRiskGridSpec()
    risk_params = template.risk_params or {}
    sl_enabled = risk_grid.sl_enabled
    tp_enabled = risk_grid.tp_enabled
    if "sl_enabled" in risk_params:
        sl_enabled = _bool_scalar_v2(value=risk_params["sl_enabled"], field_name="sl_enabled")
    if "tp_enabled" in risk_params:
        tp_enabled = _bool_scalar_v2(value=risk_params["tp_enabled"], field_name="tp_enabled")

    if sl_enabled:
        sl_values = _materialize_risk_axis_v2(
            spec=risk_grid.sl,
            axis_name="sl",
            fallback_value=risk_params.get("sl_pct"),
        )
    else:
        sl_values = (None,)

    if tp_enabled:
        tp_values = _materialize_risk_axis_v2(
            spec=risk_grid.tp,
            axis_name="tp",
            fallback_value=risk_params.get("tp_pct"),
        )
    else:
        tp_values = (None,)

    variants: list[BacktestRiskVariantV2] = []
    variant_index = 0
    for sl_pct in sl_values:
        for tp_pct in tp_values:
            variants.append(
                BacktestRiskVariantV2(
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


def _bool_scalar_v2(*, value: BacktestVariantScalar, field_name: str) -> bool:
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
        ValueError: If the value is not boolean.
    Side Effects:
        None.
    """
    if not isinstance(value, bool):
        raise ValueError(f"risk field '{field_name}' must be boolean")
    return value


def _materialize_risk_axis_v2(
    *,
    spec: GridParamSpec | None,
    axis_name: str,
    fallback_value: BacktestVariantScalar,
) -> tuple[float, ...]:
    """
    Materialize one enabled Stage B risk axis from grid spec or fallback scalar.

    Args:
        spec: Optional risk grid spec.
        axis_name: Axis name (`sl` or `tp`) for diagnostics.
        fallback_value: Optional scalar fallback from `template.risk_params`.
    Returns:
        tuple[float, ...]: Materialized numeric percentages.
    Assumptions:
        Enabled risk axes require at least one numeric value.
    Raises:
        ValueError: If the axis is missing, empty, or non-numeric.
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


def _estimate_memory_bytes_v2(
    *,
    bars: int,
    indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
) -> int:
    """
    Estimate total memory bytes for artifact runtime guard enforcement.

    Args:
        bars: Number of warmup-inclusive request-timeframe bars.
        indicator_plans: Materialized per-indicator compute plans.
    Returns:
        int: Estimated total memory bytes with reserve.
    Assumptions:
        Policy matches the existing indicators estimator (`candles + tensors + reserve`).
    Raises:
        ValueError: If the bars count is non-positive.
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


def _decode_mixed_radix_v2(*, flat_index: int, radices: tuple[int, ...]) -> tuple[int, ...]:
    """
    Decode mixed-radix coordinates from a flat index without cartesian materialization.

    Args:
        flat_index: Flat zero-based index in mixed-radix space.
        radices: Axis radices in deterministic order.
    Returns:
        tuple[int, ...]: Coordinate for each axis.
    Assumptions:
        Every radix is a positive integer.
    Raises:
        ValueError: If index/radices are invalid or out of bounds.
    Side Effects:
        None.
    """
    if len(radices) == 0:
        if flat_index != 0:
            raise ValueError("flat_index must be 0 when radices are empty")
        return ()

    total = _product_v2(values=radices)
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


def _product_v2(*, values: tuple[int, ...]) -> int:
    """
    Compute deterministic product for integer tuple values.

    Args:
        values: Integer tuple values.
    Returns:
        int: Product value (`1` for an empty tuple).
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


def _instrument_id_literal_v2(*, template: RunBacktestTemplate) -> str:
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


def _merge_grid_with_defaults_v2(
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
        Both grids target the same indicator id.
    Raises:
        ValueError: If the defaults grid uses a mismatched indicator id.
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


def _normalize_scalar_mapping_v2(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> dict[str, BacktestVariantScalar]:
    """
    Normalize scalar mapping into a deterministic key-sorted dictionary.

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
            raise ValueError("scalar mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized


def _variants_guard_error_v2(
    *,
    stage: str,
    total_variants: int,
    max_variants_per_compute: int,
    execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
) -> RoehubError:
    """
    Build canonical variants-guard overflow error for artifact-backed runtime planning.

    Args:
        stage: Stage literal where the overflow occurred.
        total_variants: Computed variant total.
        max_variants_per_compute: Configured variants guard.
        execution_profile_mode: Optional exact profile hint for background fallback persistence.
    Returns:
        RoehubError: Canonical validation error payload.
    Assumptions:
        Error shape stays stable across sync and background preflight paths.
    Raises:
        None.
    Side Effects:
        Adds an additive `execution_profile_mode` hint when profile-aware routing is active.
    """
    details: dict[str, object] = {
        "error": "max_variants_per_compute_exceeded",
        "stage": stage,
        "total_variants": total_variants,
        "max_variants_per_compute": max_variants_per_compute,
    }
    if execution_profile_mode is not None:
        details["execution_profile_mode"] = execution_profile_mode
    return RoehubError(
        code="validation_error",
        message="Backtest variants exceed configured compute budget",
        details=details,
    )


def _memory_guard_error_v2(
    *,
    stage: str,
    estimated_memory_bytes: int,
    max_compute_bytes_total: int,
    execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
) -> RoehubError:
    """
    Build canonical memory-guard overflow error for artifact-backed runtime planning.

    Args:
        stage: Stage literal where the overflow occurred.
        estimated_memory_bytes: Estimated memory bytes.
        max_compute_bytes_total: Configured memory guard budget.
        execution_profile_mode: Optional exact profile hint for background fallback persistence.
    Returns:
        RoehubError: Canonical validation error payload.
    Assumptions:
        Error shape stays stable across sync and background preflight paths.
    Raises:
        None.
    Side Effects:
        Adds an additive `execution_profile_mode` hint when profile-aware routing is active.
    """
    details: dict[str, object] = {
        "error": "max_compute_bytes_total_exceeded",
        "stage": stage,
        "estimated_memory_bytes": estimated_memory_bytes,
        "max_compute_bytes_total": max_compute_bytes_total,
    }
    if execution_profile_mode is not None:
        details["execution_profile_mode"] = execution_profile_mode
    return RoehubError(
        code="validation_error",
        message="Backtest estimated memory exceeds configured compute budget",
        details=details,
    )


def _adaptive_selector_policy_supports_requested_hybrid_profile_v2(
    *,
    policy_mode: str,
) -> bool:
    """
    Check whether selector rollout explicitly allows internal live hybrid opt-in requests.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        policy_mode: Environment-level adaptive selector rollout mode.
    Returns:
        bool: `True` when explicit requested hybrid execution is allowed for the env.
    Assumptions:
        `shadow` keeps recommendations inspectable but does not yet permit live hybrid opt-in,
        while `opt_in` and `active` do.
    Raises:
        None.
    Side Effects:
        None.
    """
    return policy_mode in {"opt_in", "active"}


def _background_auto_required_error_v2(
    *,
    execution_profile_mode: ExecutionProfileModeLiteralV2,
    stage_a_variants_total: int,
    stage_b_variants_total: int,
    estimated_memory_bytes: int,
) -> RoehubError:
    """
    Build deterministic sync-launch overflow signal for heavy-but-valid exact requests.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/wiring/modules/backtest.py

    Args:
        execution_profile_mode: Exact execution profile selected for background execution.
        stage_a_variants_total: Deterministic prepared Stage A variants count.
        stage_b_variants_total: Deterministic prepared Stage B variants count.
        estimated_memory_bytes: Deterministic estimated runtime memory footprint.
    Returns:
        RoehubError: Canonical validation payload used internally for `background_auto` routing.
    Assumptions:
        Full-budget preflight remains the source of truth for hard rejects; this error only
        signals that sync launch budgets are too small for an otherwise exact-valid request.
    Raises:
        None.
    Side Effects:
        None.
    """
    return RoehubError(
        code="validation_error",
        message="Backtest request exceeds sync launch budget and should run in background",
        details={
            "error": "background_auto_required",
            "execution_profile_mode": execution_profile_mode,
            "execution_mode": "background_auto",
            "stage_a_variants_total": stage_a_variants_total,
            "stage_b_variants_total": stage_b_variants_total,
            "estimated_memory_bytes": estimated_memory_bytes,
        },
    )


def _requested_execution_profile_not_enabled_error_v2(
    *,
    execution_profile_mode: ExecutionProfileModeLiteralV2,
    policy_mode: str | None = None,
) -> RoehubError:
    """
    Build canonical validation error for explicitly requested but rollout-disabled profiles.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        execution_profile_mode: Internal requested execution profile mode literal.
        policy_mode: Optional selector rollout mode active for the current environment.
    Returns:
        RoehubError: Canonical validation payload describing the disabled requested profile.
    Assumptions:
        Requested hybrid rollout profiles remain internal-only and must fail fast when feature
        flags or selector rollout policy do not explicitly enable live runtime.
    Raises:
        None.
    Side Effects:
        None.
    """
    details: dict[str, object] = {
        "error": "execution_profile_not_enabled",
        "execution_profile_mode": execution_profile_mode,
    }
    if policy_mode is not None:
        details["adaptive_selector_policy_mode"] = policy_mode
    return RoehubError(
        code="validation_error",
        message="Requested execution profile is not enabled for live runtime",
        details=details,
    )


__all__ = [
    "BacktestArtifactRuntimePlanV2",
    "BacktestArtifactRuntimePlannerV2",
    "build_indicator_selection_for_variant_index_v2",
    "build_signal_params_for_variant_index_v2",
    "BacktestIndicatorAxisPlanV2",
    "BacktestIndicatorPlanV2",
    "BacktestRiskVariantV2",
    "BacktestSignalAxisPlanV2",
    "BacktestSignalFeaturesAccessPlanV2",
    "BacktestStageABaseVariantV2",
    "STAGE_A_LITERAL_V2",
    "STAGE_B_LITERAL_V2",
]
