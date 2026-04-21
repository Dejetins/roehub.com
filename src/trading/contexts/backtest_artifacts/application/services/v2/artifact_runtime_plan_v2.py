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
    ExecutionProfileLaunchBudgetEvidenceV2,
    ExecutionProfileModeLiteralV2,
    ExecutionProfileParityClassificationV2,
    ExecutionProfilesCatalogV2,
    ExecutionProfileV2,
    default_execution_profiles_catalog_v2,
    execution_profile_stage_b_process_fallback_threshold_v2,
    execution_profile_supports_requested_runtime_v2,
    execution_profile_uses_hierarchical_shortlist_runtime_v2,
    execution_profile_uses_process_pool_stage_b_v2,
)

STAGE_A_LITERAL_V2 = "stage_a"
STAGE_B_LITERAL_V2 = "stage_b"
STAGE_B_EXECUTION_MODE_BYPASSED_NO_RISK_LITERAL_V2 = "bypassed_no_risk"
STAGE_B_EXECUTION_MODE_IN_PROCESS_LITERAL_V2 = "in_process"
STAGE_B_EXECUTION_MODE_PROCESS_POOL_LITERAL_V2 = "process_pool"
PlannerLaunchBudgetModeV2 = Literal["ignore", "sync_inline"]

_FLOAT32_BYTES = 4
_CANDLES_BYTES_PER_STEP = (5 * _FLOAT32_BYTES) + 8
_RESERVE_FACTOR = 0.20
_RESERVE_FIXED_BYTES = 64 * 1024**2
_COMBO_PROXY_PREFILTER_SURVIVOR_MULTIPLIER_V2 = 2
_ROW_PREFILTER_COST_WEIGHT_V2 = 1
_COMBO_PREFILTER_COST_WEIGHT_V2 = 1
_RETAINED_STAGE_A_EXACT_COST_WEIGHT_V2 = 4
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
class BacktestRuntimeStageCostModelV2:
    """
    Internal retained-frontier cost model collapsed under stable public stage vocabulary.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
    """

    row_prefilter_rows_total: int
    retained_row_variants_total: int
    combo_prefilter_variants_total: int
    retained_exact_candidates_total: int
    stage_a_cost_units: int
    retained_rows_per_indicator: tuple[int, ...] = ()
    narrowed_compute_variants_total: int | None = None

    def __post_init__(self) -> None:
        """
        Validate retained-frontier stage cost totals used only for internal planning.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Public `stage_a` remains stable while planner math separately tracks row prefilter,
            retained-row tensor envelope, combo prefilter, and retained-candidate exact work as
            positive deterministic counts. Additive parity-facing counters stay internal-only.
        Raises:
            ValueError: If one internal stage total is non-positive.
        Side Effects:
            None.
        """
        for field_name in (
            "row_prefilter_rows_total",
            "retained_row_variants_total",
            "combo_prefilter_variants_total",
            "retained_exact_candidates_total",
            "stage_a_cost_units",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"BacktestRuntimeStageCostModelV2.{field_name} must be > 0"
                )
        if self.retained_rows_per_indicator:
            if any(value <= 0 for value in self.retained_rows_per_indicator):
                raise ValueError(
                    "BacktestRuntimeStageCostModelV2.retained_rows_per_indicator must be > 0 "
                    "when provided"
                )
            if sum(self.retained_rows_per_indicator) != self.retained_row_variants_total:
                raise ValueError(
                    "BacktestRuntimeStageCostModelV2.retained_rows_per_indicator must sum to "
                    "retained_row_variants_total"
                )
        if self.narrowed_compute_variants_total is not None:
            if self.narrowed_compute_variants_total <= 0:
                raise ValueError(
                    "BacktestRuntimeStageCostModelV2.narrowed_compute_variants_total must be > "
                    "0 when provided"
                )
            if (
                self.narrowed_compute_variants_total
                > self.combo_prefilter_variants_total
            ):
                raise ValueError(
                    "BacktestRuntimeStageCostModelV2.narrowed_compute_variants_total cannot "
                    "exceed combo_prefilter_variants_total"
                )


@dataclass(frozen=True, slots=True)
class BacktestParityRetainedRowsCounterV2:
    """
    Deterministic retained-row counter for one indicator inside the parity runtime plan.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    indicator_id: str
    retained_rows: int

    def __post_init__(self) -> None:
        """
        Validate one per-indicator retained-row counter payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Parity retained-row counters are additive benchmark-facing metadata only.
        Raises:
            ValueError: If the indicator id is blank or retained rows are non-positive.
        Side Effects:
            None.
        """
        if not self.indicator_id.strip():
            raise ValueError(
                "BacktestParityRetainedRowsCounterV2.indicator_id must be non-empty"
            )
        if self.retained_rows <= 0:
            raise ValueError(
                "BacktestParityRetainedRowsCounterV2.retained_rows must be > 0"
            )


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactParityRuntimeCountersV2:
    """
    Additive benchmark-facing counters for the first-class no-risk exact parity runtime plan.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    retained_rows_per_indicator: tuple[BacktestParityRetainedRowsCounterV2, ...]
    retained_rows_total: int
    narrowed_combo_total: int
    narrowed_compute_combo_total: int
    no_risk_finalization_count: int
    exact_replay_count: int = 0
    deterministic_combo_ordering: str = "stage_a_index"
    stage_b_execution_mode: str = STAGE_B_EXECUTION_MODE_BYPASSED_NO_RISK_LITERAL_V2

    def __post_init__(self) -> None:
        """
        Validate additive no-risk parity runtime counters.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Counters are internal benchmark-facing evidence and must stay deterministic across
            sync and worker flows for the same artifact slot.
        Raises:
            ValueError: If one additive counter is invalid.
        Side Effects:
            None.
        """
        if len(self.retained_rows_per_indicator) == 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.retained_rows_per_indicator "
                "must be non-empty"
            )
        if self.retained_rows_total <= 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.retained_rows_total must be > 0"
            )
        if self.narrowed_combo_total <= 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.narrowed_combo_total must be > 0"
            )
        if self.narrowed_compute_combo_total <= 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2."
                "narrowed_compute_combo_total must be > 0"
            )
        if self.no_risk_finalization_count <= 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.no_risk_finalization_count "
                "must be > 0"
            )
        if self.exact_replay_count < 0:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.exact_replay_count must be >= 0"
            )
        if not self.deterministic_combo_ordering.strip():
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2."
                "deterministic_combo_ordering must be non-empty"
            )
        if (
            self.stage_b_execution_mode
            != STAGE_B_EXECUTION_MODE_BYPASSED_NO_RISK_LITERAL_V2
        ):
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.stage_b_execution_mode must be "
                "'bypassed_no_risk'"
            )
        if (
            sum(item.retained_rows for item in self.retained_rows_per_indicator)
            != self.retained_rows_total
        ):
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.retained_rows_total must equal the "
                "sum of retained_rows_per_indicator"
            )
        indicator_ids = tuple(
            item.indicator_id for item in self.retained_rows_per_indicator
        )
        if len(indicator_ids) != len(set(indicator_ids)):
            raise ValueError(
                "BacktestNoRiskExactParityRuntimeCountersV2.retained_rows_per_indicator must "
                "not duplicate indicator ids"
            )

    def as_mapping(self) -> Mapping[str, object]:
        """
        Export immutable additive parity counters for benchmark-facing runtime scans.

        Args:
            None.
        Returns:
            Mapping[str, object]: Immutable benchmark-facing counter payload.
        Assumptions:
            Counter export remains internal-only and deterministic.
        Raises:
            None.
        Side Effects:
            None.
        """
        return MappingProxyType(
            {
                "retained_rows_total": self.retained_rows_total,
                "narrowed_combo_total": self.narrowed_combo_total,
                "narrowed_compute_combo_total": self.narrowed_compute_combo_total,
                "no_risk_finalization_count": self.no_risk_finalization_count,
                "exact_replay_count": self.exact_replay_count,
                "deterministic_combo_ordering": self.deterministic_combo_ordering,
                "stage_b_execution_mode": self.stage_b_execution_mode,
                "retained_rows_per_indicator": MappingProxyType(
                    {
                        item.indicator_id: item.retained_rows
                        for item in self.retained_rows_per_indicator
                    }
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class BacktestArtifactRuntimePlanV2:
    """
    Deterministic artifact-backed runtime plan for Stage A enumeration and Stage B expansion.

    Docs:
      - docs/architecture/backtest/README.md
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
    stage_cost_model: BacktestRuntimeStageCostModelV2 | None = None
    launch_budget_evidence: ExecutionProfileLaunchBudgetEvidenceV2 | None = None
    parity_classification: ExecutionProfileParityClassificationV2 | None = None

    def __post_init__(self) -> None:
        """
        Validate and freeze deterministic plan invariants.

        Docs:
          - docs/architecture/backtest/README.md
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
            debug metadata is attached, it must agree with the effective execution profile. When
            launch-budget evidence or parity classification is attached, each internal additive
            marker must stay aligned with the prepared no-risk or risk-grid shape.
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
        if (
            self.stage_cost_model is not None
            and self.stage_cost_model.retained_exact_candidates_total
            > self.stage_a_variants_total
        ):
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.stage_cost_model retained frontier cannot "
                "exceed stage_a_variants_total"
            )
        if (
            self.launch_budget_evidence is not None
            and self.uses_no_risk_terminal_path()
            and self.launch_budget_evidence.workload_class != "no_risk_terminal"
        ):
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.launch_budget_evidence must classify "
                "no-risk plans as no_risk_terminal"
            )
        if (
            self.launch_budget_evidence is not None
            and not self.uses_no_risk_terminal_path()
            and self.launch_budget_evidence.workload_class == "no_risk_terminal"
        ):
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.launch_budget_evidence cannot classify "
                "risk-grid plans as no_risk_terminal"
            )
        if self.parity_classification is not None and not self.uses_no_risk_terminal_path():
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.parity_classification requires no-risk terminal "
                "path"
            )
        if (
            self.parity_classification is not None
            and self.adaptive_selector_decision is not None
            and self.execution_profile.mode != "exact_no_risk_parity"
        ):
            raise ValueError(
                "BacktestArtifactRuntimePlanV2.parity_classification requires "
                "exact_no_risk_parity execution profile"
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
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
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

    def uses_no_risk_terminal_path(self) -> bool:
        """
        Classify whether the shared runtime should finalize at the Stage A exact boundary.

        Args:
            None.
        Returns:
            bool: `True` when the prepared plan belongs to the no-risk class.
        Assumptions:
            The no-risk terminal path is valid only for a single disabled-risk cell where both TP
            and SL remain off, letting sync and worker bypass heavy generic Stage B internally
            while keeping public `stage_a` / `stage_b` vocabulary stable.
        Raises:
            None.
        Side Effects:
            None.
        """
        return _risk_variants_use_no_risk_terminal_path_v2(
            risk_variants=self.risk_variants
        )

    def stage_b_execution_mode(self) -> str:
        """
        Resolve the deterministic `stage_b_execution_mode` classification for this runtime plan.

        Args:
            None.
        Returns:
            str: Canonical `stage_b_execution_mode` literal for orchestration and NR2 benchmarks.
        Assumptions:
            No-risk runs report `bypassed_no_risk`, while risk-grid runs stay `in_process` unless
            the resolved execution profile explicitly opts into the non-default process fallback
            and the prepared workload crosses the explicit `stage_b_variants_total` threshold.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.uses_no_risk_terminal_path():
            return STAGE_B_EXECUTION_MODE_BYPASSED_NO_RISK_LITERAL_V2
        if execution_profile_uses_process_pool_stage_b_v2(
            profile=self.execution_profile,
            stage_b_variants_total=self.stage_b_variants_total,
        ):
            return STAGE_B_EXECUTION_MODE_PROCESS_POOL_LITERAL_V2
        return STAGE_B_EXECUTION_MODE_IN_PROCESS_LITERAL_V2

    def stage_b_process_fallback_threshold(self) -> str:
        """
        Resolve which explicit workload threshold activated the non-default Stage B fallback.

        Args:
            None.
        Returns:
            str: Canonical threshold literal for runtime-shape scans and benchmark traces.
        Assumptions:
            Canonical parity workloads must report `none`, while larger non-default workloads may
            report the explicit `stage_b_variants_total` threshold when process fallback is used.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.uses_no_risk_terminal_path():
            return "none"
        return execution_profile_stage_b_process_fallback_threshold_v2(
            profile=self.execution_profile,
            stage_b_variants_total=self.stage_b_variants_total,
        )

    def uses_hybrid_reduced_plan_contract(self) -> bool:
        """
        Report whether this runtime plan depends on hybrid reduced-plan semantics.

        Args:
            None.
        Returns:
            bool: `False` for the base exact runtime plan contract.
        Assumptions:
            Hybrid reduced-plan wrappers override this method to expose dependency on retained
            shortlist surfaces, while first-class parity plans must stay `False`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return False

    def parity_runtime_counters(self) -> Mapping[str, object] | None:
        """
        Return additive parity runtime counters when the plan owns a parity-first contract.

        Args:
            None.
        Returns:
            Mapping[str, object] | None: Immutable parity counters for benchmark scans, or
                `None` for non-parity plans.
        Assumptions:
            Counter exposure remains internal benchmark-facing metadata and is not a public API.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None

    def stage_a_variant_for_index(
        self,
        *,
        stage_a_index: int,
    ) -> BacktestStageABaseVariantV2:
        """
        Materialize one exact Stage A base variant by stable mixed-radix index.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
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


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactParityRuntimePlanV2(BacktestArtifactRuntimePlanV2):
    """
    First-class runtime-plan contract for canonical no-risk exact parity workloads.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
    """

    no_risk_parity_counters: BacktestNoRiskExactParityRuntimeCountersV2 | None = None

    def __post_init__(self) -> None:
        """
        Validate first-class parity runtime-plan invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Canonical parity plan must remain sync/worker-equivalent, no-risk exact, and detached
            from hybrid reduced-plan semantics.
        Raises:
            ValueError: If no-risk or parity profile invariants drift.
        Side Effects:
            Reuses base-plan normalization from `BacktestArtifactRuntimePlanV2.__post_init__`.
        """
        BacktestArtifactRuntimePlanV2.__post_init__(self)
        if not self.uses_no_risk_terminal_path():
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 requires no-risk terminal path"
            )
        if self.execution_profile.mode != "exact_no_risk_parity":
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 requires "
                "execution_profile.mode='exact_no_risk_parity'"
            )
        if self.parity_classification is None:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 requires parity_classification"
            )
        if self.no_risk_parity_counters is None:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 requires no_risk_parity_counters"
            )
        if (
            self.no_risk_parity_counters.stage_b_execution_mode
            != self.stage_b_execution_mode()
        ):
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 counters must report the same "
                "stage_b_execution_mode as the runtime plan"
            )

    @classmethod
    def from_runtime_plan(
        cls,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        no_risk_parity_counters: BacktestNoRiskExactParityRuntimeCountersV2,
    ) -> "BacktestNoRiskExactParityRuntimePlanV2":
        """
        Build one first-class parity runtime plan from the shared planner base output.

        Args:
            runtime_plan: Base planner output to promote into first-class parity plan.
            no_risk_parity_counters: Additive parity counters attached to the promoted plan.
        Returns:
            BacktestNoRiskExactParityRuntimePlanV2: Promoted parity-first runtime plan.
        Assumptions:
            Promotion keeps deterministic enumeration/state contracts unchanged and only adds
            parity-specific runtime metadata.
        Raises:
            ValueError: Propagated if promoted payload violates parity invariants.
        Side Effects:
            None.
        """
        return cls(
            indicator_plans=runtime_plan.indicator_plans,
            signal_axes=runtime_plan.signal_axes,
            risk_variants=runtime_plan.risk_variants,
            execution_profile=runtime_plan.execution_profile,
            instrument_id_literal=runtime_plan.instrument_id_literal,
            timeframe_code=runtime_plan.timeframe_code,
            direction_mode=runtime_plan.direction_mode,
            sizing_mode=runtime_plan.sizing_mode,
            execution_params=runtime_plan.execution_params,
            stage_a_variants_total=runtime_plan.stage_a_variants_total,
            stage_b_variants_total=runtime_plan.stage_b_variants_total,
            estimated_memory_bytes=runtime_plan.estimated_memory_bytes,
            indicator_estimate_calls=runtime_plan.indicator_estimate_calls,
            signal_features_access=runtime_plan.signal_features_access,
            adaptive_selector_decision=runtime_plan.adaptive_selector_decision,
            stage_cost_model=runtime_plan.stage_cost_model,
            launch_budget_evidence=runtime_plan.launch_budget_evidence,
            parity_classification=runtime_plan.parity_classification,
            no_risk_parity_counters=no_risk_parity_counters,
        )

    def uses_hybrid_reduced_plan_contract(self) -> bool:
        """
        Assert that first-class parity runtime plans are independent from hybrid reduced plans.

        Args:
            None.
        Returns:
            bool: Always `False` for first-class parity plans.
        Assumptions:
            D2 requires parity runtime plans to stop depending on reduced shortlist wrappers.
        Raises:
            None.
        Side Effects:
            None.
        """
        return False

    def parity_runtime_counters(self) -> Mapping[str, object]:
        """
        Export additive parity runtime counters attached to this first-class parity plan.

        Args:
            None.
        Returns:
            Mapping[str, object]: Immutable additive parity counters.
        Assumptions:
            Counters remain internal benchmark-facing metadata.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.no_risk_parity_counters is None:
            raise ValueError(
                "BacktestNoRiskExactParityRuntimePlanV2 requires no_risk_parity_counters"
            )
        return self.no_risk_parity_counters.as_mapping()


def runtime_plan_requires_hierarchical_shortlist_runtime_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> bool:
    """
    Resolve whether one runtime plan should enter hierarchical reduced-plan handoff.

    Args:
        runtime_plan: Prepared runtime plan candidate for sync/worker orchestration.
    Returns:
        bool: `True` only when hybrid shortlist runtime is enabled and the plan still depends on
            reduced-plan semantics.
    Assumptions:
        First-class parity plans must bypass hierarchical reduction even if historical branches
        previously used profile-level checks only. Test doubles may provide a duck-typed runtime
        plan exposing only `execution_profile`.
    Raises:
        TypeError: If a duck-typed plan exposes non-callable
            `uses_hybrid_reduced_plan_contract`.
    Side Effects:
        None.
    """
    uses_hybrid_reduced_contract = getattr(
        runtime_plan,
        "uses_hybrid_reduced_plan_contract",
        None,
    )
    if uses_hybrid_reduced_contract is not None:
        if not callable(uses_hybrid_reduced_contract):
            raise TypeError(
                "runtime_plan uses_hybrid_reduced_plan_contract attribute must be callable"
            )
        if bool(uses_hybrid_reduced_contract()):
            return True
    if isinstance(runtime_plan, BacktestNoRiskExactParityRuntimePlanV2):
        return False
    profile = getattr(runtime_plan, "execution_profile", None)
    if profile is None:
        return False
    return execution_profile_uses_hierarchical_shortlist_runtime_v2(
        profile=profile,
    )


class BacktestArtifactRuntimePlannerV2:
    """
    Build deterministic artifact-backed runtime plans and enforce guard budgets.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
        stage_a_cost_units: int | None = None,
        requested_execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
        indicator_ids: tuple[str, ...] | None = None,
        launch_budget_evidence: ExecutionProfileLaunchBudgetEvidenceV2 | None = None,
    ) -> ExecutionProfileV2:
        """
        Resolve the effective execution profile from deterministic planner cost evidence.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - apps/api/wiring/modules/backtest.py

        Args:
            stage_a_variants_total: Optional prepared Stage A variants count for future policy.
            stage_b_variants_total: Optional prepared Stage B variants count.
            estimated_memory_bytes: Optional prepared deterministic memory estimate.
            stage_a_cost_units:
                Optional retained-frontier-aware Stage A cost units collapsing row prefilter,
                combo prefilter, and retained-candidate exact work under stable public
                `stage_a` semantics for adaptive classification only.
            requested_execution_profile_mode:
                Optional internal-only requested execution profile mode supplied explicitly by the
                live caller. When present, automatic exact-profile selection is bypassed and the
                requested profile is validated against live runtime gating plus sync launch
                budgets.
            indicator_ids:
                Optional deterministic indicator ids from the prepared plan. These stay internal
                and are used only to validate `hybrid_family` plugin availability.
            launch_budget_evidence:
                Optional explicit sync-launch workload evidence prepared by the planner. Canonical
                no-risk requests may narrow Stage A and memory evidence here without changing
                public request/response contracts or full-budget guard enforcement.
        Returns:
            ExecutionProfileV2: Selected execution profile for the prepared request.
        Assumptions:
            Requested exact-profile overrides keep precedence, while requested hybrid overrides are
            allowed only when selector rollout has reached explicit `opt_in` or `active`
            semantics. The parity-only `exact_no_risk_parity` override additionally requires
            explicit planner-provided `no_risk_terminal` launch evidence so sync admission stays
            tied to narrowed exact workload math instead of raw-grid fallback totals. Persisted
            read-model metadata is not a requested override input; automatic selection uses the
            typed adaptive selector only when planning evidence is available.
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
            _validate_requested_parity_launch_evidence_v2(
                requested_profile=requested_profile,
                launch_budget_evidence=launch_budget_evidence,
            )
            requested_launch_budget_evidence = _resolved_launch_budget_evidence_v2(
                stage_a_variants_total=stage_a_variants_total,
                stage_b_variants_total=stage_b_variants_total,
                estimated_memory_bytes=estimated_memory_bytes,
                launch_budget_evidence=launch_budget_evidence,
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
                and requested_launch_budget_evidence is not None
                and not requested_profile.launch_budget.allows_evidence(
                    evidence=requested_launch_budget_evidence,
                )
            ):
                raise _background_auto_required_error_v2(
                    execution_profile_mode=requested_profile.mode,
                    stage_a_variants_total=(
                        requested_launch_budget_evidence.stage_a_variants_total
                    ),
                    stage_b_variants_total=(
                        requested_launch_budget_evidence.stage_b_variants_total
                    ),
                    estimated_memory_bytes=(
                        requested_launch_budget_evidence.estimated_memory_bytes
                    ),
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
            stage_a_cost_units=stage_a_cost_units,
            indicator_ids=indicator_ids,
        )[0]

    def _resolve_execution_profile_selection(
        self,
        *,
        stage_a_variants_total: int,
        stage_b_variants_total: int,
        estimated_memory_bytes: int,
        stage_a_cost_units: int | None = None,
        indicator_ids: tuple[str, ...] | None = None,
        launch_budget_evidence: ExecutionProfileLaunchBudgetEvidenceV2 | None = None,
        parity_classification: ExecutionProfileParityClassificationV2 | None = None,
    ) -> tuple[ExecutionProfileV2, AdaptiveSelectorDecisionV2]:
        """
        Resolve both the effective execution profile and the internal selector decision payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            stage_a_variants_total: Prepared Stage A variants count.
            stage_b_variants_total: Prepared Stage B variants count.
            estimated_memory_bytes: Deterministic memory estimate.
            stage_a_cost_units:
                Optional retained-frontier-aware Stage A cost units collapsing row prefilter,
                combo prefilter, and retained-candidate exact work under stable public
                `stage_a` semantics for adaptive classification only.
            indicator_ids: Optional deterministic indicator ids from the prepared plan.
            launch_budget_evidence:
                Optional explicit sync-launch workload evidence. Parity-first classified requests
                rely on this narrowed evidence instead of raw grid totals when validating
                `sync_inline` admission against the dedicated parity exact profile budget.
            parity_classification:
                Optional parity-first classification evidence for canonical no-risk exact
                workloads. When present, the selector must exclude this workload from hybrid
                rollout recommendations.
        Returns:
            tuple[ExecutionProfileV2, AdaptiveSelectorDecisionV2]: Effective execution profile
                plus the full selector decision payload for internal inspection.
        Assumptions:
            This helper is used only after guard math is available, so the selector can stay
            deterministic and free of runtime IO.
        Raises:
            RoehubError: If sync launch budgets are exceeded and background routing is required.
            ValueError: If parity-classified sync admission is attempted without launch evidence.
        Side Effects:
            None.
        """
        if parity_classification is not None:
            parity_profile = self._execution_profiles.profile_for_mode(
                mode="exact_no_risk_parity"
            )
            if self._launch_budget_mode == "sync_inline":
                if launch_budget_evidence is None:
                    raise ValueError(
                        "Parity-classified execution profile selection requires "
                        "launch_budget_evidence"
                    )
                if not parity_profile.launch_budget.allows_evidence(
                    evidence=launch_budget_evidence,
                ):
                    raise _background_auto_required_error_v2(
                        execution_profile_mode=parity_profile.mode,
                        stage_a_variants_total=launch_budget_evidence.stage_a_variants_total,
                        stage_b_variants_total=launch_budget_evidence.stage_b_variants_total,
                        estimated_memory_bytes=launch_budget_evidence.estimated_memory_bytes,
                    )
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
                stage_a_cost_units=stage_a_cost_units,
                parity_classification=parity_classification,
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

    def plan(
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
        Resolve one deterministic shared runtime plan with guard checks.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py

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
            Sync and worker callers both consume this shared planner surface so
            `execution_profiles` and `adaptive_selector_policy` remain centralized here.
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
        stage_cost_model = _build_stage_cost_model_v2(
            indicator_plans=indicator_plans,
            signal_axes=signal_axes,
            stage_a_variants_total=stage_a_variants_total,
            shortlist_len=shortlist_len,
        )
        stage_b_variants_total = shortlist_len * len(risk_variants)
        launch_budget_evidence = _build_launch_budget_evidence_v2(
            bars=len(candles.ts_open),
            stage_a_variants_total=stage_a_variants_total,
            stage_b_variants_total=stage_b_variants_total,
            estimated_memory_bytes=estimated_memory_bytes,
            stage_cost_model=stage_cost_model,
            risk_variants=risk_variants,
        )
        parity_classification = _build_parity_classification_v2(
            stage_cost_model=stage_cost_model,
            risk_variants=risk_variants,
        )

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
                    stage_a_cost_units=stage_cost_model.stage_a_cost_units,
                    indicator_ids=tuple(plan.indicator_id for plan in indicator_plans),
                    launch_budget_evidence=launch_budget_evidence,
                    parity_classification=parity_classification,
                )
            )
        else:
            execution_profile = self.resolve_execution_profile(
                stage_a_variants_total=stage_a_variants_total,
                stage_b_variants_total=stage_b_variants_total,
                estimated_memory_bytes=estimated_memory_bytes,
                requested_execution_profile_mode=requested_execution_profile_mode,
                indicator_ids=tuple(plan.indicator_id for plan in indicator_plans),
                launch_budget_evidence=launch_budget_evidence,
            )

        runtime_plan = BacktestArtifactRuntimePlanV2(
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
            stage_cost_model=stage_cost_model,
            launch_budget_evidence=launch_budget_evidence,
            parity_classification=parity_classification,
        )
        return _build_first_class_parity_runtime_plan_v2(
            runtime_plan=runtime_plan,
        )

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
        Preserve the historical planner API by forwarding to `plan(...)`.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py

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
            Existing callers may still use `build(...)`, but shared planner ownership remains in
            `plan(...)`.
        Raises:
            RoehubError: If guard limits are exceeded.
            ValueError: If request axes are invalid.
        Side Effects:
            Calls `plan(...)`, which may call `indicator_compute.estimate(...)` once per
            indicator block.
        """
        return self.plan(
            template=template,
            candles=candles,
            indicator_compute=indicator_compute,
            preselect=preselect,
            requested_execution_profile_mode=requested_execution_profile_mode,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        )

    def build_sync_inline_plan(
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
        Build one sync-inline plan while preserving first-class parity runtime-plan promotion.

        Args:
            template: Resolved backtest template payload.
            candles: Warmup-inclusive request-timeframe candles from pinned artifacts.
            indicator_compute: Indicator estimate port used for compute-axis materialization.
            preselect: Stage A shortlist size before Stage B expansion.
            requested_execution_profile_mode:
                Optional internal execution profile override for sync launch validation.
            defaults_provider: Optional provider for compute/signal fallback defaults.
            max_variants_per_compute: Variants guard budget.
            max_compute_bytes_total: Memory guard budget.
        Returns:
            BacktestArtifactRuntimePlanV2: Prepared sync-inline runtime plan.
        Assumptions:
            This helper is valid only for planners configured with
            `launch_budget_mode='sync_inline'`.
        Raises:
            ValueError: If planner launch budget mode is not `sync_inline`.
            RoehubError: If guard limits are exceeded.
        Side Effects:
            Delegates to `plan(...)`, which may call `indicator_compute.estimate(...)`.
        """
        if self._launch_budget_mode != "sync_inline":
            raise ValueError(
                "BacktestArtifactRuntimePlannerV2.build_sync_inline_plan requires "
                "launch_budget_mode='sync_inline'"
            )
        return self.plan(
            template=template,
            candles=candles,
            indicator_compute=indicator_compute,
            preselect=preselect,
            requested_execution_profile_mode=requested_execution_profile_mode,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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


def _risk_variant_is_no_risk_v2(
    *,
    risk_variant: BacktestRiskVariantV2,
) -> bool:
    """
    Check whether one prepared Stage B risk cell matches the canonical no-risk class.

    Args:
        risk_variant: Prepared runtime-plan risk cell.
    Returns:
        bool: `True` when both TP and SL are disabled and their percentages stay null.
    Assumptions:
        No-risk terminal-path routing must stay deterministic and shared across sync and worker
        orchestration, so the classifier relies only on the prepared runtime-plan payload.
    Raises:
        None.
    Side Effects:
        None.
    """
    risk_params = risk_variant.risk_params
    return (
        risk_params.get("sl_enabled") is False
        and risk_params.get("sl_pct") is None
        and risk_params.get("tp_enabled") is False
        and risk_params.get("tp_pct") is None
    )


def _risk_variants_use_no_risk_terminal_path_v2(
    *,
    risk_variants: tuple[BacktestRiskVariantV2, ...],
) -> bool:
    """
    Classify whether one prepared risk-variant set belongs to the canonical no-risk class.

    Args:
        risk_variants: Prepared Stage B risk variants for one runtime plan.
    Returns:
        bool: `True` when the runtime stays on the canonical single-cell no-risk terminal path.
    Assumptions:
        The no-risk class remains a single disabled-risk cell, which keeps sync launch budgeting
        and runtime-shape classification deterministic for `NR2`-style requests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return len(risk_variants) == 1 and _risk_variant_is_no_risk_v2(
        risk_variant=risk_variants[0]
    )


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
    return _estimate_memory_bytes_from_variant_counts_v2(
        bars=bars,
        indicator_variants_total=sum(plan.variants for plan in indicator_plans),
    )


def _estimate_memory_bytes_from_variant_counts_v2(
    *,
    bars: int,
    indicator_variants_total: int,
) -> int:
    """
    Estimate total runtime memory bytes from aggregate indicator-variant cardinality.

    Args:
        bars: Number of warmup-inclusive request-timeframe bars.
        indicator_variants_total: Aggregate indicator-variant rows contributing float32 tensors.
    Returns:
        int: Estimated total memory bytes with reserve.
    Assumptions:
        Planner memory budgeting stays explicit and deterministic, so both raw-grid and no-risk
        launch evidence share the same reserve policy while varying only the aggregate retained
        indicator cardinality.
    Raises:
        ValueError: If bars is non-positive or aggregate variant count is negative.
    Side Effects:
        None.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0 for memory estimate")
    if indicator_variants_total < 0:
        raise ValueError("indicator_variants_total must be >= 0 for memory estimate")

    bytes_candles = bars * _CANDLES_BYTES_PER_STEP
    bytes_indicators = bars * indicator_variants_total * _FLOAT32_BYTES
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


def _build_stage_cost_model_v2(
    *,
    indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
    stage_a_variants_total: int,
    shortlist_len: int,
) -> BacktestRuntimeStageCostModelV2:
    """
    Build retained-frontier Stage A cost evidence without widening public stage literals.

    Args:
        indicator_plans: Deterministic indicator plans used for compute-row cardinality.
        signal_axes: Deterministic signal-axis plans that survive row prefilter unchanged.
        stage_a_variants_total: Public Stage A cartesian total used by runtime progress counters.
        shortlist_len: Final Stage A shortlist envelope before Stage B risk expansion.
    Returns:
        BacktestRuntimeStageCostModelV2: Internal cost model for planner-only classification.
    Assumptions:
        Public `stage_a` still maps to the full outward stage while internal cost evidence tracks
        row prefilter, retained-row tensor envelope, combo prefilter, and retained-candidate
        exact work separately.
    Raises:
        ValueError: If one derived retained-frontier total is non-positive.
    Side Effects:
        None.
    """
    signal_variants_total = _signal_variants_total_for_axes_v2(signal_axes=signal_axes)
    row_variants = tuple(plan.variants for plan in indicator_plans)
    target_compute_variants = max(1, int(math.ceil(shortlist_len / signal_variants_total)))
    retained_row_limits = _retained_row_limits_for_stage_cost_model_v2(
        row_variants=row_variants,
        target_compute_variants=target_compute_variants,
    )
    narrowed_compute_variants_total = int(math.prod(retained_row_limits))
    combo_prefilter_variants_total = min(
        stage_a_variants_total,
        narrowed_compute_variants_total * signal_variants_total,
    )
    retained_exact_candidates_total = min(
        stage_a_variants_total,
        max(
            shortlist_len,
            shortlist_len * _COMBO_PROXY_PREFILTER_SURVIVOR_MULTIPLIER_V2,
        ),
    )
    row_prefilter_rows_total = sum(row_variants)
    stage_a_cost_units = (
        row_prefilter_rows_total * _ROW_PREFILTER_COST_WEIGHT_V2
        + combo_prefilter_variants_total * _COMBO_PREFILTER_COST_WEIGHT_V2
        + retained_exact_candidates_total * _RETAINED_STAGE_A_EXACT_COST_WEIGHT_V2
    )
    return BacktestRuntimeStageCostModelV2(
        row_prefilter_rows_total=row_prefilter_rows_total,
        retained_row_variants_total=sum(retained_row_limits),
        combo_prefilter_variants_total=combo_prefilter_variants_total,
        retained_exact_candidates_total=retained_exact_candidates_total,
        stage_a_cost_units=stage_a_cost_units,
        retained_rows_per_indicator=retained_row_limits,
        narrowed_compute_variants_total=narrowed_compute_variants_total,
    )


def _build_launch_budget_evidence_v2(
    *,
    bars: int,
    stage_a_variants_total: int,
    stage_b_variants_total: int,
    estimated_memory_bytes: int,
    stage_cost_model: BacktestRuntimeStageCostModelV2,
    risk_variants: tuple[BacktestRiskVariantV2, ...],
) -> ExecutionProfileLaunchBudgetEvidenceV2:
    """
    Build explicit sync-launch workload evidence aligned to the prepared terminal runtime shape.

    Args:
        bars: Warmup-inclusive request-timeframe bars count.
        stage_a_variants_total: Raw prepared Stage A cartesian total.
        stage_b_variants_total: Prepared Stage B variants total before runtime execution.
        estimated_memory_bytes: Raw deterministic planner memory estimate.
        stage_cost_model: Internal retained-frontier stage-cost model for the same plan.
        risk_variants: Prepared Stage B risk variants for the same plan.
    Returns:
        ExecutionProfileLaunchBudgetEvidenceV2: Explicit sync-launch evidence for requested
            profile gating.
    Assumptions:
        Canonical no-risk requests may use narrowed retained-frontier Stage A evidence and a
        no-risk-aligned memory estimate while the full-budget compute guards stay unchanged.
    Raises:
        ValueError: If one derived evidence field is invalid.
    Side Effects:
        None.
    """
    if _risk_variants_use_no_risk_terminal_path_v2(risk_variants=risk_variants):
        return ExecutionProfileLaunchBudgetEvidenceV2(
            stage_a_variants_total=stage_cost_model.combo_prefilter_variants_total,
            stage_b_variants_total=stage_b_variants_total,
            estimated_memory_bytes=_estimate_memory_bytes_from_variant_counts_v2(
                bars=bars,
                indicator_variants_total=stage_cost_model.retained_row_variants_total,
            ),
            workload_class="no_risk_terminal",
        )
    return ExecutionProfileLaunchBudgetEvidenceV2(
        stage_a_variants_total=stage_a_variants_total,
        stage_b_variants_total=stage_b_variants_total,
        estimated_memory_bytes=estimated_memory_bytes,
        workload_class="raw_grid",
    )


def _build_parity_classification_v2(
    *,
    stage_cost_model: BacktestRuntimeStageCostModelV2,
    risk_variants: tuple[BacktestRiskVariantV2, ...],
) -> ExecutionProfileParityClassificationV2 | None:
    """
    Build deterministic parity-first classification evidence for canonical no-risk workloads.

    Args:
        stage_cost_model: Internal retained-frontier cost model for the prepared planner path.
        risk_variants: Prepared Stage B risk variants for the same request.
    Returns:
        ExecutionProfileParityClassificationV2 | None:
            Explicit parity-first classification evidence for canonical no-risk exact workloads,
            otherwise `None`.
    Assumptions:
        Only the single disabled-risk terminal path with canonical two-indicator retained-row
        evidence should carry parity-first classification, and its debug reason should remain
        compact but deterministic for planner/selector review.
    Raises:
        ValueError: Propagated if the derived parity-classification payload is invalid.
    Side Effects:
        None.
    """
    if not _risk_variants_use_no_risk_terminal_path_v2(risk_variants=risk_variants):
        return None
    low_indicator_block_cardinality = (
        len(stage_cost_model.retained_rows_per_indicator) == 2
    )
    narrowed_retained_row_evidence = bool(stage_cost_model.retained_rows_per_indicator)
    notebook_shaped_cost_units = (
        stage_cost_model.narrowed_compute_variants_total is not None
    )
    if not (
        low_indicator_block_cardinality
        and narrowed_retained_row_evidence
        and notebook_shaped_cost_units
    ):
        return None
    return ExecutionProfileParityClassificationV2(
        parity_class="parity_first_no_risk_exact",
        disabled_risk_single_cell=True,
        low_indicator_block_cardinality=low_indicator_block_cardinality,
        narrowed_retained_row_evidence=narrowed_retained_row_evidence,
        notebook_shaped_cost_units=notebook_shaped_cost_units,
        nr2_classification_reason=(
            "canonical NR2 no-risk single-cell parity class; "
            f"retained_rows={stage_cost_model.retained_row_variants_total}; "
            f"combo_prefilter_variants={stage_cost_model.combo_prefilter_variants_total}"
        ),
    )


def _build_first_class_parity_runtime_plan_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> BacktestArtifactRuntimePlanV2:
    """
    Promote canonical no-risk parity workloads into a first-class runtime-plan contract.

    Args:
        runtime_plan: Shared planner output before parity-specific promotion.
    Returns:
        BacktestArtifactRuntimePlanV2: First-class parity plan for canonical no-risk workloads,
            otherwise the input runtime plan unchanged.
    Assumptions:
        D2 keeps Stage A semantics unchanged and only reifies parity runtime-shape ownership as a
        dedicated contract.
    Raises:
        ValueError: If parity profile invariants drift and required stage-cost evidence is absent.
    Side Effects:
        None.
    """
    if runtime_plan.execution_profile.mode != "exact_no_risk_parity":
        return runtime_plan
    if not runtime_plan.uses_no_risk_terminal_path():
        return runtime_plan
    if runtime_plan.parity_classification is None:
        return runtime_plan
    if runtime_plan.stage_cost_model is None:
        raise ValueError(
            "First-class parity runtime plan promotion requires stage_cost_model"
        )
    return BacktestNoRiskExactParityRuntimePlanV2.from_runtime_plan(
        runtime_plan=runtime_plan,
        no_risk_parity_counters=_build_no_risk_exact_parity_runtime_counters_v2(
            runtime_plan=runtime_plan,
            stage_cost_model=runtime_plan.stage_cost_model,
        ),
    )


def _build_no_risk_exact_parity_runtime_counters_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    stage_cost_model: BacktestRuntimeStageCostModelV2,
) -> BacktestNoRiskExactParityRuntimeCountersV2:
    """
    Build deterministic additive counters for first-class no-risk exact parity runtime plans.

    Args:
        runtime_plan: Prepared runtime plan eligible for parity promotion.
        stage_cost_model: Internal retained-frontier stage-cost model for the same plan.
    Returns:
        BacktestNoRiskExactParityRuntimeCountersV2: Additive deterministic parity counters.
    Assumptions:
        Counters remain benchmark-facing internal metadata and do not modify public API payloads.
    Raises:
        ValueError: If per-indicator retained-row evidence drifts from plan indicator ordering.
    Side Effects:
        None.
    """
    retained_rows_per_indicator = stage_cost_model.retained_rows_per_indicator
    if len(retained_rows_per_indicator) != len(runtime_plan.indicator_plans):
        raise ValueError(
            "Parity runtime counters require retained rows for every indicator plan"
        )
    if len(retained_rows_per_indicator) == 0:
        raise ValueError(
            "Parity runtime counters require at least one retained-row counter"
        )
    narrowed_compute_combo_total = stage_cost_model.narrowed_compute_variants_total
    if narrowed_compute_combo_total is None:
        signal_total = max(1, runtime_plan.signal_variants_total())
        narrowed_compute_combo_total = max(
            1,
            int(stage_cost_model.combo_prefilter_variants_total // signal_total),
        )
    return BacktestNoRiskExactParityRuntimeCountersV2(
        retained_rows_per_indicator=tuple(
            BacktestParityRetainedRowsCounterV2(
                indicator_id=plan.indicator_id,
                retained_rows=retained_rows_per_indicator[position],
            )
            for position, plan in enumerate(runtime_plan.indicator_plans)
        ),
        retained_rows_total=stage_cost_model.retained_row_variants_total,
        narrowed_combo_total=stage_cost_model.combo_prefilter_variants_total,
        narrowed_compute_combo_total=narrowed_compute_combo_total,
        no_risk_finalization_count=runtime_plan.stage_b_variants_total,
        exact_replay_count=0,
        deterministic_combo_ordering="stage_a_index",
        stage_b_execution_mode=runtime_plan.stage_b_execution_mode(),
    )


def _resolved_launch_budget_evidence_v2(
    *,
    stage_a_variants_total: int | None,
    stage_b_variants_total: int | None,
    estimated_memory_bytes: int | None,
    launch_budget_evidence: ExecutionProfileLaunchBudgetEvidenceV2 | None,
) -> ExecutionProfileLaunchBudgetEvidenceV2 | None:
    """
    Resolve the explicit launch-budget evidence used by requested sync profile gating.

    Args:
        stage_a_variants_total: Raw prepared Stage A cartesian total, if available.
        stage_b_variants_total: Raw prepared Stage B variants total, if available.
        estimated_memory_bytes: Raw deterministic planner memory estimate, if available.
        launch_budget_evidence: Optional explicit planner-prepared launch-budget evidence.
    Returns:
        ExecutionProfileLaunchBudgetEvidenceV2 | None:
            Explicit evidence when enough inputs exist, otherwise `None`.
    Assumptions:
        Requested sync launch gating should prefer explicit planner evidence and fall back to raw
        request-shape totals only when no narrower runtime-shape evidence exists.
    Raises:
        ValueError: If fallback raw evidence is materially incomplete.
    Side Effects:
        None.
    """
    if launch_budget_evidence is not None:
        return launch_budget_evidence
    if (
        stage_a_variants_total is None
        or stage_b_variants_total is None
        or estimated_memory_bytes is None
    ):
        return None
    return ExecutionProfileLaunchBudgetEvidenceV2(
        stage_a_variants_total=stage_a_variants_total,
        stage_b_variants_total=stage_b_variants_total,
        estimated_memory_bytes=estimated_memory_bytes,
        workload_class="raw_grid",
    )


def _validate_requested_parity_launch_evidence_v2(
    *,
    requested_profile: ExecutionProfileV2,
    launch_budget_evidence: ExecutionProfileLaunchBudgetEvidenceV2 | None,
) -> None:
    """
    Validate that the parity-only requested profile uses explicit narrowed no-risk evidence.

    Args:
        requested_profile: Resolved explicit requested execution profile.
        launch_budget_evidence: Optional planner-produced sync launch-budget evidence.
    Returns:
        None.
    Assumptions:
        `exact_no_risk_parity` is reserved for the canonical no-risk parity class and therefore
        must never derive sync admission from raw-grid fallback totals.
    Raises:
        ValueError: If the parity-only requested profile is used without explicit
            `no_risk_terminal` launch evidence.
    Side Effects:
        None.
    """
    if requested_profile.mode != "exact_no_risk_parity":
        return
    if launch_budget_evidence is None:
        raise ValueError(
            "Requested exact_no_risk_parity execution profile requires "
            "launch_budget_evidence"
        )
    if launch_budget_evidence.workload_class != "no_risk_terminal":
        raise ValueError(
            "Requested exact_no_risk_parity execution profile requires "
            "no_risk_terminal launch_budget_evidence"
        )


def _signal_variants_total_for_axes_v2(
    *,
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
) -> int:
    """
    Compute deterministic signal-axis cardinality for retained-frontier budgeting.

    Args:
        signal_axes: Deterministic signal-axis plans in planner order.
    Returns:
        int: Product of signal-axis cardinalities (`1` when no signal axes are configured).
    Assumptions:
        Row prefilter preserves signal-axis combinations unchanged, so planner budgeting uses the
        same multiplicative signal cardinality as the live Stage A builder.
    Raises:
        ValueError: If one signal axis has no values.
    Side Effects:
        None.
    """
    signal_variants_total = 1
    for axis in signal_axes:
        axis_cardinality = len(axis.values)
        if axis_cardinality <= 0:
            raise ValueError("signal axes must define at least one value")
        signal_variants_total *= axis_cardinality
    return signal_variants_total


def _retained_row_limits_for_stage_cost_model_v2(
    *,
    row_variants: tuple[int, ...],
    target_compute_variants: int,
) -> tuple[int, ...]:
    """
    Mirror deterministic retained-row budgeting used by the live row-prefilter frontier.

    Args:
        row_variants: Indicator-local row counts in planner order.
        target_compute_variants: Minimum compute-variant budget that should survive row pruning.
    Returns:
        tuple[int, ...]: Deterministic retained-row caps aligned to `row_variants`.
    Assumptions:
        Planner cost evidence must stay shape-compatible with the live Stage A builder without
        importing the runtime module and creating an import cycle.
    Raises:
        ValueError: If one row count or the target budget is non-positive.
    Side Effects:
        None.
    """
    if target_compute_variants <= 0:
        raise ValueError("target_compute_variants must be > 0")
    if len(row_variants) == 0:
        return ()
    if any(variants <= 0 for variants in row_variants):
        raise ValueError("row_variants must all be > 0")
    base_limit = max(
        1,
        int(math.ceil(target_compute_variants ** (1.0 / len(row_variants)))),
    )
    retained_limits = [min(variants, base_limit) for variants in row_variants]
    retained_product = math.prod(retained_limits)
    while retained_product < target_compute_variants:
        grew = False
        for index, variants in enumerate(row_variants):
            if retained_limits[index] >= variants:
                continue
            retained_limits[index] += 1
            retained_product = math.prod(retained_limits)
            grew = True
            if retained_product >= target_compute_variants:
                break
        if not grew:
            break
    return tuple(retained_limits)


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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
    "BacktestNoRiskExactParityRuntimeCountersV2",
    "BacktestNoRiskExactParityRuntimePlanV2",
    "BacktestParityRetainedRowsCounterV2",
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
    "runtime_plan_requires_hierarchical_shortlist_runtime_v2",
]
