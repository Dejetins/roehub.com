"""Deterministic adaptive selector for execution-profile cost-model routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from .execution_profile_v2 import (
    ExecutionProfileModeLiteralV2,
    ExecutionProfileParityClassificationV2,
    ExecutionProfilesCatalogV2,
    ExecutionProfileV2,
    execution_profile_uses_hierarchical_shortlist_runtime_v2,
    validate_execution_profile_mode_v2,
)

if TYPE_CHECKING:
    from .family_plugins.registry_v2 import (
        FamilyPluginRegistryResolutionV2,
        FamilyPluginRegistryV2,
    )

type AdaptiveSelectorPolicyModeLiteralV2 = Literal[
    "disabled",
    "shadow",
    "opt_in",
    "active",
]
type AdaptiveSelectorRuntimeModeLiteralV2 = Literal[
    "sync_inline",
    "background_capable",
]

ALLOWED_ADAPTIVE_SELECTOR_POLICY_MODES_V2: tuple[
    AdaptiveSelectorPolicyModeLiteralV2, ...
] = (
    "disabled",
    "shadow",
    "opt_in",
    "active",
)
ALLOWED_ADAPTIVE_SELECTOR_RUNTIME_MODES_V2: tuple[
    AdaptiveSelectorRuntimeModeLiteralV2,
    ...,
] = (
    "sync_inline",
    "background_capable",
)
_ADAPTIVE_SELECTOR_POLICY_MODE_PRIORITY_V2: dict[
    AdaptiveSelectorPolicyModeLiteralV2,
    int,
] = {
    "disabled": 0,
    "shadow": 1,
    "opt_in": 2,
    "active": 3,
}


def validate_adaptive_selector_policy_mode_v2(
    *,
    value: str,
) -> AdaptiveSelectorPolicyModeLiteralV2:
    """
    Validate one adaptive-selector rollout mode literal.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        value: Raw selector policy mode literal from config or tests.
    Returns:
        AdaptiveSelectorPolicyModeLiteralV2: Canonical approved policy mode.
    Assumptions:
        Rollout remains explicit through `disabled`, `shadow`, `opt_in`, and `active` modes so
        Milestone F2 can distinguish recommendation-only shadow behavior, internal opt-in live
        evaluation, and full active rollout without redefining selector contracts.
    Raises:
        ValueError: If the literal is blank or outside the approved policy surface.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in ALLOWED_ADAPTIVE_SELECTOR_POLICY_MODES_V2:
        raise ValueError(
            "Adaptive selector policy mode must be one of "
            f"{ALLOWED_ADAPTIVE_SELECTOR_POLICY_MODES_V2}, got {value!r}"
        )
    return cast(AdaptiveSelectorPolicyModeLiteralV2, normalized_value)


def validate_adaptive_selector_runtime_mode_v2(
    *,
    value: str,
) -> AdaptiveSelectorRuntimeModeLiteralV2:
    """
    Validate one planner-derived runtime mode literal used by the adaptive selector.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        value: Raw runtime mode literal.
    Returns:
        AdaptiveSelectorRuntimeModeLiteralV2: Canonical planner runtime mode.
    Assumptions:
        Adaptive routing depends only on whether the planner must stay `sync_inline` or may use
        one background-capable runtime path.
    Raises:
        ValueError: If the runtime mode is blank or unsupported.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if normalized_value not in ALLOWED_ADAPTIVE_SELECTOR_RUNTIME_MODES_V2:
        raise ValueError(
            "Adaptive selector runtime mode must be one of "
            f"{ALLOWED_ADAPTIVE_SELECTOR_RUNTIME_MODES_V2}, got {value!r}"
        )
    return cast(AdaptiveSelectorRuntimeModeLiteralV2, normalized_value)


@dataclass(frozen=True, slots=True)
class AdaptiveSelectorCandidatePolicyV2:
    """
    Cost-model thresholds for promoting one hybrid execution-profile candidate.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    min_grid_cardinality: int
    min_stage_a_variants_total: int
    min_stage_b_variants_total: int
    min_estimated_memory_bytes: int
    rollout_mode: AdaptiveSelectorPolicyModeLiteralV2 = "active"
    minimum_exceeded_signals: int = 3

    def __post_init__(self) -> None:
        """
        Validate one additive hybrid-promotion policy surface.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Hybrid promotion remains deterministic: the selector counts exceeded cost-model
            signals across `grid cardinality`, `stage_a`, `stage_b`, and memory rather than
            depending on wall-clock measurements, while `rollout_mode` may keep one candidate in
            `shadow`, `opt_in`, or `disabled` even when the env-level selector mode is broader.
        Raises:
            ValueError: If one threshold is non-positive, `rollout_mode` is invalid, or the
                required signal count is outside the four available cost-model dimensions.
        Side Effects:
            Normalizes `rollout_mode`.
        """
        for field_name in (
            "min_grid_cardinality",
            "min_stage_a_variants_total",
            "min_stage_b_variants_total",
            "min_estimated_memory_bytes",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"AdaptiveSelectorCandidatePolicyV2.{field_name} must be > 0")
        object.__setattr__(
            self,
            "rollout_mode",
            validate_adaptive_selector_policy_mode_v2(value=self.rollout_mode),
        )
        if self.minimum_exceeded_signals <= 0 or self.minimum_exceeded_signals > 4:
            raise ValueError(
                "AdaptiveSelectorCandidatePolicyV2.minimum_exceeded_signals must be between "
                "1 and 4"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveSelectorPolicyV2:
    """
    Startup-validated rollout and cost-model policy for the adaptive selector.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    mode: AdaptiveSelectorPolicyModeLiteralV2 = "disabled"
    hybrid_conservative: AdaptiveSelectorCandidatePolicyV2 = field(
        default_factory=lambda: AdaptiveSelectorCandidatePolicyV2(
            min_grid_cardinality=6000,
            min_stage_a_variants_total=6000,
            min_stage_b_variants_total=40000,
            min_estimated_memory_bytes=805306368,
            rollout_mode="active",
            minimum_exceeded_signals=3,
        )
    )
    hybrid_family: AdaptiveSelectorCandidatePolicyV2 = field(
        default_factory=lambda: AdaptiveSelectorCandidatePolicyV2(
            min_grid_cardinality=12000,
            min_stage_a_variants_total=12000,
            min_stage_b_variants_total=80000,
            min_estimated_memory_bytes=1073741824,
            rollout_mode="active",
            minimum_exceeded_signals=3,
        )
    )

    def __post_init__(self) -> None:
        """
        Validate additive rollout policy and normalize its explicit mode literal.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Selector policy remains conservative in this milestone: rollout mode is explicit and
            family-specific promotion stays stricter than universal `hybrid_conservative`, while
            `opt_in` remains a first-class phase between `shadow` and `active`.
        Raises:
            ValueError: If mode or one nested candidate policy is invalid.
        Side Effects:
            Normalizes `mode` to the approved lower-case literal.
        """
        object.__setattr__(
            self,
            "mode",
            validate_adaptive_selector_policy_mode_v2(value=self.mode),
        )
        if self.hybrid_conservative is None:  # type: ignore[truthy-bool]
            raise ValueError("AdaptiveSelectorPolicyV2.hybrid_conservative is required")
        if self.hybrid_family is None:  # type: ignore[truthy-bool]
            raise ValueError("AdaptiveSelectorPolicyV2.hybrid_family is required")


@dataclass(frozen=True, slots=True)
class AdaptiveSelectorPlanningEvidenceV2:
    """
    Deterministic planning-time evidence consumed by the adaptive execution selector.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    grid_cardinality: int
    stage_a_variants_total: int
    stage_b_variants_total: int
    estimated_memory_bytes: int
    runtime_mode: AdaptiveSelectorRuntimeModeLiteralV2
    indicator_ids: tuple[str, ...] = ()
    stage_a_cost_units: int | None = None
    parity_classification: ExecutionProfileParityClassificationV2 | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize the planning evidence surface used by selector policy.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            All evidence is available before runtime execution and must stay cheap to derive from
            the prepared plan; indicator ids remain internal-only metadata, while
            `stage_a_cost_units` may collapse retained-frontier row/combo/exact work back under
            stable public `stage_a` semantics for hybrid classification only.
        Raises:
            ValueError: If one cost-model dimension is non-positive or indicator ids are blank.
        Side Effects:
            Normalizes `runtime_mode` and deduplicates indicator ids in deterministic order.
        """
        for field_name in (
            "grid_cardinality",
            "stage_a_variants_total",
            "stage_b_variants_total",
            "estimated_memory_bytes",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"AdaptiveSelectorPlanningEvidenceV2.{field_name} must be > 0"
                )
        if self.stage_a_cost_units is not None and self.stage_a_cost_units <= 0:
            raise ValueError(
                "AdaptiveSelectorPlanningEvidenceV2.stage_a_cost_units must be > 0 when "
                "provided"
            )
        object.__setattr__(
            self,
            "runtime_mode",
            validate_adaptive_selector_runtime_mode_v2(value=self.runtime_mode),
        )
        normalized_indicator_ids = tuple(
            sorted(
                {
                    indicator_id.strip().lower()
                    for indicator_id in self.indicator_ids
                    if indicator_id.strip()
                }
            )
        )
        if len(normalized_indicator_ids) != len(
            {indicator_id.strip().lower() for indicator_id in self.indicator_ids}
        ):
            raise ValueError(
                "AdaptiveSelectorPlanningEvidenceV2.indicator_ids must not contain blanks"
            )
        object.__setattr__(self, "indicator_ids", normalized_indicator_ids)

    def effective_stage_a_cost_units(self) -> int:
        """
        Return the Stage A cost signal used by hybrid candidate classification.

        Args:
            None.
        Returns:
            int: Retained-frontier-aware Stage A cost units when present, else the raw Stage A
                variants total.
        Assumptions:
            Public `stage_a` vocabulary remains stable even when planner internals distinguish row
            prefilter, combo prefilter, and retained-candidate exact work.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.stage_a_cost_units or self.stage_a_variants_total


@dataclass(frozen=True, slots=True)
class AdaptiveSelectorCandidateEvaluationV2:
    """
    Compact explainability payload for one execution-profile candidate evaluation.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    mode: ExecutionProfileModeLiteralV2
    eligible: bool
    exceeded_signals: int
    reason: str

    def __post_init__(self) -> None:
        """
        Validate one compact selector-evaluation row.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Evaluation rows stay compact and deterministic so rollout debugging does not depend on
            stepping through planner conditionals manually.
        Raises:
            ValueError: If the mode literal is unsupported, the signal count is invalid, or the
                explanation is blank.
        Side Effects:
            Normalizes `mode` and `reason`.
        """
        if not self.reason.strip():
            raise ValueError("AdaptiveSelectorCandidateEvaluationV2.reason must be non-empty")
        if self.exceeded_signals < 0 or self.exceeded_signals > 4:
            raise ValueError(
                "AdaptiveSelectorCandidateEvaluationV2.exceeded_signals must be between 0 and 4"
            )
        object.__setattr__(
            self,
            "mode",
            validate_execution_profile_mode_v2(value=self.mode),
        )
        object.__setattr__(self, "reason", self.reason.strip())


@dataclass(frozen=True, slots=True)
class AdaptiveSelectorDecisionV2:
    """
    Deterministic selector decision containing both effective and recommended profiles.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    policy_mode: AdaptiveSelectorPolicyModeLiteralV2
    effective_profile: ExecutionProfileV2
    recommended_profile: ExecutionProfileV2
    exact_fallback_profile: ExecutionProfileV2
    recommendation_applied: bool
    requires_background_auto: bool
    candidate_evaluations: tuple[AdaptiveSelectorCandidateEvaluationV2, ...]

    def __post_init__(self) -> None:
        """
        Validate the compact adaptive-selector decision payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Shadow mode may recommend a different profile than the effective one, but every
            decision still retains the exact-only fallback used when rollout is disabled,
            shadow-only, opt-in-only, or otherwise ambiguous.
        Raises:
            ValueError: If the mode literal is invalid, one profile is missing, or no candidate
                evaluations are present.
        Side Effects:
            Normalizes `policy_mode`.
        """
        object.__setattr__(
            self,
            "policy_mode",
            validate_adaptive_selector_policy_mode_v2(value=self.policy_mode),
        )
        if self.effective_profile is None:  # type: ignore[truthy-bool]
            raise ValueError("AdaptiveSelectorDecisionV2.effective_profile is required")
        if self.recommended_profile is None:  # type: ignore[truthy-bool]
            raise ValueError("AdaptiveSelectorDecisionV2.recommended_profile is required")
        if self.exact_fallback_profile is None:  # type: ignore[truthy-bool]
            raise ValueError("AdaptiveSelectorDecisionV2.exact_fallback_profile is required")
        if len(self.candidate_evaluations) == 0:
            raise ValueError(
                "AdaptiveSelectorDecisionV2.candidate_evaluations must be non-empty"
            )


class AdaptiveExecutionSelectorV2(Protocol):
    """
    Contract for deterministic adaptive execution-profile selection.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    def select(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        execution_profiles: ExecutionProfilesCatalogV2,
        policy: AdaptiveSelectorPolicyV2,
    ) -> AdaptiveSelectorDecisionV2:
        """
        Select the effective execution profile from deterministic planning-time evidence.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning-time evidence already available before execution.
            execution_profiles: Startup-validated execution-profile catalog.
            policy: Startup-validated selector rollout policy.
        Returns:
            AdaptiveSelectorDecisionV2: Effective and recommended profile metadata.
        Assumptions:
            Selector logic must stay cheap, deterministic, and free of benchmark fixture reads or
            wall-clock observations on the hot path.
        Raises:
            ValueError: If any contract input is missing or invalid.
        Side Effects:
            None.
        """
        ...


@dataclass(frozen=True, slots=True)
class CostModelAdaptiveExecutionSelectorV2:
    """
    Deterministic adaptive selector that promotes hybrid profiles from planning-time evidence.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py
    """

    family_plugin_registry: FamilyPluginRegistryV2 = field(
        default_factory=lambda: _default_family_plugin_registry_v2()
    )

    def select(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        execution_profiles: ExecutionProfilesCatalogV2,
        policy: AdaptiveSelectorPolicyV2,
    ) -> AdaptiveSelectorDecisionV2:
        """
        Select the effective execution profile using a deterministic cost model.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence prepared by the shared runtime planner.
            execution_profiles: Ordered execution-profile catalog reused by the runtime.
            policy: Explicit selector rollout policy.
        Returns:
            AdaptiveSelectorDecisionV2: Effective and recommended execution-profile metadata.
        Assumptions:
            Exact selection remains the conservative fallback, while hybrid promotion may happen
            only in background-capable runtime mode and only when rollout gates are live, with
            `opt_in` staying recommendation-only unless a separate internal requested-profile
            override path explicitly asks for live hybrid execution.
        Raises:
            ValueError: If one contract input is missing.
        Side Effects:
            None.
        """
        if execution_profiles is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "CostModelAdaptiveExecutionSelectorV2.select requires execution_profiles"
            )
        if policy is None:  # type: ignore[truthy-bool]
            raise ValueError("CostModelAdaptiveExecutionSelectorV2.select requires policy")
        exact_fallback_profile, requires_background_auto, exact_evaluations = (
            self._resolve_exact_fallback(
                evidence=evidence,
                execution_profiles=execution_profiles,
            )
        )
        candidate_evaluations = list(exact_evaluations)
        recommended_profile = exact_fallback_profile

        # Parity-first classification must not be recommended as hybrid rollout
        if evidence.parity_classification is not None:
            return self._select_parity_first_profile(
                evidence=evidence,
                execution_profiles=execution_profiles,
                policy=policy,
                exact_fallback_profile=exact_fallback_profile,
                requires_background_auto=requires_background_auto,
                exact_evaluations=exact_evaluations,
            )

        hybrid_conservative_profile = execution_profiles.profile_for_mode(
            mode="hybrid_conservative"
        )
        hybrid_family_profile = execution_profiles.profile_for_mode(mode="hybrid_family")
        hybrid_conservative_rollout_mode = _bounded_adaptive_selector_policy_mode_v2(
            policy_mode=policy.mode,
            candidate_rollout_mode=policy.hybrid_conservative.rollout_mode,
        )
        hybrid_family_rollout_mode = _bounded_adaptive_selector_policy_mode_v2(
            policy_mode=policy.mode,
            candidate_rollout_mode=policy.hybrid_family.rollout_mode,
        )
        if policy.mode == "disabled":
            hybrid_conservative_evaluation = AdaptiveSelectorCandidateEvaluationV2(
                mode=hybrid_conservative_profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="adaptive selector policy disabled",
            )
            hybrid_family_evaluation = AdaptiveSelectorCandidateEvaluationV2(
                mode=hybrid_family_profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="adaptive selector policy disabled",
            )
        elif hybrid_conservative_rollout_mode == "disabled":
            hybrid_conservative_evaluation = AdaptiveSelectorCandidateEvaluationV2(
                mode=hybrid_conservative_profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="candidate rollout disabled by policy",
            )
            hybrid_family_evaluation = self._evaluate_or_skip_hybrid_candidate(
                evidence=evidence,
                profile=hybrid_family_profile,
                policy=policy.hybrid_family,
                rollout_mode=hybrid_family_rollout_mode,
            )
        else:
            hybrid_conservative_evaluation = self._evaluate_or_skip_hybrid_candidate(
                evidence=evidence,
                profile=hybrid_conservative_profile,
                policy=policy.hybrid_conservative,
                rollout_mode=hybrid_conservative_rollout_mode,
            )
            hybrid_family_evaluation = self._evaluate_or_skip_hybrid_candidate(
                evidence=evidence,
                profile=hybrid_family_profile,
                policy=policy.hybrid_family,
                rollout_mode=hybrid_family_rollout_mode,
            )
        candidate_evaluations.extend(
            (hybrid_conservative_evaluation, hybrid_family_evaluation)
        )
        if hybrid_family_rollout_mode != "disabled" and hybrid_family_evaluation.eligible:
            recommended_profile = hybrid_family_profile
        elif (
            hybrid_conservative_rollout_mode != "disabled"
            and hybrid_conservative_evaluation.eligible
        ):
            recommended_profile = hybrid_conservative_profile

        effective_profile = exact_fallback_profile
        recommendation_applied = False
        if hybrid_conservative_rollout_mode == "active" and hybrid_conservative_evaluation.eligible:
            effective_profile = hybrid_conservative_profile
        if hybrid_family_rollout_mode == "active" and hybrid_family_evaluation.eligible:
            effective_profile = hybrid_family_profile
        recommendation_applied = (
            effective_profile.mode == recommended_profile.mode
            and effective_profile.mode != exact_fallback_profile.mode
        )

        return AdaptiveSelectorDecisionV2(
            policy_mode=policy.mode,
            effective_profile=effective_profile,
            recommended_profile=recommended_profile,
            exact_fallback_profile=exact_fallback_profile,
            recommendation_applied=recommendation_applied,
            requires_background_auto=requires_background_auto,
            candidate_evaluations=tuple(candidate_evaluations),
        )

    def _evaluate_or_skip_hybrid_candidate(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        profile: ExecutionProfileV2,
        policy: AdaptiveSelectorCandidatePolicyV2,
        rollout_mode: AdaptiveSelectorPolicyModeLiteralV2,
    ) -> AdaptiveSelectorCandidateEvaluationV2:
        """
        Evaluate one hybrid candidate or return an explicit rollout-disabled explanation.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence prepared by the planner.
            profile: One hybrid execution-profile candidate from the catalog.
            policy: Candidate-specific threshold and rollout policy.
            rollout_mode: Effective rollout mode after combining environment and candidate policy.
        Returns:
            AdaptiveSelectorCandidateEvaluationV2: Eligible evaluation or explicit skip reason.
        Assumptions:
            Candidate-specific rollout caps stay additive and may keep a profile in `shadow`,
            `opt_in`, or `disabled` even when the environment-level selector mode is more
            permissive.
        Raises:
            ValueError: Propagated if the candidate profile is invalid for hybrid evaluation.
        Side Effects:
            None.
        """
        if rollout_mode == "disabled":
            return AdaptiveSelectorCandidateEvaluationV2(
                mode=profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="candidate rollout disabled by policy",
            )
        return self._evaluate_hybrid_candidate(
            evidence=evidence,
            profile=profile,
            policy=policy,
        )

    def _resolve_exact_fallback(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        execution_profiles: ExecutionProfilesCatalogV2,
    ) -> tuple[
        ExecutionProfileV2,
        bool,
        tuple[AdaptiveSelectorCandidateEvaluationV2, ...],
    ]:
        """
        Resolve the conservative exact-only fallback preserved across selector policy modes.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planner evidence.
            execution_profiles: Ordered execution-profile catalog.
        Returns:
            tuple[ExecutionProfileV2, bool, tuple[AdaptiveSelectorCandidateEvaluationV2, ...]]:
                Selected exact fallback, sync background-routing flag, and candidate evaluations.
        Assumptions:
            Existing exact launch budgets remain authoritative for conservative fallback and keep
            the exact Stage B scorer canonical, while the dedicated parity-only exact mode is
            excluded here and resolved only through explicit parity classification.
        Raises:
            ValueError: If the exact catalog is empty.
        Side Effects:
            None.
        """
        exact_profiles = execution_profiles.runtime_enabled_non_parity_exact_profiles()
        selected_profile: ExecutionProfileV2 | None = None
        evaluations: list[AdaptiveSelectorCandidateEvaluationV2] = []
        for profile in exact_profiles:
            eligible = profile.launch_budget.allows(
                stage_a_variants_total=evidence.stage_a_variants_total,
                stage_b_variants_total=evidence.stage_b_variants_total,
                estimated_memory_bytes=evidence.estimated_memory_bytes,
            )
            reason = "launch budget exceeded"
            if eligible:
                reason = "launch budget allows exact execution"
                if selected_profile is None:
                    selected_profile = profile
            elif profile.mode == exact_profiles[-1].mode:
                reason = "launch budget exceeded; retained as conservative exact fallback"
            evaluations.append(
                AdaptiveSelectorCandidateEvaluationV2(
                    mode=profile.mode,
                    eligible=eligible,
                    exceeded_signals=0,
                    reason=reason,
                )
            )
        if selected_profile is not None:
            return selected_profile, False, tuple(evaluations)
        background_profile = execution_profiles.background_exact_profile()
        return (
            background_profile,
            evidence.runtime_mode == "sync_inline",
            tuple(evaluations),
        )

    def _evaluate_hybrid_candidate(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        profile: ExecutionProfileV2,
        policy: AdaptiveSelectorCandidatePolicyV2,
    ) -> AdaptiveSelectorCandidateEvaluationV2:
        """
        Evaluate one hybrid candidate against runtime gating and cost-model thresholds.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence.
            profile: One hybrid execution-profile candidate from the catalog.
            policy: Candidate-specific cost-model thresholds.
        Returns:
            AdaptiveSelectorCandidateEvaluationV2: Explainable eligibility result.
        Assumptions:
            Hybrid candidates remain internal runtime paths; `hybrid_family` additionally depends
            on plugin availability under the live registry rules.
        Raises:
            ValueError: If the profile does not reference one hybrid mode.
        Side Effects:
            None.
        """
        if profile.mode not in {"hybrid_conservative", "hybrid_family"}:
            raise ValueError(
                "_evaluate_hybrid_candidate requires one hybrid execution profile"
            )
        if evidence.runtime_mode != "background_capable":
            return AdaptiveSelectorCandidateEvaluationV2(
                mode=profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="runtime mode requires exact sync launch",
            )
        if not execution_profile_uses_hierarchical_shortlist_runtime_v2(profile=profile):
            return AdaptiveSelectorCandidateEvaluationV2(
                mode=profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="runtime gating is not enabled",
            )
        if not profile.launch_budget.allows(
            stage_a_variants_total=evidence.stage_a_variants_total,
            stage_b_variants_total=evidence.stage_b_variants_total,
            estimated_memory_bytes=evidence.estimated_memory_bytes,
        ):
            return AdaptiveSelectorCandidateEvaluationV2(
                mode=profile.mode,
                eligible=False,
                exceeded_signals=0,
                reason="launch budget exceeded",
            )
        exceeded_signals = self._count_exceeded_signals(
            evidence=evidence,
            policy=policy,
        )
        if exceeded_signals < policy.minimum_exceeded_signals:
            return AdaptiveSelectorCandidateEvaluationV2(
                mode=profile.mode,
                eligible=False,
                exceeded_signals=exceeded_signals,
                reason=(
                    "cost model below promotion threshold "
                    f"({exceeded_signals}/{policy.minimum_exceeded_signals} signals)"
                ),
            )
        if profile.mode == "hybrid_family":
            resolution = self._resolve_family_plugin_candidate(
                evidence=evidence,
                profile=profile,
            )
            if resolution.status != "resolved":
                return AdaptiveSelectorCandidateEvaluationV2(
                    mode=profile.mode,
                    eligible=False,
                    exceeded_signals=exceeded_signals,
                    reason=f"plugin availability={resolution.status}",
                )
        return AdaptiveSelectorCandidateEvaluationV2(
            mode=profile.mode,
            eligible=True,
            exceeded_signals=exceeded_signals,
            reason="cost model and runtime gating allow hybrid execution",
        )

    def _count_exceeded_signals(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        policy: AdaptiveSelectorCandidatePolicyV2,
    ) -> int:
        """
        Count how many deterministic cost-model signals exceed one candidate's thresholds.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence.
            policy: Candidate-specific promotion thresholds.
        Returns:
            int: Number of exceeded cost-model signals in deterministic order.
        Assumptions:
            Selector signals remain `grid cardinality`, `stage_a`, `stage_b`, and memory.
        Raises:
            None.
        Side Effects:
            None.
        """
        exceeded_signals = 0
        if evidence.grid_cardinality >= policy.min_grid_cardinality:
            exceeded_signals += 1
        if evidence.effective_stage_a_cost_units() >= policy.min_stage_a_variants_total:
            exceeded_signals += 1
        if evidence.stage_b_variants_total >= policy.min_stage_b_variants_total:
            exceeded_signals += 1
        if evidence.estimated_memory_bytes >= policy.min_estimated_memory_bytes:
            exceeded_signals += 1
        return exceeded_signals

    def _select_parity_first_profile(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        execution_profiles: ExecutionProfilesCatalogV2,
        policy: AdaptiveSelectorPolicyV2,
        exact_fallback_profile: ExecutionProfileV2,
        requires_background_auto: bool,
        exact_evaluations: tuple[AdaptiveSelectorCandidateEvaluationV2, ...],
    ) -> AdaptiveSelectorDecisionV2:
        """
        Select execution profile for parity-first classified workloads.

        Parity-first workloads must NOT be recommended as hybrid_conservative or hybrid_family.
        They stay on the dedicated parity-first exact profile with explicit parity classification
        evidence.

        Docs:
          - docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence with parity classification.
            execution_profiles: Ordered execution-profile catalog.
            policy: Startup-validated selector rollout policy.
            exact_fallback_profile: Resolved exact fallback from budget evaluation.
            requires_background_auto: Whether sync launch budget was exceeded.
            exact_evaluations: Candidate evaluations from exact fallback resolution.
        Returns:
            AdaptiveSelectorDecisionV2: Decision with parity-first classification kept separate
                from hybrid rollout recommendations and generic exact fallback.
        Assumptions:
            Parity-first classification evidence was already validated by the planner, so this
            method only needs to ensure hybrid profiles are NOT recommended and the dedicated
            parity exact profile remains the effective selection.
        Raises:
            ValueError: If parity classification is missing (should have been checked before call).
        Side Effects:
            None.
        """
        if evidence.parity_classification is None:
            raise ValueError(
                "_select_parity_first_profile requires parity_classification in evidence"
            )

        parity_profile = execution_profiles.profile_for_mode(mode="exact_no_risk_parity")
        parity_reason = (
            "parity-first classification selects exact no-risk profile: "
            f"{evidence.parity_classification.nr2_classification_reason}"
        )
        parity_evaluation = AdaptiveSelectorCandidateEvaluationV2(
            mode=parity_profile.mode,
            eligible=True,
            exceeded_signals=0,
            reason=parity_reason,
        )

        # Build hybrid candidate evaluations showing they are explicitly skipped for parity class
        hybrid_conservative_profile = execution_profiles.profile_for_mode(
            mode="hybrid_conservative"
        )
        hybrid_family_profile = execution_profiles.profile_for_mode(mode="hybrid_family")

        hybrid_conservative_evaluation = AdaptiveSelectorCandidateEvaluationV2(
            mode=hybrid_conservative_profile.mode,
            eligible=False,
            exceeded_signals=0,
            reason="parity-first classification excludes hybrid rollout",
        )
        hybrid_family_evaluation = AdaptiveSelectorCandidateEvaluationV2(
            mode=hybrid_family_profile.mode,
            eligible=False,
            exceeded_signals=0,
            reason="parity-first classification excludes hybrid rollout",
        )

        candidate_evaluations = list(exact_evaluations)
        candidate_evaluations.append(parity_evaluation)
        candidate_evaluations.extend(
            (hybrid_conservative_evaluation, hybrid_family_evaluation)
        )

        # Parity-first workloads stay on the dedicated exact profile, never promoted to hybrid
        effective_profile = parity_profile
        recommended_profile = parity_profile

        return AdaptiveSelectorDecisionV2(
            policy_mode=policy.mode,
            effective_profile=effective_profile,
            recommended_profile=recommended_profile,
            exact_fallback_profile=parity_profile,
            recommendation_applied=False,
            requires_background_auto=False,
            candidate_evaluations=tuple(candidate_evaluations),
        )

    def _resolve_family_plugin_candidate(
        self,
        *,
        evidence: AdaptiveSelectorPlanningEvidenceV2,
        profile: ExecutionProfileV2,
    ) -> FamilyPluginRegistryResolutionV2:
        """
        Resolve whether `hybrid_family` is a valid adaptive candidate under live plugin rules.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-adaptive-selector-v1.md
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

        Args:
            evidence: Deterministic planning evidence containing normalized indicator ids.
            profile: Resolved `hybrid_family` execution profile candidate.
        Returns:
            FamilyPluginRegistryResolutionV2: Explicit registry resolution outcome.
        Assumptions:
            `hybrid_family` may activate only when runtime gating is live, family-plugin routing
            is enabled, and one plugin is applicable to the deterministic indicator family.
        Raises:
            ValueError: If the supplied profile is not `hybrid_family`.
        Side Effects:
            None.
        """
        if profile.mode != "hybrid_family":
            raise ValueError(
                "_resolve_family_plugin_candidate requires the 'hybrid_family' profile"
            )
        indicator_family_literal = _resolve_family_plugin_indicator_family_literal_v2(
            indicator_ids=evidence.indicator_ids
        )
        return self.family_plugin_registry.resolve_selection(
            execution_profile_mode=profile.mode,
            indicator_family_literal=indicator_family_literal,
            family_plugin_enabled=profile.feature_flags.family_plugin_enabled,
        )


def default_adaptive_selector_policy_v2() -> AdaptiveSelectorPolicyV2:
    """
    Build the default conservative adaptive-selector rollout policy.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        None.
    Returns:
        AdaptiveSelectorPolicyV2: Conservative disabled-by-default selector policy.
    Assumptions:
        Milestone F1 ships the selector foundation without changing the environment rollout by
        default; F2 may later promote `shadow`, `opt_in`, or `active` explicitly per
        environment.
    Raises:
        ValueError: If one default threshold drifts from the typed policy contract.
    Side Effects:
        None.
    """
    return AdaptiveSelectorPolicyV2()


def _default_family_plugin_registry_v2() -> FamilyPluginRegistryV2:
    """
    Build the default family-plugin registry lazily to avoid import-time cycles.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        None.
    Returns:
        FamilyPluginRegistryV2: Startup-equivalent registry for adaptive selector use.
    Assumptions:
        The selector must share the live registry semantics without introducing a second plugin
        registration path.
    Raises:
        ValueError: Propagated if registry metadata is invalid.
    Side Effects:
        Imports the family-plugin registry module lazily.
    """
    from .family_plugins.registry_v2 import build_default_family_plugin_registry_v2

    return build_default_family_plugin_registry_v2()


def _resolve_family_plugin_indicator_family_literal_v2(
    *,
    indicator_ids: tuple[str, ...],
) -> str | None:
    """
    Resolve indicator-family applicability lazily to avoid selector import cycles.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        indicator_ids: Deterministic indicator ids from the prepared runtime plan.
    Returns:
        str | None: Resolved indicator-family literal or `None` for mixed-family requests.
    Assumptions:
        Adaptive selection must reuse the same family resolution rule used by the live registry.
    Raises:
        ValueError: Propagated if one indicator id is blank.
    Side Effects:
        Imports the family-plugin contracts module lazily.
    """
    from .family_plugins.contracts_v2 import resolve_family_plugin_indicator_family_v2

    return resolve_family_plugin_indicator_family_v2(indicator_ids=indicator_ids)


def _bounded_adaptive_selector_policy_mode_v2(
    *,
    policy_mode: AdaptiveSelectorPolicyModeLiteralV2,
    candidate_rollout_mode: AdaptiveSelectorPolicyModeLiteralV2,
) -> AdaptiveSelectorPolicyModeLiteralV2:
    """
    Resolve the effective rollout mode for one candidate under the env-level selector policy.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-adaptive-selector-v1.md
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    Related:
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py

    Args:
        policy_mode: Environment-level selector rollout mode.
        candidate_rollout_mode: Candidate-specific rollout cap from config.
    Returns:
        AdaptiveSelectorPolicyModeLiteralV2: More conservative of the two rollout modes.
    Assumptions:
        Candidate-specific rollout stays narrower than or equal to the environment rollout, so
        `hybrid_family` may remain `shadow` even when `hybrid_conservative` is `opt_in` or
        `active`.
    Raises:
        None.
    Side Effects:
        None.
    """
    if (
        _ADAPTIVE_SELECTOR_POLICY_MODE_PRIORITY_V2[candidate_rollout_mode]
        <= _ADAPTIVE_SELECTOR_POLICY_MODE_PRIORITY_V2[policy_mode]
    ):
        return candidate_rollout_mode
    return policy_mode


__all__ = [
    "ALLOWED_ADAPTIVE_SELECTOR_POLICY_MODES_V2",
    "ALLOWED_ADAPTIVE_SELECTOR_RUNTIME_MODES_V2",
    "AdaptiveExecutionSelectorV2",
    "AdaptiveSelectorCandidateEvaluationV2",
    "AdaptiveSelectorCandidatePolicyV2",
    "AdaptiveSelectorDecisionV2",
    "AdaptiveSelectorPlanningEvidenceV2",
    "AdaptiveSelectorPolicyModeLiteralV2",
    "AdaptiveSelectorPolicyV2",
    "AdaptiveSelectorRuntimeModeLiteralV2",
    "CostModelAdaptiveExecutionSelectorV2",
    "default_adaptive_selector_policy_v2",
    "validate_adaptive_selector_policy_mode_v2",
    "validate_adaptive_selector_runtime_mode_v2",
]
