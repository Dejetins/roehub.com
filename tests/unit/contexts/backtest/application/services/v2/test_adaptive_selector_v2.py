from __future__ import annotations

from dataclasses import replace

import pytest

from trading.contexts.backtest.application.services.v2.adaptive_selector_v2 import (
    AdaptiveSelectorCandidateEvaluationV2,
    AdaptiveSelectorPlanningEvidenceV2,
    AdaptiveSelectorRuntimeModeLiteralV2,
    CostModelAdaptiveExecutionSelectorV2,
    default_adaptive_selector_policy_v2,
)
from trading.contexts.backtest.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestArtifactRuntimePlannerV2,
    BacktestRiskVariantV2,
    BacktestRuntimeStageCostModelV2,
    _build_launch_budget_evidence_v2,
)
from trading.contexts.backtest.application.services.v2.execution_profile_v2 import (
    ExecutionProfileFeatureFlagsV2,
    ExecutionProfileLaunchBudgetEvidenceV2,
    ExecutionProfilesCatalogV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest.application.services.v2.family_plugins.registry_v2 import (
    FamilyPluginRegistryV2,
)
from trading.platform.errors import RoehubError


def _catalog_with_live_hybrid_profiles() -> ExecutionProfilesCatalogV2:
    """
    Build a catalog with live hybrid runtime gates for adaptive-selector unit coverage.

    Args:
        None.
    Returns:
        ExecutionProfilesCatalogV2: Catalog with runtime-enabled `hybrid_conservative` and
            `hybrid_family` profiles.
    Assumptions:
        Tests keep `exact_small` as the default exact profile and only relax hybrid gates needed
        to exercise the selector contract.
    Raises:
        AssertionError: If an unexpected profile mode is missing from the default catalog.
    Side Effects:
        None.
    """
    catalog = default_execution_profiles_catalog_v2()
    updated_profiles = []
    for profile in catalog.available_profiles:
        if profile.mode == "hybrid_conservative":
            updated_profiles.append(
                replace(
                    profile,
                    feature_flags=ExecutionProfileFeatureFlagsV2(
                        runtime_enabled=True,
                        heuristic_shortlist_enabled=True,
                        parallel_stage_b_enabled=False,
                        family_plugin_enabled=False,
                    ),
                )
            )
            continue
        if profile.mode == "hybrid_family":
            updated_profiles.append(
                replace(
                    profile,
                    feature_flags=ExecutionProfileFeatureFlagsV2(
                        runtime_enabled=True,
                        heuristic_shortlist_enabled=True,
                        parallel_stage_b_enabled=False,
                        family_plugin_enabled=True,
                    ),
                )
            )
            continue
        updated_profiles.append(profile)
    return replace(catalog, available_profiles=tuple(updated_profiles))


def _policy(
    *,
    mode: str,
    hybrid_conservative_rollout_mode: str = "active",
    hybrid_family_rollout_mode: str = "active",
):
    """
    Build one selector policy with the requested rollout mode and default thresholds.

    Args:
        mode: Selector rollout mode literal.
        hybrid_conservative_rollout_mode: Candidate rollout cap for `hybrid_conservative`.
        hybrid_family_rollout_mode: Candidate rollout cap for `hybrid_family`.
    Returns:
        object: Typed adaptive-selector policy instance.
    Assumptions:
        Default thresholds are already conservative enough for the deterministic evidence used in
        these tests.
    Raises:
        ValueError: If the provided mode is invalid.
    Side Effects:
        None.
    """
    base_policy = default_adaptive_selector_policy_v2()
    return replace(
        base_policy,
        mode=mode,
        hybrid_conservative=replace(
            base_policy.hybrid_conservative,
            rollout_mode=hybrid_conservative_rollout_mode,
        ),
        hybrid_family=replace(
            base_policy.hybrid_family,
            rollout_mode=hybrid_family_rollout_mode,
        ),
    )


def _evidence(
    *,
    grid_cardinality: int,
    stage_b_variants_total: int,
    estimated_memory_bytes: int,
    runtime_mode: AdaptiveSelectorRuntimeModeLiteralV2,
    indicator_ids: tuple[str, ...],
) -> AdaptiveSelectorPlanningEvidenceV2:
    """
    Build deterministic planning evidence for selector unit tests.

    Args:
        grid_cardinality: Prepared grid cardinality and current Stage A count.
        stage_b_variants_total: Prepared Stage B variants total.
        estimated_memory_bytes: Deterministic memory estimate.
        runtime_mode: Planner runtime mode literal.
        indicator_ids: Deterministic indicator ids from the prepared plan.
    Returns:
        AdaptiveSelectorPlanningEvidenceV2: Typed selector evidence payload.
    Assumptions:
        Current planner Stage A work equals grid cardinality for these tests.
    Raises:
        ValueError: If one evidence field is invalid.
    Side Effects:
        None.
    """
    return AdaptiveSelectorPlanningEvidenceV2(
        grid_cardinality=grid_cardinality,
        stage_a_variants_total=grid_cardinality,
        stage_b_variants_total=stage_b_variants_total,
        estimated_memory_bytes=estimated_memory_bytes,
        runtime_mode=runtime_mode,
        indicator_ids=indicator_ids,
    )


def _evaluation_for_mode(
    *,
    evaluations: tuple[AdaptiveSelectorCandidateEvaluationV2, ...],
    mode: str,
) -> AdaptiveSelectorCandidateEvaluationV2:
    """
    Resolve one candidate-evaluation row by execution-profile mode.

    Args:
        evaluations: Selector candidate-evaluation payload.
        mode: Target execution-profile mode literal.
    Returns:
        AdaptiveSelectorCandidateEvaluationV2: Matching candidate-evaluation row.
    Assumptions:
        Selector decisions include one row per approved execution-profile mode.
    Raises:
        AssertionError: If the requested evaluation row is absent.
    Side Effects:
        None.
    """
    for evaluation in evaluations:
        if evaluation.mode == mode:
            return evaluation
    raise AssertionError(f"missing candidate evaluation for mode {mode!r}")


def test_selector_disabled_keeps_exact_only_behavior() -> None:
    """
    Verify `disabled` policy preserves the previous exact-only automatic behavior.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Large `ma.` requests would otherwise qualify for `hybrid_family` under live rollout
        gates, which makes `disabled` mode easy to observe.
    Raises:
        AssertionError: If disabled mode changes the effective or recommended profile.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(mode="disabled"),
    )

    assert decision.effective_profile.mode == "exact_parallel"
    assert decision.recommended_profile.mode == "exact_parallel"
    assert decision.recommendation_applied is False
    assert (
        _evaluation_for_mode(
            evaluations=decision.candidate_evaluations,
            mode="hybrid_family",
        ).reason
        == "adaptive selector policy disabled"
    )


def test_selector_shadow_reports_hybrid_recommendation_without_switching_execution() -> None:
    """
    Verify `shadow` mode recommends a hybrid profile while preserving exact execution.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The `ma.` family stays applicable to the shipped `hybrid_family` plugin registry.
    Raises:
        AssertionError: If shadow mode switches the effective profile or loses the recommendation.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(
            mode="shadow",
            hybrid_conservative_rollout_mode="active",
            hybrid_family_rollout_mode="shadow",
        ),
    )

    assert decision.effective_profile.mode == "exact_parallel"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False


def test_selector_opt_in_keeps_exact_execution_while_exposing_explicit_opt_in_phase() -> None:
    """
    Verify `opt_in` stays recommendation-only for automatic selection while remaining explicit.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The explicit prod opt-in phase should remain distinguishable from both shadow and active
        rollout even when the current automatic selector behavior still keeps exact execution.
    Raises:
        AssertionError: If opt-in mode is collapsed back into a different policy literal.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(
            mode="opt_in",
            hybrid_conservative_rollout_mode="active",
            hybrid_family_rollout_mode="shadow",
        ),
    )

    assert decision.policy_mode == "opt_in"
    assert decision.effective_profile.mode == "exact_parallel"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False


def test_selector_active_applies_conservative_hybrid_when_family_rollout_is_shadow() -> None:
    """
    Verify active rollout may execute `hybrid_conservative` while `hybrid_family` remains shadow.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Large pure `ma.` requests may satisfy both hybrid thresholds, but F2 rollout may keep the
        family path narrower than the conservative universal path.
    Raises:
        AssertionError: If active rollout applies the shadow-only family candidate or falls back
            to exact execution unexpectedly.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(
            mode="active",
            hybrid_conservative_rollout_mode="active",
            hybrid_family_rollout_mode="shadow",
        ),
    )

    assert decision.effective_profile.mode == "hybrid_conservative"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is False


def test_selector_active_keeps_small_runs_on_exact_profile() -> None:
    """
    Verify active policy keeps small requests on the conservative exact runtime path.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Small requests should stay on `exact_small` even when hybrid runtime gates are live.
    Raises:
        AssertionError: If the selector promotes a small request away from `exact_small`.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=800,
            stage_b_variants_total=6000,
            estimated_memory_bytes=134217728,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(
            mode="active",
            hybrid_conservative_rollout_mode="active",
            hybrid_family_rollout_mode="shadow",
        ),
    )

    assert decision.effective_profile.mode == "exact_small"
    assert decision.recommended_profile.mode == "exact_small"
    assert decision.recommendation_applied is False


def test_selector_active_can_choose_hybrid_family_when_plugin_is_applicable() -> None:
    """
    Verify active policy selects `hybrid_family` when family-plugin routing is truly available.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The shipped `ma.` family plugin remains registered for `hybrid_family`.
    Raises:
        AssertionError: If an applicable `ma.` request does not promote to `hybrid_family`.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2()
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(mode="active"),
    )

    assert decision.effective_profile.mode == "hybrid_family"
    assert decision.recommended_profile.mode == "hybrid_family"
    assert decision.recommendation_applied is True


def test_selector_never_picks_hybrid_family_when_plugin_is_unavailable() -> None:
    """
    Verify unsupported `hybrid_family` paths fall back to `hybrid_conservative`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `hybrid_conservative` stays available as the universal hybrid fallback when the family
        plugin registry has no matching entry.
    Raises:
        AssertionError: If the selector chooses `hybrid_family` without plugin availability.
    Side Effects:
        None.
    """
    selector = CostModelAdaptiveExecutionSelectorV2(
        family_plugin_registry=FamilyPluginRegistryV2()
    )
    decision = selector.select(
        evidence=_evidence(
            grid_cardinality=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            runtime_mode="background_capable",
            indicator_ids=("ma.fast", "ma.slow"),
        ),
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        policy=_policy(mode="active"),
    )

    assert decision.effective_profile.mode == "hybrid_conservative"
    assert decision.recommended_profile.mode == "hybrid_conservative"
    assert (
        _evaluation_for_mode(
            evaluations=decision.candidate_evaluations,
            mode="hybrid_family",
        ).reason
        == "plugin availability=missing_plugin"
    )


def test_requested_execution_profile_override_keeps_precedence_over_selector() -> None:
    """
    Verify explicit requested profile overrides still outrank adaptive automatic selection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Planner override precedence remains the shared runtime contract for internal-only
        `execution_profile_mode` metadata.
    Raises:
        AssertionError: If the planner ignores the explicit requested profile mode.
    Side Effects:
        None.
    """
    planner = BacktestArtifactRuntimePlannerV2(
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        adaptive_selector_policy=_policy(mode="active"),
    )

    profile = planner.resolve_execution_profile(
        stage_a_variants_total=20000,
        stage_b_variants_total=120000,
        estimated_memory_bytes=1200000000,
        requested_execution_profile_mode="exact_parallel",
        indicator_ids=("ma.fast", "ma.slow"),
    )

    assert profile.mode == "exact_parallel"


def test_requested_hybrid_profile_override_requires_explicit_opt_in_or_active_policy() -> None:
    """
    Verify shadow rollout keeps internal live hybrid overrides disabled until opt-in is explicit.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Making prod opt-in explicit means shadow mode may still inspect recommendations but should
        no longer silently double as live hybrid opt-in for internal overrides.
    Raises:
        AssertionError: If shadow policy accepts a live hybrid requested-profile override.
    Side Effects:
        None.
    """
    planner = BacktestArtifactRuntimePlannerV2(
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        adaptive_selector_policy=_policy(mode="shadow"),
    )

    with pytest.raises(RoehubError) as error_info:
        planner.resolve_execution_profile(
            stage_a_variants_total=20000,
            stage_b_variants_total=120000,
            estimated_memory_bytes=1200000000,
            requested_execution_profile_mode="hybrid_conservative",
            indicator_ids=("ma.fast", "ma.slow"),
        )

    details = error_info.value.details
    assert details is not None
    assert details["error"] == "execution_profile_not_enabled"
    assert details["adaptive_selector_policy_mode"] == "shadow"


def test_requested_hybrid_profile_override_is_allowed_in_opt_in_phase() -> None:
    """
    Verify explicit live hybrid overrides become allowed once rollout moves to `opt_in`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Opt-in rollout exists specifically to permit internal/manual live hybrid evaluation before
        automatic selective defaulting becomes active.
    Raises:
        AssertionError: If opt-in rollout still rejects an otherwise live hybrid override.
    Side Effects:
        None.
    """
    planner = BacktestArtifactRuntimePlannerV2(
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        adaptive_selector_policy=_policy(mode="opt_in"),
    )

    profile = planner.resolve_execution_profile(
        stage_a_variants_total=20000,
        stage_b_variants_total=120000,
        estimated_memory_bytes=1200000000,
        requested_execution_profile_mode="hybrid_conservative",
        indicator_ids=("ma.fast", "ma.slow"),
    )

    assert profile.mode == "hybrid_conservative"


def test_requested_nr2_no_risk_hybrid_profile_uses_narrowed_launch_budget_evidence() -> None:
    """
    Verify canonical `NR2` `hybrid_conservative` sync launch stays `sync_inline` once the
    planner uses explicit no-risk evidence aligned with the `bypassed_no_risk` terminal path.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Without explicit no-risk launch evidence the same request would still fall into
        `background_auto` because raw-grid Stage A and legacy memory math exceed the requested
        sync launch budget.
    Raises:
        AssertionError: If raw-grid rejection disappears or narrowed no-risk evidence fails to
            preserve the requested sync launch.
    Side Effects:
        None.
    """
    planner = BacktestArtifactRuntimePlannerV2(
        execution_profiles=_catalog_with_live_hybrid_profiles(),
        adaptive_selector_policy=_policy(mode="active"),
        launch_budget_mode="sync_inline",
    )

    with pytest.raises(RoehubError) as error_info:
        planner.resolve_execution_profile(
            stage_a_variants_total=345744,
            stage_b_variants_total=20000,
            estimated_memory_bytes=2300000000,
            requested_execution_profile_mode="hybrid_conservative",
            indicator_ids=("ma.fast", "ma.slow"),
        )

    details = error_info.value.details
    assert details is not None
    assert details["error"] == "background_auto_required"
    assert details["execution_mode"] == "background_auto"
    assert details["stage_a_variants_total"] == 345744

    profile = planner.resolve_execution_profile(
        stage_a_variants_total=345744,
        stage_b_variants_total=20000,
        estimated_memory_bytes=2300000000,
        requested_execution_profile_mode="hybrid_conservative",
        indicator_ids=("ma.fast", "ma.slow"),
        launch_budget_evidence=ExecutionProfileLaunchBudgetEvidenceV2(
            stage_a_variants_total=20164,
            stage_b_variants_total=20000,
            estimated_memory_bytes=1400000000,
            workload_class="no_risk_terminal",
        ),
    )

    assert profile.mode == "hybrid_conservative"


def test_nr2_no_risk_launch_budget_evidence_uses_planner_narrowed_workload_shape() -> None:
    """
    Verify planner-produced `NR2` launch-budget evidence narrows Stage A and memory inputs for
    the `bypassed_no_risk` terminal path instead of reusing raw-grid math.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical no-risk sync launch budgeting should stay explicit and inspectable, so the
        planner emits a typed evidence object rather than hiding this narrowing behind profile
        threshold changes.
    Raises:
        AssertionError: If the planner stops classifying the canonical no-risk shape as
            `no_risk_terminal` or reverts to raw-grid totals.
    Side Effects:
        None.
    """
    evidence = _build_launch_budget_evidence_v2(
        bars=2_500,
        stage_a_variants_total=345_744,
        stage_b_variants_total=20_000,
        estimated_memory_bytes=2_300_000_000,
        stage_cost_model=BacktestRuntimeStageCostModelV2(
            row_prefilter_rows_total=284,
            retained_row_variants_total=284,
            combo_prefilter_variants_total=20_164,
            retained_exact_candidates_total=40_000,
            stage_a_cost_units=180_448,
        ),
        risk_variants=(
            BacktestRiskVariantV2(
                risk_index=0,
                risk_params={
                    "sl_enabled": False,
                    "sl_pct": None,
                    "tp_enabled": False,
                    "tp_pct": None,
                },
            ),
        ),
    )

    assert evidence.workload_class == "no_risk_terminal"
    assert evidence.stage_a_variants_total == 20_164
    assert evidence.stage_b_variants_total == 20_000
    assert evidence.estimated_memory_bytes > 0
    assert evidence.estimated_memory_bytes < 2_300_000_000
