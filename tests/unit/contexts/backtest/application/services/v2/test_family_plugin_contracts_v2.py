from __future__ import annotations

from dataclasses import replace

import pytest

from trading.contexts.backtest.application.services.v2 import (
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorAxisPlanV2,
    BacktestIndicatorPlanV2,
    BacktestRiskVariantV2,
    ExecutionProfileFeatureFlagsV2,
    ExecutionProfileModeLiteralV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest.application.services.v2.family_plugins import (
    FamilyPluginPairCandidateV2,
    FamilyPluginPlanningContextV2,
    FamilyPluginProposalResultV2,
    FamilyPluginProxyScoreV2,
    build_family_plugin_planning_context_v2,
    resolve_family_plugin_indicator_family_v2,
)


def _build_runtime_plan(
    *,
    indicator_ids: tuple[str, ...],
    mode: ExecutionProfileModeLiteralV2 = "hybrid_family",
    family_plugin_enabled: bool = False,
) -> BacktestArtifactRuntimePlanV2:
    """
    Build a minimal deterministic runtime plan for family-plugin contract tests.

    Args:
        indicator_ids: Indicator ids exposed through the prepared runtime plan.
        mode: Execution-profile mode resolved for the runtime plan.
        family_plugin_enabled: Whether the resolved profile enables family-plugin routing.
    Returns:
        BacktestArtifactRuntimePlanV2: Minimal valid runtime plan fixture.
    Assumptions:
        Contract tests need only the planner-owned identity surface and not live runtime
        execution.
    Raises:
        ValueError: If one helper fixture literal violates runtime-plan invariants.
    Side Effects:
        None.
    """
    profile = default_execution_profiles_catalog_v2().profile_for_mode(mode=mode)
    if profile.feature_flags.family_plugin_enabled != family_plugin_enabled:
        profile = replace(
            profile,
            feature_flags=ExecutionProfileFeatureFlagsV2(
                runtime_enabled=profile.feature_flags.runtime_enabled,
                heuristic_shortlist_enabled=(
                    profile.feature_flags.heuristic_shortlist_enabled
                ),
                parallel_stage_b_enabled=profile.feature_flags.parallel_stage_b_enabled,
                family_plugin_enabled=family_plugin_enabled,
            ),
        )
    return BacktestArtifactRuntimePlanV2(
        indicator_plans=tuple(
            BacktestIndicatorPlanV2(
                indicator_id=indicator_id,
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(20,)),),
                variants=1,
            )
            for indicator_id in indicator_ids
        ),
        signal_axes=(),
        risk_variants=(
            BacktestRiskVariantV2(risk_index=0, risk_params={"tp_pct": 1.5}),
        ),
        execution_profile=profile,
        instrument_id_literal="binance:btc-usdt",
        timeframe_code="1h",
        direction_mode="long-short",
        sizing_mode="fixed_quote",
        execution_params={"fee_pct": 0.1},
        stage_a_variants_total=max(1, len(indicator_ids)),
        stage_b_variants_total=max(1, len(indicator_ids)),
        estimated_memory_bytes=1024,
        indicator_estimate_calls=max(1, len(indicator_ids)),
    )


def test_build_family_plugin_planning_context_derives_indicator_family_and_budget() -> None:
    """
    Verify planning context reuses runtime-plan metadata and derives stable family selection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Family resolution is deterministic from prepared indicator ids and budget reuses the
        typed execution-profile surface.
    Raises:
        AssertionError: If normalized family selection or budget derivation drifts.
    Side Effects:
        None.
    """
    runtime_plan = _build_runtime_plan(indicator_ids=("ma.sma", "ma.ema"))

    context = build_family_plugin_planning_context_v2(
        runtime_plan=runtime_plan,
        requested_execution_profile_mode="hybrid_family",
    )

    assert isinstance(context, FamilyPluginPlanningContextV2)
    assert context.indicator_ids == ("ma.ema", "ma.sma")
    assert context.indicator_family_literal == "ma"
    assert context.plugin_budget_ms == runtime_plan.execution_profile.family_plugin_budget_ms
    assert context.requested_execution_profile_mode == "hybrid_family"


def test_resolve_family_plugin_indicator_family_returns_none_for_mixed_families() -> None:
    """
    Verify mixed indicator families fall back to the universal proposal path deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone E avoids hidden mixed-family routing until a later selector defines it
        explicitly.
    Raises:
        AssertionError: If mixed indicator families resolve to a concrete plugin family.
    Side Effects:
        None.
    """
    assert (
        resolve_family_plugin_indicator_family_v2(
            indicator_ids=("ma.sma", "momentum.trix")
        )
        is None
    )


def test_family_plugin_proposal_result_normalizes_deterministic_ordering() -> None:
    """
    Verify proposal-only outputs deduplicate and sort row/pair/proxy coordinates deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Proposal ordering is part of the contract surface and must stay reviewable across runs.
    Raises:
        AssertionError: If one shortlist/proxy ordering drifts from the normalized contract.
    Side Effects:
        None.
    """
    proposal = FamilyPluginProposalResultV2(
        plugin_id="MA_ACCEL",
        row_shortlist=(4, 1, 4, 3),
        pair_shortlist=(
            FamilyPluginPairCandidateV2(stage_a_index=3, risk_index=2),
            FamilyPluginPairCandidateV2(stage_a_index=1, risk_index=0),
            FamilyPluginPairCandidateV2(stage_a_index=1, risk_index=0),
        ),
        proxy_scores=(
            FamilyPluginProxyScoreV2(stage_a_index=3, risk_index=1, proxy_score=0.3),
            FamilyPluginProxyScoreV2(stage_a_index=1, proxy_score=0.5),
            FamilyPluginProxyScoreV2(stage_a_index=2, risk_index=0, proxy_score=0.7),
        ),
    )

    assert proposal.plugin_id == "ma_accel"
    assert proposal.row_shortlist == (1, 3, 4)
    assert proposal.pair_shortlist == (
        FamilyPluginPairCandidateV2(stage_a_index=1, risk_index=0),
        FamilyPluginPairCandidateV2(stage_a_index=3, risk_index=2),
    )
    assert proposal.proxy_scores == (
        FamilyPluginProxyScoreV2(stage_a_index=1, proxy_score=0.5),
        FamilyPluginProxyScoreV2(stage_a_index=2, risk_index=0, proxy_score=0.7),
        FamilyPluginProxyScoreV2(stage_a_index=3, risk_index=1, proxy_score=0.3),
    )


def test_family_plugin_proposal_result_rejects_duplicate_proxy_targets() -> None:
    """
    Verify proxy-score targets stay unique so later runtime ordering never depends on ambiguity.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        A proposal may include multiple `proxy score` suggestions, but each candidate target must
        appear at most once.
    Raises:
        AssertionError: If duplicate proxy targets do not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="duplicate"):
        FamilyPluginProposalResultV2(
            plugin_id="ma_accel",
            proxy_scores=(
                FamilyPluginProxyScoreV2(stage_a_index=1, proxy_score=0.5),
                FamilyPluginProxyScoreV2(stage_a_index=1, proxy_score=0.7),
            ),
        )
