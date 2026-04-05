from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from trading.contexts.backtest.application.services.v2 import (
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorAxisPlanV2,
    BacktestIndicatorPlanV2,
    BacktestRiskVariantV2,
    ExecutionProfileFeatureFlagsV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest.application.services.v2.family_plugins import (
    FamilyAccelerationPluginV2,
    FamilyPluginApplicabilityV2,
    FamilyPluginMetadataV2,
    FamilyPluginProposalResultV2,
    FamilyPluginRegistryV2,
    build_default_family_plugin_registry_v2,
    build_family_plugin_planning_context_v2,
)


@dataclass(frozen=True, slots=True)
class _StubFamilyPlugin(FamilyAccelerationPluginV2):
    """
    Minimal proposal-only family plugin used by deterministic registry tests.
    """

    metadata: FamilyPluginMetadataV2
    proposal_result: FamilyPluginProposalResultV2

    def propose(self, *, context) -> FamilyPluginProposalResultV2:
        """
        Return the preconfigured proposal payload for registry tests.

        Args:
            context: Narrow immutable planning context.
        Returns:
            FamilyPluginProposalResultV2: Preconfigured proposal payload.
        Assumptions:
            Registry tests inspect resolution behavior and do not execute live shortlist logic.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = context
        return self.proposal_result


def _build_runtime_plan(
    *,
    indicator_ids: tuple[str, ...],
    family_plugin_enabled: bool,
) -> BacktestArtifactRuntimePlanV2:
    """
    Build a minimal runtime plan fixture for family-plugin registry tests.

    Args:
        indicator_ids: Indicator ids exposed by the prepared runtime plan.
        family_plugin_enabled: Whether the resolved execution profile enables family plugins.
    Returns:
        BacktestArtifactRuntimePlanV2: Minimal valid runtime plan fixture.
    Assumptions:
        Registry tests need planner-owned identity data but not live runtime execution.
    Raises:
        ValueError: If one helper fixture literal violates runtime-plan invariants.
    Side Effects:
        None.
    """
    profile = default_execution_profiles_catalog_v2().profile_for_mode(mode="hybrid_family")
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


def test_family_plugin_registry_resolves_plugin_by_profile_mode_and_family() -> None:
    """
    Verify registry selection depends only on resolved profile mode and indicator family.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone E registry lookup stays deterministic and explicit instead of using ad-hoc
        runtime heuristics.
    Raises:
        AssertionError: If a matching plugin does not resolve.
    Side Effects:
        None.
    """
    plugin = _StubFamilyPlugin(
        metadata=FamilyPluginMetadataV2(
            plugin_id="ma_accel",
            display_name="MA proposal",
            applicability=FamilyPluginApplicabilityV2(
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("row_shortlist", "proxy_score"),
        ),
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma_accel",
            row_shortlist=(1, 2),
        ),
    )
    registry = FamilyPluginRegistryV2(plugins=(plugin,))
    context = build_family_plugin_planning_context_v2(
        runtime_plan=_build_runtime_plan(
            indicator_ids=("ma.sma", "ma.ema"),
            family_plugin_enabled=True,
        )
    )

    resolution = registry.resolve(context=context)

    assert resolution.status == "resolved"
    assert resolution.plugin is plugin
    assert resolution.selection_key is not None
    assert resolution.selection_key.execution_profile_mode == "hybrid_family"
    assert resolution.selection_key.indicator_family_literal == "ma"


def test_family_plugin_registry_returns_missing_plugin_warning_when_no_match() -> None:
    """
    Verify registry returns an explicit missing-plugin warning instead of silent fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Missing plugin is a first-class failure-handling outcome reused by later runtime wiring.
    Raises:
        AssertionError: If missing-plugin resolution omits its warning payload.
    Side Effects:
        None.
    """
    registry = FamilyPluginRegistryV2()
    context = build_family_plugin_planning_context_v2(
        runtime_plan=_build_runtime_plan(
            indicator_ids=("ma.sma",),
            family_plugin_enabled=True,
        )
    )

    resolution = registry.resolve(context=context)

    assert resolution.status == "missing_plugin"
    assert resolution.plugin is None
    assert resolution.warning is not None
    assert resolution.warning.reason == "missing_plugin"
    assert "warning + universal fallback" in resolution.warning.message


def test_family_plugin_registry_returns_disabled_when_feature_flag_is_off() -> None:
    """
    Verify registry stays inactive while `family_plugin_enabled` remains rollout-disabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone E adds foundation only; live runtime activation remains deferred.
    Raises:
        AssertionError: If registry attempts selection while the feature flag is disabled.
    Side Effects:
        None.
    """
    plugin = _StubFamilyPlugin(
        metadata=FamilyPluginMetadataV2(
            plugin_id="ma_accel",
            display_name="MA proposal",
            applicability=FamilyPluginApplicabilityV2(
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("row_shortlist",),
        ),
        proposal_result=FamilyPluginProposalResultV2(plugin_id="ma_accel", row_shortlist=(1,)),
    )
    registry = FamilyPluginRegistryV2(plugins=(plugin,))
    context = build_family_plugin_planning_context_v2(
        runtime_plan=_build_runtime_plan(
            indicator_ids=("ma.sma",),
            family_plugin_enabled=False,
        )
    )

    resolution = registry.resolve(context=context)

    assert resolution.status == "disabled"
    assert resolution.plugin is None
    assert resolution.warning is None


def test_build_default_family_plugin_registry_v2_registers_first_ma_plugin() -> None:
    """
    Verify the shipped default registry exposes the first concrete MA-family plugin.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Concrete plugins must register through the shared registry surface rather than via
        ad-hoc runtime branching.
    Raises:
        AssertionError: If the default registry does not resolve the shipped MA-family plugin.
    Side Effects:
        None.
    """
    registry = build_default_family_plugin_registry_v2()
    context = build_family_plugin_planning_context_v2(
        runtime_plan=_build_runtime_plan(
            indicator_ids=("ma.ema", "ma.sma"),
            family_plugin_enabled=True,
        )
    )

    resolution = registry.resolve(context=context)

    assert resolution.status == "resolved"
    assert resolution.plugin is not None
    assert resolution.plugin.metadata.plugin_id == "ma.family.v1"


def test_family_plugin_registry_rejects_duplicate_selection_keys() -> None:
    """
    Verify registry fails fast when two plugins claim the same family/profile selection key.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Startup validation must catch ambiguous registry ownership before runtime planning.
    Raises:
        AssertionError: If duplicate selection keys do not raise ValueError.
    Side Effects:
        None.
    """
    duplicate_plugin_one = _StubFamilyPlugin(
        metadata=FamilyPluginMetadataV2(
            plugin_id="ma_accel_one",
            display_name="MA proposal one",
            applicability=FamilyPluginApplicabilityV2(
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("row_shortlist",),
        ),
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma_accel_one",
            row_shortlist=(1,),
        ),
    )
    duplicate_plugin_two = _StubFamilyPlugin(
        metadata=FamilyPluginMetadataV2(
            plugin_id="ma_accel_two",
            display_name="MA proposal two",
            applicability=FamilyPluginApplicabilityV2(
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("pair_shortlist",),
        ),
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma_accel_two",
        ),
    )

    with pytest.raises(ValueError, match="selection key collision"):
        FamilyPluginRegistryV2(plugins=(duplicate_plugin_one, duplicate_plugin_two))
