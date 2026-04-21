from __future__ import annotations

from dataclasses import replace

import pytest

from trading.contexts.backtest_artifacts.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorAxisPlanV2,
    BacktestIndicatorPlanV2,
    BacktestRiskVariantV2,
    BacktestSignalAxisPlanV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.execution_profile_v2 import (
    ExecutionProfileFeatureFlagsV2,
    ExecutionProfileShortlistConfigV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.family_plugins import (
    MAFamilyAccelerationPluginV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.family_plugins.contracts_v2 import (  # noqa: E501
    build_family_plugin_planning_context_v2,
)


def test_ma_family_plugin_v2_proposes_deterministic_row_shortlist_and_proxy_scores() -> None:
    """
    Verify the first MA-family plugin emits deterministic row-shortlist plus proxy-score output.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The first shipped plugin samples deterministic MA window anchors and expands them through
        the shared exact signal-space ordering.
    Raises:
        AssertionError: If deterministic proposal output drifts.
    Side Effects:
        None.
    """
    runtime_plan = _build_runtime_plan(
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40)),
                ),
                variants=4,
            ),
            BacktestIndicatorPlanV2(
                indicator_id="ma.sma",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40)),
                ),
                variants=4,
            ),
        ),
        signal_axes=(
            BacktestSignalAxisPlanV2(
                indicator_id="ma.ema",
                param_name="threshold",
                values=(0.25, 0.5),
            ),
        ),
        max_candidates=8,
    )

    proposal = MAFamilyAccelerationPluginV2().propose(
        context=build_family_plugin_planning_context_v2(runtime_plan=runtime_plan)
    )

    assert proposal.plugin_id == "ma.family.v1"
    assert proposal.row_shortlist == (0, 1, 6, 7, 24, 25, 30, 31)
    assert tuple(score.stage_a_index for score in proposal.proxy_scores) == proposal.row_shortlist
    assert all(score.proxy_score > 0.0 for score in proposal.proxy_scores)


def test_ma_family_plugin_v2_supports_vwma_without_source_axis() -> None:
    """
    Verify canonical `ma.vwma` support stays valid even though the MA definition omits `source`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.vwma` remains part of the canonical MA-family plugin applicability surface.
    Raises:
        AssertionError: If `vwma` proposals fail because the indicator lacks a `source` axis.
    Side Effects:
        None.
    """
    runtime_plan = _build_runtime_plan(
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.vwma",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40)),),
                variants=4,
            ),
        ),
        signal_axes=(),
        max_candidates=2,
    )

    proposal = MAFamilyAccelerationPluginV2().propose(
        context=build_family_plugin_planning_context_v2(runtime_plan=runtime_plan)
    )

    assert proposal.row_shortlist == (0, 3)
    assert tuple(score.stage_a_index for score in proposal.proxy_scores) == (0, 3)


def test_ma_family_plugin_v2_rejects_unknown_ma_indicator_ids() -> None:
    """
    Verify applicability is anchored to canonical `ma.*` definitions rather than prefix only.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical MA definitions are the source of truth for the first shipped family plugin.
    Raises:
        AssertionError: If unknown `ma.*` ids are accepted.
    Side Effects:
        None.
    """
    runtime_plan = _build_runtime_plan(
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.unknown",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),
                ),
                variants=2,
            ),
        ),
        signal_axes=(),
        max_candidates=2,
    )

    with pytest.raises(ValueError, match="canonical MA-family ids"):
        MAFamilyAccelerationPluginV2().propose(
            context=build_family_plugin_planning_context_v2(runtime_plan=runtime_plan)
        )


def _build_runtime_plan(
    *,
    indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
    max_candidates: int,
) -> BacktestArtifactRuntimePlanV2:
    """
    Build a minimal runtime-plan fixture for MA-family proposal tests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        indicator_plans: Planner-owned indicator plans in exact runtime order.
        signal_axes: Planner-owned signal axes in exact runtime order.
        max_candidates: Explicit shortlist cap for the profile fixture.
    Returns:
        BacktestArtifactRuntimePlanV2: Minimal valid runtime plan for MA-family proposal tests.
    Assumptions:
        Unit tests exercise only single-risk expansion and deterministic mixed-radix addressing.
    Raises:
        ValueError: Propagated if the constructed plan violates runtime-plan invariants.
    Side Effects:
        None.
    """
    profile = _hybrid_family_profile_fixture(max_candidates=max_candidates)
    stage_a_variants_total = 1
    for plan in indicator_plans:
        stage_a_variants_total *= plan.variants
    signal_variants_total = 1
    for axis in signal_axes:
        signal_variants_total *= len(axis.values)
    return BacktestArtifactRuntimePlanV2(
        indicator_plans=indicator_plans,
        signal_axes=signal_axes,
        risk_variants=(
            BacktestRiskVariantV2(risk_index=0, risk_params={"tp_pct": 2.0}),
        ),
        execution_profile=profile,
        instrument_id_literal="binance:btc-usdt",
        timeframe_code="1h",
        direction_mode="long-short",
        sizing_mode="fixed_quote",
        execution_params={"fee_pct": 0.1},
        stage_a_variants_total=stage_a_variants_total * signal_variants_total,
        stage_b_variants_total=max_candidates,
        estimated_memory_bytes=1024,
        indicator_estimate_calls=max(1, len(indicator_plans)),
    )


def _hybrid_family_profile_fixture(*, max_candidates: int):
    """
    Build one explicit opt-in `hybrid_family` execution profile for MA plugin tests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - configs/test/backtest.yaml

    Args:
        max_candidates: Explicit shortlist cap published by the profile fixture.
    Returns:
        object: Runtime-enabled `hybrid_family` profile fixture.
    Assumptions:
        Proposal tests use the same internal-only gating contract as the live runtime path.
    Raises:
        ValueError: Propagated if the constructed profile violates typed invariants.
    Side Effects:
        None.
    """
    catalog = default_execution_profiles_catalog_v2()
    base_profile = catalog.profile_for_mode(mode="hybrid_family")
    return replace(
        base_profile,
        feature_flags=ExecutionProfileFeatureFlagsV2(
            runtime_enabled=True,
            heuristic_shortlist_enabled=True,
            parallel_stage_b_enabled=False,
            family_plugin_enabled=True,
        ),
        shortlist_config=ExecutionProfileShortlistConfigV2(
            enabled=True,
            max_candidates=max_candidates,
            scoring=base_profile.shortlist_config.scoring,
            retention=base_profile.shortlist_config.retention,
        ),
    )
