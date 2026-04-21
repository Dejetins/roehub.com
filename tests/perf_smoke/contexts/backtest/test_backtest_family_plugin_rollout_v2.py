from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

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
from trading.contexts.backtest_artifacts.application.services.v2.hierarchical_shortlist_builder_v2 import (  # noqa: E501
    BacktestHierarchicalShortlistBuilderV2,
    HierarchicalShortlistRuntimePlanV2,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


class _FailIfCalledPriceLoader:
    """
    Price-loader fake that fails fast if the universal artifact path is used unexpectedly.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
    """

    def __init__(self) -> None:
        """
        Initialize one fail-fast loader recording unexpected calls.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The successful MA-family plugin path should not touch universal artifact scoring IO.
        Raises:
            None.
        Side Effects:
            Initializes an in-memory call counter.
        """
        self.calls = 0

    def load_price_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Fail immediately because the perf-smoke success path should not load price artifacts.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Never returns successfully.
        Assumptions:
            Successful family-plugin rollout evidence should avoid the universal artifact path.
        Raises:
            AssertionError: Always, because this loader must not be used on the success path.
        Side Effects:
            Increments the in-memory call counter before failing.
        """
        _ = context, timeframe
        self.calls += 1
        raise AssertionError("price artifacts must not be loaded on the family-plugin path")

    def load_mapping_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Fail immediately because the perf-smoke success path should not load mapping artifacts.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Never returns successfully.
        Assumptions:
            Successful family-plugin rollout evidence should avoid the universal artifact path.
        Raises:
            AssertionError: Always, because this loader must not be used on the success path.
        Side Effects:
            Increments the in-memory call counter before failing.
        """
        _ = context, timeframe
        self.calls += 1
        raise AssertionError("mapping artifacts must not be loaded on the family-plugin path")

    def load_hit_times_arrays(self, *, context: Any) -> Any:
        """
        Fail immediately because the perf-smoke success path should not load hit-times artifacts.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
        Returns:
            Any: Never returns successfully.
        Assumptions:
            Successful family-plugin rollout evidence should avoid the universal artifact path.
        Raises:
            AssertionError: Always, because this loader must not be used on the success path.
        Side Effects:
            Increments the in-memory call counter before failing.
        """
        _ = context
        self.calls += 1
        raise AssertionError("hit-times artifacts must not be loaded on the family-plugin path")


class _FailIfCalledSignalLoader:
    """
    Signal-loader fake that fails fast if universal shortlist scoring is used unexpectedly.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
    """

    def __init__(self) -> None:
        """
        Initialize one fail-fast loader recording unexpected calls.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The successful MA-family plugin path should not need universal signal-matrix reads.
        Raises:
            None.
        Side Effects:
            Initializes an in-memory call counter.
        """
        self.calls = 0

    def load_signal_matrix(
        self,
        *,
        context: Any,
        timeframe: str,
        indicator_id: str,
    ) -> Any:
        """
        Fail immediately because the perf-smoke success path should not load signal matrices.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Requested indicator identifier.
        Returns:
            Any: Never returns successfully.
        Assumptions:
            Successful family-plugin rollout evidence should avoid the universal shortlist path.
        Raises:
            AssertionError: Always, because this loader must not be used on the success path.
        Side Effects:
            Increments the in-memory call counter before failing.
        """
        _ = context, timeframe, indicator_id
        self.calls += 1
        raise AssertionError("signal matrices must not be loaded on the family-plugin path")

    def load_signal_rows(
        self,
        *,
        context: Any,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> Any:
        """
        Fail immediately because the perf-smoke success path should not load signal-row subsets.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Requested indicator identifier.
            row_selection: Requested exact row subset.
        Returns:
            Any: Never returns successfully.
        Assumptions:
            Successful family-plugin rollout evidence should avoid the universal shortlist path.
        Raises:
            AssertionError: Always, because this loader must not be used on the success path.
        Side Effects:
            Increments the in-memory call counter before failing.
        """
        _ = context, timeframe, indicator_id, row_selection
        self.calls += 1
        raise AssertionError("signal-row subsets must not be loaded on the family-plugin path")


def test_backtest_family_plugin_rollout_v2_reduces_ma_stage_a_space_without_fallback_io(
) -> None:
    """
    Verify the first MA-family plugin shrinks Stage A space without using universal fallback IO.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Perf-smoke evidence for the first plugin is additive: it should demonstrate bounded Stage
        A reduction for a pure MA-family request without creating a special engine.
    Raises:
        AssertionError: If the plugin path does not reduce the runtime plan or unexpectedly falls
            back to universal artifact scoring.
    Side Effects:
        None.
    """
    price_loader = _FailIfCalledPriceLoader()
    signal_loader = _FailIfCalledSignalLoader()
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    exact_plan = _runtime_plan_fixture(
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40, 80, 120)),
                ),
                variants=6,
            ),
            BacktestIndicatorPlanV2(
                indicator_id="ma.sma",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40, 80, 120)),
                ),
                variants=6,
            ),
            BacktestIndicatorPlanV2(
                indicator_id="ma.vwma",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(5, 10, 20, 40, 80, 120)),),
                variants=6,
            ),
        ),
        signal_axes=(
            BacktestSignalAxisPlanV2(
                indicator_id="ma.ema",
                param_name="threshold",
                values=(0.25, 0.5),
            ),
            BacktestSignalAxisPlanV2(
                indicator_id="ma.sma",
                param_name="threshold",
                values=(0.25, 0.5),
            ),
        ),
        max_candidates=54,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=exact_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "family_plugin"
    assert reduced_plan.family_plugin_registry_status == "resolved"
    assert reduced_plan.family_plugin_warning is None
    assert reduced_plan.family_plugin_proposal is not None
    assert reduced_plan.stage_a_variants_total == 52
    assert reduced_plan.stage_a_variants_total < exact_plan.stage_a_variants_total
    assert reduced_plan.retained_compute_variants_total == 13
    assert price_loader.calls == 0
    assert signal_loader.calls == 0


def _runtime_plan_fixture(
    *,
    indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
    max_candidates: int,
) -> BacktestArtifactRuntimePlanV2:
    """
    Build a larger pure-MA runtime-plan fixture for family-plugin perf-smoke evidence.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        indicator_plans: Planner-owned indicator plans in exact runtime order.
        signal_axes: Planner-owned signal axes in exact runtime order.
        max_candidates: Explicit shortlist cap for the opt-in `hybrid_family` profile.
    Returns:
        BacktestArtifactRuntimePlanV2: Minimal valid runtime plan for perf-smoke rollout tests.
    Assumptions:
        Perf-smoke coverage needs only single-risk expansion and deterministic mixed-radix sizing.
    Raises:
        ValueError: Propagated if the constructed plan violates runtime-plan invariants.
    Side Effects:
        None.
    """
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
        execution_profile=_hybrid_family_profile_fixture(max_candidates=max_candidates),
        instrument_id_literal="binance:btc-usdt",
        timeframe_code="15m",
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
    Build one explicit runtime-enabled `hybrid_family` profile for perf-smoke rollout tests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - configs/test/backtest.yaml

    Args:
        max_candidates: Explicit shortlist cap published by the profile fixture.
    Returns:
        object: Runtime-enabled `hybrid_family` profile fixture.
    Assumptions:
        Perf-smoke rollout evidence stays internal-only and uses the same opt-in profile gating
        as the live runtime path.
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


def _artifact_context_fixture() -> Any:
    """
    Build the minimal slot-pinned runtime context fixture used by perf-smoke builder tests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py

    Args:
        None.
    Returns:
        Any: Simple namespace matching the slot-pinned runtime protocol fields used by tests.
    Assumptions:
        Successful family-plugin rollout does not dereference artifact paths on the success path.
    Raises:
        None.
    Side Effects:
        None.
    """
    return SimpleNamespace(
        slot="slot_a",
        generation=1,
        asof_date="2025-01-01",
        manifest_hash="m" * 64,
        coordinates=SimpleNamespace(market_id=1, symbol="BTCUSDT"),
    )


def _target_time_range_fixture() -> TimeRange:
    """
    Build one deterministic request time range fixture for perf-smoke builder tests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_family_plugin_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/shared_kernel/primitives.py

    Args:
        None.
    Returns:
        TimeRange: Deterministic half-open UTC time range used by perf-smoke tests.
    Assumptions:
        Successful family-plugin rollout does not consult the actual target slice on the success
        path, but the builder contract still requires a valid time range.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    return TimeRange(
        start=UtcTimestamp(start),
        end=UtcTimestamp(end),
    )
