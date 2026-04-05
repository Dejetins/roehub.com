from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorAxisPlanV2,
    BacktestIndicatorPlanV2,
    BacktestRiskVariantV2,
    BacktestSignalAxisPlanV2,
    FamilyPluginApplicabilityV2,
    FamilyPluginCircuitBreakerV2,
    FamilyPluginMetadataV2,
    FamilyPluginProposalResultV2,
    FamilyPluginRegistryV2,
    HierarchicalShortlistRuntimePlanV2,
)
from trading.contexts.backtest.application.services.v2.execution_profile_v2 import (
    ExecutionProfileFeatureFlagsV2,
    ExecutionProfileShortlistConfigV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest.application.services.v2.hierarchical_shortlist_builder_v2 import (
    BacktestHierarchicalShortlistBuilderV2,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


class _RecordingPriceLoader:
    """
    Minimal artifact price-loader fake recording whether hybrid planning touches price artifacts.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
    """

    def __init__(self, *, close_time: np.ndarray) -> None:
        """
        Initialize one recording price-loader fake for deterministic hybrid-builder tests.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            close_time: Request-timeframe close timestamps returned for every load.
        Returns:
            None.
        Assumptions:
            Hybrid builder tests need only request-timeframe `close_time` access.
        Raises:
            None.
        Side Effects:
            Initializes an in-memory call counter.
        """
        self._close_time = np.asarray(close_time, dtype=np.int64)
        self.calls = 0

    def load_price_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Return deterministic request-timeframe prices and record the loader call.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace exposing `close_time`.
        Assumptions:
            These tests score only request-timeframe signal windows and do not need OHLCV data.
        Raises:
            ValueError: If a non-request timeframe is requested.
        Side Effects:
            Increments the in-memory call counter.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        self.calls += 1
        return SimpleNamespace(close_time=self._close_time)

    def load_mapping_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Return a deterministic identity mapping fixture for protocol completeness.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace exposing a trivial `tf_to_1m` mapping.
        Assumptions:
            Unit tests in this module do not exercise mapping loads, but the loader protocol now
            requires the method.
        Raises:
            ValueError: If a non-request timeframe is requested.
        Side Effects:
            None.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        return SimpleNamespace(
            tf_to_1m=np.arange(self._close_time.shape[0], dtype=np.int32)
        )

    def load_hit_times_arrays(self, *, context: Any) -> Any:
        """
        Return an empty hit-times fixture for protocol completeness in unit tests.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
        Returns:
            Any: Empty namespace placeholder.
        Assumptions:
            These tests do not touch hit-times arrays on the hybrid shortlist path.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = context
        return SimpleNamespace()


class _RecordingSignalLoader:
    """
    Minimal signal-loader fake exposing both full-matrix and subset-row contracts.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
    """

    def __init__(self, *, matrices_by_indicator: dict[str, np.ndarray]) -> None:
        """
        Initialize one deterministic in-memory signal artifact catalog.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            matrices_by_indicator: Full signal matrices keyed by indicator id.
        Returns:
            None.
        Assumptions:
            Matrix rows are already ordered exactly like the prepared runtime plan axes.
        Raises:
            None.
        Side Effects:
            Initializes in-memory call logs.
        """
        self._matrices_by_indicator = {
            indicator_id: np.asarray(matrix, dtype=np.int8)
            for indicator_id, matrix in matrices_by_indicator.items()
        }
        self.matrix_calls: list[str] = []
        self.row_calls: list[tuple[str, tuple[int, ...]]] = []

    def load_signal_matrix(
        self,
        *,
        context: Any,
        timeframe: str,
        indicator_id: str,
    ) -> Any:
        """
        Return one full signal matrix for hybrid block scoring and record the request.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py

        Args:
            indicator_id: Indicator identifier used to resolve the in-memory matrix.
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Indicator identifier used to resolve the in-memory matrix.
        Returns:
            Any: Namespace exposing `matrix` and `manifest.rows_count`.
        Assumptions:
            Hybrid shortlist planning loads one full matrix per indicator block.
        Raises:
            KeyError: If the indicator id is unknown in the test fixture.
        Side Effects:
            Appends one indicator id to the matrix-load log.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        matrix = self._matrices_by_indicator[indicator_id]
        self.matrix_calls.append(indicator_id)
        return SimpleNamespace(
            matrix=matrix,
            manifest=SimpleNamespace(rows_count=int(matrix.shape[0])),
        )

    def load_signal_rows(
        self,
        *,
        context: Any,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> np.ndarray:
        """
        Return deterministic selected signal rows for exact-path survivor expansion tests.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Indicator identifier used to resolve the in-memory matrix.
            row_selection: Explicit row indexes requested by the exact runtime.
        Returns:
            np.ndarray: Selected rows in the requested order.
        Assumptions:
            Hybrid builder tests call `load_signal_rows(...)` only when exact shortlist expansion
            is exercised after the reduced runtime plan is produced.
        Raises:
            KeyError: If the indicator id is unknown in the test fixture.
        Side Effects:
            Appends one `(indicator_id, row_selection)` tuple to the row-load log.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        matrix = self._matrices_by_indicator[indicator_id]
        if isinstance(row_selection, slice):
            normalized_row_selection = tuple(
                range(*row_selection.indices(matrix.shape[0]))
            )
        else:
            normalized_row_selection = tuple(int(value) for value in row_selection)
        self.row_calls.append((indicator_id, normalized_row_selection))
        return np.asarray(matrix[normalized_row_selection, :], dtype=np.int8)


class _RecordingFamilyPlugin:
    """
    Minimal proposal-only family plugin test double with optional sleep/error behavior.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
    """

    def __init__(
        self,
        *,
        proposal_result: FamilyPluginProposalResultV2,
        sleep_seconds: float = 0.0,
        raised_error: Exception | None = None,
    ) -> None:
        """
        Initialize one configurable family-plugin test double.

        Docs:
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py

        Args:
            proposal_result: Proposal payload returned on success.
            sleep_seconds: Optional artificial latency used to trigger timeout handling.
            raised_error: Optional exception raised instead of returning the proposal.
        Returns:
            None.
        Assumptions:
            Builder tests need a narrowly configurable plugin double rather than a second
            production implementation.
        Raises:
            None.
        Side Effects:
            Initializes an in-memory call counter.
        """
        self.metadata = FamilyPluginMetadataV2(
            plugin_id=proposal_result.plugin_id,
            display_name="Recording MA plugin",
            applicability=FamilyPluginApplicabilityV2(
                execution_profile_modes=("hybrid_family",),
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("row_shortlist",),
        )
        self._proposal_result = proposal_result
        self._sleep_seconds = sleep_seconds
        self._raised_error = raised_error
        self.calls = 0

    def propose(self, *, context: Any) -> FamilyPluginProposalResultV2:
        """
        Return one configured proposal payload or trigger the requested failure behavior.

        Docs:
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py

        Args:
            context: Narrow immutable planning context.
        Returns:
            FamilyPluginProposalResultV2: Configured proposal payload on success.
        Assumptions:
            Builder tests exercise routing/fallback semantics rather than plugin heuristics.
        Raises:
            Exception: Re-raises the configured `raised_error` when present.
        Side Effects:
            Increments the in-memory call counter and may sleep for the configured delay.
        """
        _ = context
        self.calls += 1
        if self._sleep_seconds > 0.0:
            time.sleep(self._sleep_seconds)
        if self._raised_error is not None:
            raise self._raised_error
        return self._proposal_result


def test_hierarchical_shortlist_builder_v2_reduces_runtime_plan_and_preserves_exact_order(
) -> None:
    """
    Verify hybrid builder prunes multi-block compute space and preserves exact Stage A ordering.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mixed-radix compute ordering remains canonical and signal-axis expansion must still emit
        exact Stage A survivors in ascending `stage_a_index`.
    Raises:
        AssertionError: If retained compute combinations or exact-order expansion drift.
    Side Effects:
        None.
    """
    runtime_plan = _runtime_plan_fixture(
        max_candidates=4,
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="alpha",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(10, 20, 30)),),
                variants=3,
            ),
            BacktestIndicatorPlanV2(
                indicator_id="beta",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(5, 15)),),
                variants=2,
            ),
        ),
        signal_axes=(
            BacktestSignalAxisPlanV2(
                indicator_id="signals.v1",
                param_name="mode",
                values=("strict", "relaxed"),
            ),
        ),
        stage_b_variants_total=8,
    )
    price_loader = _RecordingPriceLoader(close_time=np.array([2_599, 4_599], dtype=np.int64))
    signal_loader = _RecordingSignalLoader(
        matrices_by_indicator={
            "alpha": np.array([[1, -1], [1, 1], [-1, 0]], dtype=np.int8),
            "beta": np.array([[1, -1], [1, 1]], dtype=np.int8),
        }
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.retained_compute_variants_total == 2
    assert reduced_plan.stage_a_variants_total == 4
    assert tuple(
        variant.stage_a_index for variant in reduced_plan.retained_stage_a_variants
    ) == (0, 1, 2, 3)
    assert tuple(
        row.row_index for row in reduced_plan.block_results[0].retained_rows
    ) == (0, 1, 2)
    assert tuple(
        row.row_index for row in reduced_plan.block_results[1].retained_rows
    ) == (0, 1)
    assert signal_loader.matrix_calls == ["alpha", "beta"]
    assert price_loader.calls == 1


def test_hierarchical_shortlist_builder_v2_rejects_non_opt_in_profile_flags() -> None:
    """
    Verify hybrid builder fails fast when the resolved profile is not live-enabled for rollout.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone D hybrid runtime is opt-in only and exact profiles remain the canonical
        default behavior.
    Raises:
        AssertionError: If non-opt-in profiles are accepted by the hybrid builder.
    Side Effects:
        None.
    """
    catalog = default_execution_profiles_catalog_v2()
    disabled_hybrid_profile = catalog.profile_for_mode(mode="hybrid_conservative")
    runtime_plan = _runtime_plan_fixture(
        max_candidates=4,
        profile=disabled_hybrid_profile,
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="alpha",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),),
                variants=2,
            ),
        ),
        signal_axes=(),
        stage_b_variants_total=2,
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=_RecordingPriceLoader(
            close_time=np.array([2_599, 4_599], dtype=np.int64)
        ),
        signal_matrix_loader=_RecordingSignalLoader(
            matrices_by_indicator={
                "alpha": np.array([[1, -1], [1, 1]], dtype=np.int8),
            }
        ),
    )

    with pytest.raises(ValueError, match="hybrid shortlist runtime-enabled profile"):
        builder.build_runtime_plan(
            runtime_plan=runtime_plan,
            artifact_context=_artifact_context_fixture(),
            target_time_range=_target_time_range_fixture(),
        )


def test_hierarchical_shortlist_builder_v2_uses_ma_family_plugin_for_hybrid_family() -> None:
    """
    Verify `hybrid_family` keeps the shared runtime path but uses the MA-family proposal layer.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The first MA-family plugin must stay proposal-only and avoid reading universal fallback
        artifacts on the success path.
    Raises:
        AssertionError: If the reduced runtime plan does not preserve explicit family-plugin
            metadata or exact retained Stage A ordering.
    Side Effects:
        None.
    """
    price_loader = _RecordingPriceLoader(
        close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64)
    )
    signal_loader = _RecordingSignalLoader(
        matrices_by_indicator={
            "ma.ema": np.array([[1, -1, 1]], dtype=np.int8),
            "ma.sma": np.array([[1, -1, 1]], dtype=np.int8),
        }
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    runtime_plan = _runtime_plan_fixture(
        max_candidates=8,
        profile=_hybrid_family_profile_fixture(max_candidates=8),
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
        stage_b_variants_total=8,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "family_plugin"
    assert reduced_plan.family_plugin_registry_status == "resolved"
    assert reduced_plan.family_plugin_warning is None
    assert reduced_plan.family_plugin_proposal is not None
    assert reduced_plan.family_plugin_proposal.plugin_id == "ma.family.v1"
    assert reduced_plan.family_plugin_proposal.row_shortlist == (
        0,
        1,
        6,
        7,
        24,
        25,
        30,
        31,
    )
    assert reduced_plan.stage_a_variants_total == 8
    assert reduced_plan.retained_compute_variants_total == 4
    assert tuple(
        variant.stage_a_index for variant in reduced_plan.retained_stage_a_variants
    ) == (0, 1, 6, 7, 24, 25, 30, 31)
    assert price_loader.calls == 0
    assert signal_loader.matrix_calls == []


def test_hierarchical_shortlist_builder_v2_falls_back_for_mixed_family_hybrid_family() -> None:
    """
    Verify mixed-family `hybrid_family` requests degrade to the universal conservative path.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mixed-family requests must not receive MA-only behavior because the proposal layer stays
        explicit and deterministic.
    Raises:
        AssertionError: If mixed-family routing does not degrade to the universal shortlist path.
    Side Effects:
        None.
    """
    price_loader = _RecordingPriceLoader(
        close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64)
    )
    signal_loader = _RecordingSignalLoader(
        matrices_by_indicator={
            "ma.ema": np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int8),
            "momentum.trix": np.array([[1, 0, -1], [1, 1, -1]], dtype=np.int8),
        }
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    runtime_plan = _runtime_plan_fixture(
        max_candidates=2,
        profile=_hybrid_family_profile_fixture(max_candidates=2),
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),
                ),
                variants=2,
            ),
            BacktestIndicatorPlanV2(
                indicator_id="momentum.trix",
                axes=(BacktestIndicatorAxisPlanV2(name="window", values=(5, 10)),),
                variants=2,
            ),
        ),
        signal_axes=(),
        stage_b_variants_total=2,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "universal"
    assert reduced_plan.family_plugin_registry_status == "not_applicable"
    assert reduced_plan.family_plugin_warning is None
    assert reduced_plan.block_results
    assert price_loader.calls == 1
    assert signal_loader.matrix_calls == ["ma.ema", "momentum.trix"]


def test_hierarchical_shortlist_builder_v2_falls_back_on_family_plugin_timeout() -> None:
    """
    Verify family-plugin timeout records warning metadata and degrades to universal fallback.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        circuit_breaker_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Timeout handling must preserve the universal shortlist path instead of failing the run.
    Raises:
        AssertionError: If timeout warnings are omitted or fallback does not execute.
    Side Effects:
        None.
    """
    plugin = _RecordingFamilyPlugin(
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma.family.v1",
            row_shortlist=(0, 1),
        ),
        sleep_seconds=0.01,
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=_RecordingPriceLoader(
            close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64)
        ),
        signal_matrix_loader=_RecordingSignalLoader(
            matrices_by_indicator={
                "ma.ema": np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int8),
            }
        ),
        family_plugin_registry=FamilyPluginRegistryV2(plugins=(plugin,)),
    )
    runtime_plan = _runtime_plan_fixture(
        max_candidates=2,
        profile=replace(
            _hybrid_family_profile_fixture(max_candidates=2),
            family_plugin_budget_ms=1,
        ),
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),
                ),
                variants=2,
            ),
        ),
        signal_axes=(),
        stage_b_variants_total=2,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "universal"
    assert reduced_plan.family_plugin_registry_status == "resolved"
    assert reduced_plan.family_plugin_warning is not None
    assert reduced_plan.family_plugin_warning.reason == "timeout"
    assert reduced_plan.family_plugin_warning.plugin_id == "ma.family.v1"
    assert plugin.calls == 1


def test_hierarchical_shortlist_builder_v2_falls_back_on_family_plugin_error() -> None:
    """
    Verify family-plugin exceptions degrade to universal fallback with explicit warning payloads.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        circuit_breaker_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Plugin exceptions must surface as warning + universal fallback rather than breaking the
        exact shared runtime flow.
    Raises:
        AssertionError: If error warnings are omitted or the plugin is not attempted.
    Side Effects:
        None.
    """
    plugin = _RecordingFamilyPlugin(
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma.family.v1",
            row_shortlist=(0,),
        ),
        raised_error=RuntimeError("boom"),
    )
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=_RecordingPriceLoader(
            close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64)
        ),
        signal_matrix_loader=_RecordingSignalLoader(
            matrices_by_indicator={
                "ma.ema": np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int8),
            }
        ),
        family_plugin_registry=FamilyPluginRegistryV2(plugins=(plugin,)),
    )
    runtime_plan = _runtime_plan_fixture(
        max_candidates=2,
        profile=_hybrid_family_profile_fixture(max_candidates=2),
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),
                ),
                variants=2,
            ),
        ),
        signal_axes=(),
        stage_b_variants_total=2,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "universal"
    assert reduced_plan.family_plugin_registry_status == "resolved"
    assert reduced_plan.family_plugin_warning is not None
    assert reduced_plan.family_plugin_warning.reason == "error"
    assert "RuntimeError" in reduced_plan.family_plugin_warning.message
    assert plugin.calls == 1


def test_hierarchical_shortlist_builder_v2_falls_back_when_family_plugin_breaker_is_open(
) -> None:
    """
    Verify an open family-plugin circuit breaker skips plugin execution and uses fallback.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        circuit_breaker_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The per-run circuit breaker must short-circuit the family-plugin path before invocation.
    Raises:
        AssertionError: If the open-breaker warning is omitted or the plugin is still called.
    Side Effects:
        None.
    """
    plugin = _RecordingFamilyPlugin(
        proposal_result=FamilyPluginProposalResultV2(
            plugin_id="ma.family.v1",
            row_shortlist=(0,),
        ),
    )
    breaker = FamilyPluginCircuitBreakerV2(failure_threshold=1)
    breaker.record_error(plugin_id="ma.family.v1", error=RuntimeError("previous boom"))
    builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=_RecordingPriceLoader(
            close_time=np.array([2_599, 4_599, 6_599], dtype=np.int64)
        ),
        signal_matrix_loader=_RecordingSignalLoader(
            matrices_by_indicator={
                "ma.ema": np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int8),
            }
        ),
        family_plugin_registry=FamilyPluginRegistryV2(plugins=(plugin,)),
        family_plugin_circuit_breaker_factory=lambda: breaker,
    )
    runtime_plan = _runtime_plan_fixture(
        max_candidates=2,
        profile=_hybrid_family_profile_fixture(max_candidates=2),
        indicator_plans=(
            BacktestIndicatorPlanV2(
                indicator_id="ma.ema",
                axes=(
                    BacktestIndicatorAxisPlanV2(name="source", values=("close",)),
                    BacktestIndicatorAxisPlanV2(name="window", values=(10, 20)),
                ),
                variants=2,
            ),
        ),
        signal_axes=(),
        stage_b_variants_total=2,
    )

    reduced_plan = builder.build_runtime_plan(
        runtime_plan=runtime_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_fixture(),
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    assert reduced_plan.proposal_layer_source == "universal"
    assert reduced_plan.family_plugin_registry_status == "resolved"
    assert reduced_plan.family_plugin_warning is not None
    assert reduced_plan.family_plugin_warning.reason == "open_breaker"
    assert plugin.calls == 0


def _hybrid_family_profile_fixture(*, max_candidates: int):
    """
    Build one explicit opt-in `hybrid_family` profile fixture for proposal-layer tests.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - configs/test/backtest.yaml

    Args:
        max_candidates: Explicit shortlist cap published by the profile fixture.
    Returns:
        object: Runtime-enabled `hybrid_family` execution profile fixture.
    Assumptions:
        Tests keep `hybrid_family` internal-only and opt-in through profile flags plus the
        shared requested-runtime path.
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


def _runtime_plan_fixture(
    *,
    max_candidates: int,
    indicator_plans: tuple[BacktestIndicatorPlanV2, ...],
    signal_axes: tuple[BacktestSignalAxisPlanV2, ...],
    stage_b_variants_total: int,
    profile: object | None = None,
) -> BacktestArtifactRuntimePlanV2:
    """
    Build a deterministic runtime-plan fixture for hierarchical shortlist unit tests.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        max_candidates: Hybrid shortlist cap published by the profile fixture.
        indicator_plans: Prepared indicator plans in exact runtime order.
        signal_axes: Prepared signal axes in exact runtime order.
        stage_b_variants_total: Exact Stage B total used to derive shortlist envelope.
        profile: Optional explicit execution profile override.
    Returns:
        BacktestArtifactRuntimePlanV2: Immutable runtime-plan fixture.
    Assumptions:
        Unit tests exercise only single-risk expansion and deterministic mixed-radix addressing.
    Raises:
        ValueError: Propagated if the constructed runtime plan violates typed invariants.
    Side Effects:
        None.
    """
    catalog = default_execution_profiles_catalog_v2()
    hybrid_profile = replace(
        catalog.profile_for_mode(mode="hybrid_conservative"),
        feature_flags=ExecutionProfileFeatureFlagsV2(
            runtime_enabled=True,
            heuristic_shortlist_enabled=True,
            parallel_stage_b_enabled=False,
            family_plugin_enabled=False,
        ),
        shortlist_config=ExecutionProfileShortlistConfigV2(
            enabled=True,
            max_candidates=max_candidates,
            scoring=catalog.profile_for_mode(
                mode="hybrid_conservative"
            ).shortlist_config.scoring,
            retention=catalog.profile_for_mode(
                mode="hybrid_conservative"
            ).shortlist_config.retention,
        ),
    )
    selected_profile = hybrid_profile if profile is None else profile
    stage_a_variants_total = int(
        np.prod(tuple(plan.variants for plan in indicator_plans), dtype=np.int64)
    ) * max(1, int(np.prod(tuple(len(axis.values) for axis in signal_axes), dtype=np.int64)))
    return BacktestArtifactRuntimePlanV2(
        indicator_plans=indicator_plans,
        signal_axes=signal_axes,
        risk_variants=(
            BacktestRiskVariantV2(
                risk_index=0,
                risk_params={
                    "sl_enabled": True,
                    "sl_pct": 2.0,
                    "tp_enabled": True,
                    "tp_pct": 4.0,
                },
            ),
        ),
        execution_profile=selected_profile,  # type: ignore[arg-type]
        instrument_id_literal="BINANCE_SPOT_BTCUSDT",
        timeframe_code="15m",
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={
            "fee_pct": 0.0,
            "fixed_quote": 100.0,
            "init_cash_quote": 1000.0,
            "slippage_pct": 0.0,
        },
        stage_a_variants_total=stage_a_variants_total,
        stage_b_variants_total=stage_b_variants_total,
        estimated_memory_bytes=1024,
        indicator_estimate_calls=len(indicator_plans),
    )


def _artifact_context_fixture() -> Any:
    """
    Build a minimal slot-pinned runtime context fixture for hybrid shortlist tests.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py

    Args:
        None.
    Returns:
        Any: Minimal context fixture exposing canonical artifact coordinates.
    Assumptions:
        Hybrid builder tests do not need full slot metadata beyond market coordinates.
    Raises:
        None.
    Side Effects:
        None.
    """
    return SimpleNamespace(
        coordinates=SimpleNamespace(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _target_time_range_fixture() -> TimeRange:
    """
    Build the deterministic target range covering both synthetic request-timeframe bars.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        None.
    Returns:
        TimeRange: Requested runtime window selecting both synthetic close timestamps.
    Assumptions:
        Synthetic `15m` close timestamps are `[2599, 4599]` milliseconds.
    Raises:
        None.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(1970, 1, 1, 0, 0, 5, tzinfo=timezone.utc)),
    )
