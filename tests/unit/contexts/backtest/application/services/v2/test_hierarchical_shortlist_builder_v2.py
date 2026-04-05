from __future__ import annotations

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
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
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
          - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

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
          - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

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


class _RecordingSignalLoader:
    """
    Minimal signal-loader fake exposing both full-matrix and subset-row contracts.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
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
          - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

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

    def load_signal_matrix(self, *, indicator_id: str, **kwargs: Any) -> Any:
        """
        Return one full signal matrix for hybrid block scoring and record the request.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/
            test_hierarchical_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

        Args:
            indicator_id: Indicator identifier used to resolve the in-memory matrix.
            **kwargs: Ignored loader keyword arguments.
        Returns:
            Any: Namespace exposing `matrix` and `manifest.rows_count`.
        Assumptions:
            Hybrid shortlist planning loads one full matrix per indicator block.
        Raises:
            KeyError: If the indicator id is unknown in the test fixture.
        Side Effects:
            Appends one indicator id to the matrix-load log.
        """
        _ = kwargs
        matrix = self._matrices_by_indicator[indicator_id]
        self.matrix_calls.append(indicator_id)
        return SimpleNamespace(
            matrix=matrix,
            manifest=SimpleNamespace(rows_count=int(matrix.shape[0])),
        )

    def load_signal_rows(
        self,
        *,
        indicator_id: str,
        row_selection: tuple[int, ...],
        **kwargs: Any,
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
            indicator_id: Indicator identifier used to resolve the in-memory matrix.
            row_selection: Explicit row indexes requested by the exact runtime.
            **kwargs: Ignored loader keyword arguments.
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
        _ = kwargs
        matrix = self._matrices_by_indicator[indicator_id]
        normalized_row_selection = tuple(int(value) for value in row_selection)
        self.row_calls.append((indicator_id, normalized_row_selection))
        return np.asarray(matrix[normalized_row_selection, :], dtype=np.int8)


def test_hierarchical_shortlist_builder_v2_reduces_multi_indicator_runtime_plan_and_preserves_exact_order(
) -> None:
    """
    Verify hybrid builder prunes multi-block compute space and preserves exact Stage A ordering.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
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
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
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

    with pytest.raises(ValueError, match="hybrid_conservative runtime-enabled profile"):
        builder.build_runtime_plan(
            runtime_plan=runtime_plan,
            artifact_context=_artifact_context_fixture(),
            target_time_range=_target_time_range_fixture(),
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
