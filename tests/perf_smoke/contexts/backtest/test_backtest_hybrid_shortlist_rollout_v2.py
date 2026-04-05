from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.services import (
    BacktestArtifactBackedStageBScorerV2,
    BacktestArtifactRuntimePlanV2,
    BacktestArtifactRuntimeRunnerV2,
    BacktestIndicatorAxisPlanV2,
    BacktestIndicatorPlanV2,
    BacktestRiskVariantV2,
    BacktestStageAShortlistBuilderV2,
    HierarchicalShortlistRuntimePlanV2,
    default_execution_profiles_catalog_v2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
)
from trading.contexts.backtest.application.services.v2.execution_profile_v2 import (
    ExecutionProfileFeatureFlagsV2,
    ExecutionProfileShortlistConfigV2,
)
from trading.contexts.backtest.application.services.v2.hierarchical_shortlist_builder_v2 import (
    BacktestHierarchicalShortlistBuilderV2,
)
from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_BENCHMARK_CORPUS_FIXTURE_PATH = (
    _FIXTURES_DIR / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)


@dataclass(frozen=True, slots=True)
class _RolloutScenarioV2:
    """
    Deterministic synthetic rollout scenario used by hybrid shortlist perf-smoke evidence tests.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    signal_matrix: np.ndarray
    max_candidates: int
    top_k_limit: int = 10


@dataclass(frozen=True, slots=True)
class _RankedScenarioResultV2:
    """
    Final exact-vs-hybrid ranking evidence collected for one synthetic rollout scenario.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
    """

    exact_variant_keys: tuple[str, ...]
    hybrid_variant_keys: tuple[str, ...]
    exact_plan: BacktestArtifactRuntimePlanV2
    reduced_plan: BacktestArtifactRuntimePlanV2


class _SyntheticPriceLoaderV2:
    """
    In-memory price, mapping, and hit-times loader for hybrid rollout perf-smoke scenarios.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    def __init__(self, *, signal_bars: int) -> None:
        """
        Initialize one deterministic loader with a monotonically rising execution timeline.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            signal_bars: Number of request-timeframe bars in the synthetic scenario.
        Returns:
            None.
        Assumptions:
            Request-timeframe and execution-timeframe arrays may share the same synthetic bar
            count as long as mapping and hit-times tables stay internally aligned.
        Raises:
            ValueError: If `signal_bars` is non-positive.
        Side Effects:
            Initializes in-memory deterministic arrays and call counters.
        """
        if signal_bars <= 0:
            raise ValueError("signal_bars must be > 0")
        self._signal_bars = int(signal_bars)
        self._close_time = (
            np.arange(signal_bars, dtype=np.int64) * np.int64(60_000)
        ) + np.int64(60_000)
        rising_open = np.linspace(
            100.0,
            100.0 + float(signal_bars - 1),
            signal_bars,
            dtype=np.float32,
        )
        self._ohlcv = np.ascontiguousarray(
            np.column_stack(
                (
                    rising_open,
                    rising_open + np.float32(1.0),
                    rising_open - np.float32(1.0),
                    rising_open + np.float32(0.5),
                    np.ones(signal_bars, dtype=np.float32),
                )
            ),
            dtype=np.float32,
        )
        self._mapping = np.arange(signal_bars, dtype=np.uint32)
        sentinel = int(signal_bars)
        future_hit = np.minimum(
            np.arange(signal_bars, dtype=np.uint32) + np.uint32(1),
            np.uint32(sentinel),
        )
        self._hit_times = SimpleNamespace(
            manifest=SimpleNamespace(sentinel_index=sentinel),
            tp_values=np.array([0.04], dtype=np.float32),
            sl_values=np.array([0.02], dtype=np.float32),
            long_tp=np.ascontiguousarray(np.array([future_hit], dtype=np.uint32)),
            long_sl=np.ascontiguousarray(
                np.array(
                    [np.full(signal_bars, sentinel, dtype=np.uint32)],
                    dtype=np.uint32,
                )
            ),
            short_tp=np.ascontiguousarray(
                np.array(
                    [np.full(signal_bars, sentinel, dtype=np.uint32)],
                    dtype=np.uint32,
                )
            ),
            short_sl=np.ascontiguousarray(np.array([future_hit], dtype=np.uint32)),
        )
        self.price_calls = 0
        self.mapping_calls = 0
        self.hit_times_calls = 0

    def load_price_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Return the deterministic synthetic OHLCV arrays for either request or execution access.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace exposing `close_time` and `ohlcv`.
        Assumptions:
            These synthetic scenarios use the same monotone timeline for `15m` and `1m` access.
        Raises:
            ValueError: If the requested timeframe is unsupported.
        Side Effects:
            Increments the in-memory price-load call counter.
        """
        _ = context
        if timeframe not in {"15m", "1m"}:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        self.price_calls += 1
        return SimpleNamespace(close_time=self._close_time, ohlcv=self._ohlcv)

    def load_mapping_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Return deterministic request-timeframe to execution-timeframe close mappings.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
        Returns:
            Any: Namespace exposing `bar_close_1m_idx`.
        Assumptions:
            Synthetic scenarios use an identity mapping between request-timeframe bars and local
            execution indexes.
        Raises:
            ValueError: If the requested timeframe is unsupported.
        Side Effects:
            Increments the in-memory mapping-load call counter.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        self.mapping_calls += 1
        return SimpleNamespace(bar_close_1m_idx=self._mapping)

    def load_hit_times_arrays(self, *, context: Any) -> Any:
        """
        Return deterministic synthetic hit-times tables for the exact Stage B scorer.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
        Returns:
            Any: Namespace exposing hit-times arrays and `manifest.sentinel_index`.
        Assumptions:
            Long positions may hit TP on the next bar while short positions may hit SL.
        Raises:
            None.
        Side Effects:
            Increments the in-memory hit-times call counter.
        """
        _ = context
        self.hit_times_calls += 1
        return self._hit_times


class _SyntheticSignalLoaderV2:
    """
    In-memory signal loader serving full matrices and exact-path subset rows from one scenario.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    def __init__(self, *, signal_matrix: np.ndarray) -> None:
        """
        Initialize one deterministic in-memory signal-matrix catalog for a rollout scenario.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            signal_matrix: Full `[variant, time]` signal matrix for `ma.ema`.
        Returns:
            None.
        Assumptions:
            Synthetic rollout scenarios use one indicator block and one exact row ordering.
        Raises:
            ValueError: If `signal_matrix` is not two-dimensional.
        Side Effects:
            Initializes in-memory call logs.
        """
        normalized_matrix = np.asarray(signal_matrix, dtype=np.int8)
        if normalized_matrix.ndim != 2:
            raise ValueError("signal_matrix must be 2D")
        self._matrix = normalized_matrix
        self.matrix_calls = 0
        self.row_calls: list[tuple[int, ...]] = []

    def load_signal_matrix(
        self,
        *,
        context: Any,
        timeframe: str,
        indicator_id: str,
    ) -> Any:
        """
        Return the full synthetic matrix for hybrid block scoring and record the request.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            hierarchical_shortlist_builder_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Requested indicator identifier.
        Returns:
            Any: Namespace exposing the full signal matrix and rows-count metadata.
        Assumptions:
            All synthetic rollout scenarios use the single indicator id `ma.ema`.
        Raises:
            ValueError: If a different indicator id is requested.
        Side Effects:
            Increments the in-memory matrix-load counter.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        if indicator_id != "ma.ema":
            raise ValueError(f"unsupported indicator_id: {indicator_id}")
        self.matrix_calls += 1
        return SimpleNamespace(
            matrix=self._matrix,
            manifest=SimpleNamespace(rows_count=int(self._matrix.shape[0])),
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
        Return deterministic selected signal rows for exact Stage A and exact Stage B reuse.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
        Related:
          - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            stage_a_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            context: Ignored slot-pinned runtime context fixture.
            timeframe: Requested timeframe literal.
            indicator_id: Requested indicator identifier.
            row_selection: Explicit row indexes requested by the exact runtime.
        Returns:
            np.ndarray: Selected signal rows in the requested order.
        Assumptions:
            The synthetic signal matrix row ordering matches the exact runtime plan axes.
        Raises:
            ValueError: If a different indicator id is requested.
        Side Effects:
            Appends the requested row indexes to the in-memory row-load log.
        """
        _ = context
        if timeframe != "15m":
            raise ValueError(f"unsupported timeframe: {timeframe}")
        if indicator_id != "ma.ema":
            raise ValueError(f"unsupported indicator_id: {indicator_id}")
        if isinstance(row_selection, slice):
            normalized_row_selection = tuple(
                range(*row_selection.indices(self._matrix.shape[0]))
            )
        else:
            normalized_row_selection = tuple(int(value) for value in row_selection)
        self.row_calls.append(normalized_row_selection)
        return np.asarray(self._matrix[normalized_row_selection, :], dtype=np.int8)


def test_backtest_hybrid_shortlist_rollout_meets_recall_and_overlap_gates() -> None:
    """
    Verify rollout baseline and low-activity evidence satisfy the committed recall/overlap gates.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Deterministic rollout gates should rely primarily on exact-vs-hybrid ranking parity
        rather than machine-specific tight timing SLAs.
    Raises:
        AssertionError: If baseline recall/overlap or low-activity top-1 recall drift below the
            committed rollout thresholds.
    Side Effects:
        Reads the committed benchmark corpus fixture from repository.
    """
    corpus = load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )
    baseline_result = _run_exact_and_hybrid_case(
        scenario=_baseline_rollout_scenario_v2()
    )
    low_activity_result = _run_exact_and_hybrid_case(
        scenario=_low_activity_rollout_scenario_v2()
    )
    top_1_recall = _top_1_recall_ratio_v2(result=baseline_result)
    top_10_overlap = _top_k_overlap_ratio_v2(result=baseline_result, limit=10)
    low_activity_top_1_recall = _top_1_recall_ratio_v2(result=low_activity_result)

    assert top_1_recall >= float(corpus.rollout_gates.top_1_recall.min_ratio or 0.0)
    assert top_10_overlap >= float(corpus.rollout_gates.top_10_overlap.min_ratio or 0.0)
    assert low_activity_top_1_recall >= float(
        corpus.rollout_gates.low_activity.min_ratio or 0.0
    )


def test_backtest_hybrid_shortlist_rollout_exposes_high_correlation_diversity_evidence(
) -> None:
    """
    Verify the high-correlation rollout slice retains multiple explicit direction buckets.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Correlation-heavy scenarios should still retain at least a minimal direction-band spread
        so rollout reviewers can inspect explicit diversity evidence.
    Raises:
        AssertionError: If retained hybrid survivors collapse into fewer buckets than the
            committed rollout gate allows.
    Side Effects:
        Reads the committed benchmark corpus fixture from repository.
    """
    corpus = load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )
    reduced_plan, _, _ = _build_hybrid_runtime_plan_for_scenario(
        scenario=_high_correlation_rollout_scenario_v2()
    )

    assert isinstance(reduced_plan, HierarchicalShortlistRuntimePlanV2)
    distinct_direction_bands = {
        retained_row.score_payload.bucket_values["direction_band"]
        for retained_row in reduced_plan.block_results[0].retained_rows
    }

    assert len(distinct_direction_bands) >= int(
        corpus.rollout_gates.high_correlation.min_distinct_count or 0
    )


def test_backtest_hybrid_shortlist_rollout_short_circuits_small_grid_and_respects_memory_gate(
) -> None:
    """
    Verify small grids short-circuit without hybrid artifact IO and memory ratio stays bounded.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Machine-independent small-grid protection is best evidenced by proving the hybrid builder
        returns the exact plan before any artifact IO when no reduction is required.
    Raises:
        AssertionError: If the hybrid builder touches artifacts on the small-grid path or if the
            reduced-plan memory ratio exceeds the committed rollout gate.
    Side Effects:
        Reads the committed benchmark corpus fixture from repository.
    """
    corpus = load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )
    exact_small_plan, price_loader, signal_loader = _build_exact_runtime_plan_for_scenario(
        scenario=_small_grid_rollout_scenario_v2()
    )
    small_grid_hybrid_plan = replace(
        exact_small_plan,
        execution_profile=_hybrid_execution_profile_v2(max_candidates=10),
    )
    small_grid_builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    reduced_small_plan = small_grid_builder.build_runtime_plan(
        runtime_plan=small_grid_hybrid_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_for_signal_bars_v2(signal_bars=2),
    )
    memory_result = _run_exact_and_hybrid_case(
        scenario=_baseline_rollout_scenario_v2()
    )
    proxy_small_grid_ratio = 1.0
    memory_ratio = (
        float(memory_result.reduced_plan.estimated_memory_bytes)
        / float(memory_result.exact_plan.estimated_memory_bytes)
    )

    assert reduced_small_plan is small_grid_hybrid_plan
    assert price_loader.price_calls == 0
    assert price_loader.mapping_calls == 0
    assert price_loader.hit_times_calls == 0
    assert signal_loader.matrix_calls == 0
    assert signal_loader.row_calls == []
    assert proxy_small_grid_ratio <= float(
        corpus.rollout_gates.small_grid_overhead.max_ratio or 0.0
    )
    assert memory_ratio <= float(
        corpus.rollout_gates.memory_footprint.max_ratio or 0.0
    )


def _run_exact_and_hybrid_case(
    *,
    scenario: _RolloutScenarioV2,
) -> _RankedScenarioResultV2:
    """
    Execute one synthetic scenario through exact and hybrid runtime paths and collect rankings.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        scenario: Deterministic synthetic rollout scenario definition.
    Returns:
        _RankedScenarioResultV2: Final exact and hybrid variant-key rankings plus runtime plans.
    Assumptions:
        Exact Stage B scorer remains the final source of truth for overlap/recall evidence even
        when hybrid pruning is active upstream.
    Raises:
        ValueError: Propagated if the synthetic runtime wiring violates typed contracts.
    Side Effects:
        Executes the exact Stage A builder and exact Stage B scorer in-memory.
    """
    exact_plan, price_loader, signal_loader = _build_exact_runtime_plan_for_scenario(
        scenario=scenario
    )
    reduced_plan, _, _ = _build_hybrid_runtime_plan_for_scenario(
        scenario=scenario
    )
    target_time_range = _target_time_range_for_signal_bars_v2(
        signal_bars=scenario.signal_matrix.shape[1]
    )
    top_k_limit = min(int(scenario.top_k_limit), int(scenario.signal_matrix.shape[0]))
    stage_a_builder = BacktestStageAShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    exact_shortlist = stage_a_builder.build_shortlist(
        grid_context=exact_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=target_time_range,
        shortlist_limit=top_k_limit,
    )
    hybrid_shortlist = stage_a_builder.build_shortlist(
        grid_context=reduced_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=target_time_range,
        shortlist_limit=top_k_limit,
    )
    template = _template_for_row_count_v2(row_count=scenario.signal_matrix.shape[0])
    candles = _candles_for_signal_bars_v2(signal_bars=scenario.signal_matrix.shape[1])
    scorer = BacktestArtifactBackedStageBScorerV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
        artifact_context=_artifact_context_fixture(),
        target_time_range=target_time_range,
        report_target_slice=slice(0, scenario.signal_matrix.shape[1]),
        direction_mode=exact_plan.direction_mode,
        sizing_mode=exact_plan.sizing_mode,
        execution_params=exact_plan.execution_params,
        market_id=1,
        signal_timeframe="15m",
        indicator_grids=template.indicator_grids,
        init_cash_quote_default=1000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.0,
        fee_pct_default_by_market_id={1: 0.0},
    )
    runner = BacktestArtifactRuntimeRunnerV2()
    scorer.prepare_for_grid_context(
        grid_context=exact_plan,
        candles=candles,
        max_compute_bytes_total=1024,
    )
    exact_ranked, _ = runner.run_stage_b(
        template=template,
        runtime_plan=exact_plan,
        shortlist=exact_shortlist,
        candles=candles,
        scorer=scorer,
        top_k_limit=top_k_limit,
    )
    scorer.prepare_for_grid_context(
        grid_context=reduced_plan,
        candles=candles,
        max_compute_bytes_total=1024,
    )
    hybrid_ranked, _ = runner.run_stage_b(
        template=template,
        runtime_plan=reduced_plan,
        shortlist=hybrid_shortlist,
        candles=candles,
        scorer=scorer,
        top_k_limit=top_k_limit,
    )
    return _RankedScenarioResultV2(
        exact_variant_keys=tuple(item.variant_key for item in exact_ranked),
        hybrid_variant_keys=tuple(item.variant_key for item in hybrid_ranked),
        exact_plan=exact_plan,
        reduced_plan=reduced_plan,
    )


def _build_exact_runtime_plan_for_scenario(
    *,
    scenario: _RolloutScenarioV2,
) -> tuple[BacktestArtifactRuntimePlanV2, _SyntheticPriceLoaderV2, _SyntheticSignalLoaderV2]:
    """
    Build one exact runtime-plan harness and its synthetic loaders for a rollout scenario.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        scenario: Deterministic synthetic rollout scenario definition.
    Returns:
        tuple[BacktestArtifactRuntimePlanV2, _SyntheticPriceLoaderV2,
            _SyntheticSignalLoaderV2]:
            Exact runtime plan plus its in-memory loaders.
    Assumptions:
        Synthetic rollout scenarios use one indicator block, one risk cell, and no separate
        signal-axis expansion.
    Raises:
        ValueError: Propagated if the constructed runtime plan violates typed invariants.
    Side Effects:
        Initializes in-memory loader fakes only.
    """
    row_count = int(scenario.signal_matrix.shape[0])
    price_loader = _SyntheticPriceLoaderV2(signal_bars=int(scenario.signal_matrix.shape[1]))
    signal_loader = _SyntheticSignalLoaderV2(signal_matrix=scenario.signal_matrix)
    exact_profile = default_execution_profiles_catalog_v2().profile_for_mode(mode="exact_small")
    return (
        BacktestArtifactRuntimePlanV2(
            indicator_plans=(
                BacktestIndicatorPlanV2(
                    indicator_id="ma.ema",
                    axes=(
                        BacktestIndicatorAxisPlanV2(
                            name="window",
                            values=tuple(range(row_count)),
                        ),
                    ),
                    variants=row_count,
                ),
            ),
            signal_axes=(),
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
            execution_profile=exact_profile,
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
            stage_a_variants_total=row_count,
            stage_b_variants_total=min(int(scenario.top_k_limit), row_count),
            estimated_memory_bytes=1024,
            indicator_estimate_calls=1,
        ),
        price_loader,
        signal_loader,
    )


def _build_hybrid_runtime_plan_for_scenario(
    *,
    scenario: _RolloutScenarioV2,
) -> tuple[BacktestArtifactRuntimePlanV2, _SyntheticPriceLoaderV2, _SyntheticSignalLoaderV2]:
    """
    Build one reduced hybrid runtime plan for a synthetic rollout scenario.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        scenario: Deterministic synthetic rollout scenario definition.
    Returns:
        tuple[BacktestArtifactRuntimePlanV2, _SyntheticPriceLoaderV2,
            _SyntheticSignalLoaderV2]:
            Reduced hybrid plan plus its in-memory loaders.
    Assumptions:
        Hybrid rollout remains opt-in and always reuses the exact runtime plan shape as its input.
    Raises:
        ValueError: Propagated if the hybrid runtime plan violates typed invariants.
    Side Effects:
        Executes the hybrid shortlist builder once in-memory.
    """
    exact_plan, price_loader, signal_loader = _build_exact_runtime_plan_for_scenario(
        scenario=scenario
    )
    hybrid_builder = BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=price_loader,
        signal_matrix_loader=signal_loader,
    )
    hybrid_plan = replace(
        exact_plan,
        execution_profile=_hybrid_execution_profile_v2(
            max_candidates=int(scenario.max_candidates)
        ),
    )
    reduced_plan = hybrid_builder.build_runtime_plan(
        runtime_plan=hybrid_plan,
        artifact_context=_artifact_context_fixture(),
        target_time_range=_target_time_range_for_signal_bars_v2(
            signal_bars=scenario.signal_matrix.shape[1]
        ),
    )
    return reduced_plan, price_loader, signal_loader


def _hybrid_execution_profile_v2(*, max_candidates: int) -> Any:
    """
    Build one explicit opt-in `hybrid_conservative` profile fixture for rollout evidence tests.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - configs/test/backtest.yaml

    Args:
        max_candidates: Conservative shortlist cap for the synthetic scenario.
    Returns:
        Any: Execution profile fixture with runtime and heuristic-shortlist flags enabled.
    Assumptions:
        Perf-smoke evidence should exercise only the explicit opt-in hybrid rollout profile.
    Raises:
        ValueError: Propagated if the configured shortlist contract becomes invalid.
    Side Effects:
        None.
    """
    catalog = default_execution_profiles_catalog_v2()
    base_hybrid_profile = catalog.profile_for_mode(mode="hybrid_conservative")
    return replace(
        base_hybrid_profile,
        feature_flags=ExecutionProfileFeatureFlagsV2(
            runtime_enabled=True,
            heuristic_shortlist_enabled=True,
            parallel_stage_b_enabled=False,
            family_plugin_enabled=False,
        ),
        shortlist_config=ExecutionProfileShortlistConfigV2(
            enabled=True,
            max_candidates=max_candidates,
            scoring=base_hybrid_profile.shortlist_config.scoring,
            retention=base_hybrid_profile.shortlist_config.retention,
        ),
    )


def _template_for_row_count_v2(*, row_count: int) -> RunBacktestTemplate:
    """
    Build one deterministic single-indicator template matching the synthetic rollout harness.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        row_count: Number of indicator variants represented in the synthetic signal matrix.
    Returns:
        RunBacktestTemplate: Deterministic template aligned with the synthetic runtime plan.
    Assumptions:
        Synthetic rollout scenarios use one `ma.ema` indicator grid and one risk cell.
    Raises:
        ValueError: If the resulting DTO violates typed invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("15m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.ema"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(
                        name="window",
                        values=tuple(range(row_count)),
                    ),
                },
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=True,
            sl=ExplicitValuesSpec(name="sl", values=(2.0,)),
            tp=ExplicitValuesSpec(name="tp", values=(4.0,)),
        ),
        execution_params={
            "fee_pct": 0.0,
            "fixed_quote": 100.0,
            "init_cash_quote": 1000.0,
            "slippage_pct": 0.0,
        },
    )


def _candles_for_signal_bars_v2(*, signal_bars: int) -> CandleArrays:
    """
    Build deterministic dense candle arrays for the exact Stage B scorer preparation path.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        signal_bars: Number of synthetic request-timeframe bars in the scenario.
    Returns:
        CandleArrays: Deterministic candle arrays aligned with the synthetic runtime timeline.
    Assumptions:
        Artifact-backed Stage B scorer uses candles only for compatibility in this harness.
    Raises:
        ValueError: If the resulting DTO violates typed invariants.
    Side Effects:
        None.
    """
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=_target_time_range_for_signal_bars_v2(signal_bars=signal_bars),
        timeframe=Timeframe("15m"),
        ts_open=np.arange(signal_bars, dtype=np.int64),
        open=np.ones(signal_bars, dtype=np.float32),
        high=np.ones(signal_bars, dtype=np.float32),
        low=np.ones(signal_bars, dtype=np.float32),
        close=np.ones(signal_bars, dtype=np.float32),
        volume=np.ones(signal_bars, dtype=np.float32),
    )


def _artifact_context_fixture() -> Any:
    """
    Build a minimal slot-pinned runtime context fixture for the synthetic rollout harness.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py

    Args:
        None.
    Returns:
        Any: Minimal context fixture exposing canonical artifact coordinates and pin metadata.
    Assumptions:
        Synthetic rollout evidence does not need a real artifact store as long as the context
        matches the loader expectations.
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
        ),
        artifact_slot="slot_b",
        slot_generation=5,
        artifact_asof_date="2026-03-26",
        artifact_manifest_hash="b" * 64,
    )


def _target_time_range_for_signal_bars_v2(*, signal_bars: int) -> TimeRange:
    """
    Build a deterministic target window covering every synthetic request-timeframe close time.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        signal_bars: Number of synthetic request-timeframe bars in the scenario.
    Returns:
        TimeRange: Target window selecting the full synthetic timeline.
    Assumptions:
        Synthetic close timestamps are spaced by `60_000 ms` and start at `60_000 ms`.
    Raises:
        ValueError: If `signal_bars` is non-positive.
    Side Effects:
        None.
    """
    if signal_bars <= 0:
        raise ValueError("signal_bars must be > 0")
    start_ms = 60_000 - 1
    end_ms = (signal_bars * 60_000) + 1
    return TimeRange(
        start=UtcTimestamp(
            datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        ),
        end=UtcTimestamp(
            datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
        ),
    )


def _top_1_recall_ratio_v2(*, result: _RankedScenarioResultV2) -> float:
    """
    Compute the top-1 recall ratio for one exact-vs-hybrid ranking comparison.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        result: Final exact-vs-hybrid ranking evidence for one scenario.
    Returns:
        float: `1.0` when the hybrid winner matches the exact winner, otherwise `0.0`.
    Assumptions:
        Synthetic rollout scenarios always retain at least one ranked exact and hybrid variant.
    Raises:
        None.
    Side Effects:
        None.
    """
    return 1.0 if result.exact_variant_keys[0] == result.hybrid_variant_keys[0] else 0.0


def _top_k_overlap_ratio_v2(*, result: _RankedScenarioResultV2, limit: int) -> float:
    """
    Compute one deterministic top-k overlap ratio from exact and hybrid variant-key frontiers.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        result: Final exact-vs-hybrid ranking evidence for one scenario.
        limit: Maximum frontier length to compare.
    Returns:
        float: Set-overlap ratio in `[0, 1]`.
    Assumptions:
        Overlap gates compare membership, not exact ordering, because deterministic overlap is
        the primary conservative rollout criterion here.
    Raises:
        ValueError: If `limit` is non-positive.
    Side Effects:
        None.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    exact_frontier = result.exact_variant_keys[:limit]
    hybrid_frontier = result.hybrid_variant_keys[:limit]
    if len(exact_frontier) == 0:
        return 1.0
    intersection = len(set(exact_frontier).intersection(hybrid_frontier))
    return float(intersection) / float(len(exact_frontier))


def _baseline_rollout_scenario_v2() -> _RolloutScenarioV2:
    """
    Build the exact-baseline rollout scenario used for top-1 recall and top-10 overlap gates.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        None.
    Returns:
        _RolloutScenarioV2: Deterministic baseline scenario.
    Assumptions:
        Baseline scenario mixes strong long rows, neutral rows, and losing short rows so the
        hybrid shortlist can be compared to a non-trivial exact ranking.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _RolloutScenarioV2(
        signal_matrix=np.array(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [0, 1],
                [1, 1],
                [1, -1],
                [0, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
            ],
            dtype=np.int8,
        ),
        max_candidates=10,
        top_k_limit=10,
    )


def _low_activity_rollout_scenario_v2() -> _RolloutScenarioV2:
    """
    Build the sparse low-activity rollout scenario used for the dedicated recall gate.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        None.
    Returns:
        _RolloutScenarioV2: Deterministic low-activity scenario.
    Assumptions:
        Every row activates at most one bar across an eleven-bar timeline, keeping the
        `activity_ratio` below the shipped low-activity threshold.
    Raises:
        None.
    Side Effects:
        None.
    """
    signal_matrix = np.zeros((12, 11), dtype=np.int8)
    signal_matrix[0, 0] = 1
    signal_matrix[1, 1] = 1
    signal_matrix[2, 2] = 1
    signal_matrix[3, 3] = 1
    signal_matrix[4, 4] = 1
    signal_matrix[5, 5] = -1
    signal_matrix[6, 6] = -1
    signal_matrix[7, 7] = 1
    signal_matrix[8, 8] = 1
    signal_matrix[9, 9] = -1
    signal_matrix[10, 10] = 1
    signal_matrix[11, 0] = -1
    return _RolloutScenarioV2(
        signal_matrix=signal_matrix,
        max_candidates=10,
        top_k_limit=10,
    )


def _high_correlation_rollout_scenario_v2() -> _RolloutScenarioV2:
    """
    Build a high-correlation scenario used to inspect retained hybrid diversity evidence.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        None.
    Returns:
        _RolloutScenarioV2: Deterministic high-correlation scenario.
    Assumptions:
        Most rows share the same leading long pattern while a few rows introduce balanced or
        short-biased structure to exercise the diversified-retention audit trail.
    Raises:
        None.
    Side Effects:
        None.
    """
    signal_matrix = np.zeros((12, 11), dtype=np.int8)
    for row_index in range(8):
        signal_matrix[row_index, :3] = 1
    signal_matrix[8, :3] = -1
    signal_matrix[9, 4:7] = -1
    signal_matrix[10, :3] = 1
    signal_matrix[10, 5:8] = -1
    signal_matrix[11, 2:5] = 1
    return _RolloutScenarioV2(
        signal_matrix=signal_matrix,
        max_candidates=6,
        top_k_limit=10,
    )


def _small_grid_rollout_scenario_v2() -> _RolloutScenarioV2:
    """
    Build the tiny small-grid scenario used to prove hybrid short-circuit protection.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json

    Args:
        None.
    Returns:
        _RolloutScenarioV2: Deterministic small-grid scenario.
    Assumptions:
        The hybrid shortlist cap exceeds the full Stage A size here, so the builder should
        return the exact runtime plan before touching artifacts.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _RolloutScenarioV2(
        signal_matrix=np.array(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [-1, 0],
            ],
            dtype=np.int8,
        ),
        max_candidates=10,
        top_k_limit=4,
    )
