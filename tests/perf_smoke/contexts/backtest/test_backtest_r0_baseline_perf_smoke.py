from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast
from uuid import UUID

import numpy as np
import yaml

from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest.application.dto import (
    BacktestRiskGridSpec,
    RunBacktestRequest,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import CurrentUser
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactSlotResolverV2,
    BacktestStagedRunnerV1,
    CloseFillBacktestStagedScorerV1,
    YamlBacktestArtifactLoaderV2,
    load_backtest_runtime_acceleration_benchmark_corpus_v2,
    read_backtest_runtime_acceleration_benchmark_corpus_payload_v2,
    serialize_backtest_runtime_acceleration_benchmark_corpus_payload_v2,
)
from trading.contexts.backtest.application.services.v2 import (
    StageACompactTradeV2,
    StageBHitTimesSliceV2,
    run_reference_vs_fast_self_check_v2,
)
from trading.contexts.backtest.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestArtifactRuntimePlannerV2,
)
from trading.contexts.backtest.application.services.v2.execution_profile_v2 import (
    ExecutionProfileModeLiteralV2,
    ExecutionProfilesCatalogV2,
    default_execution_profiles_catalog_v2,
)
from trading.contexts.backtest.application.services.v2.stage_b_golden_fixtures_v2 import (
    load_stage_b_best_cell_replay_reference_case_v2,
)
from trading.contexts.backtest.application.use_cases import RunBacktestUseCase
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    TensorMeta,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_BENCHMARK_CORPUS_FIXTURE_PATH = (
    _FIXTURES_DIR / "backtest_runtime_acceleration_benchmark_corpus_v1.json"
)
_UNIT_STAGE_B_GOLDEN_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "unit"
    / "contexts"
    / "backtest"
    / "application"
    / "services"
    / "v2"
    / "fixtures"
    / "stage_b_golden_fixtures_v2.json"
)
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)
_PRINT_ENV_KEY = "ROEHUB_R0_BASELINE_PRINT"


@dataclass(frozen=True, slots=True)
class _R0BenchmarkScenario:
    """
    Deterministic benchmark scenario loaded from milestone-scoped fixture manifest.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    """

    scenario_id: str
    execution_class: str
    timeframe: str
    target_bars: int
    warmup_bars: int
    indicator_windows: tuple[int, ...]
    tp_values: tuple[float, ...]
    sl_values: tuple[float, ...]
    top_k: int
    preselect: int
    top_trades_n: int
    expected_clickhouse_hot_path_calls: int
    expected_v2_clickhouse_hot_path_calls: int
    expected_v2_indicator_compute_calls: int
    expected_hot_path_cost_reduction_min: int

    def __post_init__(self) -> None:
        """
        Validate one deterministic benchmark scenario fixture.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Fixture values are authored as positive finite scalars and ordered arrays.
        Raises:
            ValueError: If one scenario field violates R0 benchmark invariants.
        Side Effects:
            None.
        """
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if not self.execution_class.strip():
            raise ValueError("execution_class must be non-empty")
        if self.target_bars <= 0:
            raise ValueError("target_bars must be > 0")
        if self.warmup_bars <= 0:
            raise ValueError("warmup_bars must be > 0")
        if len(self.indicator_windows) == 0:
            raise ValueError("indicator_windows must be non-empty")
        if len(self.tp_values) == 0:
            raise ValueError("tp_values must be non-empty")
        if len(self.sl_values) == 0:
            raise ValueError("sl_values must be non-empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.preselect <= 0:
            raise ValueError("preselect must be > 0")
        if self.top_trades_n <= 0:
            raise ValueError("top_trades_n must be > 0")
        if self.expected_clickhouse_hot_path_calls < 0:
            raise ValueError("expected_clickhouse_hot_path_calls must be >= 0")
        if self.expected_v2_clickhouse_hot_path_calls < 0:
            raise ValueError("expected_v2_clickhouse_hot_path_calls must be >= 0")
        if self.expected_v2_indicator_compute_calls < 0:
            raise ValueError("expected_v2_indicator_compute_calls must be >= 0")
        if self.expected_hot_path_cost_reduction_min <= 0:
            raise ValueError("expected_hot_path_cost_reduction_min must be > 0")


class _R0ScenarioMeasurement(TypedDict):
    """
    Canonical measurement payload emitted by one deterministic R0 benchmark scenario.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    """

    scenario_id: str
    execution_class: str
    timeframe: str
    wall_clock_seconds: float
    cpu_time_seconds: float
    peak_traced_memory_bytes: int
    clickhouse_hot_path_calls: int
    indicator_compute_calls: int
    variants_returned: int


class _R10ArtifactV2ScenarioMeasurement(TypedDict):
    """
    Canonical measurement payload emitted by one artifact-backed v2 comparison scenario.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    scenario_id: str
    execution_class: str
    timeframe: str
    wall_clock_seconds: float
    cpu_time_seconds: float
    peak_traced_memory_bytes: int
    clickhouse_hot_path_calls: int
    indicator_compute_calls: int
    indicator_estimate_calls: int
    variants_returned: int


class _R10PerfComparison(TypedDict):
    """
    Deterministic comparison payload between legacy R0 baseline and artifact-backed v2 runtime.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    scenario_id: str
    baseline_hot_path_external_calls: int
    artifact_v2_hot_path_external_calls: int
    hot_path_cost_reduction: int
    baseline_wall_clock_seconds: float
    artifact_v2_wall_clock_seconds: float
    baseline_cpu_time_seconds: float
    artifact_v2_cpu_time_seconds: float


class _NullStrategyReader:
    """
    Strategy-reader stub for template-mode perf-smoke scenarios.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    """

    def load_any(self, *, strategy_id: UUID) -> None:
        """
        Return `None` because R0 benchmark scenarios use template mode only.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            None: Template-mode benchmark never resolves saved strategies.
        Assumptions:
            Template-mode route wiring bypasses strategy-reader payload usage.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return None


class _CountingCandleFeed:
    """
    Candle-feed stub producing deterministic dense `1m` candles and read counters.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic in-memory candle-feed counters.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            One `load_1m_dense` call proxies one ClickHouse hot-path read in local baseline.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.load_calls = 0

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Return deterministic dense `1m` candles for requested aligned time range.

        Args:
            market_id: Requested market identifier.
            symbol: Requested instrument symbol.
            time_range: Requested minute-aligned range.
        Returns:
            CandleArrays: Dense finite candle arrays used by timeline builder.
        Assumptions:
            Baseline fixtures keep the runtime path local and deterministic.
        Raises:
            ValueError: If time range is not divisible by one minute.
        Side Effects:
            Increments in-memory candle-read counter.
        """
        _ = market_id, symbol
        self.load_calls += 1
        return _build_dense_1m_from_time_range(time_range=time_range)


class _R0BaselineIndicatorCompute:
    """
    In-memory indicator-compute fake for deterministic R0 benchmark scenarios.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic compute counters and requested-layout trace.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Benchmark scenarios care about hot-path compute count and layout preference only.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.compute_calls = 0
        self.requested_layout_preferences: list[Layout | None] = []

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Materialize deterministic estimate payload from explicit source and param axes.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Deterministic estimate result for staged grid builder.
        Assumptions:
            R0 benchmark scenarios use explicit values only.
        Raises:
            ValueError: If estimated variants exceed guard.
        Side Effects:
            None.
        """
        axes: list[AxisDef] = []
        variants = 1

        if grid.source is not None:
            source_values = tuple(str(value) for value in grid.source.materialize())
            axes.append(AxisDef(name="source", values_enum=source_values))
            variants *= len(source_values)

        for param_name in sorted(grid.params.keys()):
            values = tuple(grid.params[param_name].materialize())
            axes.append(_axis_def(name=param_name, values=values))
            variants *= len(values)

        if variants > max_variants_guard:
            raise ValueError("variants exceed max_variants_guard")

        return EstimateResult(
            indicator_id=grid.indicator_id,
            axes=tuple(axes),
            variants=variants,
            max_variants_guard=max_variants_guard,
        )

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Return deterministic multi-variant tensor honoring requested layout preference.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Deterministic tensor used by close-fill scorer.
        Assumptions:
            Compute path is measured as part of current v1 hot-path baseline.
        Raises:
            ValueError: If bars count is non-positive.
        Side Effects:
            Increments in-memory compute call counter.
        """
        bars = int(req.candles.close.shape[0])
        if bars <= 0:
            raise ValueError("bars must be > 0")

        self.compute_calls += 1
        self.requested_layout_preferences.append(req.grid.layout_preference)
        pattern = np.asarray((1.0, 0.0, -1.0, 0.0), dtype=np.float32)
        window_spec = req.grid.params.get(
            "window",
            ExplicitValuesSpec(name="window", values=(1,)),
        )
        windows = tuple(window_spec.materialize())
        if len(windows) == 0:
            windows = (1,)
        values_time_major = np.empty((bars, len(windows)), dtype=np.float32)
        for index, raw_window in enumerate(windows):
            series = np.resize(pattern, bars).astype(np.float32)
            shift = int(int(raw_window) % 4)
            if shift > 0:
                series = np.roll(series, shift=shift)
            values_time_major[:, index] = series
        requested_layout = req.grid.layout_preference or Layout.TIME_MAJOR
        if requested_layout is Layout.VARIANT_MAJOR:
            values = np.ascontiguousarray(values_time_major.T, dtype=np.float32)
        else:
            values = np.ascontiguousarray(values_time_major, dtype=np.float32)
        return IndicatorTensor(
            indicator_id=req.grid.indicator_id,
            layout=requested_layout,
            axes=(AxisDef(name="variant", values_int=tuple(range(len(windows)))),),
            values=values,
            meta=TensorMeta(
                t=bars,
                variants=len(windows),
                nan_policy="propagate",
                compute_ms=0,
            ),
        )

    def warmup(self) -> None:
        """
        Provide no-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is unnecessary for deterministic in-memory benchmark fakes.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _EstimateOnlyArtifactIndicatorCompute(_R0BaselineIndicatorCompute):
    """
    Artifact-backed runtime compute fake that allows `estimate(...)` and forbids `compute(...)`.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic counters for artifact-backed runtime estimate-only planning.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R10-03 perf gates allow `IndicatorCompute.estimate(...)` for guard math but require
            `IndicatorCompute.compute(...)` to stay unused on sync/job hot paths.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__()
        self.estimate_calls = 0

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Count one artifact-backed planner estimate call and reuse baseline estimate semantics.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Deterministic estimate result for runtime planning.
        Assumptions:
            Artifact-backed runtime still uses estimate-only guard planning before Stage A.
        Raises:
            ValueError: If estimated variants exceed guard.
        Side Effects:
            Increments the estimate-call counter.
        """
        self.estimate_calls += 1
        return super().estimate(grid, max_variants_guard=max_variants_guard)

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Fail fast when artifact-backed runtime attempts forbidden `compute(...)` hot-path work.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Never returns because hot-path compute is forbidden here.
        Assumptions:
            `signal_tf + 1m_risk` runtime must consume shipped signals instead of materializing
            indicator tensors on the fly.
        Raises:
            AssertionError: Always, because `IndicatorCompute.compute(...)` is forbidden on the
                artifact-backed hot path.
        Side Effects:
            Increments the inherited compute-call counter for explicit diagnostics.
        """
        self.compute_calls += 1
        raise AssertionError("artifact-backed runtime must not call IndicatorCompute.compute(...)")


def test_r0_baseline_perf_smoke_collects_metric_snapshots() -> None:
    """
    Run deterministic R0 baseline scenarios and assert measurement shape/invariants.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Perf-smoke validates reproducible shape and counters, not machine-specific SLA numbers.
    Raises:
        AssertionError: If one scenario violates deterministic baseline invariants.
    Side Effects:
        Optionally prints canonical JSON measurements when `ROEHUB_R0_BASELINE_PRINT=1`.
    """
    corpus = _load_runtime_acceleration_benchmark_corpus()
    exact_baseline = corpus.slice_for_id(slice_id="exact_baseline")
    scenarios = _load_benchmark_scenarios()
    measurements = [
        _collect_legacy_scenario_measurement(scenario=scenario) for scenario in scenarios
    ]

    assert [scenario.scenario_id for scenario in scenarios] == list(
        exact_baseline.r0_scenario_ids
    )
    for scenario, measurement in zip(scenarios, measurements, strict=True):
        assert measurement["scenario_id"] == scenario.scenario_id
        assert measurement["execution_class"] == scenario.execution_class
        assert (
            measurement["clickhouse_hot_path_calls"]
            == scenario.expected_clickhouse_hot_path_calls
        )
        assert measurement["indicator_compute_calls"] >= 1
        assert 1 <= measurement["variants_returned"] <= scenario.top_k
        assert measurement["wall_clock_seconds"] >= 0.0
        assert measurement["cpu_time_seconds"] >= 0.0
        assert measurement["peak_traced_memory_bytes"] > 0

    if os.environ.get(_PRINT_ENV_KEY, "0") == "1":
        print(json.dumps({"scenarios": measurements}, sort_keys=True))


def test_r10_artifact_v2_perf_gates_reduce_hot_path_cost_vs_r0_baseline() -> None:
    """
    Compare artifact-backed v2 runtime against R0 baseline and enforce zero-call hot-path gates.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        R10-03 closes perf on deterministic counter-based evidence first: baseline hot path has
        positive external-call cost, while artifact-backed v2 must drive both families to zero.
        Wall-clock and CPU measurements are still collected for diagnostics but are not used as
        machine-specific CI gates.
    Raises:
        AssertionError: If artifact-backed runtime regresses zero-call gates or fails to reduce
            total external hot-path call cost against the approved R0 baseline.
    Side Effects:
        Creates temporary strict artifact trees and executes artifact-backed v2 runtime locally.
    """
    corpus = _load_runtime_acceleration_benchmark_corpus()
    exact_baseline = corpus.slice_for_id(slice_id="exact_baseline")
    scenarios = _load_benchmark_scenarios()

    assert [scenario.scenario_id for scenario in scenarios] == list(
        exact_baseline.r0_scenario_ids
    )

    for scenario in scenarios:
        baseline = _collect_legacy_scenario_measurement(scenario=scenario)
        artifact_v2 = _collect_artifact_v2_scenario_measurement(scenario=scenario)
        comparison = _build_perf_comparison(
            scenario_id=scenario.scenario_id,
            baseline=baseline,
            artifact_v2=artifact_v2,
        )

        assert artifact_v2["scenario_id"] == scenario.scenario_id
        assert artifact_v2["execution_class"] == scenario.execution_class
        assert (
            artifact_v2["clickhouse_hot_path_calls"]
            == scenario.expected_v2_clickhouse_hot_path_calls
        )
        assert (
            artifact_v2["indicator_compute_calls"]
            == scenario.expected_v2_indicator_compute_calls
        )
        assert artifact_v2["indicator_estimate_calls"] >= 1
        assert artifact_v2["variants_returned"] == baseline["variants_returned"]
        assert (
            comparison["hot_path_cost_reduction"]
            >= scenario.expected_hot_path_cost_reduction_min
        )


def test_r0_parity_scope_fixture_manifest_is_complete() -> None:
    """
    Verify parity-scope fixture manifest covers Stage A, legacy Stage B, and future v2 reference.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Args:
        None.
    Returns:
        None.
    Assumptions:
        R0 parity scope is document-driven and intentionally separates active
        vs reference-only cases, including the canonical R5-02 runtime-kernel
        reference path.
    Raises:
        AssertionError: If one required scope entry is missing or misclassified.
    Side Effects:
        None.
    """
    payload = json.loads((_FIXTURES_DIR / "r0_parity_scope.json").read_text(encoding="utf-8"))
    scopes = payload["parity_scopes"]
    scopes_by_id = {item["scope_id"]: item for item in scopes}
    assert [item["scope_id"] for item in scopes] == [
        "stage_a_no_risk",
        "stage_b_legacy_close_fill",
        "stage_b_signal_tf_1m_risk_reference",
    ]
    assert [item["status"] for item in scopes] == [
        "active",
        "active",
        "reference-only",
    ]
    assert scopes_by_id["stage_a_no_risk"]["reference_doc"] == (
        "docs/architecture/backtest/backtest-signals-from-indicators-v1.md"
    )
    assert scopes_by_id["stage_b_legacy_close_fill"]["reference_doc"] == (
        "docs/architecture/backtest/backtest-api-post-backtests-v1.md"
    )
    assert scopes_by_id["stage_b_signal_tf_1m_risk_reference"]["reference_doc"] == (
        "docs/architecture/backtest/backtest-runtime-kernels-v2.md"
    )
    assert scopes_by_id["stage_b_signal_tf_1m_risk_reference"]["reference_notebook"] == (
        "tests/notebook_tests/06_backtest_compute.ipynb"
    )
    assert scopes_by_id["stage_b_signal_tf_1m_risk_reference"]["golden_fixture_manifest"] == (
        "tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json"
    )
    assert scopes_by_id["stage_b_signal_tf_1m_risk_reference"]["closure_perf_smoke"] == (
        "tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py"
    )


def test_r10_parallel_tuning_contract_is_explicit_and_aligned() -> None:
    """
    Verify the post-P1 Stage A / Stage B tuning is explicit across defaults and env configs.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - configs/dev/backtest.yaml
      - configs/test/backtest.yaml
      - configs/prod/backtest.yaml
    Args:
        None.
    Returns:
        None.
    Assumptions:
        P2 tuning keeps `exact_small` serial, gives broader profiles real Stage A parallelism, and
        keeps every Stage A worker target at or below the shared `max_numba_threads` ceiling.
    Raises:
        AssertionError: If default catalog values or committed env configs drift from the tuned
            contract.
    Side Effects:
        Reads committed YAML config files from the repository.
    """
    expected_parallelism = {
        "exact_small": (1, 1),
        "exact_parallel": (4, 1),
        "hybrid_conservative": (4, 3),
        "hybrid_family": (3, 2),
    }
    catalog = default_execution_profiles_catalog_v2()

    for mode, (expected_stage_a_workers, expected_stage_b_workers) in expected_parallelism.items():
        profile = catalog.profile_for_mode(
            mode=cast(ExecutionProfileModeLiteralV2, mode)
        )
        assert profile.parallelism.stage_a_workers == expected_stage_a_workers
        assert profile.parallelism.stage_b_workers == expected_stage_b_workers

    repo_root = Path(__file__).resolve().parents[4]
    for env_name in ("dev", "test", "prod"):
        payload = yaml.safe_load(
            (repo_root / f"configs/{env_name}/backtest.yaml").read_text(encoding="utf-8")
        )
        execution_profiles = payload["backtest"]["execution_profiles"]
        cpu_ceiling = int(payload["backtest"]["cpu"]["max_numba_threads"])
        profiles_by_mode = {
            profile_payload["mode"]: profile_payload
            for profile_payload in execution_profiles["profiles"]
        }
        assert cpu_ceiling == 4

        for mode, (
            expected_stage_a_workers,
            expected_stage_b_workers,
        ) in expected_parallelism.items():
            profile_parallelism = profiles_by_mode[mode]["parallelism"]
            assert int(profile_parallelism["stage_a_workers"]) == expected_stage_a_workers
            assert int(profile_parallelism["stage_b_workers"]) == expected_stage_b_workers
            assert int(profile_parallelism["stage_a_workers"]) <= cpu_ceiling

        assert (
            int(profiles_by_mode["exact_parallel"]["parallelism"]["stage_a_workers"])
            > int(profiles_by_mode["exact_small"]["parallelism"]["stage_a_workers"])
        )
        assert (
            int(profiles_by_mode["exact_parallel"]["parallelism"]["stage_a_workers"])
            == cpu_ceiling
        )
        assert (
            int(profiles_by_mode["exact_parallel"]["parallelism"]["stage_b_workers"]) == 1
        )


def test_r5_stage_b_golden_fixture_manifest_tracks_contract_fixture_bytes() -> None:
    """
    Verify the R5-03 Stage B manifest points to the canonical contract fixture and published SHA.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        R5-03 keeps Stage B runtime validation separate from the R0 legacy close-fill baseline,
        while publishing one deterministic fixture baseline for future R6 kernel checks.
    Raises:
        AssertionError: If the contract fixture path, case order, or SHA drift unexpectedly.
    Side Effects:
        Reads fixture JSON files from repository.
    """
    corpus = _load_runtime_acceleration_benchmark_corpus()
    exact_baseline = corpus.slice_for_id(slice_id="exact_baseline")
    payload = json.loads(
        (_FIXTURES_DIR / "r5_stage_b_golden_cases.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["scope_id"] == "stage_b_signal_tf_1m_risk_reference"
    assert payload["status"] == "validation-baseline"
    assert payload["semantics"] == "signal_tf + 1m_risk"
    assert payload["contract_fixture"] == corpus.source_fixtures.stage_b_golden_fixture
    assert payload["case_order"] == list(exact_baseline.r5_stage_b_case_ids)
    assert payload["coverage"] == [
        "signal_tf + 1m_risk",
        "entry mapping request TF -> 1m",
        "TP/SL earliest hit",
        "earliest signal-exit mapping",
        "signal exit wins on equal bar",
        "SL wins TP tie",
        "entry_exec + 1",
        "exact best-cell replay",
        "metrics over compact trades",
        "sentinel_index",
        "golden fixtures",
    ]
    contract_fixture_path = _FIXTURES_DIR.parents[4] / payload["contract_fixture"]
    assert contract_fixture_path.is_file()
    assert sha256(contract_fixture_path.read_bytes()).hexdigest() == payload[
        "contract_fixture_sha256"
    ]


def test_r5_reference_vs_fast_self_check_runs_on_bounded_subset_only() -> None:
    """
    Verify perf smoke keeps the reference-vs-fast self-check on a deterministic bounded subset.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
      - tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Perf smoke must exercise the explicit self-check without moving the slow reference onto
        the default production hot path, so it validates only a smaller bounded subset.
    Raises:
        AssertionError: If the self-check stops honoring the requested bounded subset or parity.
    Side Effects:
        Reads the committed Stage B golden fixture catalog from repository.
    """
    best_cell_case = load_stage_b_best_cell_replay_reference_case_v2(
        path=_UNIT_STAGE_B_GOLDEN_FIXTURE_PATH
    )
    compact_trades = tuple(
        StageACompactTradeV2(
            entry_signal_idx=index,
            entry_exec_idx=trade.entry_exec,
            direction=trade.direction,
            sig_exit_signal_idx=None,
            sig_exit_exec_idx=trade.sig_exit_exec,
        )
        for index, trade in enumerate(best_cell_case.compact_trades)
    )
    hit_times = StageBHitTimesSliceV2(
        tp_values=np.asarray(
            tuple(float(value) - 1.0 for value in best_cell_case.level_factors.long_tp),
            dtype=np.float32,
        ),
        sl_values=np.asarray(
            tuple(1.0 - float(value) for value in best_cell_case.level_factors.long_sl),
            dtype=np.float32,
        ),
        long_tp=np.asarray(best_cell_case.hit_times.long_tp, dtype=np.int64),
        long_sl=np.asarray(best_cell_case.hit_times.long_sl, dtype=np.int64),
        short_tp=np.asarray(best_cell_case.hit_times.short_tp, dtype=np.int64),
        short_sl=np.asarray(best_cell_case.hit_times.short_sl, dtype=np.int64),
        sentinel_index=best_cell_case.hit_times.sentinel_index,
    )
    bounded_trade_count = max(1, len(compact_trades) - 1)
    bounded_tp_level_count = max(1, int(hit_times.tp_values.shape[0]) - 1)
    bounded_sl_level_count = max(1, int(hit_times.sl_values.shape[0]) - 1)

    self_check = run_reference_vs_fast_self_check_v2(
        compact_trades=compact_trades,
        hit_times=hit_times,
        exec_open=np.asarray(best_cell_case.prices.exec_open, dtype=np.float64),
        exec_close=np.asarray(best_cell_case.prices.exec_close, dtype=np.float64),
        fee_rate=float(best_cell_case.fee_rate),
        max_trade_count=bounded_trade_count,
        max_tp_level_count=bounded_tp_level_count,
        max_sl_level_count=bounded_sl_level_count,
        close_on_end=best_cell_case.close_on_end,
    )

    assert self_check.bounded_trade_count == bounded_trade_count
    assert self_check.bounded_tp_level_count == bounded_tp_level_count
    assert self_check.bounded_sl_level_count == bounded_sl_level_count
    assert self_check.bounded_trade_count < self_check.total_trade_count
    assert self_check.bounded_tp_level_count < self_check.total_tp_level_count
    assert self_check.bounded_sl_level_count < self_check.total_sl_level_count
    assert abs(self_check.max_abs_total_return_diff) <= 1e-9


def test_a3_runtime_acceleration_benchmark_corpus_manifest_is_complete() -> None:
    """
    Verify the D2+D3 benchmark corpus publishes deterministic exact and hybrid rollout slices.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
      - tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Milestone D keeps one explicit deterministic corpus for exact baseline, recall/diversity
        edge cases, small-grid overhead, and memory-footprint pressure without introducing CI SLA
        thresholds.
    Raises:
        AssertionError: If one required slice, cross-reference, or synthetic harness contract is
            missing.
    Side Effects:
        Reads committed benchmark fixtures from repository.
    """
    corpus = _load_runtime_acceleration_benchmark_corpus()
    r0_scenario_ids = tuple(
        scenario.scenario_id for scenario in _load_benchmark_scenarios()
    )
    r5_manifest = json.loads(
        (_FIXTURES_DIR / "r5_stage_b_golden_cases.json").read_text(encoding="utf-8")
    )

    assert corpus.source_fixtures.r0_benchmark_scenarios == (
        "tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json"
    )
    assert corpus.source_fixtures.r5_stage_b_manifest == (
        "tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json"
    )
    assert corpus.source_fixtures.stage_b_golden_fixture == (
        "tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json"
    )
    assert corpus.milestone_id == "A-F"
    assert corpus.epic_id == "A3+D2+D3+F2"
    assert corpus.slice_order == (
        "exact_baseline",
        "low_activity",
        "high_correlation",
        "small_grid_overhead",
        "memory_footprint",
        "medium_grids",
        "huge_grids",
        "multi_block",
    )
    assert corpus.rollout_gates.top_1_recall.metric == "top_1_recall"
    assert corpus.rollout_gates.top_1_recall.min_ratio == 0.99
    assert corpus.rollout_gates.top_10_overlap.metric == "top_10_overlap"
    assert corpus.rollout_gates.top_10_overlap.min_ratio == 0.9
    assert corpus.rollout_gates.low_activity.slice_id == "low_activity"
    assert corpus.rollout_gates.low_activity.min_ratio == 0.97
    assert corpus.rollout_gates.high_correlation.slice_id == "high_correlation"
    assert corpus.rollout_gates.high_correlation.min_distinct_count == 2
    assert corpus.rollout_gates.small_grid_overhead.max_ratio == 1.25
    assert corpus.rollout_gates.memory_footprint.max_ratio == 1.1

    exact_baseline = corpus.slice_for_id(slice_id="exact_baseline")
    assert exact_baseline.execution_profile_mode == "exact_parallel"
    assert exact_baseline.candidate_execution_profile_mode is None
    assert exact_baseline.rollout_scope == "exact_only"
    assert exact_baseline.stage_focus == ("stage_a", "stage_b", "finalizing")
    assert exact_baseline.r0_scenario_ids == r0_scenario_ids
    assert exact_baseline.r5_stage_b_case_ids == tuple(r5_manifest["case_order"])

    low_activity = corpus.slice_for_id(slice_id="low_activity")
    assert low_activity.candidate_execution_profile_mode == "hybrid_conservative"
    assert low_activity.rollout_scope == "hybrid_rollout"
    assert "low_activity" in low_activity.evaluation_focus
    assert "top_1_recall" in low_activity.evaluation_focus
    assert low_activity.synthetic_run_spec is not None
    assert low_activity.synthetic_run_spec.expected_stage_a_variants_total == 3
    assert low_activity.synthetic_run_spec.expected_stage_b_variants_total == 12

    high_correlation = corpus.slice_for_id(slice_id="high_correlation")
    assert high_correlation.candidate_execution_profile_mode == "hybrid_conservative"
    assert high_correlation.rollout_scope == "hybrid_rollout"
    assert "high_correlation" in high_correlation.evaluation_focus
    assert "diversity_evidence" in high_correlation.evaluation_focus
    assert high_correlation.synthetic_run_spec is not None
    assert high_correlation.synthetic_run_spec.expected_stage_b_variants_total == 36

    small_grid_overhead = corpus.slice_for_id(slice_id="small_grid_overhead")
    assert small_grid_overhead.execution_profile_mode == "exact_small"
    assert small_grid_overhead.candidate_execution_profile_mode == "hybrid_conservative"
    assert "small_grid_overhead" in small_grid_overhead.evaluation_focus
    assert "wall_clock_ratio" in small_grid_overhead.evaluation_focus
    assert small_grid_overhead.synthetic_run_spec is not None
    assert small_grid_overhead.synthetic_run_spec.total_candles_bars == 512
    assert small_grid_overhead.synthetic_run_spec.expected_stage_a_variants_total == 6
    assert small_grid_overhead.synthetic_run_spec.expected_stage_b_variants_total == 16
    assert small_grid_overhead.eta_fallback is not None
    assert small_grid_overhead.eta_fallback.stage_a_units_per_second == 4.0
    assert small_grid_overhead.eta_fallback.finalizing_seconds == 1

    memory_footprint = corpus.slice_for_id(slice_id="memory_footprint")
    assert memory_footprint.candidate_execution_profile_mode == "hybrid_conservative"
    assert "memory_footprint" in memory_footprint.evaluation_focus
    assert "peak_traced_memory_ratio" in memory_footprint.evaluation_focus
    assert memory_footprint.synthetic_run_spec is not None
    assert memory_footprint.synthetic_run_spec.expected_stage_a_variants_total == 10
    assert memory_footprint.synthetic_run_spec.expected_stage_b_variants_total == 90

    medium_grids = corpus.slice_for_id(slice_id="medium_grids")
    assert medium_grids.candidate_execution_profile_mode == "hybrid_conservative"
    assert "medium_grids" in medium_grids.evaluation_focus
    assert "benchmark_fallback" in medium_grids.evaluation_focus
    assert medium_grids.synthetic_run_spec is not None
    assert medium_grids.synthetic_run_spec.expected_stage_a_variants_total == 24
    assert medium_grids.synthetic_run_spec.expected_stage_b_variants_total == 96
    assert medium_grids.eta_fallback is not None
    assert medium_grids.eta_fallback.stage_b_units_per_second == 1.5

    huge_grids = corpus.slice_for_id(slice_id="huge_grids")
    assert huge_grids.candidate_execution_profile_mode == "hybrid_conservative"
    assert "huge_grids" in huge_grids.evaluation_focus
    assert huge_grids.synthetic_run_spec is not None
    assert huge_grids.synthetic_run_spec.expected_stage_a_variants_total == 96
    assert huge_grids.synthetic_run_spec.expected_stage_b_variants_total == 384
    assert huge_grids.eta_fallback is not None
    assert huge_grids.eta_fallback.finalizing_seconds == 4

    multi_block = corpus.slice_for_id(slice_id="multi_block")
    assert multi_block.candidate_execution_profile_mode == "hybrid_conservative"
    assert "multi_block" in multi_block.evaluation_focus
    assert "benchmark_fallback" in multi_block.evaluation_focus
    assert multi_block.synthetic_run_spec is not None
    assert multi_block.synthetic_run_spec.expected_stage_a_variants_total == 36
    assert multi_block.synthetic_run_spec.expected_stage_b_variants_total == 144
    assert multi_block.eta_fallback is not None
    assert multi_block.eta_fallback.stage_a_units_per_second == 2.5


def test_a3_runtime_acceleration_benchmark_corpus_serialization_is_byte_stable() -> None:
    """
    Verify the committed A3+D2+D3+F2 benchmark corpus keeps canonical byte-stable JSON formatting.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Canonical formatting is part of the reviewable benchmark-corpus contract.
    Raises:
        AssertionError: If canonical serialization drifts from the committed fixture bytes.
    Side Effects:
        Reads one committed JSON fixture from repository.
    """
    raw_payload = read_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )
    canonical_bytes = serialize_backtest_runtime_acceleration_benchmark_corpus_payload_v2(
        payload=raw_payload
    )

    assert canonical_bytes == _BENCHMARK_CORPUS_FIXTURE_PATH.read_bytes()


def _load_benchmark_scenarios() -> tuple[_R0BenchmarkScenario, ...]:
    """
    Load deterministic milestone-scoped benchmark scenarios from JSON fixture manifest.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    Args:
        None.
    Returns:
        tuple[_R0BenchmarkScenario, ...]: Ordered benchmark scenarios.
    Assumptions:
        Fixture schema is local, deterministic, and authored under version control.
    Raises:
        ValueError: If fixture shape is invalid.
    Side Effects:
        Reads one JSON fixture file from repository.
    """
    payload = json.loads(
        (_FIXTURES_DIR / "r0_benchmark_scenarios.json").read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != 1:
        raise ValueError("r0_benchmark_scenarios schema_version must be 1")
    scenarios: list[_R0BenchmarkScenario] = []
    for raw_scenario in payload.get("scenarios", []):
        scenarios.append(
            _R0BenchmarkScenario(
                scenario_id=str(raw_scenario["scenario_id"]),
                execution_class=str(raw_scenario["execution_class"]),
                timeframe=str(raw_scenario["timeframe"]),
                target_bars=int(raw_scenario["target_bars"]),
                warmup_bars=int(raw_scenario["warmup_bars"]),
                indicator_windows=tuple(
                    int(value) for value in raw_scenario["indicator_windows"]
                ),
                tp_values=tuple(float(value) for value in raw_scenario["tp_values"]),
                sl_values=tuple(float(value) for value in raw_scenario["sl_values"]),
                top_k=int(raw_scenario["top_k"]),
                preselect=int(raw_scenario["preselect"]),
                top_trades_n=int(raw_scenario["top_trades_n"]),
                expected_clickhouse_hot_path_calls=int(
                    raw_scenario["expected_clickhouse_hot_path_calls"]
                ),
                expected_v2_clickhouse_hot_path_calls=int(
                    raw_scenario["expected_v2_clickhouse_hot_path_calls"]
                ),
                expected_v2_indicator_compute_calls=int(
                    raw_scenario["expected_v2_indicator_compute_calls"]
                ),
                expected_hot_path_cost_reduction_min=int(
                    raw_scenario["expected_hot_path_cost_reduction_min"]
                ),
            )
        )
    return tuple(scenarios)


def _load_runtime_acceleration_benchmark_corpus():
    """
    Load the committed A3+D2+D3+F2 benchmark corpus for exact and hybrid rollout
    perf-smoke coverage.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - tests/perf_smoke/contexts/backtest/fixtures/
        backtest_runtime_acceleration_benchmark_corpus_v1.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py
    Args:
        None.
    Returns:
        BacktestRuntimeAccelerationBenchmarkCorpusV2: Parsed benchmark corpus contract.
    Assumptions:
        The corpus is lightweight enough to load in every benchmark/perf-smoke test that needs
        deterministic slice metadata.
    Raises:
        ValueError: If the committed benchmark corpus violates its typed contract.
    Side Effects:
        Reads one committed JSON fixture from repository.
    """
    return load_backtest_runtime_acceleration_benchmark_corpus_v2(
        path=_BENCHMARK_CORPUS_FIXTURE_PATH
    )


def _collect_legacy_scenario_measurement(
    *,
    scenario: _R0BenchmarkScenario,
) -> _R0ScenarioMeasurement:
    """
    Execute one deterministic legacy R0 baseline scenario through staged v1 runtime only.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    Args:
        scenario: Benchmark scenario fixture.
    Returns:
        _R0ScenarioMeasurement: Canonical measurement payload for one scenario.
    Assumptions:
        R10-01 removed legacy sync launch from `RunBacktestUseCase`, so the approved R0 baseline
        must now execute the legacy staged runner directly while keeping the same deterministic
        call-count proxy for one live ClickHouse timeline bootstrap.
    Raises:
        None.
    Side Effects:
        Allocates dense request-timeframe candles and executes one local legacy staged run.
    """
    request = _build_request(scenario=scenario)
    template = request.template
    assert template is not None
    candles = _build_dense_request_timeframe_candles(
        timeframe=template.timeframe,
        requested_time_range=request.time_range,
        warmup_bars=scenario.warmup_bars,
    )
    indicator_compute = _R0BaselineIndicatorCompute()
    scorer = CloseFillBacktestStagedScorerV1(
        indicator_compute=indicator_compute,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={
            "init_cash_quote": 1000.0,
            "fee_pct": 0.0,
            "slippage_pct": 0.0,
        },
        market_id=1,
        target_slice=slice(scenario.warmup_bars, candles.close.shape[0]),
        init_cash_quote_default=1000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.0,
    )
    runner = BacktestStagedRunnerV1()

    tracemalloc.start()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    response = runner.run(
        template=template,
        candles=candles,
        preselect=scenario.preselect,
        top_k=scenario.top_k,
        indicator_compute=indicator_compute,
        scorer=scorer,
        requested_time_range=request.time_range,
        top_trades_n=scenario.top_trades_n,
        max_variants_per_compute=600_000,
        max_compute_bytes_total=5 * 1024**3,
    )
    wall_clock_seconds = time.perf_counter() - started_wall
    cpu_time_seconds = time.process_time() - started_cpu
    _, peak_traced_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "scenario_id": scenario.scenario_id,
        "execution_class": scenario.execution_class,
        "timeframe": scenario.timeframe,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "cpu_time_seconds": round(cpu_time_seconds, 6),
        "peak_traced_memory_bytes": int(peak_traced_memory_bytes),
        "clickhouse_hot_path_calls": scenario.expected_clickhouse_hot_path_calls,
        "indicator_compute_calls": indicator_compute.compute_calls,
        "variants_returned": len(response.variants),
    }


def _collect_artifact_v2_scenario_measurement(
    *,
    scenario: _R0BenchmarkScenario,
) -> _R10ArtifactV2ScenarioMeasurement:
    """
    Execute one deterministic artifact-backed v2 scenario and collect closure perf metrics.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    Args:
        scenario: Benchmark scenario fixture.
    Returns:
        _R10ArtifactV2ScenarioMeasurement: Canonical measurement payload for one artifact-backed
            runtime scenario.
    Assumptions:
        Comparison scenarios reuse the same request cardinality as R0 baseline but execute through
        the real artifact-backed runtime with strict local artifacts and fail-fast zero-call
        guards for ClickHouse and `IndicatorCompute.compute(...)`.
    Raises:
        None.
    Side Effects:
        Creates a temporary strict artifact tree and executes one local artifact-backed v2 run.
    """
    request = _build_request(scenario=scenario)
    indicator_compute = _EstimateOnlyArtifactIndicatorCompute()

    with TemporaryDirectory() as tmpdir:
        artifact_loader = _write_artifact_benchmark_store_v2(
            tmp_path=Path(tmpdir),
            request=request,
        )
        use_case = RunBacktestUseCase(
            candle_feed=None,
            indicator_compute=indicator_compute,
            strategy_reader=_NullStrategyReader(),  # type: ignore[arg-type]
            artifact_slot_resolver=ArtifactSlotResolverV2(artifact_loader=artifact_loader),
            runtime_planner=_exact_baseline_runtime_planner_v2(),
            warmup_bars_default=scenario.warmup_bars,
            top_k_default=scenario.top_k,
            preselect_default=scenario.preselect,
            eager_top_reports_enabled=False,
        )

        tracemalloc.start()
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        response = use_case.execute(
            request=request,
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )
        wall_clock_seconds = time.perf_counter() - started_wall
        cpu_time_seconds = time.process_time() - started_cpu
        _, peak_traced_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "scenario_id": scenario.scenario_id,
        "execution_class": scenario.execution_class,
        "timeframe": scenario.timeframe,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "cpu_time_seconds": round(cpu_time_seconds, 6),
        "peak_traced_memory_bytes": int(peak_traced_memory_bytes),
        "clickhouse_hot_path_calls": 0,
        "indicator_compute_calls": indicator_compute.compute_calls,
        "indicator_estimate_calls": indicator_compute.estimate_calls,
        "variants_returned": len(response.variants),
    }


def _exact_baseline_runtime_planner_v2() -> BacktestArtifactRuntimePlannerV2:
    """
    Build runtime planner whose default execution profile matches the exact-baseline corpus.

    Docs:
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        None.
    Returns:
        BacktestArtifactRuntimePlannerV2: Planner honoring `exact_baseline` execution profile.
    Assumptions:
        R0 artifact-backed perf smoke reuses the `exact_baseline` corpus anchor, which is bound
        to `exact_parallel` without changing production request classification or runtime defaults.
    Raises:
        ValueError: If `exact_parallel` is not present in the default execution-profile catalog.
    Side Effects:
        None.
    """
    default_catalog = default_execution_profiles_catalog_v2()
    execution_profiles = ExecutionProfilesCatalogV2(
        default_mode="exact_parallel",
        available_profiles=default_catalog.available_profiles,
    )
    return BacktestArtifactRuntimePlannerV2(execution_profiles=execution_profiles)


def _build_perf_comparison(
    *,
    scenario_id: str,
    baseline: _R0ScenarioMeasurement,
    artifact_v2: _R10ArtifactV2ScenarioMeasurement,
) -> _R10PerfComparison:
    """
    Build a deterministic comparison payload between legacy R0 baseline and v2 runtime metrics.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    Args:
        scenario_id: Stable benchmark scenario identifier.
        baseline: Legacy baseline measurement payload.
        artifact_v2: Artifact-backed v2 measurement payload.
    Returns:
        _R10PerfComparison: Deterministic comparison payload with external-call cost deltas.
    Assumptions:
        R10-03 closure treats hot-path external-call elimination as the canonical speedup signal
        because it is deterministic and directly aligned with the approved bottlenecks.
    Raises:
        None.
    Side Effects:
        None.
    """
    baseline_hot_path_external_calls = (
        baseline["clickhouse_hot_path_calls"] + baseline["indicator_compute_calls"]
    )
    artifact_v2_hot_path_external_calls = (
        artifact_v2["clickhouse_hot_path_calls"] + artifact_v2["indicator_compute_calls"]
    )
    return {
        "scenario_id": scenario_id,
        "baseline_hot_path_external_calls": baseline_hot_path_external_calls,
        "artifact_v2_hot_path_external_calls": artifact_v2_hot_path_external_calls,
        "hot_path_cost_reduction": (
            baseline_hot_path_external_calls - artifact_v2_hot_path_external_calls
        ),
        "baseline_wall_clock_seconds": baseline["wall_clock_seconds"],
        "artifact_v2_wall_clock_seconds": artifact_v2["wall_clock_seconds"],
        "baseline_cpu_time_seconds": baseline["cpu_time_seconds"],
        "artifact_v2_cpu_time_seconds": artifact_v2["cpu_time_seconds"],
    }


def _build_request(*, scenario: _R0BenchmarkScenario) -> RunBacktestRequest:
    """
    Build deterministic template-mode request fixture for one R0 benchmark scenario.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    Args:
        scenario: Benchmark scenario fixture.
    Returns:
        RunBacktestRequest: Deterministic template-mode request.
    Assumptions:
        All scenarios use one indicator grid and explicit SL/TP axes.
    Raises:
        None.
    Side Effects:
        None.
    """
    timeframe = Timeframe(scenario.timeframe)
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timeframe.duration() * scenario.target_bars
    template = RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=timeframe,
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("momentum.roc"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(
                        name="window",
                        values=scenario.indicator_windows,
                    ),
                },
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=True,
            sl=ExplicitValuesSpec(name="sl", values=scenario.sl_values),
            tp=ExplicitValuesSpec(name="tp", values=scenario.tp_values),
        ),
        execution_params={
            "init_cash_quote": 1000.0,
            "fee_pct": 0.0,
            "slippage_pct": 0.0,
        },
    )
    return RunBacktestRequest(
        time_range=TimeRange(start=UtcTimestamp(start), end=UtcTimestamp(end)),
        template=template,
        warmup_bars=scenario.warmup_bars,
        top_k=scenario.top_k,
        preselect=scenario.preselect,
    )


def _build_dense_request_timeframe_candles(
    *,
    timeframe: Timeframe,
    requested_time_range: TimeRange,
    warmup_bars: int,
) -> CandleArrays:
    """
    Build deterministic dense request-timeframe candles for legacy staged baseline execution.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    Args:
        timeframe: Request timeframe used by the legacy staged runner.
        requested_time_range: Requested target range without warmup extension.
        warmup_bars: Warmup bars count to prepend.
    Returns:
        CandleArrays: Warmup-inclusive dense candles in request-timeframe granularity.
    Assumptions:
        Legacy R0 baseline proxies one live timeline bootstrap and then executes entirely on the
        in-memory request-timeframe candles produced here.
    Raises:
        ValueError: If warmup bars are non-positive or target duration is not an exact multiple of
            the request timeframe.
    Side Effects:
        Allocates contiguous NumPy arrays for the staged baseline run.
    """
    if warmup_bars <= 0:
        raise ValueError("warmup_bars must be > 0")

    timeframe_duration = timeframe.duration()
    target_duration = requested_time_range.duration()
    if target_duration % timeframe_duration != timedelta(0):
        raise ValueError("requested_time_range must align to the request timeframe")

    target_bars = int(target_duration // timeframe_duration)
    total_bars = target_bars + warmup_bars
    timeline_start = requested_time_range.start.value - (timeframe_duration * warmup_bars)
    timeline_end = timeline_start + (timeframe_duration * total_bars)
    timeframe_ms = int(timeframe_duration // timedelta(milliseconds=1))
    ts_open = (
        np.arange(total_bars, dtype=np.int64) * np.int64(timeframe_ms)
        + np.int64(_to_epoch_millis(timeline_start))
    )
    values = np.linspace(100.0, 140.0, total_bars, dtype=np.float32)
    wave = np.sin(np.linspace(0.0, 20.0, total_bars, dtype=np.float32)) * np.float32(1.5)
    close = np.ascontiguousarray(values + wave, dtype=np.float32)
    open_values = np.ascontiguousarray(close - np.float32(0.2), dtype=np.float32)
    high_values = np.ascontiguousarray(close + np.float32(0.8), dtype=np.float32)
    low_values = np.ascontiguousarray(close - np.float32(0.8), dtype=np.float32)
    volume = np.ascontiguousarray(np.linspace(100.0, 500.0, total_bars, dtype=np.float32))

    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(timeline_start),
            end=UtcTimestamp(timeline_end),
        ),
        timeframe=timeframe,
        ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
        open=open_values,
        high=high_values,
        low=low_values,
        close=close,
        volume=volume,
    )


def _write_artifact_benchmark_store_v2(
    *,
    tmp_path: Path,
    request: RunBacktestRequest,
) -> YamlBacktestArtifactLoaderV2:
    """
    Materialize one strict synthetic artifact store that matches the benchmark request payload.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    Args:
        tmp_path: Temporary root used for the synthetic artifact tree.
        request: Benchmark request whose timeframe, grid cardinality, and risk levels define the
            artifact contents.
    Returns:
        YamlBacktestArtifactLoaderV2: Strict artifact loader pointed at the generated store.
    Assumptions:
        Benchmark scenarios use template mode with exactly one indicator grid and explicit
        SL/TP levels. The generated store targets the same `signal_tf + 1m_risk` runtime
        contract used in production hot paths.
    Raises:
        ValueError: If the request is missing template mode or the synthetic grid assumptions are
            violated.
        OSError: If one artifact file cannot be written.
    Side Effects:
        Creates a strict `artifacts/backtest/v2` tree with `current.yaml`, manifests, arrays, and
        synthetic `1m hit-times`.
    """
    if request.template is None:
        raise ValueError("artifact benchmark store requires template mode request")
    if len(request.template.indicator_grids) != 1:
        raise ValueError("artifact benchmark store requires exactly one indicator grid")
    if request.warmup_bars is None or request.warmup_bars <= 0:
        raise ValueError("artifact benchmark store requires positive warmup_bars")
    if request.template.risk_grid is None:
        raise ValueError("artifact benchmark store requires explicit risk_grid")

    template = request.template
    risk_grid = template.risk_grid
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    slot = "slot_a"
    slot_generation = 11
    asof_date = "2026-03-29"
    timeframe = template.timeframe
    timeframe_literal = timeframe.code
    timeframe_duration = timeframe.duration()
    timeframe_minutes = int(timeframe_duration // timedelta(minutes=1))
    target_bars = int(request.time_range.duration() // timeframe_duration)
    total_timeframe_bars = target_bars + int(request.warmup_bars)
    timeline_start = request.time_range.start.value - (
        timeframe_duration * int(request.warmup_bars)
    )
    indicator_grid = template.indicator_grids[0]
    indicator_id = str(indicator_grid.indicator_id)
    window_param = indicator_grid.params["window"]
    assert window_param is not None
    indicator_windows = tuple(int(value) for value in window_param.materialize())
    assert risk_grid is not None
    tp_param = risk_grid.tp
    assert tp_param is not None
    tp_levels_pct = tuple(
        float(value) / 100.0 for value in tp_param.materialize()
    )
    sl_param = risk_grid.sl
    assert sl_param is not None
    sl_levels_pct = tuple(
        float(value) / 100.0 for value in sl_param.materialize()
    )
    if len(indicator_windows) == 0:
        raise ValueError("artifact benchmark store requires non-empty indicator window axis")
    if len(tp_levels_pct) == 0 or len(sl_levels_pct) == 0:
        raise ValueError("artifact benchmark store requires non-empty SL/TP level axes")

    one_minute_bars_total = total_timeframe_bars * timeframe_minutes
    one_minute_open_time = np.array(
        [
            _to_epoch_millis(timeline_start + timedelta(minutes=index))
            for index in range(one_minute_bars_total)
        ],
        dtype=np.int64,
    )
    one_minute_close_time = np.ascontiguousarray(
        one_minute_open_time + np.int64(60_000 - 1),
        dtype=np.int64,
    )
    price_base = np.linspace(100.0, 150.0, one_minute_bars_total, dtype=np.float32)
    price_wave = np.sin(
        np.linspace(0.0, 40.0, one_minute_bars_total, dtype=np.float32)
    ).astype(np.float32)
    one_minute_close = np.ascontiguousarray(price_base + price_wave, dtype=np.float32)
    one_minute_open = np.ascontiguousarray(one_minute_close - np.float32(0.15), dtype=np.float32)
    one_minute_high = np.ascontiguousarray(one_minute_close + np.float32(0.45), dtype=np.float32)
    one_minute_low = np.ascontiguousarray(one_minute_close - np.float32(0.45), dtype=np.float32)
    one_minute_volume = np.ascontiguousarray(
        np.linspace(50.0, 150.0, one_minute_bars_total, dtype=np.float32)
    )
    one_minute_ohlcv = np.ascontiguousarray(
        np.column_stack(
            (
                one_minute_open,
                one_minute_high,
                one_minute_low,
                one_minute_close,
                one_minute_volume,
            )
        ),
        dtype=np.float32,
    )
    timeframe_open_time = np.ascontiguousarray(
        one_minute_open_time.reshape(total_timeframe_bars, timeframe_minutes)[:, 0],
        dtype=np.int64,
    )
    timeframe_close_time = np.ascontiguousarray(
        one_minute_close_time.reshape(total_timeframe_bars, timeframe_minutes)[:, -1],
        dtype=np.int64,
    )
    timeframe_ohlcv = np.ascontiguousarray(
        np.column_stack(
            (
                one_minute_open.reshape(total_timeframe_bars, timeframe_minutes)[:, 0],
                one_minute_high.reshape(total_timeframe_bars, timeframe_minutes).max(axis=1),
                one_minute_low.reshape(total_timeframe_bars, timeframe_minutes).min(axis=1),
                one_minute_close.reshape(total_timeframe_bars, timeframe_minutes)[:, -1],
                one_minute_volume.reshape(total_timeframe_bars, timeframe_minutes).sum(axis=1),
            )
        ),
        dtype=np.float32,
    )
    bar_open_1m_idx = np.ascontiguousarray(
        np.arange(total_timeframe_bars, dtype=np.uint32) * np.uint32(timeframe_minutes),
        dtype=np.uint32,
    )
    bar_close_1m_idx = np.ascontiguousarray(
        bar_open_1m_idx + np.uint32(timeframe_minutes - 1),
        dtype=np.uint32,
    )
    signal_pattern = np.array((1, 0, -1, 0), dtype=np.int8)
    signal_matrix = np.empty((len(indicator_windows), total_timeframe_bars), dtype=np.int8)
    for row_index, _window in enumerate(indicator_windows):
        signal_row = np.resize(signal_pattern, total_timeframe_bars).astype(np.int8)
        shift = row_index % signal_pattern.shape[0]
        if shift > 0:
            signal_row = np.roll(signal_row, shift=shift)
        signal_matrix[row_index] = signal_row
    sentinel_index = one_minute_bars_total
    minute_indexes = np.arange(one_minute_bars_total, dtype=np.uint32)
    long_tp = np.empty((len(tp_levels_pct), one_minute_bars_total), dtype=np.uint32)
    long_sl = np.empty((len(sl_levels_pct), one_minute_bars_total), dtype=np.uint32)
    short_tp = np.empty((len(tp_levels_pct), one_minute_bars_total), dtype=np.uint32)
    short_sl = np.empty((len(sl_levels_pct), one_minute_bars_total), dtype=np.uint32)
    for level_index in range(len(tp_levels_pct)):
        long_tp[level_index] = np.minimum(
            minute_indexes + np.uint32(level_index + 2),
            np.uint32(sentinel_index),
        )
        short_tp[level_index] = np.minimum(
            minute_indexes + np.uint32(level_index + 3),
            np.uint32(sentinel_index),
        )
    for level_index in range(len(sl_levels_pct)):
        long_sl[level_index] = np.minimum(
            minute_indexes + np.uint32(level_index + 3),
            np.uint32(sentinel_index),
        )
        short_sl[level_index] = np.minimum(
            minute_indexes + np.uint32(level_index + 2),
            np.uint32(sentinel_index),
        )

    price_paths_1m = builder.price_paths(coordinates, slot, "1m")
    price_paths_tf = builder.price_paths(coordinates, slot, timeframe_literal)
    mapping_paths = builder.mapping_paths(coordinates, slot, timeframe_literal)
    signal_paths = builder.signal_paths(coordinates, slot, timeframe_literal, indicator_id)
    hit_times_paths = builder.hit_times_paths(coordinates, slot)

    _artifact_write_npy_v2(path=price_paths_1m.open_time, array=one_minute_open_time)
    _artifact_write_npy_v2(path=price_paths_1m.close_time, array=one_minute_close_time)
    _artifact_write_npy_v2(path=price_paths_1m.ohlcv, array=one_minute_ohlcv)
    _artifact_write_npy_v2(path=price_paths_tf.open_time, array=timeframe_open_time)
    _artifact_write_npy_v2(path=price_paths_tf.close_time, array=timeframe_close_time)
    _artifact_write_npy_v2(path=price_paths_tf.ohlcv, array=timeframe_ohlcv)
    _artifact_write_npy_v2(path=mapping_paths.bar_open_1m_idx, array=bar_open_1m_idx)
    _artifact_write_npy_v2(path=mapping_paths.bar_close_1m_idx, array=bar_close_1m_idx)
    _artifact_write_npy_v2(path=signal_paths.signals, array=signal_matrix)
    _artifact_write_npy_v2(
        path=hit_times_paths.tp_values,
        array=np.asarray(tp_levels_pct, dtype=np.float32),
    )
    _artifact_write_npy_v2(
        path=hit_times_paths.sl_values,
        array=np.asarray(sl_levels_pct, dtype=np.float32),
    )
    _artifact_write_npy_v2(path=hit_times_paths.long_tp, array=long_tp)
    _artifact_write_npy_v2(path=hit_times_paths.long_sl, array=long_sl)
    _artifact_write_npy_v2(path=hit_times_paths.short_tp, array=short_tp)
    _artifact_write_npy_v2(path=hit_times_paths.short_sl, array=short_sl)

    provenance_payload = {
        "generator": "backtest-r10-03-perf-smoke",
        "generator_version": "r10-03",
        "generated_at_utc": "2026-03-29T00:00:00Z",
        "config_sha256": "a" * 64,
        "inputs_sha256": "b" * 64,
    }
    signal_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "signal",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "indicator_id": indicator_id,
        "timeframe": timeframe_literal,
        "signals": _artifact_array_metadata_v2(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            path=signal_paths.signals,
            axis_order=("variant", "time"),
        ),
        "rows_count": int(signal_matrix.shape[0]),
        "timeline": _timeline_payload_v2(
            open_time=timeframe_open_time,
            close_time=timeframe_close_time,
        ),
        "signal_value_set": [-1, 0, 1],
        "grid": {
            "variant_key_version": 1,
            "variant_keys_sha256": "d" * 64,
            "signals_v1_params_defaults": {},
        },
        "provenance": provenance_payload,
    }
    _artifact_write_yaml_v2(path=signal_paths.manifest, payload=signal_manifest_payload)

    hit_times_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "hit_times_1m",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "timeframe": "1m",
        "timeline_bar_count": int(one_minute_bars_total),
        "sentinel_index": int(sentinel_index),
        "tp_values": _artifact_array_metadata_v2(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            path=hit_times_paths.tp_values,
            axis_order=("level",),
        ),
        "sl_values": _artifact_array_metadata_v2(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            path=hit_times_paths.sl_values,
            axis_order=("level",),
        ),
        "tables": {
            "long_tp": {
                **_artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=hit_times_paths.long_tp,
                    axis_order=("level", "time"),
                ),
                "monotonicity": "non_decreasing_by_level",
            },
            "long_sl": {
                **_artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=hit_times_paths.long_sl,
                    axis_order=("level", "time"),
                ),
                "monotonicity": "non_decreasing_by_level",
            },
            "short_tp": {
                **_artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=hit_times_paths.short_tp,
                    axis_order=("level", "time"),
                ),
                "monotonicity": "non_decreasing_by_level",
            },
            "short_sl": {
                **_artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=hit_times_paths.short_sl,
                    axis_order=("level", "time"),
                ),
                "monotonicity": "non_decreasing_by_level",
            },
        },
        "provenance": provenance_payload,
    }
    _artifact_write_yaml_v2(path=hit_times_paths.manifest, payload=hit_times_manifest_payload)

    slot_manifest_path = builder.slot_manifest_path(coordinates, slot)
    root_manifest_payload = {
        "schema_version": 1,
        "manifest_kind": "slot_root",
        "slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "identity": {
            "exchange": coordinates.exchange,
            "market_type": coordinates.market_type,
            "symbol": coordinates.symbol,
        },
        "prices": [
            {
                "timeframe": "1m",
                "open_time": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_1m.open_time,
                    axis_order=("time",),
                ),
                "close_time": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_1m.close_time,
                    axis_order=("time",),
                ),
                "ohlcv": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_1m.ohlcv,
                    axis_order=("time", "field"),
                ),
                "coverage": _timeline_payload_v2(
                    open_time=one_minute_open_time,
                    close_time=one_minute_close_time,
                ),
            },
            {
                "timeframe": timeframe_literal,
                "open_time": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_tf.open_time,
                    axis_order=("time",),
                ),
                "close_time": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_tf.close_time,
                    axis_order=("time",),
                ),
                "ohlcv": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=price_paths_tf.ohlcv,
                    axis_order=("time", "field"),
                ),
                "coverage": _timeline_payload_v2(
                    open_time=timeframe_open_time,
                    close_time=timeframe_close_time,
                ),
            },
        ],
        "mappings": [
            {
                "timeframe": timeframe_literal,
                "bar_open_1m_idx": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=mapping_paths.bar_open_1m_idx,
                    axis_order=("time",),
                ),
                "bar_close_1m_idx": _artifact_array_metadata_v2(
                    builder=builder,
                    coordinates=coordinates,
                    slot=slot,
                    path=mapping_paths.bar_close_1m_idx,
                    axis_order=("time",),
                ),
            }
        ],
        "signals": {
            "supported_timeframes": [timeframe_literal],
            "supported_indicator_ids": [indicator_id],
            "manifests": [
                {
                    "timeframe": timeframe_literal,
                    "indicator_id": indicator_id,
                    "manifest_path": _relative_slot_path_v2(
                        builder=builder,
                        coordinates=coordinates,
                        slot=slot,
                        path=signal_paths.manifest,
                    ),
                    "manifest_sha256": _file_sha256_hex_v2(signal_paths.manifest),
                }
            ],
        },
        "hit_times": {
            "timeframe": "1m",
            "manifest_path": _relative_slot_path_v2(
                builder=builder,
                coordinates=coordinates,
                slot=slot,
                path=hit_times_paths.manifest,
            ),
            "manifest_sha256": _file_sha256_hex_v2(hit_times_paths.manifest),
        },
        "signal_encoding": {
            "dtype": "int8",
            "axis_order": ["variant", "time"],
            "value_set": [-1, 0, 1],
        },
        "provenance": provenance_payload,
    }
    _artifact_write_yaml_v2(path=slot_manifest_path, payload=root_manifest_payload)

    current_pointer_payload = {
        "schema_version": 1,
        "active_slot": slot,
        "slot_generation": slot_generation,
        "asof_date": asof_date,
        "manifest_sha256": _file_sha256_hex_v2(slot_manifest_path),
        "published_at_utc": "2026-03-29T00:00:00Z",
    }
    _artifact_write_yaml_v2(
        path=builder.current_pointer_path(coordinates),
        payload=current_pointer_payload,
    )

    return loader


def _artifact_array_metadata_v2(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    path: Path,
    axis_order: tuple[str, ...],
) -> dict[str, object]:
    """
    Build strict array metadata payload for one synthetic artifact file.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    Args:
        builder: Deterministic artifact path builder.
        coordinates: Synthetic benchmark artifact coordinates.
        slot: Slot literal for the generated benchmark store.
        path: Absolute `.npy` path.
        axis_order: Canonical axis-order literal tuple for the artifact family.
    Returns:
        dict[str, object]: Strict array metadata payload mirroring runtime manifest contracts.
    Assumptions:
        Artifact arrays are already written to disk before metadata is generated.
    Raises:
        FileNotFoundError: If the target array file does not exist.
    Side Effects:
        Opens the `.npy` file through NumPy mmap to derive dtype and shape.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": _relative_slot_path_v2(
            builder=builder,
            coordinates=coordinates,
            slot=slot,
            path=path,
        ),
        "dtype": array.dtype.name,
        "shape": [int(value) for value in array.shape],
        "axis_order": [str(value) for value in axis_order],
        "sha256": _file_sha256_hex_v2(path),
    }


def _artifact_write_yaml_v2(*, path: Path, payload: dict[str, object]) -> None:
    """
    Write one YAML payload with deterministic field ordering for synthetic benchmark artifacts.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    Args:
        path: Target YAML path.
        payload: YAML payload mapping to serialize.
    Returns:
        None.
    Assumptions:
        Input mapping already uses the canonical key order expected by strict manifest tests.
    Raises:
        OSError: If the YAML file cannot be written.
    Side Effects:
        Creates parent directories and writes UTF-8 YAML content to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _artifact_write_npy_v2(*, path: Path, array: np.ndarray) -> None:
    """
    Write one `.npy` file for the synthetic artifact-backed benchmark store.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    Args:
        path: Target `.npy` file path.
        array: NumPy array payload to serialize.
    Returns:
        None.
    Assumptions:
        Arrays are already shaped and typed according to strict runtime contracts.
    Raises:
        OSError: If the `.npy` file cannot be written.
    Side Effects:
        Creates parent directories and writes binary `.npy` content to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        np.save(file_handle, array, allow_pickle=False)


def _timeline_payload_v2(*, open_time: np.ndarray, close_time: np.ndarray) -> dict[str, int]:
    """
    Build strict timeline coverage payload from paired open and close timestamp arrays.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    Args:
        open_time: Open-time timestamp array.
        close_time: Close-time timestamp array.
    Returns:
        dict[str, int]: Strict timeline coverage payload.
    Assumptions:
        Arrays are non-empty, aligned by row count, and already monotone.
    Raises:
        IndexError: If the arrays are empty.
    Side Effects:
        None.
    """
    return {
        "bar_count": int(open_time.shape[0]),
        "open_time_start": int(open_time[0]),
        "open_time_end": int(open_time[-1]),
        "close_time_start": int(close_time[0]),
        "close_time_end": int(close_time[-1]),
    }


def _relative_slot_path_v2(
    *,
    builder: BacktestArtifactPathBuilderV2,
    coordinates: ArtifactCoordinatesV2,
    slot: str,
    path: Path,
) -> str:
    """
    Convert one absolute synthetic artifact path into the canonical slot-relative path literal.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    Args:
        builder: Deterministic artifact path builder.
        coordinates: Synthetic benchmark artifact coordinates.
        slot: Slot literal for the generated benchmark store.
        path: Absolute artifact path inside the slot root.
    Returns:
        str: POSIX-style slot-relative path.
    Assumptions:
        The provided path always lives under the slot root for the generated benchmark store.
    Raises:
        ValueError: If the path is outside the slot root.
    Side Effects:
        None.
    """
    return path.relative_to(builder.slot_root(coordinates, slot)).as_posix()


def _file_sha256_hex_v2(path: Path) -> str:
    """
    Compute a lowercase SHA-256 hex digest for one synthetic benchmark artifact file.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    Args:
        path: Existing file path to hash.
    Returns:
        str: Lowercase SHA-256 hex digest.
    Assumptions:
        Files are small benchmark artifacts stored on local disk.
    Raises:
        OSError: If the file cannot be read.
    Side Effects:
        Reads the file from disk in binary mode.
    """
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_dense_1m_from_time_range(*, time_range: TimeRange) -> CandleArrays:
    """
    Build deterministic dense `1m` candles for supplied aligned range.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/indicators/application/dto/candles.py
    Args:
        time_range: Requested aligned time range.
    Returns:
        CandleArrays: Dense finite `1m` arrays covering entire range.
    Assumptions:
        Duration is divisible by one minute.
    Raises:
        ValueError: If duration is not divisible by one minute.
    Side Effects:
        Allocates numpy arrays.
    """
    duration = time_range.duration()
    if duration % _ONE_MINUTE != timedelta(0):
        raise ValueError("time_range duration must be divisible by one minute")

    count = int(duration // _ONE_MINUTE)
    start_ms = _to_epoch_millis(time_range.start.value)
    ts_open = np.arange(count, dtype=np.int64) * np.int64(60_000) + np.int64(start_ms)
    values = np.linspace(100.0, 140.0, count, dtype=np.float32)
    wave = np.sin(np.linspace(0.0, 20.0, count, dtype=np.float32)) * np.float32(1.5)
    close = np.ascontiguousarray(values + wave, dtype=np.float32)
    open_values = np.ascontiguousarray(close - np.float32(0.2), dtype=np.float32)
    high_values = np.ascontiguousarray(close + np.float32(0.8), dtype=np.float32)
    low_values = np.ascontiguousarray(close - np.float32(0.8), dtype=np.float32)
    volume = np.ascontiguousarray(np.linspace(100.0, 500.0, count, dtype=np.float32))
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=time_range,
        timeframe=Timeframe("1m"),
        ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
        open=open_values,
        high=high_values,
        low=low_values,
        close=close,
        volume=volume,
    )


def _to_epoch_millis(dt: datetime) -> int:
    """
    Convert timezone-aware datetime to epoch milliseconds.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/indicators/application/dto/candles.py
    Args:
        dt: Timezone-aware datetime.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Input datetime uses timezone information.
    Raises:
        ValueError: If datetime is naive.
    Side Effects:
        None.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = dt.astimezone(timezone.utc) - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))


def _axis_def(name: str, values: tuple[int | float | str, ...]) -> AxisDef:
    """
    Build `AxisDef` with deterministic value-family inference from materialized values.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/indicators/domain/entities/axis_def.py
      - src/trading/contexts/indicators/application/dto/estimate.py
    Args:
        name: Axis name.
        values: Materialized scalar values.
    Returns:
        AxisDef: Deterministic axis definition.
    Assumptions:
        Axis values are homogeneous and non-empty.
    Raises:
        ValueError: If values are empty or unsupported scalar type is encountered.
    Side Effects:
        None.
    """
    if len(values) == 0:
        raise ValueError("axis values must be non-empty")

    first = values[0]
    if isinstance(first, str):
        return AxisDef(name=name, values_enum=tuple(str(value) for value in values))
    if isinstance(first, int):
        return AxisDef(name=name, values_int=tuple(int(value) for value in values))
    if isinstance(first, float):
        return AxisDef(name=name, values_float=tuple(float(value) for value in values))
    raise ValueError(f"unsupported axis value type: {type(first).__name__}")
