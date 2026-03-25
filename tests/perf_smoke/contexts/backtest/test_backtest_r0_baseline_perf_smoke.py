from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestRiskGridSpec,
    RunBacktestRequest,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import CurrentUser
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
    scenarios = _load_benchmark_scenarios()
    measurements = [_collect_scenario_measurement(scenario=scenario) for scenario in scenarios]

    assert [scenario.scenario_id for scenario in scenarios] == [
        "sync-small-run",
        "large-run",
        "background-run",
    ]
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


def test_r0_parity_scope_fixture_manifest_is_complete() -> None:
    """
    Verify parity-scope fixture manifest covers Stage A, legacy Stage B, and future v2 reference.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Args:
        None.
    Returns:
        None.
    Assumptions:
        R0 parity scope is document-driven and intentionally separates active
        vs reference-only cases.
    Raises:
        AssertionError: If one required scope entry is missing or misclassified.
    Side Effects:
        None.
    """
    payload = json.loads((_FIXTURES_DIR / "r0_parity_scope.json").read_text(encoding="utf-8"))
    scopes = payload["parity_scopes"]
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
            )
        )
    return tuple(scenarios)


def _collect_scenario_measurement(
    *,
    scenario: _R0BenchmarkScenario,
) -> _R0ScenarioMeasurement:
    """
    Execute one deterministic local baseline scenario and collect R0 benchmark metrics.

    Docs:
      - docs/architecture/backtest/backtest-v2-benchmarks.md
      - tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    Args:
        scenario: Benchmark scenario fixture.
    Returns:
        _R0ScenarioMeasurement: Canonical measurement payload for one scenario.
    Assumptions:
        Local candle-feed read count proxies the current v1 ClickHouse hot-path read.
    Raises:
        None.
    Side Effects:
        Allocates in-memory benchmark inputs and executes one local backtest run.
    """
    candle_feed = _CountingCandleFeed()
    indicator_compute = _R0BaselineIndicatorCompute()
    use_case = RunBacktestUseCase(
        candle_feed=candle_feed,
        indicator_compute=indicator_compute,
        strategy_reader=_NullStrategyReader(),  # type: ignore[arg-type]
        warmup_bars_default=scenario.warmup_bars,
        top_k_default=scenario.top_k,
        preselect_default=scenario.preselect,
        top_trades_n_default=scenario.top_trades_n,
        eager_top_reports_enabled=False,
    )
    request = _build_request(scenario=scenario)

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
        "clickhouse_hot_path_calls": candle_feed.load_calls,
        "indicator_compute_calls": indicator_compute.compute_calls,
        "variants_returned": len(response.variants),
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
        top_trades_n=scenario.top_trades_n,
    )


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
