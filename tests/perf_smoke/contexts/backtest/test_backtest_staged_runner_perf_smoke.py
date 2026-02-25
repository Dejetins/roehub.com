from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.services import (
    BacktestStagedRunnerV1,
    CloseFillBacktestStagedScorerV1,
)
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
    UtcTimestamp,
)

_CATASTROPHIC_UPPER_BOUND_SECONDS = 60.0


class _PerfSmokeIndicatorCompute:
    """
    In-memory IndicatorCompute fake for staged backtest perf-smoke without external services.

    Docs:
      - docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic in-memory compute instrumentation state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Compute call count is tracked for cache-reuse invariant assertions.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.compute_calls = 0

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Materialize deterministic estimate payload from explicit source/params axes.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Deterministic estimate result for staged grid builder.
        Assumptions:
            Perf-smoke grid uses explicit values only.
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
        Return deterministic multi-variant tensor with neutral bars between sign transitions.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Deterministic time-major tensor used by scorer.
        Assumptions:
            Prepared scorer path may request full indicator axis grid in one compute call.
        Raises:
            ValueError: If bars count is non-positive.
        Side Effects:
            Increments in-memory compute call counter.
        """
        bars = int(req.candles.close.shape[0])
        if bars <= 0:
            raise ValueError("bars must be > 0")

        self.compute_calls += 1
        pattern = np.asarray((1.0, 0.0, -1.0, 0.0), dtype=np.float32)
        window_spec = req.grid.params.get(
            "window",
            ExplicitValuesSpec(name="window", values=(1,)),
        )
        windows = tuple(window_spec.materialize())
        if len(windows) == 0:
            windows = (1,)
        values = np.empty((bars, len(windows)), dtype=np.float32)
        for index, raw_window in enumerate(windows):
            series = np.resize(pattern, bars).astype(np.float32)
            shift = int(int(raw_window) % 4)
            if shift > 0:
                series = np.roll(series, shift=shift)
            values[:, index] = series
        return IndicatorTensor(
            indicator_id=req.grid.indicator_id,
            layout=Layout.TIME_MAJOR,
            axes=(AxisDef(name="variant", values_int=tuple(range(len(windows)))),),
            values=np.ascontiguousarray(values, dtype=np.float32),
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
            Warmup is unnecessary for deterministic in-memory fake.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


def test_backtest_staged_runner_perf_smoke_small_sync_grid() -> None:
    """
    Run guard-safe staged backtest perf-smoke and assert completion with deterministic invariants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Perf-smoke validates viability and shape, not strict latency SLA.
    Raises:
        AssertionError: If staged run output contracts or catastrophic-time guard fail.
    Side Effects:
        None.
    """
    candles = _build_hourly_candles(bars=512)
    indicator_compute = _PerfSmokeIndicatorCompute()
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
        target_slice=slice(32, 512),
        init_cash_quote_default=1000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.0,
    )
    runner = BacktestStagedRunnerV1()

    started = time.perf_counter()
    result = runner.run(
        template=_perf_smoke_template(),
        candles=candles,
        preselect=4,
        top_k=5,
        indicator_compute=indicator_compute,
        scorer=scorer,
        requested_time_range=candles.time_range,
        top_trades_n=2,
        max_variants_per_compute=2_000,
        max_compute_bytes_total=1 * 1024**3,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < _CATASTROPHIC_UPPER_BOUND_SECONDS
    assert result.stage_a_variants_total == 6
    assert result.stage_b_variants_total == 16
    assert len(result.variants) == 5
    assert indicator_compute.compute_calls == 1
    assert len({item.variant_key for item in result.variants}) == len(result.variants)
    assert all(len(item.variant_key) == 64 for item in result.variants)

    for variant in result.variants:
        assert np.isfinite(variant.total_return_pct)
        assert variant.report is not None
        assert variant.report.table_md is not None
        assert variant.report.table_md.startswith("|Metric|Value|")


def _perf_smoke_template() -> RunBacktestTemplate:
    """
    Build deterministic small staged template for backtest perf-smoke.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Guard-safe template with compact Stage A and Stage B cardinalities.
    Assumptions:
        Stage A uses six indicator variants; Stage B expands shortlist by four risk variants.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1h"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("momentum.roc"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(5, 10, 15, 20, 25, 30)),
                },
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=True,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0)),
            tp=ExplicitValuesSpec(name="tp", values=(2.0, 4.0)),
        ),
        execution_params={
            "init_cash_quote": 1000.0,
            "fee_pct": 0.0,
            "slippage_pct": 0.0,
        },
    )


def _build_hourly_candles(*, bars: int) -> CandleArrays:
    """
    Build deterministic dense hourly candles for staged runner perf-smoke.

    Args:
        bars: Number of hourly bars.
    Returns:
        CandleArrays: Dense finite candles payload.
    Assumptions:
        Bars count is positive and arrays remain contiguous float32/int64.
    Raises:
        ValueError: If bars count is non-positive.
    Side Effects:
        Allocates numpy arrays.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0")

    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=bars)
    hour_ms = int(timedelta(hours=1) // timedelta(milliseconds=1))
    ts_open = np.arange(bars, dtype=np.int64) * np.int64(hour_ms)

    base = np.linspace(100.0, 140.0, bars, dtype=np.float32)
    wave = np.sin(np.linspace(0.0, 40.0, bars, dtype=np.float32)) * np.float32(2.5)
    close = np.ascontiguousarray(base + wave, dtype=np.float32)
    open_values = np.ascontiguousarray(close - np.float32(0.2), dtype=np.float32)
    high_values = np.ascontiguousarray(close + np.float32(0.8), dtype=np.float32)
    low_values = np.ascontiguousarray(close - np.float32(0.8), dtype=np.float32)
    volume = np.ascontiguousarray(np.linspace(100.0, 500.0, bars, dtype=np.float32))

    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(start),
            end=UtcTimestamp(end),
        ),
        timeframe=Timeframe("1h"),
        ts_open=ts_open,
        open=open_values,
        high=high_values,
        low=low_values,
        close=close,
        volume=volume,
    )

def _axis_def(name: str, values: tuple[int | float | str, ...]) -> AxisDef:
    """
    Build `AxisDef` with deterministic value-family inference from materialized values.

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
