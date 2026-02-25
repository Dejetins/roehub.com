from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, cast

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.services import (
    TOTAL_RETURN_METRIC_LITERAL,
    BacktestGridBuilderV1,
    BacktestRunCancelledV1,
    BacktestRunControlV1,
    BacktestStagedRunnerV1,
    CloseFillBacktestStagedScorerV1,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    IndicatorVariantSelection,
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

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


class _EstimateOnlyIndicatorCompute:
    """
    IndicatorCompute fake exposing deterministic estimate axes from request grid specs.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Materialize estimate result for staged runner tests.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Deterministic axis/variants estimate.
        Assumptions:
            Fixtures use explicit values specs only.
        Raises:
            ValueError: If variants exceed guard.
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
        Return placeholder object because compute path is unused in these tests.

        Args:
            req: Compute request payload.
        Returns:
            object: Placeholder object.
        Assumptions:
            Staged runner tests use estimate-only flow.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = req
        return cast(IndicatorTensor, object())

    def warmup(self) -> None:
        """
        Provide no-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is out of scope for staged runner tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _InstrumentedSignalIndicatorCompute:
    """
    IndicatorCompute fake with deterministic signal outputs and in-memory compute call counter.

    Docs:
      - docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic in-memory compute instrumentation state.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Compute calls are counted across both Stage A and Stage B invocations.
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
            Grid axes are explicit and materializable in-memory.
        Raises:
            ValueError: If materialized variants exceed provided guard.
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
        Return deterministic multi-variant tensor with neutral bars between sign flips.

        Args:
            req: Compute request payload with explicit single-value axes.
        Returns:
            IndicatorTensor: Deterministic time-major tensor for scorer signal evaluation.
        Assumptions:
            Prepared scorer path may request full indicator axis grid in one compute call.
        Raises:
            ValueError: If bars count is non-positive.
        Side Effects:
            Increments `compute_calls` instrumentation counter.
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


class _ConstantTieScorer:
    """
    Staged scorer fake returning equal `Total Return [%]` values for tie-break tests.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, float | int | str | bool | None]],
        risk_params: Mapping[str, float | int | str | bool | None],
        indicator_variant_key: str,
        variant_key: str,
    ) -> dict[str, float]:
        """
        Return constant ranking metric to exercise deterministic tie-break logic.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Deterministic indicator key.
            variant_key: Deterministic backtest key.
        Returns:
            dict[str, float]: Constant metric payload.
        Assumptions:
            Stable sort tie-break keys decide deterministic order when metrics are equal.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (
            stage,
            candles,
            indicator_selections,
            signal_params,
            risk_params,
            indicator_variant_key,
            variant_key,
        )
        return {TOTAL_RETURN_METRIC_LITERAL: 1.0}


class _WindowScorer:
    """
    Staged scorer fake ranking variants by `window` parameter value.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, float | int | str | bool | None]],
        risk_params: Mapping[str, float | int | str | bool | None],
        indicator_variant_key: str,
        variant_key: str,
    ) -> dict[str, float]:
        """
        Return deterministic ranking metric equal to selected `window` value.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Deterministic indicator key.
            variant_key: Deterministic backtest key.
        Returns:
            dict[str, float]: Ranking payload keyed by `Total Return [%]`.
        Assumptions:
            Fixture has one indicator with integer `window` parameter.
        Raises:
            KeyError: If `window` parameter is missing.
        Side Effects:
            None.
        """
        _ = stage, candles, signal_params, risk_params, indicator_variant_key, variant_key
        window = int(indicator_selections[0].params["window"])
        return {TOTAL_RETURN_METRIC_LITERAL: float(window)}


def test_staged_runner_v1_applies_deterministic_tie_break_keys() -> None:
    """
    Verify tie-break ordering is deterministic for equal Stage A/Stage B metric values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage A tie-break key is `base_variant_key`, Stage B tie-break key is `variant_key`.
    Raises:
        AssertionError: If resulting top-k order is non-deterministic.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1()
    result = runner.run(
        template=_template_for_tie_breaks(),
        candles=_build_candles(bars=60),
        preselect=3,
        top_k=4,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_ConstantTieScorer(),
    )

    variant_keys = tuple(item.variant_key for item in result.variants)
    assert len(result.variants) == 4
    assert variant_keys == tuple(sorted(variant_keys))
    assert result.stage_a_variants_total == 4
    assert result.stage_b_variants_total == 6


def test_staged_runner_v1_applies_preselect_and_top_k_limits() -> None:
    """
    Verify staged runner keeps only Stage A shortlist and returns only Stage B top-k rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ranking metric equals indicator `window` value.
    Raises:
        AssertionError: If shortlist/top-k constraints are not respected.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1()
    result = runner.run(
        template=_template_for_top_k(),
        candles=_build_candles(bars=60),
        preselect=2,
        top_k=1,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_WindowScorer(),
    )

    assert len(result.variants) == 1
    assert result.variants[0].total_return_pct == 4.0
    assert result.stage_a_variants_total == 4
    assert result.stage_b_variants_total == 2


def test_staged_runner_v1_is_deterministic_with_parallel_scoring() -> None:
    """
    Verify CPU-parallel scoring path keeps deterministic variant ordering across runs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Final ordering is always stable by ranking key and tie-break keys.
    Raises:
        AssertionError: If repeated parallel runs produce different ordered variant keys.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1(parallel_workers=4)
    first = runner.run(
        template=_template_for_tie_breaks(),
        candles=_build_candles(bars=60),
        preselect=3,
        top_k=4,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_ConstantTieScorer(),
    )
    second = runner.run(
        template=_template_for_tie_breaks(),
        candles=_build_candles(bars=60),
        preselect=3,
        top_k=4,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_ConstantTieScorer(),
    )

    assert tuple(item.variant_key for item in first.variants) == tuple(
        item.variant_key for item in second.variants
    )


def test_staged_runner_v1_stage_b_risk_expansion_reuses_signal_cache() -> None:
    """
    Verify Stage B risk expansion reuses Stage A signals cache without extra compute calls.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Signal cache identity is based on indicator variant key and signal params only.
    Raises:
        AssertionError: If Stage B expansion triggers additional indicator compute calls.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1()
    candles = _build_candles(bars=64)
    indicator_compute = _InstrumentedSignalIndicatorCompute()
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
        target_slice=slice(8, 64),
        init_cash_quote_default=1000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.0,
    )

    result = runner.run(
        template=_template_for_signal_cache_reuse(),
        candles=candles,
        preselect=2,
        top_k=4,
        indicator_compute=indicator_compute,
        scorer=scorer,
    )

    assert result.stage_a_variants_total == 2
    assert result.stage_b_variants_total == 6
    assert indicator_compute.compute_calls == 1


class _CancellingScorer:
    """
    Deterministic scorer fake that triggers cooperative cancellation after N calls.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/run_control_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    def __init__(self, *, run_control: BacktestRunControlV1, cancel_after_calls: int) -> None:
        """
        Initialize deterministic scorer cancellation behavior.

        Args:
            run_control: Cooperative run control object.
            cancel_after_calls: Calls threshold after which scorer triggers cancellation.
        Returns:
            None.
        Assumptions:
            Call counter increments once per `score_variant` invocation.
        Raises:
            ValueError: If threshold is non-positive.
        Side Effects:
            None.
        """
        if cancel_after_calls <= 0:
            raise ValueError("cancel_after_calls must be > 0")
        self._run_control = run_control
        self._cancel_after_calls = cancel_after_calls
        self.calls = 0

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, float | int | str | bool | None]],
        risk_params: Mapping[str, float | int | str | bool | None],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic metric and trigger cooperative cancellation after threshold.

        Args:
            stage: Stage literal.
            candles: Dense candles payload.
            indicator_selections: Indicator selections payload.
            signal_params: Signal parameters payload.
            risk_params: Risk payload.
            indicator_variant_key: Indicators-only key.
            variant_key: Full variant key.
        Returns:
            Mapping[str, float]: Deterministic ranking payload.
        Assumptions:
            Cancellation request is checked by staged core before next loop iteration.
        Raises:
            None.
        Side Effects:
            Triggers cooperative cancellation on run control when threshold is reached.
        """
        _ = (
            stage,
            candles,
            indicator_selections,
            signal_params,
            risk_params,
            indicator_variant_key,
            variant_key,
        )
        self.calls += 1
        if self.calls >= self._cancel_after_calls:
            self._run_control.cancel(reason="unit_test_cancelled")
        return {TOTAL_RETURN_METRIC_LITERAL: 1.0}


def test_staged_runner_v1_cooperative_cancellation_stops_stage_loop() -> None:
    """
    Verify cooperative cancellation token interrupts Stage-A loop before full completion.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Shared staged core checks cancellation between scored variants.
    Raises:
        AssertionError: If cancellation does not interrupt staged execution.
    Side Effects:
        None.
    """
    run_control = BacktestRunControlV1(deadline_seconds=10.0)
    scorer = _CancellingScorer(run_control=run_control, cancel_after_calls=1)
    runner = BacktestStagedRunnerV1()

    with pytest.raises(BacktestRunCancelledV1):
        runner.run(
            template=_template_for_tie_breaks(),
            candles=_build_candles(bars=40),
            preselect=3,
            top_k=2,
            indicator_compute=_EstimateOnlyIndicatorCompute(),
            scorer=scorer,
            run_control=run_control,
        )
    assert scorer.calls == 1


def test_staged_runner_v1_ordering_is_independent_from_mapping_insertion_order() -> None:
    """
    Verify deterministic variant ordering is invariant to equivalent mapping insertion order.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variant keys are built from canonical sorted mapping serialization.
    Raises:
        AssertionError: If reordered template mappings change output ordering.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1()
    candles = _build_candles(bars=60)

    first = runner.run(
        template=_template_for_ordering_invariance_variant_a(),
        candles=candles,
        preselect=3,
        top_k=4,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_ConstantTieScorer(),
    )
    second = runner.run(
        template=_template_for_ordering_invariance_variant_b(),
        candles=candles,
        preselect=3,
        top_k=4,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        scorer=_ConstantTieScorer(),
    )

    assert first.stage_a_variants_total == second.stage_a_variants_total
    assert first.stage_b_variants_total == second.stage_b_variants_total
    assert tuple(item.variant_key for item in first.variants) == tuple(
        item.variant_key for item in second.variants
    )
    assert tuple(item.variant_index for item in first.variants) == tuple(
        item.variant_index for item in second.variants
    )


def test_stage_a_heap_shortlist_matches_full_sort_reference() -> None:
    """
    Verify Stage-A streaming heap shortlist is equivalent to full sort+slice reference.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage-A ranking key is `total_return_pct DESC, base_variant_key ASC`.
    Raises:
        AssertionError: If heap shortlist differs from deterministic full-sort reference.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1(parallel_workers=1)
    template = _template_for_tie_breaks()
    candles = _build_candles(bars=60)
    scorer = _ConstantTieScorer()
    preselect = 3
    grid_context = BacktestGridBuilderV1().build(
        template=template,
        candles=candles,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        preselect=preselect,
    )

    all_rows = [
        runner._score_stage_a_variant(
            base_variant=base_variant,
            candles=candles,
            scorer=scorer,
        )
        for base_variant in grid_context.iter_stage_a_variants()
    ]
    expected_rows = sorted(
        all_rows,
        key=lambda row: (-row.total_return_pct, row.base_variant.base_variant_key),
    )[:preselect]

    heap_rows = runner._score_stage_a(
        grid_context=grid_context,
        candles=candles,
        scorer=scorer,
        shortlist_limit=preselect,
    )

    assert tuple(
        (row.base_variant.base_variant_key, row.total_return_pct) for row in heap_rows
    ) == tuple((row.base_variant.base_variant_key, row.total_return_pct) for row in expected_rows)


def test_stage_b_heap_top_k_matches_full_sort_reference() -> None:
    """
    Verify Stage-B streaming heap top-K is equivalent to full sort+slice reference.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage-B ranking key is `total_return_pct DESC, variant_key ASC`.
    Raises:
        AssertionError: If heap top-K differs from deterministic full-sort reference.
    Side Effects:
        None.
    """
    runner = BacktestStagedRunnerV1(parallel_workers=1)
    template = _template_for_tie_breaks()
    candles = _build_candles(bars=60)
    scorer = _ConstantTieScorer()
    preselect = 3
    top_k = 4
    grid_context = BacktestGridBuilderV1().build(
        template=template,
        candles=candles,
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        preselect=preselect,
    )

    shortlist = runner._score_stage_a(
        grid_context=grid_context,
        candles=candles,
        scorer=scorer,
        shortlist_limit=preselect,
    )
    stage_b_tasks = tuple(
        runner._iter_stage_b_tasks(
            template=template,
            grid_context=grid_context,
            shortlist=shortlist,
        )
    )
    expected_rows = sorted(
        (
            runner._score_stage_b_task(task=task, candles=candles, scorer=scorer)
            for task in stage_b_tasks
        ),
        key=lambda row: (-row.total_return_pct, row.variant_key),
    )[:top_k]

    heap_rows, heap_tasks = runner._score_stage_b(
        template=template,
        grid_context=grid_context,
        shortlist=shortlist,
        candles=candles,
        scorer=scorer,
        top_k_limit=top_k,
    )

    assert tuple(
        (row.variant_key, row.total_return_pct, row.variant_index) for row in heap_rows
    ) == tuple((row.variant_key, row.total_return_pct, row.variant_index) for row in expected_rows)
    assert set(heap_tasks.keys()) == {row.variant_key for row in heap_rows}
    for row in heap_rows:
        assert heap_tasks[row.variant_key].variant_key == row.variant_key


def _template_for_tie_breaks() -> RunBacktestTemplate:
    """
    Build template fixture with Stage B risk expansion for tie-break ordering checks.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Deterministic staged-run template fixture.
    Assumptions:
        Risk axis SL has two values while TP is disabled.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close", "high")),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(10, 20)),
                },
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=False,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0)),
        ),
    )


def _template_for_top_k() -> RunBacktestTemplate:
    """
    Build template fixture with four Stage A variants and disabled risk expansion axes.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Deterministic staged-run template fixture.
    Assumptions:
        Stage B variants equal Stage A shortlist because risk expansion is disabled.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(1, 2, 3, 4)),
                },
            ),
        ),
    )


def _template_for_signal_cache_reuse() -> RunBacktestTemplate:
    """
    Build Stage B template where multiple risk variants share each Stage A base signal payload.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Deterministic template for signal-cache reuse invariant test.
    Assumptions:
        Stage A has two base indicator variants and Stage B expands each to three risk variants.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("momentum.roc"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(5, 10)),
                },
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=False,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0, 3.0)),
        ),
        execution_params={
            "init_cash_quote": 1000.0,
            "fee_pct": 0.0,
            "slippage_pct": 0.0,
        },
    )


def _template_for_ordering_invariance_variant_a() -> RunBacktestTemplate:
    """
    Build deterministic template with one concrete insertion order for nested mappings.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Baseline template for ordering-invariance comparison.
    Assumptions:
        Mapping content is semantically equivalent to variant B template.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(10, 20)),
                },
            ),
        ),
        signal_grids={
            "ma.sma": {
                "long_threshold": ExplicitValuesSpec(name="long_threshold", values=(0.0, 1.0)),
                "short_threshold": ExplicitValuesSpec(name="short_threshold", values=(0.0, -1.0)),
            }
        },
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=False,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0)),
        ),
        execution_params={
            "fee_pct": 0.0,
            "init_cash_quote": 1000.0,
            "slippage_pct": 0.0,
        },
    )


def _template_for_ordering_invariance_variant_b() -> RunBacktestTemplate:
    """
    Build semantically equivalent template with reversed insertion order for mappings.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Alternate insertion-order template for invariance checks.
    Assumptions:
        Canonical key normalization removes insertion-order effects.
    Raises:
        ValueError: If fixture payload violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close",)),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(10, 20)),
                },
            ),
        ),
        signal_grids={
            "ma.sma": {
                "short_threshold": ExplicitValuesSpec(name="short_threshold", values=(0.0, -1.0)),
                "long_threshold": ExplicitValuesSpec(name="long_threshold", values=(0.0, 1.0)),
            }
        },
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=False,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0)),
        ),
        execution_params={
            "slippage_pct": 0.0,
            "init_cash_quote": 1000.0,
            "fee_pct": 0.0,
        },
    )


def _build_candles(*, bars: int) -> CandleArrays:
    """
    Build deterministic dense candle arrays fixture with specified number of bars.

    Args:
        bars: Number of `1m` bars.
    Returns:
        CandleArrays: Dense finite candle arrays.
    Assumptions:
        Bars count is positive.
    Raises:
        ValueError: If bars count is not positive.
    Side Effects:
        Allocates numpy arrays.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0")

    start = datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)
    end = start + (bars * _ONE_MINUTE)
    start_ms = _to_epoch_millis(start)

    ts_open = np.arange(bars, dtype=np.int64) * np.int64(60_000) + np.int64(start_ms)
    values = np.arange(1, bars + 1, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(start=UtcTimestamp(start), end=UtcTimestamp(end)),
        timeframe=Timeframe("1m"),
        ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
        open=np.ascontiguousarray(values, dtype=np.float32),
        high=np.ascontiguousarray(values, dtype=np.float32),
        low=np.ascontiguousarray(values, dtype=np.float32),
        close=np.ascontiguousarray(values, dtype=np.float32),
        volume=np.ascontiguousarray(values, dtype=np.float32),
    )


def _to_epoch_millis(dt: datetime) -> int:
    """
    Convert timezone-aware datetime to epoch milliseconds.

    Args:
        dt: Timezone-aware datetime.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Datetime is timezone-aware.
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
    Build `AxisDef` with value-family inferred from materialized axis values.

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
