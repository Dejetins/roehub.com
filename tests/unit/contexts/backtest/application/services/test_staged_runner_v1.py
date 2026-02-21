from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, cast

import numpy as np

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.services import (
    TOTAL_RETURN_METRIC_LITERAL,
    BacktestStagedRunnerV1,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    IndicatorVariantSelection,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId
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
