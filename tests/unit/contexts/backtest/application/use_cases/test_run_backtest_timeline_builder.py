from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, cast
from uuid import UUID

import numpy as np

from trading.contexts.backtest.application.dto import (
    RunBacktestRequest,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestVariantScoreDetailsV1,
    CurrentUser,
)
from trading.contexts.backtest.application.use_cases import RunBacktestUseCase
from trading.contexts.backtest.domain.entities import ExecutionOutcomeV1, TradeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1, RiskParamsV1
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
    UserId,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


class _AlignedOnlyCandleFeed:
    """
    CandleFeed stub that accepts only minute-aligned ranges to verify use-case wiring.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic call recorder for stub assertions.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stub has no external dependencies and stores calls in-memory.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.calls: list[TimeRange] = []

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Reject non-minute-aligned calls and return deterministic dense `1m` arrays.

        Args:
            market_id: Requested market identifier.
            symbol: Requested symbol.
            time_range: Requested feed range.
        Returns:
            CandleArrays: Dense `1m` candles for supplied range.
        Assumptions:
            Range is expected to be normalized by backtest timeline builder.
        Raises:
            ValueError: If range bounds are not aligned to minute boundaries.
        Side Effects:
            Appends requested range to in-memory calls list.
        """
        _ = market_id, symbol
        if (
            time_range.start.value.second != 0
            or time_range.start.value.microsecond != 0
            or time_range.end.value.second != 0
            or time_range.end.value.microsecond != 0
        ):
            raise ValueError("time_range must be minute-aligned")
        self.calls.append(time_range)
        return _build_dense_1m_from_time_range(time_range=time_range)


class _EstimateOnlyIndicatorCompute:
    """
    IndicatorCompute stub that materializes estimate axes from request grid specs.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic estimate call recorder.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `compute` is not used by staged wiring tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.estimate_calls = 0

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Materialize axes from explicit specs and return deterministic estimate payload.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard value.
        Returns:
            EstimateResult: Deterministic estimate with axis values and variants count.
        Assumptions:
            Test fixtures use explicit axis specs only.
        Raises:
            ValueError: If variants exceed guard.
        Side Effects:
            Increments in-memory estimate calls counter.
        """
        self.estimate_calls += 1
        axes: list[AxisDef] = []
        variants = 1

        if grid.source is not None:
            source_values = tuple(str(value) for value in grid.source.materialize())
            axes.append(AxisDef(name="source", values_enum=source_values))
            variants *= len(source_values)

        for param_name in sorted(grid.params.keys()):
            values = tuple(grid.params[param_name].materialize())
            variants *= len(values)
            axes.append(_axis_def(name=param_name, values=values))

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
        Return placeholder tensor for protocol compatibility.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Placeholder casted object.
        Assumptions:
            Staged runner tests do not invoke compute.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = req
        return cast(IndicatorTensor, object())

    def warmup(self) -> None:
        """
        No-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is irrelevant for staged wiring tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _DeterministicScorer:
    """
    Staged scorer fake returning deterministic metric based on indicator selection.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
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
        Return deterministic `Total Return [%]` derived from `window` parameter value.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicator key for deterministic identity.
            variant_key: Backtest variant key for deterministic identity.
        Returns:
            dict[str, float]: Deterministic metric payload.
        Assumptions:
            Test fixture contains one indicator selection with integer `window` parameter.
        Raises:
            KeyError: If expected `window` parameter is missing.
        Side Effects:
            None.
        """
        _ = stage, candles, signal_params, risk_params, indicator_variant_key, variant_key
        window = int(indicator_selections[0].params["window"])
        return {"Total Return [%]": float(window)}


class _DeterministicScorerWithDetails:
    """
    Deterministic scorer fake that also returns Stage-B details for reporting integration tests.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
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
        Return deterministic `Total Return [%]` metric based on `window` parameter.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicator key for deterministic identity.
            variant_key: Backtest variant key for deterministic identity.
        Returns:
            dict[str, float]: Deterministic ranking metric payload.
        Assumptions:
            Fixture includes one indicator with integer `window` parameter.
        Raises:
            KeyError: If `window` parameter is absent.
        Side Effects:
            None.
        """
        _ = stage, candles, signal_params, risk_params, indicator_variant_key, variant_key
        window = int(indicator_selections[0].params["window"])
        return {"Total Return [%]": float(window)}

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, float | int | str | bool | None]],
        risk_params: Mapping[str, float | int | str | bool | None],
        indicator_variant_key: str,
        variant_key: str,
    ) -> BacktestVariantScoreDetailsV1:
        """
        Return deterministic detailed payload used by Stage-B reporting assembly.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicator key for deterministic identity.
            variant_key: Backtest variant key for deterministic identity.
        Returns:
            BacktestVariantScoreDetailsV1: Detailed scorer payload for reporting integration.
        Assumptions:
            Reporting integration test does not verify exact trade economics.
        Raises:
            KeyError: If `window` parameter is absent.
        Side Effects:
            None.
        """
        _ = stage, signal_params, risk_params, indicator_variant_key, variant_key
        window = int(indicator_selections[0].params["window"])
        metric_value = float(window)
        return BacktestVariantScoreDetailsV1(
            metrics={"Total Return [%]": metric_value},
            target_slice=slice(0, int(candles.close.shape[0])),
            execution_params=ExecutionParamsV1(
                direction_mode="long-short",
                sizing_mode="all_in",
                init_cash_quote=1000.0,
                fixed_quote=100.0,
                safe_profit_percent=30.0,
                fee_pct=0.0,
                slippage_pct=0.0,
            ),
            risk_params=RiskParamsV1(
                sl_enabled=False,
                sl_pct=None,
                tp_enabled=False,
                tp_pct=None,
            ),
            execution_outcome=_execution_outcome_with_single_trade(total_return_pct=metric_value),
        )


class _UnusedStrategyReader:
    """
    Backtest strategy reader stub for template-mode tests.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def load_any(self, *, strategy_id: UUID) -> None:
        """
        Return no snapshot because template mode does not need saved strategy lookup.

        Args:
            strategy_id: Requested saved strategy id.
        Returns:
            None: Always `None` for template mode tests.
        Assumptions:
            Caller runs only in template mode.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return None


def test_run_backtest_use_case_normalizes_non_aligned_range_via_timeline_builder() -> None:
    """
    Verify use-case normalizes non-aligned request range before candle feed call.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Timeline builder is responsible for minute normalization and warmup lookback.
    Raises:
        AssertionError: If feed call range or staged output counters are incorrect.
    Side Effects:
        None.
    """
    candle_feed = _AlignedOnlyCandleFeed()
    indicator_compute = _EstimateOnlyIndicatorCompute()
    use_case = RunBacktestUseCase(
        candle_feed=candle_feed,
        indicator_compute=indicator_compute,
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorer(),
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, 45, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 10, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20,)),
        warmup_bars=2,
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(candle_feed.calls) == 1
    normalized_range = candle_feed.calls[0]
    assert normalized_range.start == UtcTimestamp(
        datetime(2026, 2, 16, 11, 58, tzinfo=timezone.utc)
    )
    assert normalized_range.end == UtcTimestamp(datetime(2026, 2, 16, 12, 11, tzinfo=timezone.utc))
    assert response.total_indicator_compute_calls == 1
    assert response.warmup_bars == 2
    assert len(response.variants) == 1


def test_run_backtest_use_case_applies_staged_top_k_limit() -> None:
    """
    Verify use-case forwards top-k settings to staged pipeline and returns ranked variants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Deterministic scorer ranks by `window` parameter value.
    Raises:
        AssertionError: If staged ranking or top-k truncation behavior is incorrect.
    Side Effects:
        None.
    """
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorer(),
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20, 25)),
        top_k=1,
        preselect=2,
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(response.variants) == 1
    assert response.variants[0].total_return_pct == 25.0


def test_run_backtest_use_case_returns_trades_only_for_configured_top_n() -> None:
    """
    Verify reporting payload keeps full trades only for configured best N variants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variant ranking is deterministic by `Total Return [%]` descending.
    Raises:
        AssertionError: If trades are not restricted to configured top-N variants.
    Side Effects:
        None.
    """
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorerWithDetails(),
        top_trades_n_default=2,
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20, 25, 30)),
        top_k=3,
        preselect=3,
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(response.variants) == 3
    assert response.variants[0].total_return_pct == 30.0
    assert response.variants[1].total_return_pct == 25.0
    assert response.variants[2].total_return_pct == 20.0

    assert response.variants[0].report is not None
    assert response.variants[1].report is not None
    assert response.variants[2].report is not None
    assert response.variants[0].report is not None
    assert response.variants[0].report.trades is not None
    assert response.variants[1].report is not None
    assert response.variants[1].report.trades is not None
    assert response.variants[2].report is not None
    assert response.variants[2].report.trades is None
    assert response.variants[0].report.table_md is not None
    assert response.variants[0].report.table_md.startswith("|Metric|Value|")


def _execution_outcome_with_single_trade(*, total_return_pct: float) -> ExecutionOutcomeV1:
    """
    Build deterministic execution outcome fixture with one closed trade.

    Args:
        total_return_pct: Total return metric mirrored into outcome payload.
    Returns:
        ExecutionOutcomeV1: Execution fixture used by scorer-details test double.
    Assumptions:
        Trade economics are minimal and only required to satisfy domain invariants.
    Raises:
        ValueError: If execution/trade payload violates domain invariants.
    Side Effects:
        None.
    """
    trade = TradeV1(
        trade_id=1,
        direction="long",
        entry_bar_index=0,
        exit_bar_index=1,
        entry_fill_price=100.0,
        exit_fill_price=101.0,
        qty_base=1.0,
        entry_quote_amount=100.0,
        exit_quote_amount=101.0,
        entry_fee_quote=0.0,
        exit_fee_quote=0.0,
        gross_pnl_quote=1.0,
        net_pnl_quote=1.0,
        locked_profit_quote=0.0,
        exit_reason="signal_exit",
    )
    return ExecutionOutcomeV1(
        trades=(trade,),
        equity_end_quote=1000.0 + total_return_pct,
        available_quote=1000.0 + total_return_pct,
        safe_quote=0.0,
        total_return_pct=total_return_pct,
    )


def _build_template(*, windows: tuple[int, ...]) -> RunBacktestTemplate:
    """
    Build deterministic template payload for staged use-case tests.

    Args:
        windows: Explicit `window` axis values for `ma.sma` indicator grid.
    Returns:
        RunBacktestTemplate: Valid template-mode request payload.
    Assumptions:
        One indicator grid is sufficient to test staged runner wiring.
    Raises:
        ValueError: If any primitive/grid invariant fails.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={
                    "window": ExplicitValuesSpec(name="window", values=windows),
                },
            ),
        ),
    )


def _build_dense_1m_from_time_range(*, time_range: TimeRange) -> CandleArrays:
    """
    Build deterministic dense `1m` candles for supplied aligned range.

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
    values = np.arange(1, count + 1, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=time_range,
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
    Build `AxisDef` using value-family type inferred from materialized axis tuple.

    Args:
        name: Axis name.
        values: Materialized axis values.
    Returns:
        AxisDef: Deterministic axis definition instance.
    Assumptions:
        Axis values are homogeneous (`int`, `float`, or `str`).
    Raises:
        ValueError: If values are empty or contain unsupported scalar types.
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
