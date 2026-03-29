from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, cast
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
from trading.contexts.backtest.application.services import ArtifactCoordinatesV2
from trading.contexts.backtest.application.services.grid_builder_v1 import (
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.application.services.staged_core_runner_v1 import (
    BacktestStageAScoredVariantV1,
)
from trading.contexts.backtest.application.use_cases import RunBacktestUseCase
from trading.contexts.backtest.application.use_cases import run_backtest as run_backtest_module
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


@dataclass(frozen=True, slots=True)
class _FakeSlotPinnedContext:
    """
    Minimal slot-pinned context fixture used to assert sync bootstrap wiring.
    """

    coordinates: ArtifactCoordinatesV2
    artifact_slot: str
    slot_generation: int
    artifact_asof_date: str
    artifact_manifest_hash: str


class _RecordingArtifactSlotResolver:
    """
    Fake resolver recording sync bootstrap calls for slot-pinned context parity assertions.
    """

    def __init__(self, *, context: _FakeSlotPinnedContext) -> None:
        """
        Initialize resolver fake with one deterministic slot-pinned context fixture.

        Args:
            context: Slot-pinned context fixture returned for active bootstrap calls.
        Returns:
            None.
        Assumptions:
            Sync use-case tests need only `resolve_active_context(...)`.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call logs for later assertions.
        """
        self.context = context
        self.active_calls: list[ArtifactCoordinatesV2] = []

    def resolve_active_context(self, coordinates: ArtifactCoordinatesV2) -> _FakeSlotPinnedContext:
        """
        Record one sync bootstrap call and return the deterministic slot-pinned context.

        Args:
            coordinates: Requested artifact coordinates for the sync run template.
        Returns:
            _FakeSlotPinnedContext: Fixed slot-pinned context fixture.
        Assumptions:
            Sync use-case tests do not need background-pinned bootstrap behavior.
        Raises:
            None.
        Side Effects:
            Appends requested coordinates to the in-memory call log.
        """
        self.active_calls.append(coordinates)
        return self.context

    def resolve_pinned_context(
        self,
        coordinates: ArtifactCoordinatesV2,
        pinned_identity: Any,
    ) -> Any:
        """
        Reject unexpected background bootstrap calls in sync use-case tests.

        Args:
            coordinates: Ignored coordinates argument.
            pinned_identity: Ignored persisted pin payload.
        Returns:
            Any: Never returns because this path is unexpected here.
        Assumptions:
            `RunBacktestUseCase` should only use `resolve_active_context(...)`.
        Raises:
            AssertionError: Always, to signal unexpected background bootstrap usage.
        Side Effects:
            None.
        """
        _ = coordinates, pinned_identity
        raise AssertionError("sync use-case must not call resolve_pinned_context")


class _RecordingStageAShortlistBuilder:
    """
    Fake artifact-backed Stage A builder recording sync use-case wiring inputs.
    """

    def __init__(
        self,
        *,
        rows: tuple[BacktestStageAScoredVariantV1, ...],
    ) -> None:
        """
        Initialize fake builder with deterministic shortlist rows.

        Args:
            rows: Ranked Stage A rows returned for every build call.
        Returns:
            None.
        Assumptions:
            Sync orchestration tests verify wiring and not kernel economics.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def build_shortlist(
        self,
        *,
        grid_context: Any,
        artifact_context: Any,
        target_time_range: TimeRange,
        shortlist_limit: int,
        ranking: Any = None,
        batch_size: int | None = None,
        cancel_checker: Any = None,
        on_checkpoint: Any = None,
    ) -> tuple[BacktestStageAScoredVariantV1, ...]:
        """
        Record one shortlist build call and return the predefined deterministic rows.

        Args:
            grid_context: Prepared Stage A grid context.
            artifact_context: Resolved slot-pinned context.
            target_time_range: Requested trading window.
            shortlist_limit: Requested shortlist cap.
            ranking: Optional ranking config.
            batch_size: Optional chunk size override.
            cancel_checker: Optional cancellation hook.
            on_checkpoint: Optional checkpoint hook.
        Returns:
            tuple[BacktestStageAScoredVariantV1, ...]: Prebuilt deterministic Stage A rows.
        Assumptions:
            Fake builder does not execute kernels and therefore ignores runtime hooks.
        Raises:
            None.
        Side Effects:
            Appends call metadata to the in-memory log.
        """
        _ = grid_context, batch_size, cancel_checker, on_checkpoint
        self.calls.append(
            {
                "artifact_context": artifact_context,
                "target_time_range": target_time_range,
                "shortlist_limit": shortlist_limit,
                "ranking": ranking,
            }
        )
        return self.rows


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

    def score_variant_metric(
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
        Return metric-only payload required by Stage-A/Stage-B ranking hot path contracts.

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
            Metric-only output is equal to legacy `score_variant` payload for this test double.
        Raises:
            KeyError: If `window` parameter is absent.
        Side Effects:
            None.
        """
        return self.score_variant(
            stage=stage,
            candles=candles,
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            indicator_variant_key=indicator_variant_key,
            variant_key=variant_key,
        )

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


def test_run_backtest_use_case_bootstraps_active_slot_pinned_context_before_runtime() -> None:
    """
    Verify sync use-case resolves the shared slot-pinned context before runtime work starts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R6-01 bootstrap is additive here and should not change staged scoring behavior.
    Raises:
        AssertionError: If sync bootstrap coordinates or pinned identity fields drift.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    resolver = _RecordingArtifactSlotResolver(
        context=_FakeSlotPinnedContext(
            coordinates=ArtifactCoordinatesV2(
                exchange="binance",
                market_type="spot",
                symbol="BTCUSDT",
            ),
            artifact_slot="slot_a",
            slot_generation=7,
            artifact_asof_date="2026-03-29",
            artifact_manifest_hash="d" * 64,
        )
    )
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorer(),
        artifact_slot_resolver=cast(Any, resolver),
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20,)),
        top_k=1,
        preselect=1,
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert response.total_indicator_compute_calls == 1
    assert resolver.active_calls == (
        [
            ArtifactCoordinatesV2(
                exchange="binance",
                market_type="spot",
                symbol="BTCUSDT",
            )
        ]
    )
    assert resolver.context.artifact_slot == "slot_a"
    assert resolver.context.slot_generation == 7
    assert resolver.context.artifact_asof_date == "2026-03-29"
    assert resolver.context.artifact_manifest_hash == "d" * 64


def test_run_backtest_use_case_sync_summary_path_stays_summary_only() -> None:
    """
    Verify sync runtime summary path omits report/trades bodies even with legacy eager flag set.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Explicit eager flag must not re-enable runtime report/trades materialization.
    Raises:
        AssertionError: If sync runtime summary response contains eager report/trades payloads.
    Side Effects:
        None.
    """
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorerWithDetails(),
        top_trades_n_default=2,
        eager_top_reports_enabled=True,
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
    assert all(item.report is None for item in response.variants)


def test_run_backtest_use_case_uses_artifact_stage_a_shortlist_builder_when_available() -> None:
    """
    Verify sync use-case forwards pinned context and request range into artifact Stage A builder.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Additive R6-02 cutover replaces only Stage A shortlist build and leaves Stage B scoring
        on the existing path.
    Raises:
        AssertionError: If builder wiring or returned variant payload drifts.
    Side Effects:
        None.
    """
    pinned_context = _FakeSlotPinnedContext(
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_slot="slot_a",
        slot_generation=7,
        artifact_asof_date="2026-03-29",
        artifact_manifest_hash="a" * 64,
    )
    shortlist_builder = _RecordingStageAShortlistBuilder(
        rows=(
            BacktestStageAScoredVariantV1(
                base_variant=BacktestStageABaseVariant(
                    stage_a_index=0,
                    indicator_selections=(
                        IndicatorVariantSelection(
                            indicator_id="ema",
                            inputs={"source": "close"},
                            params={"window": 20},
                        ),
                    ),
                    signal_params={},
                    indicator_variant_key="1" * 64,
                    base_variant_key="2" * 64,
                ),
                total_return_pct=20.0,
            ),
        )
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
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=_DeterministicScorer(),
        artifact_slot_resolver=cast(
            Any,
            _RecordingArtifactSlotResolver(context=pinned_context),
        ),
        stage_a_shortlist_builder=cast(Any, shortlist_builder),
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(shortlist_builder.calls) == 1
    assert shortlist_builder.calls[0]["artifact_context"] == pinned_context
    assert shortlist_builder.calls[0]["target_time_range"] == request.time_range
    assert shortlist_builder.calls[0]["shortlist_limit"] == 2
    assert len(response.variants) == 1
    assert response.variants[0].total_return_pct == 20.0


def test_run_backtest_use_case_prefers_artifact_backed_stage_b_scorer_when_pinned(
    monkeypatch: Any,
) -> None:
    """
    Verify sync use-case resolves the additive artifact-backed Stage B scorer when context exists.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Resolver bootstrap and scorer construction are independent, so this test patches only the
        Stage B scorer factory and verifies the forwarded slot-pinned context.
    Raises:
        AssertionError: If sync scorer resolution does not prefer the artifact-backed builder.
    Side Effects:
        Monkeypatches the local Stage B scorer factory for the duration of the test.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    pinned_context = _FakeSlotPinnedContext(
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_slot="slot_a",
        slot_generation=7,
        artifact_asof_date="2026-03-29",
        artifact_manifest_hash="a" * 64,
    )
    requested_time_range = TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
    )
    expected_scorer = object()
    calls: list[dict[str, Any]] = []

    def _fake_builder(**kwargs: Any) -> object:
        """
        Record artifact-backed Stage B builder arguments and return a deterministic scorer stub.

        Args:
            **kwargs: Factory arguments forwarded by the sync use-case.
        Returns:
            object: Deterministic scorer sentinel.
        Assumptions:
            This wiring test validates only builder selection and argument forwarding.
        Raises:
            None.
        Side Effects:
            Appends one call payload to the in-memory log.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
        """
        calls.append(kwargs)
        return expected_scorer

    monkeypatch.setattr(
        run_backtest_module,
        "build_default_artifact_backed_stage_b_scorer_v2",
        _fake_builder,
    )
    use_case = RunBacktestUseCase(
        candle_feed=_AlignedOnlyCandleFeed(),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        strategy_reader=_UnusedStrategyReader(),
        artifact_slot_resolver=cast(Any, object()),
    )

    scorer = use_case._resolve_staged_scorer(
        template=_build_template(windows=(20, 25)),
        target_slice=slice(0, 5),
        target_time_range=requested_time_range,
        artifact_context=cast(Any, pinned_context),
    )

    assert scorer is expected_scorer
    assert len(calls) == 1
    assert calls[0]["artifact_slot_resolver"] is use_case._artifact_slot_resolver
    assert calls[0]["artifact_context"] == pinned_context
    assert calls[0]["target_time_range"] == requested_time_range
    assert calls[0]["report_target_slice"] == slice(0, 5)


def test_run_backtest_use_case_lazy_mode_omits_eager_reports_by_default() -> None:
    """
    Verify sync use-case keeps ranked variants and omits report payloads in default lazy mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Sync runtime summaries are summary-only by default.
    Raises:
        AssertionError: If report payload is unexpectedly present by default.
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
    assert all(item.report is None for item in response.variants)


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
