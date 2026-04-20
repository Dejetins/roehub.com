from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast
from unittest.mock import Mock
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
from trading.contexts.backtest.application.services.v2.artifact_runtime_core_v2 import (
    BacktestStageAScoredVariantV2,
)
from trading.contexts.backtest.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestStageABaseVariantV2,
)
from trading.contexts.backtest.application.services.v2.contracts import (
    StageANoRiskMetricsV2,
)
from trading.contexts.backtest.application.services.v2.trade_compactor_kernel import (
    StageACompactExactPayloadV2,
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


def _stage_a_compact_exact_payload() -> StageACompactExactPayloadV2:
    """
    Build one deterministic compact exact payload for sync-persistence-compatible test rows.

    Args:
        None.
    Returns:
        StageACompactExactPayloadV2: Minimal compact-trade payload accepted by parity persistence.
    Assumptions:
        Timeline/use-case tests verify orchestration only and therefore need just one valid
        compact trade retained row.
    Raises:
        ValueError: Propagated if compact payload invariants drift.
    Side Effects:
        Allocates readonly NumPy arrays for the returned payload.
    """
    return StageACompactExactPayloadV2(
        entry_signal_idx=np.asarray((0,), dtype=np.int64),
        entry_exec_idx=np.asarray((1,), dtype=np.int64),
        direction=np.asarray((1,), dtype=np.int8),
        sig_exit_signal_idx=np.asarray((1,), dtype=np.int64),
        sig_exit_exec_idx=np.asarray((1,), dtype=np.int64),
    )


def _stage_a_no_risk_metrics(*, total_return_pct: float) -> StageANoRiskMetricsV2:
    """
    Build deterministic no-risk metrics aligned to one scored Stage A row.

    Args:
        total_return_pct: Deterministic total-return payload used by the test row.
    Returns:
        StageANoRiskMetricsV2: Minimal no-risk metrics payload accepted by parity persistence.
    Assumptions:
        Tests assert ranking/order semantics only, so additive metrics may stay lightweight.
    Raises:
        ValueError: Propagated if metric invariants drift.
    Side Effects:
        None.
    """
    return StageANoRiskMetricsV2(
        total_return_pct=total_return_pct,
        max_drawdown_pct=1.0,
        return_over_max_drawdown=total_return_pct,
        profit_factor=total_return_pct + 1.0,
        trade_count=1,
        sharpe_trades=1.0,
        win_rate_pct=100.0,
        avg_trade_ret_pct=total_return_pct,
        avg_trade_exec_bars=1.0,
        exposure_pct=50.0,
    )


def _stage_a_scored_variant(
    *,
    base_variant: BacktestStageABaseVariantV2,
    total_return_pct: float,
) -> BacktestStageAScoredVariantV2:
    """
    Build one parity-persistence-compatible Stage A row for sync orchestration tests.

    Args:
        base_variant: Deterministic Stage A base variant fixture.
        total_return_pct: Deterministic ranking metric used by the row.
    Returns:
        BacktestStageAScoredVariantV2: Stage A row carrying compact exact and no-risk payloads.
    Assumptions:
        Sync tests should not fail only because lightweight rows omit additive persistence fields.
    Raises:
        ValueError: Propagated if compact payload or metric fixtures violate invariants.
    Side Effects:
        None.
    """
    return BacktestStageAScoredVariantV2(
        base_variant=base_variant,
        total_return_pct=total_return_pct,
        retained_exact_payload=_stage_a_compact_exact_payload(),
        no_risk_metrics=_stage_a_no_risk_metrics(total_return_pct=total_return_pct),
    )


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
        rows: tuple[BacktestStageAScoredVariantV2, ...],
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
        parallelism: Any = None,
        batch_size: int | None = None,
        cancel_checker: Any = None,
        on_checkpoint: Any = None,
    ) -> tuple[BacktestStageAScoredVariantV2, ...]:
        """
        Record one shortlist build call and return the predefined deterministic rows.

        Args:
            grid_context: Prepared Stage A grid context.
            artifact_context: Resolved slot-pinned context.
            target_time_range: Requested trading window.
            shortlist_limit: Requested shortlist cap.
            ranking: Optional ranking config.
            parallelism: Optional Stage A parallelism contract forwarded by runtime orchestration.
            batch_size: Optional chunk size override.
            cancel_checker: Optional cancellation hook.
            on_checkpoint: Optional checkpoint hook.
        Returns:
            tuple[BacktestStageAScoredVariantV2, ...]: Prebuilt deterministic Stage A rows.
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
                "parallelism": parallelism,
            }
        )
        return self.rows


class _RecordingArtifactTimelineBuilder:
    """
    Artifact timeline builder fake recording sync use-case inputs and returning dense candles.
    """

    def __init__(self) -> None:
        """
        Initialize artifact timeline builder fake with empty call log.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Sync use-case tests only need request-timeframe warmup handling and call recording.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        artifact_context: Any,
        market_id: MarketId,
        symbol: Symbol,
        timeframe: Timeframe,
        requested_time_range: TimeRange,
        warmup_bars: int,
    ) -> Any:
        """
        Record one build call and return deterministic warmup-inclusive `1m` candles.

        Args:
            artifact_context: Slot-pinned runtime context forwarded by the use-case.
            market_id: Requested market identifier.
            symbol: Requested symbol.
            timeframe: Requested timeframe.
            requested_time_range: Requested reporting/trading window.
            warmup_bars: Requested warmup bars count.
        Returns:
            Any: Timeline-like object exposing `candles`, `target_slice`, and `full_target_slice`.
        Assumptions:
            Tests use `1m` timeframe only and rely on minute-floor/ceil normalization here.
        Raises:
            ValueError: If `warmup_bars` is non-positive.
        Side Effects:
            Appends build metadata to the in-memory log.
        """
        if warmup_bars <= 0:
            raise ValueError("warmup_bars must be > 0")
        _ = market_id, symbol, timeframe
        normalized_target_range = _normalize_request_time_range_to_minutes(
            requested_time_range=requested_time_range
        )
        normalized_timeline_range = TimeRange(
            start=UtcTimestamp(normalized_target_range.start.value - (warmup_bars * _ONE_MINUTE)),
            end=normalized_target_range.end,
        )
        candles = _build_dense_1m_from_time_range(time_range=normalized_timeline_range)
        target_bars = int(normalized_target_range.duration() // _ONE_MINUTE)
        target_slice = slice(warmup_bars, warmup_bars + target_bars)
        self.calls.append(
            {
                "artifact_context": artifact_context,
                "requested_time_range": requested_time_range,
                "warmup_bars": warmup_bars,
                "normalized_timeline_range": normalized_timeline_range,
                "target_slice": target_slice,
            }
        )
        return SimpleNamespace(
            candles=candles,
            target_slice=target_slice,
            full_target_slice=target_slice,
        )


class _ArtifactOnlyStageAShortlistBuilder:
    """
    Artifact-only Stage A shortlist builder fake for sync use-case tests.
    """

    def build_shortlist(
        self,
        *,
        grid_context: Any,
        artifact_context: Any,
        target_time_range: TimeRange,
        shortlist_limit: int,
        ranking: Any = None,
        parallelism: Any = None,
        batch_size: int | None = None,
        cancel_checker: Any = None,
        on_checkpoint: Any = None,
    ) -> tuple[BacktestStageAScoredVariantV2, ...]:
        """
        Build deterministic shortlist rows directly from runtime-plan Stage A variants.

        Args:
            grid_context: Prepared runtime plan exposing `iter_stage_a_variants()`.
            artifact_context: Slot-pinned runtime context.
            target_time_range: Requested trading/reporting window.
            shortlist_limit: Maximum shortlist size.
            ranking: Optional ranking config.
            parallelism: Optional Stage A parallelism contract forwarded by runtime orchestration.
            batch_size: Optional chunk size override.
            cancel_checker: Optional cancellation hook.
            on_checkpoint: Optional checkpoint hook.
        Returns:
            tuple[BacktestStageAScoredVariantV2, ...]: Deterministic shortlist rows.
        Assumptions:
            Tests only need stable shortlist materialization and not real Stage A kernel math.
        Raises:
            None.
        Side Effects:
            May invoke provided cancellation/checkpoint hooks.
        """
        _ = artifact_context, target_time_range, ranking, parallelism, batch_size
        if cancel_checker is not None:
            cancel_checker("stage_a")
        base_variants = tuple(grid_context.iter_stage_a_variants())[:shortlist_limit]
        rows = tuple(
            _stage_a_scored_variant(
                base_variant=cast(Any, base_variant),
                total_return_pct=float(base_variant.indicator_selections[0].params["window"]),
            )
            for base_variant in base_variants
        )
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (-row.total_return_pct, row.base_variant.base_variant_key),
            )
        )
        if on_checkpoint is not None:
            on_checkpoint(len(rows), len(tuple(grid_context.iter_stage_a_variants())))
        return rows


class _RecordingHierarchicalShortlistBuilder:
    """
    Hybrid shortlist builder fake recording sync use-case hybrid runtime inputs.
    """

    def __init__(self, *, runtime_plan: Any) -> None:
        """
        Initialize fake builder with one deterministic reduced runtime plan.

        Args:
            runtime_plan: Runtime plan returned for every hybrid shortlist build.
        Returns:
            None.
        Assumptions:
            Sync tests only verify hybrid builder wiring and not shortlist internals here.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self.runtime_plan = runtime_plan
        self.calls: list[dict[str, Any]] = []

    def build_runtime_plan(
        self,
        *,
        runtime_plan: Any,
        artifact_context: Any,
        target_time_range: TimeRange,
    ) -> Any:
        """
        Record one hybrid shortlist build call and return the predefined reduced plan.

        Args:
            runtime_plan: Original exact runtime plan.
            artifact_context: Slot-pinned runtime context.
            target_time_range: Requested trading window.
        Returns:
            Any: Prebuilt reduced runtime plan fixture.
        Assumptions:
            Test coverage here is limited to ownership/wiring, not hybrid shortlist math.
        Raises:
            None.
        Side Effects:
            Appends one call payload to the in-memory log.
        """
        self.calls.append(
            {
                "runtime_plan": runtime_plan,
                "artifact_context": artifact_context,
                "target_time_range": target_time_range,
            }
        )
        return self.runtime_plan


class _StaticRuntimePlanner:
    """
    Runtime planner fake returning one prebuilt runtime plan for sync orchestration tests.
    """

    def __init__(self, *, runtime_plan: Any) -> None:
        """
        Initialize planner fake with one deterministic runtime-plan payload.

        Args:
            runtime_plan: Prebuilt runtime plan returned for every build call.
        Returns:
            None.
        Assumptions:
            Sync wiring tests verify planner forwarding and do not need real estimate math here.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self._runtime_plan = runtime_plan
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        template: Any,
        candles: Any,
        indicator_compute: Any,
        preselect: int,
        requested_execution_profile_mode: str | None = None,
        defaults_provider: Any = None,
        max_variants_per_compute: int,
        max_compute_bytes_total: int,
    ) -> Any:
        """
        Record one sync planner call and return the configured runtime plan.

        Args:
            template: Effective run template.
            candles: Warmup-aware artifact candles.
            indicator_compute: Indicator compute dependency.
            preselect: Requested Stage A shortlist cap.
            requested_execution_profile_mode: Optional internal profile override.
            defaults_provider: Optional runtime defaults provider.
            max_variants_per_compute: Deterministic Stage A variants guard.
            max_compute_bytes_total: Deterministic memory guard.
        Returns:
            Any: Prebuilt runtime plan fixture.
        Assumptions:
            Tests only need the resolved execution profile carried by the returned runtime plan.
        Raises:
            None.
        Side Effects:
            Appends build metadata to the in-memory call log.
        """
        _ = template, candles, indicator_compute, defaults_provider
        self.calls.append(
            {
                "preselect": preselect,
                "requested_execution_profile_mode": requested_execution_profile_mode,
                "max_variants_per_compute": max_variants_per_compute,
                "max_compute_bytes_total": max_compute_bytes_total,
            }
        )
        return self._runtime_plan


class _StaticRuntimeRunner:
    """
    Runtime runner fake returning fixed Stage B rows/tasks for sync orchestration tests.
    """

    def __init__(
        self,
        *,
        ranked_rows: tuple[Any, ...],
        ranked_tasks: Mapping[str, Any],
    ) -> None:
        """
        Initialize runner fake with deterministic Stage B payloads.

        Args:
            ranked_rows: Ranked Stage B rows returned by every run.
            ranked_tasks: Deterministic task mapping keyed by `variant_key`.
        Returns:
            None.
        Assumptions:
            Sync wiring tests only need preview-building compatibility, not real Stage B scoring.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self._ranked_rows = ranked_rows
        self._ranked_tasks = ranked_tasks
        self.calls: list[dict[str, Any]] = []

    def run_stage_b(
        self,
        *,
        template: Any,
        runtime_plan: Any,
        shortlist: Any,
        candles: Any,
        scorer: Any,
        top_k_limit: int,
        ranking: Any = None,
        cancel_checker: Any = None,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """
        Record one Stage B invocation and return fixed ranking payloads.

        Args:
            template: Effective run template.
            runtime_plan: Resolved runtime plan used for the sync run.
            shortlist: Deterministic Stage A shortlist rows.
            candles: Warmup-aware candles payload.
            scorer: Resolved scorer dependency.
            top_k_limit: Requested top-k cap.
            ranking: Optional ranking config.
            cancel_checker: Optional cancellation hook.
        Returns:
            tuple[tuple[Any, ...], Mapping[str, Any]]: Fixed ranked rows and task mapping.
        Assumptions:
            Preview-building tests need only stable `variant_key` alignment.
        Raises:
            None.
        Side Effects:
            Appends run metadata to the in-memory call log.
        """
        _ = template, candles, scorer, ranking
        self.calls.append(
            {
                "runtime_plan": runtime_plan,
                "shortlist": shortlist,
                "top_k_limit": top_k_limit,
            }
        )
        if cancel_checker is not None:
            cancel_checker("stage_b")
        return (self._ranked_rows, self._ranked_tasks)

    def run_stage_b_or_finalize_no_risk(
        self,
        *,
        template: Any,
        runtime_plan: Any,
        shortlist: Any,
        candles: Any,
        scorer: Any,
        top_k_limit: int,
        ranking: Any = None,
        cancel_checker: Any = None,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """
        Preserve compatibility with the no-risk-aware runtime runner entrypoint.

        Args:
            template: Effective run template.
            runtime_plan: Resolved runtime plan used for the sync run.
            shortlist: Deterministic Stage A shortlist rows.
            candles: Warmup-aware candles payload.
            scorer: Resolved scorer dependency.
            top_k_limit: Requested top-k cap.
            ranking: Optional ranking config.
            cancel_checker: Optional cancellation hook.
        Returns:
            tuple[tuple[Any, ...], Mapping[str, Any]]: Fixed ranked rows and task mapping.
        Assumptions:
            This fake does not model the no-risk bypass separately and reuses the same
            deterministic payload as `run_stage_b(...)`.
        Raises:
            None.
        Side Effects:
            Appends run metadata to the in-memory call log through `run_stage_b(...)`.
        """
        return self.run_stage_b(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
            candles=candles,
            scorer=scorer,
            top_k_limit=top_k_limit,
            ranking=ranking,
            cancel_checker=cancel_checker,
        )


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
        Return metric-only payload required by artifact-backed Stage B ranking loops.

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
            Metric-only output is equal to `score_variant(...)` payload for this test double.
        Raises:
            KeyError: If expected `window` parameter is missing.
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


def _default_slot_pinned_context() -> _FakeSlotPinnedContext:
    """
    Build deterministic default slot-pinned context fixture for sync use-case tests.

    Args:
        None.
    Returns:
        _FakeSlotPinnedContext: Default slot-pinned context fixture.
    Assumptions:
        Sync use-case tests all target the same `binance/spot/BTCUSDT` artifact family.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _FakeSlotPinnedContext(
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


def _build_use_case(
    *,
    indicator_compute: Any | None = None,
    staged_scorer: Any | None = None,
    artifact_slot_resolver: Any | None = None,
    artifact_timeline_builder: Any | None = None,
    stage_a_shortlist_builder: Any | None = None,
    **kwargs: Any,
) -> RunBacktestUseCase:
    """
    Build sync run use-case with deterministic artifact-runtime test doubles.

    Args:
        indicator_compute: Optional indicator compute fake.
        staged_scorer: Optional staged scorer fake.
        artifact_slot_resolver: Optional slot-pinned resolver fake.
        artifact_timeline_builder: Optional artifact timeline builder fake.
        stage_a_shortlist_builder: Optional artifact Stage A shortlist builder fake.
        **kwargs: Additional constructor kwargs forwarded to `RunBacktestUseCase`.
    Returns:
        RunBacktestUseCase: Prepared sync use-case instance.
    Assumptions:
        Tests exercise template mode only and therefore use the unused strategy reader stub.
    Raises:
        None.
    Side Effects:
        None.
    """
    resolved_artifact_slot_resolver = (
        artifact_slot_resolver
        if artifact_slot_resolver is not None
        else _RecordingArtifactSlotResolver(context=_default_slot_pinned_context())
    )
    resolved_timeline_builder = (
        artifact_timeline_builder
        if artifact_timeline_builder is not None
        else _RecordingArtifactTimelineBuilder()
    )
    resolved_stage_a_shortlist_builder = (
        stage_a_shortlist_builder
        if stage_a_shortlist_builder is not None
        else _ArtifactOnlyStageAShortlistBuilder()
    )
    return RunBacktestUseCase(
        candle_feed=None,
        indicator_compute=cast(
            Any,
            indicator_compute if indicator_compute is not None else _EstimateOnlyIndicatorCompute(),
        ),
        strategy_reader=_UnusedStrategyReader(),
        staged_scorer=cast(Any, staged_scorer),
        artifact_slot_resolver=cast(Any, resolved_artifact_slot_resolver),
        artifact_timeline_builder=cast(Any, resolved_timeline_builder),
        stage_a_shortlist_builder=cast(Any, resolved_stage_a_shortlist_builder),
        **kwargs,
    )


def test_run_backtest_use_case_routes_sync_path_through_artifact_timeline_builder() -> None:
    """
    Verify sync use-case routes through artifact timeline builder and preserves warmup handling.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Artifact timeline builder owns minute normalization and warmup expansion in R10-01.
    Raises:
        AssertionError: If builder call payload or staged output counters are incorrect.
    Side Effects:
        None.
    """
    timeline_builder = _RecordingArtifactTimelineBuilder()
    indicator_compute = _EstimateOnlyIndicatorCompute()
    use_case = _build_use_case(
        indicator_compute=indicator_compute,
        staged_scorer=_DeterministicScorer(),
        artifact_timeline_builder=timeline_builder,
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

    assert len(timeline_builder.calls) == 1
    normalized_range = timeline_builder.calls[0]["normalized_timeline_range"]
    assert timeline_builder.calls[0]["requested_time_range"] == request.time_range
    assert timeline_builder.calls[0]["warmup_bars"] == 2
    assert normalized_range.start == UtcTimestamp(
        datetime(2026, 2, 16, 11, 58, tzinfo=timezone.utc)
    )
    assert normalized_range.end == UtcTimestamp(datetime(2026, 2, 16, 12, 11, tzinfo=timezone.utc))
    assert response.total_indicator_compute_calls == 1
    assert len(response.variants) == 1


def test_run_backtest_use_case_uses_run_scoped_artifact_builders_per_public_call() -> None:
    """
    Verify sync use-case routes `preflight` and `execute` through fresh `run-scoped` builders.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Long-lived API-owned use-case instances should keep only prototype builders while each
        public call owns fresh artifact loader caches.
    Raises:
        AssertionError: If `preflight` or `execute` uses prototype builders directly.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    prototype_timeline_builder = _RecordingArtifactTimelineBuilder()
    run_scoped_timeline_builder_one = _RecordingArtifactTimelineBuilder()
    run_scoped_timeline_builder_two = _RecordingArtifactTimelineBuilder()
    prototype_timeline_builder.run_scoped = Mock(  # type: ignore[attr-defined]
        side_effect=(
            run_scoped_timeline_builder_one,
            run_scoped_timeline_builder_two,
        )
    )
    stage_a_row = _stage_a_scored_variant(
        base_variant=BacktestStageABaseVariantV2(
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
    )
    prototype_stage_a_builder = _RecordingStageAShortlistBuilder(rows=(stage_a_row,))
    run_scoped_stage_a_builder = _RecordingStageAShortlistBuilder(rows=(stage_a_row,))
    prototype_stage_a_builder.run_scoped = Mock(  # type: ignore[attr-defined]
        return_value=run_scoped_stage_a_builder
    )
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorer(),
        artifact_timeline_builder=prototype_timeline_builder,
        stage_a_shortlist_builder=prototype_stage_a_builder,
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20,)),
        warmup_bars=2,
        top_k=1,
        preselect=1,
    )
    current_user = CurrentUser(
        user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))
    )

    use_case.preflight(request=request, current_user=current_user)
    response = use_case.execute(request=request, current_user=current_user)

    assert prototype_timeline_builder.run_scoped.call_count == 2  # type: ignore[attr-defined]
    assert prototype_stage_a_builder.run_scoped.call_count == 1  # type: ignore[attr-defined]
    assert len(prototype_timeline_builder.calls) == 0
    assert len(prototype_stage_a_builder.calls) == 0
    assert len(run_scoped_timeline_builder_one.calls) == 1
    assert len(run_scoped_timeline_builder_two.calls) == 1
    assert len(run_scoped_stage_a_builder.calls) == 1
    assert len(response.variants) == 1


def test_run_backtest_use_case_uses_run_scoped_hierarchical_builder_for_hybrid_runtime() -> None:
    """
    Verify sync use-case routes hybrid shortlist runtime through a fresh `run-scoped` builder.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `hybrid_conservative` is a live API path, so its loader-owning builder must not retain
        mmap caches on the long-lived use-case singleton.
    Raises:
        AssertionError: If hybrid runtime uses the prototype builder directly.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
    """
    stage_a_row = _stage_a_scored_variant(
        base_variant=BacktestStageABaseVariantV2(
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
    )
    hybrid_runtime_plan = SimpleNamespace(
        indicator_estimate_calls=0,
        execution_profile=SimpleNamespace(
            mode="hybrid_conservative",
            parallelism=SimpleNamespace(stage_a_workers=2),
            shortlist_config=SimpleNamespace(enabled=True),
            feature_flags=SimpleNamespace(
                runtime_enabled=True,
                heuristic_shortlist_enabled=True,
                family_plugin_enabled=False,
            ),
        ),
        iter_stage_a_variants=lambda: (stage_a_row.base_variant,),
    )
    prototype_hierarchical_builder = _RecordingHierarchicalShortlistBuilder(
        runtime_plan=hybrid_runtime_plan
    )
    run_scoped_hierarchical_builder = _RecordingHierarchicalShortlistBuilder(
        runtime_plan=hybrid_runtime_plan
    )
    prototype_hierarchical_builder.run_scoped = Mock(  # type: ignore[attr-defined]
        return_value=run_scoped_hierarchical_builder
    )
    shortlist_builder = _RecordingStageAShortlistBuilder(rows=(stage_a_row,))
    runtime_runner = _StaticRuntimeRunner(
        ranked_rows=(
            SimpleNamespace(
                variant_index=stage_a_row.base_variant.stage_a_index,
                variant_key=stage_a_row.base_variant.base_variant_key,
                indicator_variant_key=stage_a_row.base_variant.indicator_variant_key,
                total_return_pct=stage_a_row.total_return_pct,
                summary_metrics_json={"Total Return [%]": stage_a_row.total_return_pct},
                best_tp_pct=None,
                best_sl_pct=None,
            ),
        ),
        ranked_tasks={
            stage_a_row.base_variant.base_variant_key: SimpleNamespace(
                indicator_selections=stage_a_row.base_variant.indicator_selections,
                signal_params=stage_a_row.base_variant.signal_params,
                risk_params={},
            )
        },
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20,)),
        warmup_bars=2,
        top_k=1,
        preselect=1,
    )
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorer(),
        stage_a_shortlist_builder=cast(Any, shortlist_builder),
        hierarchical_shortlist_builder=cast(Any, prototype_hierarchical_builder),
        runtime_planner=cast(Any, _StaticRuntimePlanner(runtime_plan=hybrid_runtime_plan)),
        runtime_runner=cast(Any, runtime_runner),
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert prototype_hierarchical_builder.run_scoped.call_count == 1  # type: ignore[attr-defined]
    assert len(prototype_hierarchical_builder.calls) == 0
    assert len(run_scoped_hierarchical_builder.calls) == 1
    assert run_scoped_hierarchical_builder.calls[0]["target_time_range"] == request.time_range
    assert len(shortlist_builder.calls) == 1
    assert len(response.variants) == 1


def test_run_backtest_use_case_bypasses_hierarchical_builder_for_exact_no_risk_parity_runtime(
) -> None:
    """
    Verify canonical parity sync orchestration bypasses hierarchical reduced-plan runtime wiring.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        D3 canonical `NR2` runs must keep the planner-owned `exact_no_risk_parity` runtime plan
        active and never call `BacktestHierarchicalShortlistBuilderV2`.
    Raises:
        AssertionError: If sync orchestration re-enters hierarchical shortlist reduction.
    Side Effects:
        None.
    """
    stage_a_row = _stage_a_scored_variant(
        base_variant=BacktestStageABaseVariantV2(
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
    )
    parity_runtime_plan = SimpleNamespace(
        indicator_estimate_calls=0,
        stage_a_variants_total=1,
        risk_variants=(
            SimpleNamespace(
                risk_params={
                    "sl_enabled": False,
                    "sl_pct": None,
                    "tp_enabled": False,
                    "tp_pct": None,
                }
            ),
        ),
        execution_profile=SimpleNamespace(
            mode="exact_no_risk_parity",
            parallelism=SimpleNamespace(stage_a_workers=1),
            shortlist_config=SimpleNamespace(enabled=True),
            feature_flags=SimpleNamespace(
                runtime_enabled=True,
                heuristic_shortlist_enabled=False,
                family_plugin_enabled=False,
            ),
        ),
        parity_classification=SimpleNamespace(
            parity_class="parity_first_no_risk_exact",
            disabled_risk_single_cell=True,
            low_indicator_block_cardinality=True,
            narrowed_retained_row_evidence=True,
            notebook_shaped_cost_units=True,
            nr2_classification_reason="canonical NR2 parity no-risk class",
        ),
        parity_runtime_counters=lambda: {
            "retained_rows_per_indicator": {"ma.ema": 1},
            "retained_rows_total": 1,
            "narrowed_combo_total": 1,
            "narrowed_compute_combo_total": 1,
            "no_risk_finalization_count": 1,
            "exact_replay_count": 0,
            "deterministic_combo_ordering": "stage_a_index",
        },
        uses_no_risk_terminal_path=lambda: True,
        stage_b_execution_mode=lambda: "bypassed_no_risk",
        stage_b_process_fallback_threshold=lambda: "none",
        iter_stage_a_variants=lambda: (stage_a_row.base_variant,),
    )
    hierarchical_shortlist_builder = _RecordingHierarchicalShortlistBuilder(
        runtime_plan=parity_runtime_plan
    )
    hierarchical_shortlist_builder.run_scoped = Mock(  # type: ignore[attr-defined]
        side_effect=AssertionError(
            "exact_no_risk_parity sync runtime must bypass hierarchical shortlist builder"
        )
    )
    shortlist_builder = _RecordingStageAShortlistBuilder(rows=(stage_a_row,))
    runtime_runner = _StaticRuntimeRunner(
        ranked_rows=(
            SimpleNamespace(
                variant_index=stage_a_row.base_variant.stage_a_index,
                variant_key=stage_a_row.base_variant.base_variant_key,
                indicator_variant_key=stage_a_row.base_variant.indicator_variant_key,
                total_return_pct=stage_a_row.total_return_pct,
                summary_metrics_json={"Total Return [%]": stage_a_row.total_return_pct},
                best_tp_pct=None,
                best_sl_pct=None,
            ),
        ),
        ranked_tasks={
            stage_a_row.base_variant.base_variant_key: SimpleNamespace(
                indicator_selections=stage_a_row.base_variant.indicator_selections,
                signal_params=stage_a_row.base_variant.signal_params,
                risk_params={},
            )
        },
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(windows=(20,)),
        warmup_bars=2,
        top_k=1,
        preselect=1,
    )
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorer(),
        stage_a_shortlist_builder=cast(Any, shortlist_builder),
        hierarchical_shortlist_builder=cast(Any, hierarchical_shortlist_builder),
        runtime_planner=cast(Any, _StaticRuntimePlanner(runtime_plan=parity_runtime_plan)),
        runtime_runner=cast(Any, runtime_runner),
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert getattr(hierarchical_shortlist_builder, "run_scoped").call_count == 0
    assert len(hierarchical_shortlist_builder.calls) == 0
    assert len(shortlist_builder.calls) == 1
    assert response.execution_profile_mode == "exact_no_risk_parity"
    sync_persistence_artifact = response.sync_persistence_artifact
    assert sync_persistence_artifact is not None
    parity_runtime_state = sync_persistence_artifact.parity_runtime_state
    assert parity_runtime_state is not None
    assert (
        parity_runtime_state.deterministic_combo_ordering == "stage_a_index"
    )


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
    use_case = _build_use_case(
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
    use_case = _build_use_case(
        indicator_compute=_EstimateOnlyIndicatorCompute(),
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
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorerWithDetails(),
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
    Verify sync use-case forwards pinned context, request range, and Stage A parallelism contract.

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
    stage_a_row = _stage_a_scored_variant(
        base_variant=BacktestStageABaseVariantV2(
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
    )
    shortlist_builder = _RecordingStageAShortlistBuilder(rows=(stage_a_row,))
    runtime_plan = SimpleNamespace(
        indicator_estimate_calls=0,
        execution_profile=SimpleNamespace(
            mode="exact_small",
            parallelism=SimpleNamespace(stage_a_workers=2),
            shortlist_config=SimpleNamespace(enabled=False),
            feature_flags=SimpleNamespace(
                runtime_enabled=True,
                heuristic_shortlist_enabled=False,
                family_plugin_enabled=False,
            ),
        )
    )
    runtime_runner = _StaticRuntimeRunner(
        ranked_rows=(
            SimpleNamespace(
                variant_index=stage_a_row.base_variant.stage_a_index,
                variant_key=stage_a_row.base_variant.base_variant_key,
                indicator_variant_key=stage_a_row.base_variant.indicator_variant_key,
                total_return_pct=stage_a_row.total_return_pct,
                summary_metrics_json={"Total Return [%]": stage_a_row.total_return_pct},
                best_tp_pct=None,
                best_sl_pct=None,
            ),
        ),
        ranked_tasks={
            stage_a_row.base_variant.base_variant_key: SimpleNamespace(
                indicator_selections=stage_a_row.base_variant.indicator_selections,
                signal_params=stage_a_row.base_variant.signal_params,
                risk_params={},
            )
        },
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
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorer(),
        artifact_slot_resolver=cast(
            Any,
            _RecordingArtifactSlotResolver(context=pinned_context),
        ),
        stage_a_shortlist_builder=cast(Any, shortlist_builder),
        runtime_planner=cast(Any, _StaticRuntimePlanner(runtime_plan=runtime_plan)),
        runtime_runner=cast(Any, runtime_runner),
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(shortlist_builder.calls) == 1
    assert shortlist_builder.calls[0]["artifact_context"] == pinned_context
    assert shortlist_builder.calls[0]["target_time_range"] == request.time_range
    assert shortlist_builder.calls[0]["shortlist_limit"] == 2
    assert shortlist_builder.calls[0]["parallelism"].stage_a_workers == 2
    assert shortlist_builder.calls[0]["parallelism"].numba_threads == min(
        2,
        run_backtest_module._DEFAULT_MAX_NUMBA_THREADS,
    )
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
    use_case = _build_use_case(
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
    use_case = _build_use_case(
        staged_scorer=_DeterministicScorerWithDetails(),
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


def _normalize_request_time_range_to_minutes(*, requested_time_range: TimeRange) -> TimeRange:
    """
    Normalize one request range to minute boundaries using floor-start and ceil-end semantics.

    Args:
        requested_time_range: Raw requested time range.
    Returns:
        TimeRange: Minute-aligned target range without warmup extension.
    Assumptions:
        Test fixtures use timezone-aware UTC timestamps.
    Raises:
        None.
    Side Effects:
        None.
    """
    start_value = requested_time_range.start.value.astimezone(timezone.utc)
    end_value = requested_time_range.end.value.astimezone(timezone.utc)
    aligned_start = start_value.replace(second=0, microsecond=0)
    aligned_end = end_value.replace(second=0, microsecond=0)
    if aligned_end != end_value:
        aligned_end += _ONE_MINUTE
    return TimeRange(
        start=UtcTimestamp(aligned_start),
        end=UtcTimestamp(aligned_end),
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
