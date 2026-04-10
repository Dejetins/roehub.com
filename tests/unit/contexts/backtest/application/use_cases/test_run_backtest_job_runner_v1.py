from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import UUID

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    BacktestRankingConfig,
    RunBacktestRequest,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    BacktestRiskVariantV1,
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.application.services.staged_core_runner_v1 import (
    BacktestStageAScoredVariantV1,
)
from trading.contexts.backtest.application.services.v2 import (
    artifact_runtime_core_v2 as artifact_runtime_core_module,
)
from trading.contexts.backtest.application.use_cases import RunBacktestJobRunnerV1
from trading.contexts.backtest.application.use_cases import (
    run_backtest_job_runner_v1 as run_backtest_job_runner_module,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobExecutionMode,
    TradeV1,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
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


class _FakeRequestDecoder:
    """
    Deterministic request decoder stub returning predefined request payload.
    """

    def __init__(self, *, request: RunBacktestRequest) -> None:
        """
        Initialize decoder stub with fixed request payload.

        Args:
            request: Prebuilt backtest request fixture.
        Returns:
            None.
        Assumptions:
            Worker tests control the request fixture shape.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._request = request

    def decode(self, *, payload: Mapping[str, Any]) -> RunBacktestRequest:
        """
        Return predefined request payload regardless of persisted JSON content.

        Args:
            payload: Persisted request payload mapping.
        Returns:
            RunBacktestRequest: Prebuilt request fixture.
        Assumptions:
            Decoder behavior is isolated from DTO validation in these tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = payload
        return self._request


@dataclass(frozen=True, slots=True)
class _FakeSlotPinnedContext:
    """
    Minimal slot-pinned context fixture used to assert background bootstrap wiring.
    """

    coordinates: ArtifactCoordinatesV2
    artifact_slot: str
    slot_generation: int
    artifact_asof_date: str
    artifact_manifest_hash: str


class _RecordingArtifactSlotResolver:
    """
    Fake resolver recording background bootstrap calls for slot-pinned context assertions.
    """

    def __init__(self, *, context: _FakeSlotPinnedContext) -> None:
        """
        Initialize resolver fake with one deterministic slot-pinned context fixture.

        Args:
            context: Slot-pinned context fixture returned for pinned bootstrap calls.
        Returns:
            None.
        Assumptions:
            Worker use-case tests need only `resolve_pinned_context(...)`.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call logs for later assertions.
        """
        self.context = context
        self.pinned_calls: list[tuple[ArtifactCoordinatesV2, Any]] = []

    def resolve_active_context(self, coordinates: ArtifactCoordinatesV2) -> Any:
        """
        Reject unexpected sync bootstrap calls in worker use-case tests.

        Args:
            coordinates: Ignored coordinates argument.
        Returns:
            Any: Never returns because this path is unexpected here.
        Assumptions:
            `RunBacktestJobRunnerV1` should only use `resolve_pinned_context(...)`.
        Raises:
            AssertionError: Always, to signal unexpected sync bootstrap usage.
        Side Effects:
            None.
        """
        _ = coordinates
        raise AssertionError("job runner must not call resolve_active_context")

    def resolve_pinned_context(
        self,
        coordinates: ArtifactCoordinatesV2,
        pinned_identity: Any,
    ) -> _FakeSlotPinnedContext:
        """
        Record one background bootstrap call and return the deterministic slot-pinned context.

        Args:
            coordinates: Requested artifact coordinates for the job template.
            pinned_identity: Persisted artifact pin converted by the use-case under test.
        Returns:
            _FakeSlotPinnedContext: Fixed slot-pinned context fixture.
        Assumptions:
            Worker tests do not need real manifest loading to verify bootstrap parity wiring.
        Raises:
            None.
        Side Effects:
            Appends requested coordinates and persisted pin payload to the in-memory call log.
        """
        self.pinned_calls.append((coordinates, pinned_identity))
        return self.context


class _RecordingStageAShortlistBuilder:
    """
    Fake artifact-backed Stage A builder recording worker use-case wiring inputs.
    """

    def __init__(
        self,
        *,
        rows: tuple[BacktestStageAScoredVariantV1, ...],
    ) -> None:
        """
        Initialize fake builder with fixed deterministic shortlist rows.

        Args:
            rows: Ranked Stage A rows returned by every build call.
        Returns:
            None.
        Assumptions:
            Worker tests verify orchestration inputs and not kernel economics here.
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
        Record one Stage A build invocation and return predefined shortlist rows.

        Args:
            grid_context: Prepared Stage A grid context.
            artifact_context: Resolved slot-pinned runtime context.
            target_time_range: Requested trading window.
            shortlist_limit: Stage A shortlist cap.
            ranking: Optional ranking config.
            batch_size: Optional chunk size override.
            cancel_checker: Optional cancellation hook.
            on_checkpoint: Optional checkpoint hook.
        Returns:
            tuple[BacktestStageAScoredVariantV1, ...]: Fixed deterministic shortlist rows.
        Assumptions:
            Fake builder bypasses hot-loop kernels and therefore ignores hooks.
        Raises:
            None.
        Side Effects:
            Appends call metadata to the in-memory log.
        """
        _ = grid_context, batch_size
        self.calls.append(
            {
                "artifact_context": artifact_context,
                "target_time_range": target_time_range,
                "shortlist_limit": shortlist_limit,
                "ranking": ranking,
            }
        )
        if cancel_checker is not None:
            cancel_checker("stage_a")
        if on_checkpoint is not None:
            on_checkpoint(len(self.rows), len(self.rows))
        return self.rows


class _FakeTimelineBuilder:
    """
    Timeline builder stub that must stay unused after R8-01 artifact cutover.
    """

    def build(
        self,
        *,
        market_id: MarketId,
        symbol: Symbol,
        timeframe: Timeframe,
        requested_time_range: TimeRange,
        warmup_bars: int,
    ) -> Any:
        """
        Fail fast when claimed worker path attempts to build a live timeline.

        Args:
            market_id: Requested market identifier.
            symbol: Requested symbol.
            timeframe: Requested timeframe.
            requested_time_range: Requested time range.
            warmup_bars: Warmup bars count.
        Returns:
            Any: Never returns because live timeline build is forbidden in these tests.
        Assumptions:
            Claimed worker execution must use slot-pinned artifact prices instead of ClickHouse.
        Raises:
            AssertionError: Always, because live timeline build is forbidden here.
        Side Effects:
            None.
        """
        _ = market_id, symbol, timeframe, requested_time_range, warmup_bars
        raise AssertionError("job runner must not build live candle timeline in R8-01")


class _NoOpIndicatorCompute:
    """
    Indicator compute placeholder used to satisfy constructor dependency.
    """

    def estimate(self, grid: Any, *, max_variants_guard: int) -> Any:
        """
        Return no-op estimate payload.

        Args:
            grid: Grid payload.
            max_variants_guard: Variants guard.
        Returns:
            Any: Placeholder payload.
        Assumptions:
            Test grid-builder fake bypasses indicator estimate calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = grid, max_variants_guard
        return None

    def compute(self, req: Any) -> Any:
        """
        Fail fast when worker hot path falls back to `IndicatorCompute.compute(...)`.

        Args:
            req: Compute request payload.
        Returns:
            Any: Never returns because compute is forbidden in claimed worker hot path.
        Assumptions:
            Stage A/Stage B runtime is artifact-backed and must not call compute here.
        Raises:
            AssertionError: Always, because worker hot path must not call compute.
        Side Effects:
            None.
        """
        _ = req
        raise AssertionError("job runner must not call IndicatorCompute.compute in R8-01")

    def warmup(self) -> None:
        """
        Execute no-op warmup.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is not relevant for these unit tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _FakeGridContext:
    """
    Minimal staged grid context fixture for Stage-A/Stage-B loops.
    """

    def __init__(
        self,
        *,
        base_variants: tuple[BacktestStageABaseVariant, ...],
        risk_variants: tuple[BacktestRiskVariantV1, ...],
        execution_profile: Any | None = None,
    ) -> None:
        """
        Initialize deterministic staged grid context payload.

        Args:
            base_variants: Stage-A base variants.
            risk_variants: Stage-B risk variants.
        Returns:
            None.
        Assumptions:
            Stage-B total is `len(base_variants) * len(risk_variants)`.
        Raises:
            ValueError: If one fixture array is empty.
        Side Effects:
            None.
        """
        if len(base_variants) == 0:
            raise ValueError("_FakeGridContext requires at least one base variant")
        if len(risk_variants) == 0:
            raise ValueError("_FakeGridContext requires at least one risk variant")
        self._base_variants = base_variants
        self.risk_variants = risk_variants
        self.stage_a_variants_total = len(base_variants)
        self.stage_b_variants_total = len(base_variants) * len(risk_variants)
        self.execution_profile = (
            execution_profile
            if execution_profile is not None
            else _build_fake_execution_profile(
                mode="exact_small",
                stage_b_workers=1,
                parallel_stage_b_enabled=False,
            )
        )

    def iter_stage_a_variants(self) -> tuple[BacktestStageABaseVariant, ...]:
        """
        Return deterministic Stage-A base variants sequence.

        Args:
            None.
        Returns:
            tuple[BacktestStageABaseVariant, ...]: Base variants fixture.
        Assumptions:
            Fixture order is deterministic and controlled by test data.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._base_variants


def _build_fake_execution_profile(
    *,
    mode: str,
    stage_b_workers: int,
    parallel_stage_b_enabled: bool,
) -> Any:
    """
    Build minimal execution-profile fixture with only fields consumed by runtime core tests.

    Args:
        mode: Stable profile mode literal.
        stage_b_workers: Configured Stage B worker count.
        parallel_stage_b_enabled: Whether process-based Stage B is enabled for the profile.
    Returns:
        Any: SimpleNamespace carrying the minimal execution-profile surface used by job-runner
            tests and runtime helpers.
    Assumptions:
        Exact profiles keep shortlist runtime disabled, while hybrid profiles would enable the
        shortlist-specific flags only when a future test explicitly requests them.
    Raises:
        None.
    Side Effects:
        None.
    """
    shortlist_enabled = mode in {"hybrid_conservative", "hybrid_family"}
    return SimpleNamespace(
        mode=mode,
        parallelism=SimpleNamespace(stage_b_workers=stage_b_workers),
        shortlist_config=SimpleNamespace(enabled=shortlist_enabled),
        feature_flags=SimpleNamespace(
            runtime_enabled=True,
            heuristic_shortlist_enabled=shortlist_enabled,
            parallel_stage_b_enabled=parallel_stage_b_enabled,
            family_plugin_enabled=mode == "hybrid_family",
        ),
    )


class _RecordingSharedRuntimePlanner:
    """
    Shared runtime planner stub returning one predefined runtime plan.
    """

    def __init__(self, *, runtime_plan: _FakeGridContext) -> None:
        """
        Initialize planner stub with one deterministic runtime plan.

        Args:
            runtime_plan: Prebuilt runtime plan fixture.
        Returns:
            None.
        Assumptions:
            The runtime plan already encodes the execution profile chosen by the shared planner.
        Raises:
            None.
        Side Effects:
            Stores in-memory planner call logs for boundary assertions.
        """
        self._runtime_plan = runtime_plan
        self.calls: list[dict[str, Any]] = []

    def plan(
        self,
        *,
        template: RunBacktestTemplate,
        candles: Any,
        indicator_compute: Any,
        preselect: int,
        requested_execution_profile_mode: str | None,
        defaults_provider: Any,
        max_variants_per_compute: int,
        max_compute_bytes_total: int,
    ) -> _FakeGridContext:
        """
        Record one shared-planner invocation and return the predefined runtime plan.

        Args:
            template: Run template payload.
            candles: Candle arrays payload.
            indicator_compute: Indicator compute dependency.
            preselect: Stage-A preselect value.
            requested_execution_profile_mode:
                Optional explicit execution profile mode forwarded from persisted job payload.
            defaults_provider: Optional defaults provider.
            max_variants_per_compute: Variants guard.
            max_compute_bytes_total: Memory guard.
        Returns:
            _FakeGridContext: Prebuilt runtime plan fixture.
        Assumptions:
            Guard checks and adaptive policy decisions are out of scope for this fake.
        Raises:
            None.
        Side Effects:
            Appends planner inputs to the in-memory call log.
        """
        self.calls.append(
            {
                "template": template,
                "candles": candles,
                "indicator_compute": indicator_compute,
                "preselect": preselect,
                "requested_execution_profile_mode": requested_execution_profile_mode,
                "defaults_provider": defaults_provider,
                "max_variants_per_compute": max_variants_per_compute,
                "max_compute_bytes_total": max_compute_bytes_total,
            }
        )
        return self._runtime_plan

    def build(
        self,
        *,
        template: RunBacktestTemplate,
        candles: Any,
        indicator_compute: Any,
        preselect: int,
        requested_execution_profile_mode: str | None,
        defaults_provider: Any,
        max_variants_per_compute: int,
        max_compute_bytes_total: int,
    ) -> _FakeGridContext:
        """
        Preserve the legacy planner fake API by forwarding to `plan(...)`.

        Args:
            template: Run template payload.
            candles: Candle arrays payload.
            indicator_compute: Indicator compute dependency.
            preselect: Stage-A preselect value.
            requested_execution_profile_mode:
                Optional explicit execution profile mode forwarded from persisted job payload.
            defaults_provider: Optional defaults provider.
            max_variants_per_compute: Variants guard.
            max_compute_bytes_total: Memory guard.
        Returns:
            _FakeGridContext: Prebuilt runtime plan fixture.
        Assumptions:
            Some tests may still call `build(...)` while the worker now calls `plan(...)`.
        Raises:
            None.
        Side Effects:
            Delegates to `plan(...)`, which records planner inputs.
        """
        return self.plan(
            template=template,
            candles=candles,
            indicator_compute=indicator_compute,
            preselect=preselect,
            requested_execution_profile_mode=requested_execution_profile_mode,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        )


class _FakePriceArraysLoader:
    """
    Artifact price loader fake returning deterministic request-timeframe arrays.
    """

    def __init__(self) -> None:
        """
        Initialize deterministic artifact prices fixture and in-memory call log.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Worker tests need only one request-timeframe price family for warmup slicing.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory call log.
        """
        self.calls: list[dict[str, Any]] = []
        self._open_time = np.asarray(
            [
                int(_utc(2026, 1, 31, 23, 58, 0).timestamp() * 1000),
                int(_utc(2026, 1, 31, 23, 59, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 0, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 1, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 2, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 3, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 4, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 5, 0).timestamp() * 1000),
            ],
            dtype=np.int64,
        )
        self._close_time = np.asarray(
            [
                int(_utc(2026, 1, 31, 23, 59, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 0, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 1, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 2, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 3, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 4, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 5, 0).timestamp() * 1000),
                int(_utc(2026, 2, 1, 0, 6, 0).timestamp() * 1000),
            ],
            dtype=np.int64,
        )
        self._ohlcv = np.asarray(
            [
                [10.0, 11.0, 9.0, 10.5, 1.0],
                [10.5, 11.5, 10.0, 11.0, 1.0],
                [11.0, 12.0, 10.5, 11.5, 1.0],
                [11.5, 12.5, 11.0, 12.0, 1.0],
                [12.0, 13.0, 11.5, 12.5, 1.0],
                [12.5, 13.5, 12.0, 13.0, 1.0],
                [13.0, 14.0, 12.5, 13.5, 1.0],
                [13.5, 14.5, 13.0, 14.0, 1.0],
            ],
            dtype=np.float32,
        )

    def load_price_arrays(self, *, context: Any, timeframe: str) -> Any:
        """
        Return deterministic artifact prices payload for worker warmup slicing.

        Args:
            context: Slot-pinned runtime context forwarded by the use-case.
            timeframe: Requested price timeframe literal.
        Returns:
            Any: Minimal artifact price payload with `open_time/close_time/ohlcv`.
        Assumptions:
            Tests exercise request timeframe only and do not need mapping/hit-times loaders here.
        Raises:
            None.
        Side Effects:
            Appends call metadata to the in-memory log.
        """
        self.calls.append({"context": context, "timeframe": timeframe})
        return SimpleNamespace(
            open_time=self._open_time,
            close_time=self._close_time,
            ohlcv=self._ohlcv,
        )


class _ArtifactOnlyStageAShortlistBuilder:
    """
    Deterministic artifact-only Stage A builder fake for claimed worker tests.
    """

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
        Build deterministic shortlist rows directly from Stage-A base variants fixture order.

        Args:
            grid_context: Prepared Stage A grid context.
            artifact_context: Resolved slot-pinned runtime context.
            target_time_range: Requested trading window.
            shortlist_limit: Maximum number of rows to retain.
            ranking: Optional ranking config.
            batch_size: Optional chunk size.
            cancel_checker: Optional cooperative cancellation hook.
            on_checkpoint: Optional checkpoint hook.
        Returns:
            tuple[BacktestStageAScoredVariantV1, ...]: Deterministic shortlist rows.
        Assumptions:
            Tests focus on worker orchestration and do not need real Stage A kernel economics.
        Raises:
            None.
        Side Effects:
            Invokes provided cancellation and checkpoint hooks.
        """
        _ = artifact_context, target_time_range, ranking, batch_size
        if cancel_checker is not None:
            cancel_checker("stage_a")
        base_variants = tuple(grid_context.iter_stage_a_variants())[:shortlist_limit]
        rows = tuple(
            BacktestStageAScoredVariantV1(
                base_variant=base_variant,
                total_return_pct=float(index + 1),
            )
            for index, base_variant in enumerate(base_variants)
        )
        if on_checkpoint is not None:
            on_checkpoint(len(rows), int(grid_context.stage_a_variants_total))
        return rows


class _DeterministicScorerWithDetails:
    """
    Deterministic scorer fake for Stage-A/Stage-B and finalizing details calls.
    """

    def __init__(self, *, stage_a_scores: Mapping[str, float]) -> None:
        """
        Initialize scorer with explicit Stage-A score mapping.

        Args:
            stage_a_scores: Mapping `base_variant_key -> total_return_pct`.
        Returns:
            None.
        Assumptions:
            Stage-B scores use one fixed value to keep ranking tie-break deterministic.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._stage_a_scores = dict(stage_a_scores)
        self._stage_b_score = 7.0

    def score_variant(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic ranking metric payload for Stage-A and Stage-B.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Metric payload with `Total Return [%]`.
        Assumptions:
            Stage-B deterministic tie-break is handled by variant-key sorting.
        Raises:
            ValueError: If Stage-A score mapping is missing requested variant key.
        Side Effects:
            None.
        """
        _ = candles, indicator_selections, signal_params, risk_params, indicator_variant_key
        if stage == "stage_a":
            return {"Total Return [%]": float(self._stage_a_scores[variant_key])}
        return {"Total Return [%]": self._stage_b_score}

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Any:
        """
        Return minimal details payload used by finalizing step.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Any: Details-like object with deterministic metrics payload.
        Assumptions:
            Reporting service fake ignores execution/risk payload structure.
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
        return SimpleNamespace(
            metrics={"Total Return [%]": self._stage_b_score},
            target_slice=slice(0, 1),
            execution_params={},
            risk_params={},
            execution_outcome={},
        )

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return metric-only payload used by staged-core ranking loops.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Deterministic ranking metrics payload.
        Assumptions:
            Metric-only path reuses deterministic values from `score_variant`.
        Raises:
            ValueError: Propagates Stage-A missing score mapping errors.
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


class _RankingAwareScorerWithDetails:
    """
    Deterministic scorer fake exposing configurable ranking metrics for jobs ordering tests.
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic multi-metric payload for Stage-A/Stage-B ranking checks.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Payload including primary/secondary metric literals.
        Assumptions:
            Stage-A fixtures use `ema.threshold` signal parameter to derive primary ranking value.
        Raises:
            ValueError: If fixture payload shape is invalid.
        Side Effects:
            None.
        """
        _ = stage, candles, indicator_selections, indicator_variant_key, variant_key
        ema_signal = signal_params.get("ema", {})
        threshold_raw = ema_signal.get("threshold", 0)
        primary = float(threshold_raw) if isinstance(threshold_raw, int | float) else 0.0
        sl_pct_raw = risk_params.get("sl_pct")
        secondary = (
            float(sl_pct_raw)
            if isinstance(sl_pct_raw, int | float) and not isinstance(sl_pct_raw, bool)
            else 0.0
        )
        total_return = 100.0 - primary
        win_rate_pct = 100.0 - (secondary * 10.0)
        sharpe_trades = total_return + (1.0 - secondary)
        return {
            "Total Return [%]": total_return,
            "total_return_pct": total_return,
            "max_drawdown_pct": primary,
            "profit_factor": secondary,
            "sharpe_trades": sharpe_trades,
            "win_rate_pct": win_rate_pct,
            "return_over_max_drawdown": (
                total_return / primary if primary != 0.0 else float("inf")
            ),
        }

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Any:
        """
        Return minimal details payload for finalizing path while preserving total-return checks.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Any: Details-like object with deterministic metrics payload.
        Assumptions:
            Finalizing consistency check uses `Total Return [%]` only.
        Raises:
            None.
        Side Effects:
            None.
        """
        metrics = self.score_variant(
            stage=stage,
            candles=candles,
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            indicator_variant_key=indicator_variant_key,
            variant_key=variant_key,
        )
        return SimpleNamespace(
            metrics=metrics,
            target_slice=slice(0, 1),
            execution_params={},
            risk_params={},
            execution_outcome={},
        )

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return metric-only payload for configurable ranking tests in staged-core loops.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Deterministic multi-metric payload.
        Assumptions:
            Metric-only path is equivalent to `score_variant` for this fake.
        Raises:
            None.
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


class _FrontierStableScorer:
    """
    Deterministic scorer fake that stabilizes top-1 Stage-B frontier after first checkpoint.
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic ranking metrics where first Stage-B candidate remains best.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Ranking payload with deterministic `Total Return [%]`.
        Assumptions:
            Stage-B task order follows shortlist order and risk-variant order.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = candles, indicator_selections, indicator_variant_key, variant_key
        threshold_raw = signal_params.get("ema", {}).get("threshold", 0)
        threshold = int(threshold_raw) if isinstance(threshold_raw, int | float) else 0
        if stage == "stage_a":
            return {"Total Return [%]": float(100 - threshold)}

        sl_enabled = risk_params.get("sl_enabled") is True
        if threshold == 1 and not sl_enabled:
            return {"Total Return [%]": 50.0}
        return {"Total Return [%]": 1.0}

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: Any,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, Any]],
        risk_params: Mapping[str, Any],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return metric-only payload used by artifact-backed Stage B ranking loop checkpoints.

        Args:
            stage: Stage literal.
            candles: Candle arrays payload.
            indicator_selections: Indicator selections.
            signal_params: Signal parameters mapping.
            risk_params: Risk payload mapping.
            indicator_variant_key: Indicators-only variant key.
            variant_key: Backtest variant key.
        Returns:
            Mapping[str, float]: Deterministic ranking payload.
        Assumptions:
            Metric-only path is equivalent to `score_variant` for this fake.
        Raises:
            None.
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


class _ParallelRankingAwareScorerWithDetails(_RankingAwareScorerWithDetails):
    """
    Ranking-aware scorer fake exposing spawned-worker snapshot hooks for exact-parallel tests.
    """

    def to_parallel_stage_b_worker_snapshot_v2(self) -> Mapping[str, str]:
        """
        Return trivial picklable scorer snapshot for spawned-worker rehydration tests.

        Args:
            None.
        Returns:
            Mapping[str, str]: Minimal snapshot payload.
        Assumptions:
            This scorer is stateless, so worker bootstrap needs only a sentinel payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"scorer": "parallel-ranking-aware"}

    @classmethod
    def from_parallel_stage_b_worker_snapshot_v2(
        cls,
        *,
        snapshot: Mapping[str, str],
    ) -> _ParallelRankingAwareScorerWithDetails:
        """
        Rehydrate deterministic scorer fake from spawned-worker snapshot payload.

        Args:
            snapshot: Minimal snapshot payload.
        Returns:
            _ParallelRankingAwareScorerWithDetails: Rehydrated scorer fake.
        Assumptions:
            Snapshot contents are fixed and validated only lightly for these unit tests.
        Raises:
            ValueError: If snapshot drift is detected.
        Side Effects:
            None.
        """
        if snapshot.get("scorer") != "parallel-ranking-aware":
            raise ValueError("unexpected parallel scorer snapshot")
        return cls()


@dataclass(frozen=True, slots=True)
class _ImmediateParallelFutureV2:
    """
    Already-resolved future stub carrying one chunk index for deterministic wait-order tests.
    """

    value: Any
    chunk_index: int

    def result(self) -> Any:
        """
        Return stored future result value.

        Args:
            None.
        Returns:
            Any: Precomputed future result payload.
        Assumptions:
            Fake executor executes submitted work synchronously before exposing the future.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.value


class _FakeProcessPoolExecutorV2:
    """
    Synchronous process-pool stub capturing exact-parallel Stage B bootstrap and chunk order.
    """

    initializer_bootstraps: list[Any] = []
    submitted_chunk_indexes: list[int] = []

    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: object,
        initializer: Any | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        """
        Store fake process-pool constructor arguments for deterministic test assertions.

        Args:
            max_workers: Requested worker count, unused beyond interface compatibility.
            mp_context: Requested multiprocessing context, unused in the fake executor.
            initializer: Optional worker initializer callable.
            initargs: Optional initializer arguments.
        Returns:
            None.
        Assumptions:
            Unit tests only need interface compatibility and bootstrap capture.
        Raises:
            None.
        Side Effects:
            Stores initializer state in memory for later synchronous execution.
        """
        del max_workers, mp_context
        self._initializer = initializer
        self._initargs = initargs

    def __enter__(self) -> _FakeProcessPoolExecutorV2:
        """
        Run captured initializer once and return fake executor instance.

        Args:
            None.
        Returns:
            _FakeProcessPoolExecutorV2: This fake executor instance.
        Assumptions:
            One shared bootstrap event is sufficient for the synchronous in-process fake.
        Raises:
            Exception: Propagates initializer failures unchanged.
        Side Effects:
            Captures the worker bootstrap payload and initializes worker-local state in-process.
        """
        if self._initializer is not None:
            _FakeProcessPoolExecutorV2.initializer_bootstraps.append(self._initargs[0])
            self._initializer(*self._initargs)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        """
        Propagate exceptions from the fake executor context body unchanged.

        Args:
            exc_type: Exception type raised inside the context body, if any.
            exc: Exception instance raised inside the context body, if any.
            exc_tb: Traceback raised inside the context body, if any.
        Returns:
            bool: Always `False` so failures remain visible to the caller.
        Assumptions:
            Fake executor should mirror real context-manager propagation behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        del exc_type, exc, exc_tb
        return False

    def submit(
        self,
        fn: Any,
        /,
        *args: object,
        **kwargs: object,
    ) -> _ImmediateParallelFutureV2:
        """
        Execute submitted chunk synchronously and expose deterministic chunk metadata.

        Args:
            fn: Submitted callable.
            *args: Positional callable arguments.
            **kwargs: Keyword callable arguments.
        Returns:
            _ImmediateParallelFutureV2: Already-resolved future carrying the chunk index.
        Assumptions:
            Exact-parallel unit tests need deterministic merge-order simulation, not real IPC.
        Raises:
            Exception: Propagates callable failures unchanged.
        Side Effects:
            Captures submitted chunk indexes in memory.
        """
        chunk = cast(Any, kwargs["chunk"])
        _FakeProcessPoolExecutorV2.submitted_chunk_indexes.append(int(chunk.chunk_index))
        return _ImmediateParallelFutureV2(
            value=fn(*args, **kwargs),
            chunk_index=int(chunk.chunk_index),
        )


def _wait_highest_chunk_first_v2(
    futures: tuple[_ImmediateParallelFutureV2, ...],
    *,
    return_when: object,
) -> tuple[set[_ImmediateParallelFutureV2], set[_ImmediateParallelFutureV2]]:
    """
    Return one completed fake future at a time in reverse chunk order.

    Args:
        futures: Pending fake futures supplied by runtime core.
        return_when: Ignored wait policy marker kept for signature compatibility.
    Returns:
        tuple[set[_ImmediateParallelFutureV2], set[_ImmediateParallelFutureV2]]:
            Completed subset and remaining futures.
    Assumptions:
        Reverse completion order stresses coordinator-side deterministic chunk merge buffering.
    Raises:
        ValueError: If called with an empty futures tuple.
    Side Effects:
        None.
    """
    del return_when
    if len(futures) == 0:
        raise ValueError("wait helper requires at least one fake future")
    completed_future = max(futures, key=lambda future: future.chunk_index)
    remaining = {future for future in futures if future is not completed_future}
    return {completed_future}, remaining


class _FakeReportingService:
    """
    Reporting service fake recording include-trades policy decisions.
    """

    def __init__(self) -> None:
        """
        Initialize reporting service fake with empty calls log.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `build_report` is called once per finalized persisted variant.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.calls: list[dict[str, Any]] = []

    def build_report(
        self,
        *,
        requested_time_range: TimeRange,
        candles: Any,
        target_slice: slice,
        execution_params: Any,
        execution_outcome: Any,
        include_table_md: bool,
        include_trades: bool,
    ) -> Any:
        """
        Return deterministic report payload with optional trades fixture.

        Args:
            requested_time_range: Requested time range.
            candles: Candle arrays payload.
            target_slice: Reporting target slice.
            execution_params: Execution params payload.
            execution_outcome: Execution outcome payload.
            include_table_md: Include markdown table flag.
            include_trades: Include trades payload flag.
        Returns:
            Any: Report-like object with `table_md` and `trades` fields.
        Assumptions:
            Finalizing requires non-empty markdown table for persisted variants.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory calls log.
        """
        _ = requested_time_range, candles, target_slice, execution_params, execution_outcome
        self.calls.append(
            {
                "include_table_md": include_table_md,
                "include_trades": include_trades,
            }
        )
        return SimpleNamespace(
            table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|7.00|",
            trades=(_sample_trade(),) if include_trades else None,
        )


class _FakeJobRepository:
    """
    Job repository fake for deterministic cancel polling behavior.
    """

    def __init__(
        self,
        *,
        default_job: BacktestJob,
        scripted_get_results: tuple[BacktestJob | None, ...] = (),
    ) -> None:
        """
        Initialize fake repository with scripted `get` responses.

        Args:
            default_job: Fallback job payload for unscripted reads.
            scripted_get_results: Optional queued `get` responses.
        Returns:
            None.
        Assumptions:
            Worker use-case reads only `get(job_id=...)` for cancel checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._default_job = default_job
        self._scripted_get_results = list(scripted_get_results)

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Return scripted or default job snapshot for cancel checks.

        Args:
            job_id: Job identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Job snapshot payload.
        Assumptions:
            `job_id` always matches configured test fixture id.
        Raises:
            None.
        Side Effects:
            Pops one scripted response when queue is non-empty.
        """
        _ = job_id, user_id
        if self._scripted_get_results:
            return self._scripted_get_results.pop(0)
        return self._default_job

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return deterministic zero blocking pins for worker cancel-polling tests.

        Args:
            market_id: Requested market id.
            symbol: Requested symbol.
            artifact_slot: Candidate slot literal.
            artifact_manifest_hash: Candidate manifest hash.
        Returns:
            int: Always `0` because worker tests do not exercise publish guard queries.
        Assumptions:
            Worker job-repository fake is used only for `get(...)` cancel checks in this test
            module.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


class _FakeLeaseRepository:
    """
    Lease repository fake recording progress/heartbeat/finish calls.
    """

    def __init__(self) -> None:
        """
        Initialize fake lease repository with empty call logs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Call order is deterministic for one claimed attempt.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.claim_next_calls: list[dict[str, Any]] = []
        self.update_progress_calls: list[dict[str, Any]] = []
        self.heartbeat_calls: list[dict[str, Any]] = []
        self.finish_calls: list[dict[str, Any]] = []

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestJob | None:
        """
        Return no claimed jobs (unused in use-case unit tests).

        Args:
            now: Claim timestamp.
            locked_by: Worker owner id.
            lease_seconds: Lease TTL.
        Returns:
            BacktestJob | None: Always `None`.
        Assumptions:
            Claim loop is out of scope for these use-case tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.claim_next_calls.append(
            {
                "now": now,
                "locked_by": locked_by,
                "lease_seconds": lease_seconds,
            }
        )
        return None

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> Any:
        """
        Record heartbeat call and return non-null payload.

        Args:
            job_id: Job id.
            now: Heartbeat timestamp.
            locked_by: Worker owner id.
            lease_seconds: Lease TTL.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful heartbeat is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.heartbeat_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "lease_seconds": lease_seconds,
            }
        )
        return object()

    def update_progress(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        stage: str,
        processed_units: int,
        total_units: int,
    ) -> Any:
        """
        Record progress call and return non-null payload.

        Args:
            job_id: Job id.
            now: Progress timestamp.
            locked_by: Worker owner id.
            stage: Stage literal.
            processed_units: Processed units.
            total_units: Total units.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful progress write is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.update_progress_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "stage": stage,
                "processed_units": processed_units,
                "total_units": total_units,
            }
        )
        return object()

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: str,
        last_error: str | None = None,
        last_error_json: Any = None,
    ) -> Any:
        """
        Record finish call and return non-null payload.

        Args:
            job_id: Job id.
            now: Finish timestamp.
            locked_by: Worker owner id.
            next_state: Target terminal state.
            last_error: Optional short error text.
            last_error_json: Optional structured error payload.
        Returns:
            Any: Non-null payload.
        Assumptions:
            Successful finish write is represented by non-null return value.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.finish_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "next_state": next_state,
                "last_error": last_error,
                "last_error_json": last_error_json,
            }
        )
        return object()


class _FakeResultsRepository:
    """
    Results repository fake with optional lease-loss simulation on snapshot writes.
    """

    def __init__(self, *, fail_replace_call_numbers: tuple[int, ...] = ()) -> None:
        """
        Initialize results repository fake.

        Args:
            fail_replace_call_numbers: 1-based replace call numbers returning lease lost.
        Returns:
            None.
        Assumptions:
            Lease-lost is represented by `False` from replace method.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._fail_replace_call_numbers = set(fail_replace_call_numbers)
        self.replace_calls: list[dict[str, Any]] = []
        self.shortlist_calls: list[dict[str, Any]] = []

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[Any, ...],
    ) -> bool:
        """
        Record snapshot replace call and optionally simulate lease loss.

        Args:
            job_id: Job id.
            now: Snapshot timestamp.
            locked_by: Worker owner id.
            rows: Snapshot rows payload.
        Returns:
            bool: `False` when configured call number simulates lease loss.
        Assumptions:
            One call can represent running snapshot or finalizing snapshot write.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.replace_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "rows": rows,
            }
        )
        call_number = len(self.replace_calls)
        return call_number not in self._fail_replace_call_numbers

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: Any,
    ) -> bool:
        """
        Record Stage-A shortlist save call and report success.

        Args:
            job_id: Job id.
            now: Save timestamp.
            locked_by: Worker owner id.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: Always `True`.
        Assumptions:
            Lease-loss simulation for shortlist save is not required in these tests.
        Raises:
            None.
        Side Effects:
            Appends call payload to in-memory log.
        """
        self.shortlist_calls.append(
            {
                "job_id": job_id,
                "now": now,
                "locked_by": locked_by,
                "shortlist": shortlist,
            }
        )
        return True


@dataclass
class _NowProvider:
    """
    Monotonic UTC now-provider fixture for deterministic use-case tests.
    """

    current: datetime
    step_seconds: int = 1

    def __call__(self) -> datetime:
        """
        Return current timestamp and advance internal cursor by fixed step.

        Args:
            None.
        Returns:
            datetime: Current UTC timestamp.
        Assumptions:
            Fixed step is positive and small enough for tests.
        Raises:
            None.
        Side Effects:
            Mutates internal timestamp cursor.
        """
        now = self.current
        self.current = now + timedelta(seconds=self.step_seconds)
        return now


def test_run_backtest_job_runner_v1_requires_shared_runtime_planner() -> None:
    """
    Verify worker use-case construction fails without an injected shared runtime planner.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Worker wiring is the canonical owner of startup-loaded planner injection, so the
        use-case must not silently fall back to local default `execution_profiles` or
        `adaptive_selector_policy`.
    Raises:
        AssertionError: If constructor allows a missing shared planner.
    Side Effects:
        None.
    """
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)

    with pytest.raises(ValueError, match="shared runtime_planner"):
        RunBacktestJobRunnerV1(
            job_repository=cast(Any, _FakeJobRepository(default_job=_build_running_job())),
            lease_repository=cast(Any, _FakeLeaseRepository()),
            results_repository=cast(Any, _FakeResultsRepository()),
            request_decoder=cast(Any, _FakeRequestDecoder(request=request)),
            indicator_compute=cast(Any, _NoOpIndicatorCompute()),
            runtime_planner=None,
            artifact_slot_resolver=cast(
                Any,
                _RecordingArtifactSlotResolver(context=_default_pinned_context()),
            ),
        )


def test_process_claimed_job_persists_stage_progress_and_finalizing_policy() -> None:
    """
    Verify succeeded flow writes stage progress semantics without eager finalizing details.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Persisted cap is `min(top_k, top_k_persisted_default)` and finalizing avoids report/trades.
    Raises:
        AssertionError: If stage progress or finalizing snapshot policy is violated.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 1.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    reporting_service = _FakeReportingService()
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=reporting_service,
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 0, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert lease_repository.finish_calls[-1]["next_state"] == "succeeded"
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_a",
        processed_units=0,
        total_units=2,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_b",
        processed_units=0,
        total_units=4,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="stage_b",
        processed_units=4,
        total_units=4,
    )
    assert _has_progress_call(
        calls=lease_repository.update_progress_calls,
        stage="finalizing",
        processed_units=0,
        total_units=1,
    )

    running_rows = results_repository.replace_calls[0]["rows"]
    assert all(row.report_table_md is None for row in running_rows)
    assert all(row.trades_json is None for row in running_rows)
    assert all(
        row.summary_metrics_json["total_return_pct"] == row.total_return_pct
        for row in running_rows
    )
    assert tuple(row.best_tp_pct for row in running_rows) == tuple(
        (
            float(row.payload_json["risk_params"]["tp_pct"])
            if row.payload_json["risk_params"]["tp_enabled"] is True
            else None
        )
        for row in running_rows
    )
    assert tuple(row.best_sl_pct for row in running_rows) == tuple(
        (
            float(row.payload_json["risk_params"]["sl_pct"])
            if row.payload_json["risk_params"]["sl_enabled"] is True
            else None
        )
        for row in running_rows
    )

    assert len(results_repository.replace_calls) == 1
    assert reporting_service.calls == []


def test_process_claimed_job_does_not_claim_additional_jobs() -> None:
    """
    Verify the use case handles only the provided claimed job and never calls `claim_next(...)`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Claiming the next queued job belongs to the outer single claim loop, not this use case.
    Raises:
        AssertionError: If the use case starts its own claim loop or changes `locked_by`.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                base_variants[0].base_variant_key: 1.0,
                base_variants[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 0, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by=" worker-test-1 ")

    observed_locked_by = {
        call["locked_by"]
        for call in (
            lease_repository.update_progress_calls
            + lease_repository.heartbeat_calls
            + lease_repository.finish_calls
            + results_repository.replace_calls
            + results_repository.shortlist_calls
        )
    }

    assert report.status == "succeeded"
    assert lease_repository.claim_next_calls == []
    assert observed_locked_by == {"worker-test-1"}


def test_process_claimed_job_applies_configured_primary_and_secondary_ranking() -> None:
    """
    Verify worker ignores compatibility-only `secondary_metric` and keeps deterministic ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Single-metric ranking now uses the primary metric plus deterministic tie-break only.
    Raises:
        AssertionError: If compatibility-only secondary metric still perturbs worker ordering.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(
        top_k=4,
        preselect=2,
        top_trades_n=1,
        ranking=BacktestRankingConfig(
            primary_metric="max_drawdown_pct",
            secondary_metric="profit_factor",
        ),
    )
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=_RankingAwareScorerWithDetails(),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=4,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 5, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    running_rows = results_repository.replace_calls[0]["rows"]
    assert tuple(
        int(row.payload_json["signal_params"]["ema"]["threshold"]) for row in running_rows
    ) == (1, 1, 2, 2)
    assert tuple(
        float(row.payload_json["risk_params"]["sl_pct"] or 0.0) for row in running_rows
    ) == (0.0, 1.0, 1.0, 0.0)


def test_process_claimed_job_applies_secondary_win_rate_ordering() -> None:
    """
    Verify alternate compatibility-only secondary metrics leave worker ordering unchanged.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Single-metric worker ordering is now independent from deprecated secondary literals.
    Raises:
        AssertionError: If deprecated secondary literals still change ranked snapshot order.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(
        top_k=4,
        preselect=2,
        top_trades_n=1,
        ranking=BacktestRankingConfig(
            primary_metric="max_drawdown_pct",
            secondary_metric="win_rate_pct",
        ),
    )
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=_RankingAwareScorerWithDetails(),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=4,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 6, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    running_rows = results_repository.replace_calls[0]["rows"]
    assert tuple(
        int(row.payload_json["signal_params"]["ema"]["threshold"]) for row in running_rows
    ) == (1, 1, 2, 2)
    assert tuple(
        float(row.payload_json["risk_params"]["sl_pct"] or 0.0) for row in running_rows
    ) == (0.0, 1.0, 1.0, 0.0)


def test_process_claimed_job_cancels_on_batch_boundary() -> None:
    """
    Verify cancel detection on batch boundary transitions job to `cancelled` and stops writes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cancel is checked before next batch and before finalizing.
    Raises:
        AssertionError: If cancel flow writes extra snapshots or misses terminal transition.
    Side Effects:
        None.
    """
    job = _build_running_job()
    cancelled_job = job.request_cancel(changed_at=job.updated_at + timedelta(seconds=1))
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 2.0,
            base_variants[1].base_variant_key: 1.0,
        }
    )
    job_repository = _FakeJobRepository(
        default_job=cancelled_job,
        scripted_get_results=(job, cancelled_job),
    )
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 10, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "cancelled"
    assert len(lease_repository.finish_calls) == 1
    assert lease_repository.finish_calls[0]["next_state"] == "cancelled"
    assert results_repository.shortlist_calls == []
    assert results_repository.replace_calls == []


def test_process_claimed_job_stops_when_lease_lost_during_snapshot_write() -> None:
    """
    Verify lease-lost snapshot write aborts processing immediately without terminal finish write.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Lease-lost is surfaced as `False` from results repository replace call.
    Raises:
        AssertionError: If worker continues writing after lease-lost condition.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 2.0,
            base_variants[1].base_variant_key: 1.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository(fail_replace_call_numbers=(1,))
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 20, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "lease_lost"
    assert len(results_repository.replace_calls) == 1
    assert lease_repository.finish_calls == []


def test_process_claimed_job_persists_running_snapshots_by_time_trigger() -> None:
    """
    Verify running snapshots are persisted when `snapshot_seconds` threshold is reached.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variants-step trigger is effectively disabled by large `snapshot_variants_step`.
    Raises:
        AssertionError: If time trigger does not persist running snapshots.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 3.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=1,
        snapshot_variants_step=10_000,
        stage_batch_size=1,
        now_provider=_NowProvider(
            current=_utc(2026, 2, 23, 10, 30, 0),
            step_seconds=2,
        ),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    running_snapshots = [
        call
        for call in results_repository.replace_calls
        if all(row.report_table_md is None for row in call["rows"])
    ]
    assert report.status == "succeeded"
    assert len(running_snapshots) >= 2


def test_process_claimed_job_persists_running_snapshots_by_variants_step() -> None:
    """
    Verify running snapshots are persisted when `snapshot_variants_step` threshold is reached.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Time trigger is effectively disabled by large `snapshot_seconds` value.
    Raises:
        AssertionError: If running snapshots are not persisted incrementally.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    scorer = _DeterministicScorerWithDetails(
        stage_a_scores={
            base_variants[0].base_variant_key: 3.0,
            base_variants[1].base_variant_key: 2.0,
        }
    )
    job_repository = _FakeJobRepository(default_job=job)
    lease_repository = _FakeLeaseRepository()
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
        ),
        scorer=scorer,
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 30, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    running_snapshots = [
        call
        for call in results_repository.replace_calls
        if all(row.report_table_md is None for row in call["rows"])
    ]
    assert report.status == "succeeded"
    assert len(running_snapshots) >= 2


def test_process_claimed_job_skips_snapshot_replace_when_frontier_signature_unchanged() -> None:
    """
    Verify cadence checkpoints skip `replace_top_variants_snapshot`
    when frontier signature is stable.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Top-1 frontier becomes stable after first Stage-B task with configured scorer fixture.
    Raises:
        AssertionError: If unchanged frontier still triggers redundant snapshot replaces.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_FrontierStableScorer(),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=1,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 40, 0)),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    running_snapshots = [
        call
        for call in results_repository.replace_calls
        if all(row.report_table_md is None for row in call["rows"])
    ]
    assert report.status == "succeeded"
    assert len(running_snapshots) == 1


def test_process_claimed_job_exact_parallel_matches_serial_stage_b_frontiers(
    monkeypatch: Any,
) -> None:
    """
    Verify exact-parallel Stage B keeps the same persisted frontier sequence as serial exact mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime core merges parallel chunk results in canonical chunk order even when worker
        completion arrives in reverse order.
    Raises:
        AssertionError: If exact-parallel changes persisted frontier ordering or skips process
            worker bootstrap.
    Side Effects:
        Monkeypatches Stage B process-pool helpers inside runtime core for deterministic coverage.
    """
    _FakeProcessPoolExecutorV2.initializer_bootstraps = []
    _FakeProcessPoolExecutorV2.submitted_chunk_indexes = []
    monkeypatch.setattr(
        artifact_runtime_core_module,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutorV2,
    )
    monkeypatch.setattr(
        artifact_runtime_core_module,
        "wait",
        _wait_highest_chunk_first_v2,
    )

    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    base_variants = _build_stage_a_variants()
    risk_variants = _build_risk_variants()
    serial_results_repository = _FakeResultsRepository()
    parallel_results_repository = _FakeResultsRepository()

    serial_use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=_build_running_job()),
        lease_repository=_FakeLeaseRepository(),
        results_repository=serial_results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
            execution_profile=_build_fake_execution_profile(
                mode="exact_small",
                stage_b_workers=1,
                parallel_stage_b_enabled=False,
            ),
        ),
        scorer=_ParallelRankingAwareScorerWithDetails(),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=2,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 41, 0)),
    )
    parallel_use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=_build_running_job()),
        lease_repository=_FakeLeaseRepository(),
        results_repository=parallel_results_repository,
        grid_context=_FakeGridContext(
            base_variants=base_variants,
            risk_variants=risk_variants,
            execution_profile=_build_fake_execution_profile(
                mode="exact_parallel",
                stage_b_workers=2,
                parallel_stage_b_enabled=True,
            ),
        ),
        scorer=_ParallelRankingAwareScorerWithDetails(),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=10_000,
        snapshot_variants_step=1,
        stage_batch_size=2,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 42, 0)),
    )

    serial_report = serial_use_case.process_claimed_job(
        job=_build_running_job(),
        locked_by="worker-test-1",
    )
    parallel_report = parallel_use_case.process_claimed_job(
        job=_build_running_job(),
        locked_by="worker-test-1",
    )

    serial_frontiers = [
        tuple(row.variant_key for row in call["rows"])
        for call in serial_results_repository.replace_calls
    ]
    parallel_frontiers = [
        tuple(row.variant_key for row in call["rows"])
        for call in parallel_results_repository.replace_calls
    ]

    assert serial_report.status == "succeeded"
    assert parallel_report.status == "succeeded"
    assert parallel_frontiers == serial_frontiers
    assert len(_FakeProcessPoolExecutorV2.initializer_bootstraps) == 1
    assert _FakeProcessPoolExecutorV2.submitted_chunk_indexes == [0, 1]


def test_process_claimed_job_builds_runtime_candles_from_pinned_artifact_prices() -> None:
    """
    Verify claimed worker runtime loads pinned `prices/<tf>` artifacts instead of live timeline IO.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `_FakeTimelineBuilder` and `_NoOpIndicatorCompute.compute(...)` already fail if the worker
        attempts legacy hot-path behavior.
    Raises:
        AssertionError: If artifact price loader is not used for runtime candle bootstrap.
    Side Effects:
        None.
    """
    job = _build_running_job()
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    assert request.template is not None
    price_arrays_loader = _FakePriceArraysLoader()
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 42, 0)),
        price_arrays_loader=price_arrays_loader,
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert len(price_arrays_loader.calls) == 1
    assert price_arrays_loader.calls[0]["context"] == _default_pinned_context()
    assert price_arrays_loader.calls[0]["timeframe"] == request.template.timeframe.code


def test_resolve_request_context_derives_internal_warmup_without_public_field() -> None:
    """
    Verify claimed worker derives internal warmup from effective indicator requirements.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public request decoding no longer supplies `warmup_bars`, so the worker must derive it
        from the resolved template.
    Raises:
        AssertionError: If derived warmup no longer follows the indicator window requirement.
    Side Effects:
        None.
    """
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    job = _build_running_job()
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                variant.base_variant_key: float(index + 1)
                for index, variant in enumerate(_build_stage_a_variants())
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 42, 0)),
    )

    context = use_case._resolve_request_context(job=job)

    assert context.request.warmup_bars is None
    assert context.warmup_bars == 10


@pytest.mark.parametrize(
    ("execution_mode",),
    (
        ("background_auto",),
        ("background_manual_legacy",),
    ),
)
def test_process_claimed_job_bootstraps_pinned_slot_context_before_runtime(
    execution_mode: BacktestJobExecutionMode,
) -> None:
    """
    Verify background use-case resolves the shared slot-pinned context before runtime work starts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R6-01 bootstrap is additive here and should not change staged job processing behavior.
    Raises:
        AssertionError: If background bootstrap coordinates or persisted pin fields drift.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    job = _build_running_job_with_artifact_pin(execution_mode=execution_mode)
    resolver = _RecordingArtifactSlotResolver(
        context=_FakeSlotPinnedContext(
            coordinates=ArtifactCoordinatesV2(
                exchange="binance",
                market_type="spot",
                symbol="BTCUSDT",
            ),
            artifact_slot="slot_b",
            slot_generation=11,
            artifact_asof_date="2026-03-29",
            artifact_manifest_hash="d" * 64,
        )
    )
    use_case = _build_use_case(
        request=_build_request(top_k=5, preselect=2, top_trades_n=1),
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 45, 0)),
        artifact_slot_resolver=resolver,
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert len(resolver.pinned_calls) == 1
    coordinates, pinned_identity = resolver.pinned_calls[0]
    assert coordinates == ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    assert pinned_identity.artifact_slot == "slot_b"
    assert pinned_identity.slot_generation == 11
    assert pinned_identity.artifact_asof_date == "2026-03-29"
    assert pinned_identity.artifact_manifest_hash == "d" * 64
    assert job.execution_mode == execution_mode


@pytest.mark.parametrize(
    ("execution_mode",),
    (
        ("background_auto",),
        ("background_manual_legacy",),
    ),
)
def test_process_claimed_job_forwards_internal_profile_override_to_shared_planner(
    execution_mode: BacktestJobExecutionMode,
) -> None:
    """
    Verify claimed worker execution delegates internal profile selection to the shared planner.

    Args:
        execution_mode: Persisted background execution mode literal under test.
    Returns:
        None.
    Assumptions:
        The worker may execute either `background_auto` or compatibility-only
        `background_manual_legacy`, but both must consume the same shared planner surface.
    Raises:
        AssertionError: If the worker stops forwarding internal planner metadata or creates a
            mode-specific profile-selection branch.
    Side Effects:
        None.
    """
    requested_mode = "hybrid_conservative"
    job = _build_running_job_with_artifact_pin(
        execution_mode=execution_mode,
        request_json={
            "mode": "template",
            "execution_profile_mode": requested_mode,
        },
    )
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    planner = _RecordingSharedRuntimePlanner(
        runtime_plan=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        )
    )
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 47, 0)),
        runtime_planner=planner,
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert len(planner.calls) == 1
    assert planner.calls[0]["requested_execution_profile_mode"] == requested_mode
    assert planner.calls[0]["preselect"] == request.preselect
    assert job.execution_mode == execution_mode


def test_process_claimed_job_treats_background_manual_legacy_as_compatibility_only(
) -> None:
    """
    Verify `background_manual_legacy` reuses the canonical shared planner contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `background_manual_legacy` remains a compatibility-only persisted literal and must not
        create a separate worker planner branch from canonical `background_auto`.
    Raises:
        AssertionError: If planner inputs differ only because the persisted execution mode is
            `background_manual_legacy`.
    Side Effects:
        None.
    """
    planner_contracts: dict[
        BacktestJobExecutionMode,
        tuple[RunBacktestTemplate, int, str | None],
    ] = {}

    for execution_mode in ("background_auto", "background_manual_legacy"):
        job = _build_running_job_with_artifact_pin(execution_mode=execution_mode)
        request = _build_request(top_k=5, preselect=2, top_trades_n=1)
        planner = _RecordingSharedRuntimePlanner(
            runtime_plan=_FakeGridContext(
                base_variants=_build_stage_a_variants(),
                risk_variants=_build_risk_variants(),
            )
        )
        use_case = _build_use_case(
            request=request,
            job_repository=_FakeJobRepository(default_job=job),
            lease_repository=_FakeLeaseRepository(),
            results_repository=_FakeResultsRepository(),
            grid_context=_FakeGridContext(
                base_variants=_build_stage_a_variants(),
                risk_variants=_build_risk_variants(),
            ),
            scorer=_DeterministicScorerWithDetails(
                stage_a_scores={
                    _build_stage_a_variants()[0].base_variant_key: 3.0,
                    _build_stage_a_variants()[1].base_variant_key: 2.0,
                }
            ),
            reporting_service=_FakeReportingService(),
            top_k_persisted_default=2,
            snapshot_seconds=None,
            snapshot_variants_step=None,
            stage_batch_size=1,
            now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 48, 0)),
            runtime_planner=planner,
        )

        report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

        assert report.status == "succeeded"
        assert len(planner.calls) == 1
        planner_contracts[execution_mode] = (
            planner.calls[0]["template"],
            planner.calls[0]["preselect"],
            planner.calls[0]["requested_execution_profile_mode"],
        )

    assert planner_contracts["background_auto"] == planner_contracts[
        "background_manual_legacy"
    ]


def test_process_claimed_job_rejects_non_background_execution_mode() -> None:
    """
    Verify the worker rejects claimed rows that are not background execution modes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The claim loop is reserved for queued/running background work, so `sync_inline` must not
        silently enter the worker path.
    Raises:
        AssertionError: If the worker accepts a non-background execution-mode literal.
    Side Effects:
        None.
    """
    use_case = _build_use_case(
        request=_build_request(top_k=5, preselect=2, top_trades_n=1),
        job_repository=_FakeJobRepository(
            default_job=_build_running_job(execution_mode="sync_inline")
        ),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 10, 49, 0)),
    )

    with pytest.raises(
        ValueError,
        match=(
            "process_claimed_job requires background_auto or "
            "background_manual_legacy execution_mode"
        ),
    ):
        use_case.process_claimed_job(
            job=_build_running_job(execution_mode="sync_inline"),
            locked_by="worker-test-1",
        )


def test_process_claimed_job_uses_artifact_stage_a_shortlist_builder_when_available() -> None:
    """
    Verify worker Stage A uses the artifact-backed shortlist builder with pinned context.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Additive R6-02 cutover replaces only Stage A shortlist construction in the worker flow.
    Raises:
        AssertionError: If builder wiring or shortlist persistence drifts.
    Side Effects:
        None.
    """
    job = _build_running_job_with_artifact_pin()
    pinned_context = _FakeSlotPinnedContext(
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_slot="slot_b",
        slot_generation=11,
        artifact_asof_date="2026-03-29",
        artifact_manifest_hash="d" * 64,
    )
    shortlist_builder = _RecordingStageAShortlistBuilder(
        rows=(
            BacktestStageAScoredVariantV1(
                base_variant=_build_stage_a_variants()[1],
                total_return_pct=2.0,
            ),
        )
    )
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    results_repository = _FakeResultsRepository()
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=job),
        lease_repository=_FakeLeaseRepository(),
        results_repository=results_repository,
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 11, 0, 0)),
        artifact_slot_resolver=_RecordingArtifactSlotResolver(context=pinned_context),
        stage_a_shortlist_builder=cast(Any, shortlist_builder),
    )

    report = use_case.process_claimed_job(job=job, locked_by="worker-test-1")

    assert report.status == "succeeded"
    assert len(shortlist_builder.calls) == 1
    assert shortlist_builder.calls[0]["artifact_context"] == pinned_context
    assert shortlist_builder.calls[0]["target_time_range"] == request.time_range
    assert shortlist_builder.calls[0]["shortlist_limit"] == 2
    assert results_repository.shortlist_calls[0]["shortlist"].stage_a_indexes == (1,)


def test_run_backtest_job_runner_v1_prefers_artifact_backed_stage_b_scorer_when_pinned(
    monkeypatch: Any,
) -> None:
    """
    Verify worker scorer resolution prefers the additive artifact-backed Stage B scorer factory.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Worker flow resolves slot-pinned context before scoring, so this test patches only the
        Stage B scorer factory and validates forwarded pinned inputs.
    Raises:
        AssertionError: If worker scorer resolution does not prefer artifact-backed wiring.
    Side Effects:
        Monkeypatches the local Stage B scorer factory for the duration of the test.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    pinned_context = _FakeSlotPinnedContext(
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_slot="slot_b",
        slot_generation=11,
        artifact_asof_date="2026-03-29",
        artifact_manifest_hash="d" * 64,
    )
    expected_scorer = object()
    calls: list[dict[str, Any]] = []
    requested_time_range = TimeRange(
        start=UtcTimestamp(_utc(2026, 2, 23, 11, 0, 0)),
        end=UtcTimestamp(_utc(2026, 2, 23, 11, 5, 0)),
    )

    def _fake_builder(**kwargs: Any) -> object:
        """
        Record worker artifact-backed scorer factory inputs and return a deterministic stub.

        Args:
            **kwargs: Factory arguments forwarded by the worker use-case.
        Returns:
            object: Deterministic scorer sentinel.
        Assumptions:
            This wiring test validates builder selection only and does not exercise Stage B math.
        Raises:
            None.
        Side Effects:
            Appends one call payload to the in-memory log.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
          - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py
        """
        calls.append(kwargs)
        return expected_scorer

    monkeypatch.setattr(
        run_backtest_job_runner_module,
        "build_default_artifact_backed_stage_b_scorer_v2",
        _fake_builder,
    )
    request = _build_request(top_k=5, preselect=2, top_trades_n=1)
    assert request.template is not None
    use_case = _build_use_case(
        request=request,
        job_repository=_FakeJobRepository(default_job=_build_running_job_with_artifact_pin()),
        lease_repository=_FakeLeaseRepository(),
        results_repository=_FakeResultsRepository(),
        grid_context=_FakeGridContext(
            base_variants=_build_stage_a_variants(),
            risk_variants=_build_risk_variants(),
        ),
        scorer=_DeterministicScorerWithDetails(
            stage_a_scores={
                _build_stage_a_variants()[0].base_variant_key: 3.0,
                _build_stage_a_variants()[1].base_variant_key: 2.0,
            }
        ),
        reporting_service=_FakeReportingService(),
        top_k_persisted_default=2,
        snapshot_seconds=None,
        snapshot_variants_step=None,
        stage_batch_size=1,
        now_provider=_NowProvider(current=_utc(2026, 2, 23, 11, 0, 0)),
        artifact_slot_resolver=cast(Any, object()),
    )
    use_case._staged_scorer = None

    scorer = use_case._resolve_staged_scorer(
        template=request.template,
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


def _build_use_case(
    *,
    request: RunBacktestRequest,
    job_repository: _FakeJobRepository,
    lease_repository: _FakeLeaseRepository,
    results_repository: _FakeResultsRepository,
    grid_context: _FakeGridContext,
    scorer: Any,
    reporting_service: _FakeReportingService,
    top_k_persisted_default: int,
    snapshot_seconds: int | None,
    snapshot_variants_step: int | None,
    stage_batch_size: int,
    now_provider: _NowProvider,
    runtime_planner: Any | None = None,
    artifact_slot_resolver: Any | None = None,
    stage_a_shortlist_builder: Any | None = None,
    price_arrays_loader: Any | None = None,
) -> RunBacktestJobRunnerV1:
    """
    Build job-runner use-case with deterministic fakes for unit tests.

    Args:
        request: Decoded request fixture.
        job_repository: Fake job repository.
        lease_repository: Fake lease repository.
        results_repository: Fake results repository.
        grid_context: Fake staged grid context.
        scorer: Fake scorer with deterministic Stage-A/Stage-B metrics.
        reporting_service: Fake reporting service.
        top_k_persisted_default: Persisted cap for top rows.
        snapshot_seconds: Optional time trigger threshold.
        snapshot_variants_step: Optional variants-step trigger threshold.
        stage_batch_size: Batch boundary size.
        now_provider: Monotonic now-provider fixture.
        runtime_planner: Optional shared runtime planner test double.
        artifact_slot_resolver: Optional shared slot-pinned context resolver test double.
        stage_a_shortlist_builder: Optional artifact-backed Stage A builder test double.
        price_arrays_loader: Optional artifact price loader test double.
    Returns:
        RunBacktestJobRunnerV1: Prepared use-case instance.
    Assumptions:
        Request decoder fake returns the provided request for any payload.
    Raises:
        None.
    Side Effects:
        None.
    """
    resolved_artifact_slot_resolver = (
        artifact_slot_resolver
        if artifact_slot_resolver is not None
        else _RecordingArtifactSlotResolver(context=_default_pinned_context())
    )
    resolved_stage_a_shortlist_builder = (
        stage_a_shortlist_builder
        if stage_a_shortlist_builder is not None
        else _ArtifactOnlyStageAShortlistBuilder()
    )
    resolved_price_arrays_loader = (
        price_arrays_loader if price_arrays_loader is not None else _FakePriceArraysLoader()
    )
    resolved_runtime_planner = (
        runtime_planner
        if runtime_planner is not None
        else _RecordingSharedRuntimePlanner(runtime_plan=grid_context)
    )
    return RunBacktestJobRunnerV1(
        job_repository=cast(Any, job_repository),
        lease_repository=cast(Any, lease_repository),
        results_repository=cast(Any, results_repository),
        request_decoder=cast(Any, _FakeRequestDecoder(request=request)),
        indicator_compute=cast(Any, _NoOpIndicatorCompute()),
        runtime_planner=cast(Any, resolved_runtime_planner),
        reporting_service=cast(Any, reporting_service),
        staged_scorer=cast(Any, scorer),
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20_000,
        top_k_persisted_default=top_k_persisted_default,
        heartbeat_seconds=1_000,
        lease_seconds=60,
        snapshot_seconds=snapshot_seconds,
        snapshot_variants_step=snapshot_variants_step,
        stage_batch_size=stage_batch_size,
        now_provider=now_provider,
        artifact_slot_resolver=cast(Any, resolved_artifact_slot_resolver),
        stage_a_shortlist_builder=cast(Any, resolved_stage_a_shortlist_builder),
        price_arrays_loader=cast(Any, resolved_price_arrays_loader),
    )


def _build_request(
    *,
    top_k: int,
    preselect: int,
    top_trades_n: int | None = None,
    ranking: BacktestRankingConfig | None = None,
) -> RunBacktestRequest:
    """
    Build deterministic template-mode request fixture for worker use-case tests.

    Args:
        top_k: Requested top-k value.
        preselect: Requested Stage-A preselect value.
        top_trades_n:
            Retained legacy test-helper argument ignored after the summary-only launch cutover.
        ranking: Optional ranking override payload.
    Returns:
        RunBacktestRequest: Template-mode request fixture.
    Assumptions:
        Indicator template uses one explicit grid and one explicit selection.
    Raises:
        None.
    Side Effects:
        None.
    """
    _ = top_trades_n
    template = RunBacktestTemplate(
        instrument_id=InstrumentId(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
        ),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ema"),
                params={"length": ExplicitValuesSpec(name="length", values=(10,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ema",
                inputs={"source": "close"},
                params={"length": 10},
            ),
        ),
        signal_grids={},
        execution_params={"slippage_pct": 0.01},
    )
    return RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(_utc(2026, 2, 1, 0, 0, 0)),
            end=UtcTimestamp(_utc(2026, 2, 2, 0, 0, 0)),
        ),
        template=template,
        top_k=top_k,
        preselect=preselect,
        ranking=ranking,
    )


def _build_running_job(
    *,
    execution_mode: BacktestJobExecutionMode = "background_auto",
    request_json: Mapping[str, Any] | None = None,
) -> BacktestJob:
    """
    Build deterministic running Backtest job fixture with persisted artifact pin metadata.

    Args:
        execution_mode: Background execution mode literal persisted on the claimed job.
        request_json: Optional persisted request payload override.
    Returns:
        BacktestJob: Running claimed job fixture pinned to immutable artifact identity.
    Assumptions:
        Worker claimed path in R8-01 always executes from persisted slot-pinned metadata.
    Raises:
        None.
    Side Effects:
        None.
    """
    created_at = _utc(2026, 2, 23, 9, 0, 0)
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000910"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=created_at,
        request_json=request_json if request_json is not None else {"mode": "template"},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-29",
        ),
        execution_mode=execution_mode,
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=5,
        ranking_primary_metric="total_return_pct",
        ranking_secondary_metric=None,
    )
    return queued.claim(
        changed_at=created_at + timedelta(seconds=5),
        locked_by="worker-test-1",
        lease_expires_at=created_at + timedelta(seconds=65),
    )


def _build_running_job_with_artifact_pin(
    *,
    execution_mode: BacktestJobExecutionMode = "background_auto",
    request_json: Mapping[str, Any] | None = None,
) -> BacktestJob:
    """
    Build deterministic running Backtest job fixture with persisted artifact pin metadata.

    Args:
        execution_mode: Background execution mode literal persisted on the claimed job.
        request_json: Optional persisted request payload override.
    Returns:
        BacktestJob: Running claimed job fixture with immutable artifact pin metadata.
    Assumptions:
        Background bootstrap tests need the same pin shape that job creation persists.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _build_running_job(
        execution_mode=execution_mode,
        request_json=request_json,
    )


def _default_pinned_context() -> _FakeSlotPinnedContext:
    """
    Build default slot-pinned runtime context fixture for artifact-only worker tests.

    Args:
        None.
    Returns:
        _FakeSlotPinnedContext: Deterministic pinned context matching default job pin metadata.
    Assumptions:
        Default worker tests use BTCUSDT spot artifacts pinned to `slot_b`.
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
        artifact_slot="slot_b",
        slot_generation=11,
        artifact_asof_date="2026-03-29",
        artifact_manifest_hash="d" * 64,
    )


def _build_stage_a_variants() -> tuple[BacktestStageABaseVariant, ...]:
    """
    Build deterministic Stage-A variants fixture.

    Args:
        None.
    Returns:
        tuple[BacktestStageABaseVariant, ...]: Stage-A variants fixture.
    Assumptions:
        Base variant keys are unique canonical 64-char literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    selection_a = IndicatorVariantSelection(
        indicator_id="ema",
        inputs={"source": "close"},
        params={"length": 10},
    )
    selection_b = IndicatorVariantSelection(
        indicator_id="ema",
        inputs={"source": "close"},
        params={"length": 20},
    )
    return (
        BacktestStageABaseVariant(
            stage_a_index=0,
            indicator_selections=(selection_a,),
            signal_params={"ema": {"threshold": 1}},
            indicator_variant_key="1" * 64,
            base_variant_key="a" * 64,
        ),
        BacktestStageABaseVariant(
            stage_a_index=1,
            indicator_selections=(selection_b,),
            signal_params={"ema": {"threshold": 2}},
            indicator_variant_key="2" * 64,
            base_variant_key="b" * 64,
        ),
    )


def _build_risk_variants() -> tuple[BacktestRiskVariantV1, ...]:
    """
    Build deterministic Stage-B risk variants fixture.

    Args:
        None.
    Returns:
        tuple[BacktestRiskVariantV1, ...]: Risk variants fixture.
    Assumptions:
        Risk payload shape follows v1 keys `sl_enabled/sl_pct/tp_enabled/tp_pct`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        BacktestRiskVariantV1(
            risk_index=0,
            risk_params={
                "sl_enabled": False,
                "sl_pct": None,
                "tp_enabled": False,
                "tp_pct": None,
            },
        ),
        BacktestRiskVariantV1(
            risk_index=1,
            risk_params={
                "sl_enabled": True,
                "sl_pct": 1.0,
                "tp_enabled": True,
                "tp_pct": 2.0,
            },
        ),
    )


def _sample_trade() -> TradeV1:
    """
    Build deterministic trade fixture for finalizing trades payload tests.

    Args:
        None.
    Returns:
        TradeV1: Deterministic closed-trade snapshot.
    Assumptions:
        Numeric values satisfy `TradeV1` domain invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return TradeV1(
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
        exit_reason="signal",
    )


def _has_progress_call(
    *,
    calls: list[dict[str, Any]],
    stage: str,
    processed_units: int,
    total_units: int,
) -> bool:
    """
    Check whether progress calls list contains expected stage counters.

    Args:
        calls: Recorded progress calls.
        stage: Expected stage.
        processed_units: Expected processed units.
        total_units: Expected total units.
    Returns:
        bool: `True` when matching call exists.
    Assumptions:
        Calls list order is deterministic but exact index is not asserted.
    Raises:
        None.
    Side Effects:
        None.
    """
    for call in calls:
        if (
            call["stage"] == stage
            and call["processed_units"] == processed_units
            and call["total_units"] == total_units
        ):
            return True
    return False


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    """
    Build timezone-aware UTC datetime helper for fixtures.

    Args:
        year: Year component.
        month: Month component.
        day: Day component.
        hour: Hour component.
        minute: Minute component.
        second: Second component.
    Returns:
        datetime: UTC-aware datetime value.
    Assumptions:
        Input components form a valid calendar datetime.
    Raises:
        ValueError: If datetime components are invalid.
    Side Effects:
        None.
    """
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
