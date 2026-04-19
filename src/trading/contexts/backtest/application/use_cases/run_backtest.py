from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1,
    BacktestRankingConfig,
    BacktestReportV1,
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestSavedOverrides,
    RunBacktestSyncPersistenceArtifact,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestStagedVariantMetricScorer,
    BacktestStagedVariantScorerWithDetails,
    BacktestStrategyReader,
    BacktestStrategySnapshot,
    BacktestVariantScoreDetailsV1,
    CurrentUser,
)
from trading.contexts.backtest.application.services import (
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactSlotResolverV2,
    BacktestHierarchicalShortlistBuilderV2,
    BacktestReportingServiceV1,
    BacktestStageAShortlistBuilderV2,
    ExecutionProfileModeLiteralV2,
    artifact_coordinates_from_market_id_v2,
    build_default_artifact_backed_stage_b_scorer_v2,
    build_default_hierarchical_shortlist_builder_v2,
    build_default_stage_a_shortlist_builder_v2,
    validate_execution_profile_mode_v2,
)
from trading.contexts.backtest.application.services.numba_runtime_v1 import (
    apply_backtest_numba_threads,
    resolve_backtest_stage_a_parallelism_v1,
)
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
from trading.contexts.backtest.application.services.v2.artifact_runtime_core_v2 import (
    STAGE_B_LITERAL_V2,
    BacktestArtifactRuntimeRunnerV2,
    BacktestStageBScoredVariantV2,
    BacktestStageBTaskV2,
    _runtime_plan_uses_no_risk_terminal_path_v2,
    persisted_stage_a_no_risk_exact_rows_v2,
)
from trading.contexts.backtest.application.services.v2.artifact_runtime_plan_v2 import (
    BacktestArtifactRuntimePlannerV2,
    BacktestArtifactRuntimePlanV2,
    runtime_plan_requires_hierarchical_shortlist_runtime_v2,
)
from trading.contexts.backtest.application.services.v2.artifact_runtime_timeline_v2 import (
    BacktestArtifactRuntimeTimelineV2,
    BacktestArtifactTimelineBuilderV2,
)
from trading.contexts.backtest.application.services.v2.price_arrays_loader import (
    MmapPriceArraysLoaderV2,
)
from trading.contexts.backtest.application.services.warmup_estimator import (
    resolve_internal_backtest_warmup_bars,
)
from trading.contexts.backtest.application.use_cases.errors import map_backtest_exception
from trading.contexts.backtest.application.use_cases.request_runtime_contract_v1 import (
    validate_signal_overrides_default_only,
    validate_template_runtime_contract,
)
from trading.contexts.backtest.domain.entities.backtest_job_results import (
    BacktestJobParityClassification,
    BacktestJobParityRetainedRowsCounter,
    BacktestJobParityRuntimeState,
)
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestValidationError,
)
from trading.contexts.backtest.domain.value_objects import build_backtest_variant_key_v1
from trading.contexts.indicators.application.dto import build_variant_key_v1
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import TimeRange

_DEFAULT_FEE_PCT_BY_MARKET_ID = {
    1: 0.075,
    2: 0.1,
    3: 0.075,
    4: 0.1,
}
_DEFAULT_MAX_NUMBA_THREADS = max(1, os.cpu_count() or 1)
MetricScorerV1 = BacktestStagedVariantMetricScorer


@dataclass(frozen=True, slots=True)
class _ResolvedRunContext:
    """
    Internal resolved request context used by run use-case orchestration.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    """

    mode: str
    strategy_id: UUID | None
    template: RunBacktestTemplate
    warmup_bars: int
    top_k: int
    preselect: int
    ranking: BacktestRankingConfig
    artifact_context: ArtifactSlotPinnedRuntimeContextV2
    spec_hash: str | None = None
    spec_payload_json: Mapping[str, Any] | None = None


class RunBacktestUseCase:
    """
    RunBacktestUseCase — staged sync backtest orchestration for saved/ad-hoc modes.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
    """

    def __init__(
        self,
        *,
        candle_feed: CandleFeed | None,
        indicator_compute: IndicatorCompute,
        strategy_reader: BacktestStrategyReader,
        candle_timeline_builder: object | None = None,
        staged_runner: object | None = None,
        staged_scorer: MetricScorerV1 | None = None,
        reporting_service: BacktestReportingServiceV1 | None = None,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        stage_a_shortlist_builder: BacktestStageAShortlistBuilderV2 | None = None,
        hierarchical_shortlist_builder: BacktestHierarchicalShortlistBuilderV2 | None = None,
        runtime_planner: BacktestArtifactRuntimePlannerV2 | None = None,
        runtime_runner: BacktestArtifactRuntimeRunnerV2 | None = None,
        artifact_timeline_builder: BacktestArtifactTimelineBuilderV2 | None = None,
        warmup_bars_default: int = 200,
        top_k_default: int = 300,
        preselect_default: int = 20000,
        ranking_primary_metric_default: str = BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
        ranking_secondary_metric_default: str | None = (
            BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1
        ),
        configurable_ranking_enabled: bool = True,
        init_cash_quote_default: float = 10000.0,
        fixed_quote_default: float = 100.0,
        safe_profit_percent_default: float = 30.0,
        slippage_pct_default: float = 0.01,
        fee_pct_default_by_market_id: Mapping[int, float] | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
        max_numba_threads: int = _DEFAULT_MAX_NUMBA_THREADS,
        eager_top_reports_enabled: bool = False,
        allowed_request_timeframes: tuple[str, ...] | None = None,
        forbidden_request_timeframes: tuple[str, ...] | None = None,
        artifact_slot_resolver: BacktestArtifactSlotResolverV2 | None = None,
    ) -> None:
        """
        Initialize staged backtest use-case dependencies and runtime defaults.

        Docs:
          - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            candle_feed: Indicators candle-feed port producing dense timeline arrays.
            indicator_compute:
                Indicators compute port used for staged grid estimate/materialization.
            strategy_reader: Backtest ACL strategy reader without owner filtering.
            candle_timeline_builder:
                Retained compatibility dependency. Production execution no longer routes through
                live `candle_timeline_builder.py` after R10-01.
            staged_runner:
                Retained compatibility dependency. Production execution no longer routes through
                `staged_runner_v1.py` after R10-01.
            staged_scorer: Optional Stage A/Stage B scorer port implementation.
            reporting_service: Optional report-builder service for variant-report endpoint.
            defaults_provider: Optional defaults provider for compute/signal grid fallback.
            stage_a_shortlist_builder:
                Optional artifact-backed Stage A shortlist builder for production runtime cutover.
            hierarchical_shortlist_builder:
                Optional shared hybrid shortlist builder used for explicit
                `hybrid_conservative` and internal `hybrid_family` rollout runs.
            runtime_planner:
                Optional artifact-backed runtime planner replacing `grid_builder_v1` in
                production paths.
            runtime_runner:
                Optional shared artifact-backed Stage B runner replacing
                `staged_core_runner_v1` in production paths.
            artifact_timeline_builder:
                Optional artifact-backed request-timeframe timeline builder replacing live
                ClickHouse timeline construction in production paths.
            warmup_bars_default:
                Retained compatibility/default-only runtime setting ignored by the active
                derived warmup path after Milestone B / EPIC B2.
            top_k_default: Runtime default top-k response limit.
            preselect_default: Runtime default preselect shortlist limit.
            ranking_primary_metric_default:
                Runtime default for ranking primary metric literal.
            ranking_secondary_metric_default:
                Retained compatibility runtime setting ignored after single-metric ranking cutover.
            configurable_ranking_enabled:
                Feature-flag guard for configurable ranking behavior rollout.
            init_cash_quote_default: Runtime default initial strategy quote balance.
            fixed_quote_default: Runtime default fixed quote notional for `fixed_quote`.
            safe_profit_percent_default: Runtime default profit-lock percent.
            slippage_pct_default: Runtime default slippage percent.
            fee_pct_default_by_market_id: Runtime default fee mapping by market id.
            max_variants_per_compute: Stage variants guard limit.
            max_compute_bytes_total: Stage memory guard limit.
            max_numba_threads:
                Runtime CPU knob for backtest runs mapped to maximum Numba threads.
            eager_top_reports_enabled:
                Retained legacy flag; sync runtime summaries stay summary-only and defer report
                bodies to explicit on-demand variant-report flows.
            allowed_request_timeframes:
                Optional runtime contract list for supported request timeframes.
            forbidden_request_timeframes:
                Optional runtime contract list for explicitly forbidden request timeframes.
            artifact_slot_resolver:
                Shared slot-pinned context bootstrap used before runtime work starts. Production
                execution requires this dependency after R10-01.
        Returns:
            None.
        Assumptions:
            Runtime defaults come from fail-fast `configs/<env>/backtest.yaml` loader.
        Raises:
            ValueError: If dependencies are missing or scalar defaults/guards are invalid.
        Side Effects:
            None.
        """
        if indicator_compute is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires indicator_compute")
        if strategy_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires strategy_reader")
        if artifact_slot_resolver is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires artifact_slot_resolver")
        if warmup_bars_default <= 0:
            raise ValueError("RunBacktestUseCase.warmup_bars_default must be > 0")
        if top_k_default <= 0:
            raise ValueError("RunBacktestUseCase.top_k_default must be > 0")
        if preselect_default <= 0:
            raise ValueError("RunBacktestUseCase.preselect_default must be > 0")
        if init_cash_quote_default <= 0.0:
            raise ValueError("RunBacktestUseCase.init_cash_quote_default must be > 0")
        if fixed_quote_default <= 0.0:
            raise ValueError("RunBacktestUseCase.fixed_quote_default must be > 0")
        if safe_profit_percent_default < 0.0 or safe_profit_percent_default > 100.0:
            raise ValueError("RunBacktestUseCase.safe_profit_percent_default must be in [0, 100]")
        if slippage_pct_default < 0.0:
            raise ValueError("RunBacktestUseCase.slippage_pct_default must be >= 0")
        if max_variants_per_compute <= 0:
            raise ValueError("RunBacktestUseCase.max_variants_per_compute must be > 0")
        if max_compute_bytes_total <= 0:
            raise ValueError("RunBacktestUseCase.max_compute_bytes_total must be > 0")
        if max_numba_threads <= 0:
            raise ValueError("RunBacktestUseCase.max_numba_threads must be > 0")
        if not isinstance(configurable_ranking_enabled, bool):
            raise ValueError("RunBacktestUseCase.configurable_ranking_enabled must be bool")
        if not isinstance(eager_top_reports_enabled, bool):
            raise ValueError("RunBacktestUseCase.eager_top_reports_enabled must be bool")

        ranking_defaults = BacktestRankingConfig(
            primary_metric=ranking_primary_metric_default,
        )

        _ = candle_feed, candle_timeline_builder, staged_runner
        resolved_stage_a_shortlist_builder = (
            stage_a_shortlist_builder
            or build_default_stage_a_shortlist_builder_v2(
                artifact_slot_resolver=artifact_slot_resolver,
                configurable_ranking_enabled=configurable_ranking_enabled,
                init_cash_quote_default=init_cash_quote_default,
                fixed_quote_default=fixed_quote_default,
                safe_profit_percent_default=safe_profit_percent_default,
                slippage_pct_default=slippage_pct_default,
                fee_pct_default_by_market_id=fee_pct_default_by_market_id,
            )
        )
        if resolved_stage_a_shortlist_builder is None:
            raise ValueError(
                "RunBacktestUseCase requires artifact-backed stage_a_shortlist_builder"
            )
        resolved_hierarchical_shortlist_builder = (
            hierarchical_shortlist_builder
            or build_default_hierarchical_shortlist_builder_v2(
                artifact_slot_resolver=artifact_slot_resolver,
                defaults_provider=defaults_provider,
            )
        )
        resolved_timeline_builder = artifact_timeline_builder or BacktestArtifactTimelineBuilderV2(
            price_arrays_loader=_build_price_arrays_loader_v2(
                artifact_slot_resolver=artifact_slot_resolver
            )
        )

        self._artifact_timeline_builder = resolved_timeline_builder
        self._indicator_compute = indicator_compute
        self._strategy_reader = strategy_reader
        self._staged_scorer = staged_scorer
        self._reporting_service = reporting_service or BacktestReportingServiceV1()
        self._defaults_provider = defaults_provider
        self._stage_a_shortlist_builder = resolved_stage_a_shortlist_builder
        self._hierarchical_shortlist_builder = resolved_hierarchical_shortlist_builder
        self._runtime_planner = runtime_planner or BacktestArtifactRuntimePlannerV2()
        self._runtime_runner = runtime_runner or BacktestArtifactRuntimeRunnerV2(
            configurable_ranking_enabled=configurable_ranking_enabled
        )
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default
        self._ranking_defaults = ranking_defaults
        self._configurable_ranking_enabled = configurable_ranking_enabled
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
        self._max_variants_per_compute = max_variants_per_compute
        self._max_compute_bytes_total = max_compute_bytes_total
        self._max_numba_threads = max_numba_threads
        self._eager_top_reports_enabled = eager_top_reports_enabled
        self._allowed_request_timeframes = _normalize_timeframe_literals(
            values=allowed_request_timeframes
        )
        self._forbidden_request_timeframes = _normalize_timeframe_literals(
            values=forbidden_request_timeframes
        )
        self._artifact_slot_resolver = artifact_slot_resolver

    def _run_scoped_artifact_timeline_builder(self) -> BacktestArtifactTimelineBuilderV2:
        """
        Resolve one timeline builder whose `artifact_loader` caches belong to the current run.

        Args:
            None.
        Returns:
            BacktestArtifactTimelineBuilderV2: Fresh run-owned timeline builder when the stored
                prototype exposes explicit `run_scoped` semantics, otherwise the original builder
                for compatibility with custom injections and test doubles.
        Assumptions:
            The default live API path wires `BacktestArtifactTimelineBuilderV2`, so returning a
            fresh builder removes process-lifetime mmap retention without changing public
            contracts.
        Raises:
            TypeError: If an injected builder exposes a non-callable `run_scoped` attribute.
        Side Effects:
            None.
        """
        run_scoped_builder = getattr(self._artifact_timeline_builder, "run_scoped", None)
        if run_scoped_builder is None:
            return self._artifact_timeline_builder
        if not callable(run_scoped_builder):
            raise TypeError("artifact_timeline_builder run_scoped attribute must be callable")
        return cast(BacktestArtifactTimelineBuilderV2, run_scoped_builder())

    def _run_scoped_stage_a_shortlist_builder(self) -> BacktestStageAShortlistBuilderV2:
        """
        Resolve one Stage A builder whose large artifact caches belong to the current run.

        Args:
            None.
        Returns:
            BacktestStageAShortlistBuilderV2: Fresh run-owned Stage A builder when the stored
                prototype exposes explicit `run_scoped` semantics, otherwise the original builder
                for compatibility with custom injections and test doubles.
        Assumptions:
            The default live API path wires `BacktestStageAShortlistBuilderV2`, so returning a
            fresh builder removes process-lifetime mmap retention while preserving same-run
            reuse.
        Raises:
            TypeError: If an injected builder exposes a non-callable `run_scoped` attribute.
        Side Effects:
            None.
        """
        run_scoped_builder = getattr(self._stage_a_shortlist_builder, "run_scoped", None)
        if run_scoped_builder is None:
            return self._stage_a_shortlist_builder
        if not callable(run_scoped_builder):
            raise TypeError("stage_a_shortlist_builder run_scoped attribute must be callable")
        return cast(BacktestStageAShortlistBuilderV2, run_scoped_builder())

    def _run_scoped_hierarchical_shortlist_builder(
        self,
    ) -> BacktestHierarchicalShortlistBuilderV2 | None:
        """
        Resolve one hierarchical builder whose large artifact caches belong to the current run.

        Args:
            None.
        Returns:
            BacktestHierarchicalShortlistBuilderV2 | None: Fresh run-owned hybrid shortlist
                builder when the stored prototype exposes explicit `run_scoped` semantics,
                otherwise the original builder, or `None` when hierarchical runtime is
                unavailable.
        Assumptions:
            The live API path may enter `hybrid_conservative` or `hybrid_family`, so hybrid
            shortlist artifact loaders must not stay attached to the long-lived use-case
            singleton.
        Raises:
            TypeError: If an injected builder exposes a non-callable `run_scoped` attribute.
        Side Effects:
            None.
        """
        if self._hierarchical_shortlist_builder is None:
            return None
        run_scoped_builder = getattr(self._hierarchical_shortlist_builder, "run_scoped", None)
        if run_scoped_builder is None:
            return self._hierarchical_shortlist_builder
        if not callable(run_scoped_builder):
            raise TypeError("hierarchical_shortlist_builder run_scoped attribute must be callable")
        return cast(BacktestHierarchicalShortlistBuilderV2, run_scoped_builder())

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control: BacktestRunControlV1 | None = None,
    ) -> RunBacktestResponse:
        """
        Execute staged sync flow and return deterministic top-k variant preview response.

        Docs:
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
          - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/application/dto/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/errors.py

        Args:
            request: Saved/ad-hoc backtest request.
            current_user: Authenticated user for ownership checks in saved mode.
            request_payload:
                Optional strict API payload snapshot accepted for compatibility with
                persisted-run orchestrators. Sync execution also honors the internal-only
                `execution_profile_mode` override when present; this field is not part of the
                public `/backtests` request contract and must stay excluded from request hashes.
                After Milestone F / EPIC F1 the persisted sync-inline launch wrapper uses this
                additive metadata to pin canonical `NR2` `POST /backtests` launches onto the
                dedicated `exact_no_risk_parity` internal engine path.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            RunBacktestResponse: Deterministic staged response with ranked top-k variants.
        Assumptions:
            Trade execution/metrics engine is delegated to scorer port implementation.
        Raises:
            RoehubError: Canonical mapped error for validation/forbidden/not-found/conflict/
                unexpected.
        Side Effects:
            Loads pinned artifact candles, resolves deterministic runtime variants, and calls the
            Stage B scorer port.
        """
        try:
            if request is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError("RunBacktestUseCase.execute requires request")
            if current_user is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError("RunBacktestUseCase.execute requires current_user")
            requested_execution_profile_mode = (
                _requested_execution_profile_mode_from_payload_v2(
                    request_payload=request_payload
                )
            )

            apply_backtest_numba_threads(max_numba_threads=self._max_numba_threads)
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)
            resolved = self._resolve_run_context(request=request, current_user=current_user)
            artifact_timeline_builder = self._run_scoped_artifact_timeline_builder()
            stage_a_shortlist_builder = self._run_scoped_stage_a_shortlist_builder()
            timeline = artifact_timeline_builder.build(
                artifact_context=resolved.artifact_context,
                market_id=resolved.template.instrument_id.market_id,
                symbol=resolved.template.instrument_id.symbol,
                timeframe=resolved.template.timeframe,
                requested_time_range=request.time_range,
                warmup_bars=resolved.warmup_bars,
            )
            runtime_plan = self._runtime_planner.build(
                template=resolved.template,
                candles=timeline.candles,
                indicator_compute=self._indicator_compute,
                preselect=resolved.preselect,
                requested_execution_profile_mode=requested_execution_profile_mode,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
            )
            effective_runtime_plan = runtime_plan
            if (
                not _runtime_plan_is_exact_no_risk_parity_v2(runtime_plan=runtime_plan)
                and runtime_plan_requires_hierarchical_shortlist_runtime_v2(
                    runtime_plan=runtime_plan
                )
            ):
                hierarchical_shortlist_builder = (
                    self._run_scoped_hierarchical_shortlist_builder()
                )
                if hierarchical_shortlist_builder is None:
                    raise ValueError(
                        "RunBacktestUseCase requires hierarchical_shortlist_builder for "
                        "hybrid shortlist runtime"
                    )
                effective_runtime_plan = (
                    hierarchical_shortlist_builder.build_runtime_plan(
                        runtime_plan=runtime_plan,
                        artifact_context=resolved.artifact_context,
                        target_time_range=request.time_range,
                    )
                )
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)
            resolved_scorer = self._resolve_staged_scorer(
                template=resolved.template,
                target_slice=timeline.target_slice,
                target_time_range=request.time_range,
                artifact_context=resolved.artifact_context,
            )
            self._prepare_scorer_for_runtime_plan(
                scorer=resolved_scorer,
                runtime_plan=effective_runtime_plan,
                candles=timeline.candles,
                run_control=run_control,
            )
            stage_a_parallelism = resolve_backtest_stage_a_parallelism_v1(
                execution_profile=effective_runtime_plan.execution_profile,
                max_numba_threads=self._max_numba_threads,
            )
            shortlist = stage_a_shortlist_builder.build_shortlist(
                grid_context=effective_runtime_plan,
                artifact_context=resolved.artifact_context,
                target_time_range=request.time_range,
                shortlist_limit=resolved.preselect,
                ranking=resolved.ranking,
                parallelism=stage_a_parallelism,
                cancel_checker=_cancel_checker_from_run_control(run_control=run_control),
            )
            ranked_rows, ranked_tasks = self._runtime_runner.run_stage_b_or_finalize_no_risk(
                template=resolved.template,
                runtime_plan=effective_runtime_plan,
                shortlist=shortlist,
                candles=timeline.candles,
                scorer=resolved_scorer,
                top_k_limit=resolved.top_k,
                ranking=resolved.ranking,
                cancel_checker=_cancel_checker_from_run_control(run_control=run_control),
            )
            variants = self._build_variant_previews(
                template=resolved.template,
                ranked_rows=ranked_rows,
                ranked_tasks=ranked_tasks,
            )
            sync_persistence_artifact = _build_sync_persistence_artifact(
                runtime_plan=effective_runtime_plan,
                shortlist=shortlist,
            )

            return RunBacktestResponse(
                mode=resolved.mode,
                instrument_id=resolved.template.instrument_id,
                timeframe=resolved.template.timeframe,
                strategy_id=resolved.strategy_id,
                top_k=resolved.top_k,
                preselect=resolved.preselect,
                direction_mode=resolved.template.direction_mode,
                sizing_mode=resolved.template.sizing_mode,
                execution_params=resolved.template.execution_params,
                variants=variants,
                total_indicator_compute_calls=runtime_plan.indicator_estimate_calls,
                artifact_slot=resolved.artifact_context.artifact_slot,
                artifact_slot_generation=resolved.artifact_context.slot_generation,
                artifact_asof_date=resolved.artifact_context.artifact_asof_date,
                artifact_manifest_hash=resolved.artifact_context.artifact_manifest_hash,
                spec_hash=resolved.spec_hash,
                spec_payload_json=resolved.spec_payload_json,
                execution_profile_mode=effective_runtime_plan.execution_profile.mode,
                sync_persistence_artifact=sync_persistence_artifact,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    def preflight(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        run_control: BacktestRunControlV1 | None = None,
    ) -> None:
        """
        Validate canonical staged guard budgets without executing Stage A or Stage B.

        Docs:
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
          - apps/api/wiring/modules/backtest.py
        Args:
            request: Saved/ad-hoc backtest request.
            current_user: Authenticated user for ownership checks in saved mode.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            None.
        Assumptions:
            Preflight reuses the same request resolution, timeline loading, and guard budgets as
            the corresponding sync or background launch path.
        Raises:
            RoehubError: Canonical mapped error for validation/forbidden/not-found/conflict/
                unexpected.
        Side Effects:
            Loads pinned artifact candles and calls `indicator_compute.estimate(...)` for
            deterministic runtime planning only.
        """
        try:
            if request is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError("RunBacktestUseCase.preflight requires request")
            if current_user is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.preflight requires current_user"
                )

            apply_backtest_numba_threads(max_numba_threads=self._max_numba_threads)
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)
            resolved = self._resolve_run_context(request=request, current_user=current_user)
            artifact_timeline_builder = self._run_scoped_artifact_timeline_builder()
            timeline = artifact_timeline_builder.build(
                artifact_context=resolved.artifact_context,
                market_id=resolved.template.instrument_id.market_id,
                symbol=resolved.template.instrument_id.symbol,
                timeframe=resolved.template.timeframe,
                requested_time_range=request.time_range,
                warmup_bars=resolved.warmup_bars,
            )
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)
            self._runtime_planner.build(
                template=resolved.template,
                candles=timeline.candles,
                indicator_compute=self._indicator_compute,
                preselect=resolved.preselect,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: BacktestVariantPayloadV1,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestReportV1:
        """
        Build on-demand deterministic report for one explicit variant payload.

        Docs:
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - apps/api/routes/backtests.py
          - apps/api/dto/backtests.py
          - src/trading/contexts/backtest/application/services/reporting_service_v1.py
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py

        Args:
            request: Saved/ad-hoc run context envelope used for instrument/timeframe/timeline.
            current_user: Authenticated user for saved-mode ownership checks.
            variant_payload: Explicit variant payload selected in UI for lazy report load.
            include_trades: Whether to include full trades payload in response report.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Deterministic report (`rows`, `table_md`, optional `trades`).
        Assumptions:
            Variant payload keeps v1 `variant_key` semantics unchanged.
        Raises:
            RoehubError: Canonical mapped error for validation/not-found/forbidden/conflict/
                unexpected.
        Side Effects:
            Reads candles, scores one explicit variant in Stage-B mode, and builds report table.
        """
        try:
            if request is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.build_variant_report requires request"
                )
            if current_user is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.build_variant_report requires current_user"
                )
            if variant_payload is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.build_variant_report requires variant_payload"
                )
            resolved = self._resolve_run_context(request=request, current_user=current_user)
            return self.build_variant_report_for_template(
                requested_time_range=request.time_range,
                template=resolved.template,
                warmup_bars=resolved.warmup_bars,
                variant_payload=variant_payload,
                include_trades=include_trades,
                run_control=run_control,
                artifact_context=resolved.artifact_context,
                template_root_path=(
                    "saved_strategy" if resolved.mode == "saved" else "body.template"
                ),
                template_already_validated=True,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    def build_variant_report_for_template(
        self,
        *,
        requested_time_range: TimeRange,
        template: RunBacktestTemplate,
        warmup_bars: int | None,
        variant_payload: BacktestVariantPayloadV1,
        include_trades: bool = False,
        run_control: BacktestRunControlV1 | None = None,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2 | None = None,
        template_root_path: str = "body.template",
        template_already_validated: bool = False,
    ) -> BacktestReportV1:
        """
        Build one lazy report from already resolved template and artifact context.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/backtest-runs-history-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
          - apps/api/routes/backtest_runs.py

        Args:
            requested_time_range: Original user request range used for reporting metrics.
            template: Effective template resolved from ad-hoc request or persisted run snapshot.
            warmup_bars: Optional warmup override from request or persisted request snapshot.
                New public call paths should pass `None`; explicit values remain
                compatibility-only for internal callers.
            variant_payload: Explicit selected variant payload for one lazy detail recompute.
            include_trades: Whether to include trades in response payload.
            run_control: Optional cooperative cancellation/deadline control object.
            artifact_context: Optional already pinned artifact runtime context.
            template_root_path: Validation root path used when runtime-contract checks run here.
            template_already_validated:
                Whether caller already ran deterministic template runtime-contract validation.
        Returns:
            BacktestReportV1: Deterministic report payload for exactly one selected variant.
        Assumptions:
            Caller may provide persisted run template + pinned artifact context to avoid reading
            live strategy storage or active `current.yaml`.
        Raises:
            BacktestValidationError: If inputs or runtime contract invariants are invalid.
            RoehubError: Canonical mapped unexpected/domain errors.
        Side Effects:
            Reads candles, scores one explicit variant, and materializes report/trades payloads.
        """
        try:
            if template is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.build_variant_report_for_template requires template"
                )
            if variant_payload is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError(
                    "RunBacktestUseCase.build_variant_report_for_template requires "
                    "variant_payload"
                )
            if not template_already_validated:
                validate_template_runtime_contract(
                    template=template,
                    defaults_provider=self._defaults_provider,
                    allowed_request_timeframes=self._allowed_request_timeframes,
                    forbidden_request_timeframes=self._forbidden_request_timeframes,
                    root_path=template_root_path,
                )

            apply_backtest_numba_threads(max_numba_threads=self._max_numba_threads)
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)

            resolved_warmup_bars = resolve_internal_backtest_warmup_bars(
                template=template,
                warmup_bars=warmup_bars,
            )
            resolved_artifact_context = (
                artifact_context
                if artifact_context is not None
                else self._bootstrap_artifact_context(template=template)
            )
            artifact_timeline_builder = self._run_scoped_artifact_timeline_builder()
            timeline = artifact_timeline_builder.build(
                artifact_context=resolved_artifact_context,
                market_id=template.instrument_id.market_id,
                symbol=template.instrument_id.symbol,
                timeframe=template.timeframe,
                requested_time_range=requested_time_range,
                warmup_bars=resolved_warmup_bars,
            )
            if run_control is not None:
                run_control.raise_if_cancelled(stage=STAGE_B_LITERAL_V2)

            scored_details = self._score_variant_payload_with_details(
                template=template,
                timeline=timeline,
                variant_payload=variant_payload,
                target_time_range=requested_time_range,
                artifact_context=resolved_artifact_context,
            )
            return self._reporting_service.build_report_from_details(
                requested_time_range=requested_time_range,
                candles=timeline.candles,
                details=scored_details,
                include_table_md=True,
                include_trades=include_trades,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    def _resolve_run_context(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
    ) -> _ResolvedRunContext:
        """
        Resolve final run-mode/template/default values before external port calls.

        Args:
            request: Input backtest request.
            current_user: Authenticated user for saved-mode ownership checks.
        Returns:
            _ResolvedRunContext: Fully resolved run context.
        Assumptions:
            Request mode exclusivity is already validated by DTO invariants.
        Raises:
            BacktestNotFoundError: If saved strategy is missing/deleted.
            BacktestForbiddenError: If saved strategy does not belong to current user.
            BacktestValidationError: If resolved template cannot be built.
        Side Effects:
            Reads saved strategy snapshot through ACL port in saved mode.
        """
        top_k = self._resolve_with_default(
            value=request.top_k,
            default=self._top_k_default,
        )
        preselect = self._resolve_with_default(
            value=request.preselect,
            default=self._preselect_default,
        )
        ranking = self._resolve_ranking_config(request=request)

        if request.strategy_id is not None:
            snapshot = self._strategy_reader.load_any(strategy_id=request.strategy_id)
            base_template = self._template_from_snapshot(
                strategy_id=request.strategy_id,
                snapshot=snapshot,
                current_user=current_user,
            )
            spec_payload_json = _build_saved_spec_payload(snapshot=snapshot)
            validate_template_runtime_contract(
                template=base_template,
                defaults_provider=self._defaults_provider,
                allowed_request_timeframes=self._allowed_request_timeframes,
                forbidden_request_timeframes=self._forbidden_request_timeframes,
                root_path="saved_strategy",
            )
            validate_signal_overrides_default_only(
                signal_grids=(
                    request.overrides.signal_grids
                    if request.overrides is not None and request.overrides.signal_grids is not None
                    else {}
                ),
                defaults_provider=self._defaults_provider,
                root_path="body.overrides.signal_grids",
            )
            template = self._apply_saved_overrides(
                base_template=base_template,
                overrides=request.overrides,
            )
            warmup_bars = resolve_internal_backtest_warmup_bars(
                template=template,
                warmup_bars=request.warmup_bars,
            )
            artifact_context = self._bootstrap_artifact_context(template=template)
            return _ResolvedRunContext(
                mode="saved",
                strategy_id=request.strategy_id,
                template=template,
                warmup_bars=warmup_bars,
                top_k=top_k,
                preselect=preselect,
                ranking=ranking,
                artifact_context=artifact_context,
                spec_hash=_build_sha256_from_payload(payload=spec_payload_json),
                spec_payload_json=MappingProxyType(spec_payload_json),
            )

        if request.template is None:  # pragma: no cover - guarded by request DTO invariant
            raise BacktestValidationError(
                "RunBacktestRequest.template is required for template mode"
            )
        validate_template_runtime_contract(
            template=request.template,
            defaults_provider=self._defaults_provider,
            allowed_request_timeframes=self._allowed_request_timeframes,
            forbidden_request_timeframes=self._forbidden_request_timeframes,
            root_path="body.template",
        )
        warmup_bars = resolve_internal_backtest_warmup_bars(
            template=request.template,
            warmup_bars=request.warmup_bars,
        )
        artifact_context = self._bootstrap_artifact_context(template=request.template)

        return _ResolvedRunContext(
            mode="template",
            strategy_id=None,
            template=request.template,
            warmup_bars=warmup_bars,
            top_k=top_k,
            preselect=preselect,
            ranking=ranking,
            artifact_context=artifact_context,
            spec_hash=None,
            spec_payload_json=None,
        )

    def _bootstrap_artifact_context(
        self,
        *,
        template: RunBacktestTemplate,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """
        Resolve the optional shared R6-01 slot-pinned context before sync runtime work starts.

        Args:
            template: Effective validated run template with canonical instrument identity.
        Returns:
            ArtifactSlotPinnedRuntimeContextV2: Bootstrapped immutable slot context.
        Assumptions:
            Production sync/detail runtime is slot-pinned and must fail fast when artifacts are
            unavailable or resolver wiring is missing.
        Raises:
            BacktestValidationError: If artifact-slot resolver wiring is missing.
            BacktestValidationError: If artifact coordinates, `current.yaml`, or slot manifest are
                unavailable or violate strict startup contracts.
        Side Effects:
            Reads strict artifact metadata from disk when resolver wiring is enabled.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
        """
        if self._artifact_slot_resolver is None:
            raise BacktestValidationError(
                "artifact-backed runtime requires artifact_slot_resolver wiring"
            )
        market_id = template.instrument_id.market_id.value
        symbol_literal = str(template.instrument_id.symbol)
        try:
            coordinates = artifact_coordinates_from_market_id_v2(
                market_id=market_id,
                symbol=symbol_literal,
            )
            return self._artifact_slot_resolver.resolve_active_context(coordinates)
        except ValueError as error:
            raise BacktestValidationError(
                "Published backtest artifacts violate shared slot-pinned context contract: "
                f"{error}"
            ) from error
        except FileNotFoundError as error:
            raise BacktestValidationError(
                "Published backtest artifacts are unavailable for requested instrument"
            ) from error

    def _template_from_snapshot(
        self,
        *,
        strategy_id: UUID,
        snapshot: BacktestStrategySnapshot | None,
        current_user: CurrentUser,
    ) -> RunBacktestTemplate:
        """
        Convert saved strategy snapshot into template after ownership/deletion checks.

        Args:
            strategy_id: Requested saved strategy identifier.
            snapshot: Loaded snapshot or `None`.
            current_user: Authenticated principal.
        Returns:
            RunBacktestTemplate: Template equivalent used by staged flow.
        Assumptions:
            Missing and deleted snapshots are hidden behind one `not_found` contract.
        Raises:
            BacktestNotFoundError: If snapshot is missing or soft-deleted.
            BacktestForbiddenError: If snapshot owner differs from current user.
        Side Effects:
            None.
        """
        if snapshot is None or snapshot.is_deleted:
            raise BacktestNotFoundError(strategy_id=strategy_id)
        if snapshot.user_id != current_user.user_id:
            raise BacktestForbiddenError(strategy_id=strategy_id)

        return RunBacktestTemplate(
            instrument_id=snapshot.instrument_id,
            timeframe=snapshot.timeframe,
            indicator_grids=snapshot.indicator_grids,
            indicator_selections=snapshot.indicator_selections,
            signal_grids=snapshot.signal_grids,
            risk_grid=snapshot.risk_grid,
            direction_mode=snapshot.direction_mode,
            sizing_mode=snapshot.sizing_mode,
            risk_params=snapshot.risk_params,
            execution_params=snapshot.execution_params,
        )

    def _apply_saved_overrides(
        self,
        *,
        base_template: RunBacktestTemplate,
        overrides: RunBacktestSavedOverrides | None,
    ) -> RunBacktestTemplate:
        """
        Merge optional saved-mode overrides over loaded snapshot template deterministically.

        Args:
            base_template: Template resolved from saved strategy snapshot.
            overrides: Optional saved-mode overrides from request payload.
        Returns:
            RunBacktestTemplate: Effective template used for staged run execution.
        Assumptions:
            Ownership/deletion checks already passed before this merge step.
        Raises:
            ValueError: Propagated from template/override value-object validation.
        Side Effects:
            None.
        """
        if overrides is None:
            return base_template

        direction_mode = (
            overrides.direction_mode
            if overrides.direction_mode is not None
            else base_template.direction_mode
        )
        sizing_mode = (
            overrides.sizing_mode
            if overrides.sizing_mode is not None
            else base_template.sizing_mode
        )
        signal_grids = _merge_signal_grids(
            base=base_template.signal_grids or {},
            updates=overrides.signal_grids or {},
        )
        risk_params = _merge_scalar_mappings(
            base=base_template.risk_params or {},
            updates=overrides.risk_params or {},
        )
        execution_params = _merge_scalar_mappings(
            base=base_template.execution_params or {},
            updates=overrides.execution_params or {},
        )
        risk_grid = (
            overrides.risk_grid if overrides.risk_grid is not None else base_template.risk_grid
        )

        return RunBacktestTemplate(
            instrument_id=base_template.instrument_id,
            timeframe=base_template.timeframe,
            indicator_grids=base_template.indicator_grids,
            indicator_selections=base_template.indicator_selections,
            signal_grids=signal_grids,
            risk_grid=risk_grid,
            direction_mode=direction_mode,
            sizing_mode=sizing_mode,
            risk_params=risk_params,
            execution_params=execution_params,
        )

    def _resolve_with_default(self, *, value: int | None, default: int) -> int:
        """
        Resolve optional positive integer override against runtime default.

        Args:
            value: Optional override from request DTO.
            default: Runtime default loaded from config.
        Returns:
            int: Effective positive integer value.
        Assumptions:
            Defaults are validated in use-case constructor.
        Raises:
            BacktestValidationError: If provided override is non-positive.
        Side Effects:
            None.
        """
        if value is None:
            return default
        if value <= 0:
            raise BacktestValidationError("Backtest request override values must be > 0")
        return value

    def _resolve_ranking_config(self, *, request: RunBacktestRequest) -> BacktestRankingConfig:
        """
        Resolve effective ranking config from request override, runtime defaults, and feature flag.

        Args:
            request: Backtest request payload.
        Returns:
            BacktestRankingConfig: Effective deterministic ranking config.
        Assumptions:
            DTO validation already normalized metric literals and duplicate checks.
        Raises:
            ValueError: If runtime ranking defaults are invalid.
        Side Effects:
            None.
        """
        if not self._configurable_ranking_enabled:
            return BacktestRankingConfig()
        if request.ranking is not None:
            return request.ranking
        return self._ranking_defaults

    def _resolve_staged_scorer(
        self,
        *,
        template: RunBacktestTemplate,
        target_slice: slice,
        target_time_range: TimeRange,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> MetricScorerV1:
        """
        Resolve artifact-backed scorer for current execution.

        Args:
            template: Resolved run template containing direction/sizing/execution settings.
            target_slice: Trading/reporting target slice inside warmup-inclusive timeline.
            target_time_range: Requested trading/reporting window for artifact-backed kernels.
            artifact_context: Slot-pinned artifact context resolved at runtime startup.
        Returns:
            MetricScorerV1: Scorer used by artifact-backed runtime.
        Assumptions:
            Injected scorer takes precedence over default artifact-backed scorer composition.
        Raises:
            ValueError: If artifact-backed scorer wiring is unavailable.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
        """
        if self._staged_scorer is not None:
            return self._staged_scorer

        artifact_backed_scorer = build_default_artifact_backed_stage_b_scorer_v2(
            artifact_slot_resolver=self._artifact_slot_resolver,
            artifact_context=artifact_context,
            template=template,
            target_time_range=target_time_range,
            report_target_slice=target_slice,
            init_cash_quote_default=self._init_cash_quote_default,
            fixed_quote_default=self._fixed_quote_default,
            safe_profit_percent_default=self._safe_profit_percent_default,
            slippage_pct_default=self._slippage_pct_default,
            fee_pct_default_by_market_id=self._fee_pct_default_by_market_id,
        )
        if artifact_backed_scorer is None:
            raise ValueError(
                "artifact-backed runtime requires slot-pinned Stage B scorer wiring"
            )
        return artifact_backed_scorer

    def _resolve_requested_time_range_for_sync_response(
        self,
        *,
        request: RunBacktestRequest,
    ) -> TimeRange | None:
        """
        Enforce summary-only sync runtime materialization for `POST /api/backtests`.

        Docs:
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - apps/api/routes/backtests.py
          - apps/api/dto/backtests.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py

        Args:
            request: Sync run request envelope.
        Returns:
            TimeRange | None: Always `None` because sync runtime summaries are summary-only.
        Assumptions:
            Dedicated variant-report flows materialize report/trades payloads on demand using the
            explicit selected variant payload and current pinned runtime context.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = request
        return None

    def _score_variant_payload_with_details(
        self,
        *,
        template: RunBacktestTemplate,
        timeline: BacktestArtifactRuntimeTimelineV2,
        variant_payload: BacktestVariantPayloadV1,
        target_time_range: TimeRange,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> BacktestVariantScoreDetailsV1:
        """
        Score one explicit variant payload with Stage-B details scorer contract.

        Docs:
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
          - src/trading/contexts/backtest/application/ports/staged_runner.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py

        Args:
            template: Resolved run template with instrument/timeframe context.
            timeline: Warmup-inclusive candle timeline object built for request range.
            variant_payload: Explicit selected variant payload.
            target_time_range: Requested trading/reporting window for artifact-backed kernels.
            artifact_context: Optional slot-pinned artifact context resolved at runtime startup.
        Returns:
            BacktestVariantScoreDetailsV1: Deterministic details for report assembly.
        Assumptions:
            Explicit payload is normalized and valid by DTO invariants.
        Raises:
            BacktestValidationError: If resolved scorer does not provide details API.
            ValueError: If variant identity payload cannot be normalized.
        Side Effects:
            Scores one Stage-B variant using deterministic scorer implementation.
        """
        scorer = self._resolve_staged_scorer(
            template=RunBacktestTemplate(
                instrument_id=template.instrument_id,
                timeframe=template.timeframe,
                indicator_grids=template.indicator_grids,
                indicator_selections=variant_payload.indicator_selections,
                signal_grids=template.signal_grids,
                risk_grid=template.risk_grid,
                direction_mode=variant_payload.direction_mode,
                sizing_mode=variant_payload.sizing_mode,
                risk_params=variant_payload.risk_params,
                execution_params=variant_payload.execution_params,
            ),
            target_slice=timeline.target_slice,
            target_time_range=target_time_range,
            artifact_context=artifact_context,
        )
        if getattr(scorer, "score_variant_with_details", None) is None:
            raise BacktestValidationError(
                "Variant-report requires scorer with deterministic details support"
            )
        details_scorer = cast(BacktestStagedVariantScorerWithDetails, scorer)
        signal_params = variant_payload.signal_params or {}
        risk_params = variant_payload.risk_params or {}
        execution_params = variant_payload.execution_params or {}

        indicator_variant_key = build_variant_key_v1(
            instrument_id=str(template.instrument_id),
            timeframe=template.timeframe.code,
            indicators=variant_payload.indicator_selections,
        )
        variant_key = build_backtest_variant_key_v1(
            indicator_variant_key=indicator_variant_key,
            direction_mode=variant_payload.direction_mode,
            sizing_mode=variant_payload.sizing_mode,
            signals=signal_params,
            risk_params=risk_params,
            execution_params=execution_params,
        )
        return details_scorer.score_variant_with_details(
            stage=STAGE_B_LITERAL_V2,
            candles=timeline.candles,
            indicator_selections=variant_payload.indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            indicator_variant_key=indicator_variant_key,
            variant_key=variant_key,
        )

    def _prepare_scorer_for_runtime_plan(
        self,
        *,
        scorer: MetricScorerV1,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        candles: Any,
        run_control: BacktestRunControlV1 | None,
    ) -> None:
        """
        Prepare scorer run context when the scorer exposes additive runtime-plan hooks.

        Args:
            scorer: Artifact-backed scorer implementation for the current run.
            runtime_plan: Deterministic artifact-backed runtime plan.
            candles: Warmup-inclusive request-timeframe candles.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            None.
        Assumptions:
            Optional scorer extension is discovered via method presence.
        Raises:
            Exception: Propagates scorer preparation errors.
        Side Effects:
            May populate scorer-local caches for prepared indicator row addressing.
        """
        prepare_method = getattr(scorer, "prepare_for_grid_context", None)
        if prepare_method is None:
            return
        prepare_method(
            grid_context=runtime_plan,
            candles=candles,
            max_compute_bytes_total=self._max_compute_bytes_total,
            run_control=run_control,
        )

    def _build_variant_previews(
        self,
        *,
        template: RunBacktestTemplate,
        ranked_rows: tuple[BacktestStageBScoredVariantV2, ...],
        ranked_tasks: Mapping[str, BacktestStageBTaskV2],
    ) -> tuple[BacktestVariantPreview, ...]:
        """
        Build summary-only variant previews from ranked artifact-backed Stage B rows.

        Args:
            template: Effective run template carrying stable direction/sizing/execution defaults.
            ranked_rows: Deterministically ranked Stage B rows.
            ranked_tasks: Deterministic `variant_key -> task` mapping for ranked rows.
        Returns:
            tuple[BacktestVariantPreview, ...]: Summary-only ranked variant previews.
        Assumptions:
            Runtime summary responses remain report/trades-free after R10-01.
        Raises:
            ValueError: If one ranked row has no matching Stage B task payload.
        Side Effects:
            None.
        """
        variants: list[BacktestVariantPreview] = []
        for row in ranked_rows:
            task = ranked_tasks.get(row.variant_key)
            if task is None:
                raise ValueError("missing Stage B task for ranked variant_key")
            variants.append(
                BacktestVariantPreview(
                    variant_index=row.variant_index,
                    variant_key=row.variant_key,
                    indicator_variant_key=row.indicator_variant_key,
                    total_return_pct=row.total_return_pct,
                    payload=BacktestVariantPayloadV1(
                        indicator_selections=task.indicator_selections,
                        signal_params=task.signal_params,
                        risk_params=task.risk_params,
                        execution_params=template.execution_params or {},
                        direction_mode=template.direction_mode,
                        sizing_mode=template.sizing_mode,
                    ),
                    report=None,
                    summary_metrics_json=row.summary_metrics_json,
                    best_tp_pct=row.best_tp_pct,
                    best_sl_pct=row.best_sl_pct,
                )
            )
        return tuple(variants)


def _requested_execution_profile_mode_from_payload_v2(
    *,
    request_payload: Mapping[str, Any] | None,
) -> ExecutionProfileModeLiteralV2 | None:
    """
    Resolve the internal-only execution-profile override from canonical request payload snapshot.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - apps/api/dto/backtests.py

    Args:
        request_payload: Optional strict API payload snapshot used by persisted/internal flows.
    Returns:
        ExecutionProfileModeLiteralV2 | None: Validated internal execution-profile mode
            override, or `None`.
    Assumptions:
        `execution_profile_mode` is internal metadata only; it is not part of the public
        `/backtests` request DTO and must remain excluded from request-hash semantics. Only this
        explicit caller-supplied payload snapshot may carry a requested live override; persisted
        job/read-model metadata must not be reinterpreted later by the worker.
    Raises:
        BacktestValidationError: If the internal override exists but is not a valid mode string.
    Side Effects:
        None.
    """
    if request_payload is None:
        return None
    raw_mode = request_payload.get("execution_profile_mode")
    if raw_mode is None:
        return None
    if not isinstance(raw_mode, str):
        raise BacktestValidationError(
            "internal execution_profile_mode override must be a string"
        )
    try:
        return validate_execution_profile_mode_v2(value=raw_mode)
    except ValueError as error:
        raise BacktestValidationError(
            "internal execution_profile_mode override is invalid"
        ) from error


def _merge_scalar_mappings(
    *,
    base: Mapping[str, int | float | str | bool | None],
    updates: Mapping[str, int | float | str | bool | None],
) -> Mapping[str, int | float | str | bool | None]:
    """
    Merge scalar mappings deterministically with update precedence and sorted keys.

    Args:
        base: Base scalar payload mapping.
        updates: Override scalar payload mapping.
    Returns:
        Mapping[str, int | float | str | bool | None]: Immutable merged scalar mapping.
    Assumptions:
        Input mappings use non-empty string keys and JSON-compatible scalar values.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    merged: dict[str, int | float | str | bool | None] = {}
    for raw_key in sorted(base.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = base[raw_key]
    for raw_key in sorted(updates.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = updates[raw_key]
    return MappingProxyType(merged)


def _merge_signal_grids(
    *,
    base: Mapping[str, Mapping[str, GridParamSpec]],
    updates: Mapping[str, Mapping[str, GridParamSpec]],
) -> Mapping[str, Mapping[str, GridParamSpec]]:
    """
    Merge nested signal-grid mappings deterministically by indicator id and param key.

    Args:
        base: Base signal-grid mapping loaded from saved strategy snapshot.
        updates: Saved-mode signal-grid overrides from request payload.
    Returns:
        Mapping[str, Mapping[str, object]]: Immutable merged nested mapping.
    Assumptions:
        Values are GridParamSpec-compatible objects validated by template DTO.
    Raises:
        ValueError: If one indicator id or param key is blank.
    Side Effects:
        None.
    """
    merged: dict[str, Mapping[str, GridParamSpec]] = {}
    indicator_ids = set(base.keys()) | set(updates.keys())
    for raw_indicator_id in sorted(indicator_ids, key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("saved-mode signal override indicator_id must be non-empty")
        merged_params: dict[str, GridParamSpec] = {}
        base_params = base.get(raw_indicator_id, {})
        updates_params = updates.get(raw_indicator_id, {})
        for raw_param_name in sorted(base_params.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("saved-mode signal override param key must be non-empty")
            merged_params[param_name] = base_params[raw_param_name]
        for raw_param_name in sorted(
            updates_params.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("saved-mode signal override param key must be non-empty")
            merged_params[param_name] = updates_params[raw_param_name]
        merged[indicator_id] = MappingProxyType(merged_params)
    return MappingProxyType(merged)


def _normalize_fee_defaults(
    *,
    values: Mapping[int, float] | None,
) -> Mapping[int, float]:
    """
    Normalize and validate runtime fee-default mapping by market id.

    Args:
        values: Optional mapping `market_id -> fee_pct`.
    Returns:
        Mapping[int, float]: Immutable normalized mapping.
    Assumptions:
        Fee values are human percent units and must be non-negative.
    Raises:
        ValueError: If one market id/fee value is invalid or mapping is empty.
    Side Effects:
        None.
    """
    source = _DEFAULT_FEE_PCT_BY_MARKET_ID if values is None else values
    normalized: dict[int, float] = {}
    for raw_market_id in sorted(source.keys()):
        market_id = int(raw_market_id)
        fee_pct = float(source[raw_market_id])
        if market_id <= 0:
            raise ValueError("fee_pct_default_by_market_id keys must be > 0")
        if fee_pct < 0.0:
            raise ValueError("fee_pct_default_by_market_id values must be >= 0")
        normalized[market_id] = fee_pct

    if len(normalized) == 0:
        raise ValueError("fee_pct_default_by_market_id must be non-empty")
    return MappingProxyType(normalized)


def _normalize_timeframe_literals(
    *,
    values: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """
    Normalize optional runtime timeframe-contract literals with stable first-seen order.

    Args:
        values: Optional tuple of raw timeframe literals.
    Returns:
        tuple[str, ...]: Normalized deduplicated lowercase timeframe literals.
    Assumptions:
        Caller owns semantic validation of timeframe values beyond normalization.
    Raises:
        ValueError: If one timeframe literal is blank.
    Side Effects:
        None.
    """
    if values is None:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value).strip().lower()
        if not value:
            raise ValueError("request timeframe literals must be non-empty")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _build_price_arrays_loader_v2(
    *,
    artifact_slot_resolver: BacktestArtifactSlotResolverV2,
) -> MmapPriceArraysLoaderV2:
    """
    Build explicit-path price loader from resolver wiring for artifact-backed runtime.

    Args:
        artifact_slot_resolver: Shared slot-pinned resolver wired at startup.
    Returns:
        MmapPriceArraysLoaderV2: Default mmap price loader for pinned artifact prices.
    Assumptions:
        Production sync/detail runtime must fail fast when resolver wiring is incomplete.
    Raises:
        ValueError: If resolver does not expose an artifact loader.
    Side Effects:
        None.
    """
    artifact_loader = getattr(artifact_slot_resolver, "artifact_loader", None)
    if artifact_loader is None:
        raise ValueError("artifact_slot_resolver must expose artifact_loader")
    return MmapPriceArraysLoaderV2(artifact_loader=artifact_loader)


def _runtime_plan_is_exact_no_risk_parity_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> bool:
    """
    Resolve whether the live runtime plan belongs to the first-class no-risk parity profile.

    Args:
        runtime_plan: Prepared artifact-backed runtime plan for the current sync execution.
    Returns:
        bool: `True` only when `execution_profile.mode` is `exact_no_risk_parity`.
    Assumptions:
        D3 canonical `NR2` orchestration must bypass `BacktestHierarchicalShortlistBuilderV2`
        entirely and keep reduced-plan semantics disabled before Stage A exact scoring.
    Raises:
        None.
    Side Effects:
        None.
    """
    execution_profile = getattr(runtime_plan, "execution_profile", None)
    if execution_profile is None:
        return False
    profile_mode = getattr(execution_profile, "mode", None)
    if profile_mode is None:
        return False
    return str(profile_mode).strip().lower() == "exact_no_risk_parity"


def _build_sync_persistence_artifact(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    shortlist: tuple[Any, ...],
) -> RunBacktestSyncPersistenceArtifact | None:
    """
    Build one internal sync persistence artifact from the already computed live Stage A state.

    Args:
        runtime_plan: Effective artifact-backed runtime plan used during the sync run.
        shortlist: Ordered live Stage A shortlist rows produced during the same sync execution.
    Returns:
        RunBacktestSyncPersistenceArtifact | None:
            Internal artifact for atomic terminal persistence, or `None` for non-parity runs.
    Assumptions:
        Only canonical `exact_no_risk_parity` sync runs persist `backtest_job_stage_a_shortlist`
        during terminal `sync_inline` writes.
    Raises:
        ValueError: If parity runtime state exists without the compact no-risk exact rows needed
            for backward-readable worker reuse.
    Side Effects:
        None.
    """
    parity_runtime_state = _parity_runtime_state_from_runtime_plan(runtime_plan=runtime_plan)
    if parity_runtime_state is None:
        return None
    if not _runtime_plan_uses_no_risk_terminal_path_v2(runtime_plan=runtime_plan):
        raise ValueError(
            "exact_no_risk_parity sync_inline persistence requires no-risk terminal runtime"
        )
    no_risk_exact_rows = persisted_stage_a_no_risk_exact_rows_v2(shortlist=shortlist)
    if no_risk_exact_rows is None:
        raise ValueError(
            "exact_no_risk_parity sync_inline persistence requires compact no-risk exact rows"
        )
    return RunBacktestSyncPersistenceArtifact(
        stage_a_indexes=tuple(row.base_variant.stage_a_index for row in shortlist),
        stage_a_variants_total=int(runtime_plan.stage_a_variants_total),
        risk_total=len(runtime_plan.risk_variants),
        preselect_used=len(shortlist),
        no_risk_exact_rows=no_risk_exact_rows,
        parity_runtime_state=parity_runtime_state,
    )


def _parity_runtime_state_from_runtime_plan(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> BacktestJobParityRuntimeState | None:
    """
    Project compact parity runtime state from the live sync runtime plan when available.

    Args:
        runtime_plan: Prepared artifact-backed runtime plan for the current sync execution.
    Returns:
        BacktestJobParityRuntimeState | None:
            Compact parity runtime state for `exact_no_risk_parity`, or `None` for other plans.
    Assumptions:
        Sync persistence must reuse the same worker-side parity contract rather than introducing
        one sync-only schema.
    Raises:
        ValueError: If a parity runtime plan omits required classification or counter metadata.
    Side Effects:
        None.
    """
    execution_profile_mode = str(runtime_plan.execution_profile.mode).strip().lower()
    if execution_profile_mode != "exact_no_risk_parity":
        return None

    raw_parity_classification = getattr(runtime_plan, "parity_classification", None)
    if raw_parity_classification is None:
        raise ValueError("exact_no_risk_parity runtime plan must expose parity_classification")
    raw_parity_runtime_counters_method = getattr(runtime_plan, "parity_runtime_counters", None)
    if raw_parity_runtime_counters_method is None or not callable(
        raw_parity_runtime_counters_method
    ):
        raise ValueError(
            "exact_no_risk_parity runtime plan must expose callable parity_runtime_counters"
        )
    raw_parity_runtime_counters = raw_parity_runtime_counters_method()
    if raw_parity_runtime_counters is None:
        raise ValueError(
            "exact_no_risk_parity runtime plan must expose parity_runtime_counters payload"
        )
    if not isinstance(raw_parity_runtime_counters, Mapping):
        raise ValueError(
            "exact_no_risk_parity runtime plan parity_runtime_counters must be mapping"
        )
    parity_runtime_counters = cast(
        Mapping[str, object],
        raw_parity_runtime_counters,
    )
    raw_retained_rows = parity_runtime_counters.get("retained_rows_per_indicator")
    if not isinstance(raw_retained_rows, Mapping) or len(raw_retained_rows) == 0:
        raise ValueError(
            "exact_no_risk_parity runtime plan must expose retained_rows_per_indicator"
        )
    stage_b_execution_mode = runtime_plan.stage_b_execution_mode()
    stage_b_process_fallback_threshold = runtime_plan.stage_b_process_fallback_threshold()
    return BacktestJobParityRuntimeState(
        execution_profile_mode=execution_profile_mode,
        parity_classification=BacktestJobParityClassification(
            parity_class=str(raw_parity_classification.parity_class),
            disabled_risk_single_cell=bool(raw_parity_classification.disabled_risk_single_cell),
            low_indicator_block_cardinality=bool(
                raw_parity_classification.low_indicator_block_cardinality
            ),
            narrowed_retained_row_evidence=bool(
                raw_parity_classification.narrowed_retained_row_evidence
            ),
            notebook_shaped_cost_units=bool(raw_parity_classification.notebook_shaped_cost_units),
            nr2_classification_reason=str(raw_parity_classification.nr2_classification_reason),
        ),
        retained_rows_per_indicator=tuple(
            BacktestJobParityRetainedRowsCounter(
                indicator_id=str(indicator_id),
                retained_rows=_coerce_positive_int(
                    name="retained_rows_per_indicator",
                    value=retained_rows,
                ),
            )
            for indicator_id, retained_rows in raw_retained_rows.items()
        ),
        retained_rows_total=_coerce_positive_int(
            name="retained_rows_total",
            value=parity_runtime_counters.get("retained_rows_total"),
        ),
        narrowed_combo_total=_coerce_positive_int(
            name="narrowed_combo_total",
            value=parity_runtime_counters.get("narrowed_combo_total"),
        ),
        narrowed_compute_combo_total=_coerce_positive_int(
            name="narrowed_compute_combo_total",
            value=parity_runtime_counters.get("narrowed_compute_combo_total"),
        ),
        no_risk_finalization_count=_coerce_positive_int(
            name="no_risk_finalization_count",
            value=parity_runtime_counters.get("no_risk_finalization_count"),
        ),
        exact_replay_count=_coerce_non_negative_int(
            name="exact_replay_count",
            value=parity_runtime_counters.get("exact_replay_count"),
        ),
        deterministic_combo_ordering=_coerce_non_empty_str(
            name="deterministic_combo_ordering",
            value=parity_runtime_counters.get("deterministic_combo_ordering"),
        ),
        stage_b_execution_mode=_coerce_non_empty_str(
            name="stage_b_execution_mode",
            value=stage_b_execution_mode,
        ),
        stage_b_process_fallback_threshold=_coerce_non_empty_str(
            name="stage_b_process_fallback_threshold",
            value=stage_b_process_fallback_threshold,
        ),
    )


def _coerce_positive_int(*, name: str, value: object) -> int:
    """
    Coerce one persisted parity counter into a positive integer.

    Args:
        name: Counter name used in deterministic error messages.
        value: Raw counter payload from runtime-plan metadata.
    Returns:
        int: Positive integer value.
    Assumptions:
        Parity runtime counters are scalar numeric values produced by the live runtime plan.
    Raises:
        ValueError: If value is not an integer greater than zero.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"exact_no_risk_parity runtime plan {name} must be positive integer")
    return value


def _coerce_non_negative_int(*, name: str, value: object) -> int:
    """
    Coerce one persisted parity counter into a non-negative integer.

    Args:
        name: Counter name used in deterministic error messages.
        value: Raw counter payload from runtime-plan metadata.
    Returns:
        int: Non-negative integer value.
    Assumptions:
        `exact_replay_count` is additive metadata and may legitimately be zero.
    Raises:
        ValueError: If value is not an integer greater than or equal to zero.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"exact_no_risk_parity runtime plan {name} must be non-negative integer"
        )
    return value


def _coerce_non_empty_str(*, name: str, value: object) -> str:
    """
    Coerce one persisted parity metadata literal into a non-empty string.

    Args:
        name: Field name used in deterministic error messages.
        value: Raw runtime-plan metadata value.
    Returns:
        str: Stripped non-empty string.
    Assumptions:
        Runtime-plan metadata already uses stable string literals for persisted parity evidence.
    Raises:
        ValueError: If value is not a non-empty string literal.
    Side Effects:
        None.
    """
    if not isinstance(value, str):
        raise ValueError(f"exact_no_risk_parity runtime plan {name} must be string")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"exact_no_risk_parity runtime plan {name} must be non-empty")
    return normalized_value


def _cancel_checker_from_run_control(
    *,
    run_control: BacktestRunControlV1 | None,
) -> Any:
    """
    Convert optional run control into a stage-aware cancel-checker callback.

    Args:
        run_control: Optional cooperative cancellation/deadline control object.
    Returns:
        Any: `None` when no run control is supplied, otherwise a stage checker callback.
    Assumptions:
        Callback shape matches Stage A/Stage B runtime helper expectations.
    Raises:
        None.
    Side Effects:
        None.
    """
    if run_control is None:
        return None

    def _checker(stage: str) -> None:
        """
        Raise when the shared run control marks the current stage as cancelled.

        Args:
            stage: Current stage literal.
        Returns:
            None.
        Assumptions:
            Caller provides stable stage literals owned by the backtest runtime.
        Raises:
            BacktestRunCancelledV1: Propagated by the shared run control when cancelled.
        Side Effects:
            None.
        """
        run_control.raise_if_cancelled(stage=stage)

    return _checker


def _build_saved_spec_payload(
    *,
    snapshot: BacktestStrategySnapshot | None,
) -> dict[str, Any]:
    """
    Extract deterministic saved-strategy spec payload used by persisted sync runs.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    Args:
        snapshot: Loaded saved strategy snapshot used for the current sync run.
    Returns:
        dict[str, Any]: Deterministic spec payload copied from the saved strategy snapshot.
    Assumptions:
        Persisted sync runs require the same saved spec snapshot that was used for execution.
    Raises:
        BacktestValidationError: If the snapshot does not carry a non-empty `spec_payload`.
    Side Effects:
        None.
    """
    spec_payload = dict(snapshot.spec_payload or {}) if snapshot is not None else {}
    if len(spec_payload) == 0:
        raise BacktestValidationError(
            "saved mode backtest requires non-empty strategy spec payload"
        )
    return spec_payload


def _build_sha256_from_payload(*, payload: Mapping[str, Any]) -> str:
    """
    Build deterministic SHA-256 hash from canonical JSON representation.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
    Args:
        payload: JSON-compatible mapping payload.
    Returns:
        str: Lowercase SHA-256 hex hash string.
    Assumptions:
        Canonical JSON uses sorted keys and compact separators.
    Raises:
        TypeError: If payload contains unsupported non-JSON values.
    Side Effects:
        None.
    """
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
