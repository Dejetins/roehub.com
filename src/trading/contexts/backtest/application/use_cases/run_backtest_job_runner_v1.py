from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, cast
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1,
    BacktestRankingConfig,
    BacktestRequestScalar,
    BacktestRiskGridSpec,
    RunBacktestRequest,
    RunBacktestSavedOverrides,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestJobLeaseRepository,
    BacktestJobRepository,
    BacktestJobRequestDecoder,
    BacktestJobResultsRepository,
    BacktestStagedVariantMetricScorer,
    BacktestStagedVariantScorer,
)
from trading.contexts.backtest.application.services import (
    STAGE_A_LITERAL,
    STAGE_B_LITERAL,
    BacktestCandleTimelineBuilder,
    BacktestGridBuilderV1,
    BacktestReportingServiceV1,
    BacktestStageABaseVariant,
    CloseFillBacktestStagedScorerV1,
)
from trading.contexts.backtest.application.services.job_runner_streaming_v1 import (
    BacktestJobSnapshotCadenceV1,
    BacktestJobTopVariantCandidateV1,
    FrontierSignatureV1,
    build_running_snapshot_rows,
)
from trading.contexts.backtest.application.services.numba_runtime_v1 import (
    apply_backtest_numba_threads,
)
from trading.contexts.backtest.application.services.staged_core_runner_v1 import (
    BacktestStageAScoredVariantV1,
    BacktestStageBScoredVariantV1,
    BacktestStageBTaskV1,
    BacktestStagedCoreRunnerV1,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
)
from trading.contexts.backtest.domain.value_objects import BacktestVariantScalar
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    RangeValuesSpec,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe

_LOG = logging.getLogger(__name__)

BacktestJobRunStatus = Literal["succeeded", "failed", "cancelled", "lease_lost"]
_DEFAULT_MAX_NUMBA_THREADS = max(1, os.cpu_count() or 1)
MetricScorerV1 = BacktestStagedVariantMetricScorer | BacktestStagedVariantScorer


@dataclass(frozen=True, slots=True)
class BacktestJobRunReportV1:
    """
    Deterministic single-attempt processing report for one claimed Backtest job.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    job_id: UUID
    attempt: int
    status: BacktestJobRunStatus
    stage_a_duration_seconds: float = 0.0
    stage_b_duration_seconds: float = 0.0
    finalizing_duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate report payload scalar invariants for metrics/logging consumers.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stage durations are monotonic non-negative wall-clock measurements.
        Raises:
            ValueError: If report status or duration values are invalid.
        Side Effects:
            None.
        """
        if self.status not in {"succeeded", "failed", "cancelled", "lease_lost"}:
            raise ValueError("BacktestJobRunReportV1.status is unsupported")
        if self.attempt < 0:
            raise ValueError("BacktestJobRunReportV1.attempt must be >= 0")
        if self.stage_a_duration_seconds < 0.0:
            raise ValueError("BacktestJobRunReportV1.stage_a_duration_seconds must be >= 0")
        if self.stage_b_duration_seconds < 0.0:
            raise ValueError("BacktestJobRunReportV1.stage_b_duration_seconds must be >= 0")
        if self.finalizing_duration_seconds < 0.0:
            raise ValueError("BacktestJobRunReportV1.finalizing_duration_seconds must be >= 0")


@dataclass(frozen=True, slots=True)
class _ResolvedJobRequestContext:
    """
    Internal resolved run settings for one claimed job attempt.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
    """

    request: RunBacktestRequest
    template: RunBacktestTemplate
    warmup_bars: int
    top_k: int
    preselect: int
    top_trades_n: int
    persisted_k: int
    ranking: BacktestRankingConfig


class RunBacktestJobRunnerV1:
    """
    Execute one claimed Backtest job attempt via streaming Stage-A/Stage-B/finalizing flow.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        lease_repository: BacktestJobLeaseRepository,
        results_repository: BacktestJobResultsRepository,
        request_decoder: BacktestJobRequestDecoder,
        candle_timeline_builder: BacktestCandleTimelineBuilder,
        indicator_compute: IndicatorCompute,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        grid_builder: BacktestGridBuilderV1 | None = None,
        reporting_service: BacktestReportingServiceV1 | None = None,
        core_runner: BacktestStagedCoreRunnerV1 | None = None,
        staged_scorer: MetricScorerV1 | None = None,
        warmup_bars_default: int = 200,
        top_k_default: int = 300,
        preselect_default: int = 20000,
        top_trades_n_default: int = 3,
        ranking_primary_metric_default: str = BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
        ranking_secondary_metric_default: str | None = (
            BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1
        ),
        configurable_ranking_enabled: bool = True,
        top_k_persisted_default: int = 300,
        init_cash_quote_default: float = 10000.0,
        fixed_quote_default: float = 100.0,
        safe_profit_percent_default: float = 30.0,
        slippage_pct_default: float = 0.01,
        fee_pct_default_by_market_id: Mapping[int, float] | None = None,
        max_variants_per_compute: int = 600_000,
        max_compute_bytes_total: int = 5 * 1024**3,
        lease_seconds: int = 60,
        heartbeat_seconds: int = 15,
        snapshot_seconds: int | None = None,
        snapshot_variants_step: int | None = None,
        stage_batch_size: int = 256,
        max_numba_threads: int = _DEFAULT_MAX_NUMBA_THREADS,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Initialize claimed-job streaming orchestrator dependencies and runtime policies.

        Args:
            job_repository: Core job repository used for cancel checks.
            lease_repository: Lease-guarded write repository.
            results_repository: Snapshot/shortlist repository.
            request_decoder: Decoder for persisted `request_json`.
            candle_timeline_builder: Candle timeline builder used by compute pipeline.
            indicator_compute: Indicator compute port for staged grid estimates/scoring.
            defaults_provider: Optional grid defaults provider.
            grid_builder: Optional custom staged grid builder.
            reporting_service: Optional report assembly service for finalizing step.
            core_runner: Shared staged scoring core used by sync and job-runner paths.
            staged_scorer: Optional custom staged scorer.
            warmup_bars_default: Runtime default warmup bars.
            top_k_default: Runtime default top-k request fallback.
            preselect_default: Runtime default Stage-A shortlist size.
            top_trades_n_default: Runtime default trades payload cap in finalizing.
            ranking_primary_metric_default:
                Runtime default for ranking primary metric literal.
            ranking_secondary_metric_default:
                Runtime default for ranking secondary metric literal.
            configurable_ranking_enabled:
                Feature-flag guard for configurable ranking behavior rollout.
            top_k_persisted_default: Persisted rows cap for worker snapshots/finalizing.
            init_cash_quote_default: Runtime execution default.
            fixed_quote_default: Runtime execution default.
            safe_profit_percent_default: Runtime execution default.
            slippage_pct_default: Runtime execution default.
            fee_pct_default_by_market_id: Runtime execution defaults mapping.
            max_variants_per_compute: Deterministic variants guard.
            max_compute_bytes_total: Deterministic memory guard.
            lease_seconds: Lease extension value used for heartbeat.
            heartbeat_seconds: Heartbeat cadence in seconds.
            snapshot_seconds: Optional time-based Stage-B snapshot trigger.
            snapshot_variants_step: Optional processed-count snapshot trigger.
            stage_batch_size: Batch boundary size for cancel/progress checks.
            max_numba_threads:
                Runtime CPU knob for jobs mapped to maximum Numba threads.
            now_provider: Optional UTC-aware current time provider for deterministic tests.
        Returns:
            None.
        Assumptions:
            Constructor performs dependency wiring and invariant checks only.
        Raises:
            ValueError: If dependencies or scalar runtime settings are invalid.
        Side Effects:
            None.
        """
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires job_repository")
        if lease_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires lease_repository")
        if results_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires results_repository")
        if request_decoder is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires request_decoder")
        if candle_timeline_builder is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires candle_timeline_builder")
        if indicator_compute is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestJobRunnerV1 requires indicator_compute")
        if warmup_bars_default <= 0:
            raise ValueError("warmup_bars_default must be > 0")
        if top_k_default <= 0:
            raise ValueError("top_k_default must be > 0")
        if preselect_default <= 0:
            raise ValueError("preselect_default must be > 0")
        if top_trades_n_default <= 0:
            raise ValueError("top_trades_n_default must be > 0")
        if top_k_persisted_default <= 0:
            raise ValueError("top_k_persisted_default must be > 0")
        if init_cash_quote_default <= 0.0:
            raise ValueError("init_cash_quote_default must be > 0")
        if fixed_quote_default <= 0.0:
            raise ValueError("fixed_quote_default must be > 0")
        if safe_profit_percent_default < 0.0 or safe_profit_percent_default > 100.0:
            raise ValueError("safe_profit_percent_default must be in [0, 100]")
        if slippage_pct_default < 0.0:
            raise ValueError("slippage_pct_default must be >= 0")
        if max_variants_per_compute <= 0:
            raise ValueError("max_variants_per_compute must be > 0")
        if max_compute_bytes_total <= 0:
            raise ValueError("max_compute_bytes_total must be > 0")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be > 0")
        if stage_batch_size <= 0:
            raise ValueError("stage_batch_size must be > 0")
        if max_numba_threads <= 0:
            raise ValueError("max_numba_threads must be > 0")
        if not isinstance(configurable_ranking_enabled, bool):
            raise ValueError("configurable_ranking_enabled must be bool")

        ranking_defaults = BacktestRankingConfig(
            primary_metric=ranking_primary_metric_default,
            secondary_metric=ranking_secondary_metric_default,
        )

        self._job_repository = job_repository
        self._lease_repository = lease_repository
        self._results_repository = results_repository
        self._request_decoder = request_decoder
        self._candle_timeline_builder = candle_timeline_builder
        self._indicator_compute = indicator_compute
        self._defaults_provider = defaults_provider
        self._grid_builder = grid_builder or BacktestGridBuilderV1()
        self._reporting_service = reporting_service or BacktestReportingServiceV1()
        self._core_runner = core_runner or BacktestStagedCoreRunnerV1(
            batch_size_default=stage_batch_size,
            configurable_ranking_enabled=configurable_ranking_enabled,
        )
        self._staged_scorer = staged_scorer
        self._warmup_bars_default = warmup_bars_default
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default
        self._top_trades_n_default = top_trades_n_default
        self._ranking_defaults = ranking_defaults
        self._configurable_ranking_enabled = configurable_ranking_enabled
        self._top_k_persisted_default = top_k_persisted_default
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
        self._max_variants_per_compute = max_variants_per_compute
        self._max_compute_bytes_total = max_compute_bytes_total
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._snapshot_cadence = BacktestJobSnapshotCadenceV1(
            snapshot_seconds=snapshot_seconds,
            snapshot_variants_step=snapshot_variants_step,
        )
        self._stage_batch_size = stage_batch_size
        self._max_numba_threads = max_numba_threads
        self._now = now_provider or _utc_now

    def process_claimed_job(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
    ) -> BacktestJobRunReportV1:
        """
        Process one already-claimed running job attempt until terminal or lease-loss state.

        Args:
            job: Claimed job snapshot returned by `claim_next(...)`.
            locked_by: Active lease owner literal.
        Returns:
            BacktestJobRunReportV1: Deterministic processing report for this attempt.
        Assumptions:
            Job was atomically claimed under `locked_by` before method invocation.
        Raises:
            ValueError: If claimed job snapshot/state is invalid.
        Side Effects:
            Performs lease-guarded job writes and staged compute work.
        """
        if job.state != "running":
            raise ValueError("process_claimed_job requires running claimed job")
        normalized_locked_by = locked_by.strip()
        if not normalized_locked_by:
            raise ValueError("process_claimed_job requires non-empty locked_by")

        stage_durations: dict[str, float] = {
            STAGE_A_LITERAL: 0.0,
            STAGE_B_LITERAL: 0.0,
            "finalizing": 0.0,
        }
        stage_started_at = perf_counter()
        current_stage = STAGE_A_LITERAL

        try:
            apply_backtest_numba_threads(max_numba_threads=self._max_numba_threads)
            context = self._resolve_request_context(job=job)
            timeline = self._candle_timeline_builder.build(
                market_id=context.template.instrument_id.market_id,
                symbol=context.template.instrument_id.symbol,
                timeframe=context.template.timeframe,
                requested_time_range=context.request.time_range,
                warmup_bars=context.warmup_bars,
            )
            scorer = self._resolve_staged_scorer(
                template=context.template,
                target_slice=timeline.target_slice,
            )
            grid_context = self._grid_builder.build(
                template=context.template,
                candles=timeline.candles,
                indicator_compute=self._indicator_compute,
                preselect=context.preselect,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
            )
            self._prepare_scorer_for_grid_context(
                scorer=scorer,
                grid_context=grid_context,
                candles=timeline.candles,
            )

            shortlist, heartbeat_at = self._run_stage_a(
                job=job,
                locked_by=normalized_locked_by,
                context=context,
                timeline=timeline,
                scorer=scorer,
                grid_context=grid_context,
            )
            stage_durations[STAGE_A_LITERAL] = max(perf_counter() - stage_started_at, 0.0)

            current_stage = STAGE_B_LITERAL
            stage_started_at = perf_counter()
            heartbeat_at = self._run_stage_b(
                job=job,
                locked_by=normalized_locked_by,
                context=context,
                timeline=timeline,
                scorer=scorer,
                grid_context=grid_context,
                shortlist=shortlist,
                last_heartbeat_at=heartbeat_at,
            )
            stage_durations[STAGE_B_LITERAL] = max(perf_counter() - stage_started_at, 0.0)

            current_stage = "finalizing"
            stage_started_at = perf_counter()
            self._run_finalizing(
                job=job,
                locked_by=normalized_locked_by,
                last_heartbeat_at=heartbeat_at,
            )
            stage_durations["finalizing"] = max(perf_counter() - stage_started_at, 0.0)
            return BacktestJobRunReportV1(
                job_id=job.job_id,
                attempt=job.attempt,
                status="succeeded",
                stage_a_duration_seconds=stage_durations[STAGE_A_LITERAL],
                stage_b_duration_seconds=stage_durations[STAGE_B_LITERAL],
                finalizing_duration_seconds=stage_durations["finalizing"],
            )
        except _BacktestJobCancelled:
            stage_durations[current_stage] = max(
                stage_durations[current_stage],
                perf_counter() - stage_started_at,
            )
            return BacktestJobRunReportV1(
                job_id=job.job_id,
                attempt=job.attempt,
                status="cancelled",
                stage_a_duration_seconds=stage_durations[STAGE_A_LITERAL],
                stage_b_duration_seconds=stage_durations[STAGE_B_LITERAL],
                finalizing_duration_seconds=stage_durations["finalizing"],
            )
        except _BacktestJobLeaseLost:
            stage_durations[current_stage] = max(
                stage_durations[current_stage],
                perf_counter() - stage_started_at,
            )
            return BacktestJobRunReportV1(
                job_id=job.job_id,
                attempt=job.attempt,
                status="lease_lost",
                stage_a_duration_seconds=stage_durations[STAGE_A_LITERAL],
                stage_b_duration_seconds=stage_durations[STAGE_B_LITERAL],
                finalizing_duration_seconds=stage_durations["finalizing"],
            )
        except Exception as error:  # noqa: BLE001
            stage_durations[current_stage] = max(
                stage_durations[current_stage],
                perf_counter() - stage_started_at,
            )
            _LOG.exception(
                "event=job_failed stage=%s job_id=%s attempt=%s locked_by=%s",
                current_stage,
                job.job_id,
                job.attempt,
                normalized_locked_by,
            )
            self._finish_failed(
                job=job,
                locked_by=normalized_locked_by,
                stage=current_stage,
                error=error,
            )
            return BacktestJobRunReportV1(
                job_id=job.job_id,
                attempt=job.attempt,
                status="failed",
                stage_a_duration_seconds=stage_durations[STAGE_A_LITERAL],
                stage_b_duration_seconds=stage_durations[STAGE_B_LITERAL],
                finalizing_duration_seconds=stage_durations["finalizing"],
            )

    def _resolve_request_context(
        self,
        *,
        job: BacktestJob,
    ) -> _ResolvedJobRequestContext:
        """
        Decode persisted request payload and resolve effective run defaults/persistence caps.

        Args:
            job: Claimed running job snapshot.
        Returns:
            _ResolvedJobRequestContext: Resolved deterministic run context.
        Assumptions:
            Persisted request payload is canonical and reproducible.
        Raises:
            ValueError: If payload is invalid or missing required template semantics.
        Side Effects:
            None.
        """
        request = self._request_decoder.decode(payload=job.request_json)
        template = self._resolve_template(job=job, request=request)
        warmup_bars = self._resolve_positive_override(
            value=request.warmup_bars,
            default=self._warmup_bars_default,
        )
        top_k = self._resolve_positive_override(
            value=request.top_k,
            default=self._top_k_default,
        )
        preselect = self._resolve_positive_override(
            value=request.preselect,
            default=self._preselect_default,
        )
        top_trades_n = self._resolve_positive_override(
            value=request.top_trades_n,
            default=self._top_trades_n_default,
        )
        ranking = self._resolve_ranking_config(request=request)
        if top_trades_n > top_k:
            top_trades_n = top_k

        return _ResolvedJobRequestContext(
            request=request,
            template=template,
            warmup_bars=warmup_bars,
            top_k=top_k,
            preselect=preselect,
            top_trades_n=top_trades_n,
            persisted_k=min(top_k, self._top_k_persisted_default),
            ranking=ranking,
        )

    def _resolve_template(
        self,
        *,
        job: BacktestJob,
        request: RunBacktestRequest,
    ) -> RunBacktestTemplate:
        """
        Resolve effective run template from request payload with saved-mode snapshot fallback.

        Args:
            job: Claimed job snapshot.
            request: Decoded request DTO.
        Returns:
            RunBacktestTemplate: Effective deterministic template for compute stages.
        Assumptions:
            Saved-mode request can omit template when snapshot payload is available.
        Raises:
            ValueError: If effective template cannot be resolved.
        Side Effects:
            None.
        """
        if request.template is not None:
            return request.template
        if job.mode != "saved":
            raise ValueError("template mode request_json must include template payload")
        if job.spec_payload_json is None:
            raise ValueError("saved mode job requires spec_payload_json when template is absent")

        base_template = _template_from_saved_spec_payload(spec_payload=job.spec_payload_json)
        return _apply_saved_overrides(
            base_template=base_template,
            overrides=request.overrides,
        )

    def _resolve_positive_override(self, *, value: int | None, default: int) -> int:
        """
        Resolve optional positive override value against runtime default.

        Args:
            value: Optional override.
            default: Runtime default fallback.
        Returns:
            int: Effective positive integer value.
        Assumptions:
            Runtime defaults are validated in constructor.
        Raises:
            ValueError: If override is non-positive.
        Side Effects:
            None.
        """
        if value is None:
            return default
        if value <= 0:
            raise ValueError("request override must be > 0")
        return value

    def _resolve_ranking_config(self, *, request: RunBacktestRequest) -> BacktestRankingConfig:
        """
        Resolve effective ranking config from request override, runtime defaults, and feature flag.

        Args:
            request: Decoded job run request payload.
        Returns:
            BacktestRankingConfig: Effective deterministic ranking config.
        Assumptions:
            Request DTO already validates ranking metric literals and duplicate checks.
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

    def _prepare_scorer_for_grid_context(
        self,
        *,
        scorer: MetricScorerV1,
        grid_context: Any,
        candles: Any,
    ) -> None:
        """
        Prepare scorer run context (batched indicator tensors) when extension is available.

        Args:
            scorer: Staged scorer implementation.
            grid_context: Prepared staged grid context.
            candles: Warmup-inclusive candle arrays.
        Returns:
            None.
        Assumptions:
            Optional scorer extension is discovered by method presence.
        Raises:
            Exception: Propagates scorer preparation errors.
        Side Effects:
            May materialize batched indicator tensors in scorer local cache.
        """
        prepare_method = getattr(scorer, "prepare_for_grid_context", None)
        if prepare_method is None:
            return
        prepare_method(
            grid_context=grid_context,
            candles=candles,
            max_compute_bytes_total=self._max_compute_bytes_total,
            run_control=None,
        )

    def _run_stage_a(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        context: _ResolvedJobRequestContext,
        timeline: Any,
        scorer: MetricScorerV1,
        grid_context: Any,
    ) -> tuple[tuple[BacktestJobTopVariantCandidateV1, ...], datetime]:
        """
        Execute streaming Stage-A shortlist build with deterministic ranking/persistence.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            context: Resolved request context.
            timeline: Built candle timeline payload.
            scorer: Deterministic staged scorer.
            grid_context: Prepared staged grid context.
        Returns:
            tuple[tuple[BacktestJobTopVariantCandidateV1, ...], datetime]:
                Ranked Stage-A shortlist candidates and last heartbeat timestamp.
        Assumptions:
            Stage-A final deterministic tie-break remains `base_variant_key ASC`.
        Raises:
            _BacktestJobCancelled: If cancel was requested before stage completion.
            _BacktestJobLeaseLost: If one lease-guarded write fails.
            ValueError: If scoring payload is invalid.
        Side Effects:
            Writes stage progress and Stage-A shortlist snapshot under lease guard.
        """
        stage_total = int(grid_context.stage_a_variants_total)
        stage_limit = min(context.preselect, stage_total)
        now = self._now()
        self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_A_LITERAL)
        self._update_progress(
            job=job,
            locked_by=locked_by,
            stage=STAGE_A_LITERAL,
            processed_units=0,
            total_units=stage_total,
            now=now,
        )
        heartbeat_at = now

        def _on_stage_a_checkpoint(processed: int, total: int) -> None:
            nonlocal heartbeat_at
            now = self._now()
            self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_A_LITERAL)
            heartbeat_at = self._heartbeat_if_due(
                job=job,
                locked_by=locked_by,
                now=now,
                last_heartbeat_at=heartbeat_at,
            )
            self._update_progress(
                job=job,
                locked_by=locked_by,
                stage=STAGE_A_LITERAL,
                processed_units=processed,
                total_units=total,
                now=now,
            )

        shortlist_rows = self._core_runner.run_stage_a(
            grid_context=grid_context,
            candles=timeline.candles,
            scorer=scorer,
            shortlist_limit=stage_limit,
            ranking=context.ranking,
            batch_size=self._stage_batch_size,
            on_checkpoint=_on_stage_a_checkpoint,
        )
        shortlist = tuple(
            BacktestJobTopVariantCandidateV1(
                variant_index=row.base_variant.stage_a_index,
                variant_key=row.base_variant.base_variant_key,
                indicator_variant_key=row.base_variant.indicator_variant_key,
                total_return_pct=row.total_return_pct,
                indicator_selections=row.base_variant.indicator_selections,
                signal_params=row.base_variant.signal_params,
                risk_params={
                    "sl_enabled": False,
                    "sl_pct": None,
                    "tp_enabled": False,
                    "tp_pct": None,
                },
            )
            for row in shortlist_rows
        )

        now = self._now()
        self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_A_LITERAL)
        stage_a_shortlist = BacktestJobStageAShortlist(
            job_id=job.job_id,
            stage_a_indexes=tuple(item.variant_index for item in shortlist),
            stage_a_variants_total=stage_total,
            risk_total=len(grid_context.risk_variants),
            preselect_used=len(shortlist),
            updated_at=now,
        )
        saved = self._results_repository.save_stage_a_shortlist(
            job_id=job.job_id,
            now=now,
            locked_by=locked_by,
            shortlist=stage_a_shortlist,
        )
        if not saved:
            raise _BacktestJobLeaseLost()

        return (shortlist, heartbeat_at)

    def _run_stage_b(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        context: _ResolvedJobRequestContext,
        timeline: Any,
        scorer: MetricScorerV1,
        grid_context: Any,
        shortlist: tuple[BacktestJobTopVariantCandidateV1, ...],
        last_heartbeat_at: datetime,
    ) -> datetime:
        """
        Execute streaming Stage-B scoring with frontier-signature-gated snapshot persistence.

        Docs:
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            context: Resolved request context.
            timeline: Built candle timeline payload.
            scorer: Deterministic staged scorer.
            grid_context: Prepared staged grid context.
            shortlist: Stage-A shortlisted base variants.
            last_heartbeat_at: Last successful heartbeat timestamp.
        Returns:
            datetime: Updated heartbeat timestamp.
        Assumptions:
            Stage-B final deterministic tie-break remains `variant_key ASC`.
        Raises:
            _BacktestJobCancelled: If cancel was requested during stage execution.
            _BacktestJobLeaseLost: If one lease-guarded write fails.
            ValueError: If scoring payload is invalid.
        Side Effects:
            Writes progress and top-variants snapshots under active lease guard.
        """
        stage_total = int(grid_context.stage_b_variants_total)
        now = self._now()
        self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_B_LITERAL)
        self._update_progress(
            job=job,
            locked_by=locked_by,
            stage=STAGE_B_LITERAL,
            processed_units=0,
            total_units=stage_total,
            now=now,
        )
        heartbeat_at = last_heartbeat_at
        processed = 0
        last_snapshot_at = now
        last_snapshot_processed = 0
        last_persisted_frontier_signature: FrontierSignatureV1 | None = None
        skipped_snapshot_writes = 0
        stage_a_shortlist = tuple(
            BacktestStageAScoredVariantV1(
                base_variant=BacktestStageABaseVariant(
                    stage_a_index=item.variant_index,
                    indicator_selections=item.indicator_selections,
                    signal_params=item.signal_params,
                    indicator_variant_key=item.indicator_variant_key,
                    base_variant_key=item.variant_key,
                ),
                total_return_pct=item.total_return_pct,
            )
            for item in shortlist
        )

        def _on_stage_b_checkpoint(
            checkpoint_processed: int,
            checkpoint_total: int,
            materialize_ranked_rows: Callable[
                [],
                tuple[BacktestStageBScoredVariantV1, ...],
            ],
            materialize_stage_b_tasks: Callable[
                [],
                Mapping[str, BacktestStageBTaskV1],
            ],
        ) -> None:
            nonlocal heartbeat_at
            nonlocal processed
            nonlocal last_snapshot_at
            nonlocal last_snapshot_processed
            nonlocal last_persisted_frontier_signature
            nonlocal skipped_snapshot_writes
            processed = checkpoint_processed
            now_local = self._now()
            self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_B_LITERAL)
            heartbeat_at = self._heartbeat_if_due(
                job=job,
                locked_by=locked_by,
                now=now_local,
                last_heartbeat_at=heartbeat_at,
            )
            self._update_progress(
                job=job,
                locked_by=locked_by,
                stage=STAGE_B_LITERAL,
                processed_units=checkpoint_processed,
                total_units=checkpoint_total,
                now=now_local,
            )
            if not self._snapshot_cadence.should_persist(
                now=now_local,
                last_persist_at=last_snapshot_at,
                processed_variants=checkpoint_processed,
                last_persist_processed_variants=last_snapshot_processed,
            ):
                return
            ranked_rows = materialize_ranked_rows()
            frontier_signature = _frontier_signature_from_ranked_rows(ranked_rows=ranked_rows)
            if frontier_signature == last_persisted_frontier_signature:
                skipped_snapshot_writes += 1
                last_snapshot_at = now_local
                last_snapshot_processed = checkpoint_processed
                return
            ranked_candidates = _ranked_candidates_from_core_rows(
                ranked_rows=ranked_rows,
                tasks_by_variant_key=materialize_stage_b_tasks(),
            )
            self._persist_running_snapshot(
                job=job,
                locked_by=locked_by,
                context=context,
                ranked_candidates=ranked_candidates,
                now=now_local,
            )
            last_persisted_frontier_signature = frontier_signature
            last_snapshot_at = now_local
            last_snapshot_processed = checkpoint_processed

        ranked_rows, ranked_tasks = self._core_runner.run_stage_b(
            template=context.template,
            grid_context=grid_context,
            shortlist=stage_a_shortlist,
            candles=timeline.candles,
            scorer=scorer,
            top_k_limit=context.persisted_k,
            ranking=context.ranking,
            batch_size=self._stage_batch_size,
            on_checkpoint=_on_stage_b_checkpoint,
        )

        now = self._now()
        self._ensure_not_cancelled(job=job, locked_by=locked_by, stage=STAGE_B_LITERAL)
        final_frontier_signature = _frontier_signature_from_ranked_rows(ranked_rows=ranked_rows)
        if final_frontier_signature != last_persisted_frontier_signature:
            ranked_candidates = _ranked_candidates_from_core_rows(
                ranked_rows=ranked_rows,
                tasks_by_variant_key=ranked_tasks,
            )
            self._persist_running_snapshot(
                job=job,
                locked_by=locked_by,
                context=context,
                ranked_candidates=ranked_candidates,
                now=now,
            )
        elif processed != last_snapshot_processed:
            skipped_snapshot_writes += 1

        if skipped_snapshot_writes > 0:
            _LOG.debug(
                "event=job_snapshot_skipped_frontier_unchanged "
                "job_id=%s attempt=%s skipped_writes=%s",
                job.job_id,
                job.attempt,
                skipped_snapshot_writes,
            )
        return heartbeat_at

    def _run_finalizing(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        last_heartbeat_at: datetime,
    ) -> None:
        """
        Execute finalizing stage without eager report/trades generation.

        Docs:
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
          - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
          - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            last_heartbeat_at: Last successful heartbeat timestamp.
        Returns:
            None.
        Assumptions:
            Finalizing runs only when job is still active and cancel was not requested.
        Raises:
            _BacktestJobCancelled: If cancel was requested before or during finalizing.
            _BacktestJobLeaseLost: If one lease-guarded write fails.
        Side Effects:
            Writes final stage progress and terminal `succeeded` state.
        """
        now = self._now()
        self._ensure_not_cancelled(job=job, locked_by=locked_by, stage="finalizing")
        self._heartbeat_if_due(
            job=job,
            locked_by=locked_by,
            now=now,
            last_heartbeat_at=last_heartbeat_at,
        )
        self._update_progress(
            job=job,
            locked_by=locked_by,
            stage="finalizing",
            processed_units=0,
            total_units=1,
            now=now,
        )

        finished = self._lease_repository.finish(
            job_id=job.job_id,
            now=self._now(),
            locked_by=locked_by,
            next_state="succeeded",
        )
        if finished is None:
            raise _BacktestJobLeaseLost()

    def _persist_running_snapshot(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        context: _ResolvedJobRequestContext,
        ranked_candidates: tuple[BacktestJobTopVariantCandidateV1, ...],
        now: datetime,
    ) -> None:
        """
        Persist deterministic running top snapshot with `report_table_md=NULL`.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            context: Resolved request context.
            ranked_candidates: Current ranked persisted candidates.
            now: Snapshot timestamp.
        Returns:
            None.
        Assumptions:
            Running snapshots do not store report markdown or trades payload.
        Raises:
            _BacktestJobLeaseLost: If lease-guarded replace write fails.
        Side Effects:
            Replaces top variants snapshot rows in persistence layer.
        """
        rows = build_running_snapshot_rows(
            job_id=job.job_id,
            now=now,
            ranked_candidates=ranked_candidates,
            direction_mode=context.template.direction_mode,
            sizing_mode=context.template.sizing_mode,
            execution_params=context.template.execution_params or {},
        )
        replaced = self._results_repository.replace_top_variants_snapshot(
            job_id=job.job_id,
            now=now,
            locked_by=locked_by,
            rows=rows,
        )
        if not replaced:
            raise _BacktestJobLeaseLost()

    def _ensure_not_cancelled(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        stage: str,
    ) -> None:
        """
        Check cancel flag and transition running job to `cancelled` terminal state if requested.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            stage: Current stage literal for diagnostics.
        Returns:
            None.
        Assumptions:
            Cancel checks are executed on batch boundaries and before finalizing.
        Raises:
            _BacktestJobCancelled: When cancel request is detected and terminal write succeeds.
            _BacktestJobLeaseLost: If terminal cancel write loses lease ownership.
        Side Effects:
            Performs one lease-guarded terminal transition write when cancel is requested.
        """
        current = self._job_repository.get(job_id=job.job_id)
        if current is None or current.cancel_requested_at is None:
            return

        finished = self._lease_repository.finish(
            job_id=job.job_id,
            now=self._now(),
            locked_by=locked_by,
            next_state="cancelled",
        )
        if finished is None:
            raise _BacktestJobLeaseLost()
        _LOG.info(
            "event=job_cancelled stage=%s job_id=%s attempt=%s locked_by=%s",
            stage,
            job.job_id,
            job.attempt,
            locked_by,
        )
        raise _BacktestJobCancelled()

    def _heartbeat_if_due(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        now: datetime,
        last_heartbeat_at: datetime,
    ) -> datetime:
        """
        Extend job lease when heartbeat cadence threshold is reached.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            now: Current timestamp.
            last_heartbeat_at: Timestamp of previous successful heartbeat.
        Returns:
            datetime: Latest successful heartbeat timestamp.
        Assumptions:
            Heartbeat uses same lease TTL policy as claim loop.
        Raises:
            _BacktestJobLeaseLost: If lease-guarded heartbeat update fails.
        Side Effects:
            Performs conditional heartbeat update in storage.
        """
        elapsed_seconds = (now - last_heartbeat_at).total_seconds()
        if elapsed_seconds < float(self._heartbeat_seconds):
            return last_heartbeat_at

        updated = self._lease_repository.heartbeat(
            job_id=job.job_id,
            now=now,
            locked_by=locked_by,
            lease_seconds=self._lease_seconds,
        )
        if updated is None:
            raise _BacktestJobLeaseLost()
        return now

    def _update_progress(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        stage: str,
        processed_units: int,
        total_units: int,
        now: datetime,
    ) -> None:
        """
        Persist deterministic stage progress under lease-owner guard.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            stage: Stage literal.
            processed_units: Processed stage units.
            total_units: Total stage units.
            now: Current timestamp.
        Returns:
            None.
        Assumptions:
            Stage units semantics follow EPIC-10 contract per stage.
        Raises:
            _BacktestJobLeaseLost: If lease-guarded progress write fails.
        Side Effects:
            Writes `stage/processed_units/total_units/progress_updated_at` fields.
        """
        updated = self._lease_repository.update_progress(
            job_id=job.job_id,
            now=now,
            locked_by=locked_by,
            stage=cast(Any, stage),
            processed_units=processed_units,
            total_units=total_units,
        )
        if updated is None:
            raise _BacktestJobLeaseLost()

    def _finish_failed(
        self,
        *,
        job: BacktestJob,
        locked_by: str,
        stage: str,
        error: Exception,
    ) -> None:
        """
        Persist failed terminal state with RoehubError-like payload and no traceback in DB.

        Args:
            job: Claimed running job snapshot.
            locked_by: Active lease owner.
            stage: Current stage literal at failure point.
            error: Captured exception.
        Returns:
            None.
        Assumptions:
            Full traceback remains only in logs; DB payload stores compact error fields.
        Raises:
            _BacktestJobLeaseLost: If failed terminal write loses lease ownership.
        Side Effects:
            Writes failed terminal transition with `last_error` and `last_error_json`.
        """
        now = self._now()
        details = {
            "stage": stage,
            "exception_type": type(error).__name__,
        }
        message = str(error).strip()
        if message:
            details["message"] = message
        payload = BacktestJobErrorPayload(
            code="backtest_job_runner_failed",
            message=message or "Backtest job runner execution failed",
            details=details,
        )
        finished = self._lease_repository.finish(
            job_id=job.job_id,
            now=now,
            locked_by=locked_by,
            next_state="failed",
            last_error=payload.message,
            last_error_json=payload,
        )
        if finished is None:
            raise _BacktestJobLeaseLost()

    def _resolve_staged_scorer(
        self,
        *,
        template: RunBacktestTemplate,
        target_slice: slice,
    ) -> MetricScorerV1:
        """
        Resolve staged scorer implementation for current run context.

        Args:
            template: Effective run template.
            target_slice: Target trading/reporting slice.
        Returns:
            MetricScorerV1: Scorer used for stage execution.
        Assumptions:
            Injected scorer takes precedence over default close-fill scorer composition.
        Raises:
            ValueError: If default scorer configuration is invalid.
        Side Effects:
            None.
        """
        if self._staged_scorer is not None:
            return self._staged_scorer
        return CloseFillBacktestStagedScorerV1(
            indicator_compute=self._indicator_compute,
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            execution_params=template.execution_params or {},
            market_id=template.instrument_id.market_id.value,
            target_slice=target_slice,
            init_cash_quote_default=self._init_cash_quote_default,
            fixed_quote_default=self._fixed_quote_default,
            safe_profit_percent_default=self._safe_profit_percent_default,
            slippage_pct_default=self._slippage_pct_default,
            fee_pct_default_by_market_id=self._fee_pct_default_by_market_id,
            max_variants_guard=self._max_variants_per_compute,
            max_compute_bytes_total=self._max_compute_bytes_total,
        )


class _BacktestJobCancelled(Exception):
    """
    Internal control-flow exception signaling successful cancel terminal transition.
    """


class _BacktestJobLeaseLost(Exception):
    """
    Internal control-flow exception signaling lease-owner guarded write mismatch.
    """

def _frontier_signature_from_ranked_rows(
    *,
    ranked_rows: tuple[BacktestStageBScoredVariantV1, ...],
) -> FrontierSignatureV1:
    """
    Build deterministic frontier signature from ranked rows for snapshot write gating.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py
    Args:
        ranked_rows: Deterministically ranked Stage-B rows.
    Returns:
        FrontierSignatureV1: Ordered `(variant_key, total_return_pct)` signature tuple.
    Assumptions:
        Row order matches final deterministic persistence ranking.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple((row.variant_key, row.total_return_pct) for row in ranked_rows)


def _ranked_candidates_from_core_rows(
    *,
    ranked_rows: tuple[BacktestStageBScoredVariantV1, ...],
    tasks_by_variant_key: Mapping[str, BacktestStageBTaskV1],
) -> tuple[BacktestJobTopVariantCandidateV1, ...]:
    """
    Convert shared core Stage-B rows/tasks into persisted job-runner candidate payloads.

    Args:
        ranked_rows: Ranked Stage-B rows from shared staged core.
        tasks_by_variant_key: Stage-B task mapping by deterministic `variant_key`.
    Returns:
        tuple[BacktestJobTopVariantCandidateV1, ...]:
            Deterministic ranked candidates for persistence and finalizing flow.
    Assumptions:
        Every ranked row has matching task payload in mapping.
    Raises:
        ValueError: If one ranked row has no corresponding task payload.
    Side Effects:
        None.
    """
    candidates: list[BacktestJobTopVariantCandidateV1] = []
    for row in ranked_rows:
        task = tasks_by_variant_key.get(row.variant_key)
        if task is None:
            raise ValueError("missing Stage-B task payload for ranked variant_key")
        candidates.append(
            BacktestJobTopVariantCandidateV1(
                variant_index=row.variant_index,
                variant_key=row.variant_key,
                indicator_variant_key=row.indicator_variant_key,
                total_return_pct=row.total_return_pct,
                indicator_selections=task.indicator_selections,
                signal_params=task.signal_params,
                risk_params=task.risk_params,
            )
        )
    return tuple(candidates)


def _normalize_fee_defaults(*, values: Mapping[int, float] | None) -> Mapping[int, float]:
    """
    Normalize runtime fee defaults mapping into deterministic market-id keyed mapping.

    Args:
        values: Optional raw mapping.
    Returns:
        Mapping[int, float]: Deterministic non-empty market fee defaults mapping.
    Assumptions:
        Missing mapping falls back to fixed v1 defaults.
    Raises:
        ValueError: If mapping keys/values are invalid.
    Side Effects:
        None.
    """
    defaults = values or {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }
    normalized: dict[int, float] = {}
    for raw_market_id in sorted(defaults.keys()):
        market_id = int(raw_market_id)
        if market_id <= 0:
            raise ValueError("fee defaults market_id keys must be > 0")
        fee_pct = float(defaults[raw_market_id])
        if fee_pct < 0.0:
            raise ValueError("fee defaults values must be >= 0")
        normalized[market_id] = fee_pct
    if len(normalized) == 0:
        raise ValueError("fee defaults mapping must be non-empty")
    return normalized


def _utc_now() -> datetime:
    """
    Build timezone-aware UTC `now` timestamp.

    Args:
        None.
    Returns:
        datetime: Current UTC timestamp.
    Assumptions:
        Worker runtime stores all lifecycle timestamps in UTC.
    Raises:
        None.
    Side Effects:
        None.
    """
    return datetime.now(timezone.utc)


def _build_variant_key_for_stage_b(
    *,
    indicator_variant_key: str,
    direction_mode: str,
    sizing_mode: str,
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
    risk_params: Mapping[str, BacktestVariantScalar],
    execution_params: Mapping[str, BacktestVariantScalar],
) -> str:
    """
    Build deterministic Stage-B variant key using existing v1 variant identity semantics.

    Args:
        indicator_variant_key: Indicators-only variant key.
        direction_mode: Effective direction mode.
        sizing_mode: Effective sizing mode.
        signal_params: Signal parameters mapping.
        risk_params: Risk parameters mapping.
        execution_params: Execution parameters mapping.
    Returns:
        str: Deterministic Stage-B `variant_key`.
    Assumptions:
        Variant key contract remains unchanged from sync staged runner path.
    Raises:
        ValueError: If payload cannot be normalized by variant key builder.
    Side Effects:
        None.
    """
    from trading.contexts.backtest.domain.value_objects import build_backtest_variant_key_v1

    return build_backtest_variant_key_v1(
        indicator_variant_key=indicator_variant_key,
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        signals=signal_params,
        risk_params=risk_params,
        execution_params=execution_params,
    )


def _template_from_saved_spec_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> RunBacktestTemplate:
    """
    Build deterministic template from persisted saved-strategy `spec_payload_json`.

    Args:
        spec_payload: Saved strategy snapshot payload from job row.
    Returns:
        RunBacktestTemplate: Template reconstructed from saved snapshot payload.
    Assumptions:
        Snapshot payload follows Strategy v1 JSON shape for `instrument/timeframe/indicators`.
    Raises:
        ValueError: If required snapshot fields cannot be parsed.
    Side Effects:
        None.
    """
    instrument_payload = spec_payload.get("instrument_id")
    if not isinstance(instrument_payload, Mapping):
        raise ValueError("saved spec payload requires instrument_id object")
    market_id_raw = instrument_payload.get("market_id")
    symbol_raw = instrument_payload.get("symbol")
    if isinstance(market_id_raw, bool) or not isinstance(market_id_raw, int):
        raise ValueError("saved spec instrument_id.market_id must be integer")
    if not isinstance(symbol_raw, str):
        raise ValueError("saved spec instrument_id.symbol must be string")

    timeframe_raw = spec_payload.get("timeframe")
    if not isinstance(timeframe_raw, str):
        raise ValueError("saved spec payload requires timeframe string")

    indicators_raw = spec_payload.get("indicators")
    if not isinstance(indicators_raw, list | tuple):
        raise ValueError("saved spec payload requires indicators array")
    if len(indicators_raw) == 0:
        raise ValueError("saved spec indicators array must be non-empty")

    grids: list[GridSpec] = []
    selections: list[IndicatorVariantSelection] = []
    seen_indicator_ids: set[str] = set()
    for indicator_item in indicators_raw:
        if not isinstance(indicator_item, Mapping):
            raise ValueError("saved spec indicator entry must be object")
        indicator_id = _indicator_id_from_payload(payload=indicator_item)
        if indicator_id in seen_indicator_ids:
            raise ValueError(f"saved spec has duplicate indicator id: {indicator_id}")
        seen_indicator_ids.add(indicator_id)

        inputs = _scalar_string_key_mapping(
            payload=indicator_item.get("inputs"),
            allow_bool=False,
            field_path=f"indicators.{indicator_id}.inputs",
        )
        params = _scalar_string_key_mapping(
            payload=indicator_item.get("params"),
            allow_bool=False,
            field_path=f"indicators.{indicator_id}.params",
        )
        source_spec: GridParamSpec | None = None
        if "source" in inputs:
            source_spec = ExplicitValuesSpec(name="source", values=(str(inputs["source"]),))

        merged_params: dict[str, GridParamSpec] = {}
        for key in sorted(params.keys()):
            merged_params[key] = ExplicitValuesSpec(name=key, values=(params[key],))
        for key in sorted(inputs.keys()):
            if key == "source" or key in merged_params:
                continue
            merged_params[key] = ExplicitValuesSpec(name=key, values=(inputs[key],))

        grids.append(
            GridSpec(
                indicator_id=IndicatorId(indicator_id),
                params=merged_params,
                source=source_spec,
            )
        )
        selections.append(
            IndicatorVariantSelection(
                indicator_id=indicator_id,
                inputs=inputs,
                params=params,
            )
        )

    ordered = sorted(zip(grids, selections, strict=True), key=lambda item: item[1].indicator_id)
    signal_grids = _signal_grids_from_spec_payload(spec_payload=spec_payload)
    risk_grid = _risk_grid_from_spec_payload(spec_payload=spec_payload)
    risk_params = _risk_params_from_spec_payload(spec_payload=spec_payload)
    execution_params = _execution_params_from_spec_payload(spec_payload=spec_payload)
    direction_mode = _optional_mode(
        payload=spec_payload,
        key="direction_mode",
        default="long-short",
    )
    sizing_mode = _optional_mode(
        payload=spec_payload,
        key="sizing_mode",
        default="all_in",
    )

    return RunBacktestTemplate(
        instrument_id=InstrumentId(
            market_id=MarketId(market_id_raw),
            symbol=Symbol(symbol_raw),
        ),
        timeframe=Timeframe(timeframe_raw),
        indicator_grids=tuple(item[0] for item in ordered),
        indicator_selections=tuple(item[1] for item in ordered),
        signal_grids=signal_grids,
        risk_grid=risk_grid,
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        risk_params=risk_params,
        execution_params=execution_params,
    )


def _apply_saved_overrides(
    *,
    base_template: RunBacktestTemplate,
    overrides: RunBacktestSavedOverrides | None,
) -> RunBacktestTemplate:
    """
    Apply optional saved-mode overrides on top of template resolved from saved snapshot.

    Args:
        base_template: Base template reconstructed from saved snapshot payload.
        overrides: Optional saved-mode overrides payload.
    Returns:
        RunBacktestTemplate: Effective template.
    Assumptions:
        Missing overrides keep base template values unchanged.
    Raises:
        ValueError: If merged payload violates template invariants.
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
        overrides.sizing_mode if overrides.sizing_mode is not None else base_template.sizing_mode
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
    risk_grid = overrides.risk_grid if overrides.risk_grid is not None else base_template.risk_grid

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


def _merge_scalar_mappings(
    *,
    base: Mapping[str, BacktestRequestScalar],
    updates: Mapping[str, BacktestRequestScalar],
) -> dict[str, BacktestRequestScalar]:
    """
    Merge scalar mappings with deterministic key ordering (`updates` override `base`).

    Args:
        base: Base scalar mapping.
        updates: Override scalar mapping.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic merged mapping.
    Assumptions:
        Keys are scalar field names from backtest template payload.
    Raises:
        ValueError: If one key is blank.
    Side Effects:
        None.
    """
    merged = dict(base)
    merged.update(updates)
    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(merged.keys()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("mapping keys must be non-empty")
        normalized[key] = merged[raw_key]
    return normalized


def _merge_signal_grids(
    *,
    base: Mapping[str, Mapping[str, GridParamSpec]],
    updates: Mapping[str, Mapping[str, GridParamSpec]],
) -> dict[str, dict[str, GridParamSpec]]:
    """
    Merge nested signal grid mappings with deterministic key ordering.

    Args:
        base: Base signal grids mapping.
        updates: Override signal grids mapping.
    Returns:
        dict[str, dict[str, GridParamSpec]]: Deterministic merged signal grids mapping.
    Assumptions:
        Nested mapping keys are normalized indicator ids and parameter names.
    Raises:
        ValueError: If one nested key is blank.
    Side Effects:
        None.
    """
    merged: dict[str, dict[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(set(base.keys()) | set(updates.keys())):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("signal grid indicator_id keys must be non-empty")
        merged_params: dict[str, GridParamSpec] = {}
        base_params = base.get(raw_indicator_id, {})
        update_params = updates.get(raw_indicator_id, {})
        for raw_param in sorted(set(base_params.keys()) | set(update_params.keys())):
            param_name = str(raw_param).strip().lower()
            if not param_name:
                raise ValueError("signal grid param keys must be non-empty")
            if raw_param in update_params:
                merged_params[param_name] = update_params[raw_param]
            else:
                merged_params[param_name] = base_params[raw_param]
        merged[indicator_id] = merged_params
    return merged


def _signal_grids_from_spec_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> dict[str, dict[str, GridParamSpec]]:
    """
    Parse optional `signal_grids` payload from saved strategy snapshot JSON.

    Args:
        spec_payload: Saved strategy snapshot payload.
    Returns:
        dict[str, dict[str, GridParamSpec]]: Deterministic signal grids mapping.
    Assumptions:
        Missing `signal_grids` payload means empty mapping.
    Raises:
        ValueError: If payload shape is invalid.
    Side Effects:
        None.
    """
    raw_signal_grids = spec_payload.get("signal_grids")
    if raw_signal_grids is None:
        return {}
    if not isinstance(raw_signal_grids, Mapping):
        raise ValueError("saved spec signal_grids must be mapping when provided")

    normalized: dict[str, dict[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(
        raw_signal_grids.keys(),
        key=lambda key: str(key).strip().lower(),
    ):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("saved spec signal_grids indicator keys must be non-empty")
        indicator_payload = raw_signal_grids[raw_indicator_id]
        if not isinstance(indicator_payload, Mapping):
            raise ValueError(f"saved spec signal_grids.{indicator_id} must be mapping")
        params: dict[str, GridParamSpec] = {}
        for raw_param_name in sorted(
            indicator_payload.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("saved spec signal param keys must be non-empty")
            params[param_name] = _grid_param_spec_from_payload_value(
                name=param_name,
                value=indicator_payload[raw_param_name],
                numeric_only=False,
            )
        normalized[indicator_id] = params
    return normalized


def _risk_grid_from_spec_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> BacktestRiskGridSpec:
    """
    Parse optional risk grid payload from saved strategy snapshot JSON.

    Args:
        spec_payload: Saved strategy snapshot payload.
    Returns:
        BacktestRiskGridSpec: Deterministic risk grid payload.
    Assumptions:
        Scalar fallback values may live under top-level `risk` mapping.
    Raises:
        ValueError: If risk payload shape is invalid.
    Side Effects:
        None.
    """
    raw_risk = spec_payload.get("risk")
    risk_payload = raw_risk if isinstance(raw_risk, Mapping) else {}
    if raw_risk is not None and not isinstance(raw_risk, Mapping):
        raise ValueError("saved spec risk must be mapping when provided")

    raw_risk_grid = spec_payload.get("risk_grid")
    risk_grid_payload = raw_risk_grid if isinstance(raw_risk_grid, Mapping) else {}
    if raw_risk_grid is not None and not isinstance(raw_risk_grid, Mapping):
        raise ValueError("saved spec risk_grid must be mapping when provided")

    sl_enabled = _bool_with_default(
        payload=risk_grid_payload,
        key="sl_enabled",
        default=_bool_with_default(payload=risk_payload, key="sl_enabled", default=False),
    )
    tp_enabled = _bool_with_default(
        payload=risk_grid_payload,
        key="tp_enabled",
        default=_bool_with_default(payload=risk_payload, key="tp_enabled", default=False),
    )
    sl_spec = _grid_param_spec_optional(
        payload=risk_grid_payload,
        key="sl",
        numeric_only=True,
    )
    tp_spec = _grid_param_spec_optional(
        payload=risk_grid_payload,
        key="tp",
        numeric_only=True,
    )
    if sl_enabled and sl_spec is None:
        raw_sl = risk_payload.get("sl_pct")
        if raw_sl is not None:
            if isinstance(raw_sl, bool) or not isinstance(raw_sl, int | float):
                raise ValueError("saved spec risk.sl_pct must be numeric")
            sl_spec = ExplicitValuesSpec(name="sl", values=(float(raw_sl),))
    if tp_enabled and tp_spec is None:
        raw_tp = risk_payload.get("tp_pct")
        if raw_tp is not None:
            if isinstance(raw_tp, bool) or not isinstance(raw_tp, int | float):
                raise ValueError("saved spec risk.tp_pct must be numeric")
            tp_spec = ExplicitValuesSpec(name="tp", values=(float(raw_tp),))

    return BacktestRiskGridSpec(
        sl_enabled=sl_enabled,
        tp_enabled=tp_enabled,
        sl=sl_spec,
        tp=tp_spec,
    )


def _risk_params_from_spec_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> dict[str, BacktestRequestScalar]:
    """
    Parse scalar `risk` payload from saved strategy snapshot JSON.

    Args:
        spec_payload: Saved strategy snapshot payload.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic scalar risk mapping.
    Assumptions:
        Missing payload means empty mapping.
    Raises:
        ValueError: If risk payload shape is invalid.
    Side Effects:
        None.
    """
    raw_risk = spec_payload.get("risk")
    if raw_risk is None:
        return {}
    if not isinstance(raw_risk, Mapping):
        raise ValueError("saved spec risk must be mapping when provided")
    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(raw_risk.keys(), key=lambda key: str(key).strip().lower()):
        key = str(raw_key).strip().lower()
        if not key:
            raise ValueError("saved spec risk keys must be non-empty")
        value = raw_risk[raw_key]
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"saved spec risk.{key} must be scalar")
        normalized[key] = value
    return normalized


def _execution_params_from_spec_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> dict[str, BacktestRequestScalar]:
    """
    Parse scalar `execution` payload from saved strategy snapshot JSON.

    Args:
        spec_payload: Saved strategy snapshot payload.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic scalar execution mapping.
    Assumptions:
        Missing payload means empty mapping.
    Raises:
        ValueError: If execution payload shape is invalid.
    Side Effects:
        None.
    """
    raw_execution = spec_payload.get("execution")
    if raw_execution is None:
        return {}
    if not isinstance(raw_execution, Mapping):
        raise ValueError("saved spec execution must be mapping when provided")
    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(raw_execution.keys(), key=lambda key: str(key).strip().lower()):
        key = str(raw_key).strip().lower()
        if not key:
            raise ValueError("saved spec execution keys must be non-empty")
        value = raw_execution[raw_key]
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"saved spec execution.{key} must be scalar")
        normalized[key] = value
    return normalized


def _grid_param_spec_optional(
    *,
    payload: Mapping[str, Any],
    key: str,
    numeric_only: bool,
) -> GridParamSpec | None:
    """
    Parse optional axis payload into grid parameter spec.

    Args:
        payload: Parent mapping payload.
        key: Axis key.
        numeric_only: Whether axis values must be numeric.
    Returns:
        GridParamSpec | None: Parsed axis spec or `None`.
    Assumptions:
        Missing key means no axis payload for this value.
    Raises:
        ValueError: If axis payload shape is invalid.
    Side Effects:
        None.
    """
    if key not in payload:
        return None
    return _grid_param_spec_from_payload_value(
        name=key,
        value=payload[key],
        numeric_only=numeric_only,
    )


def _grid_param_spec_from_payload_value(
    *,
    name: str,
    value: Any,
    numeric_only: bool,
) -> GridParamSpec:
    """
    Parse generic payload node into deterministic grid parameter spec.

    Args:
        name: Axis name.
        value: Raw payload value.
        numeric_only: Whether resulting values must be numeric.
    Returns:
        GridParamSpec: Explicit or range spec.
    Assumptions:
        Scalar non-mapping values are treated as one-value explicit axes.
    Raises:
        ValueError: If payload shape is invalid.
    Side Effects:
        None.
    """
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("axis name must be non-empty")

    if not isinstance(value, Mapping):
        if numeric_only and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ValueError(f"axis '{normalized_name}' scalar value must be numeric")
        if isinstance(value, bool):
            raise ValueError(f"axis '{normalized_name}' scalar value must not be boolean")
        return ExplicitValuesSpec(name=normalized_name, values=(cast(Any, value),))

    raw_mode = value.get("mode")
    if not isinstance(raw_mode, str):
        raise ValueError(f"axis '{normalized_name}' must include string mode")
    mode = raw_mode.strip().lower()

    if mode == "explicit":
        raw_values = value.get("values")
        if not isinstance(raw_values, list | tuple):
            raise ValueError(f"axis '{normalized_name}' explicit mode requires values array")
        parsed: list[int | float | str] = []
        for item in raw_values:
            if isinstance(item, bool):
                raise ValueError(f"axis '{normalized_name}' explicit values must not contain bool")
            if numeric_only and not isinstance(item, int | float):
                raise ValueError(f"axis '{normalized_name}' explicit values must be numeric")
            if not isinstance(item, int | float | str):
                raise ValueError(f"axis '{normalized_name}' explicit values must be scalar")
            parsed.append(cast(Any, float(item) if numeric_only else item))
        if len(parsed) == 0:
            raise ValueError(f"axis '{normalized_name}' explicit values must be non-empty")
        return ExplicitValuesSpec(name=normalized_name, values=tuple(parsed))

    if mode == "range":
        start = value.get("start")
        stop_incl = value.get("stop_incl")
        step = value.get("step")
        if (
            isinstance(start, bool)
            or not isinstance(start, int | float)
            or isinstance(stop_incl, bool)
            or not isinstance(stop_incl, int | float)
            or isinstance(step, bool)
            or not isinstance(step, int | float)
        ):
            raise ValueError(f"axis '{normalized_name}' range mode requires numeric bounds")
        return RangeValuesSpec(
            name=normalized_name,
            start=float(start),
            stop_inclusive=float(stop_incl),
            step=float(step),
        )

    raise ValueError(f"axis '{normalized_name}' mode must be explicit or range")


def _indicator_id_from_payload(*, payload: Mapping[str, Any]) -> str:
    """
    Extract normalized indicator id from indicator payload mapping.

    Args:
        payload: Indicator payload mapping.
    Returns:
        str: Normalized lowercase indicator id.
    Assumptions:
        Payload may store id under `indicator_id|id|kind|name`.
    Raises:
        ValueError: If identifier is missing or blank.
    Side Effects:
        None.
    """
    for key in ("indicator_id", "id", "kind", "name"):
        raw_value = payload.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip().lower()
    raise ValueError("indicator payload requires non-empty indicator_id/id/kind/name")


def _scalar_string_key_mapping(
    *,
    payload: Any,
    allow_bool: bool,
    field_path: str,
) -> dict[str, int | float | str]:
    """
    Parse optional string-keyed scalar mapping.

    Args:
        payload: Raw mapping payload.
        allow_bool: Whether bool values are allowed.
        field_path: Field path for deterministic errors.
    Returns:
        dict[str, int | float | str]: Deterministic scalar mapping.
    Assumptions:
        Missing payload means empty mapping.
    Raises:
        ValueError: If mapping shape or scalar values are invalid.
    Side Effects:
        None.
    """
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_path} must be mapping when provided")
    normalized: dict[str, int | float | str] = {}
    for raw_key in sorted(payload.keys(), key=lambda key: str(key).strip().lower()):
        key = str(raw_key).strip().lower()
        if not key:
            raise ValueError(f"{field_path} keys must be non-empty")
        value = payload[raw_key]
        if isinstance(value, bool) and not allow_bool:
            raise ValueError(f"{field_path}.{key} must not be boolean")
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"{field_path}.{key} must be scalar")
        normalized[key] = value
    return normalized


def _bool_with_default(
    *,
    payload: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    """
    Parse optional boolean value with default fallback.

    Args:
        payload: Parent mapping payload.
        key: Boolean key.
        default: Fallback value when key is absent.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Missing keys are represented by fallback defaults.
    Raises:
        ValueError: If payload value exists but is not boolean.
    Side Effects:
        None.
    """
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean when provided")
    return value


def _optional_mode(*, payload: Mapping[str, Any], key: str, default: str) -> str:
    """
    Parse optional mode literal from mapping with fallback default.

    Args:
        payload: Parent mapping payload.
        key: Mode key.
        default: Fallback literal when key is absent or blank.
    Returns:
        str: Normalized lowercase mode literal.
    Assumptions:
        Mode literal validation is delegated to template DTO constructors.
    Raises:
        ValueError: If provided mode value is not string.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be string when provided")
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized


__all__ = [
    "BacktestJobRunReportV1",
    "BacktestJobRunStatus",
    "RunBacktestJobRunnerV1",
]
