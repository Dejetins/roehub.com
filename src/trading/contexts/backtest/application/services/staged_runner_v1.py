from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, cast

from trading.contexts.backtest.application.dto import (
    BacktestRankingConfig,
    BacktestReportV1,
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestStagedVariantScorer,
    BacktestStagedVariantScorerWithDetails,
    BacktestVariantScoreDetailsV1,
)
from trading.contexts.backtest.application.services.grid_builder_v1 import (
    STAGE_B_LITERAL,
    BacktestGridBuildContextV1,
    BacktestGridBuilderV1,
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
    RiskParamsV1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)
from trading.shared_kernel.primitives import TimeRange

from .reporting_service_v1 import BacktestReportingServiceV1
from .run_control_v1 import BacktestRunControlV1
from .staged_core_runner_v1 import (
    BacktestStageAScoredVariantV1,
    BacktestStageBScoredVariantV1,
    BacktestStageBTaskV1,
    BacktestStagedCoreRunnerV1,
)

TOTAL_RETURN_METRIC_LITERAL = "Total Return [%]"


@dataclass(frozen=True, slots=True)
class BacktestStagedRunResultV1:
    """
    Deterministic staged-run summary used by `RunBacktestUseCase`.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    variants: tuple[BacktestVariantPreview, ...]
    stage_a_variants_total: int
    stage_b_variants_total: int
    estimated_memory_bytes: int
    indicator_estimate_calls: int

    def __post_init__(self) -> None:
        """
        Validate deterministic staged-run totals and response payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants list is already sorted by staged ranking key before construction.
        Raises:
            ValueError: If one total is invalid.
        Side Effects:
            None.
        """
        if self.stage_a_variants_total <= 0:
            raise ValueError("BacktestStagedRunResultV1.stage_a_variants_total must be > 0")
        if self.stage_b_variants_total <= 0:
            raise ValueError("BacktestStagedRunResultV1.stage_b_variants_total must be > 0")
        if self.estimated_memory_bytes <= 0:
            raise ValueError("BacktestStagedRunResultV1.estimated_memory_bytes must be > 0")
        if self.indicator_estimate_calls < 0:
            raise ValueError("BacktestStagedRunResultV1.indicator_estimate_calls must be >= 0")


@dataclass(frozen=True, slots=True)
class _StageAScoredVariant:
    """
    Internal Stage A scored row for deterministic shortlist sorting.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    base_variant: BacktestStageABaseVariant
    total_return_pct: float


@dataclass(frozen=True, slots=True)
class _StageBScoredVariant:
    """
    Internal Stage B scored row for deterministic top-K sorting.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    total_return_pct: float


@dataclass(frozen=True, slots=True)
class _StageBTask:
    """
    Internal Stage B scoring task payload for optional CPU-parallel execution.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]]
    risk_params: Mapping[str, BacktestVariantScalar]


@dataclass(frozen=True, slots=True)
class _TopVariantPayloadContext:
    """
    Internal deterministic payload context for top-ranked Stage-B variants.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
    """

    reports_by_variant_key: Mapping[str, BacktestReportV1]
    execution_params_by_variant_key: Mapping[str, Mapping[str, BacktestVariantScalar]]
    risk_params_by_variant_key: Mapping[str, Mapping[str, BacktestVariantScalar]]


class BacktestStagedRunnerV1:
    """
    Run deterministic staged pipeline (Stage A shortlist -> Stage B expand -> top-K).

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
    """

    def __init__(
        self,
        *,
        grid_builder: BacktestGridBuilderV1 | None = None,
        parallel_workers: int | None = None,
        reporting_service: BacktestReportingServiceV1 | None = None,
        core_runner: BacktestStagedCoreRunnerV1 | None = None,
    ) -> None:
        """
        Initialize staged runner with optional custom grid-builder implementation.

        Args:
            grid_builder: Optional custom grid builder for Stage A/Stage B context.
            parallel_workers: Optional number of worker threads for variant scoring.
            reporting_service: Optional reporting service used for Stage-B top-k payloads.
            core_runner: Shared staged scoring core used by sync and jobs paths.
        Returns:
            None.
        Assumptions:
            Default grid builder follows EPIC-04 deterministic guard contracts.
        Raises:
            ValueError: If provided workers count is non-positive.
        Side Effects:
            None.
        """
        if parallel_workers is not None and parallel_workers <= 0:
            raise ValueError("BacktestStagedRunnerV1.parallel_workers must be > 0")
        self._grid_builder = grid_builder or BacktestGridBuilderV1()
        self._parallel_workers = parallel_workers
        self._reporting_service = reporting_service or BacktestReportingServiceV1()
        self._core_runner = core_runner or BacktestStagedCoreRunnerV1()

    def run(
        self,
        *,
        template: RunBacktestTemplate,
        candles: CandleArrays,
        preselect: int,
        top_k: int,
        indicator_compute: IndicatorCompute,
        scorer: BacktestStagedVariantScorer,
        ranking: BacktestRankingConfig | None = None,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
        requested_time_range: TimeRange | None = None,
        top_trades_n: int = 3,
        run_control: BacktestRunControlV1 | None = None,
    ) -> BacktestStagedRunResultV1:
        """
        Execute staged ranking flow and return deterministic top-K variant previews.

        Args:
            template: Resolved backtest template payload.
            candles: Dense warmup-inclusive candle arrays.
            preselect: Stage A shortlist size.
            top_k: Stage B top-k output size.
            indicator_compute: Indicator estimate port for staged grid materialization.
            scorer: Scoring port returning `Total Return [%]` for every variant.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`)
                from request/runtime defaults.
            defaults_provider: Optional defaults provider for compute/signal fallback.
            max_variants_per_compute: Stage variants guard limit.
            max_compute_bytes_total: Stage memory guard limit.
            requested_time_range:
                Optional request range for reporting rows (`Start/End/Duration`).
            top_trades_n:
                Number of best variants for which full trades payload is included.
            run_control:
                Optional cooperative cancellation/deadline control shared with request/job
                lifecycle.
        Returns:
            BacktestStagedRunResultV1: Deterministic staged-run output summary.
        Assumptions:
            Scorer returns deterministic metrics for identical variant payloads.
        Raises:
            ValueError: If top-k/preselect values are invalid or scorer payload is invalid.
            RoehubError: Propagated from staged grid-builder guard checks.
        Side Effects:
            None.
        """
        if preselect <= 0:
            raise ValueError("BacktestStagedRunnerV1 preselect must be > 0")
        if top_k <= 0:
            raise ValueError("BacktestStagedRunnerV1 top_k must be > 0")
        if top_trades_n <= 0:
            raise ValueError("BacktestStagedRunnerV1 top_trades_n must be > 0")

        grid_context = self._grid_builder.build(
            template=template,
            candles=candles,
            indicator_compute=indicator_compute,
            preselect=preselect,
            defaults_provider=defaults_provider,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        )
        self._prepare_scorer_for_grid_context(
            scorer=scorer,
            grid_context=grid_context,
            candles=candles,
            max_compute_bytes_total=max_compute_bytes_total,
            run_control=run_control,
        )
        cancel_checker = _cancel_checker_from_run_control(run_control=run_control)

        shortlist = self._score_stage_a(
            grid_context=grid_context,
            candles=candles,
            scorer=scorer,
            shortlist_limit=preselect,
            ranking=ranking,
            cancel_checker=cancel_checker,
        )

        details_scorer = (
            _details_scorer(scorer=scorer) if requested_time_range is not None else None
        )
        retained_details_by_variant_key: Mapping[str, BacktestVariantScoreDetailsV1] = {}
        if details_scorer is None:
            top_rows, stage_b_tasks = self._score_stage_b(
                template=template,
                grid_context=grid_context,
                shortlist=shortlist,
                candles=candles,
                scorer=scorer,
                top_k_limit=top_k,
                ranking=ranking,
                cancel_checker=cancel_checker,
            )
        else:
            (
                top_rows,
                stage_b_tasks,
                retained_details_by_variant_key,
            ) = self._score_stage_b_with_retained_details(
                template=template,
                grid_context=grid_context,
                shortlist=shortlist,
                candles=candles,
                scorer=scorer,
                details_scorer=details_scorer,
                top_k_limit=top_k,
                ranking=ranking,
                cancel_checker=cancel_checker,
            )
        top_payload_context = self._build_top_reports(
            requested_time_range=requested_time_range,
            top_rows=top_rows,
            stage_b_tasks=stage_b_tasks,
            retained_details_by_variant_key=retained_details_by_variant_key,
            candles=candles,
            scorer=scorer,
            top_trades_n=top_trades_n,
        )
        variants_list: list[BacktestVariantPreview] = []
        for row in top_rows:
            task = stage_b_tasks.get(row.variant_key)
            if task is None:
                raise ValueError("missing Stage B task for top-row variant_key")

            execution_params = top_payload_context.execution_params_by_variant_key.get(
                row.variant_key,
                template.execution_params or {},
            )
            risk_params = top_payload_context.risk_params_by_variant_key.get(
                row.variant_key,
                task.risk_params,
            )
            variants_list.append(
                BacktestVariantPreview(
                    variant_index=row.variant_index,
                    variant_key=row.variant_key,
                    indicator_variant_key=row.indicator_variant_key,
                    total_return_pct=row.total_return_pct,
                    payload=BacktestVariantPayloadV1(
                        indicator_selections=task.indicator_selections,
                        signal_params=task.signal_params,
                        risk_params=risk_params,
                        execution_params=execution_params,
                        direction_mode=template.direction_mode,
                        sizing_mode=template.sizing_mode,
                    ),
                    report=top_payload_context.reports_by_variant_key.get(row.variant_key),
                )
            )
        variants = tuple(variants_list)
        return BacktestStagedRunResultV1(
            variants=variants,
            stage_a_variants_total=grid_context.stage_a_variants_total,
            stage_b_variants_total=grid_context.stage_b_variants_total,
            estimated_memory_bytes=grid_context.estimated_memory_bytes,
            indicator_estimate_calls=grid_context.indicator_estimate_calls,
        )

    def _prepare_scorer_for_grid_context(
        self,
        *,
        scorer: BacktestStagedVariantScorer,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        max_compute_bytes_total: int,
        run_control: BacktestRunControlV1 | None,
    ) -> None:
        """
        Prepare scorer hot-path context (batched indicator tensors) when scorer supports it.

        Args:
            scorer: Stage scorer port implementation.
            grid_context: Deterministic staged grid context.
            candles: Dense warmup-inclusive candles for tensor compute.
            max_compute_bytes_total: Stage memory guard budget.
            run_control: Optional cooperative cancellation/deadline control.
        Returns:
            None.
        Assumptions:
            Optional scorer extension is discovered via method presence.
        Raises:
            Exception: Propagates scorer preparation errors.
        Side Effects:
            May trigger indicator tensor precompute and in-memory cache priming.
        """
        prepare_method = getattr(scorer, "prepare_for_grid_context", None)
        if prepare_method is None:
            return
        prepare_method(
            grid_context=grid_context,
            candles=candles,
            max_compute_bytes_total=max_compute_bytes_total,
            run_control=run_control,
        )

    def _score_stage_a(
        self,
        *,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        shortlist_limit: int,
        ranking: BacktestRankingConfig | None = None,
        cancel_checker: Callable[[str], None] | None = None,
    ) -> list[BacktestStageAScoredVariantV1]:
        """
        Score Stage-A variants through shared staged core runner.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            grid_context: Built staged grid context.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
            shortlist_limit: Maximum number of Stage A rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`).
            cancel_checker: Optional cooperative cancellation callback by stage.
        Returns:
            list[BacktestStageAScoredVariantV1]:
                Sorted Stage A shortlist rows by ranking and tie-break key.
        Assumptions:
            Stage A uses base key tie-break with risk disabled.
        Raises:
            ValueError: If scorer payload lacks required ranking metric or limit is invalid.
        Side Effects:
            None.
        """
        rows = self._core_runner.run_stage_a(
            grid_context=grid_context,
            candles=candles,
            scorer=scorer,
            shortlist_limit=shortlist_limit,
            ranking=ranking,
            cancel_checker=cancel_checker,
        )
        return list(rows)

    def _score_stage_b(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: list[BacktestStageAScoredVariantV1],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        cancel_checker: Callable[[str], None] | None = None,
    ) -> tuple[list[BacktestStageBScoredVariantV1], Mapping[str, BacktestStageBTaskV1]]:
        """
        Score Stage-B variants through shared staged core runner.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            template: Resolved backtest template payload.
            grid_context: Built staged grid context.
            shortlist: Sorted Stage A shortlist rows.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`).
            cancel_checker: Optional cooperative cancellation callback by stage.
        Returns:
            tuple[list[BacktestStageBScoredVariantV1], Mapping[str, BacktestStageBTaskV1]]:
                Sorted Stage B top rows and task lookup required for report/payload building.
        Assumptions:
            Stage B tie-break key is full `variant_key`.
        Raises:
            ValueError: If scorer payload lacks required ranking metric or limit is invalid.
        Side Effects:
            None.
        """
        top_rows, top_tasks_by_variant_key = self._core_runner.run_stage_b(
            template=template,
            grid_context=grid_context,
            shortlist=tuple(shortlist),
            candles=candles,
            scorer=scorer,
            top_k_limit=top_k_limit,
            ranking=ranking,
            cancel_checker=cancel_checker,
        )
        return (list(top_rows), top_tasks_by_variant_key)

    def _score_stage_b_with_retained_details(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: list[BacktestStageAScoredVariantV1],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        details_scorer: BacktestStagedVariantScorerWithDetails,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        cancel_checker: Callable[[str], None] | None = None,
    ) -> tuple[
        list[BacktestStageBScoredVariantV1],
        Mapping[str, BacktestStageBTaskV1],
        Mapping[str, BacktestVariantScoreDetailsV1],
    ]:
        """
        Score Stage-B and retain details only for variants currently held by bounded top-k heap.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            template: Resolved backtest template payload.
            grid_context: Built staged grid context.
            shortlist: Sorted Stage A shortlist rows.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
            details_scorer: Details extension used for report payload reuse.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`).
            cancel_checker: Optional cooperative cancellation callback by stage.
        Returns:
            tuple[
                list[BacktestStageBScoredVariantV1],
                Mapping[str, BacktestStageBTaskV1],
                Mapping[str, BacktestVariantScoreDetailsV1],
            ]:
                Sorted Stage-B top rows, tasks mapping, and retained details for top rows.
        Assumptions:
            Stage B tie-break key is full `variant_key`.
        Raises:
            ValueError: If scorer payload lacks required ranking metric or limit is invalid.
        Side Effects:
            Retained details map grows only up to `top_k_limit`.
        """
        top_rows, top_tasks_by_variant_key, retained_details_by_variant_key = (
            self._core_runner.run_stage_b_with_details(
                template=template,
                grid_context=grid_context,
                shortlist=tuple(shortlist),
                candles=candles,
                scorer=scorer,
                details_scorer=details_scorer,
                top_k_limit=top_k_limit,
                ranking=ranking,
                cancel_checker=cancel_checker,
            )
        )
        return (list(top_rows), top_tasks_by_variant_key, retained_details_by_variant_key)

    def _score_stage_a_variant(
        self,
        *,
        base_variant: BacktestStageABaseVariant,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> BacktestStageAScoredVariantV1:
        """
        Score one Stage-A base variant through shared staged core helper.

        Args:
            base_variant: Stage A base variant.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
        Returns:
            _StageAScoredVariant: Scored row for stable Stage A sorting.
        Assumptions:
            Stage A always disables SL/TP for shortlist ranking.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        return self._core_runner._score_stage_a_variant(  # noqa: SLF001
            base_variant=base_variant,
            candles=candles,
            scorer=scorer,
        )

    def _iter_stage_b_tasks(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: list[BacktestStageAScoredVariantV1],
    ) -> Iterator[BacktestStageBTaskV1]:
        """
        Iterate deterministic Stage-B scoring tasks through shared staged core helper.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            template: Resolved backtest template payload.
            grid_context: Built staged grid context.
            shortlist: Stage A shortlist rows.
        Returns:
            Iterator[_StageBTask]: Deterministic Stage B task payload stream.
        Assumptions:
            Variant index follows `(shortlist_index * risk_total) + risk_index` contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        yield from self._core_runner._iter_stage_b_tasks(  # noqa: SLF001
            template=template,
            grid_context=grid_context,
            shortlist=tuple(shortlist),
        )

    def _score_stage_b_task(
        self,
        *,
        task: BacktestStageBTaskV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> BacktestStageBScoredVariantV1:
        """
        Score one Stage-B task payload through shared staged core helper.

        Args:
            task: Stage B scoring task.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
        Returns:
            _StageBScoredVariant: Scored row for Stage B stable sorting.
        Assumptions:
            Stage B scorer evaluates risk-enabled variants by task payload.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        return self._core_runner._score_stage_b_task(  # noqa: SLF001
            task=task,
            candles=candles,
            scorer=scorer,
        )

    def _score_stage_b_task_with_task(
        self,
        task: BacktestStageBTaskV1,
        *,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> tuple[BacktestStageBScoredVariantV1, BacktestStageBTaskV1]:
        """
        Score one Stage-B task and return both ranked row and original task payload.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            task: Stage B scoring task.
            candles: Dense candles forwarded to scorer.
            scorer: Stage scorer port.
        Returns:
            tuple[_StageBScoredVariant, _StageBTask]:
                Ranked row coupled with source task for top-row payload reconstruction.
        Assumptions:
            Task identity must remain unchanged after scoring for deterministic mapping.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        return (
            self._score_stage_b_task(task=task, candles=candles, scorer=scorer),
            task,
        )

    def _build_top_reports(
        self,
        *,
        requested_time_range: TimeRange | None,
        top_rows: list[BacktestStageBScoredVariantV1],
        stage_b_tasks: Mapping[str, BacktestStageBTaskV1],
        retained_details_by_variant_key: Mapping[str, BacktestVariantScoreDetailsV1],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        top_trades_n: int,
    ) -> _TopVariantPayloadContext:
        """
        Build deterministic EPIC-06 report payloads for selected Stage-B top-k variants.

        Args:
            requested_time_range: User request range for reporting rows.
            top_rows: Already ranked Stage-B top rows.
            stage_b_tasks: Stage-B task lookup by deterministic `variant_key`.
            retained_details_by_variant_key:
                Optional details retained during Stage-B scoring for current top rows.
            candles: Warmup-inclusive candles forwarded to scorer/reporting services.
            scorer: Stage scorer port optionally supporting details extension.
            top_trades_n: Number of best variants to include full trades list for.
        Returns:
            _TopVariantPayloadContext: Report and effective execution/risk payload maps.
        Assumptions:
            Ranking order in `top_rows` is deterministic and already final.
        Raises:
            ValueError: If top-row task mapping is missing or scorer detail payload mismatches.
        Side Effects:
            Uses retained Stage-B details when available; falls back to one-pass re-score only
            if retained details are unavailable.
        """
        if requested_time_range is None:
            return _TopVariantPayloadContext(
                reports_by_variant_key={},
                execution_params_by_variant_key={},
                risk_params_by_variant_key={},
            )
        details_scorer = _details_scorer(scorer=scorer)
        if len(retained_details_by_variant_key) == 0 and details_scorer is None:
            return _TopVariantPayloadContext(
                reports_by_variant_key={},
                execution_params_by_variant_key={},
                risk_params_by_variant_key={},
            )

        reports_by_variant_key: dict[str, BacktestReportV1] = {}
        execution_params_by_variant_key: dict[str, Mapping[str, BacktestVariantScalar]] = {}
        risk_params_by_variant_key: dict[str, Mapping[str, BacktestVariantScalar]] = {}
        for ranked_index, row in enumerate(top_rows):
            task = stage_b_tasks.get(row.variant_key)
            if task is None:
                raise ValueError("missing Stage B task for top-row variant_key")

            details = retained_details_by_variant_key.get(row.variant_key)
            if details is None:
                if details_scorer is None:
                    raise ValueError("missing retained details for top-row variant_key")
                details = details_scorer.score_variant_with_details(
                    stage=STAGE_B_LITERAL,
                    candles=candles,
                    indicator_selections=task.indicator_selections,
                    signal_params=task.signal_params,
                    risk_params=task.risk_params,
                    indicator_variant_key=task.indicator_variant_key,
                    variant_key=task.variant_key,
                )
            detailed_total_return_pct = _extract_total_return_pct(metrics=details.metrics)
            if abs(detailed_total_return_pct - row.total_return_pct) > 1e-12:
                raise ValueError("detailed scorer payload must match ranked total return value")

            reports_by_variant_key[row.variant_key] = self._reporting_service.build_report(
                requested_time_range=requested_time_range,
                candles=candles,
                target_slice=details.target_slice,
                execution_params=details.execution_params,
                execution_outcome=details.execution_outcome,
                include_table_md=True,
                include_trades=ranked_index < top_trades_n,
            )
            execution_params_by_variant_key[row.variant_key] = _execution_params_to_mapping(
                params=details.execution_params
            )
            risk_params_by_variant_key[row.variant_key] = _risk_params_to_mapping(
                params=details.risk_params
            )
        return _TopVariantPayloadContext(
            reports_by_variant_key=reports_by_variant_key,
            execution_params_by_variant_key=execution_params_by_variant_key,
            risk_params_by_variant_key=risk_params_by_variant_key,
        )

    def _resolve_parallel_workers(self, *, total_tasks: int) -> int:
        """
        Resolve deterministic worker count for optional CPU-parallel scoring.

        Args:
            total_tasks: Number of scoring tasks.
        Returns:
            int: Worker count (`1` for sequential execution).
        Assumptions:
            `parallel_workers=None` means auto-detect by CPU count.
        Raises:
            None.
        Side Effects:
            None.
        """
        if total_tasks <= 1:
            return 1
        if self._parallel_workers is not None:
            return min(self._parallel_workers, total_tasks)
        detected = os.cpu_count() or 1
        return min(max(detected, 1), total_tasks)


def _stage_a_rank_key(row: _StageAScoredVariant) -> tuple[float, str]:
    """
    Build deterministic Stage-A ranking key for sorted shortlist output.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        row: Stage A scored row.
    Returns:
        tuple[float, str]: Ranking key (`total_return_pct DESC`, `base_variant_key ASC`).
    Assumptions:
        Smaller tuple value means higher deterministic rank.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (-row.total_return_pct, row.base_variant.base_variant_key)


def _stage_b_rank_key(row: _StageBScoredVariant) -> tuple[float, str]:
    """
    Build deterministic Stage-B ranking key for sorted top-K output.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        row: Stage B scored row.
    Returns:
        tuple[float, str]: Ranking key (`total_return_pct DESC`, `variant_key ASC`).
    Assumptions:
        Smaller tuple value means higher deterministic rank.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (-row.total_return_pct, row.variant_key)


def _stage_a_outranks(
    *,
    candidate: _StageAScoredVariant,
    baseline: _StageAScoredVariant,
) -> bool:
    """
    Check whether Stage-A candidate is strictly better than current shortlist baseline.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        candidate: New Stage A candidate.
        baseline: Current worst retained Stage A row.
    Returns:
        bool: `True` when candidate must replace baseline in bounded heap.
    Assumptions:
        Stage A ranking key is `total_return_pct DESC, base_variant_key ASC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _stage_a_rank_key(candidate) < _stage_a_rank_key(baseline)


def _stage_b_outranks(
    *,
    candidate: _StageBScoredVariant,
    baseline: _StageBScoredVariant,
) -> bool:
    """
    Check whether Stage-B candidate is strictly better than current top-K baseline.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        candidate: New Stage B candidate.
        baseline: Current worst retained Stage B row.
    Returns:
        bool: `True` when candidate must replace baseline in bounded heap.
    Assumptions:
        Stage B ranking key is `total_return_pct DESC, variant_key ASC`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _stage_b_rank_key(candidate) < _stage_b_rank_key(baseline)


def _descending_text_key(*, value: str) -> tuple[int, ...]:
    """
    Encode text into a tuple comparable in reverse lexicographical order.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        value: Source deterministic tie-break key.
    Returns:
        tuple[int, ...]:
            Comparable tuple where lexicographically larger source strings become smaller values.
    Assumptions:
        Sentinel `0` is greater than any negated code point and handles prefix ordering.
    Raises:
        None.
    Side Effects:
        Allocates one tuple for each retained heap entry.
    """
    return (*(-ord(char) for char in value), 0)


def _details_scorer(
    *,
    scorer: BacktestStagedVariantScorer,
) -> BacktestStagedVariantScorerWithDetails | None:
    """
    Resolve optional details extension from scorer without breaking base scorer contract.

    Args:
        scorer: Base Stage A/Stage B scorer port.
    Returns:
        BacktestStagedVariantScorerWithDetails | None:
            Scorer cast to details extension when available.
    Assumptions:
        Scorer extension is detected by method presence and validated by static typing.
    Raises:
        None.
    Side Effects:
        None.
    """
    if getattr(scorer, "score_variant_with_details", None) is None:
        return None
    return cast(BacktestStagedVariantScorerWithDetails, scorer)


def _extract_total_return_pct(*, metrics: Mapping[str, float]) -> float:
    """
    Extract `Total Return [%]` metric from scorer payload and validate numeric type.

    Args:
        metrics: Scorer metrics mapping.
    Returns:
        float: Deterministic ranking value for staged sorting.
    Assumptions:
        Scorer payload contains `Total Return [%]` key.
    Raises:
        ValueError: If metric key is absent or value is not numeric scalar.
    Side Effects:
        None.
    """
    value = metrics.get(TOTAL_RETURN_METRIC_LITERAL)
    if value is None:
        raise ValueError(f"scorer payload must contain '{TOTAL_RETURN_METRIC_LITERAL}' metric")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric '{TOTAL_RETURN_METRIC_LITERAL}' must be numeric")
    return float(value)


def _execution_params_to_mapping(
    *,
    params: ExecutionParamsV1,
) -> Mapping[str, BacktestVariantScalar]:
    """
    Convert execution params value-object into deterministic scalar mapping payload.

    Args:
        params: Effective execution params value object from scorer details payload.
    Returns:
        Mapping[str, BacktestVariantScalar]: Deterministic execution scalar mapping.
    Assumptions:
        Params object already passed value-object validation invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "direction_mode": params.direction_mode,
        "sizing_mode": params.sizing_mode,
        "init_cash_quote": float(params.init_cash_quote),
        "fixed_quote": float(params.fixed_quote),
        "safe_profit_percent": float(params.safe_profit_percent),
        "fee_pct": float(params.fee_pct),
        "slippage_pct": float(params.slippage_pct),
    }


def _risk_params_to_mapping(*, params: RiskParamsV1) -> Mapping[str, BacktestVariantScalar]:
    """
    Convert risk params value-object into deterministic scalar mapping payload.

    Args:
        params: Effective risk params value object from scorer details payload.
    Returns:
        Mapping[str, BacktestVariantScalar]: Deterministic risk scalar mapping.
    Assumptions:
        Params object already passed value-object validation invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "sl_enabled": bool(params.sl_enabled),
        "sl_pct": float(params.sl_pct) if params.sl_pct is not None else None,
        "tp_enabled": bool(params.tp_enabled),
        "tp_pct": float(params.tp_pct) if params.tp_pct is not None else None,
    }


def _cancel_checker_from_run_control(
    *,
    run_control: BacktestRunControlV1 | None,
) -> Callable[[str], None] | None:
    """
    Build stage-aware cancellation checker callback from optional run-control object.

    Args:
        run_control: Optional cooperative run-control object.
    Returns:
        Callable[[str], None] | None:
            Stage checker callback or `None` when run-control is not configured.
    Assumptions:
        Callback is invoked from hot loops and must stay allocation-light.
    Raises:
        None.
    Side Effects:
        None.
    """
    if run_control is None:
        return None

    def _check(stage: str) -> None:
        """
        Check cancellation/deadline for one stage.

        Args:
            stage: Stage literal where check is executed.
        Returns:
            None.
        Assumptions:
            Stage name is a non-empty deterministic literal.
        Raises:
            BacktestRunCancelledV1: If run-control reports cancellation/deadline.
        Side Effects:
            None.
        """
        run_control.raise_if_cancelled(stage=stage)

    return _check


__all__ = [
    "BacktestStagedRunResultV1",
    "BacktestStagedRunnerV1",
    "TOTAL_RETURN_METRIC_LITERAL",
]
