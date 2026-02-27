from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from types import MappingProxyType
from typing import Callable, Iterator, Mapping

from trading.contexts.backtest.application.dto import (
    BacktestRankingConfig,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestStagedVariantScorer,
    BacktestStagedVariantScorerWithDetails,
    BacktestVariantScoreDetailsV1,
)
from trading.contexts.backtest.application.services.grid_builder_v1 import (
    STAGE_A_LITERAL,
    STAGE_B_LITERAL,
    BacktestGridBuildContextV1,
    BacktestStageABaseVariant,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection

_TOTAL_RETURN_METRIC_LITERAL = "Total Return [%]"
_MAX_DRAWDOWN_METRIC_LITERAL = "Max. Drawdown [%]"
_TOTAL_RETURN_METRIC_KEY_LITERAL = "total_return_pct"
_MAX_DRAWDOWN_METRIC_KEY_LITERAL = "max_drawdown_pct"
_RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL = "return_over_max_drawdown"
_PROFIT_FACTOR_METRIC_KEY_LITERAL = "profit_factor"
_DIRECTION_ASC_LITERAL = "ASC"
_DIRECTION_DESC_LITERAL = "DESC"
_SECONDARY_METRIC_COMPONENT_DEFAULT = 0.0

StageACheckpointCallbackV1 = Callable[[int, int], None]
StageBCheckpointCallbackV1 = Callable[
    [
        int,
        int,
        tuple["BacktestStageBScoredVariantV1", ...],
        Mapping[str, "BacktestStageBTaskV1"],
    ],
    None,
]
CancelCheckerV1 = Callable[[str], None]
StageAHeapEntryV1 = tuple[
    float,
    float,
    tuple[int, ...],
    "BacktestStageAScoredVariantV1",
]
StageBHeapEntryV1 = tuple[
    float,
    float,
    tuple[int, ...],
    "BacktestStageBScoredVariantV1",
    "BacktestStageBTaskV1",
    BacktestVariantScoreDetailsV1 | None,
]
_DEFAULT_RANKING_CONFIG_V1 = BacktestRankingConfig()
_METRIC_DIRECTION_BY_LITERAL_V1 = MappingProxyType(
    {
        _TOTAL_RETURN_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _MAX_DRAWDOWN_METRIC_KEY_LITERAL: _DIRECTION_ASC_LITERAL,
        _RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _PROFIT_FACTOR_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
    }
)
_SCORER_METRIC_KEYS_BY_LITERAL_V1 = MappingProxyType(
    {
        _TOTAL_RETURN_METRIC_KEY_LITERAL: (
            _TOTAL_RETURN_METRIC_KEY_LITERAL,
            _TOTAL_RETURN_METRIC_LITERAL,
        ),
        _MAX_DRAWDOWN_METRIC_KEY_LITERAL: (
            _MAX_DRAWDOWN_METRIC_KEY_LITERAL,
            _MAX_DRAWDOWN_METRIC_LITERAL,
        ),
        _RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL: (
            _RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL,
        ),
        _PROFIT_FACTOR_METRIC_KEY_LITERAL: (_PROFIT_FACTOR_METRIC_KEY_LITERAL,),
    }
)


@dataclass(frozen=True, slots=True)
class _ResolvedRankingPlanV1:
    """
    Pre-resolved ranking plan for one stage run to avoid repeated literal lookups in hot loops.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    primary_metric: str
    primary_direction: str
    primary_scorer_metric_keys: tuple[str, ...]
    secondary_metric: str | None
    secondary_direction: str | None
    secondary_scorer_metric_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BacktestStageAScoredVariantV1:
    """
    Deterministic scored Stage-A row shared by sync and job-runner execution paths.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    base_variant: BacktestStageABaseVariant
    total_return_pct: float


@dataclass(frozen=True, slots=True)
class BacktestStageBTaskV1:
    """
    Deterministic Stage-B task payload shared by sync and job-runner execution paths.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]]
    risk_params: Mapping[str, BacktestVariantScalar]


@dataclass(frozen=True, slots=True)
class BacktestStageBScoredVariantV1:
    """
    Deterministic scored Stage-B row shared by sync and job-runner execution paths.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    total_return_pct: float


class BacktestStagedCoreRunnerV1:
    """
    Shared staged scoring core (Stage A + Stage B) for sync and job-runner flows.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
    """

    def __init__(
        self,
        *,
        batch_size_default: int = 256,
        configurable_ranking_enabled: bool = True,
    ) -> None:
        """
        Initialize shared staged core runner with deterministic checkpoint batch size.

        Args:
            batch_size_default: Default checkpoint boundary for cancellation/progress hooks.
            configurable_ranking_enabled:
                Feature-flag guard for request/runtime ranking configuration behavior.
        Returns:
            None.
        Assumptions:
            Hook callbacks are lightweight and deterministic for identical inputs.
        Raises:
            ValueError: If `batch_size_default` is non-positive.
        Side Effects:
            None.
        """
        if batch_size_default <= 0:
            raise ValueError("BacktestStagedCoreRunnerV1.batch_size_default must be > 0")
        if not isinstance(configurable_ranking_enabled, bool):
            raise ValueError(
                "BacktestStagedCoreRunnerV1.configurable_ranking_enabled must be bool"
            )
        self._batch_size_default = batch_size_default
        self._configurable_ranking_enabled = configurable_ranking_enabled

    def run_stage_a(
        self,
        *,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        shortlist_limit: int,
        ranking: BacktestRankingConfig | None = None,
        batch_size: int | None = None,
        cancel_checker: CancelCheckerV1 | None = None,
        on_checkpoint: StageACheckpointCallbackV1 | None = None,
    ) -> tuple[BacktestStageAScoredVariantV1, ...]:
        """
        Score Stage-A variants with bounded heap shortlist and optional hooks.

        Args:
            grid_context: Deterministic staged grid context.
            candles: Dense warmup-inclusive candle arrays.
            scorer: Stage scorer contract implementation.
            shortlist_limit: Maximum number of Stage-A rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`)
                from request/runtime defaults.
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback `(processed, total)`.
        Returns:
            tuple[BacktestStageAScoredVariantV1, ...]:
                Deterministically ranked Stage-A shortlist rows.
        Assumptions:
            Final deterministic tie-break for Stage A is `base_variant_key ASC`.
        Raises:
            ValueError: If limit or batch-size is invalid or scorer payload is malformed.
        Side Effects:
            None.
        """
        if shortlist_limit <= 0:
            raise ValueError("BacktestStagedCoreRunnerV1 shortlist_limit must be > 0")
        ranking_plan = _resolve_ranking_plan(
            ranking=_effective_ranking_config(
                ranking=ranking,
                configurable_ranking_enabled=self._configurable_ranking_enabled,
            )
        )
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(grid_context.stage_a_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_A_LITERAL)

        shortlist_heap: list[StageAHeapEntryV1] = []
        processed = 0
        # HOT PATH: Stage-A scoring loop for all base grid variants.
        for base_variant in grid_context.iter_stage_a_variants():
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL)
            row, metrics = self._score_stage_a_variant_with_metrics(
                base_variant=base_variant,
                candles=candles,
                scorer=scorer,
            )
            heap_entry = _stage_a_heap_entry(
                row=row,
                metrics=metrics,
                ranking_plan=ranking_plan,
            )
            if len(shortlist_heap) < shortlist_limit:
                heappush(shortlist_heap, heap_entry)
            elif _heap_entry_outranks(candidate=heap_entry, baseline=shortlist_heap[0]):
                heapreplace(shortlist_heap, heap_entry)

            processed += 1
            if processed % effective_batch != 0 and processed != total:
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL)
            if on_checkpoint is not None:
                on_checkpoint(processed, total)

        return _stage_a_rows_from_heap(heap=shortlist_heap)

    def run_stage_b(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: tuple[BacktestStageAScoredVariantV1, ...],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        batch_size: int | None = None,
        cancel_checker: CancelCheckerV1 | None = None,
        on_checkpoint: StageBCheckpointCallbackV1 | None = None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV1, ...], Mapping[str, BacktestStageBTaskV1]]:
        """
        Score Stage-B variants with bounded top-K heap and optional checkpoint hooks.

        Args:
            template: Effective run template used for deterministic variant key build.
            grid_context: Deterministic staged grid context.
            shortlist: Deterministically sorted Stage-A shortlist rows.
            candles: Dense warmup-inclusive candle arrays.
            scorer: Stage scorer contract implementation.
            top_k_limit: Maximum number of Stage-B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`)
                from request/runtime defaults.
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint:
                Optional checkpoint callback with current ranked frontier snapshot.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV1, ...], Mapping[str, BacktestStageBTaskV1]]:
                Deterministically ranked Stage-B rows and tasks mapping by `variant_key`.
        Assumptions:
            Final deterministic tie-break for Stage B is always `variant_key ASC`.
        Raises:
            ValueError: If limits/batch-size are invalid or scorer payload is malformed.
        Side Effects:
            None.
        """
        rows, tasks, _ = self.run_stage_b_with_details(
            template=template,
            grid_context=grid_context,
            shortlist=shortlist,
            candles=candles,
            scorer=scorer,
            top_k_limit=top_k_limit,
            ranking=ranking,
            details_scorer=None,
            batch_size=batch_size,
            cancel_checker=cancel_checker,
            on_checkpoint=on_checkpoint,
        )
        return (rows, tasks)

    def run_stage_b_with_details(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: tuple[BacktestStageAScoredVariantV1, ...],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        details_scorer: BacktestStagedVariantScorerWithDetails | None = None,
        batch_size: int | None = None,
        cancel_checker: CancelCheckerV1 | None = None,
        on_checkpoint: StageBCheckpointCallbackV1 | None = None,
    ) -> tuple[
        tuple[BacktestStageBScoredVariantV1, ...],
        Mapping[str, BacktestStageBTaskV1],
        Mapping[str, BacktestVariantScoreDetailsV1],
    ]:
        """
        Score Stage-B variants and optionally retain details for variants currently in top-k heap.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            template: Effective run template used for deterministic variant key build.
            grid_context: Deterministic staged grid context.
            shortlist: Deterministically sorted Stage-A shortlist rows.
            candles: Dense warmup-inclusive candle arrays.
            scorer: Stage scorer contract implementation.
            top_k_limit: Maximum number of Stage-B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`)
                from request/runtime defaults.
            details_scorer:
                Optional scorer extension used to retain detailed execution payloads in heap.
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint:
                Optional checkpoint callback with current ranked frontier snapshot.
        Returns:
            tuple[
                tuple[BacktestStageBScoredVariantV1, ...],
                Mapping[str, BacktestStageBTaskV1],
                Mapping[str, BacktestVariantScoreDetailsV1],
            ]:
                Ranked rows, tasks mapping, and retained details for current top-k variants.
        Assumptions:
            Final deterministic tie-break for Stage B is always `variant_key ASC`.
        Raises:
            ValueError: If limits/batch-size are invalid or scorer payload is malformed.
        Side Effects:
            Retained details are bounded by heap capacity and dropped on heap evictions.
        """
        if top_k_limit <= 0:
            raise ValueError("BacktestStagedCoreRunnerV1 top_k_limit must be > 0")
        ranking_plan = _resolve_ranking_plan(
            ranking=_effective_ranking_config(
                ranking=ranking,
                configurable_ranking_enabled=self._configurable_ranking_enabled,
            )
        )
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(grid_context.stage_b_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_B_LITERAL)

        top_heap: list[StageBHeapEntryV1] = []
        processed = 0
        # HOT PATH: Stage-B scoring loop for shortlist x risk expansion tasks.
        for task in self._iter_stage_b_tasks(
            template=template,
            grid_context=grid_context,
            shortlist=shortlist,
        ):
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL)
            row, details, metrics = self._score_stage_b_task_with_optional_details(
                task=task,
                candles=candles,
                scorer=scorer,
                details_scorer=details_scorer,
            )
            heap_entry = _stage_b_heap_entry(
                row=row,
                task=task,
                details=details,
                metrics=metrics,
                ranking_plan=ranking_plan,
            )
            if len(top_heap) < top_k_limit:
                heappush(top_heap, heap_entry)
            elif _heap_entry_outranks(candidate=heap_entry, baseline=top_heap[0]):
                heapreplace(top_heap, heap_entry)

            processed += 1
            if processed % effective_batch != 0 and processed != total:
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL)
            if on_checkpoint is not None:
                on_checkpoint(
                    processed,
                    total,
                    _stage_b_rows_from_heap(heap=top_heap),
                    _stage_b_tasks_from_heap(heap=top_heap),
                )

        return (
            _stage_b_rows_from_heap(heap=top_heap),
            _stage_b_tasks_from_heap(heap=top_heap),
            _stage_b_details_from_heap(heap=top_heap),
        )

    def _iter_stage_b_tasks(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: tuple[BacktestStageAScoredVariantV1, ...],
    ) -> Iterator[BacktestStageBTaskV1]:
        """
        Iterate deterministic Stage-B scoring tasks from Stage-A shortlist.

        Args:
            template: Effective run template.
            grid_context: Deterministic staged grid context.
            shortlist: Stage-A shortlist rows.
        Returns:
            Iterator[BacktestStageBTaskV1]: Deterministic task stream.
        Assumptions:
            Variant index contract is `(shortlist_index * risk_total) + risk_index`.
        Raises:
            None.
        Side Effects:
            None.
        """
        risk_variants = grid_context.risk_variants
        risk_total = len(risk_variants)
        for shortlist_index, stage_a_row in enumerate(shortlist):
            base_variant = stage_a_row.base_variant
            for risk_variant in risk_variants:
                variant_index = (shortlist_index * risk_total) + risk_variant.risk_index
                variant_key = build_backtest_variant_key_v1(
                    indicator_variant_key=base_variant.indicator_variant_key,
                    direction_mode=template.direction_mode,
                    sizing_mode=template.sizing_mode,
                    signals=base_variant.signal_params,
                    risk_params=risk_variant.risk_params,
                    execution_params=template.execution_params,
                )
                yield BacktestStageBTaskV1(
                    variant_index=variant_index,
                    indicator_variant_key=base_variant.indicator_variant_key,
                    variant_key=variant_key,
                    indicator_selections=base_variant.indicator_selections,
                    signal_params=base_variant.signal_params,
                    risk_params=risk_variant.risk_params,
                )

    def _score_stage_a_variant(
        self,
        *,
        base_variant: BacktestStageABaseVariant,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> BacktestStageAScoredVariantV1:
        """
        Score one Stage-A base variant and return deterministic row.

        Args:
            base_variant: Stage-A base variant.
            candles: Dense warmup-inclusive candles.
            scorer: Stage scorer contract implementation.
        Returns:
            BacktestStageAScoredVariantV1: Deterministic scored row.
        Assumptions:
            Stage A always uses disabled SL/TP payload.
        Raises:
            ValueError: If scorer payload lacks ranking metric.
        Side Effects:
            None.
        """
        row, _ = self._score_stage_a_variant_with_metrics(
            base_variant=base_variant,
            candles=candles,
            scorer=scorer,
        )
        return row

    def _score_stage_a_variant_with_metrics(
        self,
        *,
        base_variant: BacktestStageABaseVariant,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> tuple[BacktestStageAScoredVariantV1, Mapping[str, float]]:
        """
        Score one Stage-A base variant and return both deterministic row and raw metrics payload.

        Args:
            base_variant: Stage-A base variant.
            candles: Dense warmup-inclusive candles.
            scorer: Stage scorer contract implementation.
        Returns:
            tuple[BacktestStageAScoredVariantV1, Mapping[str, float]]:
                Deterministic scored row and raw metrics mapping from scorer.
        Assumptions:
            Stage A always uses disabled SL/TP payload.
        Raises:
            ValueError: If scorer payload lacks required ranking metric literals.
        Side Effects:
            None.
        """
        metrics = scorer.score_variant(
            stage=STAGE_A_LITERAL,
            candles=candles,
            indicator_selections=base_variant.indicator_selections,
            signal_params=base_variant.signal_params,
            risk_params={
                "sl_enabled": False,
                "sl_pct": None,
                "tp_enabled": False,
                "tp_pct": None,
            },
            indicator_variant_key=base_variant.indicator_variant_key,
            variant_key=base_variant.base_variant_key,
        )
        return (
            BacktestStageAScoredVariantV1(
                base_variant=base_variant,
                total_return_pct=_extract_metric_value_for_literal(
                    metrics=metrics,
                    metric_literal=_TOTAL_RETURN_METRIC_KEY_LITERAL,
                ),
            ),
            metrics,
        )

    def _score_stage_b_task(
        self,
        *,
        task: BacktestStageBTaskV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> BacktestStageBScoredVariantV1:
        """
        Score one Stage-B task and return deterministic ranked row.

        Args:
            task: Stage-B task payload.
            candles: Dense warmup-inclusive candles.
            scorer: Stage scorer contract implementation.
        Returns:
            BacktestStageBScoredVariantV1: Deterministic ranked row.
        Assumptions:
            Stage B scoring uses risk-enabled payload from task.
        Raises:
            ValueError: If scorer payload lacks ranking metric.
        Side Effects:
            None.
        """
        row, _ = self._score_stage_b_task_with_metrics(
            task=task,
            candles=candles,
            scorer=scorer,
        )
        return row

    def _score_stage_b_task_with_metrics(
        self,
        *,
        task: BacktestStageBTaskV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
    ) -> tuple[BacktestStageBScoredVariantV1, Mapping[str, float]]:
        """
        Score one Stage-B task and return both deterministic row and raw metrics payload.

        Args:
            task: Stage-B task payload.
            candles: Dense warmup-inclusive candles.
            scorer: Stage scorer contract implementation.
        Returns:
            tuple[BacktestStageBScoredVariantV1, Mapping[str, float]]:
                Deterministic ranked row and raw metrics mapping from scorer.
        Assumptions:
            Stage B scoring uses risk-enabled payload from task.
        Raises:
            ValueError: If scorer payload lacks required ranking metric literals.
        Side Effects:
            None.
        """
        metrics = scorer.score_variant(
            stage=STAGE_B_LITERAL,
            candles=candles,
            indicator_selections=task.indicator_selections,
            signal_params=task.signal_params,
            risk_params=task.risk_params,
            indicator_variant_key=task.indicator_variant_key,
            variant_key=task.variant_key,
        )
        return (
            BacktestStageBScoredVariantV1(
                variant_index=task.variant_index,
                indicator_variant_key=task.indicator_variant_key,
                variant_key=task.variant_key,
                total_return_pct=_extract_metric_value_for_literal(
                    metrics=metrics,
                    metric_literal=_TOTAL_RETURN_METRIC_KEY_LITERAL,
                ),
            ),
            metrics,
        )

    def _score_stage_b_task_with_optional_details(
        self,
        *,
        task: BacktestStageBTaskV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        details_scorer: BacktestStagedVariantScorerWithDetails | None,
    ) -> tuple[
        BacktestStageBScoredVariantV1,
        BacktestVariantScoreDetailsV1 | None,
        Mapping[str, float],
    ]:
        """
        Score Stage-B task and optionally return detailed execution payload for retained top-k rows.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
        Args:
            task: Stage-B task payload.
            candles: Dense warmup-inclusive candles.
            scorer: Base Stage-B scorer contract implementation.
            details_scorer: Optional details scorer extension.
        Returns:
            tuple[
                BacktestStageBScoredVariantV1,
                BacktestVariantScoreDetailsV1 | None,
                Mapping[str, float],
            ]:
                Ranked row, optional details payload, and metrics used for ranking keys.
        Assumptions:
            Detailed payload metrics are deterministic and equivalent to ranking metric.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        if details_scorer is None:
            row, metrics = self._score_stage_b_task_with_metrics(
                task=task,
                candles=candles,
                scorer=scorer,
            )
            return (row, None, metrics)

        details = details_scorer.score_variant_with_details(
            stage=STAGE_B_LITERAL,
            candles=candles,
            indicator_selections=task.indicator_selections,
            signal_params=task.signal_params,
            risk_params=task.risk_params,
            indicator_variant_key=task.indicator_variant_key,
            variant_key=task.variant_key,
        )
        return (
            BacktestStageBScoredVariantV1(
                variant_index=task.variant_index,
                indicator_variant_key=task.indicator_variant_key,
                variant_key=task.variant_key,
                total_return_pct=_extract_metric_value_for_literal(
                    metrics=details.metrics,
                    metric_literal=_TOTAL_RETURN_METRIC_KEY_LITERAL,
                ),
            ),
            details,
            details.metrics,
        )

    def _resolve_batch_size(self, *, batch_size: int | None) -> int:
        """
        Resolve effective checkpoint batch size.

        Args:
            batch_size: Optional override.
        Returns:
            int: Effective positive checkpoint batch size.
        Assumptions:
            `None` means use constructor-level default.
        Raises:
            ValueError: If override value is non-positive.
        Side Effects:
            None.
        """
        if batch_size is None:
            return self._batch_size_default
        if batch_size <= 0:
            raise ValueError("stage batch_size must be > 0")
        return batch_size


def _effective_ranking_config(
    *,
    ranking: BacktestRankingConfig | None,
    configurable_ranking_enabled: bool,
) -> BacktestRankingConfig:
    """
    Resolve effective ranking config under explicit feature-flag guard policy.

    Args:
        ranking: Optional request/runtime ranking config.
        configurable_ranking_enabled: Feature-flag literal controlling ranking behavior switch.
    Returns:
        BacktestRankingConfig: Effective ranking config used by Stage A/Stage B loops.
    Assumptions:
        Legacy deterministic behavior is `total_return_pct DESC`.
    Raises:
        ValueError: If resulting ranking config cannot be normalized.
    Side Effects:
        None.
    """
    if not configurable_ranking_enabled:
        return _DEFAULT_RANKING_CONFIG_V1
    if ranking is None:
        return _DEFAULT_RANKING_CONFIG_V1
    return ranking


def _resolve_ranking_plan(*, ranking: BacktestRankingConfig) -> _ResolvedRankingPlanV1:
    """
    Resolve ranking literals into hot-loop plan with directions and scorer metric aliases.

    Args:
        ranking: Effective ranking config.
    Returns:
        _ResolvedRankingPlanV1: Ranking plan with pre-validated metric aliases.
    Assumptions:
        Ranking literals were normalized by DTO/config contracts before this step.
    Raises:
        ValueError: If metric literal is unsupported by staged ranking plan.
    Side Effects:
        None.
    """
    primary_metric = ranking.primary_metric
    primary_direction = _METRIC_DIRECTION_BY_LITERAL_V1.get(primary_metric)
    primary_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V1.get(primary_metric)
    if primary_direction is None or primary_keys is None:
        raise ValueError(f"unsupported primary ranking metric: {primary_metric!r}")

    secondary_metric = ranking.secondary_metric
    secondary_direction: str | None = None
    secondary_keys: tuple[str, ...] = ()
    if secondary_metric is not None:
        secondary_direction = _METRIC_DIRECTION_BY_LITERAL_V1.get(secondary_metric)
        resolved_secondary_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V1.get(secondary_metric)
        if secondary_direction is None or resolved_secondary_keys is None:
            raise ValueError(f"unsupported secondary ranking metric: {secondary_metric!r}")
        secondary_keys = resolved_secondary_keys

    return _ResolvedRankingPlanV1(
        primary_metric=primary_metric,
        primary_direction=primary_direction,
        primary_scorer_metric_keys=primary_keys,
        secondary_metric=secondary_metric,
        secondary_direction=secondary_direction,
        secondary_scorer_metric_keys=secondary_keys,
    )


def _stage_a_heap_entry(
    *,
    row: BacktestStageAScoredVariantV1,
    metrics: Mapping[str, float],
    ranking_plan: _ResolvedRankingPlanV1,
) -> StageAHeapEntryV1:
    """
    Build Stage-A heap entry where smallest tuple is current worst retained shortlist row.

    Args:
        row: Scored Stage-A row.
        metrics: Raw scorer metrics payload.
        ranking_plan: Pre-resolved ranking plan.
    Returns:
        StageAHeapEntryV1: Heap entry preserving deterministic tie-break by base key.
    Assumptions:
        Final tie-break for Stage A is `base_variant_key ASC`.
    Raises:
        ValueError: If one ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = _heap_metric_component_from_literal(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    secondary_component = _SECONDARY_METRIC_COMPONENT_DEFAULT
    if ranking_plan.secondary_metric is not None and ranking_plan.secondary_direction is not None:
        secondary_component = _heap_metric_component_from_literal(
            metrics=metrics,
            metric_literal=ranking_plan.secondary_metric,
            metric_direction=ranking_plan.secondary_direction,
            scorer_metric_keys=ranking_plan.secondary_scorer_metric_keys,
        )
    return (
        primary_component,
        secondary_component,
        _descending_text_key(value=row.base_variant.base_variant_key),
        row,
    )


def _stage_b_heap_entry(
    *,
    row: BacktestStageBScoredVariantV1,
    task: BacktestStageBTaskV1,
    details: BacktestVariantScoreDetailsV1 | None,
    metrics: Mapping[str, float],
    ranking_plan: _ResolvedRankingPlanV1,
) -> StageBHeapEntryV1:
    """
    Build Stage-B heap entry where smallest tuple is current worst retained top-k row.

    Args:
        row: Scored Stage-B row.
        task: Stage-B task payload corresponding to scored row.
        details: Optional retained details payload.
        metrics: Raw scorer metrics payload.
        ranking_plan: Pre-resolved ranking plan.
    Returns:
        StageBHeapEntryV1: Heap entry preserving deterministic tie-break by variant key.
    Assumptions:
        Final tie-break for Stage B is always `variant_key ASC`.
    Raises:
        ValueError: If one ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = _heap_metric_component_from_literal(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    secondary_component = _SECONDARY_METRIC_COMPONENT_DEFAULT
    if ranking_plan.secondary_metric is not None and ranking_plan.secondary_direction is not None:
        secondary_component = _heap_metric_component_from_literal(
            metrics=metrics,
            metric_literal=ranking_plan.secondary_metric,
            metric_direction=ranking_plan.secondary_direction,
            scorer_metric_keys=ranking_plan.secondary_scorer_metric_keys,
        )
    return (
        primary_component,
        secondary_component,
        _descending_text_key(value=row.variant_key),
        row,
        task,
        details,
    )


def _heap_entry_outranks(
    *,
    candidate: StageAHeapEntryV1 | StageBHeapEntryV1,
    baseline: StageAHeapEntryV1 | StageBHeapEntryV1,
) -> bool:
    """
    Check whether candidate heap entry outranks current baseline entry in bounded heap.

    Args:
        candidate: Candidate heap entry.
        baseline: Baseline (current worst retained) heap entry.
    Returns:
        bool: `True` when candidate must replace baseline in heap.
    Assumptions:
        Heap tuple ordering keeps worst retained row at root.
    Raises:
        None.
    Side Effects:
        None.
    """
    return candidate[:3] > baseline[:3]


def _heap_metric_component_from_literal(
    *,
    metrics: Mapping[str, float],
    metric_literal: str,
    metric_direction: str,
    scorer_metric_keys: tuple[str, ...],
) -> float:
    """
    Build deterministic heap metric component where larger value means better candidate.

    Args:
        metrics: Raw scorer metrics payload.
        metric_literal: Ranking metric literal from ranking config.
        metric_direction: Metric direction literal (`ASC` or `DESC`).
        scorer_metric_keys: Ordered scorer metric key aliases.
    Returns:
        float: Heap metric component with direction transform already applied.
    Assumptions:
        Ranking metrics are numeric scalars provided by scorer contract.
    Raises:
        ValueError: If metric literal is unsupported or metric value is absent/non-numeric.
    Side Effects:
        None.
    """
    value = _extract_metric_value(
        metrics=metrics,
        metric_literal=metric_literal,
        scorer_metric_keys=scorer_metric_keys,
    )
    if metric_direction == _DIRECTION_DESC_LITERAL:
        return value
    if metric_direction == _DIRECTION_ASC_LITERAL:
        return -value
    raise ValueError(
        f"unsupported ranking direction for '{metric_literal}': {metric_direction!r}"
    )


def _descending_text_key(*, value: str) -> tuple[int, ...]:
    """
    Encode text key into reverse-lexicographic tuple for min-heap baseline slot.

    Args:
        value: Deterministic tie-break string key.
    Returns:
        tuple[int, ...]: Reverse-comparable tuple for heap entry.
    Assumptions:
        Sentinel `0` keeps strict ordering for prefix strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (*(-ord(char) for char in value), 0)


def _extract_metric_value_for_literal(
    *,
    metrics: Mapping[str, float],
    metric_literal: str,
) -> float:
    """
    Extract numeric metric value by ranking literal using scorer metric alias mapping.

    Args:
        metrics: Raw scorer metrics payload.
        metric_literal: Ranking metric literal (`total_return_pct`, etc.).
    Returns:
        float: Numeric metric value.
    Assumptions:
        Metric literal exists in supported v1 ranking set.
    Raises:
        ValueError: If metric literal is unsupported or metric value is absent/non-numeric.
    Side Effects:
        None.
    """
    scorer_metric_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V1.get(metric_literal)
    if scorer_metric_keys is None:
        raise ValueError(f"unsupported ranking metric literal: {metric_literal!r}")
    return _extract_metric_value(
        metrics=metrics,
        metric_literal=metric_literal,
        scorer_metric_keys=scorer_metric_keys,
    )


def _extract_metric_value(
    *,
    metrics: Mapping[str, float],
    metric_literal: str,
    scorer_metric_keys: tuple[str, ...],
) -> float:
    """
    Extract one numeric metric from scorer payload using ordered alias keys.

    Args:
        metrics: Raw scorer metrics payload.
        metric_literal: Ranking metric literal for deterministic error messages.
        scorer_metric_keys: Ordered scorer key aliases.
    Returns:
        float: Numeric metric value.
    Assumptions:
        First present alias key is authoritative for deterministic scoring semantics.
    Raises:
        ValueError: If metric is absent or non-numeric.
    Side Effects:
        None.
    """
    for scorer_metric_key in scorer_metric_keys:
        value = metrics.get(scorer_metric_key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"metric '{metric_literal}' must be numeric")
        return float(value)
    raise ValueError(
        f"scorer payload must contain '{metric_literal}' metric "
        f"(aliases: {', '.join(scorer_metric_keys)})"
    )


def _stage_a_rows_from_heap(
    *,
    heap: list[StageAHeapEntryV1],
) -> tuple[BacktestStageAScoredVariantV1, ...]:
    """
    Materialize deterministic Stage-A shortlist rows from bounded heap entries.

    Args:
        heap: Internal bounded Stage-A heap entries.
    Returns:
        tuple[BacktestStageAScoredVariantV1, ...]: Deterministically ranked Stage-A rows.
    Assumptions:
        Heap ordering uses ranking components and base-key tie-break transform.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(entry[3] for entry in sorted(heap, key=lambda item: item[:3], reverse=True))


def _stage_b_rows_from_heap(
    *,
    heap: list[StageBHeapEntryV1],
) -> tuple[BacktestStageBScoredVariantV1, ...]:
    """
    Materialize deterministic Stage-B ranked rows from bounded heap entries.

    Args:
        heap: Internal bounded Stage-B heap entries.
    Returns:
        tuple[BacktestStageBScoredVariantV1, ...]: Deterministically ranked Stage-B rows.
    Assumptions:
        Heap ordering uses ranking components and final `variant_key` tie-break transform.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(entry[3] for entry in sorted(heap, key=lambda item: item[:3], reverse=True))


def _stage_b_tasks_from_heap(
    *,
    heap: list[StageBHeapEntryV1],
) -> Mapping[str, BacktestStageBTaskV1]:
    """
    Build deterministic `variant_key -> task` mapping from bounded Stage-B heap entries.

    Args:
        heap: Internal bounded Stage-B heap entries.
    Returns:
        Mapping[str, BacktestStageBTaskV1]: Deterministic task mapping.
    Assumptions:
        `variant_key` uniqueness is guaranteed by Stage-B identity builder.
    Raises:
        ValueError: If duplicate `variant_key` is detected in heap.
    Side Effects:
        None.
    """
    mapping: dict[str, BacktestStageBTaskV1] = {}
    for _, _, _, row, task, _ in heap:
        if row.variant_key in mapping:
            raise ValueError("duplicate Stage-B variant_key is not allowed")
        mapping[row.variant_key] = task
    return mapping


def _stage_b_details_from_heap(
    *,
    heap: list[StageBHeapEntryV1],
) -> Mapping[str, BacktestVariantScoreDetailsV1]:
    """
    Build deterministic `variant_key -> details` mapping from bounded Stage-B heap entries.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
    Args:
        heap: Internal bounded Stage-B heap entries.
    Returns:
        Mapping[str, BacktestVariantScoreDetailsV1]:
            Retained details payload by deterministic `variant_key`.
    Assumptions:
        Retained details are present only when details scorer extension is enabled.
    Raises:
        ValueError: If duplicate `variant_key` is detected in heap.
    Side Effects:
        None.
    """
    mapping: dict[str, BacktestVariantScoreDetailsV1] = {}
    for _, _, _, row, _, details in heap:
        if details is None:
            continue
        if row.variant_key in mapping:
            raise ValueError("duplicate Stage-B variant_key is not allowed")
        mapping[row.variant_key] = details
    return mapping


__all__ = [
    "BacktestStageAScoredVariantV1",
    "BacktestStageBScoredVariantV1",
    "BacktestStageBTaskV1",
    "BacktestStagedCoreRunnerV1",
]
