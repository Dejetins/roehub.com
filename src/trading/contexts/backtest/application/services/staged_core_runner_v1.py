from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from typing import Callable, Iterator, Mapping

from trading.contexts.backtest.application.dto import RunBacktestTemplate
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

    def __init__(self, *, batch_size_default: int = 256) -> None:
        """
        Initialize shared staged core runner with deterministic checkpoint batch size.

        Args:
            batch_size_default: Default checkpoint boundary for cancellation/progress hooks.
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
        self._batch_size_default = batch_size_default

    def run_stage_a(
        self,
        *,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        shortlist_limit: int,
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
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback `(processed, total)`.
        Returns:
            tuple[BacktestStageAScoredVariantV1, ...]:
                Deterministically ranked Stage-A shortlist rows.
        Assumptions:
            Stage-A ranking key is `Total Return [%] DESC, base_variant_key ASC`.
        Raises:
            ValueError: If limit or batch-size is invalid or scorer payload is malformed.
        Side Effects:
            None.
        """
        if shortlist_limit <= 0:
            raise ValueError("BacktestStagedCoreRunnerV1 shortlist_limit must be > 0")
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(grid_context.stage_a_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_A_LITERAL)

        shortlist_heap: list[
            tuple[float, tuple[int, ...], BacktestStageAScoredVariantV1]
        ] = []
        processed = 0
        # HOT PATH: Stage-A scoring loop for all base grid variants.
        for base_variant in grid_context.iter_stage_a_variants():
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL)
            row = self._score_stage_a_variant(
                base_variant=base_variant,
                candles=candles,
                scorer=scorer,
            )
            if len(shortlist_heap) < shortlist_limit:
                heappush(
                    shortlist_heap,
                    (
                        row.total_return_pct,
                        _descending_text_key(value=row.base_variant.base_variant_key),
                        row,
                    ),
                )
            elif _stage_a_outranks(candidate=row, baseline=shortlist_heap[0][2]):
                heapreplace(
                    shortlist_heap,
                    (
                        row.total_return_pct,
                        _descending_text_key(value=row.base_variant.base_variant_key),
                        row,
                    ),
                )

            processed += 1
            if processed % effective_batch != 0 and processed != total:
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL)
            if on_checkpoint is not None:
                on_checkpoint(processed, total)

        return tuple(sorted((entry[2] for entry in shortlist_heap), key=_stage_a_rank_key))

    def run_stage_b(
        self,
        *,
        template: RunBacktestTemplate,
        grid_context: BacktestGridBuildContextV1,
        shortlist: tuple[BacktestStageAScoredVariantV1, ...],
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        top_k_limit: int,
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
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint:
                Optional checkpoint callback with current ranked frontier snapshot.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV1, ...], Mapping[str, BacktestStageBTaskV1]]:
                Deterministically ranked Stage-B rows and tasks mapping by `variant_key`.
        Assumptions:
            Stage-B ranking key is `Total Return [%] DESC, variant_key ASC`.
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
        details_scorer: BacktestStagedVariantScorerWithDetails | None,
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
            Stage-B ranking key is `Total Return [%] DESC, variant_key ASC`.
        Raises:
            ValueError: If limits/batch-size are invalid or scorer payload is malformed.
        Side Effects:
            Retained details are bounded by heap capacity and dropped on heap evictions.
        """
        if top_k_limit <= 0:
            raise ValueError("BacktestStagedCoreRunnerV1 top_k_limit must be > 0")
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(grid_context.stage_b_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_B_LITERAL)

        top_heap: list[
            tuple[
                float,
                tuple[int, ...],
                BacktestStageBScoredVariantV1,
                BacktestStageBTaskV1,
                BacktestVariantScoreDetailsV1 | None,
            ]
        ] = []
        processed = 0
        # HOT PATH: Stage-B scoring loop for shortlist x risk expansion tasks.
        for task in self._iter_stage_b_tasks(
            template=template,
            grid_context=grid_context,
            shortlist=shortlist,
        ):
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL)
            row, details = self._score_stage_b_task_with_optional_details(
                task=task,
                candles=candles,
                scorer=scorer,
                details_scorer=details_scorer,
            )
            if len(top_heap) < top_k_limit:
                heappush(
                    top_heap,
                    (
                        row.total_return_pct,
                        _descending_text_key(value=row.variant_key),
                        row,
                        task,
                        details,
                    ),
                )
            elif _stage_b_outranks(candidate=row, baseline=top_heap[0][2]):
                heapreplace(
                    top_heap,
                    (
                        row.total_return_pct,
                        _descending_text_key(value=row.variant_key),
                        row,
                        task,
                        details,
                    ),
                )

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
        return BacktestStageAScoredVariantV1(
            base_variant=base_variant,
            total_return_pct=_extract_total_return_pct(metrics=metrics),
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
        metrics = scorer.score_variant(
            stage=STAGE_B_LITERAL,
            candles=candles,
            indicator_selections=task.indicator_selections,
            signal_params=task.signal_params,
            risk_params=task.risk_params,
            indicator_variant_key=task.indicator_variant_key,
            variant_key=task.variant_key,
        )
        return BacktestStageBScoredVariantV1(
            variant_index=task.variant_index,
            indicator_variant_key=task.indicator_variant_key,
            variant_key=task.variant_key,
            total_return_pct=_extract_total_return_pct(metrics=metrics),
        )

    def _score_stage_b_task_with_optional_details(
        self,
        *,
        task: BacktestStageBTaskV1,
        candles: CandleArrays,
        scorer: BacktestStagedVariantScorer,
        details_scorer: BacktestStagedVariantScorerWithDetails | None,
    ) -> tuple[BacktestStageBScoredVariantV1, BacktestVariantScoreDetailsV1 | None]:
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
            tuple[BacktestStageBScoredVariantV1, BacktestVariantScoreDetailsV1 | None]:
                Ranked row and optional details payload.
        Assumptions:
            Detailed payload metrics are deterministic and equivalent to ranking metric.
        Raises:
            ValueError: If scorer payload lacks required ranking metric.
        Side Effects:
            None.
        """
        if details_scorer is None:
            return (
                self._score_stage_b_task(task=task, candles=candles, scorer=scorer),
                None,
            )

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
                total_return_pct=_extract_total_return_pct(metrics=details.metrics),
            ),
            details,
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


def _stage_a_rank_key(row: BacktestStageAScoredVariantV1) -> tuple[float, str]:
    """
    Build deterministic Stage-A rank key (`total_return_pct DESC`, `base_variant_key ASC`).

    Args:
        row: Stage-A scored row.
    Returns:
        tuple[float, str]: Deterministic sortable ranking key.
    Assumptions:
        Smaller tuple means better ranked row.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (-row.total_return_pct, row.base_variant.base_variant_key)


def _stage_b_rank_key(row: BacktestStageBScoredVariantV1) -> tuple[float, str]:
    """
    Build deterministic Stage-B rank key (`total_return_pct DESC`, `variant_key ASC`).

    Args:
        row: Stage-B scored row.
    Returns:
        tuple[float, str]: Deterministic sortable ranking key.
    Assumptions:
        Smaller tuple means better ranked row.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (-row.total_return_pct, row.variant_key)


def _stage_a_outranks(
    *,
    candidate: BacktestStageAScoredVariantV1,
    baseline: BacktestStageAScoredVariantV1,
) -> bool:
    """
    Check whether candidate row outranks baseline in Stage-A ranking.

    Args:
        candidate: Candidate Stage-A row.
        baseline: Baseline Stage-A row.
    Returns:
        bool: `True` when candidate must replace baseline in heap.
    Assumptions:
        Stage-A ranking key is deterministic and total ordering.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _stage_a_rank_key(candidate) < _stage_a_rank_key(baseline)


def _stage_b_outranks(
    *,
    candidate: BacktestStageBScoredVariantV1,
    baseline: BacktestStageBScoredVariantV1,
) -> bool:
    """
    Check whether candidate row outranks baseline in Stage-B ranking.

    Args:
        candidate: Candidate Stage-B row.
        baseline: Baseline Stage-B row.
    Returns:
        bool: `True` when candidate must replace baseline in heap.
    Assumptions:
        Stage-B ranking key is deterministic and total ordering.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _stage_b_rank_key(candidate) < _stage_b_rank_key(baseline)


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


def _extract_total_return_pct(*, metrics: Mapping[str, float]) -> float:
    """
    Extract deterministic ranking metric from scorer payload.

    Args:
        metrics: Scorer metrics mapping.
    Returns:
        float: `Total Return [%]` value as float.
    Assumptions:
        Scorer contract always provides this metric for ranking.
    Raises:
        ValueError: If required metric is absent or non-numeric.
    Side Effects:
        None.
    """
    value = metrics.get(_TOTAL_RETURN_METRIC_LITERAL)
    if value is None:
        raise ValueError(f"scorer payload must contain '{_TOTAL_RETURN_METRIC_LITERAL}' metric")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric '{_TOTAL_RETURN_METRIC_LITERAL}' must be numeric")
    return float(value)


def _stage_b_rows_from_heap(
    *,
    heap: list[
        tuple[
            float,
            tuple[int, ...],
            BacktestStageBScoredVariantV1,
            BacktestStageBTaskV1,
            BacktestVariantScoreDetailsV1 | None,
        ]
    ],
) -> tuple[BacktestStageBScoredVariantV1, ...]:
    """
    Materialize deterministic ranked Stage-B rows from bounded heap entries.

    Args:
        heap: Internal bounded heap entries.
    Returns:
        tuple[BacktestStageBScoredVariantV1, ...]: Deterministically ranked rows.
    Assumptions:
        Heap may be partially filled until enough variants are processed.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(sorted((entry[2] for entry in heap), key=_stage_b_rank_key))


def _stage_b_tasks_from_heap(
    *,
    heap: list[
        tuple[
            float,
            tuple[int, ...],
            BacktestStageBScoredVariantV1,
            BacktestStageBTaskV1,
            BacktestVariantScoreDetailsV1 | None,
        ]
    ],
) -> Mapping[str, BacktestStageBTaskV1]:
    """
    Build deterministic `variant_key -> task` mapping from bounded Stage-B heap.

    Args:
        heap: Internal bounded heap entries.
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
    for _, _, row, task, _ in heap:
        if row.variant_key in mapping:
            raise ValueError("duplicate Stage-B variant_key is not allowed")
        mapping[row.variant_key] = task
    return mapping


def _stage_b_details_from_heap(
    *,
    heap: list[
        tuple[
            float,
            tuple[int, ...],
            BacktestStageBScoredVariantV1,
            BacktestStageBTaskV1,
            BacktestVariantScoreDetailsV1 | None,
        ]
    ],
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
        heap: Internal bounded heap entries.
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
    for _, _, row, _, details in heap:
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
