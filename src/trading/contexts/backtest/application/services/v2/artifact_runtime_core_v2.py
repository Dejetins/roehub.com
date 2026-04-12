"""Artifact-backed runtime ranking core for Stage A shortlist outputs and Stage B top-k."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from heapq import heappush, heapreplace
from multiprocessing import get_context
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

from trading.contexts.backtest.application.dto import BacktestRankingConfig, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestStagedVariantMetricScorer,
    BacktestStagedVariantScorerWithDetails,
    RankingMetricsV1,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection

from .artifact_runtime_plan_v2 import (
    STAGE_A_LITERAL_V2,
    STAGE_B_LITERAL_V2,
    BacktestArtifactRuntimePlanV2,
    BacktestRiskVariantV2,
    BacktestStageABaseVariantV2,
)
from .contracts import StageANoRiskMetricsV2
from .execution_profile_v2 import (
    execution_profile_stage_b_process_fallback_threshold_v2,
    execution_profile_uses_process_pool_stage_b_v2,
)
from .trade_compactor_kernel import (
    StageACompactExactPayloadV2,
    no_risk_metrics_to_ranking_payload_v2,
)

StageACheckpointCallbackV2 = Callable[[int, int], None]
StageBCheckpointRowsMaterializerV2 = Callable[
    [],
    tuple["BacktestStageBScoredVariantV2", ...],
]
StageBCheckpointTasksMaterializerV2 = Callable[
    [],
    Mapping[str, "BacktestStageBTaskV2"],
]
StageBCheckpointCallbackV2 = Callable[
    [
        int,
        int,
        StageBCheckpointRowsMaterializerV2,
        StageBCheckpointTasksMaterializerV2,
    ],
    None,
]
CancelCheckerV2 = Callable[[str], None]
MetricScorerV2 = BacktestStagedVariantMetricScorer
ScoreVariantMetricFnV2 = Callable[..., RankingMetricsV1]

_TOTAL_RETURN_METRIC_LITERAL = "Total Return [%]"
_MAX_DRAWDOWN_METRIC_LITERAL = "Max. Drawdown [%]"
_TOTAL_RETURN_METRIC_KEY_LITERAL = "total_return_pct"
_MAX_DRAWDOWN_METRIC_KEY_LITERAL = "max_drawdown_pct"
_RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL = "return_over_max_drawdown"
_PROFIT_FACTOR_METRIC_KEY_LITERAL = "profit_factor"
_SHARPE_TRADES_METRIC_KEY_LITERAL = "sharpe_trades"
_WIN_RATE_PCT_METRIC_KEY_LITERAL = "win_rate_pct"
_DIRECTION_ASC_LITERAL = "ASC"
_DIRECTION_DESC_LITERAL = "DESC"
_STAGE_A_DISABLED_RISK_PARAMS_V2: Mapping[str, BacktestVariantScalar] = MappingProxyType(
    {
        "sl_enabled": False,
        "sl_pct": None,
        "tp_enabled": False,
        "tp_pct": None,
    }
)
_DEFAULT_RANKING_CONFIG_V2 = BacktestRankingConfig()
_METRIC_DIRECTION_BY_LITERAL_V2 = MappingProxyType(
    {
        _TOTAL_RETURN_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _MAX_DRAWDOWN_METRIC_KEY_LITERAL: _DIRECTION_ASC_LITERAL,
        _RETURN_OVER_MAX_DRAWDOWN_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _PROFIT_FACTOR_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _SHARPE_TRADES_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
        _WIN_RATE_PCT_METRIC_KEY_LITERAL: _DIRECTION_DESC_LITERAL,
    }
)
_SCORER_METRIC_KEYS_BY_LITERAL_V2 = MappingProxyType(
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
        _SHARPE_TRADES_METRIC_KEY_LITERAL: (_SHARPE_TRADES_METRIC_KEY_LITERAL,),
        _WIN_RATE_PCT_METRIC_KEY_LITERAL: (_WIN_RATE_PCT_METRIC_KEY_LITERAL,),
    }
)

StageAHeapEntryV2 = tuple[
    float,
    tuple[int, ...],
    "BacktestStageAScoredVariantV2",
]
StageBHeapEntryV2 = tuple[
    float,
    tuple[int, ...],
    "BacktestStageBScoredVariantV2",
    "BacktestStageBTaskV2",
]


@dataclass(frozen=True, slots=True)
class _StageBParallelIndicatorSelectionSnapshotV2:
    """
    Picklable indicator-selection snapshot used by spawned Stage B workers.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """

    indicator_id: str
    inputs: tuple[tuple[str, int | float | str], ...]
    params: tuple[tuple[str, int | float | str], ...]


@dataclass(frozen=True, slots=True)
class _StageBParallelTaskSnapshotV2:
    """
    Picklable Stage B task snapshot for spawned exact-parallel chunk workers.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    indicator_selections: tuple[_StageBParallelIndicatorSelectionSnapshotV2, ...]
    signal_params: tuple[tuple[str, tuple[tuple[str, BacktestVariantScalar], ...]], ...]
    risk_params: tuple[tuple[str, BacktestVariantScalar], ...]
    retained_exact_payload: StageACompactExactPayloadV2 | None = None


@dataclass(frozen=True, slots=True)
class _StageBParallelChunkV2:
    """
    Canonically ordered Stage B chunk submitted to one spawned process worker.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    chunk_index: int
    task_count: int
    tasks: tuple[_StageBParallelTaskSnapshotV2, ...]


@dataclass(frozen=True, slots=True)
class _StageBParallelHeapEntrySnapshotV2:
    """
    Picklable bounded-top-k entry returned by one spawned Stage B worker chunk.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    primary_component: float
    descending_variant_key: tuple[int, ...]
    variant_index: int
    indicator_variant_key: str
    variant_key: str
    total_return_pct: float
    summary_metrics_json: tuple[tuple[str, float], ...]
    best_tp_pct: float | None
    best_sl_pct: float | None
    task: _StageBParallelTaskSnapshotV2


@dataclass(frozen=True, slots=True)
class _StageBParallelChunkResultV2:
    """
    Deterministic per-chunk Stage B result merged by the coordinator in chunk order.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    chunk_index: int
    task_count: int
    entries: tuple[_StageBParallelHeapEntrySnapshotV2, ...]


@dataclass(frozen=True, slots=True)
class _StageBParallelWorkerBootstrapV2:
    """
    Immutable bootstrap payload rehydrating one Stage B scorer per spawned worker process.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    scorer_worker_class: type[Any]
    scorer_worker_snapshot: object
    candles: CandleArrays


@dataclass(frozen=True, slots=True)
class _StageBParallelWorkerStateV2:
    """
    Process-local worker state reused across every Stage B chunk handled by that process.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """

    scorer: MetricScorerV2
    candles: CandleArrays


@dataclass(frozen=True, slots=True)
class ResolvedRankingPlanV2:
    """
    Hot-loop ranking plan with pre-resolved directions and scorer metric aliases.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    primary_metric: str
    primary_direction: str
    primary_scorer_metric_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BacktestStageAScoredVariantV2:
    """
    Deterministic scored Stage A row shared by sync and worker artifact runtime paths.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    base_variant: BacktestStageABaseVariantV2
    total_return_pct: float
    retained_exact_payload: StageACompactExactPayloadV2 | None = None
    no_risk_metrics: StageANoRiskMetricsV2 | None = None


@dataclass(frozen=True, slots=True)
class BacktestStageBTaskV2:
    """
    Deterministic Stage B task payload for artifact-backed top-k scoring.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]]
    risk_params: Mapping[str, BacktestVariantScalar]
    retained_exact_payload: StageACompactExactPayloadV2 | None = None


@dataclass(frozen=True, slots=True)
class BacktestStageBScoredVariantV2:
    """
    Deterministic scored Stage B row for artifact-backed top-k and snapshots.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    variant_index: int
    indicator_variant_key: str
    variant_key: str
    total_return_pct: float
    summary_metrics_json: Mapping[str, float]
    best_tp_pct: float | None
    best_sl_pct: float | None


class BacktestArtifactRuntimeRunnerV2:
    """
    Shared Stage B top-k runner for artifact-backed sync and worker execution paths.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    def __init__(
        self,
        *,
        batch_size_default: int = 256,
        configurable_ranking_enabled: bool = True,
    ) -> None:
        """
        Initialize shared artifact-backed runtime runner.

        Args:
            batch_size_default: Default checkpoint boundary for cancellation/progress hooks.
            configurable_ranking_enabled:
                Feature-flag guard for request/runtime ranking configuration behavior.
        Returns:
            None.
        Assumptions:
            Constructor wires scalar settings only and does not touch runtime artifacts.
        Raises:
            ValueError: If one constructor scalar is invalid.
        Side Effects:
            None.
        """
        if batch_size_default <= 0:
            raise ValueError(
                "BacktestArtifactRuntimeRunnerV2.batch_size_default must be > 0"
            )
        if not isinstance(configurable_ranking_enabled, bool):
            raise ValueError(
                "BacktestArtifactRuntimeRunnerV2.configurable_ranking_enabled must be bool"
            )
        self._batch_size_default = batch_size_default
        self._configurable_ranking_enabled = configurable_ranking_enabled

    def run_stage_b_or_finalize_no_risk(
        self,
        *,
        template: RunBacktestTemplate,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        shortlist: tuple[BacktestStageAScoredVariantV2, ...],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        batch_size: int | None = None,
        cancel_checker: CancelCheckerV2 | None = None,
        on_checkpoint: StageBCheckpointCallbackV2 | None = None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Execute the shared Stage B path or finalize no-risk runs directly from Stage A.

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan: Deterministic artifact-backed runtime plan.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed Stage B scorer contract implementation.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric` only).
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint:
                Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked final rows and task mapping by `variant_key`.
        Assumptions:
            Final deterministic tie-break remains `variant_key ASC`, while no-risk runs terminate
            after Stage A exact scoring and risk-grid runs may still activate process-based
            generic Stage B only when the resolved execution profile explicitly enables it.
        Raises:
            ValueError: If limits/batch-size are invalid or scorer payload is malformed.
        Side Effects:
            May spawn process workers only for deterministic exact-parallel risk-grid Stage B.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
        """
        if top_k_limit <= 0:
            raise ValueError("BacktestArtifactRuntimeRunnerV2 top_k_limit must be > 0")
        ranking_plan = resolve_ranking_plan_v2(
            ranking=effective_ranking_config_v2(
                ranking=ranking,
                configurable_ranking_enabled=self._configurable_ranking_enabled,
            )
        )
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(runtime_plan.stage_b_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_B_LITERAL_V2)
        if _runtime_plan_uses_no_risk_terminal_path_v2(runtime_plan=runtime_plan):
            return self._finalize_no_risk_stage_a_v2(
                template=template,
                runtime_plan=runtime_plan,
                shortlist=shortlist,
                candles=candles,
                scorer=scorer,
                top_k_limit=top_k_limit,
                ranking_plan=ranking_plan,
                total=total,
                cancel_checker=cancel_checker,
                on_checkpoint=on_checkpoint,
            )
        configure_stage_ranking_context_if_supported_v2(
            scorer=scorer,
            stage=STAGE_B_LITERAL_V2,
            ranking_plan=ranking_plan,
        )
        if self._should_run_stage_b_parallel_v2(runtime_plan=runtime_plan):
            return self._run_stage_b_parallel_v2(
                template=template,
                runtime_plan=runtime_plan,
                shortlist=shortlist,
                candles=candles,
                scorer=scorer,
                top_k_limit=top_k_limit,
                ranking_plan=ranking_plan,
                effective_batch=effective_batch,
                total=total,
                cancel_checker=cancel_checker,
                on_checkpoint=on_checkpoint,
            )
        return self._run_stage_b_serial_v2(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
            candles=candles,
            scorer=scorer,
            top_k_limit=top_k_limit,
            ranking_plan=ranking_plan,
            effective_batch=effective_batch,
            total=total,
            cancel_checker=cancel_checker,
            on_checkpoint=on_checkpoint,
        )

    def run_stage_b(
        self,
        *,
        template: RunBacktestTemplate,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        shortlist: tuple[BacktestStageAScoredVariantV2, ...],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        top_k_limit: int,
        ranking: BacktestRankingConfig | None = None,
        batch_size: int | None = None,
        cancel_checker: CancelCheckerV2 | None = None,
        on_checkpoint: StageBCheckpointCallbackV2 | None = None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Preserve the legacy Stage B entrypoint while delegating to the shared no-risk-aware path.

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan: Deterministic artifact-backed runtime plan.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed Stage B scorer contract implementation.
            top_k_limit: Maximum number of final rows retained in memory.
            ranking: Optional ranking config (`primary_metric` only).
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked final rows and task mapping by `variant_key`.
        Assumptions:
            Backward-compatible callers may still use `run_stage_b(...)`, but the shared runtime
            owns the explicit no-risk terminal-path decision.
        Raises:
            ValueError: Propagates from the shared no-risk-aware runtime path.
        Side Effects:
            Delegates to the shared runtime path, which may spawn process workers for risk-grid
            Stage B only.
        """
        return self.run_stage_b_or_finalize_no_risk(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
            candles=candles,
            scorer=scorer,
            top_k_limit=top_k_limit,
            ranking=ranking,
            batch_size=batch_size,
            cancel_checker=cancel_checker,
            on_checkpoint=on_checkpoint,
        )

    def _finalize_no_risk_stage_a_v2(
        self,
        *,
        template: RunBacktestTemplate,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        shortlist: tuple[BacktestStageAScoredVariantV2, ...],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        top_k_limit: int,
        ranking_plan: ResolvedRankingPlanV2,
        total: int,
        cancel_checker: CancelCheckerV2 | None,
        on_checkpoint: StageBCheckpointCallbackV2 | None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Materialize final no-risk rows directly from the Stage A exact shortlist.

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan: Deterministic artifact-backed runtime plan already classified as no-risk.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed scorer used for summary metrics materialization.
            top_k_limit: Maximum number of final rows retained in memory.
            ranking_plan: Pre-resolved ranking plan shared with the generic exact path.
            total: Public Stage B total preserved for progress/checkpoint semantics.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked final rows and task mapping by `variant_key`.
        Assumptions:
            Stage A shortlist order is already the exact final no-risk ranking, so the bypass
            reuses Stage A no-risk metrics when available and only falls back to scorer-based
            materialization for compatibility with older tests or staged fakes.
        Raises:
            ValueError: If the runtime plan drifts from the no-risk contract or scorer metrics are
                malformed.
        Side Effects:
            May populate scorer caches from retained Stage A exact payloads when compatibility
            fallback is needed and emits one lazy checkpoint that reports the full public Stage B
            total as processed.
        """
        if not _runtime_plan_uses_no_risk_terminal_path_v2(runtime_plan=runtime_plan):
            raise ValueError("_finalize_no_risk_stage_a_v2 requires no-risk runtime plan")
        score_variant_metric: ScoreVariantMetricFnV2 | None = None
        ranked_rows: list[BacktestStageBScoredVariantV2] = []
        ranked_tasks: dict[str, BacktestStageBTaskV2] = {}
        for shortlist_index, stage_a_row in enumerate(shortlist[:top_k_limit]):
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            task = _stage_b_task_from_variant_v2(
                base_variant=stage_a_row.base_variant,
                risk_variant=runtime_plan.risk_variants[0],
                shortlist_index=shortlist_index,
                risk_total=1,
                direction_mode=template.direction_mode,
                sizing_mode=template.sizing_mode,
                execution_params=template.execution_params or {},
                retained_exact_payload=stage_a_row.retained_exact_payload,
            )
            if stage_a_row.no_risk_metrics is not None:
                metrics = no_risk_metrics_to_ranking_payload_v2(
                    metrics=stage_a_row.no_risk_metrics
                )
            else:
                prime_retained_exact_payload_if_supported_v2(
                    scorer=scorer,
                    task=task,
                )
                if score_variant_metric is None:
                    configure_stage_ranking_context_if_supported_v2(
                        scorer=scorer,
                        stage=STAGE_A_LITERAL_V2,
                        ranking_plan=ranking_plan,
                    )
                    score_variant_metric = resolve_score_variant_metric_fn_v2(scorer=scorer)
                metrics = score_variant_metric(
                    stage=STAGE_A_LITERAL_V2,
                    candles=candles,
                    indicator_selections=task.indicator_selections,
                    signal_params=task.signal_params,
                    risk_params=_STAGE_A_DISABLED_RISK_PARAMS_V2,
                    indicator_variant_key=task.indicator_variant_key,
                    variant_key=stage_a_row.base_variant.base_variant_key,
                )
            row = _stage_b_row_from_metrics_v2(task=task, metrics=metrics)
            ranked_rows.append(row)
            ranked_tasks[row.variant_key] = task
        rows_result = tuple(ranked_rows)
        if on_checkpoint is not None:
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            tasks_result: Mapping[str, BacktestStageBTaskV2] = dict(ranked_tasks)
            on_checkpoint(
                total,
                total,
                lambda: rows_result,
                lambda: tasks_result,
            )
        return (rows_result, ranked_tasks)

    def _run_stage_b_serial_v2(
        self,
        *,
        template: RunBacktestTemplate,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        shortlist: tuple[BacktestStageAScoredVariantV2, ...],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        top_k_limit: int,
        ranking_plan: ResolvedRankingPlanV2,
        effective_batch: int,
        total: int,
        cancel_checker: CancelCheckerV2 | None,
        on_checkpoint: StageBCheckpointCallbackV2 | None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Execute the canonical in-process Stage B loop with bounded deterministic top-k state.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan: Deterministic artifact-backed runtime plan.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed Stage B scorer contract implementation.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking_plan: Pre-resolved ranking plan.
            effective_batch: Positive checkpoint cadence in processed units.
            total: Total Stage B variants expected from `runtime_plan`.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked Stage B rows and task mapping by `variant_key`.
        Assumptions:
            Final deterministic tie-break for Stage B remains `variant_key ASC`, and `RG-TTR`
            breadth scoring may still replace retained finalist rows with exact final authority
            after the cheap heap-selection pass completes.
        Raises:
            ValueError: If scorer payload is malformed.
        Side Effects:
            May prime scorer-local retained exact payload cache and exact-replay finalists only.
        """
        score_variant_metric = resolve_score_variant_metric_fn_v2(scorer=scorer)
        top_heap: list[StageBHeapEntryV2] = []
        processed = 0
        final_checkpoint_pending = False
        for task in _iter_stage_b_tasks_stream_v2(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
        ):
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            prime_retained_exact_payload_if_supported_v2(
                scorer=scorer,
                task=task,
            )
            row, metrics = score_stage_b_task_with_metrics_v2(
                task=task,
                candles=candles,
                score_variant_metric=score_variant_metric,
            )
            self._merge_stage_b_heap_entry_v2(
                heap=top_heap,
                entry=stage_b_heap_entry_v2(
                    row=row,
                    task=task,
                    metrics=metrics,
                    ranking_plan=ranking_plan,
                ),
                top_k_limit=top_k_limit,
            )

            processed += 1
            if processed % effective_batch != 0 and processed != total:
                continue
            if processed == total:
                final_checkpoint_pending = True
                continue
            self._emit_stage_b_checkpoint_v2(
                top_heap=top_heap,
                processed=processed,
                total=total,
                cancel_checker=cancel_checker,
                on_checkpoint=on_checkpoint,
            )

        rows_result, tasks_result = self._materialize_stage_b_result_v2(
            top_heap=top_heap,
            candles=candles,
            scorer=scorer,
            ranking_plan=ranking_plan,
            cancel_checker=cancel_checker,
        )
        if final_checkpoint_pending and on_checkpoint is not None:
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            on_checkpoint(
                total,
                total,
                lambda: rows_result,
                lambda: tasks_result,
            )
        return (rows_result, tasks_result)

    def _run_stage_b_parallel_v2(
        self,
        *,
        template: RunBacktestTemplate,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        shortlist: tuple[BacktestStageAScoredVariantV2, ...],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        top_k_limit: int,
        ranking_plan: ResolvedRankingPlanV2,
        effective_batch: int,
        total: int,
        cancel_checker: CancelCheckerV2 | None,
        on_checkpoint: StageBCheckpointCallbackV2 | None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Execute exact-parallel Stage B through spawned readonly workers and ordered chunk merge.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan:
                Deterministic artifact-backed runtime plan with resolved execution profile.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed Stage B scorer contract implementation.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking_plan: Pre-resolved ranking plan.
            effective_batch: Positive checkpoint cadence in processed units.
            total: Total Stage B variants expected from `runtime_plan`.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked Stage B rows and task mapping by `variant_key`.
        Assumptions:
            Worker completion order must not influence merged results or checkpoint frontiers,
            and finalist-only exact replay must happen after ordered chunk merge only.
        Raises:
            ValueError: If scorer does not expose spawned-worker snapshot support.
        Side Effects:
            Spawns process workers via the `spawn` multiprocessing context and may exact-replay
            finalists only after breadth merge finishes.
        """
        worker_factory = resolve_parallel_stage_b_worker_factory_v2(scorer=scorer)
        if worker_factory is None:
            raise ValueError(
                "resolved parallel Stage B profile requires scorer worker snapshot support"
            )
        risk_total = len(runtime_plan.risk_variants)
        if risk_total == 0 or len(shortlist) == 0:
            return self._materialize_stage_b_heap_v2(top_heap=[])
        base_variants_per_chunk = _parallel_stage_b_base_variants_per_chunk_v2(
            risk_total=risk_total,
            effective_batch=effective_batch,
        )
        chunk_count = (len(shortlist) + base_variants_per_chunk - 1) // base_variants_per_chunk
        profile = getattr(runtime_plan, "execution_profile", None)
        requested_workers = getattr(getattr(profile, "parallelism", None), "stage_b_workers", 1)
        worker_count = max(1, min(chunk_count, int(requested_workers)))
        if worker_count <= 1:
            return self._run_stage_b_serial_v2(
                template=template,
                runtime_plan=runtime_plan,
                shortlist=shortlist,
                candles=candles,
                scorer=scorer,
                top_k_limit=top_k_limit,
                ranking_plan=ranking_plan,
                effective_batch=effective_batch,
                total=total,
                cancel_checker=cancel_checker,
                on_checkpoint=on_checkpoint,
            )

        scorer_class, scorer_snapshot = worker_factory
        top_heap: list[StageBHeapEntryV2] = []
        processed = 0
        final_checkpoint_pending = False
        next_merge_chunk_index = 0
        pending_futures: dict[Future[_StageBParallelChunkResultV2], int] = {}
        ready_results: dict[int, _StageBParallelChunkResultV2] = {}
        max_inflight = max(worker_count, worker_count * 2)
        chunk_iterator = iter(
            _iter_stage_b_parallel_chunks_v2(
                template=template,
                runtime_plan=runtime_plan,
                shortlist=shortlist,
                effective_batch=effective_batch,
            )
        )
        chunks_exhausted = False

        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_initialize_stage_b_parallel_worker_v2,
            initargs=(
                _StageBParallelWorkerBootstrapV2(
                    scorer_worker_class=scorer_class,
                    scorer_worker_snapshot=scorer_snapshot,
                    candles=candles,
                ),
            ),
        ) as executor:
            while not chunks_exhausted or pending_futures:
                while not chunks_exhausted and len(pending_futures) < max_inflight:
                    if cancel_checker is not None:
                        cancel_checker(STAGE_B_LITERAL_V2)
                    try:
                        chunk = next(chunk_iterator)
                    except StopIteration:
                        chunks_exhausted = True
                        break
                    future = executor.submit(
                        _score_stage_b_parallel_chunk_v2,
                        chunk=chunk,
                        ranking_plan=ranking_plan,
                        top_k_limit=top_k_limit,
                    )
                    pending_futures[future] = chunk.chunk_index
                if not pending_futures:
                    continue
                completed, _ = wait(
                    tuple(pending_futures.keys()),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    chunk_index = pending_futures.pop(future)
                    ready_results[chunk_index] = future.result()
                while next_merge_chunk_index in ready_results:
                    if cancel_checker is not None:
                        cancel_checker(STAGE_B_LITERAL_V2)
                    chunk_result = ready_results.pop(next_merge_chunk_index)
                    for entry_snapshot in chunk_result.entries:
                        self._merge_stage_b_heap_entry_v2(
                            heap=top_heap,
                            entry=_stage_b_heap_entry_from_parallel_snapshot_v2(
                                snapshot=entry_snapshot
                            ),
                            top_k_limit=top_k_limit,
                        )
                    processed += chunk_result.task_count
                    if processed == total:
                        final_checkpoint_pending = True
                    elif processed % effective_batch == 0:
                        self._emit_stage_b_checkpoint_v2(
                            top_heap=top_heap,
                            processed=processed,
                            total=total,
                            cancel_checker=cancel_checker,
                            on_checkpoint=on_checkpoint,
                        )
                    next_merge_chunk_index += 1

        rows_result, tasks_result = self._materialize_stage_b_result_v2(
            top_heap=top_heap,
            candles=candles,
            scorer=scorer,
            ranking_plan=ranking_plan,
            cancel_checker=cancel_checker,
        )
        if final_checkpoint_pending and on_checkpoint is not None:
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            on_checkpoint(
                total,
                total,
                lambda: rows_result,
                lambda: tasks_result,
            )
        return (rows_result, tasks_result)

    def _should_run_stage_b_parallel_v2(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
    ) -> bool:
        """
        Decide whether the resolved runtime plan enables process-based exact Stage B execution.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

        Args:
            runtime_plan: Deterministic artifact-backed runtime plan.
        Returns:
            bool: `True` when Stage B should use spawned workers, otherwise `False`.
        Assumptions:
            Request classification stays unchanged; only already-resolved non-default profiles may
            activate this path by opting into process-pool Stage B explicitly.
        Raises:
            None.
        Side Effects:
            None.
        """
        return _runtime_plan_stage_b_execution_mode_v2(runtime_plan=runtime_plan) == "process_pool"

    def _merge_stage_b_heap_entry_v2(
        self,
        *,
        heap: list[StageBHeapEntryV2],
        entry: StageBHeapEntryV2,
        top_k_limit: int,
    ) -> None:
        """
        Merge one Stage B heap entry into the bounded retained frontier.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

        Args:
            heap: Mutable retained Stage B heap.
            entry: Candidate Stage B heap entry.
            top_k_limit: Maximum retained heap size.
        Returns:
            None.
        Assumptions:
            Heap root stores the current worst retained row under the deterministic comparator.
        Raises:
            None.
        Side Effects:
            Mutates the supplied heap in place.
        """
        if len(heap) < top_k_limit:
            heappush(heap, entry)
            return
        if heap_entry_outranks_v2(candidate=entry, baseline=heap[0]):
            heapreplace(heap, entry)

    def _emit_stage_b_checkpoint_v2(
        self,
        *,
        top_heap: list[StageBHeapEntryV2],
        processed: int,
        total: int,
        cancel_checker: CancelCheckerV2 | None,
        on_checkpoint: StageBCheckpointCallbackV2 | None,
    ) -> None:
        """
        Emit one lazy Stage B checkpoint from the current bounded retained frontier.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py

        Args:
            top_heap: Mutable retained Stage B heap.
            processed: Processed Stage B units so far.
            total: Total Stage B units expected.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint: Optional checkpoint callback with lazy frontier materializers.
        Returns:
            None.
        Assumptions:
            Checkpoint materialization must preserve the same deterministic row/task ordering as
            final Stage B output.
        Raises:
            None.
        Side Effects:
            May invoke cancellation and checkpoint callbacks.
        """
        if cancel_checker is not None:
            cancel_checker(STAGE_B_LITERAL_V2)
        if on_checkpoint is None:
            return

        sorted_entries_cache: tuple[StageBHeapEntryV2, ...] | None = None
        rows_cache: tuple[BacktestStageBScoredVariantV2, ...] | None = None
        tasks_cache: Mapping[str, BacktestStageBTaskV2] | None = None

        def _sorted_entries() -> tuple[StageBHeapEntryV2, ...]:
            nonlocal sorted_entries_cache
            if sorted_entries_cache is None:
                sorted_entries_cache = sorted_stage_b_heap_entries_v2(heap=top_heap)
            return sorted_entries_cache

        def _materialize_ranked_rows() -> tuple[BacktestStageBScoredVariantV2, ...]:
            nonlocal rows_cache
            if rows_cache is None:
                rows_cache = tuple(entry[2] for entry in _sorted_entries())
            return rows_cache

        def _materialize_tasks() -> Mapping[str, BacktestStageBTaskV2]:
            nonlocal tasks_cache
            if tasks_cache is None:
                tasks_cache = stage_b_tasks_from_sorted_entries_v2(entries=_sorted_entries())
            return tasks_cache

        on_checkpoint(
            processed,
            total,
            _materialize_ranked_rows,
            _materialize_tasks,
        )

    def _materialize_stage_b_result_v2(
        self,
        *,
        top_heap: list[StageBHeapEntryV2],
        candles: CandleArrays,
        scorer: MetricScorerV2,
        ranking_plan: ResolvedRankingPlanV2,
        cancel_checker: CancelCheckerV2 | None,
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Materialize the final Stage B result, exact-replaying finalists only when required.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            top_heap: Mutable retained Stage B heap built from breadth scoring.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Stage scorer implementation used by the current loop.
            ranking_plan: Pre-resolved ranking plan.
            cancel_checker: Optional cooperative cancellation callback by stage.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked final rows and task mapping by `variant_key`.
        Assumptions:
            `RG-TTR` breadth remains cheap and in-process, while final authority becomes exact only
            for the retained finalist slice after heap selection is complete.
        Raises:
            ValueError: Propagated from the details scorer when finalist exact replay drifts.
        Side Effects:
            May invoke the exact details scorer for finalists only.
        """
        sorted_entries = sorted_stage_b_heap_entries_v2(heap=top_heap)
        if not self._should_finalize_stage_b_finalists_exact_v2(
            scorer=scorer,
            ranking_plan=ranking_plan,
            entries=sorted_entries,
        ):
            return self._materialize_stage_b_heap_v2(top_heap=top_heap)

        details_scorer = resolve_stage_b_details_scorer_if_supported_v2(scorer=scorer)
        if details_scorer is None:
            return self._materialize_stage_b_heap_v2(top_heap=top_heap)

        exact_entries: list[StageBHeapEntryV2] = []
        for entry in sorted_entries:
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            task = entry[3]
            details = details_scorer.score_variant_with_details(
                stage=STAGE_B_LITERAL_V2,
                candles=candles,
                indicator_selections=task.indicator_selections,
                signal_params=task.signal_params,
                risk_params=task.risk_params,
                indicator_variant_key=task.indicator_variant_key,
                variant_key=task.variant_key,
            )
            exact_metrics = details.metrics
            exact_row = _stage_b_row_from_metrics_v2(task=task, metrics=exact_metrics)
            exact_entries.append(
                stage_b_heap_entry_v2(
                    row=exact_row,
                    task=task,
                    metrics=exact_metrics,
                    ranking_plan=ranking_plan,
                )
            )
        exact_sorted_entries = tuple(
            sorted(exact_entries, key=lambda item: item[:2], reverse=True)
        )
        return (
            tuple(entry[2] for entry in exact_sorted_entries),
            stage_b_tasks_from_sorted_entries_v2(entries=exact_sorted_entries),
        )

    def _should_finalize_stage_b_finalists_exact_v2(
        self,
        *,
        scorer: MetricScorerV2,
        ranking_plan: ResolvedRankingPlanV2,
        entries: tuple[StageBHeapEntryV2, ...],
    ) -> bool:
        """
        Decide whether Stage B finalists should be replayed exactly before final output.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py

        Args:
            scorer: Stage scorer implementation used by the current loop.
            ranking_plan: Pre-resolved ranking plan.
            entries: Sorted retained frontier entries from Stage B breadth scoring.
        Returns:
            bool: `True` when Stage B should exact-replay finalists only before returning rows.
        Assumptions:
            The finalist-only exact replay cutover applies to `total_return_pct` breadth runs such
            as `RG-TTR`; alternative-primary runs like `RG-ALT` keep their existing path.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (
            len(entries) > 0
            and ranking_plan.primary_metric == _TOTAL_RETURN_METRIC_KEY_LITERAL
            and resolve_stage_b_details_scorer_if_supported_v2(scorer=scorer) is not None
        )

    def _materialize_stage_b_heap_v2(
        self,
        *,
        top_heap: list[StageBHeapEntryV2],
    ) -> tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
        """
        Materialize final deterministic Stage B rows and task mapping from the retained heap.

        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

        Args:
            top_heap: Mutable retained Stage B heap.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked Stage B rows and task mapping by `variant_key`.
        Assumptions:
            Final row/task projections reuse the same comparator as checkpoint materialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (
            stage_b_rows_from_heap_v2(heap=top_heap),
            stage_b_tasks_from_heap_v2(heap=top_heap),
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
            ValueError: If the override value is non-positive.
        Side Effects:
            None.
        """
        if batch_size is None:
            return self._batch_size_default
        if batch_size <= 0:
            raise ValueError("stage batch_size must be > 0")
        return batch_size


_STAGE_B_PARALLEL_WORKER_STATE_V2: _StageBParallelWorkerStateV2 | None = None


def resolve_parallel_stage_b_worker_factory_v2(
    *,
    scorer: MetricScorerV2,
) -> tuple[type[Any], object] | None:
    """
    Resolve optional spawned-worker scorer rehydration hooks for exact-parallel Stage B.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        scorer: Scorer implementation selected for Stage B ranking.
    Returns:
        tuple[type[Any], object] | None: Scorer class plus picklable snapshot, or `None` when
            the scorer does not advertise spawned-worker support.
    Assumptions:
        Optional worker hooks are additive and discovered by method presence.
    Raises:
        ValueError: If the scorer exposes only one side of the snapshot contract.
    Side Effects:
        None.
    """
    snapshot_method = getattr(scorer, "to_parallel_stage_b_worker_snapshot_v2", None)
    factory_method = getattr(
        scorer.__class__,
        "from_parallel_stage_b_worker_snapshot_v2",
        None,
    )
    if snapshot_method is None and factory_method is None:
        return None
    if snapshot_method is None or factory_method is None:
        raise ValueError(
            "parallel Stage B scorer must expose both "
            "to_parallel_stage_b_worker_snapshot_v2 and "
            "from_parallel_stage_b_worker_snapshot_v2"
        )
    return scorer.__class__, snapshot_method()


def _initialize_stage_b_parallel_worker_v2(
    worker_bootstrap: _StageBParallelWorkerBootstrapV2,
) -> None:
    """
    Bootstrap one spawned Stage B worker with process-local scorer and immutable candles.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        worker_bootstrap: Immutable scorer snapshot and candle payload for one process.
    Returns:
        None.
    Assumptions:
        macOS `spawn` requires explicit per-process scorer reconstruction instead of sharing the
        parent object graph.
    Raises:
        ValueError: If scorer rehydration hooks are missing in the worker process.
    Side Effects:
        Stores worker-local scorer state in a private module global.
    """
    global _STAGE_B_PARALLEL_WORKER_STATE_V2
    factory_method = getattr(
        worker_bootstrap.scorer_worker_class,
        "from_parallel_stage_b_worker_snapshot_v2",
        None,
    )
    if factory_method is None:
        raise ValueError(
            "parallel Stage B worker bootstrap requires "
            "from_parallel_stage_b_worker_snapshot_v2"
        )
    _STAGE_B_PARALLEL_WORKER_STATE_V2 = _StageBParallelWorkerStateV2(
        scorer=cast(
            MetricScorerV2,
            factory_method(snapshot=worker_bootstrap.scorer_worker_snapshot),
        ),
        candles=worker_bootstrap.candles,
    )


def _require_stage_b_parallel_worker_state_v2() -> _StageBParallelWorkerStateV2:
    """
    Return the initialized worker-local state for one exact-parallel Stage B chunk.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        None.
    Returns:
        _StageBParallelWorkerStateV2: Process-local scorer and candle payload.
    Assumptions:
        Spawned workers must run `_initialize_stage_b_parallel_worker_v2(...)` before jobs.
    Raises:
        ValueError: If the worker was used before bootstrap completed.
    Side Effects:
        None.
    """
    if _STAGE_B_PARALLEL_WORKER_STATE_V2 is None:
        raise ValueError("parallel Stage B worker state is not initialized")
    return _STAGE_B_PARALLEL_WORKER_STATE_V2


def _parallel_stage_b_base_variants_per_chunk_v2(
    *,
    risk_total: int,
    effective_batch: int,
) -> int:
    """
    Resolve how many shortlisted base variants belong to one spawned Stage B chunk.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        risk_total: Total number of Stage B risk variants.
        effective_batch: Positive checkpoint cadence in processed units.
    Returns:
        int: Positive number of shortlisted base variants per chunk.
    Assumptions:
        Chunks stay aligned to whole base variants so worker-local caches can reuse Stage A
        payloads across all risk variants of a base variant.
    Raises:
        ValueError: If `risk_total` is non-positive.
    Side Effects:
        None.
    """
    if risk_total <= 0:
        raise ValueError("parallel Stage B requires at least one risk variant")
    return max(1, effective_batch // risk_total)


def _iter_stage_b_parallel_chunks_v2(
    *,
    template: RunBacktestTemplate,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    shortlist: tuple[BacktestStageAScoredVariantV2, ...],
    effective_batch: int,
):
    """
    Build deterministic Stage B chunks grouped by contiguous shortlisted base variants.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        template: Effective run template.
        runtime_plan: Deterministic artifact-backed runtime plan.
        shortlist: Deterministically ranked Stage A shortlist rows.
        effective_batch: Positive checkpoint cadence in processed units.
    Returns:
        Iterator[_StageBParallelChunkV2]: Canonically ordered Stage B chunks.
    Assumptions:
        Chunk boundaries stay aligned to whole shortlisted base variants so worker-local caches
        can reuse one Stage A base payload across all risk variants of that base variant.
    Raises:
        ValueError: If Stage B risk catalog is unexpectedly empty.
    Side Effects:
        None.
    """
    risk_total = len(runtime_plan.risk_variants)
    base_variants_per_chunk = _parallel_stage_b_base_variants_per_chunk_v2(
        risk_total=risk_total,
        effective_batch=effective_batch,
    )
    chunk_index = 0
    direction_mode = template.direction_mode
    sizing_mode = template.sizing_mode
    execution_params = template.execution_params or {}
    for chunk_start in range(0, len(shortlist), base_variants_per_chunk):
        chunk_shortlist = shortlist[chunk_start : chunk_start + base_variants_per_chunk]
        tasks: list[_StageBParallelTaskSnapshotV2] = []
        for local_shortlist_index, stage_a_row in enumerate(chunk_shortlist):
            base_variant = stage_a_row.base_variant
            shortlist_index = chunk_start + local_shortlist_index
            for risk_variant in runtime_plan.risk_variants:
                tasks.append(
                    _stage_b_parallel_task_snapshot_from_task_v2(
                        task=_stage_b_task_from_variant_v2(
                            base_variant=base_variant,
                            risk_variant=risk_variant,
                            shortlist_index=shortlist_index,
                            risk_total=risk_total,
                            direction_mode=direction_mode,
                            sizing_mode=sizing_mode,
                            execution_params=execution_params,
                            retained_exact_payload=stage_a_row.retained_exact_payload,
                        )
                    )
                )
        yield _StageBParallelChunkV2(
            chunk_index=chunk_index,
            task_count=len(tasks),
            tasks=tuple(tasks),
        )
        chunk_index += 1


def _score_stage_b_parallel_chunk_v2(
    *,
    chunk: _StageBParallelChunkV2,
    ranking_plan: ResolvedRankingPlanV2,
    top_k_limit: int,
) -> _StageBParallelChunkResultV2:
    """
    Score one deterministic Stage B chunk inside a spawned worker and keep local top-k only.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py

    Args:
        chunk: Canonically ordered Stage B chunk snapshot.
        ranking_plan: Pre-resolved ranking plan.
        top_k_limit: Maximum retained rows for the local chunk frontier.
    Returns:
        _StageBParallelChunkResultV2: Chunk-local bounded frontier ready for coordinator merge.
    Assumptions:
        A global top-k candidate must belong to the local top-k of its originating chunk.
    Raises:
        ValueError: If the worker scorer is unavailable or returns malformed metrics.
    Side Effects:
        Reuses worker-local scorer caches while processing this chunk and may prime retained exact
        payload cache entries ahead of exact Stage B scoring.
    """
    worker_state = _require_stage_b_parallel_worker_state_v2()
    configure_stage_ranking_context_if_supported_v2(
        scorer=worker_state.scorer,
        stage=STAGE_B_LITERAL_V2,
        ranking_plan=ranking_plan,
    )
    score_variant_metric = resolve_score_variant_metric_fn_v2(scorer=worker_state.scorer)
    local_heap: list[StageBHeapEntryV2] = []
    for task_snapshot in chunk.tasks:
        task = _stage_b_task_from_parallel_snapshot_v2(snapshot=task_snapshot)
        prime_retained_exact_payload_if_supported_v2(
            scorer=worker_state.scorer,
            task=task,
        )
        row, metrics = score_stage_b_task_with_metrics_v2(
            task=task,
            candles=worker_state.candles,
            score_variant_metric=score_variant_metric,
        )
        heap_entry = stage_b_heap_entry_v2(
            row=row,
            task=task,
            metrics=metrics,
            ranking_plan=ranking_plan,
        )
        if len(local_heap) < top_k_limit:
            heappush(local_heap, heap_entry)
        elif heap_entry_outranks_v2(candidate=heap_entry, baseline=local_heap[0]):
            heapreplace(local_heap, heap_entry)
    return _StageBParallelChunkResultV2(
        chunk_index=chunk.chunk_index,
        task_count=chunk.task_count,
        entries=tuple(
            _stage_b_parallel_heap_entry_snapshot_v2(entry=entry)
            for entry in sorted_stage_b_heap_entries_v2(heap=local_heap)
        ),
    )


def _stage_b_parallel_task_snapshot_from_task_v2(
    *,
    task: BacktestStageBTaskV2,
) -> _StageBParallelTaskSnapshotV2:
    """
    Convert one Stage B task into a fully picklable spawned-worker snapshot.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        task: Canonical in-process Stage B task payload.
    Returns:
        _StageBParallelTaskSnapshotV2: Picklable spawned-worker snapshot.
    Assumptions:
        Snapshot ordering is stable because nested mappings are serialized by sorted keys, while
        the retained exact payload stays internal and additive when available.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _StageBParallelTaskSnapshotV2(
        variant_index=task.variant_index,
        indicator_variant_key=task.indicator_variant_key,
        variant_key=task.variant_key,
        indicator_selections=tuple(
            _StageBParallelIndicatorSelectionSnapshotV2(
                indicator_id=selection.indicator_id,
                inputs=tuple(
                    sorted((str(key), value) for key, value in selection.inputs.items())
                ),
                params=tuple(
                    sorted((str(key), value) for key, value in selection.params.items())
                ),
            )
            for selection in task.indicator_selections
        ),
        signal_params=tuple(
            (
                str(indicator_id),
                tuple(sorted((str(name), value) for name, value in params.items())),
            )
            for indicator_id, params in sorted(task.signal_params.items())
        ),
        risk_params=tuple(sorted((str(name), value) for name, value in task.risk_params.items())),
        retained_exact_payload=task.retained_exact_payload,
    )


def _stage_b_task_from_parallel_snapshot_v2(
    *,
    snapshot: _StageBParallelTaskSnapshotV2,
) -> BacktestStageBTaskV2:
    """
    Rebuild one canonical Stage B task from a picklable spawned-worker snapshot.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/indicators/application/dto/variant_key.py

    Args:
        snapshot: Picklable spawned-worker task snapshot.
    Returns:
        BacktestStageBTaskV2: Canonical in-process Stage B task payload.
    Assumptions:
        Snapshot values already preserve deterministic ordering and scalar normalization, while
        the retained exact payload remains optional for persisted-compatibility fallbacks.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestStageBTaskV2(
        variant_index=snapshot.variant_index,
        indicator_variant_key=snapshot.indicator_variant_key,
        variant_key=snapshot.variant_key,
        indicator_selections=tuple(
            IndicatorVariantSelection(
                indicator_id=selection.indicator_id,
                inputs=dict(selection.inputs),
                params=dict(selection.params),
            )
            for selection in snapshot.indicator_selections
        ),
        signal_params={
            indicator_id: dict(params)
            for indicator_id, params in snapshot.signal_params
        },
        risk_params=dict(snapshot.risk_params),
        retained_exact_payload=_validated_retained_exact_payload_v2(
            retained_exact_payload=snapshot.retained_exact_payload
        ),
    )


def _validated_retained_exact_payload_v2(
    *,
    retained_exact_payload: StageACompactExactPayloadV2 | None,
) -> StageACompactExactPayloadV2 | None:
    """
    Validate the compact-trade-only retained exact payload contract accepted by the risk path.

    Args:
        retained_exact_payload: Optional internal retained payload forwarded from Stage A.
    Returns:
        StageACompactExactPayloadV2 | None: The same payload when it matches the compact-only
            contract, otherwise `None` when no retained payload exists.
    Assumptions:
        Stage B must accept only compact-trade-array retained payloads and must not silently carry
        legacy full signal-row baggage through the risk path.
    Raises:
        ValueError: If the retained payload exposes a non-compact memory-shape marker.
    Side Effects:
        None.
    """
    if retained_exact_payload is None:
        return None
    if retained_exact_payload.memory_shape_bucket != "compact_trade_arrays":
        raise ValueError(
            "retained_exact_payload must stay in the compact_trade_arrays memory shape"
        )
    return retained_exact_payload


def _stage_b_parallel_heap_entry_snapshot_v2(
    *,
    entry: StageBHeapEntryV2,
) -> _StageBParallelHeapEntrySnapshotV2:
    """
    Convert one retained Stage B heap entry into a picklable worker-result snapshot.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        entry: Canonical retained Stage B heap entry.
    Returns:
        _StageBParallelHeapEntrySnapshotV2: Picklable worker-result snapshot.
    Assumptions:
        Summary metrics are serialized by sorted keys to preserve deterministic payload order.
    Raises:
        None.
    Side Effects:
        None.
    """
    primary_component, descending_variant_key, row, task = entry
    return _StageBParallelHeapEntrySnapshotV2(
        primary_component=primary_component,
        descending_variant_key=descending_variant_key,
        variant_index=row.variant_index,
        indicator_variant_key=row.indicator_variant_key,
        variant_key=row.variant_key,
        total_return_pct=row.total_return_pct,
        summary_metrics_json=tuple(
            sorted((str(key), float(value)) for key, value in row.summary_metrics_json.items())
        ),
        best_tp_pct=row.best_tp_pct,
        best_sl_pct=row.best_sl_pct,
        task=_stage_b_parallel_task_snapshot_from_task_v2(task=task),
    )


def _stage_b_heap_entry_from_parallel_snapshot_v2(
    *,
    snapshot: _StageBParallelHeapEntrySnapshotV2,
) -> StageBHeapEntryV2:
    """
    Rebuild one canonical Stage B heap entry from a worker-result snapshot.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

    Args:
        snapshot: Picklable worker-result snapshot.
    Returns:
        StageBHeapEntryV2: Canonical retained heap entry for coordinator merge.
    Assumptions:
        Ranking components were already computed under the same deterministic ranking plan inside
        the worker process.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        snapshot.primary_component,
        snapshot.descending_variant_key,
        BacktestStageBScoredVariantV2(
            variant_index=snapshot.variant_index,
            indicator_variant_key=snapshot.indicator_variant_key,
            variant_key=snapshot.variant_key,
            total_return_pct=snapshot.total_return_pct,
            summary_metrics_json=dict(snapshot.summary_metrics_json),
            best_tp_pct=snapshot.best_tp_pct,
            best_sl_pct=snapshot.best_sl_pct,
        ),
        _stage_b_task_from_parallel_snapshot_v2(snapshot=snapshot.task),
    )


def resolve_score_variant_metric_fn_v2(*, scorer: MetricScorerV2) -> ScoreVariantMetricFnV2:
    """
    Resolve metric-only scorer function with fail-fast contract checks.

    Args:
        scorer: Scorer implementation used by Stage B ranking loop.
    Returns:
        ScoreVariantMetricFnV2: Callable producing ranking metrics payload.
    Assumptions:
        Artifact-backed production paths require metric-only scoring support.
    Raises:
        ValueError: If the scorer does not expose `score_variant_metric(...)`.
    Side Effects:
        None.
    """
    score_variant_metric = getattr(scorer, "score_variant_metric", None)
    if score_variant_metric is None:
        raise ValueError(
            "artifact-backed runtime scorer must expose score_variant_metric(...)"
        )
    return cast(ScoreVariantMetricFnV2, score_variant_metric)


def resolve_stage_b_details_scorer_if_supported_v2(
    *,
    scorer: MetricScorerV2,
) -> BacktestStagedVariantScorerWithDetails | None:
    """
    Resolve the optional details scorer used for finalist-only Stage B exact replay.

    Args:
        scorer: Scorer implementation selected for Stage B ranking.
    Returns:
        BacktestStagedVariantScorerWithDetails | None:
            Details scorer when available, otherwise `None`.
    Assumptions:
        Finalist-only exact replay is additive and must not force a stronger scorer contract on
        metric-only test doubles or legacy implementations.
    Raises:
        None.
    Side Effects:
        None.
    """
    if getattr(scorer, "score_variant_with_details", None) is None:
        return None
    return cast(BacktestStagedVariantScorerWithDetails, scorer)


def configure_stage_ranking_context_if_supported_v2(
    *,
    scorer: MetricScorerV2,
    stage: str,
    ranking_plan: ResolvedRankingPlanV2,
) -> None:
    """
    Forward the active single-metric ranking literal to scorers with ranking-context support.

    Args:
        scorer: Stage scorer implementation used by the current loop.
        stage: Stage literal (`stage_a` or `stage_b`).
        ranking_plan: Resolved ranking plan for the current loop.
    Returns:
        None.
    Assumptions:
        Scorers may ignore this hook, while Stage B fast-path scorers can use it safely.
    Raises:
        None.
    Side Effects:
        May update scorer-local in-memory ranking hints for the current run.
    """
    configure_method = getattr(scorer, "configure_stage_ranking_context", None)
    if configure_method is None:
        return
    configure_method(
        stage=stage,
        primary_metric=ranking_plan.primary_metric,
    )


def prime_retained_exact_payload_if_supported_v2(
    *,
    scorer: MetricScorerV2,
    task: BacktestStageBTaskV2,
) -> None:
    """
    Forward one retained exact-candidate payload to scorers that can reuse it for Stage B.

    Args:
        scorer: Stage scorer implementation used by the current loop.
        task: Stage B task that may carry an internal retained exact payload.
    Returns:
        None.
    Assumptions:
        The retained payload is internal-only, additive, and safe to ignore when the scorer does
        not expose a compatible priming hook or the task originated from persisted snapshots.
    Raises:
        None.
    Side Effects:
        May seed scorer-local Stage B compact-trade caches for exact replay and fast search.
    """
    if task.retained_exact_payload is None:
        return
    prime_method = getattr(scorer, "prime_retained_exact_payload", None)
    if prime_method is None:
        return
    prime_method(
        indicator_variant_key=task.indicator_variant_key,
        signal_params=task.signal_params,
        retained_exact_payload=task.retained_exact_payload,
    )


def effective_ranking_config_v2(
    *,
    ranking: BacktestRankingConfig | None,
    configurable_ranking_enabled: bool,
) -> BacktestRankingConfig:
    """
    Resolve effective ranking config under explicit feature-flag guard policy.

    Args:
        ranking: Optional request/runtime ranking config.
        configurable_ranking_enabled: Feature-flag literal controlling ranking behavior.
    Returns:
        BacktestRankingConfig: Effective ranking config used by runtime loops.
    Assumptions:
        Legacy deterministic behavior remains `total_return_pct DESC` with deterministic
        tie-breaks by canonical variant keys.
    Raises:
        ValueError: If resulting ranking config cannot be normalized.
    Side Effects:
        None.
    """
    if not configurable_ranking_enabled or ranking is None:
        return _DEFAULT_RANKING_CONFIG_V2
    return ranking


def resolve_ranking_plan_v2(*, ranking: BacktestRankingConfig) -> ResolvedRankingPlanV2:
    """
    Resolve ranking literals into a hot-loop plan with directions and aliases.

    Args:
        ranking: Effective ranking config.
    Returns:
        ResolvedRankingPlanV2: Ranking plan with pre-validated metric aliases.
    Assumptions:
        Ranking literals were normalized by DTO/config contracts before this step.
    Raises:
        ValueError: If the primary ranking metric is unsupported.
    Side Effects:
        None.
    """
    primary_metric = ranking.primary_metric
    primary_direction = _METRIC_DIRECTION_BY_LITERAL_V2.get(primary_metric)
    primary_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V2.get(primary_metric)
    if primary_direction is None or primary_keys is None:
        raise ValueError(f"unsupported primary ranking metric: {primary_metric!r}")

    return ResolvedRankingPlanV2(
        primary_metric=primary_metric,
        primary_direction=primary_direction,
        primary_scorer_metric_keys=primary_keys,
    )


def score_stage_b_task_with_metrics_v2(
    *,
    task: BacktestStageBTaskV2,
    candles: CandleArrays,
    score_variant_metric: ScoreVariantMetricFnV2,
) -> tuple[BacktestStageBScoredVariantV2, RankingMetricsV1]:
    """
    Score one Stage B task and return both deterministic row and raw metrics payload.

    Args:
        task: Stage B task payload.
        candles: Warmup-inclusive request-timeframe candles.
        score_variant_metric: Metric-only scorer callable.
    Returns:
        tuple[BacktestStageBScoredVariantV2, RankingMetricsV1]:
            Deterministic ranked row and raw scorer metrics.
    Assumptions:
        Stage B scoring uses risk-enabled payload from the task.
    Raises:
        ValueError: If scorer payload lacks required ranking metrics.
    Side Effects:
        None.
    """
    metrics = score_variant_metric(
        stage=STAGE_B_LITERAL_V2,
        candles=candles,
        indicator_selections=task.indicator_selections,
        signal_params=task.signal_params,
        risk_params=task.risk_params,
        indicator_variant_key=task.indicator_variant_key,
        variant_key=task.variant_key,
    )
    return (_stage_b_row_from_metrics_v2(task=task, metrics=metrics), metrics)


def _stage_b_row_from_metrics_v2(
    *,
    task: BacktestStageBTaskV2,
    metrics: RankingMetricsV1,
) -> BacktestStageBScoredVariantV2:
    """
    Materialize one deterministic final row from scorer metrics and task identity payload.

    Args:
        task: Final task payload carrying deterministic identity and risk metadata.
        metrics: Ranking metrics payload already produced by Stage A or Stage B scoring.
    Returns:
        BacktestStageBScoredVariantV2: Deterministic final row used by sync responses and worker
            snapshots.
    Assumptions:
        No-risk terminal-path rows reuse the same final row contract as generic Stage B output,
        differing only in how the metrics were obtained.
    Raises:
        ValueError: If ranking metrics are missing required summary fields.
    Side Effects:
        None.
    """
    return BacktestStageBScoredVariantV2(
        variant_index=task.variant_index,
        indicator_variant_key=task.indicator_variant_key,
        variant_key=task.variant_key,
        total_return_pct=extract_metric_value_for_literal_v2(
            metrics=metrics,
            metric_literal=_TOTAL_RETURN_METRIC_KEY_LITERAL,
        ),
        summary_metrics_json=summary_metrics_from_ranking_metrics_v2(metrics=metrics),
        best_tp_pct=risk_pct_from_task_v2(
            task=task,
            flag_key="tp_enabled",
            value_key="tp_pct",
        ),
        best_sl_pct=risk_pct_from_task_v2(
            task=task,
            flag_key="sl_enabled",
            value_key="sl_pct",
        ),
    )


def iter_stage_b_tasks_v2(
    *,
    template: RunBacktestTemplate,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    shortlist: tuple[BacktestStageAScoredVariantV2, ...],
) -> tuple[BacktestStageBTaskV2, ...]:
    """
    Build deterministic Stage B task stream from Stage A shortlist and risk variants.

    Args:
        template: Effective run template.
        runtime_plan: Deterministic artifact-backed runtime plan.
        shortlist: Stage A shortlist rows.
    Returns:
        tuple[BacktestStageBTaskV2, ...]: Deterministic Stage B task tuple.
    Assumptions:
        Variant index contract remains `(shortlist_index * risk_total) + risk_index`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(
        _iter_stage_b_tasks_stream_v2(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
        )
    )


def _iter_stage_b_tasks_stream_v2(
    *,
    template: RunBacktestTemplate,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    shortlist: tuple[BacktestStageAScoredVariantV2, ...],
):
    """
    Stream deterministic Stage B tasks without materializing the full exact-task tuple up front.

    Args:
        template: Effective run template.
        runtime_plan: Deterministic artifact-backed runtime plan.
        shortlist: Stage A shortlist rows.
    Returns:
        Iterator[BacktestStageBTaskV2]: Deterministic Stage B task iterator.
    Assumptions:
        Streaming preserves the canonical `(shortlist_index * risk_total) + risk_index` ordering
        while reducing transient Python object churn on the exact path and preserving any internal
        retained exact payload already attached to shortlisted Stage A rows.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    risk_variants = runtime_plan.risk_variants
    risk_total = len(risk_variants)
    direction_mode = template.direction_mode
    sizing_mode = template.sizing_mode
    execution_params = template.execution_params or {}
    for shortlist_index, stage_a_row in enumerate(shortlist):
        base_variant = stage_a_row.base_variant
        for risk_variant in risk_variants:
            yield _stage_b_task_from_variant_v2(
                base_variant=base_variant,
                risk_variant=risk_variant,
                shortlist_index=shortlist_index,
                risk_total=risk_total,
                direction_mode=direction_mode,
                sizing_mode=sizing_mode,
                execution_params=execution_params,
                retained_exact_payload=stage_a_row.retained_exact_payload,
            )


def _stage_b_task_from_variant_v2(
    *,
    base_variant: BacktestStageABaseVariantV2,
    risk_variant: BacktestRiskVariantV2,
    shortlist_index: int,
    risk_total: int,
    direction_mode: str,
    sizing_mode: str,
    execution_params: Mapping[str, BacktestVariantScalar],
    retained_exact_payload: StageACompactExactPayloadV2 | None,
) -> BacktestStageBTaskV2:
    """
    Build one deterministic Stage B task from Stage A base variant and risk variant.

    Args:
        base_variant: Stage A base variant.
        risk_variant: Stage B risk variant.
        shortlist_index: Position of the Stage A row inside the shortlisted set.
        risk_total: Total number of Stage B risk variants.
        direction_mode: Effective direction mode literal.
        sizing_mode: Effective sizing mode literal.
        execution_params: Effective execution params mapping.
        retained_exact_payload: Optional internal retained exact payload forwarded from Stage A.
    Returns:
        BacktestStageBTaskV2: Deterministic Stage B task payload.
    Assumptions:
        Full variant-key semantics remain aligned with existing v1 payload contract, while the
        retained exact payload remains internal-only and additive.
    Raises:
        None.
    Side Effects:
        None.
    """
    variant_index = (shortlist_index * risk_total) + risk_variant.risk_index
    variant_key = build_backtest_variant_key_v1(
        indicator_variant_key=base_variant.indicator_variant_key,
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        signals=base_variant.signal_params,
        risk_params=risk_variant.risk_params,
        execution_params=execution_params,
    )
    return BacktestStageBTaskV2(
        variant_index=variant_index,
        indicator_variant_key=base_variant.indicator_variant_key,
        variant_key=variant_key,
        indicator_selections=base_variant.indicator_selections,
        signal_params=base_variant.signal_params,
        risk_params=risk_variant.risk_params,
        retained_exact_payload=_validated_retained_exact_payload_v2(
            retained_exact_payload=retained_exact_payload
        ),
    )


def summary_metrics_from_ranking_metrics_v2(
    *,
    metrics: RankingMetricsV1,
) -> Mapping[str, float]:
    """
    Normalize raw scorer metrics into deterministic summary metrics payload.

    Args:
        metrics: Raw scorer metrics payload for one Stage B variant.
    Returns:
        Mapping[str, float]: Immutable mapping with deterministic summary metric keys.
    Assumptions:
        Only approved numeric summary metrics participate in persisted summary rows.
    Raises:
        ValueError: If one kept metric value is non-numeric.
    Side Effects:
        None.
    """
    normalized: dict[str, float] = {}
    metric_key_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "total_return_pct",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["total_return_pct"],
        ),
        (
            "max_drawdown_pct",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["max_drawdown_pct"],
        ),
        (
            "return_over_max_drawdown",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["return_over_max_drawdown"],
        ),
        (
            "profit_factor",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["profit_factor"],
        ),
        (
            "sharpe_trades",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["sharpe_trades"],
        ),
        (
            "win_rate_pct",
            _SCORER_METRIC_KEYS_BY_LITERAL_V2["win_rate_pct"],
        ),
        ("trade_count", ("trade_count",)),
        ("avg_trade_ret_pct", ("avg_trade_ret_pct",)),
        ("avg_trade_exec_bars", ("avg_trade_exec_bars",)),
        ("exposure_pct", ("exposure_pct",)),
    )
    for metric_key, aliases in metric_key_aliases:
        value = next(
            (
                metrics[alias]
                for alias in aliases
                if alias in metrics and metrics[alias] is not None
            ),
            None,
        )
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"summary metric {metric_key!r} must be numeric")
        normalized[metric_key] = float(value)
    total_return_pct = next(
        (
            metrics[alias]
            for alias in _SCORER_METRIC_KEYS_BY_LITERAL_V2["total_return_pct"]
            if alias in metrics and metrics[alias] is not None
        ),
        None,
    )
    if isinstance(total_return_pct, bool) or not isinstance(total_return_pct, Real):
        raise ValueError("summary metric 'total_return_pct' must be numeric")
    normalized["total_return_pct"] = float(total_return_pct)
    return MappingProxyType(normalized)


def risk_pct_from_task_v2(
    *,
    task: BacktestStageBTaskV2,
    flag_key: str,
    value_key: str,
) -> float | None:
    """
    Extract nullable best TP/SL percentage from one Stage B task risk payload.

    Args:
        task: Stage B task carrying deterministic risk params.
        flag_key: Enable flag key (`tp_enabled` or `sl_enabled`).
        value_key: Percentage key (`tp_pct` or `sl_pct`).
    Returns:
        float | None: Non-negative percentage value or `None` when axis is disabled.
    Assumptions:
        Disabled TP/SL axes keep persisted best-cell fields null.
    Raises:
        ValueError: If enabled risk percent is non-numeric or negative.
    Side Effects:
        None.
    """
    if task.risk_params.get(flag_key) is False:
        return None
    value: Any = task.risk_params.get(value_key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"risk param {value_key!r} must be numeric")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"risk param {value_key!r} must be >= 0")
    return normalized


def stage_a_heap_entry_v2(
    *,
    row: BacktestStageAScoredVariantV2,
    metrics: RankingMetricsV1,
    ranking_plan: ResolvedRankingPlanV2,
) -> StageAHeapEntryV2:
    """
    Build Stage A heap entry where the smallest tuple is the current worst retained row.

    Args:
        row: Scored Stage A row.
        metrics: Raw scorer metrics payload.
        ranking_plan: Pre-resolved single-metric ranking plan.
    Returns:
        StageAHeapEntryV2: Heap entry preserving deterministic tie-break by base key.
    Assumptions:
        Final tie-break for Stage A is `base_variant_key ASC`.
    Raises:
        ValueError: If the ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = heap_metric_component_from_literal_v2(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    return (
        primary_component,
        descending_text_key_v2(value=row.base_variant.base_variant_key),
        row,
    )


def stage_b_heap_entry_v2(
    *,
    row: BacktestStageBScoredVariantV2,
    task: BacktestStageBTaskV2,
    metrics: RankingMetricsV1,
    ranking_plan: ResolvedRankingPlanV2,
) -> StageBHeapEntryV2:
    """
    Build Stage B heap entry where the smallest tuple is the current worst retained row.

    Args:
        row: Scored Stage B row.
        task: Stage B task payload corresponding to the scored row.
        metrics: Raw scorer metrics payload.
        ranking_plan: Pre-resolved single-metric ranking plan.
    Returns:
        StageBHeapEntryV2: Heap entry preserving deterministic tie-break by variant key.
    Assumptions:
        Final tie-break for Stage B is always `variant_key ASC`.
    Raises:
        ValueError: If the ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = heap_metric_component_from_literal_v2(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    return (
        primary_component,
        descending_text_key_v2(value=row.variant_key),
        row,
        task,
    )


def heap_entry_outranks_v2(
    *,
    candidate: StageAHeapEntryV2 | StageBHeapEntryV2,
    baseline: StageAHeapEntryV2 | StageBHeapEntryV2,
) -> bool:
    """
    Check whether candidate heap entry outranks current baseline entry.

    Args:
        candidate: Candidate heap entry.
        baseline: Baseline heap entry currently retained at the root.
    Returns:
        bool: `True` when the candidate must replace the baseline.
    Assumptions:
        Heap tuple ordering keeps the worst retained row at the root.
    Raises:
        None.
    Side Effects:
        None.
    """
    return candidate[:2] > baseline[:2]


def heap_metric_component_from_literal_v2(
    *,
    metrics: RankingMetricsV1,
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
        ValueError: If metric direction is unsupported or value is absent/non-numeric.
    Side Effects:
        None.
    """
    value = extract_metric_value_v2(
        metrics=metrics,
        metric_literal=metric_literal,
        scorer_metric_keys=scorer_metric_keys,
    )
    if metric_direction == _DIRECTION_DESC_LITERAL:
        return value
    if metric_direction == _DIRECTION_ASC_LITERAL:
        return -value
    raise ValueError(f"unsupported ranking direction for '{metric_literal}': {metric_direction!r}")


def descending_text_key_v2(*, value: str) -> tuple[int, ...]:
    """
    Encode a text key into a reverse-lexicographic tuple for heap baseline ordering.

    Args:
        value: Deterministic tie-break string key.
    Returns:
        tuple[int, ...]: Reverse-comparable tuple for heap entries.
    Assumptions:
        Sentinel `0` keeps strict ordering for prefix strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (*(-ord(char) for char in value), 0)


def extract_metric_value_for_literal_v2(
    *,
    metrics: RankingMetricsV1,
    metric_literal: str,
) -> float:
    """
    Extract numeric metric value by ranking literal using ordered scorer aliases.

    Args:
        metrics: Raw scorer metrics payload.
        metric_literal: Ranking metric literal (`total_return_pct`, etc.).
    Returns:
        float: Numeric metric value.
    Assumptions:
        Metric literal exists in the supported ranking set.
    Raises:
        ValueError: If the metric literal is unsupported or absent.
    Side Effects:
        None.
    """
    scorer_metric_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V2.get(metric_literal)
    if scorer_metric_keys is None:
        raise ValueError(f"unsupported ranking metric literal: {metric_literal!r}")
    return extract_metric_value_v2(
        metrics=metrics,
        metric_literal=metric_literal,
        scorer_metric_keys=scorer_metric_keys,
    )


def extract_metric_value_v2(
    *,
    metrics: RankingMetricsV1,
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
        The first present alias key is authoritative.
    Raises:
        ValueError: If the metric is absent or non-numeric.
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


def stage_a_rows_from_heap_v2(
    *,
    heap: list[StageAHeapEntryV2],
) -> tuple[BacktestStageAScoredVariantV2, ...]:
    """
    Materialize deterministic Stage A shortlist rows from bounded heap entries.

    Args:
        heap: Internal bounded Stage A heap entries.
    Returns:
        tuple[BacktestStageAScoredVariantV2, ...]: Deterministically ranked Stage A rows.
    Assumptions:
        Heap ordering uses ranking components and base-key tie-break transform.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(entry[2] for entry in sorted(heap, key=lambda item: item[:2], reverse=True))


def stage_b_rows_from_heap_v2(
    *,
    heap: list[StageBHeapEntryV2],
) -> tuple[BacktestStageBScoredVariantV2, ...]:
    """
    Materialize deterministic Stage B ranked rows from bounded heap entries.

    Args:
        heap: Internal bounded Stage B heap entries.
    Returns:
        tuple[BacktestStageBScoredVariantV2, ...]: Deterministically ranked Stage B rows.
    Assumptions:
        Heap ordering uses ranking components and `variant_key` tie-break transform.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(entry[2] for entry in sorted_stage_b_heap_entries_v2(heap=heap))


def sorted_stage_b_heap_entries_v2(
    *,
    heap: list[StageBHeapEntryV2],
) -> tuple[StageBHeapEntryV2, ...]:
    """
    Materialize deterministic sorted Stage B heap entries for rows/tasks projections.

    Args:
        heap: Internal bounded Stage B heap entries.
    Returns:
        tuple[StageBHeapEntryV2, ...]: Deterministically sorted Stage B heap entries.
    Assumptions:
        Sorting key is fixed to ranking components and `variant_key` tie-break transform.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(sorted(heap, key=lambda item: item[:2], reverse=True))


def stage_b_tasks_from_heap_v2(
    *,
    heap: list[StageBHeapEntryV2],
) -> Mapping[str, BacktestStageBTaskV2]:
    """
    Build deterministic `variant_key -> task` mapping from bounded Stage B heap entries.

    Args:
        heap: Internal bounded Stage B heap entries.
    Returns:
        Mapping[str, BacktestStageBTaskV2]: Deterministic task mapping.
    Assumptions:
        `variant_key` uniqueness is guaranteed by the Stage B identity builder.
    Raises:
        ValueError: If duplicate `variant_key` is detected.
    Side Effects:
        None.
    """
    return stage_b_tasks_from_sorted_entries_v2(entries=sorted_stage_b_heap_entries_v2(heap=heap))


def stage_b_tasks_from_sorted_entries_v2(
    *,
    entries: tuple[StageBHeapEntryV2, ...],
) -> Mapping[str, BacktestStageBTaskV2]:
    """
    Build deterministic `variant_key -> task` mapping from sorted Stage B entries.

    Args:
        entries: Sorted Stage B heap entries.
    Returns:
        Mapping[str, BacktestStageBTaskV2]: Deterministic task mapping.
    Assumptions:
        `variant_key` uniqueness is guaranteed by Stage B identity builder.
    Raises:
        ValueError: If duplicate `variant_key` is detected.
    Side Effects:
        None.
    """
    mapping: dict[str, BacktestStageBTaskV2] = {}
    for _, _, row, task in entries:
        if row.variant_key in mapping:
            raise ValueError("duplicate Stage-B variant_key is not allowed")
        mapping[row.variant_key] = task
    return mapping


def _runtime_plan_uses_no_risk_terminal_path_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> bool:
    """
    Resolve no-risk terminal-path classification for real or duck-typed runtime plans.

    Args:
        runtime_plan: Shared runtime-plan object used by sync and worker orchestration.
    Returns:
        bool: `True` when the plan belongs to the no-risk class and should bypass generic Stage B.
    Assumptions:
        Some unit tests supply lightweight runtime-plan fakes, so the shared core must tolerate
        either the real typed helper method or the canonical single disabled-risk-cell fallback.
    Raises:
        None.
    Side Effects:
        None.
    """
    classifier = getattr(runtime_plan, "uses_no_risk_terminal_path", None)
    if callable(classifier):
        return bool(classifier())
    risk_variants = tuple(getattr(runtime_plan, "risk_variants", ()))
    if len(risk_variants) != 1:
        return False
    risk_params = getattr(risk_variants[0], "risk_params", {})
    return (
        risk_params.get("sl_enabled") is False
        and risk_params.get("sl_pct") is None
        and risk_params.get("tp_enabled") is False
        and risk_params.get("tp_pct") is None
    )


def _runtime_plan_stage_b_execution_mode_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> str:
    """
    Resolve `stage_b_execution_mode` for real or duck-typed runtime plans.

    Args:
        runtime_plan: Shared runtime-plan object used by sync and worker orchestration.
    Returns:
        str: Canonical `stage_b_execution_mode` literal used by shared runtime branching.
    Assumptions:
        Tests may supply lightweight fakes, so the shared core first prefers the typed runtime
        helper and otherwise reconstructs the same classification from the available attributes,
        with `process_pool` reserved for explicit non-default opt-in profiles whose workload
        crosses the reviewable fallback path threshold.
    Raises:
        None.
    Side Effects:
        None.
    """
    classifier = getattr(runtime_plan, "stage_b_execution_mode", None)
    if callable(classifier):
        return str(classifier())
    if _runtime_plan_uses_no_risk_terminal_path_v2(runtime_plan=runtime_plan):
        return "bypassed_no_risk"
    execution_profile = getattr(runtime_plan, "execution_profile", None)
    stage_b_variants_total = int(getattr(runtime_plan, "stage_b_variants_total", 0))
    if execution_profile is None:
        return "in_process"
    if execution_profile_uses_process_pool_stage_b_v2(
        profile=execution_profile,
        stage_b_variants_total=stage_b_variants_total,
    ):
        return "process_pool"
    return "in_process"


def _runtime_plan_stage_b_process_fallback_threshold_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
) -> str:
    """
    Resolve which explicit workload threshold activated the non-default Stage B fallback path.

    Args:
        runtime_plan: Shared runtime-plan object used by sync and worker orchestration.
    Returns:
        str: Canonical workload-threshold literal for runtime-shape traces and benchmarks.
    Assumptions:
        Shared traces must expose not only whether the fallback path exists, but also which
        explicit workload threshold activated it.
    Raises:
        None.
    Side Effects:
        None.
    """
    classifier = getattr(runtime_plan, "stage_b_process_fallback_threshold", None)
    if callable(classifier):
        return str(classifier())
    if _runtime_plan_uses_no_risk_terminal_path_v2(runtime_plan=runtime_plan):
        return "none"
    execution_profile = getattr(runtime_plan, "execution_profile", None)
    if execution_profile is None:
        return "none"
    return execution_profile_stage_b_process_fallback_threshold_v2(
        profile=execution_profile,
        stage_b_variants_total=int(getattr(runtime_plan, "stage_b_variants_total", 0)),
    )


__all__ = [
    "BacktestArtifactRuntimeRunnerV2",
    "BacktestStageAScoredVariantV2",
    "BacktestStageBScoredVariantV2",
    "BacktestStageBTaskV2",
    "ResolvedRankingPlanV2",
    "STAGE_A_LITERAL_V2",
    "STAGE_B_LITERAL_V2",
    "configure_stage_ranking_context_if_supported_v2",
    "effective_ranking_config_v2",
    "heap_entry_outranks_v2",
    "iter_stage_b_tasks_v2",
    "resolve_ranking_plan_v2",
    "resolve_score_variant_metric_fn_v2",
    "risk_pct_from_task_v2",
    "score_stage_b_task_with_metrics_v2",
    "sorted_stage_b_heap_entries_v2",
    "stage_a_heap_entry_v2",
    "stage_a_rows_from_heap_v2",
    "stage_b_heap_entry_v2",
    "stage_b_rows_from_heap_v2",
    "stage_b_tasks_from_heap_v2",
    "stage_b_tasks_from_sorted_entries_v2",
    "summary_metrics_from_ranking_metrics_v2",
]
