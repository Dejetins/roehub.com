"""Artifact-backed runtime ranking core for Stage A shortlist outputs and Stage B top-k."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

from trading.contexts.backtest.application.dto import BacktestRankingConfig, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestStagedVariantMetricScorer,
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
_SECONDARY_METRIC_COMPONENT_DEFAULT = 0.0
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
    float,
    tuple[int, ...],
    "BacktestStageAScoredVariantV2",
]
StageBHeapEntryV2 = tuple[
    float,
    float,
    tuple[int, ...],
    "BacktestStageBScoredVariantV2",
    "BacktestStageBTaskV2",
]


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
    secondary_metric: str | None
    secondary_direction: str | None
    secondary_scorer_metric_keys: tuple[str, ...]


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
        Score Stage B variants with bounded top-k heap and optional checkpoint hooks.

        Args:
            template: Effective run template used for deterministic variant-key build.
            runtime_plan: Deterministic artifact-backed runtime plan.
            shortlist: Deterministically ranked Stage A shortlist rows.
            candles: Warmup-inclusive request-timeframe candles.
            scorer: Artifact-backed Stage B scorer contract implementation.
            top_k_limit: Maximum number of Stage B rows retained in memory.
            ranking:
                Optional ranking config (`primary_metric`, optional `secondary_metric`).
            batch_size: Optional checkpoint boundary override.
            cancel_checker: Optional cooperative cancellation callback by stage.
            on_checkpoint:
                Optional checkpoint callback with lazy frontier materializers.
        Returns:
            tuple[tuple[BacktestStageBScoredVariantV2, ...], Mapping[str, BacktestStageBTaskV2]]:
                Deterministically ranked Stage B rows and task mapping by `variant_key`.
        Assumptions:
            Final deterministic tie-break for Stage B remains `variant_key ASC`.
        Raises:
            ValueError: If limits/batch-size are invalid or scorer payload is malformed.
        Side Effects:
            None.
        """
        if top_k_limit <= 0:
            raise ValueError("BacktestArtifactRuntimeRunnerV2 top_k_limit must be > 0")
        ranking_plan = resolve_ranking_plan_v2(
            ranking=effective_ranking_config_v2(
                ranking=ranking,
                configurable_ranking_enabled=self._configurable_ranking_enabled,
            )
        )
        configure_stage_ranking_context_if_supported_v2(
            scorer=scorer,
            stage=STAGE_B_LITERAL_V2,
            ranking_plan=ranking_plan,
        )
        effective_batch = self._resolve_batch_size(batch_size=batch_size)
        total = int(runtime_plan.stage_b_variants_total)
        if cancel_checker is not None:
            cancel_checker(STAGE_B_LITERAL_V2)

        score_variant_metric = resolve_score_variant_metric_fn_v2(scorer=scorer)
        top_heap: list[StageBHeapEntryV2] = []
        processed = 0
        for task in _iter_stage_b_tasks_stream_v2(
            template=template,
            runtime_plan=runtime_plan,
            shortlist=shortlist,
        ):
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            row, metrics = score_stage_b_task_with_metrics_v2(
                task=task,
                candles=candles,
                score_variant_metric=score_variant_metric,
            )
            heap_entry = stage_b_heap_entry_v2(
                row=row,
                task=task,
                metrics=metrics,
                ranking_plan=ranking_plan,
            )
            if len(top_heap) < top_k_limit:
                heappush(top_heap, heap_entry)
            elif heap_entry_outranks_v2(candidate=heap_entry, baseline=top_heap[0]):
                heapreplace(top_heap, heap_entry)

            processed += 1
            if processed % effective_batch != 0 and processed != total:
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_B_LITERAL_V2)
            if on_checkpoint is not None:
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
                        rows_cache = tuple(entry[3] for entry in _sorted_entries())
                    return rows_cache

                def _materialize_tasks() -> Mapping[str, BacktestStageBTaskV2]:
                    nonlocal tasks_cache
                    if tasks_cache is None:
                        tasks_cache = stage_b_tasks_from_sorted_entries_v2(
                            entries=_sorted_entries()
                        )
                    return tasks_cache

                on_checkpoint(
                    processed,
                    total,
                    _materialize_ranked_rows,
                    _materialize_tasks,
                )

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


def configure_stage_ranking_context_if_supported_v2(
    *,
    scorer: MetricScorerV2,
    stage: str,
    ranking_plan: ResolvedRankingPlanV2,
) -> None:
    """
    Forward active stage ranking literals to scorers with additive ranking-context support.

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
        secondary_metric=ranking_plan.secondary_metric,
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
        Legacy deterministic behavior remains `total_return_pct DESC`.
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
        ValueError: If one ranking metric is unsupported.
    Side Effects:
        None.
    """
    primary_metric = ranking.primary_metric
    primary_direction = _METRIC_DIRECTION_BY_LITERAL_V2.get(primary_metric)
    primary_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V2.get(primary_metric)
    if primary_direction is None or primary_keys is None:
        raise ValueError(f"unsupported primary ranking metric: {primary_metric!r}")

    secondary_metric = ranking.secondary_metric
    secondary_direction: str | None = None
    secondary_keys: tuple[str, ...] = ()
    if secondary_metric is not None:
        secondary_direction = _METRIC_DIRECTION_BY_LITERAL_V2.get(secondary_metric)
        resolved_secondary_keys = _SCORER_METRIC_KEYS_BY_LITERAL_V2.get(secondary_metric)
        if secondary_direction is None or resolved_secondary_keys is None:
            raise ValueError(f"unsupported secondary ranking metric: {secondary_metric!r}")
        secondary_keys = resolved_secondary_keys

    return ResolvedRankingPlanV2(
        primary_metric=primary_metric,
        primary_direction=primary_direction,
        primary_scorer_metric_keys=primary_keys,
        secondary_metric=secondary_metric,
        secondary_direction=secondary_direction,
        secondary_scorer_metric_keys=secondary_keys,
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
    return (
        BacktestStageBScoredVariantV2(
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
        ),
        metrics,
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
        while reducing transient Python object churn on the exact path.
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
    Returns:
        BacktestStageBTaskV2: Deterministic Stage B task payload.
    Assumptions:
        Full variant-key semantics remain aligned with existing v1 payload contract.
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
        ranking_plan: Pre-resolved ranking plan.
    Returns:
        StageAHeapEntryV2: Heap entry preserving deterministic tie-break by base key.
    Assumptions:
        Final tie-break for Stage A is `base_variant_key ASC`.
    Raises:
        ValueError: If one ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = heap_metric_component_from_literal_v2(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    secondary_component = _SECONDARY_METRIC_COMPONENT_DEFAULT
    if ranking_plan.secondary_metric is not None and ranking_plan.secondary_direction is not None:
        secondary_component = heap_metric_component_from_literal_v2(
            metrics=metrics,
            metric_literal=ranking_plan.secondary_metric,
            metric_direction=ranking_plan.secondary_direction,
            scorer_metric_keys=ranking_plan.secondary_scorer_metric_keys,
        )
    return (
        primary_component,
        secondary_component,
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
        ranking_plan: Pre-resolved ranking plan.
    Returns:
        StageBHeapEntryV2: Heap entry preserving deterministic tie-break by variant key.
    Assumptions:
        Final tie-break for Stage B is always `variant_key ASC`.
    Raises:
        ValueError: If one ranking metric is missing or non-numeric.
    Side Effects:
        None.
    """
    primary_component = heap_metric_component_from_literal_v2(
        metrics=metrics,
        metric_literal=ranking_plan.primary_metric,
        metric_direction=ranking_plan.primary_direction,
        scorer_metric_keys=ranking_plan.primary_scorer_metric_keys,
    )
    secondary_component = _SECONDARY_METRIC_COMPONENT_DEFAULT
    if ranking_plan.secondary_metric is not None and ranking_plan.secondary_direction is not None:
        secondary_component = heap_metric_component_from_literal_v2(
            metrics=metrics,
            metric_literal=ranking_plan.secondary_metric,
            metric_direction=ranking_plan.secondary_direction,
            scorer_metric_keys=ranking_plan.secondary_scorer_metric_keys,
        )
    return (
        primary_component,
        secondary_component,
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
    return candidate[:3] > baseline[:3]


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
    return tuple(entry[3] for entry in sorted(heap, key=lambda item: item[:3], reverse=True))


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
    return tuple(entry[3] for entry in sorted_stage_b_heap_entries_v2(heap=heap))


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
    return tuple(sorted(heap, key=lambda item: item[:3], reverse=True))


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
    for _, _, _, row, task in entries:
        if row.variant_key in mapping:
            raise ValueError("duplicate Stage-B variant_key is not allowed")
        mapping[row.variant_key] = task
    return mapping


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
