"""Artifact-backed Stage B scorer over compact trades and strict shipped `1m hit-times`."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np

from trading.contexts.backtest.application.dto import RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestSignalParamsMap,
    BacktestStagedVariantMetricScorer,
    BacktestStagedVariantScorer,
    BacktestVariantScoreDetailsV1,
    RankingMetricsV1,
)
from trading.contexts.backtest.application.services.grid_builder_v1 import (
    STAGE_A_LITERAL,
    STAGE_B_LITERAL,
    BacktestGridBuildContextV1,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
    RiskParamsV1,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import TimeRange

from .contracts import (
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactSlotResolverV2,
    BacktestPriceArraysLoaderV2,
    BacktestSignalMatrixLoaderV2,
    StageACompactTradeV2,
    StageANoRiskMetricsV2,
    StageBFastSearchResultV2,
    StageBMetricsV2,
    StageBReplayPayloadV2,
)
from .metrics_kernel import (
    build_execution_outcome_from_replay_v2,
    compute_stage_b_metrics_v2,
    stage_b_metrics_to_ranking_payload_v2,
)
from .price_arrays_loader import MmapPriceArraysLoaderV2
from .risk_exit_kernel_1m import (
    replay_risk_cell_exact_v2,
    search_risk_cells_total_return_fast_v2,
    slice_hit_times_to_execution_window_v2,
)
from .signal_aggregator_kernel import aggregate_final_signal_rows_v2
from .signal_matrix_loader import MmapSignalMatrixLoaderV2
from .stage_a_shortlist_builder_v2 import (
    PreparedIndicatorRowPlanV2,
    build_prepared_indicator_row_plan_from_grid_spec_v2,
    compute_target_slice_by_close_time_v2,
    rebase_bar_close_mapping_v2,
)
from .trade_compactor_kernel import (
    build_compact_trade_list_v2,
    compute_no_risk_metrics_v2,
    no_risk_metrics_to_ranking_payload_v2,
)

_DEFAULT_FEE_PCT_BY_MARKET_ID_V2: Mapping[int, float] = MappingProxyType(
    {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }
)
_STAGE_A_DISABLED_RISK_PARAMS_V2: Mapping[str, BacktestVariantScalar] = MappingProxyType(
    {
        "sl_enabled": False,
        "sl_pct": None,
        "tp_enabled": False,
        "tp_pct": None,
    }
)


@dataclass(frozen=True, slots=True)
class _PreparedStageABasePayloadV2:
    """
    Cached Stage A artifact-backed payload reused by Stage B scoring and details replay.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
    """

    compact_trades: tuple[StageACompactTradeV2, ...]
    no_risk_metrics: StageANoRiskMetricsV2


@dataclass(frozen=True, slots=True)
class _ExactStageBCellCacheV2:
    """
    Cached exact replay payload and metrics for one deterministic Stage B variant key.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """

    replay: StageBReplayPayloadV2
    metrics: StageBMetricsV2


class BacktestArtifactBackedStageBScorerV2(
    BacktestStagedVariantScorer,
    BacktestStagedVariantMetricScorer,
):
    """
    Artifact-backed scorer using Stage A compact trades and Stage B `1m hit-times` kernels.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    def __init__(
        self,
        *,
        price_arrays_loader: BacktestPriceArraysLoaderV2,
        signal_matrix_loader: BacktestSignalMatrixLoaderV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        target_time_range: TimeRange,
        report_target_slice: slice,
        direction_mode: str,
        sizing_mode: str,
        execution_params: Mapping[str, BacktestVariantScalar],
        market_id: int,
        signal_timeframe: str,
        indicator_grids: tuple[GridSpec, ...],
        init_cash_quote_default: float = 10000.0,
        fixed_quote_default: float = 100.0,
        safe_profit_percent_default: float = 30.0,
        slippage_pct_default: float = 0.01,
        fee_pct_default_by_market_id: Mapping[int, float] = _DEFAULT_FEE_PCT_BY_MARKET_ID_V2,
        close_on_end: bool = True,
    ) -> None:
        """
        Bootstrap artifact-backed Stage B scorer state and fail fast on artifact contract drift.

        Args:
            price_arrays_loader: Explicit-path loader for prices, mappings, and `1m hit-times`.
            signal_matrix_loader: Explicit-path loader for subset signal rows.
            artifact_context: Slot-pinned artifact context resolved at runtime startup.
            target_time_range: Requested trading/reporting window in request-timeframe terms.
            report_target_slice: Request-timeframe target slice used by details/report payloads.
            direction_mode: Runtime direction mode literal.
            sizing_mode: Runtime sizing mode literal.
            execution_params: Runtime execution scalar overrides.
            market_id: Stable market identifier used for fee-default lookup.
            signal_timeframe: Request-timeframe signal artifact family.
            indicator_grids: Template indicator grids used for lazy row-plan fallback.
            init_cash_quote_default: Runtime default initial quote balance.
            fixed_quote_default: Runtime default fixed quote notional.
            safe_profit_percent_default: Runtime default safe-profit lock percent.
            slippage_pct_default: Runtime default slippage percent.
            fee_pct_default_by_market_id: Runtime default fee percent mapping by market id.
            close_on_end: Explicit notebook-derived `close_on_end = 1` runtime switch.
        Returns:
            None.
        Assumptions:
            Constructor may load strict artifact families up front because fail-fast startup is a
            hard runtime contract for v2 artifact-backed paths.
        Raises:
            ValueError: If defaults, loader dependencies, or local artifact slice contracts drift.
        Side Effects:
            Loads pinned artifact arrays and precomputes local request/execution slices.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
        """
        if price_arrays_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactBackedStageBScorerV2 requires price_arrays_loader")
        if signal_matrix_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestArtifactBackedStageBScorerV2 requires signal_matrix_loader")
        self._price_arrays_loader = price_arrays_loader
        self._signal_matrix_loader = signal_matrix_loader
        self._artifact_context = artifact_context
        self._signal_timeframe = signal_timeframe
        self._report_target_slice = report_target_slice
        self._close_on_end = bool(close_on_end)
        self._indicator_grids_by_id = {
            str(grid.indicator_id): grid
            for grid in sorted(indicator_grids, key=lambda item: str(item.indicator_id))
        }
        self._execution_params = _resolve_execution_params_v2(
            direction_mode=direction_mode,
            sizing_mode=sizing_mode,
            execution_params=execution_params,
            market_id=market_id,
            init_cash_quote_default=init_cash_quote_default,
            fixed_quote_default=fixed_quote_default,
            safe_profit_percent_default=safe_profit_percent_default,
            slippage_pct_default=slippage_pct_default,
            fee_pct_default_by_market_id=fee_pct_default_by_market_id,
        )
        self._execution_params_mapping = MappingProxyType(
            dict(sorted((str(key), value) for key, value in (execution_params or {}).items()))
        )
        signal_prices = self._price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe=signal_timeframe,
        )
        mapping_arrays = self._price_arrays_loader.load_mapping_arrays(
            context=artifact_context,
            timeframe=signal_timeframe,
        )
        execution_prices = self._price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe="1m",
        )
        hit_times_arrays = self._price_arrays_loader.load_hit_times_arrays(context=artifact_context)
        self._signal_target_slice = compute_target_slice_by_close_time_v2(
            close_time=signal_prices.close_time,
            target_time_range=target_time_range,
        )
        self._exec_target_slice = compute_target_slice_by_close_time_v2(
            close_time=execution_prices.close_time,
            target_time_range=target_time_range,
        )
        self._local_bar_close_1m_idx = rebase_bar_close_mapping_v2(
            mapping_values=mapping_arrays.bar_close_1m_idx[self._signal_target_slice],
            exec_target_slice=self._exec_target_slice,
        )
        self._local_exec_open = np.asarray(
            execution_prices.ohlcv[self._exec_target_slice, 0],
            dtype=np.float64,
        )
        self._local_exec_close = np.asarray(
            execution_prices.ohlcv[self._exec_target_slice, 3],
            dtype=np.float64,
        )
        self._local_hit_times = slice_hit_times_to_execution_window_v2(
            hit_times_arrays=hit_times_arrays,
            exec_target_slice=self._exec_target_slice,
        )
        self._sentinel_index = int(self._local_exec_open.shape[0])
        self._prepared_row_plans_by_indicator: dict[str, PreparedIndicatorRowPlanV2] = {}
        self._stage_a_payload_cache_by_base_variant_key: dict[
            str, _PreparedStageABasePayloadV2
        ] = {}
        self._stage_b_exact_cache_by_variant_key: dict[str, _ExactStageBCellCacheV2] = {}
        self._fast_search_cache_by_base_variant_key: dict[str, StageBFastSearchResultV2] = {}
        self._ranking_primary_by_stage: dict[str, str] = {}
        self._ranking_secondary_by_stage: dict[str, str | None] = {}

    def prepare_for_grid_context(
        self,
        *,
        grid_context: BacktestGridBuildContextV1,
        candles: CandleArrays,
        max_compute_bytes_total: int,
        run_control: object | None = None,
    ) -> None:
        """
        Cache artifact row-addressing plans for the current staged run context.

        Args:
            grid_context: Deterministic staged grid context for the current run.
            candles: Warmup-inclusive request-timeframe candles, unused by artifact-backed path.
            max_compute_bytes_total: Legacy scorer compatibility argument, unused here.
            run_control: Optional cancellation hook, unused because no compute work occurs here.
        Returns:
            None.
        Assumptions:
            Artifact-backed Stage B scorer needs only deterministic row addressing and may reuse
            preloaded artifacts without extra indicator compute work.
        Raises:
            ValueError: If one indicator plan cannot be converted into artifact row metadata.
        Side Effects:
            Clears per-run caches and stores prepared row plans by indicator id.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/grid_builder_v1.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
        """
        _ = candles, max_compute_bytes_total, run_control
        self._prepared_row_plans_by_indicator = {
            plan.indicator_id: PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
            for plan in grid_context.indicator_plans
        }
        self._stage_a_payload_cache_by_base_variant_key.clear()
        self._stage_b_exact_cache_by_variant_key.clear()
        self._fast_search_cache_by_base_variant_key.clear()

    def configure_stage_ranking_context(
        self,
        *,
        stage: str,
        primary_metric: str,
        secondary_metric: str | None,
    ) -> None:
        """
        Store active staged-run ranking literals so Stage B can enable safe fast-path lookups.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            primary_metric: Active primary ranking metric literal.
            secondary_metric: Active secondary metric literal or `None`.
        Returns:
            None.
        Assumptions:
            Fast TP/SL search can safely answer Stage B hot-path scoring only when ranking uses
            `total_return_pct` with no secondary metric.
        Raises:
            None.
        Side Effects:
            Updates in-memory ranking-context hints for the current scorer instance.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        self._ranking_primary_by_stage[stage] = primary_metric
        self._ranking_secondary_by_stage[stage] = secondary_metric

    def score_variant_metric(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> RankingMetricsV1:
        """
        Score one Stage A or Stage B variant using artifacts-only kernels and exact fallback.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Warmup-inclusive request-timeframe candles, unused by artifact-backed path.
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic full backtest variant key.
        Returns:
            RankingMetricsV1: Deterministic metric payload compatible with staged runner ranking.
        Assumptions:
            Stage A remains no-risk, while Stage B uses fast total-return lookup only when the
            active ranking plan is exactly `total_return_pct DESC` with no secondary metric.
        Raises:
            ValueError: If stage, artifact row addressing, or exact replay contracts are invalid.
        Side Effects:
            Populates per-run caches for Stage A compact trades, fast search, and exact replay.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
        """
        _ = candles
        if stage == STAGE_A_LITERAL:
            base_variant_key = self._base_variant_key_v2(
                indicator_variant_key=indicator_variant_key,
                signal_params=signal_params,
            )
            payload = self._stage_a_payload_for_variant_v2(
                indicator_selections=indicator_selections,
                signal_params=signal_params,
                base_variant_key=base_variant_key,
            )
            return no_risk_metrics_to_ranking_payload_v2(metrics=payload.no_risk_metrics)
        if stage != STAGE_B_LITERAL:
            raise ValueError(f"unsupported stage literal for artifact-backed scorer: {stage!r}")

        base_variant_key = self._base_variant_key_v2(
            indicator_variant_key=indicator_variant_key,
            signal_params=signal_params,
        )
        tp_index, sl_index = self._resolve_risk_level_indexes_v2(risk_params=risk_params)
        if (
            self._can_use_stage_b_total_return_fast_path_v2()
            and tp_index is not None
            and sl_index is not None
        ):
            fast_result = self._fast_stage_b_search_for_base_variant_v2(
                indicator_selections=indicator_selections,
                signal_params=signal_params,
                base_variant_key=base_variant_key,
            )
            total_return_pct = float(fast_result.total_return_pct[tp_index, sl_index])
            return MappingProxyType(
                {
                    "total_return_pct": total_return_pct,
                    "Total Return [%]": total_return_pct,
                }
            )
        exact_cache = self._exact_stage_b_cell_cache_v2(
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            variant_key=variant_key,
            base_variant_key=base_variant_key,
        )
        return stage_b_metrics_to_ranking_payload_v2(metrics=exact_cache.metrics)

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> RankingMetricsV1:
        """
        Preserve legacy scorer compatibility by delegating to `score_variant_metric(...)`.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Warmup-inclusive request-timeframe candles.
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            risk_params: Risk payload for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic full backtest variant key.
        Returns:
            RankingMetricsV1: Deterministic metric payload for staged ranking.
        Assumptions:
            Artifact-backed scorer keeps public scorer imports stable and additive.
        Raises:
            ValueError: Propagated from `score_variant_metric(...)`.
        Side Effects:
            Reuses the same caches as `score_variant_metric(...)`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        return self.score_variant_metric(
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
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> BacktestVariantScoreDetailsV1:
        """
        Build deterministic Stage B details from exact replay only for retained explicit variants.

        Args:
            stage: Stage literal expected to be `stage_b`.
            candles: Warmup-inclusive request-timeframe candles, unused by artifact-backed path.
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            risk_params: Risk payload for this variant.
            indicator_variant_key: Deterministic compute-only indicators key.
            variant_key: Deterministic full backtest variant key.
        Returns:
            BacktestVariantScoreDetailsV1: Exact Stage B metrics plus details-compatible outcome.
        Assumptions:
            Details path is intentionally exact and limited to already selected/report variants.
        Raises:
            ValueError: If called for unsupported stage or if exact replay contracts drift.
        Side Effects:
            Populates exact replay cache for the requested variant when absent.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
        """
        _ = candles
        if stage != STAGE_B_LITERAL:
            raise ValueError("artifact-backed details scorer supports Stage B only")
        base_variant_key = self._base_variant_key_v2(
            indicator_variant_key=indicator_variant_key,
            signal_params=signal_params,
        )
        exact_cache = self._exact_stage_b_cell_cache_v2(
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            variant_key=variant_key,
            base_variant_key=base_variant_key,
        )
        execution_outcome = build_execution_outcome_from_replay_v2(
            replay=exact_cache.replay,
            metrics=exact_cache.metrics,
            execution_params=self._execution_params,
            exec_open=self._local_exec_open,
            exec_close=self._local_exec_close,
            tp_values=self._local_hit_times.tp_values,
            sl_values=self._local_hit_times.sl_values,
        )
        return BacktestVariantScoreDetailsV1(
            metrics=stage_b_metrics_to_ranking_payload_v2(metrics=exact_cache.metrics),
            target_slice=self._report_target_slice,
            execution_params=self._execution_params,
            risk_params=_resolve_risk_params_v2(risk_params=risk_params),
            execution_outcome=execution_outcome,
        )

    def _stage_a_payload_for_variant_v2(
        self,
        *,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        base_variant_key: str,
    ) -> _PreparedStageABasePayloadV2:
        """
        Load subset signal rows, aggregate final signal, and cache Stage A compact trades.

        Args:
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            base_variant_key: Deterministic Stage A cache key with risk disabled.
        Returns:
            _PreparedStageABasePayloadV2: Cached compact trades and no-risk metrics.
        Assumptions:
            Stage A compact trades are the single shared upstream contract for Stage B kernels.
        Raises:
            ValueError: If one indicator row cannot be resolved inside artifact signal matrices.
        Side Effects:
            Reads subset artifact signal rows and populates per-run cache on first access.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
          - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
        """
        cached = self._stage_a_payload_cache_by_base_variant_key.get(base_variant_key)
        if cached is not None:
            return cached
        _ = signal_params
        selected_signal_rows: dict[str, np.ndarray] = {}
        for selection in sorted(indicator_selections, key=lambda item: item.indicator_id):
            row_plan = self._row_plan_for_indicator_v2(indicator_id=selection.indicator_id)
            row_index = row_plan.row_index_for_selection(selection=selection)
            selected_rows = self._signal_matrix_loader.load_signal_rows(
                context=self._artifact_context,
                timeframe=self._signal_timeframe,
                indicator_id=selection.indicator_id,
                row_selection=(row_index,),
            )
            selected_signal_rows[selection.indicator_id] = np.asarray(
                selected_rows[:, self._signal_target_slice],
                dtype=np.int8,
            )
        final_signal = aggregate_final_signal_rows_v2(selected_signal_rows=selected_signal_rows)
        compact_trades = build_compact_trade_list_v2(
            final_signal=final_signal,
            bar_close_1m_idx=self._local_bar_close_1m_idx,
            sentinel_index=self._sentinel_index,
            direction_mode=self._execution_params.direction_mode,
        )[0]
        no_risk_metrics = compute_no_risk_metrics_v2(
            compact_trades=compact_trades,
            exec_open=self._local_exec_open,
            exec_close=self._local_exec_close,
            sentinel_index=self._sentinel_index,
            execution_params=self._execution_params,
            close_on_end=self._close_on_end,
        )
        payload = _PreparedStageABasePayloadV2(
            compact_trades=compact_trades,
            no_risk_metrics=no_risk_metrics,
        )
        self._stage_a_payload_cache_by_base_variant_key[base_variant_key] = payload
        return payload

    def _fast_stage_b_search_for_base_variant_v2(
        self,
        *,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        base_variant_key: str,
    ) -> StageBFastSearchResultV2:
        """
        Compute and cache fast TP/SL total-return matrix for one Stage A base variant.

        Args:
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            base_variant_key: Deterministic Stage A cache key with risk disabled.
        Returns:
            StageBFastSearchResultV2: Cached fast-search result for this base variant.
        Assumptions:
            Fast search is valid only over the Stage A compact trade list produced from the same
            artifact-backed signal aggregation contract.
        Raises:
            ValueError: Propagated from fast TP/SL search on contract drift.
        Side Effects:
            Populates per-run fast-search cache on first access.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
        """
        cached = self._fast_search_cache_by_base_variant_key.get(base_variant_key)
        if cached is not None:
            return cached
        payload = self._stage_a_payload_for_variant_v2(
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            base_variant_key=base_variant_key,
        )
        fast_result = search_risk_cells_total_return_fast_v2(
            compact_trades=payload.compact_trades,
            hit_times=self._local_hit_times,
            exec_open=self._local_exec_open,
            exec_close=self._local_exec_close,
            fee_rate=self._execution_params.fee_rate,
            close_on_end=self._close_on_end,
        )
        self._fast_search_cache_by_base_variant_key[base_variant_key] = fast_result
        return fast_result

    def _exact_stage_b_cell_cache_v2(
        self,
        *,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: BacktestSignalParamsMap,
        risk_params: Mapping[str, BacktestVariantScalar],
        variant_key: str,
        base_variant_key: str,
    ) -> _ExactStageBCellCacheV2:
        """
        Exact-replay one requested Stage B risk cell and cache its deterministic metrics.

        Args:
            indicator_selections: Explicit indicator selections for one deterministic variant.
            signal_params: Signal-parameter values for this variant.
            risk_params: Risk payload for this variant.
            variant_key: Deterministic full backtest variant key.
            base_variant_key: Deterministic Stage A cache key with risk disabled.
        Returns:
            _ExactStageBCellCacheV2: Cached exact replay payload and metrics.
        Assumptions:
            Exact replay is the only source of truth for non-default Stage B ranking and details.
        Raises:
            ValueError: If one risk level is not present in the shipped artifact grids.
        Side Effects:
            Populates per-run exact replay cache on first access.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
        """
        cached = self._stage_b_exact_cache_by_variant_key.get(variant_key)
        if cached is not None:
            return cached
        tp_index, sl_index = self._resolve_risk_level_indexes_v2(risk_params=risk_params)
        payload = self._stage_a_payload_for_variant_v2(
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            base_variant_key=base_variant_key,
        )
        replay = replay_risk_cell_exact_v2(
            compact_trades=payload.compact_trades,
            hit_times=self._local_hit_times,
            exec_open=self._local_exec_open,
            exec_close=self._local_exec_close,
            tp_index=tp_index,
            sl_index=sl_index,
            close_on_end=self._close_on_end,
        )
        metrics = compute_stage_b_metrics_v2(
            replay=replay,
            fee_rate=self._execution_params.fee_rate,
        )
        exact_cache = _ExactStageBCellCacheV2(replay=replay, metrics=metrics)
        self._stage_b_exact_cache_by_variant_key[variant_key] = exact_cache
        return exact_cache

    def _row_plan_for_indicator_v2(self, *, indicator_id: str) -> PreparedIndicatorRowPlanV2:
        """
        Resolve cached or lazy-built artifact row-addressing plan for one indicator id.

        Args:
            indicator_id: Canonical indicator identifier.
        Returns:
            PreparedIndicatorRowPlanV2: Prepared mixed-radix row-addressing plan.
        Assumptions:
            `prepare_for_grid_context(...)` may not run for explicit report scoring, so template
            grids remain the lazy fallback source of truth.
        Raises:
            ValueError: If the template is missing a grid for the requested indicator id.
        Side Effects:
            Stores lazily built row plans into the in-memory cache.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        cached = self._prepared_row_plans_by_indicator.get(indicator_id)
        if cached is not None:
            return cached
        grid = self._indicator_grids_by_id.get(indicator_id)
        if grid is None:
            raise ValueError(f"missing template GridSpec for indicator_id {indicator_id!r}")
        prepared = build_prepared_indicator_row_plan_from_grid_spec_v2(
            indicator_id=indicator_id,
            grid_spec=grid,
        )
        self._prepared_row_plans_by_indicator[indicator_id] = prepared
        return prepared

    def _base_variant_key_v2(
        self,
        *,
        indicator_variant_key: str,
        signal_params: BacktestSignalParamsMap,
    ) -> str:
        """
        Build deterministic Stage A cache key for one indicator/signals combination.

        Args:
            indicator_variant_key: Deterministic compute-only indicators key.
            signal_params: Signal-parameter values for this variant.
        Returns:
            str: Deterministic full variant key with risk explicitly disabled.
        Assumptions:
            Stage A cache keys must preserve existing variant-key v1 semantics unchanged.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        return build_backtest_variant_key_v1(
            indicator_variant_key=indicator_variant_key,
            direction_mode=self._execution_params.direction_mode,
            sizing_mode=self._execution_params.sizing_mode,
            signals=signal_params,
            risk_params=_STAGE_A_DISABLED_RISK_PARAMS_V2,
            execution_params=self._execution_params_mapping,
        )

    def _resolve_risk_level_indexes_v2(
        self,
        *,
        risk_params: Mapping[str, BacktestVariantScalar],
    ) -> tuple[int | None, int | None]:
        """
        Resolve TP/SL artifact-grid indexes from runtime risk payload percentages.

        Args:
            risk_params: Risk payload (`sl_enabled/sl_pct/tp_enabled/tp_pct`) for one variant.
        Returns:
            tuple[int | None, int | None]: Optional `(tp_index, sl_index)` cell coordinates.
        Assumptions:
            Risk percentages are authored in human percent units (`1.0 == 1%`) while artifact
            grids store decimal rates (`0.01`).
        Raises:
            ValueError: If one enabled TP/SL percentage is missing or absent in the shipped grid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        risk = _resolve_risk_params_v2(risk_params=risk_params)
        tp_index: int | None = None
        sl_index: int | None = None
        if risk.tp_enabled:
            if risk.tp_rate is None:
                raise ValueError("tp_rate must be set when tp_enabled is true")
            tp_index = _index_for_rate_v2(
                values=self._local_hit_times.tp_values,
                rate=float(risk.tp_rate),
                field_name="tp_pct",
            )
        if risk.sl_enabled:
            if risk.sl_rate is None:
                raise ValueError("sl_rate must be set when sl_enabled is true")
            sl_index = _index_for_rate_v2(
                values=self._local_hit_times.sl_values,
                rate=float(risk.sl_rate),
                field_name="sl_pct",
            )
        return (tp_index, sl_index)

    def _can_use_stage_b_total_return_fast_path_v2(self) -> bool:
        """
        Check whether current Stage B ranking plan can safely use fast total-return lookup only.

        Args:
            None.
        Returns:
            bool: `True` when Stage B ranking uses only `total_return_pct`.
        Assumptions:
            Secondary metrics or alternative primary metrics require exact replay metrics.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        return (
            self._ranking_primary_by_stage.get(STAGE_B_LITERAL) == "total_return_pct"
            and self._ranking_secondary_by_stage.get(STAGE_B_LITERAL) is None
        )


def build_default_artifact_backed_stage_b_scorer_v2(
    *,
    artifact_slot_resolver: BacktestArtifactSlotResolverV2 | None,
    artifact_context: ArtifactSlotPinnedRuntimeContextV2 | None,
    template: RunBacktestTemplate,
    target_time_range: TimeRange,
    report_target_slice: slice,
    init_cash_quote_default: float = 10000.0,
    fixed_quote_default: float = 100.0,
    safe_profit_percent_default: float = 30.0,
    slippage_pct_default: float = 0.01,
    fee_pct_default_by_market_id: Mapping[int, float] | None = None,
) -> BacktestArtifactBackedStageBScorerV2 | None:
    """
    Build default artifact-backed Stage B scorer from resolver wiring when v2 runtime is pinned.

    Args:
        artifact_slot_resolver: Optional slot resolver already wired by runtime startup.
        artifact_context: Optional slot-pinned artifact context for current run.
        template: Effective run template.
        target_time_range: Requested trading/reporting window.
        report_target_slice: Request-timeframe target slice used by details/report payloads.
        init_cash_quote_default: Runtime default initial quote balance.
        fixed_quote_default: Runtime default fixed quote notional.
        safe_profit_percent_default: Runtime default safe-profit lock percent.
        slippage_pct_default: Runtime default slippage percent.
        fee_pct_default_by_market_id: Runtime default fee percent mapping by market id.
    Returns:
        BacktestArtifactBackedStageBScorerV2 | None: Default artifact-backed scorer when runtime
            wiring is available, otherwise `None`.
    Assumptions:
        Legacy scorer remains the guarded fallback when artifact loader or pinned context is
        unavailable.
    Raises:
        ValueError: Propagated from scorer constructor on invalid runtime defaults.
    Side Effects:
        Loads strict artifact families during scorer bootstrap when constructed.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """
    if artifact_slot_resolver is None or artifact_context is None:
        return None
    artifact_loader = getattr(artifact_slot_resolver, "artifact_loader", None)
    if artifact_loader is None:
        return None
    typed_artifact_loader = cast(BacktestArtifactLoaderV2, artifact_loader)
    return BacktestArtifactBackedStageBScorerV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=typed_artifact_loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=typed_artifact_loader),
        artifact_context=artifact_context,
        target_time_range=target_time_range,
        report_target_slice=report_target_slice,
        direction_mode=template.direction_mode,
        sizing_mode=template.sizing_mode,
        execution_params=template.execution_params or {},
        market_id=template.instrument_id.market_id.value,
        signal_timeframe=template.timeframe.code,
        indicator_grids=template.indicator_grids,
        init_cash_quote_default=init_cash_quote_default,
        fixed_quote_default=fixed_quote_default,
        safe_profit_percent_default=safe_profit_percent_default,
        slippage_pct_default=slippage_pct_default,
        fee_pct_default_by_market_id=(
            fee_pct_default_by_market_id
            if fee_pct_default_by_market_id is not None
            else _DEFAULT_FEE_PCT_BY_MARKET_ID_V2
        ),
    )


def _resolve_execution_params_v2(
    *,
    direction_mode: str,
    sizing_mode: str,
    execution_params: Mapping[str, BacktestVariantScalar],
    market_id: int,
    init_cash_quote_default: float,
    fixed_quote_default: float,
    safe_profit_percent_default: float,
    slippage_pct_default: float,
    fee_pct_default_by_market_id: Mapping[int, float],
) -> ExecutionParamsV1:
    """
    Resolve immutable runtime execution settings for artifact-backed Stage B kernels.

    Args:
        direction_mode: Runtime direction mode literal.
        sizing_mode: Runtime sizing mode literal.
        execution_params: Runtime execution scalar overrides.
        market_id: Stable market identifier used for fee-default lookup.
        init_cash_quote_default: Runtime default initial quote balance.
        fixed_quote_default: Runtime default fixed quote notional.
        safe_profit_percent_default: Runtime default safe-profit lock percent.
        slippage_pct_default: Runtime default slippage percent.
        fee_pct_default_by_market_id: Runtime default fee percent mapping by market id.
    Returns:
        ExecutionParamsV1: Immutable execution settings shared by Stage A and Stage B kernels.
    Assumptions:
        Missing overrides fall back to the same runtime defaults already used by legacy scoring.
    Raises:
        KeyError: If fee defaults are missing for the requested market id.
        ValueError: If one execution scalar is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    return ExecutionParamsV1(
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        init_cash_quote=_resolve_number_v2(
            values=execution_params,
            primary_key="init_cash_quote",
            secondary_key="init_cash",
            default=init_cash_quote_default,
        ),
        fixed_quote=_resolve_number_v2(
            values=execution_params,
            primary_key="fixed_quote",
            secondary_key="",
            default=fixed_quote_default,
        ),
        safe_profit_percent=_resolve_number_v2(
            values=execution_params,
            primary_key="safe_profit_percent",
            secondary_key="",
            default=safe_profit_percent_default,
        ),
        fee_pct=_resolve_number_v2(
            values=execution_params,
            primary_key="fee_pct",
            secondary_key="market_fee_pct",
            default=fee_pct_default_by_market_id[market_id],
        ),
        slippage_pct=_resolve_number_v2(
            values=execution_params,
            primary_key="slippage_pct",
            secondary_key="",
            default=slippage_pct_default,
        ),
    )


def _resolve_number_v2(
    *,
    values: Mapping[str, BacktestVariantScalar],
    primary_key: str,
    secondary_key: str,
    default: float,
) -> float:
    """
    Resolve one numeric scalar from runtime override mappings with deterministic key fallback.

    Args:
        values: Runtime scalar overrides mapping.
        primary_key: Preferred override key.
        secondary_key: Secondary legacy-compatible override key.
        default: Default numeric value when neither key is present.
    Returns:
        float: Resolved numeric scalar.
    Assumptions:
        Runtime overrides are already scalar-only and may use either modern or legacy literals.
    Raises:
        ValueError: If resolved value is not numeric.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    raw_value = values.get(primary_key, values.get(secondary_key, default))
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        raise ValueError(f"{primary_key or secondary_key} must resolve to numeric scalar")
    return float(raw_value)


def _resolve_risk_params_v2(
    *,
    risk_params: Mapping[str, BacktestVariantScalar],
) -> RiskParamsV1:
    """
    Normalize risk payload mapping into immutable `RiskParamsV1`.

    Args:
        risk_params: Runtime risk scalar mapping.
    Returns:
        RiskParamsV1: Immutable normalized Stage B risk settings.
    Assumptions:
        Missing enable flags imply disabled axes and missing percentages imply `None`.
    Raises:
        ValueError: If enable flags are not booleans or enabled axes miss numeric percentages.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    sl_enabled = _bool_from_scalar_v2(
        value=risk_params.get("sl_enabled", False),
        field_name="sl_enabled",
    )
    tp_enabled = _bool_from_scalar_v2(
        value=risk_params.get("tp_enabled", False),
        field_name="tp_enabled",
    )
    sl_pct_raw = risk_params.get("sl_pct")
    tp_pct_raw = risk_params.get("tp_pct")
    sl_pct = (
        None
        if sl_pct_raw is None
        else _number_from_scalar_v2(value=sl_pct_raw, field_name="sl_pct")
    )
    tp_pct = (
        None
        if tp_pct_raw is None
        else _number_from_scalar_v2(value=tp_pct_raw, field_name="tp_pct")
    )
    return RiskParamsV1(
        sl_enabled=sl_enabled,
        sl_pct=sl_pct,
        tp_enabled=tp_enabled,
        tp_pct=tp_pct,
    )


def _bool_from_scalar_v2(*, value: BacktestVariantScalar, field_name: str) -> bool:
    """
    Resolve one explicit boolean literal from runtime scalar payloads.

    Args:
        value: Raw scalar value from runtime mapping.
        field_name: Deterministic diagnostics field label.
    Returns:
        bool: Parsed boolean literal.
    Assumptions:
        Risk enable flags remain explicit booleans in current public DTO contracts.
    Raises:
        ValueError: If the scalar is not boolean.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")
    return value


def _number_from_scalar_v2(*, value: BacktestVariantScalar, field_name: str) -> float:
    """
    Resolve one numeric scalar from runtime payloads.

    Args:
        value: Raw scalar value from runtime mapping.
        field_name: Deterministic diagnostics field label.
    Returns:
        float: Parsed numeric scalar.
    Assumptions:
        Risk percentages remain numeric human-percent literals in public DTO contracts.
    Raises:
        ValueError: If the scalar is not numeric.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _index_for_rate_v2(
    *,
    values: np.ndarray,
    rate: float,
    field_name: str,
) -> int:
    """
    Resolve exact artifact-grid index for one decimal TP/SL rate with stable tolerance.

    Args:
        values: Artifact grid values in decimal-rate form.
        rate: Required decimal rate.
        field_name: Deterministic diagnostics field label.
    Returns:
        int: Matching grid index.
    Assumptions:
        Artifact grids are shipped exactly and runtime risk percentages must map onto them.
    Raises:
        ValueError: If the requested rate is absent from the shipped grid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    normalized = np.asarray(values, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(normalized, rate, rtol=0.0, atol=1e-8))
    if matches.size == 0:
        raise ValueError(f"{field_name}={rate * 100.0:.12g}% is absent from shipped artifact grid")
    return int(matches[0])


__all__ = [
    "BacktestArtifactBackedStageBScorerV2",
    "build_default_artifact_backed_stage_b_scorer_v2",
]
