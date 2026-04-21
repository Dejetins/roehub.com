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
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
    RiskParamsV1,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import TimeRange

from .artifact_runtime_plan_v2 import (
    STAGE_A_LITERAL_V2,
    STAGE_B_LITERAL_V2,
    BacktestArtifactRuntimePlanV2,
)
from .contracts import (
    ArtifactCoordinatesV2,
    ArtifactSlotLiteralV2,
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
    StageACompactExactPayloadV2,
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
_STAGE_B_EXACT_REPLAY_SCOPE_LITERAL_V2 = "finalist-only"


@dataclass(frozen=True, slots=True)
class _PreparedStageABasePayloadV2:
    """
    Cached Stage A artifact-backed payload reused by Stage B scoring and details replay.

    Docs:
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """

    replay: StageBReplayPayloadV2
    metrics: StageBMetricsV2


@dataclass(frozen=True, slots=True)
class _ParallelStageBScorerSnapshotV2:
    """
    Picklable scorer snapshot used to rehydrate readonly Stage B workers under `spawn`.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """

    artifact_loader: BacktestArtifactLoaderV2
    coordinates: ArtifactCoordinatesV2
    artifact_slot: ArtifactSlotLiteralV2
    slot_generation: int
    artifact_asof_date: str
    artifact_manifest_hash: str
    target_time_range: TimeRange
    report_target_slice: slice
    direction_mode: str
    sizing_mode: str
    execution_params: Mapping[str, BacktestVariantScalar]
    market_id: int
    signal_timeframe: str
    prepared_row_plans: tuple["_PreparedIndicatorRowPlanSnapshotV2", ...]
    init_cash_quote_default: float
    fixed_quote_default: float
    safe_profit_percent_default: float
    slippage_pct_default: float
    fee_pct_default_by_market_id: Mapping[int, float]
    close_on_end: bool


@dataclass(frozen=True, slots=True)
class _PreparedIndicatorRowPlanSnapshotV2:
    """
    Picklable row-plan snapshot for spawned Stage B workers using readonly artifacts.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    indicator_id: str
    axis_names: tuple[str, ...]
    axis_radices: tuple[int, ...]
    axis_positions: tuple[
        tuple[str, tuple[tuple[int | float | str, int], ...]],
        ...,
    ]


class BacktestArtifactBackedStageBScorerV2(
    BacktestStagedVariantScorer,
    BacktestStagedVariantMetricScorer,
):
    """
    Artifact-backed scorer using Stage A compact trades and Stage B `1m hit-times` kernels.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
        self._target_time_range = target_time_range
        self._direction_mode = direction_mode
        self._sizing_mode = sizing_mode
        self._market_id = int(market_id)
        self._signal_timeframe = signal_timeframe
        self._report_target_slice = report_target_slice
        self._close_on_end = bool(close_on_end)
        self._indicator_grids = tuple(
            sorted(indicator_grids, key=lambda item: str(item.indicator_id))
        )
        self._indicator_grids_by_id = {
            str(grid.indicator_id): grid
            for grid in self._indicator_grids
        }
        self._init_cash_quote_default = float(init_cash_quote_default)
        self._fixed_quote_default = float(fixed_quote_default)
        self._safe_profit_percent_default = float(safe_profit_percent_default)
        self._slippage_pct_default = float(slippage_pct_default)
        self._fee_pct_default_by_market_id = MappingProxyType(
            dict(
                sorted(
                    (int(key), float(value))
                    for key, value in fee_pct_default_by_market_id.items()
                )
            )
        )
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

    def prepare_for_grid_context(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
    ) -> None:
        """
        Store active staged-run ranking literals so Stage B can enable safe fast-path lookups.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            primary_metric: Active primary ranking metric literal.
        Returns:
            None.
        Assumptions:
            Fast TP/SL search can safely answer Stage B hot-path scoring only when ranking uses
            `total_return_pct`.
        Raises:
            None.
        Side Effects:
            Updates in-memory ranking-context hints for the current scorer instance.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        self._ranking_primary_by_stage[stage] = primary_metric

    def prime_retained_exact_payload(
        self,
        *,
        indicator_variant_key: str,
        signal_params: BacktestSignalParamsMap,
        retained_exact_payload: StageACompactExactPayloadV2,
    ) -> None:
        """
        Seed the retained-candidate compact payload into the existing Stage B scorer cache.

        Args:
            indicator_variant_key: Deterministic compute-only indicators key.
            signal_params: Signal-parameter values for the retained candidate.
            retained_exact_payload: Internal compact exact payload already built by Stage A.
        Returns:
            None.
        Assumptions:
            The retained payload is internal-only and additive; when present it should warm the
            Stage A compact-trade cache without disabling the fast Stage B path for
            `primary_metric=total_return_pct`.
        Raises:
            None.
        Side Effects:
            Populates the existing Stage A base-payload cache for the corresponding base variant.
        Docs:
          - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
        """
        base_variant_key = self._base_variant_key_v2(
            indicator_variant_key=indicator_variant_key,
            signal_params=signal_params,
        )
        if base_variant_key in self._stage_a_payload_cache_by_base_variant_key:
            return
        no_risk_metrics = compute_no_risk_metrics_v2(
            compact_trades=retained_exact_payload.compact_trades,
            exec_open=self._local_exec_open,
            exec_close=self._local_exec_close,
            sentinel_index=self._sentinel_index,
            execution_params=self._execution_params,
            close_on_end=self._close_on_end,
        )
        self._stage_a_payload_cache_by_base_variant_key[base_variant_key] = (
            _PreparedStageABasePayloadV2(
                compact_trades=retained_exact_payload.compact_trades,
                no_risk_metrics=no_risk_metrics,
            )
        )

    def stage_b_exact_replay_count_v2(self) -> int:
        """
        Return the observable Stage B exact replay count for the current scorer run.

        Args:
            None.
        Returns:
            int: Number of unique Stage B variants already replayed exactly in this scorer.
        Assumptions:
            Exact replay results are cached by `variant_key`, so cache cardinality is the stable
            additive `exact_replay_count` contract exposed to perf-smoke and benchmarks.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
          - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
        """
        return len(getattr(self, "_stage_b_exact_cache_by_variant_key", {}))

    def stage_b_exact_replay_scope_v2(self) -> str:
        """
        Return the observable Stage B exact replay scope literal for benchmarks.

        Args:
            None.
        Returns:
            str: Exact replay scope literal, currently `finalist-only`.
        Assumptions:
            Stage B breadth ranking stays on the cheap path for `RG-TTR`, while final authority
            remains exact only for the retained finalist slice.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
          - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
        """
        return _STAGE_B_EXACT_REPLAY_SCOPE_LITERAL_V2

    def to_parallel_stage_b_worker_snapshot_v2(self) -> _ParallelStageBScorerSnapshotV2:
        """
        Build a picklable readonly scorer snapshot for spawned exact-parallel Stage B workers.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

        Args:
            None.
        Returns:
            _ParallelStageBScorerSnapshotV2: Picklable scorer bootstrap payload.
        Assumptions:
            Workers reopen artifact families through the same explicit-path loader contract instead
            of inheriting parent-process mmap state.
        Raises:
            ValueError: If the configured mmap loaders do not expose a shared artifact loader.
        Side Effects:
            None.
        """
        artifact_loader = getattr(self._price_arrays_loader, "artifact_loader", None)
        signal_artifact_loader = getattr(self._signal_matrix_loader, "artifact_loader", None)
        if artifact_loader is None or signal_artifact_loader is None:
            raise ValueError(
                "parallel Stage B requires mmap loaders exposing shared artifact_loader"
            )
        if artifact_loader is not signal_artifact_loader:
            raise ValueError(
                "parallel Stage B requires price and signal loaders to share artifact_loader"
            )
        prepared_row_plans = tuple(
            _prepared_indicator_row_plan_snapshot_v2(
                plan=self._row_plan_for_indicator_v2(indicator_id=indicator_id)
            )
            for indicator_id in sorted(
                {
                    *self._prepared_row_plans_by_indicator.keys(),
                    *self._indicator_grids_by_id.keys(),
                }
            )
        )
        return _ParallelStageBScorerSnapshotV2(
            artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
            coordinates=self._artifact_context.coordinates,
            artifact_slot=self._artifact_context.artifact_slot,
            slot_generation=self._artifact_context.slot_generation,
            artifact_asof_date=self._artifact_context.artifact_asof_date,
            artifact_manifest_hash=self._artifact_context.artifact_manifest_hash,
            target_time_range=self._target_time_range,
            report_target_slice=self._report_target_slice,
            direction_mode=self._direction_mode,
            sizing_mode=self._sizing_mode,
            execution_params=dict(self._execution_params_mapping),
            market_id=self._market_id,
            signal_timeframe=self._signal_timeframe,
            prepared_row_plans=prepared_row_plans,
            init_cash_quote_default=self._init_cash_quote_default,
            fixed_quote_default=self._fixed_quote_default,
            safe_profit_percent_default=self._safe_profit_percent_default,
            slippage_pct_default=self._slippage_pct_default,
            fee_pct_default_by_market_id=dict(self._fee_pct_default_by_market_id),
            close_on_end=self._close_on_end,
        )

    @classmethod
    def from_parallel_stage_b_worker_snapshot_v2(
        cls,
        *,
        snapshot: _ParallelStageBScorerSnapshotV2,
    ) -> BacktestArtifactBackedStageBScorerV2:
        """
        Rehydrate one readonly artifact-backed scorer inside a spawned Stage B worker process.

        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py

        Args:
            snapshot: Picklable scorer bootstrap payload built in the coordinator process.
        Returns:
            BacktestArtifactBackedStageBScorerV2: Rehydrated readonly scorer instance.
        Assumptions:
            Worker processes must rebuild their own mmap-backed loaders under `spawn`.
        Raises:
            ValueError: Propagated from scorer construction on artifact contract drift.
        Side Effects:
            Reopens strict artifact families inside the worker process.
        """
        slot_manifest_path = snapshot.artifact_loader.resolve_slot_manifest_path(
            snapshot.coordinates,
            snapshot.artifact_slot,
        )
        artifact_context = ArtifactSlotPinnedRuntimeContextV2(
            coordinates=snapshot.coordinates,
            artifact_slot=snapshot.artifact_slot,
            slot_generation=snapshot.slot_generation,
            artifact_asof_date=snapshot.artifact_asof_date,
            artifact_manifest_hash=snapshot.artifact_manifest_hash,
            slot_root_path=slot_manifest_path.parent,
            slot_manifest_path=slot_manifest_path,
            slot_manifest=snapshot.artifact_loader.load_slot_manifest(
                snapshot.coordinates,
                snapshot.artifact_slot,
            ),
        )
        scorer = cls(
            price_arrays_loader=MmapPriceArraysLoaderV2(
                artifact_loader=snapshot.artifact_loader
            ),
            signal_matrix_loader=MmapSignalMatrixLoaderV2(
                artifact_loader=snapshot.artifact_loader
            ),
            artifact_context=artifact_context,
            target_time_range=snapshot.target_time_range,
            report_target_slice=snapshot.report_target_slice,
            direction_mode=snapshot.direction_mode,
            sizing_mode=snapshot.sizing_mode,
            execution_params=snapshot.execution_params,
            market_id=snapshot.market_id,
            signal_timeframe=snapshot.signal_timeframe,
            indicator_grids=(),
            init_cash_quote_default=snapshot.init_cash_quote_default,
            fixed_quote_default=snapshot.fixed_quote_default,
            safe_profit_percent_default=snapshot.safe_profit_percent_default,
            slippage_pct_default=snapshot.slippage_pct_default,
            fee_pct_default_by_market_id=snapshot.fee_pct_default_by_market_id,
            close_on_end=snapshot.close_on_end,
        )
        scorer._prepared_row_plans_by_indicator = {
            row_plan.indicator_id: row_plan
            for row_plan in (
                _prepared_indicator_row_plan_from_snapshot_v2(snapshot=row_plan_snapshot)
                for row_plan_snapshot in snapshot.prepared_row_plans
            )
        }
        return scorer

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
            Stage A remains no-risk, while Stage B breadth scoring keeps the fast Stage B path
            enabled when the active ranking plan is exactly `total_return_pct DESC`, including
            cases where `retained_exact_payload` is already cached for finalist authority.
        Raises:
            ValueError: If stage, artifact row addressing, or exact replay contracts are invalid.
        Side Effects:
            Populates per-run caches for Stage A compact trades, fast search, and exact replay.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
        """
        _ = candles
        if stage == STAGE_A_LITERAL_V2:
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
        if stage != STAGE_B_LITERAL_V2:
            raise ValueError(f"unsupported stage literal for artifact-backed scorer: {stage!r}")

        base_variant_key = self._base_variant_key_v2(
            indicator_variant_key=indicator_variant_key,
            signal_params=signal_params,
        )
        tp_index, sl_index = self._resolve_risk_level_indexes_v2(risk_params=risk_params)
        if (
            self._can_use_stage_b_total_return_fast_path_v2()
            and not self._should_force_exact_stage_b_v2()
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
        """
        _ = candles
        if stage != STAGE_B_LITERAL_V2:
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
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
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/
            artifact_backed_stage_b_scorer_v2.py
        """
        return (
            self._ranking_primary_by_stage.get(STAGE_B_LITERAL_V2) == "total_return_pct"
        )

    def _should_force_exact_stage_b_v2(self) -> bool:
        """
        Check whether one Stage B base variant must bypass fast-path scoring and replay exactly.

        Args:
            None.
        Returns:
            bool: `False`, because retained exact payload must not disable breadth fast-path use.
        Assumptions:
            Exact Stage B authority remains available through the explicit details/finalist path,
            while `total_return_pct` breadth ranking must keep the fast Stage B path enabled even
            when `retained_exact_payload` is present.
        Raises:
            None.
        Side Effects:
            None.
        """
        return False


def _prepared_indicator_row_plan_snapshot_v2(
    *,
    plan: PreparedIndicatorRowPlanV2,
) -> _PreparedIndicatorRowPlanSnapshotV2:
    """
    Convert one prepared row plan into a fully picklable spawned-worker snapshot.

    Args:
        plan: In-process prepared row-addressing plan.
    Returns:
        _PreparedIndicatorRowPlanSnapshotV2: Picklable row-plan snapshot.
    Assumptions:
        Snapshot ordering must remain stable, so axis-position mappings are serialized by sorted
        keys for deterministic `spawn` payloads.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    return _PreparedIndicatorRowPlanSnapshotV2(
        indicator_id=plan.indicator_id,
        axis_names=plan.axis_names,
        axis_radices=plan.axis_radices,
        axis_positions=tuple(
            (
                axis_name,
                tuple(
                    sorted(
                        (value, int(position))
                        for value, position in axis_lookup.items()
                    )
                ),
            )
            for axis_name, axis_lookup in sorted(plan.axis_positions.items())
        ),
    )


def _prepared_indicator_row_plan_from_snapshot_v2(
    *,
    snapshot: _PreparedIndicatorRowPlanSnapshotV2,
) -> PreparedIndicatorRowPlanV2:
    """
    Rebuild one prepared row plan from a picklable spawned-worker snapshot.

    Args:
        snapshot: Picklable row-plan snapshot built in the coordinator process.
    Returns:
        PreparedIndicatorRowPlanV2: Rehydrated prepared row-addressing plan.
    Assumptions:
        Worker processes only need readonly row-addressing metadata and may restore mapping
        proxies locally after deserialization.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    return PreparedIndicatorRowPlanV2(
        indicator_id=snapshot.indicator_id,
        axis_names=snapshot.axis_names,
        axis_radices=snapshot.axis_radices,
        axis_positions=MappingProxyType(
            {
                axis_name: MappingProxyType(dict(axis_lookup))
                for axis_name, axis_lookup in snapshot.axis_positions
            }
        ),
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
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
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py
    """
    normalized = np.asarray(values, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(normalized, rate, rtol=0.0, atol=1e-8))
    if matches.size == 0:
        raise ValueError(
            f"{field_name}={rate * 100.0:.12g}% is absent from shipped artifact grid "
            f"(levels={normalized.shape[0]})"
        )
    return int(matches[0])


__all__ = [
    "BacktestArtifactBackedStageBScorerV2",
    "build_default_artifact_backed_stage_b_scorer_v2",
]
