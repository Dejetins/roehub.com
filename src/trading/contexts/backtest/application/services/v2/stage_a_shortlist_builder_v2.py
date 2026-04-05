"""Artifact-backed Stage A shortlist builder using pure aggregation and no-risk kernels."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from trading.contexts.backtest.application.dto import BacktestRankingConfig
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import TimeRange

from .artifact_runtime_core_v2 import (
    BacktestStageAScoredVariantV2,
    ResolvedRankingPlanV2,
    StageAHeapEntryV2,
    effective_ranking_config_v2,
    heap_entry_outranks_v2,
    resolve_ranking_plan_v2,
    stage_a_heap_entry_v2,
    stage_a_rows_from_heap_v2,
)
from .artifact_runtime_plan_v2 import (
    STAGE_A_LITERAL_V2,
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorPlanV2,
    BacktestSignalFeaturesAccessPlanV2,
    BacktestStageABaseVariantV2,
)
from .contracts import (
    ArtifactSignalFeaturesRowsV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactSlotResolverV2,
    BacktestPriceArraysLoaderV2,
    BacktestSignalFeaturesLoaderV2,
    BacktestSignalMatrixLoaderV2,
    artifact_market_id_from_coordinates_v2,
)
from .price_arrays_loader import MmapPriceArraysLoaderV2
from .signal_aggregator_kernel import aggregate_final_signal_rows_v2
from .signal_features_loader_v2 import MmapSignalFeaturesLoaderV2
from .signal_matrix_loader import MmapSignalMatrixLoaderV2
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

StageACancelCheckerV2 = Callable[[str], None]
StageACheckpointCallbackV2 = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class PreparedIndicatorRowPlanV2:
    """
    Pre-resolved indicator row-addressing plan for artifact-backed Stage A signal subsets.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """

    indicator_id: str
    axis_names: tuple[str, ...]
    axis_radices: tuple[int, ...]
    axis_positions: Mapping[str, Mapping[int | float | str, int]]

    @classmethod
    def from_indicator_plan(
        cls,
        *,
        plan: BacktestIndicatorPlanV2,
    ) -> PreparedIndicatorRowPlanV2:
        """
        Build a deterministic row-addressing plan from one grid-builder indicator plan.

        Args:
            plan: Grid-builder indicator plan whose axis ordering defines artifact row ordering.
        Returns:
            PreparedIndicatorRowPlanV2: Prepared row-addressing metadata for one indicator.
        Assumptions:
            Artifact signal rows preserve the same mixed-radix ordering as Stage A compute plans.
        Raises:
            ValueError: If one indicator axis contains duplicate normalized values.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/grid_builder_v1.py
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
        """
        axis_names = tuple(axis.name for axis in plan.axes)
        axis_radices = tuple(len(axis.values) for axis in plan.axes)
        axis_positions: dict[str, Mapping[int | float | str, int]] = {}
        for axis in plan.axes:
            positions: dict[int | float | str, int] = {}
            for index, raw_value in enumerate(axis.values):
                normalized_value = _normalize_indicator_scalar_v2(value=raw_value)
                if normalized_value in positions:
                    raise ValueError(
                        "indicator axis values must be unique for artifact row addressing; "
                        f"{plan.indicator_id}.{axis.name} duplicates {normalized_value!r}"
                    )
                positions[normalized_value] = index
            axis_positions[axis.name] = MappingProxyType(positions)
        return cls(
            indicator_id=plan.indicator_id,
            axis_names=axis_names,
            axis_radices=axis_radices,
            axis_positions=MappingProxyType(axis_positions),
        )

    def row_index_for_selection(
        self,
        *,
        selection: IndicatorVariantSelection,
    ) -> int:
        """
        Resolve the flattened artifact row index for one explicit indicator selection.

        Args:
            selection: Explicit indicator selection from one Stage A base variant.
        Returns:
            int: Deterministic flattened row index in the artifact signal matrix.
        Assumptions:
            Selection payload already matches the indicator id owned by this row plan.
        Raises:
            ValueError: If the selection misses one axis or contains a value outside the plan.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/grid_builder_v1.py
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
        """
        if selection.indicator_id != self.indicator_id:
            raise ValueError(
                "indicator selection id must match artifact row plan indicator_id; got "
                f"{selection.indicator_id!r}, expected {self.indicator_id!r}"
            )
        coordinates: list[int] = []
        for axis_name in self.axis_names:
            if axis_name == "source":
                raw_value = selection.inputs.get("source")
            elif axis_name in selection.params:
                raw_value = selection.params[axis_name]
            else:
                raw_value = selection.inputs.get(axis_name)
            if raw_value is None:
                raise ValueError(
                    f"indicator selection is missing axis value '{axis_name}' for "
                    f"{self.indicator_id!r}"
                )
            normalized_value = _normalize_indicator_scalar_v2(value=raw_value)
            axis_lookup = self.axis_positions.get(axis_name)
            if axis_lookup is None or normalized_value not in axis_lookup:
                raise ValueError(
                    "indicator selection axis value is outside artifact row plan: "
                    f"{self.indicator_id}.{axis_name}={normalized_value!r}"
                )
            coordinates.append(int(axis_lookup[normalized_value]))
        return _encode_mixed_radix_v2(
            coordinates=tuple(coordinates),
            radices=self.axis_radices,
        )


@dataclass(frozen=True, slots=True)
class PreparedIndicatorChunkInputsV2:
    """
    Per-indicator Stage A chunk inputs carrying signal rows and optional warm-cache feature access.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    indicator_id: str
    signal_rows: np.ndarray
    signal_features_loader: BacktestSignalFeaturesLoaderV2 | None = None
    signal_features_context: ArtifactSlotPinnedRuntimeContextV2 | None = None
    signal_features_access: BacktestSignalFeaturesAccessPlanV2 | None = None
    signal_feature_row_selection: slice | tuple[int, ...] | None = None

    def load_signal_feature_rows(self) -> ArtifactSignalFeaturesRowsV2 | None:
        """
        Materialize optional selected signal-feature rows for this chunk in variant order.

        Args:
            None.
        Returns:
            ArtifactSignalFeaturesRowsV2 | None: Selected feature rows when the additive
                `signal_features` family is available for this indicator, else `None`.
        Assumptions:
            `signal_feature_row_selection` stays aligned with `signal_rows` ordering for the same
            chunk variants, and feature matrices should stay lazy until this method is called.
        Raises:
            ValueError: If lazy feature-access metadata is only partially populated.
        Side Effects:
            May memory-map one additive feature matrix on first explicit access and returns the
            selected typed rows.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if (
            self.signal_features_loader is None
            and self.signal_features_context is None
            and self.signal_features_access is None
            and self.signal_feature_row_selection is None
        ):
            return None
        if (
            self.signal_features_loader is None
            or self.signal_features_context is None
            or self.signal_features_access is None
            or self.signal_feature_row_selection is None
        ):
            raise ValueError(
                "PreparedIndicatorChunkInputsV2 requires complete lazy signal-feature access "
                "metadata when warm-cache access is enabled"
            )
        if self.signal_features_access.optional:
            return self.signal_features_loader.try_load_signal_feature_rows(
                context=self.signal_features_context,
                timeframe=self.signal_features_access.timeframe,
                indicator_id=self.signal_features_access.indicator_id,
                row_selection=self.signal_feature_row_selection,
            )
        return self.signal_features_loader.load_signal_feature_rows(
            context=self.signal_features_context,
            timeframe=self.signal_features_access.timeframe,
            indicator_id=self.signal_features_access.indicator_id,
            row_selection=self.signal_feature_row_selection,
        )


@dataclass(frozen=True, slots=True)
class BacktestStageAShortlistBuilderV2:
    """
    Build deterministic Stage A shortlist rows from artifacts-only inputs and pure kernels.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    price_arrays_loader: BacktestPriceArraysLoaderV2
    signal_matrix_loader: BacktestSignalMatrixLoaderV2
    signal_features_loader: BacktestSignalFeaturesLoaderV2 | None = None
    configurable_ranking_enabled: bool = True
    chunk_size_default: int = 2048
    init_cash_quote_default: float = 10000.0
    fixed_quote_default: float = 100.0
    safe_profit_percent_default: float = 30.0
    slippage_pct_default: float = 0.01
    fee_pct_default_by_market_id: Mapping[int, float] = _DEFAULT_FEE_PCT_BY_MARKET_ID_V2

    def __post_init__(self) -> None:
        """
        Validate constructor dependencies and freeze deterministic runtime defaults.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Constructor wires collaborators only and does not touch artifact files.
        Raises:
            ValueError: If one dependency is missing or one scalar default is invalid.
        Side Effects:
            Freezes fee defaults into an immutable sorted mapping proxy.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
        """
        if self.price_arrays_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStageAShortlistBuilderV2 requires price_arrays_loader")
        if self.signal_matrix_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStageAShortlistBuilderV2 requires signal_matrix_loader")
        if self.chunk_size_default <= 0:
            raise ValueError("BacktestStageAShortlistBuilderV2.chunk_size_default must be > 0")
        if self.init_cash_quote_default <= 0.0:
            raise ValueError(
                "BacktestStageAShortlistBuilderV2.init_cash_quote_default must be > 0"
            )
        if self.fixed_quote_default <= 0.0:
            raise ValueError("BacktestStageAShortlistBuilderV2.fixed_quote_default must be > 0")
        if self.safe_profit_percent_default < 0.0 or self.safe_profit_percent_default > 100.0:
            raise ValueError(
                "BacktestStageAShortlistBuilderV2.safe_profit_percent_default must be in [0, 100]"
            )
        if self.slippage_pct_default < 0.0:
            raise ValueError(
                "BacktestStageAShortlistBuilderV2.slippage_pct_default must be >= 0"
            )
        normalized_fee_defaults: dict[int, float] = {}
        for market_id in sorted(self.fee_pct_default_by_market_id.keys()):
            fee_pct = self.fee_pct_default_by_market_id[market_id]
            if fee_pct < 0.0:
                raise ValueError("fee_pct_default_by_market_id values must be >= 0")
            normalized_fee_defaults[int(market_id)] = float(fee_pct)
        object.__setattr__(
            self,
            "fee_pct_default_by_market_id",
            MappingProxyType(normalized_fee_defaults),
        )

    def build_shortlist(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        target_time_range: TimeRange,
        shortlist_limit: int,
        ranking: BacktestRankingConfig | None = None,
        batch_size: int | None = None,
        cancel_checker: StageACancelCheckerV2 | None = None,
        on_checkpoint: StageACheckpointCallbackV2 | None = None,
    ) -> tuple[BacktestStageAScoredVariantV2, ...]:
        """
        Build a deterministic Stage A shortlist from artifacts-only inputs with chunked variants.

        Args:
            grid_context: Prepared Stage A grid context with deterministic variant ordering.
            artifact_context: Slot-pinned runtime context resolved once at startup.
            target_time_range: Requested trading window used for local signal/exec rebasing.
            shortlist_limit: Maximum number of retained Stage A rows.
            ranking: Optional Stage A ranking config.
            batch_size: Optional chunk override for `chunked variant processing`.
            cancel_checker: Optional cooperative cancellation callback by stage literal.
            on_checkpoint: Optional progress callback `(processed, total)` after each chunk.
        Returns:
            tuple[BacktestStageAScoredVariantV2, ...]: Deterministically ranked Stage A shortlist.
        Assumptions:
            Runtime uses artifacts-only inputs, reuses subset signal row loading, and keeps
            Stage B risk kernels out of scope.
        Raises:
            ValueError: If limits, row addressing, or artifact-local slice contracts are invalid.
        Side Effects:
            Reads pinned artifact arrays and signal row subsets through the injected loaders.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
          - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
        """
        if shortlist_limit <= 0:
            raise ValueError("BacktestStageAShortlistBuilderV2.shortlist_limit must be > 0")
        effective_batch_size = self._resolve_batch_size(batch_size=batch_size)
        ranking_plan = resolve_ranking_plan_v2(
            ranking=effective_ranking_config_v2(
                ranking=ranking,
                configurable_ranking_enabled=self.configurable_ranking_enabled,
            )
        )
        if cancel_checker is not None:
            cancel_checker(STAGE_A_LITERAL_V2)

        signal_prices = self.price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe=grid_context.timeframe_code,
        )
        mapping_arrays = self.price_arrays_loader.load_mapping_arrays(
            context=artifact_context,
            timeframe=grid_context.timeframe_code,
        )
        execution_prices = self.price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe="1m",
        )
        signal_target_slice = compute_target_slice_by_close_time_v2(
            close_time=signal_prices.close_time,
            target_time_range=target_time_range,
        )
        exec_target_slice = compute_target_slice_by_close_time_v2(
            close_time=execution_prices.close_time,
            target_time_range=target_time_range,
        )
        local_bar_close_1m_idx = rebase_bar_close_mapping_v2(
            mapping_values=mapping_arrays.bar_close_1m_idx[signal_target_slice],
            exec_target_slice=exec_target_slice,
        )
        local_exec_open = np.asarray(
            execution_prices.ohlcv[exec_target_slice, 0],
            dtype=np.float64,
        )
        local_exec_close = np.asarray(
            execution_prices.ohlcv[exec_target_slice, 3],
            dtype=np.float64,
        )
        sentinel_index = int(local_exec_open.shape[0])
        execution_params = self._resolve_execution_params(
            grid_context=grid_context,
            market_id=artifact_market_id_from_coordinates_v2(artifact_context.coordinates),
        )
        row_plans = tuple(
            PreparedIndicatorRowPlanV2.from_indicator_plan(plan=plan)
            for plan in grid_context.indicator_plans
        )

        shortlist_heap: list[StageAHeapEntryV2] = []
        chunk_variants: list[BacktestStageABaseVariantV2] = []
        total = int(grid_context.stage_a_variants_total)
        processed = 0

        for base_variant in grid_context.iter_stage_a_variants():
            chunk_variants.append(base_variant)
            if (
                len(chunk_variants) < effective_batch_size
                and (processed + len(chunk_variants)) < total
            ):
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL_V2)
            self._score_chunk_into_heap(
                row_plans=row_plans,
                chunk_variants=chunk_variants,
                grid_context=grid_context,
                artifact_context=artifact_context,
                signal_target_slice=signal_target_slice,
                local_bar_close_1m_idx=local_bar_close_1m_idx,
                sentinel_index=sentinel_index,
                local_exec_open=local_exec_open,
                local_exec_close=local_exec_close,
                execution_params=execution_params,
                ranking_plan=ranking_plan,
                shortlist_limit=shortlist_limit,
                shortlist_heap=shortlist_heap,
            )
            processed += len(chunk_variants)
            if on_checkpoint is not None:
                on_checkpoint(processed, total)
            chunk_variants.clear()

        return stage_a_rows_from_heap_v2(heap=shortlist_heap)

    def load_chunk_runtime_inputs(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
    ) -> tuple[PreparedIndicatorChunkInputsV2, ...]:
        """
        Load one Stage A chunk's per-indicator runtime inputs with optional feature warm cache.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            chunk_variants: Deterministic Stage A base variants for the current chunk.
            grid_context: Prepared runtime plan owning timeframe and warm-cache access plans.
            artifact_context: Slot-pinned runtime context used by explicit-path loaders.
            signal_target_slice: Target request slice in the signal timeline.
        Returns:
            tuple[PreparedIndicatorChunkInputsV2, ...]: Per-indicator signal rows plus additive
                optional `signal_features` access handles for the current chunk.
        Assumptions:
            Warm-cache access stays optional in Milestone C and must not fail exact execution on
            legacy slots that omit additive feature artifacts.
        Raises:
            ValueError: If row addressing drifts from the prepared indicator plans.
            FileNotFoundError: If selected signal rows are missing on disk.
        Side Effects:
            Reads selected signal rows only; additive feature matrices remain lazy until one
            consumer explicitly calls `PreparedIndicatorChunkInputsV2.load_signal_feature_rows()`.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
          - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
        """
        prepared_inputs: list[PreparedIndicatorChunkInputsV2] = []
        for plan_position, row_plan in enumerate(row_plans):
            row_indexes = tuple(
                int(value)
                for value in np.fromiter(
                    (
                        row_plan.row_index_for_selection(
                            selection=_indicator_selection_for_plan_v2(
                                base_variant=base_variant,
                                indicator_position=plan_position,
                                indicator_id=row_plan.indicator_id,
                            )
                        )
                        for base_variant in chunk_variants
                    ),
                    dtype=np.int64,
                    count=len(chunk_variants),
                ).tolist()
            )
            signal_rows = _load_chunk_signal_rows_v2(
                signal_matrix_loader=self.signal_matrix_loader,
                artifact_context=artifact_context,
                timeframe=grid_context.timeframe_code,
                indicator_id=row_plan.indicator_id,
                indicator_row_indexes=np.asarray(row_indexes, dtype=np.int64),
                signal_target_slice=signal_target_slice,
            )
            signal_features_access = _signal_features_access_plan_for_indicator_v2(
                grid_context=grid_context,
                indicator_id=row_plan.indicator_id,
            )
            prepared_inputs.append(
                PreparedIndicatorChunkInputsV2(
                    indicator_id=row_plan.indicator_id,
                    signal_rows=signal_rows,
                    signal_features_loader=(
                        self.signal_features_loader
                        if (
                            signal_features_access is not None
                            and self.signal_features_loader is not None
                        )
                        else None
                    ),
                    signal_features_context=(
                        artifact_context
                        if (
                            signal_features_access is not None
                            and self.signal_features_loader is not None
                        )
                        else None
                    ),
                    signal_features_access=signal_features_access,
                    signal_feature_row_selection=(
                        row_indexes
                        if (
                            signal_features_access is not None
                            and self.signal_features_loader is not None
                        )
                        else None
                    ),
                )
            )
        return tuple(prepared_inputs)

    def _resolve_batch_size(
        self,
        *,
        batch_size: int | None,
    ) -> int:
        """
        Resolve explicit or default chunk size used for chunked Stage A processing.

        Args:
            batch_size: Optional caller override.
        Returns:
            int: Positive chunk size.
        Assumptions:
            Chunk boundaries double as deterministic checkpoint boundaries.
        Raises:
            ValueError: If the override is non-positive.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
        """
        if batch_size is None:
            return self.chunk_size_default
        if batch_size <= 0:
            raise ValueError("BacktestStageAShortlistBuilderV2 batch_size must be > 0")
        return batch_size

    def _resolve_execution_params(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
        market_id: int,
    ) -> ExecutionParamsV1:
        """
        Resolve immutable execution settings for no-risk Stage A metric evaluation.

        Args:
            grid_context: Stage A grid context owning direction/sizing/execution mappings.
            market_id: Numeric market id used for fee fallback lookup.
        Returns:
            ExecutionParamsV1: Validated immutable execution settings.
        Assumptions:
            Missing overrides fall back to runtime defaults shared with legacy scorer wiring.
        Raises:
            KeyError: If fee defaults are missing for the requested market id.
            ValueError: If one execution scalar is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
          - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
        """
        execution_values = grid_context.execution_params
        return ExecutionParamsV1(
            direction_mode=grid_context.direction_mode,
            sizing_mode=grid_context.sizing_mode,
            init_cash_quote=_resolve_number_v2(
                values=execution_values,
                primary_key="init_cash_quote",
                secondary_key="init_cash",
                default=self.init_cash_quote_default,
            ),
            fixed_quote=_resolve_number_v2(
                values=execution_values,
                primary_key="fixed_quote",
                secondary_key="",
                default=self.fixed_quote_default,
            ),
            safe_profit_percent=_resolve_number_v2(
                values=execution_values,
                primary_key="safe_profit_percent",
                secondary_key="",
                default=self.safe_profit_percent_default,
            ),
            fee_pct=_resolve_number_v2(
                values=execution_values,
                primary_key="fee_pct",
                secondary_key="market_fee_pct",
                default=self.fee_pct_default_by_market_id[market_id],
            ),
            slippage_pct=_resolve_number_v2(
                values=execution_values,
                primary_key="slippage_pct",
                secondary_key="",
                default=self.slippage_pct_default,
            ),
        )

    def _score_chunk_into_heap(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
        local_bar_close_1m_idx: np.ndarray,
        sentinel_index: int,
        local_exec_open: np.ndarray,
        local_exec_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        ranking_plan: ResolvedRankingPlanV2,
        shortlist_limit: int,
        shortlist_heap: list[StageAHeapEntryV2],
    ) -> None:
        """
        Score one Stage A chunk and merge ranked rows into the bounded shortlist heap.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            chunk_variants: Deterministic chunk of Stage A base variants.
            grid_context: Stage A grid context with indicator/timeframe metadata.
            artifact_context: Slot-pinned runtime context for loader calls.
            signal_target_slice: Target request slice in the signal timeline.
            local_bar_close_1m_idx: Rebases `bar_close_1m_idx` for the local execution window.
            sentinel_index: Local execution sentinel index.
            local_exec_open: Local execution-bar open prices.
            local_exec_close: Local execution-bar close prices.
            execution_params: Immutable no-risk execution settings.
            ranking_plan: Pre-resolved staged ranking plan from shared Stage A machinery.
            shortlist_limit: Maximum retained shortlist size.
            shortlist_heap: Mutable bounded shortlist heap updated in place.
        Returns:
            None.
        Assumptions:
            Chunk order already matches deterministic Stage A enumeration order.
        Raises:
            ValueError: If chunk variants drift from indicator plans or row addressing fails.
        Side Effects:
            Mutates `shortlist_heap` in place.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
        """
        chunk_inputs = self.load_chunk_runtime_inputs(
            row_plans=row_plans,
            chunk_variants=chunk_variants,
            grid_context=grid_context,
            artifact_context=artifact_context,
            signal_target_slice=signal_target_slice,
        )
        selected_signal_rows = {
            prepared_input.indicator_id: prepared_input.signal_rows
            for prepared_input in chunk_inputs
        }

        final_signal = aggregate_final_signal_rows_v2(selected_signal_rows=selected_signal_rows)
        compact_trades_by_variant = build_compact_trade_list_v2(
            final_signal=final_signal,
            bar_close_1m_idx=local_bar_close_1m_idx,
            sentinel_index=sentinel_index,
            direction_mode=grid_context.direction_mode,
        )
        for base_variant, compact_trades in zip(chunk_variants, compact_trades_by_variant):
            metrics = compute_no_risk_metrics_v2(
                compact_trades=compact_trades,
                exec_open=local_exec_open,
                exec_close=local_exec_close,
                sentinel_index=sentinel_index,
                execution_params=execution_params,
            )
            row = BacktestStageAScoredVariantV2(
                base_variant=base_variant,
                total_return_pct=metrics.total_return_pct,
            )
            heap_entry = stage_a_heap_entry_v2(
                row=row,
                metrics=no_risk_metrics_to_ranking_payload_v2(metrics=metrics),
                ranking_plan=ranking_plan,
            )
            if len(shortlist_heap) < shortlist_limit:
                heappush(shortlist_heap, heap_entry)
            elif heap_entry_outranks_v2(candidate=heap_entry, baseline=shortlist_heap[0]):
                heapreplace(shortlist_heap, heap_entry)

def build_default_stage_a_shortlist_builder_v2(
    *,
    artifact_slot_resolver: BacktestArtifactSlotResolverV2 | None,
    configurable_ranking_enabled: bool,
    init_cash_quote_default: float,
    fixed_quote_default: float,
    safe_profit_percent_default: float,
    slippage_pct_default: float,
    fee_pct_default_by_market_id: Mapping[int, float] | None,
) -> BacktestStageAShortlistBuilderV2 | None:
    """
    Build the default artifact-backed Stage A shortlist builder from resolver wiring when possible.

    Args:
        artifact_slot_resolver: Optional slot resolver already wired by runtime startup.
        configurable_ranking_enabled: Feature flag for configurable ranking behavior.
        init_cash_quote_default: Runtime default initial strategy quote balance.
        fixed_quote_default: Runtime default fixed quote notional.
        safe_profit_percent_default: Runtime default profit-lock percent.
        slippage_pct_default: Runtime default slippage percent.
        fee_pct_default_by_market_id: Runtime default fee mapping by market id.
    Returns:
        BacktestStageAShortlistBuilderV2 | None: Default builder when the resolver exposes an
            artifact loader, otherwise `None`.
    Assumptions:
        This helper keeps Stage A cutover additive and leaves legacy flows untouched when v2
        runtime wiring is unavailable.
    Raises:
        ValueError: Propagated from builder constructor when one default is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """
    if artifact_slot_resolver is None:
        return None
    artifact_loader = getattr(artifact_slot_resolver, "artifact_loader", None)
    if artifact_loader is None:
        return None
    typed_artifact_loader = cast(BacktestArtifactLoaderV2, artifact_loader)
    return BacktestStageAShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=typed_artifact_loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(artifact_loader=typed_artifact_loader),
        signal_features_loader=MmapSignalFeaturesLoaderV2(
            artifact_loader=typed_artifact_loader
        ),
        configurable_ranking_enabled=configurable_ranking_enabled,
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


def build_prepared_indicator_row_plan_from_grid_spec_v2(
    *,
    indicator_id: str,
    grid_spec: GridSpec,
) -> PreparedIndicatorRowPlanV2:
    """
    Build artifact row-addressing metadata directly from one explicit indicator grid spec.

    Args:
        indicator_id: Canonical indicator id expected in the signal artifact tree.
        grid_spec: Grid spec whose source/params ordering defines artifact row ordering.
    Returns:
        PreparedIndicatorRowPlanV2: Deterministic mixed-radix row-addressing plan.
    Assumptions:
        Artifact signal rows keep Stage A ordering semantics: optional `source` axis first, then
        sorted parameter axes.
    Raises:
        ValueError: If the grid spec indicator id drifts from `indicator_id`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    grid_indicator_id = str(grid_spec.indicator_id)
    if grid_indicator_id != indicator_id:
        raise ValueError(
            "grid spec indicator id must match requested indicator_id; got "
            f"{grid_indicator_id!r}, expected {indicator_id!r}"
        )

    axis_names: list[str] = []
    axis_values_by_name: dict[str, tuple[int | float | str, ...]] = {}
    if grid_spec.source is not None:
        axis_names.append("source")
        axis_values_by_name["source"] = tuple(
            _normalize_indicator_scalar_v2(value=value)
            for value in grid_spec.source.materialize()
        )
    for param_name in sorted(grid_spec.params.keys()):
        axis_names.append(param_name)
        axis_values_by_name[param_name] = tuple(
            _normalize_indicator_scalar_v2(value=value)
            for value in grid_spec.params[param_name].materialize()
        )

    axis_radices = tuple(len(axis_values_by_name[axis_name]) for axis_name in axis_names)
    axis_positions: dict[str, Mapping[int | float | str, int]] = {}
    for axis_name in axis_names:
        positions: dict[int | float | str, int] = {}
        for index, normalized_value in enumerate(axis_values_by_name[axis_name]):
            if normalized_value in positions:
                raise ValueError(
                    "indicator axis values must be unique for artifact row addressing; "
                    f"{indicator_id}.{axis_name} duplicates {normalized_value!r}"
                )
            positions[normalized_value] = index
        axis_positions[axis_name] = MappingProxyType(positions)
    return PreparedIndicatorRowPlanV2(
        indicator_id=indicator_id,
        axis_names=tuple(axis_names),
        axis_radices=axis_radices,
        axis_positions=MappingProxyType(axis_positions),
    )


def compute_target_slice_by_close_time_v2(
    *,
    close_time: np.ndarray,
    target_time_range: TimeRange,
) -> slice:
    """
    Compute the half-open target slice using artifact `close_time` arrays directly.

    Args:
        close_time: Monotone close timestamps for one artifact family.
        target_time_range: Requested trading window `[Start, End)`.
    Returns:
        slice: Half-open target slice satisfying `Start <= close_time < End`.
    Assumptions:
        Artifact close-time arrays are already validated for monotonicity by the mmap loaders.
    Raises:
        ValueError: If the close-time array is not one-dimensional.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """
    normalized = np.asarray(close_time, dtype=np.int64)
    if normalized.ndim != 1:
        raise ValueError("artifact close_time must be a 1D array")
    if normalized.shape[0] == 0:
        return slice(0, 0)
    start_ms = _utc_timestamp_to_epoch_millis_v2(target_time_range.start.value)
    end_ms = _utc_timestamp_to_epoch_millis_v2(target_time_range.end.value)
    slice_start = int(np.searchsorted(normalized, np.int64(start_ms), side="left"))
    slice_stop = int(np.searchsorted(normalized, np.int64(end_ms), side="left"))
    if slice_stop < slice_start:
        slice_stop = slice_start
    return slice(slice_start, slice_stop)


def rebase_bar_close_mapping_v2(
    *,
    mapping_values: np.ndarray,
    exec_target_slice: slice,
) -> np.ndarray:
    """
    Rebase absolute `bar_close_1m_idx` values to the local execution timeline window.

    Args:
        mapping_values: Absolute `bar_close_1m_idx` values for the selected signal bars.
        exec_target_slice: Local execution slice used by Stage A no-risk kernels.
    Returns:
        np.ndarray: Rebases close indexes to local execution coordinates.
    Assumptions:
        Selected signal bars already satisfy the same `[Start, End)` close-time contract as the
        local execution window.
    Raises:
        ValueError: If rebased indexes fall outside the local execution slice.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
    """
    normalized = np.asarray(mapping_values, dtype=np.int64)
    rebased = normalized - int(exec_target_slice.start or 0)
    local_exec_length = int((exec_target_slice.stop or 0) - (exec_target_slice.start or 0))
    if bool(np.any((rebased < 0) | (rebased >= local_exec_length))):
        raise ValueError(
            "bar_close_1m_idx values must stay inside the local execution window after rebasing"
        )
    return rebased


def _load_chunk_signal_rows_v2(
    *,
    signal_matrix_loader: BacktestSignalMatrixLoaderV2,
    artifact_context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
    indicator_id: str,
    indicator_row_indexes: np.ndarray,
    signal_target_slice: slice,
) -> np.ndarray:
    """
    Load one chunk of selected signal rows with exact-preserving locality-oriented fast paths.

    Args:
        signal_matrix_loader: Strict runtime signal loader.
        artifact_context: Slot-pinned runtime context resolved once at startup.
        timeframe: Canonical request timeframe literal.
        indicator_id: Canonical indicator identifier.
        indicator_row_indexes: Deterministic row indexes for the current Stage A chunk.
        signal_target_slice: Local signal-timeline window selected for the run.
    Returns:
        np.ndarray: Chunk-aligned `int8` signal rows preserving the caller's deterministic order.
    Assumptions:
        When every variant in the chunk references the same artifact row, exact semantics allow
        broadcasting one mmap-backed slice instead of reindexing repeated copies.
    Raises:
        ValueError: If the chunk is empty or the loader rejects the explicit row selection.
    Side Effects:
        Reads one deterministic signal-row subset from the pinned artifact store.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """
    chunk_rows = int(indicator_row_indexes.shape[0])
    if chunk_rows <= 0:
        raise ValueError("Stage A signal chunk must contain at least one row index")
    if bool(np.all(indicator_row_indexes == indicator_row_indexes[0])):
        loaded_rows = signal_matrix_loader.load_signal_rows(
            context=artifact_context,
            timeframe=timeframe,
            indicator_id=indicator_id,
            row_selection=(int(indicator_row_indexes[0]),),
        )
        windowed_rows = np.asarray(loaded_rows[:, signal_target_slice], dtype=np.int8)
        if chunk_rows == 1:
            return windowed_rows
        return np.broadcast_to(windowed_rows, (chunk_rows, windowed_rows.shape[1]))

    unique_row_indexes, inverse_indexes = np.unique(
        indicator_row_indexes,
        return_inverse=True,
    )
    loaded_rows = signal_matrix_loader.load_signal_rows(
        context=artifact_context,
        timeframe=timeframe,
        indicator_id=indicator_id,
        row_selection=tuple(int(index) for index in unique_row_indexes),
    )
    windowed_rows = loaded_rows[:, signal_target_slice]
    return np.asarray(windowed_rows[inverse_indexes, :], dtype=np.int8)


def _indicator_selection_for_plan_v2(
    *,
    base_variant: BacktestStageABaseVariantV2,
    indicator_position: int,
    indicator_id: str,
) -> IndicatorVariantSelection:
    """
    Resolve the deterministic indicator selection expected at one plan position.

    Args:
        base_variant: Stage A base variant currently being scored.
        indicator_position: Position of the indicator plan in `grid_context.indicator_plans`.
        indicator_id: Expected indicator id for that position.
    Returns:
        IndicatorVariantSelection: Matching selection from the base variant.
    Assumptions:
        Grid builder emits indicator selections in the same order as indicator plans.
    Raises:
        ValueError: If the selection tuple length or indicator id drifts from the grid contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    if indicator_position >= len(base_variant.indicator_selections):
        raise ValueError("base variant indicator selections must align with indicator plans")
    selection = base_variant.indicator_selections[indicator_position]
    if selection.indicator_id != indicator_id:
        raise ValueError(
            "base variant indicator selection order drifted from indicator plans; got "
            f"{selection.indicator_id!r}, expected {indicator_id!r}"
        )
    return selection


def _signal_features_access_plan_for_indicator_v2(
    *,
    grid_context: BacktestArtifactRuntimePlanV2,
    indicator_id: str,
) -> BacktestSignalFeaturesAccessPlanV2 | None:
    """
    Resolve one optional warm-cache access entry from the runtime plan or plan-like test fixture.

    Args:
        grid_context: Runtime plan or compatible fixture carrying optional warm-cache metadata.
        indicator_id: Canonical indicator identifier for the requested chunk input.
    Returns:
        BacktestSignalFeaturesAccessPlanV2 | None: Matching access entry or `None` when the plan
            does not expose optional `signal_features` access for this indicator.
    Assumptions:
        Legacy test doubles may omit this additive field entirely and should keep exact behavior.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """
    resolver = getattr(grid_context, "signal_features_access_for_indicator", None)
    if callable(resolver):
        resolved = resolver(indicator_id=indicator_id)
        return cast(BacktestSignalFeaturesAccessPlanV2 | None, resolved)
    access_entries = getattr(grid_context, "signal_features_access", ())
    for access_entry in access_entries:
        if getattr(access_entry, "indicator_id", None) == indicator_id:
            return cast(BacktestSignalFeaturesAccessPlanV2, access_entry)
    return None


def _normalize_indicator_scalar_v2(*, value: object) -> int | float | str:
    """
    Normalize explicit indicator scalar values into supported mixed-radix key types.

    Args:
        value: Raw scalar value from axis definitions or explicit selections.
    Returns:
        int | float | str: Canonical scalar key.
    Assumptions:
        Indicator artifact row addressing uses only JSON-compatible scalar axis values.
    Raises:
        ValueError: If the scalar type is unsupported or bool.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    """
    if isinstance(value, bool):
        raise ValueError("indicator scalar values must not be bool")
    if isinstance(value, int | float | str):
        return value
    raise ValueError("indicator scalar values must be int, float, or str")


def _encode_mixed_radix_v2(
    *,
    coordinates: tuple[int, ...],
    radices: tuple[int, ...],
) -> int:
    """
    Encode one mixed-radix coordinate tuple into a flattened deterministic row index.

    Args:
        coordinates: Zero-based coordinate tuple.
        radices: Positive radix tuple in the same order.
    Returns:
        int: Flattened mixed-radix index.
    Assumptions:
        Artifact signal row order follows the same mixed-radix axis order as Stage A planning.
    Raises:
        ValueError: If lengths differ or one coordinate leaves its radix range.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
    """
    if len(coordinates) != len(radices):
        raise ValueError("coordinates and radices must have the same length")
    index = 0
    for position, coordinate in enumerate(coordinates):
        radix = radices[position]
        if radix <= 0:
            raise ValueError("radices must be > 0")
        if coordinate < 0 or coordinate >= radix:
            raise ValueError("coordinates must stay within [0, radix)")
        multiplier = 1
        for next_radix in radices[position + 1 :]:
            multiplier *= next_radix
        index += coordinate * multiplier
    return index


def _resolve_number_v2(
    *,
    values: Mapping[str, BacktestVariantScalar],
    primary_key: str,
    secondary_key: str,
    default: float,
) -> float:
    """
    Resolve one numeric execution scalar from override mapping with deterministic fallback.

    Args:
        values: Raw execution mapping.
        primary_key: Primary key literal.
        secondary_key: Optional fallback key literal.
        default: Default value used when neither key is present.
    Returns:
        float: Resolved numeric value.
    Assumptions:
        Bool values are rejected even though bool subclasses int in Python.
    Raises:
        ValueError: If one present value is not numeric.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
    """
    candidate = values.get(primary_key)
    if candidate is None and secondary_key:
        candidate = values.get(secondary_key)
    if candidate is None:
        return float(default)
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        raise ValueError(
            f"execution field '{primary_key}' must be numeric when provided"
        )
    return float(candidate)


def _utc_timestamp_to_epoch_millis_v2(value: object) -> int:
    """
    Convert one UTC-aware datetime-like value to epoch milliseconds.

    Args:
        value: Datetime instance carried by `UtcTimestamp.value`.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Runtime request ranges are UTC-aware and stable after DTO normalization.
    Raises:
        ValueError: If the input object is not datetime-like or is timezone-naive.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
    """
    from datetime import datetime

    if not isinstance(value, datetime):
        raise ValueError("UtcTimestamp.value must be datetime")
    if value.tzinfo is None:
        raise ValueError("UtcTimestamp.value must be timezone-aware")
    return int(value.timestamp() * 1000)


__all__ = [
    "BacktestStageAShortlistBuilderV2",
    "PreparedIndicatorChunkInputsV2",
    "PreparedIndicatorRowPlanV2",
    "build_prepared_indicator_row_plan_from_grid_spec_v2",
    "build_default_stage_a_shortlist_builder_v2",
    "compute_target_slice_by_close_time_v2",
    "rebase_bar_close_mapping_v2",
]
