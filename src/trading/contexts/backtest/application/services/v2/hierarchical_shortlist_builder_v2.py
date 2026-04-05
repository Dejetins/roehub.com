"""Hierarchical conservative shortlist runtime for explicit `hybrid_conservative` rollout.

Docs:
  - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/backtest-runtime-kernels-v2.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence, cast

import numpy as np

from trading.contexts.backtest.domain.value_objects import build_backtest_variant_key_v1
from trading.contexts.indicators.application.dto import (
    IndicatorVariantSelection,
    build_variant_key_v1,
)
from trading.shared_kernel.primitives import TimeRange

from .artifact_runtime_plan_v2 import (
    BacktestArtifactRuntimePlanV2,
    BacktestIndicatorPlanV2,
    BacktestStageABaseVariantV2,
    build_indicator_selection_for_variant_index_v2,
    build_signal_params_for_variant_index_v2,
)
from .contracts import (
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactSlotResolverV2,
    BacktestPriceArraysLoaderV2,
    BacktestSignalMatrixLoaderV2,
)
from .diversified_retention_v2 import (
    DiversifiedRetentionDecisionV2,
    DiversifiedRetentionV2,
)
from .execution_profile_v2 import (
    ExecutionProfileV2,
    execution_profile_uses_hierarchical_shortlist_runtime_v2,
)
from .generic_row_scorer_v2 import (
    GenericRowScorePayloadV2,
    GenericRowScorerV2,
    GenericRowScoringInputV2,
)
from .price_arrays_loader import MmapPriceArraysLoaderV2
from .signal_matrix_loader import MmapSignalMatrixLoaderV2
from .stage_a_shortlist_builder_v2 import compute_target_slice_by_close_time_v2


@dataclass(frozen=True, slots=True)
class HierarchicalShortlistRetainedRowV2:
    """
    One retained indicator-block row selected for hierarchical hybrid shortlist expansion.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
    """

    indicator_id: str
    row_index: int
    selection: IndicatorVariantSelection
    score_payload: GenericRowScorePayloadV2
    retained_rank: int

    def __post_init__(self) -> None:
        """
        Validate one retained indicator-block row payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Retained rows already passed deterministic scoring and retention for one indicator
            block and therefore need only stable identity checks here.
        Raises:
            ValueError: If identifiers are blank or ranks/indexes are invalid.
        Side Effects:
            None.
        """
        if not self.indicator_id.strip():
            raise ValueError("HierarchicalShortlistRetainedRowV2.indicator_id must be non-empty")
        if self.row_index < 0:
            raise ValueError("HierarchicalShortlistRetainedRowV2.row_index must be >= 0")
        if self.retained_rank <= 0:
            raise ValueError("HierarchicalShortlistRetainedRowV2.retained_rank must be > 0")


@dataclass(frozen=True, slots=True)
class HierarchicalShortlistBlockResultV2:
    """
    Audit-friendly retained rows and decisions for one indicator block.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_hierarchical_shortlist_builder_v2.py
    """

    indicator_id: str
    retained_rows: tuple[HierarchicalShortlistRetainedRowV2, ...]
    retention_decisions: tuple[DiversifiedRetentionDecisionV2, ...]

    def __post_init__(self) -> None:
        """
        Validate one indicator-block shortlist audit payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Retained rows belong to one indicator block and keep deterministic rank ordering.
        Raises:
            ValueError: If indicator ids drift across retained rows or identities duplicate.
        Side Effects:
            None.
        """
        if not self.indicator_id.strip():
            raise ValueError("HierarchicalShortlistBlockResultV2.indicator_id must be non-empty")
        seen_rows: set[int] = set()
        for retained_row in self.retained_rows:
            if retained_row.indicator_id != self.indicator_id:
                raise ValueError(
                    "HierarchicalShortlistBlockResultV2 retained_rows must share indicator_id"
                )
            if retained_row.row_index in seen_rows:
                raise ValueError(
                    "HierarchicalShortlistBlockResultV2 retained_rows must not duplicate row "
                    f"indexes for {self.indicator_id!r}"
                )
            seen_rows.add(retained_row.row_index)


@dataclass(frozen=True, slots=True)
class HierarchicalShortlistRuntimePlanV2(BacktestArtifactRuntimePlanV2):
    """
    Reduced runtime plan exposing only hybrid-retained Stage A survivors.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py
    """

    source_runtime_plan: BacktestArtifactRuntimePlanV2 | None = None
    retained_stage_a_variants: tuple[BacktestStageABaseVariantV2, ...] = ()
    block_results: tuple[HierarchicalShortlistBlockResultV2, ...] = ()
    retained_compute_variants_total: int = 0

    def __post_init__(self) -> None:
        """
        Validate reduced runtime-plan invariants after base plan normalization.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The reduced plan preserves all exact Stage B contracts while shrinking only Stage A
            enumeration space and the derived Stage B workload.
        Raises:
            ValueError: If retained variants are missing, unsorted, or cardinalities drift.
        Side Effects:
            Reuses base-plan normalization from `BacktestArtifactRuntimePlanV2.__post_init__`.
        """
        BacktestArtifactRuntimePlanV2.__post_init__(self)
        if self.source_runtime_plan is None:  # type: ignore[truthy-bool]
            raise ValueError("HierarchicalShortlistRuntimePlanV2.source_runtime_plan is required")
        if len(self.retained_stage_a_variants) == 0:
            raise ValueError(
                "HierarchicalShortlistRuntimePlanV2.retained_stage_a_variants must be non-empty"
            )
        if self.stage_a_variants_total != len(self.retained_stage_a_variants):
            raise ValueError(
                "HierarchicalShortlistRuntimePlanV2.stage_a_variants_total must match retained "
                "Stage A variants count"
            )
        if self.retained_compute_variants_total <= 0:
            raise ValueError(
                "HierarchicalShortlistRuntimePlanV2.retained_compute_variants_total must be > 0"
            )
        previous_stage_a_index: int | None = None
        seen_stage_a_indexes: set[int] = set()
        for variant in self.retained_stage_a_variants:
            if variant.stage_a_index in seen_stage_a_indexes:
                raise ValueError(
                    "HierarchicalShortlistRuntimePlanV2.retained_stage_a_variants must not "
                    f"duplicate stage_a_index {variant.stage_a_index}"
                )
            seen_stage_a_indexes.add(variant.stage_a_index)
            if (
                previous_stage_a_index is not None
                and variant.stage_a_index <= previous_stage_a_index
            ):
                raise ValueError(
                    "HierarchicalShortlistRuntimePlanV2.retained_stage_a_variants must stay "
                    "sorted by original stage_a_index"
                )
            previous_stage_a_index = variant.stage_a_index

    @classmethod
    def from_source_runtime_plan(
        cls,
        *,
        source_runtime_plan: BacktestArtifactRuntimePlanV2,
        retained_stage_a_variants: tuple[BacktestStageABaseVariantV2, ...],
        block_results: tuple[HierarchicalShortlistBlockResultV2, ...],
        retained_compute_variants_total: int,
    ) -> "HierarchicalShortlistRuntimePlanV2":
        """
        Build one reduced runtime plan reusing the exact-plan immutable contract surface.

        Args:
            source_runtime_plan: Original exact enumeration plan resolved by the planner.
            retained_stage_a_variants: Reduced Stage A survivor enumeration in exact order.
            block_results: Per-indicator shortlist debug evidence.
            retained_compute_variants_total: Count of retained compute combinations before signal
                expansion.
        Returns:
            HierarchicalShortlistRuntimePlanV2: Reduced runtime plan preserving exact Stage B
                scorer contracts.
        Assumptions:
            Stage B risk expansion limit remains the original plan's shortlist envelope.
        Raises:
            ValueError: If the retained survivor set is empty.
        Side Effects:
            None.
        """
        if len(retained_stage_a_variants) == 0:
            raise ValueError(
                "HierarchicalShortlistRuntimePlanV2 requires non-empty retained_stage_a_variants"
            )
        risk_total = len(source_runtime_plan.risk_variants)
        if risk_total <= 0:
            raise ValueError(
                "HierarchicalShortlistRuntimePlanV2 requires source plan risk variants"
            )
        original_shortlist_limit = max(
            1,
            source_runtime_plan.stage_b_variants_total // risk_total,
        )
        retained_stage_b_variants_total = (
            min(len(retained_stage_a_variants), original_shortlist_limit) * risk_total
        )
        return cls(
            indicator_plans=source_runtime_plan.indicator_plans,
            signal_axes=source_runtime_plan.signal_axes,
            risk_variants=source_runtime_plan.risk_variants,
            execution_profile=source_runtime_plan.execution_profile,
            instrument_id_literal=source_runtime_plan.instrument_id_literal,
            timeframe_code=source_runtime_plan.timeframe_code,
            direction_mode=source_runtime_plan.direction_mode,
            sizing_mode=source_runtime_plan.sizing_mode,
            execution_params=source_runtime_plan.execution_params,
            stage_a_variants_total=len(retained_stage_a_variants),
            stage_b_variants_total=retained_stage_b_variants_total,
            estimated_memory_bytes=source_runtime_plan.estimated_memory_bytes,
            indicator_estimate_calls=source_runtime_plan.indicator_estimate_calls,
            signal_features_access=source_runtime_plan.signal_features_access,
            source_runtime_plan=source_runtime_plan,
            retained_stage_a_variants=retained_stage_a_variants,
            block_results=block_results,
            retained_compute_variants_total=retained_compute_variants_total,
        )

    def iter_stage_a_variants(self) -> Iterator[BacktestStageABaseVariantV2]:
        """
        Iterate only the retained hybrid survivor variants in original exact enumeration order.

        Args:
            None.
        Returns:
            Iterator[BacktestStageABaseVariantV2]: Reduced deterministic Stage A survivor stream.
        Assumptions:
            Stage A exact kernels still consume explicit base variants and should not know whether
            they came from full exact enumeration or hybrid shortlist pruning.
        Raises:
            None.
        Side Effects:
            None.
        """
        return iter(self.retained_stage_a_variants)


@dataclass(frozen=True, slots=True)
class _HierarchicalBeamCandidateV2:
    """
    Intermediate compute-combination candidate used by deterministic beam combination.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py
    """

    row_indexes: tuple[int, ...]
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    score_total: float
    stable_identities: tuple[str, ...]

    def sort_key(self) -> tuple[float, tuple[str, ...], tuple[int, ...]]:
        """
        Return one explicit deterministic ordering key for beam selection.

        Args:
            None.
        Returns:
            tuple[float, tuple[str, ...], tuple[int, ...]]: Descending-score deterministic
                ordering key.
        Assumptions:
            All candidates in one beam round have the same arity and therefore compare fairly by
            score plus explicit stable identities.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (-self.score_total, self.stable_identities, self.row_indexes)


@dataclass(frozen=True, slots=True)
class BacktestHierarchicalShortlistBuilderV2:
    """
    Build conservative hybrid survivor runtime plans for explicit `hybrid_conservative` runs.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    price_arrays_loader: BacktestPriceArraysLoaderV2
    signal_matrix_loader: BacktestSignalMatrixLoaderV2
    row_scorer: GenericRowScorerV2 = field(default_factory=GenericRowScorerV2)
    diversified_retention: DiversifiedRetentionV2 = field(
        default_factory=DiversifiedRetentionV2
    )
    block_survivor_multiplier: int = 2

    def __post_init__(self) -> None:
        """
        Validate constructor dependencies and deterministic shortlist tuning knobs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Constructor wires collaborators only and performs no artifact IO.
        Raises:
            ValueError: If dependencies are missing or multiplier is invalid.
        Side Effects:
            None.
        """
        if self.price_arrays_loader is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2 requires price_arrays_loader"
            )
        if self.signal_matrix_loader is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2 requires signal_matrix_loader"
            )
        if self.block_survivor_multiplier <= 0:
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2.block_survivor_multiplier must be > 0"
            )

    def build_runtime_plan(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        target_time_range: TimeRange,
    ) -> BacktestArtifactRuntimePlanV2:
        """
        Build one reduced runtime plan for the explicit hybrid conservative rollout path.

        Docs:
          - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

        Args:
            runtime_plan: Original exact runtime plan resolved by the planner.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            target_time_range: Requested trading window for row-level hybrid scoring.
        Returns:
            BacktestArtifactRuntimePlanV2: Original plan when no reduction is needed, otherwise a
                reduced `HierarchicalShortlistRuntimePlanV2`.
        Assumptions:
            Hybrid rollout remains opt-in and still delegates exact Stage A and exact Stage B
            scoring to existing kernels after this reduced enumeration plan is built.
        Raises:
            ValueError: If the resolved profile is not live-enabled for hierarchical shortlist
                runtime or if artifact row-count contracts drift.
        Side Effects:
            Reads pinned price and signal artifacts to score candidate indicator rows.
        """
        profile = runtime_plan.execution_profile
        if not execution_profile_uses_hierarchical_shortlist_runtime_v2(profile=profile):
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2 requires explicit "
                "hybrid_conservative runtime-enabled profile"
            )

        signal_variants_total = _signal_variants_total_v2(runtime_plan=runtime_plan)
        compute_variants_total = _compute_variants_total_v2(runtime_plan=runtime_plan)
        conservative_stage_a_budget = self._conservative_stage_a_budget(
            runtime_plan=runtime_plan,
        )
        target_compute_variants_total = self._target_compute_variants_total(
            compute_variants_total=compute_variants_total,
            signal_variants_total=signal_variants_total,
            conservative_stage_a_budget=conservative_stage_a_budget,
        )
        if target_compute_variants_total >= compute_variants_total:
            return runtime_plan

        block_survivor_budget = self._block_survivor_budget(
            runtime_plan=runtime_plan,
            target_compute_variants_total=target_compute_variants_total,
        )
        signal_target_slice = self._signal_target_slice(
            runtime_plan=runtime_plan,
            artifact_context=artifact_context,
            target_time_range=target_time_range,
        )
        block_results = tuple(
            self._build_block_result(
                plan=plan,
                timeframe_code=runtime_plan.timeframe_code,
                profile=profile,
                artifact_context=artifact_context,
                signal_target_slice=signal_target_slice,
                block_survivor_budget=block_survivor_budget,
            )
            for plan in runtime_plan.indicator_plans
        )
        compute_candidates = self._beam_candidates(
            block_results=block_results,
            target_compute_variants_total=target_compute_variants_total,
        )
        retained_stage_a_variants = self._retained_stage_a_variants(
            runtime_plan=runtime_plan,
            compute_candidates=compute_candidates,
            signal_variants_total=signal_variants_total,
        )
        return HierarchicalShortlistRuntimePlanV2.from_source_runtime_plan(
            source_runtime_plan=runtime_plan,
            retained_stage_a_variants=retained_stage_a_variants,
            block_results=block_results,
            retained_compute_variants_total=len(compute_candidates),
        )

    def _conservative_stage_a_budget(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
    ) -> int:
        """
        Resolve the conservative Stage A survivor budget for hybrid shortlist expansion.

        Args:
            runtime_plan: Original exact runtime plan.
        Returns:
            int: Conservative Stage A survivor budget used before exact Stage A scoring.
        Assumptions:
            Hybrid rollout should not expand beyond the original exact Stage B shortlist envelope
            and must also honor the profile's explicit shortlist cap.
        Raises:
            ValueError: If risk variants are missing.
        Side Effects:
            None.
        """
        risk_total = len(runtime_plan.risk_variants)
        if risk_total <= 0:
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2 requires runtime_plan.risk_variants"
            )
        original_shortlist_limit = max(
            1,
            runtime_plan.stage_b_variants_total // risk_total,
        )
        profile_max_candidates = (
            runtime_plan.execution_profile.shortlist_config.max_candidates
            or runtime_plan.stage_a_variants_total
        )
        return min(
            runtime_plan.stage_a_variants_total,
            profile_max_candidates,
            original_shortlist_limit,
        )

    def _target_compute_variants_total(
        self,
        *,
        compute_variants_total: int,
        signal_variants_total: int,
        conservative_stage_a_budget: int,
    ) -> int:
        """
        Resolve the retained compute-combination budget before signal expansion.

        Args:
            compute_variants_total: Original compute-combination total.
            signal_variants_total: Original signal-space expansion total.
            conservative_stage_a_budget: Conservative Stage A survivor budget.
        Returns:
            int: Retained compute-combination count target.
        Assumptions:
            Every retained compute combination expands through the existing exact signal-space
            semantics, so at least one compute combination may be required even when the signal
            space itself exceeds the conservative budget.
        Raises:
            None.
        Side Effects:
            None.
        """
        if signal_variants_total <= 0:
            raise ValueError("signal_variants_total must be > 0")
        return min(
            compute_variants_total,
            max(1, conservative_stage_a_budget // signal_variants_total),
        )

    def _block_survivor_budget(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        target_compute_variants_total: int,
    ) -> int:
        """
        Resolve the per-block retained-row budget used before beam combination.

        Args:
            runtime_plan: Original exact runtime plan.
            target_compute_variants_total: Retained compute-combination count target.
        Returns:
            int: Per-indicator retained-row budget.
        Assumptions:
            The budget is intentionally conservative, using the mixed-radix root of the target
            compute space with a small multiplier to preserve diversity.
        Raises:
            None.
        Side Effects:
            None.
        """
        indicator_count = max(1, len(runtime_plan.indicator_plans))
        root_budget = int(
            math.ceil(target_compute_variants_total ** (1.0 / float(indicator_count)))
        )
        return max(1, root_budget * self.block_survivor_multiplier)

    def _signal_target_slice(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        target_time_range: TimeRange,
    ) -> slice:
        """
        Resolve the request-timeframe signal slice used for row-local hybrid scoring.

        Args:
            runtime_plan: Original exact runtime plan.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            target_time_range: Requested trading window.
        Returns:
            slice: Half-open signal-timeframe slice matching the requested trading window.
        Assumptions:
            Hybrid row scoring uses only the requested execution window, not the full warmup
            artifact timeline.
        Raises:
            ValueError: If the underlying close-time artifact is invalid.
        Side Effects:
            Reads one request-timeframe price artifact for `close_time`.
        """
        signal_prices = self.price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe=runtime_plan.timeframe_code,
        )
        return compute_target_slice_by_close_time_v2(
            close_time=signal_prices.close_time,
            target_time_range=target_time_range,
        )

    def _build_block_result(
        self,
        *,
        plan: BacktestIndicatorPlanV2,
        timeframe_code: str,
        profile: ExecutionProfileV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
        block_survivor_budget: int,
    ) -> HierarchicalShortlistBlockResultV2:
        """
        Score and retain one indicator block for later beam combination.

        Args:
            plan: One indicator plan from the original exact runtime plan.
            timeframe_code: Request-timeframe signal artifact family literal.
            profile: Resolved hybrid execution profile.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            signal_target_slice: Request-timeframe target slice used for row-local scoring.
            block_survivor_budget: Per-indicator retained-row budget.
        Returns:
            HierarchicalShortlistBlockResultV2: Retained rows plus full retention audit trail.
        Assumptions:
            Indicator-row indexes in artifact matrices match the planner's indicator-local
            mixed-radix variant indexes.
        Raises:
            ValueError: If signal row counts drift from planner invariants.
        Side Effects:
            Reads one strict signal matrix from the pinned artifact slot.
        """
        signal_matrix = self.signal_matrix_loader.load_signal_matrix(
            context=artifact_context,
            timeframe=timeframe_code,
            indicator_id=plan.indicator_id,
        )
        if signal_matrix.manifest.rows_count != plan.variants:
            raise ValueError(
                "Hierarchical shortlist requires signal rows_count to match indicator plan "
                f"variants for {plan.indicator_id!r}; got "
                f"{signal_matrix.manifest.rows_count}, expected {plan.variants}"
            )

        scored_rows = self._scorer_for_profile(profile=profile).score_rows(
            rows=self._row_inputs_for_indicator_plan(
                plan=plan,
                signal_matrix=signal_matrix.matrix,
                signal_target_slice=signal_target_slice,
            )
        )
        retention_result = self.diversified_retention.retain_rows(
            scored_rows=scored_rows,
            config=profile.shortlist_config.retention,
            max_candidates=min(block_survivor_budget, plan.variants),
        )
        retained_rows = tuple(
            HierarchicalShortlistRetainedRowV2(
                indicator_id=plan.indicator_id,
                row_index=scored_row.row_index,
                selection=build_indicator_selection_for_variant_index_v2(
                    plan=plan,
                    variant_index=scored_row.row_index,
                ),
                score_payload=scored_row,
                retained_rank=retained_rank,
            )
            for retained_rank, scored_row in enumerate(
                retention_result.retained_rows,
                start=1,
            )
        )
        return HierarchicalShortlistBlockResultV2(
            indicator_id=plan.indicator_id,
            retained_rows=retained_rows,
            retention_decisions=retention_result.decisions,
        )

    def _scorer_for_profile(
        self,
        *,
        profile: ExecutionProfileV2,
    ) -> GenericRowScorerV2:
        """
        Rehydrate the universal row scorer with the profile's explicit scoring weights.

        Args:
            profile: Resolved hybrid execution profile.
        Returns:
            GenericRowScorerV2: Row scorer carrying profile-specific shortlist weights.
        Assumptions:
            Threshold literals remain owned by the builder-level scorer while weights come from the
            resolved execution profile.
        Raises:
            None.
        Side Effects:
            None.
        """
        return GenericRowScorerV2(
            scoring=profile.shortlist_config.scoring,
            low_activity_threshold=self.row_scorer.low_activity_threshold,
            high_activity_threshold=self.row_scorer.high_activity_threshold,
            direction_balance_threshold=self.row_scorer.direction_balance_threshold,
            low_transition_ratio_threshold=(
                self.row_scorer.low_transition_ratio_threshold
            ),
            high_transition_ratio_threshold=(
                self.row_scorer.high_transition_ratio_threshold
            ),
        )

    def _row_inputs_for_indicator_plan(
        self,
        *,
        plan: BacktestIndicatorPlanV2,
        signal_matrix: np.ndarray,
        signal_target_slice: slice,
    ) -> tuple[GenericRowScoringInputV2, ...]:
        """
        Build deterministic row-scoring inputs for one indicator block.

        Args:
            plan: One indicator plan from the original exact runtime plan.
            signal_matrix: Full artifact signal matrix in `(rows, time)` order.
            signal_target_slice: Request-timeframe target slice used for row-local scoring.
        Returns:
            tuple[GenericRowScoringInputV2, ...]: Deterministic row-scoring inputs for the block.
        Assumptions:
            Row indexes in the signal matrix align exactly with the planner's indicator-local
            mixed-radix variant indexes.
        Raises:
            ValueError: If the target slice is empty or signal matrix shape drifts.
        Side Effects:
            None.
        """
        if signal_matrix.ndim != 2:
            raise ValueError("Hierarchical shortlist requires 2D signal matrices")
        if signal_matrix.shape[0] != plan.variants:
            raise ValueError(
                "Hierarchical shortlist requires signal matrix rows to match plan variants for "
                f"{plan.indicator_id!r}; got {signal_matrix.shape[0]}, expected {plan.variants}"
            )
        if (signal_target_slice.stop or 0) <= (signal_target_slice.start or 0):
            raise ValueError(
                "Hierarchical shortlist requires non-empty request-timeframe signal_target_slice"
            )
        sliced_signal_matrix = signal_matrix[:, signal_target_slice]
        return tuple(
            GenericRowScoringInputV2(
                indicator_id=plan.indicator_id,
                row_index=row_index,
                signal_row=np.asarray(sliced_signal_matrix[row_index, :], dtype=np.int8),
                stable_identity=f"{plan.indicator_id}:{row_index}",
            )
            for row_index in range(plan.variants)
        )

    def _beam_candidates(
        self,
        *,
        block_results: tuple[HierarchicalShortlistBlockResultV2, ...],
        target_compute_variants_total: int,
    ) -> tuple[_HierarchicalBeamCandidateV2, ...]:
        """
        Combine retained block rows into bounded deterministic compute candidates.

        Args:
            block_results: Ordered retained-row payloads for every indicator block.
            target_compute_variants_total: Retained compute-combination count target.
        Returns:
            tuple[_HierarchicalBeamCandidateV2, ...]: Bounded deterministic compute candidates.
        Assumptions:
            Beam combination is conservative and deterministic; it optimizes for explicit score
            ordering rather than exhaustive approximate search.
        Raises:
            ValueError: If a block has no retained rows.
        Side Effects:
            None.
        """
        beam: tuple[_HierarchicalBeamCandidateV2, ...] = (
            _HierarchicalBeamCandidateV2(
                row_indexes=(),
                indicator_selections=(),
                score_total=0.0,
                stable_identities=(),
            ),
        )
        for block_result in block_results:
            if len(block_result.retained_rows) == 0:
                raise ValueError(
                    "Hierarchical shortlist beam combination requires retained rows for every "
                    f"indicator block, missing {block_result.indicator_id!r}"
                )
            expanded: list[_HierarchicalBeamCandidateV2] = []
            for partial_candidate in beam:
                for retained_row in block_result.retained_rows:
                    expanded.append(
                        _HierarchicalBeamCandidateV2(
                            row_indexes=partial_candidate.row_indexes
                            + (retained_row.row_index,),
                            indicator_selections=partial_candidate.indicator_selections
                            + (retained_row.selection,),
                            score_total=partial_candidate.score_total
                            + retained_row.score_payload.total_score,
                            stable_identities=partial_candidate.stable_identities
                            + (retained_row.score_payload.stable_identity,),
                        )
                    )
            beam = tuple(
                sorted(
                    expanded,
                    key=lambda candidate: candidate.sort_key(),
                )[:target_compute_variants_total]
            )
        return beam

    def _retained_stage_a_variants(
        self,
        *,
        runtime_plan: BacktestArtifactRuntimePlanV2,
        compute_candidates: tuple[_HierarchicalBeamCandidateV2, ...],
        signal_variants_total: int,
    ) -> tuple[BacktestStageABaseVariantV2, ...]:
        """
        Expand retained compute candidates back into exact Stage A base variants.

        Args:
            runtime_plan: Original exact runtime plan.
            compute_candidates: Bounded retained compute candidates from beam combination.
            signal_variants_total: Signal-space expansion total from the original plan.
        Returns:
            tuple[BacktestStageABaseVariantV2, ...]: Reduced exact Stage A base variants in the
                original exact enumeration order.
        Assumptions:
            Exact Stage B remains the final source of truth, so hybrid pruning must preserve the
            canonical indicator/signal/risk payload semantics on survivors.
        Raises:
            ValueError: If no compute candidates remain after beam combination.
        Side Effects:
            None.
        """
        if len(compute_candidates) == 0:
            raise ValueError(
                "BacktestHierarchicalShortlistBuilderV2 requires non-empty compute_candidates"
            )
        indicator_radices = tuple(plan.variants for plan in runtime_plan.indicator_plans)
        retained_variants: list[BacktestStageABaseVariantV2] = []
        for compute_candidate in sorted(
            compute_candidates,
            key=lambda candidate: (
                _encode_mixed_radix_v2(
                    coordinates=candidate.row_indexes,
                    radices=indicator_radices,
                ),
                candidate.stable_identities,
            ),
        ):
            compute_index = _encode_mixed_radix_v2(
                coordinates=compute_candidate.row_indexes,
                radices=indicator_radices,
            )
            indicator_variant_key = build_variant_key_v1(
                instrument_id=runtime_plan.instrument_id_literal,
                timeframe=runtime_plan.timeframe_code,
                indicators=compute_candidate.indicator_selections,
            )
            for signal_index in range(signal_variants_total):
                signal_params = build_signal_params_for_variant_index_v2(
                    signal_axes=runtime_plan.signal_axes,
                    variant_index=signal_index,
                )
                stage_a_index = (compute_index * signal_variants_total) + signal_index
                base_variant_key = build_backtest_variant_key_v1(
                    indicator_variant_key=indicator_variant_key,
                    direction_mode=runtime_plan.direction_mode,
                    sizing_mode=runtime_plan.sizing_mode,
                    signals=signal_params,
                    risk_params=_STAGE_A_DISABLED_RISK_PARAMS_V2,
                    execution_params=runtime_plan.execution_params,
                )
                retained_variants.append(
                    BacktestStageABaseVariantV2(
                        stage_a_index=stage_a_index,
                        indicator_selections=compute_candidate.indicator_selections,
                        signal_params=signal_params,
                        indicator_variant_key=indicator_variant_key,
                        base_variant_key=base_variant_key,
                    )
                )
        return tuple(
            sorted(retained_variants, key=lambda variant: variant.stage_a_index)
        )


def build_default_hierarchical_shortlist_builder_v2(
    *,
    artifact_slot_resolver: BacktestArtifactSlotResolverV2 | None,
) -> BacktestHierarchicalShortlistBuilderV2 | None:
    """
    Build the default hierarchical shortlist builder from shared artifact runtime wiring.

    Docs:
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        artifact_slot_resolver: Optional slot resolver already wired by runtime startup.
    Returns:
        BacktestHierarchicalShortlistBuilderV2 | None: Default builder when the resolver exposes
            a shared artifact loader, otherwise `None`.
    Assumptions:
        Hybrid rollout wiring stays additive and should disappear entirely when artifact-backed
        runtime wiring is unavailable.
    Raises:
        ValueError: Propagated from builder constructor when defaults are invalid.
    Side Effects:
        None.
    """
    if artifact_slot_resolver is None:
        return None
    artifact_loader = getattr(artifact_slot_resolver, "artifact_loader", None)
    if artifact_loader is None:
        return None
    typed_artifact_loader = cast(BacktestArtifactLoaderV2, artifact_loader)
    return BacktestHierarchicalShortlistBuilderV2(
        price_arrays_loader=MmapPriceArraysLoaderV2(artifact_loader=typed_artifact_loader),
        signal_matrix_loader=MmapSignalMatrixLoaderV2(
            artifact_loader=typed_artifact_loader
        ),
    )


def _compute_variants_total_v2(*, runtime_plan: BacktestArtifactRuntimePlanV2) -> int:
    """
    Compute the original indicator-only mixed-radix variants total.

    Args:
        runtime_plan: Original exact runtime plan.
    Returns:
        int: Indicator-only compute variants total.
    Assumptions:
        Planner already validated positive indicator variant counts.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _product_v2(tuple(plan.variants for plan in runtime_plan.indicator_plans))


def _signal_variants_total_v2(*, runtime_plan: BacktestArtifactRuntimePlanV2) -> int:
    """
    Compute the original signal-space mixed-radix variants total.

    Args:
        runtime_plan: Original exact runtime plan.
    Returns:
        int: Signal-space variants total.
    Assumptions:
        Empty signal-axis sets expand to one default-only signal payload.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _product_v2(tuple(len(axis.values) for axis in runtime_plan.signal_axes))


def _product_v2(values: tuple[int, ...]) -> int:
    """
    Multiply a tuple of deterministic positive integers with empty-tuple identity `1`.

    Args:
        values: Ordered integer factors.
    Returns:
        int: Product of all factors, or `1` when `values` is empty.
    Assumptions:
        Caller already validated factor positivity.
    Raises:
        None.
    Side Effects:
        None.
    """
    result = 1
    for value in values:
        result *= int(value)
    return result


def _encode_mixed_radix_v2(
    *,
    coordinates: Sequence[int],
    radices: Sequence[int],
) -> int:
    """
    Encode one mixed-radix coordinate tuple into a flat index.

    Args:
        coordinates: Indicator-local coordinate values in radix order.
        radices: Positive radix sizes for each coordinate position.
    Returns:
        int: Deterministic flat mixed-radix index.
    Assumptions:
        Coordinates and radices use identical ordering and lengths.
    Raises:
        ValueError: If lengths drift or one coordinate is outside its radix bounds.
    Side Effects:
        None.
    """
    if len(coordinates) != len(radices):
        raise ValueError("mixed-radix coordinates and radices must have the same length")
    flat_index = 0
    stride = 1
    for coordinate, radix in zip(
        reversed(tuple(coordinates)),
        reversed(tuple(radices)),
        strict=True,
    ):
        if radix <= 0:
            raise ValueError("mixed-radix radices must be > 0")
        if coordinate < 0 or coordinate >= radix:
            raise ValueError(
                "mixed-radix coordinate is outside radix bounds; got "
                f"{coordinate} for radix {radix}"
            )
        flat_index += int(coordinate) * stride
        stride *= int(radix)
    return flat_index


_STAGE_A_DISABLED_RISK_PARAMS_V2 = {
    "sl_enabled": False,
    "sl_pct": None,
    "tp_enabled": False,
    "tp_pct": None,
}


__all__ = [
    "BacktestHierarchicalShortlistBuilderV2",
    "HierarchicalShortlistBlockResultV2",
    "HierarchicalShortlistRetainedRowV2",
    "HierarchicalShortlistRuntimePlanV2",
    "build_default_hierarchical_shortlist_builder_v2",
]
