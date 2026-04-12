"""Artifact-backed Stage A shortlist builder with row and combo proxy prefiltering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from heapq import heappush, heapreplace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, cast

import numba as nb
import numpy as np

from trading.contexts.backtest.application.dto import BacktestRankingConfig
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import TimeRange

from ..numba_runtime_v1 import (
    BacktestStageAParallelismConfigV1,
    backtest_stage_a_numba_threads_scope_v1,
    resolve_backtest_stage_a_parallelism_v1,
)
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
from .generic_row_scorer_v2 import (
    GenericRowScorerV2,
    GenericRowScoringInputV2,
    build_generic_row_signal_features_mapping_v2,
)
from .price_arrays_loader import MmapPriceArraysLoaderV2
from .signal_aggregator_kernel import aggregate_ordered_final_signal_rows_v2
from .signal_features_loader_v2 import MmapSignalFeaturesLoaderV2
from .signal_matrix_loader import MmapSignalMatrixLoaderV2
from .trade_compactor_kernel import (
    _CompactTradeBatchV2,
    build_compact_trade_batch_v2,
    compute_no_risk_metrics_for_trade_batch_v2,
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
_COMBO_PROXY_PREFILTER_SURVIVOR_MULTIPLIER_V2 = 2

StageACancelCheckerV2 = Callable[[str], None]
StageACheckpointCallbackV2 = Callable[[int, int], None]


def _close_returns_for_proxy_score_v2(
    *,
    local_signal_close: np.ndarray,
    error_prefix: str,
) -> np.ndarray:
    """
    Compute deterministic next-bar close returns for Stage A proxy-score evaluation.

    Args:
        local_signal_close: Request-timeframe close prices aligned to the signal timeline.
        error_prefix: Error-message prefix describing the calling proxy stage.
    Returns:
        np.ndarray: Float64 close-return vector aligned to `signal_row[:-1]`.
    Assumptions:
        Stage A proxy scoring evaluates only next-bar returns, so one close-return value exists
        for every signal interval except the final bar.
    Raises:
        ValueError: If close prices are not one-dimensional.
    Side Effects:
        None.
    """
    normalized_local_signal_close = np.asarray(local_signal_close, dtype=np.float64)
    if normalized_local_signal_close.ndim != 1:
        raise ValueError(f"{error_prefix} requires 1D local_signal_close")
    if normalized_local_signal_close.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    prior_close = np.ascontiguousarray(normalized_local_signal_close[:-1], dtype=np.float64)
    next_close = np.ascontiguousarray(normalized_local_signal_close[1:], dtype=np.float64)
    return np.divide(
        next_close - prior_close,
        prior_close,
        out=np.zeros_like(prior_close, dtype=np.float64),
        where=prior_close != 0.0,
    )


@nb.njit(parallel=True, cache=True)
def _batch_proxy_scores_for_signal_rows_kernel_v2(
    *,
    signal_rows_i8: np.ndarray,
    close_returns_f64: np.ndarray,
    fee_rate: float,
) -> np.ndarray:
    """
    Compute fee-adjusted proxy scores for one Stage A signal-row matrix in parallel.

    Args:
        signal_rows_i8: Int8 matrix shaped `[row, time]` on the request signal timeline.
        close_returns_f64: Float64 close-return vector aligned to `signal_rows_i8[:, :-1]`.
        fee_rate: Decimal per-side fee penalty applied per non-zero signal.
    Returns:
        np.ndarray: Float64 proxy score per row in the original deterministic row order.
    Assumptions:
        The final signal bar has no next-bar return, so the kernel evaluates only the first
        `close_returns_f64.shape[0]` intervals from each row.
    Raises:
        None.
    Side Effects:
        Allocates one float64 score vector and may trigger Numba compilation on first use.
    """
    row_count = int(signal_rows_i8.shape[0])
    interval_count = int(close_returns_f64.shape[0])
    scores = np.empty(row_count, dtype=np.float64)
    for row_index in nb.prange(row_count):
        proxy_score = 0.0
        activity_count = 0
        for interval_index in range(interval_count):
            signal_value = int(signal_rows_i8[row_index, interval_index])
            if signal_value == 0:
                continue
            proxy_score += float(signal_value) * close_returns_f64[interval_index]
            activity_count += 1
        scores[row_index] = proxy_score - (fee_rate * float(activity_count))
    return scores


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
    Per-indicator Stage A chunk inputs carrying signal rows, row addressing, and warm-cache access.

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
    signal_row_selection: tuple[int, ...] | None = None
    signal_features_loader: BacktestSignalFeaturesLoaderV2 | None = None
    signal_features_context: ArtifactSlotPinnedRuntimeContextV2 | None = None
    signal_features_access: BacktestSignalFeaturesAccessPlanV2 | None = None
    signal_feature_row_selection: slice | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate one prepared chunk input and keep retained row addressing explicit.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Explicit `signal_row_selection` remains 1:1 aligned with `signal_rows` so later
            Stage A steps can rebuild exact retained batches without retaining full
            `final_signal_row` buffers, while omitted selections fall back to batch-local ordinal
            positions for additive compatibility with older callers.
        Raises:
            ValueError: If `signal_rows` is not 2D, row selection is empty, or alignment drifts.
        Side Effects:
            Normalizes `signal_row_selection` to builtin `int` values.
        """
        normalized_signal_rows = np.asarray(self.signal_rows, dtype=np.int8)
        if normalized_signal_rows.ndim != 2:
            raise ValueError("PreparedIndicatorChunkInputsV2.signal_rows must be 2D")
        raw_row_selection = self.signal_row_selection
        normalized_row_selection = (
            tuple(range(int(normalized_signal_rows.shape[0])))
            if raw_row_selection is None
            else tuple(int(value) for value in raw_row_selection)
        )
        if len(normalized_row_selection) == 0:
            raise ValueError(
                "PreparedIndicatorChunkInputsV2.signal_row_selection must be non-empty"
            )
        if any(value < 0 for value in normalized_row_selection):
            raise ValueError(
                "PreparedIndicatorChunkInputsV2.signal_row_selection must be >= 0"
            )
        if len(normalized_row_selection) != int(normalized_signal_rows.shape[0]):
            raise ValueError(
                "PreparedIndicatorChunkInputsV2.signal_row_selection must align with signal_rows"
            )
        object.__setattr__(self, "signal_rows", normalized_signal_rows)
        object.__setattr__(self, "signal_row_selection", normalized_row_selection)

    def load_signal_feature_rows(self) -> ArtifactSignalFeaturesRowsV2 | None:
        """
        Materialize optional selected signal-feature rows for this chunk in variant order.

        Args:
            None.
        Returns:
            ArtifactSignalFeaturesRowsV2 | None: Selected feature rows when the additive
                `signal_features` family is available for this indicator, else `None`.
        Assumptions:
            `signal_row_selection` and optional `signal_feature_row_selection` stay aligned with
            `signal_rows` ordering for the same chunk variants, and feature matrices should stay
            lazy until this method is called.
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
class RetainedIndicatorRowFrontierV2:
    """
    Deterministic retained frontier for one indicator family after row-local prefiltering.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
    """

    indicator_id: str
    retained_row_indexes: tuple[int, ...]
    retained_row_lookup: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Validate one retained frontier and keep both explicit ordering and fast membership.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Retained row ordering must stay explicit and stable for reviewability, while exact
            chunk filtering still benefits from constant-time membership checks.
        Raises:
            ValueError: If the indicator id is blank, the frontier is empty, or row indexes are
                negative/duplicated.
        Side Effects:
            Normalizes retained row indexes to builtin `int` and derives a lookup set.
        """
        indicator_id = self.indicator_id.strip()
        if not indicator_id:
            raise ValueError("RetainedIndicatorRowFrontierV2.indicator_id must be non-empty")
        object.__setattr__(self, "indicator_id", indicator_id)
        normalized_row_indexes = tuple(int(value) for value in self.retained_row_indexes)
        if len(normalized_row_indexes) == 0:
            raise ValueError(
                "RetainedIndicatorRowFrontierV2.retained_row_indexes must be non-empty"
            )
        if any(value < 0 for value in normalized_row_indexes):
            raise ValueError(
                "RetainedIndicatorRowFrontierV2.retained_row_indexes must be >= 0"
            )
        if len(set(normalized_row_indexes)) != len(normalized_row_indexes):
            raise ValueError(
                "RetainedIndicatorRowFrontierV2.retained_row_indexes must be unique"
            )
        object.__setattr__(self, "retained_row_indexes", normalized_row_indexes)
        object.__setattr__(
            self,
            "retained_row_lookup",
            frozenset(normalized_row_indexes),
        )

    def contains_row_index(
        self,
        *,
        row_index: int,
    ) -> bool:
        """
        Check whether one indicator-local row index survived deterministic prefiltering.

        Args:
            row_index: Indicator-local row index to test.
        Returns:
            bool: `True` when the row remains inside the retained frontier.
        Assumptions:
            Membership checks should not depend on tuple scanning because exact Stage A may still
            inspect many variants against the retained frontier.
        Raises:
            None.
        Side Effects:
            None.
        """
        return int(row_index) in self.retained_row_lookup


@dataclass(frozen=True, slots=True)
class _RetainedExactCandidateAddressV2:
    """
    Minimal deterministic row-address metadata for one retained Stage A exact candidate.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """

    indicator_row_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        """
        Validate one retained exact-candidate address and keep it compact.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each retained exact candidate needs only per-indicator row indexes to rebuild its
            Stage A `final_signal` deterministically during exact evaluation.
        Raises:
            ValueError: If the retained address is empty or contains negative row indexes.
        Side Effects:
            Normalizes `indicator_row_indexes` to builtin `int` values.
        """
        normalized_row_indexes = tuple(int(value) for value in self.indicator_row_indexes)
        if len(normalized_row_indexes) == 0:
            raise ValueError(
                "_RetainedExactCandidateAddressV2.indicator_row_indexes must be non-empty"
            )
        if any(value < 0 for value in normalized_row_indexes):
            raise ValueError(
                "_RetainedExactCandidateAddressV2.indicator_row_indexes must be >= 0"
            )
        object.__setattr__(self, "indicator_row_indexes", normalized_row_indexes)


@dataclass(frozen=True, slots=True)
class _RetainedExactCandidateV2:
    """
    One deterministic combo proxy prefilter survivor retained for exact candidate evaluation.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py
    """

    base_variant: BacktestStageABaseVariantV2
    proxy_score: float
    retained_address: _RetainedExactCandidateAddressV2

    def __post_init__(self) -> None:
        """
        Validate one retained exact candidate emitted by combo proxy prefilter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The retained frontier stores only minimal row-address metadata so exact scoring can
            reload deterministic signal rows without retaining full `final_signal_row` buffers.
        Raises:
            ValueError: If the proxy score is non-finite.
        Side Effects:
            Normalizes `proxy_score` to builtin `float`.
        """
        proxy_score = float(self.proxy_score)
        if not math.isfinite(proxy_score):
            raise ValueError("_RetainedExactCandidateV2.proxy_score must be finite")
        object.__setattr__(self, "proxy_score", proxy_score)

    def sort_key(self) -> tuple[float, int, str]:
        """
        Return one explicit deterministic retained-frontier ordering key.

        Args:
            None.
        Returns:
            tuple[float, int, str]: Descending proxy-score order with explicit stable tie-breaks.
        Assumptions:
            The combo proxy prefilter must keep retained frontier ordering reviewable and
            reproducible across chunk sizes.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (
            -self.proxy_score,
            self.base_variant.stage_a_index,
            self.base_variant.base_variant_key,
        )


@dataclass(frozen=True, slots=True)
class StageARetainedFrontierMemoryShapeV2:
    """
    Contract-level memory-shape snapshot for the Stage A retained frontier.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-engine-vnext.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    candidate_count: int
    indicator_count_per_candidate: int
    retained_address_value_count: int
    signal_bar_count: int
    legacy_final_signal_value_count: int
    legacy_to_address_value_ratio: float | None
    stores_full_final_signal_rows: bool

    def __post_init__(self) -> None:
        """
        Validate one additive retained-frontier memory-shape snapshot.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            This snapshot measures contract-level retained payload shape only; it does not claim to
            model full Python object overhead or process RSS.
        Raises:
            ValueError: If counts are negative or the legacy-to-address ratio is invalid.
        Side Effects:
            None.
        """
        if self.candidate_count < 0:
            raise ValueError("StageARetainedFrontierMemoryShapeV2.candidate_count must be >= 0")
        if self.indicator_count_per_candidate < 0:
            raise ValueError(
                "StageARetainedFrontierMemoryShapeV2.indicator_count_per_candidate must be >= 0"
            )
        if self.retained_address_value_count < 0:
            raise ValueError(
                "StageARetainedFrontierMemoryShapeV2.retained_address_value_count must be >= 0"
            )
        if self.signal_bar_count < 0:
            raise ValueError(
                "StageARetainedFrontierMemoryShapeV2.signal_bar_count must be >= 0"
            )
        if self.legacy_final_signal_value_count < 0:
            raise ValueError(
                "StageARetainedFrontierMemoryShapeV2.legacy_final_signal_value_count must be >= 0"
            )
        if (
            self.legacy_to_address_value_ratio is not None
            and self.legacy_to_address_value_ratio <= 0.0
        ):
            raise ValueError(
                "StageARetainedFrontierMemoryShapeV2.legacy_to_address_value_ratio must be > 0"
            )


def describe_stage_a_retained_frontier_memory_shape_v2(
    *,
    retained_exact_candidates: Sequence[_RetainedExactCandidateV2],
    signal_bar_count: int,
) -> StageARetainedFrontierMemoryShapeV2:
    """
    Describe the retained frontier `memory shape` after full `final_signal_row` removal.

    Args:
        retained_exact_candidates: Deterministic retained exact candidates emitted by Stage A.
        signal_bar_count: Signal-timeline bar count that the legacy retained contract would have
            stored per survivor as a full `final_signal_row`.
    Returns:
        StageARetainedFrontierMemoryShapeV2: Additive benchmark evidence comparing retained
            addressing cardinality against the removed legacy retained payload shape.
    Assumptions:
        Exact candidates now retain only per-indicator row addresses, while the legacy contract
        retained one full `final_signal_row` value per signal bar and survivor.
    Raises:
        ValueError: If `signal_bar_count` is negative or candidate address widths are inconsistent.
    Side Effects:
        None.
    """
    if signal_bar_count < 0:
        raise ValueError("Stage A retained frontier signal_bar_count must be >= 0")
    candidate_count = len(retained_exact_candidates)
    if candidate_count == 0:
        return StageARetainedFrontierMemoryShapeV2(
            candidate_count=0,
            indicator_count_per_candidate=0,
            retained_address_value_count=0,
            signal_bar_count=int(signal_bar_count),
            legacy_final_signal_value_count=0,
            legacy_to_address_value_ratio=None,
            stores_full_final_signal_rows=False,
        )
    indicator_count_per_candidate = len(
        retained_exact_candidates[0].retained_address.indicator_row_indexes
    )
    for candidate in retained_exact_candidates[1:]:
        if (
            len(candidate.retained_address.indicator_row_indexes)
            != indicator_count_per_candidate
        ):
            raise ValueError(
                "Stage A retained frontier requires uniform retained address width"
            )
    retained_address_value_count = candidate_count * indicator_count_per_candidate
    legacy_final_signal_value_count = candidate_count * int(signal_bar_count)
    legacy_to_address_value_ratio = (
        None
        if retained_address_value_count == 0 or legacy_final_signal_value_count == 0
        else float(legacy_final_signal_value_count) / float(retained_address_value_count)
    )
    return StageARetainedFrontierMemoryShapeV2(
        candidate_count=candidate_count,
        indicator_count_per_candidate=indicator_count_per_candidate,
        retained_address_value_count=retained_address_value_count,
        signal_bar_count=int(signal_bar_count),
        legacy_final_signal_value_count=legacy_final_signal_value_count,
        legacy_to_address_value_ratio=legacy_to_address_value_ratio,
        stores_full_final_signal_rows=False,
    )


@dataclass(frozen=True, slots=True)
class StageAStreamingExactRuntimeShapeV2:
    """
    Additive runtime-shape snapshot for Stage A streaming exact scoring.

    Docs:
      - docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py
    """

    exact_scoring_mode: str
    retained_chunk_count: int
    retained_candidate_count: int
    max_retained_chunk_size: int
    deferred_replay_count: int
    execution_shape: str = "single-process parallel Stage A"
    frontier_compute_mode: str = "kernel-driven"
    stage_a_workers: int | None = None
    numba_threads_used: int | None = None

    def __post_init__(self) -> None:
        """
        Validate one additive Stage A runtime-shape snapshot.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stage A exact work should report streaming exact scoring over retained chunk batches,
            and no deferred replay remains in the active trade-list-first path.
        Raises:
            ValueError: If one count is negative or the mode literal is empty.
        Side Effects:
            None.
        """
        if not self.exact_scoring_mode:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.exact_scoring_mode must be non-empty"
            )
        if self.retained_chunk_count < 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.retained_chunk_count must be >= 0"
            )
        if self.retained_candidate_count < 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.retained_candidate_count must be >= 0"
            )
        if self.max_retained_chunk_size < 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.max_retained_chunk_size must be >= 0"
            )
        if self.deferred_replay_count < 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.deferred_replay_count must be >= 0"
            )
        if not self.execution_shape:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.execution_shape must be non-empty"
            )
        if not self.frontier_compute_mode:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.frontier_compute_mode must be non-empty"
            )
        if self.stage_a_workers is not None and self.stage_a_workers <= 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.stage_a_workers must be > 0 when provided"
            )
        if self.numba_threads_used is not None and self.numba_threads_used <= 0:
            raise ValueError(
                "StageAStreamingExactRuntimeShapeV2.numba_threads_used must be > 0 when "
                "provided"
            )


def describe_stage_a_streaming_exact_runtime_shape_v2(
    *,
    retained_chunk_sizes: Sequence[int],
    stage_a_workers: int | None = None,
    numba_threads_used: int | None = None,
) -> StageAStreamingExactRuntimeShapeV2:
    """
    Describe Stage A streaming exact scoring shape for perf-smoke benchmarks.

    Args:
        retained_chunk_sizes: Exact retained chunk sizes observed as Stage A streams trade-list-
            first exact scoring into the shortlist heap.
        stage_a_workers: Optional configured Stage A worker budget for the measured run.
        numba_threads_used: Optional effective in-process Numba thread count observed live.
    Returns:
        StageAStreamingExactRuntimeShapeV2: Additive runtime-shape evidence for streaming exact
            scoring with no deferred replay.
    Assumptions:
        The active Stage A path exact-scores each retained chunk immediately, keeps deferred
        replay count at zero, and remains a single-process kernel-driven frontier when
        `stage_a_workers` and `numba_threads_used` are supplied.
    Raises:
        ValueError: If one retained chunk size is negative.
    Side Effects:
        None.
    """
    normalized_chunk_sizes = tuple(int(value) for value in retained_chunk_sizes)
    if any(value < 0 for value in normalized_chunk_sizes):
        raise ValueError(
            "Stage A streaming exact scoring retained_chunk_sizes must be >= 0"
        )
    return StageAStreamingExactRuntimeShapeV2(
        exact_scoring_mode="streaming exact scoring",
        retained_chunk_count=len(normalized_chunk_sizes),
        retained_candidate_count=sum(normalized_chunk_sizes),
        max_retained_chunk_size=max(normalized_chunk_sizes, default=0),
        deferred_replay_count=0,
        stage_a_workers=stage_a_workers,
        numba_threads_used=numba_threads_used,
    )


def _retained_exact_candidate_addresses_for_chunk_v2(
    *,
    chunk_inputs: Sequence[PreparedIndicatorChunkInputsV2],
) -> tuple[_RetainedExactCandidateAddressV2, ...]:
    """
    Transpose per-indicator chunk selections into per-candidate retained row addresses.

    Args:
        chunk_inputs: Per-indicator Stage A chunk inputs aligned to the same variant order.
    Returns:
        tuple[_RetainedExactCandidateAddressV2, ...]: One retained row-address payload per chunk
            variant in deterministic Stage A order.
    Assumptions:
        Every prepared indicator input exposes the same variant count and preserves the authored
        indicator-plan order used by Stage A aggregation.
    Raises:
        ValueError: If the chunk is empty or indicator chunk sizes drift.
    Side Effects:
        None.
    """
    if len(chunk_inputs) == 0:
        raise ValueError("Stage A retained address derivation requires non-empty chunk_inputs")
    chunk_row_count = len(chunk_inputs[0].signal_row_selection)
    if chunk_row_count == 0:
        raise ValueError("Stage A retained address derivation requires at least one chunk row")
    for chunk_input in chunk_inputs[1:]:
        if len(chunk_input.signal_row_selection) != chunk_row_count:
            raise ValueError(
                "Stage A retained address derivation requires aligned signal_row_selection sizes"
            )
    return tuple(
        _RetainedExactCandidateAddressV2(
            indicator_row_indexes=tuple(int(value) for value in row_indexes)
        )
        for row_indexes in zip(
            *(chunk_input.signal_row_selection for chunk_input in chunk_inputs),
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class BacktestStageAShortlistBuilderV2:
    """
    Build deterministic Stage A shortlist rows with row-local prefilter before exact path.

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
    row_scorer: GenericRowScorerV2 = field(default_factory=GenericRowScorerV2)
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
        if self.row_scorer is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStageAShortlistBuilderV2 requires row_scorer")
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
        parallelism: BacktestStageAParallelismConfigV1 | None = None,
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
            parallelism:
                Optional resolved Stage A parallel contract carrying `stage_a_workers` and the
                effective Stage A Numba thread cap.
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
        effective_parallelism = (
            parallelism
            or resolve_backtest_stage_a_parallelism_v1(
                execution_profile=getattr(grid_context, "execution_profile", None)
            )
        )
        ranking_plan = resolve_ranking_plan_v2(
            ranking=effective_ranking_config_v2(
                ranking=ranking,
                configurable_ranking_enabled=self.configurable_ranking_enabled,
            )
        )
        with backtest_stage_a_numba_threads_scope_v1(parallelism=effective_parallelism):
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
            local_signal_close = np.asarray(
                signal_prices.ohlcv[signal_target_slice, 3],
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
            row_prefilter_frontier = self._build_row_prefilter_frontier(
                row_plans=row_plans,
                grid_context=grid_context,
                artifact_context=artifact_context,
                signal_target_slice=signal_target_slice,
                local_signal_close=local_signal_close,
                execution_params=execution_params,
                shortlist_limit=shortlist_limit,
                cancel_checker=cancel_checker,
            )

            shortlist_heap: list[StageAHeapEntryV2] = []
            self._stream_combo_proxy_exact_chunks_into_heap(
                row_plans=row_plans,
                grid_context=grid_context,
                artifact_context=artifact_context,
                signal_target_slice=signal_target_slice,
                local_signal_close=local_signal_close,
                local_bar_close_1m_idx=local_bar_close_1m_idx,
                sentinel_index=sentinel_index,
                local_exec_open=local_exec_open,
                local_exec_close=local_exec_close,
                execution_params=execution_params,
                row_prefilter_frontier=row_prefilter_frontier,
                ranking_plan=ranking_plan,
                shortlist_limit=shortlist_limit,
                shortlist_heap=shortlist_heap,
                retained_chunk_limit=self._target_combo_prefilter_exact_candidates(
                    grid_context=grid_context,
                    shortlist_limit=shortlist_limit,
                ),
                batch_size=effective_batch_size,
                cancel_checker=cancel_checker,
                on_checkpoint=on_checkpoint,
            )
            return stage_a_rows_from_heap_v2(heap=shortlist_heap)

    def _build_row_prefilter_frontier(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        shortlist_limit: int,
        cancel_checker: StageACancelCheckerV2 | None,
    ) -> Mapping[str, RetainedIndicatorRowFrontierV2]:
        """
        Build the deterministic retained frontier for row-local prefilter before exact path.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            grid_context: Stage A runtime plan owning indicator ordering and optional caches.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            signal_target_slice: Target request slice in the signal timeline.
            local_signal_close: Request-timeframe close prices aligned to `signal_target_slice`.
            execution_params: Immutable execution settings used for fee-aware proxy scoring.
            shortlist_limit: Final Stage A shortlist cap used to bound retained compute rows.
            cancel_checker: Optional cooperative cancellation callback by stage literal.
        Returns:
            Mapping[str, RetainedIndicatorRowFrontierV2]: Immutable retained frontier keyed by
                indicator id with explicit ranked row ordering.
        Assumptions:
            Each indicator family can be ranked independently with cheap row-local scoring before
            the existing exact Stage A path evaluates retained combinations.
        Raises:
            ValueError: If one indicator row pool is empty or artifact row shapes drift.
        Side Effects:
            Reads deterministic signal-row subsets and optional additive `signal_features` rows.
        """
        target_compute_variants = self._target_prefilter_compute_variants(
            grid_context=grid_context,
            shortlist_limit=shortlist_limit,
        )
        retained_row_limits = _retained_row_limits_v2(
            row_variants=tuple(
                int(math.prod(row_plan.axis_radices)) for row_plan in row_plans
            ),
            target_compute_variants=target_compute_variants,
        )
        scorer = self._row_scorer_for_grid_context(grid_context=grid_context)
        retained_frontier: dict[str, RetainedIndicatorRowFrontierV2] = {}
        for row_plan, retained_rows_limit in zip(
            row_plans,
            retained_row_limits,
            strict=True,
        ):
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL_V2)
            retained_frontier[row_plan.indicator_id] = RetainedIndicatorRowFrontierV2(
                indicator_id=row_plan.indicator_id,
                retained_row_indexes=self._retain_indicator_rows(
                    row_plan=row_plan,
                    grid_context=grid_context,
                    artifact_context=artifact_context,
                    signal_target_slice=signal_target_slice,
                    local_signal_close=local_signal_close,
                    execution_params=execution_params,
                    retained_rows_limit=retained_rows_limit,
                    scorer=scorer,
                ),
            )
        return MappingProxyType(retained_frontier)

    def _stream_combo_proxy_exact_chunks_into_heap(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
        local_signal_close: np.ndarray,
        local_bar_close_1m_idx: np.ndarray,
        sentinel_index: int,
        local_exec_open: np.ndarray,
        local_exec_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        row_prefilter_frontier: Mapping[str, RetainedIndicatorRowFrontierV2],
        ranking_plan: ResolvedRankingPlanV2,
        shortlist_limit: int,
        shortlist_heap: list[StageAHeapEntryV2],
        retained_chunk_limit: int,
        batch_size: int,
        cancel_checker: StageACancelCheckerV2 | None,
        on_checkpoint: StageACheckpointCallbackV2 | None,
    ) -> None:
        """
        Stream Stage A exact scoring per retained combo chunk with no deferred replay.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            grid_context: Stage A runtime plan owning deterministic variant enumeration.
            artifact_context: Slot-pinned runtime context for strict signal-row reads.
            signal_target_slice: Target request slice in the signal timeline.
            local_signal_close: Request-timeframe close prices aligned to `signal_target_slice`.
            local_bar_close_1m_idx: Rebases `bar_close_1m_idx` for the local execution window.
            sentinel_index: Local execution sentinel index.
            local_exec_open: Local execution-bar open prices.
            local_exec_close: Local execution-bar close prices.
            execution_params: Immutable no-risk execution settings.
            row_prefilter_frontier: Deterministic per-indicator retained row frontier.
            ranking_plan: Pre-resolved staged ranking plan from shared Stage A machinery.
            shortlist_limit: Maximum retained shortlist size.
            shortlist_heap: Mutable bounded shortlist heap updated in place.
            retained_chunk_limit: Maximum exact-candidate count retained inside one combo chunk.
            batch_size: Deterministic chunk size used while scanning Stage A variants.
            cancel_checker: Optional cooperative cancellation callback by stage literal.
            on_checkpoint: Optional progress callback `(processed, total)` after each scan chunk.
        Returns:
            None.
        Assumptions:
            Combo proxy prefilter narrows one retained chunk at a time, then Stage A exact work
            runs trade-list-first immediately so no deferred replay batch remains on the active
            path.
        Raises:
            ValueError: If the retained chunk limit is non-positive.
        Side Effects:
            Reads retained signal-row subsets, exact-scores each retained chunk immediately, and
            mutates `shortlist_heap` in place.
        """
        if retained_chunk_limit <= 0:
            raise ValueError(
                "Stage A combo proxy prefilter requires retained_chunk_limit > 0"
            )
        chunk_variants: list[BacktestStageABaseVariantV2] = []
        total = int(grid_context.stage_a_variants_total)
        processed = 0

        for base_variant in grid_context.iter_stage_a_variants():
            chunk_variants.append(base_variant)
            if (
                len(chunk_variants) < batch_size
                and (processed + len(chunk_variants)) < total
            ):
                continue
            if cancel_checker is not None:
                cancel_checker(STAGE_A_LITERAL_V2)
            retained_chunk_variants = self._filter_chunk_variants_by_row_prefilter(
                row_plans=row_plans,
                chunk_variants=chunk_variants,
                row_prefilter_frontier=row_prefilter_frontier,
            )
            if retained_chunk_variants:
                chunk_inputs = self.load_chunk_runtime_inputs(
                    row_plans=row_plans,
                    chunk_variants=retained_chunk_variants,
                    grid_context=grid_context,
                    artifact_context=artifact_context,
                    signal_target_slice=signal_target_slice,
                )
                final_signal = aggregate_ordered_final_signal_rows_v2(
                    ordered_signal_rows=tuple(
                        prepared_input.signal_rows for prepared_input in chunk_inputs
                    ),
                    indicator_ids=tuple(
                        prepared_input.indicator_id for prepared_input in chunk_inputs
                    ),
                )
                retained_row_indexes = self._select_combo_proxy_retained_chunk_row_indexes(
                    chunk_variants=retained_chunk_variants,
                    final_signal=final_signal,
                    local_signal_close=local_signal_close,
                    execution_params=execution_params,
                    retained_chunk_limit=retained_chunk_limit,
                )
                if retained_row_indexes:
                    retained_row_selection = np.asarray(
                        retained_row_indexes,
                        dtype=np.int64,
                    )
                    self._merge_retained_exact_payload_chunk_into_heap(
                        chunk_variants=tuple(
                            retained_chunk_variants[row_index]
                            for row_index in retained_row_indexes
                        ),
                        final_signal=np.ascontiguousarray(
                            final_signal[retained_row_selection, :],
                            dtype=np.int8,
                        ),
                        grid_context=grid_context,
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

    def _retain_indicator_rows(
        self,
        *,
        row_plan: PreparedIndicatorRowPlanV2,
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        signal_target_slice: slice,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        retained_rows_limit: int,
        scorer: GenericRowScorerV2,
    ) -> tuple[int, ...]:
        """
        Rank one indicator family's rows and retain the deterministic frontier for exact work.

        Args:
            row_plan: Prepared row-addressing metadata for one indicator family.
            grid_context: Stage A runtime plan owning timeframe and optional feature access.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            signal_target_slice: Target request slice in the signal timeline.
            local_signal_close: Request-timeframe close prices aligned to `signal_target_slice`.
            execution_params: Immutable execution settings used for fee-aware proxy scoring.
            retained_rows_limit: Maximum retained row count for this indicator family.
            scorer: Rehydrated generic row scorer used for structural row-local ranking.
        Returns:
            tuple[int, ...]: Deterministic retained row indexes in ranked frontier order.
        Assumptions:
            The retained frontier uses a cheap price-aware proxy with the generic row scorer as an
            explicit deterministic tie-break.
        Raises:
            ValueError: If the indicator pool is empty or one loaded row payload is malformed.
        Side Effects:
            Reads one indicator's selected signal rows and optional additive feature rows.
        """
        total_rows = int(math.prod(row_plan.axis_radices))
        row_indexes = np.arange(total_rows, dtype=np.int64)
        signal_rows = _load_chunk_signal_rows_v2(
            signal_matrix_loader=self.signal_matrix_loader,
            artifact_context=artifact_context,
            timeframe=grid_context.timeframe_code,
            indicator_id=row_plan.indicator_id,
            indicator_row_indexes=row_indexes,
            signal_target_slice=signal_target_slice,
        )
        signal_feature_rows = self._load_prefilter_signal_feature_rows(
            grid_context=grid_context,
            artifact_context=artifact_context,
            indicator_id=row_plan.indicator_id,
            row_indexes=tuple(int(value) for value in row_indexes.tolist()),
        )
        scored_rows = scorer.score_rows(
            rows=self._row_scoring_inputs_for_prefilter(
                indicator_id=row_plan.indicator_id,
                signal_rows=signal_rows,
                signal_feature_rows=signal_feature_rows,
            )
        )
        proxy_scores = self._prefilter_proxy_scores_for_rows(
            signal_rows=signal_rows,
            local_signal_close=local_signal_close,
            execution_params=execution_params,
        )
        scorer_sorted_row_indexes = np.asarray(
            [payload.row_index for payload in scored_rows],
            dtype=np.int64,
        )
        stable_proxy_order = np.argsort(
            -proxy_scores[scorer_sorted_row_indexes],
            kind="mergesort",
        )
        retained_count = min(retained_rows_limit, len(scored_rows))
        if retained_count <= 0:
            raise ValueError(
                f"Stage A row prefilter retained no rows for {row_plan.indicator_id!r}"
            )
        retained_row_indexes = scorer_sorted_row_indexes[stable_proxy_order[:retained_count]]
        return tuple(int(row_index) for row_index in retained_row_indexes.tolist())

    def _select_combo_proxy_retained_chunk_row_indexes(
        self,
        *,
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        final_signal: np.ndarray,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        retained_chunk_limit: int,
    ) -> tuple[int, ...]:
        """
        Select one deterministic retained chunk for immediate Stage A exact scoring.

        Args:
            chunk_variants: Deterministic Stage A base variants surviving row prefilter.
            final_signal: Aggregated Stage A `final_signal[V, T_signal]` for the same variants.
            local_signal_close: Request-timeframe close prices aligned to `final_signal`.
            execution_params: Immutable execution settings supplying the fee penalty.
            retained_chunk_limit: Maximum exact-candidate count retained inside one combo chunk.
        Returns:
            tuple[int, ...]: Selected chunk-local row indexes in original Stage A chunk order.
        Assumptions:
            Combo proxy prefilter narrows only the current retained chunk, while exact scoring
            remains authoritative because retained rows flow immediately into the trade-list-first
            path with no deferred replay.
        Raises:
            ValueError: If the retained chunk limit is non-positive or `final_signal` drifts from
                `chunk_variants`.
        Side Effects:
            None.
        """
        if retained_chunk_limit <= 0:
            raise ValueError(
                "Stage A combo proxy prefilter requires retained_chunk_limit > 0"
            )
        if int(final_signal.shape[0]) != len(chunk_variants):
            raise ValueError(
                "Stage A combo proxy prefilter requires final_signal rows to match "
                "chunk_variants"
            )
        proxy_scores = self._prefilter_proxy_scores_for_rows(
            signal_rows=final_signal,
            local_signal_close=local_signal_close,
            execution_params=execution_params,
            error_prefix="Stage A combo proxy prefilter",
        )
        stage_a_indexes = np.fromiter(
            (variant.stage_a_index for variant in chunk_variants),
            dtype=np.int64,
            count=len(chunk_variants),
        )
        stage_order = np.argsort(stage_a_indexes, kind="mergesort")
        ranked_row_indexes = stage_order[
            np.argsort(-proxy_scores[stage_order], kind="mergesort")
        ]
        retained_row_count = min(len(chunk_variants), retained_chunk_limit)
        selected_row_indexes = np.sort(
            np.asarray(
                ranked_row_indexes[:retained_row_count],
                dtype=np.int64,
            ),
            kind="mergesort",
        )
        return tuple(int(row_index) for row_index in selected_row_indexes.tolist())

    def _row_scoring_inputs_for_prefilter(
        self,
        *,
        indicator_id: str,
        signal_rows: np.ndarray,
        signal_feature_rows: ArtifactSignalFeaturesRowsV2 | None,
    ) -> tuple[GenericRowScoringInputV2, ...]:
        """
        Build deterministic row-scoring inputs for one indicator family's prefilter pool.

        Args:
            indicator_id: Indicator id whose artifact rows are being ranked.
            signal_rows: Target-sliced signal rows in artifact row order.
            signal_feature_rows: Optional additive cached feature rows in the same order.
        Returns:
            tuple[GenericRowScoringInputV2, ...]: Deterministic row-local scoring inputs.
        Assumptions:
            Optional feature rows stay strictly 1:1 aligned with `signal_rows` when available.
        Raises:
            ValueError: If the signal rows are not 2D or feature-row alignment drifts.
        Side Effects:
            None.
        """
        if signal_rows.ndim != 2:
            raise ValueError("Stage A row prefilter requires 2D signal_rows")
        if (
            signal_feature_rows is not None
            and signal_feature_rows.rows.shape[0] != signal_rows.shape[0]
        ):
            raise ValueError(
                "Stage A row prefilter requires signal_features rows to align with signal_rows"
            )
        return tuple(
            GenericRowScoringInputV2(
                indicator_id=indicator_id,
                row_index=row_index,
                stable_identity=f"{indicator_id}:{row_index}",
                signal_row=np.asarray(signal_rows[row_index, :], dtype=np.int8),
                signal_features=(
                    build_generic_row_signal_features_mapping_v2(
                        feature_names=signal_feature_rows.feature_names,
                        feature_values=tuple(
                            float(value)
                            for value in signal_feature_rows.rows[row_index, :].tolist()
                        ),
                    )
                    if signal_feature_rows is not None
                    else None
                ),
            )
            for row_index in range(int(signal_rows.shape[0]))
        )

    def _load_prefilter_signal_feature_rows(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        indicator_id: str,
        row_indexes: tuple[int, ...],
    ) -> ArtifactSignalFeaturesRowsV2 | None:
        """
        Load optional cached feature rows for one indicator family's retained frontier scoring.

        Args:
            grid_context: Stage A runtime plan owning optional feature-access metadata.
            artifact_context: Slot-pinned runtime context for strict artifact reads.
            indicator_id: Indicator id whose feature rows are requested.
            row_indexes: Deterministic row indexes aligned to the prefilter signal-row pool.
        Returns:
            ArtifactSignalFeaturesRowsV2 | None: Selected feature rows or `None` when optional
                warm-cache access is unavailable.
        Assumptions:
            `signal_features` remain non-mandatory for this milestone and must fall back to
            runtime row-local derivation when absent.
        Raises:
            FileNotFoundError: If a declared strict feature family is missing on disk.
            ValueError: If feature-row selection metadata is invalid.
        Side Effects:
            May memory-map one additive `signal_features` matrix on first explicit access.
        """
        signal_features_access = _signal_features_access_plan_for_indicator_v2(
            grid_context=grid_context,
            indicator_id=indicator_id,
        )
        if signal_features_access is None or self.signal_features_loader is None:
            return None
        if signal_features_access.optional:
            return self.signal_features_loader.try_load_signal_feature_rows(
                context=artifact_context,
                timeframe=signal_features_access.timeframe,
                indicator_id=indicator_id,
                row_selection=row_indexes,
            )
        return self.signal_features_loader.load_signal_feature_rows(
            context=artifact_context,
            timeframe=signal_features_access.timeframe,
            indicator_id=indicator_id,
            row_selection=row_indexes,
        )

    def _prefilter_proxy_score_for_row(
        self,
        *,
        row_index: int,
        signal_rows: np.ndarray,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
    ) -> float:
        """
        Compute one cheap deterministic row-local proxy score before exact evaluation.

        Args:
            row_index: Indicator-local row index inside `signal_rows`.
            signal_rows: Target-sliced signal rows in artifact row order.
            local_signal_close: Request-timeframe close prices aligned to `signal_rows` columns.
            execution_params: Immutable execution settings supplying the fee penalty.
        Returns:
            float: Fee-adjusted proxy score for deterministic single-row prefilter ranking.
        Assumptions:
            The cheap proxy approximates next-bar profitability using request-timeframe closes and
            is used only to narrow candidates before the exact path remains authoritative.
        Raises:
            ValueError: If row or price-array shapes drift.
        Side Effects:
            None.
        """
        if row_index < 0 or row_index >= int(signal_rows.shape[0]):
            raise ValueError("Stage A row prefilter row_index is outside signal_rows")
        if signal_rows.ndim != 2:
            raise ValueError("Stage A row prefilter requires 2D signal_rows")
        return self._proxy_score_for_signal_row(
            signal_row=signal_rows[row_index, :],
            local_signal_close=local_signal_close,
            execution_params=execution_params,
            error_prefix="Stage A row prefilter",
        )

    def _prefilter_proxy_scores_for_rows(
        self,
        *,
        signal_rows: np.ndarray,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        error_prefix: str = "Stage A row prefilter",
    ) -> np.ndarray:
        """
        Compute deterministic Stage A row-prefilter proxy scores for one full indicator family.

        Args:
            signal_rows: Target-sliced signal rows in artifact row order.
            local_signal_close: Request-timeframe close prices aligned to `signal_rows`.
            execution_params: Immutable execution settings supplying the fee penalty.
            error_prefix: Error-message prefix describing the calling proxy stage.
        Returns:
            np.ndarray: Float64 proxy score per row in artifact row order.
        Assumptions:
            Both row and combo proxy prefilters should batch score computation across the full
            signal matrix so the hot path is parallel-capable instead of dominated by scalar
            Python work.
        Raises:
            ValueError: If signal rows are not 2D or timeline alignment drifts.
        Side Effects:
            May trigger Numba compilation on first use.
        """
        normalized_signal_rows = np.asarray(signal_rows, dtype=np.int8)
        if normalized_signal_rows.ndim != 2:
            raise ValueError(f"{error_prefix} requires 2D signal_rows")
        if int(normalized_signal_rows.shape[1]) != int(local_signal_close.shape[0]):
            raise ValueError(
                f"{error_prefix} requires signal rows and close prices to share length"
            )
        close_returns = _close_returns_for_proxy_score_v2(
            local_signal_close=local_signal_close,
            error_prefix=error_prefix,
        )
        if close_returns.size == 0:
            return np.zeros(int(normalized_signal_rows.shape[0]), dtype=np.float64)
        return _batch_proxy_scores_for_signal_rows_kernel_v2(
            signal_rows_i8=np.ascontiguousarray(normalized_signal_rows, dtype=np.int8),
            close_returns_f64=np.ascontiguousarray(close_returns, dtype=np.float64),
            fee_rate=float(execution_params.fee_rate),
        )

    def _proxy_score_for_signal_row(
        self,
        *,
        signal_row: np.ndarray,
        local_signal_close: np.ndarray,
        execution_params: ExecutionParamsV1,
        error_prefix: str,
    ) -> float:
        """
        Compute one cheap deterministic proxy score for a single retained signal row.

        Args:
            signal_row: One deterministic signal timeline in `{-1, 0, 1}` order.
            local_signal_close: Request-timeframe close prices aligned to `signal_row`.
            execution_params: Immutable execution settings supplying the fee penalty.
            error_prefix: Error-message prefix describing the calling prefilter stage.
        Returns:
            float: Fee-adjusted proxy score for deterministic narrowing.
        Assumptions:
            Both row-level and combo-level proxy prefilters reuse the same cheap next-bar proxy so
            later tuning remains benchmarkable and explicit.
        Raises:
            ValueError: If the signal row is not 1D or does not align with close prices.
        Side Effects:
            None.
        """
        normalized_signal_row = np.asarray(signal_row, dtype=np.int8)
        if normalized_signal_row.ndim != 1:
            raise ValueError(f"{error_prefix} requires 1D signal_row")
        if int(normalized_signal_row.shape[0]) != int(local_signal_close.shape[0]):
            raise ValueError(
                f"{error_prefix} requires signal rows and close prices to share length"
            )
        close_returns = _close_returns_for_proxy_score_v2(
            local_signal_close=local_signal_close,
            error_prefix=error_prefix,
        )
        if close_returns.size == 0:
            return 0.0
        eval_signal_row = np.asarray(normalized_signal_row[:-1], dtype=np.float64)
        proxy_score = float(np.dot(eval_signal_row, close_returns))
        fee_rate = float(execution_params.fee_rate)
        activity_penalty = fee_rate * float(np.count_nonzero(eval_signal_row != 0.0))
        return proxy_score - activity_penalty

    def _target_prefilter_compute_variants(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
        shortlist_limit: int,
    ) -> int:
        """
        Resolve how many compute variants the retained frontier should preserve before exact work.

        Args:
            grid_context: Stage A runtime plan owning signal-axis metadata when available.
            shortlist_limit: Final Stage A shortlist cap requested by the caller.
        Returns:
            int: Minimum retained compute-variant budget needed before exact evaluation.
        Assumptions:
            Signal-axis variants survive the row prefilter unchanged, so only compute variants
            need to be budgeted here.
        Raises:
            ValueError: If the computed signal-axis cardinality is non-positive.
        Side Effects:
            None.
        """
        signal_axes = getattr(grid_context, "signal_axes", ())
        signal_variants_total = 1
        for signal_axis in signal_axes:
            signal_variants_total *= len(signal_axis.values)
        if signal_variants_total <= 0:
            raise ValueError(
                "Stage A row prefilter requires positive signal-axis cardinality"
            )
        stage_a_variants_total = int(
            getattr(grid_context, "stage_a_variants_total", shortlist_limit)
        )
        effective_shortlist_limit = min(shortlist_limit, stage_a_variants_total)
        return max(1, int(math.ceil(effective_shortlist_limit / signal_variants_total)))

    def _target_combo_prefilter_exact_candidates(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
        shortlist_limit: int,
    ) -> int:
        """
        Resolve how many exact survivors the combo proxy prefilter should retain per chunk.

        Args:
            grid_context: Stage A runtime plan owning deterministic variant cardinality.
            shortlist_limit: Final Stage A shortlist cap requested by the caller.
        Returns:
            int: Retained chunk budget for immediate exact survivor evaluation.
        Assumptions:
            Combo proxy prefilter should remain conservative by over-retaining a small explicit
            multiple of the final shortlist inside each chunk before Stage A exact scoring runs
            immediately with no deferred replay.
        Raises:
            ValueError: If the requested shortlist limit is non-positive.
        Side Effects:
            None.
        """
        if shortlist_limit <= 0:
            raise ValueError(
                "Stage A combo proxy prefilter requires shortlist_limit > 0"
            )
        stage_a_variants_total = int(
            getattr(grid_context, "stage_a_variants_total", shortlist_limit)
        )
        effective_shortlist_limit = min(shortlist_limit, stage_a_variants_total)
        return min(
            stage_a_variants_total,
            max(
                effective_shortlist_limit,
                effective_shortlist_limit
                * _COMBO_PROXY_PREFILTER_SURVIVOR_MULTIPLIER_V2,
            ),
        )

    def _filter_chunk_variants_by_row_prefilter(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        row_prefilter_frontier: Mapping[str, RetainedIndicatorRowFrontierV2],
    ) -> tuple[BacktestStageABaseVariantV2, ...]:
        """
        Keep only chunk variants whose indicator rows stay inside the retained frontier.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            chunk_variants: Deterministic raw Stage A variants from the current batch.
            row_prefilter_frontier: Retained row membership keyed by indicator id.
        Returns:
            tuple[BacktestStageABaseVariantV2, ...]: Exact-path candidates that survived
                deterministic single-row prefiltering.
        Assumptions:
            Chunk order stays unchanged for retained survivors to preserve deterministic exact
            ordering and checkpoint semantics.
        Raises:
            ValueError: If one retained-frontier indicator id is missing.
        Side Effects:
            None.
        """
        return tuple(
            base_variant
            for base_variant in chunk_variants
            if self._variant_passes_row_prefilter(
                row_plans=row_plans,
                base_variant=base_variant,
                row_prefilter_frontier=row_prefilter_frontier,
            )
        )

    def _variant_passes_row_prefilter(
        self,
        *,
        row_plans: Sequence[PreparedIndicatorRowPlanV2],
        base_variant: BacktestStageABaseVariantV2,
        row_prefilter_frontier: Mapping[str, RetainedIndicatorRowFrontierV2],
    ) -> bool:
        """
        Check whether one Stage A base variant belongs to the retained row frontier.

        Args:
            row_plans: Prepared per-indicator row-addressing plans.
            base_variant: One deterministic Stage A base variant.
            row_prefilter_frontier: Retained row membership keyed by indicator id.
        Returns:
            bool: `True` when every indicator row for the variant survived prefiltering.
        Assumptions:
            The retained frontier is authoritative only for Stage A narrowing; Stage B semantics
            remain unchanged and exact.
        Raises:
            ValueError: If the frontier does not contain one required indicator id.
        Side Effects:
            None.
        """
        for plan_position, row_plan in enumerate(row_plans):
            retained_frontier = row_prefilter_frontier.get(row_plan.indicator_id)
            if retained_frontier is None:
                raise ValueError(
                    "Stage A row prefilter is missing retained rows for "
                    f"{row_plan.indicator_id!r}"
                )
            row_index = row_plan.row_index_for_selection(
                selection=_indicator_selection_for_plan_v2(
                    base_variant=base_variant,
                    indicator_position=plan_position,
                    indicator_id=row_plan.indicator_id,
                )
            )
            if not retained_frontier.contains_row_index(row_index=row_index):
                return False
        return True

    def _row_scorer_for_grid_context(
        self,
        *,
        grid_context: BacktestArtifactRuntimePlanV2,
    ) -> GenericRowScorerV2:
        """
        Rehydrate the generic row scorer with runtime-plan shortlist weights when available.

        Args:
            grid_context: Stage A runtime plan that may expose execution-profile shortlist weights.
        Returns:
            GenericRowScorerV2: Deterministic row scorer used by row-local prefiltering.
        Assumptions:
            Threshold literals remain owned by the builder, while runtime profiles may override
            additive shortlist-scoring weights.
        Raises:
            None.
        Side Effects:
            None.
        """
        execution_profile = getattr(grid_context, "execution_profile", None)
        shortlist_config = getattr(execution_profile, "shortlist_config", None)
        scoring = (
            shortlist_config.scoring
            if shortlist_config is not None
            else self.row_scorer.scoring
        )
        return GenericRowScorerV2(
            scoring=scoring,
            low_activity_threshold=self.row_scorer.low_activity_threshold,
            high_activity_threshold=self.row_scorer.high_activity_threshold,
            direction_balance_threshold=self.row_scorer.direction_balance_threshold,
            low_transition_ratio_threshold=self.row_scorer.low_transition_ratio_threshold,
            high_transition_ratio_threshold=self.row_scorer.high_transition_ratio_threshold,
        )

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
                    signal_row_selection=row_indexes,
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

    def _build_retained_exact_batch(
        self,
        *,
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        final_signal: np.ndarray,
        local_bar_close_1m_idx: np.ndarray,
        sentinel_index: int,
        direction_mode: str,
    ) -> _CompactTradeBatchV2:
        """
        Build dense internal trade-list-first batch state only for retained candidates.

        Args:
            chunk_variants: Retained exact candidates aligned to `final_signal`.
            final_signal: Prepared Stage A `final_signal[V, T_signal]` rows for exact work.
            local_bar_close_1m_idx: Rebases `bar_close_1m_idx` for the local execution window.
            sentinel_index: Local execution sentinel index.
            direction_mode: Strategy direction policy used by compact-trade construction.
        Returns:
            _CompactTradeBatchV2: Dense internal exact batch aligned to `chunk_variants`.
        Assumptions:
            Trade-list-first remains internal-only and is built only after row and combo
            prefiltering retain the current chunk for streaming exact scoring with no deferred
            replay.
        Raises:
            ValueError: If `final_signal` row count drifts from `chunk_variants`.
        Side Effects:
            May trigger Numba compilation on first use.
        """
        if int(final_signal.shape[0]) != len(chunk_variants):
            raise ValueError(
                "Stage A exact retained-candidate evaluation requires final_signal rows to "
                "match chunk_variants"
            )
        return build_compact_trade_batch_v2(
            final_signal=final_signal,
            bar_close_1m_idx=local_bar_close_1m_idx,
            sentinel_index=sentinel_index,
            direction_mode=direction_mode,
        )

    def _merge_retained_exact_payload_chunk_into_heap(
        self,
        *,
        chunk_variants: Sequence[BacktestStageABaseVariantV2],
        final_signal: np.ndarray,
        grid_context: BacktestArtifactRuntimePlanV2,
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
        Exact-score one prepared `final_signal` chunk and merge results into the shortlist heap.

        Args:
            chunk_variants: Deterministic Stage A base variants aligned to `final_signal`.
            final_signal: Prepared Stage A `final_signal[V, T_signal]` rows for exact payload
                construction.
            grid_context: Stage A grid context with direction-mode metadata.
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
            The combo proxy prefilter narrows candidates first, and only the retained chunk
            receives internal compact exact payload construction before Stage A no-risk ranking,
            with shortlisted rows carrying the same exact no-risk metric payload into the direct
            no-risk finalization path.
        Raises:
            ValueError: If `final_signal` row count drifts from `chunk_variants`.
        Side Effects:
            Mutates `shortlist_heap` in place and materializes internal exact payloads only for
            rows that enter the deterministic shortlist during streaming exact scoring.
        """
        exact_batch = self._build_retained_exact_batch(
            chunk_variants=chunk_variants,
            final_signal=final_signal,
            local_bar_close_1m_idx=local_bar_close_1m_idx,
            sentinel_index=sentinel_index,
            direction_mode=grid_context.direction_mode,
        )
        exact_metrics = compute_no_risk_metrics_for_trade_batch_v2(
            compact_trade_batch=exact_batch,
            exec_open=local_exec_open,
            exec_close=local_exec_close,
            sentinel_index=sentinel_index,
            execution_params=execution_params,
        )
        for row_index, (base_variant, metrics) in enumerate(
            zip(chunk_variants, exact_metrics, strict=True)
        ):
            row = BacktestStageAScoredVariantV2(
                base_variant=base_variant,
                total_return_pct=metrics.total_return_pct,
                retained_exact_payload=None,
                no_risk_metrics=metrics,
            )
            ranking_payload = no_risk_metrics_to_ranking_payload_v2(metrics=metrics)
            heap_entry = stage_a_heap_entry_v2(
                row=row,
                metrics=ranking_payload,
                ranking_plan=ranking_plan,
            )
            if len(shortlist_heap) < shortlist_limit:
                heappush(
                    shortlist_heap,
                    stage_a_heap_entry_v2(
                        row=BacktestStageAScoredVariantV2(
                            base_variant=base_variant,
                            total_return_pct=metrics.total_return_pct,
                            retained_exact_payload=exact_batch.exact_payload_at(
                                row_index=row_index
                            ),
                            no_risk_metrics=metrics,
                        ),
                        metrics=ranking_payload,
                        ranking_plan=ranking_plan,
                    ),
                )
                continue
            if heap_entry_outranks_v2(candidate=heap_entry, baseline=shortlist_heap[0]):
                heapreplace(
                    shortlist_heap,
                    stage_a_heap_entry_v2(
                        row=BacktestStageAScoredVariantV2(
                            base_variant=base_variant,
                            total_return_pct=metrics.total_return_pct,
                            retained_exact_payload=exact_batch.exact_payload_at(
                                row_index=row_index
                            ),
                            no_risk_metrics=metrics,
                        ),
                        metrics=ranking_payload,
                        ranking_plan=ranking_plan,
                    ),
                )


def _retained_row_limits_v2(
    *,
    row_variants: Sequence[int],
    target_compute_variants: int,
) -> tuple[int, ...]:
    """
    Resolve deterministic per-indicator retained-row caps for the retained frontier.

    Args:
        row_variants: Indicator-local row counts in the original planner order.
        target_compute_variants: Minimum compute-variant budget that should survive prefiltering.
    Returns:
        tuple[int, ...]: Deterministic retained-row limits aligned to `row_variants`.
    Assumptions:
        The retained frontier should stay as small as practical while still preserving at least
        the requested compute-variant budget before exact evaluation.
    Raises:
        ValueError: If one row count or the target budget is non-positive.
    Side Effects:
        None.
    """
    if target_compute_variants <= 0:
        raise ValueError("Stage A retained frontier target_compute_variants must be > 0")
    if len(row_variants) == 0:
        return ()
    normalized_variants = tuple(int(value) for value in row_variants)
    if any(value <= 0 for value in normalized_variants):
        raise ValueError("Stage A retained frontier row_variants must all be > 0")
    base_limit = max(
        1,
        int(math.ceil(target_compute_variants ** (1.0 / len(normalized_variants)))),
    )
    retained_limits = [
        min(variants, base_limit) for variants in normalized_variants
    ]
    retained_product = math.prod(retained_limits)
    while retained_product < target_compute_variants:
        grew = False
        for index, variants in enumerate(normalized_variants):
            if retained_limits[index] >= variants:
                continue
            retained_limits[index] += 1
            retained_product = math.prod(retained_limits)
            grew = True
            if retained_product >= target_compute_variants:
                break
        if not grew:
            break
    return tuple(retained_limits)


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
