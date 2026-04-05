"""First proposal-only `MA-family` plugin for internal `hybrid_family` rollout.

Docs:
  - docs/architecture/backtest/backtest-family-accelerators-v1.md
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/indicators/indicators-ma.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Mapping

from trading.contexts.indicators.domain.definitions.ma import defs as ma_indicator_defs

from ..artifact_runtime_plan_v2 import BacktestIndicatorPlanV2
from .contracts_v2 import (
    FamilyAccelerationPluginV2,
    FamilyPluginApplicabilityV2,
    FamilyPluginMetadataV2,
    FamilyPluginPlanningContextV2,
    FamilyPluginProposalResultV2,
    FamilyPluginProxyScoreV2,
)

_SOURCE_PRIORITY_V2: tuple[str, ...] = (
    "close",
    "hlc3",
    "ohlc4",
    "hl2",
    "open",
    "high",
    "low",
    "volume",
)
_CANONICAL_MA_AXIS_NAMES_BY_INDICATOR_ID_V2: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            definition.indicator_id.value: tuple(definition.axes)
            for definition in ma_indicator_defs()
        }
    )
)
_CANONICAL_MA_INDICATOR_IDS_V2 = frozenset(_CANONICAL_MA_AXIS_NAMES_BY_INDICATOR_ID_V2.keys())


@dataclass(frozen=True, slots=True)
class _MAIndicatorRowCandidateV2:
    """
    One deterministic indicator-local MA row candidate selected from planner axis metadata.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py
    """

    variant_index: int
    proxy_score: float


@dataclass(frozen=True, slots=True)
class _MAComputeCandidateV2:
    """
    Deterministic compute candidate built from indicator-local MA row anchors.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py
    """

    variant_indexes: tuple[int, ...]
    proxy_score: float


@dataclass(frozen=True, slots=True)
class MAFamilyAccelerationPluginV2(FamilyAccelerationPluginV2):
    """
    Proposal-only MA-family plugin that samples deterministic MA window anchors.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py
    """

    metadata: FamilyPluginMetadataV2 = field(
        default_factory=lambda: FamilyPluginMetadataV2(
            plugin_id="ma.family.v1",
            display_name="MA-family proposal layer",
            applicability=FamilyPluginApplicabilityV2(
                execution_profile_modes=("hybrid_family",),
                indicator_family_literals=("ma",),
            ),
            proposal_capabilities=("row_shortlist", "proxy_score"),
        )
    )

    def __post_init__(self) -> None:
        """
        Validate that the shipped first plugin stays scoped to internal MA-family rollout.

        Docs:
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/indicators/domain/definitions/ma.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The first concrete plugin must remain narrowly registered for `hybrid_family` plus
            the canonical `ma.` indicator family only.
        Raises:
            ValueError: If shipped metadata drifts from the internal MA-family rollout contract.
        Side Effects:
            None.
        """
        if self.metadata.applicability.execution_profile_modes != ("hybrid_family",):
            raise ValueError(
                "MAFamilyAccelerationPluginV2 must register only for 'hybrid_family'"
            )
        if self.metadata.applicability.indicator_family_literals != ("ma",):
            raise ValueError(
                "MAFamilyAccelerationPluginV2 must register only for the 'ma' family"
            )

    def propose(
        self,
        *,
        context: FamilyPluginPlanningContextV2,
    ) -> FamilyPluginProposalResultV2:
        """
        Propose one deterministic MA-family row shortlist plus reviewable proxy scores.

        Docs:
          - docs/architecture/backtest/backtest-family-accelerators-v1.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            ma_family_plugin_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/indicators/domain/definitions/ma.py

        Args:
            context: Narrow immutable planning context derived from the shared runtime plan.
        Returns:
            FamilyPluginProposalResultV2: Proposal-only `row shortlist` and `proxy score`
                output for exact downstream validation.
        Assumptions:
            The first MA-family plugin remains a proposal layer: it samples deterministic MA row
            anchors from planner axis metadata and never replaces the exact Stage B scorer.
        Raises:
            ValueError: If runtime-plan indicators are outside canonical MA definitions or their
                axis payload drifts from supported MA-family shapes.
        Side Effects:
            None.
        """
        indicator_plans = _ma_runtime_indicator_plans_v2(context=context)
        shortlist_limit = _ma_row_shortlist_limit_v2(context=context)
        signal_variants_total = context.runtime_plan.signal_variants_total()
        target_compute_variants_total = max(1, shortlist_limit // signal_variants_total)
        per_indicator_limit = max(
            1,
            int(
                math.ceil(
                    target_compute_variants_total
                    ** (1.0 / float(max(1, len(indicator_plans))))
                )
            ),
        )
        indicator_candidates = tuple(
            _ma_indicator_row_candidates_v2(
                plan=plan,
                limit=per_indicator_limit,
            )
            for plan in indicator_plans
        )
        compute_candidates = _ma_compute_candidates_v2(
            indicator_candidates=indicator_candidates,
            target_compute_variants_total=target_compute_variants_total,
        )
        row_shortlist, proxy_scores = _ma_stage_a_proposal_v2(
            context=context,
            compute_candidates=compute_candidates,
            shortlist_limit=shortlist_limit,
        )
        return FamilyPluginProposalResultV2(
            plugin_id=self.metadata.plugin_id,
            row_shortlist=row_shortlist,
            proxy_scores=proxy_scores,
        )


def _ma_runtime_indicator_plans_v2(
    *,
    context: FamilyPluginPlanningContextV2,
) -> tuple[BacktestIndicatorPlanV2, ...]:
    """
    Validate that one runtime plan is fully covered by canonical MA-family indicator metadata.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        context: Narrow immutable planning context for the current runtime plan.
    Returns:
        tuple[BacktestIndicatorPlanV2, ...]: Planner-owned indicator plans after MA validation.
    Assumptions:
        The first MA-family plugin is intentionally narrow and may execute only when every
        `indicator_id` belongs to the canonical `ma.` definition set.
    Raises:
        ValueError: If the plan is mixed-family, uses unknown `ma.` ids, or exposes unsupported
            axis names for a canonical MA indicator.
    Side Effects:
        None.
    """
    if context.indicator_family_literal != "ma":
        raise ValueError(
            "MAFamilyAccelerationPluginV2 requires a pure MA-family runtime plan"
        )
    indicator_plans = context.runtime_plan.indicator_plans
    if len(indicator_plans) == 0:
        raise ValueError("MAFamilyAccelerationPluginV2 requires indicator_plans")
    for indicator_plan in indicator_plans:
        if indicator_plan.indicator_id not in _CANONICAL_MA_INDICATOR_IDS_V2:
            raise ValueError(
                "MAFamilyAccelerationPluginV2 supports only canonical MA-family ids, got "
                f"{indicator_plan.indicator_id!r}"
            )
        expected_axis_names = _CANONICAL_MA_AXIS_NAMES_BY_INDICATOR_ID_V2[
            indicator_plan.indicator_id
        ]
        actual_axis_names = tuple(axis.name for axis in indicator_plan.axes)
        if actual_axis_names != expected_axis_names:
            raise ValueError(
                "MAFamilyAccelerationPluginV2 requires axis ordering from canonical MA "
                f"definitions for {indicator_plan.indicator_id!r}; got "
                f"{actual_axis_names!r}, expected {expected_axis_names!r}"
            )
    return indicator_plans


def _ma_row_shortlist_limit_v2(
    *,
    context: FamilyPluginPlanningContextV2,
) -> int:
    """
    Resolve the bounded MA-family proposal shortlist budget from the shared runtime plan.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        context: Narrow immutable planning context for the current runtime plan.
    Returns:
        int: Conservative Stage A row budget reused by the proposal layer.
    Assumptions:
        The MA-family plugin must stay bounded by the shared shortlist envelope so universal
        fallback and exact Stage B authority remain canonical.
    Raises:
        ValueError: If the runtime plan is missing risk variants.
    Side Effects:
        None.
    """
    risk_total = len(context.runtime_plan.risk_variants)
    if risk_total <= 0:
        raise ValueError(
            "MAFamilyAccelerationPluginV2 requires runtime_plan.risk_variants"
        )
    original_shortlist_limit = max(
        1,
        context.runtime_plan.stage_b_variants_total // risk_total,
    )
    profile_max_candidates = (
        context.runtime_plan.execution_profile.shortlist_config.max_candidates
        or context.runtime_plan.stage_a_variants_total
    )
    return min(
        context.runtime_plan.stage_a_variants_total,
        profile_max_candidates,
        original_shortlist_limit,
    )


def _ma_indicator_row_candidates_v2(
    *,
    plan: BacktestIndicatorPlanV2,
    limit: int,
) -> tuple[_MAIndicatorRowCandidateV2, ...]:
    """
    Build deterministic MA indicator-local row anchors from `window` plus preferred `source`.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        plan: Planner-owned indicator plan for one canonical MA indicator.
        limit: Maximum number of indicator-local row anchors to retain.
    Returns:
        tuple[_MAIndicatorRowCandidateV2, ...]: Deterministic row anchors for the indicator.
    Assumptions:
        The first MA-family plugin keeps `source` deterministic and samples only a bounded set of
        `window` anchors instead of materializing the full cartesian space.
    Raises:
        ValueError: If `limit` is non-positive or the plan omits the canonical `window` axis.
    Side Effects:
        None.
    """
    if limit <= 0:
        raise ValueError("MAFamilyAccelerationPluginV2 row candidate limit must be > 0")
    source_axis_values: tuple[int | float | str, ...] | None = None
    window_axis_values: tuple[int | float | str, ...] | None = None
    for axis in plan.axes:
        if axis.name == "source":
            source_axis_values = axis.values
        if axis.name == "window":
            window_axis_values = axis.values
    if window_axis_values is None:
        raise ValueError(
            f"MAFamilyAccelerationPluginV2 requires a window axis for {plan.indicator_id!r}"
        )
    source_coordinate = _ma_source_axis_position_v2(values=source_axis_values)
    window_positions = _ma_window_axis_positions_v2(
        values=window_axis_values,
        limit=limit,
    )
    candidates = tuple(
        _MAIndicatorRowCandidateV2(
            variant_index=_ma_row_variant_index_v2(
                plan=plan,
                source_coordinate=source_coordinate,
                window_coordinate=window_coordinate,
            ),
            proxy_score=_ma_row_proxy_score_v2(
                source_coordinate=source_coordinate,
                source_axis_values=source_axis_values,
                window_coordinate=window_coordinate,
                window_axis_values=window_axis_values,
            ),
        )
        for window_coordinate in window_positions
    )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (-candidate.proxy_score, candidate.variant_index),
        )
    )


def _ma_source_axis_position_v2(
    *,
    values: tuple[int | float | str, ...] | None,
) -> int | None:
    """
    Select one deterministic MA `source` coordinate using canonical source-priority ordering.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        values: Optional source-axis values from one MA indicator plan.
    Returns:
        int | None: Preferred source coordinate, or `None` when the indicator has no source axis.
    Assumptions:
        The first MA-family plugin keeps `source` deterministic to avoid widening the proposal
        surface while still using canonical MA input vocabulary.
    Raises:
        ValueError: If the source axis exists but materializes to an empty sequence.
    Side Effects:
        None.
    """
    if values is None:
        return None
    if len(values) == 0:
        raise ValueError("MAFamilyAccelerationPluginV2 source axis must be non-empty")
    normalized_positions = {
        str(raw_value).strip().lower(): position
        for position, raw_value in enumerate(values)
    }
    for preferred_source in _SOURCE_PRIORITY_V2:
        position = normalized_positions.get(preferred_source)
        if position is not None:
            return position
    return 0


def _ma_window_axis_positions_v2(
    *,
    values: tuple[int | float | str, ...],
    limit: int,
) -> tuple[int, ...]:
    """
    Choose deterministic evenly spaced MA `window` coordinates from planner axis ordering.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        values: Ordered `window` axis values from one MA indicator plan.
        limit: Maximum number of window positions to retain.
    Returns:
        tuple[int, ...]: Sorted unique window coordinates in the canonical axis order.
    Assumptions:
        Planner axis ordering for `window` already matches the canonical low-to-high MA window
        progression and therefore may be sampled without extra sorting.
    Raises:
        ValueError: If `values` is empty or `limit` is non-positive.
    Side Effects:
        None.
    """
    if len(values) == 0:
        raise ValueError("MAFamilyAccelerationPluginV2 window axis must be non-empty")
    if limit <= 0:
        raise ValueError("MAFamilyAccelerationPluginV2 window limit must be > 0")
    if limit >= len(values):
        return tuple(range(len(values)))
    if limit == 1:
        return (len(values) // 2,)
    return tuple(
        sorted(
            {
                int(round(step * (len(values) - 1) / float(limit - 1)))
                for step in range(limit)
            }
        )
    )


def _ma_row_variant_index_v2(
    *,
    plan: BacktestIndicatorPlanV2,
    source_coordinate: int | None,
    window_coordinate: int,
) -> int:
    """
    Encode one MA indicator-local row index from canonical axis coordinates.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        plan: Planner-owned indicator plan for one canonical MA indicator.
        source_coordinate: Preferred `source` axis coordinate when present.
        window_coordinate: Selected `window` axis coordinate.
    Returns:
        int: Indicator-local mixed-radix row index.
    Assumptions:
        Only canonical MA axis names are present, so unsupported axes never need special routing.
    Raises:
        ValueError: If one coordinate falls outside the indicator-local axis bounds.
    Side Effects:
        None.
    """
    coordinates: list[int] = []
    radices: list[int] = []
    for axis in plan.axes:
        radices.append(len(axis.values))
        if axis.name == "source":
            if source_coordinate is None:
                raise ValueError(
                    f"MAFamilyAccelerationPluginV2 requires source coordinate for "
                    f"{plan.indicator_id!r}"
                )
            coordinate = source_coordinate
        elif axis.name == "window":
            coordinate = window_coordinate
        else:
            coordinate = 0
        if coordinate < 0 or coordinate >= len(axis.values):
            raise ValueError(
                "MAFamilyAccelerationPluginV2 axis coordinate is out of bounds for "
                f"{plan.indicator_id!r}: axis={axis.name!r}, coordinate={coordinate}, "
                f"size={len(axis.values)}"
            )
        coordinates.append(coordinate)
    return _encode_mixed_radix_v2(
        coordinates=tuple(coordinates),
        radices=tuple(radices),
    )


def _ma_row_proxy_score_v2(
    *,
    source_coordinate: int | None,
    source_axis_values: tuple[int | float | str, ...] | None,
    window_coordinate: int,
    window_axis_values: tuple[int | float | str, ...],
) -> float:
    """
    Build one cheap deterministic MA `proxy score` from `window` lag and source preference.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/indicators/domain/definitions/ma.py

    Args:
        source_coordinate: Preferred source coordinate when the plan has a source axis.
        source_axis_values: Optional source-axis values from the indicator plan.
        window_coordinate: Selected `window` coordinate.
        window_axis_values: Ordered `window` axis values from the indicator plan.
    Returns:
        float: Deterministic finite proxy score where shorter windows rank higher.
    Assumptions:
        This first MA-family heuristic stays cheap and reviewable: it uses canonical axis metadata
        only and treats shorter windows as higher-priority proposal anchors.
    Raises:
        ValueError: If `window_coordinate` is out of bounds.
    Side Effects:
        None.
    """
    if window_coordinate < 0 or window_coordinate >= len(window_axis_values):
        raise ValueError(
            "MAFamilyAccelerationPluginV2 window coordinate must be within axis bounds"
        )
    max_window_coordinate = max(1, len(window_axis_values) - 1)
    window_score = 1.0 - (float(window_coordinate) / float(max_window_coordinate))
    if source_coordinate is None or source_axis_values is None:
        return window_score
    max_source_coordinate = max(1, len(source_axis_values) - 1)
    source_score = 1.0 - (float(source_coordinate) / float(max_source_coordinate))
    return window_score + (0.05 * source_score)


def _ma_compute_candidates_v2(
    *,
    indicator_candidates: tuple[tuple[_MAIndicatorRowCandidateV2, ...], ...],
    target_compute_variants_total: int,
) -> tuple[_MAComputeCandidateV2, ...]:
    """
    Combine bounded indicator-local MA anchors into deterministic compute candidates.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        indicator_candidates: Ordered per-indicator MA row anchors.
        target_compute_variants_total: Maximum number of compute candidates to retain.
    Returns:
        tuple[_MAComputeCandidateV2, ...]: Deterministically ordered compute candidates.
    Assumptions:
        The MA-family plugin keeps proposal work bounded by composing only the retained
        indicator-local anchors instead of enumerating the full compute cartesian product.
    Raises:
        ValueError: If one indicator contributes no candidates or if the target is non-positive.
    Side Effects:
        None.
    """
    if target_compute_variants_total <= 0:
        raise ValueError(
            "MAFamilyAccelerationPluginV2 target_compute_variants_total must be > 0"
        )
    if any(len(candidates) == 0 for candidates in indicator_candidates):
        raise ValueError(
            "MAFamilyAccelerationPluginV2 requires row candidates for every indicator"
        )
    combined_candidates = tuple(
        _MAComputeCandidateV2(
            variant_indexes=tuple(
                row_candidate.variant_index for row_candidate in compute_candidate
            ),
            proxy_score=sum(
                row_candidate.proxy_score for row_candidate in compute_candidate
            ),
        )
        for compute_candidate in product(*indicator_candidates)
    )
    return tuple(
        sorted(
            combined_candidates,
            key=lambda candidate: (-candidate.proxy_score, candidate.variant_indexes),
        )[:target_compute_variants_total]
    )


def _ma_stage_a_proposal_v2(
    *,
    context: FamilyPluginPlanningContextV2,
    compute_candidates: tuple[_MAComputeCandidateV2, ...],
    shortlist_limit: int,
) -> tuple[tuple[int, ...], tuple[FamilyPluginProxyScoreV2, ...]]:
    """
    Expand bounded MA compute anchors into exact Stage A row-shortlist coordinates.

    Docs:
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py

    Args:
        context: Narrow immutable planning context for the runtime plan.
        compute_candidates: Retained MA compute candidates.
        shortlist_limit: Maximum Stage A rows the proposal may emit.
    Returns:
        tuple[tuple[int, ...], tuple[FamilyPluginProxyScoreV2, ...]]:
            Exact Stage A `row shortlist` plus aligned `proxy score` payload.
    Assumptions:
        The first MA-family plugin preserves exact signal semantics by expanding retained compute
        anchors across the shared signal-space mixed-radix order.
    Raises:
        ValueError: If `shortlist_limit` is non-positive or no compute candidates remain.
    Side Effects:
        None.
    """
    if shortlist_limit <= 0:
        raise ValueError("MAFamilyAccelerationPluginV2 shortlist_limit must be > 0")
    if len(compute_candidates) == 0:
        raise ValueError(
            "MAFamilyAccelerationPluginV2 requires non-empty compute_candidates"
        )
    signal_variants_total = context.runtime_plan.signal_variants_total()
    indicator_radices = tuple(
        plan.variants for plan in context.runtime_plan.indicator_plans
    )
    row_shortlist: list[int] = []
    proxy_scores: list[FamilyPluginProxyScoreV2] = []
    for compute_candidate in compute_candidates:
        compute_index = _encode_mixed_radix_v2(
            coordinates=compute_candidate.variant_indexes,
            radices=indicator_radices,
        )
        for signal_index in range(signal_variants_total):
            stage_a_index = (compute_index * signal_variants_total) + signal_index
            row_shortlist.append(stage_a_index)
            proxy_scores.append(
                FamilyPluginProxyScoreV2(
                    stage_a_index=stage_a_index,
                    proxy_score=compute_candidate.proxy_score,
                )
            )
            if len(row_shortlist) >= shortlist_limit:
                return (tuple(row_shortlist), tuple(proxy_scores))
    return (tuple(row_shortlist), tuple(proxy_scores))


def _encode_mixed_radix_v2(
    *,
    coordinates: tuple[int, ...],
    radices: tuple[int, ...],
) -> int:
    """
    Encode one mixed-radix coordinate tuple into a deterministic flat index.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py

    Args:
        coordinates: Per-axis coordinates in deterministic planner order.
        radices: Per-axis radix sizes matching `coordinates`.
    Returns:
        int: Zero-based flat index in the same mixed-radix space.
    Assumptions:
        Coordinate ordering matches the planner's canonical axis order.
    Raises:
        ValueError: If the coordinate/radix tuples differ in length or one coordinate falls
            outside the radix bounds.
    Side Effects:
        None.
    """
    if len(coordinates) != len(radices):
        raise ValueError(
            "_encode_mixed_radix_v2 requires coordinates and radices with the same length"
        )
    flat_index = 0
    for coordinate, radix in zip(coordinates, radices, strict=True):
        if radix <= 0:
            raise ValueError("_encode_mixed_radix_v2 radices must be > 0")
        if coordinate < 0 or coordinate >= radix:
            raise ValueError(
                "_encode_mixed_radix_v2 coordinates must stay within radix bounds"
            )
        flat_index = (flat_index * radix) + coordinate
    return flat_index


__all__ = [
    "MAFamilyAccelerationPluginV2",
]
