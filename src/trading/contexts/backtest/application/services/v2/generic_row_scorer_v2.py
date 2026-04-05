"""Deterministic universal row scorer for conservative shortlist foundation work.

Docs:
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/backtest-runtime-kernels-v2.md
  - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Sequence, cast

import numpy as np

from .contracts import SIGNAL_FEATURE_NAMES_V2
from .execution_profile_v2 import ExecutionProfileShortlistScoringConfigV2

type GenericRowMetadataScalarV2 = int | float | str | bool | None
type GenericRowBucketLabelLiteralV2 = Literal[
    "low_activity",
    "medium_activity",
    "high_activity",
    "short_bias",
    "balanced",
    "long_bias",
    "low_transition",
    "medium_transition",
    "high_transition",
]
type GenericRowScoreComponentIdLiteralV2 = Literal[
    "activity_ratio",
    "direction_balance",
    "transition_count",
    "active_span_ratio",
]
type GenericRowScoreSourceLiteralV2 = Literal["signal_features", "runtime_row_stats"]

_SIGNAL_FEATURE_INDEX_BY_NAME_V2: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(SIGNAL_FEATURE_NAMES_V2)}
)
_ALLOWED_SIGNAL_VALUES_V2 = np.array((-1, 0, 1), dtype=np.int8)


def build_generic_row_signal_features_mapping_v2(
    *,
    feature_names: Sequence[str],
    feature_values: Sequence[float],
) -> Mapping[str, float]:
    """
    Build one strict signal-feature mapping from ordered artifact-style names and row values.

    Args:
        feature_names: Ordered feature-name literals for one row.
        feature_values: Ordered numeric feature values matching `feature_names`.
    Returns:
        Mapping[str, float]: Immutable normalized mapping keyed by canonical signal-feature names.
    Assumptions:
        The additive `signal_features` artifact uses the fixed feature order from Milestone C1 and
        future runtime work should consume the same surface instead of ad-hoc row dicts.
    Raises:
        ValueError: If names are incomplete/duplicated/unsupported, lengths differ, or one value
            is not finite.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if len(feature_names) != len(feature_values):
        raise ValueError(
            "generic row signal feature names and values must have the same length"
        )
    normalized_payload: dict[str, float] = {}
    for raw_name, raw_value in zip(feature_names, feature_values, strict=True):
        normalized_name = raw_name.strip()
        if normalized_name not in _SIGNAL_FEATURE_INDEX_BY_NAME_V2:
            raise ValueError(
                "generic row signal features must match canonical names "
                f"{SIGNAL_FEATURE_NAMES_V2}, got {raw_name!r}"
            )
        if normalized_name in normalized_payload:
            raise ValueError(
                "generic row signal feature names must not contain duplicates, got "
                f"{raw_name!r}"
            )
        feature_value = float(raw_value)
        if not math.isfinite(feature_value):
            raise ValueError(
                f"generic row signal feature {normalized_name!r} must be finite"
            )
        normalized_payload[normalized_name] = feature_value
    missing_names = [
        feature_name
        for feature_name in SIGNAL_FEATURE_NAMES_V2
        if feature_name not in normalized_payload
    ]
    if missing_names:
        raise ValueError(
            "generic row signal features must include the canonical full set, missing "
            f"{tuple(missing_names)}"
        )
    return MappingProxyType(
        {
            feature_name: normalized_payload[feature_name]
            for feature_name in SIGNAL_FEATURE_NAMES_V2
        }
    )


@dataclass(frozen=True, slots=True)
class GenericRowScoringInputV2:
    """
    Universal deterministic row-scoring input for one indicator block candidate row.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
    """

    indicator_id: str
    row_index: int
    signal_row: np.ndarray
    stable_identity: str | None = None
    signal_features: Mapping[str, float] | None = None
    metadata: Mapping[str, GenericRowMetadataScalarV2] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """
        Validate one row candidate and freeze explicit metadata for deterministic scoring.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal rows are already materialized by the caller and use the canonical
            `{-1, 0, 1}` encoding from the shipped signal contract.
        Raises:
            ValueError: If identifiers are blank, row indexes are negative, signal rows are not
                1D/non-empty/int-like, or feature/metadata payloads violate deterministic bounds.
        Side Effects:
            Normalizes `signal_row`, `stable_identity`, `signal_features`, and `metadata`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
        """
        indicator_id = self.indicator_id.strip()
        if not indicator_id:
            raise ValueError("GenericRowScoringInputV2.indicator_id must be non-empty")
        object.__setattr__(self, "indicator_id", indicator_id)
        if self.row_index < 0:
            raise ValueError("GenericRowScoringInputV2.row_index must be >= 0")

        normalized_signal_row = np.asarray(self.signal_row, dtype=np.int8)
        if normalized_signal_row.ndim != 1:
            raise ValueError("GenericRowScoringInputV2.signal_row must be 1D")
        if normalized_signal_row.size == 0:
            raise ValueError("GenericRowScoringInputV2.signal_row must be non-empty")
        if not np.isin(normalized_signal_row, _ALLOWED_SIGNAL_VALUES_V2).all():
            raise ValueError(
                "GenericRowScoringInputV2.signal_row must contain only {-1, 0, 1}"
            )
        object.__setattr__(
            self,
            "signal_row",
            np.ascontiguousarray(normalized_signal_row, dtype=np.int8),
        )

        if self.stable_identity is None:
            stable_identity = f"{indicator_id}:{self.row_index}"
        else:
            stable_identity = self.stable_identity.strip()
        if not stable_identity:
            raise ValueError("GenericRowScoringInputV2.stable_identity must be non-empty")
        object.__setattr__(self, "stable_identity", stable_identity)

        if self.signal_features is not None:
            object.__setattr__(
                self,
                "signal_features",
                build_generic_row_signal_features_mapping_v2(
                    feature_names=tuple(self.signal_features.keys()),
                    feature_values=tuple(self.signal_features.values()),
                ),
            )

        normalized_metadata: dict[str, GenericRowMetadataScalarV2] = {}
        for raw_key in sorted(self.metadata.keys()):
            metadata_key = str(raw_key).strip()
            if not metadata_key:
                raise ValueError(
                    "GenericRowScoringInputV2.metadata keys must be non-empty"
                )
            metadata_value = self.metadata[raw_key]
            if isinstance(metadata_value, float) and not math.isfinite(metadata_value):
                raise ValueError(
                    f"GenericRowScoringInputV2.metadata[{metadata_key!r}] must be finite"
                )
            normalized_metadata[metadata_key] = metadata_value
        object.__setattr__(self, "metadata", MappingProxyType(normalized_metadata))


@dataclass(frozen=True, slots=True)
class GenericRowResolvedSignalFeaturesV2:
    """
    Typed signal-feature payload resolved either from cache or deterministic row derivation.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """

    nonzero_count: float
    long_count: float
    short_count: float
    activity_ratio: float
    direction_balance: float
    transition_count: float
    used_cached_signal_features: bool

    def __post_init__(self) -> None:
        """
        Validate resolved signal-feature scalars and normalize them to builtin types.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Counts are row-local deterministic aggregates and ratios stay within their canonical
            numeric bounds.
        Raises:
            ValueError: If one scalar is non-finite/negative or ratios leave expected bounds.
        Side Effects:
            Normalizes numeric fields to builtin `float`.
        Docs:
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        for field_name in (
            "nonzero_count",
            "long_count",
            "short_count",
            "activity_ratio",
            "direction_balance",
            "transition_count",
        ):
            field_value = float(getattr(self, field_name))
            if not math.isfinite(field_value):
                raise ValueError(
                    f"GenericRowResolvedSignalFeaturesV2.{field_name} must be finite"
                )
            object.__setattr__(self, field_name, field_value)
        if self.nonzero_count < 0.0:
            raise ValueError(
                "GenericRowResolvedSignalFeaturesV2.nonzero_count must be >= 0"
            )
        if self.long_count < 0.0:
            raise ValueError("GenericRowResolvedSignalFeaturesV2.long_count must be >= 0")
        if self.short_count < 0.0:
            raise ValueError("GenericRowResolvedSignalFeaturesV2.short_count must be >= 0")
        if not 0.0 <= self.activity_ratio <= 1.0:
            raise ValueError(
                "GenericRowResolvedSignalFeaturesV2.activity_ratio must be in [0, 1]"
            )
        if not -1.0 <= self.direction_balance <= 1.0:
            raise ValueError(
                "GenericRowResolvedSignalFeaturesV2.direction_balance must be in [-1, 1]"
            )
        if self.transition_count < 0.0:
            raise ValueError(
                "GenericRowResolvedSignalFeaturesV2.transition_count must be >= 0"
            )
        if not isinstance(self.used_cached_signal_features, bool):
            raise ValueError(
                "GenericRowResolvedSignalFeaturesV2.used_cached_signal_features must be bool"
            )


@dataclass(frozen=True, slots=True)
class GenericRowRuntimeStatsV2:
    """
    Cheap deterministic row-local stats derived at runtime from one signal row.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
    """

    timeline_length: int
    first_active_index: int | None
    last_active_index: int | None
    active_span: int
    active_span_ratio: float
    transition_ratio: float

    def __post_init__(self) -> None:
        """
        Validate deterministic row-local runtime stats used by generic shortlist scoring.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Runtime-derived stats stay row-local and cheap, so exact runtime paths can remain
            untouched until a later explicit rollout milestone.
        Raises:
            ValueError: If one index or normalized ratio violates the row-local contract.
        Side Effects:
            Normalizes ratios to builtin `float`.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
        """
        if self.timeline_length <= 0:
            raise ValueError("GenericRowRuntimeStatsV2.timeline_length must be > 0")
        if self.first_active_index is not None and self.first_active_index < 0:
            raise ValueError(
                "GenericRowRuntimeStatsV2.first_active_index must be >= 0 when provided"
            )
        if self.last_active_index is not None and self.last_active_index < 0:
            raise ValueError(
                "GenericRowRuntimeStatsV2.last_active_index must be >= 0 when provided"
            )
        if self.active_span < 0:
            raise ValueError("GenericRowRuntimeStatsV2.active_span must be >= 0")
        active_span_ratio = float(self.active_span_ratio)
        transition_ratio = float(self.transition_ratio)
        if not math.isfinite(active_span_ratio):
            raise ValueError(
                "GenericRowRuntimeStatsV2.active_span_ratio must be finite"
            )
        if not math.isfinite(transition_ratio):
            raise ValueError(
                "GenericRowRuntimeStatsV2.transition_ratio must be finite"
            )
        if not 0.0 <= active_span_ratio <= 1.0:
            raise ValueError(
                "GenericRowRuntimeStatsV2.active_span_ratio must be in [0, 1]"
            )
        if not 0.0 <= transition_ratio <= 1.0:
            raise ValueError(
                "GenericRowRuntimeStatsV2.transition_ratio must be in [0, 1]"
            )
        object.__setattr__(self, "active_span_ratio", active_span_ratio)
        object.__setattr__(self, "transition_ratio", transition_ratio)


@dataclass(frozen=True, slots=True)
class GenericRowScoreComponentV2:
    """
    One auditable scorer component used to build the final deterministic row score.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
    """

    component_id: GenericRowScoreComponentIdLiteralV2
    source: GenericRowScoreSourceLiteralV2
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float

    def __post_init__(self) -> None:
        """
        Validate one score component and normalize numeric fields for audit tables.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Components are bounded, explicit, and future benchmark/debug tables may render them
            without recomputing hidden transforms.
        Raises:
            ValueError: If one numeric field is non-finite or a normalized field leaves bounds.
        Side Effects:
            Normalizes numeric fields to builtin `float`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        for field_name in ("raw_value", "normalized_value", "weight", "contribution"):
            field_value = float(getattr(self, field_name))
            if not math.isfinite(field_value):
                raise ValueError(
                    f"GenericRowScoreComponentV2.{field_name} must be finite"
                )
            object.__setattr__(self, field_name, field_value)
        if not 0.0 <= self.normalized_value <= 1.0:
            raise ValueError(
                "GenericRowScoreComponentV2.normalized_value must be in [0, 1]"
            )
        if self.weight < 0.0:
            raise ValueError("GenericRowScoreComponentV2.weight must be >= 0")
        if self.contribution < 0.0:
            raise ValueError("GenericRowScoreComponentV2.contribution must be >= 0")


@dataclass(frozen=True, slots=True)
class GenericRowScorePayloadV2:
    """
    Deterministic scored row payload emitted by the universal conservative scorer.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
    """

    indicator_id: str
    row_index: int
    stable_identity: str
    total_score: float
    signal_features: GenericRowResolvedSignalFeaturesV2
    runtime_stats: GenericRowRuntimeStatsV2
    bucket_values: Mapping[str, str]
    components: tuple[GenericRowScoreComponentV2, ...]
    metadata: Mapping[str, GenericRowMetadataScalarV2] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """
        Validate final scored-row payload and freeze explicit bucket/metadata ordering.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Payloads must be safe to compare, serialize, and reuse in deterministic benchmark or
            rollout-debug tables without depending on caller iteration order.
        Raises:
            ValueError: If one identifier is invalid, total score is non-finite/outside bounds,
                or bucket values are blank.
        Side Effects:
            Normalizes `total_score` and freezes `bucket_values`/`metadata`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
        """
        if not self.indicator_id.strip():
            raise ValueError("GenericRowScorePayloadV2.indicator_id must be non-empty")
        if self.row_index < 0:
            raise ValueError("GenericRowScorePayloadV2.row_index must be >= 0")
        if not self.stable_identity.strip():
            raise ValueError("GenericRowScorePayloadV2.stable_identity must be non-empty")
        total_score = float(self.total_score)
        if not math.isfinite(total_score):
            raise ValueError("GenericRowScorePayloadV2.total_score must be finite")
        if not 0.0 <= total_score <= 1.0:
            raise ValueError("GenericRowScorePayloadV2.total_score must be in [0, 1]")
        object.__setattr__(self, "total_score", total_score)

        normalized_bucket_values: dict[str, str] = {}
        for bucket_name in sorted(self.bucket_values.keys()):
            normalized_name = str(bucket_name).strip()
            bucket_value = str(self.bucket_values[bucket_name]).strip()
            if not normalized_name:
                raise ValueError(
                    "GenericRowScorePayloadV2.bucket_values keys must be non-empty"
                )
            if not bucket_value:
                raise ValueError(
                    "GenericRowScorePayloadV2.bucket_values values must be non-empty"
                )
            normalized_bucket_values[normalized_name] = bucket_value
        object.__setattr__(self, "bucket_values", MappingProxyType(normalized_bucket_values))

        normalized_metadata: dict[str, GenericRowMetadataScalarV2] = {}
        for metadata_key in sorted(self.metadata.keys()):
            normalized_metadata[str(metadata_key)] = self.metadata[metadata_key]
        object.__setattr__(self, "metadata", MappingProxyType(normalized_metadata))
        if len(self.components) == 0:
            raise ValueError("GenericRowScorePayloadV2.components must be non-empty")

    def sort_key(self) -> tuple[float, str, str, int]:
        """
        Return one explicit deterministic sort key for score-first shortlist ordering.

        Args:
            None.
        Returns:
            tuple[float, str, str, int]: Stable ordering key using descending score and explicit
                identity tie-breaks.
        Assumptions:
            `stable_identity` is caller-controlled and stable across reruns for the same row.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        return (-self.total_score, self.stable_identity, self.indicator_id, self.row_index)


@dataclass(frozen=True, slots=True)
class GenericRowScorerV2:
    """
    Score candidate rows for any indicator block using cached features and cheap row-local stats.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    """

    scoring: ExecutionProfileShortlistScoringConfigV2 = field(
        default_factory=ExecutionProfileShortlistScoringConfigV2
    )
    low_activity_threshold: float = 0.10
    high_activity_threshold: float = 0.35
    direction_balance_threshold: float = 0.35
    low_transition_ratio_threshold: float = 0.05
    high_transition_ratio_threshold: float = 0.20

    def __post_init__(self) -> None:
        """
        Validate scorer thresholds and keep rollout knobs explicit but inactive by default.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Thresholds define deterministic bucket labels only; they do not activate heuristic
            runtime routing by themselves in Milestone D1.
        Raises:
            ValueError: If one threshold is non-finite/outside `[0, 1]` or low/high ordering
                drifts.
        Side Effects:
            Normalizes threshold fields to builtin `float`.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
        """
        for field_name in (
            "low_activity_threshold",
            "high_activity_threshold",
            "direction_balance_threshold",
            "low_transition_ratio_threshold",
            "high_transition_ratio_threshold",
        ):
            field_value = float(getattr(self, field_name))
            if not math.isfinite(field_value):
                raise ValueError(f"GenericRowScorerV2.{field_name} must be finite")
            if not 0.0 <= field_value <= 1.0:
                raise ValueError(f"GenericRowScorerV2.{field_name} must be in [0, 1]")
            object.__setattr__(self, field_name, field_value)
        if self.low_activity_threshold >= self.high_activity_threshold:
            raise ValueError(
                "GenericRowScorerV2 low/high activity thresholds must be strictly ordered"
            )
        if self.low_transition_ratio_threshold >= self.high_transition_ratio_threshold:
            raise ValueError(
                "GenericRowScorerV2 low/high transition thresholds must be strictly ordered"
            )

    def score_rows(
        self,
        *,
        rows: Sequence[GenericRowScoringInputV2],
    ) -> tuple[GenericRowScorePayloadV2, ...]:
        """
        Score one deterministic batch of candidate rows and return them in explicit sort order.

        Args:
            rows: Ordered row candidates from one or more indicator blocks.
        Returns:
            tuple[GenericRowScorePayloadV2, ...]: Deterministic scored payloads ordered by
                descending score and explicit identity tie-breaks.
        Assumptions:
            Callers pass already materialized rows and do not rely on implicit side effects such
            as artifact loading or runtime profile activation in this foundation milestone.
        Raises:
            ValueError: If duplicate `stable_identity` values are present.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        scored_rows = tuple(self.score_row(row=row) for row in rows)
        seen_identities: set[str] = set()
        for scored_row in scored_rows:
            if scored_row.stable_identity in seen_identities:
                raise ValueError(
                    "GenericRowScorerV2 rows must have unique stable_identity values, got "
                    f"{scored_row.stable_identity!r}"
                )
            seen_identities.add(scored_row.stable_identity)
        return tuple(sorted(scored_rows, key=lambda payload: payload.sort_key()))

    def score_row(
        self,
        *,
        row: GenericRowScoringInputV2,
    ) -> GenericRowScorePayloadV2:
        """
        Score one candidate row and emit a typed auditable payload with explicit components.

        Args:
            row: One universal row candidate with signal row and optional cached features.
        Returns:
            GenericRowScorePayloadV2: Typed score payload for later diversified retention or
                benchmark/debug rendering.
        Assumptions:
            Cached `signal_features` are preferred when present; cheap runtime-derived row-local
            stats remain allowed for additive conservative-shortlist work.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        runtime_stats = self._derive_runtime_stats(row=row)
        signal_features = self._resolve_signal_features(row=row)
        components = self._build_score_components(
            signal_features=signal_features,
            runtime_stats=runtime_stats,
        )
        total_weight = sum(component.weight for component in components)
        total_score = 0.0 if total_weight <= 0.0 else (
            sum(component.contribution for component in components) / total_weight
        )
        return GenericRowScorePayloadV2(
            indicator_id=row.indicator_id,
            row_index=row.row_index,
            stable_identity=cast(str, row.stable_identity),
            total_score=total_score,
            signal_features=signal_features,
            runtime_stats=runtime_stats,
            bucket_values=self._bucket_values_for_row(
                signal_features=signal_features,
                runtime_stats=runtime_stats,
            ),
            components=components,
            metadata=row.metadata,
        )

    def _resolve_signal_features(
        self,
        *,
        row: GenericRowScoringInputV2,
    ) -> GenericRowResolvedSignalFeaturesV2:
        """
        Resolve typed signal-feature scalars from cache when available or derive them from row.

        Args:
            row: One deterministic scoring input.
        Returns:
            GenericRowResolvedSignalFeaturesV2: Typed resolved feature payload.
        Assumptions:
            Cached feature order matches `SIGNAL_FEATURE_NAMES_V2`, while fallback derivation uses
            the same deterministic formulas as additive Milestone C artifacts.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
          - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
        """
        if row.signal_features is not None:
            return GenericRowResolvedSignalFeaturesV2(
                nonzero_count=row.signal_features["nonzero_count"],
                long_count=row.signal_features["long_count"],
                short_count=row.signal_features["short_count"],
                activity_ratio=row.signal_features["activity_ratio"],
                direction_balance=row.signal_features["direction_balance"],
                transition_count=row.signal_features["transition_count"],
                used_cached_signal_features=True,
            )
        signal_row = row.signal_row
        timeline_length = int(signal_row.size)
        nonzero_count = float(np.count_nonzero(signal_row != 0))
        long_count = float(np.count_nonzero(signal_row > 0))
        short_count = float(np.count_nonzero(signal_row < 0))
        activity_ratio = 0.0 if timeline_length <= 0 else nonzero_count / float(timeline_length)
        if nonzero_count <= 0.0:
            direction_balance = 0.0
        else:
            direction_balance = (long_count - short_count) / nonzero_count
        if timeline_length < 2:
            transition_count = 0.0
        else:
            transition_count = float(
                np.count_nonzero(signal_row[1:] != signal_row[:-1])
            )
        return GenericRowResolvedSignalFeaturesV2(
            nonzero_count=nonzero_count,
            long_count=long_count,
            short_count=short_count,
            activity_ratio=activity_ratio,
            direction_balance=direction_balance,
            transition_count=transition_count,
            used_cached_signal_features=False,
        )

    def _derive_runtime_stats(
        self,
        *,
        row: GenericRowScoringInputV2,
    ) -> GenericRowRuntimeStatsV2:
        """
        Derive cheap row-local runtime stats that are not shipped in cached feature artifacts.

        Args:
            row: One deterministic scoring input.
        Returns:
            GenericRowRuntimeStatsV2: Typed row-local runtime stats.
        Assumptions:
            Runtime-derived stats remain cheap and deterministic because they inspect only the
            current row and its fixed timeline length.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        signal_row = row.signal_row
        timeline_length = int(signal_row.size)
        active_indexes = np.flatnonzero(signal_row != 0)
        if active_indexes.size == 0:
            first_active_index = None
            last_active_index = None
            active_span = 0
            active_span_ratio = 0.0
        else:
            first_active_index = int(active_indexes[0])
            last_active_index = int(active_indexes[-1])
            active_span = last_active_index - first_active_index + 1
            active_span_ratio = active_span / float(timeline_length)
        if timeline_length < 2:
            transition_ratio = 0.0
        else:
            transition_count = float(np.count_nonzero(signal_row[1:] != signal_row[:-1]))
            transition_ratio = transition_count / float(timeline_length - 1)
        return GenericRowRuntimeStatsV2(
            timeline_length=timeline_length,
            first_active_index=first_active_index,
            last_active_index=last_active_index,
            active_span=active_span,
            active_span_ratio=active_span_ratio,
            transition_ratio=transition_ratio,
        )

    def _build_score_components(
        self,
        *,
        signal_features: GenericRowResolvedSignalFeaturesV2,
        runtime_stats: GenericRowRuntimeStatsV2,
    ) -> tuple[GenericRowScoreComponentV2, ...]:
        """
        Build the fixed deterministic score breakdown used by the universal row scorer.

        Args:
            signal_features: Resolved cached-or-derived signal-feature payload.
            runtime_stats: Cheap deterministic row-local runtime stats.
        Returns:
            tuple[GenericRowScoreComponentV2, ...]: Explicit ordered component breakdown.
        Assumptions:
            Component order is part of the auditable payload contract and must remain stable for
            later benchmark explanation tables.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        transition_source: GenericRowScoreSourceLiteralV2 = (
            "signal_features"
            if signal_features.used_cached_signal_features
            else "runtime_row_stats"
        )
        transition_normalized = self._normalize_ratio(
            signal_features.transition_count,
            denominator=max(runtime_stats.timeline_length - 1, 1),
        )
        direction_balance_normalized = (
            0.0
            if signal_features.nonzero_count <= 0.0
            else 1.0 - min(abs(signal_features.direction_balance), 1.0)
        )
        return (
            GenericRowScoreComponentV2(
                component_id="activity_ratio",
                source="signal_features",
                raw_value=signal_features.activity_ratio,
                normalized_value=min(max(signal_features.activity_ratio, 0.0), 1.0),
                weight=self.scoring.activity_ratio_weight,
                contribution=(
                    min(max(signal_features.activity_ratio, 0.0), 1.0)
                    * self.scoring.activity_ratio_weight
                ),
            ),
            GenericRowScoreComponentV2(
                component_id="direction_balance",
                source="signal_features",
                raw_value=signal_features.direction_balance,
                normalized_value=direction_balance_normalized,
                weight=self.scoring.direction_balance_weight,
                contribution=(
                    direction_balance_normalized
                    * self.scoring.direction_balance_weight
                ),
            ),
            GenericRowScoreComponentV2(
                component_id="transition_count",
                source=transition_source,
                raw_value=signal_features.transition_count,
                normalized_value=transition_normalized,
                weight=self.scoring.transition_ratio_weight,
                contribution=(
                    transition_normalized * self.scoring.transition_ratio_weight
                ),
            ),
            GenericRowScoreComponentV2(
                component_id="active_span_ratio",
                source="runtime_row_stats",
                raw_value=runtime_stats.active_span_ratio,
                normalized_value=runtime_stats.active_span_ratio,
                weight=self.scoring.active_span_ratio_weight,
                contribution=(
                    runtime_stats.active_span_ratio
                    * self.scoring.active_span_ratio_weight
                ),
            ),
        )

    def _bucket_values_for_row(
        self,
        *,
        signal_features: GenericRowResolvedSignalFeaturesV2,
        runtime_stats: GenericRowRuntimeStatsV2,
    ) -> Mapping[str, GenericRowBucketLabelLiteralV2]:
        """
        Build explicit deterministic bucket labels for later diversified survivor retention.

        Args:
            signal_features: Resolved cached-or-derived signal-feature payload.
            runtime_stats: Cheap deterministic row-local runtime stats.
        Returns:
            Mapping[str, GenericRowBucketLabelLiteralV2]: Immutable bucket-label mapping keyed by
                stable bucket-axis names.
        Assumptions:
            Bucket labels remain explicit and human-reviewable so future rollout slices such as
            `low_activity` can reason about survivor diversity without hidden grouping logic.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        activity_band: GenericRowBucketLabelLiteralV2
        if signal_features.activity_ratio < self.low_activity_threshold:
            activity_band = "low_activity"
        elif signal_features.activity_ratio < self.high_activity_threshold:
            activity_band = "medium_activity"
        else:
            activity_band = "high_activity"

        direction_band: GenericRowBucketLabelLiteralV2
        if signal_features.nonzero_count <= 0.0:
            direction_band = "balanced"
        elif signal_features.direction_balance > self.direction_balance_threshold:
            direction_band = "long_bias"
        elif signal_features.direction_balance < -self.direction_balance_threshold:
            direction_band = "short_bias"
        else:
            direction_band = "balanced"

        transition_band: GenericRowBucketLabelLiteralV2
        if runtime_stats.transition_ratio < self.low_transition_ratio_threshold:
            transition_band = "low_transition"
        elif runtime_stats.transition_ratio < self.high_transition_ratio_threshold:
            transition_band = "medium_transition"
        else:
            transition_band = "high_transition"
        return MappingProxyType(
            {
                "activity_band": activity_band,
                "direction_band": direction_band,
                "transition_band": transition_band,
            }
        )

    def _normalize_ratio(
        self,
        numerator: float,
        *,
        denominator: int,
    ) -> float:
        """
        Normalize one non-negative count-like scalar into `[0, 1]` with deterministic clamping.

        Args:
            numerator: Count-like scalar to normalize.
            denominator: Strict-positive normalizing denominator.
        Returns:
            float: Clamped normalized ratio in `[0, 1]`.
        Assumptions:
            Count-derived scorer components stay bounded to avoid hidden score inflation across
            different timeline lengths.
        Raises:
            ValueError: If `denominator` is not positive.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_generic_row_scorer_v2.py
        """
        if denominator <= 0:
            raise ValueError("GenericRowScorerV2 ratio denominator must be > 0")
        normalized_value = float(numerator) / float(denominator)
        return min(max(normalized_value, 0.0), 1.0)


__all__ = [
    "GenericRowBucketLabelLiteralV2",
    "GenericRowMetadataScalarV2",
    "GenericRowResolvedSignalFeaturesV2",
    "GenericRowRuntimeStatsV2",
    "GenericRowScoreComponentIdLiteralV2",
    "GenericRowScoreComponentV2",
    "GenericRowScorePayloadV2",
    "GenericRowScoreSourceLiteralV2",
    "GenericRowScorerV2",
    "GenericRowScoringInputV2",
    "build_generic_row_signal_features_mapping_v2",
]
