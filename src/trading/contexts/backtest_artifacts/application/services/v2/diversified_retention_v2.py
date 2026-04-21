"""Deterministic diversified survivor retention for conservative shortlist foundation work.

Docs:
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/README.md
  - docs/architecture/backtest/README.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .execution_profile_v2 import ExecutionProfileShortlistRetentionConfigV2
from .generic_row_scorer_v2 import GenericRowScorePayloadV2


@dataclass(frozen=True, slots=True)
class DiversifiedRetentionBucketComponentV2:
    """
    One explicit bucket-axis component used to build deterministic diversity grouping keys.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
    """

    bucket_name: str
    bucket_value: str

    def __post_init__(self) -> None:
        """
        Validate one explicit bucket-axis component used by diversified retention.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Bucket axes and values are human-reviewable string literals published by the scorer.
        Raises:
            ValueError: If one bucket name or value is blank.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        if not self.bucket_name.strip():
            raise ValueError(
                "DiversifiedRetentionBucketComponentV2.bucket_name must be non-empty"
            )
        if not self.bucket_value.strip():
            raise ValueError(
                "DiversifiedRetentionBucketComponentV2.bucket_value must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class DiversifiedRetentionBucketKeyV2:
    """
    Explicit composite diversity-bucket key built from ordered scorer bucket axes.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
    """

    components: tuple[DiversifiedRetentionBucketComponentV2, ...]

    def __post_init__(self) -> None:
        """
        Validate composite bucket identity used by deterministic diversified retention.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Bucket identity must stay explicit because future rollout work will benchmark recall,
            overlap, and diversity against the same fixed grouping semantics.
        Raises:
            ValueError: If the key is empty or contains duplicate axis names.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        if len(self.components) == 0:
            raise ValueError("DiversifiedRetentionBucketKeyV2.components must be non-empty")
        seen_names: set[str] = set()
        for component in self.components:
            if component.bucket_name in seen_names:
                raise ValueError(
                    "DiversifiedRetentionBucketKeyV2.components must not contain duplicate "
                    f"bucket names, got {component.bucket_name!r}"
                )
            seen_names.add(component.bucket_name)

    def sort_key(self) -> tuple[tuple[str, str], ...]:
        """
        Return one explicit stable ordering key for deterministic bucket iteration.

        Args:
            None.
        Returns:
            tuple[tuple[str, str], ...]: Ordered `(bucket_name, bucket_value)` pairs.
        Assumptions:
            Bucket component ordering is already validated and part of the retention contract.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        return tuple(
            (component.bucket_name, component.bucket_value)
            for component in self.components
        )


@dataclass(frozen=True, slots=True)
class DiversifiedRetentionDecisionV2:
    """
    Audit record describing whether one scored row survived deterministic diversified retention.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
    """

    row: GenericRowScorePayloadV2
    bucket_key: DiversifiedRetentionBucketKeyV2
    retained: bool
    bucket_rank: int
    selection_round: int | None
    discard_reason: str | None = None

    def __post_init__(self) -> None:
        """
        Validate one retention audit record for deterministic shortlist debugging.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            A row is either retained with a concrete round number or discarded with an explicit
            reason; both states stay machine-auditable.
        Raises:
            ValueError: If ranks are invalid or retained/discarded state is inconsistent.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
        """
        if self.bucket_rank <= 0:
            raise ValueError("DiversifiedRetentionDecisionV2.bucket_rank must be > 0")
        if self.retained:
            if self.selection_round is None or self.selection_round <= 0:
                raise ValueError(
                    "DiversifiedRetentionDecisionV2.selection_round must be > 0 for retained rows"
                )
            if self.discard_reason is not None:
                raise ValueError(
                    "DiversifiedRetentionDecisionV2.discard_reason must be None for retained rows"
                )
            return
        if self.selection_round is not None:
            raise ValueError(
                "DiversifiedRetentionDecisionV2.selection_round must be None for discarded rows"
            )
        if self.discard_reason is None or not self.discard_reason.strip():
            raise ValueError(
                "DiversifiedRetentionDecisionV2.discard_reason must be non-empty for "
                "discarded rows"
            )


@dataclass(frozen=True, slots=True)
class DiversifiedRetentionResultV2:
    """
    Final deterministic survivor set and audit trail produced by diversified retention.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
    """

    retained_rows: tuple[GenericRowScorePayloadV2, ...]
    decisions: tuple[DiversifiedRetentionDecisionV2, ...]


@dataclass(frozen=True, slots=True)
class DiversifiedRetentionV2:
    """
    Retain deterministic shortlist survivors with explicit bucket-aware round-robin grouping.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    """

    def retain_rows(
        self,
        *,
        scored_rows: Sequence[GenericRowScorePayloadV2],
        config: ExecutionProfileShortlistRetentionConfigV2,
        max_candidates: int,
    ) -> DiversifiedRetentionResultV2:
        """
        Retain deterministic survivors using explicit diversity buckets instead of raw-score cut.

        Args:
            scored_rows: Scored row payloads emitted by the universal scorer.
            config: Typed retention config carrying explicit bucket axes and optional per-bucket
                caps.
            max_candidates: Total survivor budget after diversified retention.
        Returns:
            DiversifiedRetentionResultV2: Deterministic survivor set and full audit trail.
        Assumptions:
            Retained rows are returned in score order for downstream exact scoring, while the
            audit trail preserves the round-robin diversity selection chronology.
        Raises:
            ValueError: If `max_candidates` is non-positive, duplicate identities are present, or
                one row lacks a required bucket axis from `config.diversity_buckets`.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        if max_candidates <= 0:
            raise ValueError("DiversifiedRetentionV2 max_candidates must be > 0")
        ordered_rows = tuple(sorted(scored_rows, key=lambda row: row.sort_key()))
        seen_identities: set[str] = set()
        for row in ordered_rows:
            if row.stable_identity in seen_identities:
                raise ValueError(
                    "DiversifiedRetentionV2 scored_rows must have unique stable_identity values, "
                    f"got {row.stable_identity!r}"
                )
            seen_identities.add(row.stable_identity)

        bucket_keys_by_identity: dict[str, DiversifiedRetentionBucketKeyV2] = {}
        grouped_rows: dict[DiversifiedRetentionBucketKeyV2, list[GenericRowScorePayloadV2]] = {}
        for row in ordered_rows:
            bucket_key = self._bucket_key_for_row(row=row, config=config)
            bucket_keys_by_identity[row.stable_identity] = bucket_key
            grouped_rows.setdefault(bucket_key, []).append(row)

        bucket_order = sorted(
            grouped_rows.keys(),
            key=lambda bucket_key: (
                grouped_rows[bucket_key][0].sort_key(),
                bucket_key.sort_key(),
            ),
        )

        retained_rounds: dict[str, int] = {}
        retained_bucket_ranks: dict[str, int] = {}
        bucket_offsets: dict[DiversifiedRetentionBucketKeyV2, int] = {
            bucket_key: 0 for bucket_key in bucket_order
        }
        bucket_selected_counts: dict[DiversifiedRetentionBucketKeyV2, int] = {
            bucket_key: 0 for bucket_key in bucket_order
        }
        retained_in_selection_order: list[GenericRowScorePayloadV2] = []
        selection_round = 0
        while len(retained_in_selection_order) < max_candidates:
            selection_round += 1
            selected_in_round = False
            for bucket_key in bucket_order:
                if len(retained_in_selection_order) >= max_candidates:
                    break
                if (
                    config.max_per_bucket is not None
                    and bucket_selected_counts[bucket_key] >= config.max_per_bucket
                ):
                    continue
                bucket_rows = grouped_rows[bucket_key]
                bucket_offset = bucket_offsets[bucket_key]
                if bucket_offset >= len(bucket_rows):
                    continue
                selected_row = bucket_rows[bucket_offset]
                bucket_offsets[bucket_key] = bucket_offset + 1
                bucket_selected_counts[bucket_key] += 1
                retained_in_selection_order.append(selected_row)
                retained_rounds[selected_row.stable_identity] = selection_round
                retained_bucket_ranks[selected_row.stable_identity] = bucket_selected_counts[
                    bucket_key
                ]
                selected_in_round = True
            if not selected_in_round:
                break

        retained_rows = tuple(
            sorted(retained_in_selection_order, key=lambda row: row.sort_key())
        )
        decisions = tuple(
            self._decision_for_row(
                row=row,
                bucket_key=bucket_keys_by_identity[row.stable_identity],
                bucket_rank=grouped_rows[bucket_keys_by_identity[row.stable_identity]].index(row)
                + 1,
                retained_round=retained_rounds.get(row.stable_identity),
                retained_bucket_rank=retained_bucket_ranks.get(row.stable_identity),
                max_candidates=max_candidates,
                config=config,
            )
            for row in ordered_rows
        )
        return DiversifiedRetentionResultV2(
            retained_rows=retained_rows,
            decisions=decisions,
        )

    def _bucket_key_for_row(
        self,
        *,
        row: GenericRowScorePayloadV2,
        config: ExecutionProfileShortlistRetentionConfigV2,
    ) -> DiversifiedRetentionBucketKeyV2:
        """
        Build one composite bucket key from the explicit configured diversity bucket axes.

        Args:
            row: One scored row payload with explicit scorer bucket labels.
            config: Typed retention config describing which bucket axes to use.
        Returns:
            DiversifiedRetentionBucketKeyV2: Explicit composite bucket key for this row.
        Assumptions:
            Bucket axes are configured centrally on the execution profile instead of inferred from
            ad-hoc runtime order.
        Raises:
            ValueError: If the row is missing one required bucket axis.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
        """
        components: list[DiversifiedRetentionBucketComponentV2] = []
        for bucket_name in config.diversity_buckets:
            bucket_value = row.bucket_values.get(bucket_name)
            if bucket_value is None:
                raise ValueError(
                    "DiversifiedRetentionV2 row is missing required bucket axis "
                    f"{bucket_name!r} for {row.stable_identity!r}"
                )
            components.append(
                DiversifiedRetentionBucketComponentV2(
                    bucket_name=bucket_name,
                    bucket_value=bucket_value,
                )
            )
        return DiversifiedRetentionBucketKeyV2(components=tuple(components))

    def _decision_for_row(
        self,
        *,
        row: GenericRowScorePayloadV2,
        bucket_key: DiversifiedRetentionBucketKeyV2,
        bucket_rank: int,
        retained_round: int | None,
        retained_bucket_rank: int | None,
        max_candidates: int,
        config: ExecutionProfileShortlistRetentionConfigV2,
    ) -> DiversifiedRetentionDecisionV2:
        """
        Build one explicit audit decision for a retained or discarded scored row.

        Args:
            row: One scored row payload.
            bucket_key: Composite bucket identity used during retention.
            bucket_rank: Rank of the row inside its bucket by score order.
            retained_round: Selection round when retained, else `None`.
            retained_bucket_rank: Selection rank within bucket when retained, else `None`.
            max_candidates: Global survivor budget used by the current retention run.
            config: Typed retention config for discard-reason classification.
        Returns:
            DiversifiedRetentionDecisionV2: Typed audit record for this row.
        Assumptions:
            Discard reasons stay intentionally coarse but explicit so benchmark/debug views can
            explain why a row did not survive.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
        Related:
          - tests/unit/contexts/backtest/application/services/v2/test_diversified_retention_v2.py
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
        """
        if retained_round is not None and retained_bucket_rank is not None:
            return DiversifiedRetentionDecisionV2(
                row=row,
                bucket_key=bucket_key,
                retained=True,
                bucket_rank=retained_bucket_rank,
                selection_round=retained_round,
            )
        discard_reason = "discarded_after_capacity"
        if config.max_per_bucket is not None and bucket_rank > config.max_per_bucket:
            discard_reason = "discarded_bucket_cap"
        if max_candidates <= 0:
            discard_reason = "discarded_after_capacity"
        return DiversifiedRetentionDecisionV2(
            row=row,
            bucket_key=bucket_key,
            retained=False,
            bucket_rank=bucket_rank,
            selection_round=None,
            discard_reason=discard_reason,
        )


__all__ = [
    "DiversifiedRetentionBucketComponentV2",
    "DiversifiedRetentionBucketKeyV2",
    "DiversifiedRetentionDecisionV2",
    "DiversifiedRetentionResultV2",
    "DiversifiedRetentionV2",
]
