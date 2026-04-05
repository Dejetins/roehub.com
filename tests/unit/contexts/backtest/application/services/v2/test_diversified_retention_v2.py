from __future__ import annotations

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    DiversifiedRetentionV2,
    ExecutionProfileShortlistRetentionConfigV2,
    GenericRowScorePayloadV2,
    GenericRowScorerV2,
    GenericRowScoringInputV2,
)


def _scored_retention_rows() -> tuple[GenericRowScorePayloadV2, ...]:
    """
    Build deterministic scored rows that exercise explicit diversity retention behavior.

    Args:
        None.
    Returns:
        tuple[GenericRowScorePayloadV2, ...]: Scored rows spanning repeated and distinct buckets.
    Assumptions:
        Tests need two strong rows in the same bucket so diversified retention can prove it is not
        equivalent to pure top-N truncation.
    Raises:
        None.
    Side Effects:
        None.
    """
    scorer = GenericRowScorerV2()
    return scorer.score_rows(
        rows=(
            GenericRowScoringInputV2(
                indicator_id="ma.ema",
                row_index=0,
                stable_identity="row-a",
                signal_row=np.array((1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0), dtype=np.int8),
            ),
            GenericRowScoringInputV2(
                indicator_id="ma.ema",
                row_index=1,
                stable_identity="row-b",
                signal_row=np.array((1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0), dtype=np.int8),
            ),
            GenericRowScoringInputV2(
                indicator_id="ma.ema",
                row_index=2,
                stable_identity="row-c",
                signal_row=np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), dtype=np.int8),
            ),
            GenericRowScoringInputV2(
                indicator_id="ma.ema",
                row_index=3,
                stable_identity="row-d",
                signal_row=np.array((0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0), dtype=np.int8),
            ),
        )
    )


def test_diversified_retention_v2_keeps_bucket_diversity_before_raw_top_n() -> None:
    """
    Verify diversified retention keeps explicit bucket diversity instead of raw-score truncation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `row-a` and `row-b` intentionally share the same `(activity_band, direction_band)` bucket.
    Raises:
        AssertionError: If survivors, order, or discard reasons drift.
    Side Effects:
        None.
    """
    scored_rows = _scored_retention_rows()
    retention = DiversifiedRetentionV2()

    result = retention.retain_rows(
        scored_rows=scored_rows,
        config=ExecutionProfileShortlistRetentionConfigV2(
            diversity_buckets=("activity_band", "direction_band"),
            max_per_bucket=1,
        ),
        max_candidates=3,
    )

    assert [row.stable_identity for row in result.retained_rows] == [
        "row-d",
        "row-a",
        "row-c",
    ]
    decisions_by_identity = {
        decision.row.stable_identity: decision for decision in result.decisions
    }
    assert decisions_by_identity["row-a"].retained is True
    assert decisions_by_identity["row-d"].selection_round == 1
    assert decisions_by_identity["row-b"].retained is False
    assert decisions_by_identity["row-b"].discard_reason == "discarded_bucket_cap"
    assert decisions_by_identity["row-c"].bucket_key.sort_key() == (
        ("activity_band", "low_activity"),
        ("direction_band", "long_bias"),
    )


def test_diversified_retention_v2_rejects_missing_bucket_axis() -> None:
    """
    Verify diversified retention fails fast when a scored row lacks one required bucket axis.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Bucket identity must remain explicit and complete for deterministic survivor selection.
    Raises:
        AssertionError: If missing bucket axis does not raise ValueError.
    Side Effects:
        None.
    """
    scored_row = _scored_retention_rows()[0]
    invalid_row = GenericRowScorePayloadV2(
        indicator_id=scored_row.indicator_id,
        row_index=scored_row.row_index,
        stable_identity=scored_row.stable_identity,
        total_score=scored_row.total_score,
        signal_features=scored_row.signal_features,
        runtime_stats=scored_row.runtime_stats,
        bucket_values={"activity_band": "high_activity"},
        components=scored_row.components,
        metadata=scored_row.metadata,
    )

    with pytest.raises(ValueError, match="required bucket axis"):
        DiversifiedRetentionV2().retain_rows(
            scored_rows=(invalid_row,),
            config=ExecutionProfileShortlistRetentionConfigV2(
                diversity_buckets=("activity_band", "direction_band"),
            ),
            max_candidates=1,
        )
