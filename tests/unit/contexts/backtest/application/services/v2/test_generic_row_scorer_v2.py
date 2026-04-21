from __future__ import annotations

import numpy as np
import pytest

from trading.contexts.backtest_artifacts.application.services.v2.generic_row_scorer_v2 import (
    GenericRowScorerV2,
    GenericRowScoringInputV2,
    build_generic_row_signal_features_mapping_v2,
)


def _cached_signal_features(
    *,
    nonzero_count: float,
    long_count: float,
    short_count: float,
    activity_ratio: float,
    direction_balance: float,
    transition_count: float,
):
    """
    Build one canonical cached signal-feature mapping for generic row scorer tests.

    Args:
        nonzero_count: Count of non-neutral bars in one row.
        long_count: Count of long bars in one row.
        short_count: Count of short bars in one row.
        activity_ratio: Non-neutral bar ratio over row length.
        direction_balance: Signed long/short balance in `[-1, 1]`.
        transition_count: Number of adjacent signal-state transitions.
    Returns:
        Mapping[str, float]: Immutable canonical feature mapping.
    Assumptions:
        Tests should use the same fixed feature order as shipped artifact payloads.
    Raises:
        ValueError: If canonical feature names or values are invalid.
    Side Effects:
        None.
    """
    return build_generic_row_signal_features_mapping_v2(
        feature_names=(
            "nonzero_count",
            "long_count",
            "short_count",
            "activity_ratio",
            "direction_balance",
            "transition_count",
        ),
        feature_values=(
            nonzero_count,
            long_count,
            short_count,
            activity_ratio,
            direction_balance,
            transition_count,
        ),
    )


def test_generic_row_scorer_v2_uses_cached_signal_features_and_runtime_row_stats() -> None:
    """
    Verify generic row scorer combines cached features with deterministic row-local runtime stats.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cached `signal_features` are preferred when available, while `active_span_ratio` remains a
        cheap runtime-derived stat.
    Raises:
        AssertionError: If scorer payload fields or deterministic buckets drift.
    Side Effects:
        None.
    """
    scorer = GenericRowScorerV2()
    scored_row = scorer.score_row(
        row=GenericRowScoringInputV2(
            indicator_id="ma.ema",
            row_index=3,
            stable_identity="ma.ema:3",
            signal_row=np.array((0, 1, 1, 0, -1, 0), dtype=np.int8),
            signal_features=_cached_signal_features(
                nonzero_count=3.0,
                long_count=2.0,
                short_count=1.0,
                activity_ratio=0.5,
                direction_balance=1.0 / 3.0,
                transition_count=4.0,
            ),
            metadata={"family": "ma"},
        )
    )

    assert scored_row.total_score == pytest.approx(0.6333333333)
    assert scored_row.signal_features.used_cached_signal_features is True
    assert scored_row.runtime_stats.timeline_length == 6
    assert scored_row.runtime_stats.active_span == 4
    assert scored_row.runtime_stats.active_span_ratio == pytest.approx(4.0 / 6.0)
    assert scored_row.runtime_stats.transition_ratio == pytest.approx(4.0 / 5.0)
    assert scored_row.bucket_values == {
        "activity_band": "high_activity",
        "direction_band": "balanced",
        "transition_band": "high_transition",
    }
    assert [component.component_id for component in scored_row.components] == [
        "activity_ratio",
        "direction_balance",
        "transition_count",
        "active_span_ratio",
    ]
    assert [component.source for component in scored_row.components] == [
        "signal_features",
        "signal_features",
        "signal_features",
        "runtime_row_stats",
    ]


def test_generic_row_scorer_v2_derives_features_and_sorts_by_score_then_identity() -> None:
    """
    Verify generic row scorer derives missing features and returns stable sorted payload order.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Score ordering should not depend on caller iteration order when score and identities are
        explicit.
    Raises:
        AssertionError: If derived feature values or final deterministic order drift.
    Side Effects:
        None.
    """
    scorer = GenericRowScorerV2()
    scored_rows = scorer.score_rows(
        rows=(
            GenericRowScoringInputV2(
                indicator_id="trend.adx",
                row_index=2,
                stable_identity="z-row",
                signal_row=np.array((0, 1, 0, 0), dtype=np.int8),
            ),
            GenericRowScoringInputV2(
                indicator_id="trend.adx",
                row_index=1,
                stable_identity="a-row",
                signal_row=np.array((0, 1, 0, 0), dtype=np.int8),
            ),
            GenericRowScoringInputV2(
                indicator_id="trend.adx",
                row_index=0,
                stable_identity="low-row",
                signal_row=np.array((0, 0, 0, 0), dtype=np.int8),
            ),
        )
    )

    assert [row.stable_identity for row in scored_rows] == ["a-row", "z-row", "low-row"]
    assert scored_rows[0].signal_features.used_cached_signal_features is False
    assert scored_rows[0].signal_features.activity_ratio == pytest.approx(0.25)
    assert scored_rows[0].signal_features.transition_count == pytest.approx(2.0)
    assert scored_rows[0].runtime_stats.active_span_ratio == pytest.approx(0.25)
    assert scored_rows[2].total_score == 0.0
    assert scored_rows[2].bucket_values["activity_band"] == "low_activity"


def test_generic_row_scorer_v2_rejects_incomplete_cached_signal_features() -> None:
    """
    Verify cached feature payload must contain the canonical full signal-feature set.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Universal scorer should fail fast instead of silently accepting partial cached features.
    Raises:
        AssertionError: If invalid feature payload does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="canonical full set"):
        GenericRowScoringInputV2(
            indicator_id="ma.sma",
            row_index=0,
            signal_row=np.array((0, 1, 0, -1), dtype=np.int8),
            signal_features={
                "nonzero_count": 2.0,
                "long_count": 1.0,
                "short_count": 1.0,
                "activity_ratio": 0.5,
                "direction_balance": 0.0,
            },
        )
