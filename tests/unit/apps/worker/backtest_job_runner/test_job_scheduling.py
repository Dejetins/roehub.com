from __future__ import annotations

import pytest

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobSchedulingPromotionRequired,
    classify_preflight_scheduling,
    raise_if_light_candidate_needs_heavy_slot,
    scheduling_metadata_from_preflight,
)


def test_preflight_classifies_all_full_jobs_as_heavy() -> None:
    preflight = _preflight(candidate_combinations=100)

    decision = classify_preflight_scheduling(
        preflight=preflight,
        light_max_estimated_combinations=50000,
    )

    assert decision.scheduling_class == "heavy"
    assert scheduling_metadata_from_preflight(preflight=preflight)["scheduling_class"] == "heavy"


def test_preflight_classifies_obvious_heavy_jobs_as_heavy() -> None:
    decision = classify_preflight_scheduling(
        preflight=_preflight(candidate_combinations=50001),
        light_max_estimated_combinations=50000,
    )

    assert decision.scheduling_class == "heavy"
    assert decision.estimated_combinations_upper_bound == 50001


def test_light_candidate_is_promoted_before_exact_scoring_when_actual_cost_exceeds_limit() -> None:
    with pytest.raises(BacktestJobSchedulingPromotionRequired) as exc_info:
        raise_if_light_candidate_needs_heavy_slot(
            scheduling_class="light_candidate",
            estimated_combinations_upper_bound=10,
            actual_combinations=100000,
            light_max_actual_combinations=50000,
        )

    assert exc_info.value.promotion.actual_combinations == 100000


def test_light_candidate_is_confirmed_as_light_when_actual_cost_stays_bounded() -> None:
    assert (
        raise_if_light_candidate_needs_heavy_slot(
            scheduling_class="light_candidate",
            estimated_combinations_upper_bound=10,
            actual_combinations=100,
            light_max_actual_combinations=50000,
        )
        == "light"
    )


def _preflight(*, candidate_combinations: int) -> BacktestPreflightResult:
    return BacktestPreflightResult(
        normalized_request={"risk": {"mode": "none"}},
        request_hash="d" * 64,
        result_config_hash="e" * 64,
        artifact_metadata=BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=1,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-05-13",
            hit_times_manifest_hash=None,
            published_at_utc="2026-05-13T00:00:00Z",
        ),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=1,
            candidate_combinations=candidate_combinations,
            tp_sl_cells=0,
            cost_class="small",
        ),
    )
