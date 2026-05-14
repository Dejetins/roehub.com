from __future__ import annotations

import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    load_backtest_admission_config,
)
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
    BacktestRuntimeGuardrails,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestAdmissionService,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS,
    BacktestJobSchedulingPromotionRequired,
    backtest_numba_environ,
    classify_preflight_scheduling,
    raise_if_light_candidate_needs_heavy_slot,
    scheduling_metadata_from_preflight,
)
from trading.shared_kernel.primitives import PaidLevel


def test_prod_admission_config_preserves_benchmark_top_n_for_ultra_only() -> None:
    service = BacktestAdmissionService(
        config=load_backtest_admission_config("configs/prod/backtest_admission.yaml")
    )
    guardrails = service.preflight_validation_guardrails(
        base_guardrails=BacktestRuntimeGuardrails(),
    )

    assert guardrails.max_top_n == 100
    assert service.config.policy_for(paid_level=PaidLevel("ultra")).max_top_n == 100
    assert service.config.policy_for(paid_level=PaidLevel("pro")).max_top_n == 50


def test_preflight_scheduler_metadata_preserves_conservative_upper_bound() -> None:
    preflight = _preflight(
        estimated_combinations_upper_bound=289_254_654_976,
        scheduling_class="heavy",
    )

    decision = classify_preflight_scheduling(
        preflight=preflight,
        light_max_estimated_combinations=50_000,
    )
    metadata = scheduling_metadata_from_preflight(
        preflight=preflight,
        light_max_estimated_combinations=50_000,
    )

    assert decision.scheduling_class == "heavy"
    assert decision.estimated_combinations_upper_bound == 289_254_654_976
    assert metadata["scheduling_class"] == "heavy"
    assert metadata["estimated_combinations_upper_bound"] == 289_254_654_976
    assert metadata["arity"] == 5
    assert metadata["requested_top_n"] == 50


def test_preflight_scheduler_classifies_bounded_jobs_as_light_candidate_only() -> None:
    metadata = scheduling_metadata_from_preflight(
        preflight=_preflight(
            estimated_combinations_upper_bound=512,
            scheduling_class="light_candidate",
        ),
        light_max_estimated_combinations=50_000,
    )

    assert metadata["scheduling_class"] == "light_candidate"


def test_post_prepare_light_candidate_promotes_to_heavy_before_exact_scoring() -> None:
    with pytest.raises(BacktestJobSchedulingPromotionRequired) as exc_info:
        raise_if_light_candidate_needs_heavy_slot(
            scheduling_class="light_candidate",
            estimated_combinations_upper_bound=512,
            actual_combinations=50_001,
            light_max_actual_combinations=50_000,
        )

    assert exc_info.value.promotion.estimated_combinations_upper_bound == 512
    assert exc_info.value.promotion.actual_combinations == 50_001


def test_backtest_numba_environ_uses_per_class_thread_budget() -> None:
    light_env = backtest_numba_environ(
        environ={
            ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS: "2",
            ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS: "10",
        },
        scheduling_class="light_candidate",
    )
    heavy_env = backtest_numba_environ(
        environ={
            ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS: "2",
            ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS: "10",
        },
        scheduling_class="heavy",
    )

    assert light_env[NUMBA_NUM_THREADS] == "2"
    assert heavy_env[NUMBA_NUM_THREADS] == "10"
    assert light_env["ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"] == (
        ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS
    )
    assert heavy_env["ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"] == (
        ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS
    )


def _preflight(
    *,
    estimated_combinations_upper_bound: int,
    scheduling_class: str,
) -> BacktestPreflightResult:
    return BacktestPreflightResult(
        normalized_request={"risk": {"mode": "none"}, "top_n": 50},
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
            indicator_rows=980,
            candidate_combinations=estimated_combinations_upper_bound,
            tp_sl_cells=0,
            cost_class="large",
            estimated_combinations_upper_bound=estimated_combinations_upper_bound,
            estimated_combinations=estimated_combinations_upper_bound,
            arity=5,
            row_count_upper_bounds_by_indicator={
                f"ma.dema#{index}": 196 for index in range(5)
            },
            risk_mode="none",
            requested_range={
                "start": "2020-01-11T20:08:00Z",
                "end": "2026-04-11T20:08:00Z",
            },
            requested_top_n=50,
            scheduling_class=scheduling_class,
        ),
    )
