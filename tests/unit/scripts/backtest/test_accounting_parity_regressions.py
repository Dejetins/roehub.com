from types import SimpleNamespace

import pytest

from scripts.backtest.run_api_runner_benchmark_parity import _compare_top_result_samples
from trading.contexts.backtest.application.services.v2.job_orchestration import _stage_timings


def test_elapsed_excludes_warmup_without_summing_nested_stages():
    result = _stage_timings(
        prepared_result=SimpleNamespace(
            timing=SimpleNamespace(
                subsegments={
                    "prepare_pools_core": 2.0,
                    "prepare_pools_total": 3.0,
                    "signal_row_selection": 1.0,
                }
            )
        ),
        combo_result=None,
        hit_times_result=None,
        exact_result=SimpleNamespace(
            telemetry=SimpleNamespace(
                stage_timings={"exact_scoring": 5.0, "tp_sl_exact_scoring": 5.0}
            )
        ),
        assembly_timings={"top_result_assembly": 1.0},
        elapsed=12.0,
        warmup_elapsed_s=3.0,
    )
    assert result["service_total_without_warmup"] == 9.0


def test_missing_metric_is_a_mismatch():
    assert _compare_top_result_samples(
        actual_items=[{"metrics": {}}],
        expected_items=[{"metrics": {"total_return_pct": 5.0}}],
        actual_metrics_key="metrics",
    )


def test_equal_metrics_do_not_hide_wrong_identity():
    assert _compare_top_result_samples(
        actual_items=[{"rank": 1, "indicator_rows": {"a": 2}, "metrics": {"x": 5.0}}],
        expected_items=[{"rank": 1, "indicator_rows": {"a": 1}, "metrics": {"x": 5.0}}],
        actual_metrics_key="metrics",
    )


@pytest.mark.parametrize(
    "elapsed,warmup",
    [(None, 0), (-1, None), (float("nan"), None), (12, -1), (12, 13), (12, float("inf"))],
)
def test_invalid_timing_evidence_is_rejected(elapsed, warmup):
    with pytest.raises(ValueError):
        _stage_timings(
            prepared_result=None,
            combo_result=None,
            hit_times_result=None,
            exact_result=None,
            assembly_timings={},
            elapsed=elapsed,
            warmup_elapsed_s=warmup,
        )


def test_absent_warmup_means_not_run_and_missing_stages_are_not_zero():
    result = _stage_timings(
        prepared_result=None,
        combo_result=None,
        hit_times_result=None,
        exact_result=None,
        assembly_timings={},
        elapsed=12,
        warmup_elapsed_s=None,
    )
    assert result == {"service_wall_clock_s": 12, "service_total_without_warmup": 12}


@pytest.mark.parametrize(
    "timers,status",
    [
        ({}, "not_assessed"),
        (
            {"service_wall_clock_s": 12, "sample_warmup": 3, "service_total_without_warmup": 17},
            "failed",
        ),
        (
            {"service_wall_clock_s": 12, "sample_warmup": 13, "service_total_without_warmup": 0},
            "failed",
        ),
        (
            {"service_wall_clock_s": 12, "sample_warmup": 3, "service_total_without_warmup": 9},
            "passed",
        ),
    ],
)
def test_timing_report_validates_current_attempt_only(timers, status):
    from scripts.backtest.run_api_runner_benchmark_parity import _timing_accounting

    diagnostics = {
        "telemetry": {},
        "timing_accounting": {
            "schema": "orchestration_elapsed_v2",
            "warmup": "measured_inside_interval",
        },
    }
    prior = {
        "stage_timings": {
            "service_wall_clock_s": 12,
            "sample_warmup": 3,
            "service_total_without_warmup": 9,
        },
        "exact_diagnostics": diagnostics,
    }
    latest = {"stage_timings": timers, "exact_diagnostics": diagnostics}
    assert _timing_accounting([prior, latest])["status"] == status
    assert _timing_accounting([{"stage_timings": timers}])["status"] == "not_assessed"


def test_notebook_validator_does_not_accept_api_runner_schema(tmp_path):
    from trading.contexts.backtest.application.services.v2.benchmark_accounting import (
        BenchmarkAccountingError,
        validate_canonical_benchmark_json,
    )

    path = tmp_path / "api-runner.json"
    path.write_text('{"schema": "backtest_api_runner_compute_memory_parity_v2", "jobs": []}')
    with pytest.raises(BenchmarkAccountingError, match="runs"):
        validate_canonical_benchmark_json(path)
