from __future__ import annotations

import argparse

import pytest

from scripts.backtest.run_stage_13_tp_sl_block_autotune_gate import (
    CURRENT_EXACT_LABEL,
    STAGE_09_ACCEPTED_LABEL,
    _parse_shape,
    _run_specs,
    build_stage_13_report,
)


def test_parse_shape_accepts_x_or_comma() -> None:
    assert _parse_shape("128x32") == (128, 32)
    assert _parse_shape("32,128") == (32, 128)


def test_parse_shape_rejects_non_positive() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_shape("0x64")


def test_stage_13_run_specs_do_not_duplicate_64x64_candidate() -> None:
    specs = _run_specs(((64, 64), (128, 32)))

    assert [spec.role for spec in specs] == [
        "current_exact_control",
        "stage_09_accepted_control",
        "candidate_shape",
    ]
    assert specs[-1].shape == (128, 32)


def test_stage_13_report_accepts_shape_with_wall_win_and_memory_bound() -> None:
    report = build_stage_13_report(
        run_records=[
            _run_record(label=CURRENT_EXACT_LABEL, role="current_exact_control", shape=None),
            _run_record(
                label=STAGE_09_ACCEPTED_LABEL,
                role="stage_09_accepted_control",
                shape=[64, 64],
                long_wall=10.0,
                reversal_wall=20.0,
                peak_rss=1000,
            ),
            _run_record(
                label="candidate_shape",
                role="candidate_shape",
                shape=[128, 32],
                long_wall=8.0,
                reversal_wall=16.5,
                peak_rss=1050,
            ),
        ]
    )

    assert report["decision"]["status"] == "accepted"
    assert report["decision"]["best_shape"]["shape"] == "128x32"


def test_stage_13_report_rejects_when_best_shape_is_under_threshold() -> None:
    report = build_stage_13_report(
        run_records=[
            _run_record(label=CURRENT_EXACT_LABEL, role="current_exact_control", shape=None),
            _run_record(
                label=STAGE_09_ACCEPTED_LABEL,
                role="stage_09_accepted_control",
                shape=[64, 64],
                long_wall=10.0,
                reversal_wall=20.0,
            ),
            _run_record(
                label="candidate_shape",
                role="candidate_shape",
                shape=[128, 32],
                long_wall=9.0,
                reversal_wall=17.5,
            ),
        ]
    )

    assert report["decision"]["status"] == "rejected"
    assert report["shape_evaluations"][-1]["accepted"] is False


def _run_record(
    *,
    label: str,
    role: str,
    shape: list[int] | None,
    long_wall: float = 12.0,
    reversal_wall: float = 24.0,
    peak_rss: int = 1000,
) -> dict[str, object]:
    return {
        "label": label,
        "role": role,
        "shape": shape,
        "out_dir": f"/tmp/{label}",
        "command": ["python", "benchmark.py"],
        "returncode": 0,
        "payload": {
            "pass": True,
            "git_commit": "abc123",
            "git_status_short": "",
            "host": "macstudio",
            "parity": {"pass": True},
            "instrumentation": {"pass": True},
            "memory_release": {"pass": True},
            "mixed_scheduler_smoke": {"pass": True},
            "lazy_cache_hit_memory": {"pass": True},
            "legacy_path_absence": {"pass": True},
            "dead_code_audit": {"pass": True},
            "docs_drift_audit": {"pass": True},
            "api_runner_path": {
                "pass": True,
                "jobs": [
                    _job("tp_sl_grid/arity_6/long_only", long_wall, peak_rss),
                    _job("tp_sl_grid/arity_6/long_short_reversal", reversal_wall, peak_rss),
                ]
            },
        },
    }


def _job(job_name: str, wall: float, peak_rss: int) -> dict[str, object]:
    top_sample = [
        {
            "rank": 1,
            "indicator_rows": {"ma.ema": 0},
            "best_tp_idx": 2,
            "best_sl_idx": 1,
            "metrics": {
                "best_tp_pct": 3.0,
                "best_sl_pct": 2.0,
                "total_return_pct": 10.0,
                "trade_count": 4.0,
            },
        }
    ]
    return {
        "job_name": job_name,
        "pass": True,
        "parity": {"pass": True},
        "stage_timings": {
            "service_wall_clock_s": wall,
            "service_total_without_warmup": wall - 0.1,
            "exact_scoring": wall - 0.2,
            "tp_sl_exact_scoring": wall - 0.2,
        },
        "instrumentation_counters": {
            "tp_sl_cell_block_shape": "64 x 64",
            "trade_cell_evals_per_sec": 1000.0,
            "tp_sl_cell_trade_cell_evals": 2209,
        },
        "memory": {"items": [{"peak_rss_bytes": peak_rss}]},
        "child_process_evidence": [
            {"exact_diagnostics": {"top_results_sample": top_sample}}
        ],
    }
