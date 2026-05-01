from __future__ import annotations

import argparse
import gc
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.backtest.run_iteration_4_2_exact_scoring_benchmark import (  # noqa: E402
    DEFAULT_CANONICAL_JSON,
    TARGET_RISK_MODE,
    _build_services,
    _git_commit,
    _git_status_short,
    _limit_prepared_rows,
    _load_json,
    _render_json,
    _required_mapping,
    _service_request,
)

DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_4_7_memory_cleanup"
)
DEFAULT_ARITY = 7
DEFAULT_DIRECTION_MODE = "long_short_reversal"
DEFAULT_REPEAT_COUNT = 3
DEFAULT_WARMUP_RUNS = 1
BYTES_PER_MIB = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 4.7 no-risk repeated-run memory cleanup smoke. "
            "This records service hygiene evidence, not canonical benchmark timing."
        )
    )
    parser.add_argument(
        "--canonical-json",
        type=Path,
        default=DEFAULT_CANONICAL_JSON,
        help="Canonical notebook benchmark_results.json path used for request shape.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Evidence directory for benchmark_results.json and benchmark_summary.md.",
    )
    parser.add_argument(
        "--artifact-config",
        type=Path,
        default=None,
        help=(
            "Optional backtest_artifacts.yaml path. Defaults to "
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG, then ROEHUB_ENV=prod."
        ),
    )
    parser.add_argument(
        "--arity",
        type=int,
        default=DEFAULT_ARITY,
        help="No-risk arity to smoke. Default uses the heaviest canonical arity.",
    )
    parser.add_argument(
        "--direction-mode",
        default=DEFAULT_DIRECTION_MODE,
        choices=("long_only", "long_short_reversal"),
        help="Direction mode for the repeated no-risk request.",
    )
    parser.add_argument(
        "--rows-per-indicator",
        type=int,
        default=6,
        help="Measured rows per indicator. Canonical target uses 6.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="Measured repeated-run count in one process lifecycle.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Unrecorded same-workload warmup runs before measured repeats.",
    )
    parser.add_argument(
        "--self-check-n",
        type=int,
        default=2,
        help="Self-check sample size for the smoke workload.",
    )
    parser.add_argument(
        "--no-fail-on-growth",
        action="store_true",
        help="Write evidence and return 0 even when retained RSS grows monotonically.",
    )
    args = parser.parse_args(argv)

    if args.arity <= 0:
        parser.error("--arity must be > 0")
    if args.rows_per_indicator <= 0:
        parser.error("--rows-per-indicator must be > 0")
    if args.repeat_count < 3:
        parser.error("--repeat-count must be >= 3")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.self_check_n < 0:
        parser.error("--self-check-n must be >= 0")

    canonical = _load_json(args.canonical_json)
    services = _build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=5,
        self_check_n=args.self_check_n,
    )
    request = _service_request(
        canonical_request=_required_mapping(canonical, "request"),
        arity=args.arity,
        direction_mode=args.direction_mode,
    )

    for warmup_index in range(args.warmup_runs):
        _run_one_smoke(
            services=services,
            request=request,
            rows_per_indicator=args.rows_per_indicator,
            run_index=warmup_index + 1,
            measured=False,
        )

    runs = [
        _run_one_smoke(
            services=services,
            request=request,
            rows_per_indicator=args.rows_per_indicator,
            run_index=run_index,
            measured=True,
        )
        for run_index in range(1, args.repeat_count + 1)
    ]
    payload = _build_payload(
        canonical_json=args.canonical_json,
        services=services,
        request=request,
        runs=runs,
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
        repeat_count=args.repeat_count,
        warmup_runs=args.warmup_runs,
        self_check_n=args.self_check_n,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "benchmark_results.json"
    summary_path = args.out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    if payload["pass"] or args.no_fail_on_growth:
        return 0
    return 1


def _run_one_smoke(
    *,
    services: Any,
    request: Mapping[str, Any],
    rows_per_indicator: int,
    run_index: int,
    measured: bool,
) -> dict[str, Any]:
    rss_before = _current_rss_bytes()
    run_start = time.perf_counter()
    prepared_full = services.prepare_pools.execute(
        normalized_request=request,
        artifact_metadata=services.artifact_metadata,
    )
    measured_prepared = _limit_prepared_rows(
        prepared_full,
        rows_per_indicator=rows_per_indicator,
    )
    combo_result = services.combo_planning.execute(
        prepared_result=measured_prepared,
        normalized_request=request,
    )
    exact_result = services.exact_measured.execute(
        prepared_result=measured_prepared,
        combo_planning_result=combo_result,
        normalized_request=request,
    )
    result_mapping = exact_result.as_mapping()
    result_compact = (
        bool(exact_result.memory_cleanup_evidence.result_is_compact)
        and not _contains_ndarray(result_mapping)
    )
    result_hash = exact_result.canonical_top_results_hash()
    cleanup_evidence = exact_result.memory_cleanup_evidence.as_mapping()
    stage_timings = dict(exact_result.telemetry.stage_timings)
    top_results_count = exact_result.telemetry.top_results_count
    exact_candidates_evaluated = exact_result.telemetry.exact_candidates_evaluated
    backend = exact_result.telemetry.backend_logical_name
    implementation = exact_result.telemetry.backend_implementation_id
    service_cleanup_duration_s = cleanup_evidence["cleanup_duration_s"]
    rss_after_result = _current_rss_bytes()
    gc_start = time.perf_counter()
    del result_mapping
    del exact_result
    del combo_result
    del measured_prepared
    del prepared_full
    collected = gc.collect()
    harness_cleanup_duration_s = time.perf_counter() - gc_start
    rss_after_cleanup = _current_rss_bytes()
    rss_peak = max(_peak_rss_bytes(), rss_before, rss_after_result, rss_after_cleanup)
    retained_rss_delta = rss_after_cleanup - rss_before
    run_wall_s = time.perf_counter() - run_start

    return {
        "run_index": run_index,
        "measured": measured,
        "risk_mode": TARGET_RISK_MODE,
        "direction_mode": str(request["execution"]["direction_mode"]),
        "arity": len(request["indicators"]),
        "backend": backend,
        "backend_implementation_id": implementation,
        "rows_per_indicator": rows_per_indicator,
        "top_results_count": top_results_count,
        "exact_candidates_evaluated": exact_candidates_evaluated,
        "result_hash": result_hash,
        "result_compact": result_compact,
        "service_cleanup_duration_s": service_cleanup_duration_s,
        "harness_cleanup_duration_s": harness_cleanup_duration_s,
        "cleanup_duration_s": service_cleanup_duration_s,
        "gc_collect_ran": True,
        "gc_collected_objects": collected,
        "rss_before": rss_before,
        "rss_peak": rss_peak,
        "rss_after_result": rss_after_result,
        "rss_after_cleanup": rss_after_cleanup,
        "retained_rss_delta": retained_rss_delta,
        "rss_before_mb": _bytes_to_mib(rss_before),
        "rss_peak_mb": _bytes_to_mib(rss_peak),
        "rss_after_result_mb": _bytes_to_mib(rss_after_result),
        "rss_after_cleanup_mb": _bytes_to_mib(rss_after_cleanup),
        "retained_rss_delta_mb": _bytes_to_mib(retained_rss_delta),
        "stage_timings": stage_timings,
        "run_wall_s": run_wall_s,
        "memory_cleanup_evidence": cleanup_evidence,
    }


def _build_payload(
    *,
    canonical_json: Path,
    services: Any,
    request: Mapping[str, Any],
    runs: list[dict[str, Any]],
    arity: int,
    direction_mode: str,
    rows_per_indicator: int,
    repeat_count: int,
    warmup_runs: int,
    self_check_n: int,
) -> dict[str, Any]:
    retained_rss_deltas = [int(run["retained_rss_delta"]) for run in runs]
    rss_after_cleanup_values = [int(run["rss_after_cleanup"]) for run in runs]
    monotonic_retained_rss_growth = _strictly_increasing(retained_rss_deltas)
    monotonic_rss_after_cleanup_growth = _strictly_increasing(rss_after_cleanup_values)
    worker_recycled = False
    compact_results = all(bool(run["result_compact"]) for run in runs)
    pass_payload = {
        "compact_results": compact_results,
        "repeated_run_count": len(runs) >= 3,
        "no_monotonic_retained_rss_growth": not monotonic_retained_rss_growth,
        "worker_recycled": worker_recycled,
    }
    pass_payload["overall"] = (
        pass_payload["compact_results"]
        and pass_payload["repeated_run_count"]
        and (
            pass_payload["no_monotonic_retained_rss_growth"]
            or pass_payload["worker_recycled"]
        )
    )
    return {
        "schema": "backtest_iteration_4_7_memory_cleanup_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "canonical_json": str(canonical_json),
        "artifact_config_path": str(services.artifact_config_path),
        "artifact_root": str(services.artifact_root),
        "artifact_manifest_hash": services.artifact_manifest_hash,
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "arity": arity,
            "direction_mode": direction_mode,
            "rows_per_indicator": rows_per_indicator,
            "repeat_count": repeat_count,
            "warmup_runs": warmup_runs,
            "self_check_n": self_check_n,
            "stage_scope": "memory_cleanup_evidence",
            "service_hygiene_not_canonical_stage": True,
            "canonical_stage_list_changed": False,
            "cleanup_compared_to_notebook_target": False,
        },
        "request": {
            "top_n": int(request.get("top_n", 100)),
            "benchmark_top_k": 5,
            "indicator_ids": [
                str(indicator["indicator_id"]) for indicator in request["indicators"]
            ],
        },
        "memory_cleanup_evidence": {
            "rss_before": runs[0]["rss_before"],
            "rss_peak": max(int(run["rss_peak"]) for run in runs),
            "rss_after_cleanup": runs[-1]["rss_after_cleanup"],
            "retained_rss_delta": (
                int(runs[-1]["rss_after_cleanup"]) - int(runs[0]["rss_before"])
            ),
            "rss_before_mb": _bytes_to_mib(int(runs[0]["rss_before"])),
            "rss_peak_mb": _bytes_to_mib(max(int(run["rss_peak"]) for run in runs)),
            "rss_after_cleanup_mb": _bytes_to_mib(int(runs[-1]["rss_after_cleanup"])),
            "retained_rss_delta_mb": _bytes_to_mib(
                int(runs[-1]["rss_after_cleanup"]) - int(runs[0]["rss_before"])
            ),
            "max_cleanup_duration_s": max(
                float(run["cleanup_duration_s"] or 0.0) for run in runs
            ),
            "retained_rss_delta_series": retained_rss_deltas,
            "retained_rss_delta_series_mb": [
                _bytes_to_mib(value) for value in retained_rss_deltas
            ],
            "rss_after_cleanup_series": rss_after_cleanup_values,
            "rss_after_cleanup_series_mb": [
                _bytes_to_mib(value) for value in rss_after_cleanup_values
            ],
            "repeated_run_count": len(runs),
            "monotonic_retained_rss_growth": monotonic_retained_rss_growth,
            "monotonic_rss_after_cleanup_growth": monotonic_rss_after_cleanup_growth,
            "worker_recycled": worker_recycled,
            "worker_recycle_applicable": False,
            "pass": pass_payload["overall"],
        },
        "acceptance": {
            "notebook_90_percent_comparison_applicable": False,
            "reason": (
                "memory cleanup evidence is service hygiene, not a canonical "
                "notebook stage"
            ),
            "pass_definition": (
                "compact result DTOs and no monotonic retained_rss_delta growth "
                "across at least three same-process runs, or proven worker recycle"
            ),
        },
        "runs": runs,
        "pass_detail": pass_payload,
        "pass": pass_payload["overall"],
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    evidence = _required_mapping(payload, "memory_cleanup_evidence")
    scope = _required_mapping(payload, "scope")
    lines = [
        "# Iteration 4.7 memory cleanup evidence",
        "",
        "Repeated no-risk service smoke for bounded per-job reference lifecycle.",
        "",
        "## Scope",
        "",
        "- Implemented check: `memory cleanup evidence` for no-risk scoring result compactness.",
        "- Not in scope: canonical benchmark stage addition, notebook timer comparison, "
        "scoring changes.",
        "- Cleanup is service hygiene, not a canonical benchmark stage.",
        "",
        "## Version",
        "",
        f"- Branch/commit: `{payload['git_commit']}`",
        f"- Git status: `{payload['git_status_short']}`",
        "- Service command: "
        "`uv run python scripts/backtest/run_iteration_4_7_memory_cleanup_smoke.py`",
        f"- Artifact config: `{payload['artifact_config_path']}`",
        f"- Artifact root: `{payload['artifact_root']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        f"- Canonical JSON for request shape: `{payload['canonical_json']}`",
        "",
        "## Environment",
        "",
        f"- Host: `{payload['host']}`",
        f"- Python: `{payload['python']}`",
        "- Worker lifecycle: one Python process; production worker recycle boundary is "
        "not implemented here.",
        "",
        "## Fixture",
        "",
        f"- Risk mode: `{TARGET_RISK_MODE}`",
        f"- Arity: `{scope['arity']}`",
        f"- Direction mode: `{scope['direction_mode']}`",
        f"- Rows per indicator: `{scope['rows_per_indicator']}`",
        f"- Warmup runs: `{scope['warmup_runs']}`",
        f"- Repeated run count: `{evidence['repeated_run_count']}`",
        "",
        "## Memory Cleanup Evidence",
        "",
        "Cleanup evidence is a service hygiene check, not a canonical notebook stage. It "
        "does not change the ordered stage list and is not compared with `>= 90%` "
        "notebook targets.",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| cleanup_duration_s | {float(evidence['max_cleanup_duration_s']):.9f} |",
        f"| rss_before | {int(evidence['rss_before'])} |",
        f"| rss_peak | {int(evidence['rss_peak'])} |",
        f"| rss_after_cleanup | {int(evidence['rss_after_cleanup'])} |",
        f"| retained_rss_delta | {int(evidence['retained_rss_delta'])} |",
        f"| rss_before_mb | {float(evidence['rss_before_mb']):.3f} |",
        f"| rss_peak_mb | {float(evidence['rss_peak_mb']):.3f} |",
        f"| rss_after_cleanup_mb | {float(evidence['rss_after_cleanup_mb']):.3f} |",
        f"| retained_rss_delta_mb | {float(evidence['retained_rss_delta_mb']):.3f} |",
        "| retained_rss_delta_series_mb | "
        f"`{_format_float_series(evidence['retained_rss_delta_series_mb'])}` |",
        "| rss_after_cleanup_series_mb | "
        f"`{_format_float_series(evidence['rss_after_cleanup_series_mb'])}` |",
        f"| repeated_run_count | {int(evidence['repeated_run_count'])} |",
        f"| monotonic_retained_rss_growth | `{evidence['monotonic_retained_rss_growth']}` |",
        "| monotonic_rss_after_cleanup_growth | "
        f"`{evidence['monotonic_rss_after_cleanup_growth']}` |",
        f"| worker_recycled | `{evidence['worker_recycled']}` |",
        f"| pass | `{evidence['pass']}` |",
        "",
        "Per-run retained RSS:",
        "",
        "| run | rss_before_mb | rss_peak_mb | rss_after_cleanup_mb | "
        "retained_rss_delta_mb | cleanup_duration_s | compact |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            "| {run_index} | {before:.3f} | {peak:.3f} | {after:.3f} | {delta:.3f} | "
            "{cleanup:.9f} | `{compact}` |".format(
                run_index=run["run_index"],
                before=run["rss_before_mb"],
                peak=run["rss_peak_mb"],
                after=run["rss_after_cleanup_mb"],
                delta=run["retained_rss_delta_mb"],
                cleanup=float(run["cleanup_duration_s"] or 0.0),
                compact=run["result_compact"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "- macOS RSS note: allocator caches may keep RSS above the starting value; "
            "this evidence checks the per-run `retained_rss_delta` trend and compact "
            "DTOs rather than expecting immediate OS return.",
            "",
        ]
    )
    return "\n".join(lines)


def _current_rss_bytes() -> int:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0
    output = result.stdout.strip()
    if not output:
        return 0
    try:
        return int(output.splitlines()[-1].strip()) * 1024
    except ValueError:
        return 0
def _peak_rss_bytes() -> int:
    ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return ru_maxrss
    return ru_maxrss * 1024


def _strictly_increasing(values: list[int]) -> bool:
    return len(values) >= 2 and all(
        later > earlier for earlier, later in zip(values, values[1:])
    )


def _bytes_to_mib(value: int | float) -> float:
    return float(value) / float(BYTES_PER_MIB)


def _format_float_series(values: object) -> str:
    if not isinstance(values, list):
        return str(values)
    return ", ".join(f"{float(value):.3f}" for value in values)


def _contains_ndarray(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, Mapping):
        return any(_contains_ndarray(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_ndarray(item) for item in value)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
