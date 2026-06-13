from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "backtest" / "run_api_runner_benchmark_parity.py"
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_matrix_bitset_stage_13_tp_sl_block_autotune"
)

CURRENT_EXACT_LABEL = "current_exact"
STAGE_09_ACCEPTED_LABEL = "stage_09_accepted_64x64"
STAGE_13_REQUIRED_SHAPES: tuple[tuple[int, int], ...] = (
    (64, 64),
    (128, 32),
    (32, 128),
    (128, 64),
    (64, 128),
)
SERVICE_WALL_IMPROVEMENT_MIN = 0.15
MEMORY_PEAK_REGRESSION_MAX = 0.10
TOP_RESULT_FLOAT_TOLERANCE = 1.0e-5


@dataclass(frozen=True, slots=True)
class Stage13RunSpec:
    label: str
    role: str
    shape: tuple[int, int] | None
    args: tuple[str, ...]

    @property
    def output_subdir(self) -> str:
        if self.shape is None:
            return self.label
        tp_count, sl_count = self.shape
        return f"{self.label}_{tp_count}x{sl_count}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    shapes = _unique_shapes([_parse_shape(raw) for raw in args.shape])
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    for spec in _run_specs(shapes):
        run_records.append(_run_spec(args=args, spec=spec, out_dir=out_dir))

    report = build_stage_13_report(run_records=run_records)
    results_path = out_dir / "benchmark_results.json"
    summary_path = out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(report) + "\n", encoding="utf-8")
    summary_path.write_text(render_stage_13_summary(report), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")

    accepted = _mapping(report.get("decision")).get("status") == "accepted"
    return 0 if accepted or args.no_fail_on_rejection else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 13 TP/SL block-shape autotune production gate."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--session-ttl-seconds", type=int, default=7200)
    parser.add_argument("--light-max-actual-combinations", type=int, default=50_000)
    parser.add_argument("--system-memory-cleanup-wait-seconds", type=float, default=30.0)
    parser.add_argument("--cpu-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--allow-backlog", action="store_true")
    parser.add_argument("--no-fail-on-rejection", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Do not execute child benchmarks when the expected subrun "
            "benchmark_results.json already exists; regenerate aggregate only."
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[f"{tp}x{sl}" for tp, sl in STAGE_13_REQUIRED_SHAPES],
        help="Candidate TP/SL block shape, for example 128x32. Repeatable.",
    )
    return parser


def _run_specs(shapes: Sequence[tuple[int, int]]) -> tuple[Stage13RunSpec, ...]:
    specs = [
        Stage13RunSpec(
            label=CURRENT_EXACT_LABEL,
            role="current_exact_control",
            shape=None,
            args=("--stage-13-tp-sl-current-exact-rows",),
        ),
        Stage13RunSpec(
            label=STAGE_09_ACCEPTED_LABEL,
            role="stage_09_accepted_control",
            shape=(64, 64),
            args=(
                "--stage-09-tp-sl-full-grid",
                "--tp-sl-cell-block-tp-count",
                "64",
                "--tp-sl-cell-block-sl-count",
                "64",
            ),
        ),
    ]
    for tp_count, sl_count in shapes:
        if (tp_count, sl_count) == (64, 64):
            continue
        specs.append(
            Stage13RunSpec(
                label="candidate_shape",
                role="candidate_shape",
                shape=(tp_count, sl_count),
                args=(
                    "--stage-09-tp-sl-full-grid",
                    "--tp-sl-cell-block-tp-count",
                    str(tp_count),
                    "--tp-sl-cell-block-sl-count",
                    str(sl_count),
                ),
            )
        )
    return tuple(specs)


def _run_spec(
    *,
    args: argparse.Namespace,
    spec: Stage13RunSpec,
    out_dir: Path,
) -> dict[str, Any]:
    run_out_dir = out_dir / spec.output_subdir
    payload_path = run_out_dir / "benchmark_results.json"
    if args.reuse_existing and payload_path.is_file():
        print(f"reusing {payload_path}")
        return {
            "label": spec.label,
            "role": spec.role,
            "shape": None if spec.shape is None else list(spec.shape),
            "out_dir": str(run_out_dir),
            "command": ["reuse-existing", str(payload_path)],
            "returncode": 0,
            "payload": _load_json(payload_path),
        }
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--out-dir",
        str(run_out_dir),
        "--api-base",
        str(args.api_base),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--session-ttl-seconds",
        str(args.session_ttl_seconds),
        "--light-max-actual-combinations",
        str(args.light_max_actual_combinations),
        "--system-memory-cleanup-wait-seconds",
        str(args.system_memory_cleanup_wait_seconds),
        "--cpu-sample-interval-seconds",
        str(args.cpu_sample_interval_seconds),
        "--no-fail-on-threshold",
        *spec.args,
    ]
    if args.env_file is not None:
        command.extend(("--env-file", str(args.env_file)))
    if args.allow_backlog:
        command.append("--allow-backlog")
    print(f"running {spec.output_subdir}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    payload = _load_json(payload_path) if payload_path.is_file() else None
    return {
        "label": spec.label,
        "role": spec.role,
        "shape": None if spec.shape is None else list(spec.shape),
        "out_dir": str(run_out_dir),
        "command": command,
        "returncode": completed.returncode,
        "payload": payload,
    }


def build_stage_13_report(
    *,
    run_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_summaries = [_summarize_run(record) for record in run_records]
    current = _find_run(run_summaries, role="current_exact_control")
    accepted = _find_run(run_summaries, role="stage_09_accepted_control")
    shape_runs = [
        run
        for run in run_summaries
        if run.get("shape") is not None
        and run.get("role") in {"stage_09_accepted_control", "candidate_shape"}
    ]
    rows = _shape_matrix_rows(
        current=current,
        accepted=accepted,
        shape_runs=shape_runs,
    )
    shape_evaluations = _shape_evaluations(rows=rows, shape_runs=shape_runs)
    best_shape = _best_shape(shape_evaluations)
    controls_pass = bool(current and current.get("pass")) and bool(
        accepted and accepted.get("pass")
    )
    missing_payloads = [
        str(run.get("label"))
        for run in run_summaries
        if not isinstance(run.get("payload"), Mapping)
    ]
    crashed_runs = [
        str(run.get("label"))
        for run in run_summaries
        if int(run.get("returncode") or 0) != 0 and not isinstance(run.get("payload"), Mapping)
    ]
    if missing_payloads or crashed_runs or not controls_pass:
        status = "blocked"
        reason = "missing_or_failed_control_evidence"
    elif best_shape is not None and bool(best_shape.get("accepted")):
        status = "accepted"
        reason = "best_shape_passed_service_wall_memory_and_parity_gate"
    else:
        status = "rejected"
        reason = "no_shape_met_stage_13_service_wall_gate"
    return {
        "schema": "backtest_stage_13_tp_sl_block_autotune_gate_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "acceptance": {
            "service_wall_improvement_min": SERVICE_WALL_IMPROVEMENT_MIN,
            "memory_peak_regression_max": MEMORY_PEAK_REGRESSION_MAX,
            "comparison_path": STAGE_09_ACCEPTED_LABEL,
            "required_rows": [
                "tp_sl_grid/arity_6/long_only",
                "tp_sl_grid/arity_6/long_short_reversal",
            ],
        },
        "runs": run_summaries,
        "shape_matrix": rows,
        "shape_evaluations": shape_evaluations,
        "decision": {
            "status": status,
            "reason": reason,
            "best_shape": best_shape,
            "controls_pass": controls_pass,
            "missing_payloads": missing_payloads,
            "crashed_runs": crashed_runs,
            "production_default_change": "not_allowed_unless_status_accepted",
        },
    }


def _summarize_run(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    jobs = []
    if isinstance(payload, Mapping):
        for job in _list(_mapping(payload.get("api_runner_path")).get("jobs")):
            job_map = _mapping(job)
            jobs.append(_summarize_job(job_map))
    return {
        "label": record.get("label"),
        "role": record.get("role"),
        "shape": record.get("shape"),
        "out_dir": record.get("out_dir"),
        "command": record.get("command"),
        "returncode": record.get("returncode"),
        "payload": payload if isinstance(payload, Mapping) else None,
        "payload_pass": bool(payload.get("pass")) if isinstance(payload, Mapping) else False,
        "pass": _stage_13_input_pass(payload),
        "git_commit": payload.get("git_commit") if isinstance(payload, Mapping) else None,
        "git_status_short": (
            payload.get("git_status_short") if isinstance(payload, Mapping) else None
        ),
        "host": payload.get("host") if isinstance(payload, Mapping) else None,
        "jobs": jobs,
    }


def _stage_13_input_pass(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    checks = (
        _mapping(payload.get("api_runner_path")).get("pass"),
        _mapping(payload.get("parity")).get("pass"),
        _mapping(payload.get("instrumentation")).get("pass"),
        _mapping(payload.get("memory_release")).get("pass"),
        _mapping(payload.get("mixed_scheduler_smoke")).get("pass"),
        _mapping(payload.get("lazy_cache_hit_memory")).get("pass"),
        _mapping(payload.get("legacy_path_absence")).get("pass"),
        _mapping(payload.get("dead_code_audit")).get("pass"),
        _mapping(payload.get("docs_drift_audit")).get("pass"),
    )
    return all(value is True for value in checks)


def _summarize_job(job: Mapping[str, Any]) -> dict[str, Any]:
    timings = _mapping(job.get("stage_timings"))
    counters = _mapping(job.get("instrumentation_counters"))
    top_sample = _top_sample_fingerprint(job)
    return {
        "job_name": job.get("job_name"),
        "pass": bool(job.get("pass")),
        "parity_pass": bool(_mapping(job.get("parity")).get("pass")),
        "service_wall_s": _float_or_none(timings.get("service_wall_clock_s")),
        "service_total_without_warmup_s": _float_or_none(
            timings.get("service_total_without_warmup")
        ),
        "exact_scoring_s": _float_or_none(timings.get("exact_scoring")),
        "tp_sl_exact_scoring_s": _float_or_none(timings.get("tp_sl_exact_scoring")),
        "tp_sl_cell_block_shape": counters.get("tp_sl_cell_block_shape"),
        "trade_cell_evals_per_sec": _float_or_none(
            counters.get("trade_cell_evals_per_sec")
        ),
        "tp_sl_cell_trade_cell_evals": counters.get("tp_sl_cell_trade_cell_evals"),
        "peak_rss_bytes": _peak_rss_bytes(job),
        "top_sample_hash": _stable_hash(top_sample),
        "top_sample": top_sample,
    }


def _shape_matrix_rows(
    *,
    current: Mapping[str, Any] | None,
    accepted: Mapping[str, Any] | None,
    shape_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    current_jobs = _jobs_by_name(current)
    accepted_jobs = _jobs_by_name(accepted)
    for shape_run in shape_runs:
        shape = shape_run.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        for job in _list(shape_run.get("jobs")):
            shape_job = _mapping(job)
            job_name = str(shape_job.get("job_name"))
            current_job = current_jobs.get(job_name)
            accepted_job = accepted_jobs.get(job_name)
            top_vs_current = _top_samples_match(shape_job, current_job)
            top_vs_accepted = _top_samples_match(shape_job, accepted_job)
            wall_vs_accepted = _relative_improvement(
                baseline=_float_from_mapping(accepted_job, "service_wall_s"),
                candidate=_float_from_mapping(shape_job, "service_wall_s"),
            )
            wall_vs_current = _relative_improvement(
                baseline=_float_from_mapping(current_job, "service_wall_s"),
                candidate=_float_from_mapping(shape_job, "service_wall_s"),
            )
            memory_regression = _relative_regression(
                baseline=_float_from_mapping(accepted_job, "peak_rss_bytes"),
                candidate=_float_from_mapping(shape_job, "peak_rss_bytes"),
            )
            rows.append(
                {
                    "shape": f"{shape[0]}x{shape[1]}",
                    "job_name": job_name,
                    "run_pass": bool(shape_run.get("pass")),
                    "job_pass": bool(shape_job.get("pass")),
                    "top_sample_matches_current_exact": top_vs_current,
                    "top_sample_matches_stage_09": top_vs_accepted,
                    "service_wall_s": shape_job.get("service_wall_s"),
                    "stage_09_service_wall_s": None
                    if accepted_job is None
                    else accepted_job.get("service_wall_s"),
                    "current_exact_service_wall_s": None
                    if current_job is None
                    else current_job.get("service_wall_s"),
                    "service_wall_improvement_vs_stage_09": wall_vs_accepted,
                    "service_wall_improvement_vs_current_exact": wall_vs_current,
                    "peak_rss_bytes": shape_job.get("peak_rss_bytes"),
                    "memory_peak_regression_vs_stage_09": memory_regression,
                    "memory_pass": memory_regression is not None
                    and memory_regression <= MEMORY_PEAK_REGRESSION_MAX,
                    "tp_sl_exact_scoring_s": shape_job.get("tp_sl_exact_scoring_s"),
                    "trade_cell_evals_per_sec": shape_job.get("trade_cell_evals_per_sec"),
                    "top_sample_hash": shape_job.get("top_sample_hash"),
                }
            )
    return rows


def _shape_evaluations(
    *,
    rows: Sequence[Mapping[str, Any]],
    shape_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evaluations = []
    for shape_run in shape_runs:
        shape_value = shape_run.get("shape")
        if not isinstance(shape_value, list) or len(shape_value) != 2:
            continue
        shape = f"{shape_value[0]}x{shape_value[1]}"
        shape_rows = [row for row in rows if row.get("shape") == shape]
        service_walls = [
            float(row["service_wall_s"])
            for row in shape_rows
            if isinstance(row.get("service_wall_s"), int | float)
        ]
        min_improvement = _min_float(
            row.get("service_wall_improvement_vs_stage_09") for row in shape_rows
        )
        max_memory_regression = _max_float(
            row.get("memory_peak_regression_vs_stage_09") for row in shape_rows
        )
        gate_pass = (
            bool(shape_rows)
            and bool(shape_run.get("pass"))
            and all(bool(row.get("job_pass")) for row in shape_rows)
            and all(bool(row.get("top_sample_matches_current_exact")) for row in shape_rows)
            and all(bool(row.get("top_sample_matches_stage_09")) for row in shape_rows)
            and min_improvement is not None
            and min_improvement >= SERVICE_WALL_IMPROVEMENT_MIN
            and max_memory_regression is not None
            and max_memory_regression <= MEMORY_PEAK_REGRESSION_MAX
        )
        evaluations.append(
            {
                "shape": shape,
                "accepted": gate_pass,
                "run_pass": bool(shape_run.get("pass")),
                "jobs": [row.get("job_name") for row in shape_rows],
                "total_service_wall_s": sum(service_walls) if service_walls else None,
                "min_service_wall_improvement_vs_stage_09": min_improvement,
                "max_memory_peak_regression_vs_stage_09": max_memory_regression,
            }
        )
    return evaluations


def _best_shape(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        item
        for item in evaluations
        if isinstance(item.get("total_service_wall_s"), int | float)
    ]
    if not eligible:
        return None
    return dict(min(eligible, key=lambda item: float(item["total_service_wall_s"])))


def render_stage_13_summary(report: Mapping[str, Any]) -> str:
    decision = _mapping(report.get("decision"))
    best_shape = _mapping(decision.get("best_shape"))
    lines = [
        "# Stage 13 TP/SL block autotune production gate",
        "",
        "## Короткий вывод",
        "",
        f"- Stage status: `{decision.get('status')}`.",
        f"- Reason: `{decision.get('reason')}`.",
        f"- Best shape: `{best_shape.get('shape')}`.",
        "- Production default change: `not performed by this gate report`.",
        "",
        "## Shape matrix",
        "",
        (
            "| Shape | Job | wall s | vs Stage 09 | vs current exact | "
            "peak RSS | memory vs Stage 09 | top parity | evals/s |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in _list(report.get("shape_matrix")):
        item = _mapping(row)
        top_ok = bool(item.get("top_sample_matches_current_exact")) and bool(
            item.get("top_sample_matches_stage_09")
        )
        lines.append(
            "| "
            f"`{item.get('shape')}` | "
            f"`{item.get('job_name')}` | "
            f"{_fmt_float(item.get('service_wall_s'))} | "
            f"{_fmt_pct(item.get('service_wall_improvement_vs_stage_09'))} | "
            f"{_fmt_pct(item.get('service_wall_improvement_vs_current_exact'))} | "
            f"{_fmt_int(item.get('peak_rss_bytes'))} | "
            f"{_fmt_pct(item.get('memory_peak_regression_vs_stage_09'))} | "
            f"{'pass' if top_ok else 'fail'} | "
            f"{_fmt_float(item.get('trade_cell_evals_per_sec'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            (
                "- Required: every TP/SL row keeps top sample identity/order, "
                "`best_tp`/`best_sl`, service wall improves >= 15% vs Stage 09 "
                "64x64, and peak RSS worsens <= 10%."
            ),
            f"- Controls pass: `{'yes' if decision.get('controls_pass') else 'no'}`.",
            f"- Missing payloads: `{decision.get('missing_payloads')}`.",
            f"- Crashed runs: `{decision.get('crashed_runs')}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _find_run(
    runs: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> Mapping[str, Any] | None:
    return next((run for run in runs if run.get("role") == role), None)


def _jobs_by_name(run: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if run is None:
        return {}
    return {str(job.get("job_name")): _mapping(job) for job in _list(run.get("jobs"))}


def _top_sample_fingerprint(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = _list(job.get("child_process_evidence"))
    diagnostics = {}
    for item in reversed(evidence):
        raw = _mapping(item).get("exact_diagnostics")
        if isinstance(raw, Mapping):
            diagnostics = dict(raw)
            break
    top_results = _list(diagnostics.get("top_results_sample"))
    out = []
    for item in top_results:
        result = _mapping(item)
        metrics = _mapping(result.get("metrics"))
        out.append(
            {
                "rank": result.get("rank"),
                "indicator_rows": result.get("indicator_rows"),
                "best_tp_idx": result.get("best_tp_idx"),
                "best_sl_idx": result.get("best_sl_idx"),
                "best_tp_pct": metrics.get("best_tp_pct"),
                "best_sl_pct": metrics.get("best_sl_pct"),
                "total_return_pct": metrics.get("total_return_pct"),
                "trade_count": metrics.get("trade_count"),
            }
        )
    return out


def _top_samples_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any] | None,
) -> bool:
    if right is None:
        return False
    left_items = _list(left.get("top_sample"))
    right_items = _list(right.get("top_sample"))
    if len(left_items) != len(right_items):
        return False
    for left_item, right_item in zip(left_items, right_items, strict=True):
        left_map = _mapping(left_item)
        right_map = _mapping(right_item)
        for key in ("rank", "indicator_rows", "best_tp_idx", "best_sl_idx"):
            if left_map.get(key) != right_map.get(key):
                return False
        for key in ("best_tp_pct", "best_sl_pct", "total_return_pct", "trade_count"):
            if not _float_close(left_map.get(key), right_map.get(key)):
                return False
    return True


def _peak_rss_bytes(job: Mapping[str, Any]) -> int | None:
    values = [
        int(item["peak_rss_bytes"])
        for item in _list(_mapping(job.get("memory")).get("items"))
        if isinstance(_mapping(item).get("peak_rss_bytes"), int)
    ]
    return max(values) if values else None


def _relative_improvement(
    *,
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (baseline - candidate) / baseline


def _relative_regression(
    *,
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (candidate - baseline) / baseline


def _parse_shape(raw: str) -> tuple[int, int]:
    normalized = raw.strip().lower().replace(",", "x").replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"invalid shape {raw!r}; expected TPxSL")
    try:
        tp_count = int(parts[0])
        sl_count = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {raw!r}; expected integers") from exc
    if tp_count <= 0 or sl_count <= 0:
        raise argparse.ArgumentTypeError(f"invalid shape {raw!r}; values must be positive")
    return tp_count, sl_count


def _unique_shapes(shapes: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out = []
    seen = set()
    for shape in shapes:
        if shape in seen:
            continue
        seen.add(shape)
        out.append(shape)
    return tuple(out)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return data


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _float_from_mapping(value: Mapping[str, Any] | None, key: str) -> float | None:
    if value is None:
        return None
    return _float_or_none(value.get(key))


def _float_close(left: Any, right: Any) -> bool:
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return left == right
    return abs(float(left) - float(right)) <= TOP_RESULT_FLOAT_TOLERANCE


def _min_float(values: Sequence[Any]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, int | float)]
    return min(floats) if floats else None


def _max_float(values: Sequence[Any]) -> float | None:
    floats = [float(value) for value in values if isinstance(value, int | float)]
    return max(floats) if floats else None


def _fmt_float(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):.6f}"


def _fmt_pct(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value) * 100.0:.3f}%"


def _fmt_int(value: Any) -> str:
    return "n/a" if not isinstance(value, int) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
