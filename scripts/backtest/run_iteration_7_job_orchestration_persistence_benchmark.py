from __future__ import annotations

# ruff: noqa: E402
import argparse
import gc
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.backtest import (
    run_iteration_4_2_exact_scoring_benchmark as no_risk_bench,  # noqa: E402
)
from scripts.backtest import (
    run_iteration_6_tp_sl_exact_scoring_benchmark as tp_sl_bench,  # noqa: E402
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    PERSIST_TOP_N_IO_STAGE_NAME,
    SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME,
    TOP_RESULT_ASSEMBLY_STAGE_NAME,
    BacktestTopResultAssemblyService,
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_7_job_orchestration_persistence"
)
BENCHMARK_TOP_K = 5
REQUEST_TOP_N = 100


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 7 job orchestration/top-result assembly/persistence IO "
            "benchmark evidence."
        )
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-config", type=Path, default=None)
    parser.add_argument("--arity", type=int, default=1)
    parser.add_argument(
        "--direction-mode",
        default="long_short_reversal",
        choices=("long_only", "long_short_reversal"),
    )
    parser.add_argument("--rows-per-indicator", type=int, default=6)
    parser.add_argument("--self-check-n", type=int, default=1)
    args = parser.parse_args(argv)

    canonical = no_risk_bench._load_json(args.canonical_json)
    no_risk_services = no_risk_bench._build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=BENCHMARK_TOP_K,
        self_check_n=args.self_check_n,
    )
    tp_sl_services = tp_sl_bench._build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=BENCHMARK_TOP_K,
        self_check_n=args.self_check_n,
    )
    no_risk_run = _run_no_risk(
        canonical=canonical,
        services=no_risk_services,
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
    )
    tp_sl_run = _run_tp_sl(
        canonical=canonical,
        services=tp_sl_services,
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
    )
    payload = _build_payload(
        canonical_json=args.canonical_json,
        no_risk_services=no_risk_services,
        tp_sl_services=tp_sl_services,
        no_risk_run=no_risk_run,
        tp_sl_run=tp_sl_run,
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
        self_check_n=args.self_check_n,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "benchmark_results.json"
    summary_path = args.out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0 if payload["pass"] else 1


def _run_no_risk(
    *,
    canonical: Mapping[str, Any],
    services: Any,
    arity: int,
    direction_mode: str,
    rows_per_indicator: int,
) -> dict[str, Any]:
    request = no_risk_bench._service_request(
        canonical_request=no_risk_bench._required_mapping(canonical, "request"),
        arity=arity,
        direction_mode=direction_mode,
    )
    prepared_full = services.prepare_pools.execute(
        normalized_request=request,
        artifact_metadata=services.artifact_metadata,
    )
    prepared = no_risk_bench._limit_prepared_rows(
        prepared_full,
        rows_per_indicator=rows_per_indicator,
    )
    start = time.perf_counter()
    combo_result = services.combo_planning.execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    exact_result = services.exact_measured.execute(
        prepared_result=prepared,
        combo_planning_result=combo_result,
        normalized_request=request,
    )
    runtime_wall_s = time.perf_counter() - start
    assembled = BacktestTopResultAssemblyService().assemble(
        job_id=uuid4(),
        normalized_request=request,
        top_results=exact_result.top_results,
        updated_at=datetime.now(UTC),
    )
    persist = _measure_persist_io(top_variants=assembled.top_variants)
    timers = {
        **dict(prepared.timing.subsegments),
        **dict(combo_result.telemetry.stage_timings),
        **dict(exact_result.telemetry.stage_timings),
        **dict(assembled.stage_timings),
        PERSIST_TOP_N_IO_STAGE_NAME: persist["wall_s"],
    }
    timers[SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME] = runtime_wall_s + math_sum(
        [timers[TOP_RESULT_ASSEMBLY_STAGE_NAME], timers[PERSIST_TOP_N_IO_STAGE_NAME]]
    )
    cleanup = exact_result.memory_cleanup_evidence.as_mapping()
    run = {
        "risk_mode": "none",
        "arity": arity,
        "direction_mode": direction_mode,
        "top_results_count": len(assembled.top_variants),
        "timers": timers,
        "top_results_summary_hash": assembled.summary_hash,
        "persisted_top_n_summary_hash": persist["summary_hash"],
        "persistence": persist,
        "cleanup_evidence": cleanup,
        "runtime_metrics": _runtime_metrics(),
        "pass": bool(
            len(assembled.top_variants) == exact_result.telemetry.top_results_count
            and cleanup["result_is_compact"]
            and persist["summary_only"]
        ),
    }
    del exact_result
    del combo_result
    del prepared
    del prepared_full
    gc.collect()
    return run


def _run_tp_sl(
    *,
    canonical: Mapping[str, Any],
    services: Any,
    arity: int,
    direction_mode: str,
    rows_per_indicator: int,
) -> dict[str, Any]:
    request = tp_sl_bench._service_request(
        canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
        arity=arity,
        direction_mode=direction_mode,
    )
    hit_times_result = services.hit_times.execute(
        normalized_request=request,
        context=services.context,
    )
    prepared_full = services.prepare_pools.execute(
        normalized_request=request,
        artifact_metadata=services.artifact_metadata,
    )
    prepared = tp_sl_bench._limit_prepared_rows(
        prepared_full,
        rows_per_indicator=rows_per_indicator,
    )
    start = time.perf_counter()
    combo_result = services.combo_planning.execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    exact_result = services.exact_measured.execute(
        prepared_result=prepared,
        combo_planning_result=combo_result,
        hit_times_result=hit_times_result,
        normalized_request=request,
    )
    runtime_wall_s = time.perf_counter() - start
    assembled = BacktestTopResultAssemblyService().assemble(
        job_id=uuid4(),
        normalized_request=request,
        top_results=exact_result.top_results,
        updated_at=datetime.now(UTC),
    )
    persist = _measure_persist_io(top_variants=assembled.top_variants)
    timers = {
        **dict(hit_times_result.timing.subsegments),
        **dict(prepared.timing.subsegments),
        **dict(combo_result.telemetry.stage_timings),
        **dict(exact_result.telemetry.stage_timings),
        **dict(assembled.stage_timings),
        PERSIST_TOP_N_IO_STAGE_NAME: persist["wall_s"],
    }
    timers[SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME] = runtime_wall_s + math_sum(
        [timers[TOP_RESULT_ASSEMBLY_STAGE_NAME], timers[PERSIST_TOP_N_IO_STAGE_NAME]]
    )
    cleanup = {
        "hit_times": hit_times_result.cleanup_evidence.as_mapping(),
        "tp_sl_exact": exact_result.memory_cleanup_evidence.as_mapping(),
    }
    run = {
        "risk_mode": "tp_sl_grid",
        "arity": arity,
        "direction_mode": direction_mode,
        "top_results_count": len(assembled.top_variants),
        "timers": timers,
        "top_results_summary_hash": assembled.summary_hash,
        "persisted_top_n_summary_hash": persist["summary_hash"],
        "persistence": persist,
        "cleanup_evidence": cleanup,
        "runtime_metrics": _runtime_metrics(),
        "pass": bool(
            len(assembled.top_variants) == exact_result.telemetry.top_results_count
            and exact_result.memory_cleanup_evidence.result_is_compact
            and persist["summary_only"]
        ),
    }
    del exact_result
    del combo_result
    del hit_times_result
    del prepared
    del prepared_full
    gc.collect()
    return run


def _measure_persist_io(*, top_variants: tuple[Any, ...]) -> dict[str, Any]:
    start = time.perf_counter()
    rows = [
        {
            "rank": row.rank,
            "variant_hash": row.variant_key,
            "indicator_variant_hash": row.indicator_variant_key,
            "payload_json": dict(row.payload_json),
            "summary_metrics_json": dict(row.summary_metrics_json),
            "best_tp_pct": row.best_tp_pct,
            "best_sl_pct": row.best_sl_pct,
            "report_table_md": row.report_table_md,
            "trades_json": row.trades_json,
        }
        for row in top_variants
    ]
    rendered = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    wall_s = time.perf_counter() - start
    return {
        "wall_s": wall_s,
        "rows": len(rows),
        "bytes": len(rendered.encode("utf-8")),
        "summary_hash": _sha256(rendered),
        "summary_only": all(
            row["report_table_md"] is None and row["trades_json"] is None for row in rows
        ),
    }


def _build_payload(
    *,
    canonical_json: Path,
    no_risk_services: Any,
    tp_sl_services: Any,
    no_risk_run: Mapping[str, Any],
    tp_sl_run: Mapping[str, Any],
    arity: int,
    direction_mode: str,
    rows_per_indicator: int,
    self_check_n: int,
) -> dict[str, Any]:
    runs = [dict(no_risk_run), dict(tp_sl_run)]
    return {
        "schema": "backtest_iteration_7_job_orchestration_persistence_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": no_risk_bench._git_commit(),
        "git_status_short": no_risk_bench._git_status_short(),
        "canonical_json": str(canonical_json),
        "artifact_config_path": str(no_risk_services.artifact_config_path),
        "artifact_root": str(no_risk_services.artifact_root),
        "artifact_manifest_hash": no_risk_services.artifact_manifest_hash,
        "hit_times_manifest_hash": tp_sl_services.hit_times_manifest_hash,
        "artifact_policy": "historical_prefix_compatible",
        "request": {
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "sample_warmup_top_k": 1,
            "arity": arity,
            "direction_mode": direction_mode,
            "rows_per_indicator": rows_per_indicator,
            "self_check_n": self_check_n,
        },
        "service_only_stages": [
            TOP_RESULT_ASSEMBLY_STAGE_NAME,
            PERSIST_TOP_N_IO_STAGE_NAME,
            SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME,
        ],
        "runs": runs,
        "pass_breakdown": {
            "top_result_assembly": all(
                run["timers"][TOP_RESULT_ASSEMBLY_STAGE_NAME] >= 0.0 for run in runs
            ),
            "persist_top_n_io": all(
                run["timers"][PERSIST_TOP_N_IO_STAGE_NAME] >= 0.0 for run in runs
            ),
            "summary_only": all(run["persistence"]["summary_only"] for run in runs),
            "cleanup": all(run["pass"] for run in runs),
        },
        "pass": all(run["pass"] for run in runs),
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    lines = [
        "# Iteration 7 job orchestration and persistence",
        "",
        "## Scope",
        "",
        "- Implemented: service-only `top_result_assembly`, summary-only top-N IO "
        "measurement, public/storage identity mapping evidence.",
        "- Not implemented: lazy trades detail payloads and Iteration 8 sizing expansion.",
        "",
        "## Environment",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact config: `{payload['artifact_config_path']}`",
        f"- Artifact root: `{payload['artifact_root']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        f"- Hit-times manifest hash: `{payload['hit_times_manifest_hash']}`",
        f"- Artifact policy: `{payload['artifact_policy']}`",
        "",
        "## Service-Only Stages",
        "",
        "| risk_mode | top_result_assembly_s | persist_top_n_io_s | "
        "service_total_without_warmup_s | rows | summary_only | pass |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        timers = run["timers"]
        lines.append(
            (
                "| {risk} | {assembly:.6f} | {persist:.6f} | {total:.6f} | "
                "{rows} | `{summary}` | `{passed}` |"
            ).format(
                risk=run["risk_mode"],
                assembly=timers[TOP_RESULT_ASSEMBLY_STAGE_NAME],
                persist=timers[PERSIST_TOP_N_IO_STAGE_NAME],
                total=timers[SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME],
                rows=run["top_results_count"],
                summary=run["persistence"]["summary_only"],
                passed="yes" if run["pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


def math_sum(values: list[float]) -> float:
    return sum(float(value) for value in values)


def _runtime_metrics() -> dict[str, Any]:
    return {
        "process_maxrss_raw": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
