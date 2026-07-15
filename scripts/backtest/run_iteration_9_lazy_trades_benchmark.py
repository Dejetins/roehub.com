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

from scripts.backtest import (  # noqa: E402
    run_iteration_4_2_exact_scoring_benchmark as no_risk_bench,
)
from scripts.backtest import (  # noqa: E402
    run_iteration_6_tp_sl_exact_scoring_benchmark as tp_sl_bench,
)
from trading.contexts.backtest.adapters.outbound import (  # noqa: E402
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    LocalFileBacktestLazyTradesCache,
)
from trading.contexts.backtest.application.ports import canonical_json_sha256  # noqa: E402
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
    LAZY_TRADES_CACHE_HIT_STAGE_NAME,
    LAZY_TRADES_COMPUTE_STAGE_NAME,
    BacktestLazyTradesDetailService,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.domain.entities import (  # noqa: E402
    BacktestJob,
    BacktestJobArtifactPin,
)
from trading.shared_kernel.primitives import OrganizationId, UserId  # noqa: E402

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_9_lazy_trades_detail"
)
BENCHMARK_TOP_K = 5
REQUEST_TOP_N = 100
SUMMARY_TOLERANCE = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Iteration 9 lazy trades detail benchmark evidence."
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-config", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
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
    cache_root = args.cache_root or DEFAULT_LAZY_TRADES_CACHE_ROOT
    no_risk_run = _run_case(
        canonical=canonical,
        services=no_risk_services,
        hit_times_service=tp_sl_services.hit_times,
        risk_mode="none",
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
        cache_root=cache_root,
    )
    tp_sl_run = _run_case(
        canonical=canonical,
        services=tp_sl_services,
        hit_times_service=tp_sl_services.hit_times,
        risk_mode="tp_sl_grid",
        arity=args.arity,
        direction_mode=args.direction_mode,
        rows_per_indicator=args.rows_per_indicator,
        cache_root=cache_root,
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
        cache_root=cache_root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "benchmark_results.json"
    summary_path = args.out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0 if payload["pass"] else 1


def _run_case(
    *,
    canonical: Mapping[str, Any],
    services: Any,
    hit_times_service: Any,
    risk_mode: str,
    arity: int,
    direction_mode: str,
    rows_per_indicator: int,
    cache_root: Path,
) -> dict[str, Any]:
    if risk_mode == "tp_sl_grid":
        request = tp_sl_bench._service_request(
            canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
            arity=arity,
            direction_mode=direction_mode,
        )
        prepared_full = services.prepare_pools.execute(
            normalized_request=request,
            artifact_metadata=services.artifact_metadata,
        )
        prepared = tp_sl_bench._limit_prepared_rows(
            prepared_full,
            rows_per_indicator=rows_per_indicator,
        )
        hit_times_result = services.hit_times.execute(
            normalized_request=request,
            context=services.context,
        )
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
    else:
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
        combo_result = services.combo_planning.execute(
            prepared_result=prepared,
            normalized_request=request,
        )
        exact_result = services.exact_measured.execute(
            prepared_result=prepared,
            combo_planning_result=combo_result,
            normalized_request=request,
        )
    job_id = uuid4()
    updated_at = datetime.now(UTC)
    assembled = BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=request,
        top_results=exact_result.top_results[:1],
        updated_at=updated_at,
    )
    row = assembled.top_variants[0]
    job = _job(
        job_id=job_id,
        request=request,
        artifact_metadata=services.artifact_metadata,
        engine_hash=_engine_hash(request=request, services=services),
        created_at=updated_at,
    )
    service = BacktestLazyTradesDetailService(
        prepare_pools=services.prepare_pools,
        tp_sl_hit_times=hit_times_service,
        cache=LocalFileBacktestLazyTradesCache(root=cache_root),
    )
    public_variant_key = str(row.payload_json["public_variant_key"])

    compute_runtime = _measure(
        lambda: service.execute(
            job=job,
            row=row,
            public_variant_key=public_variant_key,
        )
    )
    cache_hit_runtime = _measure(
        lambda: service.execute(
            job=job,
            row=row,
            public_variant_key=public_variant_key,
        )
    )
    compute_result = compute_runtime["value"].as_mapping()
    cache_hit_result = cache_hit_runtime["value"].as_mapping()
    parity = _summary_parity(row_summary=dict(row.summary_metrics_json), detail=compute_result)
    out = {
        "risk_mode": risk_mode,
        "arity": arity,
        "direction_mode": direction_mode,
        "sizing_mode": _sizing_mode(request=request),
        "close_on_end": bool(_mapping(request.get("execution")).get("close_on_end", True)),
        "request_hash": job.request_hash,
        "engine_params_hash": job.engine_params_hash,
        "artifact_manifest_hash": services.artifact_manifest_hash,
        "hit_times_manifest_hash": getattr(services, "hit_times_manifest_hash", None),
        "artifact_policy": "historical_prefix_compatible",
        "variant_key": public_variant_key,
        "variant_hash": row.variant_key,
        "cache_root": str(cache_root),
        "cache_key": compute_result["cache"]["cache_key"],
        "cache_ttl_seconds": DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
        "lazy_trades_compute": compute_runtime["wall_s"],
        "lazy_trades_cache_hit": cache_hit_runtime["wall_s"],
        "timing": {
            LAZY_TRADES_COMPUTE_STAGE_NAME: compute_runtime["wall_s"],
            LAZY_TRADES_CACHE_HIT_STAGE_NAME: cache_hit_runtime["wall_s"],
        },
        "cache": {
            "compute_status": compute_result["cache"]["status"],
            "hit_status": cache_hit_result["cache"]["status"],
        },
        "trade_count": len(compute_result["trades"]),
        "summary_parity": parity,
        "runtime_metrics": {
            "compute": compute_runtime["runtime_metrics"],
            "cache_hit": cache_hit_runtime["runtime_metrics"],
        },
        "pass": bool(
            compute_result["cache"]["status"] in {"miss", "expired", "read_failed"}
            and cache_hit_result["cache"]["status"] == "hit"
            and parity["pass"]
            and len(compute_result["trades"]) > 0
            and compute_runtime["wall_s"] <= 30.0
            and cache_hit_runtime["wall_s"] <= 1.0
        ),
    }
    del exact_result
    del combo_result
    del prepared
    del prepared_full
    gc.collect()
    return out


def _job(
    *,
    job_id: Any,
    request: Mapping[str, Any],
    artifact_metadata: Any,
    engine_hash: str,
    created_at: datetime,
) -> BacktestJob:
    request_json = dict(request)
    request_json["artifact_metadata"] = artifact_metadata.as_mapping()
    return BacktestJob.create_queued(
        job_id=job_id,
        organization_id=OrganizationId.from_string(
            "00000000-0000-0000-0000-000000000001"
        ),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000909"),
        mode="template",
        created_at=created_at,
        request_json=request_json,
        request_hash=canonical_json_sha256(request_json),
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash=engine_hash,
        backtest_runtime_config_hash=engine_hash,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot=artifact_metadata.artifact_slot,
            artifact_slot_generation=artifact_metadata.artifact_slot_generation,
            artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
            artifact_asof_date=artifact_metadata.artifact_asof_date,
        ),
        execution_mode="sync_inline",
        market_id=1,
        symbol=str(_mapping(request.get("coordinates")).get("symbol", "BTCUSDT")),
        timeframe=str(request.get("timeframe", "15m")),
        requested_top_n=int(request.get("top_n", REQUEST_TOP_N)),
        ranking_primary_metric=str(_mapping(request.get("ranking")).get("primary_metric")),
    )


def _summary_parity(
    *,
    row_summary: Mapping[str, Any],
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    detail_summary = _mapping(detail.get("summary_metrics"))
    checks: dict[str, Any] = {}
    for key in ("total_return_pct", "trade_count", "best_tp_pct", "best_sl_pct"):
        if key not in row_summary or key not in detail_summary:
            continue
        expected = float(row_summary[key])
        actual = float(detail_summary[key])
        checks[key] = {
            "expected": expected,
            "actual": actual,
            "abs_diff": abs(expected - actual),
            "pass": abs(expected - actual) <= SUMMARY_TOLERANCE,
        }
    return {
        "checks": checks,
        "pass": all(bool(item["pass"]) for item in checks.values()),
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
    cache_root: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numba": _numba_version(),
        "canonical_json": str(canonical_json),
        "artifact_config_path": str(no_risk_services.artifact_config_path),
        "artifact_root": str(no_risk_services.artifact_root),
        "artifact_manifest_hash": no_risk_services.artifact_manifest_hash,
        "hit_times_manifest_hash": tp_sl_services.hit_times_manifest_hash,
        "artifact_policy": "historical_prefix_compatible",
        "request": {
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "arity": arity,
            "direction_mode": direction_mode,
            "rows_per_indicator": rows_per_indicator,
        },
        "cache_root": str(cache_root),
        "cache_ttl_seconds": DEFAULT_LAZY_TRADES_CACHE_TTL_SECONDS,
        "service_only_stages": [
            LAZY_TRADES_COMPUTE_STAGE_NAME,
            LAZY_TRADES_CACHE_HIT_STAGE_NAME,
        ],
        "total_without_warmup_excludes_lazy_trades": True,
        "runs": [dict(no_risk_run), dict(tp_sl_run)],
        "pass": bool(no_risk_run["pass"] and tp_sl_run["pass"]),
    }


def _measure(fn: Any) -> dict[str, Any]:
    gc.collect()
    before = _runtime_metrics()
    start = time.perf_counter()
    value = fn()
    wall_s = time.perf_counter() - start
    after = _runtime_metrics()
    return {
        "value": value,
        "wall_s": wall_s,
        "runtime_metrics": {
            "rss_before": before["maxrss_raw"],
            "rss_after": after["maxrss_raw"],
            "process_cpu_time_s": after["process_cpu_time_s"] - before["process_cpu_time_s"],
        },
    }


def _runtime_metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "maxrss_raw": usage.ru_maxrss,
        "process_cpu_time_s": time.process_time(),
    }


def _engine_hash(*, request: Mapping[str, Any], services: Any) -> str:
    return canonical_json_sha256(
        {
            "execution": _mapping(request.get("execution")),
            "risk": _mapping(request.get("risk")),
            "artifact_config_path": str(services.artifact_config_path),
        }
    )


def _sizing_mode(*, request: Mapping[str, Any]) -> str:
    execution = _mapping(request.get("execution"))
    sizing = _mapping(execution.get("sizing"))
    return str(sizing.get("mode", "all_in"))


def _render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    lines = [
        "# Iteration 9 lazy trades detail",
        "",
        "## Version",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        f"- Hit-times manifest hash: `{payload['hit_times_manifest_hash']}`",
        f"- Artifact policy: `{payload['artifact_policy']}`",
        f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
        "",
        "## Service-only Stages",
        "",
        "| risk | variant_key | variant_hash | trades | lazy_trades_compute | "
        "lazy_trades_cache_hit | parity | pass |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            "| "
            f"`{run['risk_mode']}` | "
            f"`{run['variant_key']}` | "
            f"`{run['variant_hash']}` | "
            f"{run['trade_count']} | "
            f"{run['lazy_trades_compute']:.6f} | "
            f"{run['lazy_trades_cache_hit']:.6f} | "
            f"`{'yes' if run['summary_parity']['pass'] else 'no'}` | "
            f"`{'yes' if run['pass'] else 'no'}` |"
        )
    lines.extend(
        [
            "",
            "## Cache",
            "",
            f"- Cache root: `{payload['cache_root']}`",
            f"- Cache TTL seconds: `{payload['cache_ttl_seconds']}`",
            "- `lazy_trades_compute` and `lazy_trades_cache_hit` are service-only "
            "and are not included in `total_without_warmup`.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status_short() -> str:
    import subprocess

    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def _numba_version() -> str | None:
    try:
        import numba
    except Exception:  # noqa: BLE001
        return None
    return str(getattr(numba, "__version__", "unknown"))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
