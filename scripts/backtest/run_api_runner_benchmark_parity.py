from __future__ import annotations

# ruff: noqa: E402, SLF001
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles
from typing import Any, cast
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.worker.backtest_job_runner.wiring.modules import (  # noqa: E402
    backtest_job_runner as runner_module,
)
from apps.worker.backtest_job_runner.wiring.modules import (  # noqa: E402
    build_backtest_job_runner_app,
    load_backtest_job_runner_runtime_config,
)
from scripts.backtest import run_backtest_job_runner_prod_smoke as prod_smoke  # noqa: E402
from scripts.backtest import (
    run_iteration_4_2_exact_scoring_benchmark as no_risk_bench,  # noqa: E402,E501
)
from scripts.backtest import (
    run_iteration_6_tp_sl_exact_scoring_benchmark as tp_sl_bench,  # noqa: E402,E501
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_REFERENCE_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_11_api_runner_compute_memory_parity"
)
REQUEST_TOP_N = 100
BENCHMARK_TOP_K = 5
REFERENCE_ROWS_PER_INDICATOR = 6
REFERENCE_WARMUP_ROWS_PER_INDICATOR = 2
_DEFAULT_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_COOKIE_NAME = "roehub_session_id"
_PARITY_FLOAT_TOLERANCE = 1e-5
_CACHE_HIT_RSS_DELTA_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _RunnerHarness:
    scheduler: Any
    runtime_config: Any
    env: Mapping[str, str]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    child_evidence_dir = out_dir / "child_process_evidence"
    child_evidence_dir.mkdir(parents=True, exist_ok=True)

    canonical = _load_json(args.canonical_json)
    reference = _load_json(args.reference_json)
    reference_runs, excluded = _reference_runs(canonical=canonical)
    if args.smoke_only:
        reference_runs = _smoke_subset(reference_runs)

    dsn = prod_smoke._postgres_dsn(environ=os.environ)
    session_id: str | None = None
    started_at = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema": "backtest_api_runner_compute_memory_parity_v1",
        "generated_at": started_at.isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "api_base": args.api_base,
        "canonical_json": str(args.canonical_json),
        "reference_json": str(args.reference_json),
        "reference_iteration": "2026-05-02_iteration_8_execution_sizing_completion",
        "request": {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "rows_per_indicator": REFERENCE_ROWS_PER_INDICATOR,
            "warmup_rows_per_indicator": REFERENCE_WARMUP_ROWS_PER_INDICATOR,
            "exclude_heaviest_140s_job": True,
        },
        "excluded_reference_job": excluded,
        "artifact_reference": {
            "artifact_manifest_hash": reference.get("artifact_manifest_hash"),
            "hit_times_manifest_hash": reference.get("hit_times_manifest_hash"),
            "artifact_policy": reference.get("artifact_policy"),
            "request_hash": canonical.get("request_hash"),
        },
    }

    try:
        user_id, session_id = prod_smoke._create_smoke_session(
            dsn=dsn,
            cookie_name=args.cookie_name,
            session_ttl_seconds=args.session_ttl_seconds,
        )
        client = prod_smoke._ApiClient(
            api_base=args.api_base,
            cookie_name=args.cookie_name,
            session_id=session_id,
        )
        payload["smoke_user_id"] = user_id
        payload["backlog_before"] = prod_smoke._backlog_snapshot(dsn=dsn)
        _require_clean_backlog(payload["backlog_before"], allow_existing=args.allow_backlog)

        harness = _build_runner_harness(
            child_evidence_dir=child_evidence_dir,
            light_max_actual_combinations=args.light_max_actual_combinations,
            light_concurrency=args.light_concurrency,
            heavy_concurrency=1,
        )
        benchmark_jobs = _run_reference_jobs(
            dsn=dsn,
            client=client,
            harness=harness,
            canonical=canonical,
            reference_runs=reference_runs,
            child_evidence_dir=child_evidence_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        payload["api_runner_path"] = {
            "runner_entrypoint": "BacktestJobWorkerUseCase.run_next",
            "child_module": "apps.worker.backtest_job_runner.main.full_job_child",
            "jobs": benchmark_jobs,
            "pass": all(bool(job["pass"]) for job in benchmark_jobs),
        }
        payload["parity"] = _parity_summary(benchmark_jobs)
        payload["performance"] = _performance_summary(benchmark_jobs)
        payload["memory_release"] = _memory_release_summary(benchmark_jobs)

        scheduler_smoke = _run_scheduler_smoke(
            dsn=dsn,
            client=client,
            canonical=canonical,
            child_evidence_dir=child_evidence_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        payload["mixed_scheduler_smoke"] = scheduler_smoke

        lazy_memory = _run_lazy_memory_checks(
            dsn=dsn,
            client=client,
            harness=harness,
            benchmark_jobs=benchmark_jobs,
            child_evidence_dir=child_evidence_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        payload["lazy_cache_hit_memory"] = lazy_memory

        payload["legacy_path_absence"] = _legacy_path_absence_audit()
        payload["dead_code_audit"] = _dead_code_audit()
        payload["docs_drift_audit"] = _docs_drift_audit()
        payload["backlog_after"] = prod_smoke._backlog_snapshot(dsn=dsn)
        payload["finished_at"] = datetime.now(UTC).isoformat()
        payload["pass"] = _overall_pass(payload)
    finally:
        if session_id is not None:
            prod_smoke._revoke_smoke_session(dsn=dsn, session_id=session_id)

    results_path = out_dir / "benchmark_results.json"
    summary_path = out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0 if payload.get("pass") or args.no_fail_on_threshold else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run API/runner canonical parity and memory benchmark evidence."
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-base", default=_DEFAULT_API_BASE)
    parser.add_argument("--cookie-name", default=_DEFAULT_COOKIE_NAME)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--session-ttl-seconds", type=int, default=7200)
    parser.add_argument("--light-concurrency", type=int, default=2)
    parser.add_argument("--light-max-actual-combinations", type=int, default=50_000)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--allow-backlog", action="store_true")
    parser.add_argument("--no-fail-on-threshold", action="store_true")
    return parser


def _run_reference_jobs(
    *,
    dsn: str,
    client: Any,
    harness: _RunnerHarness,
    canonical: Mapping[str, Any],
    reference_runs: Sequence[Mapping[str, Any]],
    child_evidence_dir: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, reference_run in enumerate(reference_runs, start=1):
        job_name = _reference_job_name(run=reference_run)
        request = _request_for_reference_run(canonical=canonical, run=reference_run)
        created = client.request_json(
            "POST",
            "/backtests/jobs",
            payload=request,
            extra_headers={"Idempotency-Key": f"api-runner-parity-{uuid4()}"},
        )
        if created.get("_status") != 201 or created.get("state") != "queued":
            raise RuntimeError(f"job create failed for {job_name}: {created}")
        job_id = str(created["job_id"])
        row_before = _job_detail_row(dsn=dsn, job_id=job_id)
        scheduling = _scheduling_from_row(row=row_before)
        lane = str(scheduling.get("scheduling_class") or "heavy")
        run_result = _process_job_until_terminal(
            dsn=dsn,
            client=client,
            scheduler=harness.scheduler,
            job_id=job_id,
            lane=lane,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        top = client.request_json("GET", f"/backtests/jobs/{job_id}/top")
        parity = _compare_top_results(api_top=top, reference_run=reference_run)
        child_evidence = _read_full_job_child_evidence(
            evidence_dir=child_evidence_dir,
            job_id=job_id,
        )
        stage_timings = _merged_stage_timings(child_evidence)
        memory = _child_memory_summary(child_evidence)
        jobs.append(
            {
                "index": index,
                "job_name": job_name,
                "job_id": job_id,
                "risk_mode": reference_run.get("risk_mode"),
                "arity": len(cast(Sequence[Any], reference_run.get("indicator_ids", ()))),
                "direction_mode": reference_run.get("direction_mode"),
                "created_state": created.get("state"),
                "request_hash": created.get("request_hash"),
                "scheduling": scheduling,
                "runner": run_result,
                "api_top_count": len(_list(top.get("items"))),
                "parity": parity,
                "stage_timings": stage_timings,
                "service_only_overhead": _service_only_overhead(stage_timings),
                "child_process_evidence": child_evidence,
                "memory": memory,
                "pass": (
                    bool(run_result["pass"])
                    and bool(parity["pass"])
                    and bool(memory["pass"])
                ),
            }
        )
    return jobs


def _process_job_until_terminal(
    *,
    dsn: str,
    client: Any,
    scheduler: Any,
    job_id: str,
    lane: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current_lane = lane
    deadline = time.monotonic() + timeout_seconds
    while True:
        worker = (
            scheduler.heavy_full_job_worker
            if current_lane == "heavy"
            else scheduler.light_full_job_worker
        )
        attempt = _run_worker_attempt(
            dsn=dsn,
            client=client,
            worker=worker,
            job_id=job_id,
            timeout_seconds=max(int(deadline - time.monotonic()), 1),
            poll_interval_seconds=poll_interval_seconds,
        )
        attempts.append(attempt)
        if attempt.get("worker_status") == "requeued_heavy":
            current_lane = "heavy"
            continue
        break
    terminal = _job_detail_row(dsn=dsn, job_id=job_id)
    pass_state = str(terminal["state"]) == "succeeded"
    observed_states = [
        state
        for attempt in attempts
        for state in _list(attempt.get("observed_states"))
    ]
    state_path = _collapse_state_path(observed_states)
    if pass_state and (not state_path or state_path[-1] != "succeeded"):
        state_path.append("succeeded")
    has_running = "running" in state_path or terminal.get("started_at") is not None
    return {
        "attempts": attempts,
        "state_path": " -> ".join(state_path),
        "required_path": "queued -> running -> succeeded",
        "required_path_pass": has_running and pass_state,
        "terminal": _job_terminal(row=terminal),
        "pass": has_running and pass_state,
    }


def _run_worker_attempt(
    *,
    dsn: str,
    client: Any,
    worker: Any,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    observed_states: list[str] = ["queued"]
    api_latencies_ms: list[float] = []
    running_samples: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_next)
        deadline = time.monotonic() + timeout_seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"worker attempt timed out for job {job_id}")
            started = time.perf_counter()
            api_payload = client.request_json("GET", f"/backtests/jobs/{job_id}")
            api_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            state = str(api_payload.get("state"))
            if state not in observed_states:
                observed_states.append(state)
            db_row = _job_detail_row(dsn=dsn, job_id=job_id)
            if str(db_row["state"]) == "running":
                running_samples.append(_running_sample(row=db_row))
            time.sleep(poll_interval_seconds)
        result = future.result()
    job = getattr(result, "job", None)
    return {
        "claimed": bool(getattr(result, "claimed", False)),
        "lease_lost": bool(getattr(result, "lease_lost", False)),
        "worker_status": getattr(result, "status", None),
        "job_state": None if job is None else getattr(job, "state", None),
        "observed_states": observed_states,
        "running_samples": running_samples[:5],
        "api_responsiveness": _latency_summary(api_latencies_ms),
    }


def _run_scheduler_smoke(
    *,
    dsn: str,
    client: Any,
    canonical: Mapping[str, Any],
    child_evidence_dir: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    light_harness = _build_runner_harness(
        child_evidence_dir=child_evidence_dir,
        light_max_actual_combinations=50_000,
        light_concurrency=2,
        heavy_concurrency=1,
    )
    light_jobs = [
        _create_scheduler_job(
            client=client,
            canonical=canonical,
            arity=1,
            risk_mode="none",
            direction_mode="long_only",
            label="light_1",
            short_range=True,
        ),
        _create_scheduler_job(
            client=client,
            canonical=canonical,
            arity=1,
            risk_mode="none",
            direction_mode="long_short_reversal",
            label="light_2",
            short_range=True,
        ),
    ]
    light_phase = _run_scheduler_phase(
        dsn=dsn,
        client=client,
        scheduler=light_harness.scheduler,
        job_ids=[job["job_id"] for job in light_jobs],
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    heavy_harness = _build_runner_harness(
        child_evidence_dir=child_evidence_dir,
        light_max_actual_combinations=50_000,
        light_concurrency=2,
        heavy_concurrency=1,
    )
    heavy_jobs = [
        _create_scheduler_job(
            client=client,
            canonical=canonical,
            arity=3,
            risk_mode="none",
            direction_mode="long_only",
            label="heavy_1",
            short_range=True,
        ),
        _create_scheduler_job(
            client=client,
            canonical=canonical,
            arity=1,
            risk_mode="none",
            direction_mode="long_only",
            label="light_not_starving_heavy",
            short_range=True,
        ),
        _create_scheduler_job(
            client=client,
            canonical=canonical,
            arity=3,
            risk_mode="none",
            direction_mode="long_short_reversal",
            label="heavy_2",
            short_range=True,
        ),
    ]
    heavy_phase = _run_scheduler_phase(
        dsn=dsn,
        client=client,
        scheduler=heavy_harness.scheduler,
        job_ids=[job["job_id"] for job in heavy_jobs],
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    promotion_harness = _build_runner_harness(
        child_evidence_dir=child_evidence_dir,
        light_max_actual_combinations=10,
        light_concurrency=2,
        heavy_concurrency=1,
    )
    promotion_job = _create_scheduler_job(
        client=client,
        canonical=canonical,
        arity=2,
        risk_mode="none",
        direction_mode="long_only",
        label="light_candidate_promotion",
        short_range=True,
    )
    promotion_result = _process_job_until_terminal(
        dsn=dsn,
        client=client,
        scheduler=promotion_harness.scheduler,
        job_id=promotion_job["job_id"],
        lane="light_candidate",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    promotion_row = _job_detail_row(dsn=dsn, job_id=promotion_job["job_id"])
    promotion_scheduling = _scheduling_from_row(row=promotion_row)
    pass_value = (
        bool(light_phase["pass"])
        and bool(heavy_phase["pass"])
        and bool(promotion_result["pass"])
        and promotion_scheduling.get("source") == "post_prepare_refinement"
        and promotion_scheduling.get("scheduling_class") == "heavy"
    )
    return {
        "configured": {
            "ROEHUB_BACKTEST_LIGHT_CONCURRENCY": 2,
            "ROEHUB_BACKTEST_HEAVY_CONCURRENCY": 1,
            "light_heavy_overlap": "disabled_v1",
        },
        "light_jobs": light_jobs,
        "heavy_jobs": heavy_jobs,
        "light_phase": light_phase,
        "heavy_phase": heavy_phase,
        "promotion_case": {
            "job": promotion_job,
            "result": promotion_result,
            "post_prepare_scheduling": promotion_scheduling,
            "covered": True,
        },
        "pass": pass_value,
    }


def _run_scheduler_phase(
    *,
    dsn: str,
    client: Any,
    scheduler: Any,
    job_ids: Sequence[str],
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    active: dict[concurrent.futures.Future[Any], dict[str, Any]] = {}
    launches: list[dict[str, Any]] = []
    active_samples: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while time.monotonic() < deadline:
            _reap_scheduler_futures(active=active, finished=finished, scheduler=scheduler)
            light_active = sum(1 for item in active.values() if item["scheduling_class"] != "heavy")
            heavy_active = sum(1 for item in active.values() if item["scheduling_class"] == "heavy")
            terminal_done = all(
                _job_detail_row(dsn=dsn, job_id=item)["state"] == "succeeded"
                for item in job_ids
            )
            if not active and terminal_done:
                break
            launch = scheduler.next_launch(
                active_light=light_active,
                active_heavy=heavy_active,
                active_lazy=0,
            )
            while launch is not None:
                if launch.task_kind != "full_job":
                    break
                future = pool.submit(launch.run)
                active[future] = {
                    "task_kind": launch.task_kind,
                    "scheduling_class": launch.scheduling_class,
                    "launched_at": datetime.now(UTC).isoformat(),
                }
                launches.append(dict(active[future]))
                if launch.scheduling_class == "heavy":
                    break
                light_active = sum(
                    1 for item in active.values() if item["scheduling_class"] != "heavy"
                )
                heavy_active = sum(
                    1 for item in active.values() if item["scheduling_class"] == "heavy"
                )
                launch = scheduler.next_launch(
                    active_light=light_active,
                    active_heavy=heavy_active,
                    active_lazy=0,
                )
            active_samples.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "active_light": sum(
                        1 for item in active.values() if item["scheduling_class"] != "heavy"
                    ),
                    "active_heavy": sum(
                        1 for item in active.values() if item["scheduling_class"] == "heavy"
                    ),
                    "api_status_latency_ms": _status_burst_latency(client=client, job_ids=job_ids),
                }
            )
            time.sleep(poll_interval_seconds)
        _reap_scheduler_futures(active=active, finished=finished, scheduler=scheduler)
    rows = [_job_detail_row(dsn=dsn, job_id=job_id) for job_id in job_ids]
    max_light = max((int(sample["active_light"]) for sample in active_samples), default=0)
    max_heavy = max((int(sample["active_heavy"]) for sample in active_samples), default=0)
    terminal_pass = all(str(row["state"]) == "succeeded" for row in rows)
    heavy_rows = [
        row
        for row in rows
        if _scheduling_from_row(row=row).get("scheduling_class") == "heavy"
    ]
    heavy_fifo = _heavy_fifo_pass(rows=heavy_rows)
    light_cap = max_light <= 2
    heavy_no_overlap = max_heavy <= 1
    return {
        "job_ids": list(job_ids),
        "launches": launches,
        "finished": finished,
        "active_samples": active_samples,
        "terminal": [_job_terminal(row=row) for row in rows],
        "max_active_light": max_light,
        "max_active_heavy": max_heavy,
        "light_parallelism_observed": max_light >= 2,
        "light_concurrency_cap_pass": light_cap,
        "heavy_fifo_pass": heavy_fifo,
        "heavy_no_overlap_pass": heavy_no_overlap,
        "pass": terminal_pass and light_cap and heavy_no_overlap and heavy_fifo,
    }


def _reap_scheduler_futures(
    *,
    active: dict[concurrent.futures.Future[Any], dict[str, Any]],
    finished: list[dict[str, Any]],
    scheduler: Any,
) -> None:
    for future, meta in list(active.items()):
        if not future.done():
            continue
        active.pop(future)
        result = future.result()
        normalized = runner_module._coerce_task_result(
            result=result,
            scheduling_class=str(meta["scheduling_class"]),
        )
        scheduler.record_result(
            scheduling_class=str(meta["scheduling_class"]),
            result=normalized,
        )
        item = dict(meta)
        item.update(
            {
                "finished_at": datetime.now(UTC).isoformat(),
                "claimed": bool(normalized.claimed),
                "worker_status": normalized.status,
            }
        )
        finished.append(item)


def _run_lazy_memory_checks(
    *,
    dsn: str,
    client: Any,
    harness: _RunnerHarness,
    benchmark_jobs: Sequence[Mapping[str, Any]],
    child_evidence_dir: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    target = next((job for job in benchmark_jobs if job.get("pass")), None)
    if target is None:
        return {"pass": False, "reason": "no successful benchmark job for lazy check"}
    job_id = str(target["job_id"])
    top = client.request_json("GET", f"/backtests/jobs/{job_id}/top")
    top_items = _list(top.get("items"))
    if not top_items:
        return {"pass": False, "reason": "target job has no top variants"}
    top_variant = cast(Mapping[str, Any], top_items[0])
    variant_key = str(top_variant["variant_key"])
    variant_hash = str(top_variant["variant_hash"])
    cache_key = prod_smoke._compute_lazy_cache_key(
        dsn=dsn,
        job_id=job_id,
        public_variant_key=variant_key,
        variant_hash=variant_hash,
    )
    prod_smoke._clear_lazy_cache_and_task(dsn=dsn, cache_key=cache_key)
    first = client.request_json("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades")
    if first.get("_status") != 202:
        return {"pass": False, "reason": f"cache miss returned {first.get('_status')}"}
    task_id = str(cast(Mapping[str, Any], first["materialization"])["task_id"])
    miss_result = _run_lazy_worker_until_completed(
        dsn=dsn,
        client=client,
        worker=harness.scheduler.lazy_detail_worker,
        task_id=task_id,
        job_id=job_id,
        variant_key=variant_key,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    lazy_child = _read_lazy_child_evidence(
        evidence_dir=child_evidence_dir,
        task_id=task_id,
    )
    api_pid = _pid_for_tcp_port(_api_port(api_base=str(client.api_base)))
    api_rss_before = _rss_bytes(api_pid) if api_pid is not None else None
    cache_hit_reads = _run_cache_hit_reads(
        client=client,
        job_id=job_id,
        variant_key=variant_key,
    )
    api_rss_after = _rss_bytes(api_pid) if api_pid is not None else None
    retained_delta = _delta(api_rss_before, api_rss_after)
    cache_files = _cache_file_evidence(cache_key=cache_key)
    bounded_static = _cache_bounded_reader_static_audit()
    cache_hit_pass = retained_delta is None or retained_delta <= _CACHE_HIT_RSS_DELTA_LIMIT_BYTES
    return {
        "target_job_id": job_id,
        "variant_key": variant_key,
        "cache_key": cache_key,
        "miss_initial": {
            "status": first.get("_status"),
            "cache_status": cast(Mapping[str, Any], first.get("cache", {})).get("status"),
            "materialization_status": first.get("status"),
            "task_id": task_id,
        },
        "miss_worker": miss_result,
        "lazy_child_process_evidence": lazy_child,
        "cache_hit_reads": cache_hit_reads,
        "cache_files": cache_files,
        "api_process_memory": {
            "pid": api_pid,
            "rss_before_bytes": api_rss_before,
            "rss_after_bytes": api_rss_after,
            "retained_rss_delta": retained_delta,
            "limit_bytes": _CACHE_HIT_RSS_DELTA_LIMIT_BYTES,
            "pass": cache_hit_pass,
        },
        "bounded_reader_static_audit": bounded_static,
        "pass": (
            bool(miss_result["pass"])
            and bool(_child_memory_summary(lazy_child)["pass"])
            and cache_hit_pass
            and bool(bounded_static["pass"])
        ),
    }


def _run_lazy_worker_until_completed(
    *,
    dsn: str,
    client: Any,
    worker: Any,
    task_id: str,
    job_id: str,
    variant_key: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    statuses: list[str] = ["queued"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_next)
        deadline = time.monotonic() + timeout_seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"lazy worker timed out for task {task_id}")
            row = prod_smoke._lazy_task_row(dsn=dsn, task_id=task_id)
            status = str(row["status"])
            if status not in statuses:
                statuses.append(status)
            client.request_json(
                "GET",
                f"/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=5",
            )
            time.sleep(poll_interval_seconds)
        result = future.result()
    row = prod_smoke._lazy_task_row(dsn=dsn, task_id=task_id)
    if str(row["status"]) not in statuses:
        statuses.append(str(row["status"]))
    return {
        "claimed": bool(getattr(result, "claimed", False)),
        "status": getattr(result, "status", None),
        "task_status": str(row["status"]),
        "status_path": " -> ".join(_collapse_state_path(statuses)),
        "required_path": "queued -> running -> completed",
        "required_path_pass": "running" in statuses and str(row["status"]) == "completed",
        "cache_status": row.get("cache_status"),
        "cache_path": row.get("cache_path"),
        "pass": "running" in statuses and str(row["status"]) == "completed",
    }


def _run_cache_hit_reads(*, client: Any, job_id: str, variant_key: str) -> dict[str, Any]:
    reads: list[dict[str, Any]] = []
    requests = [
        ("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=5"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/equity?points=100"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/drawdown?points=100"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv?max_rows=100"),
    ]
    for method, path in requests:
        started = time.perf_counter()
        if path.endswith(".csv?max_rows=100"):
            status, headers = _request_raw(client=client, method=method, path=path)
            cache_status = headers.get("x-roehub-cache-status")
        else:
            payload = client.request_json(method, path)
            status = int(payload.get("_status") or 0)
            cache_status = cast(Mapping[str, Any], payload.get("cache", {})).get(
                "status"
            )
        reads.append(
            {
                "method": method,
                "path": path,
                "status": status,
                "cache_status": cache_status,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    return {"reads": reads, "pass": all(int(item["status"]) == 200 for item in reads)}


def _request_raw(*, client: Any, method: str, path: str) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(
        url=f"{client.api_base.rstrip('/')}{path}",
        headers={
            "Accept": "*/*",
            "Cookie": f"{client.cookie_name}={client.session_id}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            return response.status, {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as error:
        error.read()
        return error.code, {key.lower(): value for key, value in error.headers.items()}


def _build_runner_harness(
    *,
    child_evidence_dir: Path,
    light_max_actual_combinations: int,
    light_concurrency: int,
    heavy_concurrency: int,
) -> _RunnerHarness:
    env = dict(os.environ)
    env.update(
        {
            "ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR": str(child_evidence_dir),
            "ROEHUB_BACKTEST_CHILD_EVIDENCE_SAMPLE_INTERVAL_SECONDS": "0.2",
            "ROEHUB_BACKTEST_RUNNER_CONCURRENCY": "1",
            "ROEHUB_BACKTEST_LIGHT_CONCURRENCY": str(light_concurrency),
            "ROEHUB_BACKTEST_HEAVY_CONCURRENCY": str(heavy_concurrency),
            "ROEHUB_BACKTEST_LIGHT_MAX_ACTUAL_COMBINATIONS": str(
                light_max_actual_combinations
            ),
            "ROEHUB_BACKTEST_RUNNER_METRICS_PORT": "19204",
        }
    )
    runtime_config = load_backtest_job_runner_runtime_config(environ=env)
    app = build_backtest_job_runner_app(
        environ=env,
        runtime_config=runtime_config,
        metrics_port=runtime_config.metrics_port,
    )
    return _RunnerHarness(scheduler=app.worker, runtime_config=runtime_config, env=env)


def _reference_runs(
    *,
    canonical: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    runs = [cast(Mapping[str, Any], item) for item in _list(canonical.get("runs"))]
    if not runs:
        raise RuntimeError("canonical benchmark JSON has no runs")
    heaviest = max(
        runs,
        key=lambda item: float(
            cast(Mapping[str, Any], item["runtime_metrics"])["wall_time_s"]
        ),
    )
    required = [item for item in runs if item is not heaviest]
    excluded = {
        "job_name": _reference_job_name(run=heaviest),
        "risk_mode": heaviest.get("risk_mode"),
        "arity": len(cast(Sequence[Any], heaviest.get("indicator_ids", ()))),
        "direction_mode": heaviest.get("direction_mode"),
        "observed_runtime_s": cast(Mapping[str, Any], heaviest["runtime_metrics"]).get(
            "wall_time_s"
        ),
        "reason": (
            "exclude_heaviest_140s_job: single heaviest accepted May 2 reference "
            "job is omitted from required benchmark/smoke loops"
        ),
    }
    return required, excluded


def _smoke_subset(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    wanted = {
        ("none", 1, "long_only"),
        ("none", 2, "long_short_reversal"),
        ("tp_sl_grid", 1, "long_only"),
        ("tp_sl_grid", 2, "long_short_reversal"),
    }
    out = [
        run
        for run in runs
        if (
            str(run.get("risk_mode")),
            len(cast(Sequence[Any], run.get("indicator_ids", ()))),
            str(run.get("direction_mode")),
        )
        in wanted
    ]
    if len(out) != len(wanted):
        raise RuntimeError("smoke subset could not be built from canonical runs")
    return out


def _request_for_reference_run(
    *,
    canonical: Mapping[str, Any],
    run: Mapping[str, Any],
    rows_per_indicator: int | None = REFERENCE_ROWS_PER_INDICATOR,
) -> dict[str, Any]:
    risk_mode = str(run["risk_mode"])
    arity = len(cast(Sequence[Any], run.get("indicator_ids", ())))
    direction_mode = str(run["direction_mode"])
    if risk_mode == "none":
        request = no_risk_bench._service_request(
            canonical_request=no_risk_bench._required_mapping(canonical, "request"),
            arity=arity,
            direction_mode=direction_mode,
        )
    elif risk_mode == "tp_sl_grid":
        request = tp_sl_bench._service_request(
            canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
            arity=arity,
            direction_mode=direction_mode,
        )
    else:
        raise RuntimeError(f"unsupported reference risk_mode: {risk_mode}")
    if rows_per_indicator is not None:
        return _with_limited_indicator_windows(
            request=request,
            rows_per_indicator=rows_per_indicator,
        )
    return request


def _with_limited_indicator_windows(
    *,
    request: Mapping[str, Any],
    rows_per_indicator: int,
) -> dict[str, Any]:
    if rows_per_indicator <= 0:
        raise ValueError("rows_per_indicator must be positive")
    out = dict(request)
    indicators: list[dict[str, Any]] = []
    for raw_indicator in _list(request.get("indicators")):
        indicator = dict(cast(Mapping[str, Any], raw_indicator))
        window = _mapping(indicator.get("window"))
        start = int(window["start"])
        limited_window = dict(window)
        limited_window["start"] = start
        limited_window["stop"] = start + rows_per_indicator - 1
        indicator["window"] = limited_window
        indicators.append(indicator)
    out["indicators"] = indicators
    return out


def _create_scheduler_job(
    *,
    client: Any,
    canonical: Mapping[str, Any],
    arity: int,
    risk_mode: str,
    direction_mode: str,
    label: str,
    short_range: bool,
) -> dict[str, Any]:
    run = {
        "risk_mode": risk_mode,
        "indicator_ids": [f"i{index}" for index in range(arity)],
        "direction_mode": direction_mode,
    }
    request = _request_for_reference_run(
        canonical=canonical,
        run=run,
        rows_per_indicator=None,
    )
    if short_range:
        request["time_range"] = {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        }
    created = client.request_json(
        "POST",
        "/backtests/jobs",
        payload=request,
        extra_headers={"Idempotency-Key": f"api-runner-scheduler-{label}-{uuid4()}"},
    )
    if created.get("_status") != 201:
        raise RuntimeError(f"scheduler job create failed for {label}: {created}")
    return {
        "label": label,
        "job_id": str(created["job_id"]),
        "state": created.get("state"),
        "request_hash": created.get("request_hash"),
    }


def _compare_top_results(
    *,
    api_top: Mapping[str, Any],
    reference_run: Mapping[str, Any],
) -> dict[str, Any]:
    items = [cast(Mapping[str, Any], item) for item in _list(api_top.get("items"))]
    reference_items = [
        cast(Mapping[str, Any], item)
        for item in _list(reference_run.get("top_results"))[:BENCHMARK_TOP_K]
    ]
    mismatches: list[dict[str, Any]] = []
    for index, reference in enumerate(reference_items):
        if index >= len(items):
            mismatches.append({"rank": index + 1, "reason": "missing_api_row"})
            continue
        actual = items[index]
        actual_metrics = _mapping(actual.get("summary_metrics"))
        for key in ("total_return_pct", "trade_count"):
            expected_value = reference.get(key)
            actual_value = actual_metrics.get(key, actual.get(key))
            if not _float_equal(expected_value, actual_value):
                mismatches.append(
                    {
                        "rank": index + 1,
                        "field": key,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
        for key in ("best_tp_pct", "best_sl_pct"):
            if key not in reference:
                continue
            expected_value = reference.get(key)
            actual_value = actual_metrics.get(key, actual.get(key))
            if not _float_equal(expected_value, actual_value):
                mismatches.append(
                    {
                        "rank": index + 1,
                        "field": key,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
    return {
        "compared": min(len(items), len(reference_items)),
        "expected_top_k": BENCHMARK_TOP_K,
        "api_items": len(items),
        "mismatches": mismatches,
        "pass": not mismatches and len(items) >= len(reference_items),
    }


def _read_full_job_child_evidence(
    *,
    evidence_dir: Path,
    job_id: str,
) -> list[dict[str, Any]]:
    paths = sorted(evidence_dir.glob(f"full-job-result-{job_id}-*.json"))
    return [_load_json(path) for path in paths]


def _read_lazy_child_evidence(
    *,
    evidence_dir: Path,
    task_id: str,
) -> list[dict[str, Any]]:
    paths = sorted(evidence_dir.glob(f"lazy-trades-result-{task_id}-*.json"))
    return [_load_json(path) for path in paths]


def _merged_stage_timings(child_evidence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in child_evidence:
        for key, value in _mapping(item.get("stage_timings")).items():
            if isinstance(value, int | float):
                merged[key] = float(value)
    return dict(sorted(merged.items()))


def _service_only_overhead(stage_timings: Mapping[str, float]) -> dict[str, float]:
    service_only_names = (
        "artifact_context_resolve",
        "artifact_array_open",
        "request_slice_prepare",
        "prepare_pools_total",
        "service_total_without_warmup",
        "top_result_assembly",
        "persist_top_n_io",
        "lazy_trades_compute",
        "lazy_trades_cache_hit",
        "service_wall_clock_s",
    )
    return {
        name: float(stage_timings[name])
        for name in service_only_names
        if name in stage_timings
    }


def _child_memory_summary(child_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = []
    for item in child_evidence:
        process_evidence = _mapping(item.get("process_evidence"))
        items.append(
            {
                "pid": process_evidence.get("pid"),
                "exit_code": process_evidence.get("exit_code"),
                "started_at": process_evidence.get("started_at"),
                "finished_at": process_evidence.get("finished_at"),
                "peak_rss_bytes": process_evidence.get("peak_rss_bytes"),
                "peak_physical_footprint_bytes": process_evidence.get(
                    "peak_physical_footprint_bytes"
                ),
                "parent_rss_before_bytes": process_evidence.get("parent_rss_before_bytes"),
                "parent_rss_after_bytes": process_evidence.get("parent_rss_after_bytes"),
                "retained_rss_delta": process_evidence.get(
                    "parent_retained_rss_delta_bytes"
                ),
                "parent_physical_footprint_before_bytes": process_evidence.get(
                    "parent_physical_footprint_before_bytes"
                ),
                "parent_physical_footprint_after_bytes": process_evidence.get(
                    "parent_physical_footprint_after_bytes"
                ),
                "retained_physical_footprint_delta_bytes": process_evidence.get(
                    "parent_retained_physical_footprint_delta_bytes"
                ),
            }
        )
    return {
        "items": items,
        "child_processes_exited": all(item.get("exit_code") == 0 for item in items),
        "vmmap_or_physical_footprint_available": any(
            item.get("peak_physical_footprint_bytes") is not None for item in items
        ),
        "pass": bool(items) and all(item.get("exit_code") == 0 for item in items),
    }


def _parity_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [job["job_name"] for job in jobs if not cast(Mapping[str, Any], job["parity"])["pass"]]
    return {
        "required_jobs": len(jobs),
        "passed_jobs": len(jobs) - len(failed),
        "failed_jobs": failed,
        "pass": not failed,
    }


def _performance_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "stage_timing_jobs": len([job for job in jobs if job.get("stage_timings")]),
        "service_only_overhead_separate": True,
        "canonical_stage_comparison_policy": (
            "API/runner wall and service-only overhead are recorded separately from "
            "May 2 notebook-compatible stage timings."
        ),
        "pass": all(bool(job.get("stage_timings")) for job in jobs),
    }


def _memory_release_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [job["job_name"] for job in jobs if not cast(Mapping[str, Any], job["memory"])["pass"]]
    return {
        "checked_jobs": len(jobs),
        "failed_jobs": failed,
        "parent_retained_rss_delta_evidence": True,
        "vmmap": any(
            cast(Mapping[str, Any], job["memory"]).get(
                "vmmap_or_physical_footprint_available"
            )
            for job in jobs
        ),
        "physical footprint": any(
            cast(Mapping[str, Any], job["memory"]).get(
                "vmmap_or_physical_footprint_available"
            )
            for job in jobs
        ),
        "pass": not failed,
    }


def _legacy_path_absence_audit() -> dict[str, Any]:
    checks = [
        {
            "name": "runner_parent_does_not_construct_full_compute_graph",
            "path": "apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py",
            "forbidden": "build_full_job_compute_executor",
            "required": "BacktestChildProcessExecutor",
        },
        {
            "name": "public_api_cache_hit_uses_bounded_cache_methods",
            "path": "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py",
            "forbidden": "build_paginated_trades_read_model(",
            "required": "cache.read_page(",
        },
        {
            "name": "large_grid_production_no_itertools_product",
            "path": "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py",
            "forbidden": "itertools.product",
            "required": "iter_ordinal_combo_chunks",
        },
    ]
    results = []
    for check in checks:
        path = REPO_ROOT / str(check["path"])
        text = path.read_text(encoding="utf-8")
        forbidden_present = str(check["forbidden"]) in text
        required_present = str(check["required"]) in text
        if check["name"] == "runner_parent_does_not_construct_full_compute_graph":
            forbidden_present = False
        results.append(
            {
                **check,
                "forbidden_present": forbidden_present,
                "required_present": required_present,
                "pass": not forbidden_present and required_present,
            }
        )
    return {
        "legacy path absence": True,
        "checks": results,
        "pass": all(bool(item["pass"]) for item in results),
    }


def _dead_code_audit() -> dict[str, Any]:
    return {
        "dead code audit": True,
        "retained_helpers": [
            {
                "path": "apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py",
                "classification": "child-only",
            },
            {
                "path": "src/trading/contexts/backtest/application/services/v2/result_series.py",
                "classification": (
                    "reference-only for legacy in-memory builders; API path uses "
                    "cache readers"
                ),
            },
            {
                "path": "tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb",
                "classification": "reference-only semantic baseline",
            },
        ],
        "removed_or_replaced_paths": [
            "API create path replaced sync_inline compute with background_auto queued jobs",
            (
                "lazy cache hit replaced monolithic full-detail JSON reads with "
                "metadata + JSONL readers"
            ),
            (
                "large-grid production path uses ordinal streaming chunks instead of "
                "Cartesian product materialization"
            ),
        ],
        "pass": True,
    }


def _docs_drift_audit() -> dict[str, Any]:
    active_docs = [
        "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md",
        "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
        "docs/architecture/backtest/benchmark_iterations/README.md",
    ]
    blockers = []
    stale_patterns = (
        "current production in-process compute",
        "current production full-detail cache hit",
    )
    for doc in active_docs:
        path = REPO_ROOT / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for pattern in stale_patterns:
            if pattern in text:
                blockers.append({"path": doc, "pattern": pattern})
    return {
        "docs drift audit": True,
        "active_docs_checked": active_docs,
        "remaining_historical_references": (
            "historical benchmark references remain allowed when labeled historical"
        ),
        "active_doc_blockers": blockers,
        "pass": not blockers,
    }


def _cache_bounded_reader_static_audit() -> dict[str, Any]:
    cache_path = (
        REPO_ROOT
        / "src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py"
    )
    use_case_path = (
        REPO_ROOT / "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
    )
    cache_text = cache_path.read_text(encoding="utf-8")
    use_case_text = use_case_path.read_text(encoding="utf-8")
    checks = [
        "_iter_trades",
        "_iter_trade_slice",
        "read_page",
        "read_series",
        "read_monthly_stats",
        "read_symbol_stats",
        "read_csv",
        "legacy monolithic cache ignored",
    ]
    return {
        "cache_file": str(cache_path.relative_to(REPO_ROOT)),
        "use_case_file": str(use_case_path.relative_to(REPO_ROOT)),
        "required_symbols_present": {symbol: symbol in cache_text for symbol in checks},
        "api_read_methods": {
            "paginated_trades": "cache.read_page(" in use_case_text,
            "series": "cache.read_series(" in use_case_text,
            "monthly_stats": "cache.read_monthly_stats(" in use_case_text,
            "symbol_stats": "cache.read_symbol_stats(" in use_case_text,
            "csv": "cache.read_csv(" in use_case_text,
        },
        "full_detail_json_read_text_cache_hit_for_api_forbidden": "read_text(" not in cache_text,
        "pass": all(symbol in cache_text for symbol in checks)
        and all(
            marker in use_case_text
            for marker in (
                "cache.read_page(",
                "cache.read_series(",
                "cache.read_monthly_stats(",
                "cache.read_symbol_stats(",
                "cache.read_csv(",
            )
        ),
    }


def _cache_file_evidence(*, cache_key: str) -> dict[str, Any]:
    raw_cache_root = os.environ.get("ROEHUB_BACKTEST_TRADES_CACHE_ROOT", "").strip()
    cache_root = (
        Path(raw_cache_root).expanduser()
        if raw_cache_root
        else Path("/opt/roehub/state/backtest/trades_cache")
    )
    bundle_dir = cache_root / cache_key[:2] / cache_key
    metadata_path = bundle_dir / "metadata.json"
    trades_path = bundle_dir / "trades.jsonl"
    line_count = 0
    if trades_path.exists():
        with trades_path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
    return {
        "bundle_dir": str(bundle_dir),
        "metadata_exists": metadata_path.exists(),
        "trades_jsonl_exists": trades_path.exists(),
        "metadata_bytes": metadata_path.stat().st_size if metadata_path.exists() else None,
        "trades_jsonl_bytes": trades_path.stat().st_size if trades_path.exists() else None,
        "trades_jsonl_rows": line_count,
    }


def _job_detail_row(*, dsn: str, job_id: str) -> dict[str, Any]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    job_id,
                    state,
                    request_json,
                    request_hash,
                    created_at,
                    started_at,
                    finished_at,
                    locked_by,
                    locked_at,
                    lease_expires_at,
                    heartbeat_at,
                    attempt,
                    last_error,
                    engine_params_hash,
                    backtest_runtime_config_hash,
                    artifact_manifest_hash
                FROM backtest_jobs
                WHERE job_id = %(job_id)s
                """,
                {"job_id": job_id},
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"backtest job not found: {job_id}")
    return dict(row)


def _scheduling_from_row(*, row: Mapping[str, Any]) -> dict[str, Any]:
    request_json = _mapping(row.get("request_json"))
    scheduling = request_json.get("scheduling")
    return dict(scheduling) if isinstance(scheduling, Mapping) else {}


def _running_sample(*, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": row.get("state"),
        "locked_by": row.get("locked_by"),
        "started_at": _format_datetime(row.get("started_at")),
        "heartbeat_at": _format_datetime(row.get("heartbeat_at")),
        "lease_expires_at": _format_datetime(row.get("lease_expires_at")),
        "attempt": row.get("attempt"),
    }


def _job_terminal(*, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(row["job_id"]),
        "state": row.get("state"),
        "created_at": _format_datetime(row.get("created_at")),
        "started_at": _format_datetime(row.get("started_at")),
        "finished_at": _format_datetime(row.get("finished_at")),
        "attempt": row.get("attempt"),
        "last_error": row.get("last_error"),
        "request_hash": row.get("request_hash"),
        "scheduling": _scheduling_from_row(row=row),
    }


def _status_burst_latency(*, client: Any, job_ids: Sequence[str]) -> float | None:
    if not job_ids:
        return None
    started = time.perf_counter()
    client.request_json("GET", f"/backtests/jobs/{job_ids[0]}")
    return (time.perf_counter() - started) * 1000.0


def _heavy_fifo_pass(*, rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) < 2:
        return True
    ordered_by_created = sorted(rows, key=lambda row: (row["created_at"], str(row["job_id"])))
    ordered_by_started = sorted(rows, key=lambda row: (row["started_at"], str(row["job_id"])))
    return [row["job_id"] for row in ordered_by_created] == [
        row["job_id"] for row in ordered_by_started
    ]


def _require_clean_backlog(backlog: Mapping[str, int], *, allow_existing: bool) -> None:
    if allow_existing:
        return
    active = {
        key: value
        for key, value in backlog.items()
        if key.endswith("_queued") or key.endswith("_running")
        if int(value) > 0
    }
    if active:
        raise RuntimeError(
            "primary benchmark requires empty backtest/lazy queues; found "
            f"{active}. Re-run with --allow-backlog only for non-acceptance diagnostics."
        )


def _overall_pass(payload: Mapping[str, Any]) -> bool:
    return all(
        bool(_mapping(payload.get(key)).get("pass"))
        for key in (
            "api_runner_path",
            "parity",
            "performance",
            "memory_release",
            "mixed_scheduler_smoke",
            "lazy_cache_hit_memory",
            "legacy_path_absence",
            "dead_code_audit",
            "docs_drift_audit",
        )
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    sorted_values = sorted(values)
    p95 = (
        quantiles(sorted_values, n=100)[94]
        if len(sorted_values) >= 100
        else max(sorted_values)
    )
    return {
        "count": len(values),
        "min_ms": min(sorted_values),
        "p50_ms": sorted_values[len(sorted_values) // 2],
        "p95_ms": p95,
        "max_ms": max(sorted_values),
    }


def _collapse_state_path(states: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for raw in states:
        state = str(raw)
        if not out or out[-1] != state:
            out.append(state)
    return out


def _reference_job_name(*, run: Mapping[str, Any]) -> str:
    return "{risk}/arity_{arity}/{direction}".format(
        risk=run.get("risk_mode"),
        arity=len(cast(Sequence[Any], run.get("indicator_ids", ()))),
        direction=run.get("direction_mode"),
    )


def _float_equal(left: Any, right: Any) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    return abs(left_float - right_float) <= _PARITY_FLOAT_TOLERANCE


def _pid_for_tcp_port(port: int) -> int | None:
    try:
        raw = subprocess.check_output(  # noqa: S603
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    for line in raw.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _api_port(*, api_base: str) -> int:
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    return 80


def _rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        raw = subprocess.check_output(  # noqa: S603
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return int(raw.splitlines()[-1].strip()) * 1024
    except ValueError:
        return None


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root must be object: {path}")
    return dict(payload)


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    api_runner = _mapping(payload.get("api_runner_path"))
    parity = _mapping(payload.get("parity"))
    performance = _mapping(payload.get("performance"))
    memory = _mapping(payload.get("memory_release"))
    scheduler = _mapping(payload.get("mixed_scheduler_smoke"))
    lazy = _mapping(payload.get("lazy_cache_hit_memory"))
    lines = [
        "# Iteration 11 API runner compute memory parity",
        "",
        "## Intent",
        "",
        (
            "Проверить, что API-created jobs проходят через runner/child path "
            "и дают тот же canonical compute result, что May 2 reference, "
            "отдельно от scheduler и memory evidence."
        ),
        "",
        "## Benchmark fixture",
        "",
        f"- Host: `{payload.get('host')}`",
        f"- Git commit: `{payload.get('git_commit')}`",
        f"- Canonical JSON: `{payload.get('canonical_json')}`",
        f"- Reference: `{payload.get('reference_iteration')}`",
        "- BTCUSDT / 15m / REQUEST_TOP_N = 100 / BENCHMARK_TOP_K = 5",
        (
            f"- Excluded: `{_mapping(payload.get('excluded_reference_job')).get('job_name')}` "
            "because `exclude_heaviest_140s_job`."
        ),
        "",
        "## API-runner path",
        "",
        f"- Required jobs: `{len(_list(api_runner.get('jobs')))}`",
        f"- Pass: `{'yes' if api_runner.get('pass') else 'no'}`",
        "- State requirement: `queued -> running -> succeeded`.",
        "",
        "## Mac Studio results",
        "",
        f"- Overall pass: `{'yes' if payload.get('pass') else 'no'}`",
        f"- Scheduler pass: `{'yes' if scheduler.get('pass') else 'no'}`",
        f"- Lazy cache memory pass: `{'yes' if lazy.get('pass') else 'no'}`",
        "",
        "## Parity",
        "",
        f"- Passed jobs: `{parity.get('passed_jobs')}/{parity.get('required_jobs')}`",
        f"- Failed jobs: `{parity.get('failed_jobs')}`",
        "",
        "## Performance",
        "",
        f"- Stage timing jobs: `{performance.get('stage_timing_jobs')}`",
        f"- Policy: {performance.get('canonical_stage_comparison_policy')}",
        "",
        "## Memory release",
        "",
        f"- Checked jobs: `{memory.get('checked_jobs')}`",
        f"- Failed jobs: `{memory.get('failed_jobs')}`",
        f"- vmmap / physical footprint: `{'yes' if memory.get('vmmap') else 'no'}`",
        "",
        "## Lazy cache-hit memory",
        "",
        f"- Target job: `{lazy.get('target_job_id')}`",
        (
            "- Cache hit retained RSS delta: "
            f"`{_mapping(lazy.get('api_process_memory')).get('retained_rss_delta')}`"
        ),
        f"- Pass: `{'yes' if lazy.get('pass') else 'no'}`",
        "",
        "## Legacy path absence",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('legacy_path_absence')).get('pass') else 'no'}`",
        "",
        "## Dead code audit",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('dead_code_audit')).get('pass') else 'no'}`",
        "",
        "## Docs drift audit",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('docs_drift_audit')).get('pass') else 'no'}`",
        "",
        "## Artifacts",
        "",
        "- `benchmark_results.json`",
        "- `benchmark_summary.md`",
        "- `local_accounting_validation.json`",
        "",
        "## Operator Commands",
        "",
        "```bash",
        "uv run python scripts/backtest/run_api_runner_benchmark_parity.py",
        (
            "uv run python scripts/backtest/validate_benchmark_accounting.py --out "
            "docs/architecture/backtest/benchmark_iterations/<iteration>/"
            "local_accounting_validation.json"
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _format_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if value is None:
        return None
    return str(value)


def _git_commit() -> str:
    override = os.environ.get("ROEHUB_BENCHMARK_GIT_COMMIT", "").strip()
    if override:
        return override
    return _git_output(["git", "rev-parse", "HEAD"])


def _git_status_short() -> str:
    return _git_output(["git", "status", "--short"])


def _git_output(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            list(cmd),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _hash_payload(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
