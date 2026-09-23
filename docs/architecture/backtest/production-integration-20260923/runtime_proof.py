"""Disposable PostgreSQL acceptance through actual repositories/scheduler/child IPC."""

import argparse
import dataclasses
import datetime
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path("/Users/daniildegtyarev/.codex/worktrees/224b/roehub.com")
p = argparse.ArgumentParser()
p.add_argument("--task", type=Path, required=True)
task = json.loads(p.parse_args().task.read_text())
source = Path(task["source"])
sys.meta_path[:] = [x for x in sys.meta_path if "__editable__" not in str(x)]
sys.path[:] = [str(source / "src"), str(source), str(ROOT)] + [
    x
    for x in sys.path
    if x
    not in [
        "",
        str(ROOT),
        "/Users/daniildegtyarev/Projects/roehub.com",
        "/Users/daniildegtyarev/Projects/roehub.com/src",
    ]
]
import psycopg  # noqa: E402

from apps.worker.backtest_job_runner.wiring.modules.backtest_job_runner import (  # noqa: E402
    BacktestRunnerTaskResult,
    build_backtest_job_runner_app,
)  # noqa: E402

compare = importlib.import_module("scripts.backtest.cpu_integration.exact").compare
digest = importlib.import_module("scripts.backtest.cpu_integration.exact").digest
encode = importlib.import_module("scripts.backtest.cpu_integration.exact").encode
setup = importlib.import_module("scripts.backtest.cpu_integration.setup").setup

# The helper adds its harness root; restore selected product source precedence.
sys.path[:] = [str(source / "src"), str(source)] + [
    value for value in sys.path if value not in {str(source / "src"), str(source)}
]

from trading.contexts.backtest.application.use_cases.backtest_jobs import (  # noqa: E402
    DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
    _job_request_json,
    _market_id,
    _ranking_primary_metric,
    _symbol,
)
from trading.contexts.backtest.domain.entities import (  # noqa: E402
    BacktestJob,
    BacktestJobArtifactPin,
)
from trading.shared_kernel.primitives import OrganizationId, UserId  # noqa: E402

DSN = os.environ["S6_DSN"]


def now():
    return datetime.datetime.now(datetime.UTC)


user = UserId(UUID(int=6001))
org = OrganizationId(UUID(int=6002))
pf, env = setup(Path(task["dataset"]))
env = {
    **os.environ,
    **env,
    "STRATEGY_PG_DSN": DSN,
    "S6_SOURCE": str(source),
    "S6_POLICY": task["policy"],
    "S6_SCORING_MARKER": task["scratch"] + "/scoring-start",
    "PYTHONPATH": str(source) + os.pathsep + str(source / "src"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "NUMBA_CACHE_DIR": task["cache"],
    "NUMBA_NUM_THREADS": "12",
    "NUMBA_THREADING_LAYER": "workqueue",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "ROEHUB_BACKTEST_NUMBA_NUM_THREADS": "12",
    "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "12",
    "ROEHUB_BACKTEST_RUNNER_CONCURRENCY": "1",
    "ROEHUB_BACKTEST_LIGHT_CONCURRENCY": "0",
    "ROEHUB_BACKTEST_HEAVY_CONCURRENCY": "1",
    "ROEHUB_BACKTEST_RUNNER_HEARTBEAT_INTERVAL_SECONDS": "0.1",
    "ROEHUB_BACKTEST_RUNNER_LEASE_SECONDS": "60",
    "ROEHUB_BACKTEST_TRADES_CACHE_ROOT": task["scratch"] + "/lazy",
    "TMPDIR": task["scratch"],
}
env.pop("ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR", None)
Path(task["scratch"]).mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = task["scratch"]
__import__("tempfile").tempdir = task["scratch"]
app = build_backtest_job_runner_app(environ=env)
scheduler: Any = app.worker
worker = scheduler.heavy_full_job_worker
repo = worker.job_repository
actual = worker.executor


class Capture:
    result: Any = None

    def execute(self, **kw):
        self.result = actual.execute(**kw)
        return self.result


capture = Capture()
worker = dataclasses.replace(worker, executor=capture)
scheduler.heavy_full_job_worker = worker
records = []
jobs = {}
with psycopg.connect(DSN) as c:
    c.execute("DELETE FROM backtest_jobs")


def enqueue(case, index):
    pre = pf.execute(case["request_payload"])
    meta = pre.artifact_metadata
    jid = UUID(int=task["trace_index"] * 100 + index + 10000)
    assert meta.artifact_slot in ("slot_a", "slot_b")
    pin = BacktestJobArtifactPin(
        artifact_slot="slot_a" if meta.artifact_slot == "slot_a" else "slot_b",
        artifact_slot_generation=meta.artifact_slot_generation,
        artifact_manifest_hash=meta.artifact_manifest_hash,
        artifact_asof_date=meta.artifact_asof_date,
    )
    job = BacktestJob.create_queued(
        job_id=jid,
        organization_id=org,
        user_id=user,
        mode="template",
        created_at=now(),
        request_json=_job_request_json(
            preflight=pre,
            key_hash=None,
            light_max_estimated_combinations=DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
            strategy_name=None,
        ),
        request_hash=pre.request_hash,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash=pre.result_config_hash,
        backtest_runtime_config_hash=pre.result_config_hash,
        artifact_pin=pin,
        execution_mode="background_auto",
        market_id=_market_id(preflight=pre),
        symbol=_symbol(preflight=pre),
        timeframe=str(pre.normalized_request["timeframe"]),
        requested_top_n=int(pre.normalized_request["top_n"]),
        ranking_primary_metric=_ranking_primary_metric(preflight=pre),
        ranking_secondary_metric=None,
    )
    repo.create(job=job)
    jobs[jid] = (case, time.perf_counter())
    return jid


# Frozen arrivals are a simultaneous queue burst; insertion times retained separately.
for i, case in enumerate(task["cases"]):
    enqueue(case, i)
start = time.perf_counter()
remaining = len(jobs)
empty_attempts = 0
while remaining:
    assert scheduler.next_launch(active_light=0, active_heavy=1, active_lazy=0) is None
    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    t = time.perf_counter()
    result = launch.run()
    duration = time.perf_counter() - t
    if not result.claimed:
        empty_attempts += 1
        assert empty_attempts < 10
        scheduler.record_result(
            scheduling_class=launch.scheduling_class,
            result=BacktestRunnerTaskResult(task_kind=launch.task_kind, claimed=False),
        )
        continue
    remaining -= 1
    job = result.job
    jid = job.job_id if job else next(iter(jobs))
    rows = repo.list_top_variants(job_id=jid, organization_id=org)
    assert job and job.state == "succeeded", (
        None if job is None else (job.state, job.last_error, job.last_error_json)
    )
    assert all(row.updated_at == job.updated_at == job.finished_at for row in rows)
    expected_rows = tuple(
        dataclasses.replace(row, updated_at=job.finished_at) for row in capture.result.top_variants
    )
    parity = compare(encode(expected_rows), encode(rows))
    assert parity["status"] == "pass", parity
    records.append(
        {
            "job": str(jid),
            "case": jobs[jid][0]["case_id"],
            "service_s": duration,
            "end_to_end_s": time.perf_counter() - jobs[jid][1],
            "wait_s": t - jobs[jid][1],
            "state": None if job is None else job.state,
            "lease_lost": result.lease_lost,
            "rows": len(rows),
            "parity": parity,
            "readback_digest": digest(encode(rows)),
            "prefix": capture.result.exact_diagnostics["telemetry"].get("prefix_traversal"),
        }
    )
    scheduler.record_result(
        scheduling_class=launch.scheduling_class,
        result=BacktestRunnerTaskResult(task_kind="full_job", claimed=True, status="completed"),
    )
    capture.result = None
with psycopg.connect(DSN) as c:
    size_row = c.execute("SELECT pg_database_size(current_database())").fetchone()
    assert size_row is not None
    size = size_row[0]
    persisted_row = c.execute("SELECT count(*) FROM backtest_job_top_variants").fetchone()
    assert persisted_row is not None
    persisted = persisted_row[0]

print(
    json.dumps(
        {
            "records": records,
            "makespan_s": time.perf_counter() - start,
            "database_bytes": size,
            "persisted_rows": persisted,
            "child_module": actual.child_module,
            "cleanup_files": [p.name for p in Path(task["scratch"]).glob("roehub-*")],
        },
        separators=(",", ":"),
    )
)
