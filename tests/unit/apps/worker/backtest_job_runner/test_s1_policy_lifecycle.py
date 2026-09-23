from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from apps.worker.backtest_job_runner.wiring.modules import process_observation
from apps.worker.backtest_job_runner.wiring.modules.child_ipc import BacktestChildSuccessResult
from apps.worker.backtest_job_runner.wiring.modules.child_process import (
    BacktestChildProcessError,
    BacktestChildProcessExecutor,
)
from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
    BacktestCpuCapacity,
    full_job_resource_environ,
)
from tests.unit.apps.worker.backtest_job_runner.test_child_process_executor import _preflight
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    backtest_numba_environ,
    resolve_backtest_numba_thread_decision,
)
from trading.contexts.backtest.application.services.v2.job_scratch import BacktestJobScratch
from trading.contexts.backtest.application.use_cases import BacktestJobCancellationRequested

ROOT = Path(__file__).resolve().parents[5]


def test_policy_is_immutable_with_production_bounds():
    policy = BacktestComputePolicy()
    assert policy.threads.num_threads == 12
    assert policy.cost_permutation_min_rows == 32
    with pytest.raises(FrozenInstanceError):
        setattr(policy, "cost_permutation_min_rows", 1)
    with pytest.raises(ValueError):
        BacktestComputePolicy(integer_tape_min_bars=0)


@pytest.mark.parametrize("budget", [1, 2, 6, 12])
def test_selected_budget_survives_child_preimport(budget):
    env = {
        "ROEHUB_BACKTEST_NUMBA_NUM_THREADS": str(budget),
        "ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "18",
    }
    parent = backtest_numba_environ(environ=env, scheduling_class="heavy")
    # The effective envelope is authoritative in the child even if selectors differ.
    parent["ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS"] = "1"
    child = backtest_numba_environ(environ=parent, scheduling_class="heavy", inherited=True)
    assert child["NUMBA_NUM_THREADS"] == str(budget)
    assert env["NUMBA_NUM_THREADS"] == "18"
    assert (
        resolve_backtest_numba_thread_decision(environ=parent, scheduling_class="heavy").num_threads
        == 1
    )


def test_budget_precedence_cap_and_invalid_envelope():
    env = {"ROEHUB_BACKTEST_NUMBA_NUM_THREADS": "4", "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "6"}
    capacity = BacktestCpuCapacity(18, 8, None, 6)
    assert full_job_resource_environ(environ=env, capacity=capacity)["NUMBA_NUM_THREADS"] == "6"
    with pytest.raises(ValueError, match="capacity"):
        full_job_resource_environ(environ=env, capacity=BacktestCpuCapacity(18, 4, None, None))
    assert (
        full_job_resource_environ(environ={}, capacity=BacktestCpuCapacity(2, None, None, None))[
            "NUMBA_NUM_THREADS"
        ]
        == "12"
    )
    with pytest.raises(ValueError, match="maximum"):
        backtest_numba_environ(
            environ={"ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS": "6", "NUMBA_NUM_THREADS": "12"},
            scheduling_class="heavy",
            inherited=True,
        )


def test_scratch_clear_releases_inputs_and_rejects_reuse():
    import weakref

    class Input:
        pass

    scratch = BacktestJobScratch(uuid4())
    value = Input()
    ref = weakref.ref(value)
    scratch.retain("input", value)
    del value
    assert ref() is not None
    scratch.clear()
    assert ref() is None and scratch.closed and scratch.retained_count == 0
    scratch.clear()
    with pytest.raises(RuntimeError):
        scratch.retain("next", object())


@pytest.mark.parametrize("mode", ["success", "failure", "malformed", "timeout", "cancel"])
def test_real_child_partial_ipc_cleanup_and_reaping(tmp_path, mode):
    record = tmp_path / "record.json"
    env = {
        **os.environ,
        "PYTHONPATH": f'{ROOT / "src"}:{ROOT}',
        "PYTHONDONTWRITEBYTECODE": "1",
        "S1_CHILD_RECORD": str(record),
        "S1_CHILD_MODE": mode,
        "ROEHUB_BACKTEST_NUMBA_NUM_THREADS": "2",
        "ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR": str(tmp_path / "evidence"),
        "ROEHUB_BACKTEST_CHILD_EVIDENCE_COLLECT_RSS": "0",
        "ROEHUB_BACKTEST_CHILD_EVIDENCE_SAMPLE_INTERVAL_SECONDS": ".01",
    }
    env.pop("ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS", None)
    parent_env = dict(os.environ)
    event = threading.Event()
    thread = None
    if mode == "cancel":

        def cancel():
            deadline = time.monotonic() + 5
            while not record.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            event.set()

        thread = threading.Thread(target=cancel)
        thread.start()
    executor = BacktestChildProcessExecutor(
        environ=env,
        scheduling_class="heavy",
        light_max_actual_combinations=50000,
        timeout_seconds=5 if mode == "timeout" else 10,
        child_module="tests.unit.apps.worker.backtest_job_runner.s1_child",
    )
    try:
        if mode == "success":
            result = executor.execute(
                job_id=uuid4(),
                preflight=_preflight(),
                updated_at=datetime.now(UTC),
                cancel_event=event,
            )
            assert isinstance(result, BacktestChildSuccessResult)
            assert result.cleanup_evidence["threads"] == "2"
        else:
            expected = (
                BacktestJobCancellationRequested
                if mode == "cancel"
                else json.JSONDecodeError
                if mode == "malformed"
                else BacktestChildProcessError
            )
            with pytest.raises(expected):
                executor.execute(
                    job_id=uuid4(),
                    preflight=_preflight(),
                    updated_at=datetime.now(UTC),
                    cancel_event=event,
                )
    finally:
        if thread:
            thread.join(timeout=6)
    data = json.loads(record.read_text())
    assert Path(data["module"]).is_relative_to(ROOT)
    assert not Path(data["output"]).parent.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(data["pid"], 0)
    assert dict(os.environ) == parent_env
    evidence = next(
        item for path in (tmp_path / "evidence").glob("*.json")
        if (item := json.loads(path.read_text())).get("schema")
        == "roehub_child_process_evidence_v1"
    )
    assert evidence["pid"] == data["pid"]


def test_real_observer_exception_reaps_child(tmp_path, monkeypatch):
    child = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        child.append(process)
        return process

    monkeypatch.setattr(process_observation.subprocess, "Popen", launch)
    calls = 0

    def rss(pid):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected observer failure")
        return 1

    monkeypatch.setattr(process_observation, "_rss_bytes", rss)
    with pytest.raises(RuntimeError, match="observer failure"):
        process_observation.run_observed_subprocess(
            cmd=[sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            env={**os.environ, "ROEHUB_BACKTEST_CHILD_EVIDENCE_COLLECT_RSS": "1"},
            timeout_seconds=10,
            evidence_prefix="injection",
            metadata={},
        )
    assert len(child) == 1 and child[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child[0].pid, 0)


def test_lease_loss_cancels_real_child_without_partial_persistence(tmp_path):
    from typing import Any, cast

    from tests.unit.contexts.backtest.application.use_cases.test_backtest_job_worker_use_case import (  # noqa: E501
        _LeaseRepository,
        _PreflightService,
        _queued_job,
        _Repository,
    )
    from trading.contexts.backtest.application.use_cases import BacktestJobWorkerUseCase

    record = tmp_path / "lease-child.json"
    repository = _Repository(job=_queued_job(), finish_returns_none=True)

    class LostLease(_LeaseRepository):
        def heartbeat(self, **kwargs):
            if record.exists():
                return None
            return self.repository.job

    env = {
        **os.environ,
        "PYTHONPATH": f'{ROOT / "src"}:{ROOT}',
        "PYTHONDONTWRITEBYTECODE": "1",
        "S1_CHILD_RECORD": str(record),
        "S1_CHILD_MODE": "cancel",
        "ROEHUB_BACKTEST_NUMBA_NUM_THREADS": "2",
        "ROEHUB_BACKTEST_CHILD_EVIDENCE_COLLECT_RSS": "0",
    }
    env.pop("ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS", None)
    executor = BacktestChildProcessExecutor(
        environ=env,
        scheduling_class="heavy",
        light_max_actual_combinations=50000,
        timeout_seconds=8,
        child_module="tests.unit.apps.worker.backtest_job_runner.s1_child",
    )
    use_case = BacktestJobWorkerUseCase(
        lease_repository=LostLease(repository),
        job_repository=cast(Any, repository),
        preflight_service=cast(Any, _PreflightService()),
        executor=executor,
        lease_seconds=60,
        heartbeat_interval_seconds=0.01,
        locked_by="s1-local",
    )
    result = use_case.run_next()
    assert result.lease_lost and result.status == "cancelled"
    assert repository.top_rows == () and repository.terminal_commits == 0
    data = json.loads(record.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(data["pid"], 0)
    assert not Path(data["output"]).exists()
