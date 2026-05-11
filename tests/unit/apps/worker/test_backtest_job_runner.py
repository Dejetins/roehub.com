from __future__ import annotations

import asyncio
import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from apps.worker.backtest_job_runner.wiring.modules import (
    BacktestJobRunnerApp,
    BacktestJobRunnerMetrics,
    BacktestJobRunnerRuntimeConfig,
    load_backtest_job_runner_runtime_config,
)
from trading.contexts.backtest.application.use_cases import BacktestJobWorkerResult

backtest_job_runner_main = importlib.import_module(
    "apps.worker.backtest_job_runner.main.main"
)


class _NoOpApp:
    async def run(self, _stop_event: asyncio.Event) -> None:
        return None


def _runtime_config(**overrides: Any) -> BacktestJobRunnerRuntimeConfig:
    payload = {
        "enabled": True,
        "concurrency": 1,
        "poll_interval_seconds": 0.001,
        "empty_backoff_seconds": 0.001,
        "lease_seconds": 120,
        "heartbeat_interval_seconds": 30.0,
        "max_jobs_per_process": 1,
        "metrics_port": 19204,
    }
    payload.update(overrides)
    return BacktestJobRunnerRuntimeConfig(**payload)


def test_load_config_requires_concurrency_one() -> None:
    try:
        load_backtest_job_runner_runtime_config(
            environ={"ROEHUB_BACKTEST_RUNNER_CONCURRENCY": "2"}
        )
    except ValueError as error:
        assert "ROEHUB_BACKTEST_RUNNER_CONCURRENCY=1" in str(error)
    else:
        raise AssertionError("expected concurrency validation failure")


def test_run_async_exits_zero_when_runner_disabled(monkeypatch) -> None:
    calls = {"build": 0}

    monkeypatch.setattr(
        backtest_job_runner_main,
        "load_backtest_job_runner_runtime_config",
        lambda *, environ: _runtime_config(enabled=False),
    )

    def _build_app(**_kwargs: Any) -> _NoOpApp:
        calls["build"] += 1
        return _NoOpApp()

    monkeypatch.setattr(backtest_job_runner_main, "build_backtest_job_runner_app", _build_app)
    monkeypatch.setattr(
        backtest_job_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    exit_code = asyncio.run(backtest_job_runner_main._run_async(metrics_port=None))

    assert exit_code == 0
    assert calls["build"] == 0


def test_run_async_metrics_port_cli_override_has_priority(monkeypatch) -> None:
    received_metrics_ports: list[int] = []

    monkeypatch.setattr(
        backtest_job_runner_main,
        "load_backtest_job_runner_runtime_config",
        lambda *, environ: _runtime_config(metrics_port=19204),
    )

    def _build_app(
        *,
        environ: Any,
        runtime_config: BacktestJobRunnerRuntimeConfig,
        metrics_port: int,
    ) -> _NoOpApp:
        _ = environ, runtime_config
        received_metrics_ports.append(metrics_port)
        return _NoOpApp()

    monkeypatch.setattr(backtest_job_runner_main, "build_backtest_job_runner_app", _build_app)
    monkeypatch.setattr(
        backtest_job_runner_main,
        "_install_signal_handlers",
        lambda _stop_event: None,
    )

    cli_override_exit_code = asyncio.run(
        backtest_job_runner_main._run_async(metrics_port=19304)
    )
    env_default_exit_code = asyncio.run(backtest_job_runner_main._run_async(metrics_port=None))

    assert cli_override_exit_code == 0
    assert env_default_exit_code == 0
    assert received_metrics_ports == [19304, 19204]


def test_runner_app_starts_metrics_and_exits_after_max_jobs(monkeypatch) -> None:
    started_ports: list[int] = []
    job = SimpleNamespace(
        job_id=uuid4(),
        state="succeeded",
        request_json={"admission": {"paid_level": "pro"}},
        created_at=datetime.now(UTC) - timedelta(seconds=5),
        started_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    worker = _Worker(results=[BacktestJobWorkerResult(job=cast(Any, job), claimed=True)])
    metrics = BacktestJobRunnerMetrics()
    app = BacktestJobRunnerApp(
        runtime_config=_runtime_config(max_jobs_per_process=1),
        worker=cast(Any, worker),
        metrics=metrics,
        metrics_port=19204,
    )

    monkeypatch.setattr(
        "apps.worker.backtest_job_runner.wiring.modules.backtest_job_runner.start_http_server",
        lambda port, *, registry: started_ports.append(port),
    )

    asyncio.run(app.run(asyncio.Event()))

    assert started_ports == [19204]
    assert worker.calls == 1
    assert metrics.tasks_claimed_total.labels(
        task_kind="full_job",
        paid_level="pro",
    )._value.get() == 1
    assert metrics.tasks_finished_total.labels(
        task_kind="full_job",
        status="succeeded",
    )._value.get() == 1
    assert metrics.last_success_unixtime.labels(task_kind="full_job")._value.get() > 0


class _Worker:
    def __init__(self, *, results: list[BacktestJobWorkerResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run_next(self) -> BacktestJobWorkerResult:
        self.calls += 1
        if not self._results:
            return BacktestJobWorkerResult(job=None, claimed=False)
        return self._results.pop(0)
