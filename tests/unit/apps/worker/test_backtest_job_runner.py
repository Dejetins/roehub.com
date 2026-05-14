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
    BacktestRunnerTaskResult,
    BacktestRunnerTaskScheduler,
    load_backtest_job_runner_runtime_config,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobWorkerResult,
    BacktestLazyTradesMaterializationWorkerResult,
)

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
        "light_concurrency": 0,
        "heavy_concurrency": 1,
        "poll_interval_seconds": 0.001,
        "empty_backoff_seconds": 0.001,
        "lease_seconds": 120,
        "heartbeat_interval_seconds": 30.0,
        "max_jobs_per_process": 1,
        "metrics_port": 19204,
        "child_timeout_seconds": 21600.0,
        "light_max_estimated_combinations": 50000,
        "light_max_actual_combinations": 50000,
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
    scheduler = _SingleLaunchScheduler(worker=worker)
    metrics = BacktestJobRunnerMetrics()
    app = BacktestJobRunnerApp(
        runtime_config=_runtime_config(max_jobs_per_process=1),
        worker=cast(Any, scheduler),
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


def test_runner_metrics_records_lazy_detail_cache_status() -> None:
    metrics = BacktestJobRunnerMetrics()
    result = BacktestRunnerTaskResult(
        task_kind="lazy_detail",
        claimed=True,
        status="completed",
        created_at=datetime.now(UTC) - timedelta(seconds=5),
        started_at=datetime.now(UTC) - timedelta(seconds=1),
        cache_status="miss",
    )

    metrics.observe_result(result=result, duration_seconds=0.25)

    assert metrics.tasks_claimed_total.labels(
        task_kind="lazy_detail",
        paid_level="unknown",
    )._value.get() == 1
    assert metrics.tasks_finished_total.labels(
        task_kind="lazy_detail",
        status="completed",
    )._value.get() == 1
    assert metrics.lazy_trades_cache_total.labels(status="miss")._value.get() == 1


def test_task_scheduler_uses_heavy_lane_only_for_full_jobs() -> None:
    heavy_worker = _TaskWorker(
        results=[
            BacktestJobWorkerResult(
                job=cast(Any, _job(state="succeeded")),
                claimed=True,
            )
        ]
    )
    lazy_worker = _TaskWorker(results=[])
    scheduler = BacktestRunnerTaskScheduler(
        heavy_full_job_worker=cast(Any, heavy_worker),
        lazy_detail_worker=cast(Any, lazy_worker),
    )

    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)

    assert launch is not None
    assert launch.scheduling_class == "heavy"
    result = launch.run()
    assert isinstance(result, BacktestJobWorkerResult)
    assert result.claimed is True
    assert heavy_worker.calls == 1
    assert scheduler.next_launch(active_light=1, active_heavy=0, active_lazy=0) is None


class _SingleLaunchScheduler:
    def __init__(self, *, worker: _Worker) -> None:
        self._worker = worker
        self._launched = False

    def next_launch(
        self,
        *,
        active_light: int,
        active_heavy: int,
        active_lazy: int,
    ) -> Any:
        _ = active_light, active_heavy, active_lazy
        if self._launched:
            return None
        self._launched = True
        return SimpleNamespace(
            task_kind="full_job",
            scheduling_class="heavy",
            run=self._worker.run_next,
        )

    def record_result(
        self,
        *,
        scheduling_class: str,
        result: BacktestRunnerTaskResult,
    ) -> None:
        _ = scheduling_class, result


class _Worker:
    def __init__(self, *, results: list[BacktestJobWorkerResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run_next(self) -> BacktestJobWorkerResult:
        self.calls += 1
        if not self._results:
            return BacktestJobWorkerResult(job=None, claimed=False)
        return self._results.pop(0)


class _TaskWorker:
    def __init__(self, *, results: list[Any]) -> None:
        self._results = list(results)
        self.calls = 0

    def run_next(self) -> Any:
        self.calls += 1
        if not self._results:
            return BacktestLazyTradesMaterializationWorkerResult(task=None, claimed=False)
        return self._results.pop(0)


def _lazy_task() -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        created_at=datetime.now(UTC) - timedelta(seconds=5),
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )


def _job(*, state: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        request_json={"admission": {"paid_level": "pro"}},
        created_at=datetime.now(UTC) - timedelta(seconds=5),
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )
