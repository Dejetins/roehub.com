from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import cast

from apps.worker.backtest_job_runner.wiring.modules.backtest_job_runner import (
    BacktestRunnerTaskResult,
    BacktestRunnerTaskScheduler,
    build_backtest_job_runner_app,
)
from apps.worker.backtest_job_runner.wiring.modules.full_job_compute import (
    build_full_job_compute_executor,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobWorkerUseCase,
    BacktestLazyTradesMaterializationWorkerUseCase,
)


def test_scheduler_probes_heavy_before_light_and_batches_light_slots() -> None:
    scheduler = _scheduler()

    first = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    assert first is not None
    assert first.scheduling_class == "heavy"
    scheduler.record_result(
        scheduling_class="heavy",
        result=BacktestRunnerTaskResult(task_kind="full_job", claimed=False),
    )

    second = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    assert second is not None
    assert second.scheduling_class == "light_candidate"
    third = scheduler.next_launch(active_light=1, active_heavy=0, active_lazy=0)
    assert third is not None
    assert third.scheduling_class == "light_candidate"


def test_scheduler_rechecks_heavy_after_light_anti_starvation_limit() -> None:
    scheduler = _scheduler(full_job_anti_starvation_limit=1)
    scheduler.record_result(
        scheduling_class="heavy",
        result=BacktestRunnerTaskResult(task_kind="full_job", claimed=False),
    )
    scheduler.record_result(
        scheduling_class="light_candidate",
        result=BacktestRunnerTaskResult(
            task_kind="full_job",
            claimed=True,
            scheduling_class="light_candidate",
        ),
    )

    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)

    assert launch is not None
    assert launch.scheduling_class == "heavy"


def test_scheduler_returns_to_full_poll_after_empty_lazy_probe() -> None:
    scheduler = _scheduler()

    _record_empty_full_probe(scheduler=scheduler, scheduling_class="heavy")
    _record_empty_full_probe(scheduler=scheduler, scheduling_class="light_candidate")
    _record_empty_full_probe(scheduler=scheduler, scheduling_class="heavy")
    lazy = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    assert lazy is not None
    assert lazy.task_kind == "lazy_detail"

    scheduler.record_result(
        scheduling_class="none",
        result=BacktestRunnerTaskResult(task_kind="lazy_detail", claimed=False),
    )
    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)

    assert launch is not None
    assert launch.scheduling_class == "heavy"


def test_scheduler_limits_consecutive_lazy_claims_before_full_probe() -> None:
    scheduler = _scheduler(lazy_detail_anti_starvation_limit=2)

    _record_empty_full_probe(scheduler=scheduler, scheduling_class="heavy")
    _record_empty_full_probe(scheduler=scheduler, scheduling_class="light_candidate")
    _record_empty_full_probe(scheduler=scheduler, scheduling_class="heavy")
    for _ in range(2):
        lazy = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
        assert lazy is not None
        assert lazy.task_kind == "lazy_detail"
        scheduler.record_result(
            scheduling_class="none",
            result=BacktestRunnerTaskResult(task_kind="lazy_detail", claimed=True),
        )

    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)

    assert launch is not None
    assert launch.scheduling_class == "heavy"


def test_production_runner_wiring_does_not_construct_full_compute_service_in_parent() -> None:
    source = inspect.getsource(build_backtest_job_runner_app)

    assert "BacktestRuntimeJobOrchestrationService" not in source
    assert "BacktestChildProcessExecutor" in source
    assert "scheduling_classes=(\"heavy\",)" in source


def test_child_compute_wiring_uses_canonical_selection_configs() -> None:
    source = inspect.getsource(build_full_job_compute_executor)

    assert "row_prefilter_top_fraction=1.0" in source
    assert "row_prefilter_min_nonzero=1" in source
    assert "combo_top_frac=1.0" in source
    assert "combo_min_confirm=1" in source


@dataclass
class _FakeFullWorker:
    def run_next(self) -> BacktestRunnerTaskResult:
        return BacktestRunnerTaskResult(task_kind="full_job", claimed=False)


@dataclass
class _FakeLazyWorker:
    def run_next(self) -> BacktestRunnerTaskResult:
        return BacktestRunnerTaskResult(task_kind="lazy_detail", claimed=False)


def _record_empty_full_probe(
    *,
    scheduler: BacktestRunnerTaskScheduler,
    scheduling_class: str,
) -> None:
    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    assert launch is not None
    assert launch.scheduling_class == scheduling_class
    scheduler.record_result(
        scheduling_class=scheduling_class,
        result=BacktestRunnerTaskResult(
            task_kind="full_job",
            claimed=False,
            scheduling_class=scheduling_class,
        ),
    )


def _scheduler(
    *,
    lazy_detail_anti_starvation_limit: int = 5,
    full_job_anti_starvation_limit: int = 4,
) -> BacktestRunnerTaskScheduler:
    return BacktestRunnerTaskScheduler(
        light_full_job_worker=cast(BacktestJobWorkerUseCase, _FakeFullWorker()),
        heavy_full_job_worker=cast(BacktestJobWorkerUseCase, _FakeFullWorker()),
        lazy_detail_worker=cast(
            BacktestLazyTradesMaterializationWorkerUseCase,
            _FakeLazyWorker(),
        ),
        light_concurrency=2,
        heavy_concurrency=1,
        lazy_detail_anti_starvation_limit=lazy_detail_anti_starvation_limit,
        full_job_anti_starvation_limit=full_job_anti_starvation_limit,
    )
