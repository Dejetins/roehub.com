from __future__ import annotations

import inspect
import os
import subprocess
import sys
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


def test_scheduler_uses_single_heavy_full_job_lane() -> None:
    scheduler = _scheduler()

    first = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    assert first is not None
    assert first.scheduling_class == "heavy"

    assert scheduler.next_launch(active_light=0, active_heavy=1, active_lazy=0) is None
    assert scheduler.next_launch(active_light=1, active_heavy=0, active_lazy=0) is None


def test_scheduler_rechecks_heavy_after_successful_full_job() -> None:
    scheduler = _scheduler()
    scheduler.record_result(
        scheduling_class="heavy",
        result=BacktestRunnerTaskResult(
            task_kind="full_job",
            claimed=True,
            scheduling_class="heavy",
        ),
    )

    launch = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)

    assert launch is not None
    assert launch.scheduling_class == "heavy"


def test_scheduler_returns_to_full_poll_after_empty_lazy_probe() -> None:
    scheduler = _scheduler()

    _record_empty_full_probe(scheduler=scheduler, scheduling_class="heavy")
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
    assert "light_full_job_worker" not in source
    assert "scheduling_classes=None" in source


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
) -> BacktestRunnerTaskScheduler:
    return BacktestRunnerTaskScheduler(
        heavy_full_job_worker=cast(BacktestJobWorkerUseCase, _FakeFullWorker()),
        lazy_detail_worker=cast(
            BacktestLazyTradesMaterializationWorkerUseCase,
            _FakeLazyWorker(),
        ),
        heavy_concurrency=1,
        lazy_detail_anti_starvation_limit=lazy_detail_anti_starvation_limit,
    )


def test_child_derivative_warmup_preserves_admitted_scoring_thread_budget(tmp_path) -> None:
    # A fresh interpreter proves both the pre-import ceiling and warmup thread mask.
    result = subprocess.run(
        [sys.executable, "-c", """
import os
import numba
from apps.worker.backtest_job_runner.wiring.modules.full_job_compute import (
    build_full_job_compute_executor,
)
executor = build_full_job_compute_executor(environ=os.environ)
assert executor.compute_policy.threads.num_threads == 2
assert numba.get_num_threads() == 2, "derivative warmup changed the admitted child budget"
assert os.environ["ROEHUB_NUMBA_NUM_THREADS"] == "1"
"""],
        env={
            **os.environ,
            "ROEHUB_ENV": "test",
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": "configs/test/backtest_artifacts.yaml",
            "ROEHUB_INDICATORS_CONFIG": "configs/test/indicators.yaml",
            "NUMBA_NUM_THREADS": "2",
            "NUMBA_THREADING_LAYER": "workqueue",
            "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS": "2",
            "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE": "test_admitted_budget",
            "ROEHUB_NUMBA_NUM_THREADS": "1",
            "ROEHUB_NUMBA_CACHE_DIR": str(tmp_path / "numba-cache"),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_parent_metadata_composition_does_not_warm_indicator_compute(monkeypatch) -> None:
    from apps.worker.backtest_job_runner.wiring.modules.lazy_trades_compute import (
        build_lazy_trades_compute_service,
    )
    from trading.contexts.indicators.adapters.outbound import NumbaIndicatorCompute

    def forbidden_warmup(self):
        raise AssertionError("metadata-only parent must not initialize compute")

    monkeypatch.setattr(NumbaIndicatorCompute, "warmup", forbidden_warmup)
    service = build_lazy_trades_compute_service(
        environ={
            "ROEHUB_ENV": "test",
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": "configs/test/backtest_artifacts.yaml",
            "ROEHUB_INDICATORS_CONFIG": "configs/test/indicators.yaml",
        },
        prepare_derivatives=False,
    )
    assert service.derivative_builder is None
    assert service.source_resolver is not None
