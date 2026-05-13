from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
    BacktestChildSuccessResult,
)
from apps.worker.backtest_job_runner.wiring.modules.child_process import (
    BacktestChildProcessError,
    BacktestChildProcessExecutor,
)
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobHeavyPromotion,
)


def test_child_process_executor_returns_bounded_success_result() -> None:
    executor = _executor(mode="success")

    result = executor.execute(
        job_id=uuid4(),
        preflight=_preflight(),
        updated_at=datetime.now(UTC),
    )

    assert isinstance(result, BacktestChildSuccessResult)
    assert len(result.top_variants) == 1
    assert result.cleanup_evidence["worker_recycle_strategy"] == (
        "disposable child process"
    )


def test_child_process_executor_maps_promotion_without_terminal_result() -> None:
    executor = _executor(mode="promote")

    result = executor.execute(
        job_id=uuid4(),
        preflight=_preflight(),
        updated_at=datetime.now(UTC),
    )

    assert isinstance(result, BacktestJobHeavyPromotion)
    assert result.actual_combinations == 100000


def test_child_process_executor_raises_bounded_error_on_child_failure() -> None:
    executor = _executor(mode="failure")

    with pytest.raises(BacktestChildProcessError) as exc_info:
        executor.execute(
            job_id=uuid4(),
            preflight=_preflight(),
            updated_at=datetime.now(UTC),
        )

    assert "child process failed" in str(exc_info.value)
    assert "returncode=7" in str(exc_info.value)


def test_child_process_executor_raises_timeout() -> None:
    executor = _executor(mode="timeout", timeout_seconds=0.01)

    with pytest.raises(BacktestChildProcessError) as exc_info:
        executor.execute(
            job_id=uuid4(),
            preflight=_preflight(),
            updated_at=datetime.now(UTC),
        )

    assert "child process timeout" in str(exc_info.value)


def _executor(*, mode: str, timeout_seconds: float = 5.0) -> BacktestChildProcessExecutor:
    return BacktestChildProcessExecutor(
        environ={**os.environ, "FAKE_CHILD_MODE": mode},
        scheduling_class="light_candidate",
        light_max_actual_combinations=50000,
        timeout_seconds=timeout_seconds,
        python_executable=sys.executable,
        child_module="tests.unit.apps.worker.backtest_job_runner.fake_child",
    )


def _preflight() -> BacktestPreflightResult:
    return BacktestPreflightResult(
        normalized_request={"risk": {"mode": "none"}, "top_n": 1},
        request_hash="d" * 64,
        result_config_hash="e" * 64,
        artifact_metadata=BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=1,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-05-13",
            hit_times_manifest_hash=None,
            published_at_utc="2026-05-13T00:00:00Z",
        ),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=1,
            candidate_combinations=1,
            tp_sl_cells=0,
            cost_class="small",
        ),
    )
