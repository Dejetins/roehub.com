from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.use_cases.lazy_trades_materialization_worker import (
    BacktestLazyTradesMaterializationExecutionResult,
)

log = logging.getLogger(__name__)


class BacktestLazyTradesChildProcessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesChildProcessExecutor:
    environ: Mapping[str, str]
    timeout_seconds: float
    python_executable: str = sys.executable
    child_module: str = "apps.worker.backtest_job_runner.main.lazy_trades_child"

    def execute(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
    ) -> BacktestLazyTradesMaterializationExecutionResult:
        started = datetime.now().timestamp()
        with tempfile.TemporaryDirectory(prefix="roehub-lazy-trades-child-") as tmp_dir:
            output_path = Path(tmp_dir) / "result.json"
            cmd = [
                self.python_executable,
                "-m",
                self.child_module,
                "--task-id",
                str(task.task_id),
                "--job-id",
                str(task.job_id),
                "--owner-user-id",
                str(task.owner_user_id),
                "--variant-key",
                task.public_variant_key,
                "--output-json",
                str(output_path),
            ]
            log.info(
                "starting lazy trades child process: task_id=%s job_id=%s",
                task.task_id,
                task.job_id,
            )
            try:
                completed = subprocess.run(
                    cmd,
                    env={**self.environ, "PYTHONUNBUFFERED": "1"},
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise BacktestLazyTradesChildProcessError(
                    f"lazy trades child process timeout after {self.timeout_seconds:.0f}s"
                ) from error
            elapsed = datetime.now().timestamp() - started
            if completed.returncode != 0:
                stderr_tail = _bounded_tail(value=completed.stderr, limit=4000)
                raise BacktestLazyTradesChildProcessError(
                    "lazy trades child process failed "
                    f"returncode={completed.returncode} stderr_tail={stderr_tail!r}"
                )
            if not output_path.exists():
                stdout_tail = _bounded_tail(value=completed.stdout, limit=4000)
                stderr_tail = _bounded_tail(value=completed.stderr, limit=4000)
                raise BacktestLazyTradesChildProcessError(
                    "lazy trades child process did not write result "
                    f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
                )
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise BacktestLazyTradesChildProcessError(
                    "lazy trades child process result must be JSON object"
                )
            log.info(
                "lazy trades child process exited: task_id=%s cache_status=%s "
                "elapsed_seconds=%.3f",
                task.task_id,
                payload.get("cache_status"),
                elapsed,
            )
            return BacktestLazyTradesMaterializationExecutionResult(
                cache_status=str(payload.get("cache_status") or "unknown"),
                cache_path=None
                if payload.get("cache_path") is None
                else str(payload.get("cache_path")),
            )


def _bounded_tail(*, value: str | None, limit: int) -> str:
    if value is None:
        return ""
    return value[-limit:]
