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
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestPreflightResult
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestSchedulingClass,
    backtest_numba_environ,
)

from .child_ipc import (
    child_result_from_mapping,
    preflight_to_mapping,
)

log = logging.getLogger(__name__)


class BacktestChildProcessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BacktestChildProcessExecutor:
    environ: Mapping[str, str]
    scheduling_class: BacktestSchedulingClass
    light_max_actual_combinations: int
    timeout_seconds: float
    python_executable: str = sys.executable
    child_module: str = "apps.worker.backtest_job_runner.main.full_job_child"

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
    ) -> object:
        _ = updated_at
        started = datetime.now().timestamp()
        with tempfile.TemporaryDirectory(prefix="roehub-backtest-child-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            preflight_path = tmp_path / "preflight.json"
            output_path = tmp_path / "result.json"
            with preflight_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    preflight_to_mapping(preflight=preflight),
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            cmd = [
                self.python_executable,
                "-m",
                self.child_module,
                "--job-id",
                str(job_id),
                "--preflight-json",
                str(preflight_path),
                "--output-json",
                str(output_path),
                "--scheduling-class",
                self.scheduling_class,
                "--light-max-actual-combinations",
                str(self.light_max_actual_combinations),
            ]
            env = backtest_numba_environ(
                environ={**self.environ, "PYTHONUNBUFFERED": "1"},
                scheduling_class=self.scheduling_class,
            )
            log.info(
                "starting backtest child process: job_id=%s scheduling_class=%s "
                "numba_threads=%s numba_thread_source=%s",
                job_id,
                self.scheduling_class,
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"),
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"),
            )
            try:
                completed = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise BacktestChildProcessError(
                    f"child process timeout after {self.timeout_seconds:.0f}s"
                ) from error
            elapsed = datetime.now().timestamp() - started
            if completed.returncode != 0:
                stderr_tail = _bounded_tail(value=completed.stderr, limit=4000)
                raise BacktestChildProcessError(
                    "child process failed "
                    f"returncode={completed.returncode} stderr_tail={stderr_tail!r}"
                )
            if not output_path.exists():
                stdout_tail = _bounded_tail(value=completed.stdout, limit=4000)
                stderr_tail = _bounded_tail(value=completed.stderr, limit=4000)
                raise BacktestChildProcessError(
                    "child process did not write result "
                    f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
                )
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise BacktestChildProcessError("child process result must be JSON object")
            result = child_result_from_mapping(payload=payload)
            log.info(
                "backtest child process exited: job_id=%s status=%s elapsed_seconds=%.3f",
                job_id,
                payload.get("status"),
                elapsed,
            )
            return result


def _bounded_tail(*, value: str | None, limit: int) -> str:
    if value is None:
        return ""
    return value[-limit:]
