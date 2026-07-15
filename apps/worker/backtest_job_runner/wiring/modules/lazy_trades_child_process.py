from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.use_cases.lazy_trades_materialization_worker import (
    BacktestLazyTradesMaterializationExecutionResult,
)

from .process_observation import run_observed_subprocess

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
                "--organization-id",
                str(task.organization_id),
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
            completed = run_observed_subprocess(
                cmd=cmd,
                env={**self.environ, "PYTHONUNBUFFERED": "1"},
                timeout_seconds=self.timeout_seconds,
                evidence_prefix=f"lazy-trades-{task.task_id}",
                metadata={
                    "task_kind": "lazy_trades",
                    "task_id": str(task.task_id),
                    "job_id": str(task.job_id),
                    "public_variant_key": task.public_variant_key,
                    "child_module": self.child_module,
                },
            )
            if completed.evidence.get("timed_out"):
                raise BacktestLazyTradesChildProcessError(
                    f"lazy trades child process timeout after {self.timeout_seconds:.0f}s"
                )
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
            _write_result_evidence(
                env=self.environ,
                task=task,
                payload=payload,
                process_evidence=completed.evidence,
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


def _write_result_evidence(
    *,
    env: Mapping[str, str],
    task: BacktestLazyTradesMaterializationTask,
    payload: Mapping[str, object],
    process_evidence: Mapping[str, object],
) -> None:
    raw_dir = env.get("ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR", "").strip()
    if not raw_dir:
        return
    evidence_dir = Path(raw_dir).expanduser()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "roehub_lazy_trades_child_result_evidence_v1",
        "task_id": str(task.task_id),
        "job_id": str(task.job_id),
        "public_variant_key": task.public_variant_key,
        "cache_status": payload.get("cache_status"),
        "cache_path": payload.get("cache_path"),
        "process_evidence": dict(process_evidence),
    }
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = evidence_dir / f"lazy-trades-result-{task.task_id}-{suffix}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
