from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
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
from .process_observation import run_observed_subprocess

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
        scheduling_class: BacktestSchedulingClass = "heavy"
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
                scheduling_class,
                "--light-max-actual-combinations",
                str(self.light_max_actual_combinations),
            ]
            env = backtest_numba_environ(
                environ={**self.environ, "PYTHONUNBUFFERED": "1"},
                scheduling_class=scheduling_class,
            )
            log.info(
                "starting backtest child process: job_id=%s scheduling_class=%s "
                "numba_threads=%s numba_thread_source=%s",
                job_id,
                scheduling_class,
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"),
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"),
            )
            completed = run_observed_subprocess(
                cmd=cmd,
                env=env,
                timeout_seconds=self.timeout_seconds,
                evidence_prefix=f"full-job-{job_id}",
                metadata={
                    "task_kind": "full_job",
                    "job_id": str(job_id),
                    "scheduling_class": scheduling_class,
                    "child_module": self.child_module,
                    "numba_threads": env.get(
                        "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"
                    ),
                    "numba_thread_source": env.get(
                        "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"
                    ),
                },
            )
            if completed.evidence.get("timed_out"):
                raise BacktestChildProcessError(
                    f"child process timeout after {self.timeout_seconds:.0f}s"
                )
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
            _write_result_evidence(
                env=env,
                job_id=job_id,
                payload=payload,
                process_evidence=completed.evidence,
            )
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


def _write_result_evidence(
    *,
    env: Mapping[str, str],
    job_id: UUID,
    payload: Mapping[str, object],
    process_evidence: Mapping[str, object],
) -> None:
    raw_dir = env.get("ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR", "").strip()
    if not raw_dir:
        return
    evidence_dir = Path(raw_dir).expanduser()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    top_variants = payload.get("top_variants")
    top_variants_count = len(top_variants) if isinstance(top_variants, list) else 0
    raw_stage_timings = payload.get("stage_timings")
    stage_timings = dict(raw_stage_timings) if isinstance(raw_stage_timings, Mapping) else {}
    raw_cleanup_evidence = payload.get("cleanup_evidence")
    cleanup_evidence = (
        dict(raw_cleanup_evidence) if isinstance(raw_cleanup_evidence, Mapping) else {}
    )
    raw_exact_diagnostics = payload.get("exact_diagnostics")
    exact_diagnostics = (
        dict(raw_exact_diagnostics)
        if isinstance(raw_exact_diagnostics, Mapping)
        else {}
    )
    evidence = {
        "schema": "roehub_full_job_child_result_evidence_v1",
        "job_id": str(job_id),
        "status": payload.get("status"),
        "stage_timings": stage_timings,
        "summary_hash": payload.get("summary_hash"),
        "cleanup_evidence": cleanup_evidence,
        "exact_diagnostics": exact_diagnostics,
        "top_variants_count": top_variants_count,
        "process_evidence": dict(process_evidence),
    }
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = evidence_dir / f"full-job-result-{job_id}-{suffix}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
