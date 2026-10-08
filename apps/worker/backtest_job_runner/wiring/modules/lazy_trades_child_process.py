from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobRepository,
)
from trading.contexts.backtest.application.ports import (
    BacktestLazyTradesMaterializationTask,
)
from trading.contexts.backtest.application.ports.backtest_job_repositories import (
    ArtifactReaderReservation,
)
from trading.contexts.backtest.application.use_cases.lazy_trades_materialization_worker import (
    BacktestLazyTradesMaterializationExecutionResult,
)

from .compute_resources import BacktestAttemptDirectory, BacktestScratchLimits, write_attempt_json
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
    job_repository: PostgresBacktestJobRepository | None = None

    def execute(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
        cancel_event: threading.Event | None = None,
    ) -> BacktestLazyTradesMaterializationExecutionResult:
        if self.job_repository is None:
            return self._execute_child(task=task, cancel_event=cancel_event)
        repo = self.job_repository
        job = repo.get(
            job_id=task.job_id, organization_id=task.organization_id, user_id=task.owner_user_id
        )
        if job is None or job.artifact_pin is None:
            raise ValueError("lazy source job or pin is missing")
        coordinates = job.request_json["coordinates"]
        from .lazy_trades_compute import build_lazy_trades_compute_service

        service = build_lazy_trades_compute_service(
            environ=self.environ, prepare_derivatives=False
        )
        metadata = service.select_replay_metadata(job=job)
        estimated = 0
        if job.input_recipe_json is not None:
            from trading.contexts.backtest.application.dto.input_recipe import BacktestInputRecipe

            recipe = BacktestInputRecipe.from_mapping(job.input_recipe_json)
            bars = next(domain.shape[0] for domain in recipe.snapshot.consumed_domains
                        if domain.role == f"prices.{recipe.snapshot.signal_timeframe}.open_time")
            selected_rows = len({row.indicator_id for row in recipe.rows})
            estimated = bars * (selected_rows + 16) + selected_rows * 8192 + 16 * 1024**2

        def validate_lease() -> None:
            if task.locked_by is None:
                raise ValueError("lazy attempt requires its exact lease owner")
            with repo.transaction():
                repo.validate_attempt_lease(
                    owner_kind="lazy",
                    organization_id=task.organization_id.value,
                    owner_id=task.task_id,
                    attempt=task.attempt,
                    locked_by=task.locked_by,
                )

        reader = None
        attempt = None
        try:
            with repo.transaction():
                if task.locked_by is None:
                    raise ValueError("lazy attempt requires its exact lease owner")
                repo.validate_attempt_lease(
                    owner_kind="lazy",
                    organization_id=task.organization_id.value,
                    owner_id=task.task_id,
                    attempt=task.attempt,
                    locked_by=task.locked_by,
                )
                reader = repo.reserve_artifact_reader(
                    reader=ArtifactReaderReservation(
                        coordinates["exchange"],
                        coordinates["market_type"],
                        coordinates["symbol"],
                        metadata.artifact_slot,
                        metadata.artifact_slot_generation,
                        metadata.artifact_manifest_hash,
                        "lazy",
                        task.organization_id.value,
                        task.task_id,
                        uuid4(),
                        task.attempt,
                        uuid4(),
                    )
                )
                # Identity revalidation is part of reader admission, before commit
                # and before the lazy subprocess can hash or mmap any payload.
                from trading.contexts.backtest.adapters.outbound import (
                    BacktestArtifactPathBuilderV2,
                    load_backtest_artifacts_runtime_config,
                    resolve_backtest_artifacts_config_path,
                )
                from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
                    FilesystemBacktestArtifactArrayLoader,
                )
                from trading.contexts.backtest.application.dto import (
                    BacktestCoordinates,
                )
                from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (  # noqa: E501
                    YamlBacktestArtifactLoaderV2,
                )

                config = load_backtest_artifacts_runtime_config(
                    Path(resolve_backtest_artifacts_config_path(environ=self.environ))
                )
                loader = FilesystemBacktestArtifactArrayLoader(
                    artifact_loader=YamlBacktestArtifactLoaderV2(
                        path_resolver=BacktestArtifactPathBuilderV2(
                            root=config.artifact_root_path()
                        )
                    )
                )
                context = loader.resolve_context(
                    coordinates=BacktestCoordinates(**coordinates),
                    artifact_metadata=metadata,
                )
                context.close_mmaps()
                attempt = BacktestAttemptDirectory.reserve(
                    root=Path(
                        self.environ.get(
                            "ROEHUB_BACKTEST_SCRATCH_ROOT",
                            str(Path(tempfile.gettempdir()) / "roehub-backtest-attempts"),
                        )
                    ),
                    owner={
                        key: None if value is None else str(value)
                        for key, value in reader.parameters().items()
                    },
                    estimated_bytes=estimated,
                    limits=BacktestScratchLimits.from_environ(self.environ),
                )
                validate_lease()
            return self._execute_child(
                task=task, cancel_event=cancel_event, attempt=attempt,
                validate_lease=validate_lease,
                source_metadata=metadata.as_mapping(),
            )
        finally:
            try:
                if attempt is not None and reader is not None and not attempt.child_reaped:
                    repo.quarantine_artifact_reader(reader=reader)
                    raise RuntimeError("unreaped lazy child retains source reservation")
                if reader is not None:
                    repo.release_artifact_reader(reader=reader)
                if attempt is not None:
                    attempt.cleanup_after_reap()
            finally:
                if attempt is not None and attempt.path.exists():
                    os.close(attempt.lock_fd)

    def _execute_child(
        self,
        *,
        task: BacktestLazyTradesMaterializationTask,
        cancel_event: threading.Event | None,
        attempt: BacktestAttemptDirectory | None = None,
        validate_lease: Callable[[], None] | None = None,
        source_metadata: Mapping[str, object] | None = None,
    ) -> BacktestLazyTradesMaterializationExecutionResult:
        started = datetime.now().timestamp()
        directory = (
            tempfile.TemporaryDirectory(prefix="roehub-lazy-trades-child-")
            if attempt is None
            else nullcontext(str(attempt.path))
        )
        with directory as tmp_dir:
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
            if attempt is not None:
                input_path = Path(tmp_dir) / "input-context.json"
                write_attempt_json(input_path, {
                    "metadata": source_metadata, "owner": attempt.owner,
                    "max_generated_bytes": max(1, attempt.generated_bytes),
                    "max_compute_bytes": int(self.environ.get(
                        "ROEHUB_BACKTEST_ATTEMPT_COMPUTE_BYTES", str(512 * 1024**2))),
                })
                cmd.extend(["--input-context-json", str(input_path)])
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
                cancel_event=cancel_event,
                pass_fds=() if attempt is None else (attempt.lock_fd,),
                output_directory=Path(tmp_dir),
                before_start=validate_lease,
                on_started=(
                    None if attempt is None else lambda: setattr(attempt, "child_reaped", False)
                ),
                on_reaped=(
                    None if attempt is None else lambda: setattr(attempt, "child_reaped", True)
                ),
                metadata={
                    "task_kind": "lazy_trades",
                    "task_id": str(task.task_id),
                    "job_id": str(task.job_id),
                    "public_variant_key": task.public_variant_key,
                    "child_module": self.child_module,
                },
            )
            if completed.evidence.get("cancelled"):
                raise BacktestLazyTradesChildProcessError("lazy child lost its lease")
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
