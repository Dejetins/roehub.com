from __future__ import annotations

import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping
from uuid import UUID, uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobRepository,
)
from trading.contexts.backtest.application.dto import BacktestPreflightResult
from trading.contexts.backtest.application.ports.backtest_job_repositories import (
    ArtifactReaderReservation,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestSchedulingClass,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobCancellationRequested,
)
from trading.contexts.backtest.domain.entities import BacktestJob
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    BacktestPreparedArtifactSet,
)

from .child_ipc import (
    BacktestChildSuccessResult,
    PreparedInputEvent,
    child_result_from_mapping,
    preflight_to_mapping,
)
from .compute_resources import (
    BacktestAttemptDirectory,
    BacktestScratchLimits,
    discover_cpu_capacity,
    full_job_resource_environ,
    write_attempt_json,
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
    job_repository: PostgresBacktestJobRepository | None = None

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        cancel_event: threading.Event | None = None,
        job: BacktestJob | None = None,
        locked_by: str | None = None,
    ) -> object:
        started = time.perf_counter()
        if self.job_repository is None:
            if preflight.input_recipe is not None:
                raise ValueError("recipe child requires durable ownership repository")
            return self._execute_child(
                job_id=job_id, preflight=preflight, cancel_event=cancel_event
            )
        if job is None or job.job_id != job_id or locked_by is None:
            raise ValueError("durable child requires exact claimed job and lease owner")
        repo = self.job_repository
        owner_token, parent_incarnation = uuid4(), uuid4()

        def validate_lease() -> None:
            with repo.transaction():
                repo.validate_attempt_lease(
                    owner_kind="job",
                    organization_id=job.organization_id.value,
                    owner_id=job_id,
                    attempt=job.attempt,
                    locked_by=locked_by,
                )

        attempt = None
        reader = None
        try:
            with repo.transaction():
                repo.validate_attempt_lease(
                    owner_kind="job",
                    organization_id=job.organization_id.value,
                    owner_id=job_id,
                    attempt=job.attempt,
                    locked_by=locked_by,
                )
                previous = repo.get_artifact_reader(
                    organization_id=job.organization_id.value,
                    owner_id=job_id,
                    owner_kind="job",
                )
                if previous is None:
                    coordinates = preflight.normalized_request["coordinates"]
                    pin = preflight.artifact_metadata
                    reader = repo.reserve_artifact_reader(
                        reader=ArtifactReaderReservation(
                            coordinates["exchange"],
                            coordinates["market_type"],
                            coordinates["symbol"],
                            pin.artifact_slot,
                            pin.artifact_slot_generation,
                            pin.artifact_manifest_hash,
                            "job",
                            job.organization_id.value,
                            job_id,
                            owner_token,
                            job.attempt,
                            parent_incarnation,
                        )
                    )
                else:
                    reader = repo.transfer_artifact_reader(
                        previous=previous,
                        attempt=job.attempt,
                        parent_incarnation=parent_incarnation,
                        owner_token=owner_token,
                        locked_by=locked_by,
                    )
                expected = preflight.artifact_metadata
                if (reader.slot, reader.generation, reader.manifest_sha256) != (
                    expected.artifact_slot,
                    expected.artifact_slot_generation,
                    expected.artifact_manifest_hash,
                ) or (
                    job.artifact_pin is not None
                    and (
                        job.artifact_pin.artifact_slot,
                        job.artifact_pin.artifact_slot_generation,
                        job.artifact_pin.artifact_manifest_hash,
                    )
                    != (reader.slot, reader.generation, reader.manifest_sha256)
                ):
                    raise ValueError("attempt preflight differs from its durable source pin")
                limits = BacktestScratchLimits.from_environ(self.environ)
                estimated = _estimated_attempt_bytes(preflight=preflight, environ=self.environ)
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
                    limits=limits,
                )

                validate_lease()

            def acknowledge(prepared: BacktestPreparedArtifactSet) -> str:
                if cancel_event is not None and cancel_event.is_set():
                    raise BacktestJobCancellationRequested(
                        "cancelled before prepared acknowledgement"
                    )
                return repo.acknowledge_prepared_inputs(
                    job=job,
                    reader=reader,
                    locked_by=locked_by,
                    prepared=prepared,
                )

            result = self._execute_child(
                job_id=job_id,
                preflight=preflight,
                cancel_event=cancel_event,
                attempt=attempt,
                acknowledge=acknowledge,
                validate_lease=validate_lease,
            )
        finally:
            cleanup_started = time.perf_counter()
            try:
                if attempt is not None and reader is not None and not attempt.child_reaped:
                    repo.quarantine_artifact_reader(reader=reader)
                    raise RuntimeError("unreaped child retains its input reservation")
                # Keep the lifetime marker if DB loss makes release uncertain.
                if reader is not None:
                    repo.release_artifact_reader(reader=reader)
                if attempt is not None:
                    attempt.cleanup_after_reap()
            finally:
                if attempt is not None and attempt.path.exists():
                    os.close(attempt.lock_fd)
            cleanup_elapsed = time.perf_counter() - cleanup_started
        if isinstance(result, BacktestChildSuccessResult):
            result = replace(
                result,
                stage_timings={
                    **result.stage_timings,
                    "parent_attempt_cleanup": cleanup_elapsed,
                    "parent_attempt_including_cleanup": time.perf_counter() - started,
                },
            )
        return result

    def _execute_child(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        cancel_event: threading.Event | None,
        attempt: BacktestAttemptDirectory | None = None,
        acknowledge: Callable[[BacktestPreparedArtifactSet], str] | None = None,
        validate_lease: Callable[[], None] | None = None,
    ) -> object:
        scheduling_class: BacktestSchedulingClass = "heavy"
        started = datetime.now().timestamp()
        directory = (
            tempfile.TemporaryDirectory(prefix="roehub-backtest-child-")
            if attempt is None
            else nullcontext(str(attempt.path))
        )
        with directory as tmp_dir:
            tmp_path = Path(tmp_dir)
            preflight_path = tmp_path / "preflight.json"
            output_path = tmp_path / "result.json"
            write_attempt_json(preflight_path, preflight_to_mapping(preflight=preflight))
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
            capacity = discover_cpu_capacity(self.environ)
            env = full_job_resource_environ(
                capacity=capacity,
                environ={**self.environ, "PYTHONUNBUFFERED": "1"},
            )
            log.info(
                "starting backtest child process: job_id=%s scheduling_class=%s "
                "numba_threads=%s numba_thread_source=%s",
                job_id,
                scheduling_class,
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"),
                env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"),
            )
            parent_channel, child_channel = socket.socketpair()
            callback = _PreparedInputReceiver(parent_channel, acknowledge)
            fds = ()
            if attempt is not None:
                fds = (attempt.lock_fd, child_channel.fileno())
                ownership_path = tmp_path / "attempt.json"
                write_attempt_json(
                    ownership_path,
                    {
                        **dict(attempt.owner),
                        "max_generated_bytes": max(1, attempt.generated_bytes),
                        "max_compute_bytes": int(
                            self.environ.get(
                                "ROEHUB_BACKTEST_ATTEMPT_COMPUTE_BYTES", str(512 * 1024**2)
                            )
                        ),
                    },
                )
                cmd += [
                    "--attempt-json",
                    str(ownership_path),
                    "--prepared-fd",
                    str(child_channel.fileno()),
                ]
            try:
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
                        "cpu_capacity": asdict(capacity),
                        "numba_threads": env.get("ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"),
                        "numba_thread_source": env.get(
                            "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"
                        ),
                    },
                    cancel_event=cancel_event,
                    pass_fds=fds,
                    output_directory=tmp_path,
                    before_start=validate_lease,
                    on_poll=callback.poll if attempt is not None else None,
                    on_started=(
                        None if attempt is None else lambda: setattr(attempt, "child_reaped", False)
                    ),
                    on_reaped=(
                        None if attempt is None else lambda: setattr(attempt, "child_reaped", True)
                    ),
                )
            finally:
                parent_channel.close()
                child_channel.close()
            if completed.evidence.get("cancelled"):
                raise BacktestJobCancellationRequested(
                    f"child process cancelled for job_id={job_id}"
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
    raw_instrumentation_counters = payload.get("instrumentation_counters")
    instrumentation_counters = (
        dict(raw_instrumentation_counters)
        if isinstance(raw_instrumentation_counters, Mapping)
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
        "instrumentation_counters": instrumentation_counters,
        "top_variants_count": top_variants_count,
        "process_evidence": dict(process_evidence),
    }
    # Opt-in benchmark evidence is capped independently of requested job size.
    # Normal child diagnostics remain a five-row sample.
    if env.get("ROEHUB_BACKTEST_BENCHMARK_FULL_TOP") == "1":
        rows = top_variants if isinstance(top_variants, list) else []
        evidence["benchmark_full_top"] = {
            "count": len(rows),
            "complete": isinstance(top_variants, list) and len(rows) <= 50,
            "items": [
                {
                    "rank": row["rank"], "variant_hash": row["variant_key"],
                    "canonical_variant_params": row["payload_json"]["canonical_variant_params"],
                    "summary_metrics": row["summary_metrics_json"],
                    "best_tp_pct": row["best_tp_pct"], "best_sl_pct": row["best_sl_pct"],
                }
                for row in rows[:50]
            ],
        }
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = evidence_dir / f"full-job-result-{job_id}-{suffix}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _PreparedInputReceiver:
    def __init__(
        self,
        channel: socket.socket,
        acknowledge: Callable[[BacktestPreparedArtifactSet], str] | None,
    ) -> None:
        self.channel = channel
        channel.setblocking(False)
        self.acknowledge = acknowledge
        self.pending = bytearray()
        self.acknowledged = False

    def poll(self) -> None:
        if self.acknowledged:
            return
        try:
            data = self.channel.recv(65536)
        except BlockingIOError:
            return
        if not data:
            return
        self.pending.extend(data)
        if len(self.pending) > 16 * 1024**2:
            raise ValueError("prepared input event exceeds bounded IPC size")
        if b"\n" not in self.pending:
            return
        if not self.pending.endswith(b"\n") or self.pending.count(b"\n") != 1:
            raise ValueError("invalid prepared input event framing")
        event = PreparedInputEvent.from_mapping(json.loads(self.pending))
        if self.acknowledge is None:
            raise ValueError("unsolicited prepared input event")
        digest = self.acknowledge(event.prepared)
        if digest != event.prepared.content_sha256:
            raise ValueError("durable acknowledgement differs from prepared input")
        self.channel.sendall(digest.encode() + b"\n")
        self.acknowledged = True
        self.pending.clear()


def _estimated_attempt_bytes(
    *,
    preflight: BacktestPreflightResult,
    environ: Mapping[str, str],
) -> int:
    """Bound only absent derivatives using manifest metadata; no candle payload reads."""
    from trading.contexts.backtest.adapters.outbound import (
        BacktestArtifactPathBuilderV2,
        load_backtest_artifacts_runtime_config,
        resolve_backtest_artifacts_config_path,
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
        FilesystemBacktestArtifactArrayLoader,
    )
    from trading.contexts.backtest.application.dto import BacktestCoordinates
    from trading.contexts.backtest.application.services.v2.prepare_pools import _risk_level_matches
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (  # noqa: E501
        YamlBacktestArtifactLoaderV2,
    )

    recipe = preflight.input_recipe
    config = load_backtest_artifacts_runtime_config(
        Path(resolve_backtest_artifacts_config_path(environ=environ))
    )
    loader = FilesystemBacktestArtifactArrayLoader(
        artifact_loader=YamlBacktestArtifactLoaderV2(
            path_resolver=BacktestArtifactPathBuilderV2(root=config.artifact_root_path())
        )
    )
    context = loader.resolve_context(
        coordinates=BacktestCoordinates(**preflight.normalized_request["coordinates"]),
        artifact_metadata=preflight.artifact_metadata,
    )
    if recipe is None:
        return 0
    timeframe = recipe.snapshot.signal_timeframe
    refs = tuple(ref for ref in context.references if ref.domain.timeframe == timeframe)
    rows = {
        (ref.domain.role.removeprefix("signals."), row)
        for ref in refs
        if ref.domain.role.startswith("signals.")
        for row in ref.row_ids
    }
    missing_rows = sum((row.indicator_id, row.row_id) not in rows for row in recipe.rows)
    bars = next(
        domain.shape[0]
        for domain in recipe.snapshot.consumed_domains
        if domain.role == f"prices.{timeframe}.open_time"
    )
    missing_levels = 0
    for family, levels in (("tp", recipe.tp_levels_pct), ("sl", recipe.sl_levels_pct)):
        for level in levels:
            if not all(
                any(
                    _risk_level_matches(level, value)
                    for ref in refs
                    if ref.domain.role == f"hit_times.{suffix}"
                    for value in ref.risk_values
                )
                for suffix in (f"{family}_values", f"long_{family}", f"short_{family}")
            ):
                missing_levels += 1
    if not missing_rows and not missing_levels:
        return 0
    return missing_rows * (bars + 8192) + missing_levels * (8 * bars + 8192) + 16 * 1024**2
