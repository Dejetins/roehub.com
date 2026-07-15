from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from apps.worker.job_runtime.control import DockerCommandRunner
from apps.worker.job_runtime.oci_runner import OciJobRunner, OciRuntimeError
from trading.integration.job_runtime_postgres import PostgresJobRuntimeCatalog


class JobRuntimeRecovery:
    """Own stale attempts before removing their OCI and scratch resources."""

    def __init__(
        self,
        *,
        catalog: PostgresJobRuntimeCatalog,
        runtime_root: Path,
        command_runner: DockerCommandRunner,
        docker_command: Sequence[str] = ("docker",),
        control_timeout_seconds: float = 30.0,
        environ: Mapping[str, str] | None = None,
        recovery_owner_id: str | None = None,
    ) -> None:
        root_candidate = runtime_root.expanduser()
        if root_candidate.is_symlink():
            raise ValueError("job runtime recovery root is unsafe")
        root = root_candidate.resolve()
        if root == Path("/"):
            raise ValueError("job runtime recovery root is unsafe")
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not docker_command or control_timeout_seconds <= 0:
            raise ValueError("job runtime recovery configuration is invalid")
        self._catalog = catalog
        self._runtime_root = root
        self._docker = tuple(docker_command)
        self._command_runner = command_runner
        self._control_timeout = control_timeout_seconds
        self._environ = dict(os.environ if environ is None else environ)
        self._recovery_owner_id = recovery_owner_id or f"recovery.{uuid4().hex}"

    def recover(
        self,
        *,
        now: datetime,
        worker_heartbeat_before: datetime,
        recovery_claimed_before: datetime,
    ) -> tuple[UUID, ...]:
        recovered: list[UUID] = []
        claims = self._catalog.claim_stale_for_recovery(
            now=now,
            worker_heartbeat_before=worker_heartbeat_before,
            recovery_claimed_before=recovery_claimed_before,
            recovery_owner_id=self._recovery_owner_id,
        )
        for claim in claims:
            envelope = claim.envelope
            container_name = OciJobRunner.container_name(envelope.attempt_id)
            self._remove_container(
                container_name=OciJobRunner.exporter_name(envelope.attempt_id),
                expected_attempt=str(envelope.attempt_id),
                expected_runtime="JobOutputExporter/v1",
            )
            self._remove_container(
                container_name=container_name,
                expected_attempt=str(envelope.attempt_id),
                expected_runtime="JobEnvelope/v1",
            )
            self._remove_container(
                container_name=OciJobRunner.keeper_name(envelope.attempt_id),
                expected_attempt=str(envelope.attempt_id),
                expected_runtime="JobOutputKeeper/v1",
            )
            self._remove_volume(
                OciJobRunner.output_volume_name(envelope.attempt_id),
                expected_attempt=str(envelope.attempt_id),
            )
            self._remove_scratch(envelope.attempt_id)
            self._catalog.complete_recovery(claim=claim, completed_at=now)
            recovered.append(envelope.attempt_id)
        return tuple(recovered)

    def _remove_container(
        self,
        *,
        container_name: str,
        expected_attempt: str,
        expected_runtime: str,
    ) -> None:
        inspected = self._run(
            (*self._docker, "container", "inspect", container_name),
            allow_missing=True,
        )
        if inspected.returncode != 0:
            if "No such container" in inspected.stderr:
                return
            raise OciRuntimeError(code="job.recovery_container_inspect_failed")
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as error:
            raise OciRuntimeError(code="job.recovery_inspect_invalid") from error
        if not isinstance(payload, list) or len(payload) != 1:
            raise OciRuntimeError(code="job.recovery_inspect_invalid")
        config = payload[0].get("Config") if isinstance(payload[0], dict) else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        if not isinstance(labels, dict) or (
            labels.get("io.roehub.runtime") != expected_runtime
            or labels.get("io.roehub.attempt") != expected_attempt
        ):
            raise OciRuntimeError(code="job.recovery_container_identity_mismatch")
        removed = self._run((*self._docker, "rm", "-f", container_name))
        if removed.returncode != 0:
            raise OciRuntimeError(code="job.recovery_container_remove_failed")

    def _remove_volume(self, volume_name: str, *, expected_attempt: str) -> None:
        inspected = self._run(
            (*self._docker, "volume", "inspect", volume_name),
            allow_missing=True,
        )
        if inspected.returncode != 0:
            if "No such volume" in inspected.stderr:
                return
            raise OciRuntimeError(code="job.recovery_volume_inspect_failed")
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as error:
            raise OciRuntimeError(code="job.recovery_volume_inspect_invalid") from error
        if not isinstance(payload, list) or len(payload) != 1:
            raise OciRuntimeError(code="job.recovery_volume_inspect_invalid")
        options = payload[0].get("Options") if isinstance(payload[0], dict) else None
        labels = payload[0].get("Labels") if isinstance(payload[0], dict) else None
        if (
            not isinstance(options, dict)
            or options.get("type") != "tmpfs"
            or options.get("device") != "tmpfs"
            or not isinstance(labels, dict)
            or labels.get("io.roehub.runtime") != "JobEnvelope/v1"
            or labels.get("io.roehub.attempt") != expected_attempt
        ):
            raise OciRuntimeError(code="job.recovery_volume_identity_mismatch")
        removed = self._run((*self._docker, "volume", "rm", "-f", volume_name))
        if removed.returncode != 0:
            raise OciRuntimeError(code="job.recovery_volume_remove_failed")

    def _remove_scratch(self, attempt_id: UUID) -> None:
        attempt_root = self._runtime_root / attempt_id.hex
        if attempt_root.is_symlink():
            raise OciRuntimeError(code="job.recovery_scratch_unsafe")
        if attempt_root.exists():
            shutil.rmtree(attempt_root)

    def _run(
        self, command: Sequence[str], *, allow_missing: bool = False
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._command_runner.run(
                command,
                environ=self._environ,
                timeout_seconds=self._control_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise OciRuntimeError(code="job.docker_control_timeout") from error
        except OSError as error:
            raise OciRuntimeError(code="job.docker_control_unavailable") from error
        if not allow_missing and completed.returncode != 0:
            raise OciRuntimeError(code="job.docker_command_failed")
        return completed


__all__ = ["JobRuntimeRecovery"]
