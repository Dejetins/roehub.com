from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.worker.job_runtime.control import DockerCommandRunner
from trading.integration import (
    MAX_JOB_OUTPUT_BYTES,
    JobEnvelope,
    JobOutcome,
    JobOutputDescriptor,
    StrategyRuntimeDecision,
    sha256_job_payload,
)

_CONTAINER_UID_GID = "65532:65532"
_INPUT_TARGET = "/job/input"
_OUTPUT_TARGET = "/job/output"
_MAX_STRATEGY_DECISIONS_BYTES = 1024 * 1024
_OUTPUT_INODE_LIMIT = 1024
_KEEPER_MEMORY_BYTES = 16 * 1024 * 1024
_KEEPER_PIDS = 8
_KEEPER_LIFETIME_SECONDS = "2147483647"
_DEFAULT_DOCKER_CONTROL_TIMEOUT_SECONDS = 30.0
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class OciRuntimeError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class OciPolicyEvidence:
    image_digest: str
    user: str
    read_only_root: bool
    network_mode: str
    memory_bytes: int
    memory_swap_bytes: int
    cpu_millis: int
    pids: int
    bind_mount_targets: tuple[str, ...]
    output_volume_name: str
    output_volume_options: str
    output_keeper_name: str
    scratch_tmpfs: str
    log_driver: str
    docker_socket_visible: bool
    secret_environment_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OciExecutionResult:
    outcome: JobOutcome
    completed_at: datetime
    exit_code: int | None
    error_code: str | None
    outputs: tuple[JobOutputDescriptor, ...]
    strategy_decisions: tuple[StrategyRuntimeDecision, ...]
    policy: OciPolicyEvidence


class OciJobRunner:
    """Run one immutable image under a fail-closed Docker/OCI policy."""

    def __init__(
        self,
        *,
        utility_image_digest: str,
        command_runner: DockerCommandRunner,
        docker_command: Sequence[str] = ("docker",),
        poll_interval_seconds: float = 0.05,
        control_timeout_seconds: float = _DEFAULT_DOCKER_CONTROL_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not docker_command:
            raise ValueError("docker command cannot be empty")
        if _IMAGE_DIGEST_PATTERN.fullmatch(utility_image_digest) is None:
            raise ValueError("utility image must be bound by sha256 digest")
        if poll_interval_seconds <= 0 or control_timeout_seconds <= 0:
            raise ValueError("runtime timeouts must be positive")
        self._docker = tuple(docker_command)
        self._command_runner = command_runner
        self._utility_image_digest = utility_image_digest
        self._poll_interval = poll_interval_seconds
        self._control_timeout = control_timeout_seconds
        self._environ = dict(os.environ if environ is None else environ)

    @staticmethod
    def container_name(attempt_id: object) -> str:
        return f"roehub-job-{str(attempt_id).replace('-', '')}"

    @staticmethod
    def output_volume_name(attempt_id: object) -> str:
        return f"roehub-job-output-{str(attempt_id).replace('-', '')}"

    @staticmethod
    def exporter_name(attempt_id: object) -> str:
        return f"roehub-job-export-{str(attempt_id).replace('-', '')}"

    @staticmethod
    def keeper_name(attempt_id: object) -> str:
        return f"roehub-job-keeper-{str(attempt_id).replace('-', '')}"

    def run(
        self,
        *,
        envelope: JobEnvelope,
        input_root: Path,
        output_root: Path,
        cancellation_requested: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> OciExecutionResult:
        input_path, output_path = self._prepare_roots(
            envelope=envelope,
            input_root=input_root,
            output_root=output_root,
        )
        container_name = self.container_name(envelope.attempt_id)
        output_volume_name = self.output_volume_name(envelope.attempt_id)
        keeper_name = self.keeper_name(envelope.attempt_id)
        outcome: JobOutcome = "crashed"
        error_code: str | None = "job.container_create_failed"
        exit_code: int | None = None
        outputs: tuple[JobOutputDescriptor, ...] = ()
        decisions: tuple[StrategyRuntimeDecision, ...] = ()
        policy: OciPolicyEvidence | None = None
        exporter_cleanup_required = True
        try:
            self._create_output_volume(
                volume_name=output_volume_name,
                max_bytes=envelope.limits.output_bytes,
                attempt_id=str(envelope.attempt_id),
            )
            self._run(
                self._create_keeper_command(
                    envelope=envelope,
                    keeper_name=keeper_name,
                    output_volume_name=output_volume_name,
                )
            )
            self._inspect_keeper_policy(
                envelope=envelope,
                keeper_name=keeper_name,
                output_volume_name=output_volume_name,
            )
            command = self._create_command(
                envelope=envelope,
                input_root=input_path,
                container_name=container_name,
                output_volume_name=output_volume_name,
            )
            self._run(command)
            policy = self._inspect_policy(
                container_name=container_name,
                envelope=envelope,
                input_root=input_path,
                output_volume_name=output_volume_name,
                keeper_name=keeper_name,
            )
            self._run((*self._docker, "start", container_name))
            started = time.monotonic()
            heartbeat_at = started
            first_poll = True
            while True:
                state = self._container_state(container_name)
                if not bool(state.get("Running")):
                    exit_code = int(state.get("ExitCode", 255))
                    if bool(state.get("OOMKilled")) or (
                        exit_code != 0
                        and self._output_volume_exhausted(
                            envelope=envelope,
                            output_volume_name=output_volume_name,
                        )
                    ):
                        outcome = "resource_exhausted"
                        error_code = "job.resource_exhausted"
                    elif exit_code == 0:
                        outcome = "succeeded"
                        error_code = None
                    else:
                        outcome = "crashed"
                        error_code = "job.container_crashed"
                    break
                now_monotonic = time.monotonic()
                if cancellation_requested is not None and cancellation_requested():
                    outcome = "canceled"
                    error_code = "job.canceled"
                    self._best_effort((*self._docker, "kill", container_name))
                    exit_code = 137
                    break
                deadline_seconds = max(
                    0.0,
                    (envelope.deadline - datetime.now(UTC)).total_seconds(),
                )
                if (
                    now_monotonic - started >= envelope.limits.wall_time_seconds
                    or deadline_seconds <= 0
                ):
                    outcome = "timed_out"
                    error_code = "job.deadline_exceeded"
                    self._best_effort((*self._docker, "kill", container_name))
                    exit_code = 137
                    break
                if heartbeat is not None and now_monotonic - heartbeat_at >= 1.0:
                    heartbeat()
                    heartbeat_at = now_monotonic
                time.sleep(min(self._poll_interval, 0.01) if first_poll else self._poll_interval)
                first_poll = False
            if outcome == "succeeded":
                if not bool(self._container_state(keeper_name).get("Running")):
                    raise OciRuntimeError(code="job.output_keeper_stopped")
                self._copy_outputs(
                    envelope=envelope,
                    output_volume_name=output_volume_name,
                    output_root=output_path,
                )
                exporter_cleanup_required = False
                outputs = self._collect_outputs(
                    root=output_path,
                    max_bytes=envelope.limits.output_bytes,
                )
                decisions = self._load_strategy_decisions(
                    root=output_path,
                    required=envelope.capability == "custom_strategy",
                )
            else:
                self._purge_output(output_path)
        except OciRuntimeError:
            raise
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise OciRuntimeError(code="job.runtime_boundary_failed") from error
        finally:
            try:
                self._cleanup_resources(
                    attempt_id=str(envelope.attempt_id),
                    exporter_name=self.exporter_name(envelope.attempt_id),
                    container_name=container_name,
                    keeper_name=keeper_name,
                    output_volume_name=output_volume_name,
                    trusted_identity=policy is not None,
                    cleanup_exporter=exporter_cleanup_required,
                )
            except OciRuntimeError as error:
                cleanup_error = self._cleanup_failure(error)
                if cleanup_error is error:
                    raise
                raise cleanup_error from error
        if policy is None:
            raise OciRuntimeError(code="job.runtime_policy_missing")
        return OciExecutionResult(
            outcome=outcome,
            completed_at=datetime.now(UTC),
            exit_code=exit_code,
            error_code=error_code,
            outputs=outputs,
            strategy_decisions=decisions,
            policy=policy,
        )

    def _prepare_roots(
        self, *, envelope: JobEnvelope, input_root: Path, output_root: Path
    ) -> tuple[Path, Path]:
        input_candidate = input_root.expanduser()
        output_candidate = output_root.expanduser()
        if input_candidate.is_symlink() or output_candidate.is_symlink():
            raise OciRuntimeError(code="job.mount_root_unsafe")
        input_path = input_candidate.resolve()
        output_path = output_candidate.resolve()
        if (
            input_path == output_path
            or input_path in output_path.parents
            or output_path in input_path.parents
        ):
            raise OciRuntimeError(code="job.mount_roots_overlap")
        input_path.mkdir(parents=True, exist_ok=True, mode=0o750)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o770)
        output_path.chmod(0o770)
        if any(output_path.iterdir()):
            raise OciRuntimeError(code="job.output_not_empty")
        for candidate in input_path.rglob("*"):
            if candidate.is_symlink() or not (candidate.is_file() or candidate.is_dir()):
                raise OciRuntimeError(code="job.input_type_unsafe")
        envelope_path = input_path / "envelope.json"
        if envelope_path.exists():
            raise OciRuntimeError(code="job.host_input_path_occupied")
        envelope_path.write_bytes(envelope.canonical_bytes() + b"\n")
        envelope_path.chmod(0o440)
        return input_path, output_path

    def _create_command(
        self,
        *,
        envelope: JobEnvelope,
        input_root: Path,
        container_name: str,
        output_volume_name: str,
    ) -> tuple[str, ...]:
        limits = envelope.limits
        return (
            *self._docker,
            "create",
            "--pull",
            "never",
            "--name",
            container_name,
            "--label",
            "io.roehub.runtime=JobEnvelope/v1",
            "--label",
            f"io.roehub.attempt={envelope.attempt_id}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--log-driver",
            "none",
            "--memory",
            str(limits.memory_bytes),
            "--memory-swap",
            str(limits.memory_bytes),
            "--cpus",
            str(limits.cpu_millis / 1000),
            "--pids-limit",
            str(limits.pids),
            "--user",
            _CONTAINER_UID_GID,
            "--workdir",
            "/tmp",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmpfs_bytes},nr_inodes=1024",
            "--mount",
            f"type=bind,source={input_root},target={_INPUT_TARGET},readonly",
            "--mount",
            f"type=volume,source={output_volume_name},target={_OUTPUT_TARGET},volume-nocopy",
            envelope.image_digest,
            *envelope.command,
        )

    def _create_keeper_command(
        self,
        *,
        envelope: JobEnvelope,
        keeper_name: str,
        output_volume_name: str,
    ) -> tuple[str, ...]:
        return (
            *self._docker,
            "run",
            "-d",
            "--pull",
            "never",
            "--name",
            keeper_name,
            "--label",
            "io.roehub.runtime=JobOutputKeeper/v1",
            "--label",
            f"io.roehub.attempt={envelope.attempt_id}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--log-driver",
            "none",
            "--memory",
            str(_KEEPER_MEMORY_BYTES),
            "--memory-swap",
            str(_KEEPER_MEMORY_BYTES),
            "--pids-limit",
            str(_KEEPER_PIDS),
            "--user",
            _CONTAINER_UID_GID,
            "--mount",
            f"type=volume,source={output_volume_name},target={_OUTPUT_TARGET},readonly,volume-nocopy",
            "--entrypoint",
            "/bin/sleep",
            self._utility_image_digest,
            _KEEPER_LIFETIME_SECONDS,
        )

    def _inspect_policy(
        self,
        *,
        container_name: str,
        envelope: JobEnvelope,
        input_root: Path,
        output_volume_name: str,
        keeper_name: str,
    ) -> OciPolicyEvidence:
        inspect = self._docker_json((*self._docker, "container", "inspect", container_name))
        if len(inspect) != 1:
            raise OciRuntimeError(code="job.container_inspect_invalid")
        payload = inspect[0]
        config = payload.get("Config")
        host = payload.get("HostConfig")
        mounts = payload.get("Mounts")
        if (
            not isinstance(config, dict)
            or not isinstance(host, dict)
            or not isinstance(mounts, list)
        ):
            raise OciRuntimeError(code="job.container_policy_missing")
        environment_names = tuple(
            sorted(
                str(item).split("=", 1)[0]
                for item in config.get("Env", [])
                if isinstance(item, str)
            )
        )
        secret_names = tuple(
            name
            for name in environment_names
            if any(
                marker in name.upper()
                for marker in ("PASSWORD", "TOKEN", "SECRET", "CREDENTIAL", "API_KEY", "DSN")
            )
        )
        bind_mounts = tuple(
            item for item in mounts if isinstance(item, dict) and item.get("Type") == "bind"
        )
        mount_targets = tuple(sorted(str(item.get("Destination")) for item in bind_mounts))
        output_mounts = tuple(
            item
            for item in mounts
            if isinstance(item, dict)
            and item.get("Type") == "volume"
            and item.get("Destination") == _OUTPUT_TARGET
        )
        docker_socket_visible = any(
            "docker.sock" in str(item.get(field, "")).lower()
            for item in mounts
            if isinstance(item, dict)
            for field in ("Source", "Destination")
        )
        raw_tmpfs = host.get("Tmpfs")
        tmpfs = cast(dict[str, Any], raw_tmpfs) if isinstance(raw_tmpfs, dict) else {}
        raw_log_config = host.get("LogConfig")
        log_config = (
            cast(dict[str, Any], raw_log_config) if isinstance(raw_log_config, dict) else {}
        )
        volume_payload = self._docker_json((*self._docker, "volume", "inspect", output_volume_name))
        if len(volume_payload) != 1:
            raise OciRuntimeError(code="job.output_volume_inspect_invalid")
        volume_options = volume_payload[0].get("Options")
        volume_labels = volume_payload[0].get("Labels")
        if not isinstance(volume_options, dict) or not isinstance(volume_labels, dict):
            raise OciRuntimeError(code="job.output_volume_policy_missing")
        options_value = str(volume_options.get("o", ""))
        evidence = OciPolicyEvidence(
            image_digest=str(payload.get("Image")),
            user=str(config.get("User")),
            read_only_root=host.get("ReadonlyRootfs") is True,
            network_mode=str(host.get("NetworkMode")),
            memory_bytes=int(host.get("Memory", 0)),
            memory_swap_bytes=int(host.get("MemorySwap", 0)),
            cpu_millis=int(host.get("NanoCpus", 0)) // 1_000_000,
            pids=int(host.get("PidsLimit", 0)),
            bind_mount_targets=mount_targets,
            output_volume_name=output_volume_name,
            output_volume_options=options_value,
            output_keeper_name=keeper_name,
            scratch_tmpfs=str(tmpfs.get("/tmp", "")),
            log_driver=str(log_config.get("Type", "")),
            docker_socket_visible=docker_socket_visible,
            secret_environment_names=secret_names,
        )
        expected = envelope.limits
        input_mount_ok = (
            len(bind_mounts) == 1
            and bind_mounts[0].get("Destination") == _INPUT_TARGET
            and bind_mounts[0].get("RW") is False
            and Path(str(bind_mounts[0].get("Source"))).resolve() == input_root
        )
        checks = {
            "image": evidence.image_digest == envelope.image_digest,
            "command": config.get("Cmd") == list(envelope.command),
            "user": evidence.user == _CONTAINER_UID_GID,
            "rootfs": evidence.read_only_root,
            "network": evidence.network_mode == "none",
            "memory": evidence.memory_bytes == expected.memory_bytes,
            "swap": evidence.memory_swap_bytes == expected.memory_bytes,
            "cpu": evidence.cpu_millis == expected.cpu_millis,
            "pids": evidence.pids == expected.pids,
            "mounts": input_mount_ok
            and evidence.bind_mount_targets == (_INPUT_TARGET,)
            and len(output_mounts) == 1
            and output_mounts[0].get("Name") == output_volume_name
            and output_mounts[0].get("RW") is True,
            "output_volume": volume_options.get("type") == "tmpfs"
            and volume_options.get("device") == "tmpfs"
            and volume_labels.get("io.roehub.runtime") == "JobEnvelope/v1"
            and volume_labels.get("io.roehub.attempt") == str(envelope.attempt_id)
            and {
                f"size={expected.output_bytes}",
                f"nr_inodes={_OUTPUT_INODE_LIMIT}",
            }.issubset(set(options_value.split(","))),
            "scratch_tmpfs": self._tmpfs_matches(
                evidence.scratch_tmpfs,
                size=expected.tmpfs_bytes,
                inodes=1024,
            ),
            "logs": evidence.log_driver == "none",
            "socket": not evidence.docker_socket_visible,
            "environment": not evidence.secret_environment_names,
            "capabilities": "ALL" in (host.get("CapDrop") or []),
            "privileges": "no-new-privileges" in (host.get("SecurityOpt") or []),
        }
        failed = next((name for name, passed in checks.items() if not passed), None)
        if failed is not None:
            raise OciRuntimeError(code=f"job.container_policy_{failed}_mismatch")
        return evidence

    def _inspect_keeper_policy(
        self,
        *,
        envelope: JobEnvelope,
        keeper_name: str,
        output_volume_name: str,
    ) -> None:
        inspect = self._docker_json((*self._docker, "container", "inspect", keeper_name))
        if len(inspect) != 1:
            raise OciRuntimeError(code="job.output_keeper_inspect_invalid")
        payload = inspect[0]
        config = payload.get("Config")
        host = payload.get("HostConfig")
        mounts = payload.get("Mounts")
        state = payload.get("State")
        if (
            not isinstance(config, dict)
            or not isinstance(host, dict)
            or not isinstance(mounts, list)
            or not isinstance(state, dict)
        ):
            raise OciRuntimeError(code="job.output_keeper_policy_missing")
        labels = config.get("Labels")
        output_mounts = tuple(
            item
            for item in mounts
            if isinstance(item, dict)
            and item.get("Type") == "volume"
            and item.get("Destination") == _OUTPUT_TARGET
        )
        raw_log_config = host.get("LogConfig")
        log_config = raw_log_config if isinstance(raw_log_config, dict) else {}
        checks = {
            "identity": isinstance(labels, dict)
            and labels.get("io.roehub.runtime") == "JobOutputKeeper/v1"
            and labels.get("io.roehub.attempt") == str(envelope.attempt_id),
            "image": payload.get("Image") == self._utility_image_digest,
            "command": config.get("Entrypoint") == ["/bin/sleep"]
            and config.get("Cmd") == [_KEEPER_LIFETIME_SECONDS],
            "running": state.get("Running") is True,
            "user": config.get("User") == _CONTAINER_UID_GID,
            "rootfs": host.get("ReadonlyRootfs") is True,
            "network": host.get("NetworkMode") == "none",
            "memory": host.get("Memory") == _KEEPER_MEMORY_BYTES
            and host.get("MemorySwap") == _KEEPER_MEMORY_BYTES,
            "pids": host.get("PidsLimit") == _KEEPER_PIDS,
            "mount": len(output_mounts) == 1
            and output_mounts[0].get("Name") == output_volume_name
            and output_mounts[0].get("RW") is False,
            "logs": log_config.get("Type") == "none",
            "capabilities": "ALL" in (host.get("CapDrop") or []),
            "privileges": "no-new-privileges" in (host.get("SecurityOpt") or []),
        }
        failed = next((name for name, passed in checks.items() if not passed), None)
        if failed is not None:
            raise OciRuntimeError(code=f"job.output_keeper_policy_{failed}_mismatch")

    @staticmethod
    def _tmpfs_matches(options: str, *, size: int, inodes: int) -> bool:
        values = set(options.split(","))
        return {
            "rw",
            "noexec",
            "nosuid",
            "nodev",
            f"size={size}",
            f"nr_inodes={inodes}",
        }.issubset(values)

    def _container_state(self, container_name: str) -> dict[str, Any]:
        inspect = self._docker_json((*self._docker, "container", "inspect", container_name))
        if len(inspect) != 1 or not isinstance(inspect[0].get("State"), dict):
            raise OciRuntimeError(code="job.container_state_invalid")
        return cast(dict[str, Any], inspect[0]["State"])

    def _output_volume_exhausted(self, *, envelope: JobEnvelope, output_volume_name: str) -> bool:
        completed = self._run(
            (
                *self._docker,
                "run",
                "--rm",
                "--name",
                self.exporter_name(envelope.attempt_id),
                "--label",
                "io.roehub.runtime=JobOutputExporter/v1",
                "--label",
                f"io.roehub.attempt={envelope.attempt_id}",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--log-driver",
                "none",
                "--memory",
                str(32 * 1024 * 1024),
                "--memory-swap",
                str(32 * 1024 * 1024),
                "--pids-limit",
                "16",
                "--mount",
                f"type=volume,source={output_volume_name},target=/source,readonly,volume-nocopy",
                "--entrypoint",
                "/bin/df",
                self._utility_image_digest,
                "-Pk",
                "/source",
            )
        )
        lines = completed.stdout.splitlines()
        if len(lines) != 2:
            raise OciRuntimeError(code="job.output_volume_usage_invalid")
        columns = lines[1].split()
        if len(columns) < 6 or not columns[3].isdigit():
            raise OciRuntimeError(code="job.output_volume_usage_invalid")
        return int(columns[3]) == 0

    def _create_output_volume(self, *, volume_name: str, max_bytes: int, attempt_id: str) -> None:
        self._run(
            (
                *self._docker,
                "volume",
                "create",
                "--driver",
                "local",
                "--opt",
                "type=tmpfs",
                "--opt",
                "device=tmpfs",
                "--opt",
                f"o=size={max_bytes},nr_inodes={_OUTPUT_INODE_LIMIT}",
                "--label",
                "io.roehub.runtime=JobEnvelope/v1",
                "--label",
                f"io.roehub.attempt={attempt_id}",
                volume_name,
            )
        )

    def _copy_outputs(
        self,
        *,
        envelope: JobEnvelope,
        output_volume_name: str,
        output_root: Path,
    ) -> None:
        self._run(
            (
                *self._docker,
                "run",
                "--rm",
                "--name",
                self.exporter_name(envelope.attempt_id),
                "--label",
                "io.roehub.runtime=JobOutputExporter/v1",
                "--label",
                f"io.roehub.attempt={envelope.attempt_id}",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--log-driver",
                "none",
                "--memory",
                str(64 * 1024 * 1024),
                "--memory-swap",
                str(64 * 1024 * 1024),
                "--pids-limit",
                "32",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=volume,source={output_volume_name},target=/source,readonly,volume-nocopy",
                "--mount",
                f"type=bind,source={output_root},target=/destination",
                "--entrypoint",
                "/bin/cp",
                self._utility_image_digest,
                "-a",
                "/source/.",
                "/destination/",
            )
        )

    def _collect_outputs(self, *, root: Path, max_bytes: int) -> tuple[JobOutputDescriptor, ...]:
        outputs: list[JobOutputDescriptor] = []
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise OciRuntimeError(code="job.output_type_unsafe")
            if path.is_dir():
                continue
            relative = path.relative_to(root).as_posix()
            if not path.is_file():
                raise OciRuntimeError(code="job.output_type_unsafe")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                file_stat = os.fstat(stream.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise OciRuntimeError(code="job.output_type_unsafe")
                total += file_stat.st_size
                if total > max_bytes or total > MAX_JOB_OUTPUT_BYTES:
                    raise OciRuntimeError(code="job.output_limit_exceeded")
                payload = stream.read(file_stat.st_size + 1)
                if len(payload) != file_stat.st_size:
                    raise OciRuntimeError(code="job.output_changed_during_read")
            if stat.S_IMODE(file_stat.st_mode) & 0o111:
                raise OciRuntimeError(code="job.output_executable_forbidden")
            media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            outputs.append(
                JobOutputDescriptor(
                    path=relative,
                    digest=sha256_job_payload(payload),
                    size_bytes=len(payload),
                    media_type=media_type,
                )
            )
            if len(outputs) > 256:
                raise OciRuntimeError(code="job.output_file_limit_exceeded")
        return tuple(outputs)

    @staticmethod
    def _load_strategy_decisions(
        *, root: Path, required: bool
    ) -> tuple[StrategyRuntimeDecision, ...]:
        path = root / "strategy-decisions.json"
        if not path.exists():
            if required:
                raise OciRuntimeError(code="job.strategy_decisions_missing")
            return ()
        if path.stat().st_size > _MAX_STRATEGY_DECISIONS_BYTES:
            raise OciRuntimeError(code="job.strategy_decisions_too_large")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise OciRuntimeError(code="job.strategy_decisions_invalid")
        try:
            return tuple(StrategyRuntimeDecision.model_validate(item) for item in payload)
        except ValueError as error:
            raise OciRuntimeError(code="job.strategy_decisions_invalid") from error

    @staticmethod
    def _purge_output(root: Path) -> None:
        for path in tuple(root.iterdir()):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _run_allow_failure(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self._command_runner.run(
                command,
                environ=self._environ,
                timeout_seconds=self._control_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise OciRuntimeError(code="job.docker_control_timeout") from error
        except OSError as error:
            raise OciRuntimeError(code="job.docker_control_unavailable") from error

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        completed = self._run_allow_failure(command)
        if completed.returncode != 0:
            raise OciRuntimeError(code="job.docker_command_failed")
        return completed

    def _docker_json(self, command: Sequence[str]) -> list[dict[str, Any]]:
        completed = self._run(command)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise OciRuntimeError(code="job.docker_response_invalid")
        return cast(list[dict[str, Any]], payload)

    def _best_effort(self, command: Sequence[str]) -> None:
        try:
            self._command_runner.run(
                command,
                environ=self._environ,
                timeout_seconds=self._control_timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _cleanup_resources(
        self,
        *,
        attempt_id: str,
        exporter_name: str,
        container_name: str,
        keeper_name: str,
        output_volume_name: str,
        trusted_identity: bool,
        cleanup_exporter: bool,
    ) -> None:
        first_error: OciRuntimeError | None = None
        removals: list[Callable[[], None]] = []
        if cleanup_exporter:
            removals.append(
                lambda: self._remove_container_if_present(
                    exporter_name,
                    attempt_id=attempt_id,
                    runtime="JobOutputExporter/v1",
                )
            )
        if trusted_identity:
            removals.append(lambda: self._remove_trusted_containers((container_name, keeper_name)))
        else:
            removals.extend(
                (
                    lambda: self._remove_container_if_present(
                        container_name,
                        attempt_id=attempt_id,
                        runtime="JobEnvelope/v1",
                    ),
                    lambda: self._remove_container_if_present(
                        keeper_name,
                        attempt_id=attempt_id,
                        runtime="JobOutputKeeper/v1",
                    ),
                )
            )
        removals.append(
            lambda: self._remove_volume_if_present(
                output_volume_name,
                attempt_id=attempt_id,
                trusted_identity=trusted_identity,
            )
        )
        for remove in removals:
            try:
                remove()
            except OciRuntimeError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    @staticmethod
    def _cleanup_failure(error: OciRuntimeError) -> OciRuntimeError:
        if error.code in {
            "job.container_cleanup_failed",
            "job.container_cleanup_identity_mismatch",
            "job.output_volume_cleanup_failed",
            "job.output_volume_cleanup_identity_mismatch",
        }:
            return error
        return OciRuntimeError(code="job.cleanup_boundary_failed")

    def _remove_trusted_containers(self, container_names: Sequence[str]) -> None:
        completed = self._run_allow_failure((*self._docker, "rm", "-f", *container_names))
        if completed.returncode != 0:
            raise OciRuntimeError(code="job.container_cleanup_failed")

    def _remove_container_if_present(
        self,
        container_name: str,
        *,
        attempt_id: str,
        runtime: str,
        trusted_identity: bool = False,
    ) -> None:
        if trusted_identity:
            completed = self._run_allow_failure((*self._docker, "rm", "-f", container_name))
            if completed.returncode == 0 or "No such container" in completed.stderr:
                return
            raise OciRuntimeError(code="job.container_cleanup_failed")
        inspected = self._run_allow_failure((*self._docker, "container", "inspect", container_name))
        if inspected.returncode != 0 and "No such container" in inspected.stderr:
            return
        if inspected.returncode != 0:
            raise OciRuntimeError(code="job.container_cleanup_failed")
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as error:
            raise OciRuntimeError(code="job.container_cleanup_failed") from error
        config = payload[0].get("Config") if len(payload) == 1 else None
        labels = config.get("Labels") if isinstance(config, dict) else None
        if not isinstance(labels, dict) or (
            labels.get("io.roehub.runtime") != runtime
            or labels.get("io.roehub.attempt") != attempt_id
        ):
            raise OciRuntimeError(code="job.container_cleanup_identity_mismatch")
        completed = self._run_allow_failure((*self._docker, "rm", "-f", container_name))
        if completed.returncode != 0:
            raise OciRuntimeError(code="job.container_cleanup_failed")

    def _remove_volume_if_present(
        self,
        volume_name: str,
        *,
        attempt_id: str,
        trusted_identity: bool,
    ) -> None:
        if trusted_identity:
            completed = self._run_allow_failure((*self._docker, "volume", "rm", "-f", volume_name))
            if completed.returncode == 0 or "No such volume" in completed.stderr:
                return
            raise OciRuntimeError(code="job.output_volume_cleanup_failed")
        inspected = self._run_allow_failure((*self._docker, "volume", "inspect", volume_name))
        if inspected.returncode != 0 and "No such volume" in inspected.stderr:
            return
        if inspected.returncode != 0:
            raise OciRuntimeError(code="job.output_volume_cleanup_failed")
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as error:
            raise OciRuntimeError(code="job.output_volume_cleanup_failed") from error
        labels = payload[0].get("Labels") if len(payload) == 1 else None
        if not isinstance(labels, dict) or (
            labels.get("io.roehub.runtime") != "JobEnvelope/v1"
            or labels.get("io.roehub.attempt") != attempt_id
        ):
            raise OciRuntimeError(code="job.output_volume_cleanup_identity_mismatch")
        completed = self._run_allow_failure((*self._docker, "volume", "rm", "-f", volume_name))
        if completed.returncode != 0:
            raise OciRuntimeError(code="job.output_volume_cleanup_failed")


__all__ = [
    "OciExecutionResult",
    "OciJobRunner",
    "OciPolicyEvidence",
    "OciRuntimeError",
]
