from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

import pytest

from apps.control_agent.auth import ServiceIdentityAuthorizer
from apps.control_agent.job_rpc import (
    start_job_control_server,
    stop_job_control_server,
)
from apps.control_agent.job_runtime_backend import (
    ControlAgentJobDockerRunner,
    classify_job_command,
)
from apps.worker.job_runtime.control_agent_client import ControlAgentJobUnixClient
from trading.contexts.operations import ControlOperationError


def _private(path: Path, value: str) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
    return path


class _Runner:
    def run_typed(
        self,
        *,
        operation: str,
        arguments: tuple[str, ...],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        assert operation == "container.inspect"
        assert arguments == (
            "container",
            "inspect",
            "roehub-job-" + "1" * 32,
        )
        assert timeout_seconds == 5
        return subprocess.CompletedProcess(["docker", *arguments], 1, "", "No such container")


def test_job_runtime_uses_authenticated_typed_unix_rpc(tmp_path: Path) -> None:
    api = _private(tmp_path / "api", "a" * 48)
    owner = _private(tmp_path / "owner", "b" * 48)
    job = _private(tmp_path / "job", "c" * 48)
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=api,
        owner_token_file=owner,
        job_token_file=job,
        replay_state_dir=tmp_path / "replay",
    )
    socket_root = Path(gettempdir()) / f"rj-{uuid4().hex[:8]}"
    socket_root.mkdir(mode=0o700)
    socket_path = socket_root / "job.sock"
    server, thread = start_job_control_server(
        socket_path=socket_path,
        runner=_Runner(),  # type: ignore[arg-type]
        authorizer=authorizer,
    )
    try:
        client = ControlAgentJobUnixClient(
            socket_path=socket_path,
            identity_key="c" * 48,
        )
        client.ping()
        command = (
            "docker",
            "container",
            "inspect",
            "roehub-job-" + "1" * 32,
        )
        result = client.run(command, environ={"UNSAFE_SECRET": "not-sent"}, timeout_seconds=5)
    finally:
        stop_job_control_server(
            server=server,
            thread=thread,
            socket_path=socket_path,
        )
        socket_root.rmdir()

    assert result.returncode == 1
    assert result.stderr == "No such container"


@pytest.mark.parametrize(
    "command",
    [
        ("docker", "run", "--privileged", "alpine", "sh"),
        ("docker", "run", "--env", "TOKEN=value", "alpine", "id"),
        (
            "docker",
            "run",
            "--ipc",
            "host",
            "--pull",
            "never",
            "--network",
            "none",
            "sha256:" + "1" * 64,
        ),
        ("docker", "exec", "container", "sh"),
        ("sh", "-c", "docker ps"),
    ],
)
def test_job_docker_grammar_rejects_privilege_environment_and_shell(
    command: tuple[str, ...],
) -> None:
    with pytest.raises(ControlOperationError, match="control_agent.job_command_rejected"):
        classify_job_command(command)


def test_direct_job_runner_uses_the_same_closed_grammar() -> None:
    runner = ControlAgentJobDockerRunner()
    with pytest.raises(ControlOperationError, match="control_agent.job_command_rejected"):
        runner.run(
            ("docker", "run", "--privileged", "alpine", "id"),
            environ={},
            timeout_seconds=5,
        )


def test_job_docker_grammar_distinguishes_container_command_from_control_flags() -> None:
    name = "roehub-job-" + "1" * 32
    digest = "sha256:" + "2" * 64
    command = (
        "docker",
        "create",
        "--pull",
        "never",
        "--name",
        name,
        "--label",
        "io.roehub.runtime=JobEnvelope/v1",
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
        "67108864",
        "--memory-swap",
        "67108864",
        "--pids-limit",
        "32",
        "--mount",
        "type=bind,source=/tmp/input,target=/job/input,readonly",
        digest,
        "/bin/sh",
        "-c",
        "test ! -e /var/run/docker.sock",
    )

    assert classify_job_command(command) == "container.create"

    unsafe_mount = tuple(
        "type=bind,source=/var/run/docker.sock,target=/job/input,readonly"
        if token == "type=bind,source=/tmp/input,target=/job/input,readonly"
        else token
        for token in command
    )
    with pytest.raises(ControlOperationError, match="control_agent.job_command_rejected"):
        classify_job_command(unsafe_mount)
