"""Closed Docker grammar and execution owned by the control-agent process."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from trading.contexts.operations import ControlOperationError

JobDockerOperation = Literal[
    "container.create",
    "container.inspect",
    "container.kill",
    "container.remove",
    "container.start",
    "container.run",
    "volume.create",
    "volume.inspect",
    "volume.remove",
]

_JOB_NAME = re.compile(r"^roehub-job-(?:output-|export-|keeper-)?[0-9a-f]{32}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN = {
    "--add-host",
    "--cap-add",
    "--device",
    "--env",
    "-e",
    "--env-file",
    "--privileged",
    "--pid",
    "--uts",
    "--volume",
    "-v",
}
_SAFE_ENVIRONMENT = ("DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST", "PATH")
_CONTAINER_FLAGS = {
    "-d": 0,
    "--rm": 0,
    "--read-only": 0,
    "--pull": 1,
    "--name": 1,
    "--label": 1,
    "--network": 1,
    "--cap-drop": 1,
    "--security-opt": 1,
    "--log-driver": 1,
    "--memory": 1,
    "--memory-swap": 1,
    "--cpus": 1,
    "--pids-limit": 1,
    "--user": 1,
    "--workdir": 1,
    "--tmpfs": 1,
    "--mount": 1,
    "--entrypoint": 1,
}


def classify_job_command(command: Sequence[str]) -> JobDockerOperation:
    """Classify and validate the exact OCI-job Docker grammar before transport."""

    if len(command) < 3 or Path(command[0]).name != "docker":
        raise ControlOperationError(code="control_agent.job_command_rejected")
    tokens = tuple(command[1:])
    if len(tokens) > 160 or any(
        not isinstance(token, str)
        or not token
        or len(token) > 4096
        or "\x00" in token
        for token in tokens
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    docker_control_tokens = tokens
    if tokens[0] in {"create", "run"}:
        image_index = next(
            (
                index
                for index, token in enumerate(tokens)
                if _IMAGE_DIGEST.fullmatch(token)
            ),
            len(tokens),
        )
        docker_control_tokens = tokens[:image_index]
    if any(token in _FORBIDDEN for token in tokens) or any(
        "docker.sock" in token.lower() for token in docker_control_tokens
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")

    if tokens[:2] == ("container", "inspect"):
        operation: JobDockerOperation = "container.inspect"
        names = tokens[2:]
    elif tokens[:2] == ("volume", "inspect"):
        operation = "volume.inspect"
        names = tokens[2:]
    elif tokens[:3] == ("volume", "rm", "-f"):
        operation = "volume.remove"
        names = tokens[3:]
    elif tokens[:2] == ("volume", "create"):
        operation = "volume.create"
        names = (tokens[-1],)
        _validate_volume_create(tokens)
        _require_tokens(
            tokens,
            (
                "--driver",
                "local",
                "type=tmpfs",
                "device=tmpfs",
                "io.roehub.runtime=JobEnvelope/v1",
                "--label",
            ),
        )
        if not any(token.startswith("o=size=") and "nr_inodes=1024" in token for token in tokens):
            raise ControlOperationError(code="control_agent.job_command_rejected")
    elif tokens[0] == "create":
        operation = "container.create"
        names = (_option_value(tokens, "--name"),)
        _validate_container_flag_grammar(tokens)
        _validate_hardened_container(tokens, require_output_bind=True)
    elif tokens[0] == "run":
        operation = "container.run"
        names = (_option_value(tokens, "--name"),)
        _validate_container_flag_grammar(tokens)
        _validate_hardened_container(tokens, require_output_bind=False)
    elif tokens[0] == "start":
        operation = "container.start"
        names = tokens[1:]
    elif tokens[0] == "kill":
        operation = "container.kill"
        names = tokens[1:]
    elif tokens[:2] == ("rm", "-f"):
        operation = "container.remove"
        names = tokens[2:]
    else:
        raise ControlOperationError(code="control_agent.job_command_rejected")
    if not names or any(_JOB_NAME.fullmatch(name) is None for name in names):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    return operation


def _validate_container_flag_grammar(tokens: Sequence[str]) -> None:
    try:
        image_index = next(
            index for index, token in enumerate(tokens) if _IMAGE_DIGEST.fullmatch(token)
        )
    except StopIteration as error:
        raise ControlOperationError(code="control_agent.job_command_rejected") from error
    index = 1
    values: dict[str, list[str]] = {}
    while index < image_index:
        flag = tokens[index]
        arity = _CONTAINER_FLAGS.get(flag)
        if arity is None or index + arity >= image_index:
            raise ControlOperationError(code="control_agent.job_command_rejected")
        if arity:
            values.setdefault(flag, []).append(tokens[index + 1])
        index += arity + 1
    exact_values = {
        "--pull": "never",
        "--network": "none",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--log-driver": "none",
    }
    if any(
        any(value != expected for value in values.get(flag, ()))
        for flag, expected in exact_values.items()
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    if any(value != "/tmp" for value in values.get("--workdir", ())):
        raise ControlOperationError(code="control_agent.job_command_rejected")


def _validate_volume_create(tokens: Sequence[str]) -> None:
    allowed = {"--driver", "--opt", "--label"}
    index = 2
    values: dict[str, list[str]] = {}
    while index < len(tokens) - 1:
        flag = tokens[index]
        if flag not in allowed or index + 1 >= len(tokens) - 1:
            raise ControlOperationError(code="control_agent.job_command_rejected")
        values.setdefault(flag, []).append(tokens[index + 1])
        index += 2
    if values.get("--driver") != ["local"]:
        raise ControlOperationError(code="control_agent.job_command_rejected")
    options = set(values.get("--opt", ()))
    if not {"type=tmpfs", "device=tmpfs"}.issubset(options) or not any(
        option.startswith("o=size=") and option.endswith(",nr_inodes=1024")
        for option in options
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")


def _validate_hardened_container(
    tokens: Sequence[str],
    *,
    require_output_bind: bool,
) -> None:
    _require_tokens(
        tokens,
        (
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
            "--memory-swap",
            "--pids-limit",
            "--mount",
        ),
    )
    if not any(_IMAGE_DIGEST.fullmatch(token) for token in tokens):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    mounts = [
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == "--mount"
    ]
    if any(
        "type=bind" in mount and "readonly" not in mount and "target=/destination" not in mount
        for mount in mounts
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    if require_output_bind and not any(
        "type=bind" in mount and "target=/job/input" in mount and "readonly" in mount
        for mount in mounts
    ):
        raise ControlOperationError(code="control_agent.job_command_rejected")
    for label in (
        "io.roehub.runtime=JobEnvelope/v1",
        "io.roehub.runtime=JobOutputKeeper/v1",
        "io.roehub.runtime=JobOutputExporter/v1",
    ):
        if label in tokens:
            break
    else:
        raise ControlOperationError(code="control_agent.job_command_rejected")


def _require_tokens(tokens: Sequence[str], required: Sequence[str]) -> None:
    if any(token not in tokens for token in required):
        raise ControlOperationError(code="control_agent.job_command_rejected")


def _option_value(tokens: Sequence[str], option: str) -> str:
    try:
        index = tokens.index(option)
        return tokens[index + 1]
    except (ValueError, IndexError) as error:
        raise ControlOperationError(code="control_agent.job_command_rejected") from error


class ControlAgentJobDockerRunner:
    """Execute only validated OCI-job Docker operations in the control-agent."""

    def run(
        self,
        command: Sequence[str],
        *,
        environ: Mapping[str, str],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        operation = classify_job_command(command)
        return self.run_typed(
            operation=operation,
            arguments=tuple(command[1:]),
            timeout_seconds=timeout_seconds,
            environ=environ,
        )

    def run_typed(
        self,
        *,
        operation: JobDockerOperation,
        arguments: Sequence[str],
        timeout_seconds: float,
        environ: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = ("docker", *arguments)
        if classify_job_command(command) != operation or not 0 < timeout_seconds <= 60:
            raise ControlOperationError(code="control_agent.job_command_rejected")
        source = os.environ if environ is None else environ
        safe_environment = {
            key: value for key, value in source.items() if key in _SAFE_ENVIRONMENT
        }
        return subprocess.run(
            list(command),
            env=safe_environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )


__all__ = [
    "ControlAgentJobDockerRunner",
    "JobDockerOperation",
    "classify_job_command",
]
