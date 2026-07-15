"""Product DockerCommandRunner client for the control-agent job socket."""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from apps.control_agent.job_rpc import JobControlRequest, JobControlResponse
from apps.control_agent.job_runtime_backend import classify_job_command
from trading.contexts.operations import ControlOperationError
from trading.contexts.operations.auth import mint_service_assertion


class ControlAgentJobUnixClient:
    """Send the closed OCI job grammar over an authenticated local socket."""

    def __init__(
        self,
        *,
        socket_path: Path,
        identity_key: str,
    ) -> None:
        if len(identity_key) < 32 or len(identity_key) > 512:
            raise ControlOperationError(code="control_agent.job_client_invalid")
        self._socket_path = socket_path.expanduser().resolve()
        self._identity_key = identity_key

    def ping(self) -> None:
        response = self._call_authenticated(
            operation="ping",
            arguments=(),
            timeout_seconds=5.0,
        )
        if response.returncode != 0:
            raise ControlOperationError(code="control_agent.job_unavailable")

    def run(
        self,
        command: Sequence[str],
        *,
        environ: Mapping[str, str],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        del environ
        operation = classify_job_command(command)
        response = self._call_authenticated(
            operation=operation,
            arguments=tuple(command[1:]),
            timeout_seconds=timeout_seconds,
        )
        if response.returncode is None:
            raise ControlOperationError(code="control_agent.job_response_invalid")
        return subprocess.CompletedProcess(
            list(command),
            response.returncode,
            response.stdout,
            response.stderr,
        )

    def _call_authenticated(
        self,
        *,
        operation: Literal["ping"]
        | Literal[
            "container.create",
            "container.inspect",
            "container.kill",
            "container.remove",
            "container.start",
            "container.run",
            "volume.create",
            "volume.inspect",
            "volume.remove",
        ],
        arguments: tuple[str, ...],
        timeout_seconds: float,
    ) -> JobControlResponse:
        unsigned = JobControlRequest(
            credential="0" * 32,
            operation=operation,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
        )
        request = unsigned.model_copy(
            update={
                "credential": mint_service_assertion(
                    identity="job_runtime",
                    identity_key=self._identity_key,
                    request_digest=unsigned.authorization_digest,
                )
            }
        )
        encoded = request.model_dump_json(by_alias=True).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(timeout_seconds + 1)
                connection.connect(str(self._socket_path))
                connection.sendall(encoded)
                response_bytes = b""
                while not response_bytes.endswith(b"\n"):
                    chunk = connection.recv(64 * 1024)
                    if not chunk:
                        break
                    response_bytes += chunk
                    if len(response_bytes) > 1024 * 1024:
                        raise ControlOperationError(
                            code="control_agent.job_response_too_large"
                        )
        except (OSError, TimeoutError) as error:
            raise ControlOperationError(code="control_agent.job_unavailable") from error
        try:
            response = JobControlResponse.model_validate_json(response_bytes)
        except ValidationError as error:
            raise ControlOperationError(
                code="control_agent.job_response_invalid"
            ) from error
        if response.status != "ok":
            raise ControlOperationError(
                code=response.error_code or "control_agent.job_rejected"
            )
        return response


__all__ = ["ControlAgentJobUnixClient"]
