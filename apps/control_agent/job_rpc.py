"""Authenticated typed Unix-socket RPC for OCI job Docker control."""

from __future__ import annotations

import hashlib
import json
import os
import socketserver
from pathlib import Path
from threading import Thread
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.control_agent.auth import ServiceIdentityAuthorizer
from apps.control_agent.job_runtime_backend import (
    ControlAgentJobDockerRunner,
    JobDockerOperation,
)
from trading.contexts.operations import ControlOperationError

_MAX_BYTES = 1024 * 1024


class JobControlRequest(BaseModel):
    """One closed Docker operation emitted by the trusted OCI policy engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.job-control-request/v1alpha1"] = Field(
        default="io.roehub.job-control-request/v1alpha1",
        alias="schema",
    )
    identity: Literal["job_runtime"] = "job_runtime"
    credential: str = Field(min_length=32, max_length=512)
    operation: JobDockerOperation | Literal["ping"]
    arguments: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=30.0, gt=0, le=60)

    @property
    def authorization_digest(self) -> str:
        payload = self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"credential"},
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class JobControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.job-control-response/v1alpha1"] = Field(
        default="io.roehub.job-control-response/v1alpha1",
        alias="schema",
    )
    status: Literal["ok", "error"]
    returncode: int | None = None
    stdout: str = Field(default="", max_length=512 * 1024)
    stderr: str = Field(default="", max_length=512 * 1024)
    error_code: str | None = None


class _JobServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str,
        *,
        runner: ControlAgentJobDockerRunner,
        authorizer: ServiceIdentityAuthorizer,
    ) -> None:
        self.runner = runner
        self.authorizer = authorizer
        super().__init__(socket_path, _JobHandler)


class _JobHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = cast(_JobServer, self.server)
        raw = self.rfile.readline(_MAX_BYTES + 1)
        if not raw or len(raw) > _MAX_BYTES or not raw.endswith(b"\n"):
            self._write_error("control_agent.job_request_invalid")
            return
        try:
            request = JobControlRequest.model_validate_json(raw)
            server.authorizer.authorize(
                identity="job_runtime",
                credential=request.credential,
                action=None,
                request_digest=request.authorization_digest,
            )
            if request.operation == "ping":
                if request.arguments:
                    raise ControlOperationError(
                        code="control_agent.job_command_rejected"
                    )
                response = JobControlResponse(status="ok", returncode=0)
            else:
                completed = server.runner.run_typed(
                    operation=request.operation,
                    arguments=request.arguments,
                    timeout_seconds=request.timeout_seconds,
                )
                response = JobControlResponse(
                    status="ok",
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
        except ValidationError:
            self._write_error("control_agent.job_request_invalid")
            return
        except ControlOperationError as error:
            self._write_error(error.code)
            return
        encoded = response.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"
        self.wfile.write(encoded)

    def _write_error(self, code: str) -> None:
        response = JobControlResponse(status="error", error_code=code)
        self.wfile.write(
            response.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"
        )


def start_job_control_server(
    *,
    socket_path: Path,
    runner: ControlAgentJobDockerRunner,
    authorizer: ServiceIdentityAuthorizer,
) -> tuple[_JobServer, Thread]:
    path = socket_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.exists():
        if not path.is_socket():
            raise ControlOperationError(code="control_agent.job_socket_unsafe")
        path.unlink()
    server = _JobServer(str(path), runner=runner, authorizer=authorizer)
    os.chmod(path, 0o660)
    thread = Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        name="roehub-job-control",
        daemon=True,
    )
    thread.start()
    return server, thread


def stop_job_control_server(
    *,
    server: _JobServer,
    thread: Thread,
    socket_path: Path,
) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    socket_path.expanduser().resolve().unlink(missing_ok=True)


__all__ = [
    "JobControlRequest",
    "JobControlResponse",
    "start_job_control_server",
    "stop_job_control_server",
]
