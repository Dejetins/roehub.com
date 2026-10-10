"""Authenticated JSON-over-Unix-socket server for privileged operations."""

from __future__ import annotations

import os
import socketserver
from pathlib import Path
from threading import Thread
from typing import cast

from pydantic import ValidationError

from apps.control_agent.auth import ServiceIdentityAuthorizer
from trading.contexts.operations import ControlOperationError, ControlOperationService
from trading.contexts.operations.adapters import AppendOnlyOperationJournal
from trading.contexts.operations.contracts import ControlAgentRequest, ControlAgentResponse

_MAX_REQUEST_BYTES = 1024 * 1024


class _ControlAgentServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str,
        *,
        service: ControlOperationService,
        journal: AppendOnlyOperationJournal,
        authorizer: ServiceIdentityAuthorizer,
    ) -> None:
        self.service = service
        self.journal = journal
        self.authorizer = authorizer
        super().__init__(socket_path, _ControlAgentHandler)


class _ControlAgentHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        control_server = cast(_ControlAgentServer, self.server)
        raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            self._write_error("control_agent.request_invalid")
            return
        try:
            request = ControlAgentRequest.model_validate_json(raw)
            action = request.operation.action if request.operation is not None else None
            control_server.authorizer.authorize(
                identity=request.identity,
                credential=request.credential,
                action=action,
                request_digest=request.authorization_digest,
            )
            if request.method == "submit":
                assert request.operation is not None
                response = ControlAgentResponse(
                    status="ok",
                    result=control_server.service.submit(request.operation),
                )
            elif request.method == "get":
                assert request.operation_id is not None
                response = ControlAgentResponse(
                    status="ok",
                    result=control_server.service.get(request.operation_id),
                )
            elif request.method == "reconcile":
                assert request.operation_id is not None
                response = ControlAgentResponse(
                    status="ok",
                    result=control_server.service.reconcile(request.operation_id),
                )
            else:
                entries = tuple(
                    dict(entry)
                    for entry in control_server.journal.entries(
                        after_sequence=request.after_sequence
                    )
                )
                response = ControlAgentResponse(status="ok", journal_entries=entries)
        except ValidationError:
            self._write_error("control_agent.request_invalid")
            return
        except ControlOperationError as error:
            self._write_error(error.code)
            return
        encoded = response.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"
        self.wfile.write(encoded)

    def _write_error(self, code: str) -> None:
        response = ControlAgentResponse(status="error", error_code=code)
        self.wfile.write(
            response.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"
        )


def serve(
    *,
    socket_path: Path,
    service: ControlOperationService,
    journal: AppendOnlyOperationJournal,
    authorizer: ServiceIdentityAuthorizer,
) -> None:
    """Serve until terminated, replacing only a stale local socket node."""

    path, server = _build_server(
        socket_path=socket_path,
        service=service,
        journal=journal,
        authorizer=authorizer,
    )
    try:
        os.chmod(path, 0o660)
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        path.unlink(missing_ok=True)


def start_control_agent_server(
    *,
    socket_path: Path,
    service: ControlOperationService,
    journal: AppendOnlyOperationJournal,
    authorizer: ServiceIdentityAuthorizer,
) -> tuple[_ControlAgentServer, Thread]:
    """Start the same authenticated server for a bounded runtime composition."""

    _path, server = _build_server(
        socket_path=socket_path,
        service=service,
        journal=journal,
        authorizer=authorizer,
    )
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return server, thread


def stop_control_agent_server(
    *,
    server: _ControlAgentServer,
    thread: Thread,
    socket_path: Path,
) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    socket_path.unlink(missing_ok=True)


def _build_server(
    *,
    socket_path: Path,
    service: ControlOperationService,
    journal: AppendOnlyOperationJournal,
    authorizer: ServiceIdentityAuthorizer,
) -> tuple[Path, _ControlAgentServer]:
    path = socket_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.exists():
        if not path.is_socket():
            raise ControlOperationError(code="control_agent.socket_unsafe")
        path.unlink()
    server = _ControlAgentServer(
        str(path), service=service, journal=journal, authorizer=authorizer
    )
    os.chmod(path, 0o660)
    return path, server


__all__ = ["serve", "start_control_agent_server", "stop_control_agent_server"]
