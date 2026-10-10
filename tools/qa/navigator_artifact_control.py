"""Storage-only local Navigator control service using the production signed transport.

Credentials and installation settings must be provisioned explicitly before launch.
This entrypoint never seeds data, activates scheduling, or connects to Docker.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from apps.control_agent.artifact_storage_wiring import build_artifact_storage_service
from apps.control_agent.auth import ServiceIdentityAuthorizer
from apps.control_agent.server import serve
from trading.contexts.operations import ControlOperationError, ControlOperationService
from trading.contexts.operations.adapters import AppendOnlyOperationJournal
from trading.contexts.operations.contracts import OperationRequest, OperationResult


class DisabledHostOperations:
    """The local artifact service has no general installation-control capability."""

    def execute(self, request: OperationRequest) -> OperationResult:
        raise ControlOperationError(code="operation.action_not_allowed")

    def reconcile(self, request: OperationRequest) -> OperationResult:
        raise ControlOperationError(code="operation.action_not_allowed")


def _serve(socket: Path) -> None:
    # Production serve can replace a stale node; this local launcher requires the
    # operator to resolve it, so a second launch cannot steal an existing socket.
    if socket.exists() or socket.is_symlink():
        raise RuntimeError("Local artifact control socket already exists")
    storage = build_artifact_storage_service(os.environ)
    if storage is None:
        raise RuntimeError("Explicit artifact installation configuration is required")
    state = Path(os.environ["ROEHUB_LOCAL_ARTIFACT_CONTROL_STATE"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = AppendOnlyOperationJournal(path=state / "operations.jsonl")
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=Path(os.environ["ROEHUB_CONTROL_AGENT_API_IDENTITY_FILE"]),
        owner_token_file=Path(os.environ["ROEHUB_LOCAL_ARTIFACT_OWNER_IDENTITY_FILE"]),
        replay_state_dir=state / "replay",
    )
    serve(
        socket_path=socket,
        service=ControlOperationService(backend=DisabledHostOperations(), journal=journal),
        journal=journal,
        authorizer=authorizer,
        storage_service=storage,
    )


def main() -> None:
    socket = Path(os.environ["ROEHUB_CONTROL_AGENT_SOCKET"]).expanduser().absolute()
    lock = socket.with_name(socket.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _serve(socket)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
