"""Unix-socket client for API and host CLI callers."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from ..auth import mint_service_assertion
from ..contracts import (
    ControlAgentRequest,
    ControlAgentResponse,
    ControlOperationError,
    OperationRequest,
    OperationResult,
)


class ControlAgentUnixClient:
    """Send one closed operation envelope per local socket connection."""

    def __init__(
        self,
        *,
        socket_path: Path,
        identity: Literal["api", "installation_owner"],
        identity_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        if len(identity_key) < 32 or len(identity_key) > 512 or timeout_seconds <= 0:
            raise ControlOperationError(code="control_agent.client_configuration_invalid")
        self._socket_path = socket_path.expanduser().resolve()
        self._identity: Literal["api", "installation_owner"] = identity
        self._identity_key = identity_key
        self._timeout = timeout_seconds

    def submit(self, operation: OperationRequest) -> OperationResult:
        response = self._call_authenticated(
            method="submit",
            operation=operation,
        )
        if response.result is None:
            raise ControlOperationError(code="control_agent.response_invalid")
        return response.result

    def get(self, operation_id: UUID) -> OperationResult:
        response = self._call_authenticated(
            method="get",
            operation_id=operation_id,
        )
        if response.result is None:
            raise ControlOperationError(code="control_agent.response_invalid")
        return response.result

    def reconcile(self, operation_id: UUID) -> OperationResult:
        response = self._call_authenticated(
            method="reconcile",
            operation_id=operation_id,
        )
        if response.result is None:
            raise ControlOperationError(code="control_agent.response_invalid")
        return response.result

    def journal(self, *, after_sequence: int = 0) -> tuple[dict[str, object], ...]:
        response = self._call_authenticated(
            method="journal",
            after_sequence=after_sequence,
        )
        return response.journal_entries

    def _call_authenticated(
        self,
        *,
        method: Literal["submit", "get", "reconcile", "journal"],
        operation: OperationRequest | None = None,
        operation_id: UUID | None = None,
        after_sequence: int = 0,
    ) -> ControlAgentResponse:
        unsigned = ControlAgentRequest(
            identity=self._identity,
            credential="0" * 32,
            method=method,
            operation=operation,
            operation_id=operation_id,
            after_sequence=after_sequence,
        )
        request = unsigned.model_copy(
            update={
                "credential": mint_service_assertion(
                    identity=self._identity,
                    identity_key=self._identity_key,
                    request_digest=unsigned.authorization_digest,
                )
            }
        )
        return self._call(request)

    def _call(self, request: ControlAgentRequest) -> ControlAgentResponse:
        encoded = request.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout)
                connection.connect(str(self._socket_path))
                connection.sendall(encoded)
                response_bytes = b""
                while not response_bytes.endswith(b"\n"):
                    chunk = connection.recv(64 * 1024)
                    if not chunk:
                        break
                    response_bytes += chunk
                    if len(response_bytes) > 1024 * 1024:
                        raise ControlOperationError(code="control_agent.response_too_large")
        except (OSError, TimeoutError) as error:
            raise ControlOperationError(code="control_agent.unavailable") from error
        try:
            response = ControlAgentResponse.model_validate_json(response_bytes)
        except ValidationError as error:
            raise ControlOperationError(code="control_agent.response_invalid") from error
        if response.status != "ok":
            raise ControlOperationError(
                code=response.error_code or "control_agent.operation_rejected"
            )
        return response


__all__ = ["ControlAgentUnixClient"]
