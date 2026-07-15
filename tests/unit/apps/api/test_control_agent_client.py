from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import pytest

from apps.api.control_agent_client import (
    ApiControlAgentClient,
    _validated_audit_event,
    build_control_agent_audit_runtime_from_environ,
)
from trading.contexts.operations import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
)


class _Transport:
    def submit(self, request: OperationRequest) -> OperationResult:
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code="topology.started",
        )

    def journal(self, *, after_sequence: int = 0) -> tuple[dict[str, object], ...]:
        assert after_sequence == 3
        return (
            {"sequence": 4, "entry_hash": "a" * 64, "state": "succeeded"},
            {"sequence": 5, "entry_hash": "b" * 64, "state": "accepted"},
        )


class _Sink:
    def __init__(self) -> None:
        self.hashes: list[str] = []

    def append_control_event(
        self, *, entry_hash: str, payload: Mapping[str, object]
    ) -> None:
        assert payload["entry_hash"] == entry_hash
        self.hashes.append(entry_hash)

    def current_sequence(self) -> int:
        return 3


def test_api_client_submits_typed_operation_and_reconciles_journal() -> None:
    client = ApiControlAgentClient(transport=_Transport())  # type: ignore[arg-type]
    result = client.submit(
        OperationRequest(operation_id=uuid4(), action=OperationAction.START)
    )
    sink = _Sink()

    cursor = client.reconcile_audit(sink=sink, after_sequence=3)

    assert result.state == OperationState.SUCCEEDED
    assert cursor == 5
    assert sink.hashes == ["a" * 64, "b" * 64]


def test_api_control_agent_runtime_is_optional_but_partial_config_fails_closed() -> None:
    assert build_control_agent_audit_runtime_from_environ(environ={}) is None
    assert (
        build_control_agent_audit_runtime_from_environ(
            environ={"ROEHUB_STORAGE_POSTGRES_DSN": "postgresql://local/runtime"}
        )
        is None
    )

    with pytest.raises(
        ControlOperationError,
        match="control_agent.api_configuration_incomplete",
    ):
        build_control_agent_audit_runtime_from_environ(
            environ={"ROEHUB_CONTROL_AGENT_SOCKET": "/tmp/control-agent.sock"}
        )


def test_api_control_audit_rejects_sensitive_payload_keys() -> None:
    with pytest.raises(
        ControlOperationError,
        match="control_agent.audit_payload_sensitive",
    ):
        _validated_audit_event(
            entry_hash="a" * 64,
            payload={"entry_hash": "a" * 64, "token": "must-not-persist"},
        )
