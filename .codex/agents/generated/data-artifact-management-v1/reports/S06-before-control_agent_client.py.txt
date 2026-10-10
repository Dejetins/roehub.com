"""API-side adapter for authenticated typed control-agent operations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from apps.control_agent.auth import read_private_credential
from apps.migrations.bootstrap import normalize_psycopg_dsn
from trading.contexts.operations import ControlOperationError, OperationRequest, OperationResult
from trading.contexts.operations.adapters import ControlAgentUnixClient


class ControlAuditSink(Protocol):
    """Persist one journal event idempotently by its hash."""

    def append_control_event(
        self,
        *,
        entry_hash: str,
        payload: Mapping[str, object],
    ) -> None: ...

    def current_sequence(self) -> int: ...


class PostgresControlAuditSink:
    """Persist the independent journal transactionally with a durable cursor."""

    def __init__(self, *, dsn: str) -> None:
        self._dsn = normalize_psycopg_dsn(dsn=dsn)

    def current_sequence(self) -> int:
        try:
            with psycopg.connect(self._dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT sequence FROM control_operation_audit_cursor "
                        "WHERE singleton = TRUE"
                    )
                    row = cursor.fetchone()
        except psycopg.Error as error:
            raise ControlOperationError(
                code="control_agent.audit_store_unavailable"
            ) from error
        if row is None:
            raise ControlOperationError(code="control_agent.audit_cursor_missing")
        return int(row[0])

    def append_control_event(
        self,
        *,
        entry_hash: str,
        payload: Mapping[str, object],
    ) -> None:
        event = _validated_audit_event(entry_hash=entry_hash, payload=payload)
        try:
            with psycopg.connect(self._dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT sequence, entry_hash FROM control_operation_audit_cursor "
                        "WHERE singleton = TRUE FOR UPDATE"
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise ControlOperationError(
                            code="control_agent.audit_cursor_missing"
                        )
                    current_sequence = int(row[0])
                    current_hash = str(row[1]) if row[1] is not None else None
                    sequence = cast(int, event["sequence"])
                    if sequence <= current_sequence:
                        cursor.execute(
                            "SELECT entry_hash FROM control_operation_audit_events "
                            "WHERE sequence = %s",
                            (sequence,),
                        )
                        existing = cursor.fetchone()
                        if existing is None or str(existing[0]) != entry_hash:
                            raise ControlOperationError(
                                code="control_agent.audit_replay_conflict"
                            )
                        return
                    if sequence != current_sequence + 1:
                        raise ControlOperationError(code="control_agent.audit_sequence_gap")
                    previous_hash = cast(str, event["previous_hash"])
                    expected_previous = current_hash or "0" * 64
                    if previous_hash != expected_previous:
                        raise ControlOperationError(code="control_agent.audit_chain_mismatch")
                    cursor.execute(
                        """
                        INSERT INTO control_operation_audit_events (
                            entry_hash,
                            sequence,
                            operation_id,
                            action,
                            state,
                            detail_code,
                            recorded_at,
                            payload
                        ) VALUES (
                            %(entry_hash)s,
                            %(sequence)s,
                            %(operation_id)s,
                            %(action)s,
                            %(state)s,
                            %(detail_code)s,
                            %(recorded_at)s,
                            %(payload)s::jsonb
                        )
                        ON CONFLICT (entry_hash) DO NOTHING
                        """,
                        {
                            **event,
                            "payload": Jsonb(dict(payload)),
                        },
                    )
                    cursor.execute(
                        "UPDATE control_operation_audit_cursor "
                        "SET sequence = %s, entry_hash = %s "
                        "WHERE singleton = TRUE",
                        (sequence, entry_hash),
                    )
        except ControlOperationError:
            raise
        except psycopg.Error as error:
            raise ControlOperationError(
                code="control_agent.audit_store_unavailable"
            ) from error


class ApiControlAgentClient:
    """Submit operations and reconcile the independent journal into API audit state."""

    def __init__(self, *, transport: ControlAgentUnixClient) -> None:
        self._transport = transport

    def submit(self, request: OperationRequest) -> OperationResult:
        return self._transport.submit(request)

    def get(self, operation_id: UUID) -> OperationResult:
        return self._transport.get(operation_id)

    def reconcile(self, operation_id: UUID) -> OperationResult:
        return self._transport.reconcile(operation_id)

    def reconcile_audit(
        self,
        *,
        sink: ControlAuditSink,
        after_sequence: int | None = None,
    ) -> int:
        if after_sequence is None:
            after_sequence = sink.current_sequence()
        last_sequence = after_sequence
        for entry in self._transport.journal(after_sequence=after_sequence):
            sink.append_control_event(
                entry_hash=str(entry["entry_hash"]),
                payload=entry,
            )
            last_sequence = int(str(entry["sequence"]))
        return last_sequence


def build_api_control_agent_client_from_environ(
    *, environ: Mapping[str, str] | None = None
) -> ApiControlAgentClient:
    values = os.environ if environ is None else environ
    socket_path = Path(values["ROEHUB_CONTROL_AGENT_SOCKET"])
    identity_file = Path(values["ROEHUB_CONTROL_AGENT_API_IDENTITY_FILE"])
    return ApiControlAgentClient(
        transport=ControlAgentUnixClient(
            socket_path=socket_path,
            identity="api",
            identity_key=read_private_credential(identity_file),
        )
    )


def build_control_agent_audit_runtime_from_environ(
    *, environ: Mapping[str, str] | None = None
) -> tuple[ApiControlAgentClient, PostgresControlAuditSink] | None:
    values = os.environ if environ is None else environ
    control_keys = (
        "ROEHUB_CONTROL_AGENT_SOCKET",
        "ROEHUB_CONTROL_AGENT_API_IDENTITY_FILE",
    )
    control_present = tuple(
        bool(values.get(key, "").strip()) for key in control_keys
    )
    if not any(control_present):
        return None
    if not all(control_present) or not values.get(
        "ROEHUB_STORAGE_POSTGRES_DSN", ""
    ).strip():
        raise ControlOperationError(code="control_agent.api_configuration_incomplete")
    return (
        build_api_control_agent_client_from_environ(environ=values),
        PostgresControlAuditSink(dsn=values["ROEHUB_STORAGE_POSTGRES_DSN"]),
    )


def _validated_audit_event(
    *,
    entry_hash: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    forbidden = {
        "authorization",
        "ciphertext",
        "cookie",
        "credential",
        "dsn",
        "environment",
        "identity",
        "password",
        "plaintext",
        "token",
    }
    for key, value in _walk(payload):
        if key.lower() in forbidden:
            raise ControlOperationError(code="control_agent.audit_payload_sensitive")
        if isinstance(value, str) and len(value) > 8192:
            raise ControlOperationError(code="control_agent.audit_payload_invalid")
    try:
        raw_sequence = payload["sequence"]
        if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool):
            raise TypeError
        sequence = raw_sequence
        operation_id = UUID(str(payload["operation_id"]))
        action = str(payload["action"])
        state = str(payload["state"])
        detail_code = str(payload["detail_code"])
        recorded_at = datetime.fromisoformat(str(payload["recorded_at"]))
        previous_hash = str(payload["previous_hash"])
    except (KeyError, TypeError, ValueError) as error:
        raise ControlOperationError(code="control_agent.audit_payload_invalid") from error
    if (
        len(entry_hash) != 64
        or payload.get("entry_hash") != entry_hash
        or sequence < 1
        or len(previous_hash) != 64
    ):
        raise ControlOperationError(code="control_agent.audit_payload_invalid")
    return {
        "entry_hash": entry_hash,
        "sequence": sequence,
        "operation_id": operation_id,
        "action": action,
        "state": state,
        "detail_code": detail_code,
        "recorded_at": recorded_at,
        "previous_hash": previous_hash,
    }


def _walk(value: object) -> tuple[tuple[str, object], ...]:
    found: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.append((str(key), item))
            found.extend(_walk(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk(item))
    return tuple(found)


__all__ = [
    "ApiControlAgentClient",
    "ControlAuditSink",
    "PostgresControlAuditSink",
    "build_api_control_agent_client_from_environ",
    "build_control_agent_audit_runtime_from_environ",
]
