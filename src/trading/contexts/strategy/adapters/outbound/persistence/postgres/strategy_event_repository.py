from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import StrategyEventRepository
from trading.contexts.strategy.domain.entities import StrategyEvent
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class PostgresStrategyEventRepository(StrategyEventRepository):
    """
    PostgresStrategyEventRepository — explicit SQL adapter for append-only Strategy v1 events.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_event_repository.py
      - src/trading/contexts/strategy/domain/entities/strategy_event.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        events_table: str = "strategy_events",
    ) -> None:
        """
        Initialize event repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            events_table: Event table name.
        Returns:
            None.
        Assumptions:
            Table schema follows Strategy v1 migration contract.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyEventRepository requires gateway")
        normalized_table = events_table.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategyEventRepository requires non-empty events_table")
        self._gateway = gateway
        self._events_table = normalized_table

    def append(self, *, event: StrategyEvent) -> StrategyEvent:
        """
        Append one immutable event row.

        Args:
            event: Event snapshot to append.
        Returns:
            StrategyEvent: Persisted event snapshot.
        Assumptions:
            Event stream is append-only in repository API.
        Raises:
            StrategyStorageError: If append returns no row or mapping fails.
        Side Effects:
            Executes one SQL insert statement.
        """
        query = f"""
        INSERT INTO {self._events_table}
        (
            event_id,
            user_id,
            strategy_id,
            run_id,
            ts,
            event_type,
            payload_json
        )
        VALUES
        (
            %(event_id)s,
            %(user_id)s,
            %(strategy_id)s,
            %(run_id)s,
            %(ts)s,
            %(event_type)s,
            %(payload_json)s::jsonb
        )
        RETURNING
            event_id,
            user_id,
            strategy_id,
            run_id,
            ts,
            event_type,
            payload_json
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "event_id": str(event.event_id),
                "user_id": str(event.user_id),
                "strategy_id": str(event.strategy_id),
                "run_id": str(event.run_id) if event.run_id is not None else None,
                "ts": event.ts,
                "event_type": event.event_type,
                "payload_json": _json_dumps(payload=event.payload_json),
            },
        )
        if row is None:
            raise StrategyStorageError("PostgresStrategyEventRepository.append returned no row")
        return _map_event_row(row=row)

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        List strategy-level event stream in deterministic timestamp order.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered event snapshots.
        Assumptions:
            Ordering is deterministic by `ORDER BY ts ASC, event_id ASC`.
        Raises:
            StrategyStorageError: If one of rows cannot be mapped.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            event_id,
            user_id,
            strategy_id,
            run_id,
            ts,
            event_type,
            payload_json
        FROM {self._events_table}
        WHERE user_id = %(user_id)s
          AND strategy_id = %(strategy_id)s
        ORDER BY ts ASC, event_id ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={
                "user_id": str(user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return tuple(_map_event_row(row=row) for row in rows)

    def list_for_run(self, *, user_id: UserId, run_id: UUID) -> tuple[StrategyEvent, ...]:
        """
        List run-level event stream in deterministic timestamp order.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            tuple[StrategyEvent, ...]: Ordered event snapshots.
        Assumptions:
            Query selects only rows with non-null `run_id` equal to input.
        Raises:
            StrategyStorageError: If one of rows cannot be mapped.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            event_id,
            user_id,
            strategy_id,
            run_id,
            ts,
            event_type,
            payload_json
        FROM {self._events_table}
        WHERE user_id = %(user_id)s
          AND run_id = %(run_id)s
        ORDER BY ts ASC, event_id ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={
                "user_id": str(user_id),
                "run_id": str(run_id),
            },
        )
        return tuple(_map_event_row(row=row) for row in rows)



def _map_event_row(*, row: Mapping[str, Any]) -> StrategyEvent:
    """
    Map SQL row into immutable StrategyEvent domain object.

    Args:
        row: SQL row mapping.
    Returns:
        StrategyEvent: Mapped event snapshot.
    Assumptions:
        Row schema follows Strategy v1 event table contract.
    Raises:
        StrategyStorageError: If mapping fails.
    Side Effects:
        None.
    """
    try:
        run_id_raw = row["run_id"]
        run_id = None if run_id_raw is None else UUID(str(run_id_raw))
        payload_json = _coerce_json_mapping(value=row["payload_json"], field_name="payload_json")
        return StrategyEvent(
            event_id=UUID(str(row["event_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            strategy_id=UUID(str(row["strategy_id"])),
            run_id=run_id,
            ts=row["ts"],
            event_type=str(row["event_type"]),
            payload_json=payload_json,
        )
    except Exception as error:  # noqa: BLE001
        raise StrategyStorageError(
            "PostgresStrategyEventRepository cannot map event row"
        ) from error



def _coerce_json_mapping(*, value: Any, field_name: str) -> Mapping[str, Any]:
    """
    Convert gateway JSON field into mapping payload.

    Args:
        value: Raw gateway value.
        field_name: Field name used in deterministic errors.
    Returns:
        Mapping[str, Any]: Parsed JSON mapping.
    Assumptions:
        psycopg may return dict, text JSON, bytes, or memoryview.
    Raises:
        StrategyStorageError: If value cannot be decoded as JSON mapping.
    Side Effects:
        None.
    """
    parsed = _coerce_json_value(value=value, field_name=field_name)
    if not isinstance(parsed, Mapping):
        raise StrategyStorageError(f"{field_name} must be JSON object")
    return parsed



def _coerce_json_value(*, value: Any, field_name: str) -> Any:
    """
    Normalize JSON field value into Python payload object.

    Args:
        value: Raw gateway value.
        field_name: Field name used in deterministic errors.
    Returns:
        Any: Parsed JSON-compatible object.
    Assumptions:
        JSON payload is serialized using UTF-8 representation.
    Raises:
        StrategyStorageError: If value cannot be decoded.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes().decode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise StrategyStorageError(f"{field_name} is invalid JSON text") from error
    raise StrategyStorageError(f"{field_name} has unsupported JSON value type")



def _json_dumps(*, payload: Any) -> str:
    """
    Serialize payload into deterministic JSON text for explicit SQL `::jsonb` casts.

    Args:
        payload: JSON-serializable payload.
    Returns:
        str: Deterministic JSON text.
    Assumptions:
        Payload is serializable via Python JSON module.
    Raises:
        StrategyStorageError: If payload is not serializable.
    Side Effects:
        None.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise StrategyStorageError("JSON payload is not serializable") from error
