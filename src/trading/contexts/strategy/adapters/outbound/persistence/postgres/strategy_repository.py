from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import StrategyRepository
from trading.contexts.strategy.domain.entities import Strategy, StrategySpecV1
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class PostgresStrategyRepository(StrategyRepository):
    """
    PostgresStrategyRepository — explicit SQL adapter for immutable Strategy v1 storage.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/gateway.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        strategies_table: str = "strategy_strategies",
    ) -> None:
        """
        Initialize strategy repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            strategies_table: Strategy table name.
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
            raise ValueError("PostgresStrategyRepository requires gateway")
        normalized_table = strategies_table.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategyRepository requires non-empty strategies_table")
        self._gateway = gateway
        self._strategies_table = normalized_table

    def create(self, *, strategy: Strategy) -> Strategy:
        """
        Insert immutable strategy row and return persisted snapshot.

        Args:
            strategy: Strategy aggregate to persist.
        Returns:
            Strategy: Persisted row mapped into domain object.
        Assumptions:
            `spec_json` is immutable and insert-only.
        Raises:
            StrategyStorageError: If insert returns no row or row mapping fails.
        Side Effects:
            Executes one SQL insert statement.
        """
        query = f"""
        INSERT INTO {self._strategies_table}
        (
            strategy_id,
            user_id,
            name,
            instrument_id,
            instrument_key,
            market_type,
            symbol,
            timeframe,
            indicators_json,
            spec_json,
            created_at,
            is_deleted
        )
        VALUES
        (
            %(strategy_id)s,
            %(user_id)s,
            %(name)s,
            %(instrument_id)s::jsonb,
            %(instrument_key)s,
            %(market_type)s,
            %(symbol)s,
            %(timeframe)s,
            %(indicators_json)s::jsonb,
            %(spec_json)s::jsonb,
            %(created_at)s,
            %(is_deleted)s
        )
        RETURNING
            strategy_id,
            user_id,
            name,
            instrument_id,
            instrument_key,
            market_type,
            symbol,
            timeframe,
            indicators_json,
            spec_json,
            created_at,
            is_deleted
        """
        spec_json = strategy.spec.to_json()
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "strategy_id": str(strategy.strategy_id),
                "user_id": str(strategy.user_id),
                "name": strategy.name,
                "instrument_id": _json_dumps(payload=strategy.spec.instrument_id.as_dict()),
                "instrument_key": strategy.spec.instrument_key,
                "market_type": strategy.spec.market_type,
                "symbol": str(strategy.spec.instrument_id.symbol),
                "timeframe": strategy.spec.timeframe.code,
                "indicators_json": _json_dumps(payload=list(strategy.spec.indicators)),
                "spec_json": _json_dumps(payload=spec_json),
                "created_at": strategy.created_at,
                "is_deleted": strategy.is_deleted,
            },
        )
        if row is None:
            raise StrategyStorageError("PostgresStrategyRepository.create returned no row")
        return _map_strategy_row(row=row)

    def find_by_strategy_id(self, *, user_id: UserId, strategy_id: UUID) -> Strategy | None:
        """
        Find strategy by owner and strategy id.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            Strategy | None: Mapped strategy snapshot or `None`.
        Assumptions:
            Strategy id is unique in table.
        Raises:
            StrategyStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            strategy_id,
            user_id,
            name,
            instrument_id,
            instrument_key,
            market_type,
            symbol,
            timeframe,
            indicators_json,
            spec_json,
            created_at,
            is_deleted
        FROM {self._strategies_table}
        WHERE user_id = %(user_id)s
          AND strategy_id = %(strategy_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "strategy_id": str(strategy_id),
            },
        )
        if row is None:
            return None
        return _map_strategy_row(row=row)

    def list_for_user(
        self,
        *,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        """
        List user strategies with deterministic ordering by creation time and identifier.

        Args:
            user_id: Strategy owner identifier.
            include_deleted: Include soft-deleted rows when true.
        Returns:
            tuple[Strategy, ...]: Sorted strategy snapshots.
        Assumptions:
            `ORDER BY created_at ASC, strategy_id ASC` guarantees deterministic ordering.
        Raises:
            StrategyStorageError: If one of rows cannot be mapped.
        Side Effects:
            Executes one SQL select statement.
        """
        where_deleted = "" if include_deleted else "AND is_deleted = FALSE"
        query = f"""
        SELECT
            strategy_id,
            user_id,
            name,
            instrument_id,
            instrument_key,
            market_type,
            symbol,
            timeframe,
            indicators_json,
            spec_json,
            created_at,
            is_deleted
        FROM {self._strategies_table}
        WHERE user_id = %(user_id)s
          {where_deleted}
        ORDER BY created_at ASC, strategy_id ASC
        """
        rows = self._gateway.fetch_all(query=query, parameters={"user_id": str(user_id)})
        return tuple(_map_strategy_row(row=row) for row in rows)

    def soft_delete(self, *, user_id: UserId, strategy_id: UUID) -> bool:
        """
        Soft-delete strategy by updating only `is_deleted = TRUE`.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            bool: `True` when one row changed, otherwise `False`.
        Assumptions:
            This is the only allowed mutable operation for strategy rows.
        Raises:
            StrategyStorageError: If SQL execution fails unexpectedly.
        Side Effects:
            Executes one SQL update statement.
        """
        query = f"""
        UPDATE {self._strategies_table}
        SET is_deleted = TRUE
        WHERE user_id = %(user_id)s
          AND strategy_id = %(strategy_id)s
          AND is_deleted = FALSE
        RETURNING strategy_id
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return row is not None



def _map_strategy_row(*, row: Mapping[str, Any]) -> Strategy:
    """
    Map SQL row payload into immutable Strategy domain aggregate.

    Args:
        row: SQL row mapping.
    Returns:
        Strategy: Mapped domain strategy aggregate.
    Assumptions:
        Row schema matches Strategy v1 storage contract.
    Raises:
        StrategyStorageError: If mapping fails.
    Side Effects:
        None.
    """
    try:
        spec_payload = _coerce_json_mapping(value=row["spec_json"], field_name="spec_json")
        spec_payload = dict(spec_payload)

        instrument_payload = _coerce_json_mapping(
            value=row["instrument_id"],
            field_name="instrument_id",
        )
        instrument_market_id = instrument_payload.get("market_id")
        instrument_symbol = instrument_payload.get("symbol")
        if not isinstance(instrument_market_id, int) or not isinstance(instrument_symbol, str):
            raise StrategyStorageError("Strategy row instrument_id payload is malformed")

        spec_payload.setdefault("instrument_id", {
            "market_id": instrument_market_id,
            "symbol": instrument_symbol,
        })

        instrument_key = str(row["instrument_key"])
        market_type = str(row["market_type"])
        timeframe = str(row["timeframe"])
        symbol = str(row["symbol"])
        indicators_payload = _coerce_json_sequence(
            value=row["indicators_json"],
            field_name="indicators_json",
        )

        spec_payload.setdefault("instrument_key", instrument_key)
        spec_payload.setdefault("market_type", market_type)
        spec_payload.setdefault("timeframe", timeframe)
        spec_payload.setdefault("symbol", symbol)
        spec_payload.setdefault("indicators", indicators_payload)

        spec = StrategySpecV1.from_json(payload=spec_payload)
        return Strategy(
            strategy_id=UUID(str(row["strategy_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            name=str(row["name"]),
            spec=spec,
            created_at=row["created_at"],
            is_deleted=bool(row["is_deleted"]),
        )
    except StrategyStorageError:
        raise
    except Exception as error:  # noqa: BLE001
        raise StrategyStorageError("PostgresStrategyRepository cannot map strategy row") from error



def _coerce_json_mapping(*, value: Any, field_name: str) -> Mapping[str, Any]:
    """
    Convert gateway value into JSON mapping payload.

    Args:
        value: Raw gateway row value.
        field_name: Field name used in deterministic errors.
    Returns:
        Mapping[str, Any]: Parsed JSON mapping.
    Assumptions:
        psycopg may return parsed object, text JSON, or memoryview.
    Raises:
        StrategyStorageError: If value cannot be coerced to mapping.
    Side Effects:
        None.
    """
    parsed = _coerce_json_value(value=value, field_name=field_name)
    if not isinstance(parsed, Mapping):
        raise StrategyStorageError(f"{field_name} must be JSON object")
    return parsed



def _coerce_json_sequence(*, value: Any, field_name: str) -> Sequence[Any]:
    """
    Convert gateway value into JSON sequence payload.

    Args:
        value: Raw gateway row value.
        field_name: Field name used in deterministic errors.
    Returns:
        Sequence[Any]: Parsed JSON sequence.
    Assumptions:
        psycopg may return parsed object, text JSON, or memoryview.
    Raises:
        StrategyStorageError: If value cannot be coerced to sequence.
    Side Effects:
        None.
    """
    parsed = _coerce_json_value(value=value, field_name=field_name)
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes, bytearray)):
        raise StrategyStorageError(f"{field_name} must be JSON array")
    return parsed



def _coerce_json_value(*, value: Any, field_name: str) -> Any:
    """
    Normalize JSON field value from SQL row into Python object.

    Args:
        value: Raw gateway value.
        field_name: Field name used in deterministic errors.
    Returns:
        Any: Parsed JSON-compatible object.
    Assumptions:
        Raw value may already be parsed or represented as bytes/text.
    Raises:
        StrategyStorageError: If JSON decoding fails.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
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
    Serialize payload into deterministic JSON string for explicit SQL `::jsonb` casts.

    Args:
        payload: JSON-serializable payload.
    Returns:
        str: Deterministic JSON text.
    Assumptions:
        Payload is serializable by Python JSON module.
    Raises:
        StrategyStorageError: If payload is not JSON-serializable.
    Side Effects:
        None.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise StrategyStorageError("JSON payload is not serializable") from error
