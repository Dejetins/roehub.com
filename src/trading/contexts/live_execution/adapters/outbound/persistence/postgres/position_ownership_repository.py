from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.live_execution.application.ports import (
    StrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.domain import (
    StrategyPositionOwnership,
    StrategyPositionOwnershipConflictError,
    StrategyPositionOwnershipState,
    StrategyPositionOwnershipStorageError,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import UserId


class PostgresStrategyPositionOwnershipRepository(StrategyPositionOwnershipRepository):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table_name: str = "strategy_position_ownership",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyPositionOwnershipRepository requires gateway")
        normalized_table = table_name.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategyPositionOwnershipRepository requires table_name")
        self._gateway = gateway
        self._table_name = normalized_table

    def reserve(self, *, ownership: StrategyPositionOwnership) -> StrategyPositionOwnership:
        query = f"""
        INSERT INTO {self._table_name}
        (
            ownership_id,
            owner_user_id,
            exchange_connection_id,
            strategy_id,
            live_profile_id,
            strategy_run_id,
            market_type,
            instrument_key,
            position_mode,
            state,
            acquired_at,
            released_at,
            expires_at,
            reason
        )
        VALUES
        (
            %(ownership_id)s,
            %(owner_user_id)s,
            %(exchange_connection_id)s,
            %(strategy_id)s,
            %(live_profile_id)s,
            %(strategy_run_id)s,
            %(market_type)s,
            %(instrument_key)s,
            %(position_mode)s,
            %(state)s,
            %(acquired_at)s,
            %(released_at)s,
            %(expires_at)s,
            %(reason)s
        )
        RETURNING
            ownership_id,
            owner_user_id,
            exchange_connection_id,
            strategy_id,
            live_profile_id,
            strategy_run_id,
            market_type,
            instrument_key,
            position_mode,
            state,
            acquired_at,
            released_at,
            expires_at,
            reason
        """
        try:
            row = self._gateway.fetch_one(
                query=query,
                parameters=_ownership_parameters(ownership=ownership),
            )
        except Exception as error:  # noqa: BLE001
            if _is_unique_violation(error=error):
                existing = self.get_blocking_for_scope(
                    owner_user_id=ownership.owner_user_id,
                    exchange_connection_id=ownership.exchange_connection_id,
                    market_type=ownership.market_type,
                    instrument_key=ownership.instrument_key,
                )
                if existing is not None:
                    raise StrategyPositionOwnershipConflictError(existing=existing) from error
            raise StrategyPositionOwnershipStorageError(
                "PostgresStrategyPositionOwnershipRepository.reserve failed"
            ) from error
        if row is None:
            raise StrategyPositionOwnershipStorageError(
                "PostgresStrategyPositionOwnershipRepository.reserve returned no row"
            )
        return _map_ownership(row=row)

    def update_state(
        self,
        *,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        state: str,
        reason: str,
        changed_at: datetime,
    ) -> StrategyPositionOwnership | None:
        released_at_sql = "released_at = %(changed_at)s," if state == "released" else ""
        query = f"""
        UPDATE {self._table_name}
        SET
            state = %(state)s,
            reason = %(reason)s,
            {released_at_sql}
            expires_at = expires_at
        WHERE owner_user_id = %(owner_user_id)s
          AND strategy_run_id = %(strategy_run_id)s
          AND state <> 'released'
        RETURNING
            ownership_id,
            owner_user_id,
            exchange_connection_id,
            strategy_id,
            live_profile_id,
            strategy_run_id,
            market_type,
            instrument_key,
            position_mode,
            state,
            acquired_at,
            released_at,
            expires_at,
            reason
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_run_id": str(strategy_run_id),
                "state": state,
                "reason": reason,
                "changed_at": changed_at,
            },
        )
        if row is None:
            return None
        return _map_ownership(row=row)

    def get_for_run(
        self, *, owner_user_id: UserId, strategy_run_id: UUID
    ) -> StrategyPositionOwnership | None:
        query = f"""
        SELECT
            ownership_id,
            owner_user_id,
            exchange_connection_id,
            strategy_id,
            live_profile_id,
            strategy_run_id,
            market_type,
            instrument_key,
            position_mode,
            state,
            acquired_at,
            released_at,
            expires_at,
            reason
        FROM {self._table_name}
        WHERE owner_user_id = %(owner_user_id)s
          AND strategy_run_id = %(strategy_run_id)s
        ORDER BY acquired_at DESC, ownership_id DESC
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_run_id": str(strategy_run_id),
            },
        )
        return _map_ownership(row=row) if row is not None else None

    def get_blocking_for_scope(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        market_type: str,
        instrument_key: str,
    ) -> StrategyPositionOwnership | None:
        query = f"""
        SELECT
            ownership_id,
            owner_user_id,
            exchange_connection_id,
            strategy_id,
            live_profile_id,
            strategy_run_id,
            market_type,
            instrument_key,
            position_mode,
            state,
            acquired_at,
            released_at,
            expires_at,
            reason
        FROM {self._table_name}
        WHERE owner_user_id = %(owner_user_id)s
          AND exchange_connection_id = %(exchange_connection_id)s
          AND market_type = %(market_type)s
          AND instrument_key = %(instrument_key)s
          AND state IN ('reserved', 'active', 'releasing', 'stale_requires_repair')
        ORDER BY acquired_at DESC, ownership_id DESC
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "exchange_connection_id": str(exchange_connection_id),
                "market_type": market_type,
                "instrument_key": instrument_key,
            },
        )
        return _map_ownership(row=row) if row is not None else None


def _ownership_parameters(*, ownership: StrategyPositionOwnership) -> dict[str, object]:
    return {
        "ownership_id": str(ownership.ownership_id),
        "owner_user_id": str(ownership.owner_user_id),
        "exchange_connection_id": str(ownership.exchange_connection_id),
        "strategy_id": str(ownership.strategy_id),
        "live_profile_id": (
            str(ownership.live_profile_id) if ownership.live_profile_id is not None else None
        ),
        "strategy_run_id": str(ownership.strategy_run_id),
        "market_type": ownership.market_type,
        "instrument_key": ownership.instrument_key,
        "position_mode": ownership.position_mode,
        "state": ownership.state,
        "acquired_at": ownership.acquired_at,
        "released_at": ownership.released_at,
        "expires_at": ownership.expires_at,
        "reason": ownership.reason,
    }


def _map_ownership(*, row: Mapping[str, Any]) -> StrategyPositionOwnership:
    try:
        state = cast(StrategyPositionOwnershipState, str(row["state"]))
        live_profile_raw = row.get("live_profile_id")
        return StrategyPositionOwnership(
            ownership_id=UUID(str(row["ownership_id"])),
            owner_user_id=UserId.from_string(str(row["owner_user_id"])),
            exchange_connection_id=UUID(str(row["exchange_connection_id"])),
            strategy_id=UUID(str(row["strategy_id"])),
            live_profile_id=UUID(str(live_profile_raw)) if live_profile_raw else None,
            strategy_run_id=UUID(str(row["strategy_run_id"])),
            market_type=str(row["market_type"]),
            instrument_key=str(row["instrument_key"]),
            position_mode=str(row["position_mode"]),
            state=state,
            acquired_at=_coerce_utc_datetime(value=row["acquired_at"], field_name="acquired_at"),
            released_at=_coerce_optional_utc_datetime(
                value=row["released_at"],
                field_name="released_at",
            ),
            expires_at=_coerce_optional_utc_datetime(
                value=row["expires_at"],
                field_name="expires_at",
            ),
            reason=str(row["reason"]),
        )
    except Exception as error:  # noqa: BLE001
        raise StrategyPositionOwnershipStorageError(
            "cannot map strategy_position_ownership row"
        ) from error


def _coerce_optional_utc_datetime(*, value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _coerce_utc_datetime(value=value, field_name=field_name)


def _coerce_utc_datetime(*, value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyPositionOwnershipStorageError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise StrategyPositionOwnershipStorageError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _is_unique_violation(*, error: Exception) -> bool:
    sql_state = getattr(error, "sqlstate", None)
    if sql_state == "23505":
        return True
    message = str(error).lower()
    return "strategy_position_ownership_one_blocking" in message
