from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import (
    LiveStrategyProfileRepository,
)
from trading.contexts.strategy.domain.entities.live_strategy_profile import (
    LiveStrategyProfile,
)
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresLiveStrategyProfileRepository(LiveStrategyProfileRepository):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table_name: str = "strategy_live_profiles",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresLiveStrategyProfileRepository requires gateway")
        normalized_table = table_name.strip()
        if not normalized_table:
            raise ValueError("PostgresLiveStrategyProfileRepository requires table")
        self._gateway = gateway
        self._table_name = normalized_table

    def create(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile | None:
        row = self._gateway.fetch_one(
            query=f"""
            INSERT INTO {self._table_name}
            (
                profile_id,
                organization_id,
                owner_user_id,
                strategy_id,
                mode,
                exchange_connection_id,
                sizing_method,
                sizing_value,
                max_position_notional,
                max_orders_per_run,
                max_notional_per_run,
                readiness_status,
                readiness_reason,
                created_at,
                updated_at
            )
            VALUES
            (
                %(profile_id)s,
                %(organization_id)s,
                %(owner_user_id)s,
                %(strategy_id)s,
                %(mode)s,
                %(exchange_connection_id)s,
                %(sizing_method)s,
                %(sizing_value)s,
                %(max_position_notional)s,
                %(max_orders_per_run)s,
                %(max_notional_per_run)s,
                %(readiness_status)s,
                %(readiness_reason)s,
                %(created_at)s,
                %(updated_at)s
            )
            ON CONFLICT (organization_id, owner_user_id, strategy_id) DO NOTHING
            RETURNING
                profile_id,
                organization_id,
                owner_user_id,
                strategy_id,
                mode,
                exchange_connection_id,
                sizing_method,
                sizing_value,
                max_position_notional,
                max_orders_per_run,
                max_notional_per_run,
                readiness_status,
                readiness_reason,
                created_at,
                updated_at
            """,
            parameters=_profile_parameters(profile=profile),
        )
        return _map_profile(row=row) if row is not None else None

    def get_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
    ) -> LiveStrategyProfile | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT
                profile_id,
                organization_id,
                owner_user_id,
                strategy_id,
                mode,
                exchange_connection_id,
                sizing_method,
                sizing_value,
                max_position_notional,
                max_orders_per_run,
                max_notional_per_run,
                readiness_status,
                readiness_reason,
                created_at,
                updated_at
            FROM {self._table_name}
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return _map_profile(row=row) if row is not None else None

    def update(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile:
        row = self._gateway.fetch_one(
            query=f"""
            UPDATE {self._table_name}
               SET mode = %(mode)s,
                   exchange_connection_id = %(exchange_connection_id)s,
                   sizing_method = %(sizing_method)s,
                   sizing_value = %(sizing_value)s,
                   max_position_notional = %(max_position_notional)s,
                   max_orders_per_run = %(max_orders_per_run)s,
                   max_notional_per_run = %(max_notional_per_run)s,
                   readiness_status = %(readiness_status)s,
                   readiness_reason = %(readiness_reason)s,
                   updated_at = %(updated_at)s
             WHERE profile_id = %(profile_id)s
               AND organization_id = %(organization_id)s
               AND owner_user_id = %(owner_user_id)s
               AND strategy_id = %(strategy_id)s
            RETURNING
                profile_id,
                organization_id,
                owner_user_id,
                strategy_id,
                mode,
                exchange_connection_id,
                sizing_method,
                sizing_value,
                max_position_notional,
                max_orders_per_run,
                max_notional_per_run,
                readiness_status,
                readiness_reason,
                created_at,
                updated_at
            """,
            parameters=_profile_parameters(profile=profile),
        )
        if row is None:
            raise StrategyStorageError("live profile update returned no row")
        return _map_profile(row=row)


def _profile_parameters(*, profile: LiveStrategyProfile) -> dict[str, object]:
    return {
        "profile_id": str(profile.profile_id),
        "organization_id": str(profile.organization_id),
        "owner_user_id": str(profile.owner_user_id),
        "strategy_id": str(profile.strategy_id),
        "mode": profile.mode,
        "exchange_connection_id": (
            str(profile.exchange_connection_id)
            if profile.exchange_connection_id is not None
            else None
        ),
        "sizing_method": profile.sizing_method,
        "sizing_value": profile.sizing_value,
        "max_position_notional": profile.max_position_notional,
        "max_orders_per_run": profile.max_orders_per_run,
        "max_notional_per_run": profile.max_notional_per_run,
        "readiness_status": profile.readiness_status,
        "readiness_reason": profile.readiness_reason,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _map_profile(*, row: Mapping[str, Any]) -> LiveStrategyProfile:
    return LiveStrategyProfile(
        profile_id=UUID(str(row["profile_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        mode=str(row["mode"]),  # type: ignore[arg-type]
        exchange_connection_id=(
            UUID(str(row["exchange_connection_id"]))
            if row["exchange_connection_id"] is not None
            else None
        ),
        sizing_method=str(row["sizing_method"]),  # type: ignore[arg-type]
        sizing_value=_decimal(value=row["sizing_value"]),
        max_position_notional=(
            _decimal(value=row["max_position_notional"])
            if row["max_position_notional"] is not None
            else None
        ),
        max_orders_per_run=int(row["max_orders_per_run"]),
        max_notional_per_run=_decimal(value=row["max_notional_per_run"]),
        readiness_status=str(row["readiness_status"]),  # type: ignore[arg-type]
        readiness_reason=str(row["readiness_reason"]),
        created_at=_normalize_datetime(value=row["created_at"]),
        updated_at=_normalize_datetime(value=row["updated_at"]),
    )


def _decimal(*, value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _normalize_datetime(*, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyStorageError("live profile datetime is invalid")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
