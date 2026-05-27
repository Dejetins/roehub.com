from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import (
    StrategyExchangeBindingRepository,
)
from trading.contexts.strategy.domain.entities import StrategyExchangeBinding
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId


class PostgresStrategyExchangeBindingRepository(StrategyExchangeBindingRepository):
    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        table_name: str = "strategy_exchange_bindings",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyExchangeBindingRepository requires gateway")
        normalized_table = table_name.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategyExchangeBindingRepository requires table")
        self._gateway = gateway
        self._table_name = normalized_table

    def create(
        self, *, binding: StrategyExchangeBinding
    ) -> StrategyExchangeBinding | None:
        active = self._gateway.fetch_one(
            query=f"""
            SELECT binding_id
            FROM {self._table_name}
            WHERE owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
              AND exchange_connection_id = %(exchange_connection_id)s
              AND usage_mode = %(usage_mode)s
              AND binding_status = 'active'
            LIMIT 1
            """,
            parameters=_binding_parameters(binding=binding),
        )
        if active is not None:
            return None
        row = self._gateway.fetch_one(
            query=f"""
            INSERT INTO {self._table_name}
            (
                binding_id,
                owner_user_id,
                strategy_id,
                exchange_connection_id,
                usage_mode,
                binding_status,
                created_at,
                updated_at,
                disabled_at,
                archived_at
            )
            VALUES
            (
                %(binding_id)s,
                %(owner_user_id)s,
                %(strategy_id)s,
                %(exchange_connection_id)s,
                %(usage_mode)s,
                %(binding_status)s,
                %(created_at)s,
                %(updated_at)s,
                %(disabled_at)s,
                %(archived_at)s
            )
            RETURNING
                binding_id,
                owner_user_id,
                strategy_id,
                exchange_connection_id,
                usage_mode,
                binding_status,
                created_at,
                updated_at,
                disabled_at,
                archived_at
            """,
            parameters=_binding_parameters(binding=binding),
        )
        if row is None:
            raise StrategyStorageError("binding insert returned no row")
        return _map_binding(row=row)

    def get(
        self, *, owner_user_id: UserId, strategy_id: UUID, binding_id: UUID
    ) -> StrategyExchangeBinding | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT
                binding_id,
                owner_user_id,
                strategy_id,
                exchange_connection_id,
                usage_mode,
                binding_status,
                created_at,
                updated_at,
                disabled_at,
                archived_at
            FROM {self._table_name}
            WHERE owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
              AND binding_id = %(binding_id)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "binding_id": str(binding_id),
            },
        )
        return _map_binding(row=row) if row is not None else None

    def list_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> tuple[StrategyExchangeBinding, ...]:
        rows = self._gateway.fetch_all(
            query=f"""
            SELECT
                binding_id,
                owner_user_id,
                strategy_id,
                exchange_connection_id,
                usage_mode,
                binding_status,
                created_at,
                updated_at,
                disabled_at,
                archived_at
            FROM {self._table_name}
            WHERE owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
            ORDER BY created_at ASC, binding_id ASC
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return tuple(_map_binding(row=row) for row in rows)

    def disable(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        binding_id: UUID,
        disabled_at: datetime,
    ) -> StrategyExchangeBinding | None:
        row = self._gateway.fetch_one(
            query=f"""
            UPDATE {self._table_name}
               SET binding_status = 'disabled',
                   updated_at = %(disabled_at)s,
                   disabled_at = %(disabled_at)s
             WHERE owner_user_id = %(owner_user_id)s
               AND strategy_id = %(strategy_id)s
               AND binding_id = %(binding_id)s
               AND binding_status = 'active'
            RETURNING
                binding_id,
                owner_user_id,
                strategy_id,
                exchange_connection_id,
                usage_mode,
                binding_status,
                created_at,
                updated_at,
                disabled_at,
                archived_at
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "binding_id": str(binding_id),
                "disabled_at": disabled_at,
            },
        )
        if row is not None:
            return _map_binding(row=row)
        return self.get(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            binding_id=binding_id,
        )


def _binding_parameters(*, binding: StrategyExchangeBinding) -> dict[str, object]:
    return {
        "binding_id": str(binding.binding_id),
        "owner_user_id": str(binding.owner_user_id),
        "strategy_id": str(binding.strategy_id),
        "exchange_connection_id": str(binding.exchange_connection_id),
        "usage_mode": binding.usage_mode,
        "binding_status": binding.binding_status,
        "created_at": binding.created_at,
        "updated_at": binding.updated_at,
        "disabled_at": binding.disabled_at,
        "archived_at": binding.archived_at,
    }


def _map_binding(*, row: Mapping[str, Any]) -> StrategyExchangeBinding:
    return StrategyExchangeBinding(
        binding_id=UUID(str(row["binding_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        usage_mode="trading",
        binding_status=str(row["binding_status"]),  # type: ignore[arg-type]
        created_at=_normalize_datetime(value=row["created_at"]),
        updated_at=_normalize_datetime(value=row["updated_at"]),
        disabled_at=_normalize_optional_datetime(value=row["disabled_at"]),
        archived_at=_normalize_optional_datetime(value=row["archived_at"]),
    )


def _normalize_datetime(*, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyStorageError("binding datetime is invalid")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_optional_datetime(*, value: object) -> datetime | None:
    if value is None:
        return None
    return _normalize_datetime(value=value)
