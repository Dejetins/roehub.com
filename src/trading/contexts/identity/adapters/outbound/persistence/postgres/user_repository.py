from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.contexts.identity.domain.entities import User
from trading.contexts.identity.domain.value_objects import TelegramUserId
from trading.shared_kernel.primitives import PaidLevel, UserId


class PostgresIdentityUserRepository(UserRepository):
    """
    PostgresIdentityUserRepository — Postgres adapter for identity user storage port.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/user_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
      - migrations/postgres/0001_identity_v1.sql
    """

    def __init__(
        self,
        *,
        gateway: IdentityPostgresGateway,
        users_table: str = "identity_users",
    ) -> None:
        """
        Initialize repository with SQL gateway and target users table.

        Args:
            gateway: SQL gateway abstraction.
            users_table: Target users table name.
        Returns:
            None.
        Assumptions:
            Table has schema compatible with migration `0001_identity_v1.sql`.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresIdentityUserRepository requires gateway")
        normalized_table = users_table.strip()
        if not normalized_table:
            raise ValueError("PostgresIdentityUserRepository requires non-empty users_table")

        self._gateway = gateway
        self._users_table = normalized_table

    def find_by_user_id(self, *, user_id: UserId) -> User | None:
        """
        Find identity user by stable user id.

        Args:
            user_id: Stable user identifier.
        Returns:
            User | None: User snapshot or None when user is absent.
        Assumptions:
            `user_id` column is primary key.
        Raises:
            ValueError: If row mapping is malformed.
        Side Effects:
            Executes one SQL SELECT query.
        """
        query = f"""
        SELECT
            user_id,
            telegram_user_id,
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        FROM {self._users_table}
        WHERE user_id = %(user_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"user_id": str(user_id)},
        )
        if row is None:
            return None
        return _map_user_row(row=row)

    def upsert_telegram_login(
        self,
        *,
        telegram_user_id: TelegramUserId,
        login_at: datetime,
    ) -> User:
        """
        Create or update user by Telegram id and update last login timestamp.

        Args:
            telegram_user_id: Telegram identity key.
            login_at: Current UTC login timestamp.
        Returns:
            User: Persisted user snapshot after upsert.
        Assumptions:
            `telegram_user_id` has unique index.
        Raises:
            ValueError: If returned row is missing required columns.
        Side Effects:
            Executes one SQL upsert statement with possible row insert/update.
        """
        query = f"""
        INSERT INTO {self._users_table}
        (
            user_id,
            telegram_user_id,
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        )
        VALUES
        (
            %(user_id)s,
            %(telegram_user_id)s,
            %(paid_level)s,
            %(created_at)s,
            %(last_login_at)s,
            FALSE
        )
        ON CONFLICT (telegram_user_id)
        DO UPDATE
        SET
            last_login_at = EXCLUDED.last_login_at,
            is_deleted = FALSE
        RETURNING
            user_id,
            telegram_user_id,
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(uuid4()),
                "telegram_user_id": telegram_user_id.value,
                "paid_level": str(PaidLevel.free()),
                "created_at": login_at,
                "last_login_at": login_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentityUserRepository upsert returned no row")
        return _map_user_row(row=row)



def _map_user_row(*, row: Mapping[str, Any]) -> User:
    """
    Map SQL row mapping to immutable domain `User` entity.

    Args:
        row: SQL result mapping.
    Returns:
        User: Domain user entity.
    Assumptions:
        Row contains schema from `identity_users` table.
    Raises:
        ValueError: If required fields are missing or malformed.
    Side Effects:
        None.
    """
    try:
        return User(
            user_id=UserId.from_string(str(row["user_id"])),
            telegram_user_id=TelegramUserId(int(row["telegram_user_id"])),
            paid_level=PaidLevel(str(row["paid_level"])),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            is_deleted=bool(row["is_deleted"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PostgresIdentityUserRepository cannot map user row") from error
