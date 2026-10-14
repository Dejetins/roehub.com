from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.user_repository import UserRepository
from trading.contexts.identity.domain.entities import User
from trading.shared_kernel.primitives import PaidLevel, UserId


class PostgresIdentityUserRepository(UserRepository):
    """
    PostgresIdentityUserRepository — Postgres adapter for identity user storage port.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
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
            Table has schema compatible with active identity users migration.
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



    def create_local_user(
        self,
        *,
        user_id: UserId,
        created_at: datetime,
    ) -> User:
        query = f"""
        INSERT INTO {self._users_table}
        (user_id, paid_level, created_at, last_login_at, is_deleted)
        VALUES (%(user_id)s, 'free', %(created_at)s, %(created_at)s, FALSE)
        RETURNING user_id, paid_level, created_at, last_login_at, is_deleted
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"user_id": str(user_id), "created_at": created_at},
        )
        if row is None:
            raise ValueError("PostgresIdentityUserRepository local insert returned no row")
        return _map_user_row(row=row)

    def record_local_login(
        self,
        *,
        user_id: UserId,
        login_at: datetime,
    ) -> User:
        return _update_existing_user_login(
            gateway=self._gateway,
            users_table=self._users_table,
            user_id=user_id,
            login_at=login_at,
        )


def _update_existing_user_login(
    *,
    gateway: IdentityPostgresGateway,
    users_table: str,
    user_id: UserId,
    login_at: datetime,
) -> User:
    """
    Update last-login timestamp for existing local Roehub user.

    Args:
        gateway: SQL gateway abstraction.
        users_table: Target users table name.
        user_id: Stable local Roehub user identifier.
        login_at: Current UTC login timestamp.
    Returns:
        User: Persisted user snapshot after login reactivation.
    Assumptions:
        User row already exists and is uniquely addressable by `user_id`.
    Raises:
        ValueError: If update returns no row or malformed row mapping.
    Side Effects:
        Executes one SQL UPDATE query.
    """
    query = f"""
    UPDATE {users_table}
    SET
        last_login_at = %(last_login_at)s,
        is_deleted = FALSE
    WHERE user_id = %(user_id)s
    RETURNING
        user_id,
        paid_level,
        created_at,
        last_login_at,
        is_deleted
    """
    row = gateway.fetch_one(
        query=query,
        parameters={
            "user_id": str(user_id),
            "last_login_at": login_at,
        },
    )
    if row is None:
        raise ValueError("PostgresIdentityUserRepository login update returned no row")
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
            paid_level=PaidLevel(str(row["paid_level"])),
            created_at=_normalize_utc_datetime(value=row["created_at"], field_name="created_at"),
            last_login_at=_normalize_optional_utc_datetime(
                value=row["last_login_at"],
                field_name="last_login_at",
            ),
            is_deleted=bool(row["is_deleted"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PostgresIdentityUserRepository cannot map user row") from error


def _normalize_utc_datetime(*, value: Any, field_name: str) -> datetime:
    """
    Normalize DB timestamp into timezone-aware UTC datetime.

    Args:
        value: Raw timestamp value returned by psycopg row mapping.
        field_name: Logical field label used in deterministic error messages.
    Returns:
        datetime: UTC-normalized datetime value.
    Assumptions:
        PostgreSQL `timestamptz` values are timezone-aware and may use non-UTC offsets.
    Raises:
        ValueError: If value is missing timezone information or has unsupported type.
    Side Effects:
        None.
    """
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _normalize_optional_utc_datetime(*, value: Any, field_name: str) -> datetime | None:
    """
    Normalize nullable DB timestamp into optional UTC datetime.

    Args:
        value: Optional raw timestamp value returned by psycopg row mapping.
        field_name: Logical field label used in deterministic error messages.
    Returns:
        datetime | None: UTC-normalized timestamp or `None`.
    Assumptions:
        Null values are represented as `None`.
    Raises:
        ValueError: If non-null value has unsupported type or timezone shape.
    Side Effects:
        None.
    """
    if value is None:
        return None
    return _normalize_utc_datetime(value=value, field_name=field_name)
