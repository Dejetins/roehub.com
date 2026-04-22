from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

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
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
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
        keycloak_subject_column: str = "keycloak_subject",
    ) -> None:
        """
        Initialize repository with SQL gateway and target users table.

        Args:
            gateway: SQL gateway abstraction.
            users_table: Target users table name.
            keycloak_subject_column: Column storing external Keycloak subject binding.
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
        normalized_keycloak_subject_column = keycloak_subject_column.strip()
        if not normalized_table:
            raise ValueError("PostgresIdentityUserRepository requires non-empty users_table")
        if not normalized_keycloak_subject_column:
            raise ValueError(
                "PostgresIdentityUserRepository requires non-empty keycloak_subject_column"
            )

        self._gateway = gateway
        self._users_table = normalized_table
        self._keycloak_subject_column = normalized_keycloak_subject_column

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

    def find_by_keycloak_subject(self, *, keycloak_subject: str) -> User | None:
        """
        Find identity user by external Keycloak subject binding.

        Args:
            keycloak_subject: Opaque external subject from Keycloak.
        Returns:
            User | None: User snapshot or None when subject is absent.
        Assumptions:
            Keycloak subject column is unique in active schema.
        Raises:
            ValueError: If subject is blank or row mapping is malformed.
        Side Effects:
            Executes one SQL SELECT query.
        """
        normalized_keycloak_subject = _normalize_keycloak_subject(
            keycloak_subject=keycloak_subject
        )
        query = f"""
        SELECT
            user_id,
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        FROM {self._users_table}
        WHERE {self._keycloak_subject_column} = %(keycloak_subject)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"keycloak_subject": normalized_keycloak_subject},
        )
        if row is None:
            return None
        return _map_user_row(row=row)

    def upsert_keycloak_login(
        self,
        *,
        keycloak_subject: str,
        login_at: datetime,
    ) -> User:
        """
        Create or update local Roehub user by external Keycloak subject.

        Args:
            keycloak_subject: Opaque external subject from Keycloak.
            login_at: Current UTC login timestamp.
        Returns:
            User: Persisted user snapshot after upsert.
        Assumptions:
            Schema contains `keycloak_subject` column before this method is called in runtime.
        Raises:
            ValueError: If returned row is missing required columns.
        Side Effects:
            Executes one SQL upsert statement with possible row insert/update.
        """
        normalized_keycloak_subject = _normalize_keycloak_subject(
            keycloak_subject=keycloak_subject
        )
        existing_user = self.find_by_keycloak_subject(
            keycloak_subject=normalized_keycloak_subject
        )
        if existing_user is not None:
            return _update_existing_user_login(
                gateway=self._gateway,
                users_table=self._users_table,
                user_id=existing_user.user_id,
                login_at=login_at,
            )

        generated_user_id = UserId(uuid4())
        query = f"""
        INSERT INTO {self._users_table}
        (
            user_id,
            {self._keycloak_subject_column},
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        )
        VALUES
        (
            %(user_id)s,
            %(keycloak_subject)s,
            %(paid_level)s,
            %(created_at)s,
            %(last_login_at)s,
            FALSE
        )
        RETURNING
            user_id,
            paid_level,
            created_at,
            last_login_at,
            is_deleted
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(generated_user_id),
                "keycloak_subject": normalized_keycloak_subject,
                "paid_level": str(PaidLevel.free()),
                "created_at": login_at,
                "last_login_at": login_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentityUserRepository upsert returned no row")
        return _map_user_row(row=row)


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


def _normalize_keycloak_subject(*, keycloak_subject: str) -> str:
    """
    Normalize external Keycloak subject value for repository queries.

    Args:
        keycloak_subject: Raw Keycloak `sub` value.
    Returns:
        str: Non-empty stripped subject string.
    Assumptions:
        Subject is opaque and must not be transformed beyond whitespace trim.
    Raises:
        ValueError: If subject is blank after normalization.
    Side Effects:
        None.
    """
    normalized_subject = keycloak_subject.strip()
    if not normalized_subject:
        raise ValueError("keycloak_subject must be non-empty")
    return normalized_subject
