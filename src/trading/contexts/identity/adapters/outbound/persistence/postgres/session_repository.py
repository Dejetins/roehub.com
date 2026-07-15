from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.session_repository import (
    IdentitySession,
    SessionRepository,
)
from trading.shared_kernel.primitives import UserId


class PostgresIdentitySessionRepository(SessionRepository):
    """
    PostgresIdentitySessionRepository — Postgres adapter for persisted Roehub sessions.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
      - migrations/postgres/0005_identity_keycloak_cutover_v1.sql
    """

    def __init__(
        self,
        *,
        gateway: IdentityPostgresGateway,
        sessions_table: str = "identity_sessions",
    ) -> None:
        """
        Initialize repository with SQL gateway and target sessions table.

        Args:
            gateway: SQL gateway abstraction.
            sessions_table: Target sessions table name.
        Returns:
            None.
        Assumptions:
            Table schema matches active identity sessions migration.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresIdentitySessionRepository requires gateway")
        normalized_sessions_table = sessions_table.strip()
        if not normalized_sessions_table:
            raise ValueError(
                "PostgresIdentitySessionRepository requires non-empty sessions_table"
            )

        self._gateway = gateway
        self._sessions_table = normalized_sessions_table

    def create_session(
        self,
        *,
        user_id: UserId,
        now: datetime,
        idle_ttl_seconds: int,
        absolute_ttl_seconds: int,
    ) -> IdentitySession:
        """
        Create one persisted Roehub session row in Postgres.

        Args:
            user_id: Stable local Roehub user identifier.
            now: Session creation timestamp in UTC.
            idle_ttl_seconds: Idle expiration budget in seconds.
            absolute_ttl_seconds: Absolute expiration budget in seconds.
        Returns:
            IdentitySession: Persisted session snapshot after insert.
        Assumptions:
            Session TTL values already match final runtime policy.
        Raises:
            ValueError: If TTL values are invalid or returned row mapping fails.
        Side Effects:
            Executes one SQL INSERT statement.
        """
        _validate_session_ttls(
            idle_ttl_seconds=idle_ttl_seconds,
            absolute_ttl_seconds=absolute_ttl_seconds,
        )
        idle_expires_at = now + timedelta(seconds=idle_ttl_seconds)
        absolute_expires_at = now + timedelta(seconds=absolute_ttl_seconds)
        generated_session_id = uuid4()
        query = f"""
        INSERT INTO {self._sessions_table}
        (
            session_id,
            user_id,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        )
        VALUES
        (
            %(session_id)s,
            %(user_id)s,
            %(created_at)s,
            %(last_seen_at)s,
            %(idle_expires_at)s,
            %(absolute_expires_at)s,
            NULL
        )
        RETURNING
            session_id,
            user_id,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "session_id": str(generated_session_id),
                "user_id": str(user_id),
                "created_at": now,
                "last_seen_at": now,
                "idle_expires_at": idle_expires_at,
                "absolute_expires_at": absolute_expires_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentitySessionRepository insert returned no row")
        return _map_identity_session_row(row=row)

    def find_by_session_id(self, *, session_id: UUID) -> IdentitySession | None:
        """
        Find persisted Roehub session row by session id.

        Args:
            session_id: Opaque Roehub session identifier.
        Returns:
            IdentitySession | None: Session snapshot or `None` when absent.
        Assumptions:
            Session id is unique primary key of target sessions table.
        Raises:
            ValueError: If stored row mapping fails.
        Side Effects:
            Executes one SQL SELECT statement.
        """
        query = f"""
        SELECT
            session_id,
            user_id,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        FROM {self._sessions_table}
        WHERE session_id = %(session_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"session_id": str(session_id)},
        )
        if row is None:
            return None
        return _map_identity_session_row(row=row)

    def revoke_session(self, *, session_id: UUID, revoked_at: datetime) -> IdentitySession | None:
        """
        Revoke one persisted Roehub session row in Postgres.

        Args:
            session_id: Opaque Roehub session identifier.
            revoked_at: Revocation timestamp in UTC.
        Returns:
            IdentitySession | None: Updated revoked session snapshot or `None` when absent.
        Assumptions:
            Revocation keeps original session row for auditability.
        Raises:
            ValueError: If stored row mapping fails.
        Side Effects:
            Executes one SQL UPDATE statement.
        """
        query = f"""
        UPDATE {self._sessions_table}
        SET revoked_at = %(revoked_at)s
        WHERE session_id = %(session_id)s
        RETURNING
            session_id,
            user_id,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "session_id": str(session_id),
                "revoked_at": revoked_at,
            },
        )
        if row is None:
            return None
        return _map_identity_session_row(row=row)

    def revoke_user_sessions(self, *, user_id: UserId, revoked_at: datetime) -> int:
        query = f"""
        UPDATE {self._sessions_table}
        SET revoked_at = %(revoked_at)s
        WHERE user_id = %(user_id)s AND revoked_at IS NULL
        RETURNING session_id
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={"user_id": str(user_id), "revoked_at": revoked_at},
        )
        return len(rows)


def _map_identity_session_row(*, row: Mapping[str, Any]) -> IdentitySession:
    """
    Map SQL row mapping to immutable `IdentitySession` snapshot.

    Args:
        row: SQL result mapping.
    Returns:
        IdentitySession: Persisted session snapshot.
    Assumptions:
        Row contains schema from `identity_sessions` table.
    Raises:
        ValueError: If required fields are missing or malformed.
    Side Effects:
        None.
    """
    try:
        return IdentitySession(
            session_id=UUID(str(row["session_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            created_at=_normalize_utc_datetime(value=row["created_at"], field_name="created_at"),
            last_seen_at=_normalize_utc_datetime(
                value=row["last_seen_at"],
                field_name="last_seen_at",
            ),
            idle_expires_at=_normalize_utc_datetime(
                value=row["idle_expires_at"],
                field_name="idle_expires_at",
            ),
            absolute_expires_at=_normalize_utc_datetime(
                value=row["absolute_expires_at"],
                field_name="absolute_expires_at",
            ),
            revoked_at=_normalize_optional_utc_datetime(
                value=row["revoked_at"],
                field_name="revoked_at",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PostgresIdentitySessionRepository cannot map session row") from error


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


def _validate_session_ttls(*, idle_ttl_seconds: int, absolute_ttl_seconds: int) -> None:
    """
    Validate idle and absolute TTL parameters for session creation.

    Args:
        idle_ttl_seconds: Idle expiration budget in seconds.
        absolute_ttl_seconds: Absolute expiration budget in seconds.
    Returns:
        None.
    Assumptions:
        Both TTL values are passed as integral second counts.
    Raises:
        ValueError: If TTL values are non-positive or absolute TTL is shorter than idle TTL.
    Side Effects:
        None.
    """
    if idle_ttl_seconds <= 0:
        raise ValueError("idle_ttl_seconds must be > 0")
    if absolute_ttl_seconds <= 0:
        raise ValueError("absolute_ttl_seconds must be > 0")
    if absolute_ttl_seconds < idle_ttl_seconds:
        raise ValueError("absolute_ttl_seconds must be >= idle_ttl_seconds")
