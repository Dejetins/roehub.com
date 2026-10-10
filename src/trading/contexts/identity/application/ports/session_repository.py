from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class IdentitySession:
    """
    IdentitySession — persisted Roehub browser session snapshot.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py
    """

    session_id: UUID
    user_id: UserId
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        """
        Validate persisted session timestamp invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            All datetime fields use timezone-aware UTC values.
        Raises:
            ValueError: If datetime ordering or timezone invariants are violated.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="created_at", value=self.created_at)
        _ensure_utc_datetime(name="last_seen_at", value=self.last_seen_at)
        _ensure_utc_datetime(name="idle_expires_at", value=self.idle_expires_at)
        _ensure_utc_datetime(name="absolute_expires_at", value=self.absolute_expires_at)
        if self.revoked_at is not None:
            _ensure_utc_datetime(name="revoked_at", value=self.revoked_at)
        if self.last_seen_at < self.created_at:
            raise ValueError("IdentitySession.last_seen_at cannot be before created_at")
        if self.idle_expires_at < self.last_seen_at:
            raise ValueError("IdentitySession.idle_expires_at cannot be before last_seen_at")
        if self.absolute_expires_at < self.idle_expires_at:
            raise ValueError("IdentitySession.absolute_expires_at cannot be before idle_expires_at")
        if self.revoked_at is not None and self.revoked_at < self.created_at:
            raise ValueError("IdentitySession.revoked_at cannot be before created_at")

    def is_active_at(self, *, at: datetime) -> bool:
        """
        Check whether session is active for one timestamp.

        Args:
            at: Point-in-time to evaluate in UTC.
        Returns:
            bool: `True` when session is neither revoked nor expired.
        Assumptions:
            Caller provides timezone-aware UTC datetime.
        Raises:
            ValueError: If `at` is not timezone-aware UTC.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="at", value=at)
        if self.revoked_at is not None and self.revoked_at <= at:
            return False
        if self.idle_expires_at <= at:
            return False
        if self.absolute_expires_at <= at:
            return False
        return True


@dataclass(frozen=True, slots=True)
class BrowserDevice:
    device_id: UUID
    label: str
    session_id: UUID  # Internal credential reference. Never serialize this field.
    last_seen_at: datetime
    expires_at: datetime
    active: bool


class SessionRepository(Protocol):
    """
    SessionRepository — port for persisted Roehub browser sessions.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py
    """

    def bind_device(self, *, session_id: UUID, device_key: str, label: str, now: datetime) -> UUID:
        """Atomically bind a live session and revoke prior logins in that browser profile."""
        ...

    def list_devices(
        self, *, user_id: UserId, now: datetime, before: UUID | None, limit: int
    ) -> tuple[BrowserDevice, ...]:
        """One row per browser profile, ordered by stable device ID descending."""
        ...

    def legacy_session_count(self, *, user_id: UserId, now: datetime) -> int:
        """Count valid sessions whose browser was never observed."""
        ...

    def revoke_device(self, *, user_id: UserId, device_id: UUID | None, now: datetime) -> None:
        """Owner-scoped, idempotent revocation; None targets unassigned legacy logins."""
        ...

    def touch_session(self, *, session_id: UUID, now: datetime) -> None:
        """Record authenticated activity at most once per minute; never extend expiry."""
        ...

    def create_session(
        self,
        *,
        user_id: UserId,
        now: datetime,
        idle_ttl_seconds: int,
        absolute_ttl_seconds: int,
    ) -> IdentitySession:
        """
        Create one persisted session for a local Roehub user.

        Args:
            user_id: Stable local Roehub user identifier.
            now: Session creation timestamp in UTC.
            idle_ttl_seconds: Idle expiration budget in seconds.
            absolute_ttl_seconds: Absolute expiration budget in seconds.
        Returns:
            IdentitySession: Persisted session snapshot.
        Assumptions:
            `absolute_ttl_seconds` is greater than or equal to `idle_ttl_seconds`.
        Raises:
            ValueError: If repository cannot persist or validate session state.
        Side Effects:
            Writes one session record into identity storage.
        """
        ...

    def find_by_session_id(self, *, session_id: UUID) -> IdentitySession | None:
        """
        Find persisted session by opaque session identifier.

        Args:
            session_id: Opaque Roehub session identifier.
        Returns:
            IdentitySession | None: Session snapshot or `None` when absent.
        Assumptions:
            Session id uniquely identifies one persisted session row.
        Raises:
            ValueError: If repository cannot map stored session row.
        Side Effects:
            None.
        """
        ...

    def revoke_session(self, *, session_id: UUID, revoked_at: datetime) -> IdentitySession | None:
        """
        Revoke persisted session for logout or administrative invalidation.

        Args:
            session_id: Opaque Roehub session identifier.
            revoked_at: Revocation timestamp in UTC.
        Returns:
            IdentitySession | None: Updated revoked session snapshot or `None` when absent.
        Assumptions:
            Revocation does not delete row history.
        Raises:
            ValueError: If repository cannot persist or map revoked session state.
        Side Effects:
            Writes one revocation update in identity storage.
        """
        ...

    def revoke_user_sessions(self, *, user_id: UserId, revoked_at: datetime) -> int:
        """Revoke every active session for recovery or account-wide invalidation."""

        ...


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone awareness and UTC offset for session datetime fields.

    Args:
        name: Field name for deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        UTC datetimes are represented with timezone info and zero offset.
    Raises:
        ValueError: If datetime is naive or not in UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
