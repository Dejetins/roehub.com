from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from threading import RLock
from uuid import UUID, uuid4

from trading.contexts.identity.application.ports.session_repository import (
    BrowserDevice,
    IdentitySession,
    SessionRepository,
)
from trading.shared_kernel.primitives import UserId


class InMemoryIdentitySessionRepository(SessionRepository):
    """
    InMemoryIdentitySessionRepository — deterministic in-memory session store for dev/test.

    Docs:
      - docs/architecture/identity/oidc-authentication-provider-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/session_repository.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory session storage.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Repository instance is process-local and not shared between test cases.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._sessions: dict[str, IdentitySession] = {}
        self._devices: dict[UUID, tuple[UserId, str, str, UUID]] = {}
        self._session_devices: dict[UUID, UUID] = {}
        self._device_lock = RLock()

    def bind_device(self, *, session_id: UUID, device_key: str, label: str, now: datetime) -> UUID:
        with self._device_lock:
            session = self._sessions.get(str(session_id))
            if session is None or not session.is_active_at(at=now):
                raise ValueError("inactive_session")
            if session_id in self._session_devices:
                return self._session_devices[session_id]
            device_id = next(
                (
                    key
                    for key, value in self._devices.items()
                    if value[:2] == (session.user_id, device_key)
                ),
                uuid4(),
            )
            for sid, did in self._session_devices.items():
                if did == device_id and sid != session_id:
                    self.revoke_session(session_id=sid, revoked_at=now)
            self._devices[device_id] = (session.user_id, device_key, label, session_id)
            self._session_devices[session_id] = device_id
            self._sessions[str(session_id)] = replace(
                session, last_seen_at=max(session.last_seen_at, now)
            )
            return device_id

    def list_devices(
        self, *, user_id: UserId, now: datetime, before: UUID | None, limit: int
    ) -> tuple[BrowserDevice, ...]:
        with self._device_lock:
            result = []
            for did, (owner, _key, label, sid) in sorted(self._devices.items(), reverse=True):
                if owner != user_id or (before is not None and did >= before):
                    continue
                session = self._sessions[str(sid)]
                result.append(
                    BrowserDevice(
                        did,
                        label,
                        sid,
                        session.last_seen_at,
                        min(session.idle_expires_at, session.absolute_expires_at),
                        session.is_active_at(at=now),
                    )
                )
            return tuple(result[:limit])

    def legacy_session_count(self, *, user_id: UserId, now: datetime) -> int:
        return sum(
            s.user_id == user_id
            and s.session_id not in self._session_devices
            and s.is_active_at(at=now)
            for s in self._sessions.values()
        )

    def revoke_device(self, *, user_id: UserId, device_id: UUID | None, now: datetime) -> None:
        with self._device_lock:
            for session in tuple(self._sessions.values()):
                if (
                    session.user_id == user_id
                    and session.revoked_at is None
                    and self._session_devices.get(session.session_id) == device_id
                ):
                    self.revoke_session(session_id=session.session_id, revoked_at=now)

    def touch_session(self, *, session_id: UUID, now: datetime) -> None:
        with self._device_lock:
            session = self._sessions.get(str(session_id))
            if (
                session
                and session.is_active_at(at=now)
                and now - session.last_seen_at >= timedelta(seconds=60)
            ):
                self._sessions[str(session_id)] = replace(session, last_seen_at=now)

    def create_session(
        self,
        *,
        user_id: UserId,
        now: datetime,
        idle_ttl_seconds: int,
        absolute_ttl_seconds: int,
    ) -> IdentitySession:
        """
        Create one in-memory session snapshot.

        Args:
            user_id: Stable local Roehub user identifier.
            now: Session creation timestamp in UTC.
            idle_ttl_seconds: Idle expiration budget in seconds.
            absolute_ttl_seconds: Absolute expiration budget in seconds.
        Returns:
            IdentitySession: Persisted in-memory session snapshot.
        Assumptions:
            `absolute_ttl_seconds` is greater than or equal to `idle_ttl_seconds`.
        Raises:
            ValueError: If TTL values are invalid or domain invariants fail.
        Side Effects:
            Mutates in-memory session dictionary.
        """
        _validate_session_ttls(
            idle_ttl_seconds=idle_ttl_seconds,
            absolute_ttl_seconds=absolute_ttl_seconds,
        )
        session = IdentitySession(
            session_id=uuid4(),
            user_id=user_id,
            created_at=now,
            last_seen_at=now,
            idle_expires_at=now + timedelta(seconds=idle_ttl_seconds),
            absolute_expires_at=now + timedelta(seconds=absolute_ttl_seconds),
            revoked_at=None,
        )
        with self._device_lock:
            self._sessions[str(session.session_id)] = session
        return session

    def find_by_session_id(self, *, session_id: UUID) -> IdentitySession | None:
        """
        Find one in-memory session snapshot by session id.

        Args:
            session_id: Opaque Roehub session identifier.
        Returns:
            IdentitySession | None: Stored session snapshot or `None`.
        Assumptions:
            Session id dictionary key uses canonical UUID string format.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._sessions.get(str(session_id))

    def revoke_session(self, *, session_id: UUID, revoked_at: datetime) -> IdentitySession | None:
        """
        Revoke one in-memory session snapshot.

        Args:
            session_id: Opaque Roehub session identifier.
            revoked_at: Revocation timestamp in UTC.
        Returns:
            IdentitySession | None: Updated revoked session snapshot or `None`.
        Assumptions:
            Revocation should preserve original row data for later inspection.
        Raises:
            ValueError: If revocation timestamp breaks session invariants.
        Side Effects:
            Mutates in-memory session dictionary.
        """
        with self._device_lock:
            existing_session = self._sessions.get(str(session_id))
            if existing_session is None:
                return None
            revoked_session = replace(existing_session, revoked_at=revoked_at)
            self._sessions[str(session_id)] = revoked_session
            return revoked_session

    def revoke_user_sessions(self, *, user_id: UserId, revoked_at: datetime) -> int:
        with self._device_lock:
            revoked_count = 0
            for key, session in tuple(self._sessions.items()):
                if session.user_id != user_id or session.revoked_at is not None:
                    continue
                self._sessions[key] = replace(session, revoked_at=revoked_at)
                revoked_count += 1
            return revoked_count


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
