from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentitySessionRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    session_repository,
)
from trading.shared_kernel.primitives import UserId


def test_map_identity_session_row_normalizes_non_utc_timestamps() -> None:
    """
    Verify Postgres session row mapper normalizes timezone-aware timestamps to UTC.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Psycopg row mappings may carry non-UTC timezone offsets from database driver output.
    Raises:
        AssertionError: If mapped session timestamps are not normalized to UTC.
    Side Effects:
        None.
    """
    local_tz = timezone(timedelta(hours=3))
    created_at_local = datetime(2026, 4, 22, 15, 0, 0, tzinfo=local_tz)
    session = session_repository._map_identity_session_row(
        row={
            "session_id": str(uuid4()),
            "user_id": str(uuid4()),
            "created_at": created_at_local,
            "last_seen_at": created_at_local + timedelta(minutes=5),
            "idle_expires_at": created_at_local + timedelta(minutes=35),
            "absolute_expires_at": created_at_local + timedelta(hours=12),
            "revoked_at": None,
        }
    )

    assert session.created_at.utcoffset() == timedelta(0)
    assert session.last_seen_at.utcoffset() == timedelta(0)
    assert session.idle_expires_at.utcoffset() == timedelta(0)
    assert session.absolute_expires_at.utcoffset() == timedelta(0)
    assert session.created_at == created_at_local.astimezone(timezone.utc)


def test_in_memory_session_repository_creates_active_session_with_expected_ttls() -> None:
    """
    Verify in-memory session repository creates Roehub session with idle and absolute expiry.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        In-memory repository should mirror final persisted session lifecycle semantics for tests.
    Raises:
        AssertionError: If created session shape or activity state differs from expectations.
    Side Effects:
        None.
    """
    repository = InMemoryIdentitySessionRepository()
    now = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)

    session = repository.create_session(
        user_id=UserId(uuid4()),
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )

    assert session.created_at == now
    assert session.last_seen_at == now
    assert session.idle_expires_at == now + timedelta(seconds=1800)
    assert session.absolute_expires_at == now + timedelta(seconds=43200)
    assert session.revoked_at is None
    assert session.is_active_at(at=now + timedelta(minutes=10)) is True
    assert repository.find_by_session_id(session_id=session.session_id) == session


def test_in_memory_session_repository_revokes_existing_session() -> None:
    """
    Verify in-memory session repository persists revocation timestamp for one session.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Revoked session stays readable for audit/debugging after logout.
    Raises:
        AssertionError: If revocation does not update stored session snapshot.
    Side Effects:
        None.
    """
    repository = InMemoryIdentitySessionRepository()
    now = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    session = repository.create_session(
        user_id=UserId(uuid4()),
        now=now,
        idle_ttl_seconds=1800,
        absolute_ttl_seconds=43200,
    )

    revoked = repository.revoke_session(
        session_id=session.session_id,
        revoked_at=now + timedelta(minutes=15),
    )

    assert revoked is not None
    assert revoked.revoked_at == now + timedelta(minutes=15)
    assert revoked.is_active_at(at=now + timedelta(minutes=16)) is False
    assert repository.find_by_session_id(session_id=session.session_id) == revoked


def test_postgres_session_repository_create_session_persists_ttl_fields() -> None:
    """
    Verify Postgres repository create-session path persists TTL timestamps and
    returns active snapshot.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Gateway callback returns inserted row as mapping for repository mapper.
    Raises:
        AssertionError: If SQL parameters or returned session TTL semantics differ.
    Side Effects:
        None.
    """
    now = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)

    def fetch_one_handler(
        query: str,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """
        Return deterministic inserted-row mapping for create-session SQL query.

        Args:
            query: SQL query text captured by fake gateway.
            parameters: SQL bind parameters mapping.
        Returns:
            Mapping[str, Any]: One deterministic inserted row mapping.
        Assumptions:
            Create-session repository uses INSERT with RETURNING clause.
        Raises:
            AssertionError: If query text does not target expected sessions table.
        Side Effects:
            None.
        """
        assert "INSERT INTO identity_sessions" in query
        return {
            "session_id": parameters["session_id"],
            "user_id": parameters["user_id"],
            "created_at": parameters["created_at"],
            "last_seen_at": parameters["last_seen_at"],
            "idle_expires_at": parameters["idle_expires_at"],
            "absolute_expires_at": parameters["absolute_expires_at"],
            "revoked_at": None,
        }

    gateway = _RecordingGateway(fetch_one_handler=fetch_one_handler)
    repository = session_repository.PostgresIdentitySessionRepository(gateway=gateway)

    session = repository.create_session(
        user_id=UserId(uuid4()),
        now=now,
        idle_ttl_seconds=120,
        absolute_ttl_seconds=3600,
    )

    assert gateway.last_parameters is not None
    assert gateway.last_parameters["created_at"] == now
    assert gateway.last_parameters["idle_expires_at"] == now + timedelta(seconds=120)
    assert gateway.last_parameters["absolute_expires_at"] == now + timedelta(seconds=3600)
    assert session.created_at == now
    assert session.idle_expires_at == now + timedelta(seconds=120)
    assert session.absolute_expires_at == now + timedelta(seconds=3600)
    assert session.is_active_at(at=now + timedelta(seconds=119)) is True
    assert session.is_active_at(at=now + timedelta(seconds=120)) is False


def test_postgres_session_repository_revoke_session_persists_revoked_at() -> None:
    """
    Verify Postgres repository revoke-session path stores revocation timestamp
    and deactivates session.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Gateway callback returns updated row as mapping for repository mapper.
    Raises:
        AssertionError: If SQL parameters or revocation semantics differ.
    Side Effects:
        None.
    """
    created_at = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    revoked_at = created_at + timedelta(minutes=5)
    session_id = uuid4()
    user_id = uuid4()

    def fetch_one_handler(
        query: str,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """
        Return deterministic updated-row mapping for revoke-session SQL query.

        Args:
            query: SQL query text captured by fake gateway.
            parameters: SQL bind parameters mapping.
        Returns:
            Mapping[str, Any]: One deterministic updated row mapping.
        Assumptions:
            Revoke-session repository uses UPDATE with RETURNING clause.
        Raises:
            AssertionError: If query text does not target expected sessions table.
        Side Effects:
            None.
        """
        assert "UPDATE identity_sessions" in query
        return {
            "session_id": str(session_id),
            "user_id": str(user_id),
            "created_at": created_at,
            "last_seen_at": created_at,
            "idle_expires_at": created_at + timedelta(minutes=30),
            "absolute_expires_at": created_at + timedelta(hours=12),
            "revoked_at": parameters["revoked_at"],
        }

    gateway = _RecordingGateway(fetch_one_handler=fetch_one_handler)
    repository = session_repository.PostgresIdentitySessionRepository(gateway=gateway)

    revoked_session = repository.revoke_session(
        session_id=session_id,
        revoked_at=revoked_at,
    )

    assert gateway.last_parameters is not None
    assert gateway.last_parameters["session_id"] == str(session_id)
    assert gateway.last_parameters["revoked_at"] == revoked_at
    assert revoked_session is not None
    assert revoked_session.revoked_at == revoked_at
    assert revoked_session.is_active_at(at=revoked_at) is False


class _RecordingGateway:
    """
    _RecordingGateway captures SQL calls and returns deterministic `fetch_one` mapping.
    """

    def __init__(
        self,
        *,
        fetch_one_handler: Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None],
    ) -> None:
        """
        Initialize fake gateway with one fetch-one callback.

        Args:
            fetch_one_handler: Callback returning row mapping by query/parameters.
        Returns:
            None.
        Assumptions:
            Session repository tests require only `fetch_one` behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._fetch_one_handler = fetch_one_handler
        self.last_query: str | None = None
        self.last_parameters: Mapping[str, Any] | None = None

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Capture one SQL fetch-one call and return callback-provided row mapping.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: Callback-provided row mapping.
        Assumptions:
            Caller passes deterministic SQL text and parameter mapping.
        Raises:
            None.
        Side Effects:
            Stores last query/parameters for test assertions.
        """
        self.last_query = query
        self.last_parameters = parameters
        return self._fetch_one_handler(query, parameters)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Disallow unexpected execute calls in current session repository tests.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Current tests should not use execute-only gateway calls.
        Raises:
            AssertionError: Always raised when called unexpectedly.
        Side Effects:
            None.
        """
        _ = (query, parameters)
        raise AssertionError("Unexpected execute() call in _RecordingGateway")

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Disallow unexpected fetch-all calls in current session repository tests.

        Args:
            query: SQL query text.
            parameters: SQL bind parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Always empty tuple.
        Assumptions:
            Current tests should not use fetch-all gateway calls.
        Raises:
            AssertionError: Always raised when called unexpectedly.
        Side Effects:
            None.
        """
        _ = (query, parameters)
        raise AssertionError("Unexpected fetch_all() call in _RecordingGateway")
