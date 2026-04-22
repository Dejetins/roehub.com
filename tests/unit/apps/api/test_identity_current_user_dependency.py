from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
    _resolve_session_id,
)
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application.ports import IdentityClock
from trading.contexts.identity.application.ports.current_user import (
    CurrentUserPrincipal,
)

_SESSION_COOKIE_NAME = "roehub_session_id"
_BASE_NOW = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)


class _FixedClock(IdentityClock):
    """
    Deterministic UTC clock for current-user dependency tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize deterministic clock with one UTC timestamp.

        Args:
            now_value: Initial timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Tests provide already-normalized UTC values.
        Raises:
            ValueError: If datetime is naive or not UTC.
        Side Effects:
            None.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def now(self) -> datetime:
        """
        Return deterministic UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current fixed UTC timestamp.
        Assumptions:
            Time does not auto-progress during one test.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_resolve_session_id_reads_cookie_value() -> None:
    """
    Verify helper resolves opaque session id from configured cookie.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Browser auth path stores Roehub session id only in cookie.
    Raises:
        AssertionError: If helper fails to read normalized cookie value.
    Side Effects:
        None.
    """
    request = _build_request(
        cookie_header=f"{_SESSION_COOKIE_NAME}=session-123",
        authorization_header=None,
    )

    resolved = _resolve_session_id(
        request=request,
        cookie_name=_SESSION_COOKIE_NAME,
    )

    assert resolved == "session-123"


def test_resolve_session_id_ignores_authorization_header_without_cookie() -> None:
    """
    Verify helper does not fallback to Authorization header on browser path.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Final web auth model is cookie-only for protected browser/API requests.
    Raises:
        AssertionError: If helper still accepts bearer header fallback.
    Side Effects:
        None.
    """
    request = _build_request(
        cookie_header=None,
        authorization_header="Bearer provider-token",
    )

    resolved = _resolve_session_id(
        request=request,
        cookie_name=_SESSION_COOKIE_NAME,
    )

    assert resolved is None


def test_dependency_returns_principal_from_local_session_and_user_snapshot() -> None:
    """
    Verify dependency resolves principal through Roehub session and local user repositories.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        User snapshot is created on successful Keycloak login before protected request.
    Raises:
        AssertionError: If principal payload does not match persisted local state.
    Side Effects:
        None.
    """
    client, session_id, user_id = _build_dependency_test_client()
    client.cookies.set(_SESSION_COOKIE_NAME, session_id)

    response = client.get("/protected")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": user_id,
        "paid_level": "free",
    }


def test_dependency_rejects_missing_cookie_with_deterministic_payload() -> None:
    """
    Verify dependency returns 401 with missing-session error when cookie is absent.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Protected endpoints must reject unauthenticated requests before route logic runs.
    Raises:
        AssertionError: If error payload differs from stable contract.
    Side Effects:
        None.
    """
    client, _session_id, _user_id = _build_dependency_test_client()

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "missing_session_id",
            "message": "Session id is required",
        }
    }


def test_dependency_rejects_invalid_session_id_with_deterministic_payload() -> None:
    """
    Verify dependency returns 401 when cookie contains malformed session identifier.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Roehub session id format is UUID and should fail fast on malformed values.
    Raises:
        AssertionError: If invalid session id does not map to stable 401 payload.
    Side Effects:
        None.
    """
    client, _session_id, _user_id = _build_dependency_test_client()
    client.cookies.set(_SESSION_COOKIE_NAME, "not-a-uuid")

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "invalid_session_id",
            "message": "Session id must be UUID",
        }
    }


def test_dependency_rejects_expired_session_with_deterministic_payload() -> None:
    """
    Verify dependency returns 401 when local session is expired by TTL policy.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Session activity is evaluated against persisted idle and absolute expiry timestamps.
    Raises:
        AssertionError: If expired session is not treated as unauthorized.
    Side Effects:
        None.
    """
    client, session_id, _user_id = _build_dependency_test_client(
        clock_now=_BASE_NOW + timedelta(minutes=20),
        session_created_at=_BASE_NOW,
        idle_ttl_seconds=300,
        absolute_ttl_seconds=600,
    )
    client.cookies.set(_SESSION_COOKIE_NAME, session_id)

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "inactive_session",
            "message": "Session is inactive",
        }
    }


def test_dependency_rejects_revoked_session_with_deterministic_payload() -> None:
    """
    Verify dependency returns 401 when local session was explicitly revoked.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logout revokes local Roehub session before protected request is evaluated again.
    Raises:
        AssertionError: If revoked session is still treated as active.
    Side Effects:
        None.
    """
    client, session_id, _user_id = _build_dependency_test_client(
        revoked_at=_BASE_NOW + timedelta(minutes=2),
        clock_now=_BASE_NOW + timedelta(minutes=2),
    )
    client.cookies.set(_SESSION_COOKIE_NAME, session_id)

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "inactive_session",
            "message": "Session is inactive",
        }
    }


def _build_dependency_test_client(
    *,
    clock_now: datetime = _BASE_NOW + timedelta(minutes=1),
    session_created_at: datetime = _BASE_NOW,
    idle_ttl_seconds: int = 1800,
    absolute_ttl_seconds: int = 43200,
    revoked_at: datetime | None = None,
) -> tuple[TestClient, str, str]:
    """
    Build FastAPI test client backed by in-memory Roehub user and session repositories.

    Args:
        clock_now: UTC timestamp used by dependency for session activity evaluation.
        session_created_at: UTC timestamp used when initial session is persisted.
        idle_ttl_seconds: Session idle TTL in seconds.
        absolute_ttl_seconds: Session absolute TTL in seconds.
        revoked_at: Optional UTC timestamp used to revoke persisted session before request.
    Returns:
        tuple[TestClient, str, str]: Test client, persisted session id, and local user id.
    Assumptions:
        Route shape is minimal and exists only to exercise dependency behavior.
    Raises:
        ValueError: If provided datetimes are naive or not UTC.
    Side Effects:
        Creates in-memory FastAPI application and repositories.
    """
    validated_clock_now = _ensure_utc_datetime(value=clock_now, field_name="clock_now")
    validated_session_created_at = _ensure_utc_datetime(
        value=session_created_at,
        field_name="session_created_at",
    )
    validated_revoked_at = (
        None
        if revoked_at is None
        else _ensure_utc_datetime(value=revoked_at, field_name="revoked_at")
    )
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()
    user = user_repository.upsert_keycloak_login(
        keycloak_subject="keycloak-subject-1",
        login_at=validated_session_created_at,
    )
    session = session_repository.create_session(
        user_id=user.user_id,
        now=validated_session_created_at,
        idle_ttl_seconds=idle_ttl_seconds,
        absolute_ttl_seconds=absolute_ttl_seconds,
    )
    if validated_revoked_at is not None:
        session_repository.revoke_session(
            session_id=session.session_id,
            revoked_at=validated_revoked_at,
        )

    current_user = RoehubSessionCurrentUser(
        session_repository=session_repository,
        user_repository=user_repository,
        clock=_FixedClock(now_value=validated_clock_now),
    )
    dependency = RequireCurrentUserDependency(
        current_user=current_user,
        cookie_name=_SESSION_COOKIE_NAME,
    )

    app = FastAPI()

    @app.get("/protected")
    def get_protected(
        principal: CurrentUserPrincipal = Depends(dependency),
    ) -> dict[str, str]:
        """
        Return serialized authenticated principal for dependency assertions.

        Args:
            principal: Authenticated principal resolved by dependency.
        Returns:
            dict[str, str]: Minimal JSON payload derived from authenticated principal.
        Assumptions:
            Dependency already mapped unauthorized states to HTTP 401.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "user_id": str(principal.user_id),
            "paid_level": str(principal.paid_level),
        }

    return TestClient(app), str(session.session_id), str(user.user_id)


def _build_request(
    *,
    cookie_header: str | None,
    authorization_header: str | None,
) -> Request:
    """
    Build minimal Starlette request with optional Cookie and Authorization headers.

    Args:
        cookie_header: Raw Cookie header value or `None`.
        authorization_header: Raw Authorization header value or `None`.
    Returns:
        Request: Deterministic request object used by dependency helper tests.
    Assumptions:
        Header values are already normalized as plain strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    headers: list[tuple[bytes, bytes]] = []
    if cookie_header is not None:
        headers.append((b"cookie", cookie_header.encode("utf-8")))
    if authorization_header is not None:
        headers.append((b"authorization", authorization_header.encode("utf-8")))
    return Request(
        scope={
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
        }
    )


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate timezone-aware UTC datetime and return the normalized value.

    Args:
        value: Datetime value to validate.
        field_name: Field name used in deterministic error message.
    Returns:
        datetime: Original datetime after UTC validation.
    Assumptions:
        Tests operate only on timezone-aware UTC datetimes.
    Raises:
        ValueError: If datetime is naive or not UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must be UTC datetime")
    return value
