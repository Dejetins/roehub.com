from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from uuid import UUID

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_identity_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound.persistence.in_memory import (
    InMemoryIdentitySessionRepository,
    InMemoryIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user import (
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.application import IdentityClock

_KEYCLOAK_AUTH_URL = "https://auth.roehub.local/realms/roehub/protocol/openid-connect/auth"
_KEYCLOAK_TOKEN_URL = "https://auth.roehub.local/realms/roehub/protocol/openid-connect/token"
_KEYCLOAK_INTROSPECTION_URL = (
    "https://auth.roehub.local/realms/roehub/protocol/openid-connect/token/introspect"
)
_KEYCLOAK_CLIENT_ID = "roehub-api"
_KEYCLOAK_CLIENT_SECRET = "test-client-secret"
_KEYCLOAK_REDIRECT_URI = "http://127.0.0.1:8010/auth/callback"
_KEYCLOAK_LOGOUT_REDIRECT_URI = "http://127.0.0.1:8010/login"
_SESSION_COOKIE_NAME = "roehub_session_id"
_KEYCLOAK_SUBJECT = "keycloak-subject-1"
_BASE_NOW = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)


class _MutableClock(IdentityClock):
    """
    Mutable deterministic UTC clock for identity route tests.
    """

    def __init__(self, *, now_value: datetime) -> None:
        """
        Initialize deterministic clock with initial UTC value.

        Args:
            now_value: Initial timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Tests advance time explicitly with `set_now`.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            None.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def set_now(self, *, now_value: datetime) -> None:
        """
        Replace deterministic clock value.

        Args:
            now_value: New timezone-aware UTC datetime.
        Returns:
            None.
        Assumptions:
            Tests control timeline for login/logout deterministically.
        Raises:
            ValueError: If datetime is naive or non-UTC.
        Side Effects:
            Mutates internal clock state.
        """
        self._now_value = _ensure_utc_datetime(value=now_value, field_name="now_value")

    def now(self) -> datetime:
        """
        Return current deterministic UTC timestamp.

        Args:
            None.
        Returns:
            datetime: Current fixed UTC datetime.
        Assumptions:
            Time does not auto-progress during a test.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._now_value


def test_current_user_dependency_rejects_missing_cookie() -> None:
    """
    Verify protected endpoint returns 401 when opaque session cookie is missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Current-user dependency is backed by local Roehub session storage.
    Raises:
        AssertionError: If endpoint does not return expected 401 payload.
    Side Effects:
        None.
    """
    client, _clock, _user_repository, _session_repository = _build_identity_test_client()

    response = client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "missing_session_id",
            "message": "Session id is required",
        }
    }


def test_current_user_dependency_rejects_unknown_session_cookie() -> None:
    """
    Verify protected endpoint returns 401 for unknown persisted Roehub session id.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Session cookie format may be valid UUID even when no local session row exists.
    Raises:
        AssertionError: If endpoint accepts non-existent local session.
    Side Effects:
        None.
    """
    client, _clock, _user_repository, _session_repository = _build_identity_test_client()
    client.cookies.set(_SESSION_COOKIE_NAME, "00000000-0000-0000-0000-000000000001")

    response = client.get("/auth/current-user")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "session_not_found",
            "message": "Session is not found",
        }
    }


def test_get_auth_login_redirects_to_keycloak_and_sets_state_cookie() -> None:
    """
    Verify login endpoint redirects to Keycloak authorize URL and writes state cookies.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Login flow uses OIDC code grant and callback state correlation.
    Raises:
        AssertionError: If redirect query or state cookies are missing.
    Side Effects:
        None.
    """
    client, _clock, _user_repository, _session_repository = _build_identity_test_client()

    response = client.get("/auth/login?next=/strategies", follow_redirects=False)

    assert response.status_code == 307
    location = response.headers["location"]
    parsed_location = urlparse(location)
    assert f"{parsed_location.scheme}://{parsed_location.netloc}{parsed_location.path}" == (
        _KEYCLOAK_AUTH_URL
    )
    query = parse_qs(parsed_location.query)
    assert query["client_id"] == [_KEYCLOAK_CLIENT_ID]
    assert query["redirect_uri"] == [_KEYCLOAK_REDIRECT_URI]
    assert query["response_type"] == ["code"]
    assert query["scope"] == ["openid profile email"]
    assert len(query["state"][0]) >= 16
    assert client.cookies.get("roehub_oidc_state") == query["state"][0]
    stored_next_cookie = client.cookies.get("roehub_oidc_next")
    assert stored_next_cookie is not None
    assert stored_next_cookie.strip('"') == "/strategies"


def test_get_auth_callback_creates_local_user_and_session_cookie() -> None:
    """
    Verify callback exchanges code, upserts local user, issues opaque session
    cookie, and current-user reads Roehub DB state.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Introspection may return provider claims, but Roehub DB remains paid-level source of truth.
    Raises:
        AssertionError: If callback fails to persist local auth state or leaks provider token.
    Side Effects:
        Performs deterministic mock token and introspection exchanges.
    """
    captured_token_form_data: dict[str, list[str]] = {}
    captured_introspection_form_data: dict[str, list[str]] = {}
    client, _clock, user_repository, session_repository = _build_identity_test_client(
        oidc_http_transport=_build_oidc_transport(
            captured_token_form_data=captured_token_form_data,
            captured_introspection_form_data=captured_introspection_form_data,
            introspection_paid_level="pro",
        )
    )
    login_response = client.get("/auth/login?next=/strategies", follow_redirects=False)
    assert login_response.status_code == 307
    state = client.cookies.get("roehub_oidc_state")
    assert state is not None

    callback_response = client.get(
        f"/auth/callback?code=test-auth-code&state={state}",
        follow_redirects=False,
    )

    assert callback_response.status_code == 307
    assert callback_response.headers["location"] == "/strategies"
    assert captured_token_form_data["grant_type"] == ["authorization_code"]
    assert captured_token_form_data["code"] == ["test-auth-code"]
    assert captured_token_form_data["client_id"] == [_KEYCLOAK_CLIENT_ID]
    assert captured_token_form_data["client_secret"] == [_KEYCLOAK_CLIENT_SECRET]
    assert captured_token_form_data["redirect_uri"] == [_KEYCLOAK_REDIRECT_URI]
    assert captured_introspection_form_data["token"] == ["oidc-access-token"]
    assert captured_introspection_form_data["token_type_hint"] == ["access_token"]
    assert client.cookies.get("roehub_oidc_state") is None
    assert client.cookies.get("roehub_oidc_next") is None

    session_cookie_value = client.cookies.get(_SESSION_COOKIE_NAME)
    assert session_cookie_value is not None
    assert session_cookie_value != "oidc-access-token"
    parsed_session_id = UUID(session_cookie_value)
    persisted_user = user_repository.find_by_keycloak_subject(
        keycloak_subject=_KEYCLOAK_SUBJECT
    )
    assert persisted_user is not None
    persisted_session = session_repository.find_by_session_id(session_id=parsed_session_id)
    assert persisted_session is not None
    assert persisted_session.user_id == persisted_user.user_id

    current_user_response = client.get("/auth/current-user")

    assert current_user_response.status_code == 200
    assert current_user_response.json() == {
        "user_id": str(persisted_user.user_id),
        "paid_level": "free",
    }


def test_post_auth_password_login_creates_local_user_and_session_cookie() -> None:
    """
    Verify same-origin password login exchanges credentials with Keycloak and issues
    only the opaque Roehub session cookie to the browser.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Keycloak remains credential authority; Roehub persists only user/session state.
    Raises:
        AssertionError: If password-login leaks provider tokens or skips introspection.
    Side Effects:
        Performs deterministic mock token and introspection exchanges.
    """
    captured_token_form_data: dict[str, list[str]] = {}
    captured_introspection_form_data: dict[str, list[str]] = {}
    client, _clock, user_repository, session_repository = _build_identity_test_client(
        oidc_http_transport=_build_oidc_transport(
            captured_token_form_data=captured_token_form_data,
            captured_introspection_form_data=captured_introspection_form_data,
            introspection_paid_level="pro",
        )
    )

    response = client.post(
        "/auth/password-login",
        json={
            "username": "quant_trader@example.com",
            "password": "secret-password",
            "next": "/dashboard",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"next": "/dashboard"}
    assert captured_token_form_data["grant_type"] == ["password"]
    assert captured_token_form_data["username"] == ["quant_trader@example.com"]
    assert captured_token_form_data["password"] == ["secret-password"]
    assert captured_token_form_data["client_id"] == [_KEYCLOAK_CLIENT_ID]
    assert captured_token_form_data["client_secret"] == [_KEYCLOAK_CLIENT_SECRET]
    assert captured_token_form_data["scope"] == ["openid profile email"]
    assert captured_introspection_form_data["token"] == ["oidc-access-token"]
    assert captured_introspection_form_data["token_type_hint"] == ["access_token"]

    session_cookie_value = client.cookies.get(_SESSION_COOKIE_NAME)
    assert session_cookie_value is not None
    assert session_cookie_value != "oidc-access-token"
    parsed_session_id = UUID(session_cookie_value)
    persisted_user = user_repository.find_by_keycloak_subject(
        keycloak_subject=_KEYCLOAK_SUBJECT
    )
    assert persisted_user is not None
    persisted_session = session_repository.find_by_session_id(session_id=parsed_session_id)
    assert persisted_session is not None
    assert persisted_session.user_id == persisted_user.user_id

    current_user_response = client.get("/auth/current-user")

    assert current_user_response.status_code == 200
    assert current_user_response.json() == {
        "user_id": str(persisted_user.user_id),
        "paid_level": "free",
    }


def test_get_auth_callback_falls_back_to_jwt_subject_when_introspection_inactive() -> None:
    """
    Verify callback can resolve user subject from access-token payload when
    introspection returns inactive token.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Access token comes from successful code exchange for confidential client.
    Raises:
        AssertionError: If callback does not create local session via fallback path.
    Side Effects:
        Performs deterministic mock token and introspection exchanges.
    """
    fallback_access_token = _build_jwt_with_subject(subject=_KEYCLOAK_SUBJECT)
    client, _clock, user_repository, session_repository = _build_identity_test_client(
        oidc_http_transport=_build_oidc_transport(
            token_access_token=fallback_access_token,
            introspection_active=False,
        )
    )
    login_response = client.get("/auth/login?next=/strategies", follow_redirects=False)
    assert login_response.status_code == 307
    state = client.cookies.get("roehub_oidc_state")
    assert state is not None

    callback_response = client.get(
        f"/auth/callback?code=test-auth-code&state={state}",
        follow_redirects=False,
    )

    assert callback_response.status_code == 307
    assert callback_response.headers["location"] == "/strategies"
    session_cookie_value = client.cookies.get(_SESSION_COOKIE_NAME)
    assert session_cookie_value is not None
    persisted_user = user_repository.find_by_keycloak_subject(
        keycloak_subject=_KEYCLOAK_SUBJECT
    )
    assert persisted_user is not None
    persisted_session = session_repository.find_by_session_id(
        session_id=UUID(session_cookie_value)
    )
    assert persisted_session is not None
    assert persisted_session.user_id == persisted_user.user_id

    current_user_response = client.get("/auth/current-user")
    assert current_user_response.status_code == 200
    assert current_user_response.json() == {
        "user_id": str(persisted_user.user_id),
        "paid_level": "free",
    }


def test_get_auth_callback_rejects_state_mismatch() -> None:
    """
    Verify callback endpoint rejects mismatched state with deterministic 401 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Callback state must match one-time state cookie value from `/auth/login`.
    Raises:
        AssertionError: If state mismatch is not rejected.
    Side Effects:
        None.
    """
    client, _clock, _user_repository, _session_repository = _build_identity_test_client()
    login_response = client.get("/auth/login?next=/strategies", follow_redirects=False)
    assert login_response.status_code == 307

    response = client.get(
        "/auth/callback?code=test-auth-code&state=wrong-state",
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "oidc_state_mismatch",
            "message": "OIDC state validation failed",
        }
    }


def test_post_auth_logout_revokes_local_session_and_clears_auth_cookie() -> None:
    """
    Verify logout endpoint revokes persisted Roehub session and clears opaque session cookie.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logout is local-session invalidation plus browser cookie cleanup.
    Raises:
        AssertionError: If persisted session stays active or cookie-clearing headers are missing.
    Side Effects:
        Performs deterministic mock token and introspection exchanges.
    """
    client, clock, _user_repository, session_repository = _build_identity_test_client(
        oidc_http_transport=_build_oidc_transport()
    )
    login_response = client.get("/auth/login?next=/strategies", follow_redirects=False)
    assert login_response.status_code == 307
    state = client.cookies.get("roehub_oidc_state")
    assert state is not None

    callback_response = client.get(
        f"/auth/callback?code=test-auth-code&state={state}",
        follow_redirects=False,
    )
    assert callback_response.status_code == 307
    session_cookie_value = client.cookies.get(_SESSION_COOKIE_NAME)
    assert session_cookie_value is not None
    parsed_session_id = UUID(session_cookie_value)

    clock.set_now(now_value=_BASE_NOW.replace(minute=5))
    response = client.post("/auth/logout")

    assert response.status_code == 204
    set_cookie_header = response.headers.get("set-cookie", "")
    assert f"{_SESSION_COOKIE_NAME}=" in set_cookie_header
    revoked_session = session_repository.find_by_session_id(session_id=parsed_session_id)
    assert revoked_session is not None
    assert revoked_session.revoked_at == clock.now()

    client.cookies.set(_SESSION_COOKIE_NAME, session_cookie_value)
    current_user_response = client.get("/auth/current-user")

    assert current_user_response.status_code == 401
    assert current_user_response.json() == {
        "detail": {
            "error": "inactive_session",
            "message": "Session is inactive",
        }
    }


def _build_identity_test_client(
    *,
    oidc_http_transport: httpx.BaseTransport | None = None,
) -> tuple[
    TestClient,
    _MutableClock,
    InMemoryIdentityUserRepository,
    InMemoryIdentitySessionRepository,
]:
    """
    Build test client with identity router and in-memory Roehub auth storage.

    Args:
        oidc_http_transport: Optional httpx transport override for token/introspection flow.
    Returns:
        tuple[
            TestClient,
            _MutableClock,
            InMemoryIdentityUserRepository,
            InMemoryIdentitySessionRepository,
        ]:
            FastAPI test client, mutable clock, local user repository, and
            local session repository.
    Assumptions:
        Test app uses final browser auth model: opaque session cookie plus local session lookup.
    Raises:
        ValueError: If dependency construction is invalid.
    Side Effects:
        Creates in-memory FastAPI application.
    """
    clock = _MutableClock(now_value=_BASE_NOW)
    user_repository = InMemoryIdentityUserRepository()
    session_repository = InMemoryIdentitySessionRepository()
    current_user_port = RoehubSessionCurrentUser(
        session_repository=session_repository,
        user_repository=user_repository,
        clock=clock,
    )
    current_user_dependency = RequireCurrentUserDependency(
        current_user=current_user_port,
        cookie_name=_SESSION_COOKIE_NAME,
    )

    app = FastAPI()
    app.include_router(
        build_identity_router(
            keycloak_auth_url=_KEYCLOAK_AUTH_URL,
            keycloak_token_url=_KEYCLOAK_TOKEN_URL,
            keycloak_introspection_url=_KEYCLOAK_INTROSPECTION_URL,
            keycloak_client_id=_KEYCLOAK_CLIENT_ID,
            keycloak_client_secret=_KEYCLOAK_CLIENT_SECRET,
            keycloak_redirect_uri=_KEYCLOAK_REDIRECT_URI,
            keycloak_logout_redirect_uri=_KEYCLOAK_LOGOUT_REDIRECT_URI,
            current_user_dependency=current_user_dependency,
            user_repository=user_repository,
            session_repository=session_repository,
            clock=clock,
            cookie_name=_SESSION_COOKIE_NAME,
            cookie_secure=False,
            session_idle_ttl_seconds=1800,
            session_absolute_ttl_seconds=43200,
            cookie_samesite="lax",
            cookie_path="/",
            oidc_http_transport=oidc_http_transport,
        )
    )
    return TestClient(app), clock, user_repository, session_repository


def _build_oidc_transport(
    *,
    captured_token_form_data: dict[str, list[str]] | None = None,
    captured_introspection_form_data: dict[str, list[str]] | None = None,
    introspection_paid_level: str = "free",
    token_access_token: str = "oidc-access-token",
    introspection_active: bool = True,
) -> httpx.MockTransport:
    """
    Build deterministic transport handling both token exchange and introspection calls.

    Args:
        captured_token_form_data: Optional container receiving token-exchange form fields.
        captured_introspection_form_data: Optional container receiving introspection form fields.
        introspection_paid_level: Paid-level claim returned by introspection payload.
        token_access_token: Access token value returned by token endpoint.
        introspection_active: Active-flag value returned by introspection payload.
    Returns:
        httpx.MockTransport: Transport returning deterministic OIDC responses by URL.
    Assumptions:
        Callback flow uses one token exchange followed by one backend introspection call.
    Raises:
        AssertionError: If request shape or target URL differs from expected contract.
    Side Effects:
        Mutates optional capture dictionaries for test assertions.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """
        Return deterministic OIDC response based on request URL.

        Args:
            request: Outbound request sent by auth callback flow.
        Returns:
            httpx.Response: Deterministic token or introspection payload.
        Assumptions:
            Requests are URL-encoded form POSTs.
        Raises:
            AssertionError: If request URL/method differs from expected contract.
        Side Effects:
            Mutates optional capture dictionaries for assertions.
        """
        assert request.method == "POST"
        if str(request.url) == _KEYCLOAK_TOKEN_URL:
            if captured_token_form_data is not None:
                captured_token_form_data.update(parse_qs(request.content.decode("utf-8")))
            return httpx.Response(
                status_code=200,
                json={
                    "access_token": token_access_token,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        if str(request.url) == _KEYCLOAK_INTROSPECTION_URL:
            if captured_introspection_form_data is not None:
                captured_introspection_form_data.update(
                    parse_qs(request.content.decode("utf-8"))
                )
            return httpx.Response(
                status_code=200,
                json={
                    "active": introspection_active,
                    "sub": _KEYCLOAK_SUBJECT,
                    "paid_level": introspection_paid_level,
                },
            )
        raise AssertionError(f"Unexpected OIDC request URL: {request.url}")

    return httpx.MockTransport(handler)


def _ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate timezone-aware UTC datetime and return the same value.

    Args:
        value: Datetime value to validate.
        field_name: Field name used in deterministic error messages.
    Returns:
        datetime: Original validated datetime.
    Assumptions:
        Route tests operate only on UTC datetimes.
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


def _build_jwt_with_subject(*, subject: str) -> str:
    """
    Build deterministic unsigned JWT-like token containing one `sub` claim.

    Args:
        subject: Subject claim value to embed in payload.
    Returns:
        str: Compact JWT string with deterministic `alg=none` header.
    Assumptions:
        Test helper token is consumed only by payload parser in callback fallback.
    Raises:
        ValueError: If subject is empty.
    Side Effects:
        None.
    """
    normalized_subject = subject.strip()
    if not normalized_subject:
        raise ValueError("_build_jwt_with_subject requires non-empty subject")
    header_segment = _encode_json_to_base64url(payload={"alg": "none", "typ": "JWT"})
    payload_segment = _encode_json_to_base64url(payload={"sub": normalized_subject})
    return f"{header_segment}.{payload_segment}.signature"


def _encode_json_to_base64url(*, payload: dict[str, str]) -> str:
    """
    Encode compact JSON payload to URL-safe base64 without padding.

    Args:
        payload: Flat JSON object encoded into one JWT segment.
    Returns:
        str: URL-safe base64 token segment without trailing `=` padding.
    Assumptions:
        Payload is JSON-serializable and deterministic for tests.
    Raises:
        ValueError: If payload is empty.
    Side Effects:
        None.
    """
    if not payload:
        raise ValueError("_encode_json_to_base64url requires non-empty payload")
    encoded_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded_bytes).decode("ascii").rstrip("=")
