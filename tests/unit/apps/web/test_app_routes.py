from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app


def _build_test_client(*, api_result: CurrentUserApiResult | None = None) -> TestClient:
    """
    Build web TestClient with deterministic internal API client override.

    Args:
        api_result: Optional fixed result returned by mocked current-user API adapter.
    Returns:
        TestClient: Configured client with fake API adapter in app state.
    Assumptions:
        Internal API adapter exposes `fetch_current_user(cookie_header=...)`.
    Raises:
        None.
    Side Effects:
        Creates isolated FastAPI app instance for each test.
    """
    app = create_app(environ={"WEB_API_BASE_URL": "http://api.local"})
    resolved_api_result = api_result or CurrentUserApiResult(
        status_code=200,
        user=WebCurrentUser(
            user_id="00000000-0000-0000-0000-000000000321",
            paid_level="free",
        ),
        error_message=None,
    )
    app.state.current_user_api_client = SimpleNamespace(
        fetch_current_user=lambda *, cookie_header: resolved_api_result
    )
    return TestClient(app)



def test_create_app_fails_fast_when_web_api_base_url_is_missing() -> None:
    """
    Verify web app startup fails fast when `WEB_API_BASE_URL` is not configured.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime config validation executes during app factory call.
    Raises:
        AssertionError: If startup unexpectedly succeeds without required env var.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="WEB_API_BASE_URL"):
        create_app(environ={})



def test_protected_page_redirects_to_login_on_unauthorized_current_user() -> None:
    """
    Verify login gate redirects protected page requests to `/login?next=...` on 401.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Internal API client returns HTTP 401 for unauthenticated browser cookie state.
    Raises:
        AssertionError: If redirect status or target differs from contract.
    Side Effects:
        None.
    """
    client = _build_test_client(
        api_result=CurrentUserApiResult(status_code=401, user=None, error_message=None)
    )

    response = client.get("/backtests/jobs", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/login?next=%2Fbacktests%2Fjobs"



def test_login_page_sanitizes_external_next_parameter() -> None:
    """
    Verify login page strips external redirect target and keeps safe fallback `/`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Sanitized next path is serialized into inline JavaScript as `nextPath`.
    Raises:
        AssertionError: If external URL is preserved in rendered page.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/login?next=https://evil.example/path")
    match = re.search(r'const nextPath = "([^"]+)";', response.text)

    assert response.status_code == 200
    assert match is not None
    assert match.group(1) == "/"
    assert "https://evil.example/path" not in response.text



def test_logout_page_contains_api_logout_call_and_login_redirect() -> None:
    """
    Verify logout page JavaScript calls API logout endpoint and redirects to `/login`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logout flow is browser-driven JavaScript call to `/api/auth/logout`.
    Raises:
        AssertionError: If expected API path or redirect target is missing in HTML.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/logout")

    assert response.status_code == 200
    assert "/api/auth/logout" in response.text
    assert "window.location.assign('/login')" in response.text
