from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app

# WEB-EPIC-07 mapping:
# - Scope 2: smoke tests for login gate redirects and SSR pages that expose
#   required data-hooks, assets entrypoints, and /api/* literals without network I/O.


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

@pytest.mark.parametrize(
    ("path", "expected_location"),
    [
        ("/backtests", "/login?next=%2Fbacktests"),
        ("/backtests/jobs", "/login?next=%2Fbacktests%2Fjobs"),
        (
            "/backtests/jobs/00000000-0000-0000-0000-000000000777",
            "/login?next=%2Fbacktests%2Fjobs%2F00000000-0000-0000-0000-000000000777",
        ),
        ("/strategies", "/login?next=%2Fstrategies"),
        ("/strategies/new", "/login?next=%2Fstrategies%2Fnew"),
        (
            "/strategies/00000000-0000-0000-0000-000000000123",
            "/login?next=%2Fstrategies%2F00000000-0000-0000-0000-000000000123",
        ),
    ],
)
def test_protected_page_redirects_to_login_on_unauthorized_current_user(
    path: str,
    expected_location: str,
) -> None:
    """
    Verify login gate redirects protected page requests to `/login?next=...` on 401.

    Args:
        path: Protected route path requested by test client.
        expected_location: Expected redirect location with guarded `next` query.
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

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == expected_location



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


def test_strategies_list_page_renders_required_strategy_ui_hooks() -> None:
    """
    Verify `/strategies` renders list-page hooks and API paths for Strategy UI module.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authorized user receives HTML page that bootstraps browser-side API calls.
    Raises:
        AssertionError: If required hooks or API path literals are missing from SSR output.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/strategies")

    assert response.status_code == 200
    assert 'data-strategy-page="list"' in response.text
    assert "/assets/strategy_ui.js" in response.text
    assert "/strategies/new" in response.text
    assert "/api/strategies" in response.text
    assert "/api/strategies/clone" in response.text


def test_backtests_page_renders_required_backtest_ui_hooks() -> None:
    """
    Verify `/backtests` renders required hooks and API literals for sync backtest UI module.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtests page is protected SSR that performs browser-side API calls to `/api/*`.
    Raises:
        AssertionError: If required hooks/literals for preflight/run/prefill flow are missing.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/backtests")

    assert response.status_code == 200
    assert 'data-backtest-page="sync"' in response.text
    assert "/assets/backtest_ui.js" in response.text
    assert "/api/backtests" in response.text
    assert "/api/indicators/estimate" in response.text
    assert "/api/strategies" in response.text
    assert "/api/market-data/markets" in response.text
    assert "/api/market-data/instruments" in response.text
    assert "/api/indicators" in response.text
    assert "/strategies/new" in response.text
    assert "sessionStorage" in response.text
    assert "prefill" in response.text
    assert "Indicator params support both explicit values and range axes." in response.text
    assert "Source is selected from allowed values." in response.text


def test_backtest_jobs_list_page_renders_required_jobs_ui_hooks() -> None:
    """
    Verify `/backtests/jobs` renders list-page hooks and required jobs API literals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Jobs list page is SSR shell and browser performs JSON requests to `/api/backtests/jobs`.
    Raises:
        AssertionError: If required list hooks or literals are missing in rendered HTML.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/backtests/jobs")

    assert response.status_code == 200
    assert 'data-backtest-jobs-page="list"' in response.text
    assert "/assets/backtest_jobs_ui.js" in response.text
    assert "/api/backtests/jobs" in response.text
    assert "base64url(json)" in response.text
    assert "next_cursor" in response.text
    assert "Jobs disabled by config" in response.text
    assert "/backtests?run_type=job" in response.text


def test_backtest_job_details_page_renders_job_id_and_required_jobs_literals() -> None:
    """
    Verify `/backtests/jobs/{job_id}` renders details hooks and route job identifier.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Details page route is SSR-only and job payload is fetched browser-side.
    Raises:
        AssertionError: If route job id or required jobs API path literals are missing.
    Side Effects:
        None.
    """
    client = _build_test_client()
    job_id = "00000000-0000-0000-0000-000000000456"

    response = client.get(f"/backtests/jobs/{job_id}")

    assert response.status_code == 200
    assert 'data-backtest-jobs-page="details"' in response.text
    assert f'data-job-id="{job_id}"' in response.text
    assert "/api/backtests/jobs/" in response.text
    assert "/api/backtests/jobs/{job_id}/top" in response.text
    assert "/api/backtests/jobs/{job_id}/cancel" in response.text
    assert "limit=50" in response.text
    assert "Jobs disabled by config" in response.text
    assert "sessionStorage" in response.text
    assert "prefill" in response.text


def test_strategy_builder_page_renders_required_reference_api_hooks() -> None:
    """
    Verify `/strategies/new` keeps builder hooks and exposes prefill-query integration hooks.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Builder page supports optional `prefill` query parameter without changing base hooks.
    Raises:
        AssertionError: If required endpoint literals or prefill hooks are absent from SSR output.
    Side Effects:
        None.
    """
    client = _build_test_client()

    response = client.get("/strategies/new?prefill=sample-prefill-id")

    assert response.status_code == 200
    assert 'data-strategy-page="builder"' in response.text
    assert "/api/strategies" in response.text
    assert "/api/market-data/markets" in response.text
    assert "/api/market-data/instruments" in response.text
    assert "/api/indicators" in response.text
    assert 'data-prefill-query-param="prefill"' in response.text
    assert 'data-prefill-storage="sessionStorage"' in response.text
    assert "<textarea" not in response.text


def test_strategy_details_page_renders_required_strategy_id_and_hooks() -> None:
    """
    Verify `/strategies/{strategy_id}` renders details hooks with route strategy identifier.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Page receives strategy id from route and performs browser-side API loading.
    Raises:
        AssertionError: If strategy-id hook or required API literals are missing.
    Side Effects:
        None.
    """
    client = _build_test_client()
    strategy_id = "00000000-0000-0000-0000-000000000123"

    response = client.get(f"/strategies/{strategy_id}")

    assert response.status_code == 200
    assert 'data-strategy-page="details"' in response.text
    assert f'data-strategy-id="{strategy_id}"' in response.text
    assert "/api/strategies/{strategy_id}" in response.text
    assert "/api/strategies/clone" in response.text
