from __future__ import annotations

import re
from types import SimpleNamespace
from typing import cast
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app
from apps.web.main.i18n import LOCALE_COOKIE_NAME, assert_catalog_keys_match


def _build_test_client(*, api_result: CurrentUserApiResult | None = None) -> TestClient:
    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
        }
    )
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


def test_create_app_fails_fast_when_required_web_api_urls_are_missing() -> None:
    with pytest.raises(ValueError, match="WEB_API_BASE_URL"):
        create_app(environ={})

    with pytest.raises(ValueError, match="WEB_API_UPSTREAM_URL"):
        create_app(environ={"WEB_API_BASE_URL": "http://web.local"})


def test_stage_1_route_map_is_registered_without_user_badge_partial() -> None:
    client = _build_test_client()

    app = cast(FastAPI, client.app)
    route_paths = {getattr(route, "path", None) for route in app.routes}

    assert {
        "/",
        "/login",
        "/logout",
        "/register",
        "/dashboard",
        "/settings",
        "/strategies",
        "/strategies/new",
        "/strategies/{strategy_id}",
        "/monitoring",
        "/backtests",
        "/backtests/new",
        "/backtests/{job_id}",
        "/api/{upstream_path:path}",
        "/favicon.ico",
    }.issubset(route_paths)
    assert "/_partial/user_badge" not in route_paths


def test_same_origin_api_proxy_strips_prefix_and_forwards_cookie() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = request.url.query.decode()
        captured["cookie"] = request.headers.get("cookie")
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={"ok": True},
        )

    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
        }
    )
    app.state.api_proxy_transport = httpx.MockTransport(handler)
    client = TestClient(app)

    response = client.get(
        "/api/auth/current-user?verbose=1",
        headers={"cookie": "session=abc; mode=dev"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert captured["method"] == "GET"
    assert captured["path"] == "/auth/current-user"
    assert captured["query"] == "verbose=1"
    assert captured["cookie"] == "session=abc; mode=dev"


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/login",
        "/logout",
        "/register",
    ],
)
def test_public_shell_routes_render(path: str) -> None:
    client = _build_test_client()

    response = client.get(path)

    assert response.status_code == 200
    assert "Roehub" in response.text
    assert 'lang="en"' in response.text
    assert 'data-locale="en"' in response.text
    assert "/assets/vendor/htmx.min.js" in response.text
    assert "https://unpkg.com" not in response.text
    assert "terminal-orange" in response.text


@pytest.mark.parametrize(
    ("path", "expected_location"),
    [
        ("/dashboard", "/login?next=%2Fdashboard"),
        ("/settings", "/login?next=%2Fsettings"),
        ("/strategies", "/login?next=%2Fstrategies"),
        ("/strategies/new", "/login?next=%2Fstrategies%2Fnew"),
        (
            "/strategies/00000000-0000-0000-0000-000000000123",
            "/login?next=%2Fstrategies%2F00000000-0000-0000-0000-000000000123",
        ),
        ("/monitoring", "/login?next=%2Fmonitoring"),
        ("/backtests", "/login?next=%2Fbacktests"),
        ("/backtests/new", "/login?next=%2Fbacktests%2Fnew"),
        ("/backtests/job-123", "/login?next=%2Fbacktests%2Fjob-123"),
    ],
)
def test_protected_page_redirects_to_login_on_unauthorized_current_user(
    path: str,
    expected_location: str,
) -> None:
    client = _build_test_client(
        api_result=CurrentUserApiResult(status_code=401, user=None, error_message=None)
    )

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == expected_location


def test_protected_redirect_preserves_safe_local_query_in_next_parameter() -> None:
    client = _build_test_client(
        api_result=CurrentUserApiResult(status_code=401, user=None, error_message=None)
    )

    response = client.get("/settings?theme=graphite", follow_redirects=False)

    assert response.status_code == 307
    expected_location = f"/login?next={quote('/settings?theme=graphite', safe='')}"
    assert response.headers["location"] == expected_location


def test_default_locale_is_english_with_language_switcher() -> None:
    client = _build_test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert 'lang="en"' in response.text
    assert 'data-locale="en"' in response.text
    assert "Strategy and backtest operations terminal." in response.text
    assert 'data-locale-option="en"' in response.text
    assert 'data-locale-option="ru"' in response.text


def test_locale_query_selects_russian_and_sets_cookie_without_localized_routes() -> None:
    client = _build_test_client()

    response = client.get("/?locale=ru")

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Терминал управления стратегиями" in response.text
    assert f"{LOCALE_COOKIE_NAME}=ru" in response.headers["set-cookie"]
    assert 'href="/dashboard"' in response.text
    assert 'href="/strategies"' in response.text


def test_locale_cookie_selects_russian() -> None:
    client = _build_test_client()

    response = client.get("/", headers={"cookie": f"{LOCALE_COOKIE_NAME}=ru"})

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Регистрация" in response.text


def test_invalid_locale_cookie_falls_back_to_english() -> None:
    client = _build_test_client()

    response = client.get("/", headers={"cookie": f"{LOCALE_COOKIE_NAME}=javascript"})

    assert response.status_code == 200
    assert 'lang="en"' in response.text
    assert 'data-locale="en"' in response.text
    assert "Strategy and backtest operations terminal." in response.text


def test_accept_language_is_used_when_cookie_is_absent() -> None:
    client = _build_test_client()

    response = client.get("/", headers={"accept-language": "ru-RU,ru;q=0.9,en;q=0.1"})

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text


def test_locale_catalogs_have_matching_keys() -> None:
    assert_catalog_keys_match()


@pytest.mark.parametrize(
    ("path", "active_nav_key", "placeholder_id"),
    [
        ("/dashboard", "/dashboard", "dashboard"),
        ("/settings", "/settings", "settings"),
        ("/strategies", "/strategies", "strategies"),
        ("/strategies/new", "/strategies", "strategies-new"),
        ("/strategies/abc", "/strategies", "strategy-detail"),
        ("/monitoring", "/monitoring", "monitoring"),
        ("/backtests", "/backtests", "backtests"),
        ("/backtests/new", "/backtests", "backtests-new"),
        ("/backtests/job-123", "/backtests", "backtest-result"),
    ],
)
def test_authorized_protected_routes_render_placeholders_with_active_nav(
    path: str,
    active_nav_key: str,
    placeholder_id: str,
) -> None:
    client = _build_test_client()

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert f'data-placeholder-page="{placeholder_id}"' in response.text
    assert "/api/auth/current-user" in response.text
    assert _nav_item_is_active(html=response.text, nav_key=active_nav_key)


def test_login_page_sanitizes_external_next_parameter() -> None:
    client = _build_test_client()

    response = client.get("/login?next=https://evil.example/path")

    assert response.status_code == 200
    assert "/api/auth/login?next=%2F" in response.text
    assert 'id="keycloak-login-link"' in response.text
    assert "window.location.assign" not in response.text
    assert "https://evil.example/path" not in response.text


def test_register_page_uses_keycloak_backed_entrypoint_without_local_password_form() -> None:
    client = _build_test_client()

    response = client.get("/register?next=/settings")

    assert response.status_code == 200
    assert 'id="keycloak-register-link"' in response.text
    assert "/api/auth/login?next=%2Fsettings" in response.text
    assert "<form" not in response.text
    assert 'type="password"' not in response.text


@pytest.mark.parametrize(
    ("path", "expected_redirect"),
    [
        ("/logout", "/login"),
        ("/logout?next=/strategies", "/strategies"),
        ("/logout?next=https://evil.example/path", "/login"),
    ],
)
def test_logout_page_uses_external_auth_script_and_sanitized_redirect(
    path: str,
    expected_redirect: str,
) -> None:
    client = _build_test_client()

    response = client.get(path)

    assert response.status_code == 200
    assert 'action="/api/auth/logout"' in response.text
    assert 'src="/assets/js/pages/auth.js"' in response.text
    assert f'data-redirect-path="{expected_redirect}"' in response.text
    assert "fetch('/api/auth/logout'" not in response.text
    assert "https://evil.example/path" not in response.text


@pytest.mark.parametrize(
    "asset_path",
    [
        "/assets/vendor/htmx.min.js",
        "/assets/js/pages/auth.js",
        "/assets/css/tokens.css",
        "/assets/css/themes.css",
        "/assets/css/base.css",
        "/assets/css/layout.css",
        "/assets/css/shell.css",
    ],
)
def test_shell_assets_are_self_hosted(asset_path: str) -> None:
    client = _build_test_client()

    response = client.get(asset_path)

    assert response.status_code == 200
    assert response.content


def test_user_badge_partial_route_is_not_public() -> None:
    client = _build_test_client()

    response = client.get("/_partial/user_badge")

    assert response.status_code == 404


def test_favicon_route_avoids_browser_404_noise() -> None:
    client = _build_test_client()

    response = client.get("/favicon.ico")

    assert response.status_code == 204


def _nav_item_is_active(*, html: str, nav_key: str) -> bool:
    pattern = rf'data-nav-key="{re.escape(nav_key)}"[\s\S]*?data-active="true"'
    return re.search(pattern, html) is not None
