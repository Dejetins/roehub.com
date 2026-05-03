from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.web.main.api_client import (
    AccountPreferencesApiResult,
    CurrentUserApiResult,
    WebAccountPreferences,
    WebCurrentUser,
)
from apps.web.main.app import create_app
from apps.web.main.i18n import LOCALE_COOKIE_NAME, assert_catalog_keys_match

REPO_ROOT = Path(__file__).resolve().parents[4]
WEB_DIST_ROOT = REPO_ROOT / "apps" / "web" / "dist"


def _build_test_client(
    *,
    api_result: CurrentUserApiResult | None = None,
    preferences_result: AccountPreferencesApiResult | None = None,
) -> TestClient:
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
    resolved_preferences_result = preferences_result or AccountPreferencesApiResult(
        status_code=200,
        preferences=WebAccountPreferences(
            theme="terminal-orange",
            locale="en",
            density="compact",
        ),
        error_message=None,
    )
    app.state.account_preferences_api_client = SimpleNamespace(
        fetch_preferences=lambda *, cookie_header: resolved_preferences_result
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
    assert "/assets/css/components.css" in response.text
    assert "/assets/js/components/shell.js" in response.text
    assert "/assets/site.css" not in response.text
    assert "https://unpkg.com" not in response.text
    assert "terminal-orange" in response.text


def test_stage_3_landing_route_renders_without_current_user_api_dependency() -> None:
    def fail_current_user_call(*, cookie_header: str | None) -> CurrentUserApiResult:
        raise AssertionError("landing must not call current-user API")

    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
        }
    )
    app.state.current_user_api_client = SimpleNamespace(fetch_current_user=fail_current_user_call)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'data-landing-page' in response.text
    assert "ROEHUB" in response.text
    assert 'href="/register"' in response.text
    assert 'href="/login"' in response.text
    assert 'id="platform-map"' in response.text
    assert "/assets/css/pages/landing.css" in response.text
    assert "/assets/js/pages/landing.js" in response.text
    assert "/api/auth/current-user" not in response.text


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
    assert "Built for systematic traders" in response.text
    assert "Get started free" in response.text
    assert 'data-locale-option="en"' in response.text
    assert 'data-locale-option="ru"' in response.text


def test_locale_query_selects_russian_and_sets_cookie_without_localized_routes() -> None:
    client = _build_test_client()

    response = client.get("/?locale=ru")

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Для системных трейдеров" in response.text
    assert "Начать бесплатно" in response.text
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
    assert "Built for systematic traders" in response.text


def test_accept_language_is_used_when_cookie_is_absent() -> None:
    client = _build_test_client()

    response = client.get("/", headers={"accept-language": "ru-RU,ru;q=0.9,en;q=0.1"})

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text


def test_locale_catalogs_have_matching_keys() -> None:
    assert_catalog_keys_match()


def test_stage_7_monitoring_route_renders_live_page_with_api_and_stream_contracts() -> None:
    client = _build_test_client()

    response = client.get("/monitoring")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "data-monitoring-page" in response.text
    assert 'data-monitor-endpoint="/api/ui/strategies/monitor"' in response.text
    assert 'data-stream-path="/api/stream/strategies"' in response.text
    assert 'data-run-path-template="/api/strategies/{strategy_id}/run"' in response.text
    assert 'data-stop-path-template="/api/strategies/{strategy_id}/stop"' in response.text
    assert (
        'data-snapshot-path-template="/api/ui/strategies/{strategy_id}/snapshot"'
        in response.text
    )
    assert (
        'data-positions-path-template="/api/ui/strategies/{strategy_id}/positions?limit=50"'
        in response.text
    )
    assert (
        'data-fills-path-template="/api/ui/strategies/{strategy_id}/fills?limit=50"'
        in response.text
    )
    assert "/assets/css/pages/monitoring.css" in response.text
    assert "/assets/js/pages/monitoring.js" in response.text
    assert "data-ui-kit-placeholder" not in response.text
    assert _nav_item_is_active(html=response.text, nav_key="/monitoring")


def test_stage_7_monitoring_route_has_russian_locale_copy() -> None:
    client = _build_test_client()

    response = client.get("/monitoring?locale=ru")

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Мониторинг" in response.text
    assert "Загрузка" in response.text
    assert "/api/stream/strategies" in response.text


def test_backtest_result_route_renders_result_page_hooks_without_initial_trades() -> None:
    client = _build_test_client()

    response = client.get("/backtests/job-123")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'data-backtest-result-page' in response.text
    assert 'data-job-id="job-123"' in response.text
    assert "/api/backtests/jobs/job-123/summary" in response.text
    assert "/api/backtests/jobs/job-123/variants/{variant_key}/equity?points=1200" in response.text
    assert "/api/backtests/jobs/job-123/variants/{variant_key}/trades.csv" in response.text
    assert "/assets/css/pages/backtests.css" in response.text
    assert "/assets/js/pages/backtests_result.js" in response.text
    assert 'data-placeholder-page="backtest-result"' not in response.text
    assert '"trades":' not in response.text
    assert _nav_item_is_active(html=response.text, nav_key="/backtests")


def test_authorized_dashboard_route_renders_stage_4_page_with_active_nav() -> None:
    client = _build_test_client()

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'data-dashboard' in response.text
    assert 'data-dashboard-endpoint="/api/ui/dashboard/summary"' in response.text
    assert 'data-poll-interval-ms="12000"' in response.text
    assert "/assets/css/pages/dashboard.css" in response.text
    assert "/assets/js/pages/dashboard.js" in response.text
    assert "/api/ui/dashboard/summary" in response.text
    assert "data-ui-kit-placeholder" not in response.text
    assert _nav_item_is_active(html=response.text, nav_key="/dashboard")


def test_authorized_settings_route_renders_stage_5_account_page_with_preferences() -> None:
    client = _build_test_client(
        preferences_result=AccountPreferencesApiResult(
            status_code=200,
            preferences=WebAccountPreferences(
                theme="graphite",
                locale="ru",
                density="comfortable",
            ),
            error_message=None,
        )
    )

    response = client.get("/settings")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert 'data-theme="graphite"' in response.text
    assert 'data-settings' in response.text
    assert 'data-preferences-endpoint="/api/ui/account/preferences"' in response.text
    assert 'data-exchange-keys-endpoint="/api/exchange-keys"' in response.text
    assert "/assets/css/pages/settings.css" in response.text
    assert "/assets/js/pages/settings.js" in response.text
    assert "data-ui-kit-placeholder" not in response.text
    assert "route-secret" not in response.text
    assert 'name="api_secret"' in response.text
    assert 'name="api_secret" value=' not in response.text
    assert _nav_item_is_active(html=response.text, nav_key="/settings")


@pytest.mark.parametrize(
    ("path", "page_type", "expected_text"),
    [
        ("/strategies", "list", "Strategy library"),
        ("/strategies/new", "create", "New strategy"),
        ("/strategies/abc", "detail", "Strategy detail"),
    ],
)
def test_stage_6_strategy_routes_render_new_pages_without_old_strategy_asset(
    path: str,
    page_type: str,
    expected_text: str,
) -> None:
    client = _build_test_client()

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert f'data-strategy-page="{page_type}"' in response.text
    assert expected_text in response.text
    assert 'src="/assets/js/pages/strategies.js"' in response.text
    assert 'href="/assets/css/pages/strategies.css"' in response.text
    assert "/assets/strategy_ui.js" not in response.text
    assert "data-ui-kit-placeholder" not in response.text
    assert _nav_item_is_active(html=response.text, nav_key="/strategies")


def test_stage_6_strategy_create_route_has_russian_locale_copy() -> None:
    client = _build_test_client()

    response = client.get("/strategies/new?locale=ru")

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Новая стратегия" in response.text
    assert "Блоки индикаторов" in response.text
    assert 'data-api-create-path="/api/strategies"' in response.text


def test_stage_6_strategy_pages_use_existing_strategy_api_paths() -> None:
    client = _build_test_client()

    list_response = client.get("/strategies")
    create_response = client.get("/strategies/new")
    detail_response = client.get("/strategies/strategy-123")

    assert 'data-api-list-path="/api/strategies"' in list_response.text
    assert 'data-api-clone-path="/api/strategies/clone"' in list_response.text
    assert 'data-api-delete-path-template="/api/strategies/{strategy_id}"' in list_response.text
    assert 'data-api-create-path="/api/strategies"' in create_response.text
    assert 'data-api-get-path-template="/api/strategies/{strategy_id}"' in detail_response.text
    assert 'data-api-clone-path="/api/strategies/clone"' in detail_response.text
    assert 'data-api-delete-path-template="/api/strategies/{strategy_id}"' in detail_response.text


def test_stage_8_backtests_routes_render_history_and_configurator_split() -> None:
    client = _build_test_client()

    history_response = client.get("/backtests")
    run_response = client.get("/backtests/new")

    assert history_response.status_code == 200
    assert run_response.status_code == 200
    assert history_response.headers["cache-control"] == "no-store"
    assert run_response.headers["cache-control"] == "no-store"

    assert "data-backtests-history-page" in history_response.text
    assert 'data-api-jobs-path="/api/backtests/jobs"' in history_response.text
    assert 'data-api-counters-path="/api/ui/backtests/counters"' in history_response.text
    assert "/assets/js/pages/backtests_history.js" in history_response.text
    assert "/assets/css/pages/backtests.css" in history_response.text
    assert "/assets/backtest_ui.js" not in history_response.text
    assert "/api/backtests/jobs/{job_id}/top" not in history_response.text
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades" not in history_response.text

    assert "data-backtests-run-page" in run_response.text
    assert 'data-api-defaults-path="/api/backtests/runtime-defaults"' in run_response.text
    assert 'data-api-preflight-path="/api/backtests/preflight"' in run_response.text
    assert 'data-api-jobs-path="/api/backtests/jobs"' in run_response.text
    assert 'data-api-markets-path="/api/market-data/markets"' in run_response.text
    assert 'data-api-indicators-path="/api/indicators"' in run_response.text
    assert "/assets/js/pages/backtests_run.js" in run_response.text
    assert "/assets/css/pages/backtests.css" in run_response.text
    assert "backtest_presets" in run_response.text
    assert "/assets/backtest_ui.js" not in run_response.text
    assert "/api/backtests/jobs/{job_id}/top" not in run_response.text
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades" not in run_response.text
    assert _nav_item_is_active(html=history_response.text, nav_key="/backtests")
    assert _nav_item_is_active(html=run_response.text, nav_key="/backtests")


def test_stage_8_backtest_configurator_has_russian_locale_copy() -> None:
    client = _build_test_client()

    response = client.get("/backtests/new?locale=ru")

    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert 'data-locale="ru"' in response.text
    assert "Конфигуратор" in response.text
    assert "Сборка валидированного request draft" in response.text
    assert 'data-api-jobs-path="/api/backtests/jobs"' in response.text


def test_stage_2_shell_exposes_theme_and_locale_client_hooks() -> None:
    client = _build_test_client()

    response = client.get("/monitoring")

    assert response.status_code == 200
    assert 'data-theme="terminal-orange"' in response.text
    assert 'data-theme-control' in response.text
    assert 'data-theme-option="terminal-orange"' in response.text
    assert 'data-theme-option="graphite"' in response.text
    assert 'data-theme-option="matrix-green"' in response.text
    assert 'data-theme-option="high-contrast"' in response.text
    assert 'data-locale-control' in response.text
    assert 'data-locale-option="en"' in response.text
    assert 'data-locale-option="ru"' in response.text
    assert 'id="rh-locale-catalogs"' in response.text
    assert 'data-i18n="theme.current"' in response.text
    assert 'data-i18n="page.monitoring.title"' in response.text


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
        "/assets/css/components.css",
        "/assets/css/shell.css",
        "/assets/css/pages/dashboard.css",
        "/assets/css/pages/landing.css",
        "/assets/css/pages/monitoring.css",
        "/assets/css/pages/settings.css",
        "/assets/css/pages/strategies.css",
        "/assets/css/pages/backtests.css",
        "/assets/js/core/api.js",
        "/assets/js/core/poller.js",
        "/assets/js/core/sse.js",
        "/assets/js/core/dom.js",
        "/assets/js/core/locale.js",
        "/assets/js/core/theme.js",
        "/assets/js/core/notifications.js",
        "/assets/js/core/formatters.js",
        "/assets/js/core/validators.js",
        "/assets/js/components/shell.js",
        "/assets/js/charts/timeseries.js",
        "/assets/js/pages/dashboard.js",
        "/assets/js/pages/landing.js",
        "/assets/js/pages/monitoring.js",
        "/assets/js/pages/settings.js",
        "/assets/js/pages/strategies.js",
        "/assets/js/pages/backtests_history.js",
        "/assets/js/pages/backtests_result.js",
        "/assets/js/pages/backtests_run.js",
    ],
)
def test_shell_assets_are_self_hosted(asset_path: str) -> None:
    client = _build_test_client()

    response = client.get(asset_path)

    assert response.status_code == 200
    assert response.content


def test_stage_2_js_core_contract_literals_are_present() -> None:
    api_js = (WEB_DIST_ROOT / "js" / "core" / "api.js").read_text(encoding="utf-8")
    poller_js = (WEB_DIST_ROOT / "js" / "core" / "poller.js").read_text(encoding="utf-8")
    locale_js = (WEB_DIST_ROOT / "js" / "core" / "locale.js").read_text(encoding="utf-8")

    assert 'credentials: "include"' in api_js
    assert "redirectToLogin" in api_js
    assert "validation_error" in api_js
    assert "conflict" in api_js
    assert "forbidden" in api_js
    assert "unauthorized" in api_js
    assert "timeout" in api_js
    assert "setCsrfTokenProvider" in api_js
    assert "X-CSRF-Token" in api_js

    assert "this.inFlight" in poller_js
    assert "this.documentRef.hidden" in poller_js
    assert "hiddenIntervalMs ?? 5000" in poller_js

    assert 'DEFAULT_LOCALE = "en"' in locale_js
    assert '"ru"' in locale_js
    assert "roehub_locale" in locale_js
    assert "data-locale-option" in locale_js


def test_stage_4_dashboard_js_uses_one_summary_request_and_safe_poller() -> None:
    dashboard_js = (WEB_DIST_ROOT / "js" / "pages" / "dashboard.js").read_text(
        encoding="utf-8"
    )

    assert 'DEFAULT_ENDPOINT = "/api/ui/dashboard/summary"' in dashboard_js
    assert "DEFAULT_POLL_INTERVAL_MS = 12000" in dashboard_js
    assert "createPoller" in dashboard_js
    assert "apiRequest(endpoint" in dashboard_js
    assert "redirectOnUnauthorized: true" in dashboard_js
    assert "innerHTML" not in dashboard_js


def test_stage_7_monitoring_js_uses_sse_and_safe_polling_fallback() -> None:
    monitoring_js = (WEB_DIST_ROOT / "js" / "pages" / "monitoring.js").read_text(
        encoding="utf-8"
    )

    assert 'DEFAULT_MONITOR_ENDPOINT = "/api/ui/strategies/monitor"' in monitoring_js
    assert "createPoller" in monitoring_js
    assert "createEventStream" in monitoring_js
    assert "closeStream(state)" in monitoring_js
    assert "redirectOnUnauthorized: true" in monitoring_js
    assert "data-monitoring-mobile-tab" in monitoring_js
    assert "drawTimeSeries" in monitoring_js
    assert "innerHTML" not in monitoring_js


def test_stage_5_settings_assets_keep_secret_values_write_only() -> None:
    settings_js = (WEB_DIST_ROOT / "js" / "pages" / "settings.js").read_text(
        encoding="utf-8"
    )

    assert "/api/exchange-keys" not in settings_js
    assert "form.elements.api_secret.value = \"\"" in settings_js
    assert "form.elements.passphrase.value = \"\"" in settings_js
    assert "innerHTML" not in settings_js
    assert "api_secret" not in (WEB_DIST_ROOT / "css" / "pages" / "settings.css").read_text(
        encoding="utf-8"
    )


def test_financial_theme_tokens_are_invariant_across_theme_overrides() -> None:
    tokens_css = (WEB_DIST_ROOT / "css" / "tokens.css").read_text(encoding="utf-8")
    themes_css = (WEB_DIST_ROOT / "css" / "themes.css").read_text(encoding="utf-8")

    assert "--rh-financial-positive" in tokens_css
    assert "--rh-financial-negative" in tokens_css
    assert "--rh-financial-neutral" in tokens_css
    assert "--rh-financial-positive" not in themes_css
    assert "--rh-financial-negative" not in themes_css
    assert "--rh-financial-neutral" not in themes_css


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
