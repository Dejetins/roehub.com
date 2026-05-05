from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app
from apps.web.main.i18n import LOCALE_COOKIE_NAME, catalog_key_sets

_WEB_ROOT = Path(__file__).resolve().parents[4] / "apps" / "web"


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


def test_public_landing_renders_terminal_shell_and_local_assets() -> None:
    client = _build_test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert '<html lang="en" data-locale="en" data-theme="terminal-orange">' in response.text
    assert 'data-shell-header' in response.text
    assert 'data-nav-key="dashboard"' in response.text
    assert 'data-nav-key="settings"' in response.text
    assert "/assets/vendor/htmx.min.js" in response.text
    assert "/assets/css/components.css" in response.text
    assert "/assets/css/pages/landing.css" in response.text
    assert "/assets/js/core/theme.js" in response.text
    assert "/assets/js/core/locale.js" in response.text
    assert "/assets/js/components/dropdown.js" in response.text
    assert "/assets/js/pages/auth.js" in response.text
    assert "/assets/js/pages/landing.js" in response.text
    assert "https://unpkg.com" not in response.text
    assert 'data-auth-modal' in response.text
    assert 'data-page="landing"' in response.text
    assert 'data-landing-root' in response.text
    assert 'href="/register"' in response.text
    assert 'data-auth-open' in response.text
    assert "Roehub platform" in response.text
    assert 'id="landing-capabilities-title"' not in response.text
    assert "Roehub: research, validate, automate, execute." in response.text
    assert "ROEHUB WEB" in response.text
    assert "&gt;_ ROEHUB WEB" not in response.text
    assert '<div class="command-bar"' not in response.text
    assert 'data-shell-status-bar' in response.text
    assert '<span class="user-badge__value">Authentication required</span>' not in response.text
    assert 'id="theme-switcher-trigger"' in response.text
    assert 'data-theme-value="terminal-orange"' in response.text
    assert "site.css" not in response.text


def test_public_landing_does_not_require_current_user_api_without_auth_cookie() -> None:
    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
        }
    )

    def fail_fetch_current_user(*, cookie_header: str | None) -> CurrentUserApiResult:
        raise AssertionError(f"unexpected current-user lookup: {cookie_header}")

    app.state.current_user_api_client = SimpleNamespace(fetch_current_user=fail_fetch_current_user)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Guest" in response.text


def test_public_landing_shows_current_user_when_auth_cookie_is_present() -> None:
    client = _build_test_client()

    response = client.get("/", cookies={"roehub_session_id": "session-123"})

    assert response.status_code == 200
    assert '<span class="user-badge__value">FREE</span>' in response.text
    assert "Logout" in response.text
    assert 'data-auth-next="/dashboard"\n            >Login</button>' not in response.text


@pytest.mark.parametrize(
    ("path", "expected_location"),
    [
        ("/dashboard", "/login?next=%2Fdashboard"),
        ("/settings", "/login?next=%2Fsettings"),
        ("/backtests", "/login?next=%2Fbacktests"),
        ("/backtests/new", "/login?next=%2Fbacktests%2Fnew"),
        ("/backtests/abc123", "/login?next=%2Fbacktests%2Fabc123"),
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
    client = _build_test_client(
        api_result=CurrentUserApiResult(status_code=401, user=None, error_message=None)
    )

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == expected_location


def test_login_route_sanitizes_external_next_and_preopens_modal() -> None:
    client = _build_test_client()

    response = client.get("/login?next=https://evil.example/path")

    assert response.status_code == 200
    assert "/api/auth/login?next=%2F" in response.text
    assert 'id="keycloak-login-link"' in response.text
    assert 'data-open-on-load="true"' in response.text
    assert "Sign in to Roehub" in response.text
    assert "https://evil.example/path" not in response.text


def test_register_route_is_separate_keycloak_backed_entrypoint() -> None:
    client = _build_test_client()

    response = client.get("/register?next=/settings")

    assert response.status_code == 200
    assert 'data-page="register"' in response.text
    assert 'data-register-entrypoint="/api/auth/login?next=%2Fsettings"' in response.text
    assert "Create your Roehub account" in response.text
    assert "<input" not in response.text
    assert "<form" not in response.text


def test_favicon_route_avoids_browser_404_noise() -> None:
    client = _build_test_client()

    response = client.get("/favicon.ico")

    assert response.status_code == 204


@pytest.mark.parametrize(
    ("path", "expected_redirect"),
    [
        ("/logout", "/login"),
        ("/logout?next=/strategies", "/strategies"),
        ("/logout?next=https://evil.example/path", "/login"),
    ],
)
def test_logout_page_uses_external_auth_module_and_sanitized_redirect(
    path: str,
    expected_redirect: str,
) -> None:
    client = _build_test_client()

    response = client.get(path)

    assert response.status_code == 200
    assert 'data-auth-logout' in response.text
    assert f'data-logout-redirect="{expected_redirect}"' in response.text
    assert "/assets/js/pages/auth.js" in response.text
    assert "https://evil.example/path" not in response.text
    assert "<script>" not in response.text


def test_user_badge_partial_route_is_not_publicly_registered() -> None:
    client = _build_test_client()

    response = client.get("/_partial/user_badge")

    assert response.status_code == 404


def test_authorized_settings_route_renders_stage_5_workstation() -> None:
    client = _build_test_client()

    settings_response = client.get("/settings")

    assert settings_response.status_code == 200
    assert 'data-page="settings"' in settings_response.text
    assert 'data-nav-key="settings"' in settings_response.text
    assert 'nav-tab--active"' in settings_response.text
    assert 'data-profile-endpoint="/api/ui/account/profile"' in settings_response.text
    assert 'data-preferences-endpoint="/api/ui/account/preferences"' in settings_response.text
    assert 'data-exchange-keys-endpoint="/api/exchange-keys"' in settings_response.text
    assert '<div class="command-bar"' not in settings_response.text
    assert '<footer class="status-bar">' not in settings_response.text
    assert "shell-status-panel app-bottom-status shell-global-status" in settings_response.text
    assert "/assets/css/pages/settings.css" in settings_response.text
    assert "/assets/js/pages/settings.js" in settings_response.text
    for panel in [
        "command_bar",
        "profile",
        "exchange_keys",
        "limits",
        "integrations",
        "notifications",
        "security",
        "sessions",
        "audit",
        "top_actions",
    ]:
        assert f'data-settings-panel="{panel}"' in settings_response.text
    assert 'role="listbox"' in settings_response.text
    assert '<select' not in settings_response.text
    assert '<meter' not in settings_response.text
    assert "settings-cli-meter" in settings_response.text
    assert 'data-security-focus' not in settings_response.text
    assert '<span aria-hidden="true">&gt;_</span>' not in settings_response.text


def test_authorized_placeholder_routes_render_active_navigation() -> None:
    client = _build_test_client()

    strategies_response = client.get("/strategies/new")

    assert strategies_response.status_code == 200
    assert 'data-nav-key="strategies"' in strategies_response.text
    assert 'nav-tab--active"' in strategies_response.text
    assert 'id="placeholder-market-listbox"' not in strategies_response.text
    assert "BTCUSDT" not in strategies_response.text
    assert "ETHUSDT" not in strategies_response.text
    assert "SOLUSDT" not in strategies_response.text


def test_authorized_dashboard_renders_stage_4_workstation_shell() -> None:
    client = _build_test_client()

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert 'data-page="dashboard"' in response.text
    assert 'data-dashboard-root' in response.text
    assert 'data-summary-endpoint="/api/ui/dashboard/summary"' in response.text
    assert "/assets/css/pages/dashboard.css" in response.text
    assert "/assets/js/pages/dashboard.js" in response.text
    for panel in [
        "selected_strategy_snapshot",
        "equity_pnl_series",
        "metric_grid",
        "open_positions",
        "recent_executions",
        "health_risk",
        "alerts",
        "symbol_allocation",
        "strategy_list",
    ]:
        assert f'data-dashboard-panel="{panel}"' in response.text
    assert "Monitoring strategies" in response.text
    assert "Strategy list" in response.text
    assert 'role="listbox"' in response.text
    assert "data-dashboard-refresh-preset" in response.text
    assert "shell-status-panel app-bottom-status dashboard-status-line" in response.text
    assert "dashboard-workstation__command" not in response.text
    assert "dashboard-command-actions" not in response.text
    assert "dashboard-status-line" in response.text
    assert "dashboard-pagination" not in response.text
    assert "dashboard-page-button" not in response.text
    assert '<footer class="status-bar">' not in response.text
    assert '<span aria-hidden="true">&gt;_</span>' not in response.text
    assert "data-selected-action" not in response.text
    assert "<select" not in response.text


def test_locale_cookie_selects_russian_shell_without_localizing_routes() -> None:
    client = _build_test_client()

    response = client.get("/", cookies={LOCALE_COOKIE_NAME: "ru"})

    assert response.status_code == 200
    assert '<html lang="ru" data-locale="ru" data-theme="terminal-orange">' in response.text
    assert "Открыть вход в Roehub" in response.text
    assert 'href="/strategies"' in response.text
    assert "/api/auth/login" in response.text


def test_invalid_locale_cookie_falls_back_to_english() -> None:
    client = _build_test_client()

    response = client.get("/", cookies={LOCALE_COOKIE_NAME: "de"})

    assert response.status_code == 200
    assert '<html lang="en" data-locale="en" data-theme="terminal-orange">' in response.text
    assert "Open Roehub login" in response.text


def test_locale_switch_sets_cookie_and_keeps_route_path() -> None:
    client = _build_test_client()

    response = client.get(
        "/locale?locale=ru&next=/settings",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/settings"
    assert f"{LOCALE_COOKIE_NAME}=ru" in response.headers["set-cookie"]


def test_locale_catalog_keys_match() -> None:
    key_sets = catalog_key_sets()

    assert key_sets["en"] == key_sets["ru"]


def test_stage_2_design_system_assets_exist_and_keep_contract_literals() -> None:
    expected_assets = [
        "dist/css/tokens.css",
        "dist/css/themes.css",
        "dist/css/base.css",
        "dist/css/layout.css",
        "dist/css/components.css",
        "dist/js/core/theme.js",
        "dist/js/core/api.js",
        "dist/js/core/poller.js",
        "dist/js/core/sse.js",
        "dist/js/core/dom.js",
        "dist/js/core/locale.js",
        "dist/js/core/notifications.js",
        "dist/js/core/formatters.js",
        "dist/js/core/validators.js",
        "dist/js/core/refresh.js",
        "dist/js/pages/dashboard.js",
        "dist/css/pages/dashboard.css",
        "dist/js/components/dropdown.js",
        "dist/js/components/listbox.js",
        "dist/js/components/combobox.js",
        "dist/js/components/refresh-control.js",
        "dist/js/pages/settings.js",
        "dist/css/pages/settings.css",
    ]

    for asset in expected_assets:
        assert (_WEB_ROOT / asset).is_file(), asset

    tokens_css = (_WEB_ROOT / "dist/css/tokens.css").read_text(encoding="utf-8")
    themes_css = (_WEB_ROOT / "dist/css/themes.css").read_text(encoding="utf-8")
    theme_js = (_WEB_ROOT / "dist/js/core/theme.js").read_text(encoding="utf-8")
    api_js = (_WEB_ROOT / "dist/js/core/api.js").read_text(encoding="utf-8")
    poller_js = (_WEB_ROOT / "dist/js/core/poller.js").read_text(encoding="utf-8")
    locale_js = (_WEB_ROOT / "dist/js/core/locale.js").read_text(encoding="utf-8")
    refresh_js = (_WEB_ROOT / "dist/js/core/refresh.js").read_text(encoding="utf-8")
    components_css = (_WEB_ROOT / "dist/css/components.css").read_text(encoding="utf-8")
    dashboard_js = (_WEB_ROOT / "dist/js/pages/dashboard.js").read_text(encoding="utf-8")
    dashboard_css = (_WEB_ROOT / "dist/css/pages/dashboard.css").read_text(encoding="utf-8")

    assert "--rh-financial-positive" in tokens_css
    assert "--rh-financial-negative" in tokens_css
    assert "--rh-workstation-panel-bg" in tokens_css
    assert "--rh-cli-meter-fill" in tokens_css
    assert ".rh-cli-meter" in components_css
    assert ".dashboard-cli-meter" in components_css
    assert "--rh-financial-positive" not in themes_css
    assert "--rh-financial-negative" not in themes_css
    for theme in ["terminal-orange", "graphite", "matrix-green", "high-contrast"]:
        assert theme in theme_js
        assert f'data-theme="{theme}"' in tokens_css or theme in themes_css

    for status in ["401", "403", "409", "422", "timeout"]:
        assert status in api_js
    assert "x-csrf-token" in api_js
    assert "this.running" in poller_js
    assert "document.hidden" in poller_js
    assert "retry_after_seconds" in poller_js
    assert "roehub_locale" in locale_js
    assert "DEFAULT_LOCALE = \"en\"" in locale_js
    for preset in ['"10s"', '"15s"', '"30s"', '"1m"', '"5m"']:
        assert preset in refresh_js
        assert preset in dashboard_js
    assert "/api/ui/dashboard/summary" in dashboard_js
    assert "createPoller" in dashboard_js
    assert "activeRequest" in dashboard_js
    assert "hiddenTabPause" in dashboard_js
    for placeholder in ["metric_1", "metric_2", "metric_3", "metric_4"]:
        assert placeholder in dashboard_js
    assert "rh-cli-meter dashboard-cli-meter" not in dashboard_js
    assert "dashboard-page-button" not in dashboard_css
    assert "data-selected-action" not in dashboard_js
    assert ".command-bar" not in dashboard_css
    assert "--rh-financial-positive" in dashboard_css
    assert "--rh-financial-negative" in dashboard_css
