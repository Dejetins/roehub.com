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
    main_html = response.text.split('<main id="main-content"', maxsplit=1)[1].split(
        "</main>", maxsplit=1
    )[0]
    assert '<html lang="en" data-locale="en" data-theme="terminal-orange">' in response.text
    assert "<title>Roehub CLI | Roehub</title>" in response.text
    assert 'data-shell-header' in response.text
    assert 'data-nav-key="dashboard"' in response.text
    assert 'data-nav-key="settings"' in response.text
    assert "/assets/vendor/htmx.min.js" in response.text
    assert "/assets/css/components.css" in response.text
    assert "/assets/css/motion-config.css" in response.text
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
    assert "Roehub platform" not in response.text
    assert 'id="landing-capabilities-title"' not in response.text
    assert "Roehub: research, validate, automate, execute." not in main_html
    assert "Roehub unifies backtesting, strategy management" not in main_html
    assert "Roehub CLI" not in main_html
    assert "booting" not in main_html
    assert 'data-cli-state' not in main_html
    assert "landing-cli__toolbar" not in main_html
    assert 'data-cli-stream' in main_html
    assert 'data-cli-log' in main_html
    assert "&gt; roehub init --workspace cloud" in main_html
    assert "ROEHUB WEB" not in response.text
    assert "&gt;_ ROEHUB WEB" not in response.text
    assert '<div class="command-bar"' not in response.text
    assert 'data-shell-status-bar' in response.text
    assert "WEB SSR:" not in response.text
    assert "Account:" not in response.text
    assert "Mode:" not in response.text
    assert '<span class="user-badge__value">Authentication required</span>' not in response.text
    assert 'id="theme-switcher-trigger"' in response.text
    assert "/assets/css/shell.css?v=" in response.text
    theme_trigger = response.text.split('id="theme-switcher-trigger"', maxsplit=1)[1].split(
        "</button>", maxsplit=1
    )[0]
    assert ">Theme" not in theme_trigger
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
    assert '<span class="user-badge__value">Free</span>' in response.text
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
    main_html = response.text.split('<main id="main-content"', maxsplit=1)[1].split(
        "</main>", maxsplit=1
    )[0]
    assert 'data-page="login"' in main_html
    assert 'data-landing-root' in main_html
    assert 'data-cli-stream' in main_html
    assert "/assets/css/pages/landing.css" in response.text
    assert "/assets/js/pages/landing.js" in response.text
    assert "Roehub: research, validate, automate, execute." not in main_html
    assert "Roehub uses a Keycloak-backed account flow" not in main_html
    assert "landing-cli__toolbar" not in main_html
    assert 'data-cli-state' not in main_html


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
    assert "WEB SSR:" not in settings_response.text
    assert "Account:" not in settings_response.text
    assert "/assets/css/pages/settings.css" in settings_response.text
    assert "/assets/js/pages/settings.js" in settings_response.text
    assert ">Home</a>" in settings_response.text
    assert ">HOME</a>" not in settings_response.text
    for panel in [
        "profile",
        "profile_actions",
        "exchange_keys",
        "limits",
        "integrations",
        "notifications",
        "security",
        "sessions",
        "audit",
    ]:
        assert f'data-settings-panel="{panel}"' in settings_response.text
    assert 'data-settings-panel="command_bar"' not in settings_response.text
    assert 'data-settings-panel="top_actions"' not in settings_response.text
    assert '<header class="settings-command"' not in settings_response.text
    assert settings_response.text.index('data-profile-edit') > settings_response.text.index(
        'data-settings-panel="profile"'
    )
    assert "data-save-all" not in settings_response.text
    assert "data-settings-custom-interval" not in settings_response.text
    assert 'data-settings-panel="preferences"' not in settings_response.text
    assert "data-settings-theme-option" not in settings_response.text
    assert "data-settings-locale-option" not in settings_response.text
    assert "data-settings-refresh-option" not in settings_response.text
    assert "Custom interval seconds" not in settings_response.text
    assert 'role="listbox"' in settings_response.text
    assert '<select' not in settings_response.text
    assert '<meter' not in settings_response.text
    assert "settings-cli-meter" in settings_response.text
    assert 'data-security-focus' not in settings_response.text
    assert '<span aria-hidden="true">&gt;_</span>' not in settings_response.text


def test_authorized_strategy_routes_render_stage_6_workstation_and_aliases() -> None:
    client = _build_test_client()

    strategies_response = client.get("/strategies")
    main_html = strategies_response.text.split('<main id="main-content"', maxsplit=1)[1].split(
        "</main>", maxsplit=1
    )[0]

    assert strategies_response.status_code == 200
    assert 'data-page="strategies"' in strategies_response.text
    assert 'data-strategies-root' in strategies_response.text
    assert 'data-dashboard-endpoint="/api/ui/strategies/dashboard"' in strategies_response.text
    assert 'data-api-create-path="/api/strategies"' not in strategies_response.text
    assert 'data-api-clone-path="/api/strategies/clone"' in strategies_response.text
    assert 'data-api-run-path-template="/api/strategies/{strategy_id}/run"' in (
        strategies_response.text
    )
    assert 'data-api-stop-path-template="/api/strategies/{strategy_id}/stop"' in (
        strategies_response.text
    )
    assert 'data-nav-key="strategies"' in strategies_response.text
    assert 'nav-tab--active"' in strategies_response.text
    assert "/assets/css/pages/strategies.css" in strategies_response.text
    assert "/assets/js/pages/strategies.js" in strategies_response.text
    assert "/assets/strategy_ui.js" not in strategies_response.text
    assert "<select" not in strategies_response.text
    assert 'role="listbox"' in strategies_response.text
    assert 'data-strategy-control="branded dropdown"' in strategies_response.text
    assert 'data-saved-search' in strategies_response.text
    assert 'data-strategy-saved-rows' in strategies_response.text
    assert 'data-strategy-load' not in strategies_response.text
    assert "Load statistics" not in strategies_response.text
    assert 'id="strategy-manage-trigger"' in strategies_response.text
    assert 'data-strategy-create-open' not in strategies_response.text
    assert 'class="strategies-command"' not in strategies_response.text
    assert 'data-strategies-panel="command_status"' not in strategies_response.text
    assert 'data-strategies-panel="best_worst_days"' not in strategies_response.text
    assert 'class="terminal-panel strategies-drawdown"' not in strategies_response.text
    assert 'class="terminal-panel strategies-equity"' not in strategies_response.text
    assert 'class="strategies-chart-mini"' not in strategies_response.text
    assert 'data-strategies-panel="stat_tiles"' not in strategies_response.text
    assert 'data-strategies-panel="symbol_results"' not in strategies_response.text
    assert 'class="terminal-panel strategies-long-short"' not in strategies_response.text
    assert 'class="terminal-panel strategies-risk"' not in strategies_response.text
    assert 'class="terminal-panel strategies-monthly"' not in strategies_response.text
    assert 'class="terminal-panel strategies-hours"' not in strategies_response.text
    assert 'class="terminal-panel strategies-symbols"' not in strategies_response.text
    for panel in [
        "selected_strategy",
        "visual_workspace",
        "statistics_workspace",
        "saved_strategies",
        "metric_grid",
        "long_short",
        "risk_execution",
        "monthly_stats",
        "hourly_results",
        "trades",
    ]:
        assert f'data-strategies-panel="{panel}"' in strategies_response.text
    for mode in ["trades", "equity", "drawdown"]:
        assert f'data-chart-mode="{mode}"' in strategies_response.text
    for mode in ["overall", "long_short", "hourly", "risk", "monthly"]:
        assert f'data-stat-mode="{mode}"' in strategies_response.text
    assert "Protected workspace placeholder" not in main_html
    assert "strategy_ui.js" not in main_html
    assert 'data-strategy-create-panel' not in strategies_response.text
    assert 'class="strategies-stat"' not in strategies_response.text
    assert 'class="strategies-risk-list"' not in strategies_response.text
    assert "<th>Metric</th>" in main_html
    assert "<th>Value</th>" in main_html
    assert "<th>Source</th>" in main_html
    assert 'data-strategies-refresh-preset' in strategies_response.text
    assert "shell-status-panel app-bottom-status shell-global-status strategies-status-line" in (
        strategies_response.text
    )
    assert main_html.index('data-strategies-panel="statistics_workspace"') < main_html.index(
        'data-strategies-panel="selected_strategy"'
    )
    assert main_html.index('data-strategies-panel="visual_workspace"') < main_html.index(
        'data-strategies-panel="saved_strategies"'
    )
    assert main_html.index('data-strategies-panel="saved_strategies"') < main_html.index(
        'data-strategies-panel="trades"'
    )

    new_response = client.get("/strategies/new")
    assert new_response.status_code == 200
    assert 'data-page="strategies"' in new_response.text
    assert 'data-initial-mode="dashboard"' in new_response.text
    assert 'data-strategy-create-panel' not in new_response.text
    assert "/assets/strategy_ui.js" not in new_response.text

    strategy_id = "00000000-0000-0000-0000-000000000123"
    detail_response = client.get(f"/strategies/{strategy_id}")
    assert detail_response.status_code == 200
    assert 'data-page="strategies"' in detail_response.text
    assert f'data-initial-strategy-id="{strategy_id}"' in detail_response.text
    assert "/assets/strategy_ui.js" not in detail_response.text


def test_authorized_backtest_routes_render_stage_8_workstation_and_aliases() -> None:
    client = _build_test_client()

    response = client.get("/backtests")
    main_html = response.text.split('<main id="main-content"', maxsplit=1)[1].split(
        "</main>", maxsplit=1
    )[0]

    assert response.status_code == 200
    assert 'data-page="backtests"' in response.text
    assert 'data-backtests-root' in response.text
    assert 'data-workstation-endpoint="/api/ui/backtests/workstation"' in response.text
    assert 'data-runtime-defaults-endpoint="/api/backtests/runtime-defaults"' in response.text
    assert 'data-preflight-endpoint="/api/backtests/preflight"' in response.text
    assert 'data-jobs-endpoint="/api/backtests/jobs"' in response.text
    assert (
        'data-job-cancel-endpoint-template="/api/backtests/jobs/{job_id}/cancel"'
        in response.text
    )
    assert 'data-job-delete-endpoint-template="/api/backtests/jobs/{job_id}"' in response.text
    assert (
        'data-job-summary-endpoint-template="/api/backtests/jobs/{job_id}/summary"'
        in response.text
    )
    assert (
        'data-variant-endpoint-template="/api/backtests/jobs/{job_id}/variants/{variant_key}"'
        in response.text
    )
    assert (
        'data-variant-equity-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/equity"'
        in response.text
    )
    assert (
        'data-variant-drawdown-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/drawdown"'
        in response.text
    )
    assert (
        'data-variant-monthly-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/monthly-stats"'
        in response.text
    )
    assert (
        'data-variant-symbol-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/symbol-stats"'
        in response.text
    )
    assert (
        'data-variant-trades-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/trades?page={page}&page_size={page_size}"'
        in response.text
    )
    assert (
        'data-variant-trades-csv-endpoint-template="/api/backtests/jobs/{job_id}'
        '/variants/{variant_key}/trades.csv"'
        in response.text
    )
    assert 'data-variant-open-delay-ms="180"' in response.text
    assert 'data-variant-open-duration-ms="650"' in response.text
    assert 'data-variant-preview-limit="5"' in response.text
    assert "data-indicator-add-menu" in response.text
    assert "data-risk-grid" in response.text
    assert 'data-risk-side-enabled="tp"' in response.text
    assert 'data-risk-side-enabled="sl"' in response.text
    assert "data-clear-job-filters" in response.text
    assert 'data-current-value="sizing_mode"' in response.text
    assert 'data-backtest-menu="market"' in response.text
    assert 'data-backtest-menu="market_type"' in response.text
    assert 'id="backtest-instrument-market-trigger"' in response.text
    assert 'id="backtest-instrument-market-type-trigger"' in response.text
    assert response.text.count('data-backtest-option="timeframe"') >= 10
    for timeframe in ("15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "2d", "3d"):
        assert f'data-value="{timeframe}"' in response.text
    assert 'data-sizing-field="equity_pct"' in response.text
    assert 'data-sizing-field="quote_amount"' in response.text
    assert 'data-sizing-bounds-row' in response.text
    assert "backtests-field-pair--dates" in response.text
    assert "data-job-picker-menu" in response.text
    assert "data-backtest-option=\"job_exchange\"" in response.text
    assert "data-backtest-option=\"job_market_type\"" in response.text
    assert "data-clear-symbols" not in response.text
    assert "data-symbol-count" not in response.text
    assert "data-selected-symbols" not in response.text
    assert "data-job-launched-from" in response.text
    assert "data-job-symbol" in response.text
    assert "data-load-more-jobs" in response.text
    assert "data-job-cancel-dialog" in response.text
    assert "data-job-cancel-confirm" in response.text
    assert "/assets/css/pages/backtests.css" in response.text
    assert "/assets/js/pages/backtests.js" in response.text
    assert "/assets/backtest_ui.js" not in response.text
    assert "<select" not in response.text
    assert 'role="listbox"' in response.text
    assert 'data-backtest-control="branded dropdown"' in response.text
    assert "Protected workspace placeholder" not in main_html
    assert "backtest_ui.js" not in main_html
    for panel in [
        "config",
        "ai_configurator",
        "instruments",
        "indicators",
        "optimization",
        "jobs_variants",
    ]:
        assert f'data-backtests-panel="{panel}"' in response.text
    for removed_fragment in [
        'data-backtests-panel="command_bar"',
        'data-backtests-panel="recent_events"',
        "backtest-preset-trigger",
        "mean_rev_opt_v2",
        "backtests.results.actions",
        "data-cancel-job",
        "Added RSI",
        "Configuration is ready",
    ]:
        assert removed_fragment not in response.text
    assert main_html.index('data-backtests-panel="config"') < main_html.index(
        'data-backtests-panel="ai_configurator"'
    )
    assert main_html.index('data-backtests-panel="instruments"') < main_html.index(
        'data-backtests-panel="indicators"'
    )
    assert main_html.index('data-backtests-panel="optimization"') < main_html.index(
        'data-backtests-panel="jobs_variants"'
    )
    assert 'data-backtests-refresh-preset' in response.text
    assert "data-footer-capital" not in response.text

    new_response = client.get("/backtests/new")
    assert new_response.status_code == 200
    assert 'data-page="backtests"' in new_response.text
    assert 'data-initial-mode="create"' in new_response.text
    assert "/assets/backtest_ui.js" not in new_response.text

    job_id = "00000000-0000-0000-0000-000000000234"
    detail_response = client.get(f"/backtests/{job_id}")
    assert detail_response.status_code == 200
    assert 'data-page="backtests"' in detail_response.text
    assert 'data-initial-mode="selected_job"' in detail_response.text
    assert f'data-initial-job-id="{job_id}"' in detail_response.text


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
    assert "PnL / equity monitoring (live)" not in response.text
    assert "PnL / equity monitoring" in response.text
    assert ">4H</button>" in response.text
    assert ">1W</button>" in response.text
    assert ">1M</button>" in response.text
    assert "dashboard-live-indicator" not in response.text
    assert "data-alert-sort-level" in response.text
    assert "dashboard-table--alerts" in response.text
    assert "dashboard-table--allocation" in response.text
    assert 'role="listbox"' in response.text
    assert "data-dashboard-refresh-preset" in response.text
    assert (
        "shell-status-panel app-bottom-status shell-global-status dashboard-status-line"
        in response.text
    )
    assert "dashboard-workstation__command" not in response.text
    assert "dashboard-command-actions" not in response.text
    assert "dashboard-status-line" in response.text
    assert 'data-dashboard-loading hidden aria-hidden="true"' in response.text
    assert "<span>Loading dashboard summary</span>" not in response.text
    assert "data-footer-account" not in response.text
    assert "data-footer-mode" not in response.text
    assert "WEB SSR:" not in response.text
    assert "Account:" not in response.text
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
        "dist/css/motion-config.css",
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
        "dist/js/pages/strategies.js",
        "dist/css/pages/strategies.css",
        "dist/js/pages/backtests.js",
        "dist/css/pages/backtests.css",
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
    motion_config_css = (_WEB_ROOT / "dist/css/motion-config.css").read_text(encoding="utf-8")
    layout_css = (_WEB_ROOT / "dist/css/layout.css").read_text(encoding="utf-8")
    base_css = (_WEB_ROOT / "dist/css/base.css").read_text(encoding="utf-8")
    theme_js = (_WEB_ROOT / "dist/js/core/theme.js").read_text(encoding="utf-8")
    api_js = (_WEB_ROOT / "dist/js/core/api.js").read_text(encoding="utf-8")
    poller_js = (_WEB_ROOT / "dist/js/core/poller.js").read_text(encoding="utf-8")
    locale_js = (_WEB_ROOT / "dist/js/core/locale.js").read_text(encoding="utf-8")
    refresh_js = (_WEB_ROOT / "dist/js/core/refresh.js").read_text(encoding="utf-8")
    components_css = (_WEB_ROOT / "dist/css/components.css").read_text(encoding="utf-8")
    dashboard_js = (_WEB_ROOT / "dist/js/pages/dashboard.js").read_text(encoding="utf-8")
    dashboard_css = (_WEB_ROOT / "dist/css/pages/dashboard.css").read_text(encoding="utf-8")
    strategies_js = (_WEB_ROOT / "dist/js/pages/strategies.js").read_text(encoding="utf-8")
    strategies_css = (_WEB_ROOT / "dist/css/pages/strategies.css").read_text(encoding="utf-8")
    backtests_js = (_WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")
    backtests_css = (_WEB_ROOT / "dist/css/pages/backtests.css").read_text(encoding="utf-8")

    assert "--rh-financial-positive" in tokens_css
    assert "--rh-financial-negative" in tokens_css
    assert "--rh-canvas: #000000" in tokens_css
    assert "--rh-workstation-panel-bg" in tokens_css
    assert "[hidden]" in base_css
    assert "display: none !important;" in base_css
    assert "background: var(--rh-canvas)" in layout_css
    assert "--rh-cli-meter-fill" in tokens_css
    assert "--rh-page-transition-duration:" in motion_config_css
    assert ".page-shell:not(:has(.landing-page))" in layout_css
    assert "@view-transition" in layout_css
    assert "prefers-reduced-motion: reduce" in layout_css
    assert "view-transition-name: none" in layout_css
    assert ".rh-cli-meter" in components_css
    assert ".dashboard-cli-meter" in components_css
    assert "--rh-financial-positive" not in themes_css
    assert "--rh-financial-negative" not in themes_css
    for theme in ["terminal-orange", "graphite"]:
        assert theme in theme_js
        assert f'data-theme="{theme}"' in tokens_css or theme in themes_css
    assert "matrix-green" not in theme_js
    assert "high-contrast" not in theme_js
    assert 'data-theme="matrix-green"' not in themes_css
    assert 'data-theme="high-contrast"' not in themes_css

    for status in ["401", "403", "409", "422", "timeout"]:
        assert status in api_js
    assert "x-csrf-token" in api_js
    assert "this.running" in poller_js
    assert "document.hidden" in poller_js
    assert "retry_after_seconds" in poller_js
    assert "roehub_locale" in locale_js
    assert "DEFAULT_LOCALE = \"en\"" in locale_js
    assert "__roehubDropdownDelegatesInitialized" in (
        _WEB_ROOT / "dist/js/components/dropdown.js"
    ).read_text(encoding="utf-8")
    assert "__roehubListboxDelegatesInitialized" in (
        _WEB_ROOT / "dist/js/components/listbox.js"
    ).read_text(encoding="utf-8")
    for preset in ['"10s"', '"15s"', '"30s"', '"1m"', '"5m"']:
        assert preset in refresh_js
        assert preset in dashboard_js
    assert "/api/ui/dashboard/summary" in dashboard_js
    assert "createPoller" in dashboard_js
    assert "/api/ui/strategies/dashboard" in strategies_js
    assert "/api/strategies/clone" in strategies_js
    assert "source_strategy_id" in strategies_js
    assert "createPoller" in strategies_js
    assert "activeRequest" in strategies_js
    assert "hiddenTabPause" in strategies_js
    assert "manualRefreshRetrySeconds" in strategies_js
    assert "button.disabled = isRunning;" in strategies_js
    assert "positionStatusRefreshMenu" in strategies_js
    assert "closeStatusRefreshMenu" in strategies_js
    assert "data-saved-search" in strategies_js
    assert "strategies.saved.no_matches" in strategies_js
    assert "grid-row: 1;" in strategies_css
    assert "data-chart-mode" in strategies_js
    assert "data-stat-mode" in strategies_js
    assert "strategy_ui.js" not in strategies_js
    assert "--rh-financial-positive" in strategies_css
    assert "--rh-financial-negative" in strategies_css
    assert "/api/ui/backtests/workstation" in backtests_js
    assert "/api/backtests/preflight" in backtests_js
    assert "/api/backtests/jobs" in backtests_js
    assert "Idempotency-Key" in backtests_js
    assert "request_hash" in backtests_js
    assert "refresh_status" in backtests_js
    assert "retry_after_seconds" in backtests_js
    assert "data-delete-job-id" in backtests_js
    assert "data-load-more-jobs" in backtests_js
    assert "delayedVariantOpen" in backtests_js
    assert "top_limit" in backtests_js
    assert "variantPreviewLimit" in backtests_js
    assert "queueVariantPanelAnimation" in backtests_js
    assert "variantEmptyText" in backtests_js
    assert "backtests.variants.none_passed_quality_gate" in backtests_js
    assert "parameterCombinationCount(root)" in backtests_js
    assert "riskCombinationCount(root)" in backtests_js
    assert "data-clear-job-filters" in backtests_js
    assert "data-variant-closing" in backtests_js
    assert "trashIcon(t(\"backtests.actions.delete\"))" in backtests_js
    assert "&times;" not in backtests_js
    assert "bindStatusBar" in backtests_js
    assert "backtests-variant-frame--static" in backtests_js
    assert '"backtests-variant-frame backtests-variant-frame--static is-open"' not in backtests_js
    assert "jobId && jobId === state.selectedJobId" in backtests_js
    assert "state.resultDetails = null;" in backtests_js
    assert "signedDrawdownPercent" in backtests_js
    assert "integerOrDash" in backtests_js
    assert "backtests-download-button" in backtests_js
    assert "createPoller" in backtests_js
    assert "activeRequest" in backtests_js
    assert "hiddenTabPause" in backtests_js
    assert "manualRefreshRetrySeconds" in backtests_js
    assert "configSeeded" in backtests_js
    assert "symbolQuery" in backtests_js
    assert "[data-backtest-menu='" in backtests_js
    assert "buildSizingPayload" in backtests_js
    assert "toggleStatusRefreshMenu" in backtests_js
    assert "data-symbol-select" in backtests_js
    assert "filterSymbols(root, state.symbolQuery);" in backtests_js
    assert "state.symbolQuery = event.target.value || \"\";" in backtests_js
    assert "<small>${escapeHtml(symbol.status)}</small>" not in backtests_js
    assert "button.disabled = isRunning;" in backtests_js
    assert "renderBacktestSeries" in backtests_js
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity" in backtests_js
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown" in backtests_js
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats" in backtests_js
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats" in backtests_js
    assert "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=" in backtests_js
    assert "data-result-chart" in backtests_js
    assert "data-trades-rows" in backtests_js
    assert "data-result-refresh" in backtests_js
    assert "trades.csv" in backtests_js
    assert "activeResultRequest" in backtests_js
    assert "backtest_ui.js" not in backtests_js
    assert "grid-row: 1;" in backtests_css
    assert "height: var(--rh-workarea-height);" in backtests_css
    assert ".backtests-instrument-controls" in backtests_css
    assert "grid-template-columns: minmax(0, 1fr);" in backtests_css
    assert ".backtests-field-pair--dates" in backtests_css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in backtests_css
    assert ".backtests-icon-button" in backtests_css
    assert ".backtests-variant-frame" in backtests_css
    assert "max-height var(--backtests-variant-open-duration" in backtests_css
    assert ".backtests-variant-frame.backtests-variant-frame--static" in backtests_css
    assert "--rh-financial-positive" in backtests_css
    assert "--rh-financial-negative" in backtests_css
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
