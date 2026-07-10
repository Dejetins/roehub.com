from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app

_WORKBENCH_CSS = Path(__file__).resolve().parents[4] / "apps/web/dist/css/workbench.css"


def _authorized_client() -> TestClient:
    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
        }
    )
    app.state.current_user_api_client = SimpleNamespace(
        fetch_current_user=lambda *, cookie_header: CurrentUserApiResult(
            status_code=200,
            user=WebCurrentUser(
                user_id="00000000-0000-0000-0000-000000000321",
                paid_level="free",
            ),
            error_message=None,
        )
    )
    return TestClient(app)


def test_workbench_shell_has_one_primary_sidebar_and_no_global_status_footer() -> None:
    response = _authorized_client().get("/dashboard")

    assert response.status_code == 200
    assert response.text.count("data-workbench-sidebar") == 1
    assert response.text.count('class="workbench-sidebar"') == 1
    assert 'class="workbench-topbar"' in response.text
    assert 'class="workbench-mobile-nav"' in response.text
    assert "shell-global-status" not in response.text
    assert 'id="command-dialog"' in response.text
    assert response.headers["cache-control"] == "private, no-store"


def test_new_route_entries_reuse_real_production_read_models() -> None:
    client = _authorized_client()

    monitoring = client.get("/monitoring")
    models = client.get("/models")
    connections = client.get("/connections")

    assert monitoring.status_code == 200
    assert 'data-page="monitoring"' in monitoring.text
    assert 'data-summary-endpoint="/api/ui/dashboard/summary"' in monitoring.text
    assert 'data-nav-key="monitoring"' in monitoring.text
    assert 'nav-tab--active is-active"' in monitoring.text

    assert models.status_code == 200
    assert 'data-page="strategies"' in models.text
    assert 'data-initial-mode="rl_ml"' in models.text
    assert 'data-nav-key="models"' in models.text
    assert 'nav-tab--active is-active"' in models.text

    assert connections.status_code == 200
    assert 'data-page="connections"' in connections.text
    assert 'data-settings-scope="connections"' in connections.text
    assert 'data-exchange-keys-endpoint="/api/ui/account/exchange-connections"' in connections.text
    assert 'data-audit-endpoint="/api/ui/account/audit-events"' in connections.text
    assert 'data-nav-key="connections"' in connections.text
    assert 'nav-tab--active is-active"' in connections.text


def test_login_gateway_is_public_and_authenticated_root_redirects_to_overview() -> None:
    client = _authorized_client()

    public = client.get("/login")
    authenticated_root = client.get("/", cookies={"roehub_session_id": "opaque"})

    assert public.status_code == 200
    assert 'data-auth-gateway' in public.text
    assert 'data-auth-continue' in public.text
    assert 'data-workbench-sidebar' not in public.text
    assert authenticated_root.url.path == "/dashboard"


def test_shell_exposes_six_local_themes_and_mutation_reconciliation_contract() -> None:
    response = _authorized_client().get("/connections")

    for theme in ("abyss", "graphite", "slate", "frost", "paper", "sand"):
        assert f'data-theme-value="{theme}"' in response.text
    assert "/assets/js/core/workbench-shell.js" in response.text


def test_compact_actions_keep_minimum_desktop_and_mobile_target_widths() -> None:
    css = _WORKBENCH_CSS.read_text(encoding="utf-8")

    assert ".rh-button--compact { min-width: 36px; min-height: 36px; }" in css
    assert ".rh-button--compact { min-width: 44px; }" in css
