from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
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
    assert "data-auth-gateway" in public.text
    assert "data-auth-continue" in public.text
    assert "data-workbench-sidebar" not in public.text
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
    assert "grid-template-columns: minmax(0, 1fr) 44px 44px" in css
    assert ".strategies-action-group { flex-wrap: wrap; }" in css


# Foundation route seam: unit doubles supplement the real HTTP/browser fixture.
def _platform_client(tmp_path, monkeypatch, *, status=200, enabled=True):
    import json

    import apps.web.main.app as web_module

    (tmp_path / ".vite").mkdir(exist_ok=True)
    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / "assets/main.js").write_text("/* fixture asset */")
    (tmp_path / "assets/main.css").write_text("body {}")
    (tmp_path / ".vite/manifest.json").write_text(
        json.dumps(
            {
                "src/main.tsx": {"file": "assets/main.js", "css": ["assets/main.css"]},
            }
        )
    )
    monkeypatch.setattr(web_module, "DEFAULT_PLATFORM_DIST", tmp_path)
    app = create_app(
        environ={
            "WEB_API_BASE_URL": "http://web.local",
            "WEB_API_UPSTREAM_URL": "http://api.local",
            "WEB_BACKTESTS_CLIENT_ENABLED": str(enabled).lower(),
        }
    )
    app.state.current_user_api_client = SimpleNamespace(
        fetch_current_user=lambda **_: CurrentUserApiResult(
            status_code=status,
            user=WebCurrentUser(user_id="local-subject", paid_level="free")
            if status == 200
            else None,
            error_message=None if status == 200 else "Identity unavailable",
        )
    )
    return TestClient(app)


def test_platform_routes_remain_gated_and_keep_ssr_destinations(tmp_path, monkeypatch):
    client = _platform_client(tmp_path, monkeypatch)
    for path in ("/backtests", "/backtests/new", "/backtests/local-job?variant=a%2Fb"):
        response = client.get(path)
        assert response.status_code == 200
        assert 'id="platform-root"' in response.text
        assert response.headers["cache-control"] == "private, no-store"
        assert "/platform-assets/assets/main.js" in response.text
    for path in ("/dashboard", "/strategies", "/strategies/local-id", "/settings"):
        response = client.get(path)
        assert response.status_code == 200
        assert 'id="platform-root"' not in response.text
    assert client.get("/backtests/a/b").status_code == 404
    assert client.get("/platform-assets/assets/main.js").status_code == 200
    assert client.get("/platform-assets/index.html").status_code == 404
    assert client.get("/platform-assets/.vite/manifest.json").status_code == 404

    revised = client.get("/backtests/new", params={"asset_version": 'layout&"<v2>'})
    assert revised.status_code == 200
    assert revised.headers["cache-control"] == "private, no-store"
    assert 'main.css?v=layout%26%22%3Cv2%3E' in revised.text
    assert 'main.js?v=layout%26%22%3Cv2%3E' in revised.text


def test_platform_anonymous_expired_and_unavailable_identity(tmp_path, monkeypatch):
    for cookie in (None, "roehub_session_id=expired"):
        client = _platform_client(tmp_path, monkeypatch, status=401)
        response = client.get(
            "/backtests/local-job?variant=a%2Fb",
            follow_redirects=False,
            headers={} if cookie is None else {"Cookie": cookie},
        )
        assert response.status_code == 307
        assert response.headers["location"] == (
            "/login?next=%2Fbacktests%2Flocal-job%3Fvariant%3Da%252Fb"
        )
        assert response.headers["cache-control"] == "private, no-store"
        assert "platform-bootstrap" not in response.text
    unavailable = _platform_client(tmp_path, monkeypatch, status=503).get("/backtests")
    assert unavailable.status_code == 502
    assert "platform-bootstrap" not in unavailable.text
    assert unavailable.headers["cache-control"] == "private, no-store"


def test_platform_disabled_returns_original_ssr_and_locale_is_server_resolved(
    tmp_path, monkeypatch
):
    client = _platform_client(tmp_path, monkeypatch, enabled=False)
    for path in ("/backtests", "/backtests/new", "/backtests/local-job"):
        response = client.get(path)
        assert 'data-page="backtests"' in response.text
        assert "platform-root" not in response.text
    enabled = _platform_client(tmp_path, monkeypatch)
    response = enabled.get("/locale?locale=ru&next=%2Fbacktests%2Fnew")
    assert '<html lang="ru">' in response.text
    assert '"locale": "ru"' in response.text
    external = enabled.get("/locale?locale=en&next=https://elsewhere.test", follow_redirects=False)
    assert external.headers["location"] == "/"


def test_platform_proxy_preserves_status_and_origin_without_granting_auth(tmp_path, monkeypatch):
    import httpx

    client = _platform_client(tmp_path, monkeypatch)
    observed = []

    def upstream(request):
        observed.append(request)
        return httpx.Response(401, json={"detail": "Unauthorized"})

    app = client.app
    assert isinstance(app, FastAPI)
    app.state.api_proxy_transport = httpx.MockTransport(upstream)
    response = client.post(
        "/api/backtests/jobs?limit=2",
        json={"example": True},
        headers={
            "Origin": "http://testserver",
            "Idempotency-Key": "same-key",
            "Cookie": "roehub_session_id=expired",
        },
    )
    assert response.status_code == 401
    assert str(observed[0].url) == "http://api.local/backtests/jobs?limit=2"
    assert observed[0].headers["origin"] == "http://testserver"
    assert observed[0].headers["idempotency-key"] == "same-key"
    assert observed[0].headers["cookie"] == "roehub_session_id=expired"


def test_platform_flag_requires_complete_build_and_rejects_invalid_values(tmp_path, monkeypatch):
    import pytest

    import apps.web.main.app as web_module

    monkeypatch.setattr(web_module, "DEFAULT_PLATFORM_DIST", tmp_path)
    environ = {"WEB_API_BASE_URL": "http://web.local", "WEB_API_UPSTREAM_URL": "http://api.local"}
    create_app(environ=environ)  # Missing build cannot affect default SSR startup.
    with pytest.raises(ValueError, match="complete platform-web build"):
        create_app(environ={**environ, "WEB_BACKTESTS_CLIENT_ENABLED": "true"})
    with pytest.raises(ValueError, match="must be true or false"):
        create_app(environ={**environ, "WEB_BACKTESTS_CLIENT_ENABLED": "yes"})
