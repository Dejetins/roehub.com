"""FastAPI application factory for Roehub Web SSR service."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from apps.web.main.api_client import (
    CurrentUserApiClient,
    CurrentUserApiResult,
    HttpxCurrentUserApiClient,
    WebCurrentUser,
)
from apps.web.main.i18n import (
    DEFAULT_LOCALE,
    LOCALE_COOKIE_NAME,
    SUPPORTED_LOCALES,
    load_catalog,
    normalize_locale,
    resolve_locale,
    translate,
)
from apps.web.main.security import sanitize_next_path
from apps.web.main.settings import WebRuntimeSettings, resolve_web_runtime_settings

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES_PATH = _PACKAGE_ROOT / "templates"
_DIST_PATH = _PACKAGE_ROOT / "dist"
_DEFAULT_RUNBOOKS_PATH = Path(__file__).resolve().parents[3] / "docs/runbooks/generated/ru"
_RUNBOOK_ID = re.compile(r"^[a-z][a-z0-9.-]{1,127}$")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class _NavItem:
    key: str
    label_key: str
    short_label: str
    path: str
    active_path: str


@dataclass(frozen=True)
class _ProtectedPage:
    page_path: str
    active_path: str
    title_key: str
    description_key: str


_NAV_ITEMS: tuple[_NavItem, ...] = (
    _NavItem(
        key="dashboard",
        label_key="nav.dashboard",
        short_label="OV",
        path="/dashboard",
        active_path="/dashboard",
    ),
    _NavItem(
        key="strategies",
        label_key="nav.strategies",
        short_label="ST",
        path="/strategies",
        active_path="/strategies",
    ),
    _NavItem(
        key="backtests",
        label_key="nav.backtests",
        short_label="BT",
        path="/backtests",
        active_path="/backtests",
    ),
    _NavItem(
        key="monitoring",
        label_key="nav.monitoring",
        short_label="LV",
        path="/monitoring",
        active_path="/monitoring",
    ),
    _NavItem(
        key="models",
        label_key="nav.models",
        short_label="ML",
        path="/models",
        active_path="/models",
    ),
    _NavItem(
        key="connections",
        label_key="nav.connections",
        short_label="CN",
        path="/connections",
        active_path="/connections",
    ),
    _NavItem(
        key="admin",
        label_key="nav.admin",
        short_label="AD",
        path="/admin",
        active_path="/admin",
    ),
    _NavItem(
        key="settings",
        label_key="nav.settings",
        short_label="SE",
        path="/settings",
        active_path="/settings",
    ),
)
_PROTECTED_PAGES: dict[str, _ProtectedPage] = {
    "/dashboard": _ProtectedPage(
        page_path="/dashboard",
        active_path="/dashboard",
        title_key="page.dashboard.title",
        description_key="page.dashboard.desc",
    ),
    "/settings": _ProtectedPage(
        page_path="/settings",
        active_path="/settings",
        title_key="page.settings.title",
        description_key="page.settings.desc",
    ),
    "/strategies": _ProtectedPage(
        page_path="/strategies",
        active_path="/strategies",
        title_key="page.strategies.title",
        description_key="page.strategies.desc",
    ),
    "/strategies/new": _ProtectedPage(
        page_path="/strategies/new",
        active_path="/strategies",
        title_key="page.strategies.new_title",
        description_key="page.strategies.new_desc",
    ),
    "/backtests": _ProtectedPage(
        page_path="/backtests",
        active_path="/backtests",
        title_key="page.backtests.title",
        description_key="page.backtests.desc",
    ),
    "/backtests/new": _ProtectedPage(
        page_path="/backtests/new",
        active_path="/backtests",
        title_key="page.backtests.title",
        description_key="page.backtests.desc",
    ),
    "/monitoring": _ProtectedPage(
        page_path="/monitoring",
        active_path="/monitoring",
        title_key="page.monitoring.title",
        description_key="page.monitoring.desc",
    ),
    "/models": _ProtectedPage(
        page_path="/models",
        active_path="/models",
        title_key="page.models.title",
        description_key="page.models.desc",
    ),
    "/connections": _ProtectedPage(
        page_path="/connections",
        active_path="/connections",
        title_key="page.connections.title",
        description_key="page.connections.desc",
    ),
    "/admin": _ProtectedPage(
        page_path="/admin",
        active_path="/admin",
        title_key="page.admin.title",
        description_key="page.admin.desc",
    ),
}
_THEME_OPTIONS: tuple[dict[str, str], ...] = (
    {"key": "abyss", "label_key": "theme.abyss"},
    {"key": "graphite", "label_key": "theme.graphite"},
    {"key": "slate", "label_key": "theme.slate"},
    {"key": "frost", "label_key": "theme.frost"},
    {"key": "paper", "label_key": "theme.paper"},
    {"key": "sand", "label_key": "theme.sand"},
)
_REFRESH_PRESETS: tuple[dict[str, str], ...] = (
    {"key": "off", "label": "Off"},
    {"key": "10s", "label": "10s"},
    {"key": "15s", "label": "15s"},
    {"key": "30s", "label": "30s"},
    {"key": "1m", "label": "1m"},
    {"key": "5m", "label": "5m"},
)


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """Build FastAPI web app with SSR templates, assets, auth gate, and API proxy."""
    effective_environ = os.environ if environ is None else environ
    runtime_settings = resolve_web_runtime_settings(environ=effective_environ)

    templates = Jinja2Templates(directory=str(_TEMPLATES_PATH))
    app = FastAPI(title="Roehub Web", version="1.0.0")
    app.mount("/assets", StaticFiles(directory=str(_DIST_PATH)), name="assets")
    app.state.current_user_api_client = HttpxCurrentUserApiClient(
        api_base_url=runtime_settings.api_base_url
    )
    app.state.api_proxy_transport = None
    app.state.asset_version = _resolve_asset_version(environ=effective_environ)
    app.state.runbooks_path = Path(
        effective_environ.get("ROEHUB_RUNBOOKS_PATH", str(_DEFAULT_RUNBOOKS_PATH))
    )
    app.state.plugin_panel_lab_enabled = (
        effective_environ.get("ROEHUB_PLUGIN_PANEL_LAB", "false").lower() == "true"
    )
    lab_instance_id = effective_environ.get(
        "ROEHUB_PLUGIN_PANEL_LAB_INSTANCE_ID",
        "00000000-0000-0000-0000-000000000000",
    )
    try:
        app.state.plugin_panel_lab_instance_id = str(UUID(lab_instance_id))
    except ValueError as error:
        raise ValueError("ROEHUB_PLUGIN_PANEL_LAB_INSTANCE_ID must be a UUID") from error
    _register_routes(app=app, templates=templates, runtime_settings=runtime_settings)
    return app


def _register_routes(
    *,
    app: FastAPI,
    templates: Jinja2Templates,
    runtime_settings: WebRuntimeSettings,
) -> None:
    """Register public shell, auth UX, protected placeholders, and `/api/*` proxy."""

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, bool]:
        return {"live": True}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> Response:
        try:
            response = httpx.get(
                f"{runtime_settings.api_upstream_url}/health",
                timeout=2.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return Response(
                content=json.dumps(
                    {"ready": False, "reason": "api_unavailable"},
                    sort_keys=True,
                ),
                status_code=503,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"ready": True}, sort_keys=True),
            status_code=200,
            media_type="application/json",
        )

    @app.get("/", response_class=HTMLResponse)
    def get_landing_page(request: Request) -> Response:
        if _resolve_optional_public_current_user(request=request) is not None:
            return RedirectResponse(url="/dashboard")
        return _render_public_page(
            request=request,
            templates=templates,
            template_name="pages/login.html",
            page_path="/",
            active_path="/",
            title_key="auth.login_title",
            open_login_modal=False,
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def get_favicon() -> Response:
        return FileResponse(_DIST_PATH / "favicon.svg", media_type="image/svg+xml")

    @app.get("/locale", include_in_schema=False)
    def set_locale(
        request: Request,
        locale: str | None = None,
        next: str | None = None,
    ) -> Response:
        selected_locale = normalize_locale(locale)
        redirect_path = sanitize_next_path(raw_next=next)
        response = RedirectResponse(url=redirect_path)
        response.set_cookie(
            key=LOCALE_COOKIE_NAME,
            value=selected_locale,
            max_age=60 * 60 * 24 * 365,
            path="/",
            secure=False,
            httponly=False,
            samesite="lax",
        )
        return response

    @app.get("/login", response_class=HTMLResponse)
    def get_login_page(request: Request, next: str | None = None) -> Response:
        if _resolve_optional_public_current_user(request=request) is not None:
            return RedirectResponse(url="/dashboard")
        return _render_public_page(
            request=request,
            templates=templates,
            template_name="pages/login.html",
            page_path="/login",
            active_path="/",
            title_key="auth.login_title",
            open_login_modal=False,
            auth_next_path=sanitize_next_path(raw_next=next),
        )

    @app.get("/logout", response_class=HTMLResponse)
    def get_logout_page(request: Request, next: str | None = None) -> Response:
        post_logout_redirect_path = sanitize_next_path(raw_next=next, default_path="/login")
        context = _build_template_context(
            request=request,
            page_path="/logout",
            active_path="/",
            page_title_key="auth.logout",
            current_user=None,
            error_message=None,
            auth_next_path=post_logout_redirect_path,
        )
        context["post_logout_redirect_path"] = post_logout_redirect_path
        return templates.TemplateResponse(request, "pages/logout.html", context=context)

    @app.get("/register", response_class=HTMLResponse)
    def get_register_page(request: Request, next: str | None = None) -> Response:
        safe_next_path = sanitize_next_path(raw_next=next, default_path="/dashboard")
        context = _build_template_context(
            request=request,
            page_path="/register",
            active_path="/",
            page_title_key="auth.register_title",
            current_user=None,
            error_message=None,
            auth_next_path=safe_next_path,
        )
        context["register_entrypoint_url"] = _build_oidc_login_url(next_path=safe_next_path)
        return templates.TemplateResponse(request, "pages/register.html", context=context)

    @app.api_route(
        "/api/{upstream_path:path}",
        methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
    )
    async def proxy_api_request(request: Request, upstream_path: str) -> Response:
        upstream_suffix = (
            f"api/{upstream_path}" if upstream_path.startswith("v1/") else upstream_path
        )
        upstream_url = f"{runtime_settings.api_upstream_url}/{upstream_suffix}"
        request_body = await request.body()
        request_headers = _build_proxy_request_headers(request=request)
        transport = getattr(request.app.state, "api_proxy_transport", None)
        client_kwargs: dict[str, Any] = {"timeout": 30.0}
        if transport is not None:
            client_kwargs["transport"] = transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as http_client:
                upstream_response = await http_client.request(
                    request.method,
                    upstream_url,
                    content=request_body,
                    headers=request_headers,
                    params=request.query_params,
                )
        except httpx.HTTPError:
            return Response(
                status_code=502,
                content="API proxy request failed",
                media_type="text/plain",
            )

        proxied_response = Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
        )
        for header_name, header_value in upstream_response.headers.multi_items():
            if header_name.lower() in _HOP_BY_HOP_HEADERS:
                continue
            proxied_response.headers.append(header_name, header_value)
        return proxied_response

    @app.get("/dashboard", response_class=HTMLResponse)
    def get_dashboard_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/dashboard",
            active_path="/dashboard",
            page_title_key="page.dashboard.title",
            page_description_key="page.dashboard.desc",
            template_name="pages/dashboard.html",
        )

    @app.get("/__qa/plugin-panels", response_class=HTMLResponse, include_in_schema=False)
    def get_plugin_panel_lab_page(request: Request) -> Response:
        if not bool(getattr(request.app.state, "plugin_panel_lab_enabled", False)):
            raise HTTPException(status_code=404)
        instance_id = str(request.app.state.plugin_panel_lab_instance_id)
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/__qa/plugin-panels",
            active_path="/dashboard",
            page_title_key="plugin_panel_lab.page_title",
            page_description_key="plugin_panel_lab.page_description",
            template_name="pages/plugin_panel_lab.html",
            template_context={
                "plugin_panel_instance_id": instance_id,
                "plugin_panel_query_endpoint": (
                    f"/api/v1/plugins/data-sources/{instance_id}:query"
                ),
            },
        )

    @app.get("/settings", response_class=HTMLResponse)
    def get_settings_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/settings",
            active_path="/settings",
            page_title_key="page.settings.title",
            page_description_key="page.settings.desc",
            template_name="pages/settings.html",
        )

    @app.get("/strategies", response_class=HTMLResponse)
    def get_strategies_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            active_path="/strategies",
            page_title_key="page.strategies.title",
            page_description_key="page.strategies.desc",
            template_name="pages/strategies.html",
            template_context={
                "strategy_initial_mode": request.query_params.get("mode") or "dashboard",
                "strategy_initial_id": request.query_params.get("strategy_id") or "",
            },
        )

    @app.get("/strategies/new", response_class=HTMLResponse)
    def get_new_strategy_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            active_path="/strategies",
            page_title_key="page.strategies.new_title",
            page_description_key="page.strategies.new_desc",
            template_name="pages/strategies.html",
            template_context={
                "strategy_initial_mode": "dashboard",
                "strategy_initial_id": "",
            },
        )

    @app.get("/strategies/{strategy_id}", response_class=HTMLResponse)
    def get_strategy_details_page(request: Request, strategy_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            active_path="/strategies",
            page_title_key="page.strategies.title",
            page_description_key="page.strategies.desc",
            template_name="pages/strategies.html",
            template_context={
                "strategy_initial_mode": "dashboard",
                "strategy_initial_id": strategy_id,
            },
        )

    @app.get("/backtests", response_class=HTMLResponse)
    def get_backtests_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            active_path="/backtests",
            page_title_key="page.backtests.title",
            page_description_key="page.backtests.desc",
            template_name="pages/backtests.html",
            template_context={
                "backtest_initial_mode": request.query_params.get("mode") or "workstation",
                "backtest_initial_job_id": request.query_params.get("job_id") or "",
            },
        )

    @app.get("/backtests/new", response_class=HTMLResponse)
    def get_new_backtest_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            active_path="/backtests",
            page_title_key="page.backtests.title",
            page_description_key="page.backtests.desc",
            template_name="pages/backtests.html",
            template_context={
                "backtest_initial_mode": "create",
                "backtest_initial_job_id": "",
            },
        )

    @app.get("/backtests/{job_id}", response_class=HTMLResponse)
    def get_backtest_deep_link(request: Request, job_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            active_path="/backtests",
            page_title_key="page.backtests.title",
            page_description_key="page.backtests.desc",
            template_name="pages/backtests.html",
            template_context={
                "backtest_initial_mode": "selected_job",
                "backtest_initial_job_id": job_id,
            },
        )

    @app.get("/monitoring", response_class=HTMLResponse)
    def get_monitoring_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/monitoring",
            active_path="/monitoring",
            page_title_key="page.monitoring.title",
            page_description_key="page.monitoring.desc",
            template_name="pages/monitoring.html",
        )

    @app.get("/models", response_class=HTMLResponse)
    def get_models_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/models",
            active_path="/models",
            page_title_key="page.models.title",
            page_description_key="page.models.desc",
            template_name="pages/strategies.html",
            template_context={
                "strategy_initial_mode": "rl_ml",
                "strategy_initial_id": request.query_params.get("strategy_id") or "",
            },
        )

    @app.get("/connections", response_class=HTMLResponse)
    def get_connections_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/connections",
            active_path="/connections",
            page_title_key="page.connections.title",
            page_description_key="page.connections.desc",
            template_name="pages/connections.html",
        )

    @app.get("/admin", response_class=HTMLResponse)
    def get_admin_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/admin",
            active_path="/admin",
            page_title_key="page.admin.title",
            page_description_key="page.admin.desc",
            template_name="pages/admin.html",
        )

    @app.get("/runbooks/{runbook_id}", response_class=HTMLResponse)
    def get_runbook_page(request: Request, runbook_id: str) -> Response:
        if _RUNBOOK_ID.fullmatch(runbook_id) is None:
            raise HTTPException(status_code=404)
        runbook_path = Path(request.app.state.runbooks_path) / f"{runbook_id}.md"
        if not runbook_path.is_file():
            raise HTTPException(status_code=404)
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path=f"/runbooks/{runbook_id}",
            active_path="/admin",
            page_title_key="page.admin.title",
            template_name="pages/runbook.html",
            template_context={
                "runbook_id": runbook_id,
                "runbook_content": runbook_path.read_text(encoding="utf-8"),
            },
        )


def _render_public_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    template_name: str,
    page_path: str,
    active_path: str,
    title_key: str,
    open_login_modal: bool,
    auth_next_path: str = "/dashboard",
) -> Response:
    current_user = _resolve_optional_public_current_user(request=request)
    context = _build_template_context(
        request=request,
        page_path=page_path,
        active_path=active_path,
        page_title_key=title_key,
        current_user=current_user,
        error_message=None,
        auth_next_path=auth_next_path,
    )
    context["open_login_modal"] = open_login_modal
    return templates.TemplateResponse(request, template_name, context=context)


def _render_protected_placeholder(
    *,
    request: Request,
    templates: Jinja2Templates,
    page: _ProtectedPage,
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    return _render_protected_page(
        request=request,
        templates=templates,
        page_path=page.page_path,
        active_path=page.active_path,
        page_title_key=page.title_key,
        page_description_key=page.description_key,
        template_name="pages/placeholder.html",
        template_context=template_context,
    )


def _resolve_current_user_api_client(*, request: Request) -> CurrentUserApiClient:
    api_client = getattr(request.app.state, "current_user_api_client", None)
    if api_client is None:
        raise ValueError("current_user_api_client is not configured in application state")
    return api_client


def _resolve_optional_public_current_user(*, request: Request) -> WebCurrentUser | None:
    cookie_header = request.headers.get("cookie")
    if not _has_auth_cookie(cookie_header=cookie_header):
        return None
    api_client = _resolve_current_user_api_client(request=request)
    api_result = api_client.fetch_current_user(cookie_header=cookie_header)
    if api_result.status_code == 200:
        return api_result.user
    return None


def _has_auth_cookie(*, cookie_header: str | None) -> bool:
    if not cookie_header:
        return False
    return any(
        cookie.strip().startswith("roehub_session_id=")
        for cookie in cookie_header.split(";")
    )


def _build_proxy_request_headers(*, request: Request) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for header_name, header_value in request.headers.items():
        if header_name.lower() in _HOP_BY_HOP_HEADERS:
            continue
        forwarded_headers[header_name] = header_value
    existing_header_names = {header_name.lower() for header_name in forwarded_headers}
    if "x-forwarded-host" not in existing_header_names:
        forwarded_host = request.headers.get("host")
        if forwarded_host:
            forwarded_headers["x-forwarded-host"] = forwarded_host
    if "x-forwarded-proto" not in existing_header_names:
        forwarded_headers["x-forwarded-proto"] = request.url.scheme
    return forwarded_headers


def _render_protected_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    page_path: str,
    active_path: str,
    page_title_key: str,
    page_description_key: str | None = None,
    template_name: str = "pages/placeholder.html",
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    api_client = _resolve_current_user_api_client(request=request)
    api_result = api_client.fetch_current_user(cookie_header=request.headers.get("cookie"))

    if api_result.status_code == 401:
        return _build_login_redirect_response(current_path=request.url.path)

    current_user = api_result.user if api_result.status_code == 200 else None
    error_message = _build_api_error_message(api_result=api_result)
    status_code = 200 if current_user is not None else 502

    context = _build_template_context(
        request=request,
        page_path=page_path,
        active_path=active_path,
        page_title_key=page_title_key,
        current_user=current_user,
        error_message=error_message,
        auth_next_path=page_path,
    )
    if page_description_key is not None:
        context["page_description"] = translate(
            locale=context["locale"],
            key=page_description_key,
        )
    if template_context is not None:
        context.update(template_context)
    response = templates.TemplateResponse(
        request,
        template_name,
        context=context,
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "private, no-store"
    return response


def _build_login_redirect_response(*, current_path: str) -> RedirectResponse:
    safe_next_path = sanitize_next_path(raw_next=current_path)
    query = urlencode({"next": safe_next_path})
    return RedirectResponse(url=f"/login?{query}")


def _build_api_error_message(*, api_result: CurrentUserApiResult) -> str | None:
    if api_result.status_code == 200:
        return None
    if api_result.status_code == 401:
        return None
    if api_result.error_message is None:
        return "Identity API request failed"
    return api_result.error_message


def _build_template_context(
    *,
    request: Request,
    page_path: str,
    active_path: str,
    page_title_key: str,
    current_user: WebCurrentUser | None,
    error_message: str | None,
    auth_next_path: str,
) -> dict[str, Any]:
    locale = resolve_locale(request=request)
    current_browser_path = _build_current_browser_path(request=request)
    page_title = translate(locale=locale, key=page_title_key)
    safe_auth_next_path = sanitize_next_path(raw_next=auth_next_path)
    locale_catalogs = {
        catalog_locale: load_catalog(locale=catalog_locale)
        for catalog_locale in SUPPORTED_LOCALES
    }
    asset_version = str(getattr(request.app.state, "asset_version", "dev"))
    return {
        "request": request,
        "page_path": page_path,
        "active_path": active_path,
        "page_title": page_title,
        "current_user": current_user,
        "error_message": error_message,
        "locale": locale,
        "default_locale": DEFAULT_LOCALE,
        "supported_locales": SUPPORTED_LOCALES,
        "locale_catalogs_json": json.dumps(locale_catalogs, ensure_ascii=False, sort_keys=True),
        "asset_url": lambda path: _build_asset_url(path=path, asset_version=asset_version),
        "theme_options": _THEME_OPTIONS,
        "refresh_presets": _REFRESH_PRESETS,
        "t": lambda key: translate(locale=locale, key=key),
        "nav_items": _build_nav_items(locale=locale, active_path=active_path),
        "language_options": _build_language_options(
            locale=locale,
            current_browser_path=current_browser_path,
        ),
        "open_login_modal": False,
        "auth_next_path": safe_auth_next_path,
        "oidc_login_url": _build_oidc_login_url(next_path=safe_auth_next_path),
    }


def _build_current_browser_path(*, request: Request) -> str:
    if request.url.query:
        return sanitize_next_path(raw_next=f"{request.url.path}?{request.url.query}")
    return sanitize_next_path(raw_next=request.url.path)


def _build_nav_items(*, locale: str, active_path: str) -> list[dict[str, str | bool]]:
    return [
        {
            "key": item.key,
            "label": translate(locale=locale, key=item.label_key),
            "short_label": item.short_label,
            "path": item.path,
            "is_active": item.active_path == active_path,
        }
        for item in _NAV_ITEMS
    ]


def _resolve_asset_version(*, environ: Mapping[str, str]) -> str:
    explicit_version = (
        environ.get("WEB_ASSET_VERSION")
        or environ.get("ROEHUB_WEB_ASSET_VERSION")
        or ""
    ).strip()
    if explicit_version:
        return explicit_version
    try:
        return str(
            max(
                path.stat().st_mtime_ns
                for path in _DIST_PATH.rglob("*")
                if path.is_file()
            )
        )
    except (OSError, ValueError):
        return "dev"


def _build_asset_url(*, path: str, asset_version: str) -> str:
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}v={asset_version}"


def _build_language_options(
    *,
    locale: str,
    current_browser_path: str,
) -> list[dict[str, str | bool]]:
    return [
        {
            "code": option,
            "label": option.upper(),
            "is_active": option == locale,
            "url": f"/locale?{urlencode({'locale': option, 'next': current_browser_path})}",
        }
        for option in SUPPORTED_LOCALES
    ]


def _build_oidc_login_url(*, next_path: str) -> str:
    safe_next_path = sanitize_next_path(raw_next=next_path)
    return f"/api/auth/oidc/login?{urlencode({'next': safe_next_path})}"
