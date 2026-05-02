"""FastAPI application factory for Roehub Web SSR service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from apps.web.main.api_client import (
    CurrentUserApiClient,
    CurrentUserApiResult,
    HttpxCurrentUserApiClient,
    WebCurrentUser,
)
from apps.web.main.security import sanitize_next_path
from apps.web.main.settings import WebRuntimeSettings, resolve_web_runtime_settings

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES_PATH = _PACKAGE_ROOT / "templates"
_DIST_PATH = _PACKAGE_ROOT / "dist"
_DEFAULT_THEME = "terminal-orange"
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
    href: str
    label: str


@dataclass(frozen=True)
class _ThemeOption:
    key: str
    label: str


_PRIMARY_NAV_ITEMS = (
    _NavItem(key="/", href="/", label="Главная"),
    _NavItem(key="/dashboard", href="/dashboard", label="Панель"),
    _NavItem(key="/strategies", href="/strategies", label="Стратегии"),
    _NavItem(key="/backtests", href="/backtests", label="Backtests"),
    _NavItem(key="/monitoring", href="/monitoring", label="Мониторинг"),
    _NavItem(key="/settings", href="/settings", label="Настройки"),
)
_THEME_OPTIONS = (
    _ThemeOption(key="terminal-orange", label="Orange"),
    _ThemeOption(key="graphite", label="Graphite"),
    _ThemeOption(key="matrix-green", label="Matrix"),
    _ThemeOption(key="high-contrast", label="Contrast"),
)
_THEME_KEYS = {theme.key for theme in _THEME_OPTIONS}


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """Build FastAPI web app with SSR templates, static assets, and auth shell routes."""
    effective_environ = os.environ if environ is None else environ
    runtime_settings = resolve_web_runtime_settings(environ=effective_environ)

    templates = Jinja2Templates(directory=str(_TEMPLATES_PATH))
    app = FastAPI(title="Roehub Web", version="1.0.0")
    app.mount("/assets", StaticFiles(directory=str(_DIST_PATH)), name="assets")
    app.state.current_user_api_client = HttpxCurrentUserApiClient(
        api_base_url=runtime_settings.api_base_url
    )
    app.state.api_proxy_transport = None
    _register_routes(app=app, templates=templates, runtime_settings=runtime_settings)
    return app


def _register_routes(
    *,
    app: FastAPI,
    templates: Jinja2Templates,
    runtime_settings: WebRuntimeSettings,
) -> None:
    """Register web shell, auth entrypoints, protected placeholders, and API proxy."""

    @app.get("/", response_class=HTMLResponse)
    def get_landing_page(request: Request) -> Response:
        return _render_public_page(
            request=request,
            templates=templates,
            page_path="/",
            page_title="Roehub",
            template_name="pages/landing.html",
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def get_favicon() -> Response:
        return Response(status_code=204)

    @app.get("/login", response_class=HTMLResponse)
    def get_login_page(request: Request, next: str | None = None) -> Response:
        safe_next_path = sanitize_next_path(raw_next=next)
        context = _build_template_context(
            request=request,
            page_path="/login",
            page_title="Вход",
            current_user=None,
            error_message=None,
        )
        context["oidc_login_url"] = _build_oidc_login_url(next_path=safe_next_path)
        return templates.TemplateResponse(request, "pages/login.html", context=context)

    @app.get("/register", response_class=HTMLResponse)
    def get_register_page(request: Request, next: str | None = None) -> Response:
        safe_next_path = sanitize_next_path(raw_next=next, default_path="/dashboard")
        context = _build_template_context(
            request=request,
            page_path="/register",
            page_title="Регистрация",
            current_user=None,
            error_message=None,
        )
        context["oidc_register_url"] = _build_oidc_login_url(next_path=safe_next_path)
        return templates.TemplateResponse(request, "pages/register.html", context=context)

    @app.get("/logout", response_class=HTMLResponse)
    def get_logout_page(request: Request, next: str | None = None) -> Response:
        post_logout_redirect_path = sanitize_next_path(raw_next=next, default_path="/login")
        context = _build_template_context(
            request=request,
            page_path="/logout",
            page_title="Выход",
            current_user=None,
            error_message=None,
        )
        context["post_logout_redirect_path"] = post_logout_redirect_path
        context["logout_url"] = "/api/auth/logout"
        return templates.TemplateResponse(request, "pages/logout.html", context=context)

    @app.api_route(
        "/api/{upstream_path:path}",
        methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
    )
    async def proxy_api_request(request: Request, upstream_path: str) -> Response:
        upstream_url = f"{runtime_settings.api_upstream_url}/{upstream_path}"
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
        except httpx.HTTPError as error:
            return Response(
                status_code=502,
                content=f"API proxy request failed: {error}",
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
            page_title="Панель",
            page_description="Protected dashboard placeholder for Stage 2.",
        )

    @app.get("/settings", response_class=HTMLResponse)
    def get_settings_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/settings",
            page_title="Настройки",
            page_description="Protected settings placeholder for account preferences.",
        )

    @app.get("/strategies", response_class=HTMLResponse)
    def get_strategies_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title="Стратегии",
            page_description="Protected strategies placeholder for the Stage 6 package.",
        )

    @app.get("/strategies/new", response_class=HTMLResponse)
    def get_new_strategy_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title="Новая стратегия",
            page_description="Protected create-strategy entrypoint placeholder.",
            template_context={"placeholder_id": "strategies-new"},
        )

    @app.get("/strategies/{strategy_id}", response_class=HTMLResponse)
    def get_strategy_details_page(request: Request, strategy_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title="Стратегия",
            page_description="Protected strategy detail placeholder.",
            template_context={"placeholder_id": "strategy-detail", "entity_id": strategy_id},
        )

    @app.get("/monitoring", response_class=HTMLResponse)
    def get_monitoring_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/monitoring",
            page_title="Мониторинг",
            page_description="Protected monitoring placeholder for runtime state.",
        )

    @app.get("/backtests", response_class=HTMLResponse)
    def get_backtests_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title="Backtests",
            page_description="Protected backtests history placeholder.",
        )

    @app.get("/backtests/new", response_class=HTMLResponse)
    def get_new_backtest_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title="Новый backtest",
            page_description="Protected backtest configurator placeholder.",
            template_context={"placeholder_id": "backtests-new"},
        )

    @app.get("/backtests/{job_id}", response_class=HTMLResponse)
    def get_backtest_result_page(request: Request, job_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title="Backtest result",
            page_description="Protected backtest result placeholder.",
            template_context={"placeholder_id": "backtest-result", "entity_id": job_id},
        )


def _resolve_current_user_api_client(*, request: Request) -> CurrentUserApiClient:
    api_client = getattr(request.app.state, "current_user_api_client", None)
    if api_client is None:
        raise ValueError("current_user_api_client is not configured in application state")
    return api_client


def _build_proxy_request_headers(*, request: Request) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for header_name, header_value in request.headers.items():
        if header_name.lower() in _HOP_BY_HOP_HEADERS:
            continue
        forwarded_headers[header_name] = header_value
    return forwarded_headers


def _render_public_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    page_path: str,
    page_title: str,
    template_name: str,
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    context = _build_template_context(
        request=request,
        page_path=page_path,
        page_title=page_title,
        current_user=None,
        error_message=None,
    )
    if template_context is not None:
        context.update(template_context)
    return templates.TemplateResponse(request, template_name, context=context)


def _render_protected_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    page_path: str,
    page_title: str,
    page_description: str,
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    api_client = _resolve_current_user_api_client(request=request)
    api_result = api_client.fetch_current_user(cookie_header=request.headers.get("cookie"))

    if api_result.status_code == 401:
        return _build_login_redirect_response(current_path=_build_current_path(request=request))

    current_user = api_result.user if api_result.status_code == 200 else None
    error_message = _build_api_error_message(api_result=api_result)
    status_code = 200 if current_user is not None else 502

    context = _build_template_context(
        request=request,
        page_path=page_path,
        page_title=page_title,
        current_user=current_user,
        error_message=error_message,
    )
    context["page_description"] = page_description
    context["placeholder_id"] = page_path.strip("/").replace("/", "-") or "home"
    if template_context is not None:
        context.update(template_context)

    response = templates.TemplateResponse(
        request,
        "pages/placeholder.html",
        context=context,
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _build_login_redirect_response(*, current_path: str) -> RedirectResponse:
    safe_next_path = sanitize_next_path(raw_next=current_path)
    query = urlencode({"next": safe_next_path})
    return RedirectResponse(url=f"/login?{query}")


def _build_oidc_login_url(*, next_path: str) -> str:
    safe_next_path = sanitize_next_path(raw_next=next_path)
    query = urlencode({"next": safe_next_path})
    return f"/api/auth/login?{query}"


def _build_current_path(*, request: Request) -> str:
    query = str(request.url.query)
    if not query:
        return request.url.path
    return f"{request.url.path}?{query}"


def _build_api_error_message(*, api_result: CurrentUserApiResult) -> str | None:
    if api_result.status_code in (200, 401):
        return None
    if api_result.error_message is None:
        return "Identity API request failed"
    return api_result.error_message


def _build_template_context(
    *,
    request: Request,
    page_path: str,
    page_title: str,
    current_user: WebCurrentUser | None,
    error_message: str | None,
) -> dict[str, Any]:
    current_theme = _resolve_theme(request=request)
    return {
        "request": request,
        "page_path": page_path,
        "page_title": page_title,
        "current_user": current_user,
        "error_message": error_message,
        "current_theme": current_theme,
        "nav_items": _build_nav_items(page_path=page_path),
        "auth_actions": _build_auth_actions(page_path=page_path, current_user=current_user),
        "theme_options": _build_theme_options(request=request, current_theme=current_theme),
    }


def _build_nav_items(*, page_path: str) -> list[dict[str, str | bool]]:
    return [
        {
            "key": item.key,
            "href": item.href,
            "label": item.label,
            "active": item.key == page_path,
        }
        for item in _PRIMARY_NAV_ITEMS
    ]


def _build_auth_actions(
    *,
    page_path: str,
    current_user: WebCurrentUser | None,
) -> list[dict[str, str | bool]]:
    if current_user is not None:
        return [
            {
                "key": "/logout",
                "href": "/logout",
                "label": "Выйти",
                "active": page_path == "/logout",
            }
        ]
    return [
        {
            "key": "/login",
            "href": "/login",
            "label": "Войти",
            "active": page_path == "/login",
        },
        {
            "key": "/register",
            "href": "/register",
            "label": "Регистрация",
            "active": page_path == "/register",
        },
    ]


def _resolve_theme(*, request: Request) -> str:
    requested_theme = request.query_params.get("theme")
    if requested_theme in _THEME_KEYS:
        return requested_theme
    return _DEFAULT_THEME


def _build_theme_options(
    *,
    request: Request,
    current_theme: str,
) -> list[dict[str, str | bool]]:
    query_params = dict(request.query_params)
    theme_options: list[dict[str, str | bool]] = []
    for theme in _THEME_OPTIONS:
        query_params["theme"] = theme.key
        theme_options.append(
            {
                "key": theme.key,
                "label": theme.label,
                "href": f"{request.url.path}?{urlencode(query_params)}",
                "active": theme.key == current_theme,
            }
        )
    return theme_options
