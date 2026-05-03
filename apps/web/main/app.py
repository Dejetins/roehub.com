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
    AccountPreferencesApiClient,
    AccountPreferencesApiResult,
    CurrentUserApiClient,
    CurrentUserApiResult,
    HttpxAccountPreferencesApiClient,
    HttpxCurrentUserApiClient,
    WebAccountPreferences,
    WebCurrentUser,
)
from apps.web.main.i18n import (
    LOCALE_COOKIE_NAME,
    SUPPORTED_LOCALES,
    build_translator,
    load_catalogs,
    normalize_locale,
    resolve_locale,
    translate,
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
    label_key: str


@dataclass(frozen=True)
class _ThemeOption:
    key: str
    label_key: str


@dataclass(frozen=True)
class _LocaleOption:
    key: str
    label_key: str


_PRIMARY_NAV_ITEMS = (
    _NavItem(key="/", href="/", label_key="nav.home"),
    _NavItem(key="/dashboard", href="/dashboard", label_key="nav.dashboard"),
    _NavItem(key="/strategies", href="/strategies", label_key="nav.strategies"),
    _NavItem(key="/backtests", href="/backtests", label_key="nav.backtests"),
    _NavItem(key="/monitoring", href="/monitoring", label_key="nav.monitoring"),
    _NavItem(key="/settings", href="/settings", label_key="nav.settings"),
)
_THEME_OPTIONS = (
    _ThemeOption(key="terminal-orange", label_key="theme.terminal_orange"),
    _ThemeOption(key="graphite", label_key="theme.graphite"),
    _ThemeOption(key="matrix-green", label_key="theme.matrix_green"),
    _ThemeOption(key="high-contrast", label_key="theme.high_contrast"),
)
_THEME_KEYS = {theme.key for theme in _THEME_OPTIONS}
_LOCALE_OPTIONS = tuple(
    _LocaleOption(key=locale, label_key=f"locale.{locale}") for locale in SUPPORTED_LOCALES
)


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
    app.state.account_preferences_api_client = HttpxAccountPreferencesApiClient(
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
            page_title_key="page.landing.title",
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
            page_title_key="page.login.title",
            current_user=None,
            error_message=None,
        )
        context["oidc_login_url"] = _build_oidc_login_url(next_path=safe_next_path)
        return _render_template_response(
            request=request,
            templates=templates,
            template_name="pages/login.html",
            context=context,
        )

    @app.get("/register", response_class=HTMLResponse)
    def get_register_page(request: Request, next: str | None = None) -> Response:
        safe_next_path = sanitize_next_path(raw_next=next, default_path="/dashboard")
        context = _build_template_context(
            request=request,
            page_path="/register",
            page_title_key="page.register.title",
            current_user=None,
            error_message=None,
        )
        context["oidc_register_url"] = _build_oidc_login_url(next_path=safe_next_path)
        return _render_template_response(
            request=request,
            templates=templates,
            template_name="pages/register.html",
            context=context,
        )

    @app.get("/logout", response_class=HTMLResponse)
    def get_logout_page(request: Request, next: str | None = None) -> Response:
        post_logout_redirect_path = sanitize_next_path(raw_next=next, default_path="/login")
        context = _build_template_context(
            request=request,
            page_path="/logout",
            page_title_key="page.logout.title",
            current_user=None,
            error_message=None,
        )
        context["post_logout_redirect_path"] = post_logout_redirect_path
        context["logout_url"] = "/api/auth/logout"
        return _render_template_response(
            request=request,
            templates=templates,
            template_name="pages/logout.html",
            context=context,
        )

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
            page_title_key="page.dashboard.title",
            page_description_key="page.dashboard.description",
            template_name="pages/dashboard.html",
        )

    @app.get("/settings", response_class=HTMLResponse)
    def get_settings_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/settings",
            page_title_key="page.settings.title",
            page_description_key="page.settings.description",
            template_name="pages/settings.html",
            load_account_preferences=True,
        )

    @app.get("/strategies", response_class=HTMLResponse)
    def get_strategies_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title_key="page.strategies.title",
            page_description_key="page.strategies.description",
            template_name="pages/strategies.html",
        )

    @app.get("/strategies/new", response_class=HTMLResponse)
    def get_new_strategy_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title_key="page.strategy_new.title",
            page_description_key="page.strategy_new.description",
            template_name="pages/strategy_create.html",
        )

    @app.get("/strategies/{strategy_id}", response_class=HTMLResponse)
    def get_strategy_details_page(request: Request, strategy_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title_key="page.strategy_detail.title",
            page_description_key="page.strategy_detail.description",
            template_name="pages/strategy_detail.html",
            template_context={"strategy_id": strategy_id},
        )

    @app.get("/monitoring", response_class=HTMLResponse)
    def get_monitoring_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/monitoring",
            page_title_key="page.monitoring.title",
            page_description_key="page.monitoring.description",
            template_name="pages/monitoring.html",
        )

    @app.get("/backtests", response_class=HTMLResponse)
    def get_backtests_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title_key="page.backtests.title",
            page_description_key="page.backtests.description",
            template_name="pages/backtests_history.html",
        )

    @app.get("/backtests/new", response_class=HTMLResponse)
    def get_new_backtest_page(request: Request) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title_key="page.backtest_new.title",
            page_description_key="page.backtest_new.description",
            template_name="pages/backtests_run.html",
        )

    @app.get("/backtests/{job_id}", response_class=HTMLResponse)
    def get_backtest_result_page(request: Request, job_id: str) -> Response:
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title_key="page.backtest_result.title",
            page_description_key="page.backtest_result.description",
            template_name="pages/backtests_result.html",
            template_context={"job_id": job_id},
        )


def _resolve_current_user_api_client(*, request: Request) -> CurrentUserApiClient:
    api_client = getattr(request.app.state, "current_user_api_client", None)
    if api_client is None:
        raise ValueError("current_user_api_client is not configured in application state")
    return api_client


def _resolve_account_preferences_api_client(*, request: Request) -> AccountPreferencesApiClient:
    api_client = getattr(request.app.state, "account_preferences_api_client", None)
    if api_client is None:
        raise ValueError("account_preferences_api_client is not configured in application state")
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
    page_title_key: str,
    template_name: str,
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    context = _build_template_context(
        request=request,
        page_path=page_path,
        page_title_key=page_title_key,
        current_user=None,
        error_message=None,
    )
    if template_context is not None:
        context.update(template_context)
    return _render_template_response(
        request=request,
        templates=templates,
        template_name=template_name,
        context=context,
    )


def _render_protected_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    page_path: str,
    page_title_key: str,
    page_description_key: str,
    template_name: str = "pages/placeholder.html",
    load_account_preferences: bool = False,
    template_context: Mapping[str, Any] | None = None,
) -> Response:
    api_client = _resolve_current_user_api_client(request=request)
    api_result = api_client.fetch_current_user(cookie_header=request.headers.get("cookie"))

    if api_result.status_code == 401:
        return _build_login_redirect_response(current_path=_build_current_path(request=request))

    current_user = api_result.user if api_result.status_code == 200 else None
    preferences_result = _fetch_account_preferences(
        request=request,
        enabled=load_account_preferences and current_user is not None,
    )
    account_preferences = (
        preferences_result.preferences if preferences_result is not None else None
    )
    current_locale, should_set_locale_cookie = _resolve_locale_state(
        request=request,
        account_locale=account_preferences.locale if account_preferences is not None else None,
    )
    error_message = _build_api_error_message(api_result=api_result, locale=current_locale)
    status_code = 200 if current_user is not None else 502

    context = _build_template_context(
        request=request,
        page_path=page_path,
        page_title_key=page_title_key,
        current_user=current_user,
        error_message=error_message,
        locale=current_locale,
        should_set_locale_cookie=should_set_locale_cookie,
        account_preferences=account_preferences,
    )
    context["page_description"] = translate(locale=current_locale, key=page_description_key)
    context["page_description_key"] = page_description_key
    context["placeholder_id"] = page_path.strip("/").replace("/", "-") or "home"
    if template_context is not None:
        context.update(template_context)

    response = _render_template_response(
        request,
        templates=templates,
        template_name=template_name,
        context=context,
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _render_template_response(
    request: Request,
    *,
    templates: Jinja2Templates,
    template_name: str,
    context: dict[str, Any],
    status_code: int = 200,
) -> Response:
    response = templates.TemplateResponse(
        request,
        template_name,
        context=context,
        status_code=status_code,
    )
    if context.get("should_set_locale_cookie") is True:
        response.set_cookie(
            LOCALE_COOKIE_NAME,
            str(context["current_locale"]),
            max_age=31_536_000,
            httponly=False,
            samesite="lax",
        )
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


def _build_api_error_message(
    *,
    api_result: CurrentUserApiResult,
    locale: str,
) -> str | None:
    if api_result.status_code in (200, 401):
        return None
    if api_result.error_message is None:
        return translate(locale=locale, key="error.identity_api")
    return api_result.error_message


def _fetch_account_preferences(
    *,
    request: Request,
    enabled: bool,
) -> AccountPreferencesApiResult | None:
    if not enabled:
        return None
    api_client = _resolve_account_preferences_api_client(request=request)
    result = api_client.fetch_preferences(cookie_header=request.headers.get("cookie"))
    if result.status_code != 200:
        return None
    return result


def _build_template_context(
    *,
    request: Request,
    page_path: str,
    page_title_key: str,
    current_user: WebCurrentUser | None,
    error_message: str | None,
    locale: str | None = None,
    should_set_locale_cookie: bool | None = None,
    account_preferences: WebAccountPreferences | None = None,
) -> dict[str, Any]:
    current_theme = _resolve_theme(
        request=request,
        account_theme=account_preferences.theme if account_preferences is not None else None,
    )
    if locale is None or should_set_locale_cookie is None:
        current_locale, should_set_cookie = _resolve_locale_state(
            request=request,
            account_locale=account_preferences.locale if account_preferences is not None else None,
        )
    else:
        current_locale = locale
        should_set_cookie = should_set_locale_cookie
    t = build_translator(locale=current_locale)
    return {
        "request": request,
        "page_path": page_path,
        "page_title_key": page_title_key,
        "page_title": t(page_title_key),
        "current_user": current_user,
        "error_message": error_message,
        "current_theme": current_theme,
        "current_locale": current_locale,
        "should_set_locale_cookie": should_set_cookie,
        "t": t,
        "nav_items": _build_nav_items(page_path=page_path, locale=current_locale),
        "auth_actions": _build_auth_actions(
            page_path=page_path,
            current_user=current_user,
            locale=current_locale,
        ),
        "theme_options": _build_theme_options(
            request=request,
            current_theme=current_theme,
            locale=current_locale,
        ),
        "locale_options": _build_locale_options(request=request, current_locale=current_locale),
        "client_i18n_catalogs": load_catalogs(),
    }


def _resolve_locale_state(
    *,
    request: Request,
    account_locale: str | None = None,
) -> tuple[str, bool]:
    raw_query_locale = request.query_params.get("locale")
    resolved_locale = normalize_locale(account_locale) or resolve_locale(
        query_locale=raw_query_locale,
        cookie_locale=request.cookies.get(LOCALE_COOKIE_NAME),
        accept_language=request.headers.get("accept-language"),
    )
    should_set_cookie = normalize_locale(raw_query_locale) is not None
    return resolved_locale, should_set_cookie


def _build_nav_items(*, page_path: str, locale: str) -> list[dict[str, str | bool]]:
    return [
        {
            "key": item.key,
            "href": item.href,
            "label_key": item.label_key,
            "label": translate(locale=locale, key=item.label_key),
            "active": item.key == page_path,
        }
        for item in _PRIMARY_NAV_ITEMS
    ]


def _build_auth_actions(
    *,
    page_path: str,
    current_user: WebCurrentUser | None,
    locale: str,
) -> list[dict[str, str | bool]]:
    if current_user is not None:
        return [
            {
                "key": "/logout",
                "href": "/logout",
                "label_key": "auth.logout",
                "label": translate(locale=locale, key="auth.logout"),
                "active": page_path == "/logout",
            }
        ]
    return [
        {
            "key": "/login",
            "href": "/login",
            "label_key": "auth.login",
            "label": translate(locale=locale, key="auth.login"),
            "active": page_path == "/login",
        },
        {
            "key": "/register",
            "href": "/register",
            "label_key": "auth.register",
            "label": translate(locale=locale, key="auth.register"),
            "active": page_path == "/register",
        },
    ]


def _resolve_theme(*, request: Request, account_theme: str | None = None) -> str:
    if account_theme in _THEME_KEYS:
        return account_theme
    requested_theme = request.query_params.get("theme")
    if requested_theme in _THEME_KEYS:
        return requested_theme
    return _DEFAULT_THEME


def _build_theme_options(
    *,
    request: Request,
    current_theme: str,
    locale: str,
) -> list[dict[str, str | bool]]:
    theme_options: list[dict[str, str | bool]] = []
    for theme in _THEME_OPTIONS:
        theme_options.append(
            {
                "key": theme.key,
                "label": translate(locale=locale, key=theme.label_key),
                "href": _build_control_href(request=request, updates={"theme": theme.key}),
                "active": theme.key == current_theme,
            }
        )
    return theme_options


def _build_locale_options(
    *,
    request: Request,
    current_locale: str,
) -> list[dict[str, str | bool]]:
    return [
        {
            "key": locale.key,
            "label": translate(locale=current_locale, key=locale.label_key),
            "href": _build_control_href(request=request, updates={"locale": locale.key}),
            "active": locale.key == current_locale,
        }
        for locale in _LOCALE_OPTIONS
    ]


def _build_control_href(*, request: Request, updates: Mapping[str, str]) -> str:
    query_params = dict(request.query_params)
    if "next" in query_params:
        query_params["next"] = sanitize_next_path(raw_next=query_params["next"])
    query_params.update(updates)
    query = urlencode(query_params)
    if not query:
        return request.url.path
    return f"{request.url.path}?{query}"
