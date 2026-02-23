"""FastAPI application factory for Roehub Web SSR service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

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

_BOT_USERNAME = "RoehubAuth_bot"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES_PATH = _PACKAGE_ROOT / "templates"
_DIST_PATH = _PACKAGE_ROOT / "dist"



def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """
    Build FastAPI web app with SSR templates, static assets, and auth page skeleton.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
      - docs/architecture/roadmap/milestone-6-epics-v1.md
    Related:
      - apps/web/main/main.py
      - apps/web/main/settings.py
      - apps/web/main/api_client.py

    Args:
        environ: Optional process environment mapping override.
    Returns:
        FastAPI: Configured Roehub Web application instance.
    Assumptions:
        `WEB_API_BASE_URL` is configured before web process startup.
    Raises:
        ValueError: If runtime settings are invalid.
    Side Effects:
        Reads environment mapping and configures static/template runtime wiring.
    """
    effective_environ = os.environ if environ is None else environ
    runtime_settings = resolve_web_runtime_settings(environ=effective_environ)

    templates = Jinja2Templates(directory=str(_TEMPLATES_PATH))
    app = FastAPI(title="Roehub Web", version="1.0.0")
    app.mount("/assets", StaticFiles(directory=str(_DIST_PATH)), name="assets")
    app.state.current_user_api_client = HttpxCurrentUserApiClient(
        api_base_url=runtime_settings.api_base_url
    )
    _register_routes(app=app, templates=templates, runtime_settings=runtime_settings)
    return app



def _register_routes(
    *,
    app: FastAPI,
    templates: Jinja2Templates,
    runtime_settings: WebRuntimeSettings,
) -> None:
    """
    Register all web routes for landing, auth UX, and protected skeleton pages.

    Args:
        app: FastAPI application instance.
        templates: Template renderer for SSR pages.
        runtime_settings: Validated runtime settings payload.
    Returns:
        None.
    Assumptions:
        API client is already attached to `app.state.current_user_api_client`.
    Raises:
        ValueError: If runtime settings are invalid.
    Side Effects:
        Adds HTTP routes to the FastAPI application.
    """
    _ = runtime_settings

    @app.get("/", response_class=HTMLResponse)
    def get_landing_page(request: Request) -> Response:
        """
        Render public landing page.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML landing page response.
        Assumptions:
            Landing route is always public and does not require auth checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        return templates.TemplateResponse(
            request,
            "landing.html",
            context=_build_template_context(
                request=request,
                page_path="/",
                page_title="Roehub",
                current_user=None,
                error_message=None,
            ),
        )

    @app.get("/login", response_class=HTMLResponse)
    def get_login_page(request: Request, next: str | None = None) -> Response:
        """
        Render login page with Telegram Login Widget and guarded next redirect target.

        Args:
            request: HTTP request object.
            next: Optional redirect target requested by caller.
        Returns:
            Response: HTML login page response.
        Assumptions:
            Login widget uses Telegram Variant A callback with browser-side API call.
        Raises:
            None.
        Side Effects:
            None.
        """
        safe_next_path = sanitize_next_path(raw_next=next)
        context = _build_template_context(
            request=request,
            page_path="/login",
            page_title="Login",
            current_user=None,
            error_message=None,
        )
        context["bot_username"] = _BOT_USERNAME
        context["next_path"] = safe_next_path
        return templates.TemplateResponse(request, "login.html", context=context)

    @app.get("/logout", response_class=HTMLResponse)
    def get_logout_page(request: Request) -> Response:
        """
        Render logout page that calls API logout endpoint via browser-side JavaScript.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML logout page response.
        Assumptions:
            API endpoint `/api/auth/logout` clears auth cookie.
        Raises:
            None.
        Side Effects:
            None.
        """
        return templates.TemplateResponse(
            request,
            "logout.html",
            context=_build_template_context(
                request=request,
                page_path="/logout",
                page_title="Logout",
                current_user=None,
                error_message=None,
            ),
        )

    @app.get("/strategies", response_class=HTMLResponse)
    def get_strategies_page(request: Request) -> Response:
        """
        Render protected strategies skeleton page behind current-user login gate.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML strategies page or login redirect response.
        Assumptions:
            Identity API determines current user via forwarded cookies.
        Raises:
            None.
        Side Effects:
            May perform server-side API request to `/api/auth/current-user`.
        """
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/strategies",
            page_title="Strategies",
            page_description=(
                "Strategy UI arrives in WEB-EPIC-04. This page is a protected skeleton."
            ),
        )

    @app.get("/backtests", response_class=HTMLResponse)
    def get_backtests_page(request: Request) -> Response:
        """
        Render protected backtests skeleton page behind current-user login gate.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML backtests page or login redirect response.
        Assumptions:
            Identity API determines current user via forwarded cookies.
        Raises:
            None.
        Side Effects:
            May perform server-side API request to `/api/auth/current-user`.
        """
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests",
            page_title="Backtests",
            page_description=(
                "Backtest UI arrives in WEB-EPIC-05. This page is a protected skeleton."
            ),
        )

    @app.get("/backtests/jobs", response_class=HTMLResponse)
    def get_backtest_jobs_page(request: Request) -> Response:
        """
        Render protected backtest jobs skeleton page behind current-user login gate.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML jobs page or login redirect response.
        Assumptions:
            Identity API determines current user via forwarded cookies.
        Raises:
            None.
        Side Effects:
            May perform server-side API request to `/api/auth/current-user`.
        """
        return _render_protected_page(
            request=request,
            templates=templates,
            page_path="/backtests/jobs",
            page_title="Backtest Jobs",
            page_description=(
                "Backtest jobs UI arrives in WEB-EPIC-06. This page is a protected skeleton."
            ),
        )

    @app.get("/_partial/user_badge", response_class=HTMLResponse)
    def get_user_badge_partial(request: Request) -> Response:
        """
        Render HTMX partial snippet for authenticated user badge.

        Args:
            request: HTTP request object.
        Returns:
            Response: HTML badge snippet with status reflecting auth/API result.
        Assumptions:
            Endpoint is consumed by HTMX from protected SSR pages.
        Raises:
            None.
        Side Effects:
            Performs server-side identity API call for current user lookup.
        """
        api_client = _resolve_current_user_api_client(request=request)
        api_result = api_client.fetch_current_user(
            cookie_header=request.headers.get("cookie")
        )
        if api_result.status_code == 200 and api_result.user is not None:
            return templates.TemplateResponse(
                request,
                "partials/user_badge.html",
                context={"request": request, "current_user": api_result.user},
            )
        if api_result.status_code == 401:
            return HTMLResponse(
                status_code=401,
                content='<span class="user-badge user-badge--guest">guest</span>',
            )
        return HTMLResponse(
            status_code=502,
            content='<span class="user-badge user-badge--error">api unavailable</span>',
        )



def _resolve_current_user_api_client(*, request: Request) -> CurrentUserApiClient:
    """
    Resolve server-side current-user API adapter from FastAPI application state.

    Args:
        request: HTTP request object.
    Returns:
        CurrentUserApiClient: Bound API adapter instance.
    Assumptions:
        `create_app` sets `app.state.current_user_api_client` at startup.
    Raises:
        ValueError: If API client is not configured in app state.
    Side Effects:
        None.
    """
    api_client = getattr(request.app.state, "current_user_api_client", None)
    if api_client is None:
        raise ValueError("current_user_api_client is not configured in application state")
    return api_client



def _render_protected_page(
    *,
    request: Request,
    templates: Jinja2Templates,
    page_path: str,
    page_title: str,
    page_description: str,
) -> Response:
    """
    Enforce login gate and render protected page skeleton with user badge.

    Args:
        request: HTTP request object.
        templates: Jinja2 template renderer.
        page_path: Canonical route path for navigation state.
        page_title: Page title shown in skeleton content.
        page_description: Placeholder page description.
    Returns:
        Response: Login redirect or SSR protected page response.
    Assumptions:
        Auth state comes exclusively from `/api/auth/current-user` response status.
    Raises:
        None.
    Side Effects:
        Performs server-side request to identity API.
    """
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
        page_title=page_title,
        current_user=current_user,
        error_message=error_message,
    )
    context["page_description"] = page_description
    return templates.TemplateResponse(
        request,
        "protected_page.html",
        context=context,
        status_code=status_code,
    )



def _build_login_redirect_response(*, current_path: str) -> RedirectResponse:
    """
    Build deterministic redirect response to `/login` with guarded `next` target.

    Args:
        current_path: Requested protected path.
    Returns:
        RedirectResponse: Redirect to login route with encoded safe next parameter.
    Assumptions:
        `current_path` originates from trusted server-side request routing.
    Raises:
        None.
    Side Effects:
        None.
    """
    safe_next_path = sanitize_next_path(raw_next=current_path)
    query = urlencode({"next": safe_next_path})
    return RedirectResponse(url=f"/login?{query}")



def _build_api_error_message(*, api_result: CurrentUserApiResult) -> str | None:
    """
    Convert API lookup failure into human-readable SSR error banner message.

    Args:
        api_result: Current-user lookup result from internal API client.
    Returns:
        str | None: Error message suitable for UI banner or None when no error.
    Assumptions:
        Unauthorized responses are handled by redirect and never shown as error banners.
    Raises:
        None.
    Side Effects:
        None.
    """
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
    page_title: str,
    current_user: WebCurrentUser | None,
    error_message: str | None,
) -> dict[str, Any]:
    """
    Build base template context shared by all SSR page handlers.

    Args:
        request: HTTP request object.
        page_path: Current route path for navigation highlighting.
        page_title: Human-readable page title.
        current_user: Optional authenticated user summary for badge rendering.
        error_message: Optional API/web error banner text.
    Returns:
        dict[str, Any]: Template context payload.
    Assumptions:
        Template names consume the same core keys across all pages.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "request": request,
        "page_path": page_path,
        "page_title": page_title,
        "current_user": current_user,
        "error_message": error_message,
    }
