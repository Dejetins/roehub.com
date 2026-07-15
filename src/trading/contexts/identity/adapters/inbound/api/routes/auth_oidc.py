from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict
from starlette.responses import Response

from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports import (
    CurrentUserPrincipal,
    IdentityClock,
    SessionRepository,
)
from trading.contexts.identity.application.use_cases import (
    OidcAuthenticationError,
    OidcAuthenticationService,
)

_ATTEMPT_COOKIE = "roehub_oidc_attempt"
_CSRF_COOKIE = "roehub_csrf"
_RECENT_AUTH_WINDOW = timedelta(minutes=5)


class OidcStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Literal[True] = True
    provider_id: str
    display_name: str
    local_fallback_available: Literal[True] = True


def build_auth_oidc_router(
    *,
    service: OidcAuthenticationService,
    current_user_dependency: RequireCurrentUserDependency,
    session_repository: SessionRepository,
    clock: IdentityClock,
    cookie_name: str,
    cookie_secure: bool,
    session_absolute_ttl_seconds: int,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
) -> APIRouter:
    """Build provider-neutral OIDC login, verified linking and callback routes."""

    router = APIRouter(prefix="/auth/oidc", tags=["identity-oidc"])

    @router.get("/status", response_model=OidcStatusResponse)
    def get_status() -> OidcStatusResponse:
        return OidcStatusResponse(
            provider_id=service.provider_id,
            display_name=service.provider_display_name,
        )

    @router.get("/login")
    def get_login(next: str = Query(default="/", max_length=1024)) -> Response:
        try:
            start = service.begin_login(next_path=next)
        except OidcAuthenticationError as error:
            return _error_response(error=error)
        response = RedirectResponse(start.authorization_url, status_code=303)
        _set_attempt_cookie(response=response, attempt_id=start.attempt_id, secure=cookie_secure)
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.get("/link")
    def get_link(
        next: str = Query(default="/account", max_length=1024),
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> Response:
        if (
            principal.session_created_at is None
            or principal.session_created_at + _RECENT_AUTH_WINDOW < clock.now()
        ):
            return _public_error(status_code=403, code="recent_auth_required")
        try:
            start = service.begin_link(user_id=principal.user_id, next_path=next)
        except OidcAuthenticationError as error:
            return _error_response(error=error)
        response = RedirectResponse(start.authorization_url, status_code=303)
        _set_attempt_cookie(response=response, attempt_id=start.attempt_id, secure=cookie_secure)
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.get("/callback")
    def get_callback(
        request: Request,
        code: str | None = Query(default=None, min_length=1, max_length=4096),
        state: str | None = Query(default=None, min_length=1, max_length=512),
        error: str | None = Query(default=None, max_length=128),
    ) -> Response:
        del error
        attempt_id = _attempt_id(request.cookies.get(_ATTEMPT_COOKIE))
        if attempt_id is None:
            return _public_error(status_code=400, code="oidc_attempt_invalid")
        if code is None or state is None:
            service.cancel(attempt_id=attempt_id)
            response = _public_error(status_code=401, code="oidc_authentication_failed")
            _clear_attempt_cookie(response=response, secure=cookie_secure)
            return response
        callback_user_id = _active_session_user_id(
            request=request,
            repository=session_repository,
            clock=clock,
            cookie_name=cookie_name,
        )
        try:
            result = service.complete(
                attempt_id=attempt_id,
                state=state,
                code=code,
                callback_user_id=callback_user_id,
            )
        except OidcAuthenticationError as auth_error:
            response = _error_response(error=auth_error)
            _clear_attempt_cookie(response=response, secure=cookie_secure)
            return response
        response = RedirectResponse(result.next_path, status_code=303)
        _clear_attempt_cookie(response=response, secure=cookie_secure)
        if result.session is not None:
            response.set_cookie(
                key=cookie_name,
                value=str(result.session.session_id),
                max_age=session_absolute_ttl_seconds,
                expires=session_absolute_ttl_seconds,
                httponly=True,
                secure=cookie_secure,
                samesite=cookie_samesite,
                path=cookie_path,
            )
            response.set_cookie(
                key=_CSRF_COOKIE,
                value=secrets.token_urlsafe(24),
                max_age=session_absolute_ttl_seconds,
                expires=session_absolute_ttl_seconds,
                httponly=False,
                secure=cookie_secure,
                samesite=cookie_samesite,
                path=cookie_path,
            )
        response.headers["Cache-Control"] = "no-store"
        return response

    return router


def _active_session_user_id(
    *,
    request: Request,
    repository: SessionRepository,
    clock: IdentityClock,
    cookie_name: str,
):
    raw = request.cookies.get(cookie_name)
    if raw is None:
        return None
    try:
        session_id = UUID(raw)
    except ValueError:
        return None
    session = repository.find_by_session_id(session_id=session_id)
    if session is None or not session.is_active_at(at=clock.now()):
        return None
    return session.user_id


def _attempt_id(raw: str | None) -> UUID | None:
    if raw is None:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _set_attempt_cookie(
    *, response: RedirectResponse, attempt_id: UUID, secure: bool
) -> None:
    response.set_cookie(
        key=_ATTEMPT_COOKIE,
        value=str(attempt_id),
        max_age=600,
        expires=600,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _clear_attempt_cookie(*, response: Response, secure: bool) -> None:
    response.delete_cookie(
        key=_ATTEMPT_COOKIE,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )


def _error_response(*, error: OidcAuthenticationError) -> Response:
    if error.provider_unavailable:
        return _public_error(status_code=503, code="oidc_provider_unavailable")
    if error.code in {"oidc_identity_conflict"}:
        return _public_error(status_code=409, code=error.code)
    if error.code in {
        "oidc_invitation_required",
        "oidc_verified_email_required",
        "oidc_link_session_required",
    }:
        return _public_error(status_code=403, code=error.code)
    return _public_error(status_code=401, code="oidc_authentication_failed")


def _public_error(*, status_code: int, code: str) -> Response:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": "Authentication request could not complete."}},
        headers={"Cache-Control": "no-store"},
    )
