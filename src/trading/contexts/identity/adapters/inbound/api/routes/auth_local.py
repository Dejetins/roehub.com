from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports import (
    CurrentUserPrincipal,
    IdentityClock,
    SessionRepository,
)
from trading.contexts.identity.application.use_cases import (
    LocalAuthError,
    LocalAuthOptions,
    LocalAuthResult,
    LocalAuthService,
)

_CSRF_COOKIE_NAME = "roehub_csrf"
_RECENT_AUTH_WINDOW = timedelta(minutes=5)


class LocalAuthStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap_required: bool
    passkey_available: bool
    password_available: bool
    registration_open: bool


class LocalAuthOptionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    challenge_id: UUID
    public_key: dict[str, Any] = Field(alias="publicKey")


class BootstrapOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: str = Field(min_length=16, max_length=256)
    username: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=2, max_length=120)
    installation_name: str = Field(min_length=2, max_length=120)
    organization_slug: str = Field(min_length=3, max_length=64)
    organization_name: str = Field(min_length=2, max_length=120)
    password: str | None = Field(default=None, max_length=1024)


class CredentialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenge_id: UUID
    credential: dict[str, Any]


class PasswordLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class RecoveryLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    recovery_code: str = Field(min_length=1, max_length=128)


class AuthenticatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authenticated: Literal[True] = True


class BootstrapCompleteResponse(AuthenticatedResponse):
    recovery_codes: tuple[str, ...]


def build_auth_local_router(
    *,
    service: LocalAuthService,
    current_user_dependency: RequireCurrentUserDependency,
    session_repository: SessionRepository,
    clock: IdentityClock,
    cookie_name: str,
    cookie_secure: bool,
    session_absolute_ttl_seconds: int,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
) -> APIRouter:
    """Build same-origin passkey-first local-auth routes."""

    router = APIRouter(prefix="/auth/local", tags=["identity-local-auth"])

    @router.get("/status", response_model=LocalAuthStatusResponse)
    def get_status(response: Response) -> LocalAuthStatusResponse:
        response.headers["Cache-Control"] = "no-store"
        return LocalAuthStatusResponse.model_validate(service.status(), from_attributes=True)

    @router.post("/bootstrap/options", response_model=LocalAuthOptionsResponse)
    def post_bootstrap_options(
        payload: BootstrapOptionsRequest, request: Request, response: Response
    ) -> LocalAuthOptionsResponse:
        _enforce_same_origin(request=request)
        response.headers["Cache-Control"] = "no-store"
        try:
            options = service.begin_bootstrap(**payload.model_dump())
        except LocalAuthError as error:
            raise _public_error(error) from error
        return _options_response(options)

    @router.post("/bootstrap/complete", response_model=BootstrapCompleteResponse)
    def post_bootstrap_complete(
        payload: CredentialRequest, request: Request, response: Response
    ) -> BootstrapCompleteResponse:
        _enforce_same_origin(request=request)
        try:
            result = service.complete_bootstrap(**payload.model_dump())
        except LocalAuthError as error:
            raise _public_error(error) from error
        _set_auth_cookies(
            response=response,
            result=result,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
            session_absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        return BootstrapCompleteResponse(recovery_codes=result.recovery_codes)

    @router.post("/passkey/options", response_model=LocalAuthOptionsResponse)
    def post_passkey_options(request: Request, response: Response) -> LocalAuthOptionsResponse:
        _enforce_same_origin(request=request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return _options_response(service.begin_passkey_login())
        except LocalAuthError as error:
            raise _public_error(error) from error

    @router.post("/passkey/complete", response_model=AuthenticatedResponse)
    def post_passkey_complete(
        payload: CredentialRequest, request: Request, response: Response
    ) -> AuthenticatedResponse:
        _enforce_same_origin(request=request)
        try:
            result = service.complete_passkey_login(**payload.model_dump())
        except LocalAuthError as error:
            raise _public_error(error) from error
        _set_auth_cookies(
            response=response,
            result=result,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
            session_absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        return AuthenticatedResponse()

    @router.post("/password", response_model=AuthenticatedResponse)
    def post_password(
        payload: PasswordLoginRequest, request: Request, response: Response
    ) -> AuthenticatedResponse:
        _enforce_same_origin(request=request)
        try:
            result = service.password_login(**payload.model_dump())
        except LocalAuthError as error:
            raise _public_error(error) from error
        _set_auth_cookies(
            response=response,
            result=result,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
            session_absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        return AuthenticatedResponse()

    @router.post("/recovery", response_model=AuthenticatedResponse)
    def post_recovery(
        payload: RecoveryLoginRequest, request: Request, response: Response
    ) -> AuthenticatedResponse:
        _enforce_same_origin(request=request)
        try:
            result = service.recovery_login(**payload.model_dump())
        except LocalAuthError as error:
            raise _public_error(error) from error
        _set_auth_cookies(
            response=response,
            result=result,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
            session_absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        return AuthenticatedResponse()

    @router.post("/passkeys/options", response_model=LocalAuthOptionsResponse)
    def post_passkeys_options(
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> LocalAuthOptionsResponse:
        _enforce_authenticated_mutation(request=request)
        _enforce_recent_principal(principal=principal, now=clock.now())
        response.headers["Cache-Control"] = "no-store"
        try:
            return _options_response(service.begin_passkey_registration(user_id=principal.user_id))
        except LocalAuthError as error:
            raise _public_error(error) from error

    @router.post("/passkeys/complete", status_code=204, response_model=None)
    def post_passkeys_complete(
        payload: CredentialRequest,
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> None:
        _enforce_authenticated_mutation(request=request)
        _enforce_recent_principal(principal=principal, now=clock.now())
        try:
            service.complete_passkey_registration(
                challenge_id=payload.challenge_id,
                user_id=principal.user_id,
                credential=payload.credential,
            )
        except LocalAuthError as error:
            raise _public_error(error) from error
        response.headers["Cache-Control"] = "no-store"

    @router.post("/recent-auth/options", response_model=LocalAuthOptionsResponse)
    def post_recent_auth_options(
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> LocalAuthOptionsResponse:
        _enforce_authenticated_mutation(request=request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return _options_response(service.begin_recent_auth(user_id=principal.user_id))
        except LocalAuthError as error:
            raise _public_error(error) from error

    @router.post("/recent-auth/complete", response_model=AuthenticatedResponse)
    def post_recent_auth_complete(
        payload: CredentialRequest,
        request: Request,
        response: Response,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> AuthenticatedResponse:
        _enforce_authenticated_mutation(request=request)
        current_session_id = _session_id(request=request, cookie_name=cookie_name)
        try:
            result = service.complete_recent_auth(
                challenge_id=payload.challenge_id,
                user_id=principal.user_id,
                credential=payload.credential,
                session_id_to_rotate=current_session_id,
            )
        except LocalAuthError as error:
            raise _public_error(error) from error
        _set_auth_cookies(
            response=response,
            result=result,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
            session_absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        return AuthenticatedResponse()

    @router.post("/logout", status_code=204, response_model=None)
    def post_logout(request: Request, response: Response) -> None:
        _enforce_authenticated_mutation(request=request)
        try:
            session_id = _session_id(request=request, cookie_name=cookie_name)
        except HTTPException:
            session_id = None
        if session_id is not None:
            session_repository.revoke_session(session_id=session_id, revoked_at=clock.now())
        response.delete_cookie(key=cookie_name, path=cookie_path)
        response.delete_cookie(key=_CSRF_COOKIE_NAME, path=cookie_path)
        response.headers["Cache-Control"] = "no-store"

    return router


def _options_response(value: LocalAuthOptions) -> LocalAuthOptionsResponse:
    return LocalAuthOptionsResponse(
        challenge_id=value.challenge_id,
        publicKey=dict(value.public_key),
    )


def _set_auth_cookies(
    *,
    response: Response,
    result: LocalAuthResult,
    cookie_name: str,
    cookie_secure: bool,
    cookie_samesite: Literal["lax", "strict", "none"],
    cookie_path: str,
    session_absolute_ttl_seconds: int,
) -> None:
    response.set_cookie(
        key=cookie_name,
        value=str(result.session.session_id),
        max_age=session_absolute_ttl_seconds,
        expires=session_absolute_ttl_seconds,
        path=cookie_path,
        secure=cookie_secure,
        httponly=True,
        samesite=cookie_samesite,
    )
    response.set_cookie(
        key=_CSRF_COOKIE_NAME,
        value=secrets.token_urlsafe(32),
        max_age=session_absolute_ttl_seconds,
        expires=session_absolute_ttl_seconds,
        path=cookie_path,
        secure=cookie_secure,
        httponly=False,
        samesite=cookie_samesite,
    )
    response.headers["Cache-Control"] = "no-store"


def _enforce_same_origin(*, request: Request) -> None:
    rejection_reason = same_origin_rejection_reason(
        request=request, fail_closed_without_origin=True
    )
    if rejection_reason is not None:
        raise HTTPException(
            status_code=403,
            detail={"error": "csrf_required", "message": "Request was rejected."},
        )


def _enforce_authenticated_mutation(*, request: Request) -> None:
    _enforce_same_origin(request=request)
    cookie_value = request.cookies.get(_CSRF_COOKIE_NAME)
    header_value = request.headers.get("x-csrf-token")
    if (
        cookie_value is None
        or header_value is None
        or not secrets.compare_digest(cookie_value, header_value)
    ):
        raise HTTPException(
            status_code=403,
            detail={"error": "csrf_required", "message": "Request was rejected."},
        )


def _enforce_recent_principal(*, principal: CurrentUserPrincipal, now: Any) -> None:
    if (
        principal.session_created_at is None
        or principal.session_created_at + _RECENT_AUTH_WINDOW < now
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "recent_auth_required",
                "message": "Fresh passkey verification is required.",
            },
        )


def _session_id(*, request: Request, cookie_name: str) -> UUID:
    try:
        return UUID(request.cookies.get(cookie_name, ""))
    except ValueError as error:
        raise HTTPException(status_code=401, detail={"error": "authentication_failed"}) from error


def _public_error(error: LocalAuthError) -> HTTPException:
    if error.code in {
        "bootstrap_unavailable",
        "invalid_bootstrap_input",
        "weak_password",
    }:
        return HTTPException(
            status_code=400,
            detail={"error": error.code, "message": "Request could not be completed."},
        )
    return HTTPException(
        status_code=401,
        detail={"error": "authentication_failed", "message": "Authentication failed."},
    )
