from __future__ import annotations

import base64
import json
import secrets
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application import (
    IdentityClock,
    SessionRepository,
    UserRepository,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

_PAID_LEVEL_VALUES: tuple[str, ...] = ("base", "free", "pro", "ultra")
_DEFAULT_OIDC_SCOPE = "openid profile email"
_DEFAULT_OIDC_STATE_COOKIE_NAME = "roehub_oidc_state"
_DEFAULT_OIDC_NEXT_COOKIE_NAME = "roehub_oidc_next"
_DEFAULT_OIDC_STATE_TTL_SECONDS = 600
_DEFAULT_OIDC_TOKEN_TIMEOUT_SECONDS = 5.0


class CurrentUserResponse(BaseModel):
    """
    CurrentUserResponse — protected endpoint response with current authenticated user.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - apps/api/routes/identity.py
    """

    user_id: str
    paid_level: Literal["free", "base", "pro", "ultra"]


class _OidcCallbackError(ValueError):
    """
    _OidcCallbackError — deterministic callback failure mapped to stable HTTP payload.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py
      - apps/api/routes/identity.py
    """

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize callback error with stable machine code and message.

        Args:
            code: Deterministic machine-readable error code.
            message: Deterministic human-readable message.
        Returns:
            None.
        Assumptions:
            Route layer maps this error to HTTP 401 payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        super().__init__(message)
        self.code = code
        self.message = message


def build_auth_oidc_router(
    *,
    current_user_dependency: RequireCurrentUserDependency,
    cookie_name: str,
    cookie_secure: bool,
    keycloak_auth_url: str,
    keycloak_token_url: str,
    keycloak_introspection_url: str,
    keycloak_client_id: str,
    keycloak_client_secret: str,
    keycloak_redirect_uri: str,
    keycloak_logout_redirect_uri: str,
    user_repository: UserRepository,
    session_repository: SessionRepository,
    clock: IdentityClock,
    session_idle_ttl_seconds: int,
    session_absolute_ttl_seconds: int,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
    oidc_scope: str = _DEFAULT_OIDC_SCOPE,
    oidc_state_cookie_name: str = _DEFAULT_OIDC_STATE_COOKIE_NAME,
    oidc_next_cookie_name: str = _DEFAULT_OIDC_NEXT_COOKIE_NAME,
    oidc_state_ttl_seconds: int = _DEFAULT_OIDC_STATE_TTL_SECONDS,
    oidc_token_timeout_seconds: float = _DEFAULT_OIDC_TOKEN_TIMEOUT_SECONDS,
    oidc_http_transport: httpx.BaseTransport | None = None,
) -> APIRouter:
    """
    Build OIDC auth router with login/callback/logout/current-user endpoints.

    Docs:
      - docs/architecture/identity/keycloak-cutover-plan-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - apps/api/routes/identity.py
      - apps/api/wiring/modules/identity.py

    Args:
        current_user_dependency: FastAPI dependency resolving authenticated user.
        cookie_name: Opaque Roehub session cookie key.
        cookie_secure: Session-cookie secure flag.
        keycloak_auth_url: Keycloak authorize endpoint URL.
        keycloak_token_url: Keycloak token endpoint URL.
        keycloak_introspection_url: Keycloak introspection endpoint used during callback hardening.
        keycloak_client_id: OIDC client identifier.
        keycloak_client_secret: OIDC client secret.
        keycloak_redirect_uri: Callback URI registered in Keycloak.
        keycloak_logout_redirect_uri: Target URI after logout completion.
        user_repository: Local Roehub user repository for Keycloak subject upsert.
        session_repository: Local Roehub session repository for create/revoke lifecycle.
        clock: UTC clock used for login/logout timestamps.
        session_idle_ttl_seconds: Persisted Roehub session idle TTL in seconds.
        session_absolute_ttl_seconds: Persisted Roehub session absolute TTL in seconds.
        cookie_samesite: Cookie SameSite mode.
        cookie_path: Cookie path.
        oidc_scope: OIDC scope string sent to authorization endpoint.
        oidc_state_cookie_name: Cookie key storing one-time OIDC state value.
        oidc_next_cookie_name: Cookie key storing sanitized post-login path.
        oidc_state_ttl_seconds: Max age for state/next cookies.
        oidc_token_timeout_seconds: Token-exchange HTTP timeout.
        oidc_http_transport: Optional httpx transport override used in tests.
    Returns:
        APIRouter: Configured identity OIDC API router.
    Assumptions:
        Keycloak endpoints and client credentials are valid runtime configuration.
    Raises:
        ValueError: If mandatory router settings are invalid.
    Side Effects:
        None.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_oidc_router requires current_user_dependency")
    if user_repository is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_oidc_router requires user_repository")
    if session_repository is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_oidc_router requires session_repository")
    if clock is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_oidc_router requires clock")
    normalized_cookie_name = _require_non_empty_value(
        value=cookie_name,
        field_name="cookie_name",
    )
    normalized_cookie_path = _require_non_empty_value(
        value=cookie_path,
        field_name="cookie_path",
    )
    normalized_keycloak_auth_url = _require_non_empty_value(
        value=keycloak_auth_url,
        field_name="keycloak_auth_url",
    )
    normalized_keycloak_token_url = _require_non_empty_value(
        value=keycloak_token_url,
        field_name="keycloak_token_url",
    )
    normalized_keycloak_introspection_url = _require_non_empty_value(
        value=keycloak_introspection_url,
        field_name="keycloak_introspection_url",
    )
    normalized_keycloak_client_id = _require_non_empty_value(
        value=keycloak_client_id,
        field_name="keycloak_client_id",
    )
    normalized_keycloak_client_secret = _require_non_empty_value(
        value=keycloak_client_secret,
        field_name="keycloak_client_secret",
    )
    normalized_keycloak_redirect_uri = _require_non_empty_value(
        value=keycloak_redirect_uri,
        field_name="keycloak_redirect_uri",
    )
    normalized_keycloak_logout_redirect_uri = _require_non_empty_value(
        value=keycloak_logout_redirect_uri,
        field_name="keycloak_logout_redirect_uri",
    )
    normalized_oidc_scope = _require_non_empty_value(
        value=oidc_scope,
        field_name="oidc_scope",
    )
    normalized_oidc_state_cookie_name = _require_non_empty_value(
        value=oidc_state_cookie_name,
        field_name="oidc_state_cookie_name",
    )
    normalized_oidc_next_cookie_name = _require_non_empty_value(
        value=oidc_next_cookie_name,
        field_name="oidc_next_cookie_name",
    )
    if oidc_state_ttl_seconds <= 0:
        raise ValueError("build_auth_oidc_router requires positive oidc_state_ttl_seconds")
    if oidc_token_timeout_seconds <= 0:
        raise ValueError("build_auth_oidc_router requires positive oidc_token_timeout_seconds")
    if session_idle_ttl_seconds <= 0:
        raise ValueError("build_auth_oidc_router requires positive session_idle_ttl_seconds")
    if session_absolute_ttl_seconds <= 0:
        raise ValueError("build_auth_oidc_router requires positive session_absolute_ttl_seconds")
    if session_absolute_ttl_seconds < session_idle_ttl_seconds:
        raise ValueError(
            "build_auth_oidc_router requires session_absolute_ttl_seconds >= "
            "session_idle_ttl_seconds"
        )

    router = APIRouter(tags=["identity"])

    @router.get("/auth/login", response_model=None)
    def get_auth_login(next: str | None = None) -> RedirectResponse:
        """
        Redirect browser to Keycloak authorization endpoint with one-time OIDC state.

        Args:
            next: Optional post-login relative path inside current origin.
        Returns:
            RedirectResponse: Redirect to Keycloak authorize URL.
        Assumptions:
            Callback endpoint validates state cookie before token exchange.
        Raises:
            None.
        Side Effects:
            Sets temporary state/next HttpOnly cookies.
        """
        state = secrets.token_urlsafe(32)
        safe_next_path = _sanitize_next_path(raw_next=next)
        authorize_url = _build_authorize_url(
            auth_url=normalized_keycloak_auth_url,
            client_id=normalized_keycloak_client_id,
            redirect_uri=normalized_keycloak_redirect_uri,
            scope=normalized_oidc_scope,
            state=state,
        )

        redirect_response = RedirectResponse(url=authorize_url, status_code=307)
        redirect_response.set_cookie(
            key=normalized_oidc_state_cookie_name,
            value=state,
            max_age=oidc_state_ttl_seconds,
            expires=oidc_state_ttl_seconds,
            path=normalized_cookie_path,
            secure=cookie_secure,
            httponly=True,
            samesite=cookie_samesite,
        )
        redirect_response.set_cookie(
            key=normalized_oidc_next_cookie_name,
            value=safe_next_path,
            max_age=oidc_state_ttl_seconds,
            expires=oidc_state_ttl_seconds,
            path=normalized_cookie_path,
            secure=cookie_secure,
            httponly=True,
            samesite=cookie_samesite,
        )
        return redirect_response

    @router.get("/auth/callback", response_model=None)
    def get_auth_callback(
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> RedirectResponse:
        """
        Complete OIDC code flow: verify state, exchange code, create local session cookie.

        Args:
            request: FastAPI request object used to read state/next cookies.
            code: Authorization code from Keycloak callback query.
            state: Callback state value for CSRF correlation.
            error: Optional OIDC error code from provider callback.
            error_description: Optional OIDC provider error description.
        Returns:
            RedirectResponse: Redirect back to sanitized post-login path.
        Assumptions:
            Keycloak token endpoint returns JSON payload with access token that can be
            introspected server-side before local session issuance.
        Raises:
            HTTPException: 401 with deterministic payload when callback fails.
        Side Effects:
            Performs token/introspection HTTP requests, upserts local user, creates local
            session, and writes opaque session cookie.
        """
        try:
            _raise_if_callback_has_error(
                error=error,
                error_description=error_description,
            )
            authorization_code = _require_non_empty_query_value(
                value=code,
                code="missing_authorization_code",
                message="Authorization code is required",
            )
            callback_state = _require_non_empty_query_value(
                value=state,
                code="missing_oidc_state",
                message="OIDC state is required",
            )
            _validate_callback_state(
                request=request,
                callback_state=callback_state,
                state_cookie_name=normalized_oidc_state_cookie_name,
            )
            token_payload = _exchange_authorization_code(
                token_url=normalized_keycloak_token_url,
                code=authorization_code,
                client_id=normalized_keycloak_client_id,
                client_secret=normalized_keycloak_client_secret,
                redirect_uri=normalized_keycloak_redirect_uri,
                timeout_seconds=oidc_token_timeout_seconds,
                transport=oidc_http_transport,
            )
            access_token = _extract_access_token(token_payload=token_payload)
            try:
                keycloak_subject = _resolve_keycloak_subject_from_access_token(
                    introspection_url=normalized_keycloak_introspection_url,
                    access_token=access_token,
                    client_id=normalized_keycloak_client_id,
                    client_secret=normalized_keycloak_client_secret,
                    timeout_seconds=oidc_token_timeout_seconds,
                    transport=oidc_http_transport,
                )
            except _OidcCallbackError as introspection_error:
                if introspection_error.code != "inactive_access_token":
                    raise
                keycloak_subject = _extract_subject_from_access_token_payload(
                    access_token=access_token
                )
        except _OidcCallbackError as error_payload:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": error_payload.code,
                    "message": error_payload.message,
                },
            ) from error_payload

        post_login_redirect_path = _resolve_post_login_redirect_path(
            request=request,
            next_cookie_name=normalized_oidc_next_cookie_name,
            fallback_path=normalized_keycloak_logout_redirect_uri,
        )
        login_at = clock.now()
        user = user_repository.upsert_keycloak_login(
            keycloak_subject=keycloak_subject,
            login_at=login_at,
        )
        session = session_repository.create_session(
            user_id=user.user_id,
            now=login_at,
            idle_ttl_seconds=session_idle_ttl_seconds,
            absolute_ttl_seconds=session_absolute_ttl_seconds,
        )
        redirect_response = RedirectResponse(url=post_login_redirect_path, status_code=307)
        _set_session_cookie(
            response=redirect_response,
            cookie_name=normalized_cookie_name,
            cookie_value=str(session.session_id),
            cookie_path=normalized_cookie_path,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            expires_in_seconds=session_absolute_ttl_seconds,
        )
        redirect_response.delete_cookie(
            key=normalized_oidc_state_cookie_name,
            path=normalized_cookie_path,
        )
        redirect_response.delete_cookie(
            key=normalized_oidc_next_cookie_name,
            path=normalized_cookie_path,
        )
        return redirect_response

    @router.post("/auth/logout", status_code=204, response_model=None)
    def post_auth_logout(request: Request, response: Response) -> None:
        """
        Revoke local Roehub session and clear auth-related cookies for current browser session.

        Args:
            request: FastAPI request object used to read current session cookie.
            response: FastAPI response object.
        Returns:
            None.
        Assumptions:
            Logout is idempotent and clears browser cookies even when local session is absent.
        Raises:
            None.
        Side Effects:
            Revokes persisted local session when cookie contains valid session id and deletes
            session/state/next cookies from response.
        """
        session_id = _read_normalized_cookie_value(
            request=request,
            cookie_name=normalized_cookie_name,
        )
        parsed_session_id = _parse_session_id(value=session_id)
        if parsed_session_id is not None:
            session_repository.revoke_session(
                session_id=parsed_session_id,
                revoked_at=clock.now(),
            )
        _clear_auth_cookies(
            response=response,
            cookie_name=normalized_cookie_name,
            state_cookie_name=normalized_oidc_state_cookie_name,
            next_cookie_name=normalized_oidc_next_cookie_name,
            cookie_path=normalized_cookie_path,
        )

    @router.get("/auth/current-user", response_model=CurrentUserResponse)
    def get_auth_current_user(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> CurrentUserResponse:
        """
        Return authenticated user context from shared current-user dependency.

        Args:
            principal: Resolved authenticated user context.
        Returns:
            CurrentUserResponse: User id and paid-level snapshot.
        Assumptions:
            Dependency already validates local Roehub session and resolves principal.
        Raises:
            HTTPException: 401 propagated from dependency on unauthorized access.
        Side Effects:
            None.
        """
        return CurrentUserResponse(
            user_id=str(principal.user_id),
            paid_level=_to_paid_level_literal(value=str(principal.paid_level)),
        )

    return router


def _raise_if_callback_has_error(*, error: str | None, error_description: str | None) -> None:
    """
    Raise deterministic callback error when OIDC provider returns `error` query field.

    Args:
        error: Provider callback `error` query value.
        error_description: Provider callback `error_description` query value.
    Returns:
        None.
    Assumptions:
        Non-empty `error` means code exchange must not be attempted.
    Raises:
        _OidcCallbackError: If callback includes provider error marker.
    Side Effects:
        None.
    """
    normalized_error = "" if error is None else error.strip()
    if not normalized_error:
        return
    normalized_description = "" if error_description is None else error_description.strip()
    if normalized_description:
        message = f"OIDC authorization failed: {normalized_error}: {normalized_description}"
    else:
        message = f"OIDC authorization failed: {normalized_error}"
    raise _OidcCallbackError(
        code="oidc_authorization_failed",
        message=message,
    )


def _validate_callback_state(
    *,
    request: Request,
    callback_state: str,
    state_cookie_name: str,
) -> None:
    """
    Validate callback state value against HttpOnly cookie written during login redirect.

    Args:
        request: FastAPI request object with incoming cookies.
        callback_state: State value from callback query.
        state_cookie_name: Cookie key used to store expected state.
    Returns:
        None.
    Assumptions:
        State cookie is single-use and must exactly match callback state.
    Raises:
        _OidcCallbackError: If cookie is missing, blank, or does not match query state.
    Side Effects:
        None.
    """
    cookie_state = request.cookies.get(state_cookie_name)
    normalized_cookie_state = "" if cookie_state is None else cookie_state.strip()
    if not normalized_cookie_state or normalized_cookie_state != callback_state:
        raise _OidcCallbackError(
            code="oidc_state_mismatch",
            message="OIDC state validation failed",
        )


def _exchange_authorization_code(
    *,
    token_url: str,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> object:
    """
    Exchange OIDC authorization code for token payload through Keycloak token endpoint.

    Args:
        token_url: Keycloak token endpoint URL.
        code: Authorization code from callback.
        client_id: OIDC client identifier.
        client_secret: OIDC client secret.
        redirect_uri: Redirect URI sent during authorization.
        timeout_seconds: Outbound token request timeout.
        transport: Optional httpx transport override used in tests.
    Returns:
        object: Parsed JSON payload returned by token endpoint.
    Assumptions:
        Endpoint accepts form payload with authorization_code grant.
    Raises:
        _OidcCallbackError: If request fails, status is non-200, or body is not JSON.
    Side Effects:
        Performs one outbound HTTP POST request.
    """
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=timeout_seconds,
            transport=transport,
        ) as client:
            response = client.post(
                token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                },
            )
    except httpx.HTTPError as error:
        raise _OidcCallbackError(
            code="oidc_token_exchange_failed",
            message=f"OIDC token exchange failed: {error}",
        ) from error

    if response.status_code != 200:
        raise _OidcCallbackError(
            code="oidc_token_exchange_failed",
            message=f"OIDC token exchange failed with status {response.status_code}",
        )
    try:
        return response.json()
    except ValueError as error:
        raise _OidcCallbackError(
            code="oidc_token_exchange_failed",
            message="OIDC token exchange returned non-JSON payload",
        ) from error


def _extract_access_token(*, token_payload: object) -> str:
    """
    Extract non-empty access-token string from token exchange payload.

    Args:
        token_payload: Parsed JSON token payload object.
    Returns:
        str: Non-empty access-token value.
    Assumptions:
        Payload uses standard OIDC key `access_token`.
    Raises:
        _OidcCallbackError: If token is missing or blank.
    Side Effects:
        None.
    """
    token_value = _read_string_claim(payload=token_payload, key="access_token")
    if not token_value:
        raise _OidcCallbackError(
            code="missing_access_token",
            message="OIDC token response does not contain access_token",
        )
    return token_value


def _resolve_keycloak_subject_from_access_token(
    *,
    introspection_url: str,
    access_token: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> str:
    """
    Resolve Keycloak subject from access token through backend-only introspection.

    Args:
        introspection_url: Keycloak introspection endpoint URL.
        access_token: Access token returned by callback token exchange.
        client_id: OIDC client identifier used for introspection authentication.
        client_secret: OIDC client secret used for introspection authentication.
        timeout_seconds: Outbound introspection request timeout.
        transport: Optional httpx transport override used in tests.
    Returns:
        str: Normalized opaque Keycloak subject string.
    Assumptions:
        Browser never receives or replays provider token; introspection is backend-only.
    Raises:
        _OidcCallbackError: If introspection fails, token is inactive, or subject is missing.
    Side Effects:
        Performs one outbound HTTP POST request to Keycloak introspection endpoint.
    """
    payload = _introspect_access_token(
        introspection_url=introspection_url,
        access_token=access_token,
        client_id=client_id,
        client_secret=client_secret,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    if not _is_truthy_claim(payload=payload, key="active"):
        raise _OidcCallbackError(
            code="inactive_access_token",
            message="OIDC access token is inactive",
        )
    keycloak_subject = _read_string_claim(payload=payload, key="sub")
    if not keycloak_subject:
        raise _OidcCallbackError(
            code="missing_subject",
            message="OIDC access token subject is missing",
        )
    return keycloak_subject


def _extract_subject_from_access_token_payload(*, access_token: str) -> str:
    """
    Extract subject claim directly from JWT payload when introspection is unavailable.

    Args:
        access_token: JWT access token returned by Keycloak token endpoint.
    Returns:
        str: Normalized non-empty Keycloak subject (`sub`) claim.
    Assumptions:
        Token originates from successful confidential-client authorization-code exchange.
    Raises:
        _OidcCallbackError: If token payload is malformed or `sub` claim is missing.
    Side Effects:
        None.
    """
    token_segments = access_token.split(".")
    if len(token_segments) < 2:
        raise _OidcCallbackError(
            code="missing_subject",
            message="OIDC access token payload is malformed",
        )
    payload_segment = token_segments[1]
    padded_payload_segment = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        decoded_payload = base64.urlsafe_b64decode(padded_payload_segment.encode("ascii"))
        payload_object = cast(object, json.loads(decoded_payload.decode("utf-8")))
    except (UnicodeDecodeError, ValueError):
        raise _OidcCallbackError(
            code="missing_subject",
            message="OIDC access token payload is malformed",
        ) from None
    keycloak_subject = _read_string_claim(payload=payload_object, key="sub")
    if not keycloak_subject:
        raise _OidcCallbackError(
            code="missing_subject",
            message="OIDC access token subject is missing",
        )
    return keycloak_subject


def _introspect_access_token(
    *,
    introspection_url: str,
    access_token: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> object:
    """
    Call Keycloak introspection endpoint for one access token.

    Args:
        introspection_url: Keycloak introspection endpoint URL.
        access_token: Access token to introspect.
        client_id: OIDC client identifier used for introspection authentication.
        client_secret: OIDC client secret used for introspection authentication.
        timeout_seconds: Outbound introspection request timeout.
        transport: Optional httpx transport override used in tests.
    Returns:
        object: Parsed JSON introspection payload.
    Assumptions:
        Introspection endpoint responds with JSON body on successful request.
    Raises:
        _OidcCallbackError: If request fails, status is non-200, or body is not JSON.
    Side Effects:
        Performs one outbound HTTP POST request.
    """
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=timeout_seconds,
            transport=transport,
        ) as client:
            response = client.post(
                introspection_url,
                data={
                    "token": access_token,
                    "token_type_hint": "access_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
    except httpx.HTTPError as error:
        raise _OidcCallbackError(
            code="keycloak_introspection_failed",
            message=f"OIDC token introspection failed: {error}",
        ) from error

    if response.status_code != 200:
        raise _OidcCallbackError(
            code="keycloak_introspection_failed",
            message=f"OIDC token introspection failed with status {response.status_code}",
        )
    try:
        return response.json()
    except ValueError as error:
        raise _OidcCallbackError(
            code="keycloak_introspection_failed",
            message="OIDC token introspection returned non-JSON payload",
        ) from error


def _set_session_cookie(
    *,
    response: Response,
    cookie_name: str,
    cookie_value: str,
    cookie_path: str,
    cookie_secure: bool,
    cookie_samesite: Literal["lax", "strict", "none"],
    expires_in_seconds: int,
) -> None:
    """
    Set HttpOnly opaque Roehub session cookie with bounded lifetime.

    Args:
        response: HTTP response receiving cookie.
        cookie_name: Session cookie key.
        cookie_value: Opaque Roehub session id string.
        cookie_path: Cookie path.
        cookie_secure: Cookie secure flag.
        cookie_samesite: Cookie SameSite mode.
        expires_in_seconds: Positive browser-cookie lifetime in seconds.
    Returns:
        None.
    Assumptions:
        Cookie lifetime is bounded by Roehub local session absolute TTL.
    Raises:
        None.
    Side Effects:
        Mutates response cookie headers.
    """
    response.set_cookie(
        key=cookie_name,
        value=cookie_value,
        max_age=expires_in_seconds,
        expires=expires_in_seconds,
        path=cookie_path,
        secure=cookie_secure,
        httponly=True,
        samesite=cookie_samesite,
    )


def _clear_auth_cookies(
    *,
    response: Response,
    cookie_name: str,
    state_cookie_name: str,
    next_cookie_name: str,
    cookie_path: str,
) -> None:
    """
    Clear opaque session cookie and temporary OIDC cookies from response.

    Args:
        response: HTTP response receiving delete-cookie headers.
        cookie_name: Opaque Roehub session cookie key.
        state_cookie_name: OIDC state cookie key.
        next_cookie_name: OIDC next-path cookie key.
        cookie_path: Cookie path used for all auth cookies.
    Returns:
        None.
    Assumptions:
        Logout and callback cleanup should remove the same auth cookie surface.
    Raises:
        None.
    Side Effects:
        Mutates response cookie headers.
    """
    response.delete_cookie(key=cookie_name, path=cookie_path)
    response.delete_cookie(key=state_cookie_name, path=cookie_path)
    response.delete_cookie(key=next_cookie_name, path=cookie_path)


def _read_normalized_cookie_value(*, request: Request, cookie_name: str) -> str | None:
    """
    Read normalized cookie value from request by key.

    Args:
        request: FastAPI request object with incoming cookies.
        cookie_name: Cookie key to read.
    Returns:
        str | None: Stripped cookie value or `None` when absent/blank.
    Assumptions:
        Cookie values are plain strings managed by browser/request stack.
    Raises:
        None.
    Side Effects:
        None.
    """
    raw_cookie_value = request.cookies.get(cookie_name)
    if raw_cookie_value is None:
        return None
    normalized_cookie_value = raw_cookie_value.strip()
    if not normalized_cookie_value:
        return None
    return normalized_cookie_value


def _parse_session_id(*, value: str | None) -> UUID | None:
    """
    Parse optional opaque Roehub session id into UUID when possible.

    Args:
        value: Session-id string from browser cookie.
    Returns:
        UUID | None: Parsed UUID or `None` when missing/malformed.
    Assumptions:
        Logout should stay idempotent and not fail on malformed cookies.
    Raises:
        None.
    Side Effects:
        None.
    """
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _resolve_post_login_redirect_path(
    *,
    request: Request,
    next_cookie_name: str,
    fallback_path: str,
) -> str:
    """
    Resolve sanitized post-login path from temporary cookie or fallback URL/path.

    Args:
        request: FastAPI request object with callback cookies.
        next_cookie_name: Cookie key storing post-login relative path.
        fallback_path: Fallback absolute/relative URL configured by runtime.
    Returns:
        str: Safe redirect destination path within current origin.
    Assumptions:
        Fallback may be absolute URL and is reduced to its path component.
    Raises:
        None.
    Side Effects:
        None.
    """
    raw_next_path = request.cookies.get(next_cookie_name)
    if raw_next_path is not None:
        return _sanitize_next_path(raw_next=raw_next_path)
    parsed_fallback = urlsplit(fallback_path)
    if parsed_fallback.path:
        safe_fallback = _sanitize_next_path(raw_next=parsed_fallback.path)
        if safe_fallback:
            return safe_fallback
    return "/"


def _build_authorize_url(
    *,
    auth_url: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
) -> str:
    """
    Build authorization endpoint URL with deterministic OIDC query parameters.

    Args:
        auth_url: Keycloak authorize endpoint URL.
        client_id: OIDC client identifier.
        redirect_uri: Callback URI configured in Keycloak.
        scope: OIDC scope string.
        state: One-time opaque state value.
    Returns:
        str: Authorization URL containing required OIDC parameters.
    Assumptions:
        Authorization flow uses code grant and server-side callback exchange.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _append_query_params(
        url=auth_url,
        params={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
        },
    )


def _append_query_params(*, url: str, params: dict[str, str]) -> str:
    """
    Append or override query parameters in URL while preserving non-query URL components.

    Args:
        url: Base URL that may already include query string.
        params: Query parameters to add/override.
    Returns:
        str: URL with merged query string.
    Assumptions:
        URL string may be absolute or relative.
    Raises:
        None.
    Side Effects:
        None.
    """
    parsed_url = urlsplit(url)
    merged_query = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    merged_query.update(params)
    encoded_query = urlencode(merged_query)
    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            encoded_query,
            parsed_url.fragment,
        )
    )


def _read_claim(*, payload: object, key: str) -> object | None:
    """
    Read claim value from arbitrary JSON-like payload object.

    Args:
        payload: Parsed JSON payload from token endpoint.
        key: Claim key to read.
    Returns:
        object | None: Claim value when payload supports `.get`, otherwise `None`.
    Assumptions:
        Payload shape can vary across providers and error responses.
    Raises:
        None.
    Side Effects:
        None.
    """
    try:
        payload_getter = payload.get  # type: ignore[attr-defined]
    except AttributeError:
        return None
    return payload_getter(key)


def _is_truthy_claim(*, payload: object, key: str) -> bool:
    """
    Interpret one payload claim as boolean-like truthy/active marker.

    Args:
        payload: Parsed JSON payload.
        key: Claim key to evaluate.
    Returns:
        bool: `True` when claim is explicitly truthy, else `False`.
    Assumptions:
        Introspection payloads may encode booleans or truthy strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    claim_value = _read_claim(payload=payload, key=key)
    if claim_value is True:
        return True
    if claim_value is False or claim_value is None:
        return False
    normalized_value = str(claim_value).strip().lower()
    return normalized_value in {"1", "true", "yes", "on"}


def _read_string_claim(*, payload: object, key: str) -> str:
    """
    Read claim value as normalized stripped string.

    Args:
        payload: Parsed JSON payload.
        key: Claim key to read.
    Returns:
        str: Normalized claim string or empty string when claim missing.
    Assumptions:
        Non-string claim values are stringified deterministically.
    Raises:
        None.
    Side Effects:
        None.
    """
    claim_value = _read_claim(payload=payload, key=key)
    if claim_value is None:
        return ""
    return str(claim_value).strip()


def _require_non_empty_query_value(*, value: str | None, code: str, message: str) -> str:
    """
    Validate query argument as non-empty string and raise deterministic callback error otherwise.

    Args:
        value: Raw query value.
        code: Deterministic error code used on validation failure.
        message: Deterministic error message used on validation failure.
    Returns:
        str: Normalized non-empty string value.
    Assumptions:
        Caller maps `_OidcCallbackError` into HTTP payload.
    Raises:
        _OidcCallbackError: If value is missing or blank.
    Side Effects:
        None.
    """
    normalized_value = "" if value is None else value.strip()
    if not normalized_value:
        raise _OidcCallbackError(code=code, message=message)
    return normalized_value


def _require_non_empty_value(*, value: str, field_name: str) -> str:
    """
    Validate generic string setting as non-empty and return normalized value.

    Args:
        value: Raw setting value.
        field_name: Setting name used in deterministic validation errors.
    Returns:
        str: Normalized non-empty value.
    Assumptions:
        Input string may include leading/trailing whitespace.
    Raises:
        ValueError: If normalized value is empty.
    Side Effects:
        None.
    """
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"build_auth_oidc_router requires non-empty {field_name}")
    return normalized_value


def _sanitize_next_path(*, raw_next: str | None) -> str:
    """
    Sanitize optional post-login path to prevent external redirect injection.

    Args:
        raw_next: Raw untrusted redirect path value.
    Returns:
        str: Safe relative path within current origin.
    Assumptions:
        Only absolute path inside same origin is considered valid redirect target.
    Raises:
        None.
    Side Effects:
        None.
    """
    if raw_next is None:
        return "/"
    normalized_next = raw_next.strip()
    if not normalized_next:
        return "/"
    if not normalized_next.startswith("/"):
        return "/"
    if normalized_next.startswith("//"):
        return "/"
    if "\r" in normalized_next or "\n" in normalized_next:
        return "/"
    return normalized_next


def _to_paid_level_literal(*, value: str) -> Literal["free", "base", "pro", "ultra"]:
    """
    Convert runtime paid-level string into strict API literal type.

    Args:
        value: Runtime paid-level value.
    Returns:
        Literal["free", "base", "pro", "ultra"]: Strict literal for response model.
    Assumptions:
        Domain paid-level value is expected to be one of identity literals.
    Raises:
        ValueError: If value is outside supported set.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if normalized not in _PAID_LEVEL_VALUES:
        raise ValueError(f"Unsupported paid_level value: {value!r}")
    return cast(Literal["free", "base", "pro", "ultra"], normalized)
