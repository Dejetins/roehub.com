from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from trading.contexts.identity.adapters.inbound.api.deps.current_user import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.application.ports.telegram_auth_payload_validator import (
    TelegramAuthValidationError,
)
from trading.contexts.identity.application.use_cases.telegram_login import TelegramLoginUseCase


class TelegramLoginRequest(BaseModel):
    """
    TelegramLoginRequest — API payload for Telegram Login Widget Variant A.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/telegram/
        telegram_login_widget_payload_validator.py
      - apps/api/routes/identity.py
    """

    id: int
    auth_date: int
    hash: str
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None
    photo_url: str | None = None


class TelegramLoginResponse(BaseModel):
    """
    TelegramLoginResponse — API response for successful Telegram login.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/shared_kernel/primitives/user_id.py
      - src/trading/shared_kernel/primitives/paid_level.py
    """

    user_id: str
    paid_level: Literal["free", "base", "pro", "ultra"]


class CurrentUserResponse(BaseModel):
    """
    CurrentUserResponse — protected endpoint response with current authenticated user.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/current_user.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - apps/api/routes/identity.py
    """

    user_id: str
    paid_level: Literal["free", "base", "pro", "ultra"]


_PAID_LEVEL_VALUES: tuple[str, ...] = ("base", "free", "pro", "ultra")


def build_auth_telegram_router(
    *,
    telegram_login: TelegramLoginUseCase,
    current_user_dependency: RequireCurrentUserDependency,
    cookie_name: str,
    cookie_secure: bool,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
) -> APIRouter:
    """
    Build identity router with Telegram login, logout, and current-user endpoints.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
      - apps/api/main/app.py

    Args:
        telegram_login: Telegram login use-case.
        current_user_dependency: FastAPI dependency resolving authenticated user.
        cookie_name: JWT cookie key.
        cookie_secure: Cookie secure flag.
        cookie_samesite: Cookie SameSite mode.
        cookie_path: Cookie path.
    Returns:
        APIRouter: Configured identity API router.
    Assumptions:
        Login use-case returns JWT token and expiration metadata.
    Raises:
        ValueError: If mandatory router settings are invalid.
    Side Effects:
        None.
    """
    if telegram_login is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_telegram_router requires telegram_login")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_auth_telegram_router requires current_user_dependency")
    normalized_cookie_name = cookie_name.strip()
    normalized_cookie_path = cookie_path.strip()
    if not normalized_cookie_name:
        raise ValueError("build_auth_telegram_router requires non-empty cookie_name")
    if not normalized_cookie_path:
        raise ValueError("build_auth_telegram_router requires non-empty cookie_path")

    router = APIRouter(tags=["identity"])

    @router.post("/auth/telegram/login", response_model=TelegramLoginResponse)
    def post_auth_telegram_login(
        request: TelegramLoginRequest,
        response: Response,
    ) -> TelegramLoginResponse:
        """
        Validate Telegram payload, upsert user, issue JWT, and set HttpOnly cookie.

        Args:
            request: Telegram login widget payload.
            response: FastAPI response object for setting cookie.
        Returns:
            TelegramLoginResponse: Stable user identity payload.
        Assumptions:
            Request payload originates from Telegram Login Widget Variant A.
        Raises:
            HTTPException: 401 for invalid Telegram payload, 422 for malformed request conversion.
        Side Effects:
            Sets HttpOnly authentication cookie on success.
        """
        payload = _build_telegram_payload(request=request)
        try:
            result = telegram_login.login(payload=payload)
        except TelegramAuthValidationError as error:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": error.code,
                    "message": error.message,
                },
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_telegram_payload",
                    "message": str(error),
                },
            ) from error

        max_age_seconds = _resolve_cookie_max_age_seconds(expires_at=result.jwt_expires_at)
        response.set_cookie(
            key=normalized_cookie_name,
            value=result.jwt_token,
            max_age=max_age_seconds,
            expires=max_age_seconds,
            path=normalized_cookie_path,
            secure=cookie_secure,
            httponly=True,
            samesite=cookie_samesite,
        )

        return TelegramLoginResponse(
            user_id=str(result.user_id),
            paid_level=_to_paid_level_literal(value=str(result.paid_level)),
        )

    @router.post("/auth/logout", status_code=204, response_model=None)
    def post_auth_logout(response: Response) -> None:
        """
        Clear identity JWT cookie and end current authenticated session.

        Args:
            response: FastAPI response object.
        Returns:
            None.
        Assumptions:
            Logout in v1 is client-cookie invalidation only.
        Raises:
            None.
        Side Effects:
            Deletes authentication cookie from response.
        """
        response.delete_cookie(key=normalized_cookie_name, path=normalized_cookie_path)

    @router.get("/auth/current-user", response_model=CurrentUserResponse)
    def get_auth_current_user(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> CurrentUserResponse:
        """
        Return authenticated user context from CurrentUser dependency.

        Args:
            principal: Resolved authenticated user context.
        Returns:
            CurrentUserResponse: User id and paid-level snapshot.
        Assumptions:
            Dependency already enforces JWT validation and active-user checks.
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



def _build_telegram_payload(*, request: TelegramLoginRequest) -> dict[str, str]:
    """
    Build deterministic payload mapping for Telegram validator/use-case.

    Args:
        request: Parsed API request payload.
    Returns:
        dict[str, str]: Deterministically ordered mapping with string values.
    Assumptions:
        Optional fields may be omitted when absent.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload: dict[str, str] = {
        "auth_date": str(request.auth_date),
        "hash": request.hash,
        "id": str(request.id),
    }
    if request.first_name is not None:
        payload["first_name"] = request.first_name
    if request.last_name is not None:
        payload["last_name"] = request.last_name
    if request.username is not None:
        payload["username"] = request.username
    if request.photo_url is not None:
        payload["photo_url"] = request.photo_url

    return dict(sorted(payload.items(), key=lambda item: item[0]))



def _resolve_cookie_max_age_seconds(*, expires_at: datetime) -> int:
    """
    Resolve non-negative cookie max-age from token expiration timestamp.

    Args:
        expires_at: UTC expiration datetime from login result.
    Returns:
        int: Positive max-age in seconds.
    Assumptions:
        Expiration is set in the future relative to current UTC time.
    Raises:
        ValueError: If expiration datetime is naive/non-UTC.
    Side Effects:
        Reads current system UTC time.
    """
    offset = expires_at.utcoffset()
    if expires_at.tzinfo is None or offset is None:
        raise ValueError("_resolve_cookie_max_age_seconds requires UTC expiration datetime")
    if offset.total_seconds() != 0:
        raise ValueError("_resolve_cookie_max_age_seconds requires UTC expiration datetime")

    now = datetime.now(timezone.utc)
    seconds = int((expires_at - now).total_seconds())
    return max(1, seconds)


def _to_paid_level_literal(*, value: str) -> Literal["free", "base", "pro", "ultra"]:
    """
    Convert runtime paid-level string into strict API literal type.

    Args:
        value: Runtime paid-level value.
    Returns:
        Literal["free", "base", "pro", "ultra"]: Strict literal for response model.
    Assumptions:
        Domain paid-level value is expected to be one of identity v1 literals.
    Raises:
        ValueError: If value is outside supported set.
    Side Effects:
        None.
    """
    normalized = value.strip().lower()
    if normalized not in _PAID_LEVEL_VALUES:
        raise ValueError(f"Unsupported paid_level value: {value!r}")
    return cast(Literal["free", "base", "pro", "ultra"], normalized)
