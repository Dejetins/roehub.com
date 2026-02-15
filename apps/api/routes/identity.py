"""
Identity API routes.

Docs:
  - docs/architecture/identity/identity-telegram-login-user-model-v1.md
  - docs/architecture/identity/identity-2fa-totp-policy-v1.md
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.inbound.api.routes import (
    build_auth_telegram_router,
    build_two_factor_totp_router,
)
from trading.contexts.identity.application.use_cases import (
    SetupTwoFactorTotpUseCase,
    TelegramLoginUseCase,
    VerifyTwoFactorTotpUseCase,
)


def build_identity_router(
    *,
    telegram_login: TelegramLoginUseCase,
    two_factor_setup: SetupTwoFactorTotpUseCase,
    two_factor_verify: VerifyTwoFactorTotpUseCase,
    current_user_dependency: RequireCurrentUserDependency,
    cookie_name: str,
    cookie_secure: bool,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
) -> APIRouter:
    """
    Build identity router facade for FastAPI app composition root.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related: trading.contexts.identity.adapters.inbound.api.routes.auth_telegram,
      trading.contexts.identity.adapters.inbound.api.routes.two_factor_totp,
      apps.api.wiring.modules.identity,
      apps.api.main.app

    Args:
        telegram_login: Telegram login use-case.
        two_factor_setup: 2FA setup use-case.
        two_factor_verify: 2FA verify use-case.
        current_user_dependency: FastAPI dependency resolving authenticated principal.
        cookie_name: JWT cookie key.
        cookie_secure: Cookie secure flag.
        cookie_samesite: Cookie SameSite mode.
        cookie_path: Cookie path.
    Returns:
        APIRouter: Configured identity router.
    Assumptions:
        Use-case and dependency are pre-wired with runtime settings.
    Raises:
        ValueError: If underlying route builder validates invalid arguments.
    Side Effects:
        None.
    """
    router = APIRouter()
    router.include_router(
        build_auth_telegram_router(
            telegram_login=telegram_login,
            current_user_dependency=current_user_dependency,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            cookie_samesite=cookie_samesite,
            cookie_path=cookie_path,
        )
    )
    router.include_router(
        build_two_factor_totp_router(
            setup_use_case=two_factor_setup,
            verify_use_case=two_factor_verify,
            current_user_dependency=current_user_dependency,
        )
    )
    return router
