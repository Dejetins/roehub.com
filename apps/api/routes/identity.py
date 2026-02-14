"""
Identity API routes.

Docs: docs/architecture/identity/identity-telegram-login-user-model-v1.md
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.inbound.api.routes import build_auth_telegram_router
from trading.contexts.identity.application.use_cases import TelegramLoginUseCase


def build_identity_router(
    *,
    telegram_login: TelegramLoginUseCase,
    current_user_dependency: RequireCurrentUserDependency,
    cookie_name: str,
    cookie_secure: bool,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
) -> APIRouter:
    """
    Build identity router facade for FastAPI app composition root.

    Docs: docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related: trading.contexts.identity.adapters.inbound.api.routes.auth_telegram,
      apps.api.wiring.modules.identity,
      apps.api.main.app

    Args:
        telegram_login: Telegram login use-case.
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
    return build_auth_telegram_router(
        telegram_login=telegram_login,
        current_user_dependency=current_user_dependency,
        cookie_name=cookie_name,
        cookie_secure=cookie_secure,
        cookie_samesite=cookie_samesite,
        cookie_path=cookie_path,
    )
