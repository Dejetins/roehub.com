"""
Identity API routes.

Docs:
  - docs/architecture/identity/identity-telegram-login-user-model-v1.md
  - docs/architecture/identity/identity-2fa-totp-policy-v1.md
  - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
    RequireTwoFactorEnabledDependency,
)
from trading.contexts.identity.adapters.inbound.api.routes import (
    build_auth_telegram_router,
    build_exchange_keys_router,
    build_two_factor_totp_router,
)
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
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
    create_exchange_key_use_case: CreateExchangeKeyUseCase | None = None,
    list_exchange_keys_use_case: ListExchangeKeysUseCase | None = None,
    delete_exchange_key_use_case: DeleteExchangeKeyUseCase | None = None,
    two_factor_enabled_dependency: RequireTwoFactorEnabledDependency | None = None,
) -> APIRouter:
    """
    Build identity router facade for FastAPI app composition root.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      - apps/api/wiring/modules/identity.py

    Args:
        telegram_login: Telegram login use-case.
        two_factor_setup: 2FA setup use-case.
        two_factor_verify: 2FA verify use-case.
        current_user_dependency: FastAPI dependency resolving authenticated principal.
        cookie_name: JWT cookie key.
        cookie_secure: Cookie secure flag.
        cookie_samesite: Cookie SameSite mode.
        cookie_path: Cookie path.
        create_exchange_key_use_case: Optional exchange-key create use-case.
        list_exchange_keys_use_case: Optional exchange-key list use-case.
        delete_exchange_key_use_case: Optional exchange-key delete use-case.
        two_factor_enabled_dependency: Optional auth+2FA dependency for exchange keys routes.
    Returns:
        APIRouter: Configured identity router.
    Assumptions:
        Exchange keys routes are included only when all exchange dependencies are provided.
    Raises:
        ValueError: If exchange route dependencies are partially configured.
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

    configured_exchange_dependencies = sum(
        dependency is not None
        for dependency in (
            create_exchange_key_use_case,
            list_exchange_keys_use_case,
            delete_exchange_key_use_case,
            two_factor_enabled_dependency,
        )
    )
    if configured_exchange_dependencies not in {0, 4}:
        raise ValueError(
            "build_identity_router requires all exchange keys dependencies or none of them"
        )
    if configured_exchange_dependencies == 4:
        assert create_exchange_key_use_case is not None
        assert list_exchange_keys_use_case is not None
        assert delete_exchange_key_use_case is not None
        assert two_factor_enabled_dependency is not None
        router.include_router(
            build_exchange_keys_router(
                create_use_case=create_exchange_key_use_case,
                list_use_case=list_exchange_keys_use_case,
                delete_use_case=delete_exchange_key_use_case,
                two_factor_dependency=two_factor_enabled_dependency,
            )
        )
    return router
