"""
Identity API routes.

Docs:
  - docs/architecture/identity/local-auth-sessions-recovery-v1.md
  - docs/architecture/identity/oidc-authentication-provider-v1.md
  - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from trading.contexts.identity.adapters.inbound.api.deps import (
    RequireCurrentUserDependency,
)
from trading.contexts.identity.adapters.inbound.api.routes import (
    build_auth_current_user_router,
    build_auth_local_router,
    build_auth_oidc_router,
    build_exchange_keys_router,
    build_organizations_router,
)
from trading.contexts.identity.application import (
    AccountSettingsRepository,
    IdentityClock,
    SessionRepository,
    UserRepository,
)
from trading.contexts.identity.application.use_cases import (
    CreateExchangeKeyUseCase,
    DeleteExchangeKeyUseCase,
    ListExchangeKeysUseCase,
    LocalAuthService,
    OidcAuthenticationService,
    OrganizationAccessService,
)


def build_identity_router(
    *,
    current_user_dependency: RequireCurrentUserDependency,
    user_repository: UserRepository,
    session_repository: SessionRepository,
    clock: IdentityClock,
    cookie_name: str,
    cookie_secure: bool,
    session_idle_ttl_seconds: int,
    session_absolute_ttl_seconds: int,
    cookie_samesite: Literal["lax", "strict", "none"] = "lax",
    cookie_path: str = "/",
    oidc_authentication_service: OidcAuthenticationService | None = None,
    audit_events_repository: AccountSettingsRepository | None = None,
    create_exchange_key_use_case: CreateExchangeKeyUseCase | None = None,
    list_exchange_keys_use_case: ListExchangeKeysUseCase | None = None,
    delete_exchange_key_use_case: DeleteExchangeKeyUseCase | None = None,
    organization_access_service: OrganizationAccessService | None = None,
    local_auth_service: LocalAuthService | None = None,
) -> APIRouter:
    """
    Build identity router facade for FastAPI app composition root.

    Docs:
      - docs/architecture/identity/local-auth-sessions-recovery-v1.md
      - docs/architecture/identity/oidc-authentication-provider-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      - apps/api/wiring/modules/identity.py

    Args:
        current_user_dependency: FastAPI dependency resolving authenticated principal.
        audit_events_repository: Audit event writer for credential mutation events.
        user_repository: Local Roehub user repository.
        session_repository: Local Roehub session repository for create/revoke lifecycle.
        clock: UTC clock used for login/logout timestamps.
        cookie_name: Opaque session cookie key.
        cookie_secure: Cookie secure flag.
        session_idle_ttl_seconds: Persisted Roehub session idle TTL in seconds.
        session_absolute_ttl_seconds: Persisted Roehub session absolute TTL in seconds.
        cookie_samesite: Cookie SameSite mode.
        cookie_path: Cookie path.
        oidc_authentication_service: Optional provider-neutral OIDC orchestration service.
        create_exchange_key_use_case: Optional exchange-key create use-case.
        list_exchange_keys_use_case: Optional exchange-key list use-case.
        delete_exchange_key_use_case: Optional exchange-key delete use-case.
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
        build_auth_current_user_router(
            current_user_dependency=current_user_dependency,
        )
    )
    if oidc_authentication_service is not None:
        router.include_router(
            build_auth_oidc_router(
                service=oidc_authentication_service,
                current_user_dependency=current_user_dependency,
                cookie_name=cookie_name,
                cookie_secure=cookie_secure,
                session_repository=session_repository,
                clock=clock,
                session_absolute_ttl_seconds=session_absolute_ttl_seconds,
                cookie_samesite=cookie_samesite,
                cookie_path=cookie_path,
            )
        )
    if local_auth_service is not None:
        router.include_router(
            build_auth_local_router(
                service=local_auth_service,
                current_user_dependency=current_user_dependency,
                session_repository=session_repository,
                clock=clock,
                cookie_name=cookie_name,
                cookie_secure=cookie_secure,
                session_absolute_ttl_seconds=session_absolute_ttl_seconds,
                cookie_samesite=cookie_samesite,
                cookie_path=cookie_path,
            )
        )

    configured_exchange_dependencies = sum(
        dependency is not None
        for dependency in (
            create_exchange_key_use_case,
            list_exchange_keys_use_case,
            delete_exchange_key_use_case,
        )
    )
    if configured_exchange_dependencies not in {0, 3}:
        raise ValueError(
            "build_identity_router requires all exchange keys dependencies or none of them"
        )
    if configured_exchange_dependencies == 3:
        assert create_exchange_key_use_case is not None
        assert list_exchange_keys_use_case is not None
        assert delete_exchange_key_use_case is not None
        if audit_events_repository is None:
            raise ValueError(
                "build_identity_router requires audit_events_repository for exchange keys"
            )
        router.include_router(
            build_exchange_keys_router(
                create_use_case=create_exchange_key_use_case,
                list_use_case=list_exchange_keys_use_case,
                delete_use_case=delete_exchange_key_use_case,
                current_user_dependency=current_user_dependency,
                audit_events_repository=audit_events_repository,
                clock=clock,
            )
        )
    if organization_access_service is not None:
        router.include_router(
            build_organizations_router(
                service=organization_access_service,
                current_user_dependency=current_user_dependency,
                clock=clock,
            )
        )
    return router
