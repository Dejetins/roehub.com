from __future__ import annotations

from typing import Mapping

from fastapi import APIRouter

from apps.api.routes.ui_account import build_ui_account_router as build_ui_account_api_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.adapters.outbound import (
    InMemoryAccountSettingsRepository,
    PostgresAccountSettingsRepository,
    PsycopgIdentityPostgresGateway,
    SystemIdentityClock,
)
from trading.contexts.identity.application import AccountSettingsRepository
from trading.contexts.identity.application.use_cases.account_settings import (
    AccountSettingsUseCase,
)

_IDENTITY_PG_DSN_KEY = "IDENTITY_PG_DSN"


def build_ui_account_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires current_user_dependency")
    return build_ui_account_api_router(
        account_settings=build_account_settings_use_case(environ=environ),
        current_user_dependency=current_user_dependency,
    )


def build_account_settings_use_case(*, environ: Mapping[str, str]) -> AccountSettingsUseCase:
    return AccountSettingsUseCase(
        repository=_build_account_settings_repository(environ=environ),
        clock=SystemIdentityClock(),
    )


def _build_account_settings_repository(
    *,
    environ: Mapping[str, str],
) -> AccountSettingsRepository:
    postgres_dsn = environ.get(_IDENTITY_PG_DSN_KEY, "").strip()
    if postgres_dsn:
        return PostgresAccountSettingsRepository(
            gateway=PsycopgIdentityPostgresGateway(dsn=postgres_dsn)
        )
    return InMemoryAccountSettingsRepository()
