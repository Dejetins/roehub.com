from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import build_ui_account_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application import AccountSettingsRepository, IdentityClock
from trading.contexts.identity.application.use_cases import (
    AccountSettingsUseCase,
    ListExchangeKeysUseCase,
)

_UI_ACCOUNT_ALLOWED_ORIGINS_KEY = "UI_ACCOUNT_ALLOWED_ORIGINS"
_DEFAULT_ALLOWED_UI_ORIGINS = (
    "http://127.0.0.1:8010",
    "http://localhost:8010",
    "http://web.local",
)


@dataclass(frozen=True, slots=True)
class UiAccountApiModule:
    router: APIRouter


def build_ui_account_api_module(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
    account_settings_repository: AccountSettingsRepository,
    list_exchange_keys_use_case: ListExchangeKeysUseCase,
    clock: IdentityClock,
) -> UiAccountApiModule:
    """
    Build UI account API module with owner-scoped identity settings dependencies.

    Docs:
      - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
    Related:
      - apps/api/routes/ui_account.py
      - apps/api/main/app.py
    """
    account_settings_use_case = AccountSettingsUseCase(
        repository=account_settings_repository,
        clock=clock,
    )
    return UiAccountApiModule(
        router=build_ui_account_router(
            account_settings_use_case=account_settings_use_case,
            list_exchange_keys_use_case=list_exchange_keys_use_case,
            current_user_dependency=current_user_dependency,
            allowed_ui_origins=_resolve_allowed_ui_origins(environ=environ),
        )
    )


def _resolve_allowed_ui_origins(*, environ: Mapping[str, str]) -> tuple[str, ...]:
    raw_value = environ.get(_UI_ACCOUNT_ALLOWED_ORIGINS_KEY, "").strip()
    if not raw_value:
        return _DEFAULT_ALLOWED_UI_ORIGINS
    return tuple(
        origin.strip().rstrip("/")
        for origin in raw_value.split(",")
        if origin.strip()
    )
