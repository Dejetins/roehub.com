from __future__ import annotations

from typing import Mapping

from fastapi import APIRouter

from apps.api.exchange_control_client import build_exchange_control_client_from_environ
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
from trading.contexts.notifications.application import (
    InMemoryNotificationTelegramBindingStore,
    NotificationTelegramBindingService,
)
from trading.contexts.strategy.adapters.outbound import (
    InMemoryStrategyExchangeBindingRepository,
    InMemoryStrategyRepository,
    PostgresStrategyExchangeBindingRepository,
    PostgresStrategyRepository,
    PsycopgStrategyPostgresGateway,
)
from trading.contexts.strategy.application.use_cases import (
    StrategyExchangeBindingService,
)

_IDENTITY_PG_DSN_KEY = "IDENTITY_PG_DSN"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


def build_ui_account_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_account_router requires current_user_dependency")
    clock = SystemIdentityClock()
    return build_ui_account_api_router(
        account_settings=build_account_settings_use_case(environ=environ, clock=clock),
        current_user_dependency=current_user_dependency,
        clock=clock,
        exchange_control_client=build_exchange_control_client_from_environ(environ=environ),
        strategy_binding_service=build_strategy_exchange_binding_service(environ=environ),
        telegram_binding_service=build_notification_telegram_binding_service(),
    )


def build_account_settings_use_case(
    *, environ: Mapping[str, str], clock: SystemIdentityClock | None = None
) -> AccountSettingsUseCase:
    return AccountSettingsUseCase(
        repository=_build_account_settings_repository(environ=environ),
        clock=clock or SystemIdentityClock(),
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


def build_strategy_exchange_binding_service(
    *,
    environ: Mapping[str, str],
) -> StrategyExchangeBindingService:
    postgres_dsn = environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    if postgres_dsn:
        gateway = PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
        return StrategyExchangeBindingService(
            strategy_repository=PostgresStrategyRepository(gateway=gateway),
            binding_repository=PostgresStrategyExchangeBindingRepository(gateway=gateway),
        )
    return StrategyExchangeBindingService(
        strategy_repository=InMemoryStrategyRepository(),
        binding_repository=InMemoryStrategyExchangeBindingRepository(),
    )


def build_notification_telegram_binding_service() -> NotificationTelegramBindingService:
    return NotificationTelegramBindingService(
        store=InMemoryNotificationTelegramBindingStore()
    )
