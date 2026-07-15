from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from trading.contexts.notifications.adapters.inbound.telegram import TelegramUpdateMapper
from trading.contexts.notifications.adapters.outbound.persistence.postgres import (
    PostgresNotificationProviderRepository,
    PostgresNotificationTelegramBindingStore,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.adapters.outbound.providers import (
    OpenBaoTelegramRecipientSecretStore,
    PostgresTelegramRecipientScopeResolver,
    TelegramBotApiUpdateSource,
)
from trading.contexts.notifications.application import (
    NotificationTelegramBindingService,
    TelegramCommandHandler,
)
from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.application.telegram_worker import TelegramProviderWorker
from trading.contexts.notifications.domain import NotificationProviderInstance
from trading.platform.secrets import OpenBaoSecretResolver, SecretKind, SecureTokenFile
from trading.shared_kernel.primitives import OrganizationId

_OPENBAO_ADDRESS_KEY = "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS"
_OPENBAO_TOKEN_FILE_KEY = "ROEHUB_TELEGRAM_WORKER_OPENBAO_TOKEN_FILE"
_OPENBAO_ROOT_KEY = "ROEHUB_OPENBAO_ROOT"


@dataclass(frozen=True, slots=True)
class TelegramBotWorkerRuntimeConfig:
    enabled: bool
    poll_interval_seconds: int
    long_poll_timeout_seconds: int
    telegram_api_base_url: str
    telegram_connect_timeout_seconds: float


def load_telegram_bot_worker_runtime_config(
    *, config_path: Path
) -> TelegramBotWorkerRuntimeConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("telegram bot worker config must be a mapping")
    notifications = _mapping(payload.get("notifications"), "notifications")
    telegram_bot = _mapping(notifications.get("telegram_bot"), "notifications.telegram_bot")
    providers = _mapping(notifications.get("providers"), "notifications.providers")
    telegram = _mapping(providers.get("telegram"), "notifications.providers.telegram")
    return TelegramBotWorkerRuntimeConfig(
        enabled=_bool(telegram_bot.get("enabled"), default=False),
        poll_interval_seconds=_int(telegram_bot.get("poll_interval_seconds"), default=5),
        long_poll_timeout_seconds=_int(
            telegram_bot.get("long_poll_timeout_seconds"), default=30
        ),
        telegram_api_base_url=_text(
            telegram.get("api_base_url"), default="https://api.telegram.org"
        ),
        telegram_connect_timeout_seconds=_float(
            telegram.get("connect_timeout_seconds"), default=3.0
        ),
    )


def build_telegram_command_handler(
    *,
    repository: NotificationRepository,
    binding_service: NotificationTelegramBindingService,
) -> TelegramCommandHandler:
    return TelegramCommandHandler(
        repository=repository,
        binding_service=binding_service,
    )


def build_telegram_provider_workers(
    *,
    repository: NotificationRepository,
    postgres_dsn: str,
    runtime_config: TelegramBotWorkerRuntimeConfig,
    environ: Mapping[str, str] | None = None,
) -> tuple[TelegramProviderWorker, ...]:
    env = os.environ if environ is None else environ
    gateway = PsycopgNotificationPostgresGateway(dsn=postgres_dsn)
    provider_repository = PostgresNotificationProviderRepository(gateway=gateway)
    binding_store = PostgresNotificationTelegramBindingStore(gateway=gateway)
    secret_resolver = OpenBaoSecretResolver(
        address=_required(env, _OPENBAO_ADDRESS_KEY),
        token_source=SecureTokenFile(Path(_required(env, _OPENBAO_TOKEN_FILE_KEY))),
        secret_root=env.get(_OPENBAO_ROOT_KEY, "kv/roehub").strip() or "kv/roehub",
    )
    scope_resolver = PostgresTelegramRecipientScopeResolver(gateway=gateway)
    workers: list[TelegramProviderWorker] = []
    for instance in provider_repository.list_active_instances():
        if instance.provider_key != "telegram_bot_api":
            continue
        if instance.secret_ref is None:
            raise ValueError("Telegram provider instance secret reference is unavailable")
        update_source = TelegramBotApiUpdateSource(
            api_base_url=_instance_text(
                instance, "api_base_url", runtime_config.telegram_api_base_url
            ),
            credential_source=_credential_source(
                resolver=secret_resolver, secret_ref=instance.secret_ref
            ),
            connect_timeout_seconds=runtime_config.telegram_connect_timeout_seconds,
        )
        workers.append(
            TelegramProviderWorker(
                provider_instance_id=instance.instance_id,
                organization_id=instance.organization_id,
                provider_repository=provider_repository,
                update_source=update_source,
                scope_resolver=scope_resolver,
                command_handler_factory=_handler_factory(
                    instance=instance,
                    repository=repository,
                    binding_store=binding_store,
                    recipient_secret_store=OpenBaoTelegramRecipientSecretStore(
                        secret_resolver=secret_resolver
                    ),
                ),
                mapper=TelegramUpdateMapper(),
                long_poll_timeout_seconds=runtime_config.long_poll_timeout_seconds,
            )
        )
    return tuple(workers)


def openbao_service_input_presence(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, bool]:
    environment = os.environ if environ is None else environ
    return {
        _OPENBAO_ADDRESS_KEY: bool(
            environment.get(_OPENBAO_ADDRESS_KEY, "").strip()
        ),
        _OPENBAO_TOKEN_FILE_KEY: bool(
            environment.get(_OPENBAO_TOKEN_FILE_KEY, "").strip()
        ),
        _OPENBAO_ROOT_KEY: bool(environment.get(_OPENBAO_ROOT_KEY, "").strip()),
    }


def _handler_factory(
    *,
    instance: NotificationProviderInstance,
    repository: NotificationRepository,
    binding_store: PostgresNotificationTelegramBindingStore,
    recipient_secret_store: OpenBaoTelegramRecipientSecretStore,
) -> Callable[[OrganizationId], TelegramCommandHandler]:
    def build(organization_id: OrganizationId) -> TelegramCommandHandler:
        if instance.organization_id is not None and instance.organization_id != organization_id:
            raise ValueError("Telegram provider instance belongs to another organization")
        return build_telegram_command_handler(
            repository=repository,
            binding_service=NotificationTelegramBindingService(
                store=binding_store,
                organization_id=organization_id,
                provider_instance_id=instance.instance_id,
                recipient_secret_store=recipient_secret_store,
            ),
        )

    return build


def _credential_source(
    *, resolver: OpenBaoSecretResolver, secret_ref: str
) -> Callable[[], str]:
    def resolve() -> str:
        return resolver.resolve(
            secret_ref, expected_kind=SecretKind.TELEGRAM
        ).reveal_text()

    return resolve


def _instance_text(
    instance: NotificationProviderInstance, key: str, default: str
) -> str:
    value = instance.config_json.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _required(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise ValueError(f"Telegram bot worker requires {key}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("expected bool config value")


def _int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError("expected int config value")


def _float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("expected numeric config value")


def _text(value: object, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("expected non-empty text config value")
