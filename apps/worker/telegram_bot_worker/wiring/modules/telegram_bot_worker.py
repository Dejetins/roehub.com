from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.application import (
    InMemoryNotificationTelegramBindingStore,
    NotificationTelegramBindingService,
    TelegramCommandHandler,
)
from trading.contexts.notifications.application.ports import NotificationRepository

_PREFERRED_TELEGRAM_CREDENTIAL_KEY = "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN"
_FALLBACK_TELEGRAM_CREDENTIAL_KEY = "TELEGRAM_BOT_TOKEN"


@dataclass(frozen=True, slots=True)
class TelegramBotWorkerRuntimeConfig:
    enabled: bool
    poll_interval_seconds: int
    long_poll_timeout_seconds: int
    telegram_enabled: bool
    telegram_api_base_url: str


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
        telegram_enabled=_bool(telegram.get("enabled"), default=False),
        telegram_api_base_url=_text(
            telegram.get("api_base_url"), default="https://api.telegram.org"
        ),
    )


def build_telegram_command_handler(
    *,
    repository: NotificationRepository | None = None,
    binding_service: NotificationTelegramBindingService | None = None,
) -> TelegramCommandHandler:
    return TelegramCommandHandler(
        repository=repository or InMemoryNotificationRepository(),
        binding_service=binding_service
        or NotificationTelegramBindingService(
            store=InMemoryNotificationTelegramBindingStore()
        ),
    )


def telegram_bot_credential_presence(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, bool]:
    env = os.environ if environ is None else environ
    return {
        _PREFERRED_TELEGRAM_CREDENTIAL_KEY: bool(
            env.get(_PREFERRED_TELEGRAM_CREDENTIAL_KEY, "").strip()
        ),
        _FALLBACK_TELEGRAM_CREDENTIAL_KEY: bool(
            env.get(_FALLBACK_TELEGRAM_CREDENTIAL_KEY, "").strip()
        ),
    }


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


def _text(value: object, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("expected non-empty text config value")
