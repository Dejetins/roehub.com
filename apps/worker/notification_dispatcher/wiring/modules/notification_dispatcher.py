from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import yaml
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from trading.contexts.notifications.adapters import (
    FakeNotificationProvider,
    LogOnlyNotificationProvider,
    TelegramBotApiNotificationProvider,
    TelegramNotificationProviderConfig,
)
from trading.contexts.notifications.application import (
    NotificationDispatcher,
    NotificationDispatcherConfig,
)
from trading.contexts.notifications.application.ports import (
    NotificationProvider,
    NotificationRepository,
)

_PREFERRED_TELEGRAM_CREDENTIAL_KEY = "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN"
_FALLBACK_TELEGRAM_CREDENTIAL_KEY = "TELEGRAM_BOT_TOKEN"


@dataclass(frozen=True, slots=True)
class NotificationDispatcherRuntimeConfig:
    enabled: bool
    provider_mode: str
    batch_size: int
    lease_seconds: int
    retry_backoff_seconds: int
    max_attempts: int
    metrics_port: int
    telegram_enabled: bool
    telegram_api_base_url: str
    telegram_send_timeout_s: float

    def dispatcher_config(self) -> NotificationDispatcherConfig:
        return NotificationDispatcherConfig(
            batch_size=self.batch_size,
            lease_seconds=self.lease_seconds,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_attempts=self.max_attempts,
        )


class SystemNotificationDispatcherClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class NotificationDispatcherPrometheusMetrics:
    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.deliveries_claimed_total = Counter(
            "notification_dispatcher_deliveries_claimed_total",
            "Notification dispatcher claimed deliveries",
            ("provider",),
            registry=self.registry,
        )
        self.delivery_results_total = Counter(
            "notification_dispatcher_delivery_results_total",
            "Notification dispatcher delivery results",
            ("provider", "status"),
            registry=self.registry,
        )
        self.delivery_latency_seconds = Histogram(
            "notification_dispatcher_delivery_latency_seconds",
            "Notification delivery latency from creation to terminal dispatcher state",
            ("provider",),
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0, 300.0),
            registry=self.registry,
        )
        self.pending_age_seconds = Gauge(
            "notification_dispatcher_pending_age_seconds",
            "Oldest due notification delivery age in seconds",
            registry=self.registry,
        )
        self.unknown_deliveries = Gauge(
            "notification_dispatcher_unknown_deliveries",
            "Notification deliveries currently in unknown provider state",
            registry=self.registry,
        )

    def on_delivery_claimed(self, *, provider_key: str) -> None:
        self.deliveries_claimed_total.labels(provider=provider_key).inc()

    def on_delivery_result(self, *, provider_key: str, status: str) -> None:
        self.delivery_results_total.labels(provider=provider_key, status=status).inc()

    def observe_delivery_latency_seconds(
        self, *, provider_key: str, seconds: float
    ) -> None:
        self.delivery_latency_seconds.labels(provider=provider_key).observe(seconds)

    def set_pending_age_seconds(self, *, seconds: float) -> None:
        self.pending_age_seconds.set(seconds)

    def set_unknown_count(self, *, count: int) -> None:
        self.unknown_deliveries.set(count)


def load_notification_dispatcher_runtime_config(
    *, config_path: Path
) -> NotificationDispatcherRuntimeConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("notification dispatcher config must be a mapping")
    return notification_dispatcher_runtime_config_from_mapping(payload=payload)


def notification_dispatcher_runtime_config_from_mapping(
    *, payload: Mapping[str, object]
) -> NotificationDispatcherRuntimeConfig:
    notifications = _mapping(payload.get("notifications"), "notifications")
    dispatcher = _mapping(notifications.get("dispatcher"), "notifications.dispatcher")
    providers = _mapping(notifications.get("providers"), "notifications.providers")
    telegram = _mapping(providers.get("telegram"), "notifications.providers.telegram")
    metrics = _mapping(notifications.get("metrics"), "notifications.metrics")
    return NotificationDispatcherRuntimeConfig(
        enabled=_bool(dispatcher.get("enabled"), default=False),
        provider_mode=_text(dispatcher.get("provider_mode"), default="log_only"),
        batch_size=_int(dispatcher.get("batch_size"), default=100),
        lease_seconds=_int(dispatcher.get("lease_seconds"), default=30),
        retry_backoff_seconds=_int(dispatcher.get("retry_backoff_seconds"), default=60),
        max_attempts=_int(dispatcher.get("max_attempts"), default=3),
        metrics_port=_int(metrics.get("port"), default=9210),
        telegram_enabled=_bool(telegram.get("enabled"), default=False),
        telegram_api_base_url=_text(
            telegram.get("api_base_url"), default="https://api.telegram.org"
        ),
        telegram_send_timeout_s=_float(telegram.get("send_timeout_s"), default=2.0),
    )


def build_notification_dispatcher(
    *,
    repository: NotificationRepository,
    runtime_config: NotificationDispatcherRuntimeConfig,
    environ: Mapping[str, str] | None = None,
    metrics: NotificationDispatcherPrometheusMetrics | None = None,
) -> NotificationDispatcher:
    env = environ or os.environ
    providers: list[NotificationProvider] = [
        LogOnlyNotificationProvider(),
        FakeNotificationProvider(),
    ]
    credential = _resolve_telegram_credential(environ=env)
    providers.append(
        TelegramBotApiNotificationProvider(
            config=TelegramNotificationProviderConfig(
                enabled=runtime_config.telegram_enabled,
                credential=credential,
                api_base_url=runtime_config.telegram_api_base_url,
                send_timeout_s=runtime_config.telegram_send_timeout_s,
            )
        )
    )
    return NotificationDispatcher(
        repository=repository,
        providers=tuple(providers),
        clock=SystemNotificationDispatcherClock(),
        config=runtime_config.dispatcher_config(),
        metrics=metrics,
    )


def telegram_credential_presence(*, environ: Mapping[str, str]) -> dict[str, bool]:
    return {
        _PREFERRED_TELEGRAM_CREDENTIAL_KEY: bool(
            environ.get(_PREFERRED_TELEGRAM_CREDENTIAL_KEY, "").strip()
        ),
        _FALLBACK_TELEGRAM_CREDENTIAL_KEY: bool(
            environ.get(_FALLBACK_TELEGRAM_CREDENTIAL_KEY, "").strip()
        ),
    }


def _resolve_telegram_credential(*, environ: Mapping[str, str]) -> str | None:
    preferred = environ.get(_PREFERRED_TELEGRAM_CREDENTIAL_KEY, "").strip()
    if preferred:
        return preferred
    fallback = environ.get(_FALLBACK_TELEGRAM_CREDENTIAL_KEY, "").strip()
    if fallback:
        return fallback
    return None


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
