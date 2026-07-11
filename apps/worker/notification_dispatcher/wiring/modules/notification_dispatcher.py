from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import yaml
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from trading.contexts.notifications.adapters import (
    FakeNotificationProvider,
    LogOnlyNotificationProvider,
    PostgresNotificationRepository,
    PsycopgNotificationPostgresGateway,
    TelegramApiHealthProbeConfig,
    TelegramBotApiHealthProbe,
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
_PREFERRED_POSTGRES_DSN_KEY = "NOTIFICATIONS_PG_DSN"
_FALLBACK_POSTGRES_DSN_KEYS = ("STRATEGY_PG_DSN", "POSTGRES_DSN")

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NotificationDispatcherRuntimeConfig:
    enabled: bool
    provider_mode: str
    batch_size: int
    lease_seconds: int
    retry_backoff_seconds: int
    max_attempts: int
    poll_interval_seconds: float
    empty_backoff_seconds: float
    metrics_port: int
    telegram_enabled: bool
    telegram_api_base_url: str
    telegram_send_timeout_s: float
    telegram_healthcheck_enabled: bool
    telegram_healthcheck_interval_seconds: float
    telegram_healthcheck_timeout_s: float

    def __post_init__(self) -> None:
        if self.telegram_healthcheck_interval_seconds <= 0:
            raise ValueError("Telegram healthcheck interval must be > 0")
        if self.telegram_healthcheck_timeout_s <= 0:
            raise ValueError("Telegram healthcheck timeout must be > 0")

    def dispatcher_config(self) -> NotificationDispatcherConfig:
        return NotificationDispatcherConfig(
            batch_size=self.batch_size,
            lease_seconds=self.lease_seconds,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_attempts=self.max_attempts,
            allowed_provider_keys=_allowed_provider_keys(provider_mode=self.provider_mode),
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
        self.telegram_api_up = Gauge(
            "notifications_telegram_api_up",
            "Whether the latest Telegram Bot API getMe probe succeeded",
            registry=self.registry,
        )
        self.telegram_api_probe_latency_seconds = Gauge(
            "notifications_telegram_api_probe_latency_seconds",
            "Latency of the latest Telegram Bot API getMe probe",
            registry=self.registry,
        )
        self.telegram_api_last_success_unixtime = Gauge(
            "notifications_telegram_api_last_success_unixtime",
            "Unix timestamp of the latest successful Telegram Bot API probe",
            registry=self.registry,
        )
        self.telegram_api_probe_total = Counter(
            "notifications_telegram_api_probe_total",
            "Telegram Bot API probes by bounded result",
            ("result",),
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

    def on_telegram_api_probe(
        self, *, up: bool, latency_seconds: float, checked_at_unixtime: float
    ) -> None:
        self.telegram_api_up.set(1 if up else 0)
        self.telegram_api_probe_latency_seconds.set(latency_seconds)
        self.telegram_api_probe_total.labels(result="success" if up else "failure").inc()
        if up:
            self.telegram_api_last_success_unixtime.set(checked_at_unixtime)


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
        poll_interval_seconds=_float(dispatcher.get("poll_interval_seconds"), default=2.0),
        empty_backoff_seconds=_float(dispatcher.get("empty_backoff_seconds"), default=5.0),
        metrics_port=_int(metrics.get("port"), default=9210),
        telegram_enabled=_bool(telegram.get("enabled"), default=False),
        telegram_api_base_url=_text(
            telegram.get("api_base_url"), default="https://api.telegram.org"
        ),
        telegram_send_timeout_s=_float(telegram.get("send_timeout_s"), default=2.0),
        telegram_healthcheck_enabled=_bool(
            telegram.get("healthcheck_enabled"), default=False
        ),
        telegram_healthcheck_interval_seconds=_float(
            telegram.get("healthcheck_interval_seconds"), default=30.0
        ),
        telegram_healthcheck_timeout_s=_float(
            telegram.get("healthcheck_timeout_s"), default=5.0
        ),
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


@dataclass(frozen=True, slots=True)
class NotificationDispatcherApp:
    dispatcher: NotificationDispatcher
    runtime_config: NotificationDispatcherRuntimeConfig
    metrics: NotificationDispatcherPrometheusMetrics
    telegram_health_probe: TelegramBotApiHealthProbe | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        start_http_server(self.runtime_config.metrics_port, registry=self.metrics.registry)
        log.info(
            "notification-dispatcher metrics server started on port %s",
            self.runtime_config.metrics_port,
        )
        tasks = [self._run_dispatcher_loop(stop_event)]
        if self.telegram_health_probe is not None:
            tasks.append(self._run_telegram_health_probe_loop(stop_event))
        await asyncio.gather(*tasks)

    async def _run_dispatcher_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            result = self.dispatcher.drain_once()
            wait_seconds = (
                self.runtime_config.poll_interval_seconds
                if result.claimed
                else self.runtime_config.empty_backoff_seconds
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            except TimeoutError:
                continue

    async def _run_telegram_health_probe_loop(
        self, stop_event: asyncio.Event
    ) -> None:
        if self.telegram_health_probe is None:
            return
        previous_probe_up: bool | None = None
        while not stop_event.is_set():
            probe_result = await asyncio.to_thread(self.telegram_health_probe.probe)
            self.metrics.on_telegram_api_probe(
                up=probe_result.up,
                latency_seconds=probe_result.latency_seconds,
                checked_at_unixtime=time.time(),
            )
            state_changed = (
                previous_probe_up is None or previous_probe_up != probe_result.up
            )
            log_method = (
                log.info
                if state_changed and probe_result.up
                else log.warning
                if state_changed
                else log.debug
            )
            log_method(
                "telegram-api probe up=%s latency_ms=%s error_code=%s state_changed=%s",
                probe_result.up,
                round(probe_result.latency_seconds * 1000, 1),
                probe_result.error_code or "none",
                state_changed,
            )
            previous_probe_up = probe_result.up
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.runtime_config.telegram_healthcheck_interval_seconds,
                )
            except TimeoutError:
                continue


def build_notification_dispatcher_app(
    *,
    config_path: Path,
    environ: Mapping[str, str] | None = None,
) -> NotificationDispatcherApp:
    env = os.environ if environ is None else environ
    runtime_config = load_notification_dispatcher_runtime_config(config_path=config_path)
    gateway = PsycopgNotificationPostgresGateway(
        dsn=resolve_notification_postgres_dsn(environ=env)
    )
    repository = PostgresNotificationRepository(gateway=gateway)
    metrics = NotificationDispatcherPrometheusMetrics()
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        environ=env,
        metrics=metrics,
    )
    credential = _resolve_telegram_credential(environ=env)
    telegram_health_probe = None
    if runtime_config.telegram_healthcheck_enabled:
        telegram_health_probe = TelegramBotApiHealthProbe(
            config=TelegramApiHealthProbeConfig(
                enabled=True,
                credential=credential,
                api_base_url=runtime_config.telegram_api_base_url,
                timeout_s=runtime_config.telegram_healthcheck_timeout_s,
            )
        )
    return NotificationDispatcherApp(
        dispatcher=dispatcher,
        runtime_config=runtime_config,
        metrics=metrics,
        telegram_health_probe=telegram_health_probe,
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


def postgres_dsn_presence(*, environ: Mapping[str, str]) -> dict[str, bool]:
    return {
        key: bool(environ.get(key, "").strip())
        for key in (_PREFERRED_POSTGRES_DSN_KEY, *_FALLBACK_POSTGRES_DSN_KEYS)
    }


def resolve_notification_postgres_dsn(*, environ: Mapping[str, str]) -> str:
    for key in (_PREFERRED_POSTGRES_DSN_KEY, *_FALLBACK_POSTGRES_DSN_KEYS):
        value = environ.get(key, "").strip()
        if value:
            return value
    raise ValueError("notification dispatcher requires NOTIFICATIONS_PG_DSN or fallback DSN")


def _resolve_telegram_credential(*, environ: Mapping[str, str]) -> str | None:
    preferred = environ.get(_PREFERRED_TELEGRAM_CREDENTIAL_KEY, "").strip()
    if preferred:
        return preferred
    fallback = environ.get(_FALLBACK_TELEGRAM_CREDENTIAL_KEY, "").strip()
    if fallback:
        return fallback
    return None


def _allowed_provider_keys(*, provider_mode: str) -> frozenset[str] | None:
    normalized = provider_mode.strip()
    if normalized == "log_only":
        return frozenset({"log_only", "fake"})
    if normalized == "fake":
        return frozenset({"fake"})
    if normalized == "telegram_bot_api":
        return frozenset({"telegram_bot_api"})
    if normalized == "all":
        return None
    raise ValueError("unsupported notification dispatcher provider_mode")


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
