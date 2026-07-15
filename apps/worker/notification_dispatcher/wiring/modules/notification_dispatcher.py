from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping

import yaml
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from trading.contexts.notifications.adapters import (
    FakeNotificationProvider,
    HttpNotificationProvider,
    HttpNotificationProviderConfig,
    LogOnlyNotificationProvider,
    PostgresNotificationRepository,
    PsycopgNotificationPostgresGateway,
    TelegramBotApiNotificationProvider,
    TelegramNotificationProviderConfig,
)
from trading.contexts.notifications.adapters.outbound.persistence.postgres import (
    PostgresNotificationProviderRepository,
)
from trading.contexts.notifications.adapters.outbound.providers import (
    PostgresOpenBaoTelegramRecipientResolver,
)
from trading.contexts.notifications.application import (
    NotificationDispatcher,
    NotificationDispatcherConfig,
)
from trading.contexts.notifications.application.ports import (
    NotificationProvider,
    NotificationProviderRepository,
    NotificationRepository,
)
from trading.contexts.notifications.domain import (
    NotificationProviderDescriptor,
    NotificationProviderHealth,
    NotificationProviderInstance,
)
from trading.platform.secrets import (
    OpenBaoSecretResolver,
    SecretKind,
    SecureTokenFile,
)

_PREFERRED_POSTGRES_DSN_KEY = "NOTIFICATIONS_PG_DSN"
_FALLBACK_POSTGRES_DSN_KEYS = ("STRATEGY_PG_DSN", "POSTGRES_DSN")
_OPENBAO_ADDRESS_KEY = "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS"
_OPENBAO_TOKEN_FILE_KEY = "ROEHUB_NOTIFICATIONS_OPENBAO_TOKEN_FILE"
_OPENBAO_ROOT_KEY = "ROEHUB_OPENBAO_ROOT"

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NotificationDispatcherRuntimeConfig:
    enabled: bool
    provider_mode: str
    batch_size: int
    lease_seconds: int
    retry_backoff_seconds: int
    max_retry_backoff_seconds: int
    retry_jitter_ratio: float
    max_attempts: int
    poll_interval_seconds: float
    empty_backoff_seconds: float
    metrics_port: int
    provider_healthcheck_interval_seconds: float
    telegram_api_base_url: str
    telegram_connect_timeout_seconds: float
    telegram_overall_timeout_seconds: float

    def __post_init__(self) -> None:
        if self.provider_healthcheck_interval_seconds <= 0:
            raise ValueError("provider healthcheck interval must be > 0")
        if not 0 < self.telegram_connect_timeout_seconds <= 3:
            raise ValueError("Telegram connect timeout must be in (0, 3]")
        if not 0 < self.telegram_overall_timeout_seconds <= 10:
            raise ValueError("Telegram overall timeout must be in (0, 10]")

    def dispatcher_config(self) -> NotificationDispatcherConfig:
        return NotificationDispatcherConfig(
            batch_size=self.batch_size,
            lease_seconds=self.lease_seconds,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_retry_backoff_seconds=self.max_retry_backoff_seconds,
            retry_jitter_ratio=self.retry_jitter_ratio,
            max_attempts=self.max_attempts,
            allowed_provider_keys=_allowed_provider_keys(provider_mode=self.provider_mode),
        )


class SystemNotificationDispatcherClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class NotificationDispatcherPrometheusMetrics:
    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        instance_labels = ("provider", "provider_instance")
        self.deliveries_claimed_total = Counter(
            "notification_dispatcher_deliveries_claimed_total",
            "Notification dispatcher claimed deliveries",
            instance_labels,
            registry=self.registry,
        )
        self.delivery_results_total = Counter(
            "notification_dispatcher_delivery_results_total",
            "Notification dispatcher delivery results",
            (*instance_labels, "status"),
            registry=self.registry,
        )
        self.delivery_unknown_total = Counter(
            "notifications_delivery_unknown_total",
            "Notification deliveries entering unknown provider state",
            ("category", "provider_instance"),
            registry=self.registry,
        )
        self.delivery_latency_seconds = Histogram(
            "notification_dispatcher_delivery_latency_seconds",
            "Notification delivery latency from creation to terminal dispatcher state",
            instance_labels,
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
        self.provider_ready = Gauge(
            "notification_provider_instance_ready",
            "Whether the provider instance passed its latest redacted health probe",
            instance_labels,
            registry=self.registry,
        )
        self.provider_health_total = Counter(
            "notification_provider_instance_health_total",
            "Provider instance health probes by bounded status",
            (*instance_labels, "status"),
            registry=self.registry,
        )

    def on_delivery_claimed(
        self, *, provider_key: str, provider_instance_id: str
    ) -> None:
        self.deliveries_claimed_total.labels(
            provider=provider_key, provider_instance=provider_instance_id
        ).inc()

    def on_delivery_result(
        self,
        *,
        provider_key: str,
        provider_instance_id: str,
        category: str,
        status: str,
    ) -> None:
        self.delivery_results_total.labels(
            provider=provider_key,
            provider_instance=provider_instance_id,
            status=status,
        ).inc()
        if status == "unknown":
            self.delivery_unknown_total.labels(
                category=category[:64], provider_instance=provider_instance_id
            ).inc()

    def observe_delivery_latency_seconds(
        self, *, provider_key: str, provider_instance_id: str, seconds: float
    ) -> None:
        self.delivery_latency_seconds.labels(
            provider=provider_key, provider_instance=provider_instance_id
        ).observe(seconds)

    def set_pending_age_seconds(self, *, seconds: float) -> None:
        self.pending_age_seconds.set(seconds)

    def set_unknown_count(self, *, count: int) -> None:
        self.unknown_deliveries.set(count)

    def on_provider_health(
        self, *, provider_key: str, provider_instance_id: str, status: str
    ) -> None:
        labels = {
            "provider": provider_key,
            "provider_instance": provider_instance_id,
        }
        self.provider_ready.labels(**labels).set(1 if status == "ready" else 0)
        self.provider_health_total.labels(**labels, status=status).inc()


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
        retry_backoff_seconds=_int(dispatcher.get("retry_backoff_seconds"), default=5),
        max_retry_backoff_seconds=_int(
            dispatcher.get("max_retry_backoff_seconds"), default=300
        ),
        retry_jitter_ratio=_float(dispatcher.get("retry_jitter_ratio"), default=0.2),
        max_attempts=_int(dispatcher.get("max_attempts"), default=5),
        poll_interval_seconds=_float(dispatcher.get("poll_interval_seconds"), default=2.0),
        empty_backoff_seconds=_float(dispatcher.get("empty_backoff_seconds"), default=5.0),
        metrics_port=_int(metrics.get("port"), default=9210),
        provider_healthcheck_interval_seconds=_float(
            providers.get("healthcheck_interval_seconds"), default=30.0
        ),
        telegram_api_base_url=_text(
            telegram.get("api_base_url"), default="https://api.telegram.org"
        ),
        telegram_connect_timeout_seconds=_float(
            telegram.get("connect_timeout_seconds"), default=3.0
        ),
        telegram_overall_timeout_seconds=_float(
            telegram.get("overall_timeout_seconds"), default=10.0
        ),
    )


def build_notification_dispatcher(
    *,
    repository: NotificationRepository,
    runtime_config: NotificationDispatcherRuntimeConfig,
    providers: tuple[NotificationProvider, ...] | None = None,
    metrics: NotificationDispatcherPrometheusMetrics | None = None,
) -> NotificationDispatcher:
    effective_providers = providers or (
        LogOnlyNotificationProvider(),
        FakeNotificationProvider(),
    )
    return NotificationDispatcher(
        repository=repository,
        providers=effective_providers,
        clock=SystemNotificationDispatcherClock(),
        config=runtime_config.dispatcher_config(),
        metrics=metrics,
    )


@dataclass(frozen=True, slots=True)
class NotificationDispatcherApp:
    dispatcher: NotificationDispatcher
    runtime_config: NotificationDispatcherRuntimeConfig
    metrics: NotificationDispatcherPrometheusMetrics
    providers: tuple[NotificationProvider, ...] = ()
    provider_repository: NotificationProviderRepository | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        start_http_server(self.runtime_config.metrics_port, registry=self.metrics.registry)
        log.info(
            "notification-dispatcher metrics server started on port %s",
            self.runtime_config.metrics_port,
        )
        tasks = [self._run_dispatcher_loop(stop_event)]
        if self.providers and self.provider_repository is not None:
            tasks.append(self._run_provider_health_probe_loop(stop_event))
        await asyncio.gather(*tasks)

    async def _run_dispatcher_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            result = await asyncio.to_thread(self.dispatcher.drain_once)
            wait_seconds = (
                self.runtime_config.poll_interval_seconds
                if result.claimed
                else self.runtime_config.empty_backoff_seconds
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            except TimeoutError:
                continue

    async def _run_provider_health_probe_loop(self, stop_event: asyncio.Event) -> None:
        if self.provider_repository is None:
            return
        while not stop_event.is_set():
            for provider in self.providers:
                try:
                    health = await asyncio.to_thread(provider.health)
                except Exception:  # noqa: BLE001
                    health = NotificationProviderHealth(
                        instance_id=provider.provider_instance_id,
                        status="degraded",
                        checked_at=datetime.now(UTC),
                        error_code="provider_transport_error",
                    )
                self.provider_repository.record_health(health=health)
                self.metrics.on_provider_health(
                    provider_key=provider.provider_key,
                    provider_instance_id=str(provider.provider_instance_id),
                    status=health.status,
                )
                log_method = log.info if health.status == "ready" else log.warning
                log_method(
                    "notification provider health provider=%s instance=%s status=%s "
                    "error_code=%s",
                    provider.provider_key,
                    provider.provider_instance_id,
                    health.status,
                    health.error_code or "none",
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.runtime_config.provider_healthcheck_interval_seconds,
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
    provider_repository = PostgresNotificationProviderRepository(gateway=gateway)
    providers = _build_runtime_providers(
        provider_repository=provider_repository,
        gateway=gateway,
        runtime_config=runtime_config,
        environ=env,
    )
    metrics = NotificationDispatcherPrometheusMetrics()
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        providers=providers,
        metrics=metrics,
    )
    return NotificationDispatcherApp(
        dispatcher=dispatcher,
        runtime_config=runtime_config,
        metrics=metrics,
        providers=providers,
        provider_repository=provider_repository,
    )


def postgres_dsn_presence(*, environ: Mapping[str, str]) -> dict[str, bool]:
    return {
        key: bool(environ.get(key, "").strip())
        for key in (_PREFERRED_POSTGRES_DSN_KEY, *_FALLBACK_POSTGRES_DSN_KEYS)
    }


def openbao_service_input_presence(*, environ: Mapping[str, str]) -> dict[str, bool]:
    return {
        _OPENBAO_ADDRESS_KEY: bool(environ.get(_OPENBAO_ADDRESS_KEY, "").strip()),
        _OPENBAO_TOKEN_FILE_KEY: bool(
            environ.get(_OPENBAO_TOKEN_FILE_KEY, "").strip()
        ),
        _OPENBAO_ROOT_KEY: bool(environ.get(_OPENBAO_ROOT_KEY, "").strip()),
    }


def resolve_notification_postgres_dsn(*, environ: Mapping[str, str]) -> str:
    for key in (_PREFERRED_POSTGRES_DSN_KEY, *_FALLBACK_POSTGRES_DSN_KEYS):
        value = environ.get(key, "").strip()
        if value:
            return value
    raise ValueError("notification dispatcher requires NOTIFICATIONS_PG_DSN or fallback DSN")


def _build_runtime_providers(
    *,
    provider_repository: NotificationProviderRepository,
    gateway: PsycopgNotificationPostgresGateway,
    runtime_config: NotificationDispatcherRuntimeConfig,
    environ: Mapping[str, str],
) -> tuple[NotificationProvider, ...]:
    instances = provider_repository.list_active_instances()
    needs_openbao = any(instance.secret_ref is not None for instance in instances)
    secret_resolver: OpenBaoSecretResolver | None = None
    recipient_resolver: PostgresOpenBaoTelegramRecipientResolver | None = None
    if needs_openbao:
        address = _required(environ, _OPENBAO_ADDRESS_KEY)
        token_path = Path(_required(environ, _OPENBAO_TOKEN_FILE_KEY))
        secret_root = environ.get(_OPENBAO_ROOT_KEY, "kv/roehub").strip() or "kv/roehub"
        secret_resolver = OpenBaoSecretResolver(
            address=address,
            token_source=SecureTokenFile(token_path),
            secret_root=secret_root,
        )
        recipient_resolver = PostgresOpenBaoTelegramRecipientResolver(
            gateway=gateway,
            secret_resolver=secret_resolver,
        )

    providers: list[NotificationProvider] = []
    for instance in instances:
        if instance.provider_key == "log_only":
            providers.append(
                LogOnlyNotificationProvider(
                    provider_instance_id=instance.instance_id,
                    organization_id=instance.organization_id,
                )
            )
        elif instance.provider_key == "fake":
            providers.append(
                FakeNotificationProvider(
                    provider_instance_id=instance.instance_id,
                    organization_id=instance.organization_id,
                )
            )
        elif instance.provider_key == "telegram_bot_api":
            if secret_resolver is None or recipient_resolver is None:
                raise ValueError("Telegram provider instances require OpenBao service inputs")
            providers.append(
                _telegram_provider(
                    instance=instance,
                    runtime_config=runtime_config,
                    secret_resolver=secret_resolver,
                    recipient_resolver=recipient_resolver,
                )
            )
        else:
            package = provider_repository.get_package(package_id=instance.package_id)
            if package is None:
                raise ValueError("Notification provider package is unavailable")
            providers.append(
                _http_provider(
                    instance=instance,
                    descriptor=package.descriptor,
                    runtime_config=runtime_config,
                    secret_resolver=secret_resolver,
                )
            )
    return tuple(providers)


def _telegram_provider(
    *,
    instance: NotificationProviderInstance,
    runtime_config: NotificationDispatcherRuntimeConfig,
    secret_resolver: OpenBaoSecretResolver,
    recipient_resolver: PostgresOpenBaoTelegramRecipientResolver,
) -> TelegramBotApiNotificationProvider:
    if instance.secret_ref is None:
        raise ValueError("Telegram provider instance secret reference is unavailable")
    api_base_url = _instance_text(
        instance, "api_base_url", runtime_config.telegram_api_base_url
    )
    connect_timeout = _instance_float(
        instance,
        "connect_timeout_seconds",
        runtime_config.telegram_connect_timeout_seconds,
    )
    overall_timeout = _instance_float(
        instance,
        "overall_timeout_seconds",
        runtime_config.telegram_overall_timeout_seconds,
    )
    return TelegramBotApiNotificationProvider(
        config=TelegramNotificationProviderConfig(
            instance=instance,
            api_base_url=api_base_url,
            connect_timeout_seconds=connect_timeout,
            overall_timeout_seconds=overall_timeout,
        ),
        credential_source=_credential_source(
            secret_resolver=secret_resolver,
            secret_ref=instance.secret_ref,
        ),
        recipient_resolver=recipient_resolver.resolve,
    )


def _credential_source(
    *, secret_resolver: OpenBaoSecretResolver, secret_ref: str
) -> Callable[[], str]:
    def resolve() -> str:
        return secret_resolver.resolve(
            secret_ref, expected_kind=SecretKind.TELEGRAM
        ).reveal_text()

    return resolve


def _http_provider(
    *,
    instance: NotificationProviderInstance,
    descriptor: NotificationProviderDescriptor,
    runtime_config: NotificationDispatcherRuntimeConfig,
    secret_resolver: OpenBaoSecretResolver | None,
) -> HttpNotificationProvider:
    endpoint_url = _instance_text(instance, "endpoint_url", "")
    if not endpoint_url:
        raise ValueError("Custom notification provider requires endpoint_url")
    health_url = _instance_text(instance, "health_url", "") or None
    credential_source: Callable[[], str] | None = None
    if instance.secret_ref is not None:
        if secret_resolver is None:
            raise ValueError("Custom provider secret reference requires OpenBao")
        credential_source = _plugin_credential_source(
            secret_resolver=secret_resolver,
            secret_ref=instance.secret_ref,
        )
    return HttpNotificationProvider(
        config=HttpNotificationProviderConfig(
            instance=instance,
            descriptor=descriptor,
            endpoint_url=endpoint_url,
            health_url=health_url,
            connect_timeout_seconds=_instance_float(
                instance,
                "connect_timeout_seconds",
                runtime_config.telegram_connect_timeout_seconds,
            ),
            overall_timeout_seconds=_instance_float(
                instance,
                "overall_timeout_seconds",
                runtime_config.telegram_overall_timeout_seconds,
            ),
        ),
        credential_source=credential_source,
    )


def _plugin_credential_source(
    *, secret_resolver: OpenBaoSecretResolver, secret_ref: str
) -> Callable[[], str]:
    def resolve() -> str:
        return secret_resolver.resolve(
            secret_ref, expected_kind=SecretKind.PLUGIN
        ).reveal_text()

    return resolve


def _allowed_provider_keys(*, provider_mode: str) -> frozenset[str] | None:
    normalized = provider_mode.strip()
    if normalized == "log_only":
        return frozenset({"log_only", "fake"})
    if normalized == "fake":
        return frozenset({"fake"})
    if normalized == "active_instances":
        return None
    raise ValueError("unsupported notification dispatcher provider_mode")


def _required(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise ValueError(f"notification dispatcher requires {key}")
    return value


def _instance_text(
    instance: NotificationProviderInstance, key: str, default: str
) -> str:
    value = instance.config_json.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _instance_float(
    instance: NotificationProviderInstance, key: str, default: float
) -> float:
    value = instance.config_json.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


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
