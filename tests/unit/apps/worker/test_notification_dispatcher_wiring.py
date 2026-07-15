from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from prometheus_client import CollectorRegistry, generate_latest

from apps.worker.notification_dispatcher.wiring.modules.notification_dispatcher import (
    NotificationDispatcherApp,
    NotificationDispatcherPrometheusMetrics,
    NotificationDispatcherRuntimeConfig,
    build_notification_dispatcher,
    load_notification_dispatcher_runtime_config,
    openbao_service_input_presence,
    postgres_dsn_presence,
    resolve_notification_postgres_dsn,
)
from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
)
from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderHealth,
)
from trading.contexts.notifications.domain.notification import NotificationProviderKey
from trading.shared_kernel.primitives import OrganizationId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_LOG_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")
_TELEGRAM_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000003")


def test_notification_dispatcher_configs_are_disabled_and_redacted_by_default() -> None:
    for config_path in (
        Path("configs/dev/notifications.yaml"),
        Path("configs/test/notifications.yaml"),
        Path("configs/prod/notifications.yaml"),
    ):
        runtime_config = load_notification_dispatcher_runtime_config(
            config_path=config_path
        )

        expected_enabled = config_path == Path("configs/prod/notifications.yaml")
        assert runtime_config.enabled is expected_enabled
        assert runtime_config.provider_mode == (
            "active_instances" if expected_enabled else "log_only"
        )
        assert runtime_config.telegram_api_base_url == "https://api.telegram.org"
        assert runtime_config.provider_healthcheck_interval_seconds == 30.0
        assert runtime_config.telegram_connect_timeout_seconds == 3.0
        assert runtime_config.telegram_overall_timeout_seconds == 10.0


def test_openbao_service_input_presence_reports_only_booleans() -> None:
    presence = openbao_service_input_presence(
        environ={
            "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS": "http://openbao:8200",
            "ROEHUB_NOTIFICATIONS_OPENBAO_TOKEN_FILE": "/run/secrets/notifications-token",
            "ROEHUB_OPENBAO_ROOT": "kv/roehub",
        }
    )

    assert presence == {
        "ROEHUB_NOTIFICATIONS_OPENBAO_ADDRESS": True,
        "ROEHUB_NOTIFICATIONS_OPENBAO_TOKEN_FILE": True,
        "ROEHUB_OPENBAO_ROOT": True,
    }
    assert "notifications-token" not in repr(presence)


def test_postgres_dsn_presence_and_resolution_report_only_booleans() -> None:
    environ = {
        "NOTIFICATIONS_PG_DSN": "",
        "STRATEGY_PG_DSN": "postgresql://example",
        "POSTGRES_DSN": "",
    }

    assert postgres_dsn_presence(environ=environ) == {
        "NOTIFICATIONS_PG_DSN": False,
        "STRATEGY_PG_DSN": True,
        "POSTGRES_DSN": False,
    }
    assert resolve_notification_postgres_dsn(environ=environ) == "postgresql://example"


def test_composition_root_drains_backlog_with_log_only_provider() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery())
    runtime_config = load_notification_dispatcher_runtime_config(
        config_path=Path("configs/test/notifications.yaml")
    )
    metrics = NotificationDispatcherPrometheusMetrics(registry=CollectorRegistry())
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        metrics=metrics,
    )

    result = dispatcher.drain_once()
    payload = generate_latest(metrics.registry).decode("utf-8")

    assert result.sent == 1
    assert repository.deliveries[delivery.delivery_id].status == "sent"
    assert len(repository.attempts) == 1
    assert "notification_dispatcher_deliveries_claimed_total" in payload
    assert "notification_dispatcher_delivery_results_total" in payload
    assert "notification_dispatcher_delivery_latency_seconds" in payload
    assert "notification_dispatcher_pending_age_seconds" in payload
    assert "notification_dispatcher_unknown_deliveries" in payload
    assert "notification_provider_instance_ready" in payload
    assert "notification_provider_instance_health_total" in payload
    assert "notifications_delivery_unknown_total" in payload


def test_log_only_provider_mode_skips_telegram_deliveries_without_claiming() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery(provider_key="telegram_bot_api"))
    runtime_config = load_notification_dispatcher_runtime_config(
        config_path=Path("configs/test/notifications.yaml")
    )
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
    )

    result = dispatcher.drain_once()

    assert result.scanned == 1
    assert result.claimed == 0
    assert repository.deliveries[delivery.delivery_id].status == "pending"
    assert repository.deliveries[delivery.delivery_id].attempt_count == 0


def test_slow_telegram_probe_does_not_block_dispatcher_loop() -> None:
    dispatcher = _FastDispatcher()
    provider = _BlockingProvider()
    provider_repository = _ProviderRepositoryProbe()
    app = NotificationDispatcherApp(
        dispatcher=dispatcher,  # type: ignore[arg-type]
        runtime_config=_runtime_config(),
        metrics=NotificationDispatcherPrometheusMetrics(
            registry=CollectorRegistry()
        ),
        providers=(provider,),
        provider_repository=provider_repository,  # type: ignore[arg-type]
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        tasks = (
            asyncio.create_task(app._run_dispatcher_loop(stop_event)),
            asyncio.create_task(app._run_provider_health_probe_loop(stop_event)),
        )
        assert await asyncio.to_thread(provider.started.wait, 1.0)
        calls_when_probe_started = dispatcher.calls
        for _ in range(50):
            if dispatcher.calls > calls_when_probe_started:
                break
            await asyncio.sleep(0.01)
        assert dispatcher.calls > calls_when_probe_started
        provider.release.set()
        stop_event.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_provider_health_exception_degrades_only_affected_instance() -> None:
    provider_repository = _ProviderRepositoryProbe()
    provider = _FailingProvider()
    app = NotificationDispatcherApp(
        dispatcher=_FastDispatcher(),  # type: ignore[arg-type]
        runtime_config=_runtime_config(),
        metrics=NotificationDispatcherPrometheusMetrics(
            registry=CollectorRegistry()
        ),
        providers=(provider,),
        provider_repository=provider_repository,  # type: ignore[arg-type]
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        task = asyncio.create_task(app._run_provider_health_probe_loop(stop_event))
        for _ in range(50):
            if provider_repository.health:
                break
            await asyncio.sleep(0.01)
        stop_event.set()
        await task

    asyncio.run(scenario())

    assert len(provider_repository.health) == 1
    assert provider_repository.health[0].instance_id == _LOG_INSTANCE_ID
    assert provider_repository.health[0].status == "degraded"
    assert provider_repository.health[0].error_code == "provider_transport_error"


class _FastDispatcher:
    def __init__(self) -> None:
        self.calls = 0

    def drain_once(self) -> object:
        self.calls += 1
        return type("Batch", (), {"claimed": 0})()


class _BlockingProvider:
    provider_instance_id = _LOG_INSTANCE_ID
    provider_key = "log_only"
    organization_id = None

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def descriptor(self) -> NotificationProviderDescriptor:
        return NotificationProviderDescriptor(
            provider_key="log_only",
            display_name="Blocking provider",
            package_version="1.0.0",
            config_schema={"type": "object"},
            channels=("telegram",),
            templates=("plain_text.v1",),
            error_codes=("provider_disabled",),
        )

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        _ = delivery
        raise AssertionError("not used")

    def health(self) -> NotificationProviderHealth:
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return NotificationProviderHealth(
            instance_id=self.provider_instance_id,
            status="ready",
            checked_at=datetime.now(timezone.utc),
        )


class _FailingProvider(_BlockingProvider):
    def health(self) -> NotificationProviderHealth:
        raise RuntimeError("controlled health failure")


class _ProviderRepositoryProbe:
    def __init__(self) -> None:
        self.health: list[NotificationProviderHealth] = []

    def record_health(self, *, health: NotificationProviderHealth) -> None:
        self.health.append(health)


def _runtime_config() -> NotificationDispatcherRuntimeConfig:
    return NotificationDispatcherRuntimeConfig(
        enabled=True,
        provider_mode="log_only",
        batch_size=1,
        lease_seconds=30,
        retry_backoff_seconds=60,
        max_retry_backoff_seconds=300,
        retry_jitter_ratio=0.0,
        max_attempts=3,
        poll_interval_seconds=0.01,
        empty_backoff_seconds=0.01,
        metrics_port=0,
        provider_healthcheck_interval_seconds=30.0,
        telegram_api_base_url="https://api.telegram.org",
        telegram_connect_timeout_seconds=3.0,
        telegram_overall_timeout_seconds=10.0,
    )


def _delivery(*, provider_key: NotificationProviderKey = "log_only") -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=(
            _TELEGRAM_INSTANCE_ID
            if provider_key == "telegram_bot_api"
            else _LOG_INSTANCE_ID
        ),
        event_id=UUID("44444444-4444-4444-8444-444444444444"),
        report_run_id=None,
        command_id=None,
        route_id=uuid4(),
        provider_key=provider_key,
        channel_key="telegram",
        recipient_address_ref="telegram_ref:user:stage03",
        template_key="strategy_signal",
        rendered_payload_json={"text": "Stage 03 composition smoke"},
        status="pending",
        attempt_count=0,
        created_at=datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc),
    )
