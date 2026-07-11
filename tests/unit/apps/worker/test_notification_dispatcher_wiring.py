from __future__ import annotations

import asyncio
import time
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
    postgres_dsn_presence,
    resolve_notification_postgres_dsn,
    telegram_credential_presence,
)
from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
    TelegramApiHealthProbeResult,
)
from trading.contexts.notifications.domain import NotificationDelivery
from trading.contexts.notifications.domain.notification import NotificationProviderKey


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
        assert runtime_config.provider_mode == "log_only"
        assert runtime_config.telegram_enabled is False
        assert runtime_config.telegram_api_base_url == "https://api.telegram.org"
        assert runtime_config.telegram_healthcheck_enabled is (
            config_path == Path("configs/prod/notifications.yaml")
        )
        assert runtime_config.telegram_healthcheck_interval_seconds == 30.0
        assert runtime_config.telegram_healthcheck_timeout_s == 5.0


def test_telegram_credential_presence_reports_only_booleans() -> None:
    presence = telegram_credential_presence(
        environ={
            "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN": "stage03-credential",
            "TELEGRAM_BOT_TOKEN": "",
        }
    )

    assert presence == {
        "ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN": True,
        "TELEGRAM_BOT_TOKEN": False,
    }
    assert "stage03-credential" not in repr(presence)


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
        environ={},
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
    assert "notifications_telegram_api_up" in payload
    assert "notifications_telegram_api_probe_latency_seconds" in payload
    assert "notifications_telegram_api_last_success_unixtime" in payload
    assert "notifications_telegram_api_probe_total" in payload


def test_log_only_provider_mode_skips_telegram_deliveries_without_claiming() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery(provider_key="telegram_bot_api"))
    runtime_config = load_notification_dispatcher_runtime_config(
        config_path=Path("configs/test/notifications.yaml")
    )
    dispatcher = build_notification_dispatcher(
        repository=repository,
        runtime_config=runtime_config,
        environ={},
    )

    result = dispatcher.drain_once()

    assert result.scanned == 1
    assert result.claimed == 0
    assert repository.deliveries[delivery.delivery_id].status == "pending"
    assert repository.deliveries[delivery.delivery_id].attempt_count == 0


def test_slow_telegram_probe_does_not_block_dispatcher_loop() -> None:
    dispatcher = _FastDispatcher()
    app = NotificationDispatcherApp(
        dispatcher=dispatcher,  # type: ignore[arg-type]
        runtime_config=_runtime_config(),
        metrics=NotificationDispatcherPrometheusMetrics(
            registry=CollectorRegistry()
        ),
        telegram_health_probe=_SlowProbe(),  # type: ignore[arg-type]
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        tasks = (
            asyncio.create_task(app._run_dispatcher_loop(stop_event)),
            asyncio.create_task(app._run_telegram_health_probe_loop(stop_event)),
        )
        await asyncio.sleep(0.05)
        assert dispatcher.calls >= 2
        stop_event.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


class _FastDispatcher:
    def __init__(self) -> None:
        self.calls = 0

    def drain_once(self) -> object:
        self.calls += 1
        return type("Batch", (), {"claimed": 0})()


class _SlowProbe:
    def probe(self) -> TelegramApiHealthProbeResult:
        time.sleep(0.15)
        return TelegramApiHealthProbeResult(up=True, latency_seconds=0.15)


def _runtime_config() -> NotificationDispatcherRuntimeConfig:
    return NotificationDispatcherRuntimeConfig(
        enabled=True,
        provider_mode="log_only",
        batch_size=1,
        lease_seconds=30,
        retry_backoff_seconds=60,
        max_attempts=3,
        poll_interval_seconds=0.01,
        empty_backoff_seconds=0.01,
        metrics_port=0,
        telegram_enabled=False,
        telegram_api_base_url="https://api.telegram.org",
        telegram_send_timeout_s=2.0,
        telegram_healthcheck_enabled=True,
        telegram_healthcheck_interval_seconds=30.0,
        telegram_healthcheck_timeout_s=5.0,
    )


def _delivery(*, provider_key: NotificationProviderKey = "log_only") -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
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
