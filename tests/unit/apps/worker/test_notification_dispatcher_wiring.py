from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from prometheus_client import CollectorRegistry, generate_latest

from apps.worker.notification_dispatcher.wiring.modules.notification_dispatcher import (
    NotificationDispatcherPrometheusMetrics,
    build_notification_dispatcher,
    load_notification_dispatcher_runtime_config,
    postgres_dsn_presence,
    resolve_notification_postgres_dsn,
    telegram_credential_presence,
)
from trading.contexts.notifications.adapters import InMemoryNotificationRepository
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
