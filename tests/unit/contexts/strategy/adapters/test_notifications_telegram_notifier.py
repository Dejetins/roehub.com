from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.domain import NotificationRoute
from trading.contexts.strategy.adapters.outbound import NotificationsTelegramNotifier
from trading.contexts.strategy.adapters.outbound.messaging.telegram import TelegramNotifierHooks
from trading.contexts.strategy.application import StrategyTelegramNotificationV1
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-4000-8000-000000000900")
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000901")


def test_notifications_telegram_notifier_creates_event_and_pending_delivery() -> None:
    repository = InMemoryNotificationRepository()
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000910")
    repository.upsert_route(route=_route(owner_user_id=owner_user_id, mode="critical_only"))
    hook_calls: list[str] = []

    notifier = NotificationsTelegramNotifier(
        repository=repository,
        hooks=TelegramNotifierHooks(on_notify_sent=lambda: hook_calls.append("sent")),
        now_factory=lambda: _now(),
    )

    notifier.notify(notification=_notification(owner_user_id=owner_user_id))

    assert hook_calls == ["sent"]
    assert len(repository.events) == 1
    event = next(iter(repository.events.values()))
    assert event.source_context == "strategy"
    assert event.source_event_type == "failed"
    assert event.category == "strategy_run_failed"
    assert event.severity == "warning"
    assert event.scope_json["strategy_id"] == "00000000-0000-0000-0000-00000000a910"

    assert len(repository.deliveries) == 1
    delivery = next(iter(repository.deliveries.values()))
    assert delivery.event_id == event.event_id
    assert delivery.provider_key == "telegram_bot_api"
    assert delivery.status == "pending"
    assert delivery.template_key == "strategy_run_failed.v1"
    message_text = delivery.rendered_payload_json["message_text"]
    assert isinstance(message_text, str)
    assert message_text.startswith("FAILED |")


def test_notifications_telegram_notifier_uses_skip_hook_without_active_route() -> None:
    repository = InMemoryNotificationRepository()
    hook_calls: list[str] = []
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000911")
    notifier = NotificationsTelegramNotifier(
        repository=repository,
        hooks=TelegramNotifierHooks(on_notify_skipped=lambda: hook_calls.append("skipped")),
        now_factory=lambda: _now(),
    )

    notifier.notify(notification=_notification(owner_user_id=owner_user_id))

    assert hook_calls == ["skipped"]
    assert len(repository.events) == 1
    assert repository.deliveries == {}


def _route(*, owner_user_id: UserId, mode: str) -> NotificationRoute:
    now = _now()
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="user",
        owner_user_id=owner_user_id,
        channel_key="telegram",
        provider_key="telegram_bot_api",
        mode=mode,  # type: ignore[arg-type]
        category_filter=(),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:test",
        status="active",
        created_at=now,
        updated_at=now,
    )


def _notification(*, owner_user_id: UserId) -> StrategyTelegramNotificationV1:
    strategy_id = UUID("00000000-0000-0000-0000-00000000a910")
    run_id = UUID("00000000-0000-0000-0000-00000000b910")
    return StrategyTelegramNotificationV1(
        organization_id=_ORGANIZATION_ID,
        user_id=owner_user_id,
        ts=_now(),
        strategy_id=strategy_id,
        run_id=run_id,
        event_type="failed",
        instrument_key="binance:BTCUSDT",
        timeframe="1m",
        message_text=(
            f"FAILED | strategy_id={strategy_id} | run_id={run_id} | error=test failure"
        ),
    )


def _now() -> datetime:
    return datetime(2026, 6, 29, 21, 30, tzinfo=timezone.utc)
