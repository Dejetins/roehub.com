from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from trading.contexts.notifications.application import (
    NotificationRepository,
    NotificationSourceRouter,
    SyntheticNotificationSourceFact,
)
from trading.contexts.notifications.application.source_router import decide_route
from trading.contexts.notifications.domain import NotificationDelivery
from trading.contexts.notifications.domain.notification import (
    NotificationCategory,
    NotificationSeverity,
)
from trading.contexts.strategy.application.ports import (
    StrategyTelegramNotificationV1,
    TelegramNotifier,
)

from .telegram_notifier_hooks import TelegramNotifierHooks

log = logging.getLogger(__name__)


class NotificationsTelegramNotifier(TelegramNotifier):
    """
    NotificationsTelegramNotifier — Strategy rollback-compatible notifier backed by notifications.

    Docs:
      - docs/architecture/notifications/web-execution-telegram-notifications-v1.md
      - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
    Related:
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - src/trading/contexts/notifications/application/source_router.py
    """

    def __init__(
        self,
        *,
        repository: NotificationRepository,
        router: NotificationSourceRouter | None = None,
        hooks: TelegramNotifierHooks | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("NotificationsTelegramNotifier requires repository")
        self._repository = repository
        self._router = router if router is not None else NotificationSourceRouter()
        self._hooks = hooks if hooks is not None else TelegramNotifierHooks()
        self._now_factory = (
            now_factory
            if now_factory is not None
            else lambda: datetime.now(timezone.utc)
        )

    def notify(self, *, notification: StrategyTelegramNotificationV1) -> None:
        try:
            now = self._now_factory()
            fact = _source_fact(notification=notification)
            event = self._repository.record_event(
                event=self._router.event_from_fact(fact=fact, now=now)
            )
            routes = self._repository.list_active_routes(
                owner_user_id=notification.user_id,
                recipient_kind="user",
                category=event.category,
            )

            deliveries = 0
            for route in routes:
                decision, reason = decide_route(event=event, route=route)
                if decision != "deliver" and reason == "provider_not_synthetic":
                    decision = "deliver"
                if decision != "deliver":
                    continue
                self._repository.record_delivery(
                    delivery=NotificationDelivery(
                        delivery_id=uuid4(),
                        event_id=event.event_id,
                        report_run_id=None,
                        command_id=None,
                        route_id=route.route_id,
                        provider_key=route.provider_key,
                        channel_key=route.channel_key,
                        recipient_address_ref=route.recipient_address_ref,
                        template_key=f"{event.category}.v1",
                        rendered_payload_json={
                            "category": event.category,
                            "severity": event.severity,
                            "source_context": event.source_context,
                            "message_text": notification.message_text,
                        },
                        status="pending",
                        attempt_count=0,
                        created_at=now,
                    )
                )
                deliveries += 1

            if deliveries == 0:
                _emit_hook(self._hooks.on_notify_skipped)
                log.warning(
                    (
                        "strategy notification queued none "
                        "reason=no_active_notifications_route event_type=%s "
                        "strategy_id=%s run_id=%s"
                    ),
                    notification.event_type,
                    notification.strategy_id,
                    notification.run_id,
                )
                return

            _emit_hook(self._hooks.on_notify_sent)
            log.info(
                (
                    "strategy notification queued through notifications "
                    "event_type=%s strategy_id=%s run_id=%s deliveries=%s"
                ),
                notification.event_type,
                notification.strategy_id,
                notification.run_id,
                deliveries,
            )
        except Exception:  # noqa: BLE001
            _emit_hook(self._hooks.on_notify_error)
            log.exception(
                (
                    "strategy notification queueing through notifications failed "
                    "event_type=%s strategy_id=%s run_id=%s"
                ),
                notification.event_type,
                notification.strategy_id,
                notification.run_id,
            )


def _source_fact(
    *,
    notification: StrategyTelegramNotificationV1,
) -> SyntheticNotificationSourceFact:
    category, severity = _category_and_severity(event_type=notification.event_type)
    return SyntheticNotificationSourceFact(
        fact_id=f"{notification.event_type}:{notification.strategy_id}:{notification.run_id}",
        owner_user_id=notification.user_id,
        recipient_kind="user",
        source_context="strategy",
        source_event_type=notification.event_type,
        category=category,
        severity=severity,
        occurred_at=notification.ts,
        scope_json={
            "strategy_id": str(notification.strategy_id),
            "run_id": str(notification.run_id),
            "instrument_key": notification.instrument_key,
            "timeframe": notification.timeframe,
        },
        payload_json={
            "event_type": notification.event_type,
            "message_text": notification.message_text,
        },
    )


def _category_and_severity(
    *, event_type: str
) -> tuple[NotificationCategory, NotificationSeverity]:
    if event_type == "failed":
        return "strategy_run_failed", "warning"
    if event_type == "signal":
        return "strategy_signal", "info"
    if event_type in {"trade_open", "trade_close"}:
        return "trade_fill", "info"
    return "strategy_signal", "info"


def _emit_hook(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()
