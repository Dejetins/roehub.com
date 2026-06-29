from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationReportRun,
    NotificationRoute,
    TelegramUpdate,
)
from trading.shared_kernel.primitives import UserId


@dataclass(slots=True)
class InMemoryNotificationRepository:
    events: dict[UUID, NotificationEvent] = field(default_factory=dict)
    routes: dict[UUID, NotificationRoute] = field(default_factory=dict)
    deliveries: dict[UUID, NotificationDelivery] = field(default_factory=dict)
    attempts: dict[UUID, NotificationDeliveryAttempt] = field(default_factory=dict)
    telegram_updates: dict[int, TelegramUpdate] = field(default_factory=dict)
    report_runs: dict[UUID, NotificationReportRun] = field(default_factory=dict)

    def record_event(self, *, event: NotificationEvent) -> NotificationEvent:
        existing = self.get_event_by_dedupe_key(dedupe_key=event.dedupe_key)
        if existing is not None:
            return existing
        self.events[event.event_id] = event
        return event

    def get_event_by_dedupe_key(self, *, dedupe_key: str) -> NotificationEvent | None:
        for event in self.events.values():
            if event.dedupe_key == dedupe_key:
                return event
        return None

    def upsert_route(self, *, route: NotificationRoute) -> NotificationRoute:
        self.routes[route.route_id] = route
        return route

    def list_active_routes(
        self, *, owner_user_id: UserId | None, recipient_kind: str, category: str
    ) -> tuple[NotificationRoute, ...]:
        return tuple(
            route
            for route in self.routes.values()
            if route.status == "active"
            and route.recipient_kind == recipient_kind
            and route.owner_user_id == owner_user_id
            and (not route.category_filter or category in route.category_filter)
        )

    def record_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        self.deliveries[delivery.delivery_id] = delivery
        return delivery

    def list_due_deliveries(
        self, *, now: datetime, limit: int
    ) -> tuple[NotificationDelivery, ...]:
        due: list[NotificationDelivery] = []
        for delivery in sorted(self.deliveries.values(), key=lambda item: item.created_at):
            if len(due) >= limit:
                break
            if delivery.status in {"pending", "retry"} and (
                delivery.next_attempt_at is None or delivery.next_attempt_at <= now
            ):
                due.append(delivery)
                continue
            if (
                delivery.status == "claimed"
                and delivery.lease_until is not None
                and delivery.lease_until <= now
            ):
                due.append(delivery)
        return tuple(due)

    def update_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        self.deliveries[delivery.delivery_id] = delivery
        return delivery

    def count_deliveries_by_status(self, *, status: str) -> int:
        return sum(1 for delivery in self.deliveries.values() if delivery.status == status)

    def claim_delivery(
        self, *, delivery_id: UUID, lease_until: datetime, now: datetime
    ) -> NotificationDelivery | None:
        delivery = self.deliveries.get(delivery_id)
        if delivery is None:
            return None
        if delivery.status == "claimed":
            if delivery.lease_until is None or delivery.lease_until > now:
                return None
        elif delivery.status not in {"pending", "retry"}:
            return None
        elif delivery.next_attempt_at is not None and delivery.next_attempt_at > now:
            return None
        claimed = NotificationDelivery(
            delivery_id=delivery.delivery_id,
            event_id=delivery.event_id,
            report_run_id=delivery.report_run_id,
            command_id=delivery.command_id,
            route_id=delivery.route_id,
            provider_key=delivery.provider_key,
            channel_key=delivery.channel_key,
            recipient_address_ref=delivery.recipient_address_ref,
            template_key=delivery.template_key,
            rendered_payload_json=delivery.rendered_payload_json,
            status="claimed",
            attempt_count=delivery.attempt_count + 1,
            next_attempt_at=delivery.next_attempt_at,
            lease_until=lease_until,
            last_error_code=delivery.last_error_code,
            provider_message_id=delivery.provider_message_id,
            created_at=delivery.created_at,
            sent_at=delivery.sent_at,
        )
        _ = now
        self.deliveries[delivery_id] = claimed
        return claimed

    def record_delivery_attempt(
        self, *, attempt: NotificationDeliveryAttempt
    ) -> NotificationDeliveryAttempt:
        self.attempts[attempt.attempt_id] = attempt
        return attempt

    def record_telegram_update(self, *, update: TelegramUpdate) -> TelegramUpdate:
        existing = self.get_telegram_update(telegram_update_id=update.telegram_update_id)
        if existing is not None:
            return existing
        self.telegram_updates[update.telegram_update_id] = update
        return update

    def get_telegram_update(
        self, *, telegram_update_id: int
    ) -> TelegramUpdate | None:
        return self.telegram_updates.get(telegram_update_id)

    def record_report_run(self, *, report_run: NotificationReportRun) -> NotificationReportRun:
        self.report_runs[report_run.report_run_id] = report_run
        return report_run
