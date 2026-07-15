from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from uuid import UUID

from trading.contexts.notifications.application.delivery_counters import (
    NotificationDeliveryCounters,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationReportRun,
    NotificationRoute,
    TelegramUpdate,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


@dataclass(slots=True)
class InMemoryNotificationRepository:
    events: dict[UUID, NotificationEvent] = field(default_factory=dict)
    routes: dict[UUID, NotificationRoute] = field(default_factory=dict)
    deliveries: dict[UUID, NotificationDelivery] = field(default_factory=dict)
    attempts: dict[UUID, NotificationDeliveryAttempt] = field(default_factory=dict)
    telegram_updates: dict[tuple[UUID, int], TelegramUpdate] = field(default_factory=dict)
    report_runs: dict[UUID, NotificationReportRun] = field(default_factory=dict)

    def record_event(self, *, event: NotificationEvent) -> NotificationEvent:
        existing = self.get_event_by_dedupe_key(
            organization_id=event.organization_id, dedupe_key=event.dedupe_key
        )
        if existing is not None:
            return existing
        self.events[event.event_id] = event
        return event

    def get_event_by_dedupe_key(
        self, *, organization_id: OrganizationId, dedupe_key: str
    ) -> NotificationEvent | None:
        for event in self.events.values():
            if event.organization_id == organization_id and event.dedupe_key == dedupe_key:
                return event
        return None

    def upsert_route(self, *, route: NotificationRoute) -> NotificationRoute:
        existing = self.routes.get(route.route_id)
        if existing is not None and (
            existing.owner_user_id != route.owner_user_id
            or existing.recipient_kind != route.recipient_kind
            or existing.organization_id != route.organization_id
            or existing.provider_instance_id != route.provider_instance_id
        ):
            raise ValueError("NotificationRoute owner and recipient kind are immutable")
        self.routes[route.route_id] = route
        return route

    def get_route(
        self, *, organization_id: OrganizationId, route_id: UUID
    ) -> NotificationRoute | None:
        route = self.routes.get(route_id)
        if route is None or route.organization_id != organization_id:
            return None
        return route

    def list_active_routes(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId | None,
        recipient_kind: str,
        category: str,
    ) -> tuple[NotificationRoute, ...]:
        return tuple(
            route
            for route in self.routes.values()
            if route.status == "active"
            and route.organization_id == organization_id
            and route.recipient_kind == recipient_kind
            and route.owner_user_id == owner_user_id
            and (not route.category_filter or category in route.category_filter)
        )

    def list_active_report_routes(self) -> tuple[NotificationRoute, ...]:
        return tuple(
            route
            for route in self.routes.values()
            if route.status == "active"
            and route.recipient_kind == "user"
            and route.owner_user_id is not None
            and route.mode in {"reports", "all"}
            and (not route.category_filter or "portfolio_report" in route.category_filter)
        )

    def record_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        route = self.routes.get(delivery.route_id)
        if route is not None and (
            route.organization_id != delivery.organization_id
            or route.provider_instance_id != delivery.provider_instance_id
            or route.provider_key != delivery.provider_key
        ):
            raise ValueError("NotificationDelivery route scope mismatch")
        if delivery.event_id is not None:
            event = self.events.get(delivery.event_id)
            if event is not None and event.organization_id != delivery.organization_id:
                raise ValueError("NotificationDelivery event scope mismatch")
        if delivery.report_run_id is not None:
            report_run = self.report_runs.get(delivery.report_run_id)
            if (
                report_run is not None
                and report_run.organization_id != delivery.organization_id
            ):
                raise ValueError("NotificationDelivery report scope mismatch")
        if delivery.replayed_from_delivery_id is not None:
            original = self.deliveries.get(delivery.replayed_from_delivery_id)
            if original is None:
                raise ValueError("NotificationDelivery replay source is unavailable")
            if (
                original.organization_id != delivery.organization_id
                or original.provider_instance_id != delivery.provider_instance_id
            ):
                raise ValueError("NotificationDelivery replay scope mismatch")
        existing = self.deliveries.get(delivery.delivery_id)
        if existing is not None:
            if existing != delivery:
                raise ValueError("NotificationDelivery identity conflict")
            return existing
        self.deliveries[delivery.delivery_id] = delivery
        return delivery

    def get_delivery(
        self, *, organization_id: OrganizationId, delivery_id: UUID
    ) -> NotificationDelivery | None:
        delivery = self.deliveries.get(delivery_id)
        if delivery is None or delivery.organization_id != organization_id:
            return None
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
        return tuple(due)

    def recover_expired_claims(self, *, now: datetime) -> int:
        recovered = 0
        for delivery_id, delivery in tuple(self.deliveries.items()):
            if (
                delivery.status == "claimed"
                and delivery.lease_until is not None
                and delivery.lease_until <= now
            ):
                self.deliveries[delivery_id] = replace(
                    delivery,
                    status="unknown",
                    lease_until=None,
                    next_attempt_at=None,
                    last_error_code="provider_shutdown",
                )
                recovered += 1
        return recovered

    def update_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery:
        existing = self.deliveries.get(delivery.delivery_id)
        if existing is None:
            raise ValueError("NotificationDelivery is unavailable")
        if (
            existing.organization_id != delivery.organization_id
            or existing.provider_instance_id != delivery.provider_instance_id
            or existing.provider_key != delivery.provider_key
            or existing.route_id != delivery.route_id
        ):
            raise ValueError("NotificationDelivery immutable scope mismatch")
        self.deliveries[delivery.delivery_id] = delivery
        return delivery

    def count_deliveries_by_status(self, *, status: str) -> int:
        return sum(1 for delivery in self.deliveries.values() if delivery.status == status)

    def get_delivery_counters(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, now: datetime
    ) -> NotificationDeliveryCounters:
        owned_route_ids = {
            route.route_id
            for route in self.routes.values()
            if route.organization_id == organization_id
            and route.owner_user_id == owner_user_id
            and route.recipient_kind == "user"
        }
        sent = tuple(
            delivery
            for delivery in self.deliveries.values()
            if delivery.route_id in owned_route_ids
            and delivery.status == "sent"
            and delivery.provider_key == "telegram_bot_api"
        )
        last_sent_at = max(
            (delivery.sent_at for delivery in sent if delivery.sent_at is not None),
            default=None,
        )
        return NotificationDeliveryCounters(
            telegram_sent_total=len(sent),
            telegram_sent_last_24h=sum(
                1
                for delivery in sent
                if delivery.sent_at is not None
                and delivery.sent_at >= now - timedelta(hours=24)
            ),
            last_telegram_sent_at=last_sent_at,
        )

    def claim_delivery(
        self, *, delivery_id: UUID, lease_until: datetime, now: datetime
    ) -> NotificationDelivery | None:
        delivery = self.deliveries.get(delivery_id)
        if delivery is None:
            return None
        if delivery.status not in {"pending", "retry"}:
            return None
        elif delivery.next_attempt_at is not None and delivery.next_attempt_at > now:
            return None
        claimed = NotificationDelivery(
            delivery_id=delivery.delivery_id,
            organization_id=delivery.organization_id,
            provider_instance_id=delivery.provider_instance_id,
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
        delivery = self.deliveries.get(attempt.delivery_id)
        if delivery is None or (
            delivery.organization_id != attempt.organization_id
            or delivery.provider_instance_id != attempt.provider_instance_id
            or delivery.provider_key != attempt.provider_key
        ):
            raise ValueError("NotificationDeliveryAttempt scope mismatch")
        self.attempts[attempt.attempt_id] = attempt
        return attempt

    def record_telegram_update(self, *, update: TelegramUpdate) -> TelegramUpdate:
        existing = self.get_telegram_update(
            organization_id=update.organization_id,
            provider_instance_id=update.provider_instance_id,
            telegram_update_id=update.telegram_update_id,
        )
        if existing is not None:
            return existing
        self.telegram_updates[(update.provider_instance_id, update.telegram_update_id)] = update
        return update

    def get_telegram_update(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        telegram_update_id: int,
    ) -> TelegramUpdate | None:
        update = self.telegram_updates.get((provider_instance_id, telegram_update_id))
        if update is None or update.organization_id != organization_id:
            return None
        return update

    def record_telegram_command_response(
        self,
        *,
        update: TelegramUpdate,
        route: NotificationRoute,
        delivery: NotificationDelivery,
    ) -> tuple[TelegramUpdate, NotificationRoute, NotificationDelivery]:
        if (
            route.organization_id != update.organization_id
            or route.provider_instance_id != update.provider_instance_id
            or delivery.organization_id != update.organization_id
            or delivery.provider_instance_id != update.provider_instance_id
            or delivery.route_id != route.route_id
        ):
            raise ValueError("Telegram command response scope mismatch")
        existing = self.get_telegram_update(
            organization_id=update.organization_id,
            provider_instance_id=update.provider_instance_id,
            telegram_update_id=update.telegram_update_id,
        )
        if existing is not None:
            existing_delivery = self.get_delivery(
                organization_id=update.organization_id,
                delivery_id=delivery.delivery_id,
            )
            existing_route = self.get_route(
                organization_id=update.organization_id,
                route_id=route.route_id,
            )
            if existing_delivery is None or existing_route is None:
                raise ValueError("Telegram command response is partially persisted")
            return existing, existing_route, existing_delivery
        recorded_route = self.upsert_route(route=route)
        recorded_update = self.record_telegram_update(update=update)
        recorded_delivery = self.record_delivery(delivery=delivery)
        return recorded_update, recorded_route, recorded_delivery

    def record_report_run(self, *, report_run: NotificationReportRun) -> NotificationReportRun:
        existing = self.get_report_run_by_dedupe_key(
            organization_id=report_run.organization_id, dedupe_key=report_run.dedupe_key
        )
        if existing is not None:
            return existing
        self.report_runs[report_run.report_run_id] = report_run
        return report_run

    def get_report_run_by_dedupe_key(
        self, *, organization_id: OrganizationId, dedupe_key: str
    ) -> NotificationReportRun | None:
        for report_run in self.report_runs.values():
            if (
                report_run.organization_id == organization_id
                and report_run.dedupe_key == dedupe_key
            ):
                return report_run
        return None
