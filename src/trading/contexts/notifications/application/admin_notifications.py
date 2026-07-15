from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.application.source_router import (
    NotificationSourceRouter,
    SyntheticNotificationSourceFact,
    decide_route,
)
from trading.contexts.notifications.domain import NotificationDelivery
from trading.shared_kernel.primitives import OrganizationId


class NotificationAdminClock(Protocol):
    def now(self) -> datetime: ...


class NotificationAdminMetrics(Protocol):
    def on_admin_notification(
        self, *, category: str, severity: str, status: str
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NotificationAdminDrillResult:
    events_created: int
    deliveries_created: int
    suppressed_routes: int
    categories: tuple[str, ...]


class NotificationAdminDrillService:
    def __init__(
        self,
        *,
        repository: NotificationRepository,
        clock: NotificationAdminClock,
        metrics: NotificationAdminMetrics | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._router = NotificationSourceRouter()
        self._metrics = metrics

    def create_synthetic_admin_deliveries(
        self, *, organization_id: OrganizationId
    ) -> NotificationAdminDrillResult:
        now = self._clock.now()
        events_created = 0
        deliveries_created = 0
        suppressed_routes = 0
        categories: list[str] = []

        for fact in synthetic_admin_notification_facts(
            organization_id=organization_id, now=now
        ):
            event = self._router.event_from_fact(fact=fact, now=now)
            stored_event = self._repository.record_event(event=event)
            events_created += 1
            categories.append(stored_event.category)
            routes = self._repository.list_active_routes(
                organization_id=organization_id,
                owner_user_id=None,
                recipient_kind="admin",
                category=stored_event.category,
            )
            for route in routes:
                decision, _reason = decide_route(event=stored_event, route=route)
                if decision != "deliver":
                    suppressed_routes += 1
                    continue
                delivery = NotificationDelivery(
                    delivery_id=uuid4(),
                    organization_id=organization_id,
                    provider_instance_id=route.provider_instance_id,
                    event_id=stored_event.event_id,
                    report_run_id=None,
                    command_id=None,
                    route_id=route.route_id,
                    provider_key=route.provider_key,
                    channel_key=route.channel_key,
                    recipient_address_ref=route.recipient_address_ref,
                    template_key=f"{stored_event.category}.v1",
                    rendered_payload_json={
                        "category": stored_event.category,
                        "severity": stored_event.severity,
                        "source_context": stored_event.source_context,
                        "runbook": "docs/runbooks/notifications-admin-alerts.md",
                    },
                    status="pending",
                    attempt_count=0,
                    created_at=now,
                )
                self._repository.record_delivery(delivery=delivery)
                deliveries_created += 1
                if self._metrics is not None:
                    self._metrics.on_admin_notification(
                        category=stored_event.category,
                        severity=stored_event.severity,
                        status="pending",
                    )

        return NotificationAdminDrillResult(
            events_created=events_created,
            deliveries_created=deliveries_created,
            suppressed_routes=suppressed_routes,
            categories=tuple(categories),
        )


def synthetic_admin_notification_facts(
    *, organization_id: OrganizationId, now: datetime
) -> tuple[SyntheticNotificationSourceFact, ...]:
    return (
        SyntheticNotificationSourceFact(
            fact_id="stage07-admin-critical",
            organization_id=organization_id,
            owner_user_id=None,
            recipient_kind="admin",
            source_context="ops",
            source_event_type="admin_critical",
            category="admin_critical",
            severity="critical",
            occurred_at=now,
            scope_json={"alert": "dispatcher_down"},
            payload_json={"runbook": "notifications-admin-alerts"},
        ),
        SyntheticNotificationSourceFact(
            fact_id="stage07-admin-alert",
            organization_id=organization_id,
            owner_user_id=None,
            recipient_kind="admin",
            source_context="ops",
            source_event_type="admin_alert",
            category="admin_alert",
            severity="warning",
            occurred_at=now,
            scope_json={"alert": "retry_rate"},
            payload_json={"runbook": "notifications-admin-alerts"},
        ),
        SyntheticNotificationSourceFact(
            fact_id="stage07-admin-report",
            organization_id=organization_id,
            owner_user_id=None,
            recipient_kind="admin",
            source_context="notifications",
            source_event_type="admin_report",
            category="admin_report",
            severity="info",
            occurred_at=now,
            scope_json={"period": "day"},
            payload_json={"runbook": "notifications-admin-alerts"},
        ),
    )
