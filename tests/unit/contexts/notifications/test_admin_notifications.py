from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
    LogOnlyNotificationProvider,
)
from trading.contexts.notifications.application import (
    NotificationAdminDrillService,
    NotificationDispatcher,
)
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")


def _now() -> datetime:
    return datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


@dataclass(slots=True)
class CapturingAdminMetrics:
    records: list[tuple[str, str, str]]

    def __init__(self) -> None:
        self.records = []

    def on_admin_notification(
        self, *, category: str, severity: str, status: str
    ) -> None:
        self.records.append((category, severity, status))


def test_admin_drill_creates_admin_only_deliveries_and_log_attempts() -> None:
    repository = InMemoryNotificationRepository()
    repository.upsert_route(route=_admin_route())
    repository.upsert_route(route=_user_route())
    metrics = CapturingAdminMetrics()
    service = NotificationAdminDrillService(
        repository=repository,
        clock=FixedClock(_now()),
        metrics=metrics,
    )

    drill = service.create_synthetic_admin_deliveries(
        organization_id=_ORGANIZATION_ID
    )
    dispatch = NotificationDispatcher(
        repository=repository,
        providers=(LogOnlyNotificationProvider(),),
        clock=FixedClock(_now()),
    ).drain_once()

    assert drill.events_created == 3
    assert drill.deliveries_created == 3
    assert drill.categories == ("admin_critical", "admin_alert", "admin_report")
    assert len(repository.events) == 3
    assert len(repository.deliveries) == 3
    assert dispatch.sent == 3
    assert len(repository.attempts) == 3
    assert all(event.recipient_kind == "admin" for event in repository.events.values())
    assert all(
        repository.routes[delivery.route_id].recipient_kind == "admin"
        for delivery in repository.deliveries.values()
    )
    assert metrics.records == [
        ("admin_critical", "critical", "pending"),
        ("admin_alert", "warning", "pending"),
        ("admin_report", "info", "pending"),
    ]


def _admin_route() -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="admin",
        owner_user_id=None,
        channel_key="telegram",
        provider_key="log_only",
        mode="all",
        category_filter=("admin_critical", "admin_alert", "admin_report"),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:admin:stage07",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )


def _user_route() -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="user",
        owner_user_id=UserId(UUID("11111111-1111-4111-8111-111111111111")),
        channel_key="telegram",
        provider_key="log_only",
        mode="all",
        category_filter=("admin_critical", "admin_alert", "admin_report"),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:user:stage07",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )
