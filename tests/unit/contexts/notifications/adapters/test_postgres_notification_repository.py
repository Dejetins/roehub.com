from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import PostgresNotificationRepository
from trading.contexts.notifications.domain import NotificationDelivery, NotificationRoute
from trading.shared_kernel.primitives import UserId


class _Gateway:
    def __init__(self) -> None:
        self.routes: dict[str, Mapping[str, Any]] = {}
        self.deliveries: dict[str, Mapping[str, Any]] = {}

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if "INSERT INTO notification_routes" in query:
            row = _decode_jsonb(parameters)
            self.routes[str(row["route_id"])] = row
            return row
        if "SELECT route_id" in query and "FROM notification_routes" in query:
            return self.routes.get(str(parameters["route_id"]))
        if "INSERT INTO notification_deliveries" in query:
            row = _decode_jsonb(parameters)
            self.deliveries[str(row["delivery_id"])] = row
            return row
        if "UPDATE notification_deliveries SET" in query and "status = 'claimed'" in query:
            row = self.deliveries.get(str(parameters["delivery_id"]))
            if row is None:
                return None
            claimed = dict(row)
            claimed["status"] = "claimed"
            claimed["attempt_count"] = int(claimed["attempt_count"]) + 1
            claimed["lease_until"] = parameters["lease_until"]
            self.deliveries[str(claimed["delivery_id"])] = claimed
            return claimed
        raise AssertionError(query)

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        raise AssertionError(query)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        raise AssertionError(query)


def test_postgres_notification_repository_maps_route_and_claims_delivery() -> None:
    gateway = _Gateway()
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime(2026, 6, 29, 16, 0, tzinfo=timezone.utc)
    route = repository.upsert_route(route=_route(now=now))
    stored_route = repository.get_route(route_id=route.route_id)
    delivery = repository.record_delivery(delivery=_delivery(route_id=route.route_id, now=now))

    claimed = repository.claim_delivery(
        delivery_id=delivery.delivery_id,
        lease_until=datetime(2026, 6, 29, 16, 1, tzinfo=timezone.utc),
        now=now,
    )

    assert stored_route == route
    assert claimed is not None
    assert claimed.status == "claimed"
    assert claimed.attempt_count == 1


def _route(*, now: datetime) -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        recipient_kind="user",
        owner_user_id=UserId(UUID("11111111-1111-4111-8111-111111111111")),
        channel_key="telegram",
        provider_key="log_only",
        mode="all",
        category_filter=("strategy_signal",),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:user:postgres",
        status="active",
        created_at=now,
        updated_at=now,
    )


def _delivery(*, route_id: UUID, now: datetime) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
        event_id=UUID("22222222-2222-4222-8222-222222222222"),
        report_run_id=None,
        command_id=None,
        route_id=route_id,
        provider_key="log_only",
        channel_key="telegram",
        recipient_address_ref="telegram_ref:user:postgres",
        template_key="strategy_signal.v1",
        rendered_payload_json={"text": "stage09"},
        status="pending",
        attempt_count=0,
        created_at=now,
    )


def _decode_jsonb(parameters: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(parameters)
    for key, value in tuple(row.items()):
        if hasattr(value, "obj"):
            row[key] = value.obj
    return row
