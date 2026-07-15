from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import PostgresNotificationRepository
from trading.contexts.notifications.domain import NotificationDelivery, NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")


class _Gateway:
    def __init__(self) -> None:
        self.routes: dict[str, Mapping[str, Any]] = {}
        self.deliveries: dict[str, Mapping[str, Any]] = {}
        self.fetch_all_queries: list[str] = []
        self.last_route_upsert_query = ""

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if "AS telegram_sent_last_24h" in query:
            return {
                "telegram_sent_total": 2,
                "telegram_sent_last_24h": 1,
                "last_telegram_sent_at": parameters["now"],
            }
        if "INSERT INTO notification_routes" in query:
            self.last_route_upsert_query = query
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
        self.fetch_all_queries.append(query)
        if "FROM notification_routes" in query:
            owner_user_id = parameters["owner_user_id"]
            recipient_kind = parameters["recipient_kind"]
            category = parameters["category"]
            return tuple(
                route
                for route in self.routes.values()
                if route["status"] == "active"
                and route["recipient_kind"] == recipient_kind
                and route["owner_user_id"] == owner_user_id
                and (
                    not route["category_filter"]
                    or category in route["category_filter"]
                )
            )
        raise AssertionError(query)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        raise AssertionError(query)


def test_postgres_notification_repository_maps_route_and_claims_delivery() -> None:
    gateway = _Gateway()
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime(2026, 6, 29, 16, 0, tzinfo=timezone.utc)
    route = repository.upsert_route(route=_route(now=now))
    stored_route = repository.get_route(
        organization_id=_ORGANIZATION_ID, route_id=route.route_id
    )
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
    assert "owner_user_id IS NOT DISTINCT FROM EXCLUDED.owner_user_id" in (
        gateway.last_route_upsert_query
    )


def test_postgres_notification_repository_lists_active_routes_with_typed_owner_filter() -> None:
    gateway = _Gateway()
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime(2026, 6, 29, 16, 0, tzinfo=timezone.utc)
    route = repository.upsert_route(route=_route(now=now))

    routes = repository.list_active_routes(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=route.owner_user_id,
        recipient_kind="user",
        category="strategy_signal",
    )

    assert routes == (route,)
    assert gateway.fetch_all_queries
    assert "%(owner_user_id)s::uuid" in gateway.fetch_all_queries[-1]


def test_postgres_notification_repository_reads_owner_scoped_delivery_counters() -> None:
    gateway = _Gateway()
    repository = PostgresNotificationRepository(gateway=gateway)
    now = datetime(2026, 6, 29, 16, 0, tzinfo=timezone.utc)

    counters = repository.get_delivery_counters(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=UserId(UUID("11111111-1111-4111-8111-111111111111")),
        now=now,
    )

    assert counters.telegram_sent_total == 2
    assert counters.telegram_sent_last_24h == 1
    assert counters.last_telegram_sent_at == now


def _route(*, now: datetime) -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
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
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
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
