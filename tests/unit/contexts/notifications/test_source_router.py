from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.application import (
    NotificationSourceRouter,
    SyntheticNotificationSourceFact,
    synthetic_notification_matrix,
)
from trading.contexts.notifications.domain import NotificationRoute, NotificationValidationError
from trading.contexts.notifications.domain.notification import NotificationMode
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")


def _now() -> datetime:
    return datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)


def _user_id() -> UserId:
    return UserId(UUID("11111111-1111-4111-8111-111111111111"))


def _user_routes(owner_user_id: UserId) -> tuple[NotificationRoute, ...]:
    now = _now()
    return (
        _route(owner_user_id=owner_user_id, mode="critical_only", categories=()),
        _route(owner_user_id=owner_user_id, mode="signals", categories=("strategy_signal",)),
        _route(owner_user_id=owner_user_id, mode="trades", categories=()),
        _route(owner_user_id=owner_user_id, mode="reports", categories=("portfolio_report",)),
        _route(owner_user_id=owner_user_id, mode="all", categories=("stats_response",), now=now),
    )


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
        recipient_address_ref="telegram_ref:admin:stage02",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )


def _route(
    *,
    owner_user_id: UserId,
    mode: NotificationMode,
    categories: tuple[str, ...],
    now: datetime | None = None,
) -> NotificationRoute:
    timestamp = now or _now()
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="user",
        owner_user_id=owner_user_id,
        channel_key="telegram",
        provider_key="log_only",
        mode=mode,
        category_filter=categories,
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref=f"telegram_ref:user:{mode}:stage02",
        status="active",
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate() -> None:
    owner_user_id = _user_id()
    routes = (*_user_routes(owner_user_id), _admin_route())
    router = NotificationSourceRouter()
    repository = InMemoryNotificationRepository()
    delivered_categories: set[str] = set()

    for route in routes:
        repository.upsert_route(route=route)
    for fact in _matrix(owner_user_id):
        result = router.route(fact=fact, routes=routes, now=_now())
        repository.record_event(event=result.event)
        for delivery in result.deliveries:
            repository.record_delivery(delivery=delivery)
        for attempt in result.attempts:
            repository.record_delivery_attempt(attempt=attempt)
        assert result.decisions
        assert result.deliveries, fact.category
        assert len(result.deliveries) == len(result.attempts)
        delivered_categories.add(result.event.category)

    assert delivered_categories == {
        "strategy_run_failed",
        "strategy_signal",
        "trade_fill",
        "execution_rejected",
        "execution_terminal",
        "execution_unknown",
        "kill_switch",
        "portfolio_report",
        "stats_response",
        "admin_critical",
        "admin_alert",
        "admin_report",
    }
    assert len(repository.events) == 26
    assert len(repository.deliveries) >= 26
    assert len(repository.attempts) == len(repository.deliveries)
    assert {
        fact.source_event_type
        for fact in _matrix(owner_user_id)
    } >= {
        "producer_signal_rejected",
        "producer_order_rejected",
        "producer_manual_exit",
        "producer_reconciliation_pending",
        "producer_strategy_stopped",
        "producer_strategy_restarted",
        "producer_soak_failed",
        "producer_soak_succeeded",
        "producer_resource_threshold_breached",
    }


def test_router_proves_user_admin_route_separation() -> None:
    owner_user_id = _user_id()
    router = NotificationSourceRouter()
    user_fact = _matrix(owner_user_id)[0]
    admin_fact = _matrix(owner_user_id)[-1]
    routes = (_route(owner_user_id=owner_user_id, mode="all", categories=()), _admin_route())

    user_result = router.route(fact=user_fact, routes=routes, now=_now())
    admin_result = router.route(fact=admin_fact, routes=routes, now=_now())

    user_decisions = {
        decision.route.recipient_kind: decision.decision for decision in user_result.decisions
    }
    admin_decisions = {
        decision.route.recipient_kind: decision.decision for decision in admin_result.decisions
    }

    assert user_decisions == {
        "user": "deliver",
        "admin": "suppress",
    }
    assert admin_decisions == {
        "user": "suppress",
        "admin": "deliver",
    }


def test_router_applies_user_preference_modes() -> None:
    owner_user_id = _user_id()
    routes = (
        _route(owner_user_id=owner_user_id, mode="off", categories=()),
        _route(owner_user_id=owner_user_id, mode="critical_only", categories=()),
        _route(owner_user_id=owner_user_id, mode="signals", categories=()),
        _route(owner_user_id=owner_user_id, mode="trades", categories=()),
        _route(owner_user_id=owner_user_id, mode="reports", categories=()),
        _route(owner_user_id=owner_user_id, mode="all", categories=()),
    )
    facts_by_category = {
        fact.category: fact
        for fact in _matrix(owner_user_id)
        if fact.recipient_kind == "user"
    }
    router = NotificationSourceRouter()

    expected_modes_by_category = {
        "strategy_run_failed": {"critical_only", "all"},
        "strategy_signal": {"signals", "all"},
        "trade_fill": {"trades", "all"},
        "execution_rejected": {"critical_only", "trades", "all"},
        "execution_terminal": {"critical_only", "trades", "all"},
        "portfolio_report": {"reports", "all"},
        "stats_response": {"all"},
    }

    for category, expected_modes in expected_modes_by_category.items():
        result = router.route(fact=facts_by_category[category], routes=routes, now=_now())
        delivered_modes = {
            decision.route.mode for decision in result.decisions if decision.decision == "deliver"
        }
        assert delivered_modes == expected_modes

    critical_facts_by_category = {
        fact.category: fact
        for fact in _matrix(owner_user_id)
        if fact.category in {"execution_unknown", "kill_switch"}
    }
    for category, fact in critical_facts_by_category.items():
        result = router.route(fact=fact, routes=routes, now=_now())
        delivered_modes = {
            decision.route.mode for decision in result.decisions if decision.decision == "deliver"
        }
        assert delivered_modes == {"critical_only", "all"}, category


def test_router_rejects_secret_like_synthetic_payloads() -> None:
    router = NotificationSourceRouter()
    fact = SyntheticNotificationSourceFact(
        fact_id="strategy:signal:secret",
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_user_id(),
        recipient_kind="user",
        source_context="strategy",
        source_event_type="signal",
        category="strategy_signal",
        severity="info",
        occurred_at=_now(),
        scope_json={"strategy_id": str(uuid4())},
        payload_json={"bot_token": "redacted"},
    )

    with pytest.raises(NotificationValidationError, match="sensitive_notification_key_rejected"):
        router.event_from_fact(fact=fact, now=_now())


def _matrix(owner_user_id: UserId) -> tuple[SyntheticNotificationSourceFact, ...]:
    return synthetic_notification_matrix(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner_user_id,
        now=_now(),
    )
