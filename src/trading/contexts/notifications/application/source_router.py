from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping
from uuid import UUID, uuid4

from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationRoute,
    build_notification_dedupe_key,
    sanitize_notification_mapping,
)
from trading.contexts.notifications.domain.notification import (
    NotificationCategory,
    NotificationRecipientKind,
    NotificationSeverity,
    NotificationSourceContext,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_SIGNAL_CATEGORIES = frozenset({"strategy_signal"})
_TRADE_CATEGORIES = frozenset({"trade_fill", "execution_rejected", "execution_terminal"})
_REPORT_CATEGORIES = frozenset({"portfolio_report", "admin_report"})
_CRITICAL_USER_CATEGORIES = frozenset(
    {
        "strategy_run_failed",
        "execution_rejected",
        "execution_terminal",
        "execution_unknown",
        "kill_switch",
    }
)
_ADMIN_CATEGORIES = frozenset({"admin_critical", "admin_alert", "admin_report"})


@dataclass(frozen=True, slots=True)
class SyntheticNotificationSourceFact:
    fact_id: str
    organization_id: OrganizationId
    owner_user_id: UserId | None
    recipient_kind: NotificationRecipientKind
    source_context: NotificationSourceContext
    source_event_type: str
    category: NotificationCategory
    severity: NotificationSeverity
    occurred_at: datetime
    scope_json: Mapping[str, object]
    payload_json: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NotificationRouteDecision:
    event: NotificationEvent
    route: NotificationRoute
    decision: str
    reason: str


@dataclass(frozen=True, slots=True)
class NotificationSyntheticFlowResult:
    event: NotificationEvent
    decisions: tuple[NotificationRouteDecision, ...]
    deliveries: tuple[NotificationDelivery, ...]
    attempts: tuple[NotificationDeliveryAttempt, ...]


class NotificationSourceRouter:
    def route(
        self,
        *,
        fact: SyntheticNotificationSourceFact,
        routes: tuple[NotificationRoute, ...],
        now: datetime,
    ) -> NotificationSyntheticFlowResult:
        event = self.event_from_fact(fact=fact, now=now)
        decisions: list[NotificationRouteDecision] = []
        deliveries: list[NotificationDelivery] = []
        attempts: list[NotificationDeliveryAttempt] = []

        for route in routes:
            decision, reason = decide_route(event=event, route=route)
            decisions.append(
                NotificationRouteDecision(
                    event=event,
                    route=route,
                    decision=decision,
                    reason=reason,
                )
            )
            if decision != "deliver":
                continue
            delivery = NotificationDelivery(
                delivery_id=uuid4(),
                organization_id=event.organization_id,
                provider_instance_id=route.provider_instance_id,
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
                },
                status="pending",
                attempt_count=0,
                created_at=now,
            )
            attempt = NotificationDeliveryAttempt(
                attempt_id=uuid4(),
                organization_id=event.organization_id,
                provider_instance_id=route.provider_instance_id,
                delivery_id=delivery.delivery_id,
                provider_key=route.provider_key,
                started_at=now,
                status="sent",
                finished_at=now,
                redacted_request_hash="a" * 64,
                redacted_response_hash="b" * 64,
            )
            deliveries.append(delivery)
            attempts.append(attempt)

        return NotificationSyntheticFlowResult(
            event=event,
            decisions=tuple(decisions),
            deliveries=tuple(deliveries),
            attempts=tuple(attempts),
        )

    def event_from_fact(
        self, *, fact: SyntheticNotificationSourceFact, now: datetime
    ) -> NotificationEvent:
        source_id = fact.fact_id.strip()
        dedupe_key = build_notification_dedupe_key(
            organization_id=fact.organization_id,
            source_context=fact.source_context,
            source_event_type=fact.source_event_type,
            source_id=source_id,
        )
        return NotificationEvent(
            event_id=uuid4(),
            organization_id=fact.organization_id,
            owner_user_id=fact.owner_user_id,
            recipient_kind=fact.recipient_kind,
            source_context=fact.source_context,
            source_event_type=fact.source_event_type,
            category=fact.category,
            severity=fact.severity,
            scope_json=sanitize_notification_mapping(fact.scope_json),
            payload_json=sanitize_notification_mapping(fact.payload_json),
            dedupe_key=dedupe_key,
            occurred_at=fact.occurred_at,
            created_at=now,
        )


def decide_route(*, event: NotificationEvent, route: NotificationRoute) -> tuple[str, str]:
    if event.organization_id != route.organization_id:
        return "suppress", "organization_mismatch"
    if route.status != "active":
        return "suppress", "route_not_active"
    if route.provider_key not in {"fake", "log_only"}:
        return "suppress", "provider_not_synthetic"
    if not _recipient_matches(event=event, route=route):
        return "suppress", "recipient_mismatch"
    if route.category_filter and event.category not in route.category_filter:
        return "suppress", "category_filtered"
    if not _mode_allows(route=route, event=event):
        return "suppress", "mode_filtered"
    return "deliver", "matched"


def _recipient_matches(*, event: NotificationEvent, route: NotificationRoute) -> bool:
    if event.recipient_kind == "both":
        return True
    return event.recipient_kind == route.recipient_kind


def _mode_allows(*, route: NotificationRoute, event: NotificationEvent) -> bool:
    if route.recipient_kind == "admin":
        return event.category in _ADMIN_CATEGORIES and route.mode in {"critical_only", "all"}
    if route.mode == "off":
        return False
    if route.mode == "all":
        return True
    if route.mode == "critical_only":
        return event.severity == "critical" or event.category in _CRITICAL_USER_CATEGORIES
    if route.mode == "signals":
        return event.category in _SIGNAL_CATEGORIES
    if route.mode == "trades":
        return event.category in _TRADE_CATEGORIES
    if route.mode == "reports":
        return event.category in _REPORT_CATEGORIES
    return False


def synthetic_notification_matrix(
    *, organization_id: OrganizationId, owner_user_id: UserId, now: datetime
) -> tuple[SyntheticNotificationSourceFact, ...]:
    def _fact(*args: object) -> SyntheticNotificationSourceFact:
        return _scoped_fact(organization_id, *args)  # type: ignore[arg-type]

    strategy_id = UUID("22222222-2222-4222-8222-222222222222")
    exchange_connection_id = UUID("33333333-3333-4333-8333-333333333333")
    base_scope = {
        "strategy_id": str(strategy_id),
        "exchange_connection_id": str(exchange_connection_id),
        "market_type": "futures",
        "symbol": "BTCUSDT",
    }
    return (
        _fact(
            owner_user_id,
            "user",
            "strategy",
            "failed",
            "strategy_run_failed",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "strategy",
            "signal",
            "strategy_signal",
            "info",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_fill",
            "trade_fill",
            "info",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_rejected",
            "execution_rejected",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_signal_rejected",
            "execution_rejected",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_order_rejected",
            "execution_rejected",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_terminal",
            "execution_terminal",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_manual_exit",
            "execution_terminal",
            "info",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "both",
            "live_execution",
            "producer_reconciliation_pending",
            "execution_unknown",
            "critical",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_strategy_stopped",
            "execution_terminal",
            "warning",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "user",
            "live_execution",
            "producer_strategy_restarted",
            "execution_terminal",
            "info",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "both",
            "live_execution",
            "producer_unknown",
            "execution_unknown",
            "critical",
            now,
            base_scope,
        ),
        _fact(
            owner_user_id,
            "both",
            "live_execution",
            "producer_kill_switch",
            "kill_switch",
            "critical",
            now,
            base_scope,
        ),
        _fact(
            None,
            "admin",
            "live_execution",
            "producer_soak_failed",
            "admin_critical",
            "critical",
            now,
            {"stage": "12.4", "window": "6h"},
        ),
        _fact(
            None,
            "admin",
            "live_execution",
            "producer_soak_succeeded",
            "admin_report",
            "info",
            now,
            {"stage": "12.4", "window": "6h"},
        ),
        _fact(
            None,
            "admin",
            "live_execution",
            "producer_resource_threshold_breached",
            "admin_alert",
            "warning",
            now,
            {"stage": "12.3", "surface": "resource"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "portfolio_weekly",
            "portfolio_report",
            "info",
            now,
            {"period": "week"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "portfolio_monthly",
            "portfolio_report",
            "info",
            now,
            {"period": "month"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "stats_today",
            "stats_response",
            "info",
            now,
            {"period": "today"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "stats_week",
            "stats_response",
            "info",
            now,
            {"period": "week"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "stats_month",
            "stats_response",
            "info",
            now,
            {"period": "month"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "strategy_stats_week",
            "stats_response",
            "info",
            now,
            {"period": "week", "scope": "strategy"},
        ),
        _fact(
            owner_user_id,
            "user",
            "notifications",
            "exchange_stats_month",
            "stats_response",
            "info",
            now,
            {"period": "month", "scope": "exchange"},
        ),
        _fact(
            None,
            "admin",
            "ops",
            "admin_critical",
            "admin_critical",
            "critical",
            now,
            {"alert": "dispatcher_down"},
        ),
        _fact(
            None,
            "admin",
            "ops",
            "admin_alert",
            "admin_alert",
            "warning",
            now,
            {"alert": "retry_rate"},
        ),
        _fact(
            None,
            "admin",
            "notifications",
            "admin_report",
            "admin_report",
            "info",
            now,
            {"period": "day"},
        ),
    )


def _scoped_fact(
    organization_id: OrganizationId,
    owner_user_id: UserId | None,
    recipient_kind: NotificationRecipientKind,
    source_context: NotificationSourceContext,
    source_event_type: str,
    category: NotificationCategory,
    severity: NotificationSeverity,
    now: datetime,
    scope_json: Mapping[str, object],
) -> SyntheticNotificationSourceFact:
    fact_id = f"{source_context}:{source_event_type}:{recipient_kind}"
    return SyntheticNotificationSourceFact(
        fact_id=fact_id,
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        recipient_kind=recipient_kind,
        source_context=source_context,
        source_event_type=source_event_type,
        category=category,
        severity=severity,
        occurred_at=now,
        scope_json=scope_json,
        payload_json={"summary": f"synthetic {category}"},
    )
