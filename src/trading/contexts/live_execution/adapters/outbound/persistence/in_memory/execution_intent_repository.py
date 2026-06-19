from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID

from trading.contexts.live_execution.application.ports import ExecutionIntentRepository
from trading.contexts.live_execution.domain import (
    ExecutionIntent,
    ExecutionNotificationOutboxEvent,
    ExecutionProducerOutcomeLink,
    ExecutionRiskAuditEvent,
    ExecutionSourceEvent,
)
from trading.shared_kernel.primitives import UserId


class InMemoryExecutionIntentRepository(ExecutionIntentRepository):
    def __init__(self) -> None:
        self.source_events: list[ExecutionSourceEvent] = []
        self.intents: list[ExecutionIntent] = []
        self.risk_audit_events: list[ExecutionRiskAuditEvent] = []
        self.notifications: list[ExecutionNotificationOutboxEvent] = []

    def record_source_event(self, *, event: ExecutionSourceEvent) -> ExecutionSourceEvent:
        existing = self.get_source_event_by_idempotency(
            owner_user_id=event.owner_user_id,
            source_type=event.source_type,
            idempotency_key_hash=event.idempotency_key_hash,
        )
        if existing is not None:
            return existing
        self.source_events.append(event)
        return event

    def get_source_event_by_id(
        self, *, owner_user_id: UserId, source_event_id: UUID
    ) -> ExecutionSourceEvent | None:
        return next(
            (
                item
                for item in self.source_events
                if item.owner_user_id == owner_user_id and item.source_event_id == source_event_id
            ),
            None,
        )

    def get_source_event_by_idempotency(
        self,
        *,
        owner_user_id: UserId,
        source_type: str,
        idempotency_key_hash: str,
    ) -> ExecutionSourceEvent | None:
        return next(
            (
                item
                for item in self.source_events
                if item.owner_user_id == owner_user_id
                and item.source_type == source_type
                and item.idempotency_key_hash == idempotency_key_hash
            ),
            None,
        )

    def update_source_event_outcome(
        self,
        *,
        owner_user_id: UserId,
        source_event_id: UUID,
        outcome: str,
        outcome_reason: str,
        intent_id: UUID | None,
    ) -> ExecutionSourceEvent | None:
        for index, item in enumerate(self.source_events):
            if item.owner_user_id == owner_user_id and item.source_event_id == source_event_id:
                updated = replace(
                    item,
                    outcome=outcome,
                    outcome_reason=outcome_reason,
                    intent_id=intent_id,
                )
                self.source_events[index] = updated
                return updated
        return None

    def record_intent(self, *, intent: ExecutionIntent) -> ExecutionIntent:
        existing = self.get_intent_by_idempotency(
            owner_user_id=intent.owner_user_id,
            idempotency_key_hash=intent.idempotency_key_hash,
        )
        if existing is not None:
            return existing
        self.intents.append(intent)
        return intent

    def get_intent_by_idempotency(
        self, *, owner_user_id: UserId, idempotency_key_hash: str
    ) -> ExecutionIntent | None:
        return next(
            (
                item
                for item in self.intents
                if item.owner_user_id == owner_user_id
                and item.idempotency_key_hash == idempotency_key_hash
            ),
            None,
        )

    def get_intent_by_id(
        self, *, owner_user_id: UserId, intent_id: UUID
    ) -> ExecutionIntent | None:
        return next(
            (
                item
                for item in self.intents
                if item.owner_user_id == owner_user_id and item.intent_id == intent_id
            ),
            None,
        )

    def claim_intent_for_dispatch(
        self, *, intent_id: UUID, now: datetime, retry_budget: int
    ) -> ExecutionIntent | None:
        for index, item in enumerate(self.intents):
            if (
                item.intent_id == intent_id
                and item.status in {"accepted", "retry"}
                and item.risk_status == "accepted"
                and item.dispatch_attempt_count < retry_budget
            ):
                updated = replace(
                    item,
                    status="dispatching",
                    status_reason="dispatch_publish_pending",
                    dispatch_attempt_count=item.dispatch_attempt_count + 1,
                    dispatch_last_error=None,
                    dispatch_updated_at=now,
                )
                self.intents[index] = updated
                return updated
        return None

    def mark_intent_dispatched(
        self,
        *,
        intent_id: UUID,
        stream_name: str,
        redis_message_id: str,
        now: datetime,
    ) -> ExecutionIntent | None:
        return self._replace_intent(
            intent_id=intent_id,
            status="dispatched",
            status_reason="redis_xadd_ok",
            dispatch_stream_name=stream_name,
            dispatch_redis_message_id=redis_message_id,
            dispatch_last_error=None,
            dispatch_updated_at=now,
        )

    def mark_intent_dispatch_retry(
        self, *, intent_id: UUID, reason: str, now: datetime
    ) -> ExecutionIntent | None:
        return self._replace_intent(
            intent_id=intent_id,
            status="retry",
            status_reason=reason,
            dispatch_last_error=reason,
            dispatch_updated_at=now,
        )

    def mark_intent_quarantined(
        self, *, intent_id: UUID, reason: str, stream_name: str | None, now: datetime
    ) -> ExecutionIntent | None:
        return self._replace_intent(
            intent_id=intent_id,
            status="quarantined",
            status_reason=reason,
            dispatch_stream_name=stream_name,
            dispatch_last_error=reason,
            dispatch_updated_at=now,
        )

    def record_risk_audit_event(
        self, *, event: ExecutionRiskAuditEvent
    ) -> ExecutionRiskAuditEvent:
        self.risk_audit_events.append(event)
        return event

    def record_notification_outbox(
        self, *, event: ExecutionNotificationOutboxEvent
    ) -> ExecutionNotificationOutboxEvent:
        existing = next(
            (
                item
                for item in self.notifications
                if item.owner_user_id == event.owner_user_id
                and item.event_type == event.event_type
                and item.source_event_id == event.source_event_id
                and item.intent_id == event.intent_id
                and item.order_id == event.order_id
                and item.reason == event.reason
            ),
            None,
        )
        if existing is not None:
            return existing
        self.notifications.append(event)
        return event

    def list_recent_notifications(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
        strategy_id: UUID | None = None,
    ) -> tuple[ExecutionNotificationOutboxEvent, ...]:
        items = [item for item in self.notifications if item.owner_user_id == owner_user_id]
        if strategy_id is not None:
            items = [
                item
                for item in items
                if self._notification_matches_strategy(item=item, strategy_id=strategy_id)
            ]
        return tuple(sorted(items, key=lambda item: item.created_at, reverse=True)[:limit])

    def list_producer_outcome_links_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID, limit: int
    ) -> tuple[ExecutionProducerOutcomeLink, ...]:
        rows: list[ExecutionProducerOutcomeLink] = []
        for event in self.source_events:
            if event.owner_user_id != owner_user_id:
                continue
            if event.source_ref_json.get("strategy_id") != str(strategy_id):
                continue
            intent = (
                self.get_intent_by_id(owner_user_id=owner_user_id, intent_id=event.intent_id)
                if event.intent_id is not None
                else None
            )
            notification = next(
                (
                    item
                    for item in sorted(
                        self.notifications,
                        key=lambda notification: notification.created_at,
                        reverse=True,
                    )
                    if item.source_event_id == event.source_event_id
                ),
                None,
            )
            rows.append(
                ExecutionProducerOutcomeLink(
                    source_event_id=event.source_event_id,
                    owner_user_id=event.owner_user_id,
                    source_type=event.source_type,
                    source_event_ref=event.source_event_ref,
                    source_event_received_at=event.received_at,
                    strategy_signal_id=event.strategy_signal_id,
                    outcome=event.outcome,
                    outcome_reason=event.outcome_reason,
                    intent_id=event.intent_id,
                    intent_status=intent.status if intent is not None else None,
                    intent_status_reason=intent.status_reason if intent is not None else None,
                    risk_status=intent.risk_status if intent is not None else None,
                    risk_reason=intent.risk_reason if intent is not None else None,
                    order_status=None,
                    order_status_reason=None,
                    fill_count=None,
                    latest_fill_at=None,
                    reconciliation_status=None,
                    reconciliation_reason=None,
                    notification_event_type=(
                        notification.event_type if notification is not None else None
                    ),
                    notification_reason=notification.reason if notification is not None else None,
                    updated_at=event.received_at,
                )
            )
        return tuple(sorted(rows, key=lambda item: item.updated_at, reverse=True)[:limit])

    def _replace_intent(self, *, intent_id: UUID, **updates: object) -> ExecutionIntent | None:
        for index, item in enumerate(self.intents):
            if item.intent_id == intent_id:
                updated = replace(item, **updates)
                self.intents[index] = updated
                return updated
        return None

    def _notification_matches_strategy(
        self, *, item: ExecutionNotificationOutboxEvent, strategy_id: UUID
    ) -> bool:
        if item.source_event_id is None:
            return False
        event = self.get_source_event_by_id(
            owner_user_id=item.owner_user_id,
            source_event_id=item.source_event_id,
        )
        return event is not None and event.source_ref_json.get("strategy_id") == str(strategy_id)
