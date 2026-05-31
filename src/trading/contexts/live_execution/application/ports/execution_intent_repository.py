from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import (
    ExecutionIntent,
    ExecutionRiskAuditEvent,
    ExecutionSourceEvent,
)
from trading.shared_kernel.primitives import UserId


class ExecutionIntentRepository(Protocol):
    def record_source_event(self, *, event: ExecutionSourceEvent) -> ExecutionSourceEvent: ...

    def get_source_event_by_id(
        self, *, owner_user_id: UserId, source_event_id: UUID
    ) -> ExecutionSourceEvent | None: ...

    def get_source_event_by_idempotency(
        self,
        *,
        owner_user_id: UserId,
        source_type: str,
        idempotency_key_hash: str,
    ) -> ExecutionSourceEvent | None: ...

    def update_source_event_outcome(
        self,
        *,
        owner_user_id: UserId,
        source_event_id: UUID,
        outcome: str,
        outcome_reason: str,
        intent_id: UUID | None,
    ) -> ExecutionSourceEvent | None: ...

    def record_intent(self, *, intent: ExecutionIntent) -> ExecutionIntent: ...

    def get_intent_by_id(
        self, *, owner_user_id: UserId, intent_id: UUID
    ) -> ExecutionIntent | None: ...

    def get_intent_by_idempotency(
        self, *, owner_user_id: UserId, idempotency_key_hash: str
    ) -> ExecutionIntent | None: ...

    def claim_intent_for_dispatch(
        self, *, intent_id: UUID, now: datetime, retry_budget: int
    ) -> ExecutionIntent | None: ...

    def mark_intent_dispatched(
        self,
        *,
        intent_id: UUID,
        stream_name: str,
        redis_message_id: str,
        now: datetime,
    ) -> ExecutionIntent | None: ...

    def mark_intent_dispatch_retry(
        self, *, intent_id: UUID, reason: str, now: datetime
    ) -> ExecutionIntent | None: ...

    def mark_intent_quarantined(
        self, *, intent_id: UUID, reason: str, stream_name: str | None, now: datetime
    ) -> ExecutionIntent | None: ...

    def record_risk_audit_event(
        self, *, event: ExecutionRiskAuditEvent
    ) -> ExecutionRiskAuditEvent: ...
