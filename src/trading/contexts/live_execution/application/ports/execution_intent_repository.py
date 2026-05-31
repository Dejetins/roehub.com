from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import ExecutionIntent, ExecutionSourceEvent
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

    def get_intent_by_idempotency(
        self, *, owner_user_id: UserId, idempotency_key_hash: str
    ) -> ExecutionIntent | None: ...
