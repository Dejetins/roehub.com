from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from trading.contexts.live_execution.application.ports import ExecutionIntentRepository
from trading.contexts.live_execution.domain import ExecutionIntent, ExecutionSourceEvent
from trading.shared_kernel.primitives import UserId


class InMemoryExecutionIntentRepository(ExecutionIntentRepository):
    def __init__(self) -> None:
        self.source_events: list[ExecutionSourceEvent] = []
        self.intents: list[ExecutionIntent] = []

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
