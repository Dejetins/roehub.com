from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID

from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.domain import NotificationDelivery
from trading.shared_kernel.primitives import OrganizationId


@dataclass(frozen=True, slots=True)
class ReplayNotificationDeliveryCommand:
    organization_id: OrganizationId
    original_delivery_id: UUID
    replay_delivery_id: UUID

    def __post_init__(self) -> None:
        if self.original_delivery_id == self.replay_delivery_id:
            raise ValueError("notification replay requires a new delivery identity")


class NotificationDeliveryReplayService:
    """Create an explicit, durable replay without mutating the ambiguous delivery."""

    def __init__(self, *, repository: NotificationRepository) -> None:
        self._repository = repository

    def replay(
        self,
        *,
        command: ReplayNotificationDeliveryCommand,
        now: datetime,
    ) -> NotificationDelivery:
        existing = self._repository.get_delivery(
            organization_id=command.organization_id,
            delivery_id=command.replay_delivery_id,
        )
        if existing is not None:
            if existing.replayed_from_delivery_id != command.original_delivery_id:
                raise ValueError("notification replay identity is already in use")
            return existing

        original = self._repository.get_delivery(
            organization_id=command.organization_id,
            delivery_id=command.original_delivery_id,
        )
        if original is None:
            raise ValueError("notification replay source is unavailable")
        if original.status not in {"unknown", "dead_letter"}:
            raise ValueError("notification replay requires unknown or dead_letter source")

        replay = replace(
            original,
            delivery_id=command.replay_delivery_id,
            replayed_from_delivery_id=original.delivery_id,
            status="pending",
            attempt_count=0,
            next_attempt_at=None,
            lease_until=None,
            last_error_code=None,
            provider_message_id=None,
            sent_at=None,
            created_at=now,
        )
        recorded = self._repository.record_delivery(delivery=replay)
        if recorded.replayed_from_delivery_id != original.delivery_id:
            raise ValueError("notification replay lineage was not persisted")
        return recorded


__all__ = [
    "NotificationDeliveryReplayService",
    "ReplayNotificationDeliveryCommand",
]
