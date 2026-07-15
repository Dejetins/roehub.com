from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderHealth,
)
from trading.shared_kernel.primitives import OrganizationId

LOG_ONLY_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")
FAKE_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000002")


@dataclass(frozen=True, slots=True)
class LogOnlyNotificationProvider:
    provider_key: str = "log_only"
    provider_instance_id: UUID = LOG_ONLY_PROVIDER_INSTANCE_ID
    organization_id: OrganizationId | None = None

    @property
    def descriptor(self) -> NotificationProviderDescriptor:
        return NotificationProviderDescriptor(
            provider_key=self.provider_key,
            display_name="Log only",
            package_version="1.0.0",
            config_schema={"type": "object", "additionalProperties": False},
            channels=("telegram", "email", "webhook", "push", "in_app"),
            templates=("plain_text.v1",),
            error_codes=("provider_disabled",),
        )

    def health(self) -> NotificationProviderHealth:
        return NotificationProviderHealth(
            instance_id=self.provider_instance_id,
            status="ready",
            checked_at=datetime.now(UTC),
        )

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        request_hash = _stable_delivery_hash(delivery=delivery, prefix="request")
        response_hash = _stable_delivery_hash(delivery=delivery, prefix="response")
        return NotificationProviderResult(
            status="sent",
            provider_message_id=f"{self.provider_key}:{delivery.delivery_id}",
            redacted_request_hash=request_hash,
            redacted_response_hash=response_hash,
        )


def _stable_delivery_hash(*, delivery: NotificationDelivery, prefix: str) -> str:
    payload = (
        f"{prefix}:{delivery.delivery_id}:{delivery.route_id}:"
        f"{delivery.provider_key}:{delivery.template_key}"
    )
    return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FakeNotificationProvider(LogOnlyNotificationProvider):
    provider_key: str = "fake"
    provider_instance_id: UUID = FAKE_PROVIDER_INSTANCE_ID
