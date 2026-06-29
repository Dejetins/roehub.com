from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import NotificationDelivery


@dataclass(frozen=True, slots=True)
class LogOnlyNotificationProvider:
    provider_key: str = "log_only"

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
