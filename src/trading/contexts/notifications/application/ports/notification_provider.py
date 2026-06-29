from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from trading.contexts.notifications.domain import NotificationDelivery

NotificationProviderResultStatus = Literal[
    "sent", "retry", "unknown", "dead_letter", "suppressed"
]


@dataclass(frozen=True, slots=True)
class NotificationProviderResult:
    status: NotificationProviderResultStatus
    error_code: str | None = None
    provider_message_id: str | None = None
    retry_after_seconds: int | None = None
    redacted_request_hash: str | None = None
    redacted_response_hash: str | None = None


class NotificationProvider(Protocol):
    @property
    def provider_key(self) -> str: ...

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult: ...
