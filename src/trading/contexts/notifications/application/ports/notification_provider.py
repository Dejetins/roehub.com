from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from trading.contexts.notifications.domain import (
    NOTIFICATION_PROVIDER_ERROR_CODES,
    NotificationDelivery,
    NotificationProviderDescriptor,
    NotificationProviderHealth,
)
from trading.shared_kernel.primitives import OrganizationId

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

    def __post_init__(self) -> None:
        if (
            self.error_code is not None
            and self.error_code not in NOTIFICATION_PROVIDER_ERROR_CODES
        ):
            raise ValueError("notification provider result error code is not bounded")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ValueError(
                "notification provider retry_after_seconds must be non-negative"
            )


class NotificationProvider(Protocol):
    @property
    def provider_instance_id(self) -> UUID: ...

    @property
    def provider_key(self) -> str: ...

    @property
    def organization_id(self) -> OrganizationId | None: ...

    @property
    def descriptor(self) -> NotificationProviderDescriptor: ...

    def health(self) -> NotificationProviderHealth: ...

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult: ...
