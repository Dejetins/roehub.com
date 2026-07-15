from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping
from uuid import UUID

from trading.contexts.live_execution.domain.execution_source import ExecutionSourceType
from trading.shared_kernel.primitives import OrganizationId, UserId

ExecutionNotificationEventType = Literal[
    "producer_rejected",
    "producer_signal_rejected",
    "producer_order_rejected",
    "producer_fill",
    "producer_manual_exit",
    "producer_unknown",
    "producer_reconciliation_pending",
    "producer_kill_switch",
    "producer_terminal",
    "producer_strategy_stopped",
    "producer_strategy_restarted",
    "producer_soak_failed",
    "producer_soak_succeeded",
    "producer_resource_threshold_breached",
]
ExecutionNotificationSeverity = Literal["info", "warning", "critical"]
ExecutionNotificationStatus = Literal["pending", "sent", "failed"]

SUPPORTED_NOTIFICATION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "producer_rejected",
        "producer_signal_rejected",
        "producer_order_rejected",
        "producer_fill",
        "producer_manual_exit",
        "producer_unknown",
        "producer_reconciliation_pending",
        "producer_kill_switch",
        "producer_terminal",
        "producer_strategy_stopped",
        "producer_strategy_restarted",
        "producer_soak_failed",
        "producer_soak_succeeded",
        "producer_resource_threshold_breached",
    }
)
SUPPORTED_NOTIFICATION_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
_SENSITIVE_LABEL_PARTS = frozenset(
    {"secret", "token", "authorization", "cookie", "api_key", "apikey", "signature", "passphrase"}
)


class ExecutionNotificationValidationError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionNotificationOutboxEvent:
    notification_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    source_type: ExecutionSourceType
    event_type: ExecutionNotificationEventType
    severity: ExecutionNotificationSeverity
    reason: str
    source_event_id: UUID | None
    intent_id: UUID | None
    order_id: UUID | None
    strategy_signal_id: UUID | None
    labels_json: Mapping[str, str]
    status: ExecutionNotificationStatus
    created_at: datetime
    sent_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.event_type not in SUPPORTED_NOTIFICATION_EVENT_TYPES:
            raise ExecutionNotificationValidationError(reason="unsupported_notification_event_type")
        if self.severity not in SUPPORTED_NOTIFICATION_SEVERITIES:
            raise ExecutionNotificationValidationError(reason="unsupported_notification_severity")
        if self.status not in {"pending", "sent", "failed"}:
            raise ExecutionNotificationValidationError(reason="unsupported_notification_status")
        if not self.reason.strip():
            raise ExecutionNotificationValidationError(reason="notification_reason_required")
        _validate_labels(self.labels_json)


def sanitize_notification_labels(labels: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        label_key = str(key).strip()
        if not label_key:
            raise ExecutionNotificationValidationError(reason="notification_label_key_required")
        lowered_key = label_key.lower()
        if any(part in lowered_key for part in _SENSITIVE_LABEL_PARTS):
            raise ExecutionNotificationValidationError(
                reason="sensitive_notification_label_rejected"
            )
        label_value = str(value).strip()
        if not label_value:
            continue
        normalized[label_key[:80]] = label_value[:160]
        if len(normalized) > 16:
            raise ExecutionNotificationValidationError(reason="notification_labels_too_large")
    return normalized


def _validate_labels(labels: Mapping[str, str]) -> None:
    if len(labels) > 16:
        raise ExecutionNotificationValidationError(reason="notification_labels_too_large")
    for key, value in labels.items():
        if not str(key).strip():
            raise ExecutionNotificationValidationError(reason="notification_label_key_required")
        if any(part in str(key).strip().lower() for part in _SENSITIVE_LABEL_PARTS):
            raise ExecutionNotificationValidationError(
                reason="sensitive_notification_label_rejected"
            )
        if not isinstance(value, str) or not value.strip():
            raise ExecutionNotificationValidationError(reason="notification_label_value_required")
