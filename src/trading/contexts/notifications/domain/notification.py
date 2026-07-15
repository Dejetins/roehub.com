from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Literal, Mapping, TypeAlias
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

NotificationRecipientKind = Literal["user", "admin", "both"]
NotificationRouteRecipientKind = Literal["user", "admin"]
NotificationSeverity = Literal["info", "warning", "critical"]
NotificationCategory = Literal[
    "strategy_run_failed",
    "strategy_signal",
    "trade_fill",
    "execution_rejected",
    "execution_terminal",
    "execution_unknown",
    "kill_switch",
    "portfolio_report",
    "stats_response",
    "system_alert",
    "admin_critical",
    "admin_alert",
    "admin_report",
]
NotificationSourceContext = Literal[
    "strategy",
    "live_execution",
    "rl_trading",
    "market_data",
    "ops",
    "identity",
    "notifications",
]
NotificationChannelKey = Literal["telegram", "email", "webhook", "push", "in_app"]
NotificationProviderKey: TypeAlias = str
NotificationMode = Literal["off", "critical_only", "trades", "signals", "reports", "all"]
NotificationRouteStatus = Literal["active", "paused", "requires_rebind", "disabled"]
NotificationDeliveryStatus = Literal[
    "pending",
    "claimed",
    "sent",
    "failed",
    "retry",
    "dead_letter",
    "suppressed",
    "unknown",
]
TelegramUpdateStatus = Literal["pending", "handled", "ignored", "failed", "dead_letter"]
NotificationReportType = Literal["portfolio_weekly", "portfolio_monthly", "stats_on_demand"]
NotificationReportQualityStatus = Literal["complete", "partial", "unavailable"]
NotificationReportStatus = Literal["pending", "rendered", "sent", "failed", "suppressed"]

SUPPORTED_RECIPIENT_KINDS: frozenset[str] = frozenset({"user", "admin", "both"})
SUPPORTED_ROUTE_RECIPIENT_KINDS: frozenset[str] = frozenset({"user", "admin"})
SUPPORTED_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
SUPPORTED_CATEGORIES: frozenset[str] = frozenset(
    {
        "strategy_run_failed",
        "strategy_signal",
        "trade_fill",
        "execution_rejected",
        "execution_terminal",
        "execution_unknown",
        "kill_switch",
        "portfolio_report",
        "stats_response",
        "system_alert",
        "admin_critical",
        "admin_alert",
        "admin_report",
    }
)
SUPPORTED_SOURCE_CONTEXTS: frozenset[str] = frozenset(
    {"strategy", "live_execution", "rl_trading", "market_data", "ops", "identity", "notifications"}
)
SUPPORTED_CHANNEL_KEYS: frozenset[str] = frozenset(
    {"telegram", "email", "webhook", "push", "in_app"}
)
SUPPORTED_MODES: frozenset[str] = frozenset(
    {"off", "critical_only", "trades", "signals", "reports", "all"}
)
SUPPORTED_ROUTE_STATUSES: frozenset[str] = frozenset(
    {"active", "paused", "requires_rebind", "disabled"}
)
SUPPORTED_DELIVERY_STATUSES: frozenset[str] = frozenset(
    {"pending", "claimed", "sent", "failed", "retry", "dead_letter", "suppressed", "unknown"}
)
SUPPORTED_TELEGRAM_UPDATE_STATUSES: frozenset[str] = frozenset(
    {"pending", "handled", "ignored", "failed", "dead_letter"}
)
SUPPORTED_REPORT_TYPES: frozenset[str] = frozenset(
    {"portfolio_weekly", "portfolio_monthly", "stats_on_demand"}
)
SUPPORTED_REPORT_QUALITY_STATUSES: frozenset[str] = frozenset(
    {"complete", "partial", "unavailable"}
)
SUPPORTED_REPORT_STATUSES: frozenset[str] = frozenset(
    {"pending", "rendered", "sent", "failed", "suppressed"}
)
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "secret",
        "token",
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "signature",
        "passphrase",
        "password",
        "chat_id",
        "chatid",
    }
)
_HASH_HEX_CHARS = frozenset("0123456789abcdef")


class NotificationValidationError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    event_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId | None
    recipient_kind: NotificationRecipientKind
    source_context: NotificationSourceContext
    source_event_type: str
    category: NotificationCategory
    severity: NotificationSeverity
    scope_json: Mapping[str, object]
    payload_json: Mapping[str, object]
    dedupe_key: str
    occurred_at: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        _require_supported(self.recipient_kind, SUPPORTED_RECIPIENT_KINDS, "recipient_kind")
        if self.recipient_kind in {"user", "both"} and self.owner_user_id is None:
            raise NotificationValidationError(reason="owner_user_id_required_for_user_event")
        if self.recipient_kind == "admin" and self.owner_user_id is not None:
            raise NotificationValidationError(reason="admin_event_must_not_have_owner_user_id")
        _require_supported(self.source_context, SUPPORTED_SOURCE_CONTEXTS, "source_context")
        _require_non_empty_text(self.source_event_type, "source_event_type")
        _require_supported(self.category, SUPPORTED_CATEGORIES, "category")
        _require_supported(self.severity, SUPPORTED_SEVERITIES, "severity")
        _validate_mapping(self.scope_json, "scope_json")
        _validate_mapping(self.payload_json, "payload_json")
        _require_dedupe_key(self.dedupe_key)


@dataclass(frozen=True, slots=True)
class NotificationRoute:
    route_id: UUID
    organization_id: OrganizationId
    provider_instance_id: UUID
    recipient_kind: NotificationRouteRecipientKind
    owner_user_id: UserId | None
    channel_key: NotificationChannelKey
    provider_key: NotificationProviderKey
    mode: NotificationMode
    category_filter: tuple[str, ...]
    scope_filter_json: Mapping[str, object]
    schedule_json: Mapping[str, object]
    recipient_address_ref: str
    status: NotificationRouteStatus
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_supported(self.recipient_kind, SUPPORTED_ROUTE_RECIPIENT_KINDS, "recipient_kind")
        if self.recipient_kind == "user" and self.owner_user_id is None:
            raise NotificationValidationError(reason="user_route_requires_owner_user_id")
        if self.recipient_kind == "admin" and self.owner_user_id is not None:
            raise NotificationValidationError(reason="admin_route_must_not_have_owner_user_id")
        _require_supported(self.channel_key, SUPPORTED_CHANNEL_KEYS, "channel_key")
        _require_provider_key(self.provider_key)
        _require_supported(self.mode, SUPPORTED_MODES, "mode")
        _require_supported(self.status, SUPPORTED_ROUTE_STATUSES, "status")
        for category in self.category_filter:
            _require_supported(category, SUPPORTED_CATEGORIES, "category_filter")
        _validate_mapping(self.scope_filter_json, "scope_filter_json")
        _validate_mapping(self.schedule_json, "schedule_json")
        _require_redacted_ref(self.recipient_address_ref, "recipient_address_ref")


@dataclass(frozen=True, slots=True)
class NotificationDelivery:
    delivery_id: UUID
    organization_id: OrganizationId
    provider_instance_id: UUID
    event_id: UUID | None
    report_run_id: UUID | None
    command_id: UUID | None
    route_id: UUID
    provider_key: NotificationProviderKey
    channel_key: NotificationChannelKey
    recipient_address_ref: str
    template_key: str
    rendered_payload_json: Mapping[str, object]
    status: NotificationDeliveryStatus
    attempt_count: int
    created_at: datetime
    next_attempt_at: datetime | None = None
    lease_until: datetime | None = None
    last_error_code: str | None = None
    provider_message_id: str | None = None
    sent_at: datetime | None = None
    replayed_from_delivery_id: UUID | None = None

    def __post_init__(self) -> None:
        source_ref_count = sum(
            ref is not None for ref in (self.event_id, self.report_run_id, self.command_id)
        )
        if source_ref_count != 1:
            raise NotificationValidationError(reason="delivery_requires_exactly_one_source_ref")
        _require_provider_key(self.provider_key)
        _require_supported(self.channel_key, SUPPORTED_CHANNEL_KEYS, "channel_key")
        _require_redacted_ref(self.recipient_address_ref, "recipient_address_ref")
        _require_non_empty_text(self.template_key, "template_key")
        _validate_mapping(self.rendered_payload_json, "rendered_payload_json")
        _require_supported(self.status, SUPPORTED_DELIVERY_STATUSES, "status")
        if self.attempt_count < 0:
            raise NotificationValidationError(reason="attempt_count_must_be_non_negative")
        if self.replayed_from_delivery_id == self.delivery_id:
            raise NotificationValidationError(reason="delivery_cannot_replay_itself")


@dataclass(frozen=True, slots=True)
class NotificationDeliveryAttempt:
    attempt_id: UUID
    organization_id: OrganizationId
    provider_instance_id: UUID
    delivery_id: UUID
    provider_key: NotificationProviderKey
    started_at: datetime
    status: NotificationDeliveryStatus
    finished_at: datetime | None = None
    http_status: int | None = None
    error_code: str | None = None
    retry_after_seconds: int | None = None
    redacted_request_hash: str | None = None
    redacted_response_hash: str | None = None

    def __post_init__(self) -> None:
        _require_provider_key(self.provider_key)
        _require_supported(self.status, SUPPORTED_DELIVERY_STATUSES, "status")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise NotificationValidationError(reason="http_status_out_of_range")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise NotificationValidationError(reason="retry_after_seconds_must_be_non_negative")
        _require_redacted_hash(self.redacted_request_hash, "redacted_request_hash")
        _require_redacted_hash(self.redacted_response_hash, "redacted_response_hash")


@dataclass(frozen=True, slots=True)
class TelegramUpdate:
    organization_id: OrganizationId
    provider_instance_id: UUID
    telegram_update_id: int
    received_at: datetime
    chat_id_ref: str
    owner_user_id: UserId | None
    command_name: str | None
    command_args_json: Mapping[str, object]
    status: TelegramUpdateStatus
    idempotency_key: str
    created_at: datetime
    handled_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.telegram_update_id < 0:
            raise NotificationValidationError(reason="telegram_update_id_must_be_non_negative")
        _require_redacted_ref(self.chat_id_ref, "chat_id_ref")
        if self.command_name is not None:
            _require_non_empty_text(self.command_name, "command_name")
        _validate_mapping(self.command_args_json, "command_args_json")
        _require_supported(self.status, SUPPORTED_TELEGRAM_UPDATE_STATUSES, "status")
        _require_dedupe_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class NotificationReportRun:
    report_run_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    report_type: NotificationReportType
    period_start: datetime
    period_end: datetime
    scope_json: Mapping[str, object]
    quality_status: NotificationReportQualityStatus
    status: NotificationReportStatus
    dedupe_key: str
    created_at: datetime
    rendered_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_supported(self.report_type, SUPPORTED_REPORT_TYPES, "report_type")
        if self.period_start >= self.period_end:
            raise NotificationValidationError(reason="report_period_must_be_non_empty")
        _validate_mapping(self.scope_json, "scope_json")
        _require_supported(
            self.quality_status, SUPPORTED_REPORT_QUALITY_STATUSES, "quality_status"
        )
        _require_supported(self.status, SUPPORTED_REPORT_STATUSES, "status")
        _require_dedupe_key(self.dedupe_key)


def build_notification_dedupe_key(
    *,
    organization_id: OrganizationId,
    source_context: str,
    source_event_type: str,
    source_id: str,
) -> str:
    _require_non_empty_text(source_context, "source_context")
    _require_non_empty_text(source_event_type, "source_event_type")
    _require_non_empty_text(source_id, "source_id")
    digest = sha256(
        f"{organization_id}:{source_context}:{source_event_type}:{source_id}".encode()
    ).hexdigest()
    return f"{source_context}:{source_event_type}:{digest}"


def sanitize_notification_mapping(
    values: Mapping[str, object], *, max_items: int = 32
) -> dict[str, object]:
    if len(values) > max_items:
        raise NotificationValidationError(reason="notification_mapping_too_large")
    sanitized: dict[str, object] = {}
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if not key:
            raise NotificationValidationError(reason="notification_mapping_key_required")
        _reject_sensitive_key(key)
        sanitized[key[:96]] = _sanitize_value(value)
    return sanitized


def _validate_mapping(values: Mapping[str, object], field: str) -> None:
    try:
        sanitize_notification_mapping(values)
    except NotificationValidationError as exc:
        raise NotificationValidationError(reason=f"{field}_{exc.reason}") from exc


def _sanitize_value(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        _reject_sensitive_value(text)
        return text[:512]
    if isinstance(value, Mapping):
        return sanitize_notification_mapping(value, max_items=16)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, UUID):
        return str(value)
    raise NotificationValidationError(reason="unsupported_notification_mapping_value")


def _require_supported(value: str, supported: frozenset[str], field: str) -> None:
    if value not in supported:
        raise NotificationValidationError(reason=f"unsupported_{field}")


def _require_non_empty_text(value: str, field: str) -> None:
    if not value.strip():
        raise NotificationValidationError(reason=f"{field}_required")


def _require_provider_key(value: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9._-]{2,127}", value):
        raise NotificationValidationError(reason="unsupported_provider_key")


def _require_dedupe_key(value: str) -> None:
    text = value.strip()
    if not 16 <= len(text) <= 240:
        raise NotificationValidationError(reason="dedupe_key_invalid_length")
    _reject_sensitive_value(text)


def _require_redacted_ref(value: str, field: str) -> None:
    text = value.strip()
    if not 8 <= len(text) <= 180:
        raise NotificationValidationError(reason=f"{field}_invalid_length")
    _reject_sensitive_value(text)


def _require_redacted_hash(value: str | None, field: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(char not in _HASH_HEX_CHARS for char in value):
        raise NotificationValidationError(reason=f"{field}_must_be_sha256_hex")


def _reject_sensitive_key(key: str) -> None:
    lowered = key.casefold()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        raise NotificationValidationError(reason="sensitive_notification_key_rejected")


def _reject_sensitive_value(value: str) -> None:
    lowered = value.casefold()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        raise NotificationValidationError(reason="sensitive_notification_value_rejected")
