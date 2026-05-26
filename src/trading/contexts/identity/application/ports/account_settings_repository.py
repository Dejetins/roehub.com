from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

ThemePreference = Literal["terminal-orange", "graphite", "matrix-green", "high-contrast"]
LocalePreference = Literal["en", "ru"]
DensityPreference = Literal["compact", "comfortable"]
AutorefreshPreset = Literal["off", "10s", "15s", "30s", "1m", "5m", "custom"]
NotificationMode = Literal["off", "on", "critical"]
IntegrationMode = Literal["off", "alerts", "critical"]
AuditEventType = Literal[
    "profile_updated",
    "preferences_updated",
    "integration_updated",
    "notifications_updated",
    "exchange_key_created",
    "exchange_key_deleted",
    "exchange_connection_created",
    "exchange_connection_validated",
    "exchange_connection_validation_failed",
    "exchange_credential_rotated",
    "exchange_connection_disabled",
    "exchange_connection_archived",
    "exchange_connection_deleted",
    "exchange_connection_reclassified",
]


@dataclass(frozen=True, slots=True)
class AccountProfileSettings:
    owner_user_id: UserId
    username: str | None
    email: str | None
    timezone: str
    telegram_discord: str | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountPreferences:
    owner_user_id: UserId
    theme: ThemePreference
    locale: LocalePreference
    density: DensityPreference
    autorefresh_preset: AutorefreshPreset
    refresh_interval_seconds: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountIntegrationSettings:
    owner_user_id: UserId
    integration_key: Literal["telegram", "discord", "slack"]
    mode: IntegrationMode
    webhook_url_masked: str | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountNotificationSettings:
    owner_user_id: UserId
    channel_key: Literal[
        "telegram",
        "email",
        "push",
        "trade_fills",
        "risk_alerts",
        "daily_report",
        "system",
    ]
    mode: NotificationMode
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountSessionView:
    session_id: str
    owner_user_id: UserId
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    revoked_at: datetime | None
    device: str
    ip_address: str
    location: str
    is_current: bool


@dataclass(frozen=True, slots=True)
class AccountAuditEvent:
    event_id: UUID
    owner_user_id: UserId
    created_at: datetime
    event_type: AuditEventType
    summary: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CursorPage[T]:
    items: tuple[T, ...]
    next_cursor: str | None


class AccountSettingsRepository(Protocol):
    def get_profile(self, *, owner_user_id: UserId, now: datetime) -> AccountProfileSettings:
        ...

    def save_profile(
        self,
        *,
        owner_user_id: UserId,
        username: str | None,
        email: str | None,
        timezone: str,
        telegram_discord: str | None,
        updated_at: datetime,
    ) -> AccountProfileSettings:
        ...

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        ...

    def save_preferences(
        self,
        *,
        owner_user_id: UserId,
        theme: ThemePreference,
        locale: LocalePreference,
        density: DensityPreference,
        autorefresh_preset: AutorefreshPreset,
        refresh_interval_seconds: int,
        updated_at: datetime,
    ) -> AccountPreferences:
        ...

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegrationSettings, ...]:
        ...

    def save_integration(
        self,
        *,
        owner_user_id: UserId,
        integration_key: Literal["telegram", "discord", "slack"],
        mode: IntegrationMode,
        webhook_url_masked: str | None,
        updated_at: datetime,
    ) -> AccountIntegrationSettings:
        ...

    def list_notifications(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountNotificationSettings, ...]:
        ...

    def save_notification(
        self,
        *,
        owner_user_id: UserId,
        channel_key: Literal[
            "telegram",
            "email",
            "push",
            "trade_fills",
            "risk_alerts",
            "daily_report",
            "system",
        ],
        mode: NotificationMode,
        updated_at: datetime,
    ) -> AccountNotificationSettings:
        ...

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> CursorPage[AccountSessionView]:
        ...

    def append_audit_event(
        self,
        *,
        owner_user_id: UserId,
        event_type: AuditEventType,
        summary: str,
        metadata: dict[str, str],
        created_at: datetime,
    ) -> AccountAuditEvent:
        ...

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
    ) -> CursorPage[AccountAuditEvent]:
        ...
