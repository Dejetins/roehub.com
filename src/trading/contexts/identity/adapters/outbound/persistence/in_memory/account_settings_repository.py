from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountAuditEvent,
    AccountIntegrationSettings,
    AccountNotificationSettings,
    AccountPreferences,
    AccountProfileSettings,
    AccountSessionView,
    AccountSettingsRepository,
    AuditEventType,
    AutorefreshPreset,
    CursorPage,
    DensityPreference,
    IntegrationMode,
    LocalePreference,
    NotificationMode,
    ThemePreference,
)
from trading.shared_kernel.primitives import UserId


class InMemoryAccountSettingsRepository(AccountSettingsRepository):
    def __init__(self) -> None:
        self._profiles: dict[str, AccountProfileSettings] = {}
        self._preferences: dict[str, AccountPreferences] = {}
        self._integrations: dict[tuple[str, str], AccountIntegrationSettings] = {}
        self._notifications: dict[tuple[str, str], AccountNotificationSettings] = {}
        self._sessions: dict[str, AccountSessionView] = {}
        self._audit_events: list[AccountAuditEvent] = []

    def get_profile(self, *, owner_user_id: UserId, now: datetime) -> AccountProfileSettings:
        key = str(owner_user_id)
        existing = self._profiles.get(key)
        if existing is not None:
            return existing
        profile = AccountProfileSettings(
            owner_user_id=owner_user_id,
            username="quant_trader",
            email="quant_trader@example.com",
            timezone="Europe/Moscow",
            telegram_discord="@quant_trader / quant_trader#4319",
            updated_at=now,
        )
        self._profiles[key] = profile
        return profile

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
        profile = AccountProfileSettings(
            owner_user_id=owner_user_id,
            username=username,
            email=email,
            timezone=timezone,
            telegram_discord=telegram_discord,
            updated_at=updated_at,
        )
        self._profiles[str(owner_user_id)] = profile
        return profile

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        key = str(owner_user_id)
        existing = self._preferences.get(key)
        if existing is not None:
            return existing
        preferences = AccountPreferences(
            owner_user_id=owner_user_id,
            theme="terminal-orange",
            locale="en",
            density="compact",
            autorefresh_preset="15s",
            refresh_interval_seconds=15,
            updated_at=now,
        )
        self._preferences[key] = preferences
        return preferences

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
        preferences = AccountPreferences(
            owner_user_id=owner_user_id,
            theme=theme,
            locale=locale,
            density=density,
            autorefresh_preset=autorefresh_preset,
            refresh_interval_seconds=refresh_interval_seconds,
            updated_at=updated_at,
        )
        self._preferences[str(owner_user_id)] = preferences
        return preferences

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegrationSettings, ...]:
        return tuple(
            self._integrations.get((str(owner_user_id), key))
            or self._default_integration(owner_user_id=owner_user_id, key=key, now=now)
            for key in ("telegram", "discord", "slack")
        )

    def save_integration(
        self,
        *,
        owner_user_id: UserId,
        integration_key: Literal["telegram", "discord", "slack"],
        mode: IntegrationMode,
        webhook_url_masked: str | None,
        updated_at: datetime,
    ) -> AccountIntegrationSettings:
        integration = AccountIntegrationSettings(
            owner_user_id=owner_user_id,
            integration_key=integration_key,  # type: ignore[arg-type]
            mode=mode,
            webhook_url_masked=webhook_url_masked,
            updated_at=updated_at,
        )
        self._integrations[(str(owner_user_id), integration_key)] = integration
        return integration

    def list_notifications(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountNotificationSettings, ...]:
        return tuple(
            self._notifications.get((str(owner_user_id), key))
            or self._default_notification(owner_user_id=owner_user_id, key=key, now=now)
            for key in (
                "telegram",
                "email",
                "push",
                "trade_fills",
                "risk_alerts",
                "daily_report",
                "system",
            )
        )

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
        notification = AccountNotificationSettings(
            owner_user_id=owner_user_id,
            channel_key=channel_key,  # type: ignore[arg-type]
            mode=mode,
            updated_at=updated_at,
        )
        self._notifications[(str(owner_user_id), channel_key)] = notification
        return notification

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> CursorPage[AccountSessionView]:
        sessions = [
            session
            for session in self._sessions.values()
            if str(session.owner_user_id) == str(owner_user_id)
        ]
        if not sessions:
            sessions = [
                AccountSessionView(
                    session_id="current",
                    owner_user_id=owner_user_id,
                    created_at=now,
                    last_seen_at=now,
                    idle_expires_at=now,
                    absolute_expires_at=now,
                    revoked_at=None,
                    device="Roehub Web / current browser",
                    ip_address="127.0.0.1",
                    location="local-dev",
                    is_current=True,
                )
            ]
        ordered = sorted(
            sessions,
            key=lambda item: (item.last_seen_at, item.session_id),
            reverse=True,
        )
        return _page_items(items=tuple(ordered), cursor=cursor, limit=limit)

    def append_audit_event(
        self,
        *,
        owner_user_id: UserId,
        event_type: AuditEventType,
        summary: str,
        metadata: dict[str, str],
        created_at: datetime,
    ) -> AccountAuditEvent:
        event = AccountAuditEvent(
            event_id=uuid4(),
            owner_user_id=owner_user_id,
            created_at=created_at,
            event_type=event_type,
            summary=summary,
            metadata=dict(metadata),
        )
        self._audit_events.append(event)
        return event

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
    ) -> CursorPage[AccountAuditEvent]:
        indexed_events = [
            (index, event)
            for index, event in enumerate(self._audit_events)
            if str(event.owner_user_id) == str(owner_user_id)
        ]
        ordered = [
            event
            for _index, event in sorted(
                indexed_events,
                key=lambda item: (item[1].created_at, item[0]),
                reverse=True,
            )
        ]
        return _page_items(items=tuple(ordered), cursor=cursor, limit=limit)

    def record_session(self, *, session: AccountSessionView) -> None:
        self._sessions[session.session_id] = session

    def _default_integration(
        self,
        *,
        owner_user_id: UserId,
        key: str,
        now: datetime,
    ) -> AccountIntegrationSettings:
        defaults = {
            "telegram": ("alerts", "https://...e97fb"),
            "discord": ("alerts", None),
            "slack": ("off", None),
        }
        mode, webhook_url_masked = defaults[key]
        integration = AccountIntegrationSettings(
            owner_user_id=owner_user_id,
            integration_key=key,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            webhook_url_masked=webhook_url_masked,
            updated_at=now,
        )
        self._integrations[(str(owner_user_id), key)] = integration
        return integration

    def _default_notification(
        self,
        *,
        owner_user_id: UserId,
        key: str,
        now: datetime,
    ) -> AccountNotificationSettings:
        notification = AccountNotificationSettings(
            owner_user_id=owner_user_id,
            channel_key=key,  # type: ignore[arg-type]
            mode="on",
            updated_at=now,
        )
        self._notifications[(str(owner_user_id), key)] = notification
        return notification


def _page_items[T](
    *,
    items: tuple[T, ...],
    cursor: str | None,
    limit: int,
) -> CursorPage[T]:
    start = 0
    if cursor is not None:
        try:
            start = max(0, int(cursor))
        except ValueError:
            start = 0
    selected = items[start : start + limit + 1]
    next_cursor = str(start + limit) if len(selected) > limit else None
    return CursorPage(items=tuple(selected[:limit]), next_cursor=next_cursor)
