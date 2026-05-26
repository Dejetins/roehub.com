from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
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


class PostgresAccountSettingsRepository(AccountSettingsRepository):
    def __init__(self, *, gateway: IdentityPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresAccountSettingsRepository requires gateway")
        self._gateway = gateway

    def get_profile(self, *, owner_user_id: UserId, now: datetime) -> AccountProfileSettings:
        row = self._gateway.fetch_one(
            query="""
            SELECT owner_user_id, username, email, timezone, telegram_discord, updated_at
            FROM identity_user_profile_overrides
            WHERE owner_user_id = %(owner_user_id)s
            """,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        if row is None:
            return AccountProfileSettings(
                owner_user_id=owner_user_id,
                username="quant_trader",
                email="quant_trader@example.com",
                timezone="Europe/Moscow",
                telegram_discord="@quant_trader / quant_trader#4319",
                updated_at=now,
            )
        return _map_profile(row=row)

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
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO identity_user_profile_overrides
              (owner_user_id, username, email, timezone, telegram_discord, updated_at)
            VALUES
              (%(owner_user_id)s, %(username)s, %(email)s, %(timezone)s,
               %(telegram_discord)s, %(updated_at)s)
            ON CONFLICT (owner_user_id) DO UPDATE SET
              username = EXCLUDED.username,
              email = EXCLUDED.email,
              timezone = EXCLUDED.timezone,
              telegram_discord = EXCLUDED.telegram_discord,
              updated_at = EXCLUDED.updated_at
            RETURNING owner_user_id, username, email, timezone, telegram_discord, updated_at
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "username": username,
                "email": email,
                "timezone": timezone,
                "telegram_discord": telegram_discord,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("identity_user_profile_overrides upsert returned no row")
        return _map_profile(row=row)

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        row = self._gateway.fetch_one(
            query="""
            SELECT owner_user_id, theme, locale, density, autorefresh_preset,
                   refresh_interval_seconds, updated_at
            FROM identity_user_preferences
            WHERE owner_user_id = %(owner_user_id)s
            """,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        if row is None:
            return AccountPreferences(
                owner_user_id=owner_user_id,
                theme="terminal-orange",
                locale="en",
                density="compact",
                autorefresh_preset="15s",
                refresh_interval_seconds=15,
                updated_at=now,
            )
        return _map_preferences(row=row)

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
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO identity_user_preferences
              (owner_user_id, theme, locale, density, autorefresh_preset,
               refresh_interval_seconds, updated_at)
            VALUES
              (%(owner_user_id)s, %(theme)s, %(locale)s, %(density)s,
               %(autorefresh_preset)s, %(refresh_interval_seconds)s, %(updated_at)s)
            ON CONFLICT (owner_user_id) DO UPDATE SET
              theme = EXCLUDED.theme,
              locale = EXCLUDED.locale,
              density = EXCLUDED.density,
              autorefresh_preset = EXCLUDED.autorefresh_preset,
              refresh_interval_seconds = EXCLUDED.refresh_interval_seconds,
              updated_at = EXCLUDED.updated_at
            RETURNING owner_user_id, theme, locale, density, autorefresh_preset,
                      refresh_interval_seconds, updated_at
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "theme": theme,
                "locale": locale,
                "density": density,
                "autorefresh_preset": autorefresh_preset,
                "refresh_interval_seconds": refresh_interval_seconds,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("identity_user_preferences upsert returned no row")
        return _map_preferences(row=row)

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegrationSettings, ...]:
        rows = {
            str(row["integration_key"]): _map_integration(row=row)
            for row in self._gateway.fetch_all(
                query="""
                SELECT owner_user_id, integration_key, mode, webhook_url_masked, updated_at
                FROM identity_integrations
                WHERE owner_user_id = %(owner_user_id)s
                ORDER BY integration_key ASC
                """,
                parameters={"owner_user_id": str(owner_user_id)},
            )
        }
        return tuple(
            rows.get(key)
            or AccountIntegrationSettings(
                owner_user_id=owner_user_id,
                integration_key=key,  # type: ignore[arg-type]
                mode="off" if key == "slack" else "alerts",
                webhook_url_masked=None,
                updated_at=now,
            )
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
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO identity_integrations
              (owner_user_id, integration_key, mode, webhook_url_masked, updated_at)
            VALUES
              (%(owner_user_id)s, %(integration_key)s, %(mode)s,
               %(webhook_url_masked)s, %(updated_at)s)
            ON CONFLICT (owner_user_id, integration_key) DO UPDATE SET
              mode = EXCLUDED.mode,
              webhook_url_masked = EXCLUDED.webhook_url_masked,
              updated_at = EXCLUDED.updated_at
            RETURNING owner_user_id, integration_key, mode, webhook_url_masked, updated_at
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "integration_key": integration_key,
                "mode": mode,
                "webhook_url_masked": webhook_url_masked,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("identity_integrations upsert returned no row")
        return _map_integration(row=row)

    def list_notifications(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountNotificationSettings, ...]:
        rows = {
            str(row["channel_key"]): _map_notification(row=row)
            for row in self._gateway.fetch_all(
                query="""
                SELECT owner_user_id, channel_key, mode, updated_at
                FROM identity_notification_preferences
                WHERE owner_user_id = %(owner_user_id)s
                ORDER BY channel_key ASC
                """,
                parameters={"owner_user_id": str(owner_user_id)},
            )
        }
        return tuple(
            rows.get(key)
            or AccountNotificationSettings(
                owner_user_id=owner_user_id,
                channel_key=key,  # type: ignore[arg-type]
                mode="on",
                updated_at=now,
            )
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
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO identity_notification_preferences
              (owner_user_id, channel_key, mode, updated_at)
            VALUES (%(owner_user_id)s, %(channel_key)s, %(mode)s, %(updated_at)s)
            ON CONFLICT (owner_user_id, channel_key) DO UPDATE SET
              mode = EXCLUDED.mode,
              updated_at = EXCLUDED.updated_at
            RETURNING owner_user_id, channel_key, mode, updated_at
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "channel_key": channel_key,
                "mode": mode,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("identity_notification_preferences upsert returned no row")
        return _map_notification(row=row)

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> CursorPage[AccountSessionView]:
        offset = _cursor_offset(cursor=cursor)
        rows = self._gateway.fetch_all(
            query="""
            SELECT session_id, user_id, created_at, last_seen_at, idle_expires_at,
                   absolute_expires_at, revoked_at
            FROM identity_sessions
            WHERE user_id = %(owner_user_id)s
            ORDER BY last_seen_at DESC, session_id DESC
            LIMIT %(limit_plus_one)s
            OFFSET %(offset)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "limit_plus_one": limit + 1,
                "offset": offset,
            },
        )
        if not rows:
            item = AccountSessionView(
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
            return CursorPage(items=(item,), next_cursor=None)
        mapped = tuple(_map_session(row=row) for row in rows)
        next_cursor = str(offset + limit) if len(mapped) > limit else None
        return CursorPage(items=mapped[:limit], next_cursor=next_cursor)

    def append_audit_event(
        self,
        *,
        owner_user_id: UserId,
        event_type: AuditEventType,
        summary: str,
        metadata: dict[str, str],
        created_at: datetime,
    ) -> AccountAuditEvent:
        event_id = uuid4()
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO identity_audit_events
              (
                event_id,
                owner_user_id,
                created_at,
                event_type,
                summary,
                target_id,
                metadata_json
              )
            VALUES
              (%(event_id)s, %(owner_user_id)s, %(created_at)s, %(event_type)s,
               %(summary)s, %(target_id)s, %(metadata_json)s)
            RETURNING event_id, owner_user_id, created_at, event_type, summary, metadata_json
            """,
            parameters={
                "event_id": str(event_id),
                "owner_user_id": str(owner_user_id),
                "created_at": created_at,
                "event_type": event_type,
                "summary": summary,
                "target_id": metadata.get("connection_id"),
                "metadata_json": Jsonb(metadata),
            },
        )
        if row is None:
            raise ValueError("identity_audit_events insert returned no row")
        return _map_audit_event(row=row)

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
    ) -> CursorPage[AccountAuditEvent]:
        offset = _cursor_offset(cursor=cursor)
        rows = self._gateway.fetch_all(
            query="""
            SELECT event_id, owner_user_id, created_at, event_type, summary, metadata_json
            FROM identity_audit_events
            WHERE owner_user_id = %(owner_user_id)s
            ORDER BY created_at DESC, event_id DESC
            LIMIT %(limit_plus_one)s
            OFFSET %(offset)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "limit_plus_one": limit + 1,
                "offset": offset,
            },
        )
        mapped = tuple(_map_audit_event(row=row) for row in rows)
        next_cursor = str(offset + limit) if len(mapped) > limit else None
        return CursorPage(items=mapped[:limit], next_cursor=next_cursor)


def _cursor_offset(*, cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        return max(0, int(cursor))
    except ValueError:
        return 0


def _map_profile(*, row: Mapping[str, Any]) -> AccountProfileSettings:
    return AccountProfileSettings(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        username=_optional_str(row["username"]),
        email=_optional_str(row["email"]),
        timezone=str(row["timezone"]),
        telegram_discord=_optional_str(row["telegram_discord"]),
        updated_at=_utc(row["updated_at"]),
    )


def _map_preferences(*, row: Mapping[str, Any]) -> AccountPreferences:
    return AccountPreferences(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        theme=str(row["theme"]),  # type: ignore[arg-type]
        locale=str(row["locale"]),  # type: ignore[arg-type]
        density=str(row["density"]),  # type: ignore[arg-type]
        autorefresh_preset=str(row["autorefresh_preset"]),  # type: ignore[arg-type]
        refresh_interval_seconds=int(row["refresh_interval_seconds"]),
        updated_at=_utc(row["updated_at"]),
    )


def _map_integration(*, row: Mapping[str, Any]) -> AccountIntegrationSettings:
    return AccountIntegrationSettings(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        integration_key=str(row["integration_key"]),  # type: ignore[arg-type]
        mode=str(row["mode"]),  # type: ignore[arg-type]
        webhook_url_masked=_optional_str(row["webhook_url_masked"]),
        updated_at=_utc(row["updated_at"]),
    )


def _map_notification(*, row: Mapping[str, Any]) -> AccountNotificationSettings:
    return AccountNotificationSettings(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        channel_key=str(row["channel_key"]),  # type: ignore[arg-type]
        mode=str(row["mode"]),  # type: ignore[arg-type]
        updated_at=_utc(row["updated_at"]),
    )


def _map_session(*, row: Mapping[str, Any]) -> AccountSessionView:
    return AccountSessionView(
        session_id=str(row["session_id"]),
        owner_user_id=UserId.from_string(str(row["user_id"])),
        created_at=_utc(row["created_at"]),
        last_seen_at=_utc(row["last_seen_at"]),
        idle_expires_at=_utc(row["idle_expires_at"]),
        absolute_expires_at=_utc(row["absolute_expires_at"]),
        revoked_at=_optional_utc(row["revoked_at"]),
        device="Roehub Web",
        ip_address="masked",
        location="unknown",
        is_current=False,
    )


def _map_audit_event(*, row: Mapping[str, Any]) -> AccountAuditEvent:
    metadata = row.get("metadata_json")
    safe_metadata = {
        str(key): str(value)
        for key, value in (metadata.items() if isinstance(metadata, Mapping) else [])
    }
    return AccountAuditEvent(
        event_id=UUID(str(row["event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        created_at=_utc(row["created_at"]),
        event_type=str(row["event_type"]),  # type: ignore[arg-type]
        summary=str(row["summary"]),
        metadata=safe_metadata,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("timestamp must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    return _utc(value)
