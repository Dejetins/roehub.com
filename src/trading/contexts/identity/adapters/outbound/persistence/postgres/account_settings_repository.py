from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from psycopg.types.json import Json

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    IdentityPostgresGateway,
)
from trading.contexts.identity.application.ports.account_settings_repository import (
    SUPPORTED_ACCOUNT_INTEGRATIONS,
    AccountAuditCursor,
    AccountAuditEvent,
    AccountIntegration,
    AccountPreferences,
    AccountProfile,
    AccountSessionCursor,
    AccountSettingsRepository,
)
from trading.contexts.identity.application.ports.session_repository import IdentitySession
from trading.shared_kernel.primitives import UserId


class PostgresIdentityAccountSettingsRepository(AccountSettingsRepository):
    """
    Postgres account settings repository backed by identity-owned tables.

    Docs:
      - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
    Related:
      - migrations/postgres/0006_identity_account_settings_v1.sql
      - apps/api/routes/ui_account.py
    """

    def __init__(
        self,
        *,
        gateway: IdentityPostgresGateway,
        preferences_table: str = "identity_user_preferences",
        profile_table: str = "identity_user_profile_overrides",
        integrations_table: str = "identity_integrations",
        audit_table: str = "identity_audit_events",
        sessions_table: str = "identity_sessions",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresIdentityAccountSettingsRepository requires gateway")
        self._gateway = gateway
        self._preferences_table = _normalize_table_name(preferences_table)
        self._profile_table = _normalize_table_name(profile_table)
        self._integrations_table = _normalize_table_name(integrations_table)
        self._audit_table = _normalize_table_name(audit_table)
        self._sessions_table = _normalize_table_name(sessions_table)

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        query = f"""
        SELECT
            owner_user_id,
            theme,
            locale,
            density,
            email_notifications_enabled,
            trade_alerts_enabled,
            product_updates_enabled,
            updated_at
        FROM {self._preferences_table}
        WHERE owner_user_id = %(owner_user_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        if row is None:
            return AccountPreferences(
                owner_user_id=owner_user_id,
                theme="terminal-orange",
                locale="en",
                density="compact",
                email_notifications_enabled=True,
                trade_alerts_enabled=True,
                product_updates_enabled=False,
                updated_at=now,
            )
        return _map_preferences_row(row=row)

    def upsert_preferences(
        self,
        *,
        owner_user_id: UserId,
        theme: str,
        locale: str,
        density: str,
        email_notifications_enabled: bool,
        trade_alerts_enabled: bool,
        product_updates_enabled: bool,
        updated_at: datetime,
    ) -> AccountPreferences:
        query = f"""
        INSERT INTO {self._preferences_table}
        (
            owner_user_id,
            theme,
            locale,
            density,
            email_notifications_enabled,
            trade_alerts_enabled,
            product_updates_enabled,
            created_at,
            updated_at
        )
        VALUES
        (
            %(owner_user_id)s,
            %(theme)s,
            %(locale)s,
            %(density)s,
            %(email_notifications_enabled)s,
            %(trade_alerts_enabled)s,
            %(product_updates_enabled)s,
            %(updated_at)s,
            %(updated_at)s
        )
        ON CONFLICT (owner_user_id) DO UPDATE SET
            theme = EXCLUDED.theme,
            locale = EXCLUDED.locale,
            density = EXCLUDED.density,
            email_notifications_enabled = EXCLUDED.email_notifications_enabled,
            trade_alerts_enabled = EXCLUDED.trade_alerts_enabled,
            product_updates_enabled = EXCLUDED.product_updates_enabled,
            updated_at = EXCLUDED.updated_at
        RETURNING
            owner_user_id,
            theme,
            locale,
            density,
            email_notifications_enabled,
            trade_alerts_enabled,
            product_updates_enabled,
            updated_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "theme": theme,
                "locale": locale,
                "density": density,
                "email_notifications_enabled": email_notifications_enabled,
                "trade_alerts_enabled": trade_alerts_enabled,
                "product_updates_enabled": product_updates_enabled,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentityAccountSettingsRepository preference upsert failed")
        return _map_preferences_row(row=row)

    def get_profile(self, *, owner_user_id: UserId) -> AccountProfile:
        query = f"""
        SELECT owner_user_id, display_name, timezone, updated_at
        FROM {self._profile_table}
        WHERE owner_user_id = %(owner_user_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        if row is None:
            return AccountProfile(
                owner_user_id=owner_user_id,
                display_name=None,
                timezone="UTC",
                updated_at=None,
            )
        return _map_profile_row(row=row)

    def upsert_profile(
        self,
        *,
        owner_user_id: UserId,
        display_name: str | None,
        timezone: str,
        updated_at: datetime,
    ) -> AccountProfile:
        query = f"""
        INSERT INTO {self._profile_table}
        (owner_user_id, display_name, timezone, created_at, updated_at)
        VALUES (%(owner_user_id)s, %(display_name)s, %(timezone)s, %(updated_at)s, %(updated_at)s)
        ON CONFLICT (owner_user_id) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            timezone = EXCLUDED.timezone,
            updated_at = EXCLUDED.updated_at
        RETURNING owner_user_id, display_name, timezone, updated_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "owner_user_id": str(owner_user_id),
                "display_name": display_name,
                "timezone": timezone,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentityAccountSettingsRepository profile upsert failed")
        return _map_profile_row(row=row)

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegration, ...]:
        query = f"""
        SELECT owner_user_id, provider, enabled, updated_at
        FROM {self._integrations_table}
        WHERE owner_user_id = %(owner_user_id)s
        ORDER BY provider ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        by_provider = {}
        for row in rows:
            integration = _map_integration_row(row=row)
            by_provider[integration.provider] = integration
        integrations: list[AccountIntegration] = []
        for provider in SUPPORTED_ACCOUNT_INTEGRATIONS:
            integrations.append(
                by_provider.get(
                    provider,
                    AccountIntegration(
                        owner_user_id=owner_user_id,
                        provider=provider,
                        enabled=False,
                        updated_at=now,
                    ),
                )
            )
        return tuple(integrations)

    def upsert_integrations(
        self,
        *,
        owner_user_id: UserId,
        integrations: tuple[tuple[str, bool], ...],
        updated_at: datetime,
    ) -> tuple[AccountIntegration, ...]:
        query = f"""
        INSERT INTO {self._integrations_table}
        (owner_user_id, provider, enabled, settings_json, created_at, updated_at)
        VALUES (
            %(owner_user_id)s,
            %(provider)s,
            %(enabled)s,
            '{{}}'::jsonb,
            %(updated_at)s,
            %(updated_at)s
        )
        ON CONFLICT (owner_user_id, provider) DO UPDATE SET
            enabled = EXCLUDED.enabled,
            updated_at = EXCLUDED.updated_at
        """
        for provider, enabled in integrations:
            self._gateway.execute(
                query=query,
                parameters={
                    "owner_user_id": str(owner_user_id),
                    "provider": provider,
                    "enabled": enabled,
                    "updated_at": updated_at,
                },
            )
        return self.list_integrations(owner_user_id=owner_user_id, now=updated_at)

    def append_audit_event(
        self,
        *,
        event_id: UUID,
        owner_user_id: UserId,
        event_type: str,
        metadata: Mapping[str, Any],
        created_at: datetime,
    ) -> AccountAuditEvent:
        query = f"""
        INSERT INTO {self._audit_table}
        (event_id, owner_user_id, event_type, event_version, metadata_json, created_at)
        VALUES (
            %(event_id)s,
            %(owner_user_id)s,
            %(event_type)s,
            1,
            %(metadata_json)s,
            %(created_at)s
        )
        RETURNING event_id, owner_user_id, event_type, metadata_json, created_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "event_id": str(event_id),
                "owner_user_id": str(owner_user_id),
                "event_type": event_type,
                "metadata_json": Json(dict(metadata)),
                "created_at": created_at,
            },
        )
        if row is None:
            raise ValueError("PostgresIdentityAccountSettingsRepository audit insert failed")
        return _map_audit_row(row=row)

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountAuditCursor | None,
        limit: int,
    ) -> tuple[AccountAuditEvent, ...]:
        cursor_clause = ""
        parameters: dict[str, Any] = {
            "owner_user_id": str(owner_user_id),
            "limit": limit,
        }
        if cursor is not None:
            cursor_clause = """
              AND (created_at, event_id) < (%(cursor_created_at)s, %(cursor_event_id)s)
            """
            parameters["cursor_created_at"] = cursor.created_at
            parameters["cursor_event_id"] = str(cursor.event_id)
        query = f"""
        SELECT event_id, owner_user_id, event_type, metadata_json, created_at
        FROM {self._audit_table}
        WHERE owner_user_id = %(owner_user_id)s
        {cursor_clause}
        ORDER BY created_at DESC, event_id DESC
        LIMIT %(limit)s
        """
        rows = self._gateway.fetch_all(query=query, parameters=parameters)
        return tuple(_map_audit_row(row=row) for row in rows)

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountSessionCursor | None,
        limit: int,
    ) -> tuple[IdentitySession, ...]:
        cursor_clause = ""
        parameters: dict[str, Any] = {
            "owner_user_id": str(owner_user_id),
            "limit": limit,
        }
        if cursor is not None:
            cursor_clause = """
              AND (created_at, session_id) < (%(cursor_created_at)s, %(cursor_session_id)s)
            """
            parameters["cursor_created_at"] = cursor.created_at
            parameters["cursor_session_id"] = str(cursor.session_id)
        query = f"""
        SELECT
            session_id,
            user_id,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        FROM {self._sessions_table}
        WHERE user_id = %(owner_user_id)s
        {cursor_clause}
        ORDER BY created_at DESC, session_id DESC
        LIMIT %(limit)s
        """
        rows = self._gateway.fetch_all(query=query, parameters=parameters)
        return tuple(_map_session_row(row=row) for row in rows)


def _normalize_table_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("table name must be non-empty")
    return normalized


def _map_preferences_row(*, row: Mapping[str, Any]) -> AccountPreferences:
    return AccountPreferences(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        theme=str(row["theme"]),
        locale=str(row["locale"]),
        density=str(row["density"]),
        email_notifications_enabled=bool(row["email_notifications_enabled"]),
        trade_alerts_enabled=bool(row["trade_alerts_enabled"]),
        product_updates_enabled=bool(row["product_updates_enabled"]),
        updated_at=_normalize_utc_datetime(value=row["updated_at"], field_name="updated_at"),
    )


def _map_profile_row(*, row: Mapping[str, Any]) -> AccountProfile:
    return AccountProfile(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        display_name=str(row["display_name"]) if row["display_name"] is not None else None,
        timezone=str(row["timezone"]),
        updated_at=_normalize_optional_utc_datetime(
            value=row["updated_at"],
            field_name="updated_at",
        ),
    )


def _map_integration_row(*, row: Mapping[str, Any]) -> AccountIntegration:
    return AccountIntegration(
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        provider=str(row["provider"]),
        enabled=bool(row["enabled"]),
        updated_at=_normalize_utc_datetime(value=row["updated_at"], field_name="updated_at"),
    )


def _map_audit_row(*, row: Mapping[str, Any]) -> AccountAuditEvent:
    return AccountAuditEvent(
        event_id=UUID(str(row["event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        event_type=str(row["event_type"]),
        metadata=_normalize_json_mapping(row["metadata_json"]),
        created_at=_normalize_utc_datetime(value=row["created_at"], field_name="created_at"),
    )


def _map_session_row(*, row: Mapping[str, Any]) -> IdentitySession:
    return IdentitySession(
        session_id=UUID(str(row["session_id"])),
        user_id=UserId.from_string(str(row["user_id"])),
        created_at=_normalize_utc_datetime(value=row["created_at"], field_name="created_at"),
        last_seen_at=_normalize_utc_datetime(
            value=row["last_seen_at"],
            field_name="last_seen_at",
        ),
        idle_expires_at=_normalize_utc_datetime(
            value=row["idle_expires_at"],
            field_name="idle_expires_at",
        ),
        absolute_expires_at=_normalize_utc_datetime(
            value=row["absolute_expires_at"],
            field_name="absolute_expires_at",
        ),
        revoked_at=_normalize_optional_utc_datetime(
            value=row["revoked_at"],
            field_name="revoked_at",
        ),
    )


def _normalize_json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalize_utc_datetime(*, value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _normalize_optional_utc_datetime(*, value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _normalize_utc_datetime(value=value, field_name=field_name)
