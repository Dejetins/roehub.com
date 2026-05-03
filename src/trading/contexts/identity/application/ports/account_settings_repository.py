from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.identity.application.ports.session_repository import IdentitySession
from trading.shared_kernel.primitives import UserId

SUPPORTED_ACCOUNT_THEMES = (
    "terminal-orange",
    "graphite",
    "matrix-green",
    "high-contrast",
)
SUPPORTED_ACCOUNT_LOCALES = ("en", "ru")
SUPPORTED_ACCOUNT_DENSITIES = ("compact", "comfortable")
SUPPORTED_ACCOUNT_INTEGRATIONS = ("telegram", "email_digest", "webhook_alerts")


@dataclass(frozen=True, slots=True)
class AccountPreferences:
    owner_user_id: UserId
    theme: str
    locale: str
    density: str
    email_notifications_enabled: bool
    trade_alerts_enabled: bool
    product_updates_enabled: bool
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountProfile:
    owner_user_id: UserId
    display_name: str | None
    timezone: str
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class AccountIntegration:
    owner_user_id: UserId
    provider: str
    enabled: bool
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountAuditEvent:
    event_id: UUID
    owner_user_id: UserId
    event_type: str
    metadata: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AccountAuditCursor:
    created_at: datetime
    event_id: UUID


@dataclass(frozen=True, slots=True)
class AccountSessionCursor:
    created_at: datetime
    session_id: UUID


class AccountSettingsRepository(Protocol):
    """
    AccountSettingsRepository stores owner-scoped account UI state in identity storage.

    Docs:
      - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
    Related:
      - apps/api/routes/ui_account.py
      - migrations/postgres/0006_identity_account_settings_v1.sql
    """

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        ...

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
        ...

    def get_profile(self, *, owner_user_id: UserId) -> AccountProfile:
        ...

    def upsert_profile(
        self,
        *,
        owner_user_id: UserId,
        display_name: str | None,
        timezone: str,
        updated_at: datetime,
    ) -> AccountProfile:
        ...

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegration, ...]:
        ...

    def upsert_integrations(
        self,
        *,
        owner_user_id: UserId,
        integrations: tuple[tuple[str, bool], ...],
        updated_at: datetime,
    ) -> tuple[AccountIntegration, ...]:
        ...

    def append_audit_event(
        self,
        *,
        event_id: UUID,
        owner_user_id: UserId,
        event_type: str,
        metadata: Mapping[str, Any],
        created_at: datetime,
    ) -> AccountAuditEvent:
        ...

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountAuditCursor | None,
        limit: int,
    ) -> tuple[AccountAuditEvent, ...]:
        ...

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountSessionCursor | None,
        limit: int,
    ) -> tuple[IdentitySession, ...]:
        ...
