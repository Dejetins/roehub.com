from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.identity.adapters.outbound.persistence.in_memory.session_repository import (
    InMemoryIdentitySessionRepository,
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


class InMemoryIdentityAccountSettingsRepository(AccountSettingsRepository):
    """
    In-memory account settings repository for dev/test identity wiring.

    Docs:
      - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/account_settings_repository.py
      - apps/api/routes/ui_account.py
    """

    def __init__(self, *, session_repository: InMemoryIdentitySessionRepository) -> None:
        if session_repository is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "InMemoryIdentityAccountSettingsRepository requires session_repository"
            )
        self._session_repository = session_repository
        self._preferences: dict[str, AccountPreferences] = {}
        self._profiles: dict[str, AccountProfile] = {}
        self._integrations: dict[tuple[str, str], AccountIntegration] = {}
        self._audit_events: dict[str, AccountAuditEvent] = {}

    def get_preferences(self, *, owner_user_id: UserId, now: datetime) -> AccountPreferences:
        user_key = str(owner_user_id)
        existing = self._preferences.get(user_key)
        if existing is not None:
            return existing
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
        preferences = AccountPreferences(
            owner_user_id=owner_user_id,
            theme=theme,
            locale=locale,
            density=density,
            email_notifications_enabled=email_notifications_enabled,
            trade_alerts_enabled=trade_alerts_enabled,
            product_updates_enabled=product_updates_enabled,
            updated_at=updated_at,
        )
        self._preferences[str(owner_user_id)] = preferences
        return preferences

    def get_profile(self, *, owner_user_id: UserId) -> AccountProfile:
        return self._profiles.get(
            str(owner_user_id),
            AccountProfile(
                owner_user_id=owner_user_id,
                display_name=None,
                timezone="UTC",
                updated_at=None,
            ),
        )

    def upsert_profile(
        self,
        *,
        owner_user_id: UserId,
        display_name: str | None,
        timezone: str,
        updated_at: datetime,
    ) -> AccountProfile:
        profile = AccountProfile(
            owner_user_id=owner_user_id,
            display_name=display_name,
            timezone=timezone,
            updated_at=updated_at,
        )
        self._profiles[str(owner_user_id)] = profile
        return profile

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
        now: datetime,
    ) -> tuple[AccountIntegration, ...]:
        integrations: list[AccountIntegration] = []
        for provider in SUPPORTED_ACCOUNT_INTEGRATIONS:
            integrations.append(
                self._integrations.get(
                    (str(owner_user_id), provider),
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
        for provider, enabled in integrations:
            integration = AccountIntegration(
                owner_user_id=owner_user_id,
                provider=provider,
                enabled=enabled,
                updated_at=updated_at,
            )
            self._integrations[(str(owner_user_id), provider)] = integration
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
        event = AccountAuditEvent(
            event_id=event_id,
            owner_user_id=owner_user_id,
            event_type=event_type,
            metadata=dict(metadata),
            created_at=created_at,
        )
        self._audit_events[str(event_id)] = event
        return event

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountAuditCursor | None,
        limit: int,
    ) -> tuple[AccountAuditEvent, ...]:
        rows = [
            event
            for event in self._audit_events.values()
            if event.owner_user_id == owner_user_id
        ]
        rows.sort(key=lambda item: (item.created_at, str(item.event_id)), reverse=True)
        if cursor is not None:
            rows = [
                row
                for row in rows
                if (row.created_at, str(row.event_id))
                < (cursor.created_at, str(cursor.event_id))
            ]
        return tuple(rows[:limit])

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: AccountSessionCursor | None,
        limit: int,
    ) -> tuple[IdentitySession, ...]:
        rows = [
            session
            for session in self._session_repository._sessions.values()
            if session.user_id == owner_user_id
        ]
        rows.sort(key=lambda item: (item.created_at, str(item.session_id)), reverse=True)
        if cursor is not None:
            rows = [
                row
                for row in rows
                if (row.created_at, str(row.session_id))
                < (cursor.created_at, str(cursor.session_id))
            ]
        return tuple(replace(row) for row in rows[:limit])
