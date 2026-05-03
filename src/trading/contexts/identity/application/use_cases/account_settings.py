from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.identity.application.ports.account_settings_repository import (
    SUPPORTED_ACCOUNT_DENSITIES,
    SUPPORTED_ACCOUNT_INTEGRATIONS,
    SUPPORTED_ACCOUNT_LOCALES,
    SUPPORTED_ACCOUNT_THEMES,
    AccountAuditCursor,
    AccountAuditEvent,
    AccountIntegration,
    AccountPreferences,
    AccountSessionCursor,
    AccountSettingsRepository,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.session_repository import IdentitySession
from trading.shared_kernel.primitives import PaidLevel, UserId

_DEFAULT_PAGE_LIMIT = 25
_MAX_PAGE_LIMIT = 100


@dataclass(frozen=True, slots=True)
class AccountProfileView:
    owner_user_id: UserId
    paid_level: PaidLevel
    display_name: str | None
    timezone: str
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class AccountLimitsView:
    paid_level: PaidLevel
    exchange_keys_used: int
    exchange_keys_limit: int
    active_strategies_used: int
    active_strategies_limit: int
    webhook_events_used: int
    webhook_events_limit: int


@dataclass(frozen=True, slots=True)
class AccountSessionView:
    session_id: UUID
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    revoked_at: datetime | None
    status: str


@dataclass(frozen=True, slots=True)
class AccountSessionsPage:
    items: tuple[AccountSessionView, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class AccountAuditEventsPage:
    items: tuple[AccountAuditEvent, ...]
    next_cursor: str | None


class AccountSettingsOperationError(ValueError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        field_errors: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.field_errors = dict(field_errors or {})

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
        }
        if self.field_errors:
            payload["field_errors"] = self.field_errors
        return payload


class AccountSettingsUseCase:
    """
    AccountSettingsUseCase owns owner-scoped settings read/write behavior.

    Docs:
      - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
    Related:
      - apps/api/routes/ui_account.py
      - src/trading/contexts/identity/application/ports/account_settings_repository.py
    """

    def __init__(
        self,
        *,
        repository: AccountSettingsRepository,
        clock: IdentityClock,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("AccountSettingsUseCase requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("AccountSettingsUseCase requires clock")
        self._repository = repository
        self._clock = clock

    def get_profile(
        self,
        *,
        owner_user_id: UserId,
        paid_level: PaidLevel,
    ) -> AccountProfileView:
        profile = self._repository.get_profile(owner_user_id=owner_user_id)
        return AccountProfileView(
            owner_user_id=profile.owner_user_id,
            paid_level=paid_level,
            display_name=profile.display_name,
            timezone=profile.timezone,
            updated_at=profile.updated_at,
        )

    def update_profile(
        self,
        *,
        owner_user_id: UserId,
        paid_level: PaidLevel,
        display_name: str | None,
        timezone_name: str,
    ) -> AccountProfileView:
        now = self._clock.now()
        profile = self._repository.upsert_profile(
            owner_user_id=owner_user_id,
            display_name=_normalize_display_name(display_name=display_name),
            timezone=_normalize_timezone(timezone_name=timezone_name),
            updated_at=now,
        )
        self._append_audit(
            owner_user_id=owner_user_id,
            event_type="account.profile.updated",
            metadata={"fields": ["display_name", "timezone"]},
            now=now,
        )
        return AccountProfileView(
            owner_user_id=profile.owner_user_id,
            paid_level=paid_level,
            display_name=profile.display_name,
            timezone=profile.timezone,
            updated_at=profile.updated_at,
        )

    def get_limits(
        self,
        *,
        paid_level: PaidLevel,
        exchange_keys_used: int,
    ) -> AccountLimitsView:
        plan = _limits_for_paid_level(paid_level=paid_level)
        return AccountLimitsView(
            paid_level=paid_level,
            exchange_keys_used=exchange_keys_used,
            exchange_keys_limit=plan["exchange_keys"],
            active_strategies_used=0,
            active_strategies_limit=plan["active_strategies"],
            webhook_events_used=0,
            webhook_events_limit=plan["webhook_events"],
        )

    def get_preferences(self, *, owner_user_id: UserId) -> AccountPreferences:
        return self._repository.get_preferences(
            owner_user_id=owner_user_id,
            now=self._clock.now(),
        )

    def update_preferences(
        self,
        *,
        owner_user_id: UserId,
        updates: Mapping[str, str],
    ) -> AccountPreferences:
        current = self.get_preferences(owner_user_id=owner_user_id)
        theme = updates.get("theme", current.theme)
        locale = updates.get("locale", current.locale)
        density = updates.get("density", current.density)
        _validate_choice(field_name="theme", value=theme, allowed=SUPPORTED_ACCOUNT_THEMES)
        _validate_choice(field_name="locale", value=locale, allowed=SUPPORTED_ACCOUNT_LOCALES)
        _validate_choice(field_name="density", value=density, allowed=SUPPORTED_ACCOUNT_DENSITIES)
        now = self._clock.now()
        updated = self._repository.upsert_preferences(
            owner_user_id=owner_user_id,
            theme=theme,
            locale=locale,
            density=density,
            email_notifications_enabled=current.email_notifications_enabled,
            trade_alerts_enabled=current.trade_alerts_enabled,
            product_updates_enabled=current.product_updates_enabled,
            updated_at=now,
        )
        self._append_audit(
            owner_user_id=owner_user_id,
            event_type="account.preferences.updated",
            metadata={"fields": sorted(updates)},
            now=now,
        )
        return updated

    def update_notifications(
        self,
        *,
        owner_user_id: UserId,
        email_notifications_enabled: bool,
        trade_alerts_enabled: bool,
        product_updates_enabled: bool,
    ) -> AccountPreferences:
        current = self.get_preferences(owner_user_id=owner_user_id)
        now = self._clock.now()
        updated = self._repository.upsert_preferences(
            owner_user_id=owner_user_id,
            theme=current.theme,
            locale=current.locale,
            density=current.density,
            email_notifications_enabled=email_notifications_enabled,
            trade_alerts_enabled=trade_alerts_enabled,
            product_updates_enabled=product_updates_enabled,
            updated_at=now,
        )
        self._append_audit(
            owner_user_id=owner_user_id,
            event_type="account.notifications.updated",
            metadata={"fields": ["email", "trade_alerts", "product_updates"]},
            now=now,
        )
        return updated

    def list_integrations(self, *, owner_user_id: UserId) -> tuple[AccountIntegration, ...]:
        return self._repository.list_integrations(
            owner_user_id=owner_user_id,
            now=self._clock.now(),
        )

    def update_integrations(
        self,
        *,
        owner_user_id: UserId,
        integrations: tuple[tuple[str, bool], ...],
    ) -> tuple[AccountIntegration, ...]:
        providers = [provider for provider, _enabled in integrations]
        if len(providers) != len(set(providers)):
            raise AccountSettingsOperationError(
                status_code=422,
                code="validation_error",
                message="Validation failed",
                field_errors={"providers": "Duplicate integration provider."},
            )
        for provider in providers:
            _validate_choice(
                field_name="provider",
                value=provider,
                allowed=SUPPORTED_ACCOUNT_INTEGRATIONS,
            )
        now = self._clock.now()
        updated = self._repository.upsert_integrations(
            owner_user_id=owner_user_id,
            integrations=integrations,
            updated_at=now,
        )
        self._append_audit(
            owner_user_id=owner_user_id,
            event_type="account.integrations.updated",
            metadata={"providers": sorted(providers)},
            now=now,
        )
        return updated

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int | None,
    ) -> AccountSessionsPage:
        page_limit = _normalize_limit(limit=limit)
        parsed_cursor = _decode_session_cursor(raw_cursor=cursor)
        rows = self._repository.list_sessions(
            owner_user_id=owner_user_id,
            cursor=parsed_cursor,
            limit=page_limit + 1,
        )
        page_rows = rows[:page_limit]
        now = self._clock.now()
        items = tuple(_to_session_view(session=session, now=now) for session in page_rows)
        next_cursor = None
        if len(rows) > page_limit and page_rows:
            next_cursor = _encode_session_cursor(session=page_rows[-1])
        return AccountSessionsPage(items=items, next_cursor=next_cursor)

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int | None,
    ) -> AccountAuditEventsPage:
        page_limit = _normalize_limit(limit=limit)
        parsed_cursor = _decode_audit_cursor(raw_cursor=cursor)
        rows = self._repository.list_audit_events(
            owner_user_id=owner_user_id,
            cursor=parsed_cursor,
            limit=page_limit + 1,
        )
        page_rows = rows[:page_limit]
        next_cursor = None
        if len(rows) > page_limit and page_rows:
            next_cursor = _encode_audit_cursor(event=page_rows[-1])
        return AccountAuditEventsPage(items=page_rows, next_cursor=next_cursor)

    def _append_audit(
        self,
        *,
        owner_user_id: UserId,
        event_type: str,
        metadata: Mapping[str, Any],
        now: datetime,
    ) -> AccountAuditEvent:
        return self._repository.append_audit_event(
            event_id=uuid4(),
            owner_user_id=owner_user_id,
            event_type=event_type,
            metadata=metadata,
            created_at=now,
        )


def _normalize_display_name(*, display_name: str | None) -> str | None:
    if display_name is None:
        return None
    normalized = display_name.strip()
    if not normalized:
        return None
    if len(normalized) > 80:
        raise AccountSettingsOperationError(
            status_code=422,
            code="validation_error",
            message="Validation failed",
            field_errors={"display_name": "Display name must be 80 characters or fewer."},
        )
    return normalized


def _normalize_timezone(*, timezone_name: str) -> str:
    normalized = timezone_name.strip() or "UTC"
    if len(normalized) > 64 or any(char.isspace() for char in normalized):
        raise AccountSettingsOperationError(
            status_code=422,
            code="validation_error",
            message="Validation failed",
            field_errors={"timezone": "Timezone must be a compact IANA identifier."},
        )
    return normalized


def _validate_choice(*, field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value in allowed:
        return
    raise AccountSettingsOperationError(
        status_code=422,
        code="validation_error",
        message="Validation failed",
        field_errors={field_name: f"Unsupported {field_name}: {value}."},
    )


def _normalize_limit(*, limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_PAGE_LIMIT
    if limit < 1 or limit > _MAX_PAGE_LIMIT:
        raise AccountSettingsOperationError(
            status_code=422,
            code="validation_error",
            message="Validation failed",
            field_errors={"limit": "Limit must be between 1 and 100."},
        )
    return limit


def _to_session_view(*, session: IdentitySession, now: datetime) -> AccountSessionView:
    if session.revoked_at is not None and session.revoked_at <= now:
        status = "revoked"
    elif session.is_active_at(at=now):
        status = "active"
    else:
        status = "expired"
    return AccountSessionView(
        session_id=session.session_id,
        created_at=session.created_at,
        last_seen_at=session.last_seen_at,
        idle_expires_at=session.idle_expires_at,
        absolute_expires_at=session.absolute_expires_at,
        revoked_at=session.revoked_at,
        status=status,
    )


def _limits_for_paid_level(*, paid_level: PaidLevel) -> dict[str, int]:
    value = str(paid_level)
    if value == "ultra":
        return {"exchange_keys": 25, "active_strategies": 200, "webhook_events": 5000}
    if value == "pro":
        return {"exchange_keys": 10, "active_strategies": 50, "webhook_events": 1000}
    if value == "base":
        return {"exchange_keys": 5, "active_strategies": 10, "webhook_events": 250}
    return {"exchange_keys": 2, "active_strategies": 3, "webhook_events": 50}


def _encode_session_cursor(*, session: IdentitySession) -> str:
    return _encode_cursor(
        payload={
            "created_at": session.created_at.isoformat(),
            "id": str(session.session_id),
        }
    )


def _decode_session_cursor(*, raw_cursor: str | None) -> AccountSessionCursor | None:
    payload = _decode_cursor(raw_cursor=raw_cursor)
    if payload is None:
        return None
    try:
        return AccountSessionCursor(
            created_at=_parse_cursor_datetime(value=str(payload["created_at"])),
            session_id=UUID(str(payload["id"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _invalid_cursor_error() from error


def _encode_audit_cursor(*, event: AccountAuditEvent) -> str:
    return _encode_cursor(
        payload={
            "created_at": event.created_at.isoformat(),
            "id": str(event.event_id),
        }
    )


def _decode_audit_cursor(*, raw_cursor: str | None) -> AccountAuditCursor | None:
    payload = _decode_cursor(raw_cursor=raw_cursor)
    if payload is None:
        return None
    try:
        return AccountAuditCursor(
            created_at=_parse_cursor_datetime(value=str(payload["created_at"])),
            event_id=UUID(str(payload["id"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _invalid_cursor_error() from error


def _encode_cursor(*, payload: Mapping[str, str]) -> str:
    raw_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw_payload).decode("ascii").rstrip("=")


def _decode_cursor(*, raw_cursor: str | None) -> dict[str, Any] | None:
    if raw_cursor is None or not raw_cursor.strip():
        return None
    try:
        padded = raw_cursor.strip() + "=" * (-len(raw_cursor.strip()) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise _invalid_cursor_error() from error
    if not isinstance(payload, dict):
        raise _invalid_cursor_error()
    return payload


def _parse_cursor_datetime(*, value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("cursor datetime must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _invalid_cursor_error() -> AccountSettingsOperationError:
    return AccountSettingsOperationError(
        status_code=422,
        code="validation_error",
        message="Validation failed",
        field_errors={"cursor": "Cursor is invalid."},
    )
