from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

UserNotificationMode = Literal["off", "critical_only", "signals", "trades", "reports", "all"]

_DEFAULT_TIMEZONE = "UTC"
_DEFAULT_UNBOUND_RECIPIENT_REF = "telegram_ref:unbound"


@dataclass(frozen=True, slots=True)
class UserNotificationReportSchedule:
    weekly_enabled: bool
    monthly_enabled: bool
    timezone: str


@dataclass(frozen=True, slots=True)
class UserNotificationSettingsView:
    route_id: UUID
    mode: UserNotificationMode
    status: Literal["active", "paused", "requires_rebind", "disabled"]
    recipient_address_ref: str | None
    report_schedule: UserNotificationReportSchedule
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class UserNotificationSettingsUpdate:
    mode: UserNotificationMode
    weekly_enabled: bool
    monthly_enabled: bool
    timezone: str | None = None
    recipient_address_ref: str | None = None


@dataclass(frozen=True, slots=True)
class UserNotificationSettingsService:
    repository: NotificationRepository

    def get_settings(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        now: datetime,
        default_timezone: str | None,
    ) -> UserNotificationSettingsView:
        route = self.repository.get_route(
            organization_id=organization_id,
            route_id=_user_route_id(
                organization_id=organization_id, owner_user_id=owner_user_id
            ),
        )
        if route is None:
            timezone = _normalize_timezone(default_timezone)
            return UserNotificationSettingsView(
                route_id=_user_route_id(
                    organization_id=organization_id, owner_user_id=owner_user_id
                ),
                mode="off",
                status="disabled",
                recipient_address_ref=None,
                report_schedule=UserNotificationReportSchedule(
                    weekly_enabled=False,
                    monthly_enabled=False,
                    timezone=timezone,
                ),
                updated_at=now,
            )
        return _settings_view(route=route, fallback_timezone=default_timezone)

    def update_settings(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        owner_user_id: UserId,
        update: UserNotificationSettingsUpdate,
        now: datetime,
        default_timezone: str | None,
    ) -> UserNotificationSettingsView:
        route_id = _user_route_id(
            organization_id=organization_id, owner_user_id=owner_user_id
        )
        existing = self.repository.get_route(
            organization_id=organization_id, route_id=route_id
        )
        timezone = _normalize_timezone(update.timezone or default_timezone)
        recipient_address_ref = (
            update.recipient_address_ref
            or (existing.recipient_address_ref if existing is not None else None)
            or _DEFAULT_UNBOUND_RECIPIENT_REF
        )
        status = _route_status(mode=update.mode, recipient_address_ref=recipient_address_ref)
        route = NotificationRoute(
            route_id=route_id,
            organization_id=organization_id,
            provider_instance_id=provider_instance_id,
            recipient_kind="user",
            owner_user_id=owner_user_id,
            channel_key="telegram",
            provider_key="telegram_bot_api",
            mode=update.mode,
            category_filter=(),
            scope_filter_json={},
            schedule_json={
                "weekly": {"enabled": update.weekly_enabled},
                "monthly": {"enabled": update.monthly_enabled},
                "timezone": timezone,
            },
            recipient_address_ref=recipient_address_ref,
            status=status,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        saved = self.repository.upsert_route(route=route)
        return _settings_view(route=saved, fallback_timezone=timezone)


def _user_route_id(*, organization_id: OrganizationId, owner_user_id: UserId) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"roehub:notifications:user-route:{organization_id}:{owner_user_id}",
    )


def _settings_view(
    *,
    route: NotificationRoute,
    fallback_timezone: str | None,
) -> UserNotificationSettingsView:
    schedule = route.schedule_json
    return UserNotificationSettingsView(
        route_id=route.route_id,
        mode=route.mode,
        status=route.status,
        recipient_address_ref=route.recipient_address_ref,
        report_schedule=UserNotificationReportSchedule(
            weekly_enabled=_schedule_enabled(schedule.get("weekly")),
            monthly_enabled=_schedule_enabled(schedule.get("monthly")),
            timezone=_normalize_timezone(schedule.get("timezone") or fallback_timezone),
        ),
        updated_at=route.updated_at,
    )


def _schedule_enabled(value: object) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled"))
    return False


def _normalize_timezone(value: object) -> str:
    if not isinstance(value, str):
        return _DEFAULT_TIMEZONE
    normalized = value.strip()
    return normalized or _DEFAULT_TIMEZONE


def _route_status(
    *,
    mode: UserNotificationMode,
    recipient_address_ref: str,
) -> Literal["active", "requires_rebind", "disabled"]:
    if mode == "off":
        return "disabled"
    if recipient_address_ref == _DEFAULT_UNBOUND_RECIPIENT_REF:
        return "requires_rebind"
    return "active"
