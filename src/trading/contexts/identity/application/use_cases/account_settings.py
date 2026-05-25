from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from trading.contexts.identity.application.ports.account_settings_repository import (
    AccountAuditEvent,
    AccountIntegrationSettings,
    AccountNotificationSettings,
    AccountPreferences,
    AccountProfileSettings,
    AccountSessionView,
    AccountSettingsRepository,
    AutorefreshPreset,
    CursorPage,
    DensityPreference,
    IntegrationMode,
    LocalePreference,
    NotificationMode,
    ThemePreference,
)
from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.shared_kernel.primitives import UserId

_THEMES: set[str] = {"terminal-orange", "graphite", "matrix-green", "high-contrast"}
_LOCALES: set[str] = {"en", "ru"}
_DENSITIES: set[str] = {"compact", "comfortable"}
_AUTOREFRESH_PRESETS: Mapping[str, int] = {
    "off": 0,
    "10s": 10,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "5m": 300,
}
_MIN_CUSTOM_AUTOREFRESH_SECONDS = 10
_MAX_CUSTOM_AUTOREFRESH_SECONDS = 1800


class AccountSettingsValidationError(ValueError):
    def __init__(self, *, code: str, message: str, field: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field


@dataclass(frozen=True, slots=True)
class AccountSettingsUseCase:
    repository: AccountSettingsRepository
    clock: IdentityClock

    def get_profile(self, *, owner_user_id: UserId) -> AccountProfileSettings:
        return self.repository.get_profile(owner_user_id=owner_user_id, now=self.clock.now())

    def update_profile(
        self,
        *,
        owner_user_id: UserId,
        username: str | None,
        email: str | None,
        timezone: str | None,
        telegram_discord: str | None,
    ) -> AccountProfileSettings:
        now = self.clock.now()
        profile = self.repository.save_profile(
            owner_user_id=owner_user_id,
            username=_optional_trim(username),
            email=_optional_trim(email),
            timezone=_normalize_timezone(timezone),
            telegram_discord=_optional_trim(telegram_discord),
            updated_at=now,
        )
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="profile_updated",
            summary="Profile settings updated",
            metadata={"surface": "settings"},
            created_at=now,
        )
        return profile

    def get_preferences(self, *, owner_user_id: UserId) -> AccountPreferences:
        return self.repository.get_preferences(owner_user_id=owner_user_id, now=self.clock.now())

    def update_preferences(
        self,
        *,
        owner_user_id: UserId,
        theme: str,
        locale: str,
        density: str,
        autorefresh_preset: str,
        refresh_interval_seconds: int | None,
    ) -> AccountPreferences:
        now = self.clock.now()
        normalized_preset, normalized_interval = _normalize_autorefresh(
            preset=autorefresh_preset,
            refresh_interval_seconds=refresh_interval_seconds,
        )
        preferences = self.repository.save_preferences(
            owner_user_id=owner_user_id,
            theme=_normalize_theme(theme),
            locale=_normalize_locale(locale),
            density=_normalize_density(density),
            autorefresh_preset=normalized_preset,
            refresh_interval_seconds=normalized_interval,
            updated_at=now,
        )
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="preferences_updated",
            summary="Account preferences updated",
            metadata={
                "theme": preferences.theme,
                "locale": preferences.locale,
                "autorefresh": preferences.autorefresh_preset,
            },
            created_at=now,
        )
        return preferences

    def get_limits(
        self,
        *,
        owner_user_id: UserId,
        plan: str,
        exchange_connections_used: int,
        api_keys_used: int,
    ) -> dict[str, int | str]:
        _ = owner_user_id
        return {
            "plan": plan,
            "exchange_connections_used": exchange_connections_used,
            "exchange_connections_limit": 10,
            "api_keys_used": api_keys_used,
            "api_keys_limit": 10,
            "active_strategies_used": 0,
            "active_strategies_limit": 50,
            "webhook_events_used": 0,
            "webhook_events_limit": 100,
        }

    def list_integrations(
        self,
        *,
        owner_user_id: UserId,
    ) -> tuple[AccountIntegrationSettings, ...]:
        return self.repository.list_integrations(owner_user_id=owner_user_id, now=self.clock.now())

    def update_integration(
        self,
        *,
        owner_user_id: UserId,
        integration_key: str,
        mode: str,
        webhook_url: str | None,
    ) -> AccountIntegrationSettings:
        now = self.clock.now()
        integration = self.repository.save_integration(
            owner_user_id=owner_user_id,
            integration_key=_normalize_integration_key(integration_key),
            mode=_normalize_integration_mode(mode),
            webhook_url_masked=_mask_webhook_url(webhook_url),
            updated_at=now,
        )
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="integration_updated",
            summary=f"Integration {integration.integration_key} updated",
            metadata={"integration": integration.integration_key, "mode": integration.mode},
            created_at=now,
        )
        return integration

    def list_notifications(
        self,
        *,
        owner_user_id: UserId,
    ) -> tuple[AccountNotificationSettings, ...]:
        return self.repository.list_notifications(owner_user_id=owner_user_id, now=self.clock.now())

    def update_notification(
        self,
        *,
        owner_user_id: UserId,
        channel_key: str,
        mode: str,
    ) -> AccountNotificationSettings:
        now = self.clock.now()
        notification = self.repository.save_notification(
            owner_user_id=owner_user_id,
            channel_key=_normalize_channel_key(channel_key),
            mode=_normalize_notification_mode(mode),
            updated_at=now,
        )
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="notifications_updated",
            summary=f"Notification {notification.channel_key} updated",
            metadata={"channel": notification.channel_key, "mode": notification.mode},
            created_at=now,
        )
        return notification

    def list_sessions(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
    ) -> CursorPage[AccountSessionView]:
        return self.repository.list_sessions(
            owner_user_id=owner_user_id,
            cursor=cursor,
            limit=_normalize_limit(limit),
            now=self.clock.now(),
        )

    def list_audit_events(
        self,
        *,
        owner_user_id: UserId,
        cursor: str | None,
        limit: int,
    ) -> CursorPage[AccountAuditEvent]:
        return self.repository.list_audit_events(
            owner_user_id=owner_user_id,
            cursor=cursor,
            limit=_normalize_limit(limit),
        )

    def record_exchange_connection_validation(
        self,
        *,
        owner_user_id: UserId,
        exchange_name: str,
        validation_status: str,
    ) -> None:
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="exchange_connection_validated",
            summary="Exchange connection validation completed",
            metadata={
                "exchange": exchange_name,
                "validation_status": validation_status,
            },
            created_at=self.clock.now(),
        )

    def record_exchange_connection_archive(
        self,
        *,
        owner_user_id: UserId,
        connection_id: str,
        exchange_name: str,
        market_type: str,
        environment: str,
        previous_status: str,
        new_status: str,
        reason: str,
    ) -> None:
        self.repository.append_audit_event(
            owner_user_id=owner_user_id,
            event_type="exchange_connection_archived",
            summary="Exchange connection archived",
            metadata={
                "connection_id": connection_id,
                "exchange_name": exchange_name,
                "market_type": market_type,
                "environment": environment,
                "previous_status": previous_status,
                "new_status": new_status,
                "reason": reason,
            },
            created_at=self.clock.now(),
        )


def _optional_trim(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_timezone(value: str | None) -> str:
    normalized = _optional_trim(value)
    return normalized or "Europe/Moscow"


def _normalize_theme(value: str) -> ThemePreference:
    normalized = value.strip()
    if normalized not in _THEMES:
        raise AccountSettingsValidationError(
            code="unsupported_theme",
            message="Unsupported theme preference.",
            field="theme",
        )
    return normalized  # type: ignore[return-value]


def _normalize_locale(value: str) -> LocalePreference:
    normalized = value.strip().lower()
    if normalized not in _LOCALES:
        raise AccountSettingsValidationError(
            code="unsupported_locale",
            message="Unsupported locale preference.",
            field="locale",
        )
    return normalized  # type: ignore[return-value]


def _normalize_density(value: str) -> DensityPreference:
    normalized = value.strip().lower()
    if normalized not in _DENSITIES:
        raise AccountSettingsValidationError(
            code="unsupported_density",
            message="Unsupported density preference.",
            field="density",
        )
    return normalized  # type: ignore[return-value]


def _normalize_autorefresh(
    *,
    preset: str,
    refresh_interval_seconds: int | None,
) -> tuple[AutorefreshPreset, int]:
    normalized_preset = preset.strip().lower()
    if normalized_preset in _AUTOREFRESH_PRESETS:
        return (
            normalized_preset,  # type: ignore[return-value]
            _AUTOREFRESH_PRESETS[normalized_preset],
        )
    if normalized_preset != "custom":
        raise AccountSettingsValidationError(
            code="unsupported_autorefresh_preset",
            message="Unsupported autorefresh preset.",
            field="autorefresh_preset",
        )
    if refresh_interval_seconds is None:
        raise AccountSettingsValidationError(
            code="missing_custom_interval",
            message="Custom autorefresh interval is required.",
            field="refresh_interval_seconds",
        )
    if refresh_interval_seconds < _MIN_CUSTOM_AUTOREFRESH_SECONDS:
        raise AccountSettingsValidationError(
            code="autorefresh_interval_too_low",
            message="Autorefresh interval is below the safe minimum.",
            field="refresh_interval_seconds",
        )
    if refresh_interval_seconds > _MAX_CUSTOM_AUTOREFRESH_SECONDS:
        raise AccountSettingsValidationError(
            code="autorefresh_interval_too_high",
            message="Autorefresh interval is above the supported maximum.",
            field="refresh_interval_seconds",
        )
    return "custom", refresh_interval_seconds


def _normalize_integration_key(value: str) -> Literal["telegram", "discord", "slack"]:
    normalized = value.strip().lower()
    if normalized not in {"telegram", "discord", "slack"}:
        raise AccountSettingsValidationError(
            code="unsupported_integration",
            message="Unsupported integration.",
            field="integration_key",
        )
    return normalized  # type: ignore[return-value]


def _normalize_integration_mode(value: str) -> IntegrationMode:
    normalized = value.strip().lower()
    if normalized not in {"off", "alerts", "critical"}:
        raise AccountSettingsValidationError(
            code="unsupported_integration_mode",
            message="Unsupported integration mode.",
            field="mode",
        )
    return normalized  # type: ignore[return-value]


def _normalize_channel_key(
    value: str,
) -> Literal["telegram", "email", "push", "trade_fills", "risk_alerts", "daily_report", "system"]:
    normalized = value.strip().lower()
    allowed = {
        "telegram",
        "email",
        "push",
        "trade_fills",
        "risk_alerts",
        "daily_report",
        "system",
    }
    if normalized not in allowed:
        raise AccountSettingsValidationError(
            code="unsupported_notification_channel",
            message="Unsupported notification channel.",
            field="channel_key",
        )
    return normalized  # type: ignore[return-value]


def _normalize_notification_mode(value: str) -> NotificationMode:
    normalized = value.strip().lower()
    if normalized not in {"off", "on", "critical"}:
        raise AccountSettingsValidationError(
            code="unsupported_notification_mode",
            message="Unsupported notification mode.",
            field="mode",
        )
    return normalized  # type: ignore[return-value]


def _normalize_limit(value: int) -> int:
    return min(max(value, 1), 50)


def _mask_webhook_url(value: str | None) -> str | None:
    normalized = _optional_trim(value)
    if normalized is None:
        return None
    if len(normalized) <= 12:
        return "****"
    return f"{normalized[:8]}...{normalized[-4:]}"
