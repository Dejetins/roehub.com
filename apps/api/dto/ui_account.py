from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ThemeValue = Literal["terminal-orange", "graphite", "matrix-green", "high-contrast"]
LocaleValue = Literal["en", "ru"]
DensityValue = Literal["compact", "comfortable"]
AutorefreshPresetValue = Literal["off", "10s", "15s", "30s", "1m", "5m", "custom"]


class AccountProfileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    username: str | None
    email: str | None
    timezone: str
    locale: LocaleValue
    telegram_discord: str | None
    subscription_status: str
    updated_at: datetime


class UpdateAccountProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str | None = Field(default=None, max_length=80)
    email: str | None = Field(default=None, max_length=160)
    timezone: str | None = Field(default=None, max_length=80)
    telegram_discord: str | None = Field(default=None, max_length=160)


class AccountLimitsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: str
    exchange_connections_used: int
    exchange_connections_limit: int
    api_keys_used: int
    api_keys_limit: int
    active_strategies_used: int
    active_strategies_limit: int
    webhook_events_used: int
    webhook_events_limit: int


class CreateExchangeConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exchange_name: Literal["binance", "bybit"]
    market_type: Literal["spot", "futures"]
    environment: Literal["mainnet", "testnet"] = "mainnet"
    label: str | None = Field(default=None, max_length=80)
    permissions: Literal["read", "trade"] = "read"
    api_key: str = Field(min_length=1)
    api_secret: str = Field(min_length=1)
    passphrase: str | None = None


class RotateExchangeConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)
    api_secret: str = Field(min_length=1)
    passphrase: str | None = None


class ExchangeConnectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    credential_version_id: str
    exchange_name: Literal["binance", "bybit"]
    market_type: Literal["spot", "futures"]
    environment: Literal["mainnet", "testnet"]
    label: str | None
    permissions: Literal["read", "trade"]
    api_key: str
    status: Literal["active", "disabled"]
    status_reason: str | None
    validation_status: Literal[
        "valid_readonly",
        "valid_trade_enabled",
        "invalid_credentials",
        "invalid_permissions",
        "invalid_ip_restriction",
        "unsupported_account_mode",
        "skipped_external_validation",
    ]
    validation_reason: str | None
    ip_restriction_status: str
    last_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None


class ExchangeConnectionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ExchangeConnectionResponse]
    next_cursor: str | None = None


class AccountIntegrationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integration_key: Literal["telegram", "discord", "slack"]
    label: str
    status: Literal["connected", "disconnected"]
    mode: Literal["off", "alerts", "critical"]
    webhook_url_masked: str | None
    updated_at: datetime


class UpdateAccountIntegrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integration_key: Literal["telegram", "discord", "slack"]
    mode: Literal["off", "alerts", "critical"]
    webhook_url: str | None = Field(default=None, max_length=512)


class AccountIntegrationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AccountIntegrationResponse]


class AccountNotificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_key: Literal[
        "telegram",
        "email",
        "push",
        "trade_fills",
        "risk_alerts",
        "daily_report",
        "system",
    ]
    label: str
    mode: Literal["off", "on", "critical"]
    updated_at: datetime


class UpdateAccountNotificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_key: Literal[
        "telegram",
        "email",
        "push",
        "trade_fills",
        "risk_alerts",
        "daily_report",
        "system",
    ]
    mode: Literal["off", "on", "critical"]


class AccountNotificationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AccountNotificationResponse]


class AccountAutorefreshPreferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_key: AutorefreshPresetValue
    refresh_interval_seconds: int
    allowed_presets: list[Literal["off", "10s", "15s", "30s", "1m", "5m"]]
    min_custom_interval_seconds: int
    max_custom_interval_seconds: int


class AccountPreferencesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: ThemeValue
    locale: LocaleValue
    density: DensityValue
    autorefresh: AccountAutorefreshPreferenceResponse
    updated_at: datetime


class UpdateAccountPreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    locale: str
    density: str = "compact"
    autorefresh_preset: str
    refresh_interval_seconds: int | None = Field(default=None, ge=0, le=3600)


class AccountSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    revoked_at: datetime | None
    device: str
    ip_address: str
    location: str
    is_current: bool


class AccountSessionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AccountSessionResponse]
    next_cursor: str | None


class AccountAuditEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    created_at: datetime
    event_type: str
    summary: str
    metadata: dict[str, str]


class AccountAuditEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AccountAuditEventResponse]
    next_cursor: str | None
