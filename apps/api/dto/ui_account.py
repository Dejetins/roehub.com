from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

"""
UI account API contract.

Browser-visible paths are `/api/ui/account/...` through the web proxy. Backend
router paths are `/ui/account/...`, without a second `/api` prefix. All endpoints
are owner-scoped by `RequireCurrentUserDependency`; mutation endpoints require
the UI account same-origin guard. DTO additions are compatible-change and do not
affect request hashes or cache identity.
"""

ThemeLiteral = Literal["terminal-orange", "graphite", "matrix-green", "high-contrast"]
LocaleLiteral = Literal["en", "ru"]
DensityLiteral = Literal["compact", "comfortable"]
IntegrationProviderLiteral = Literal["telegram", "email_digest", "webhook_alerts"]
SessionStatusLiteral = Literal["active", "expired", "revoked"]


class AccountProfileResponse(BaseModel):
    user_id: str
    paid_level: str
    display_name: str | None
    timezone: str
    updated_at: datetime | None


class UpdateAccountProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=80)
    timezone: str = Field(default="UTC", min_length=1, max_length=64)


class AccountLimitsResponse(BaseModel):
    paid_level: str
    exchange_keys_used: int
    exchange_keys_limit: int
    active_strategies_used: int
    active_strategies_limit: int
    webhook_events_used: int
    webhook_events_limit: int


class AccountPreferencesResponse(BaseModel):
    theme: ThemeLiteral
    locale: LocaleLiteral
    density: DensityLiteral
    updated_at: datetime


class UpdateAccountPreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: ThemeLiteral | None = None
    locale: LocaleLiteral | None = None
    density: DensityLiteral | None = None


class AccountNotificationsResponse(BaseModel):
    email_notifications_enabled: bool
    trade_alerts_enabled: bool
    product_updates_enabled: bool
    updated_at: datetime


class UpdateAccountNotificationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_notifications_enabled: bool
    trade_alerts_enabled: bool
    product_updates_enabled: bool


class AccountIntegrationResponse(BaseModel):
    provider: IntegrationProviderLiteral
    enabled: bool
    updated_at: datetime


class AccountIntegrationsResponse(BaseModel):
    integrations: list[AccountIntegrationResponse]


class UpdateAccountIntegrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: IntegrationProviderLiteral
    enabled: bool


class UpdateAccountIntegrationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integrations: list[UpdateAccountIntegrationItem] = Field(default_factory=list)


class AccountSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    revoked_at: datetime | None
    status: SessionStatusLiteral


class AccountSessionsResponse(BaseModel):
    items: list[AccountSessionResponse]
    next_cursor: str | None


class AccountAuditEventResponse(BaseModel):
    event_id: str
    event_type: str
    metadata: dict[str, object]
    created_at: datetime


class AccountAuditEventsResponse(BaseModel):
    items: list[AccountAuditEventResponse]
    next_cursor: str | None
