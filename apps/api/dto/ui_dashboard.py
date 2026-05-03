from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DashboardSourceStatus = Literal["available", "degraded", "unavailable"]


class DashboardSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DashboardSourceStatus
    code: str
    message: str
    updated_at: str | None = None


class DashboardAccountResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: DashboardSourceResponse
    user_id: str
    paid_level: str


class DashboardStrategyItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    name: str
    state: str
    instrument_key: str | None = None
    timeframe: str | None = None
    updated_at: str | None = None


class DashboardStrategiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: DashboardSourceResponse
    total_count: int | None = Field(default=None, ge=0)
    active_count: int | None = Field(default=None, ge=0)
    items: list[DashboardStrategyItemResponse]


class DashboardBacktestJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: str
    pipeline_stage: str
    progress_percent: int = Field(ge=0, le=100)
    symbol: str | None = None
    timeframe: str | None = None
    risk_mode: str | None = None
    primary_metric: str | None = None
    updated_at: str
    links: dict[str, str]


class DashboardBacktestsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: DashboardSourceResponse
    active_count: int | None = Field(default=None, ge=0)
    items: list[DashboardBacktestJobResponse]
    next_cursor: str | None = None


class DashboardAlertResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str
    severity: str
    title: str
    created_at: str
    link: str | None = None


class DashboardAlertsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: DashboardSourceResponse
    items: list[DashboardAlertResponse]
    next_cursor: str | None = None


class DashboardSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    poll_interval_seconds: int = Field(ge=10, le=15)
    sources: dict[str, DashboardSourceResponse]
    account: DashboardAccountResponse
    strategies: DashboardStrategiesResponse
    backtests: DashboardBacktestsResponse
    alerts: DashboardAlertsResponse
    links: dict[str, str]


__all__ = [
    "DashboardAccountResponse",
    "DashboardAlertResponse",
    "DashboardAlertsResponse",
    "DashboardBacktestJobResponse",
    "DashboardBacktestsResponse",
    "DashboardSourceResponse",
    "DashboardSourceStatus",
    "DashboardStrategiesResponse",
    "DashboardStrategyItemResponse",
    "DashboardSummaryResponse",
]
