from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceStatus = Literal["available", "degraded", "unavailable"]


class StrategyMonitoringSourceResponse(BaseModel):
    status: SourceStatus
    code: str
    message: str
    updated_at: str | None = None


class StrategyMonitoringLimitsResponse(BaseModel):
    strategies: int
    alerts: int
    positions: int
    fills: int
    equity_points: int


class StrategyMonitorItemResponse(BaseModel):
    strategy_id: str
    name: str
    state: str
    run_id: str | None
    instrument_key: str
    timeframe: str
    checkpoint_ts_open: str | None
    lag_seconds: int | None = Field(ge=0, default=None)
    updated_at: str


class StrategyMonitorResponse(BaseModel):
    source: StrategyMonitoringSourceResponse
    generated_at: str
    poll_interval_seconds: int
    items: list[StrategyMonitorItemResponse]
    selected_strategy_id: str | None
    next_cursor: str | None
    limits: StrategyMonitoringLimitsResponse
    links: dict[str, str]


class StrategySnapshotSpecResponse(BaseModel):
    instrument_key: str
    market_type: str
    timeframe: str
    signal_template: str


class StrategySnapshotRunResponse(BaseModel):
    run_id: str | None
    state: str
    started_at: str | None
    stopped_at: str | None
    checkpoint_ts_open: str | None
    updated_at: str | None
    last_error: str | None


class StrategySnapshotMetricResponse(BaseModel):
    key: str
    value: str
    tone: Literal["neutral", "positive", "negative"] = "neutral"
    updated_at: str | None = None


class StrategyMonitoringAlertResponse(BaseModel):
    alert_id: str
    severity: Literal["info", "warning", "critical"]
    title: str
    created_at: str


class StrategySnapshotResponse(BaseModel):
    source: StrategyMonitoringSourceResponse
    generated_at: str
    strategy_id: str
    name: str
    spec: StrategySnapshotSpecResponse
    run: StrategySnapshotRunResponse
    metrics: list[StrategySnapshotMetricResponse]
    alerts: list[StrategyMonitoringAlertResponse]
    links: dict[str, str]


class StrategyPositionItemResponse(BaseModel):
    position_id: str
    symbol: str
    side: str
    quantity: str
    entry_price: str | None
    unrealized_pnl: str | None
    updated_at: str


class StrategyPositionsResponse(BaseModel):
    source: StrategyMonitoringSourceResponse
    strategy_id: str
    limit: int
    items: list[StrategyPositionItemResponse]


class StrategyFillItemResponse(BaseModel):
    fill_id: str
    symbol: str
    side: str
    price: str
    quantity: str
    realized_pnl: str | None
    created_at: str


class StrategyFillsResponse(BaseModel):
    source: StrategyMonitoringSourceResponse
    strategy_id: str
    limit: int
    items: list[StrategyFillItemResponse]
    next_cursor: str | None


class StrategyEquityPointResponse(BaseModel):
    x: str
    value: float


class StrategyEquityResponse(BaseModel):
    source: StrategyMonitoringSourceResponse
    strategy_id: str
    range: str
    points: int
    items: list[StrategyEquityPointResponse]


__all__ = [
    "StrategyEquityResponse",
    "StrategyFillItemResponse",
    "StrategyFillsResponse",
    "StrategyMonitorItemResponse",
    "StrategyMonitorResponse",
    "StrategyMonitoringAlertResponse",
    "StrategyMonitoringLimitsResponse",
    "StrategyMonitoringSourceResponse",
    "StrategyPositionItemResponse",
    "StrategyPositionsResponse",
    "StrategySnapshotMetricResponse",
    "StrategySnapshotResponse",
    "StrategySnapshotRunResponse",
    "StrategySnapshotSpecResponse",
]
