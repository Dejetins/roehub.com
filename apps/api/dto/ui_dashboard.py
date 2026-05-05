from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceStatus = Literal["available", "degraded", "unavailable"]
RefreshStatus = Literal["fresh", "degraded", "rate_limited"]
FinancialDirection = Literal["positive", "negative", "neutral"]
PanelState = Literal["ready", "empty", "degraded", "unavailable"]


class DashboardSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: SourceStatus
    generated_at: datetime | None = None
    age_seconds: int | None = None
    detail: str | None = None
    retry_after_seconds: int | None = None
    next_allowed_refresh_at: datetime | None = None


class DashboardStrategyActionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_start: bool
    can_stop: bool
    can_restart: bool
    can_open_settings: bool


class DashboardSelectedStrategySnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    strategy_id: str | None
    name: str | None
    version: str | None
    exchange: str | None
    symbols: list[str]
    direction: str | None
    mode: str | None
    timeframe: str | None
    capital: str | None
    leverage: str | None
    status: Literal["live", "paper", "stopped", "degraded", "unknown"]
    latest_update: datetime | None
    uptime_seconds: int | None
    actions: DashboardStrategyActionsResponse
    degradation_reason: str | None = None


class DashboardEquityPnlPointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    equity: float | None
    total_pnl: float | None
    realized_pnl: float | None
    unrealized_pnl: float | None
    marker: Literal["buy", "sell"] | None = None


class DashboardEquityPnlSeriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: Literal["ready", "empty", "degraded", "unavailable"]
    range: Literal["1d", "6h", "1h"]
    max_points: int
    points: list[DashboardEquityPnlPointResponse]
    degradation_reason: str | None = None


class DashboardMetricResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    value: float | int | None
    formatted: str
    direction: FinancialDirection
    status: SourceStatus
    source: str


class DashboardPositionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: Literal["long", "short"]
    entry: float | None
    mark: float | None
    pnl: float | None
    pnl_percent: float | None
    roe_percent: float | None
    leverage: float | None
    opened_at: datetime | None
    can_close: bool


class DashboardPositionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    limit: int
    items: list[DashboardPositionResponse]
    degradation_reason: str | None = None


class DashboardExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    symbol: str
    side: Literal["buy", "sell"]
    price: float | None
    quantity: float | None
    fee: float | None
    realized_pnl: float | None
    reason: str | None


class DashboardExecutionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    limit: int
    items: list[DashboardExecutionResponse]
    next_cursor: str | None = None
    degradation_reason: str | None = None


class DashboardHealthCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    state: Literal["ok", "warn", "error", "unknown"]
    value: str
    ratio: float | None = None
    source: str


class DashboardHealthRiskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: Literal["ok", "warn", "error", "unknown"]
    checks: list[DashboardHealthCheckResponse]
    degradation_reason: str | None = None


class DashboardAlertResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    severity: Literal["info", "warn", "error"]
    message: str
    source: str
    strategy_id: str | None
    acknowledged: bool


class DashboardAlertsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    items: list[DashboardAlertResponse]
    next_cursor: str | None = None
    degradation_reason: str | None = None


class DashboardSymbolAllocationItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    pnl: float | None
    pnl_percent: float | None
    share_percent: float | None
    bar_ratio: float | None
    direction: FinancialDirection


class DashboardSymbolAllocationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    items: list[DashboardSymbolAllocationItemResponse]
    degradation_reason: str | None = None


class DashboardStrategyListFiltersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    exchange: str
    mode: str
    query: str
    sort: Literal["pnl", "activity", "name", "open_positions"]


class DashboardStrategyListTotalsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    running: int
    stopped: int
    degraded: int
    symbols: int
    strategies: int
    open_positions: int | None


class DashboardStrategyListRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    name: str
    version: str | None
    exchange: str | None
    symbols: list[str]
    latest_activity: datetime | None
    pnl: float | None
    pnl_percent: float | None
    mode: str | None
    open_positions: int | None
    status: Literal["live", "paper", "stopped", "degraded", "unknown"]
    mini_sparkline: list[float]
    sparkline_state: Literal["ready", "empty", "degraded", "unavailable"]


class DashboardStrategyListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    filters: DashboardStrategyListFiltersResponse
    totals: DashboardStrategyListTotalsResponse
    items: list[DashboardStrategyListRowResponse]
    next_cursor: str | None = None
    degradation_reason: str | None = None


class DashboardFooterStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_status: Literal["ok", "degraded", "unknown"]
    account_tier: str
    mode: str
    api_label: str
    latency_ms: int | None
    server_time: datetime


class DashboardRefreshControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manual_refresh_available: bool
    autorefresh_enabled: bool
    interval_seconds: int
    preset_key: str
    generated_at: datetime
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None
    last_refresh_reason: Literal["initial", "auto", "manual"]
    refresh_status: RefreshStatus


class DashboardSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    refresh_status: RefreshStatus
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None
    sources: list[DashboardSourceResponse]
    selected_strategy_snapshot: DashboardSelectedStrategySnapshotResponse
    equity_pnl_series: DashboardEquityPnlSeriesResponse
    metric_grid: list[DashboardMetricResponse]
    open_positions: DashboardPositionsResponse
    recent_executions: DashboardExecutionsResponse
    health_risk: DashboardHealthRiskResponse
    alerts: DashboardAlertsResponse
    symbol_allocation: DashboardSymbolAllocationResponse
    strategy_list: DashboardStrategyListResponse
    footer_status: DashboardFooterStatusResponse
    refresh_control: DashboardRefreshControlResponse
