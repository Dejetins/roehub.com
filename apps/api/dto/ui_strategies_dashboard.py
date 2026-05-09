from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceStatus = Literal["available", "degraded", "unavailable"]
RefreshStatus = Literal["fresh", "degraded", "rate_limited"]
FinancialDirection = Literal["positive", "negative", "neutral"]
PanelState = Literal["ready", "empty", "degraded", "unavailable"]
StrategyRunStatus = Literal["live", "stopped", "degraded", "unknown"]


class StrategyDashboardSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: SourceStatus
    generated_at: datetime | None = None
    age_seconds: int | None = None
    detail: str | None = None
    retry_after_seconds: int | None = None
    next_allowed_refresh_at: datetime | None = None


class StrategyDashboardActionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_create: bool
    can_clone: bool
    can_delete: bool
    can_run: bool
    can_stop: bool


class StrategyDashboardSelectedStrategyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    strategy_id: str | None
    name: str | None
    version: str | None
    exchange: str | None
    market_type: str | None
    symbols: list[str]
    timeframe: str | None
    direction: str | None
    capital_usdt: float | None
    commission_percent: float | None
    slippage_percent: float | None
    created_at: datetime | None
    updated_at: datetime | None
    status: StrategyRunStatus
    run_state: str | None
    latest_update: datetime | None
    actions: StrategyDashboardActionsResponse
    degradation_reason: str | None = None


class StrategyDashboardSelectorFiltersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["active", "all"]
    cursor: str | None
    limit: int
    query: str
    sort: Literal["updated", "name", "status"]


class StrategyDashboardSelectorTotalsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: int
    active: int
    stopped: int
    degraded: int
    symbols: int


class StrategyDashboardSelectorRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    name: str
    version: str | None
    exchange: str | None
    market_type: str | None
    symbols: list[str]
    timeframe: str | None
    status: StrategyRunStatus
    run_state: str | None
    latest_activity: datetime | None


class StrategyDashboardSelectorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    filters: StrategyDashboardSelectorFiltersResponse
    totals: StrategyDashboardSelectorTotalsResponse
    items: list[StrategyDashboardSelectorRowResponse]
    selected_strategy_id: str | None
    next_cursor: str | None = None
    degradation_reason: str | None = None


class StrategyChartCandleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None


class StrategyChartMarkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    label: str
    side: Literal["buy", "sell", "tp", "sl"]
    price: float | None


class StrategyChartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    symbol: str | None
    range: Literal["1y", "6m", "1m"]
    max_points: int
    candles: list[StrategyChartCandleResponse]
    markers: list[StrategyChartMarkerResponse]
    degradation_reason: str | None = None


class StrategyMetricResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    value: float | int | None
    formatted: str
    direction: FinancialDirection
    status: SourceStatus
    source: str


class StrategyMetricGridResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    items: list[StrategyMetricResponse]
    degradation_reason: str | None = None


class StrategyMonthlyStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    columns: list[str]
    rows: list[dict[str, str | float | int | None]]
    summary: list[StrategyMetricResponse]
    degradation_reason: str | None = None


class StrategyBreakdownRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    long_value: str | None = None
    short_value: str | None = None
    total_value: str | None = None
    direction: FinancialDirection = "neutral"


class StrategyBreakdownPanelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    rows: list[StrategyBreakdownRowResponse]
    degradation_reason: str | None = None


class StrategySeriesPointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    value: float | None


class StrategySeriesPanelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    title: str
    max_points: int
    points: list[StrategySeriesPointResponse]
    degradation_reason: str | None = None


class StrategyHourlyResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour_bucket: str
    win_rate_percent: float | None
    pnl_percent: float | None
    direction: FinancialDirection


class StrategyHourlyResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    items: list[StrategyHourlyResultResponse]
    total: StrategyHourlyResultResponse | None = None
    degradation_reason: str | None = None


class StrategyTradeRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_number: int
    symbol: str
    side: Literal["long", "short"]
    entry_time: datetime | None
    entry: float | None
    exit_time: datetime | None
    exit: float | None
    pnl_percent: float | None
    pnl_usdt: float | None
    bars: int | None
    hold_time: str | None
    phase: str | None
    reason: str | None
    note: str | None


class StrategyTradesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    limit: int
    items: list[StrategyTradeRowResponse]
    next_cursor: str | None = None
    degradation_reason: str | None = None


class StrategySymbolResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    trades: int | None
    win_rate_percent: float | None
    pnl_percent: float | None
    pnl_usdt: float | None
    direction: FinancialDirection


class StrategySymbolResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    items: list[StrategySymbolResultResponse]
    total: StrategySymbolResultResponse | None = None
    degradation_reason: str | None = None


class StrategyDashboardFooterStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_status: Literal["ok", "degraded", "unknown"]
    data_status: Literal["actual", "stale", "degraded", "unknown"]
    api_label: str
    latency_ms: int | None
    capital_usdt: float | None
    server_time: datetime


class StrategyDashboardRefreshControlResponse(BaseModel):
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


class StrategyDashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    refresh_status: RefreshStatus
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None
    sources: list[StrategyDashboardSourceResponse]
    selected_strategy: StrategyDashboardSelectedStrategyResponse
    strategy_selector: StrategyDashboardSelectorResponse
    chart: StrategyChartResponse
    metric_grid: StrategyMetricGridResponse
    monthly_stats: StrategyMonthlyStatsResponse
    long_short: StrategyBreakdownPanelResponse
    risk_execution: StrategyBreakdownPanelResponse
    drawdown: StrategySeriesPanelResponse
    equity_curve: StrategySeriesPanelResponse
    hourly_results: StrategyHourlyResultsResponse
    trades: StrategyTradesResponse
    symbol_results: StrategySymbolResultsResponse
    footer_status: StrategyDashboardFooterStatusResponse
    refresh_control: StrategyDashboardRefreshControlResponse
