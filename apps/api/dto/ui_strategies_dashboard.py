from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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


class StrategyDashboardLiveProfileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    mode: Literal["monitor_only", "paper", "live"]
    exchange_connection_id: str | None
    sizing_method: Literal["fixed_quote", "fixed_equity_pct"]
    sizing_value: Decimal
    max_position_notional: Decimal | None
    max_orders_per_run: int
    max_notional_per_run: Decimal
    readiness_status: Literal["ready", "blocked"]
    readiness_reason: str
    updated_at: datetime | None
    degradation_reason: str | None = None


class StrategyDashboardCompatibilityReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    compatibility_state: Literal["launchable", "not_launchable", "degraded"]
    compatibility_reason_codes: list[str]
    market_data_state: Literal["ready", "missing", "stale", "pending"]
    market_data_reason_codes: list[str]
    market_data_stream_name: str | None
    market_data_age_seconds: int | None
    launch_blocked: bool
    launch_blocked_reason: str
    checked_at: datetime | None
    degradation_reason: str | None = None


class StrategyDashboardExchangeAccountReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    status: Literal["fresh", "stale", "degraded", "config_mismatch"]
    reason_codes: list[str]
    exchange_connection_id: str | None
    instrument_key: str | None
    market_type: str | None
    account_snapshot_id: str | None
    config_guard_result_id: str | None
    age_seconds: int | None
    checked_at: datetime | None
    ready_for_risk: bool
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


class StrategySignalJournalRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str
    strategy_run_id: str
    live_profile_id: str | None
    mode: Literal["monitor_only", "paper", "live"]
    outcome: Literal["warmup", "no_signal", "signal", "blocked"]
    signal_action: Literal["none", "open", "close", "reduce", "reverse"]
    side: Literal["buy", "sell"] | None
    reason_code: str
    reference_price: Decimal
    instrument_key: str
    market_type: str
    timeframe: str
    bar_ts_open: datetime
    bar_ts_close: datetime
    source_message_id: str
    created_at: datetime | None


class StrategySignalJournalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    limit: int
    items: list[StrategySignalJournalRowResponse]
    degradation_reason: str | None = None


class StrategyDashboardPaperAccountingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    reserved_budget: Decimal | None
    position_quantity: Decimal | None
    average_entry_price: Decimal | None
    equity: Decimal | None
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    fee_total: Decimal | None
    funding_total: Decimal | None
    fee_model: str | None
    funding_model: str | None
    pnl_complete: bool
    completeness_reason: str
    updated_at: datetime | None
    degradation_reason: str | None = None


class StrategyExecutionOutcomeLinkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_id: str
    source_type: str
    source_event_ref: str
    strategy_signal_id: str | None
    outcome: str
    outcome_reason: str
    intent_id: str | None
    intent_status: str | None
    intent_status_reason: str | None
    risk_status: str | None
    risk_reason: str | None
    order_status: str | None
    order_status_reason: str | None
    notification_event_type: str | None
    notification_reason: str | None
    updated_at: datetime


class StrategyExecutionOutcomeLinksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: PanelState
    limit: int
    items: list[StrategyExecutionOutcomeLinkResponse]
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
    live_profile: StrategyDashboardLiveProfileResponse
    compatibility_readiness: StrategyDashboardCompatibilityReadinessResponse
    exchange_account_readiness: StrategyDashboardExchangeAccountReadinessResponse
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
    signal_journal: StrategySignalJournalResponse
    paper_accounting: StrategyDashboardPaperAccountingResponse
    execution_outcomes: StrategyExecutionOutcomeLinksResponse
    footer_status: StrategyDashboardFooterStatusResponse
    refresh_control: StrategyDashboardRefreshControlResponse
