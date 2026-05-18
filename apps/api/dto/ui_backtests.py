from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

BacktestPanelState = Literal["ready", "empty", "degraded", "unavailable"]
BacktestRefreshStatus = Literal["fresh", "degraded", "rate_limited"]


class BacktestWorkstationSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["available", "degraded", "unavailable"]
    generated_at: str | None = None
    detail: str | None = None
    retry_after_seconds: int | None = None
    next_allowed_refresh_at: str | None = None


class BacktestConfigDraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinates: dict[str, Any]
    timeframe: str
    time_range: dict[str, str]
    indicators: list[dict[str, Any]]
    risk: dict[str, Any]
    execution: dict[str, Any]
    ranking: dict[str, Any]
    top_n: int


class BacktestOptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str
    status: Literal["available", "disabled"] = "available"


class BacktestInstrumentUniverseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: BacktestPanelState
    markets: list[BacktestOptionResponse]
    market_types: list[BacktestOptionResponse]
    symbols: list[BacktestOptionResponse]
    timeframes: list[BacktestOptionResponse]
    selected_symbols: list[str]
    degradation_reason: str | None = None


class BacktestIndicatorCatalogRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    label: str
    family: str
    min_value: float | int | None
    max_value: float | int | None
    step: float | int | None
    sources: list[str]
    param_specs: dict[str, Any]
    status: Literal["available", "disabled"] = "available"


class BacktestIndicatorCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: BacktestPanelState
    items: list[BacktestIndicatorCatalogRowResponse]
    total_combinations_estimate: int
    degradation_reason: str | None = None


class BacktestOptimizationOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: BacktestPanelState
    active_job_id: str | None
    progress_percent: int
    processed_units: int
    total_units: int
    completed_jobs: int
    running_jobs: int
    queued_jobs: int
    estimated_remaining: str | None
    degradation_reason: str | None = None


class BacktestRecentEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    level: Literal["info", "warning", "error"]
    message: str
    job_id: str | None = None


class BacktestRecentEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: BacktestPanelState
    items: list[BacktestRecentEventResponse]
    degradation_reason: str | None = None


class BacktestJobTableFiltersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str | None
    cursor: str | None
    query: str
    exchange: str | None
    market_type: str | None
    symbol: str | None
    launched_from: str | None
    launched_to: str | None
    limit: int
    sort: Literal["created_desc"]


class BacktestJobTableRowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: str
    strategy: str
    exchange: str
    market_type: str
    symbol: str
    created_at: str
    cancel_requested_at: str | None = None
    indicator_summary: str
    period: str
    direction: str
    combinations: int | None
    best_return_pct: float | None
    best_sharpe: float | None
    avg_drawdown_pct: float | None
    profit_factor: float | None
    win_rate_pct: float | None
    trades_count: int | None
    progress_percent: int
    refresh_status: str
    retry_after_seconds: int
    links: dict[str, Any]
    actions: dict[str, bool]


class BacktestJobTableResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    state: BacktestPanelState
    filters: BacktestJobTableFiltersResponse
    items: list[BacktestJobTableRowResponse]
    next_cursor: str | None = None
    degradation_reason: str | None = None


class BacktestFooterStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api: str
    worker: str
    queue: str
    generated_at: str
    data: str


class BacktestRefreshControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manual: bool
    autorefresh_presets: list[str]
    default_preset: str
    generated_at: str
    refresh_status: BacktestRefreshStatus
    next_allowed_refresh_at: str | None
    retry_after_seconds: int | None


class BacktestWorkstationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    refresh_status: BacktestRefreshStatus
    next_allowed_refresh_at: str | None
    retry_after_seconds: int | None
    sources: list[BacktestWorkstationSourceResponse]
    runtime_defaults: dict[str, Any]
    config_draft: BacktestConfigDraftResponse
    instrument_universe: BacktestInstrumentUniverseResponse
    indicator_catalog: BacktestIndicatorCatalogResponse
    optimization_overview: BacktestOptimizationOverviewResponse
    recent_events: BacktestRecentEventsResponse
    job_table: BacktestJobTableResponse
    footer_status: BacktestFooterStatusResponse
    refresh_control: BacktestRefreshControlResponse


__all__ = [
    "BacktestConfigDraftResponse",
    "BacktestFooterStatusResponse",
    "BacktestIndicatorCatalogResponse",
    "BacktestIndicatorCatalogRowResponse",
    "BacktestInstrumentUniverseResponse",
    "BacktestJobTableFiltersResponse",
    "BacktestJobTableResponse",
    "BacktestJobTableRowResponse",
    "BacktestOptionResponse",
    "BacktestOptimizationOverviewResponse",
    "BacktestRecentEventResponse",
    "BacktestRecentEventsResponse",
    "BacktestRefreshControlResponse",
    "BacktestWorkstationResponse",
    "BacktestWorkstationSourceResponse",
]
