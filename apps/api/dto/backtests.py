from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from trading.contexts.backtest.application.dto import (
    BacktestJobCreateResult,
    BacktestJobListResult,
    BacktestJobReadModel,
    BacktestJobTopResult,
    BacktestJobTopVariantReadModel,
    BacktestLazyTradesDetailReadModel,
    BacktestLazyTradesMaterializationReadModel,
    BacktestLazyTradesResultReadModel,
    BacktestPreflightResult,
    BacktestRuntimeDefaults,
)


def _public_cache_metadata(value: Any) -> dict[str, Any]:
    """Expose cache status/identity/TTL without filesystem paths or raw IO errors."""
    if not isinstance(value, dict):
        raise ValueError("cache metadata must be an object")
    return {key: value[key] for key in ("status", "cache_key", "ttl_seconds", "ttl_hours")
            if key in value}


PublicBacktestCacheMetadata = Annotated[
    dict[str, Any], BeforeValidator(_public_cache_metadata),
]


def _public_materialization_metadata(value: Any) -> dict[str, Any]:
    """Keep task state public while retaining raw failure diagnostics internally."""
    if not isinstance(value, dict):
        raise ValueError("materialization metadata must be an object")
    public = {key: value[key] for key in (
        "task_id", "correlation_id", "status", "retryable", "retry_after_seconds",
        "priority_class", "created_at", "updated_at", "started_at", "finished_at",
        "attempt", "request_identity",
    ) if key in value}
    failed = value.get("status") == "failed"
    public["last_error"] = "Result preparation failed." if failed else None
    public["last_error_json"] = {"code": "backtest.materialization_failed"} if failed else None
    return public


PublicBacktestMaterializationMetadata = Annotated[
    dict[str, Any], BeforeValidator(_public_materialization_metadata),
]


class BacktestRuntimeDefaultsResponse(BaseModel):
    """
    API response model for `GET /backtests/runtime-defaults`.
    """

    supported_timeframes: list[str]
    risk_modes: list[str]
    direction_modes: list[str]
    sizing_modes: list[str]
    ranking_metrics: list[str]
    ranking_default: dict[str, Any]
    top_n_default: int
    quality_constraints_default: dict[str, Any]
    guardrails: dict[str, Any]
    execution_defaults: dict[str, Any]
    supported_indicator_ids: list[str]
    indicator_sources: dict[str, list[str]]
    indicator_param_specs: dict[str, Any]
    hit_times_grid: dict[str, Any]
    direction_market_compatibility: dict[str, Any]
    links: dict[str, Any]


class BacktestInputReadinessResponse(BaseModel):
    """Bounded metadata assessment; checksum verification still precedes worker execution."""

    status: Literal["ready", "requires_materialization"]
    requested_signal_rows: int = Field(ge=0)
    missing_signal_rows: int = Field(ge=0)
    requested_risk_levels: int = Field(ge=0)
    risk_coverage: Literal["not_required", "worker_verification_required", "covered"]
    estimated_generated_bytes_upper_bound: int = Field(ge=0)
    payload_validation: Literal["pending"]


class BacktestPreflightResponse(BaseModel):
    """
    API response model for `POST /backtests/preflight`.
    """

    input_readiness: BacktestInputReadinessResponse | None = None
    normalized_request: dict[str, Any]
    request_hash: str
    result_config_hash: str
    artifact_metadata: dict[str, Any]
    cost_estimate: dict[str, Any]
    warnings: list[dict[str, str]]
    errors: list[dict[str, str]]
    funding_readiness: dict[str, Any]
    direction_market_compatibility: dict[str, Any]


class BacktestJobProgressResponse(BaseModel):
    pipeline_stage: str
    percent: int
    processed_units: int
    total_units: int
    updated_at: str | None


class BacktestJobResponse(BaseModel):
    """
    API response model for public `/backtests/jobs` job reads.
    """

    job_id: str
    organization_id: str
    state: str
    request_hash: str
    result_config_hash: str
    artifact_metadata: dict[str, Any]
    progress: BacktestJobProgressResponse
    request: dict[str, Any]
    requested_top_n: int | None
    ranking: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    cancel_requested_at: str | None
    updated_at: str
    refresh_status: str
    generated_at: str
    next_allowed_refresh_at: str
    retry_after_seconds: int
    terminal_summary: dict[str, Any]
    links: dict[str, Any]
    idempotent_replay: bool | None = None


class BacktestJobsListResponse(BaseModel):
    items: list[BacktestJobResponse]
    next_cursor: str | None


class BacktestTopVariantResponse(BaseModel):
    rank: int
    variant_key: str
    variant_hash: str
    indicator_variant_hash: str | None
    summary_metrics: dict[str, Any]
    best_tp_pct: float | None
    best_sl_pct: float | None
    canonical_variant_params: dict[str, Any]
    readable_params: dict[str, Any]
    links: dict[str, Any]
    actions: dict[str, Any]
    funding_manifest_hash: str | None = None
    funding: dict[str, Any] = Field(default_factory=dict)


class BacktestTopVariantsResponse(BaseModel):
    items: list[BacktestTopVariantResponse]


class BacktestLazyTradesDetailResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    engine_params_hash: str
    artifact_manifest_hash: str
    summary_metrics: dict[str, Any]
    canonical_variant_params: dict[str, Any]
    readable_params: dict[str, Any]
    trades: list[dict[str, Any]]
    chart_overlay: dict[str, Any]
    cache: PublicBacktestCacheMetadata
    timing: dict[str, Any]
    funding_manifest_hash: str | None = None
    funding: dict[str, Any] = Field(default_factory=dict)


class BacktestLazyTradesMaterializationResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    status: str
    materialization: PublicBacktestMaterializationMetadata
    cache: PublicBacktestCacheMetadata
    timing: dict[str, Any]
    pagination: dict[str, Any]


BacktestLazyTradesResponse = (
    BacktestLazyTradesDetailResponse | BacktestLazyTradesMaterializationResponse
)


class BacktestResultSummaryResponse(BaseModel):
    job: BacktestJobResponse
    top_variants: BacktestTopVariantsResponse
    selected_variant_key: str | None
    refresh_status: str
    retry_after_seconds: int
    links: dict[str, Any]


class BacktestResultSeriesResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    kind: str
    points: list[dict[str, Any]]
    requested_points: int
    max_points: int
    returned_points: int
    source_points: int
    downsampled: bool
    cache: PublicBacktestCacheMetadata
    timing: dict[str, Any]


class BacktestResultStatsResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    kind: str
    items: list[dict[str, Any]]
    bounds: dict[str, Any]
    cache: PublicBacktestCacheMetadata
    timing: dict[str, Any]


class BacktestPaginatedTradesResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    items: list[dict[str, Any]]
    pagination: dict[str, Any]
    summary_metrics: dict[str, Any]
    cache: PublicBacktestCacheMetadata
    timing: dict[str, Any]


def build_backtest_runtime_defaults_response(
    *,
    defaults: BacktestRuntimeDefaults,
) -> BacktestRuntimeDefaultsResponse:
    return BacktestRuntimeDefaultsResponse.model_validate(defaults.as_mapping())


def build_backtest_preflight_response(
    *,
    result: BacktestPreflightResult,
) -> BacktestPreflightResponse:
    return BacktestPreflightResponse.model_validate(result.as_mapping())


def build_backtest_job_response(
    *,
    result: BacktestJobCreateResult | BacktestJobReadModel,
) -> BacktestJobResponse:
    return BacktestJobResponse.model_validate(result.as_mapping())


def build_backtest_jobs_list_response(
    *,
    result: BacktestJobListResult,
) -> BacktestJobsListResponse:
    return BacktestJobsListResponse.model_validate(result.as_mapping())


def build_backtest_top_variants_response(
    *,
    result: BacktestJobTopResult,
) -> BacktestTopVariantsResponse:
    return BacktestTopVariantsResponse.model_validate(result.as_mapping())


def build_backtest_top_variant_response(
    *,
    result: BacktestJobTopVariantReadModel,
) -> BacktestTopVariantResponse:
    return BacktestTopVariantResponse.model_validate(result.as_mapping())


def build_backtest_lazy_trades_detail_response(
    *,
    result: BacktestLazyTradesDetailReadModel,
) -> BacktestLazyTradesDetailResponse:
    return BacktestLazyTradesDetailResponse.model_validate(result.as_mapping())


def build_backtest_lazy_trades_response(
    *,
    result: BacktestLazyTradesResultReadModel,
) -> BacktestLazyTradesResponse:
    if isinstance(result, BacktestLazyTradesMaterializationReadModel):
        return BacktestLazyTradesMaterializationResponse.model_validate(result.as_mapping())
    return BacktestLazyTradesDetailResponse.model_validate(result.as_mapping())


def build_backtest_lazy_trades_materialization_response(
    *,
    result: BacktestLazyTradesMaterializationReadModel,
) -> BacktestLazyTradesMaterializationResponse:
    return BacktestLazyTradesMaterializationResponse.model_validate(result.as_mapping())


def build_backtest_result_summary_response(*, result: Any) -> BacktestResultSummaryResponse:
    return BacktestResultSummaryResponse.model_validate(result.as_mapping())


def build_backtest_result_series_response(
    *,
    result: Any,
) -> BacktestResultSeriesResponse | BacktestLazyTradesMaterializationResponse:
    if isinstance(result, BacktestLazyTradesMaterializationReadModel):
        return build_backtest_lazy_trades_materialization_response(result=result)
    return BacktestResultSeriesResponse.model_validate(result.as_mapping())


def build_backtest_result_stats_response(
    *,
    result: Any,
) -> BacktestResultStatsResponse | BacktestLazyTradesMaterializationResponse:
    if isinstance(result, BacktestLazyTradesMaterializationReadModel):
        return build_backtest_lazy_trades_materialization_response(result=result)
    return BacktestResultStatsResponse.model_validate(result.as_mapping())


def build_backtest_paginated_trades_response(
    *,
    result: Any,
) -> BacktestPaginatedTradesResponse | BacktestLazyTradesMaterializationResponse:
    if isinstance(result, BacktestLazyTradesMaterializationReadModel):
        return build_backtest_lazy_trades_materialization_response(result=result)
    return BacktestPaginatedTradesResponse.model_validate(result.as_mapping())


__all__ = [
    "BacktestLazyTradesDetailResponse",
    "BacktestLazyTradesMaterializationResponse",
    "BacktestLazyTradesResponse",
    "BacktestPaginatedTradesResponse",
    "BacktestJobProgressResponse",
    "BacktestJobResponse",
    "BacktestJobsListResponse",
    "BacktestPreflightResponse",
    "BacktestResultSeriesResponse",
    "BacktestResultStatsResponse",
    "BacktestResultSummaryResponse",
    "BacktestRuntimeDefaultsResponse",
    "BacktestTopVariantResponse",
    "BacktestTopVariantsResponse",
    "build_backtest_job_response",
    "build_backtest_lazy_trades_detail_response",
    "build_backtest_lazy_trades_materialization_response",
    "build_backtest_lazy_trades_response",
    "build_backtest_jobs_list_response",
    "build_backtest_paginated_trades_response",
    "build_backtest_preflight_response",
    "build_backtest_result_series_response",
    "build_backtest_result_stats_response",
    "build_backtest_result_summary_response",
    "build_backtest_runtime_defaults_response",
    "build_backtest_top_variant_response",
    "build_backtest_top_variants_response",
]
