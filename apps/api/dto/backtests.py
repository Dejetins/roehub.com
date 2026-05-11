from __future__ import annotations

from typing import Any

from pydantic import BaseModel

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
    guardrails: dict[str, Any]
    execution_defaults: dict[str, Any]
    supported_indicator_ids: list[str]
    indicator_sources: dict[str, list[str]]
    indicator_param_specs: dict[str, Any]
    hit_times_grid: dict[str, Any]
    links: dict[str, Any]


class BacktestPreflightResponse(BaseModel):
    """
    API response model for `POST /backtests/preflight`.
    """

    normalized_request: dict[str, Any]
    request_hash: str
    result_config_hash: str
    artifact_metadata: dict[str, Any]
    cost_estimate: dict[str, Any]
    warnings: list[dict[str, str]]
    errors: list[dict[str, str]]


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
    cache: dict[str, Any]
    timing: dict[str, Any]


class BacktestLazyTradesMaterializationResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    status: str
    materialization: dict[str, Any]
    cache: dict[str, Any]
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
    cache: dict[str, Any]
    timing: dict[str, Any]


class BacktestResultStatsResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    kind: str
    items: list[dict[str, Any]]
    bounds: dict[str, Any]
    cache: dict[str, Any]
    timing: dict[str, Any]


class BacktestPaginatedTradesResponse(BaseModel):
    job_id: str
    variant_key: str
    variant_hash: str
    items: list[dict[str, Any]]
    pagination: dict[str, Any]
    summary_metrics: dict[str, Any]
    cache: dict[str, Any]
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
