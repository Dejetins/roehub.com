from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from trading.contexts.backtest.application.dto import (
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


__all__ = [
    "BacktestPreflightResponse",
    "BacktestRuntimeDefaultsResponse",
    "build_backtest_preflight_response",
    "build_backtest_runtime_defaults_response",
]
