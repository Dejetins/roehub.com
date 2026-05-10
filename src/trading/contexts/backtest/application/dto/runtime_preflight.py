from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

JsonMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestCoordinates:
    """
    Normalized public backtest artifact coordinates.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/preflight.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/
        artifact_context_resolver.py
    """

    exchange: str
    market_type: str
    symbol: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class BacktestRuntimeGuardrails:
    """
    Public preflight guardrails used by backtest runtime-defaults and validation.
    """

    max_active_jobs_per_user: int = 1
    max_queued_jobs_per_user: int = 3
    max_active_jobs_global: int = 1
    max_top_n: int = 100
    max_indicator_arity: int = 10
    max_indicator_rows: int = 1000
    max_candidate_combinations: int = 300_000_000_000
    max_tp_sl_cells: int = 2209
    lazy_trades_rate_limit: str = "30/10min"
    job_queue_timeout_seconds: int = 300
    job_wall_timeout_seconds: int = 900
    lazy_trades_timeout_seconds: int = 30

    def as_mapping(self) -> dict[str, int | str]:
        return {
            "max_active_jobs_per_user": self.max_active_jobs_per_user,
            "max_queued_jobs_per_user": self.max_queued_jobs_per_user,
            "max_active_jobs_global": self.max_active_jobs_global,
            "max_top_n": self.max_top_n,
            "max_indicator_arity": self.max_indicator_arity,
            "max_indicator_rows": self.max_indicator_rows,
            "max_candidate_combinations": self.max_candidate_combinations,
            "max_tp_sl_cells": self.max_tp_sl_cells,
            "lazy_trades_rate_limit": self.lazy_trades_rate_limit,
            "job_queue_timeout_seconds": self.job_queue_timeout_seconds,
            "job_wall_timeout_seconds": self.job_wall_timeout_seconds,
            "lazy_trades_timeout_seconds": self.lazy_trades_timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class BacktestExecutionDefaults:
    """
    Result-affecting execution defaults for request normalization.
    """

    direction_mode: str = "long_short_reversal"
    fee_rate: float = 0.00075
    slippage_rate: float = 0.0001
    initial_cash_quote: float = 10000.0
    sizing: JsonMapping | None = None
    profit_lock: JsonMapping | None = None
    close_on_end: bool = True

    def as_mapping(self) -> dict[str, Any]:
        return {
            "direction_mode": self.direction_mode,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "initial_cash_quote": self.initial_cash_quote,
            "sizing": dict(self.sizing or {"mode": "fixed_equity_pct", "equity_pct": 10.0}),
            "profit_lock": dict(self.profit_lock or {"enabled": False}),
            "close_on_end": self.close_on_end,
        }


@dataclass(frozen=True, slots=True)
class BacktestRuntimeDefaults:
    """
    Public runtime-defaults contract exposed by `GET /backtests/runtime-defaults`.
    """

    supported_timeframes: tuple[str, ...]
    risk_modes: tuple[str, ...]
    direction_modes: tuple[str, ...]
    sizing_modes: tuple[str, ...]
    ranking_metrics: tuple[str, ...]
    ranking_default: JsonMapping
    top_n_default: int
    guardrails: BacktestRuntimeGuardrails
    execution_defaults: BacktestExecutionDefaults
    supported_indicator_ids: tuple[str, ...]
    indicator_sources: Mapping[str, tuple[str, ...]]
    indicator_param_specs: JsonMapping
    hit_times_grid: JsonMapping
    links: JsonMapping

    def as_mapping(self) -> dict[str, Any]:
        return {
            "supported_timeframes": list(self.supported_timeframes),
            "risk_modes": list(self.risk_modes),
            "direction_modes": list(self.direction_modes),
            "sizing_modes": list(self.sizing_modes),
            "ranking_metrics": list(self.ranking_metrics),
            "ranking_default": dict(self.ranking_default),
            "top_n_default": self.top_n_default,
            "guardrails": self.guardrails.as_mapping(),
            "execution_defaults": self.execution_defaults.as_mapping(),
            "supported_indicator_ids": list(self.supported_indicator_ids),
            "indicator_sources": {
                indicator_id: list(values)
                for indicator_id, values in sorted(self.indicator_sources.items())
            },
            "indicator_param_specs": dict(self.indicator_param_specs),
            "hit_times_grid": dict(self.hit_times_grid),
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class BacktestValidationIssue:
    """
    Deterministic validation item for preflight responses and API error details.
    """

    path: str
    code: str
    message: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BacktestCostEstimate:
    """
    Request-cost estimate produced before any scoring work starts.
    """

    indicator_rows: int
    candidate_combinations: int
    tp_sl_cells: int
    cost_class: str

    def as_mapping(self) -> dict[str, int | str]:
        return {
            "indicator_rows": self.indicator_rows,
            "candidate_combinations": self.candidate_combinations,
            "tp_sl_cells": self.tp_sl_cells,
            "cost_class": self.cost_class,
        }


@dataclass(frozen=True, slots=True)
class BacktestArtifactMetadata:
    """
    Public artifact context selected by current pointer and slot manifests.
    """

    artifact_slot: str
    artifact_slot_generation: int
    artifact_manifest_hash: str
    artifact_asof_date: str
    hit_times_manifest_hash: str | None
    published_at_utc: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "artifact_slot": self.artifact_slot,
            "artifact_slot_generation": self.artifact_slot_generation,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "artifact_asof_date": self.artifact_asof_date,
            "hit_times_manifest_hash": self.hit_times_manifest_hash,
            "published_at_utc": self.published_at_utc,
        }


@dataclass(frozen=True, slots=True)
class BacktestPreflightResult:
    """
    Successful preflight normalization result.
    """

    normalized_request: JsonMapping
    request_hash: str
    result_config_hash: str
    artifact_metadata: BacktestArtifactMetadata
    cost_estimate: BacktestCostEstimate
    warnings: tuple[BacktestValidationIssue, ...] = ()
    errors: tuple[BacktestValidationIssue, ...] = ()

    def as_mapping(self) -> dict[str, Any]:
        return {
            "normalized_request": dict(self.normalized_request),
            "request_hash": self.request_hash,
            "result_config_hash": self.result_config_hash,
            "artifact_metadata": self.artifact_metadata.as_mapping(),
            "cost_estimate": self.cost_estimate.as_mapping(),
            "warnings": [warning.as_mapping() for warning in self.warnings],
            "errors": [error.as_mapping() for error in self.errors],
        }


__all__ = [
    "BacktestArtifactMetadata",
    "BacktestCoordinates",
    "BacktestCostEstimate",
    "BacktestExecutionDefaults",
    "BacktestPreflightResult",
    "BacktestRuntimeDefaults",
    "BacktestRuntimeGuardrails",
    "BacktestValidationIssue",
    "JsonMapping",
]
