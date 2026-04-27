from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

NO_RISK_METRIC_NAMES: tuple[str, ...] = (
    "total_return_pct",
    "max_drawdown_pct",
    "return_over_max_drawdown",
    "profit_factor",
    "trade_count",
    "sharpe_trades",
    "win_rate_pct",
    "avg_trade_ret_pct",
    "avg_trade_exec_bars",
    "exposure_pct",
)


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactConfig:
    """
    Internal knobs for Iteration 4 no-risk exact scoring.
    """

    benchmark_top_k: int = 5
    self_check_n: int = 2
    combo_chunk_size: int = 4096
    bars_per_year_exec: float = 365.0 * 24.0 * 60.0
    fee_penalty_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.benchmark_top_k <= 0:
            raise ValueError("benchmark_top_k must be > 0")
        if self.self_check_n < 0:
            raise ValueError("self_check_n must be >= 0")
        if self.combo_chunk_size <= 0:
            raise ValueError("combo_chunk_size must be > 0")
        if self.bars_per_year_exec <= 0.0:
            raise ValueError("bars_per_year_exec must be > 0")
        if self.fee_penalty_multiplier < 0.0:
            raise ValueError("fee_penalty_multiplier must be >= 0")


@dataclass(frozen=True, slots=True)
class BacktestNoRiskPriceContext:
    """
    Internal execution-price arrays needed by no-risk scoring.
    """

    execution_open_1m: np.ndarray
    execution_close_1m: np.ndarray

    def __post_init__(self) -> None:
        execution_open_1m = np.ascontiguousarray(
            np.asarray(self.execution_open_1m, dtype=np.float32)
        )
        execution_close_1m = np.ascontiguousarray(
            np.asarray(self.execution_close_1m, dtype=np.float32)
        )
        if execution_open_1m.ndim != 1 or execution_close_1m.ndim != 1:
            raise ValueError("execution price arrays must be one-dimensional")
        if int(execution_open_1m.shape[0]) != int(execution_close_1m.shape[0]):
            raise ValueError("execution open/close arrays must have equal length")
        if int(execution_open_1m.shape[0]) == 0:
            raise ValueError("execution price arrays must not be empty")
        object.__setattr__(self, "execution_open_1m", execution_open_1m)
        object.__setattr__(self, "execution_close_1m", execution_close_1m)

    @classmethod
    def from_ohlcv_1m(cls, ohlcv_1m: np.ndarray) -> BacktestNoRiskPriceContext:
        ohlcv = np.asarray(ohlcv_1m, dtype=np.float32)
        if ohlcv.ndim != 2 or int(ohlcv.shape[1]) < 4:
            raise ValueError("ohlcv_1m must be a 2D array with open/close columns")
        return cls(
            execution_open_1m=ohlcv[:, 0],
            execution_close_1m=ohlcv[:, 3],
        )


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactTelemetry:
    """
    Stage timings and benchmark boundary metadata for Iteration 4.
    """

    stage_timings: Mapping[str, float]
    request_top_n: int
    benchmark_top_k: int
    top_results_count: int
    heap_capacity: int
    exact_backend_display_name: str
    implementation_id: str
    exact_candidates_evaluated: int
    combo_chunks_processed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_timings",
            MappingProxyType({str(key): float(value) for key, value in self.stage_timings.items()}),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "stage_timings": dict(self.stage_timings),
            "request_top_n": self.request_top_n,
            "benchmark_top_k": self.benchmark_top_k,
            "top_results_count": self.top_results_count,
            "heap_capacity": self.heap_capacity,
            "exact_backend_display_name": self.exact_backend_display_name,
            "implementation_id": self.implementation_id,
            "exact_candidates_evaluated": self.exact_candidates_evaluated,
            "combo_chunks_processed": self.combo_chunks_processed,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactResult:
    """
    In-memory notebook-compatible no-risk exact scoring result.
    """

    top_results: tuple[Mapping[str, Any], ...]
    self_check: Mapping[str, Any]
    telemetry: BacktestNoRiskExactTelemetry

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_results",
            tuple(MappingProxyType(dict(item)) for item in self.top_results),
        )
        object.__setattr__(self, "self_check", MappingProxyType(dict(self.self_check)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "top_results": [dict(item) for item in self.top_results],
            "self_check": dict(self.self_check),
            "telemetry": self.telemetry.as_mapping(),
        }


__all__ = [
    "NO_RISK_METRIC_NAMES",
    "BacktestNoRiskExactConfig",
    "BacktestNoRiskExactResult",
    "BacktestNoRiskExactTelemetry",
    "BacktestNoRiskPriceContext",
]
