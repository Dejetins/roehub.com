from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

TP_SL_EXACT_METRIC_NAMES: tuple[str, ...] = (
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
    "best_tp_pct",
    "best_sl_pct",
)


@dataclass(frozen=True, slots=True)
class BacktestTpSlExactConfig:
    """
    Internal runtime knobs for the artifact-backed TP/SL exact scoring boundary.
    """

    benchmark_top_k: int = 5
    default_request_top_n: int = 100
    run_self_check: bool = False
    self_check_sample_size: int = 2
    self_check_return_tolerance: float = 1e-4
    default_fee_rate: float = 0.00075
    default_slippage_rate: float = 0.0001
    default_initial_cash_quote: float = 10000.0
    default_fixed_quote: float = 100.0
    default_safe_profit_percent: float = 30.0

    def __post_init__(self) -> None:
        if self.benchmark_top_k <= 0:
            raise ValueError("benchmark_top_k must be > 0")
        if self.default_request_top_n <= 0:
            raise ValueError("default_request_top_n must be > 0")
        if self.self_check_sample_size < 0:
            raise ValueError("self_check_sample_size must be >= 0")
        if self.self_check_return_tolerance < 0.0:
            raise ValueError("self_check_return_tolerance must be >= 0")
        if self.default_fee_rate < 0.0:
            raise ValueError("default_fee_rate must be >= 0")
        if self.default_slippage_rate < 0.0:
            raise ValueError("default_slippage_rate must be >= 0")
        if self.default_initial_cash_quote <= 0.0:
            raise ValueError("default_initial_cash_quote must be > 0")
        if self.default_fixed_quote <= 0.0:
            raise ValueError("default_fixed_quote must be > 0")
        if self.default_safe_profit_percent < 0.0:
            raise ValueError("default_safe_profit_percent must be >= 0")

    @property
    def heap_capacity(self) -> int:
        return self.benchmark_top_k


@dataclass(frozen=True, slots=True)
class BacktestTpSlExecutionContext:
    timeframe: str
    hit_times_path: str
    time_slice_start_15m: int
    time_slice_stop_15m: int
    trade_T_length: int
    eval_T_length: int
    sentinel_index: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "hit_times_path": self.hit_times_path,
            "time_slice_start_15m": self.time_slice_start_15m,
            "time_slice_stop_15m": self.time_slice_stop_15m,
            "trade_T_length": self.trade_T_length,
            "eval_T_length": self.eval_T_length,
            "sentinel_index": self.sentinel_index,
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlTopResult:
    """
    Compact retained TP/SL row. No public variant identity is assembled here.
    """

    rank: int
    score: float
    indicator_rows: Mapping[str, int]
    best_tp_idx: int
    best_sl_idx: int
    metrics: Mapping[str, float]
    metadata: Mapping[str, int | float | str | bool | None]

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.best_tp_idx < 0:
            raise ValueError("best_tp_idx must be >= 0")
        if self.best_sl_idx < 0:
            raise ValueError("best_sl_idx must be >= 0")
        object.__setattr__(
            self,
            "indicator_rows",
            MappingProxyType(
                {
                    str(indicator_id): int(row_id)
                    for indicator_id, row_id in self.indicator_rows.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({str(key): float(value) for key, value in self.metrics.items()}),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(key): _compact_scalar(value)
                    for key, value in self.metadata.items()
                }
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "indicator_rows": dict(self.indicator_rows),
            "best_tp_idx": self.best_tp_idx,
            "best_sl_idx": self.best_sl_idx,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {
            "best_sl_pct": float(self.metrics["best_sl_pct"]),
            "best_tp_pct": float(self.metrics["best_tp_pct"]),
            "total_return_pct": float(self.metrics["total_return_pct"]),
            "trade_count": int(round(float(self.metrics["trade_count"]))),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlSelfCheckSummary:
    enabled: bool
    status: str
    sample_size: int = 0
    mismatches: int = 0
    max_abs_return_diff: float | None = None
    backend_logical_name: str | None = None
    backend_implementation_id: str | None = None
    direction_mode: str | None = None
    trade_count_equal: bool | None = None
    best_cell_equal: bool | None = None
    valid_tp_sl_indexes: bool | None = None
    return_tolerance: float | None = None
    first_mismatch: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if self.mismatches < 0:
            raise ValueError("mismatches must be >= 0")
        if self.max_abs_return_diff is not None and self.max_abs_return_diff < 0.0:
            raise ValueError("max_abs_return_diff must be >= 0")
        if self.return_tolerance is not None and self.return_tolerance < 0.0:
            raise ValueError("return_tolerance must be >= 0")
        if self.first_mismatch is not None:
            object.__setattr__(
                self,
                "first_mismatch",
                MappingProxyType(dict(self.first_mismatch)),
            )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "sample_size": self.sample_size,
            "mismatches": self.mismatches,
            "max_abs_return_diff": self.max_abs_return_diff,
            "backend_logical_name": self.backend_logical_name,
            "backend_implementation_id": self.backend_implementation_id,
            "direction_mode": self.direction_mode,
            "trade_count_equal": self.trade_count_equal,
            "best_cell_equal": self.best_cell_equal,
            "valid_tp_sl_indexes": self.valid_tp_sl_indexes,
            "return_tolerance": self.return_tolerance,
            "first_mismatch": None
            if self.first_mismatch is None
            else dict(self.first_mismatch),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlExactTelemetry:
    stage_timings: Mapping[str, float]
    request_top_n: int
    benchmark_top_k: int
    heap_capacity: int
    top_results_count: int
    exact_candidates_evaluated: int
    risk_mode: str
    direction_mode: str
    backend_id: str
    arity: int
    status: str
    backend_logical_name: str | None = None
    backend_implementation_id: str | None = None
    metric_names: tuple[str, ...] = TP_SL_EXACT_METRIC_NAMES
    sample_metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.request_top_n <= 0:
            raise ValueError("request_top_n must be > 0")
        if self.benchmark_top_k <= 0:
            raise ValueError("benchmark_top_k must be > 0")
        if self.heap_capacity <= 0:
            raise ValueError("heap_capacity must be > 0")
        if self.top_results_count < 0:
            raise ValueError("top_results_count must be >= 0")
        if self.top_results_count > self.benchmark_top_k:
            raise ValueError("top_results_count must be <= benchmark_top_k")
        if self.exact_candidates_evaluated < 0:
            raise ValueError("exact_candidates_evaluated must be >= 0")
        object.__setattr__(
            self,
            "stage_timings",
            MappingProxyType({str(key): float(value) for key, value in self.stage_timings.items()}),
        )
        object.__setattr__(self, "metric_names", tuple(str(name) for name in self.metric_names))
        if self.sample_metrics is not None:
            object.__setattr__(
                self,
                "sample_metrics",
                MappingProxyType(
                    {str(key): float(value) for key, value in self.sample_metrics.items()}
                ),
            )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "stage_timings": dict(self.stage_timings),
            "request_top_n": self.request_top_n,
            "benchmark_top_k": self.benchmark_top_k,
            "heap_capacity": self.heap_capacity,
            "top_results_count": self.top_results_count,
            "exact_candidates_evaluated": self.exact_candidates_evaluated,
            "risk_mode": self.risk_mode,
            "direction_mode": self.direction_mode,
            "backend_id": self.backend_id,
            "arity": self.arity,
            "status": self.status,
            "backend_logical_name": self.backend_logical_name,
            "backend_implementation_id": self.backend_implementation_id,
            "metric_names": list(self.metric_names),
            "sample_metrics": None
            if self.sample_metrics is None
            else dict(self.sample_metrics),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlMemoryCleanupEvidence:
    checked_reference_names: tuple[str, ...]
    retained_heavy_reference_names: tuple[str, ...]
    result_contains_heavy_references: bool
    cleanup_duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.cleanup_duration_s is not None and self.cleanup_duration_s < 0.0:
            raise ValueError("cleanup_duration_s must be >= 0")
        object.__setattr__(
            self,
            "checked_reference_names",
            tuple(str(name) for name in self.checked_reference_names),
        )
        object.__setattr__(
            self,
            "retained_heavy_reference_names",
            tuple(str(name) for name in self.retained_heavy_reference_names),
        )

    @property
    def result_is_compact(self) -> bool:
        return (
            not self.result_contains_heavy_references
            and len(self.retained_heavy_reference_names) == 0
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "checked_reference_names": list(self.checked_reference_names),
            "retained_heavy_reference_names": list(self.retained_heavy_reference_names),
            "result_contains_heavy_references": self.result_contains_heavy_references,
            "result_is_compact": self.result_is_compact,
            "cleanup_duration_s": self.cleanup_duration_s,
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlExactResult:
    execution_context: BacktestTpSlExecutionContext
    top_results: tuple[BacktestTpSlTopResult, ...]
    telemetry: BacktestTpSlExactTelemetry
    self_check: BacktestTpSlSelfCheckSummary
    memory_cleanup_evidence: BacktestTpSlMemoryCleanupEvidence

    def __post_init__(self) -> None:
        top_results = tuple(self.top_results)
        if len(top_results) != self.telemetry.top_results_count:
            raise ValueError("top_results length must match telemetry.top_results_count")
        object.__setattr__(self, "top_results", top_results)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "execution_context": self.execution_context.as_mapping(),
            "top_results": [top_result.as_mapping() for top_result in self.top_results],
            "telemetry": self.telemetry.as_mapping(),
            "self_check": self.self_check.as_mapping(),
            "memory_cleanup_evidence": self.memory_cleanup_evidence.as_mapping(),
        }

    def canonical_top_results_payload(self) -> list[dict[str, Any]]:
        return [top_result.as_canonical_mapping() for top_result in self.top_results]


def _compact_scalar(value: object) -> int | float | str | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _compact_scalar(scalar)
    raise TypeError(f"metadata value {value!r} is not a compact scalar")


def canonical_tp_sl_top_results_payload(
    top_results: Sequence[BacktestTpSlTopResult | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for top_result in top_results:
        if isinstance(top_result, BacktestTpSlTopResult):
            payload.append(top_result.as_canonical_mapping())
            continue
        payload.append(
            {
                "best_sl_pct": float(top_result["best_sl_pct"]),
                "best_tp_pct": float(top_result["best_tp_pct"]),
                "total_return_pct": float(top_result["total_return_pct"]),
                "trade_count": int(top_result["trade_count"]),
            }
        )
    return payload


__all__ = [
    "BacktestTpSlExactConfig",
    "BacktestTpSlExactResult",
    "BacktestTpSlExactTelemetry",
    "BacktestTpSlExecutionContext",
    "BacktestTpSlMemoryCleanupEvidence",
    "BacktestTpSlSelfCheckSummary",
    "BacktestTpSlTopResult",
    "TP_SL_EXACT_METRIC_NAMES",
    "canonical_tp_sl_top_results_payload",
]
