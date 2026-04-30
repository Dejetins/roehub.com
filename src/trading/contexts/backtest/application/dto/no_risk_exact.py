from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

CompactScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactConfig:
    """
    Internal runtime knobs for the no-risk exact scoring boundary.
    """

    benchmark_top_k: int = 5
    default_request_top_n: int = 100
    run_self_check: bool = False

    def __post_init__(self) -> None:
        if self.benchmark_top_k <= 0:
            raise ValueError("benchmark_top_k must be > 0")
        if self.default_request_top_n <= 0:
            raise ValueError("default_request_top_n must be > 0")

    @property
    def heap_capacity(self) -> int:
        return self.benchmark_top_k


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExecutionContext:
    """
    Compact no-risk price/execution context summary for later exact stages.
    """

    timeframe: str
    execution_timeframe: str
    time_slice_start_15m: int
    time_slice_stop_15m: int
    trade_T_length: int
    eval_T_length: int
    t_exec_limit_1m: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "time_slice_start_15m": self.time_slice_start_15m,
            "time_slice_stop_15m": self.time_slice_stop_15m,
            "trade_T_length": self.trade_T_length,
            "eval_T_length": self.eval_T_length,
            "t_exec_limit_1m": self.t_exec_limit_1m,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskTopResult:
    """
    Compact top-row placeholder shape; no arrays or candidate buffers are retained.
    """

    rank: int
    score: float | None
    indicator_rows: Mapping[str, int]
    metrics: Mapping[str, float]
    metadata: Mapping[str, CompactScalar]

    def __post_init__(self) -> None:
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
                {str(key): _compact_scalar(value) for key, value in self.metadata.items()}
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "indicator_rows": dict(self.indicator_rows),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskSelfCheckSummary:
    """
    Summary-only self-check telemetry. Iteration 4.1 does not run the self-check.
    """

    enabled: bool
    status: str
    sample_size: int = 0
    mismatches: int = 0
    max_abs_diff: float | None = None

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if self.mismatches < 0:
            raise ValueError("mismatches must be >= 0")
        if self.max_abs_diff is not None and self.max_abs_diff < 0.0:
            raise ValueError("max_abs_diff must be >= 0")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "sample_size": self.sample_size,
            "mismatches": self.mismatches,
            "max_abs_diff": self.max_abs_diff,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactTelemetry:
    """
    Scalar no-risk exact boundary telemetry.
    """

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
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskMemoryCleanupEvidence:
    """
    Service hygiene evidence that the returned DTO surface is compact.
    """

    checked_reference_names: tuple[str, ...]
    retained_heavy_reference_names: tuple[str, ...]
    result_contains_heavy_references: bool

    def __post_init__(self) -> None:
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
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactResult:
    """
    Compact Iteration 4.1 no-risk exact boundary result.
    """

    execution_context: BacktestNoRiskExecutionContext
    top_results: tuple[BacktestNoRiskTopResult, ...]
    telemetry: BacktestNoRiskExactTelemetry
    self_check: BacktestNoRiskSelfCheckSummary
    memory_cleanup_evidence: BacktestNoRiskMemoryCleanupEvidence

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


def _compact_scalar(value: object) -> CompactScalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata value {value!r} is not a compact scalar")


__all__ = [
    "BacktestNoRiskExactConfig",
    "BacktestNoRiskExactResult",
    "BacktestNoRiskExactTelemetry",
    "BacktestNoRiskExecutionContext",
    "BacktestNoRiskMemoryCleanupEvidence",
    "BacktestNoRiskSelfCheckSummary",
    "BacktestNoRiskTopResult",
    "CompactScalar",
]
