from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

CompactScalar = str | int | float | bool | None
NO_RISK_CANONICAL_INTEGER_FIELDS = frozenset({"confirm_count", "trade_count"})
NO_RISK_CANONICAL_PROXY_FIELDS = ("confirm_count", "proxy_score")


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactConfig:
    """
    Internal runtime knobs for the no-risk exact scoring boundary.
    """

    benchmark_top_k: int = 5
    default_request_top_n: int = 50
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

    def as_canonical_mapping(self) -> dict[str, Any]:
        """
        Return the notebook-compatible top result row used for result hashing.
        """

        payload: dict[str, Any] = {}
        for key, value in self.metrics.items():
            payload[key] = _canonical_no_risk_field_value(key=key, value=value)
        for key in NO_RISK_CANONICAL_PROXY_FIELDS:
            if key in self.metadata:
                payload[key] = _canonical_no_risk_field_value(
                    key=key,
                    value=self.metadata[key],
                )
        for indicator_id in self.indicator_rows:
            payload[indicator_id] = _canonical_indicator_metadata(
                indicator_id=indicator_id,
                row_id=self.indicator_rows[indicator_id],
                metadata=self.metadata,
            )
        return _canonical_mapping(payload)


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
    backend_logical_name: str | None = None
    backend_implementation_id: str | None = None
    direction_mode: str | None = None
    trade_count_equal: bool | None = None
    return_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if self.mismatches < 0:
            raise ValueError("mismatches must be >= 0")
        if self.max_abs_diff is not None and self.max_abs_diff < 0.0:
            raise ValueError("max_abs_diff must be >= 0")
        if self.return_tolerance is not None and self.return_tolerance < 0.0:
            raise ValueError("return_tolerance must be >= 0")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "sample_size": self.sample_size,
            "mismatches": self.mismatches,
            "max_abs_diff": self.max_abs_diff,
            "backend_logical_name": self.backend_logical_name,
            "backend_implementation_id": self.backend_implementation_id,
            "direction_mode": self.direction_mode,
            "trade_count_equal": self.trade_count_equal,
            "return_tolerance": self.return_tolerance,
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
    backend_logical_name: str | None = None
    backend_implementation_id: str | None = None
    metric_names: tuple[str, ...] = ()
    sample_metrics: Mapping[str, float] | None = None
    numba_num_threads: int | None = None
    numba_thread_source: str | None = None

    def __post_init__(self) -> None:
        if self.request_top_n <= 0:
            raise ValueError("request_top_n must be > 0")
        if self.benchmark_top_k <= 0:
            raise ValueError("benchmark_top_k must be > 0")
        if self.heap_capacity <= 0:
            raise ValueError("heap_capacity must be > 0")
        if self.top_results_count < 0:
            raise ValueError("top_results_count must be >= 0")
        if self.top_results_count > self.heap_capacity:
            raise ValueError("top_results_count must be <= heap_capacity")
        if self.exact_candidates_evaluated < 0:
            raise ValueError("exact_candidates_evaluated must be >= 0")
        if self.numba_num_threads is not None and self.numba_num_threads <= 0:
            raise ValueError("numba_num_threads must be > 0 when provided")
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
            "numba_num_threads": self.numba_num_threads,
            "numba_thread_source": self.numba_thread_source,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskMemoryCleanupEvidence:
    """
    Service hygiene evidence that the returned DTO surface is compact.
    """

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

    def canonical_top_results_payload(self) -> list[dict[str, Any]]:
        return canonical_no_risk_top_results_payload(self.top_results)

    def canonical_top_results_hash(self) -> str:
        return canonical_no_risk_top_results_hash(self.top_results)


def canonical_no_risk_top_results_payload(
    top_results: Sequence[BacktestNoRiskTopResult | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for top_result in top_results:
        if isinstance(top_result, BacktestNoRiskTopResult):
            payload.append(top_result.as_canonical_mapping())
            continue
        normalized = _normalize_canonical_json_value(top_result)
        if not isinstance(normalized, dict):
            raise TypeError("canonical top result row must normalize to a mapping")
        payload.append(normalized)
    return payload


def canonical_no_risk_top_results_hash(
    top_results: Sequence[BacktestNoRiskTopResult | Mapping[str, Any]],
) -> str:
    return canonical_no_risk_json_hash(canonical_no_risk_top_results_payload(top_results))


def canonical_no_risk_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        _normalize_canonical_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_no_risk_field_value(*, key: str, value: object) -> CompactScalar:
    if key in NO_RISK_CANONICAL_INTEGER_FIELDS:
        return int(value)  # type: ignore[arg-type]
    return _normalize_canonical_scalar(value)


def _canonical_indicator_metadata(
    *,
    indicator_id: str,
    row_id: int,
    metadata: Mapping[str, CompactScalar],
) -> dict[str, Any]:
    prefix = f"{indicator_id}."
    indicator_metadata = {
        key[len(prefix) :]: value
        for key, value in metadata.items()
        if key.startswith(prefix)
    }
    indicator_metadata.setdefault("indicator_id", indicator_id)
    indicator_metadata.setdefault("row_id", row_id)
    return _canonical_mapping(indicator_metadata)


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_canonical_json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("canonical mapping did not normalize to a dict")
    return normalized


def _normalize_canonical_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_canonical_json_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_canonical_json_value(item) for item in value]
    return _normalize_canonical_scalar(value)


def _normalize_canonical_scalar(value: object) -> CompactScalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _normalize_canonical_scalar(scalar)
    return str(value)


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
    "canonical_no_risk_json_hash",
    "canonical_no_risk_top_results_hash",
    "canonical_no_risk_top_results_payload",
]
