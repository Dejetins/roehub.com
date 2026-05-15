from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "service_warmup",
    "numba_warmup",
    "sample_warmup",
    "total_without_warmup",
    "load_hit_times",
    "tp_sl_grid_validation",
    "prepare_pools_core",
    "build_exact_context",
    "build_proxy_context",
    "combo_iteration",
    "proxy_filter",
    "self_check",
    "exact_scoring",
    "tp_sl_exact_scoring",
    "heap_update",
    "top_result_proxy_fill",
)
CANONICAL_STAGE_SET = frozenset(CANONICAL_STAGE_ORDER)
CANONICAL_STAGE_ALIASES: Mapping[str, str] = {
    "prepare_pools": "prepare_pools_core",
    "total": "total_without_warmup",
}
SERVICE_ONLY_TELEMETRY_FIELDS: tuple[str, ...] = (
    "artifact_context_resolve",
    "artifact_array_open",
    "request_slice_prepare",
    "prepare_pools_total",
    "service_total_without_warmup",
    "top_result_assembly",
    "tp_sl_full_metrics_second_pass",
    "persist_top_n_io",
    "lazy_trades_compute",
    "lazy_trades_cache_hit",
)
SERVICE_ONLY_TELEMETRY_SET = frozenset(SERVICE_ONLY_TELEMETRY_FIELDS)
NO_RISK_TOTAL_COMPONENT_STAGES: tuple[str, ...] = (
    "prepare_pools_core",
    "build_exact_context",
    "build_proxy_context",
    "combo_iteration",
    "proxy_filter",
    "self_check",
    "exact_scoring",
    "heap_update",
    "top_result_proxy_fill",
)
TP_SL_TOTAL_COMPONENT_STAGES: tuple[str, ...] = (
    "load_hit_times",
    "tp_sl_grid_validation",
    "prepare_pools_core",
    "build_exact_context",
    "build_proxy_context",
    "combo_iteration",
    "proxy_filter",
    "self_check",
    "exact_scoring",
    "heap_update",
)
DEFAULT_REQUEST_TOP_N = 10
DEFAULT_BENCHMARK_TOP_K = 5
DEFAULT_SAMPLE_WARMUP_TOP_K = 1


class BenchmarkAccountingError(ValueError):
    """
    Raised when benchmark accounting cannot be made notebook-compatible.
    """


def canonical_required_stages(risk_mode: str) -> tuple[str, ...]:
    if risk_mode == "none":
        return (
            "service_warmup",
            "numba_warmup",
            "sample_warmup",
            "total_without_warmup",
            *NO_RISK_TOTAL_COMPONENT_STAGES,
        )
    if risk_mode == "tp_sl_grid":
        return (
            "service_warmup",
            "numba_warmup",
            "sample_warmup",
            "total_without_warmup",
            "load_hit_times",
            "tp_sl_grid_validation",
            "prepare_pools_core",
            "build_exact_context",
            "build_proxy_context",
            "combo_iteration",
            "proxy_filter",
            "self_check",
            "exact_scoring",
            "tp_sl_exact_scoring",
            "heap_update",
            "top_result_proxy_fill",
        )
    raise BenchmarkAccountingError(f"unsupported risk_mode for benchmark accounting: {risk_mode!r}")


def notebook_total_component_stages(risk_mode: str) -> tuple[str, ...]:
    if risk_mode == "none":
        return NO_RISK_TOTAL_COMPONENT_STAGES
    if risk_mode == "tp_sl_grid":
        return TP_SL_TOTAL_COMPONENT_STAGES
    raise BenchmarkAccountingError(f"unsupported risk_mode for benchmark accounting: {risk_mode!r}")


def normalize_stage_name(stage_name: str) -> str:
    return CANONICAL_STAGE_ALIASES.get(stage_name, stage_name)


def split_benchmark_timers(
    timers: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    canonical: dict[str, float] = {}
    service_only: dict[str, float] = {}
    for raw_stage_name, raw_seconds in timers.items():
        stage_name = normalize_stage_name(str(raw_stage_name))
        seconds = _coerce_seconds(stage_name=raw_stage_name, raw_seconds=raw_seconds)
        if stage_name in CANONICAL_STAGE_SET:
            _store_unique_stage(
                target=canonical,
                stage_name=stage_name,
                seconds=seconds,
                raw_stage_name=str(raw_stage_name),
            )
            continue
        if stage_name in SERVICE_ONLY_TELEMETRY_SET:
            _store_unique_stage(
                target=service_only,
                stage_name=stage_name,
                seconds=seconds,
                raw_stage_name=str(raw_stage_name),
            )
            continue
        raise BenchmarkAccountingError(
            f"unknown benchmark stage after alias normalization: {raw_stage_name!r}"
        )
    return canonical, service_only


def normalize_canonical_timers(timers: Mapping[str, Any]) -> dict[str, float]:
    canonical, service_only = split_benchmark_timers(timers)
    if service_only:
        joined = ", ".join(sorted(service_only))
        raise BenchmarkAccountingError(
            f"service-only telemetry must not be part of canonical timers: {joined}"
        )
    return canonical


def validate_canonical_timer_targets(
    *,
    timers: Mapping[str, Any],
    risk_mode: str,
) -> dict[str, float]:
    canonical = normalize_canonical_timers(timers)
    _require_stages(
        timers=canonical,
        required_stages=canonical_required_stages(risk_mode),
        scope=f"canonical {risk_mode} target",
    )
    return canonical


def build_benchmark_accounting_record(
    *,
    timers: Mapping[str, Any],
    risk_mode: str,
    request_top_n: int = DEFAULT_REQUEST_TOP_N,
    benchmark_top_k: int = DEFAULT_BENCHMARK_TOP_K,
    sample_warmup_top_k: int = DEFAULT_SAMPLE_WARMUP_TOP_K,
    top_results_count: int,
    heap_capacity: int | None = None,
) -> dict[str, Any]:
    """
    Build a runner evidence fragment with canonical and service-only totals separated.
    """

    resolved_heap_capacity = benchmark_top_k if heap_capacity is None else heap_capacity
    _validate_runner_sizes(
        request_top_n=request_top_n,
        benchmark_top_k=benchmark_top_k,
        sample_warmup_top_k=sample_warmup_top_k,
        top_results_count=top_results_count,
        heap_capacity=resolved_heap_capacity,
    )

    canonical_timers, service_only_timers = split_benchmark_timers(timers)
    component_stages = notebook_total_component_stages(risk_mode)
    _require_stages(
        timers=canonical_timers,
        required_stages=component_stages,
        scope=f"notebook-compatible {risk_mode} total",
    )
    _require_stages(
        timers=service_only_timers,
        required_stages=("service_total_without_warmup",),
        scope="service-only telemetry",
    )

    notebook_total = math.fsum(canonical_timers[stage] for stage in component_stages)
    canonical_timers["total_without_warmup"] = notebook_total
    ordered_canonical_timers = {
        stage: canonical_timers[stage]
        for stage in CANONICAL_STAGE_ORDER
        if stage in canonical_timers
    }
    ordered_service_only_timers = {
        stage: service_only_timers[stage]
        for stage in SERVICE_ONLY_TELEMETRY_FIELDS
        if stage in service_only_timers
    }

    return {
        "schema": "backtest_benchmark_accounting_v1",
        "risk_mode": risk_mode,
        "request": {"top_n": request_top_n},
        "benchmark_top_k": benchmark_top_k,
        "sample_warmup_top_k": sample_warmup_top_k,
        "top_results_count": top_results_count,
        "heap_capacity": resolved_heap_capacity,
        "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
        "canonical_timers": ordered_canonical_timers,
        "service_only_telemetry": ordered_service_only_timers,
        "total_without_warmup": notebook_total,
        "service_total_without_warmup": ordered_service_only_timers[
            "service_total_without_warmup"
        ],
        "total_component_stages": list(component_stages),
        "service_total_compared_to_canonical": False,
    }


def validate_canonical_benchmark_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise BenchmarkAccountingError("canonical benchmark JSON must contain list field 'runs'")
    request = data.get("request")
    if not isinstance(request, Mapping):
        raise BenchmarkAccountingError(
            "canonical benchmark JSON must contain mapping field 'request'"
        )

    request_top_n = request.get("top_n")
    stage_presence: Counter[str] = Counter()
    top_results_counts: set[int] = set()
    risk_modes: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise BenchmarkAccountingError(f"run {index} must be a mapping")
        risk_mode = str(run.get("risk_mode", ""))
        risk_modes.add(risk_mode)
        timers = run.get("timers")
        if not isinstance(timers, Mapping):
            raise BenchmarkAccountingError(f"run {index} must contain mapping field 'timers'")
        normalized_timers = validate_canonical_timer_targets(
            timers=timers,
            risk_mode=risk_mode,
        )
        stage_presence.update(normalized_timers.keys())
        top_results = run.get("top_results")
        if isinstance(top_results, list):
            top_results_counts.add(len(top_results))

    return {
        "schema": "backtest_benchmark_accounting_validation_v1",
        "canonical_json": str(path),
        "run_count": len(runs),
        "risk_modes": sorted(risk_modes),
        "request": {"top_n": request_top_n},
        "benchmark_top_k": DEFAULT_BENCHMARK_TOP_K,
        "sample_warmup_top_k": DEFAULT_SAMPLE_WARMUP_TOP_K,
        "top_results_count_values": sorted(top_results_counts),
        "heap_capacity": DEFAULT_BENCHMARK_TOP_K,
        "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
        "stage_aliases": dict(CANONICAL_STAGE_ALIASES),
        "service_only_telemetry_fields": list(SERVICE_ONLY_TELEMETRY_FIELDS),
        "stage_presence": dict(stage_presence),
        "total_alias_is_historical": True,
        "prepare_pools_alias_normalized": stage_presence["prepare_pools_core"] == len(runs),
        "service_total_compared_to_canonical": False,
    }


def _coerce_seconds(*, stage_name: object, raw_seconds: Any) -> float:
    if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, (int, float)):
        raise BenchmarkAccountingError(f"benchmark stage {stage_name!r} must be numeric seconds")
    seconds = float(raw_seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise BenchmarkAccountingError(
            f"benchmark stage {stage_name!r} must be finite non-negative seconds"
        )
    return seconds


def _store_unique_stage(
    *,
    target: dict[str, float],
    stage_name: str,
    seconds: float,
    raw_stage_name: str,
) -> None:
    existing = target.get(stage_name)
    if existing is None:
        target[stage_name] = seconds
        return
    if not math.isclose(existing, seconds, rel_tol=0.0, abs_tol=1e-12):
        raise BenchmarkAccountingError(
            f"conflicting benchmark aliases for stage {stage_name!r}: "
            f"{existing} vs {seconds} from {raw_stage_name!r}"
        )


def _require_stages(
    *,
    timers: Mapping[str, float],
    required_stages: tuple[str, ...],
    scope: str,
) -> None:
    missing = [stage for stage in required_stages if stage not in timers]
    if missing:
        joined = ", ".join(missing)
        raise BenchmarkAccountingError(f"missing required {scope} stage(s): {joined}")


def _validate_runner_sizes(
    *,
    request_top_n: int,
    benchmark_top_k: int,
    sample_warmup_top_k: int,
    top_results_count: int,
    heap_capacity: int,
) -> None:
    if request_top_n <= 0:
        raise BenchmarkAccountingError("request.top_n must be > 0")
    if benchmark_top_k <= 0:
        raise BenchmarkAccountingError("benchmark_top_k must be > 0")
    if sample_warmup_top_k <= 0:
        raise BenchmarkAccountingError("sample_warmup_top_k must be > 0")
    if heap_capacity < benchmark_top_k:
        raise BenchmarkAccountingError("heap_capacity must be >= benchmark_top_k")
    if top_results_count < 0:
        raise BenchmarkAccountingError("top_results_count must be >= 0")
    if top_results_count > benchmark_top_k:
        raise BenchmarkAccountingError("top_results_count must be <= benchmark_top_k")


__all__ = [
    "CANONICAL_STAGE_ALIASES",
    "CANONICAL_STAGE_ORDER",
    "DEFAULT_BENCHMARK_TOP_K",
    "DEFAULT_REQUEST_TOP_N",
    "DEFAULT_SAMPLE_WARMUP_TOP_K",
    "NO_RISK_TOTAL_COMPONENT_STAGES",
    "SERVICE_ONLY_TELEMETRY_FIELDS",
    "TP_SL_TOTAL_COMPONENT_STAGES",
    "BenchmarkAccountingError",
    "build_benchmark_accounting_record",
    "canonical_required_stages",
    "normalize_canonical_timers",
    "normalize_stage_name",
    "notebook_total_component_stages",
    "split_benchmark_timers",
    "validate_canonical_benchmark_json",
    "validate_canonical_timer_targets",
]
