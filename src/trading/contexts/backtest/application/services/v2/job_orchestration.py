from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestCoordinates,
    BacktestPreflightResult,
    BacktestPreparePoolsResult,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant

from .combo_planning import MATRIX_BITSET_NO_RISK_V1_BACKEND
from .job_scheduling import (
    DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
    BacktestSchedulingClass,
)
from .matrix_backend.bitsets import build_runtime_bitset_pack_telemetry
from .matrix_backend.row_signatures import build_row_signature_telemetry
from .prepare_pools import build_signal_segments, row_metadata_order_hash
from .top_result_assembly import (
    TOP_RESULT_ASSEMBLY_STAGE_NAME,
    BacktestTopResultAssemblyService,
)

PERSIST_TOP_N_IO_STAGE_NAME = "persist_top_n_io"
SAMPLE_WARMUP_STAGE_NAME = "sample_warmup"
SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME = "service_total_without_warmup"
SAMPLE_WARMUP_ROWS_PER_INDICATOR = 2
SAMPLE_WARMUP_TOP_N = 1
MATRIX_BACKEND_MODE_ENV_KEY = "ROEHUB_BACKTEST_MATRIX_BACKEND_MODE"
MATRIX_BACKEND_MODE_OFF = "off"
MATRIX_BACKEND_MODE_STAGE_04_NO_RISK_MVP = "stage_04_no_risk_mvp"
MATRIX_BACKEND_MODE_STAGE_05_NO_RISK_REVERSAL_ARITY6 = (
    "stage_05_no_risk_reversal_arity6"
)


@dataclass(frozen=True, slots=True)
class BacktestJobExecutionResult:
    top_variants: tuple[BacktestJobTopVariant, ...]
    stage_timings: Mapping[str, float]
    summary_hash: str
    cleanup_evidence: Mapping[str, Any]
    exact_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    instrumentation_counters: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "top_variants_count": len(self.top_variants),
            "stage_timings": dict(self.stage_timings),
            "summary_hash": self.summary_hash,
            "cleanup_evidence": dict(self.cleanup_evidence),
            "exact_diagnostics": dict(self.exact_diagnostics),
            "instrumentation_counters": dict(self.instrumentation_counters),
        }


@dataclass(frozen=True, slots=True)
class BacktestRuntimeJobOrchestrationService:
    """
    Child-only full-job executor around the accepted artifact runtime services.
    """

    prepare_pools: Any
    combo_planning: Any
    no_risk_exact: Any
    tp_sl_hit_times: Any
    tp_sl_exact: Any
    artifact_array_loader: Any
    top_result_assembly: BacktestTopResultAssemblyService = BacktestTopResultAssemblyService()

    def execute(
        self,
        *,
        job_id: UUID,
        preflight: BacktestPreflightResult,
        updated_at: datetime,
        scheduling_class: BacktestSchedulingClass = "heavy",
        light_max_actual_combinations: int = DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
    ) -> BacktestJobExecutionResult:
        normalized_request = preflight.normalized_request
        risk = normalized_request.get("risk")
        risk_mode = risk.get("mode") if isinstance(risk, Mapping) else None
        start = time.perf_counter()
        hit_times_result = None
        exact_result = None
        combo_result = None
        prepared_result = None
        warmup_result = None
        warmup_combo_result = None
        warmup_prepared_result = None
        warmup_elapsed_s: float | None = None
        row_signature_telemetry = None
        row_signature_elapsed_s: float | None = None
        bitset_pack_telemetry = None
        bitset_pack_elapsed_s: float | None = None
        try:
            prepared_result = self.prepare_pools.execute(
                normalized_request=normalized_request,
                artifact_metadata=preflight.artifact_metadata,
            )
            row_signature_started = time.perf_counter()
            row_signature_telemetry = build_row_signature_telemetry(
                prepared_result.indicator_pools
            )
            row_signature_elapsed_s = time.perf_counter() - row_signature_started
            bitset_pack_started = time.perf_counter()
            bitset_pack_telemetry = build_runtime_bitset_pack_telemetry(
                prepared_result.indicator_pools
            )
            bitset_pack_elapsed_s = time.perf_counter() - bitset_pack_started
            _ = scheduling_class, light_max_actual_combinations
            confirmed_scheduling_class: BacktestSchedulingClass = "heavy"
            if risk_mode == "none":
                requested_no_risk_backend_id = _matrix_backend_override(
                    normalized_request=normalized_request,
                    prepared_result=prepared_result,
                )
                warmup_elapsed_s = self._run_no_risk_sample_warmup(
                    prepared_result=prepared_result,
                    normalized_request=normalized_request,
                    requested_backend_id=requested_no_risk_backend_id,
                )
                combo_result = _execute_combo_planning(
                    self.combo_planning,
                    prepared_result=prepared_result,
                    normalized_request=normalized_request,
                    requested_backend_id=requested_no_risk_backend_id,
                )
                exact_result = self.no_risk_exact.execute(
                    prepared_result=prepared_result,
                    combo_planning_result=combo_result,
                    normalized_request=normalized_request,
                )
            elif risk_mode == "tp_sl_grid":
                context = self.artifact_array_loader.resolve_context(
                    coordinates=_coordinates_from_preflight(preflight=preflight),
                    artifact_metadata=preflight.artifact_metadata,
                )
                hit_times_result = self.tp_sl_hit_times.execute(
                    normalized_request=normalized_request,
                    context=context,
                )
                warmup_elapsed_s = self._run_tp_sl_sample_warmup(
                    prepared_result=prepared_result,
                    hit_times_result=hit_times_result,
                    normalized_request=normalized_request,
                )
                combo_result = self.combo_planning.execute(
                    prepared_result=prepared_result,
                    normalized_request=normalized_request,
                )
                exact_result = self.tp_sl_exact.execute(
                    prepared_result=prepared_result,
                    combo_planning_result=combo_result,
                    hit_times_result=hit_times_result,
                    normalized_request=normalized_request,
                )
            else:
                raise ValueError(f"unsupported risk.mode for job execution: {risk_mode!r}")

            assembly = self.top_result_assembly.assemble(
                job_id=job_id,
                normalized_request=normalized_request,
                top_results=exact_result.top_results,
                updated_at=updated_at,
            )
            stage_timings = _stage_timings(
                prepared_result=prepared_result,
                combo_result=combo_result,
                hit_times_result=hit_times_result,
                exact_result=exact_result,
                assembly_timings=assembly.stage_timings,
                elapsed=time.perf_counter() - start,
                warmup_elapsed_s=warmup_elapsed_s,
            )
            cleanup_evidence = {
                "runtime_result": exact_result.memory_cleanup_evidence.as_mapping(),
                "result_contains_heavy_references": False,
                "worker_recycle_required": True,
                "worker_recycle_strategy": "disposable child process",
                "scheduling_class": confirmed_scheduling_class,
            }
            exact_diagnostics = {
                "telemetry": exact_result.telemetry.as_mapping(),
                "combo_planning": combo_result.telemetry.as_mapping(),
                "row_signatures": row_signature_telemetry.as_mapping(),
                "signal_bitsets": bitset_pack_telemetry.as_mapping(),
                "self_check": exact_result.self_check.as_mapping(),
                "top_results_sample": [
                    item.as_mapping() for item in exact_result.top_results[:5]
                ],
            }
            instrumentation_counters = _instrumentation_counters(
                preflight=preflight,
                prepared_result=prepared_result,
                combo_result=combo_result,
                hit_times_result=hit_times_result,
                exact_result=exact_result,
                stage_timings=stage_timings,
                row_signature_telemetry=row_signature_telemetry,
                row_signature_elapsed_s=row_signature_elapsed_s,
                bitset_pack_telemetry=bitset_pack_telemetry,
                bitset_pack_elapsed_s=bitset_pack_elapsed_s,
            )
            return BacktestJobExecutionResult(
                top_variants=assembly.top_variants,
                stage_timings=stage_timings,
                summary_hash=assembly.summary_hash,
                cleanup_evidence=cleanup_evidence,
                exact_diagnostics=exact_diagnostics,
                instrumentation_counters=instrumentation_counters,
            )
        finally:
            del warmup_result
            del warmup_combo_result
            del warmup_prepared_result
            del hit_times_result
            del exact_result
            del combo_result
            del row_signature_telemetry
            del bitset_pack_telemetry
            del prepared_result
            gc.collect()

    def _run_no_risk_sample_warmup(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        normalized_request: Mapping[str, Any],
        requested_backend_id: str | None,
    ) -> float:
        started = time.perf_counter()
        warmup_prepared = _limit_prepared_rows_for_warmup(prepared_result)
        warmup_request = _sample_warmup_request(normalized_request=normalized_request)
        warmup_combo = _execute_combo_planning(
            self.combo_planning,
            prepared_result=warmup_prepared,
            normalized_request=warmup_request,
            requested_backend_id=requested_backend_id,
        )
        warmup_result = self.no_risk_exact.execute(
            prepared_result=warmup_prepared,
            combo_planning_result=warmup_combo,
            normalized_request=warmup_request,
        )
        del warmup_result
        del warmup_combo
        del warmup_prepared
        gc.collect()
        return time.perf_counter() - started

    def _run_tp_sl_sample_warmup(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        hit_times_result: Any,
        normalized_request: Mapping[str, Any],
    ) -> float:
        started = time.perf_counter()
        warmup_prepared = _limit_prepared_rows_for_warmup(prepared_result)
        warmup_request = _sample_warmup_request(normalized_request=normalized_request)
        warmup_combo = self.combo_planning.execute(
            prepared_result=warmup_prepared,
            normalized_request=warmup_request,
        )
        warmup_result = self.tp_sl_exact.execute(
            prepared_result=warmup_prepared,
            combo_planning_result=warmup_combo,
            hit_times_result=hit_times_result,
            normalized_request=warmup_request,
        )
        del warmup_result
        del warmup_combo
        del warmup_prepared
        gc.collect()
        return time.perf_counter() - started


def _execute_combo_planning(
    combo_planning: Any,
    *,
    prepared_result: BacktestPreparePoolsResult,
    normalized_request: Mapping[str, Any],
    requested_backend_id: str | None,
) -> Any:
    if requested_backend_id is None:
        return combo_planning.execute(
            prepared_result=prepared_result,
            normalized_request=normalized_request,
        )
    return combo_planning.execute(
        prepared_result=prepared_result,
        normalized_request=normalized_request,
        requested_backend_id=requested_backend_id,
    )


def _matrix_backend_override(
    *,
    normalized_request: Mapping[str, Any],
    prepared_result: BacktestPreparePoolsResult,
) -> str | None:
    mode = os.environ.get(MATRIX_BACKEND_MODE_ENV_KEY, MATRIX_BACKEND_MODE_OFF).strip()
    if not mode or mode == MATRIX_BACKEND_MODE_OFF:
        return None
    if mode not in {
        MATRIX_BITSET_NO_RISK_V1_BACKEND,
        MATRIX_BACKEND_MODE_STAGE_04_NO_RISK_MVP,
        MATRIX_BACKEND_MODE_STAGE_05_NO_RISK_REVERSAL_ARITY6,
    }:
        allowed = (
            MATRIX_BACKEND_MODE_OFF,
            MATRIX_BACKEND_MODE_STAGE_04_NO_RISK_MVP,
            MATRIX_BACKEND_MODE_STAGE_05_NO_RISK_REVERSAL_ARITY6,
            MATRIX_BITSET_NO_RISK_V1_BACKEND,
        )
        raise ValueError(
            f"{MATRIX_BACKEND_MODE_ENV_KEY} must be one of "
            f"{allowed!r}"
        )
    risk = normalized_request.get("risk")
    execution = normalized_request.get("execution")
    risk_mode = risk.get("mode") if isinstance(risk, Mapping) else None
    direction_mode = (
        execution.get("direction_mode") if isinstance(execution, Mapping) else None
    )
    arity = len(prepared_result.indicator_ids)
    if (
        mode == MATRIX_BACKEND_MODE_STAGE_04_NO_RISK_MVP
        and risk_mode == "none"
        and direction_mode == "long_only"
        and arity in (2, 3)
    ):
        return MATRIX_BITSET_NO_RISK_V1_BACKEND
    if (
        mode == MATRIX_BACKEND_MODE_STAGE_05_NO_RISK_REVERSAL_ARITY6
        and risk_mode == "none"
        and direction_mode in {"long_only", "long_short_reversal"}
        and arity == 6
    ):
        return MATRIX_BITSET_NO_RISK_V1_BACKEND
    if (
        mode == MATRIX_BITSET_NO_RISK_V1_BACKEND
        and risk_mode == "none"
        and direction_mode in {"long_only", "long_short_reversal"}
        and arity in (2, 3, 6)
    ):
        return MATRIX_BITSET_NO_RISK_V1_BACKEND
    return None


def _stage_timings(
    *,
    prepared_result: Any,
    combo_result: Any,
    hit_times_result: Any,
    exact_result: Any,
    assembly_timings: Mapping[str, float],
    elapsed: float,
    warmup_elapsed_s: float | None,
) -> dict[str, float]:
    timers: dict[str, float] = {}
    if prepared_result is not None:
        timers.update(dict(prepared_result.timing.subsegments))
    if combo_result is not None:
        timers.update(dict(combo_result.telemetry.stage_timings))
    if hit_times_result is not None:
        timers.update(dict(hit_times_result.timing.subsegments))
    if exact_result is not None:
        timers.update(dict(exact_result.telemetry.stage_timings))
    timers.update(dict(assembly_timings))
    timers[SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME] = math.fsum(timers.values())
    if warmup_elapsed_s is not None:
        timers[SAMPLE_WARMUP_STAGE_NAME] = float(warmup_elapsed_s)
    timers.setdefault(TOP_RESULT_ASSEMBLY_STAGE_NAME, 0.0)
    timers.setdefault(PERSIST_TOP_N_IO_STAGE_NAME, 0.0)
    timers["service_wall_clock_s"] = elapsed
    return timers


def _instrumentation_counters(
    *,
    preflight: BacktestPreflightResult,
    prepared_result: BacktestPreparePoolsResult,
    combo_result: Any,
    hit_times_result: Any,
    exact_result: Any,
    stage_timings: Mapping[str, float],
    row_signature_telemetry: Any,
    row_signature_elapsed_s: float | None,
    bitset_pack_telemetry: Any,
    bitset_pack_elapsed_s: float | None,
) -> dict[str, Any]:
    combo_telemetry = combo_result.telemetry
    exact_telemetry = exact_result.telemetry
    artifact_load_s = _sum_optional_seconds(
        stage_timings,
        ("artifact_context_resolve", "artifact_array_open"),
    )
    exact_candidates = int(exact_telemetry.exact_candidates_evaluated)
    exact_scoring_s = _optional_seconds(stage_timings, "exact_scoring")
    tp_sl_exact_scoring_s = _optional_seconds(stage_timings, "tp_sl_exact_scoring")
    tp_count, sl_count = _tp_sl_grid_counts(hit_times_result=hit_times_result)
    tp_sl_cells = int(preflight.cost_estimate.tp_sl_cells)
    row_signature_mapping = row_signature_telemetry.as_mapping()
    bitset_pack_mapping = bitset_pack_telemetry.as_mapping()

    return {
        "artifact_load_ms": _seconds_to_ms(artifact_load_s),
        "signals_pack_ms": _seconds_to_ms(bitset_pack_elapsed_s),
        "signals_pack_bytes": bitset_pack_mapping["packed_bytes"],
        "signals_pack_estimated_peak_bytes": bitset_pack_mapping["estimated_peak_bytes"],
        "signals_pack_arrays_released": bitset_pack_mapping[
            "arrays_released_before_return"
        ],
        "bitset_word_count": bitset_pack_mapping["word_count"],
        "bitset_padding_valid": bitset_pack_mapping["padding_valid"],
        "bitset_consensus_sample_count": bitset_pack_mapping[
            "consensus_sample_count"
        ],
        "bitset_consensus_sample_mismatches": bitset_pack_mapping[
            "consensus_sample_mismatches"
        ],
        "bitset_consensus_sample_parity": bitset_pack_mapping[
            "consensus_sample_parity"
        ],
        "row_signature_ms": _seconds_to_ms(row_signature_elapsed_s),
        "combo_iteration_ms": _stage_ms(stage_timings, "combo_iteration"),
        "proxy_filter_ms": _stage_ms(stage_timings, "proxy_filter"),
        "exact_scoring_ms": _stage_ms(stage_timings, "exact_scoring"),
        "tp_sl_exact_scoring_ms": _stage_ms(stage_timings, "tp_sl_exact_scoring"),
        "top_result_assembly_ms": _stage_ms(stage_timings, "top_result_assembly"),
        "rows_before_prefilter": None,
        "rows_after_prefilter": _rows_after_prefilter(prepared_result),
        "unique_rows_after_dedup": row_signature_mapping["unique_rows_after_dedup"],
        "duplicate_signal_row_ids": row_signature_mapping["duplicate_signal_row_ids"],
        "row_signature_collision_count": row_signature_mapping[
            "row_signature_collision_count"
        ],
        "consensus_signature_count": row_signature_mapping["consensus_signature_count"],
        "consensus_signature_mode": row_signature_mapping["consensus_signature_mode"],
        "candidate_upper_bound_after_row_dedup": row_signature_mapping[
            "candidate_upper_bound_after_row_dedup"
        ],
        "combo_count_planned": int(combo_telemetry.cartesian_combinations),
        "candidates_after_proxy": int(combo_telemetry.proxy_candidates_selected),
        "exact_candidates": exact_candidates,
        "avg_segments_per_candidate": None,
        "avg_trades_per_candidate": None,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "tp_sl_cells": tp_sl_cells,
        "exact_candidates_per_sec": _rate(
            numerator=exact_candidates,
            denominator_s=exact_scoring_s,
        ),
        "trade_cell_evals_per_sec": _rate(
            numerator=exact_candidates * tp_sl_cells,
            denominator_s=tp_sl_exact_scoring_s,
        ),
    }


def _optional_seconds(timers: Mapping[str, float], key: str) -> float | None:
    value = timers.get(key)
    if value is None:
        return None
    return float(value)


def _sum_optional_seconds(
    timers: Mapping[str, float],
    keys: Sequence[str],
) -> float | None:
    values = [_optional_seconds(timers, key) for key in keys]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return math.fsum(present)


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _stage_ms(timers: Mapping[str, float], key: str) -> float | None:
    return _seconds_to_ms(_optional_seconds(timers, key))


def _rate(*, numerator: int, denominator_s: float | None) -> float | None:
    if denominator_s is None or denominator_s <= 0.0:
        return None
    return float(numerator) / denominator_s


def _rows_after_prefilter(prepared_result: BacktestPreparePoolsResult) -> int:
    return sum(int(pool.row_ids.shape[0]) for pool in prepared_result.indicator_pools)


def _tp_sl_grid_counts(*, hit_times_result: Any) -> tuple[int | None, int | None]:
    if hit_times_result is None:
        return None, None
    hit_times = hit_times_result.hit_times
    return int(hit_times.tp_values.shape[0]), int(hit_times.sl_values.shape[0])


def _limit_prepared_rows_for_warmup(
    prepared_result: BacktestPreparePoolsResult,
) -> BacktestPreparePoolsResult:
    rows_per_indicator = min(
        SAMPLE_WARMUP_ROWS_PER_INDICATOR,
        *(int(pool.row_ids.shape[0]) for pool in prepared_result.indicator_pools),
    )
    if rows_per_indicator <= 0:
        raise ValueError("sample warmup requires at least one prepared row per indicator")
    pools: list[PreparedIndicatorPool] = []
    row_slice = slice(0, rows_per_indicator)
    for pool in prepared_result.indicator_pools:
        trade_t = np.ascontiguousarray(pool.trade_T[row_slice])
        change_count = np.ascontiguousarray(pool.change_count[row_slice])
        pools.append(
            PreparedIndicatorPool(
                indicator_id=pool.indicator_id,
                row_ids=np.ascontiguousarray(pool.row_ids[row_slice]),
                filtered_row_ids=np.ascontiguousarray(pool.filtered_row_ids[row_slice]),
                trade_T=trade_t,
                eval_T=np.ascontiguousarray(pool.eval_T[row_slice]),
                segments=build_signal_segments(trade_t, change_count=change_count),
                row_score=np.ascontiguousarray(pool.row_score[row_slice]),
                score_adj=np.ascontiguousarray(pool.score_adj[row_slice]),
                nonzero=np.ascontiguousarray(pool.nonzero[row_slice]),
                proxy=np.ascontiguousarray(pool.proxy[row_slice]),
                change_count=change_count,
                metadata=pool.metadata[row_slice],
            )
        )
    limited_pools = tuple(pools)
    return replace(
        prepared_result,
        indicator_pools=limited_pools,
        row_metadata_order_hash=row_metadata_order_hash(limited_pools),
    )


def _sample_warmup_request(*, normalized_request: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(normalized_request)
    request["top_n"] = SAMPLE_WARMUP_TOP_N
    return request


def _coordinates_from_preflight(*, preflight: BacktestPreflightResult) -> BacktestCoordinates:
    coordinates = preflight.normalized_request.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("normalized_request.coordinates must be object")
    return BacktestCoordinates(
        exchange=str(coordinates["exchange"]),
        market_type=str(coordinates["market_type"]),
        symbol=str(coordinates["symbol"]),
    )


__all__ = [
    "PERSIST_TOP_N_IO_STAGE_NAME",
    "SAMPLE_WARMUP_STAGE_NAME",
    "SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME",
    "BacktestJobExecutionResult",
    "BacktestRuntimeJobOrchestrationService",
]
