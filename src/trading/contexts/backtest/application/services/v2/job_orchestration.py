from __future__ import annotations

import gc
import math
import time
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

from .job_scheduling import (
    DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
    BacktestSchedulingClass,
)
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


@dataclass(frozen=True, slots=True)
class BacktestJobExecutionResult:
    top_variants: tuple[BacktestJobTopVariant, ...]
    stage_timings: Mapping[str, float]
    summary_hash: str
    cleanup_evidence: Mapping[str, Any]
    exact_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "top_variants_count": len(self.top_variants),
            "stage_timings": dict(self.stage_timings),
            "summary_hash": self.summary_hash,
            "cleanup_evidence": dict(self.cleanup_evidence),
            "exact_diagnostics": dict(self.exact_diagnostics),
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
        try:
            prepared_result = self.prepare_pools.execute(
                normalized_request=normalized_request,
                artifact_metadata=preflight.artifact_metadata,
            )
            _ = scheduling_class, light_max_actual_combinations
            confirmed_scheduling_class: BacktestSchedulingClass = "heavy"
            if risk_mode == "none":
                warmup_elapsed_s = self._run_no_risk_sample_warmup(
                    prepared_result=prepared_result,
                    normalized_request=normalized_request,
                )
                combo_result = self.combo_planning.execute(
                    prepared_result=prepared_result,
                    normalized_request=normalized_request,
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
                "self_check": exact_result.self_check.as_mapping(),
                "top_results_sample": [
                    item.as_mapping() for item in exact_result.top_results[:5]
                ],
            }
            return BacktestJobExecutionResult(
                top_variants=assembly.top_variants,
                stage_timings=stage_timings,
                summary_hash=assembly.summary_hash,
                cleanup_evidence=cleanup_evidence,
                exact_diagnostics=exact_diagnostics,
            )
        finally:
            del warmup_result
            del warmup_combo_result
            del warmup_prepared_result
            del hit_times_result
            del exact_result
            del combo_result
            del prepared_result
            gc.collect()

    def _run_no_risk_sample_warmup(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        normalized_request: Mapping[str, Any],
    ) -> float:
        started = time.perf_counter()
        warmup_prepared = _limit_prepared_rows_for_warmup(prepared_result)
        warmup_request = _sample_warmup_request(normalized_request=normalized_request)
        warmup_combo = self.combo_planning.execute(
            prepared_result=warmup_prepared,
            normalized_request=warmup_request,
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
