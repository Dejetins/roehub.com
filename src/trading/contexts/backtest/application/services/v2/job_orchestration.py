from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestCoordinates,
    BacktestPreflightResult,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant

from .job_scheduling import (
    DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
    BacktestSchedulingClass,
    estimated_combinations_upper_bound_from_job_request,
    raise_if_light_candidate_needs_heavy_slot,
)
from .top_result_assembly import (
    TOP_RESULT_ASSEMBLY_STAGE_NAME,
    BacktestTopResultAssemblyService,
)

PERSIST_TOP_N_IO_STAGE_NAME = "persist_top_n_io"
SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME = "service_total_without_warmup"


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
        try:
            prepared_result = self.prepare_pools.execute(
                normalized_request=normalized_request,
                artifact_metadata=preflight.artifact_metadata,
            )
            combo_result = self.combo_planning.execute(
                prepared_result=prepared_result,
                normalized_request=normalized_request,
            )
            confirmed_scheduling_class = raise_if_light_candidate_needs_heavy_slot(
                scheduling_class=scheduling_class,
                estimated_combinations_upper_bound=(
                    estimated_combinations_upper_bound_from_job_request(
                        request_json=normalized_request
                    )
                    or preflight.cost_estimate.candidate_combinations
                ),
                actual_combinations=combo_result.telemetry.cartesian_combinations,
                light_max_actual_combinations=light_max_actual_combinations,
            )
            if risk_mode == "none":
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
            del hit_times_result
            del exact_result
            del combo_result
            del prepared_result
            gc.collect()


def _stage_timings(
    *,
    prepared_result: Any,
    combo_result: Any,
    hit_times_result: Any,
    exact_result: Any,
    assembly_timings: Mapping[str, float],
    elapsed: float,
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
    timers.setdefault(TOP_RESULT_ASSEMBLY_STAGE_NAME, 0.0)
    timers.setdefault(PERSIST_TOP_N_IO_STAGE_NAME, 0.0)
    timers["service_wall_clock_s"] = elapsed
    return timers


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
    "SERVICE_TOTAL_WITHOUT_WARMUP_STAGE_NAME",
    "BacktestJobExecutionResult",
    "BacktestRuntimeJobOrchestrationService",
]
