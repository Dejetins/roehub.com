from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningResult,
    BacktestNoRiskExactConfig,
    BacktestNoRiskExactResult,
    BacktestNoRiskExactTelemetry,
    BacktestNoRiskExecutionContext,
    BacktestNoRiskMemoryCleanupEvidence,
    BacktestNoRiskSelfCheckSummary,
    BacktestNoRiskTopResult,
    BacktestPreparePoolsResult,
)

NO_RISK_EXACT_BOUNDARY_STAGE_NAME = "no_risk_exact_boundary"
NO_RISK_EXACT_BOUNDARY_STATUS = "boundary_ready"
NO_RISK_SELF_CHECK_NOT_RUN_STATUS = "not_run"
CANONICAL_EXECUTION_TIMEFRAME_V1 = "1m"


class BacktestNoRiskExactRejected(ValueError):
    """
    Deterministic internal rejection for unsupported no-risk exact boundary inputs.
    """


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactScoringService:
    """
    Internal service shell for Iteration 4.1 no-risk exact scoring.
    """

    config: BacktestNoRiskExactConfig = BacktestNoRiskExactConfig()

    def execute(
        self,
        *,
        prepared_result: BacktestPreparePoolsResult,
        combo_planning_result: BacktestComboPlanningResult,
        normalized_request: Mapping[str, Any],
    ) -> BacktestNoRiskExactResult:
        """
        Validate the no-risk boundary and return a compact placeholder result.

        Iteration 4.1 intentionally does not implement exact scoring kernels,
        heap maintenance, or top-result proxy fill.
        """

        stage_start = time.perf_counter()
        risk_mode = _risk_mode_from_normalized(normalized_request)
        if risk_mode != "none":
            raise BacktestNoRiskExactRejected(
                f"no-risk exact boundary requires risk.mode='none'; got {risk_mode!r}"
            )

        backend = combo_planning_result.backend
        if backend.risk_mode != "none":
            raise BacktestNoRiskExactRejected(
                f"combo planning backend risk_mode must be 'none'; got {backend.risk_mode!r}"
            )
        arity = len(prepared_result.indicator_ids)
        if backend.arity != arity:
            raise BacktestNoRiskExactRejected(
                f"combo planning arity {backend.arity} does not match prepared arity {arity}"
            )

        request_top_n = _request_top_n_from_normalized(
            normalized_request,
            default_request_top_n=self.config.default_request_top_n,
        )
        top_results: tuple[BacktestNoRiskTopResult, ...] = ()
        stage_timings = {
            NO_RISK_EXACT_BOUNDARY_STAGE_NAME: time.perf_counter() - stage_start,
        }
        telemetry = BacktestNoRiskExactTelemetry(
            stage_timings=stage_timings,
            request_top_n=request_top_n,
            benchmark_top_k=self.config.benchmark_top_k,
            heap_capacity=self.config.heap_capacity,
            top_results_count=len(top_results),
            exact_candidates_evaluated=combo_planning_result.telemetry.exact_candidates_evaluated,
            risk_mode=risk_mode,
            direction_mode=backend.direction_mode,
            backend_id=backend.backend_id,
            arity=arity,
            status=NO_RISK_EXACT_BOUNDARY_STATUS,
        )
        return BacktestNoRiskExactResult(
            execution_context=_execution_context_from_prepared(prepared_result),
            top_results=top_results,
            telemetry=telemetry,
            self_check=BacktestNoRiskSelfCheckSummary(
                enabled=self.config.run_self_check,
                status=NO_RISK_SELF_CHECK_NOT_RUN_STATUS,
            ),
            memory_cleanup_evidence=BacktestNoRiskMemoryCleanupEvidence(
                checked_reference_names=(
                    "prepared_result",
                    "combo_planning_result",
                    "prepared_pools",
                    "exact_context",
                    "proxy_context",
                ),
                retained_heavy_reference_names=(),
                result_contains_heavy_references=False,
            ),
        )


def _execution_context_from_prepared(
    prepared_result: BacktestPreparePoolsResult,
) -> BacktestNoRiskExecutionContext:
    return BacktestNoRiskExecutionContext(
        timeframe=prepared_result.timeframe,
        execution_timeframe=CANONICAL_EXECUTION_TIMEFRAME_V1,
        time_slice_start_15m=prepared_result.time_slice_start_15m,
        time_slice_stop_15m=prepared_result.time_slice_stop_15m,
        trade_T_length=prepared_result.trade_T_length,
        eval_T_length=prepared_result.eval_T_length,
        t_exec_limit_1m=prepared_result.execution_mapping.t_exec_limit_1m,
    )


def _risk_mode_from_normalized(normalized_request: Mapping[str, Any]) -> str:
    risk = normalized_request.get("risk")
    if not isinstance(risk, Mapping):
        raise BacktestNoRiskExactRejected("normalized_request.risk must be a mapping")
    return str(risk.get("mode"))


def _request_top_n_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    default_request_top_n: int,
) -> int:
    raw_top_n = normalized_request.get("top_n", default_request_top_n)
    if isinstance(raw_top_n, bool) or not isinstance(raw_top_n, int):
        raise BacktestNoRiskExactRejected("normalized_request.top_n must be an integer")
    if raw_top_n <= 0:
        raise BacktestNoRiskExactRejected("normalized_request.top_n must be > 0")
    return raw_top_n


__all__ = [
    "CANONICAL_EXECUTION_TIMEFRAME_V1",
    "NO_RISK_EXACT_BOUNDARY_STAGE_NAME",
    "NO_RISK_EXACT_BOUNDARY_STATUS",
    "NO_RISK_SELF_CHECK_NOT_RUN_STATUS",
    "BacktestNoRiskExactRejected",
    "BacktestNoRiskExactScoringService",
]
