from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from trading.contexts.backtest.application.dto import BacktestPreflightResult

BacktestSchedulingClass = Literal["light_candidate", "light", "heavy"]

SCHEDULING_METADATA_KEY = "scheduling"
DEFAULT_LIGHT_ESTIMATED_COMBINATIONS = 50_000
DEFAULT_LIGHT_ACTUAL_COMBINATIONS = 50_000
DEFAULT_FULL_JOB_NUMBA_NUM_THREADS = 12
ROEHUB_BACKTEST_NUMBA_NUM_THREADS = "ROEHUB_BACKTEST_NUMBA_NUM_THREADS"
ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS = "ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS"
ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS = "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS"
ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS = (
    "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS"
)
ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE = (
    "ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"
)
NUMBA_NUM_THREADS = "NUMBA_NUM_THREADS"


@dataclass(frozen=True, slots=True)
class BacktestSchedulingDecision:
    scheduling_class: BacktestSchedulingClass
    estimated_combinations_upper_bound: int


@dataclass(frozen=True, slots=True)
class BacktestNumbaThreadDecision:
    num_threads: int
    source: str


@dataclass(frozen=True, slots=True)
class BacktestJobHeavyPromotion:
    estimated_combinations_upper_bound: int
    actual_combinations: int
    reason: str = "light_candidate_exceeded_actual_threshold"


class BacktestJobSchedulingPromotionRequired(Exception):
    def __init__(
        self,
        *,
        estimated_combinations_upper_bound: int,
        actual_combinations: int,
    ) -> None:
        self.promotion = BacktestJobHeavyPromotion(
            estimated_combinations_upper_bound=estimated_combinations_upper_bound,
            actual_combinations=actual_combinations,
        )
        super().__init__(
            "light_candidate job exceeded actual light threshold before exact scoring"
        )


def classify_preflight_scheduling(
    *,
    preflight: BacktestPreflightResult,
    light_max_estimated_combinations: int = DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
) -> BacktestSchedulingDecision:
    if light_max_estimated_combinations <= 0:
        raise ValueError("light_max_estimated_combinations must be > 0")
    estimate = int(
        preflight.cost_estimate.estimated_combinations_upper_bound
        or preflight.cost_estimate.candidate_combinations
    )
    return BacktestSchedulingDecision(
        scheduling_class="heavy",
        estimated_combinations_upper_bound=max(estimate, 0),
    )


def scheduling_metadata_from_preflight(
    *,
    preflight: BacktestPreflightResult,
    light_max_estimated_combinations: int = DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
) -> dict[str, Any]:
    decision = classify_preflight_scheduling(
        preflight=preflight,
        light_max_estimated_combinations=light_max_estimated_combinations,
    )
    return {
        "version": 1,
        "source": "preflight",
        "scheduling_class": decision.scheduling_class,
        "estimated_combinations_upper_bound": (
            decision.estimated_combinations_upper_bound
        ),
        "estimated_combinations": int(
            preflight.cost_estimate.estimated_combinations
            or preflight.cost_estimate.candidate_combinations
        ),
        "arity": preflight.cost_estimate.arity,
        "row_count_upper_bounds_by_indicator": dict(
            preflight.cost_estimate.row_count_upper_bounds_by_indicator or {}
        ),
        "risk_mode": preflight.cost_estimate.risk_mode,
        "requested_range": None
        if preflight.cost_estimate.requested_range is None
        else dict(preflight.cost_estimate.requested_range),
        "requested_top_n": preflight.cost_estimate.requested_top_n,
    }


def scheduling_class_from_job_request(
    *,
    request_json: Mapping[str, Any],
) -> BacktestSchedulingClass:
    raw_scheduling = request_json.get(SCHEDULING_METADATA_KEY)
    if not isinstance(raw_scheduling, Mapping):
        return "heavy"
    raw_class = raw_scheduling.get("scheduling_class")
    if raw_class in {"light_candidate", "light", "heavy"}:
        return "heavy"
    return "heavy"


def estimated_combinations_upper_bound_from_job_request(
    *,
    request_json: Mapping[str, Any],
) -> int:
    raw_scheduling = request_json.get(SCHEDULING_METADATA_KEY)
    if not isinstance(raw_scheduling, Mapping):
        return 0
    raw_estimate = raw_scheduling.get("estimated_combinations_upper_bound")
    if isinstance(raw_estimate, bool) or not isinstance(raw_estimate, int):
        return 0
    return max(raw_estimate, 0)


def raise_if_light_candidate_needs_heavy_slot(
    *,
    scheduling_class: BacktestSchedulingClass,
    estimated_combinations_upper_bound: int,
    actual_combinations: int,
    light_max_actual_combinations: int = DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
) -> BacktestSchedulingClass:
    if light_max_actual_combinations <= 0:
        raise ValueError("light_max_actual_combinations must be > 0")
    if scheduling_class != "light_candidate":
        return scheduling_class
    if actual_combinations > light_max_actual_combinations:
        raise BacktestJobSchedulingPromotionRequired(
            estimated_combinations_upper_bound=estimated_combinations_upper_bound,
            actual_combinations=actual_combinations,
        )
    return "light"


def resolve_backtest_numba_thread_decision(
    *,
    environ: Mapping[str, str],
    scheduling_class: BacktestSchedulingClass,
) -> BacktestNumbaThreadDecision:
    _ = scheduling_class
    for key in (
        ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS,
        ROEHUB_BACKTEST_NUMBA_NUM_THREADS,
    ):
        raw_value = environ.get(key)
        if raw_value is None or not raw_value.strip():
            continue
        return BacktestNumbaThreadDecision(
            num_threads=_positive_env_int(key=key, raw_value=raw_value),
            source=key,
        )
    return BacktestNumbaThreadDecision(
        num_threads=DEFAULT_FULL_JOB_NUMBA_NUM_THREADS,
        source="default_full_job_budget",
    )


def backtest_numba_environ(
    *,
    environ: Mapping[str, str],
    scheduling_class: BacktestSchedulingClass,
) -> dict[str, str]:
    decision = resolve_backtest_numba_thread_decision(
        environ=environ,
        scheduling_class=scheduling_class,
    )
    return {
        **dict(environ),
        NUMBA_NUM_THREADS: str(decision.num_threads),
        ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS: str(decision.num_threads),
        ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE: decision.source,
    }


def _positive_env_int(*, key: str, raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value
