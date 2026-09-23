"""Immutable job resource bounds for the production compute algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from .job_scheduling import (
    DEFAULT_FULL_JOB_NUMBA_NUM_THREADS,
    BacktestNumbaThreadDecision,
)


@dataclass(frozen=True, slots=True)
class BacktestComputePolicy:
    threads: BacktestNumbaThreadDecision = BacktestNumbaThreadDecision(
        num_threads=DEFAULT_FULL_JOB_NUMBA_NUM_THREADS, source="default_full_job_budget"
    )
    cost_permutation_min_rows: int = 32
    cost_permutation_max_bytes: int = 64 * 1024 * 1024
    local_top_k_max_bytes: int = 64 * 1024 * 1024
    integer_tape_min_bars: int = 32
    integer_tape_max_bytes: int = 64 * 1024 * 1024
    prefix_guard_max_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.threads.num_threads <= 0:
            raise ValueError("compute thread budget must be positive")
        if self.cost_permutation_min_rows < 1 or self.cost_permutation_max_bytes < 0:
            raise ValueError("invalid cost permutation bounds")
        if self.local_top_k_max_bytes < 0:
            raise ValueError("invalid local top-K scratch bound")
        if self.integer_tape_min_bars < 1 or self.integer_tape_max_bytes < 0:
            raise ValueError("invalid integer tape bounds")
        if self.prefix_guard_max_bytes < 0:
            raise ValueError("invalid prefix guard scratch bound")
