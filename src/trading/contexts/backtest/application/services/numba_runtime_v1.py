from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, cast


@dataclass(frozen=True, slots=True)
class BacktestStageAParallelismConfigV1:
    """
    Immutable Stage A parallelism contract resolved from runtime profile and process ceiling.

    Args:
        stage_a_workers:
            Resolved live Stage A worker budget after applying the process-level thread ceiling.
        numba_threads: Effective Numba thread count applied while Stage A is running.
    Returns:
        None.
    Assumptions:
        `stage_a_workers` and `numba_threads` describe the same live Stage A thread budget after
        the process-wide `max_numba_threads` ceiling has been applied.
    Raises:
        ValueError: If either value is non-positive.
    Side Effects:
        None.
    """

    stage_a_workers: int
    numba_threads: int

    def __post_init__(self) -> None:
        """
        Validate strict-positive Stage A worker and thread counts.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Runtime orchestration resolves explicit integer worker counts before Stage A starts.
        Raises:
            ValueError: If `stage_a_workers` or `numba_threads` is non-positive.
        Side Effects:
            None.
        """
        if self.stage_a_workers <= 0:
            raise ValueError("BacktestStageAParallelismConfigV1.stage_a_workers must be > 0")
        if self.numba_threads <= 0:
            raise ValueError("BacktestStageAParallelismConfigV1.numba_threads must be > 0")


def apply_backtest_numba_threads(*, max_numba_threads: int) -> int:
    """
    Apply backtest CPU limit by setting effective Numba thread count.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/runbooks/indicators-numba-cache-and-threads.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py

    Args:
        max_numba_threads: Positive target value for `numba.set_num_threads(...)`.
    Returns:
        int: Effective Numba thread count after apply.
    Assumptions:
        Numba runtime is available in the current process.
    Raises:
        ValueError: If `max_numba_threads` is non-positive.
    Side Effects:
        Mutates process-level Numba runtime thread setting.
    """
    if max_numba_threads <= 0:
        raise ValueError("max_numba_threads must be > 0")
    import numba

    numba.set_num_threads(max_numba_threads)
    numba_runtime = cast(Any, numba)
    return int(numba_runtime.get_num_threads())


def _current_backtest_numba_threads() -> int:
    """
    Read the current effective Numba thread count for the active process.

    Args:
        None.
    Returns:
        int: Current Numba thread count.
    Assumptions:
        Numba runtime was already imported successfully in the current interpreter.
    Raises:
        ValueError: If the runtime reports a non-positive thread count.
    Side Effects:
        Imports `numba` on first use.
    """
    import numba

    numba_runtime = cast(Any, numba)
    current_threads = int(numba_runtime.get_num_threads())
    if current_threads <= 0:
        raise ValueError("current Numba thread count must be > 0")
    return current_threads


def current_backtest_numba_threads_v1() -> int:
    """
    Read the current effective Numba thread count for live single-process Stage A work.

    Args:
        None.
    Returns:
        int: Current effective Numba thread count for the active Python process.
    Assumptions:
        Perf-smoke and unit tests may need an explicit observable `numba_threads_used` value to
        prove that `stage_a_workers` maps to real in-process kernel parallelism.
    Raises:
        ValueError: Propagated if the runtime reports a non-positive thread count.
    Side Effects:
        Imports `numba` on first use through the shared runtime accessor.
    """
    return _current_backtest_numba_threads()


def resolve_backtest_stage_a_parallelism_v1(
    *,
    execution_profile: Any | None,
    max_numba_threads: int | None = None,
) -> BacktestStageAParallelismConfigV1:
    """
    Resolve the explicit Stage A parallelism contract from profile metadata and thread ceiling.

    Args:
        execution_profile:
            Resolved execution profile exposing `parallelism.stage_a_workers`, or `None` for
            compatibility-only callers that rely on the already applied process thread ceiling.
        max_numba_threads:
            Optional process-level Numba thread ceiling. When omitted, the current effective
            runtime thread count becomes the ceiling for the returned Stage A contract.
    Returns:
        BacktestStageAParallelismConfigV1: Immutable Stage A runtime contract with the configured
            worker target clamped to the effective thread ceiling.
    Assumptions:
        Stage A ordering stays deterministic because this helper only resolves the live
        worker/thread budget and does not alter variant enumeration or checkpoint boundaries.
    Raises:
        ValueError: If the ceiling is non-positive or the execution profile exposes invalid
            `stage_a_workers`.
    Side Effects:
        Reads the current Numba thread count when `max_numba_threads` is omitted.
    """
    effective_max_numba_threads = (
        current_backtest_numba_threads_v1()
        if max_numba_threads is None
        else int(max_numba_threads)
    )
    if effective_max_numba_threads <= 0:
        raise ValueError("max_numba_threads must be > 0")
    stage_a_workers = effective_max_numba_threads
    if execution_profile is not None:
        parallelism = getattr(execution_profile, "parallelism", None)
        if parallelism is None:
            raise ValueError("execution_profile.parallelism is required for Stage A runtime")
        raw_stage_a_workers = getattr(parallelism, "stage_a_workers", None)
        if raw_stage_a_workers is None:
            raise ValueError(
                "execution_profile.parallelism.stage_a_workers is required for Stage A runtime"
            )
        stage_a_workers = int(raw_stage_a_workers)
    if stage_a_workers <= 0:
        raise ValueError("stage_a_workers must be > 0")
    resolved_stage_a_workers = min(stage_a_workers, effective_max_numba_threads)
    return BacktestStageAParallelismConfigV1(
        stage_a_workers=resolved_stage_a_workers,
        numba_threads=resolved_stage_a_workers,
    )


@contextmanager
def backtest_stage_a_numba_threads_scope_v1(
    *,
    parallelism: BacktestStageAParallelismConfigV1,
) -> Iterator[int]:
    """
    Apply Stage A-specific Numba threads for one scope and restore the previous process setting.

    Args:
        parallelism: Resolved Stage A parallel contract for the current runtime plan.
    Returns:
        Iterator[int]: Context manager yielding the applied Stage A thread count.
    Assumptions:
        The caller already selected the execution profile for the current run and wants Stage A to
        use its own `stage_a_workers` contract without permanently mutating later stages.
    Raises:
        ValueError: Propagated if the resolved contract is invalid.
    Side Effects:
        Temporarily mutates the process-level Numba thread setting and restores the prior value on
        exit.
    """
    previous_threads = current_backtest_numba_threads_v1()
    apply_backtest_numba_threads(max_numba_threads=parallelism.numba_threads)
    try:
        yield parallelism.numba_threads
    finally:
        apply_backtest_numba_threads(max_numba_threads=previous_threads)

__all__ = [
    "BacktestStageAParallelismConfigV1",
    "apply_backtest_numba_threads",
    "backtest_stage_a_numba_threads_scope_v1",
    "current_backtest_numba_threads_v1",
    "resolve_backtest_stage_a_parallelism_v1",
]
