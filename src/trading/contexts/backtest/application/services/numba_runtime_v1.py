from __future__ import annotations

from typing import Any, cast


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


__all__ = ["apply_backtest_numba_threads"]
