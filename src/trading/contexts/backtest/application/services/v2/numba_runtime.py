from __future__ import annotations

import os
from typing import Mapping

import numba as nb

from .job_scheduling import (
    NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE,
    ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS,
    ROEHUB_BACKTEST_NUMBA_NUM_THREADS,
)


def current_backtest_numba_telemetry(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    env = os.environ if environ is None else environ
    thread_count = int(nb.get_num_threads())
    return {
        "numba_num_threads": thread_count,
        "numba_thread_source": _thread_source(environ=env),
    }


def _thread_source(*, environ: Mapping[str, str]) -> str:
    explicit_source = environ.get(ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE)
    if explicit_source:
        return explicit_source
    for key in (
        ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS,
        ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS,
        ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS,
        ROEHUB_BACKTEST_NUMBA_NUM_THREADS,
        NUMBA_NUM_THREADS,
    ):
        if environ.get(key):
            return key
    return "numba_default"


__all__ = ["current_backtest_numba_telemetry"]
