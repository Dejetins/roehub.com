from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    rolling_mean_grid_f64,
)
from trading.contexts.indicators.adapters.outbound.compute_numba.warmup import (
    ComputeNumbaWarmupRunner,
)
from trading.platform.config import IndicatorsComputeNumbaConfig


def test_numba_rolling_mean_perf_smoke(tmp_path: Path) -> None:
    """
    Run lightweight perf-smoke for rolling mean kernel after warmup.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Perf-smoke verifies runtime viability only, not strict latency SLA.
    Raises:
        AssertionError: If output shape is wrong or execution time is non-positive.
    Side Effects:
        Triggers Numba warmup and JIT compilation.
    """
    config = IndicatorsComputeNumbaConfig(
        numba_num_threads=1,
        numba_cache_dir=tmp_path / "numba-cache",
        max_compute_bytes_total=5 * 1024**3,
    )
    warmup = ComputeNumbaWarmupRunner(config=config)
    warmup.warmup()

    source = np.linspace(1.0, 100.0, 8_192, dtype=np.float64)
    windows = np.array([5, 10, 20, 50, 100], dtype=np.int64)

    started = time.perf_counter()
    out = rolling_mean_grid_f64(source, windows)
    elapsed = time.perf_counter() - started

    assert out.shape == (8_192, 5)
    assert elapsed > 0.0
