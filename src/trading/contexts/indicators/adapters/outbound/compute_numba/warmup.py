"""
Numba runtime configuration and warmup runner for indicators compute.

Docs: docs/architecture/indicators/indicators-compute-engine-core.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
  trading.platform.config.indicators_compute_numba
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numba
import numpy as np

from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    compute_ma_grid_f32,
    compute_momentum_grid_f32,
    compute_trend_grid_f32,
    compute_volatility_grid_f32,
    compute_volume_grid_f32,
    ewma_grid_f64,
    is_supported_ma_indicator,
    is_supported_momentum_indicator,
    is_supported_trend_indicator,
    is_supported_volatility_indicator,
    is_supported_volume_indicator,
    rolling_mean_grid_f64,
    rolling_sum_grid_f64,
    write_series_grid_time_major,
    write_series_grid_variant_major,
)
from trading.platform.config import IndicatorsComputeNumbaConfig

log = logging.getLogger(__name__)


def apply_numba_runtime_config(*, config: IndicatorsComputeNumbaConfig) -> int:
    """
    Apply numba threads/cache settings and validate cache directory writability.

    Args:
        config: Validated indicators Numba runtime config.
    Returns:
        int: Effective numba thread count after applying configuration.
    Assumptions:
        Numba runtime is available in current interpreter.
    Raises:
        ValueError: If cache directory is not writable.
    Side Effects:
        Mutates process env (`NUMBA_CACHE_DIR`) and numba runtime state.
    """
    os.environ["NUMBA_CACHE_DIR"] = str(config.numba_cache_dir)

    cache_dir = ensure_numba_cache_dir_writable(path=config.numba_cache_dir)
    numba_config = cast(Any, numba.config)
    setattr(numba_config, "CACHE_DIR", str(cache_dir))
    numba.set_num_threads(config.numba_num_threads)
    return int(numba.get_num_threads())


def ensure_numba_cache_dir_writable(*, path: Path) -> Path:
    """
    Ensure provided cache directory exists and supports write operations.

    Args:
        path: Candidate Numba cache directory.
    Returns:
        Path: Normalized cache directory path.
    Assumptions:
        Caller passes path resolved from runtime config.
    Raises:
        ValueError: If path cannot be created or written.
    Side Effects:
        Creates directory tree when missing and touches a short-lived probe file.
    """
    normalized = Path(path)
    try:
        normalized.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"NUMBA_CACHE_DIR is not writable: {normalized}"
        ) from error

    try:
        with NamedTemporaryFile(
            mode="w",
            prefix=".numba_write_probe_",
            dir=normalized,
            delete=True,
            encoding="utf-8",
        ) as probe:
            probe.write("ok")
            probe.flush()
    except OSError as error:
        raise ValueError(
            f"NUMBA_CACHE_DIR is not writable: {normalized}"
        ) from error
    return normalized


class ComputeNumbaWarmupRunner:
    """
    Idempotent warmup runner for indicators compute common Numba kernels.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
      trading.contexts.indicators.adapters.outbound.compute_numba.engine
    """

    def __init__(self, *, config: IndicatorsComputeNumbaConfig) -> None:
        """
        Store config and initialize warmup state.

        Args:
            config: Indicators compute numba config.
        Returns:
            None.
        Assumptions:
            Warmup can be safely executed multiple times; runner keeps idempotent flag.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._config = config
        self._is_warm = False

    def warmup(self) -> None:
        """
        Apply runtime config and eagerly compile core Numba kernels.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup inputs are deterministic and side-effect free for business logic.
        Raises:
            ValueError: If runtime config cannot be applied.
        Side Effects:
            JIT-compiles Numba kernels and emits one structured log message.
        """
        if self._is_warm:
            return

        warmup_started = time.perf_counter()
        effective_threads = apply_numba_runtime_config(config=self._config)
        self._run_kernel_warmup()
        elapsed_seconds = time.perf_counter() - warmup_started
        log.info(
            "compute_numba warmup complete",
            extra={
                "warmup_done": True,
                "warmup_seconds": round(elapsed_seconds, 6),
                "numba_num_threads_effective": effective_threads,
                "numba_cache_dir": str(self._config.numba_cache_dir),
                "kernels": [
                    "rolling_sum_grid_f64",
                    "rolling_mean_grid_f64",
                    "ewma_grid_f64",
                    "write_series_grid_time_major",
                    "write_series_grid_variant_major",
                    "compute_ma_grid_f32",
                    "compute_trend_grid_f32",
                    "compute_volatility_grid_f32",
                    "compute_momentum_grid_f32",
                    "compute_volume_grid_f32",
                ],
            },
        )
        self._is_warm = True

    def _run_kernel_warmup(self) -> None:
        """
        Execute short deterministic workloads to trigger kernel compilation.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup shapes are small enough to keep startup overhead minimal.
        Raises:
            None.
        Side Effects:
            Compiles Numba kernels and allocates temporary arrays.
        """
        t_size = 2_048
        windows = np.array([5, 10, 20, 50, 100], dtype=np.int64)
        source_f64 = np.linspace(1.0, 2.0, t_size, dtype=np.float64)

        _ = rolling_sum_grid_f64(source_f64, windows)
        _ = rolling_mean_grid_f64(source_f64, windows)
        _ = ewma_grid_f64(source_f64, windows, False)

        variants = windows.shape[0]
        variant_series = np.ascontiguousarray(
            np.repeat(source_f64.reshape(1, t_size), repeats=variants, axis=0).astype(np.float32)
        )
        out_time_major = np.empty((t_size, variants), dtype=np.float32)
        out_variant_major = np.empty((variants, t_size), dtype=np.float32)
        write_series_grid_time_major(out_time_major, variant_series)
        write_series_grid_variant_major(out_variant_major, variant_series)

        source_f32 = source_f64.astype(np.float32, copy=False)
        volume_f32 = np.linspace(10.0, 20.0, t_size, dtype=np.float32)
        source_variants = np.ascontiguousarray(
            np.repeat(source_f32.reshape(1, t_size), repeats=variants, axis=0)
        )
        mults_f64 = np.linspace(1.5, 2.5, variants, dtype=np.float64)
        annualizations_i64 = np.array([252, 365, 252, 365, 252], dtype=np.int64)

        for indicator_id in (
            "ma.sma",
            "ma.ema",
            "ma.wma",
            "ma.rma",
            "ma.vwma",
            "ma.dema",
            "ma.tema",
            "ma.zlema",
            "ma.hma",
        ):
            if is_supported_ma_indicator(indicator_id=indicator_id):
                _ = compute_ma_grid_f32(
                    indicator_id=indicator_id,
                    source=source_f32,
                    windows=windows,
                    volume=volume_f32,
                )

        if is_supported_volatility_indicator(indicator_id="volatility.atr"):
            _ = compute_volatility_grid_f32(
                indicator_id="volatility.atr",
                high=source_f32 + np.float32(1.0),
                low=source_f32 - np.float32(1.0),
                close=source_f32,
                windows=windows,
            )
        if is_supported_volatility_indicator(indicator_id="volatility.bbands"):
            _ = compute_volatility_grid_f32(
                indicator_id="volatility.bbands",
                source_variants=source_variants,
                windows=windows,
                mults=mults_f64,
            )
        if is_supported_volatility_indicator(indicator_id="volatility.hv"):
            _ = compute_volatility_grid_f32(
                indicator_id="volatility.hv",
                source_variants=source_variants,
                windows=windows,
                annualizations=annualizations_i64,
            )

        if is_supported_momentum_indicator(indicator_id="momentum.rsi"):
            _ = compute_momentum_grid_f32(
                indicator_id="momentum.rsi",
                source_variants=source_variants,
                windows=windows,
            )
        if is_supported_momentum_indicator(indicator_id="momentum.macd"):
            _ = compute_momentum_grid_f32(
                indicator_id="momentum.macd",
                source_variants=source_variants,
                fast_windows=np.array([8, 10, 12, 14, 16], dtype=np.int64),
                slow_windows=np.array([20, 22, 24, 26, 28], dtype=np.int64),
                signal_windows=np.array([5, 7, 9, 11, 13], dtype=np.int64),
            )

        high_f32 = source_f32 + np.float32(1.2)
        low_f32 = source_f32 - np.float32(1.2)
        close_f32 = source_f32 + np.float32(0.3)

        if is_supported_trend_indicator(indicator_id="trend.adx"):
            _ = compute_trend_grid_f32(
                indicator_id="trend.adx",
                high=high_f32,
                low=low_f32,
                close=close_f32,
                windows=windows,
                smoothings=np.array([5, 7, 9, 11, 13], dtype=np.int64),
            )
        if is_supported_trend_indicator(indicator_id="trend.supertrend"):
            _ = compute_trend_grid_f32(
                indicator_id="trend.supertrend",
                high=high_f32,
                low=low_f32,
                close=close_f32,
                windows=windows,
                mults=mults_f64,
            )
        if is_supported_trend_indicator(indicator_id="trend.linreg_slope"):
            _ = compute_trend_grid_f32(
                indicator_id="trend.linreg_slope",
                source_variants=source_variants,
                windows=windows,
            )
        if is_supported_trend_indicator(indicator_id="trend.psar"):
            _ = compute_trend_grid_f32(
                indicator_id="trend.psar",
                high=high_f32,
                low=low_f32,
                accel_starts=np.array([0.01, 0.015, 0.02, 0.025, 0.03], dtype=np.float64),
                accel_steps=np.array([0.01, 0.015, 0.02, 0.025, 0.03], dtype=np.float64),
                accel_maxes=np.array([0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float64),
            )

        if is_supported_volume_indicator(indicator_id="volume.vwap"):
            _ = compute_volume_grid_f32(
                indicator_id="volume.vwap",
                high=high_f32,
                low=low_f32,
                close=close_f32,
                volume=volume_f32,
                windows=windows,
            )
        if is_supported_volume_indicator(indicator_id="volume.vwap_deviation"):
            _ = compute_volume_grid_f32(
                indicator_id="volume.vwap_deviation",
                high=high_f32,
                low=low_f32,
                close=close_f32,
                volume=volume_f32,
                windows=windows,
                mults=mults_f64,
            )


__all__ = [
    "ComputeNumbaWarmupRunner",
    "apply_numba_runtime_config",
    "ensure_numba_cache_dir_writable",
]
