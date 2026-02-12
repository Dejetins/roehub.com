"""
Common Numba kernels and guard helpers for indicator compute engine.

Docs: docs/architecture/indicators/indicators-compute-engine-core.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.adapters.outbound.compute_numba.warmup
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from trading.contexts.indicators.domain.errors import ComputeBudgetExceeded

WORKSPACE_FACTOR_DEFAULT = 0.20
WORKSPACE_FIXED_BYTES_DEFAULT = 67_108_864
FLOAT32_DTYPE_BYTES = 4


@nb.njit(cache=True)
def is_nan(value: float) -> bool:
    """
    Return whether the provided scalar is NaN.

    Args:
        value: Floating-point scalar.
    Returns:
        bool: True when value is NaN.
    Assumptions:
        The caller passes numeric values compatible with `math.isnan`.
    Raises:
        None.
    Side Effects:
        None.
    """
    return math.isnan(value)


@nb.njit(cache=True)
def nan_to_zero(value: float) -> float:
    """
    Convert NaN scalar into zero while preserving finite values.

    Args:
        value: Floating-point scalar.
    Returns:
        float: Zero for NaN; original value otherwise.
    Assumptions:
        Used in kernels where NaN should be neutralized for accumulation.
    Raises:
        None.
    Side Effects:
        None.
    """
    if is_nan(value):
        return 0.0
    return value


@nb.njit(cache=True)
def zero_to_nan(value: float) -> float:
    """
    Convert exact zero value to NaN.

    Args:
        value: Floating-point scalar.
    Returns:
        float: NaN for zero; original value otherwise.
    Assumptions:
        Caller uses this helper only when zero is a sentinel value.
    Raises:
        None.
    Side Effects:
        None.
    """
    if value == 0.0:
        return np.nan
    return value


@nb.njit(cache=True)
def first_valid_index(values: np.ndarray) -> int:
    """
    Return index of first non-NaN value or `-1` when not found.

    Args:
        values: One-dimensional float array.
    Returns:
        int: Index of first finite value, or `-1`.
    Assumptions:
        Input array is one-dimensional and numeric.
    Raises:
        None.
    Side Effects:
        None.
    """
    for index in range(values.shape[0]):
        if not is_nan(values[index]):
            return index
    return -1


@nb.njit(parallel=True, cache=True)
def rolling_sum_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute rolling sums for multiple integer windows using float64 accumulators.

    Args:
        source: One-dimensional source series (`float64` preferred).
        windows: One-dimensional integer windows.
    Returns:
        np.ndarray: Matrix of shape `(T, W)` in float64.
    Assumptions:
        `windows` contains strictly positive integers.
    Raises:
        None.
    Side Effects:
        None.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = windows[window_index]
        running_sum = 0.0
        nan_count = 0
        for time_index in range(t_size):
            incoming = float(source[time_index])
            if is_nan(incoming):
                nan_count += 1
            else:
                running_sum += incoming

            if time_index >= window:
                outgoing = float(source[time_index - window])
                if is_nan(outgoing):
                    nan_count -= 1
                else:
                    running_sum -= outgoing

            if time_index + 1 < window or nan_count > 0:
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = running_sum
    return out


@nb.njit(parallel=True, cache=True)
def rolling_mean_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute rolling means for multiple integer windows.

    Args:
        source: One-dimensional source series (`float64` preferred).
        windows: One-dimensional integer windows.
    Returns:
        np.ndarray: Matrix of shape `(T, W)` in float64.
    Assumptions:
        `windows` contains strictly positive integers.
    Raises:
        None.
    Side Effects:
        None.
    """
    rolling_sum = rolling_sum_grid_f64(source, windows)
    t_size = rolling_sum.shape[0]
    w_size = rolling_sum.shape[1]
    out = np.empty((t_size, w_size), dtype=np.float64)
    for window_index in nb.prange(w_size):
        window = windows[window_index]
        for time_index in range(t_size):
            value = float(rolling_sum[time_index, window_index])
            if is_nan(value):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = value / window
    return out


@nb.njit(parallel=True, cache=True)
def ewma_grid_f64(source: np.ndarray, windows: np.ndarray, use_rma_alpha: bool) -> np.ndarray:
    """
    Compute EWMA/RMA matrix for multiple windows in one pass per window.

    Args:
        source: One-dimensional source series (`float64` preferred).
        windows: One-dimensional integer windows.
        use_rma_alpha: When True uses `1 / w`, otherwise `2 / (w + 1)`.
    Returns:
        np.ndarray: Matrix of shape `(T, W)` in float64.
    Assumptions:
        `windows` contains strictly positive integers.
    Raises:
        None.
    Side Effects:
        None.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)
    for window_index in nb.prange(w_size):
        window = windows[window_index]
        alpha = 1.0 / window if use_rma_alpha else 2.0 / (window + 1.0)
        previous = np.nan
        for time_index in range(t_size):
            value = float(source[time_index])
            if is_nan(value):
                previous = np.nan
                out[time_index, window_index] = np.nan
                continue
            if is_nan(previous):
                previous = value
            else:
                previous = alpha * value + (1.0 - alpha) * previous
            out[time_index, window_index] = previous
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def write_series_grid_time_major(values: np.ndarray, variant_series: np.ndarray) -> None:
    """
    Fill TIME_MAJOR output tensor `(T, V)` from variant-major source matrix `(V, T)`.

    Args:
        values: Output matrix `(T, V)`.
        variant_series: Source matrix `(V, T)`.
    Returns:
        None.
    Assumptions:
        Shapes are already validated by the caller.
    Raises:
        None.
    Side Effects:
        Writes into `values` buffer in-place.
    """
    variants = variant_series.shape[0]
    t_size = variant_series.shape[1]
    for variant_index in nb.prange(variants):
        for time_index in range(t_size):
            values[time_index, variant_index] = variant_series[variant_index, time_index]


@nb.njit(parallel=True, fastmath=True, cache=True)
def write_series_grid_variant_major(values: np.ndarray, variant_series: np.ndarray) -> None:
    """
    Fill VARIANT_MAJOR output tensor `(V, T)` from source matrix `(V, T)`.

    Args:
        values: Output matrix `(V, T)`.
        variant_series: Source matrix `(V, T)`.
    Returns:
        None.
    Assumptions:
        Shapes are already validated by the caller.
    Raises:
        None.
    Side Effects:
        Writes into `values` buffer in-place.
    """
    variants = variant_series.shape[0]
    t_size = variant_series.shape[1]
    for variant_index in nb.prange(variants):
        for time_index in range(t_size):
            values[variant_index, time_index] = variant_series[variant_index, time_index]


def estimate_tensor_bytes(*, t: int, variants: int, dtype_bytes: int = FLOAT32_DTYPE_BYTES) -> int:
    """
    Estimate output tensor bytes for `(T, V)` shape and fixed dtype size.

    Args:
        t: Time dimension length.
        variants: Variant count.
        dtype_bytes: Number of bytes per scalar.
    Returns:
        int: Estimated output bytes.
    Assumptions:
        All arguments are strictly positive integers.
    Raises:
        ValueError: If any argument is non-positive.
    Side Effects:
        None.
    """
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if variants <= 0:
        raise ValueError(f"variants must be > 0, got {variants}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")
    return int(t) * int(variants) * int(dtype_bytes)


def estimate_total_bytes(
    *,
    bytes_out: int,
    workspace_factor: float = WORKSPACE_FACTOR_DEFAULT,
    workspace_fixed_bytes: int = WORKSPACE_FIXED_BYTES_DEFAULT,
) -> int:
    """
    Estimate total compute bytes with reserved workspace margin.

    Args:
        bytes_out: Output tensor bytes.
        workspace_factor: Proportional workspace reserve factor.
        workspace_fixed_bytes: Fixed workspace reserve in bytes.
    Returns:
        int: Estimated total bytes.
    Assumptions:
        Reservation model is conservative and deterministic.
    Raises:
        ValueError: If arguments are invalid.
    Side Effects:
        None.
    """
    if bytes_out <= 0:
        raise ValueError(f"bytes_out must be > 0, got {bytes_out}")
    if workspace_factor < 0:
        raise ValueError(f"workspace_factor must be >= 0, got {workspace_factor}")
    if workspace_fixed_bytes < 0:
        raise ValueError(f"workspace_fixed_bytes must be >= 0, got {workspace_fixed_bytes}")
    workspace_est = int(math.ceil(bytes_out * workspace_factor)) + workspace_fixed_bytes
    return bytes_out + workspace_est


def check_total_budget_or_raise(
    *,
    t: int,
    variants: int,
    bytes_out: int,
    bytes_total_est: int,
    max_compute_bytes_total: int,
) -> None:
    """
    Enforce single total memory budget and raise deterministic domain error.

    Args:
        t: Time dimension length.
        variants: Variant count.
        bytes_out: Output tensor bytes.
        bytes_total_est: Estimated total bytes including workspace.
        max_compute_bytes_total: Allowed total memory budget in bytes.
    Returns:
        None.
    Assumptions:
        Caller passes precomputed deterministic estimates.
    Raises:
        ComputeBudgetExceeded: If estimate exceeds budget.
        ValueError: If budget is non-positive.
    Side Effects:
        None.
    """
    if max_compute_bytes_total <= 0:
        raise ValueError(
            "max_compute_bytes_total must be > 0, "
            f"got {max_compute_bytes_total}"
        )
    if bytes_total_est > max_compute_bytes_total:
        raise ComputeBudgetExceeded(
            t=t,
            variants=variants,
            bytes_out=bytes_out,
            bytes_total_est=bytes_total_est,
            max_compute_bytes_total=max_compute_bytes_total,
        )


__all__ = [
    "FLOAT32_DTYPE_BYTES",
    "WORKSPACE_FACTOR_DEFAULT",
    "WORKSPACE_FIXED_BYTES_DEFAULT",
    "check_total_budget_or_raise",
    "estimate_tensor_bytes",
    "estimate_total_bytes",
    "ewma_grid_f64",
    "first_valid_index",
    "is_nan",
    "nan_to_zero",
    "rolling_mean_grid_f64",
    "rolling_sum_grid_f64",
    "write_series_grid_time_major",
    "write_series_grid_variant_major",
    "zero_to_nan",
]
