"""
Numba kernels for volatility-family indicators.

Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numpy.volatility,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.domain.definitions.volatility
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import PRECISION_MODE_FLOAT64, SUPPORTED_PRECISION_MODES, is_nan

_SUPPORTED_VOLATILITY_IDS = {
    "volatility.tr",
    "volatility.atr",
    "volatility.stddev",
    "volatility.variance",
    "volatility.hv",
    "volatility.bbands",
    "volatility.bbands_bandwidth",
    "volatility.bbands_percent_b",
}


@nb.njit(cache=True)
def _rolling_mean_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-mean series with warmup and NaN-window propagation policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-mean series.
    Assumptions:
        Any NaN inside active window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
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
            out[time_index] = np.nan
        else:
            out[time_index] = running_sum / float(window)

    return out


@nb.njit(cache=True)
def _rolling_variance_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-variance (`ddof=0`) series with NaN-window policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling variance series.
    Assumptions:
        Any NaN inside active window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    running_sum = 0.0
    running_sum_sq = 0.0
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if is_nan(incoming):
            nan_count += 1
        else:
            running_sum += incoming
            running_sum_sq += incoming * incoming

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if is_nan(outgoing):
                nan_count -= 1
            else:
                running_sum -= outgoing
                running_sum_sq -= outgoing * outgoing

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
            continue

        mean = running_sum / float(window)
        variance = (running_sum_sq / float(window)) - (mean * mean)
        if variance < 0.0 and variance > -1e-12:
            variance = 0.0
        out[time_index] = variance

    return out


@nb.njit(cache=True)
def _true_range_series_f64(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute True Range series with EPIC-07 NaN boundary policy.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series used for previous close.
    Returns:
        np.ndarray: Float64 true-range series.
    Assumptions:
        Inputs are aligned one-dimensional series with identical length.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        if is_nan(high_value) or is_nan(low_value):
            out[time_index] = np.nan
            continue

        hl = high_value - low_value
        if time_index == 0:
            out[time_index] = hl
            continue

        previous_close = float(close[time_index - 1])
        if is_nan(previous_close):
            out[time_index] = hl
            continue

        hc = abs(high_value - previous_close)
        lc = abs(low_value - previous_close)
        if hc > hl:
            hl = hc
        if lc > hl:
            hl = lc
        out[time_index] = hl
    return out


@nb.njit(cache=True)
def _rma_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one RMA series with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
        window: Positive integer smoothing window (`alpha = 1 / window`).
    Returns:
        np.ndarray: Float64 RMA series.
    Assumptions:
        First valid source value is used as seed after each reset.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    alpha = 1.0 / float(window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    previous = np.nan
    for time_index in range(t_size):
        value = float(source[time_index])
        if is_nan(value):
            previous = np.nan
            out[time_index] = np.nan
            continue
        if is_nan(previous):
            previous = value
        else:
            previous = (alpha * value) + ((1.0 - alpha) * previous)
        out[time_index] = previous
    return out


@nb.njit(parallel=True, cache=True)
def _atr_variants_f64(tr: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute ATR matrix for per-variant windows.

    Args:
        tr: True-range series.
        windows: Per-variant windows vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Windows are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = tr.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        out[variant_index, :] = _rma_series_f64(tr, window)
    return out


@nb.njit(parallel=True, cache=True)
def _variance_or_stddev_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    return_stddev: bool,
) -> np.ndarray:
    """
    Compute rolling variance or rolling stddev matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant windows vector.
        return_stddev: Whether to return stddev (True) or variance (False).
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Windows are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporary vectors.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        variance = _rolling_variance_series_f64(source_variants[variant_index, :], window)
        if return_stddev:
            for time_index in range(t_size):
                value = float(variance[time_index])
                if is_nan(value):
                    out[variant_index, time_index] = np.nan
                else:
                    out[variant_index, time_index] = math.sqrt(value)
        else:
            out[variant_index, :] = variance
    return out


@nb.njit(parallel=True, cache=True)
def _hv_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    annualizations: np.ndarray,
) -> np.ndarray:
    """
    Compute historical volatility matrix for per-variant parameters.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant windows vector.
        annualizations: Per-variant annualization factors.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are pre-validated as positive integers.
    Raises:
        None.
    Side Effects:
        Allocates per-variant temporary returns vectors and one output matrix.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        series = source_variants[variant_index, :]
        returns = np.empty(t_size, dtype=np.float64)
        returns[0] = np.nan
        for time_index in range(1, t_size):
            current = float(series[time_index])
            previous = float(series[time_index - 1])
            if is_nan(current) or is_nan(previous) or current <= 0.0 or previous <= 0.0:
                returns[time_index] = np.nan
            else:
                returns[time_index] = math.log(current / previous)

        variance = _rolling_variance_series_f64(returns, int(windows[variant_index]))
        annualization_root = math.sqrt(float(annualizations[variant_index]))
        for time_index in range(t_size):
            value = float(variance[time_index])
            if is_nan(value):
                out[variant_index, time_index] = np.nan
            else:
                out[variant_index, time_index] = (math.sqrt(value) * annualization_root) * 100.0
    return out


@nb.njit(parallel=True, cache=True)
def _bbands_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    mults: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Compute Bollinger-derived matrix for selected output mode.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant windows vector.
        mults: Per-variant multipliers vector.
        mode: `0=middle`, `1=bandwidth`, `2=percent_b`.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are pre-validated and `mode` is one of supported values.
    Raises:
        None.
    Side Effects:
        Allocates per-variant temporary vectors and one output matrix.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        _bbands_series_into_f64(
            out[variant_index, :],
            source_variants[variant_index, :],
            int(windows[variant_index]),
            float(mults[variant_index]),
            mode,
        )
    return out


@nb.njit(cache=True)
def _bbands_series_into_f64(
    out: np.ndarray,
    source: np.ndarray,
    window: int,
    mult: float,
    mode: int,
) -> None:
    """
    Compute Bollinger-derived series into a preallocated output buffer.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py

    Args:
        out: Preallocated float64 output vector.
        source: Float64 source series.
        window: Rolling window.
        mult: Bollinger multiplier.
        mode: `0=middle`, `1=bandwidth`, `2=percent_b`.
    Returns:
        None.
    Assumptions:
        Parameters are pre-validated and `mode` is one of supported values.
    Raises:
        None.
    Side Effects:
        Writes selected Bollinger output into `out` in-place.
    """
    t_size = source.shape[0]
    middle = _rolling_mean_series_f64(source, window)
    variance = _rolling_variance_series_f64(source, window)

    if mode == 0:
        for time_index in range(t_size):
            out[time_index] = float(middle[time_index])
        return

    if mode == 1:
        for time_index in range(t_size):
            middle_value = float(middle[time_index])
            variance_value = float(variance[time_index])
            if is_nan(middle_value) or is_nan(variance_value) or middle_value == 0.0:
                out[time_index] = np.nan
                continue
            sigma = math.sqrt(variance_value)
            out[time_index] = (2.0 * mult * sigma) / middle_value
        return

    for time_index in range(t_size):
        source_value = float(source[time_index])
        middle_value = float(middle[time_index])
        variance_value = float(variance[time_index])
        if is_nan(source_value) or is_nan(middle_value) or is_nan(variance_value):
            out[time_index] = np.nan
            continue
        sigma = math.sqrt(variance_value)
        denominator = 2.0 * mult * sigma
        if denominator == 0.0:
            out[time_index] = np.nan
            continue
        lower = middle_value - (mult * sigma)
        out[time_index] = (source_value - lower) / denominator


def is_supported_volatility_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by volatility kernels.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by volatility kernels.
    Assumptions:
        Identifier normalization is delegated to `_normalize_volatility_indicator_id`.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized_id = _normalize_volatility_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_VOLATILITY_IDS


def compute_volatility_grid_f32(
    *,
    indicator_id: str,
    source_variants: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    mults: np.ndarray | None = None,
    annualizations: np.ndarray | None = None,
    precision: str = PRECISION_MODE_FLOAT64,
) -> np.ndarray:
    """
    Compute volatility indicator matrix `(V, T)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/domain/definitions/volatility.py

    Args:
        indicator_id: Volatility indicator identifier.
        source_variants: Optional `(V, T)` matrix for source-based indicators.
        high: Optional `high` series for OHLC indicators.
        low: Optional `low` series for OHLC indicators.
        close: Optional `close` series for OHLC indicators.
        windows: Optional per-variant window values.
        mults: Optional per-variant multiplier values.
        annualizations: Optional per-variant annualization values.
        precision: Precision mode (`float32`, `mixed`, `float64`) from engine policy dispatch.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)`.
    Assumptions:
        Parameter vectors, when required, are materialized in deterministic variant order.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_volatility_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_VOLATILITY_IDS:
        raise ValueError(f"unsupported volatility indicator_id: {indicator_id!r}")
    _validate_precision_mode(precision=precision)
    core_dtype = np.float64 if precision == PRECISION_MODE_FLOAT64 else np.float32

    if normalized_id == "volatility.tr":
        high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
        low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
        close_f64 = _prepare_series(name="close", values=close, dtype=core_dtype)
        _ensure_same_length(high=high_f64, low=low_f64, close=close_f64)
        tr = _true_range_series_f64(high_f64, low_f64, close_f64)
        out_f64 = np.ascontiguousarray(tr.reshape(1, tr.shape[0]))
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volatility.atr":
        high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
        low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
        close_f64 = _prepare_series(name="close", values=close, dtype=core_dtype)
        _ensure_same_length(high=high_f64, low=low_f64, close=close_f64)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        tr = _true_range_series_f64(high_f64, low_f64, close_f64)
        out_f64 = _atr_variants_f64(tr, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    source_f64 = _prepare_source_variants(values=source_variants, dtype=core_dtype)
    variants = source_f64.shape[0]

    if normalized_id == "volatility.stddev":
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        out_f64 = _variance_or_stddev_variants_f64(source_f64, windows_i64, True)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volatility.variance":
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        out_f64 = _variance_or_stddev_variants_f64(source_f64, windows_i64, False)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volatility.hv":
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        annualizations_i64 = _prepare_int_variants(
            name="annualizations",
            values=annualizations,
            expected_size=variants,
        )
        out_f64 = _hv_variants_f64(source_f64, windows_i64, annualizations_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    windows_i64 = _prepare_int_variants(
        name="windows",
        values=windows,
        expected_size=variants,
    )
    mults_f64 = _prepare_float_variants(
        name="mults",
        values=mults,
        expected_size=variants,
    )
    mode = 0
    if normalized_id == "volatility.bbands_bandwidth":
        mode = 1
    elif normalized_id == "volatility.bbands_percent_b":
        mode = 2

    out_f64 = _bbands_variants_f64(source_f64, windows_i64, mults_f64, mode)
    return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))


def _validate_precision_mode(*, precision: str) -> None:
    """
    Validate volatility kernel precision mode against shared precision policy constants.

    Docs: docs/architecture/indicators/indicators-kernels-f32-migration-plan-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        precision: Precision mode candidate.
    Returns:
        None.
    Assumptions:
        Engine-level dispatch passes normalized lowercase mode values.
    Raises:
        ValueError: If precision mode is unsupported.
    Side Effects:
        None.
    """
    if precision not in SUPPORTED_PRECISION_MODES:
        raise ValueError(
            "unsupported precision mode: "
            f"{precision!r}; expected one of {SUPPORTED_PRECISION_MODES!r}"
        )


def _prepare_series(
    *,
    name: str,
    values: np.ndarray | None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """
    Normalize mandatory one-dimensional series input.

    Args:
        name: Logical input name for deterministic error messages.
        values: Series input.
        dtype: Target floating dtype for normalized output.
    Returns:
        np.ndarray: Float64 C-contiguous one-dimensional array.
    Assumptions:
        Series may include NaN values from CandleFeed holes.
    Raises:
        ValueError: If input is missing or not one-dimensional.
    Side Effects:
        Allocates normalized array.
    """
    if values is None:
        raise ValueError(f"{name} series is required")
    out = np.ascontiguousarray(values, dtype=dtype)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return out


def _prepare_source_variants(
    *,
    values: np.ndarray | None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """
    Normalize source variants matrix for source-parameterized indicators.

    Args:
        values: Variant-major source matrix.
        dtype: Target floating dtype for normalized output.
    Returns:
        np.ndarray: Float64 C-contiguous two-dimensional array `(V, T)`.
    Assumptions:
        Variant order already matches materialized axis order from compute engine.
    Raises:
        ValueError: If matrix is missing, not two-dimensional, or empty.
    Side Effects:
        Allocates normalized matrix.
    """
    if values is None:
        raise ValueError("source_variants matrix is required")
    out = np.ascontiguousarray(values, dtype=dtype)
    if out.ndim != 2:
        raise ValueError("source_variants must be a 2D array")
    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError("source_variants must be non-empty on both dimensions")
    return out


def _prepare_int_variants(
    *,
    name: str,
    values: np.ndarray | None,
    expected_size: int | None = None,
) -> np.ndarray:
    """
    Normalize integer per-variant parameter vector.

    Args:
        name: Logical parameter name for deterministic error messages.
        values: Parameter vector.
        expected_size: Optional expected length for strict alignment checks.
    Returns:
        np.ndarray: Int64 C-contiguous vector.
    Assumptions:
        Parameter values are integer-like and positive for current volatility formulas.
    Raises:
        ValueError: If vector is missing, malformed, non-positive, or misaligned.
    Side Effects:
        Allocates normalized vector.
    """
    if values is None:
        raise ValueError(f"{name} vector is required")
    out = np.ascontiguousarray(values, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if out.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    if np.any(out <= 0):
        raise ValueError(f"{name} must contain only positive integers")
    if expected_size is not None and out.shape[0] != expected_size:
        raise ValueError(
            f"{name} length must match variants: "
            f"expected={expected_size}, got={out.shape[0]}"
        )
    return out


def _prepare_float_variants(
    *,
    name: str,
    values: np.ndarray | None,
    expected_size: int | None = None,
) -> np.ndarray:
    """
    Normalize float per-variant parameter vector.

    Args:
        name: Logical parameter name for deterministic error messages.
        values: Parameter vector.
        expected_size: Optional expected length for strict alignment checks.
    Returns:
        np.ndarray: Float64 C-contiguous vector.
    Assumptions:
        Parameter values are finite floats materialized by grid builder.
    Raises:
        ValueError: If vector is missing, malformed, contains non-finite values, or misaligned.
    Side Effects:
        Allocates normalized vector.
    """
    if values is None:
        raise ValueError(f"{name} vector is required")
    out = np.ascontiguousarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if out.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    if expected_size is not None and out.shape[0] != expected_size:
        raise ValueError(
            f"{name} length must match variants: "
            f"expected={expected_size}, got={out.shape[0]}"
        )
    return out


def _ensure_same_length(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> None:
    """
    Validate equal length for OHLC aligned series.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
    Returns:
        None.
    Assumptions:
        Inputs are one-dimensional arrays.
    Raises:
        ValueError: If series lengths differ.
    Side Effects:
        None.
    """
    if high.shape[0] != low.shape[0] or high.shape[0] != close.shape[0]:
        raise ValueError("high, low, close lengths must match")


def _normalize_volatility_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize volatility indicator identifier.

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized indicator id.
    Assumptions:
        Volatility indicator aliases are not used in v1.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized = indicator_id.strip().lower()
    if not normalized:
        raise ValueError("indicator_id must be non-empty")
    return normalized


__all__ = [
    "compute_volatility_grid_f32",
    "is_supported_volatility_indicator",
]
