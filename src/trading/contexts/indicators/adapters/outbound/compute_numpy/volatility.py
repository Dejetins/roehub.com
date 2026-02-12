"""
Numpy oracle implementation for volatility-family indicators.

Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels.volatility,
  trading.contexts.indicators.domain.definitions.volatility,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine
"""

from __future__ import annotations

import math

import numpy as np

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


def is_supported_volatility_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by volatility oracle implementation.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py,
      src/trading/contexts/indicators/domain/definitions/volatility.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by volatility oracle.
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
) -> np.ndarray:
    """
    Compute volatility indicator matrix `(V, T)` using pure NumPy/Python loops.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py,
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
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)`.
    Assumptions:
        Parameter vectors, when required, are already materialized in deterministic variant order.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_volatility_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_VOLATILITY_IDS:
        raise ValueError(f"unsupported volatility indicator_id: {indicator_id!r}")

    if normalized_id == "volatility.tr":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        tr = _true_range_series_f64(high=high_f64, low=low_f64, close=close_f64)
        out_f64 = np.ascontiguousarray(tr.reshape(1, tr.shape[0]))
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volatility.atr":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)

        tr = _true_range_series_f64(high=high_f64, low=low_f64, close=close_f64)
        variants = windows_i64.shape[0]
        t_size = tr.shape[0]
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index, window_raw in enumerate(windows_i64):
            out_f64[variant_index, :] = _rma_series_f64(source=tr, window=int(window_raw))
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    source_f64 = _prepare_source_variants(values=source_variants)
    variants = source_f64.shape[0]
    t_size = source_f64.shape[1]

    if normalized_id in {"volatility.stddev", "volatility.variance"}:
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            variance = _rolling_variance_series_f64(
                source=source_f64[variant_index, :],
                window=int(windows_i64[variant_index]),
            )
            if normalized_id == "volatility.variance":
                out_f64[variant_index, :] = variance
                continue

            stddev = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                value = float(variance[time_index])
                if math.isnan(value):
                    stddev[time_index] = np.nan
                else:
                    stddev[time_index] = math.sqrt(value)
            out_f64[variant_index, :] = stddev
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
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            series = source_f64[variant_index, :]
            returns = np.empty(t_size, dtype=np.float64)
            returns[0] = np.nan
            for time_index in range(1, t_size):
                current = float(series[time_index])
                previous = float(series[time_index - 1])
                if (
                    math.isnan(current)
                    or math.isnan(previous)
                    or current <= 0.0
                    or previous <= 0.0
                ):
                    returns[time_index] = np.nan
                else:
                    returns[time_index] = math.log(current / previous)

            variance = _rolling_variance_series_f64(
                source=returns,
                window=int(windows_i64[variant_index]),
            )
            annualization = int(annualizations_i64[variant_index])
            if annualization <= 0:
                raise ValueError(
                    "annualizations must contain only positive integers"
                )
            annualization_root = math.sqrt(float(annualization))
            hv = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                value = float(variance[time_index])
                if math.isnan(value):
                    hv[time_index] = np.nan
                else:
                    hv[time_index] = (math.sqrt(value) * annualization_root) * 100.0
            out_f64[variant_index, :] = hv
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in {
        "volatility.bbands",
        "volatility.bbands_bandwidth",
        "volatility.bbands_percent_b",
    }:
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
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            series = source_f64[variant_index, :]
            window = int(windows_i64[variant_index])
            mult = float(mults_f64[variant_index])

            middle = _rolling_mean_series_f64(source=series, window=window)
            variance = _rolling_variance_series_f64(source=series, window=window)
            sigma = np.empty(t_size, dtype=np.float64)
            upper = np.empty(t_size, dtype=np.float64)
            lower = np.empty(t_size, dtype=np.float64)

            for time_index in range(t_size):
                variance_value = float(variance[time_index])
                middle_value = float(middle[time_index])
                if math.isnan(variance_value) or math.isnan(middle_value):
                    sigma[time_index] = np.nan
                    upper[time_index] = np.nan
                    lower[time_index] = np.nan
                    continue
                sigma_value = math.sqrt(variance_value)
                sigma[time_index] = sigma_value
                upper[time_index] = middle_value + (mult * sigma_value)
                lower[time_index] = middle_value - (mult * sigma_value)

            if normalized_id == "volatility.bbands":
                out_f64[variant_index, :] = middle
                continue

            if normalized_id == "volatility.bbands_bandwidth":
                bandwidth = np.empty(t_size, dtype=np.float64)
                for time_index in range(t_size):
                    middle_value = float(middle[time_index])
                    upper_value = float(upper[time_index])
                    lower_value = float(lower[time_index])
                    if (
                        math.isnan(middle_value)
                        or math.isnan(upper_value)
                        or math.isnan(lower_value)
                        or middle_value == 0.0
                    ):
                        bandwidth[time_index] = np.nan
                    else:
                        bandwidth[time_index] = (upper_value - lower_value) / middle_value
                out_f64[variant_index, :] = bandwidth
                continue

            percent_b = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                source_value = float(series[time_index])
                upper_value = float(upper[time_index])
                lower_value = float(lower[time_index])
                denominator = upper_value - lower_value
                if (
                    math.isnan(source_value)
                    or math.isnan(upper_value)
                    or math.isnan(lower_value)
                    or denominator == 0.0
                ):
                    percent_b[time_index] = np.nan
                else:
                    percent_b[time_index] = (source_value - lower_value) / denominator
            out_f64[variant_index, :] = percent_b

        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    raise ValueError(f"unsupported volatility indicator_id: {indicator_id!r}")


def _prepare_series(*, name: str, values: np.ndarray | None) -> np.ndarray:
    """
    Normalize mandatory one-dimensional series input.

    Args:
        name: Logical input name for deterministic error messages.
        values: Series input.
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
    out = np.ascontiguousarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return out


def _prepare_source_variants(*, values: np.ndarray | None) -> np.ndarray:
    """
    Normalize source variants matrix for source-parameterized indicators.

    Args:
        values: Variant-major source matrix.
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
    out = np.ascontiguousarray(values, dtype=np.float64)
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


def _rolling_mean_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    running_sum = 0.0
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if math.isnan(incoming):
            nan_count += 1
        else:
            running_sum += incoming

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if math.isnan(outgoing):
                nan_count -= 1
            else:
                running_sum -= outgoing

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
        else:
            out[time_index] = running_sum / float(window)

    return out


def _rolling_variance_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    running_sum = 0.0
    running_sum_sq = 0.0
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if math.isnan(incoming):
            nan_count += 1
        else:
            running_sum += incoming
            running_sum_sq += incoming * incoming

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if math.isnan(outgoing):
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


def _true_range_series_f64(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
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
        ValueError: If input lengths mismatch.
    Side Effects:
        Allocates one output array.
    """
    if high.shape[0] != low.shape[0] or high.shape[0] != close.shape[0]:
        raise ValueError("high, low, close lengths must match")

    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        if math.isnan(high_value) or math.isnan(low_value):
            out[time_index] = np.nan
            continue

        hl = high_value - low_value
        if time_index == 0:
            out[time_index] = hl
            continue

        previous_close = float(close[time_index - 1])
        if math.isnan(previous_close):
            out[time_index] = hl
            continue

        hc = abs(high_value - previous_close)
        lc = abs(low_value - previous_close)
        out[time_index] = max(hl, hc, lc)
    return out


def _rma_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    alpha = 1.0 / float(window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    previous = np.nan
    for time_index in range(t_size):
        value = float(source[time_index])
        if math.isnan(value):
            previous = np.nan
            out[time_index] = np.nan
            continue
        if math.isnan(previous):
            previous = value
        else:
            previous = (alpha * value) + ((1.0 - alpha) * previous)
        out[time_index] = previous
    return out


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
