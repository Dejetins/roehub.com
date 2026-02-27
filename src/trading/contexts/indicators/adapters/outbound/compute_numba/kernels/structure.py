"""
Numba kernels for structure/normalization indicators.

Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
Related: docs/architecture/indicators/indicators_formula.yaml,
  trading.contexts.indicators.adapters.outbound.compute_numpy.structure,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.adapters.outbound.compute_numba.kernels.volatility
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import PRECISION_MODE_FLOAT64, SUPPORTED_PRECISION_MODES, is_nan
from .volatility import compute_volatility_grid_f32

_SUPPORTED_STRUCTURE_IDS = {
    "structure.candle_body",
    "structure.candle_body_atr",
    "structure.candle_body_pct",
    "structure.candle_lower_wick",
    "structure.candle_lower_wick_atr",
    "structure.candle_lower_wick_pct",
    "structure.candle_range",
    "structure.candle_range_atr",
    "structure.candle_stats",
    "structure.candle_stats_atr_norm",
    "structure.candle_upper_wick",
    "structure.candle_upper_wick_atr",
    "structure.candle_upper_wick_pct",
    "structure.distance_to_ma_norm",
    "structure.percent_rank",
    "structure.pivot_high",
    "structure.pivot_low",
    "structure.pivots",
    "structure.zscore",
}

_CANDLE_STATS_OUTPUT_MODE = {
    "structure.candle_body": 0,
    "structure.candle_range": 1,
    "structure.candle_upper_wick": 2,
    "structure.candle_lower_wick": 3,
    "structure.candle_stats": 4,
    "structure.candle_body_pct": 4,
    "structure.candle_upper_wick_pct": 5,
    "structure.candle_lower_wick_pct": 6,
}

_CANDLE_STATS_ATR_OUTPUT_MODE = {
    "structure.candle_stats_atr_norm": 0,
    "structure.candle_body_atr": 0,
    "structure.candle_range_atr": 1,
    "structure.candle_upper_wick_atr": 2,
    "structure.candle_lower_wick_atr": 3,
}

_PIVOTS_OUTPUT_MODE = {
    "structure.pivots": 0,
    "structure.pivot_high": 0,
    "structure.pivot_low": 1,
}


@nb.njit(cache=True)
def _rolling_mean_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-mean series with warmup and NaN-window propagation.

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
    Compute one rolling-variance (`ddof=0`) with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-variance series.
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
def _ema_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one EMA series with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
        window: Positive integer smoothing window.
    Returns:
        np.ndarray: Float64 EMA series.
    Assumptions:
        `reset-on-NaN` means any NaN input clears state until next valid seed.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    alpha = 2.0 / (float(window) + 1.0)
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


@nb.njit(cache=True)
def _percent_rank_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling percent-rank (`tie_mode='le'`, `scale=100.0`) for one window.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 percent-rank series.
    Assumptions:
        Any NaN inside active window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        out[time_index] = np.nan

    for time_index in range(t_size):
        if time_index + 1 < window:
            continue

        start = time_index + 1 - window
        current = float(source[time_index])
        if is_nan(current):
            continue

        valid = True
        count_le = 0
        for idx in range(start, time_index + 1):
            value = float(source[idx])
            if is_nan(value):
                valid = False
                break
            if value <= current:
                count_le += 1

        if not valid:
            continue

        out[time_index] = (float(count_le) / float(window)) * 100.0

    return out


@nb.njit(cache=True)
def _candle_stats_series_f64(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Compute one candle-stats output series selected by mode.

    Args:
        open_: Open-price series.
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        mode: Output selector (`0=body,1=range,2=upper,3=lower,4=body_pct,5=upper_pct,6=lower_pct`).
    Returns:
        np.ndarray: Float64 selected output series.
    Assumptions:
        Any NaN in OHLC emits NaN output; `range==0` makes `*_pct` outputs NaN.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = open_.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        open_value = float(open_[time_index])
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])

        if (
            is_nan(open_value)
            or is_nan(high_value)
            or is_nan(low_value)
            or is_nan(close_value)
        ):
            out[time_index] = np.nan
            continue

        range_value = high_value - low_value
        body_value = abs(close_value - open_value)

        oc_max = open_value
        if close_value > oc_max:
            oc_max = close_value

        oc_min = open_value
        if close_value < oc_min:
            oc_min = close_value

        upper_wick_value = high_value - oc_max
        lower_wick_value = oc_min - low_value

        if mode == 0:
            out[time_index] = body_value
            continue
        if mode == 1:
            out[time_index] = range_value
            continue
        if mode == 2:
            out[time_index] = upper_wick_value
            continue
        if mode == 3:
            out[time_index] = lower_wick_value
            continue

        if range_value == 0.0:
            out[time_index] = np.nan
            continue

        if mode == 4:
            out[time_index] = body_value / range_value
            continue
        if mode == 5:
            out[time_index] = upper_wick_value / range_value
            continue
        out[time_index] = lower_wick_value / range_value

    return out


@nb.njit(cache=True)
def _pivot_series_f64(values: np.ndarray, left: int, right: int, mode: int) -> np.ndarray:
    """
    Compute one pivot series with strict confirmation and shift-confirm semantics.

    Args:
        values: Input high or low series.
        left: Left confirmation window.
        right: Right confirmation window.
        mode: `0=pivot_high`, `1=pivot_low`.
    Returns:
        np.ndarray: Float64 sparse pivot-value series.
    Assumptions:
        `shift_confirm=true`: pivot at center index is emitted only on confirm index `center+right`.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = values.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        out[time_index] = np.nan

    for time_index in range(t_size):
        center = time_index - right
        start = center - left
        end = center + right

        if center < 0 or start < 0 or end >= t_size:
            continue

        candidate = float(values[center])
        if is_nan(candidate):
            continue

        valid = True
        for idx in range(start, end + 1):
            value = float(values[idx])
            if is_nan(value):
                valid = False
                break
            if idx == center:
                continue
            if mode == 0:
                if value >= candidate:
                    valid = False
                    break
            elif value <= candidate:
                valid = False
                break

        if valid:
            out[time_index] = candidate

    return out


@nb.njit(parallel=True, cache=True)
def _zscore_variants_f64(source_variants: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute z-score matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 z-score matrix `(V, T)`.
    Assumptions:
        `sd==0` produces NaN to enforce deterministic div-by-zero policy.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporary vectors.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        series = source_variants[variant_index, :]
        window = int(windows[variant_index])
        mean = _rolling_mean_series_f64(series, window)
        variance = _rolling_variance_series_f64(series, window)

        for time_index in range(t_size):
            source_value = float(series[time_index])
            mean_value = float(mean[time_index])
            variance_value = float(variance[time_index])

            if (
                is_nan(source_value)
                or is_nan(mean_value)
                or is_nan(variance_value)
                or variance_value == 0.0
            ):
                out[variant_index, time_index] = np.nan
                continue

            out[variant_index, time_index] = (source_value - mean_value) / math.sqrt(
                variance_value
            )

    return out


@nb.njit(parallel=True, cache=True)
def _percent_rank_variants_f64(source_variants: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute rolling percent-rank matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 percent-rank matrix `(V, T)`.
    Assumptions:
        Percent rank uses `tie_mode='le'` and `scale=100.0`.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _percent_rank_series_f64(
            source_variants[variant_index, :],
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _candle_stats_atr_norm_variants_f64(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_variants: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Compute ATR-normalized candle-stats matrix for per-variant ATR vectors.

    Args:
        open_: Open-price series.
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        atr_variants: Variant-major ATR matrix `(V, T)`.
        mode: Raw candle-stats output selector (`0=body,1=range,2=upper,3=lower`).
    Returns:
        np.ndarray: Float64 normalized matrix `(V, T)`.
    Assumptions:
        `atr==0` and NaN propagate to NaN output.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and one raw-series vector.
    """
    variants = atr_variants.shape[0]
    t_size = atr_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    raw = _candle_stats_series_f64(open_, high, low, close, mode)

    for variant_index in nb.prange(variants):
        for time_index in range(t_size):
            raw_value = float(raw[time_index])
            atr_value = float(atr_variants[variant_index, time_index])
            if is_nan(raw_value) or is_nan(atr_value) or atr_value == 0.0:
                out[variant_index, time_index] = np.nan
            else:
                out[variant_index, time_index] = raw_value / atr_value

    return out


@nb.njit(parallel=True, cache=True)
def _pivots_variants_f64(
    values: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Compute pivot matrix for per-variant `(left, right)` parameters.

    Args:
        values: Input high or low series.
        lefts: Per-variant left windows.
        rights: Per-variant right windows.
        mode: `0=pivot_high`, `1=pivot_low`.
    Returns:
        np.ndarray: Float64 pivot matrix `(V, T)`.
    Assumptions:
        Right-window confirmation follows `shift_confirm=true` semantics.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = lefts.shape[0]
    t_size = values.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _pivot_series_f64(
            values,
            int(lefts[variant_index]),
            int(rights[variant_index]),
            mode,
        )

    return out


@nb.njit(parallel=True, cache=True)
def _distance_to_ma_norm_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    atr_variants: np.ndarray,
) -> np.ndarray:
    """
    Compute `(source - ema(source, window)) / atr(window)` per variant.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant window vector.
        atr_variants: Variant-major ATR matrix `(V, T)`.
    Returns:
        np.ndarray: Float64 normalized-distance matrix `(V, T)`.
    Assumptions:
        EMA uses reset-on-NaN semantics and `atr==0` produces NaN.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant EMA vectors.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        source = source_variants[variant_index, :]
        ema = _ema_series_f64(source, int(windows[variant_index]))

        for time_index in range(t_size):
            source_value = float(source[time_index])
            ema_value = float(ema[time_index])
            atr_value = float(atr_variants[variant_index, time_index])

            if (
                is_nan(source_value)
                or is_nan(ema_value)
                or is_nan(atr_value)
                or atr_value == 0.0
            ):
                out[variant_index, time_index] = np.nan
            else:
                out[variant_index, time_index] = (source_value - ema_value) / atr_value

    return out


def is_supported_structure_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by structure kernels.

    Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      docs/architecture/indicators/indicators_formula.yaml

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by structure kernels.
    Assumptions:
        Indicator aliases are explicit wrapper ids in v1.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized_id = _normalize_structure_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_STRUCTURE_IDS


def compute_structure_grid_f32(
    *,
    indicator_id: str,
    source_variants: np.ndarray | None = None,
    open: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    atr_windows: np.ndarray | None = None,
    lefts: np.ndarray | None = None,
    rights: np.ndarray | None = None,
    precision: str = PRECISION_MODE_FLOAT64,
) -> np.ndarray:
    """
    Compute structure indicator matrix `(V, T)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      docs/architecture/indicators/indicators_formula.yaml

    Args:
        indicator_id: Structure indicator identifier.
        source_variants: Optional `(V, T)` source matrix for source-based indicators.
        open: Optional open-price series.
        high: Optional high-price series.
        low: Optional low-price series.
        close: Optional close-price series.
        windows: Optional per-variant `window` vector.
        atr_windows: Optional per-variant `atr_window` vector.
        lefts: Optional per-variant `left` vector.
        rights: Optional per-variant `right` vector.
        precision: Precision mode (`float32`, `mixed`, `float64`) from engine policy dispatch.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)`.
    Assumptions:
        v1 returns one primary output per indicator id; wrappers expose non-primary outputs.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_structure_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_STRUCTURE_IDS:
        raise ValueError(f"unsupported structure indicator_id: {indicator_id!r}")
    _validate_precision_mode(precision=precision)
    core_dtype = np.float64 if precision == PRECISION_MODE_FLOAT64 else np.float32

    if normalized_id in _CANDLE_STATS_OUTPUT_MODE:
        open_f64 = _prepare_series(name="open", values=open, dtype=core_dtype)
        high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
        low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
        close_f64 = _prepare_series(name="close", values=close, dtype=core_dtype)
        _ensure_same_length_ohlc(open=open_f64, high=high_f64, low=low_f64, close=close_f64)

        mode = _CANDLE_STATS_OUTPUT_MODE[normalized_id]
        series = _candle_stats_series_f64(open_f64, high_f64, low_f64, close_f64, mode)
        out_f64 = np.ascontiguousarray(series.reshape(1, series.shape[0]))
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in _CANDLE_STATS_ATR_OUTPUT_MODE:
        open_f64 = _prepare_series(name="open", values=open, dtype=core_dtype)
        high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
        low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
        close_f64 = _prepare_series(name="close", values=close, dtype=core_dtype)
        _ensure_same_length_ohlc(open=open_f64, high=high_f64, low=low_f64, close=close_f64)

        atr_windows_i64 = _prepare_int_variants(name="atr_windows", values=atr_windows)
        atr_variants_f32 = compute_volatility_grid_f32(
            indicator_id="volatility.atr",
            high=high_f64,
            low=low_f64,
            close=close_f64,
            windows=atr_windows_i64,
            precision=precision,
        )
        atr_variants_f64 = np.ascontiguousarray(atr_variants_f32, dtype=np.float64)

        mode = _CANDLE_STATS_ATR_OUTPUT_MODE[normalized_id]
        out_f64 = _candle_stats_atr_norm_variants_f64(
            open_f64,
            high_f64,
            low_f64,
            close_f64,
            atr_variants_f64,
            mode,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in {"structure.zscore", "structure.percent_rank"}:
        source_f64 = _prepare_source_variants(values=source_variants, dtype=core_dtype)
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=source_f64.shape[0],
        )

        if normalized_id == "structure.zscore":
            out_f64 = _zscore_variants_f64(source_f64, windows_i64)
        else:
            out_f64 = _percent_rank_variants_f64(source_f64, windows_i64)

        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in _PIVOTS_OUTPUT_MODE:
        high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
        low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
        _ensure_same_length_hl(high=high_f64, low=low_f64)

        lefts_i64 = _prepare_int_variants(name="lefts", values=lefts)
        rights_i64 = _prepare_int_variants(
            name="rights",
            values=rights,
            expected_size=lefts_i64.shape[0],
        )

        mode = _PIVOTS_OUTPUT_MODE[normalized_id]
        if mode == 0:
            out_f64 = _pivots_variants_f64(high_f64, lefts_i64, rights_i64, mode)
        else:
            out_f64 = _pivots_variants_f64(low_f64, lefts_i64, rights_i64, mode)

        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    source_f64 = _prepare_source_variants(values=source_variants, dtype=core_dtype)
    windows_i64 = _prepare_int_variants(
        name="windows",
        values=windows,
        expected_size=source_f64.shape[0],
    )

    high_f64 = _prepare_series(name="high", values=high, dtype=core_dtype)
    low_f64 = _prepare_series(name="low", values=low, dtype=core_dtype)
    close_f64 = _prepare_series(name="close", values=close, dtype=core_dtype)
    _ensure_same_length_hlc(high=high_f64, low=low_f64, close=close_f64)
    if high_f64.shape[0] != source_f64.shape[1]:
        raise ValueError(
            "source_variants time dimension must match OHLC length: "
            f"source_t={source_f64.shape[1]}, ohlc_t={high_f64.shape[0]}"
        )

    atr_variants_f32 = compute_volatility_grid_f32(
        indicator_id="volatility.atr",
        high=high_f64,
        low=low_f64,
        close=close_f64,
        windows=windows_i64,
        precision=precision,
    )
    atr_variants_f64 = np.ascontiguousarray(atr_variants_f32, dtype=np.float64)

    out_f64 = _distance_to_ma_norm_variants_f64(source_f64, windows_i64, atr_variants_f64)
    return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))


def _validate_precision_mode(*, precision: str) -> None:
    """
    Validate structure kernel precision mode against shared precision policy constants.

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
        Series may include NaN holes from CandleFeed ACL.
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
        Variant order already matches deterministic axis materialization.
    Raises:
        ValueError: If matrix is missing, malformed, or empty.
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
        expected_size: Optional expected variant count for alignment checks.
    Returns:
        np.ndarray: Int64 C-contiguous vector.
    Assumptions:
        Structure windows/left/right values are strictly positive integers.
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


def _ensure_same_length_ohlc(
    *,
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> None:
    """
    Validate equal length for OHLC aligned series.

    Args:
        open: Open-price series.
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
    t_size = open.shape[0]
    if high.shape[0] != t_size or low.shape[0] != t_size or close.shape[0] != t_size:
        raise ValueError("open, high, low, close lengths must match")


def _ensure_same_length_hl(*, high: np.ndarray, low: np.ndarray) -> None:
    """
    Validate equal length for high/low aligned series.

    Args:
        high: High-price series.
        low: Low-price series.
    Returns:
        None.
    Assumptions:
        Inputs are one-dimensional arrays.
    Raises:
        ValueError: If series lengths differ.
    Side Effects:
        None.
    """
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low lengths must match")


def _ensure_same_length_hlc(*, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
    """
    Validate equal length for HLC aligned series.

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


def _normalize_structure_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize structure indicator identifier.

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized indicator id.
    Assumptions:
        Structure wrapper ids are stable public ids in v1.
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
    "compute_structure_grid_f32",
    "is_supported_structure_indicator",
]
