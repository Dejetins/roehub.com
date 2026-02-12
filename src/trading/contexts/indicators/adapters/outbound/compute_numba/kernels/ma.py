"""
Numba kernels for MA-family indicators.

Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels._common,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.adapters.outbound.compute_numpy.ma
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import ewma_grid_f64, is_nan, rolling_mean_grid_f64, rolling_sum_grid_f64

_SUPPORTED_MA_IDS = {
    "ma.sma",
    "ma.ema",
    "ma.wma",
    "ma.lwma",
    "ma.rma",
    "ma.smma",
    "ma.vwma",
    "ma.dema",
    "ma.tema",
    "ma.zlema",
    "ma.hma",
}


@nb.njit(cache=True)
def _ewma_series_f64(source: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute one EWMA-like series with NaN reset policy.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        alpha: Smoothing factor in `(0, 1]`.
    Returns:
        np.ndarray: One-dimensional float64 result with NaN reset behavior.
    Assumptions:
        `source` is contiguous and `alpha` is validated by the caller.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
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
def _wma_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one linear-WMA series with rolling-window NaN policy.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        window: Positive integer window size.
    Returns:
        np.ndarray: One-dimensional float64 linear-WMA result.
    Assumptions:
        `window` is positive and `source` is contiguous.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        out[time_index] = np.nan

    if t_size == 0:
        return out
    if window > t_size:
        return out

    denominator = (float(window) * (float(window) + 1.0)) / 2.0
    sum_x = 0.0
    weighted = 0.0
    nan_count = 0

    for offset in range(window):
        raw_value = float(source[offset])
        weight = float(offset + 1)
        if is_nan(raw_value):
            nan_count += 1
            continue
        sum_x += raw_value
        weighted += weight * raw_value

    if nan_count == 0:
        out[window - 1] = weighted / denominator

    for time_index in range(window, t_size):
        incoming = float(source[time_index])
        outgoing = float(source[time_index - window])

        incoming_value = 0.0
        outgoing_value = 0.0

        if is_nan(incoming):
            nan_count += 1
        else:
            incoming_value = incoming

        if is_nan(outgoing):
            nan_count -= 1
        else:
            outgoing_value = outgoing

        weighted = weighted - sum_x + (float(window) * incoming_value)
        sum_x = sum_x - outgoing_value + incoming_value

        if nan_count == 0:
            out[time_index] = weighted / denominator
        else:
            out[time_index] = np.nan

    return out


@nb.njit(parallel=True, cache=True)
def _wma_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute linear-WMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        Window values are pre-validated by wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and temporary vectors per window worker.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = int(windows[window_index])
        series = _wma_series_f64(source, window)
        for time_index in range(t_size):
            out[time_index, window_index] = series[time_index]

    return out


@nb.njit(parallel=True, cache=True)
def _vwma_grid_f64(source: np.ndarray, volume: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute VWMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 price/source series.
        volume: One-dimensional float64 volume series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        `source` and `volume` have identical length and aligned indices.
    Raises:
        None.
    Side Effects:
        Allocates intermediate `pv`, rolling sums, and one output matrix.
    """
    t_size = source.shape[0]
    pv = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        price = float(source[time_index])
        vol = float(volume[time_index])
        if is_nan(price) or is_nan(vol):
            pv[time_index] = np.nan
        else:
            pv[time_index] = price * vol

    numerator = rolling_sum_grid_f64(pv, windows)
    denominator = rolling_sum_grid_f64(volume, windows)

    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        for time_index in range(t_size):
            num = float(numerator[time_index, window_index])
            den = float(denominator[time_index, window_index])
            if is_nan(num) or is_nan(den) or den == 0.0:
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = num / den

    return out


@nb.njit(parallel=True, cache=True)
def _dema_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute DEMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        EWMA primitive follows NaN-reset semantics.
    Raises:
        None.
    Side Effects:
        Allocates intermediate EMA matrices and output matrix.
    """
    ema1 = ewma_grid_f64(source, windows, False)
    t_size = ema1.shape[0]
    w_size = ema1.shape[1]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = int(windows[window_index])
        alpha = 2.0 / (float(window) + 1.0)
        ema2 = _ewma_series_f64(ema1[:, window_index], alpha)
        for time_index in range(t_size):
            value_ema1 = float(ema1[time_index, window_index])
            value_ema2 = float(ema2[time_index])
            if is_nan(value_ema1) or is_nan(value_ema2):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = (2.0 * value_ema1) - value_ema2

    return out


@nb.njit(parallel=True, cache=True)
def _tema_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute TEMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        EWMA primitive follows NaN-reset semantics.
    Raises:
        None.
    Side Effects:
        Allocates intermediate EMA matrices and output matrix.
    """
    ema1 = ewma_grid_f64(source, windows, False)
    t_size = ema1.shape[0]
    w_size = ema1.shape[1]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = int(windows[window_index])
        alpha = 2.0 / (float(window) + 1.0)
        ema2 = _ewma_series_f64(ema1[:, window_index], alpha)
        ema3 = _ewma_series_f64(ema2, alpha)

        for time_index in range(t_size):
            value_ema1 = float(ema1[time_index, window_index])
            value_ema2 = float(ema2[time_index])
            value_ema3 = float(ema3[time_index])
            if is_nan(value_ema1) or is_nan(value_ema2) or is_nan(value_ema3):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = (
                    (3.0 * value_ema1) - (3.0 * value_ema2) + value_ema3
                )

    return out


@nb.njit(parallel=True, cache=True)
def _zlema_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute ZLEMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        Lag is `floor((window - 1) / 2)` for each window.
    Raises:
        None.
    Side Effects:
        Allocates per-window adjusted series and one output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = int(windows[window_index])
        lag = int((window - 1) // 2)
        alpha = 2.0 / (float(window) + 1.0)

        adjusted = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            current = float(source[time_index])
            if lag == 0:
                adjusted[time_index] = current
                continue

            lagged_index = time_index - lag
            if lagged_index < 0:
                adjusted[time_index] = np.nan
                continue

            lagged = float(source[lagged_index])
            if is_nan(current) or is_nan(lagged):
                adjusted[time_index] = np.nan
            else:
                adjusted[time_index] = current + (current - lagged)

        zlema = _ewma_series_f64(adjusted, alpha)
        for time_index in range(t_size):
            out[time_index, window_index] = zlema[time_index]

    return out


@nb.njit(parallel=True, cache=True)
def _hma_grid_f64(source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute HMA matrix for multiple windows.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: One-dimensional float64 source series.
        windows: One-dimensional positive integer windows.
    Returns:
        np.ndarray: Float64 matrix with shape `(T, W)`.
    Assumptions:
        `w2=floor(window/2)` and `sqrt_w=floor(sqrt(window))` with lower bound `1`.
    Raises:
        None.
    Side Effects:
        Allocates intermediate WMA series and output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index in nb.prange(w_size):
        window = int(windows[window_index])
        window_half = int(window // 2)
        if window_half < 1:
            window_half = 1

        window_sqrt = int(math.sqrt(float(window)))
        if window_sqrt < 1:
            window_sqrt = 1

        wma_half = _wma_series_f64(source, window_half)
        wma_full = _wma_series_f64(source, window)

        diff = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            value_half = float(wma_half[time_index])
            value_full = float(wma_full[time_index])
            if is_nan(value_half) or is_nan(value_full):
                diff[time_index] = np.nan
            else:
                diff[time_index] = (2.0 * value_half) - value_full

        hma = _wma_series_f64(diff, window_sqrt)
        for time_index in range(t_size):
            out[time_index, window_index] = hma[time_index]

    return out


def is_supported_ma_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by MA kernels.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by MA kernels.
    Assumptions:
        Identifier normalization is delegated to `_normalize_ma_indicator_id`.
    Raises:
        None.
    Side Effects:
        None.
    """
    normalized_id = _normalize_ma_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_MA_IDS


def compute_ma_grid_f32(
    *,
    indicator_id: str,
    source: np.ndarray,
    windows: np.ndarray,
    volume: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute MA indicator matrix `(T, W)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        indicator_id: MA indicator identifier (`ma.sma`, `ma.ema`, ...).
        source: Source series vector.
        windows: Window values vector.
        volume: Optional volume vector required by `ma.vwma`.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(T, W)`.
    Assumptions:
        Caller provides aligned vectors for `source` and optional `volume`.
    Raises:
        ValueError: If indicator id is unsupported, windows are invalid, or volume is missing.
    Side Effects:
        Allocates intermediate float64 matrices before casting to float32.
    """
    normalized_id = _normalize_ma_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_MA_IDS:
        raise ValueError(f"unsupported MA indicator_id: {indicator_id!r}")

    source_f64, windows_i64 = _prepare_source_and_windows(source=source, windows=windows)

    if normalized_id == "ma.sma":
        out_f64 = rolling_mean_grid_f64(source_f64, windows_i64)
    elif normalized_id == "ma.ema":
        out_f64 = ewma_grid_f64(source_f64, windows_i64, False)
    elif normalized_id in {"ma.rma", "ma.smma"}:
        out_f64 = ewma_grid_f64(source_f64, windows_i64, True)
    elif normalized_id in {"ma.wma", "ma.lwma"}:
        out_f64 = _wma_grid_f64(source_f64, windows_i64)
    elif normalized_id == "ma.vwma":
        if volume is None:
            raise ValueError("ma.vwma requires volume series")
        volume_f64 = _prepare_volume(volume=volume, expected_length=source_f64.shape[0])
        out_f64 = _vwma_grid_f64(source_f64, volume_f64, windows_i64)
    elif normalized_id == "ma.dema":
        out_f64 = _dema_grid_f64(source_f64, windows_i64)
    elif normalized_id == "ma.tema":
        out_f64 = _tema_grid_f64(source_f64, windows_i64)
    elif normalized_id == "ma.zlema":
        out_f64 = _zlema_grid_f64(source_f64, windows_i64)
    elif normalized_id == "ma.hma":
        out_f64 = _hma_grid_f64(source_f64, windows_i64)
    else:  # pragma: no cover
        raise ValueError(f"unsupported MA indicator_id: {indicator_id!r}")

    out_f32 = np.ascontiguousarray(out_f64.astype(np.float32, copy=False))
    return out_f32


def _prepare_source_and_windows(
    *,
    source: np.ndarray,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize source/windows inputs for MA kernels.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        source: Source series vector.
        windows: Window values vector.
    Returns:
        tuple[np.ndarray, np.ndarray]: `(source_f64, windows_i64)` contiguous arrays.
    Assumptions:
        Source can contain NaNs and windows must be positive integers.
    Raises:
        ValueError: If array shapes are invalid or window values are non-positive.
    Side Effects:
        Allocates normalized contiguous arrays.
    """
    source_f64 = np.ascontiguousarray(source, dtype=np.float64)
    if source_f64.ndim != 1:
        raise ValueError("source must be a 1D array")

    windows_i64 = np.ascontiguousarray(windows, dtype=np.int64)
    if windows_i64.ndim != 1:
        raise ValueError("windows must be a 1D array")
    if windows_i64.shape[0] == 0:
        raise ValueError("windows must contain at least one value")
    if np.any(windows_i64 <= 0):
        raise ValueError("windows must contain only positive integers")

    return source_f64, windows_i64


def _prepare_volume(*, volume: np.ndarray, expected_length: int) -> np.ndarray:
    """
    Normalize volume vector for VWMA kernels.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        volume: Volume vector.
        expected_length: Required length matching source series length.
    Returns:
        np.ndarray: Float64 C-contiguous volume vector.
    Assumptions:
        Volume can contain NaN values per CandleFeed policy.
    Raises:
        ValueError: If volume shape or length is invalid.
    Side Effects:
        Allocates one normalized array.
    """
    volume_f64 = np.ascontiguousarray(volume, dtype=np.float64)
    if volume_f64.ndim != 1:
        raise ValueError("volume must be a 1D array")
    if volume_f64.shape[0] != expected_length:
        raise ValueError(
            "volume length must match source length: "
            f"expected={expected_length}, got={volume_f64.shape[0]}"
        )
    return volume_f64


def _normalize_ma_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize MA indicator identifier and resolve aliases.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/definitions/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized id with aliases resolved.
    Assumptions:
        Alias mapping is stable for v1 (`ma.smma` -> `ma.rma`).
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized = indicator_id.strip().lower()
    if not normalized:
        raise ValueError("indicator_id must be non-empty")
    if normalized == "ma.smma":
        return "ma.rma"
    return normalized


__all__ = [
    "compute_ma_grid_f32",
    "is_supported_ma_indicator",
]
