"""
Numpy oracle implementation for MA-family indicators.

Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels.ma,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine
"""

from __future__ import annotations

import math

import numpy as np

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


def is_supported_ma_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by MA oracle implementation.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by MA oracle.
    Assumptions:
        Identifier normalization is delegated to `_normalize_ma_indicator_id`.
    Raises:
        ValueError: If identifier is blank.
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
    Compute MA indicator matrix `(T, W)` using pure NumPy/Python loops.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: MA indicator identifier.
        source: Source series vector.
        windows: Window values vector.
        volume: Optional volume vector for `ma.vwma`.
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
        out_f64 = _sma_grid_f64(source=source_f64, windows=windows_i64)
    elif normalized_id == "ma.ema":
        out_f64 = _ema_like_grid_f64(source=source_f64, windows=windows_i64, use_rma_alpha=False)
    elif normalized_id in {"ma.rma", "ma.smma"}:
        out_f64 = _ema_like_grid_f64(source=source_f64, windows=windows_i64, use_rma_alpha=True)
    elif normalized_id in {"ma.wma", "ma.lwma"}:
        out_f64 = _wma_grid_f64(source=source_f64, windows=windows_i64)
    elif normalized_id == "ma.vwma":
        if volume is None:
            raise ValueError("ma.vwma requires volume series")
        volume_f64 = _prepare_volume(volume=volume, expected_length=source_f64.shape[0])
        out_f64 = _vwma_grid_f64(source=source_f64, volume=volume_f64, windows=windows_i64)
    elif normalized_id == "ma.dema":
        out_f64 = _dema_grid_f64(source=source_f64, windows=windows_i64)
    elif normalized_id == "ma.tema":
        out_f64 = _tema_grid_f64(source=source_f64, windows=windows_i64)
    elif normalized_id == "ma.zlema":
        out_f64 = _zlema_grid_f64(source=source_f64, windows=windows_i64)
    elif normalized_id == "ma.hma":
        out_f64 = _hma_grid_f64(source=source_f64, windows=windows_i64)
    else:  # pragma: no cover
        raise ValueError(f"unsupported MA indicator_id: {indicator_id!r}")

    return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))


def _prepare_source_and_windows(
    *,
    source: np.ndarray,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize source/windows inputs for MA oracle functions.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

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
    Normalize volume vector for VWMA oracle function.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

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


def _rolling_sum_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-sum series with window-NaN policy.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py

    Args:
        source: Float64 source series.
        window: Positive rolling window.
    Returns:
        np.ndarray: Float64 rolling-sum series.
    Assumptions:
        Any NaN inside window produces NaN output.
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
            out[time_index] = running_sum

    return out


def _sma_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute SMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        Windows are pre-validated by wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        rolling_sum = _rolling_sum_series_f64(source=source, window=window)
        for time_index in range(t_size):
            value = float(rolling_sum[time_index])
            if math.isnan(value):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = value / float(window)

    return out


def _ewma_series_f64(*, source: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute one EWMA-like series with NaN reset policy.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        alpha: Smoothing factor in `(0, 1]`.
    Returns:
        np.ndarray: Float64 EWMA series.
    Assumptions:
        NaN source value resets state and emits NaN on the same index.
    Raises:
        ValueError: If alpha is outside `(0, 1]`.
    Side Effects:
        Allocates one output array.
    """
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

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


def _ema_like_grid_f64(
    *,
    source: np.ndarray,
    windows: np.ndarray,
    use_rma_alpha: bool,
) -> np.ndarray:
    """
    Compute EMA/RMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
        use_rma_alpha: True for RMA alpha (`1/w`), False for EMA alpha (`2/(w+1)`).
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        Windows are pre-validated by wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        alpha = (1.0 / float(window)) if use_rma_alpha else (2.0 / (float(window) + 1.0))
        series = _ewma_series_f64(source=source, alpha=alpha)
        out[:, window_index] = series

    return out


def _wma_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one linear-WMA series with rolling-window NaN policy.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        window: Positive rolling window.
    Returns:
        np.ndarray: Float64 WMA series.
    Assumptions:
        Any NaN inside window produces NaN output.
    Raises:
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.full(t_size, np.nan, dtype=np.float64)
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
        if math.isnan(raw_value):
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

        if math.isnan(incoming):
            nan_count += 1
        else:
            incoming_value = incoming

        if math.isnan(outgoing):
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


def _wma_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute linear-WMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        Windows are pre-validated by wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        out[:, window_index] = _wma_series_f64(source=source, window=window)

    return out


def _vwma_grid_f64(*, source: np.ndarray, volume: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute VWMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        volume: Float64 volume series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        Any NaN in source/volume inside window yields NaN output.
    Raises:
        ValueError: If source and volume lengths differ.
    Side Effects:
        Allocates intermediate vectors and output matrix.
    """
    if source.shape[0] != volume.shape[0]:
        raise ValueError("source and volume lengths must match")

    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    pv = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        price = float(source[time_index])
        vol = float(volume[time_index])
        if math.isnan(price) or math.isnan(vol):
            pv[time_index] = np.nan
        else:
            pv[time_index] = price * vol

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        numerator = _rolling_sum_series_f64(source=pv, window=window)
        denominator = _rolling_sum_series_f64(source=volume, window=window)

        for time_index in range(t_size):
            num = float(numerator[time_index])
            den = float(denominator[time_index])
            if math.isnan(num) or math.isnan(den) or den == 0.0:
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = num / den

    return out


def _dema_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute DEMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        EMA stages use NaN reset policy.
    Raises:
        None.
    Side Effects:
        Allocates intermediate EMA arrays.
    """
    ema1 = _ema_like_grid_f64(source=source, windows=windows, use_rma_alpha=False)
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        alpha = 2.0 / (float(window) + 1.0)
        ema2 = _ewma_series_f64(source=ema1[:, window_index], alpha=alpha)

        for time_index in range(t_size):
            value_ema1 = float(ema1[time_index, window_index])
            value_ema2 = float(ema2[time_index])
            if math.isnan(value_ema1) or math.isnan(value_ema2):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = (2.0 * value_ema1) - value_ema2

    return out


def _tema_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute TEMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        EMA stages use NaN reset policy.
    Raises:
        None.
    Side Effects:
        Allocates intermediate EMA arrays.
    """
    ema1 = _ema_like_grid_f64(source=source, windows=windows, use_rma_alpha=False)
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        alpha = 2.0 / (float(window) + 1.0)
        ema2 = _ewma_series_f64(source=ema1[:, window_index], alpha=alpha)
        ema3 = _ewma_series_f64(source=ema2, alpha=alpha)

        for time_index in range(t_size):
            value_ema1 = float(ema1[time_index, window_index])
            value_ema2 = float(ema2[time_index])
            value_ema3 = float(ema3[time_index])
            if math.isnan(value_ema1) or math.isnan(value_ema2) or math.isnan(value_ema3):
                out[time_index, window_index] = np.nan
            else:
                out[time_index, window_index] = (
                    (3.0 * value_ema1) - (3.0 * value_ema2) + value_ema3
                )

    return out


def _zlema_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute ZLEMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        Lag is `floor((window - 1) / 2)` for each window.
    Raises:
        None.
    Side Effects:
        Allocates per-window adjusted vectors and output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
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
            if math.isnan(current) or math.isnan(lagged):
                adjusted[time_index] = np.nan
            else:
                adjusted[time_index] = current + (current - lagged)

        out[:, window_index] = _ewma_series_f64(source=adjusted, alpha=alpha)

    return out


def _hma_grid_f64(*, source: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute HMA matrix in float64.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py

    Args:
        source: Float64 source series.
        windows: Positive integer windows vector.
    Returns:
        np.ndarray: Float64 matrix `(T, W)`.
    Assumptions:
        `w2=floor(window/2)` and `sqrt_w=floor(sqrt(window))` with lower bound `1`.
    Raises:
        None.
    Side Effects:
        Allocates intermediate WMA vectors and output matrix.
    """
    t_size = source.shape[0]
    w_size = windows.shape[0]
    out = np.empty((t_size, w_size), dtype=np.float64)

    for window_index, window_raw in enumerate(windows):
        window = int(window_raw)
        window_half = max(1, window // 2)
        window_sqrt = max(1, int(math.sqrt(float(window))))

        wma_half = _wma_series_f64(source=source, window=window_half)
        wma_full = _wma_series_f64(source=source, window=window)

        diff = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            value_half = float(wma_half[time_index])
            value_full = float(wma_full[time_index])
            if math.isnan(value_half) or math.isnan(value_full):
                diff[time_index] = np.nan
            else:
                diff[time_index] = (2.0 * value_half) - value_full

        out[:, window_index] = _wma_series_f64(source=diff, window=window_sqrt)

    return out


def _normalize_ma_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize MA indicator identifier and resolve aliases.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/domain/definitions/ma.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py

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
