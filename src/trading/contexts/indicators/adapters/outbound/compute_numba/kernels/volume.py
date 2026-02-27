"""
Numba kernels for volume-family indicators.

Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
Related: docs/architecture/indicators/indicators_formula.yaml,
  trading.contexts.indicators.adapters.outbound.compute_numpy.volume,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.domain.definitions.volume
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import is_nan

_SUPPORTED_VOLUME_IDS = {
    "volume.ad_line",
    "volume.cmf",
    "volume.mfi",
    "volume.obv",
    "volume.volume_sma",
    "volume.vwap",
    "volume.vwap_deviation",
}


@nb.njit(cache=True)
def _rolling_sum_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-sum series with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-sum series.
    Assumptions:
        Any NaN inside the active window yields NaN output for that index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
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
            out[time_index] = running_sum

    return out


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
        Window values are validated by Python wrapper before kernel execution.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    rolling_sum = _rolling_sum_series_f64(source, window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        value = float(rolling_sum[time_index])
        if is_nan(value):
            out[time_index] = np.nan
        else:
            out[time_index] = value / float(window)

    return out


@nb.njit(cache=True)
def _rolling_variance_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-variance series (`ddof=0`) with NaN-window policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-variance series.
    Assumptions:
        Any NaN inside the active window yields NaN output for that index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
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
def _rolling_std_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-standard-deviation series (`ddof=0`) with NaN-window policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-standard-deviation series.
    Assumptions:
        Standard deviation is square root of rolling variance from `_rolling_variance_series_f64`.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    variance = _rolling_variance_series_f64(source, window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        value = float(variance[time_index])
        if is_nan(value):
            out[time_index] = np.nan
        else:
            out[time_index] = math.sqrt(value)

    return out


@nb.njit(cache=True)
def _shift_series_f64(source: np.ndarray, periods: int) -> np.ndarray:
    """
    Shift one series by integer periods with NaN fill for out-of-range indices.

    Args:
        source: Float64 source series.
        periods: Shift periods (`>0` past, `<0` future).
    Returns:
        np.ndarray: Shifted float64 series.
    Assumptions:
        Shift semantics follow `docs/architecture/indicators/indicators_formula.yaml`.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        source_index = time_index - periods
        if source_index < 0 or source_index >= t_size:
            out[time_index] = np.nan
        else:
            out[time_index] = float(source[source_index])

    return out


@nb.njit(cache=True)
def _diff_series_f64(source: np.ndarray, periods: int) -> np.ndarray:
    """
    Compute discrete difference series with NaN reset boundaries.

    Args:
        source: Float64 source series.
        periods: Positive lag periods.
    Returns:
        np.ndarray: Float64 difference series.
    Assumptions:
        If current or lagged value is NaN, output is NaN at current index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        if time_index < periods:
            out[time_index] = np.nan
            continue

        current = float(source[time_index])
        lagged = float(source[time_index - periods])
        if is_nan(current) or is_nan(lagged):
            out[time_index] = np.nan
        else:
            out[time_index] = current - lagged

    return out


@nb.njit(cache=True)
def _cumsum_reset_series_f64(source: np.ndarray) -> np.ndarray:
    """
    Compute cumulative sum series with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
    Returns:
        np.ndarray: Float64 cumulative-sum series.
    Assumptions:
        NaN source values emit NaN output and reset accumulation state.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    running = np.nan

    for time_index in range(t_size):
        value = float(source[time_index])
        if is_nan(value):
            running = np.nan
            out[time_index] = np.nan
            continue

        if is_nan(running):
            running = value
        else:
            running += value
        out[time_index] = running

    return out


@nb.njit(cache=True)
def _typical_price_series_f64(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute typical price series `(high + low + close) / 3` with NaN propagation.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
    Returns:
        np.ndarray: Float64 typical-price series.
    Assumptions:
        Any NaN in OHLC inputs yields NaN at corresponding output index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])
        if is_nan(high_value) or is_nan(low_value) or is_nan(close_value):
            out[time_index] = np.nan
        else:
            out[time_index] = (high_value + low_value + close_value) / 3.0

    return out


@nb.njit(cache=True)
def _ad_line_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute Accumulation/Distribution line with reset-on-NaN cumulative semantics.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
    Returns:
        np.ndarray: Float64 AD-line series.
    Assumptions:
        Division by zero in `(high-low)` yields NaN and resets cumulative state.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    running = np.nan

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])
        volume_value = float(volume[time_index])

        if (
            is_nan(high_value)
            or is_nan(low_value)
            or is_nan(close_value)
            or is_nan(volume_value)
        ):
            out[time_index] = np.nan
            running = np.nan
            continue

        denominator = high_value - low_value
        if denominator == 0.0:
            out[time_index] = np.nan
            running = np.nan
            continue

        numerator = ((close_value - low_value) - (high_value - close_value))
        mfm = numerator / denominator
        mfv = mfm * volume_value

        if is_nan(mfv):
            out[time_index] = np.nan
            running = np.nan
            continue

        if is_nan(running):
            running = mfv
        else:
            running += mfv
        out[time_index] = running

    return out


@nb.njit(cache=True)
def _obv_series_f64(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Compute On-Balance Volume with reset-on-NaN cumulative semantics.

    Args:
        close: Close-price series.
        volume: Volume series.
    Returns:
        np.ndarray: Float64 OBV series.
    Assumptions:
        NaN in close/volume emits NaN and resets cumulative state.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors and one output vector.
    """
    delta_close = _diff_series_f64(close, 1)
    t_size = close.shape[0]
    signed_volume = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        delta = float(delta_close[time_index])
        volume_value = float(volume[time_index])

        if is_nan(delta) or is_nan(volume_value):
            signed_volume[time_index] = np.nan
            continue

        direction = 0.0
        if delta > 0.0:
            direction = 1.0
        elif delta < 0.0:
            direction = -1.0

        signed_volume[time_index] = direction * volume_value

    return _cumsum_reset_series_f64(signed_volume)


@nb.njit(cache=True)
def _cmf_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute Chaikin Money Flow for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        window: Rolling window.
    Returns:
        np.ndarray: Float64 CMF series.
    Assumptions:
        Formula follows spec via `mfv = diff(ad_line, 1)` and rolling sums.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    ad_line = _ad_line_series_f64(high, low, close, volume)
    mfv = _diff_series_f64(ad_line, 1)
    sum_mfv = _rolling_sum_series_f64(mfv, window)
    sum_volume = _rolling_sum_series_f64(volume, window)

    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        numerator = float(sum_mfv[time_index])
        denominator = float(sum_volume[time_index])
        if is_nan(numerator) or is_nan(denominator) or denominator == 0.0:
            out[time_index] = np.nan
        else:
            out[time_index] = numerator / denominator

    return out


@nb.njit(cache=True)
def _mfi_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute Money Flow Index for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        window: Rolling window.
    Returns:
        np.ndarray: Float64 MFI series.
    Assumptions:
        NaN boundaries propagate through rolling sums via NaN-valued pos/neg flows.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    typical_price = _typical_price_series_f64(high, low, close)
    prev_tp = _shift_series_f64(typical_price, 1)
    t_size = high.shape[0]

    positive_flow = np.empty(t_size, dtype=np.float64)
    negative_flow = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        tp_value = float(typical_price[time_index])
        prev_tp_value = float(prev_tp[time_index])
        volume_value = float(volume[time_index])

        if is_nan(tp_value) or is_nan(prev_tp_value) or is_nan(volume_value):
            positive_flow[time_index] = np.nan
            negative_flow[time_index] = np.nan
            continue

        raw_money_flow = tp_value * volume_value
        if tp_value > prev_tp_value:
            positive_flow[time_index] = raw_money_flow
            negative_flow[time_index] = 0.0
        elif tp_value < prev_tp_value:
            positive_flow[time_index] = 0.0
            negative_flow[time_index] = raw_money_flow
        else:
            positive_flow[time_index] = 0.0
            negative_flow[time_index] = 0.0

    sum_positive = _rolling_sum_series_f64(positive_flow, window)
    sum_negative = _rolling_sum_series_f64(negative_flow, window)

    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        positive_value = float(sum_positive[time_index])
        negative_value = float(sum_negative[time_index])

        if is_nan(positive_value) or is_nan(negative_value):
            out[time_index] = np.nan
            continue

        if negative_value == 0.0:
            out[time_index] = 100.0
            continue

        money_ratio = positive_value / negative_value
        out[time_index] = 100.0 - (100.0 / (1.0 + money_ratio))

    return out


@nb.njit(cache=True)
def _vwap_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling VWAP for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        window: Rolling window.
    Returns:
        np.ndarray: Float64 VWAP series.
    Assumptions:
        Denominator `rolling_sum(volume)` equal to zero yields NaN output.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    typical_price = _typical_price_series_f64(high, low, close)
    t_size = high.shape[0]

    price_volume = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        price_value = float(typical_price[time_index])
        volume_value = float(volume[time_index])
        if is_nan(price_value) or is_nan(volume_value):
            price_volume[time_index] = np.nan
        else:
            price_volume[time_index] = price_value * volume_value

    numerator = _rolling_sum_series_f64(price_volume, window)
    denominator = _rolling_sum_series_f64(volume, window)

    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        numerator_value = float(numerator[time_index])
        denominator_value = float(denominator[time_index])
        if is_nan(numerator_value) or is_nan(denominator_value) or denominator_value == 0.0:
            out[time_index] = np.nan
        else:
            out[time_index] = numerator_value / denominator_value

    return out


@nb.njit(cache=True)
def _vwap_deviation_upper_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    mult: float,
) -> np.ndarray:
    """
    Compute VWAP deviation primary output (`vwap_upper`) for one variant.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        window: Rolling window.
        mult: Band multiplier.
    Returns:
        np.ndarray: Float64 VWAP upper-band series.
    Assumptions:
        Primary output in v1 is `vwap_upper`, which depends on both `window` and `mult`.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    _vwap_deviation_upper_series_into_f64(
        out,
        high,
        low,
        close,
        volume,
        window,
        mult,
    )
    return out


@nb.njit(cache=True)
def _vwap_deviation_upper_series_into_f64(
    out: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    mult: float,
) -> None:
    """
    Compute VWAP deviation primary output (`vwap_upper`) into preallocated buffer.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py

    Args:
        out: Preallocated float64 output vector.
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        window: Rolling window.
        mult: Band multiplier.
    Returns:
        None.
    Assumptions:
        Primary output in v1 is `vwap_upper`, which depends on both `window` and `mult`.
    Raises:
        None.
    Side Effects:
        Writes VWAP deviation upper-band values into `out` in-place.
    """
    t_size = high.shape[0]
    vwap = _vwap_series_f64(high, low, close, volume, window)
    diff = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])
        vwap_value = float(vwap[time_index])
        if is_nan(high_value) or is_nan(low_value) or is_nan(close_value) or is_nan(vwap_value):
            diff[time_index] = np.nan
            continue
        typical_price_value = (high_value + low_value + close_value) / 3.0
        diff[time_index] = typical_price_value - vwap_value

    stdev = _rolling_std_series_f64(diff, window)

    for time_index in range(t_size):
        vwap_value = float(vwap[time_index])
        stdev_value = float(stdev[time_index])
        if is_nan(vwap_value) or is_nan(stdev_value):
            out[time_index] = np.nan
        else:
            out[time_index] = vwap_value + (mult * stdev_value)


@nb.njit(parallel=True, cache=True)
def _cmf_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    """
    Compute CMF matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Window vector is positive and aligned with variants.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _cmf_series_f64(
            high,
            low,
            close,
            volume,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _mfi_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    """
    Compute MFI matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Window vector is positive and aligned with variants.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _mfi_series_f64(
            high,
            low,
            close,
            volume,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _volume_sma_variants_f64(volume: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute volume SMA matrix for per-variant windows.

    Args:
        volume: Volume series.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Window vector is positive and aligned with variants.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = volume.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _rolling_mean_series_f64(
            volume,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _vwap_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    """
    Compute VWAP matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Window vector is positive and aligned with variants.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _vwap_series_f64(
            high,
            low,
            close,
            volume,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _vwap_deviation_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    windows: np.ndarray,
    mults: np.ndarray,
) -> np.ndarray:
    """
    Compute VWAP upper-band matrix for per-variant `(window, mult)` pairs.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
        windows: Per-variant window vector.
        mults: Per-variant multiplier vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameter vectors are aligned by variant order in Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        _vwap_deviation_upper_series_into_f64(
            out[variant_index, :],
            high,
            low,
            close,
            volume,
            int(windows[variant_index]),
            float(mults[variant_index]),
        )

    return out


def is_supported_volume_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by volume kernels.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by volume kernels.
    Assumptions:
        Identifier normalization is delegated to `_normalize_volume_indicator_id`.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized_id = _normalize_volume_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_VOLUME_IDS


def compute_volume_grid_f32(
    *,
    indicator_id: str,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    mults: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute volume indicator matrix `(V, T)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/domain/definitions/volume.py

    Args:
        indicator_id: Volume indicator identifier.
        high: Optional high-price series.
        low: Optional low-price series.
        close: Optional close-price series.
        volume: Optional volume series.
        windows: Optional per-variant window values.
        mults: Optional per-variant multiplier values.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)` in deterministic variant order.
    Assumptions:
        Wrapper receives already-materialized per-variant vectors from compute engine.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_volume_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_VOLUME_IDS:
        raise ValueError(f"unsupported volume indicator_id: {indicator_id!r}")

    volume_f64 = _prepare_series(name="volume", values=volume)

    if normalized_id == "volume.obv":
        close_f64 = _prepare_series(name="close", values=close)
        _ensure_same_length_cv(close=close_f64, volume=volume_f64)
        out_f64 = np.ascontiguousarray(_obv_series_f64(close_f64, volume_f64).reshape(1, -1))
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volume.volume_sma":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _volume_sma_variants_f64(volume_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    high_f64 = _prepare_series(name="high", values=high)
    low_f64 = _prepare_series(name="low", values=low)
    close_f64 = _prepare_series(name="close", values=close)
    _ensure_same_length(high=high_f64, low=low_f64, close=close_f64, volume=volume_f64)

    if normalized_id == "volume.ad_line":
        out_f64 = np.ascontiguousarray(
            _ad_line_series_f64(high_f64, low_f64, close_f64, volume_f64).reshape(1, -1)
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    windows_i64 = _prepare_int_variants(name="windows", values=windows)

    if normalized_id == "volume.cmf":
        out_f64 = _cmf_variants_f64(high_f64, low_f64, close_f64, volume_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volume.mfi":
        out_f64 = _mfi_variants_f64(high_f64, low_f64, close_f64, volume_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "volume.vwap":
        out_f64 = _vwap_variants_f64(high_f64, low_f64, close_f64, volume_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    mults_f64 = _prepare_float_variants(
        name="mults",
        values=mults,
        expected_size=windows_i64.shape[0],
    )
    if np.any(mults_f64 <= 0.0):
        raise ValueError("mults must contain only positive values")

    out_f64 = _vwap_deviation_variants_f64(
        high_f64,
        low_f64,
        close_f64,
        volume_f64,
        windows_i64,
        mults_f64,
    )
    return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))


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
        Parameter values are integer-like and positive for volume formulas.
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
    volume: np.ndarray,
) -> None:
    """
    Validate equal length for OHLCV aligned series.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        volume: Volume series.
    Returns:
        None.
    Assumptions:
        Inputs are one-dimensional arrays.
    Raises:
        ValueError: If series lengths differ.
    Side Effects:
        None.
    """
    if (
        high.shape[0] != low.shape[0]
        or high.shape[0] != close.shape[0]
        or high.shape[0] != volume.shape[0]
    ):
        raise ValueError("high, low, close, volume lengths must match")


def _ensure_same_length_cv(*, close: np.ndarray, volume: np.ndarray) -> None:
    """
    Validate equal length for close/volume aligned series.

    Args:
        close: Close-price series.
        volume: Volume series.
    Returns:
        None.
    Assumptions:
        Inputs are one-dimensional arrays.
    Raises:
        ValueError: If series lengths differ.
    Side Effects:
        None.
    """
    if close.shape[0] != volume.shape[0]:
        raise ValueError("close and volume lengths must match")


def _normalize_volume_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize volume indicator identifier.

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized indicator id.
    Assumptions:
        Volume indicator aliases are not used in v1.
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
    "compute_volume_grid_f32",
    "is_supported_volume_indicator",
]
