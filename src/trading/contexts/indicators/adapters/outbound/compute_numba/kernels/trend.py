"""
Numba kernels for trend-family indicators.

Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
Related: docs/architecture/indicators/indicators_formula.yaml,
  trading.contexts.indicators.adapters.outbound.compute_numpy.trend,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.domain.definitions.trend
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import is_nan

_SUPPORTED_TREND_IDS = {
    "trend.adx",
    "trend.aroon",
    "trend.chandelier_exit",
    "trend.donchian",
    "trend.ichimoku",
    "trend.keltner",
    "trend.linreg_slope",
    "trend.psar",
    "trend.supertrend",
    "trend.vortex",
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
def _rolling_min_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-minimum series with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-minimum series.
    Assumptions:
        Any NaN inside the active window yields NaN output for that index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if is_nan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if is_nan(outgoing):
                nan_count -= 1

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
            continue

        start = time_index + 1 - window
        minimum = float(source[start])
        for index in range(start + 1, time_index + 1):
            candidate = float(source[index])
            if candidate < minimum:
                minimum = candidate
        out[time_index] = minimum

    return out


@nb.njit(cache=True)
def _rolling_max_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-maximum series with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling-maximum series.
    Assumptions:
        Any NaN inside the active window yields NaN output for that index.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if is_nan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if is_nan(outgoing):
                nan_count -= 1

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
            continue

        start = time_index + 1 - window
        maximum = float(source[start])
        for index in range(start + 1, time_index + 1):
            candidate = float(source[index])
            if candidate >= maximum:
                maximum = candidate
        out[time_index] = maximum

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
def _rolling_time_since_max_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling bars-since-maximum with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 vector with values in `[0, window - 1]` or NaN.
    Assumptions:
        Ties choose the most recent maximum (smallest bars-since value).
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if is_nan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if is_nan(outgoing):
                nan_count -= 1

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
            continue

        start = time_index + 1 - window
        max_value = float(source[start])
        max_index = start
        for index in range(start + 1, time_index + 1):
            candidate = float(source[index])
            if candidate >= max_value:
                max_value = candidate
                max_index = index
        out[time_index] = float(time_index - max_index)

    return out


@nb.njit(cache=True)
def _rolling_time_since_min_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling bars-since-minimum with warmup and NaN-window propagation.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 vector with values in `[0, window - 1]` or NaN.
    Assumptions:
        Ties choose the most recent minimum (smallest bars-since value).
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0

    for time_index in range(t_size):
        incoming = float(source[time_index])
        if is_nan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if is_nan(outgoing):
                nan_count -= 1

        if time_index + 1 < window or nan_count > 0:
            out[time_index] = np.nan
            continue

        start = time_index + 1 - window
        min_value = float(source[start])
        min_index = start
        for index in range(start + 1, time_index + 1):
            candidate = float(source[index])
            if candidate <= min_value:
                min_value = candidate
                min_index = index
        out[time_index] = float(time_index - min_index)

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
def _ema_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute EMA with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
        window: Positive integer EMA window.
    Returns:
        np.ndarray: Float64 EMA series.
    Assumptions:
        First valid source value after each NaN gap is used as EMA seed.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
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
def _rma_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute Wilder RMA with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
        window: Positive integer RMA window.
    Returns:
        np.ndarray: Float64 RMA series.
    Assumptions:
        First valid source value after each NaN gap is used as RMA seed.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
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


@nb.njit(cache=True)
def _true_range_series_f64(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute True Range with NaN-aware previous-close semantics.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
    Returns:
        np.ndarray: Float64 true-range series.
    Assumptions:
        When previous close is NaN, true range falls back to `high - low`.
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
def _adx_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
    smoothing: int,
) -> np.ndarray:
    """
    Compute ADX primary output (`adx`) for one `(window, smoothing)` variant.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: Wilder smoothing window for TR/DM chains.
        smoothing: Wilder smoothing window for DX chain.
    Returns:
        np.ndarray: Float64 ADX series.
    Assumptions:
        NaN holes reset stateful chains through NaN values in intermediate series.
    Raises:
        None.
    Side Effects:
        Allocates several intermediate vectors.
    """
    t_size = high.shape[0]
    plus_dm = np.empty(t_size, dtype=np.float64)
    minus_dm = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        if is_nan(high_value) or is_nan(low_value):
            plus_dm[time_index] = np.nan
            minus_dm[time_index] = np.nan
            continue

        if time_index == 0:
            plus_dm[time_index] = 0.0
            minus_dm[time_index] = 0.0
            continue

        previous_high = float(high[time_index - 1])
        previous_low = float(low[time_index - 1])
        if is_nan(previous_high) or is_nan(previous_low):
            plus_dm[time_index] = 0.0
            minus_dm[time_index] = 0.0
            continue

        up_move = high_value - previous_high
        down_move = previous_low - low_value

        if up_move > down_move and up_move > 0.0:
            plus_dm[time_index] = up_move
        else:
            plus_dm[time_index] = 0.0

        if down_move > up_move and down_move > 0.0:
            minus_dm[time_index] = down_move
        else:
            minus_dm[time_index] = 0.0

    tr = _true_range_series_f64(high, low, close)
    atr = _rma_series_f64(tr, window)
    plus_dm_smoothed = _rma_series_f64(plus_dm, window)
    minus_dm_smoothed = _rma_series_f64(minus_dm, window)

    plus_di = np.empty(t_size, dtype=np.float64)
    minus_di = np.empty(t_size, dtype=np.float64)
    dx = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        atr_value = float(atr[time_index])
        plus_value = float(plus_dm_smoothed[time_index])
        minus_value = float(minus_dm_smoothed[time_index])

        if (
            is_nan(atr_value)
            or is_nan(plus_value)
            or is_nan(minus_value)
            or atr_value == 0.0
        ):
            plus_di[time_index] = np.nan
            minus_di[time_index] = np.nan
            dx[time_index] = np.nan
            continue

        plus_di_value = 100.0 * (plus_value / atr_value)
        minus_di_value = 100.0 * (minus_value / atr_value)
        plus_di[time_index] = plus_di_value
        minus_di[time_index] = minus_di_value

        denominator = plus_di_value + minus_di_value
        if denominator == 0.0:
            dx[time_index] = np.nan
        else:
            dx[time_index] = 100.0 * (abs(plus_di_value - minus_di_value) / denominator)

    return _rma_series_f64(dx, smoothing)


@nb.njit(cache=True)
def _aroon_osc_series_f64(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Compute Aroon oscillator primary output (`aroon_osc`) for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        window: Rolling window.
    Returns:
        np.ndarray: Float64 Aroon oscillator series.
    Assumptions:
        Rolling helpers return NaN for warmup and any window containing NaN.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    since_hh = _rolling_time_since_max_series_f64(high, window)
    since_ll = _rolling_time_since_min_series_f64(low, window)
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_bars = float(since_hh[time_index])
        low_bars = float(since_ll[time_index])
        if is_nan(high_bars) or is_nan(low_bars):
            out[time_index] = np.nan
            continue

        aroon_up = ((float(window) - high_bars) / float(window)) * 100.0
        aroon_down = ((float(window) - low_bars) / float(window)) * 100.0
        out[time_index] = aroon_up - aroon_down

    return out


@nb.njit(cache=True)
def _chandelier_long_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
    mult: float,
) -> np.ndarray:
    """
    Compute Chandelier Exit primary output (`chandelier_long`) for one variant.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: Rolling/ATR window.
        mult: ATR multiplier.
    Returns:
        np.ndarray: Float64 Chandelier long-stop series.
    Assumptions:
        ATR uses Wilder RMA with reset-on-NaN semantics.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    tr = _true_range_series_f64(high, low, close)
    atr = _rma_series_f64(tr, window)
    hh = _rolling_max_series_f64(high, window)
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        hh_value = float(hh[time_index])
        atr_value = float(atr[time_index])
        if is_nan(hh_value) or is_nan(atr_value):
            out[time_index] = np.nan
        else:
            out[time_index] = hh_value - (mult * atr_value)

    return out


@nb.njit(cache=True)
def _donchian_mid_series_f64(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Compute Donchian primary output (`donchian_mid`) for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        window: Rolling window.
    Returns:
        np.ndarray: Float64 Donchian midpoint series.
    Assumptions:
        Midpoint is `(rolling_max(high) + rolling_min(low)) * 0.5`.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    upper = _rolling_max_series_f64(high, window)
    lower = _rolling_min_series_f64(low, window)
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        upper_value = float(upper[time_index])
        lower_value = float(lower[time_index])
        if is_nan(upper_value) or is_nan(lower_value):
            out[time_index] = np.nan
        else:
            out[time_index] = (upper_value + lower_value) * 0.5

    return out


@nb.njit(cache=True)
def _ichimoku_span_a_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    conversion_window: int,
    base_window: int,
    displacement: int,
) -> np.ndarray:
    """
    Compute Ichimoku primary output (`span_a`) for one parameter variant.

    Args:
        high: High-price series.
        low: Low-price series.
        conversion_window: Tenkan-sen window.
        base_window: Kijun-sen window.
        displacement: Forward displacement in bars.
    Returns:
        np.ndarray: Float64 shifted span-a series.
    Assumptions:
        Shift semantics use `periods=-displacement` as fixed in formulas spec.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    conv_h = _rolling_max_series_f64(high, conversion_window)
    conv_l = _rolling_min_series_f64(low, conversion_window)
    base_h = _rolling_max_series_f64(high, base_window)
    base_l = _rolling_min_series_f64(low, base_window)

    t_size = high.shape[0]
    span_a_raw = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        conv_h_value = float(conv_h[time_index])
        conv_l_value = float(conv_l[time_index])
        base_h_value = float(base_h[time_index])
        base_l_value = float(base_l[time_index])

        if (
            is_nan(conv_h_value)
            or is_nan(conv_l_value)
            or is_nan(base_h_value)
            or is_nan(base_l_value)
        ):
            span_a_raw[time_index] = np.nan
            continue

        conversion = (conv_h_value + conv_l_value) * 0.5
        base = (base_h_value + base_l_value) * 0.5
        span_a_raw[time_index] = (conversion + base) * 0.5

    return _shift_series_f64(span_a_raw, -displacement)


@nb.njit(cache=True)
def _keltner_middle_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute Keltner primary output (`middle`) for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: EMA window.
    Returns:
        np.ndarray: Float64 Keltner middle-line series.
    Assumptions:
        Middle line is EMA of typical price `(h + l + c) / 3`.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    t_size = high.shape[0]
    typical_price = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])
        if is_nan(high_value) or is_nan(low_value) or is_nan(close_value):
            typical_price[time_index] = np.nan
        else:
            typical_price[time_index] = (high_value + low_value + close_value) / 3.0

    return _ema_series_f64(typical_price, window)


@nb.njit(cache=True)
def _linreg_slope_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling linear-regression slope for `x=0..window-1`.

    Args:
        source: Float64 source series.
        window: Positive integer rolling window.
    Returns:
        np.ndarray: Float64 slope series.
    Assumptions:
        Any NaN inside window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        out[time_index] = np.nan

    n = float(window)
    sum_x = (n * (n - 1.0)) * 0.5
    sum_x2 = ((n - 1.0) * n * ((2.0 * n) - 1.0)) / 6.0
    denominator = (n * sum_x2) - (sum_x * sum_x)
    if denominator == 0.0:
        return out

    for time_index in range(window - 1, t_size):
        start = time_index + 1 - window
        sum_y = 0.0
        sum_xy = 0.0
        invalid = False

        for offset in range(window):
            value = float(source[start + offset])
            if is_nan(value):
                invalid = True
                break
            sum_y += value
            sum_xy += float(offset) * value

        if invalid:
            out[time_index] = np.nan
        else:
            out[time_index] = ((n * sum_xy) - (sum_x * sum_y)) / denominator

    return out


@nb.njit(cache=True)
def _psar_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    accel_start: float,
    accel_step: float,
    accel_max: float,
) -> np.ndarray:
    """
    Compute Parabolic SAR primary output (`psar`) with NaN-driven state resets.

    Args:
        high: High-price series.
        low: Low-price series.
        accel_start: Initial acceleration factor.
        accel_step: Step for acceleration increase.
        accel_max: Maximum acceleration factor.
    Returns:
        np.ndarray: Float64 PSAR series.
    Assumptions:
        NaN on inputs emits NaN output and fully resets trend direction/state.
    Raises:
        None.
    Side Effects:
        Allocates one output vector.
    """
    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    direction = 0
    sar = np.nan
    ep = np.nan
    af = accel_start
    prev_high = np.nan
    prev_low = np.nan

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])

        if is_nan(high_value) or is_nan(low_value):
            out[time_index] = np.nan
            direction = 0
            sar = np.nan
            ep = np.nan
            af = accel_start
            prev_high = np.nan
            prev_low = np.nan
            continue

        if direction == 0:
            direction = 1
            sar = low_value
            ep = high_value
            af = accel_start
            out[time_index] = sar
            prev_high = high_value
            prev_low = low_value
            continue

        sar = sar + (af * (ep - sar))

        if direction > 0:
            if not is_nan(prev_low) and sar > prev_low:
                sar = prev_low

            if low_value < sar:
                direction = -1
                sar = ep
                ep = low_value
                af = accel_start
            else:
                if high_value > ep:
                    ep = high_value
                    af = min(accel_max, af + accel_step)
        else:
            if not is_nan(prev_high) and sar < prev_high:
                sar = prev_high

            if high_value > sar:
                direction = 1
                sar = ep
                ep = high_value
                af = accel_start
            else:
                if low_value < ep:
                    ep = low_value
                    af = min(accel_max, af + accel_step)

        out[time_index] = sar
        prev_high = high_value
        prev_low = low_value

    return out


@nb.njit(cache=True)
def _supertrend_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
    mult: float,
) -> np.ndarray:
    """
    Compute SuperTrend primary output (`supertrend`) with NaN-driven state resets.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: ATR window.
        mult: ATR multiplier.
    Returns:
        np.ndarray: Float64 SuperTrend line series.
    Assumptions:
        NaN on required inputs emits NaN output and fully resets trend direction/state.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors and one output vector.
    """
    tr = _true_range_series_f64(high, low, close)
    atr = _rma_series_f64(tr, window)

    t_size = high.shape[0]
    out = np.empty(t_size, dtype=np.float64)

    direction = 0
    final_upper = np.nan
    final_lower = np.nan
    prev_close = np.nan

    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        close_value = float(close[time_index])
        atr_value = float(atr[time_index])

        if (
            is_nan(high_value)
            or is_nan(low_value)
            or is_nan(close_value)
            or is_nan(atr_value)
        ):
            out[time_index] = np.nan
            direction = 0
            final_upper = np.nan
            final_lower = np.nan
            prev_close = np.nan
            continue

        hl2 = (high_value + low_value) * 0.5
        basic_upper = hl2 + (mult * atr_value)
        basic_lower = hl2 - (mult * atr_value)

        if direction == 0:
            direction = 1 if close_value >= hl2 else -1
            final_upper = basic_upper
            final_lower = basic_lower
            out[time_index] = final_lower if direction > 0 else final_upper
            prev_close = close_value
            continue

        if basic_upper < final_upper or prev_close > final_upper:
            final_upper = basic_upper
        if basic_lower > final_lower or prev_close < final_lower:
            final_lower = basic_lower

        if direction > 0:
            if close_value < final_lower:
                direction = -1
                out[time_index] = final_upper
            else:
                out[time_index] = final_lower
        else:
            if close_value > final_upper:
                direction = 1
                out[time_index] = final_lower
            else:
                out[time_index] = final_upper

        prev_close = close_value

    return out


@nb.njit(cache=True)
def _vortex_plus_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute Vortex primary output (`vi_plus`) for one window.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: Rolling sum window.
    Returns:
        np.ndarray: Float64 `vi_plus` series.
    Assumptions:
        Rolling sums follow warmup and NaN-window propagation policy.
    Raises:
        None.
    Side Effects:
        Allocates intermediate vectors.
    """
    t_size = high.shape[0]
    prev_low = _shift_series_f64(low, 1)
    vm_plus = np.empty(t_size, dtype=np.float64)

    for time_index in range(t_size):
        high_value = float(high[time_index])
        prev_low_value = float(prev_low[time_index])
        if is_nan(high_value) or is_nan(prev_low_value):
            vm_plus[time_index] = np.nan
        else:
            vm_plus[time_index] = abs(high_value - prev_low_value)

    tr = _true_range_series_f64(high, low, close)
    sum_vm_plus = _rolling_sum_series_f64(vm_plus, window)
    sum_tr = _rolling_sum_series_f64(tr, window)

    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        numerator = float(sum_vm_plus[time_index])
        denominator = float(sum_tr[time_index])
        if is_nan(numerator) or is_nan(denominator) or denominator == 0.0:
            out[time_index] = np.nan
        else:
            out[time_index] = numerator / denominator

    return out


@nb.njit(parallel=True, cache=True)
def _adx_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
    smoothings: np.ndarray,
) -> np.ndarray:
    """
    Compute ADX primary output matrix for per-variant `(window, smoothing)` pairs.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        windows: Per-variant window vector.
        smoothings: Per-variant smoothing vector.
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
        out[variant_index, :] = _adx_series_f64(
            high,
            low,
            close,
            int(windows[variant_index]),
            int(smoothings[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _aroon_variants_f64(high: np.ndarray, low: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute Aroon oscillator matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
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
        out[variant_index, :] = _aroon_osc_series_f64(
            high,
            low,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _chandelier_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
    mults: np.ndarray,
) -> np.ndarray:
    """
    Compute Chandelier long-stop matrix for per-variant `(window, mult)` pairs.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
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
        out[variant_index, :] = _chandelier_long_series_f64(
            high,
            low,
            close,
            int(windows[variant_index]),
            float(mults[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _donchian_variants_f64(high: np.ndarray, low: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute Donchian midpoint matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
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
        out[variant_index, :] = _donchian_mid_series_f64(
            high,
            low,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _ichimoku_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    conversion_windows: np.ndarray,
    base_windows: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """
    Compute Ichimoku span-a matrix for per-variant parameter vectors.

    Args:
        high: High-price series.
        low: Low-price series.
        conversion_windows: Per-variant conversion windows.
        base_windows: Per-variant base windows.
        displacements: Per-variant displacements.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameter vectors are aligned by variant order in Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = conversion_windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _ichimoku_span_a_series_f64(
            high,
            low,
            int(conversion_windows[variant_index]),
            int(base_windows[variant_index]),
            int(displacements[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _keltner_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
    mults: np.ndarray,
) -> np.ndarray:
    """
    Compute Keltner middle-line matrix for per-variant `(window, mult)` pairs.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        windows: Per-variant window vector.
        mults: Per-variant multiplier vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        `mults` are validated for alignment even though primary output uses only `window`.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        _ = float(mults[variant_index])
        out[variant_index, :] = _keltner_middle_series_f64(
            high,
            low,
            close,
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _linreg_variants_f64(source_variants: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute linear-regression slope matrix for source-variant rows and windows.

    Args:
        source_variants: Float64 source matrix `(V, T)`.
        windows: Per-variant window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Window vector is aligned with source variant order.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _linreg_slope_series_f64(
            source_variants[variant_index, :],
            int(windows[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _psar_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    accel_starts: np.ndarray,
    accel_steps: np.ndarray,
    accel_maxes: np.ndarray,
) -> np.ndarray:
    """
    Compute PSAR matrix for per-variant acceleration parameters.

    Args:
        high: High-price series.
        low: Low-price series.
        accel_starts: Per-variant start accelerations.
        accel_steps: Per-variant acceleration steps.
        accel_maxes: Per-variant maximum accelerations.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameter vectors are aligned by variant order in Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix.
    """
    variants = accel_starts.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        out[variant_index, :] = _psar_series_f64(
            high,
            low,
            float(accel_starts[variant_index]),
            float(accel_steps[variant_index]),
            float(accel_maxes[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _supertrend_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
    mults: np.ndarray,
) -> np.ndarray:
    """
    Compute SuperTrend matrix for per-variant `(window, mult)` pairs.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
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
        out[variant_index, :] = _supertrend_series_f64(
            high,
            low,
            close,
            int(windows[variant_index]),
            float(mults[variant_index]),
        )

    return out


@nb.njit(parallel=True, cache=True)
def _vortex_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    """
    Compute Vortex `vi_plus` matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
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
        out[variant_index, :] = _vortex_plus_series_f64(
            high,
            low,
            close,
            int(windows[variant_index]),
        )

    return out


def is_supported_trend_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by trend kernels.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by trend kernels.
    Assumptions:
        Identifier normalization is delegated to `_normalize_trend_indicator_id`.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized_id = _normalize_trend_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_TREND_IDS


def compute_trend_grid_f32(
    *,
    indicator_id: str,
    source_variants: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    smoothings: np.ndarray | None = None,
    mults: np.ndarray | None = None,
    conversion_windows: np.ndarray | None = None,
    base_windows: np.ndarray | None = None,
    span_b_windows: np.ndarray | None = None,
    displacements: np.ndarray | None = None,
    accel_starts: np.ndarray | None = None,
    accel_steps: np.ndarray | None = None,
    accel_maxes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute trend indicator matrix `(V, T)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md
    Related:
      docs/architecture/indicators/indicators_formula.yaml,
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/domain/definitions/trend.py

    Args:
        indicator_id: Trend indicator identifier.
        source_variants: Optional `(V, T)` source matrix for `trend.linreg_slope`.
        high: Optional high-price series.
        low: Optional low-price series.
        close: Optional close-price series.
        windows: Optional per-variant window values.
        smoothings: Optional per-variant smoothing values.
        mults: Optional per-variant multiplier values.
        conversion_windows: Optional per-variant Ichimoku conversion windows.
        base_windows: Optional per-variant Ichimoku base windows.
        span_b_windows: Optional per-variant Ichimoku span-b windows (validated for v1 parity).
        displacements: Optional per-variant Ichimoku displacement values.
        accel_starts: Optional per-variant PSAR acceleration starts.
        accel_steps: Optional per-variant PSAR acceleration steps.
        accel_maxes: Optional per-variant PSAR acceleration maxima.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)` in deterministic variant order.
    Assumptions:
        Wrapper receives already-materialized per-variant vectors from compute engine.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_trend_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_TREND_IDS:
        raise ValueError(f"unsupported trend indicator_id: {indicator_id!r}")

    if normalized_id == "trend.linreg_slope":
        source_f64 = _prepare_source_variants(values=source_variants)
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=source_f64.shape[0],
        )
        out_f64 = _linreg_variants_f64(source_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    high_f64 = _prepare_series(name="high", values=high)
    low_f64 = _prepare_series(name="low", values=low)

    if normalized_id == "trend.psar":
        _ensure_same_length_hl(high=high_f64, low=low_f64)
        accel_starts_f64 = _prepare_float_variants(name="accel_starts", values=accel_starts)
        accel_steps_f64 = _prepare_float_variants(
            name="accel_steps",
            values=accel_steps,
            expected_size=accel_starts_f64.shape[0],
        )
        accel_maxes_f64 = _prepare_float_variants(
            name="accel_maxes",
            values=accel_maxes,
            expected_size=accel_starts_f64.shape[0],
        )

        if np.any(accel_starts_f64 <= 0.0):
            raise ValueError("accel_starts must contain only positive values")
        if np.any(accel_steps_f64 <= 0.0):
            raise ValueError("accel_steps must contain only positive values")
        if np.any(accel_maxes_f64 <= 0.0):
            raise ValueError("accel_maxes must contain only positive values")
        if np.any(accel_maxes_f64 < accel_starts_f64):
            raise ValueError("accel_maxes must be >= accel_starts per variant")

        out_f64 = _psar_variants_f64(
            high_f64,
            low_f64,
            accel_starts_f64,
            accel_steps_f64,
            accel_maxes_f64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    close_f64 = _prepare_series(name="close", values=close)
    _ensure_same_length(high=high_f64, low=low_f64, close=close_f64)

    if normalized_id == "trend.adx":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        smoothings_i64 = _prepare_int_variants(
            name="smoothings",
            values=smoothings,
            expected_size=windows_i64.shape[0],
        )
        out_f64 = _adx_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
            smoothings_i64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.aroon":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _aroon_variants_f64(high_f64, low_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.chandelier_exit":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        mults_f64 = _prepare_float_variants(
            name="mults",
            values=mults,
            expected_size=windows_i64.shape[0],
        )
        if np.any(mults_f64 <= 0.0):
            raise ValueError("mults must contain only positive values")
        out_f64 = _chandelier_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
            mults_f64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.donchian":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _donchian_variants_f64(high_f64, low_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.ichimoku":
        conversion_i64 = _prepare_int_variants(
            name="conversion_windows",
            values=conversion_windows,
        )
        base_i64 = _prepare_int_variants(
            name="base_windows",
            values=base_windows,
            expected_size=conversion_i64.shape[0],
        )
        span_b_i64 = _prepare_int_variants(
            name="span_b_windows",
            values=span_b_windows,
            expected_size=conversion_i64.shape[0],
        )
        _ = span_b_i64
        displacements_i64 = _prepare_int_variants(
            name="displacements",
            values=displacements,
            expected_size=conversion_i64.shape[0],
        )
        out_f64 = _ichimoku_variants_f64(
            high_f64,
            low_f64,
            conversion_i64,
            base_i64,
            displacements_i64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.keltner":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        mults_f64 = _prepare_float_variants(
            name="mults",
            values=mults,
            expected_size=windows_i64.shape[0],
        )
        if np.any(mults_f64 <= 0.0):
            raise ValueError("mults must contain only positive values")
        out_f64 = _keltner_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
            mults_f64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.supertrend":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        mults_f64 = _prepare_float_variants(
            name="mults",
            values=mults,
            expected_size=windows_i64.shape[0],
        )
        if np.any(mults_f64 <= 0.0):
            raise ValueError("mults must contain only positive values")
        out_f64 = _supertrend_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
            mults_f64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "trend.vortex":
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _vortex_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    raise ValueError(f"unsupported trend indicator_id: {indicator_id!r}")


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
    Normalize source variants matrix for source-parameterized trend indicators.

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
        Parameter values are integer-like and positive for trend formulas.
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


def _ensure_same_length(*, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
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


def _ensure_same_length_hl(*, high: np.ndarray, low: np.ndarray) -> None:
    """
    Validate equal length for HL-aligned series.

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


def _normalize_trend_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize trend indicator identifier.

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized indicator id.
    Assumptions:
        Trend indicator aliases are not used in v1.
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
    "compute_trend_grid_f32",
    "is_supported_trend_indicator",
]
