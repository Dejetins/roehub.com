"""
Numba kernels for momentum-family indicators.

Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numpy.momentum,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine,
  trading.contexts.indicators.domain.definitions.momentum
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ._common import is_nan

_SUPPORTED_MOMENTUM_IDS = {
    "momentum.rsi",
    "momentum.roc",
    "momentum.cci",
    "momentum.williams_r",
    "momentum.trix",
    "momentum.fisher",
    "momentum.stoch",
    "momentum.stoch_rsi",
    "momentum.ppo",
    "momentum.macd",
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
def _rolling_min_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one rolling-minimum series with NaN-window propagation policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling minimum series.
    Assumptions:
        Any NaN inside active window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
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
    Compute one rolling-maximum series with NaN-window propagation policy.

    Args:
        source: Float64 source series.
        window: Positive integer window.
    Returns:
        np.ndarray: Float64 rolling maximum series.
    Assumptions:
        Any NaN inside active window yields NaN output for that position.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
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
            if candidate > maximum:
                maximum = candidate
        out[time_index] = maximum
    return out


@nb.njit(cache=True)
def _ema_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one EMA series with reset-on-NaN state policy.

    Args:
        source: Float64 source series.
        window: Positive integer smoothing window (`alpha = 2 / (window + 1)`).
    Returns:
        np.ndarray: Float64 EMA series.
    Assumptions:
        First valid source value is used as seed after each reset.
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
def _rsi_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one RSI series using RMA averages with reset-on-NaN semantics.

    Args:
        source: Float64 source series.
        window: Positive integer RSI window (`alpha = 1 / window` for RMA parts).
    Returns:
        np.ndarray: Float64 RSI series.
    Assumptions:
        Delta stage emits NaN at first valid point and after each NaN-reset boundary.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    alpha = 1.0 / float(window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    previous_source = np.nan
    avg_gain = np.nan
    avg_loss = np.nan

    for time_index in range(t_size):
        current = float(source[time_index])
        if is_nan(current):
            previous_source = np.nan
            avg_gain = np.nan
            avg_loss = np.nan
            out[time_index] = np.nan
            continue

        if is_nan(previous_source):
            previous_source = current
            out[time_index] = np.nan
            continue

        delta = current - previous_source
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        if is_nan(avg_gain):
            avg_gain = gain
        else:
            avg_gain = (alpha * gain) + ((1.0 - alpha) * avg_gain)

        if is_nan(avg_loss):
            avg_loss = loss
        else:
            avg_loss = (alpha * loss) + ((1.0 - alpha) * avg_loss)

        if avg_loss == 0.0:
            if avg_gain == 0.0:
                out[time_index] = np.nan
            else:
                out[time_index] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[time_index] = 100.0 - (100.0 / (1.0 + rs))

        previous_source = current
    return out


@nb.njit(cache=True)
def _roc_series_f64(source: np.ndarray, window: int) -> np.ndarray:
    """
    Compute one ROC series with fixed warmup and NaN/zero-denominator policy.

    Args:
        source: Float64 source series.
        window: Positive integer lookback lag.
    Returns:
        np.ndarray: Float64 ROC series.
    Assumptions:
        Warmup for ROC is `t < window`.
    Raises:
        None.
    Side Effects:
        Allocates one output array.
    """
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        if time_index < window:
            out[time_index] = np.nan
            continue
        current = float(source[time_index])
        previous = float(source[time_index - window])
        if is_nan(current) or is_nan(previous) or previous == 0.0:
            out[time_index] = np.nan
        else:
            out[time_index] = 100.0 * ((current / previous) - 1.0)
    return out


@nb.njit(cache=True)
def _stoch_k_series_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_window: int,
    smoothing: int,
    d_window: int,
) -> np.ndarray:
    """
    Compute stochastic `k` output with rolling pipeline semantics from EPIC-07.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        k_window: Lookback window for `hh/ll` envelope.
        smoothing: Smoothing window for `k_raw -> k`.
        d_window: D-line window (`k -> d`) computed for completeness.
    Returns:
        np.ndarray: Float64 stochastic `k` series.
    Assumptions:
        Rolling stages propagate NaN when any value inside the active window is NaN.
    Raises:
        None.
    Side Effects:
        Allocates intermediate arrays.
    """
    t_size = high.shape[0]
    hh = _rolling_max_series_f64(high, k_window)
    ll = _rolling_min_series_f64(low, k_window)
    k_raw = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        close_value = float(close[time_index])
        hh_value = float(hh[time_index])
        ll_value = float(ll[time_index])
        denominator = hh_value - ll_value
        if is_nan(close_value) or is_nan(hh_value) or is_nan(ll_value) or denominator == 0.0:
            k_raw[time_index] = np.nan
        else:
            k_raw[time_index] = 100.0 * ((close_value - ll_value) / denominator)
    k = _rolling_mean_series_f64(k_raw, smoothing)
    _ = _rolling_mean_series_f64(k, d_window)
    return k


@nb.njit(parallel=True, cache=True)
def _rsi_or_roc_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    compute_rsi: bool,
) -> np.ndarray:
    """
    Compute RSI or ROC matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant windows vector.
        compute_rsi: Whether to compute RSI (`True`) or ROC (`False`).
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Windows are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        if compute_rsi:
            out[variant_index, :] = _rsi_series_f64(source_variants[variant_index, :], window)
        else:
            out[variant_index, :] = _roc_series_f64(source_variants[variant_index, :], window)
    return out


@nb.njit(parallel=True, cache=True)
def _williams_or_cci_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
    compute_williams: bool,
) -> np.ndarray:
    """
    Compute Williams %R or CCI matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        windows: Per-variant windows vector.
        compute_williams: Whether to compute Williams %R (`True`) or CCI (`False`).
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Windows are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        if compute_williams:
            hh = _rolling_max_series_f64(high, window)
            ll = _rolling_min_series_f64(low, window)
            series = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                hh_value = float(hh[time_index])
                ll_value = float(ll[time_index])
                close_value = float(close[time_index])
                denominator = hh_value - ll_value
                if (
                    is_nan(hh_value)
                    or is_nan(ll_value)
                    or is_nan(close_value)
                    or denominator == 0.0
                ):
                    series[time_index] = np.nan
                else:
                    series[time_index] = -100.0 * ((hh_value - close_value) / denominator)
            out[variant_index, :] = series
            continue

        tp = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            high_value = float(high[time_index])
            low_value = float(low[time_index])
            close_value = float(close[time_index])
            if is_nan(high_value) or is_nan(low_value) or is_nan(close_value):
                tp[time_index] = np.nan
            else:
                tp[time_index] = (high_value + low_value + close_value) / 3.0

        cci = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            if time_index + 1 < window:
                cci[time_index] = np.nan
                continue

            start = time_index + 1 - window
            sum_tp = 0.0
            invalid = False
            for index in range(start, time_index + 1):
                value = float(tp[index])
                if is_nan(value):
                    invalid = True
                    break
                sum_tp += value
            if invalid:
                cci[time_index] = np.nan
                continue

            mean_tp = sum_tp / float(window)
            mean_deviation = 0.0
            for index in range(start, time_index + 1):
                mean_deviation += abs(float(tp[index]) - mean_tp)
            mean_deviation = mean_deviation / float(window)
            denominator = 0.015 * mean_deviation
            if denominator == 0.0:
                cci[time_index] = np.nan
            else:
                cci[time_index] = (float(tp[time_index]) - mean_tp) / denominator

        out[variant_index, :] = cci

    return out


@nb.njit(parallel=True, cache=True)
def _fisher_variants_f64(high: np.ndarray, low: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Compute Fisher transform matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        windows: Per-variant windows vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Windows are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)

    hl2 = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        high_value = float(high[time_index])
        low_value = float(low[time_index])
        if is_nan(high_value) or is_nan(low_value):
            hl2[time_index] = np.nan
        else:
            hl2[time_index] = (high_value + low_value) * 0.5

    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        hh = _rolling_max_series_f64(high, window)
        ll = _rolling_min_series_f64(low, window)
        fisher = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            price = float(hl2[time_index])
            hh_value = float(hh[time_index])
            ll_value = float(ll[time_index])
            denominator = hh_value - ll_value
            if is_nan(price) or is_nan(hh_value) or is_nan(ll_value) or denominator == 0.0:
                fisher[time_index] = np.nan
                continue
            normalized = (2.0 * ((price - ll_value) / denominator)) - 1.0
            if normalized > 0.999:
                normalized = 0.999
            elif normalized < -0.999:
                normalized = -0.999
            fisher[time_index] = 0.5 * math.log((1.0 + normalized) / (1.0 - normalized))
        out[variant_index, :] = fisher
    return out


@nb.njit(parallel=True, cache=True)
def _stoch_variants_f64(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_windows: np.ndarray,
    smoothings: np.ndarray,
    d_windows: np.ndarray,
) -> np.ndarray:
    """
    Compute stochastic `k` matrix for per-variant windows.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        k_windows: Per-variant K-window vector.
        smoothings: Per-variant smoothing-window vector.
        d_windows: Per-variant D-window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = k_windows.shape[0]
    t_size = high.shape[0]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        out[variant_index, :] = _stoch_k_series_f64(
            high,
            low,
            close,
            int(k_windows[variant_index]),
            int(smoothings[variant_index]),
            int(d_windows[variant_index]),
        )
    return out


@nb.njit(parallel=True, cache=True)
def _stoch_rsi_variants_f64(
    source_variants: np.ndarray,
    rsi_windows: np.ndarray,
    k_windows: np.ndarray,
    smoothings: np.ndarray,
    d_windows: np.ndarray,
) -> np.ndarray:
    """
    Compute stochastic-RSI `k` matrix for per-variant parameters.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        rsi_windows: Per-variant RSI-window vector.
        k_windows: Per-variant K-window vector.
        smoothings: Per-variant smoothing-window vector.
        d_windows: Per-variant D-window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        rsi_series = _rsi_series_f64(
            source_variants[variant_index, :],
            int(rsi_windows[variant_index]),
        )
        hh = _rolling_max_series_f64(rsi_series, int(k_windows[variant_index]))
        ll = _rolling_min_series_f64(rsi_series, int(k_windows[variant_index]))
        k_raw = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            rsi_value = float(rsi_series[time_index])
            hh_value = float(hh[time_index])
            ll_value = float(ll[time_index])
            denominator = hh_value - ll_value
            if is_nan(rsi_value) or is_nan(hh_value) or is_nan(ll_value) or denominator == 0.0:
                k_raw[time_index] = np.nan
            else:
                k_raw[time_index] = 100.0 * ((rsi_value - ll_value) / denominator)
        k = _rolling_mean_series_f64(k_raw, int(smoothings[variant_index]))
        _ = _rolling_mean_series_f64(k, int(d_windows[variant_index]))
        out[variant_index, :] = k
    return out


@nb.njit(parallel=True, cache=True)
def _trix_variants_f64(
    source_variants: np.ndarray,
    windows: np.ndarray,
    signal_windows: np.ndarray,
) -> np.ndarray:
    """
    Compute TRIX main-line matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        windows: Per-variant TRIX-window vector.
        signal_windows: Per-variant signal-window vector.
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        window = int(windows[variant_index])
        signal_window = int(signal_windows[variant_index])
        source = source_variants[variant_index, :]
        ema1 = _ema_series_f64(source, window)
        ema2 = _ema_series_f64(ema1, window)
        ema3 = _ema_series_f64(ema2, window)
        trix = np.empty(t_size, dtype=np.float64)
        trix[0] = np.nan
        for time_index in range(1, t_size):
            current = float(ema3[time_index])
            previous = float(ema3[time_index - 1])
            if is_nan(current) or is_nan(previous) or previous == 0.0:
                trix[time_index] = np.nan
            else:
                trix[time_index] = 100.0 * ((current / previous) - 1.0)
        _ = _ema_series_f64(trix, signal_window)
        out[variant_index, :] = trix
    return out


@nb.njit(parallel=True, cache=True)
def _macd_or_ppo_variants_f64(
    source_variants: np.ndarray,
    fast_windows: np.ndarray,
    slow_windows: np.ndarray,
    signal_windows: np.ndarray,
    compute_ppo: bool,
) -> np.ndarray:
    """
    Compute MACD or PPO main-line matrix for per-variant windows.

    Args:
        source_variants: Variant-major source matrix `(V, T)`.
        fast_windows: Per-variant fast-window vector.
        slow_windows: Per-variant slow-window vector.
        signal_windows: Per-variant signal-window vector.
        compute_ppo: Whether to compute PPO (`True`) or MACD (`False`).
    Returns:
        np.ndarray: Float64 matrix `(V, T)`.
    Assumptions:
        Parameters are positive integers validated by Python wrapper.
    Raises:
        None.
    Side Effects:
        Allocates one output matrix and per-variant temporaries.
    """
    variants = source_variants.shape[0]
    t_size = source_variants.shape[1]
    out = np.empty((variants, t_size), dtype=np.float64)
    for variant_index in nb.prange(variants):
        source = source_variants[variant_index, :]
        fast = _ema_series_f64(source, int(fast_windows[variant_index]))
        slow = _ema_series_f64(source, int(slow_windows[variant_index]))
        main = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            fast_value = float(fast[time_index])
            slow_value = float(slow[time_index])
            if is_nan(fast_value) or is_nan(slow_value):
                main[time_index] = np.nan
                continue
            if compute_ppo:
                if slow_value == 0.0:
                    main[time_index] = np.nan
                else:
                    main[time_index] = 100.0 * ((fast_value - slow_value) / slow_value)
            else:
                main[time_index] = fast_value - slow_value
        _ = _ema_series_f64(main, int(signal_windows[variant_index]))
        out[variant_index, :] = main
    return out


def is_supported_momentum_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by momentum kernels.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by momentum kernels.
    Assumptions:
        Identifier normalization is delegated to `_normalize_momentum_indicator_id`.
    Raises:
        ValueError: If identifier is blank.
    Side Effects:
        None.
    """
    normalized_id = _normalize_momentum_indicator_id(indicator_id=indicator_id)
    return normalized_id in _SUPPORTED_MOMENTUM_IDS


def compute_momentum_grid_f32(
    *,
    indicator_id: str,
    source_variants: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    signal_windows: np.ndarray | None = None,
    fast_windows: np.ndarray | None = None,
    slow_windows: np.ndarray | None = None,
    rsi_windows: np.ndarray | None = None,
    k_windows: np.ndarray | None = None,
    smoothings: np.ndarray | None = None,
    d_windows: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute momentum indicator matrix `(V, T)` as float32 contiguous array.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py,
      src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py,
      src/trading/contexts/indicators/domain/definitions/momentum.py

    Args:
        indicator_id: Momentum indicator identifier.
        source_variants: Optional `(V, T)` matrix for source-based indicators.
        high: Optional `high` series for OHLC/HL indicators.
        low: Optional `low` series for OHLC/HL indicators.
        close: Optional `close` series for OHLC indicators.
        windows: Optional per-variant window values.
        signal_windows: Optional per-variant signal window values.
        fast_windows: Optional per-variant fast window values.
        slow_windows: Optional per-variant slow window values.
        rsi_windows: Optional per-variant RSI window values.
        k_windows: Optional per-variant stochastic K window values.
        smoothings: Optional per-variant smoothing window values.
        d_windows: Optional per-variant D window values.
    Returns:
        np.ndarray: Float32 C-contiguous matrix `(V, T)`.
    Assumptions:
        Parameter vectors, when required, are materialized in deterministic variant order.
    Raises:
        ValueError: If indicator id is unsupported or required inputs are missing/invalid.
    Side Effects:
        Allocates intermediate float64 arrays before casting to float32.
    """
    normalized_id = _normalize_momentum_indicator_id(indicator_id=indicator_id)
    if normalized_id not in _SUPPORTED_MOMENTUM_IDS:
        raise ValueError(f"unsupported momentum indicator_id: {indicator_id!r}")

    if normalized_id in {"momentum.rsi", "momentum.roc"}:
        source_f64 = _prepare_source_variants(values=source_variants)
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=source_f64.shape[0],
        )
        out_f64 = _rsi_or_roc_variants_f64(
            source_f64,
            windows_i64,
            normalized_id == "momentum.rsi",
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in {"momentum.cci", "momentum.williams_r"}:
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        _ensure_same_length(high=high_f64, low=low_f64, close=close_f64)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _williams_or_cci_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            windows_i64,
            normalized_id == "momentum.williams_r",
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.fisher":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        _ensure_same_length_hl(high=high_f64, low=low_f64)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        out_f64 = _fisher_variants_f64(high_f64, low_f64, windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.stoch":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        _ensure_same_length(high=high_f64, low=low_f64, close=close_f64)

        k_windows_i64 = _prepare_int_variants(name="k_windows", values=k_windows)
        smoothings_i64 = _prepare_int_variants(
            name="smoothings",
            values=smoothings,
            expected_size=k_windows_i64.shape[0],
        )
        d_windows_i64 = _prepare_int_variants(
            name="d_windows",
            values=d_windows,
            expected_size=k_windows_i64.shape[0],
        )

        out_f64 = _stoch_variants_f64(
            high_f64,
            low_f64,
            close_f64,
            k_windows_i64,
            smoothings_i64,
            d_windows_i64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.stoch_rsi":
        source_f64 = _prepare_source_variants(values=source_variants)
        variants = source_f64.shape[0]
        rsi_windows_i64 = _prepare_int_variants(
            name="rsi_windows",
            values=rsi_windows,
            expected_size=variants,
        )
        k_windows_i64 = _prepare_int_variants(
            name="k_windows",
            values=k_windows,
            expected_size=variants,
        )
        smoothings_i64 = _prepare_int_variants(
            name="smoothings",
            values=smoothings,
            expected_size=variants,
        )
        d_windows_i64 = _prepare_int_variants(
            name="d_windows",
            values=d_windows,
            expected_size=variants,
        )
        out_f64 = _stoch_rsi_variants_f64(
            source_f64,
            rsi_windows_i64,
            k_windows_i64,
            smoothings_i64,
            d_windows_i64,
        )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.trix":
        source_f64 = _prepare_source_variants(values=source_variants)
        variants = source_f64.shape[0]
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        signal_windows_i64 = _prepare_int_variants(
            name="signal_windows",
            values=signal_windows,
            expected_size=variants,
        )
        out_f64 = _trix_variants_f64(source_f64, windows_i64, signal_windows_i64)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    source_f64 = _prepare_source_variants(values=source_variants)
    variants = source_f64.shape[0]
    fast_windows_i64 = _prepare_int_variants(
        name="fast_windows",
        values=fast_windows,
        expected_size=variants,
    )
    slow_windows_i64 = _prepare_int_variants(
        name="slow_windows",
        values=slow_windows,
        expected_size=variants,
    )
    signal_windows_i64 = _prepare_int_variants(
        name="signal_windows",
        values=signal_windows,
        expected_size=variants,
    )
    out_f64 = _macd_or_ppo_variants_f64(
        source_f64,
        fast_windows_i64,
        slow_windows_i64,
        signal_windows_i64,
        normalized_id == "momentum.ppo",
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
        Parameter values are integer-like and positive for current momentum formulas.
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
    Validate equal length for HL aligned series.

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


def _normalize_momentum_indicator_id(*, indicator_id: str) -> str:
    """
    Normalize momentum indicator identifier.

    Args:
        indicator_id: Raw indicator identifier.
    Returns:
        str: Lowercase normalized indicator id.
    Assumptions:
        Momentum indicator aliases are not used in v1.
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
    "compute_momentum_grid_f32",
    "is_supported_momentum_indicator",
]
