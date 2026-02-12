"""
Numpy oracle implementation for momentum-family indicators.

Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels.momentum,
  trading.contexts.indicators.domain.definitions.momentum,
  trading.contexts.indicators.adapters.outbound.compute_numba.engine
"""

from __future__ import annotations

import math

import numpy as np

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


def is_supported_momentum_indicator(*, indicator_id: str) -> bool:
    """
    Return whether indicator id is supported by momentum oracle implementation.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py,
      src/trading/contexts/indicators/domain/definitions/momentum.py

    Args:
        indicator_id: Indicator identifier.
    Returns:
        bool: True when id is supported by momentum oracle.
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
    Compute momentum indicator matrix `(V, T)` using pure NumPy/Python loops.

    Docs: docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py,
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
        variants = source_f64.shape[0]
        t_size = source_f64.shape[1]
        windows_i64 = _prepare_int_variants(
            name="windows",
            values=windows,
            expected_size=variants,
        )
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            series = source_f64[variant_index, :]
            window = int(windows_i64[variant_index])
            if normalized_id == "momentum.rsi":
                out_f64[variant_index, :] = _rsi_series_f64(source=series, window=window)
                continue
            out_f64[variant_index, :] = _roc_series_f64(source=series, window=window)
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in {"momentum.cci", "momentum.williams_r"}:
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)

        t_size = high_f64.shape[0]
        variants = windows_i64.shape[0]
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        if high_f64.shape[0] != low_f64.shape[0] or high_f64.shape[0] != close_f64.shape[0]:
            raise ValueError("high, low, close lengths must match")

        for variant_index, window_raw in enumerate(windows_i64):
            window = int(window_raw)
            if normalized_id == "momentum.williams_r":
                hh = _rolling_max_series_f64(source=high_f64, window=window)
                ll = _rolling_min_series_f64(source=low_f64, window=window)
                out = np.empty(t_size, dtype=np.float64)
                for time_index in range(t_size):
                    hh_value = float(hh[time_index])
                    ll_value = float(ll[time_index])
                    close_value = float(close_f64[time_index])
                    denominator = hh_value - ll_value
                    if (
                        math.isnan(hh_value)
                        or math.isnan(ll_value)
                        or math.isnan(close_value)
                        or denominator == 0.0
                    ):
                        out[time_index] = np.nan
                    else:
                        out[time_index] = -100.0 * ((hh_value - close_value) / denominator)
                out_f64[variant_index, :] = out
                continue

            tp = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                high_value = float(high_f64[time_index])
                low_value = float(low_f64[time_index])
                close_value = float(close_f64[time_index])
                if math.isnan(high_value) or math.isnan(low_value) or math.isnan(close_value):
                    tp[time_index] = np.nan
                else:
                    tp[time_index] = (high_value + low_value + close_value) / 3.0

            out = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                if time_index + 1 < window:
                    out[time_index] = np.nan
                    continue

                start = time_index + 1 - window
                sum_tp = 0.0
                invalid = False
                for index in range(start, time_index + 1):
                    value = float(tp[index])
                    if math.isnan(value):
                        invalid = True
                        break
                    sum_tp += value
                if invalid:
                    out[time_index] = np.nan
                    continue

                mean_tp = sum_tp / float(window)
                mean_deviation = 0.0
                for index in range(start, time_index + 1):
                    mean_deviation += abs(float(tp[index]) - mean_tp)
                mean_deviation = mean_deviation / float(window)
                denominator = 0.015 * mean_deviation
                if denominator == 0.0:
                    out[time_index] = np.nan
                else:
                    out[time_index] = (float(tp[time_index]) - mean_tp) / denominator

            out_f64[variant_index, :] = out
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.fisher":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        windows_i64 = _prepare_int_variants(name="windows", values=windows)
        if high_f64.shape[0] != low_f64.shape[0]:
            raise ValueError("high and low lengths must match")

        t_size = high_f64.shape[0]
        variants = windows_i64.shape[0]
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        hl2 = np.empty(t_size, dtype=np.float64)
        for time_index in range(t_size):
            high_value = float(high_f64[time_index])
            low_value = float(low_f64[time_index])
            if math.isnan(high_value) or math.isnan(low_value):
                hl2[time_index] = np.nan
            else:
                hl2[time_index] = (high_value + low_value) * 0.5

        for variant_index, window_raw in enumerate(windows_i64):
            window = int(window_raw)
            hh = _rolling_max_series_f64(source=high_f64, window=window)
            ll = _rolling_min_series_f64(source=low_f64, window=window)
            fisher = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                price = float(hl2[time_index])
                hh_value = float(hh[time_index])
                ll_value = float(ll[time_index])
                denominator = hh_value - ll_value
                if (
                    math.isnan(price)
                    or math.isnan(hh_value)
                    or math.isnan(ll_value)
                    or denominator == 0.0
                ):
                    fisher[time_index] = np.nan
                    continue
                normalized = (2.0 * ((price - ll_value) / denominator)) - 1.0
                if normalized > 0.999:
                    normalized = 0.999
                elif normalized < -0.999:
                    normalized = -0.999
                fisher[time_index] = 0.5 * math.log((1.0 + normalized) / (1.0 - normalized))
            out_f64[variant_index, :] = fisher
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.stoch":
        high_f64 = _prepare_series(name="high", values=high)
        low_f64 = _prepare_series(name="low", values=low)
        close_f64 = _prepare_series(name="close", values=close)
        if high_f64.shape[0] != low_f64.shape[0] or high_f64.shape[0] != close_f64.shape[0]:
            raise ValueError("high, low, close lengths must match")

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

        t_size = high_f64.shape[0]
        variants = k_windows_i64.shape[0]
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            out_f64[variant_index, :] = _stoch_k_series_f64(
                high=high_f64,
                low=low_f64,
                close=close_f64,
                k_window=int(k_windows_i64[variant_index]),
                smoothing=int(smoothings_i64[variant_index]),
                d_window=int(d_windows_i64[variant_index]),
            )
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id == "momentum.stoch_rsi":
        source_f64 = _prepare_source_variants(values=source_variants)
        variants = source_f64.shape[0]
        t_size = source_f64.shape[1]
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
        out_f64 = np.empty((variants, t_size), dtype=np.float64)
        for variant_index in range(variants):
            rsi_series = _rsi_series_f64(
                source=source_f64[variant_index, :],
                window=int(rsi_windows_i64[variant_index]),
            )
            hh = _rolling_max_series_f64(
                source=rsi_series,
                window=int(k_windows_i64[variant_index]),
            )
            ll = _rolling_min_series_f64(
                source=rsi_series,
                window=int(k_windows_i64[variant_index]),
            )
            k_raw = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                rsi_value = float(rsi_series[time_index])
                hh_value = float(hh[time_index])
                ll_value = float(ll[time_index])
                denominator = hh_value - ll_value
                if (
                    math.isnan(rsi_value)
                    or math.isnan(hh_value)
                    or math.isnan(ll_value)
                    or denominator == 0.0
                ):
                    k_raw[time_index] = np.nan
                else:
                    k_raw[time_index] = 100.0 * ((rsi_value - ll_value) / denominator)
            k = _rolling_mean_series_f64(source=k_raw, window=int(smoothings_i64[variant_index]))
            _ = _rolling_mean_series_f64(source=k, window=int(d_windows_i64[variant_index]))
            out_f64[variant_index, :] = k
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    if normalized_id in {"momentum.trix", "momentum.ppo", "momentum.macd"}:
        source_f64 = _prepare_source_variants(values=source_variants)
        variants = source_f64.shape[0]
        t_size = source_f64.shape[1]
        out_f64 = np.empty((variants, t_size), dtype=np.float64)

        if normalized_id == "momentum.trix":
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
            for variant_index in range(variants):
                source_row = source_f64[variant_index, :]
                ema1 = _ema_series_f64(source=source_row, window=int(windows_i64[variant_index]))
                ema2 = _ema_series_f64(source=ema1, window=int(windows_i64[variant_index]))
                ema3 = _ema_series_f64(source=ema2, window=int(windows_i64[variant_index]))
                trix = np.empty(t_size, dtype=np.float64)
                trix[0] = np.nan
                for time_index in range(1, t_size):
                    current = float(ema3[time_index])
                    previous = float(ema3[time_index - 1])
                    if math.isnan(current) or math.isnan(previous) or previous == 0.0:
                        trix[time_index] = np.nan
                    else:
                        trix[time_index] = 100.0 * ((current / previous) - 1.0)
                _ = _ema_series_f64(source=trix, window=int(signal_windows_i64[variant_index]))
                out_f64[variant_index, :] = trix
            return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

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
        for variant_index in range(variants):
            source_row = source_f64[variant_index, :]
            fast = _ema_series_f64(source=source_row, window=int(fast_windows_i64[variant_index]))
            slow = _ema_series_f64(source=source_row, window=int(slow_windows_i64[variant_index]))
            main = np.empty(t_size, dtype=np.float64)
            for time_index in range(t_size):
                fast_value = float(fast[time_index])
                slow_value = float(slow[time_index])
                if math.isnan(fast_value) or math.isnan(slow_value):
                    main[time_index] = np.nan
                    continue
                if normalized_id == "momentum.macd":
                    main[time_index] = fast_value - slow_value
                    continue
                if slow_value == 0.0:
                    main[time_index] = np.nan
                else:
                    main[time_index] = 100.0 * ((fast_value - slow_value) / slow_value)
            _ = _ema_series_f64(source=main, window=int(signal_windows_i64[variant_index]))
            out_f64[variant_index, :] = main
        return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))

    raise ValueError(f"unsupported momentum indicator_id: {indicator_id!r}")


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


def _rolling_min_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0
    for time_index in range(t_size):
        incoming = float(source[time_index])
        if math.isnan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if math.isnan(outgoing):
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


def _rolling_max_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    nan_count = 0
    for time_index in range(t_size):
        incoming = float(source[time_index])
        if math.isnan(incoming):
            nan_count += 1

        if time_index >= window:
            outgoing = float(source[time_index - window])
            if math.isnan(outgoing):
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


def _ema_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    alpha = 2.0 / (float(window) + 1.0)

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


def _rsi_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    alpha = 1.0 / float(window)
    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    previous_source = np.nan
    avg_gain = np.nan
    avg_loss = np.nan

    for time_index in range(t_size):
        current = float(source[time_index])
        if math.isnan(current):
            previous_source = np.nan
            avg_gain = np.nan
            avg_loss = np.nan
            out[time_index] = np.nan
            continue

        if math.isnan(previous_source):
            previous_source = current
            out[time_index] = np.nan
            continue

        delta = current - previous_source
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        if math.isnan(avg_gain):
            avg_gain = gain
        else:
            avg_gain = (alpha * gain) + ((1.0 - alpha) * avg_gain)

        if math.isnan(avg_loss):
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


def _roc_series_f64(*, source: np.ndarray, window: int) -> np.ndarray:
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
        ValueError: If window is non-positive.
    Side Effects:
        Allocates one output array.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    t_size = source.shape[0]
    out = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        if time_index < window:
            out[time_index] = np.nan
            continue
        current = float(source[time_index])
        previous = float(source[time_index - window])
        if math.isnan(current) or math.isnan(previous) or previous == 0.0:
            out[time_index] = np.nan
        else:
            out[time_index] = 100.0 * ((current / previous) - 1.0)
    return out


def _stoch_k_series_f64(
    *,
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
        ValueError: If window arguments are non-positive.
    Side Effects:
        Allocates intermediate arrays.
    """
    if k_window <= 0 or smoothing <= 0 or d_window <= 0:
        raise ValueError("k_window, smoothing, d_window must be > 0")
    if high.shape[0] != low.shape[0] or high.shape[0] != close.shape[0]:
        raise ValueError("high, low, close lengths must match")

    t_size = high.shape[0]
    hh = _rolling_max_series_f64(source=high, window=k_window)
    ll = _rolling_min_series_f64(source=low, window=k_window)
    k_raw = np.empty(t_size, dtype=np.float64)
    for time_index in range(t_size):
        close_value = float(close[time_index])
        hh_value = float(hh[time_index])
        ll_value = float(ll[time_index])
        denominator = hh_value - ll_value
        if (
            math.isnan(close_value)
            or math.isnan(hh_value)
            or math.isnan(ll_value)
            or denominator == 0.0
        ):
            k_raw[time_index] = np.nan
        else:
            k_raw[time_index] = 100.0 * ((close_value - ll_value) / denominator)
    k = _rolling_mean_series_f64(source=k_raw, window=smoothing)
    _ = _rolling_mean_series_f64(source=k, window=d_window)
    return k


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
