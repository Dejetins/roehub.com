from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    BacktestCandleTimelineBuilder,
    compute_target_slice_by_bar_close_ts,
    normalize_1m_load_time_range,
    rollup_1m_candles_best_effort,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MINUTE = timedelta(minutes=1)


class _RecordingCandleFeed:
    """
    Deterministic CandleFeed stub that records load calls for assertion.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
    """

    def __init__(self, *, candles: CandleArrays) -> None:
        """
        Store deterministic dense `1m` candles returned on each load call.

        Args:
            candles: Dense candle arrays returned by stub.
        Returns:
            None.
        Assumptions:
            Test config controls candles payload deterministically.
        Raises:
            ValueError: If candles payload is missing.
        Side Effects:
            None.
        """
        if candles is None:  # type: ignore[truthy-bool]
            raise ValueError("_RecordingCandleFeed requires candles")
        self.calls: list[TimeRange] = []
        self._candles = candles

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Record call arguments and return deterministic dense `1m` candles.

        Args:
            market_id: Market identifier (unused in stub assertions).
            symbol: Instrument symbol (unused in stub assertions).
            time_range: Requested aligned range captured for assertions.
        Returns:
            CandleArrays: Preconfigured dense candles payload.
        Assumptions:
            Caller already validates API-level invariants.
        Raises:
            None.
        Side Effects:
            Appends one item to in-memory calls list.
        """
        _ = market_id, symbol
        self.calls.append(time_range)
        return self._candles


def test_normalize_1m_load_time_range_applies_floor_ceil_and_warmup() -> None:
    """
    Verify range normalization applies warmup duration and minute floor/ceil deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Warmup bars are measured in target timeframe units.
    Raises:
        AssertionError: If normalized boundaries differ from BKT-EPIC-02 contract.
    Side Effects:
        None.
    """
    requested_range = TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 16, 10, 2, 30, 123000, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 16, 10, 33, 5, 1000, tzinfo=timezone.utc)),
    )

    normalized = normalize_1m_load_time_range(
        requested_time_range=requested_range,
        timeframe=Timeframe("5m"),
        warmup_bars=2,
    )

    assert normalized.start == UtcTimestamp(datetime(2026, 2, 16, 9, 52, tzinfo=timezone.utc))
    assert normalized.end == UtcTimestamp(datetime(2026, 2, 16, 10, 34, tzinfo=timezone.utc))


def test_rollup_1m_best_effort_keeps_bucket_with_missing_minutes() -> None:
    """
    Verify missing `1m` points inside bucket do not drop derived bucket in best-effort mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Availability criterion is finite `close` in source `1m` arrays.
    Raises:
        AssertionError: If rolled OHLCV aggregation violates best-effort contract.
    Side Effects:
        None.
    """
    candles_1m = _build_dense_1m_candles(
        start=datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc),
        close_values=(
            100.0,
            101.0,
            None,
            103.0,
            104.0,
            200.0,
            201.0,
            202.0,
            203.0,
            204.0,
            300.0,
            301.0,
            302.0,
            303.0,
            304.0,
            999.0,
        ),
    )

    rolled = rollup_1m_candles_best_effort(candles_1m=candles_1m, timeframe=Timeframe("5m"))

    assert tuple(rolled.ts_open.tolist()) == (
        _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)),
        _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 5, tzinfo=timezone.utc)),
        _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 10, tzinfo=timezone.utc)),
    )
    assert rolled.open[0] == pytest.approx(100.0)
    assert rolled.close[0] == pytest.approx(104.0)
    assert rolled.high[0] == pytest.approx(104.0)
    assert rolled.low[0] == pytest.approx(100.0)
    assert rolled.volume[0] == pytest.approx(12.0)
    assert bool(np.all(np.isfinite(rolled.close)))


def test_rollup_1m_carry_forward_fills_empty_bucket_with_prev_close() -> None:
    """
    Verify fully empty bucket is emitted as carry-forward (`OHLC=prev_close`, `volume=0`).

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Previous close exists from earlier non-empty bucket in same rolled range.
    Raises:
        AssertionError: If carry-forward values diverge from deterministic policy.
    Side Effects:
        None.
    """
    candles_1m = _build_dense_1m_candles(
        start=datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc),
        close_values=(
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            None,
            None,
            None,
            None,
            None,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            888.0,
        ),
    )

    rolled = rollup_1m_candles_best_effort(candles_1m=candles_1m, timeframe=Timeframe("5m"))

    assert rolled.open[1] == pytest.approx(14.0)
    assert rolled.high[1] == pytest.approx(14.0)
    assert rolled.low[1] == pytest.approx(14.0)
    assert rolled.close[1] == pytest.approx(14.0)
    assert rolled.volume[1] == pytest.approx(0.0)
    assert bool(np.all(np.isfinite(rolled.open)))
    assert bool(np.all(np.isfinite(rolled.high)))
    assert bool(np.all(np.isfinite(rolled.low)))
    assert bool(np.all(np.isfinite(rolled.close)))
    assert bool(np.all(np.isfinite(rolled.volume)))


def test_rollup_1m_raises_validation_error_when_no_market_data_exists() -> None:
    """
    Verify deterministic validation error is raised when all source buckets are empty.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        No finite `close` values in entire source range means no market data.
    Raises:
        AssertionError: If expected error is not raised or message is non-deterministic.
    Side Effects:
        None.
    """
    candles_1m = _build_dense_1m_candles(
        start=datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc),
        close_values=tuple(None for _ in range(16)),
    )

    with pytest.raises(BacktestValidationError, match="no market data for requested range"):
        rollup_1m_candles_best_effort(candles_1m=candles_1m, timeframe=Timeframe("5m"))


def test_compute_target_slice_by_bar_close_ts_uses_half_open_range() -> None:
    """
    Verify target slice includes bars by `Start <= bar_close_ts < End` rule.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Input candles are already sorted by `ts_open`.
    Raises:
        AssertionError: If computed slice bounds violate close-based rule.
    Side Effects:
        None.
    """
    rolled = CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 0, 20, tzinfo=timezone.utc)),
        ),
        timeframe=Timeframe("5m"),
        ts_open=np.asarray(
            [
                _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)),
                _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 5, tzinfo=timezone.utc)),
                _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 10, tzinfo=timezone.utc)),
                _datetime_to_epoch_millis(datetime(2026, 2, 16, 0, 15, tzinfo=timezone.utc)),
            ],
            dtype=np.int64,
        ),
        open=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        high=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        low=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        close=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        volume=np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
    )

    target_slice = compute_target_slice_by_bar_close_ts(
        candles=rolled,
        target_time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 0, 9, 30, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 0, 17, tzinfo=timezone.utc)),
        ),
    )

    assert target_slice.start == 1
    assert target_slice.stop == 3


def test_builder_calls_feed_with_normalized_1m_range() -> None:
    """
    Verify builder requests minute-aligned warmup range from CandleFeed and returns rolled data.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Feed returns deterministic dense `1m` candles for provided aligned range.
    Raises:
        AssertionError: If builder call contract or no-NaN invariant is broken.
    Side Effects:
        None.
    """
    requested_range = TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 16, 10, 2, 30, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 16, 10, 21, 40, tzinfo=timezone.utc)),
    )
    expected_normalized = normalize_1m_load_time_range(
        requested_time_range=requested_range,
        timeframe=Timeframe("5m"),
        warmup_bars=2,
    )
    dense_candles = _build_dense_1m_candles_from_time_range(time_range=expected_normalized)
    feed = _RecordingCandleFeed(candles=dense_candles)
    builder = BacktestCandleTimelineBuilder(candle_feed=feed)

    timeline = builder.build(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        timeframe=Timeframe("5m"),
        requested_time_range=requested_range,
        warmup_bars=2,
    )

    assert len(feed.calls) == 1
    assert feed.calls[0] == expected_normalized
    assert timeline.normalized_1m_time_range == expected_normalized
    assert timeline.candles.timeframe == Timeframe("5m")
    assert bool(np.all(np.isfinite(timeline.candles.close)))


def _build_dense_1m_candles_from_time_range(*, time_range: TimeRange) -> CandleArrays:
    """
    Build deterministic dense `1m` CandleArrays for supplied minute-aligned time range.

    Args:
        time_range: Minute-aligned range used for timeline length.
    Returns:
        CandleArrays: Dense source arrays with finite OHLCV values.
    Assumptions:
        Range duration is divisible by one minute.
    Raises:
        ValueError: If range duration cannot be represented as minute count.
    Side Effects:
        Allocates numpy arrays.
    """
    minutes = int((time_range.duration() // _MINUTE))
    close_values = tuple(float(index + 1) for index in range(minutes))
    return _build_dense_1m_candles(
        start=time_range.start.value,
        close_values=close_values,
        time_range=time_range,
    )


def _build_dense_1m_candles(
    *,
    start: datetime,
    close_values: tuple[float | None, ...],
    time_range: TimeRange | None = None,
) -> CandleArrays:
    """
    Build deterministic dense `1m` CandleArrays with optional NaN holes in OHLCV.

    Args:
        start: Start datetime for first minute candle.
        close_values: Per-minute close values; `None` produces NaN hole.
        time_range: Optional explicit range override.
    Returns:
        CandleArrays: Dense `1m` arrays for rollup tests.
    Assumptions:
        `close_values` order defines deterministic timestamp order.
    Raises:
        ValueError: If start datetime is not timezone-aware.
    Side Effects:
        Allocates numpy arrays.
    """
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start must be timezone-aware")

    timeline_length = len(close_values)
    start_ms = _datetime_to_epoch_millis(start)
    ts_open = np.arange(timeline_length, dtype=np.int64) * np.int64(60_000) + np.int64(start_ms)
    open_values: list[float] = []
    high_values: list[float] = []
    low_values: list[float] = []
    close_series: list[float] = []
    volume_values: list[float] = []
    for idx, value in enumerate(close_values):
        if value is None:
            open_values.append(np.nan)
            high_values.append(np.nan)
            low_values.append(np.nan)
            close_series.append(np.nan)
            volume_values.append(np.nan)
            continue

        finite_value = float(value)
        open_values.append(finite_value)
        high_values.append(finite_value)
        low_values.append(finite_value)
        close_series.append(finite_value)
        volume_values.append(float(idx + 1))

    resolved_time_range = time_range or TimeRange(
        start=UtcTimestamp(start),
        end=UtcTimestamp(start + timeline_length * _MINUTE),
    )
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=resolved_time_range,
        timeframe=Timeframe("1m"),
        ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
        open=np.ascontiguousarray(np.asarray(open_values, dtype=np.float32), dtype=np.float32),
        high=np.ascontiguousarray(np.asarray(high_values, dtype=np.float32), dtype=np.float32),
        low=np.ascontiguousarray(np.asarray(low_values, dtype=np.float32), dtype=np.float32),
        close=np.ascontiguousarray(np.asarray(close_series, dtype=np.float32), dtype=np.float32),
        volume=np.ascontiguousarray(np.asarray(volume_values, dtype=np.float32), dtype=np.float32),
    )


def _datetime_to_epoch_millis(dt: datetime) -> int:
    """
    Convert timezone-aware datetime to epoch milliseconds with integer arithmetic.

    Args:
        dt: Input timezone-aware datetime.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Input datetime is timezone-aware.
    Raises:
        ValueError: If datetime is naive.
    Side Effects:
        None.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = dt.astimezone(timezone.utc) - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))
