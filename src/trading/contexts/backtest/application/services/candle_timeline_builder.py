from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange, UtcTimestamp

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_BASE_TIMEFRAME = Timeframe("1m")
_BASE_TIMEFRAME_MS = int(_BASE_TIMEFRAME.duration() // timedelta(milliseconds=1))
_NO_MARKET_DATA_MESSAGE = "no market data for requested range"


@dataclass(frozen=True, slots=True)
class BacktestCandleTimeline:
    """
    Backtest candle timeline payload for BKT-EPIC-02 orchestration.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
    """

    candles: CandleArrays
    normalized_1m_time_range: TimeRange
    target_slice: slice

    def __post_init__(self) -> None:
        """
        Validate deterministic timeline payload invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `target_slice` uses half-open Python slicing semantics and references `candles`.
        Raises:
            ValueError: If payload fields are missing or slice boundaries are invalid.
        Side Effects:
            None.
        """
        if self.candles is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestCandleTimeline.candles is required")
        if self.normalized_1m_time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestCandleTimeline.normalized_1m_time_range is required")
        if self.target_slice.start is None or self.target_slice.stop is None:
            raise ValueError("BacktestCandleTimeline.target_slice requires explicit start/stop")
        if self.target_slice.start < 0:
            raise ValueError("BacktestCandleTimeline.target_slice.start must be >= 0")
        if self.target_slice.stop < self.target_slice.start:
            raise ValueError("BacktestCandleTimeline.target_slice.stop must be >= start")


class BacktestCandleTimelineBuilder:
    """
    Build deterministic backtest candle timeline from dense 1m feed for BKT-EPIC-02.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def __init__(self, *, candle_feed: CandleFeed) -> None:
        """
        Store CandleFeed dependency used as dense 1m source of truth adapter.

        Args:
            candle_feed: Indicators CandleFeed contract for `load_1m_dense(...)`.
        Returns:
            None.
        Assumptions:
            Feed returns deterministic dense `1m` arrays with NaN holes.
        Raises:
            ValueError: If dependency is missing.
        Side Effects:
            None.
        """
        if candle_feed is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestCandleTimelineBuilder requires candle_feed")
        self._candle_feed = candle_feed

    def build(
        self,
        *,
        market_id: MarketId,
        symbol: Symbol,
        timeframe: Timeframe,
        requested_time_range: TimeRange,
        warmup_bars: int,
    ) -> BacktestCandleTimeline:
        """
        Build backtest candles for selected timeframe with best-effort rollup policy.

        Docs:
          - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
          - docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
        Related:
          - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/
            market_data_candle_feed.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - src/trading/shared_kernel/primitives/time_range.py

        Args:
            market_id: Stable market identifier.
            symbol: Market-local instrument symbol.
            timeframe: Target timeframe requested by backtest template.
            requested_time_range: User-provided target range `[Start, End)`.
            warmup_bars: Warmup bars count in target timeframe units.
        Returns:
            BacktestCandleTimeline: Rolled candles with normalized load range and target slice.
        Assumptions:
            Warmup lookback is measured in bars of `timeframe`.
        Raises:
            BacktestValidationError: If range normalization/rollup cannot produce market data.
        Side Effects:
            Calls CandleFeed port once to load dense `1m` candles.
        """
        normalized_1m_time_range = normalize_1m_load_time_range(
            requested_time_range=requested_time_range,
            timeframe=timeframe,
            warmup_bars=warmup_bars,
        )
        candles_1m = self._candle_feed.load_1m_dense(
            market_id=market_id,
            symbol=symbol,
            time_range=normalized_1m_time_range,
        )
        rolled_candles = rollup_1m_candles_best_effort(
            candles_1m=candles_1m,
            timeframe=timeframe,
        )
        target_slice = compute_target_slice_by_bar_close_ts(
            candles=rolled_candles,
            target_time_range=requested_time_range,
        )
        return BacktestCandleTimeline(
            candles=rolled_candles,
            normalized_1m_time_range=normalized_1m_time_range,
            target_slice=target_slice,
        )


def normalize_1m_load_time_range(
    *,
    requested_time_range: TimeRange,
    timeframe: Timeframe,
    warmup_bars: int,
) -> TimeRange:
    """
    Normalize user range into minute-aligned dense `1m` load window with warmup lookback.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/shared_kernel/primitives/time_range.py
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/
        market_data_candle_feed.py

    Args:
        requested_time_range: User-provided target range `[Start, End)`.
        timeframe: Backtest target timeframe.
        warmup_bars: Warmup bars count in target timeframe units.
    Returns:
        TimeRange: Normalized minute-aligned range for internal `load_1m_dense`.
    Assumptions:
        `warmup_bars` is strictly positive.
    Raises:
        BacktestValidationError: If warmup bars are non-positive or normalized bounds are invalid.
    Side Effects:
        None.
    """
    if warmup_bars <= 0:
        raise BacktestValidationError("warmup_bars must be > 0")

    warmup_duration = timeframe.duration() * warmup_bars
    normalized_start = _floor_datetime_to_minute(
        dt=requested_time_range.start.value - warmup_duration
    )
    normalized_end = _ceil_datetime_to_minute(dt=requested_time_range.end.value)
    if normalized_start >= normalized_end:
        raise BacktestValidationError("normalized 1m load range must satisfy start < end")

    return TimeRange(
        start=UtcTimestamp(normalized_start),
        end=UtcTimestamp(normalized_end),
    )


def rollup_1m_candles_best_effort(
    *,
    candles_1m: CandleArrays,
    timeframe: Timeframe,
) -> CandleArrays:
    """
    Roll dense `1m` candles to target timeframe using best-effort and carry-forward policies.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - src/trading/contexts/backtest/domain/errors/backtest_errors.py

    Args:
        candles_1m: Dense `1m` candles with potential NaN holes.
        timeframe: Target timeframe for rollup.
    Returns:
        CandleArrays: Deterministic rolled candles with no NaN values.
    Assumptions:
        Availability criterion in v1 is `close` being finite.
    Raises:
        BacktestValidationError: If there is no market data in the entire range.
    Side Effects:
        Allocates numpy arrays for rolled series.
    """
    if candles_1m.timeframe.code != _BASE_TIMEFRAME.code:
        raise BacktestValidationError("rollup source timeframe must be 1m")
    if candles_1m.ts_open.shape[0] == 0:
        raise BacktestValidationError(_NO_MARKET_DATA_MESSAGE)

    start_ms = _utc_timestamp_to_epoch_millis(ts=candles_1m.time_range.start)
    end_ms = _utc_timestamp_to_epoch_millis(ts=candles_1m.time_range.end)
    timeframe_ms = _timeframe_millis(timeframe=timeframe)
    first_bucket_open_ms = _utc_timestamp_to_epoch_millis(
        ts=timeframe.bucket_open(candles_1m.time_range.start)
    )

    ts_open_values: list[int] = []
    open_values: list[float] = []
    high_values: list[float] = []
    low_values: list[float] = []
    close_values: list[float] = []
    volume_values: list[float] = []
    prev_close: float | None = None
    bucket_open_ms = first_bucket_open_ms

    while True:
        bucket_close_ms = bucket_open_ms + timeframe_ms
        if bucket_close_ms >= end_ms:
            break
        if bucket_close_ms < start_ms:
            bucket_open_ms += timeframe_ms
            continue

        bucket_ohlcv = _extract_bucket_ohlcv(
            candles_1m=candles_1m,
            bucket_open_ms=bucket_open_ms,
            bucket_close_ms=bucket_close_ms,
            range_start_ms=start_ms,
        )
        if bucket_ohlcv is None:
            if prev_close is not None:
                ts_open_values.append(bucket_open_ms)
                open_values.append(prev_close)
                high_values.append(prev_close)
                low_values.append(prev_close)
                close_values.append(prev_close)
                volume_values.append(0.0)
            bucket_open_ms += timeframe_ms
            continue

        bucket_open, bucket_high, bucket_low, bucket_close, bucket_volume = bucket_ohlcv
        prev_close = bucket_close
        ts_open_values.append(bucket_open_ms)
        open_values.append(bucket_open)
        high_values.append(bucket_high)
        low_values.append(bucket_low)
        close_values.append(bucket_close)
        volume_values.append(bucket_volume)
        bucket_open_ms += timeframe_ms

    if len(ts_open_values) == 0:
        raise BacktestValidationError(_NO_MARKET_DATA_MESSAGE)

    ts_open = np.ascontiguousarray(np.asarray(ts_open_values, dtype=np.int64), dtype=np.int64)
    rolled_open = np.ascontiguousarray(np.asarray(open_values, dtype=np.float32), dtype=np.float32)
    rolled_high = np.ascontiguousarray(np.asarray(high_values, dtype=np.float32), dtype=np.float32)
    rolled_low = np.ascontiguousarray(np.asarray(low_values, dtype=np.float32), dtype=np.float32)
    rolled_close = np.ascontiguousarray(
        np.asarray(close_values, dtype=np.float32),
        dtype=np.float32,
    )
    rolled_volume = np.ascontiguousarray(
        np.asarray(volume_values, dtype=np.float32),
        dtype=np.float32,
    )
    _ensure_finite_derived_series(
        open_values=rolled_open,
        high_values=rolled_high,
        low_values=rolled_low,
        close_values=rolled_close,
        volume_values=rolled_volume,
    )

    derived_time_range = TimeRange(
        start=_epoch_millis_to_utc_timestamp(ms=int(ts_open[0])),
        end=_epoch_millis_to_utc_timestamp(ms=int(ts_open[-1]) + timeframe_ms),
    )
    return CandleArrays(
        market_id=candles_1m.market_id,
        symbol=candles_1m.symbol,
        time_range=derived_time_range,
        timeframe=timeframe,
        ts_open=ts_open,
        open=rolled_open,
        high=rolled_high,
        low=rolled_low,
        close=rolled_close,
        volume=rolled_volume,
    )


def compute_target_slice_by_bar_close_ts(
    *,
    candles: CandleArrays,
    target_time_range: TimeRange,
) -> slice:
    """
    Compute `[Start, End)` target slice by `bar_close_ts` semantics for backtest reporting.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/shared_kernel/primitives/time_range.py
      - src/trading/shared_kernel/primitives/timeframe.py

    Args:
        candles: Rolled candles for one timeframe.
        target_time_range: User-provided target range `[Start, End)`.
    Returns:
        slice: Half-open index slice satisfying `Start <= bar_close_ts < End`.
    Assumptions:
        `candles.ts_open` is sorted in non-decreasing order.
    Raises:
        None.
    Side Effects:
        None.
    """
    if candles.ts_open.shape[0] == 0:
        return slice(0, 0)

    timeframe_ms = _timeframe_millis(timeframe=candles.timeframe)
    target_start_ms = _utc_timestamp_to_epoch_millis(ts=target_time_range.start)
    target_end_ms = _utc_timestamp_to_epoch_millis(ts=target_time_range.end)
    bar_close_ts = candles.ts_open.astype(np.int64, copy=False) + np.int64(timeframe_ms)

    slice_start = int(np.searchsorted(bar_close_ts, np.int64(target_start_ms), side="left"))
    slice_stop = int(np.searchsorted(bar_close_ts, np.int64(target_end_ms), side="left"))
    if slice_stop < slice_start:
        slice_stop = slice_start
    return slice(slice_start, slice_stop)


def _extract_bucket_ohlcv(
    *,
    candles_1m: CandleArrays,
    bucket_open_ms: int,
    bucket_close_ms: int,
    range_start_ms: int,
) -> tuple[float, float, float, float, float] | None:
    """
    Aggregate one bucket from dense `1m` arrays ignoring missing minutes by close-finite rule.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - src/trading/shared_kernel/primitives/timeframe.py

    Args:
        candles_1m: Dense `1m` candle arrays.
        bucket_open_ms: Bucket open timestamp in epoch milliseconds.
        bucket_close_ms: Bucket close timestamp in epoch milliseconds.
        range_start_ms: Dense range start in epoch milliseconds.
    Returns:
        tuple[float, float, float, float, float] | None:
            Aggregated OHLCV tuple or `None` when no available minutes exist.
    Assumptions:
        Source arrays follow `1m` regular grid and index-to-minute mapping.
    Raises:
        None.
    Side Effects:
        None.
    """
    start_idx = max(0, int((bucket_open_ms - range_start_ms) // _BASE_TIMEFRAME_MS))
    end_idx = min(
        int(candles_1m.ts_open.shape[0]),
        int((bucket_close_ms - range_start_ms) // _BASE_TIMEFRAME_MS),
    )
    if start_idx >= end_idx:
        return None

    bucket_open_values = candles_1m.open[start_idx:end_idx]
    bucket_high_values = candles_1m.high[start_idx:end_idx]
    bucket_low_values = candles_1m.low[start_idx:end_idx]
    bucket_close_values = candles_1m.close[start_idx:end_idx]
    bucket_volume_values = candles_1m.volume[start_idx:end_idx]

    available_mask = np.isfinite(bucket_close_values)
    if not np.any(available_mask):
        return None

    available_idx = np.flatnonzero(available_mask)
    first_idx = int(available_idx[0])
    last_idx = int(available_idx[-1])
    open_value = float(bucket_open_values[first_idx])
    close_value = float(bucket_close_values[last_idx])
    high_value = float(np.max(bucket_high_values[available_mask]))
    low_value = float(np.min(bucket_low_values[available_mask]))
    volume_value = float(np.sum(bucket_volume_values[available_mask], dtype=np.float64))

    if not np.all(
        np.isfinite(
            np.asarray(
                [open_value, high_value, low_value, close_value, volume_value],
                dtype=np.float64,
            )
        )
    ):
        return None
    return (open_value, high_value, low_value, close_value, volume_value)


def _ensure_finite_derived_series(
    *,
    open_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    volume_values: np.ndarray,
) -> None:
    """
    Enforce `no NaN` invariant for backtest derived candles.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/domain/errors/backtest_errors.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        open_values: Rolled open series.
        high_values: Rolled high series.
        low_values: Rolled low series.
        close_values: Rolled close series.
        volume_values: Rolled volume series.
    Returns:
        None.
    Assumptions:
        Arrays are float series aligned by index.
    Raises:
        BacktestValidationError: If any output value is non-finite.
    Side Effects:
        None.
    """
    if not np.all(np.isfinite(open_values)):
        raise BacktestValidationError("derived candles must not contain NaN in open")
    if not np.all(np.isfinite(high_values)):
        raise BacktestValidationError("derived candles must not contain NaN in high")
    if not np.all(np.isfinite(low_values)):
        raise BacktestValidationError("derived candles must not contain NaN in low")
    if not np.all(np.isfinite(close_values)):
        raise BacktestValidationError("derived candles must not contain NaN in close")
    if not np.all(np.isfinite(volume_values)):
        raise BacktestValidationError("derived candles must not contain NaN in volume")


def _floor_datetime_to_minute(*, dt: datetime) -> datetime:
    """
    Floor UTC datetime to minute boundary without float rounding.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/
        market_data_candle_feed.py

    Args:
        dt: Timezone-aware UTC datetime.
    Returns:
        datetime: Floored datetime at `...:..:00.000Z`.
    Assumptions:
        Input datetime is timezone-aware.
    Raises:
        None.
    Side Effects:
        None.
    """
    epoch_ms = _datetime_to_epoch_millis(dt=dt)
    floored_ms = (epoch_ms // _BASE_TIMEFRAME_MS) * _BASE_TIMEFRAME_MS
    return _epoch_millis_to_datetime(ms=floored_ms)


def _ceil_datetime_to_minute(*, dt: datetime) -> datetime:
    """
    Ceil UTC datetime to minute boundary without float rounding.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/
        market_data_candle_feed.py

    Args:
        dt: Timezone-aware UTC datetime.
    Returns:
        datetime: Ceiled datetime at minute boundary.
    Assumptions:
        Input datetime is timezone-aware.
    Raises:
        None.
    Side Effects:
        None.
    """
    epoch_ms = _datetime_to_epoch_millis(dt=dt)
    quotient, remainder = divmod(epoch_ms, _BASE_TIMEFRAME_MS)
    ceiled_ms = epoch_ms if remainder == 0 else (quotient + 1) * _BASE_TIMEFRAME_MS
    return _epoch_millis_to_datetime(ms=ceiled_ms)


def _timeframe_millis(*, timeframe: Timeframe) -> int:
    """
    Convert timeframe duration to integer milliseconds.

    Docs:
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py

    Args:
        timeframe: Shared-kernel timeframe.
    Returns:
        int: Duration in milliseconds.
    Assumptions:
        Timeframe duration is positive.
    Raises:
        BacktestValidationError: If timeframe duration is non-positive.
    Side Effects:
        None.
    """
    timeframe_ms = int(timeframe.duration() // timedelta(milliseconds=1))
    if timeframe_ms <= 0:
        raise BacktestValidationError("timeframe duration must be > 0")
    return timeframe_ms


def _utc_timestamp_to_epoch_millis(*, ts: UtcTimestamp) -> int:
    """
    Convert `UtcTimestamp` to epoch milliseconds without floating-point rounding.

    Docs:
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py

    Args:
        ts: UTC timestamp value-object.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Timestamp already satisfies UTC shared-kernel invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _datetime_to_epoch_millis(dt=ts.value)


def _epoch_millis_to_utc_timestamp(*, ms: int) -> UtcTimestamp:
    """
    Convert epoch milliseconds to `UtcTimestamp` using integer arithmetic.

    Docs:
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py

    Args:
        ms: Epoch milliseconds.
    Returns:
        UtcTimestamp: Converted UTC timestamp value-object.
    Assumptions:
        Millisecond value is within representable datetime bounds.
    Raises:
        ValueError: If resulting timestamp violates shared-kernel UTC invariants.
    Side Effects:
        None.
    """
    return UtcTimestamp(_epoch_millis_to_datetime(ms=ms))


def _datetime_to_epoch_millis(*, dt: datetime) -> int:
    """
    Convert timezone-aware datetime to epoch milliseconds without float operations.

    Docs:
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py

    Args:
        dt: Timezone-aware datetime.
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
    normalized = dt.astimezone(timezone.utc)
    delta = normalized - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))


def _epoch_millis_to_datetime(*, ms: int) -> datetime:
    """
    Convert epoch milliseconds to timezone-aware UTC datetime.

    Docs:
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/utc_timestamp.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py

    Args:
        ms: Epoch milliseconds.
    Returns:
        datetime: UTC datetime with millisecond precision.
    Assumptions:
        Input is integer milliseconds from unix epoch.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _EPOCH_UTC + timedelta(milliseconds=ms)


__all__ = [
    "BacktestCandleTimeline",
    "BacktestCandleTimelineBuilder",
    "compute_target_slice_by_bar_close_ts",
    "normalize_1m_load_time_range",
    "rollup_1m_candles_best_effort",
]
