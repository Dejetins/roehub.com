"""
Indicators CandleFeed adapter (`market_data_acl`) backed by market_data canonical store.

Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
Related: src/trading/contexts/indicators/application/ports/feeds/candle_feed.py,
  src/trading/contexts/indicators/application/dto/candle_arrays.py,
  src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py

Policy literals:
- timeframe: "1m"
- timeline: "[start, end)"
- missing candles: "NaN"
- duplicates: "last-wins"
- out-of-range: "ignore"
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from trading.contexts.indicators.application.dto import CandleArrays
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.domain.errors import GridValidationError
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_TIMEFRAME_1M = Timeframe("1m")


class MarketDataCandleFeed(CandleFeed):
    """
    ACL implementation of `CandleFeed` using market_data canonical 1m candles.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/application/ports/feeds/candle_feed.py,
      src/trading/contexts/indicators/application/dto/candle_arrays.py
    """

    def __init__(self, *, canonical_candle_reader: CanonicalCandleReader) -> None:
        """
        Store market_data reader dependency for `market_data_acl` bridge.

        Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
        Related:
          src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py

        Args:
            canonical_candle_reader: Port for canonical 1m candle retrieval.
        Returns:
            None.
        Assumptions:
            Reader returns CandleWithMeta payloads for `[start, end)`.
        Raises:
            ValueError: If reader dependency is missing.
        Side Effects:
            None.
        """
        if canonical_candle_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataCandleFeed requires canonical_candle_reader")
        self._canonical_candle_reader = canonical_candle_reader

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Materialize dense contiguous OHLCV arrays on strict `1m` `[start, end)` grid.

        Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
        Related: src/trading/contexts/indicators/application/dto/candle_arrays.py,
          src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py

        Args:
            market_id: Stable market identifier.
            symbol: Instrument symbol in selected market.
            time_range: Half-open dense timeline range `[start, end)`.
        Returns:
            CandleArrays: Dense `ts_open` + OHLCV arrays with "NaN" holes.
        Assumptions:
            Duplicate timestamps follow deterministic "last-wins" policy after stable sort.
            Candles outside `[start, end)` are deterministically "ignore"-ed.
        Raises:
            GridValidationError: If `time_range` duration is not aligned to `1m`.
        Side Effects:
            None.
        """
        step_ms = _timeframe_millis(timeframe=_TIMEFRAME_1M)
        start_ms = _utc_timestamp_to_epoch_millis(ts=time_range.start)
        end_ms = _utc_timestamp_to_epoch_millis(ts=time_range.end)
        duration_ms = end_ms - start_ms
        _validate_time_range_alignment(duration_ms=duration_ms, step_ms=step_ms)

        timeline_size = int(duration_ms // step_ms)
        ts_open = np.arange(timeline_size, dtype=np.int64) * np.int64(step_ms) + np.int64(start_ms)
        dense_open = _nan_float32(size=timeline_size)
        dense_high = _nan_float32(size=timeline_size)
        dense_low = _nan_float32(size=timeline_size)
        dense_close = _nan_float32(size=timeline_size)
        dense_volume = _nan_float32(size=timeline_size)

        instrument_id = InstrumentId(market_id=market_id, symbol=symbol)
        sparse_rows = tuple(
            self._canonical_candle_reader.read_1m(
                instrument_id=instrument_id,
                time_range=time_range,
            )
        )
        if sparse_rows:
            sparse_ts, sparse_open, sparse_high, sparse_low, sparse_close, sparse_volume = (
                _extract_sparse_columns(rows=sparse_rows)
            )
            order = np.argsort(sparse_ts, kind="stable")
            sorted_ts = sparse_ts[order]
            sorted_open = sparse_open[order]
            sorted_high = sparse_high[order]
            sorted_low = sparse_low[order]
            sorted_close = sparse_close[order]
            sorted_volume = sparse_volume[order]

            in_range_mask = (sorted_ts >= start_ms) & (sorted_ts < end_ms)
            aligned_mask = ((sorted_ts - start_ms) % step_ms) == 0
            valid_mask = in_range_mask & aligned_mask
            if np.any(valid_mask):
                valid_ts = sorted_ts[valid_mask]
                valid_open = sorted_open[valid_mask]
                valid_high = sorted_high[valid_mask]
                valid_low = sorted_low[valid_mask]
                valid_close = sorted_close[valid_mask]
                valid_volume = sorted_volume[valid_mask]

                dense_idx = ((valid_ts - start_ms) // step_ms).astype(np.int64, copy=False)
                dense_open[dense_idx] = valid_open
                dense_high[dense_idx] = valid_high
                dense_low[dense_idx] = valid_low
                dense_close[dense_idx] = valid_close
                dense_volume[dense_idx] = valid_volume

        return CandleArrays(
            market_id=market_id,
            symbol=symbol,
            time_range=time_range,
            timeframe=_TIMEFRAME_1M,
            ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
            open=np.ascontiguousarray(dense_open, dtype=np.float32),
            high=np.ascontiguousarray(dense_high, dtype=np.float32),
            low=np.ascontiguousarray(dense_low, dtype=np.float32),
            close=np.ascontiguousarray(dense_close, dtype=np.float32),
            volume=np.ascontiguousarray(dense_volume, dtype=np.float32),
        )


def _validate_time_range_alignment(*, duration_ms: int, step_ms: int) -> None:
    """
    Validate strict dense timeline alignment for `1m` `[start, end)` contract.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/domain/errors/grid_validation_error.py

    Args:
        duration_ms: Requested range duration in milliseconds.
        step_ms: Timeframe duration in milliseconds.
    Returns:
        None.
    Assumptions:
        `step_ms` comes from `Timeframe("1m")`.
    Raises:
        GridValidationError: If duration is non-positive or not divisible by `step_ms`.
    Side Effects:
        None.
    """
    if step_ms <= 0:
        raise GridValidationError(f"timeframe duration must be > 0, got step_ms={step_ms}")
    if duration_ms <= 0:
        raise GridValidationError(f"time_range duration must be > 0, got duration_ms={duration_ms}")
    if duration_ms % step_ms != 0:
        raise GridValidationError(
            "time_range [start, end) must align to timeframe 1m: "
            f"duration_ms={duration_ms}, timeframe_ms={step_ms}"
        )


def _timeframe_millis(*, timeframe: Timeframe) -> int:
    """
    Convert timeframe duration into integer milliseconds.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/shared_kernel/primitives/timeframe.py

    Args:
        timeframe: Shared-kernel timeframe primitive.
    Returns:
        int: Duration in milliseconds.
    Assumptions:
        Timeframe duration is positive and millisecond-aligned.
    Raises:
        GridValidationError: If duration is not positive.
    Side Effects:
        None.
    """
    frame_ms = int(timeframe.duration() // timedelta(milliseconds=1))
    if frame_ms <= 0:
        raise GridValidationError(f"timeframe duration must be > 0, got timeframe_ms={frame_ms}")
    return frame_ms


def _utc_timestamp_to_epoch_millis(*, ts: UtcTimestamp) -> int:
    """
    Convert `UtcTimestamp` to epoch milliseconds without floating-point rounding.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/shared_kernel/primitives/utc_timestamp.py

    Args:
        ts: Timestamp value-object constrained to UTC with millisecond precision.
    Returns:
        int: Unix epoch milliseconds.
    Assumptions:
        Timestamp is timezone-aware UTC per shared-kernel invariants.
    Raises:
        None.
    Side Effects:
        None.
    """
    delta = ts.value - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))


def _nan_float32(*, size: int) -> np.ndarray:
    """
    Allocate contiguous float32 vector pre-filled with `NaN`.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        size: Target vector length.
    Returns:
        np.ndarray: C-contiguous float32 vector with all values equal to NaN.
    Assumptions:
        Size is non-negative and controlled by validated timeline math.
    Raises:
        None.
    Side Effects:
        Allocates one numpy array.
    """
    return np.full(size, np.nan, dtype=np.float32, order="C")


def _extract_sparse_columns(
    *,
    rows: Sequence[CandleWithMeta],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sparse candle columns into numpy vectors for vectorized materialization.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/market_data/application/dto/candle_with_meta.py

    Args:
        rows: Sparse canonical candles from market_data reader.
    Returns:
        tuple[np.ndarray, ...]: `ts_open` int64 plus OHLCV float32 vectors.
    Assumptions:
        Each row provides complete candle fields (`open/high/low/close/volume_base`).
    Raises:
        None.
    Side Effects:
        Allocates numpy arrays.
    """
    ts_values: list[int] = []
    open_values: list[float] = []
    high_values: list[float] = []
    low_values: list[float] = []
    close_values: list[float] = []
    volume_values: list[float] = []

    for row in rows:
        candle = row.candle
        ts_values.append(_utc_timestamp_to_epoch_millis(ts=candle.ts_open))
        open_values.append(float(candle.open))
        high_values.append(float(candle.high))
        low_values.append(float(candle.low))
        close_values.append(float(candle.close))
        volume_values.append(float(candle.volume_base))

    return (
        np.asarray(ts_values, dtype=np.int64),
        np.asarray(open_values, dtype=np.float32),
        np.asarray(high_values, dtype=np.float32),
        np.asarray(low_values, dtype=np.float32),
        np.asarray(close_values, dtype=np.float32),
        np.asarray(volume_values, dtype=np.float32),
    )
