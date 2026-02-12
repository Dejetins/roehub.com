"""
Unit tests for indicators CandleFeed `market_data_acl` adapter.

Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
Related:
  src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py,
  src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence

import numpy as np
import pytest

from trading.contexts.indicators.adapters.outbound.feeds.market_data_acl import (
    MarketDataCandleFeed,
)
from trading.contexts.indicators.domain.errors import GridValidationError
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

_MARKET_ID = MarketId(1)
_SYMBOL = Symbol("BTCUSDT")
_START = UtcTimestamp(datetime(2026, 2, 11, 10, 0, tzinfo=timezone.utc))


class _CanonicalReaderStub(CanonicalCandleReader):
    """
    Deterministic in-memory canonical reader stub for `market_data_acl` tests.
    """

    def __init__(self, *, rows: Sequence[CandleWithMeta]) -> None:
        """
        Persist static rows and request-capture fields.

        Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
        Related:
          src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py

        Args:
            rows: Static canonical candle payload to stream from `read_1m`.
        Returns:
            None.
        Assumptions:
            Rows are immutable for one test case.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._rows = tuple(rows)
        self.last_instrument_id: InstrumentId | None = None
        self.last_time_range: TimeRange | None = None

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Return fixed rows while capturing request metadata for assertions.

        Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
        Related: src/trading/contexts/market_data/application/dto/candle_with_meta.py

        Args:
            instrument_id: Requested instrument identifier.
            time_range: Requested half-open interval `[start, end)`.
        Returns:
            Iterator[CandleWithMeta]: Iterator over fixture rows.
        Assumptions:
            Adapter under test may pass any range/instrument; stub does not filter.
        Raises:
            None.
        Side Effects:
            Stores the latest call arguments for test assertions.
        """
        self.last_instrument_id = instrument_id
        self.last_time_range = time_range
        return iter(self._rows)


def _utc_ts(*, minute_offset: int, second: int = 0) -> UtcTimestamp:
    """
    Build deterministic UTC timestamp relative to fixture `_START`.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/shared_kernel/primitives/utc_timestamp.py

    Args:
        minute_offset: Integer minute offset from `_START`.
        second: Optional extra second component for misalignment tests.
    Returns:
        UtcTimestamp: Normalized UTC timestamp.
    Assumptions:
        `_START` remains fixed for all tests in this module.
    Raises:
        ValueError: If resulting datetime violates `UtcTimestamp` invariants.
    Side Effects:
        None.
    """
    base = _START.value + timedelta(minutes=minute_offset, seconds=second)
    return UtcTimestamp(base)


def _time_range(*, end_minute_offset: int, end_second: int = 0) -> TimeRange:
    """
    Build deterministic `[start, end)` range for adapter tests.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/shared_kernel/primitives/time_range.py

    Args:
        end_minute_offset: Minute offset for range end relative to `_START`.
        end_second: Optional second component for non-aligned ranges.
    Returns:
        TimeRange: Requested half-open interval.
    Assumptions:
        Start is always `_START`.
    Raises:
        ValueError: If end is not strictly greater than start.
    Side Effects:
        None.
    """
    return TimeRange(
        start=_START,
        end=_utc_ts(minute_offset=end_minute_offset, second=end_second),
    )


def _epoch_ms(ts: UtcTimestamp) -> int:
    """
    Convert UTC timestamp primitive to epoch milliseconds.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/shared_kernel/primitives/utc_timestamp.py

    Args:
        ts: Timestamp primitive.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Timestamp already normalized to UTC and milliseconds.
    Raises:
        None.
    Side Effects:
        None.
    """
    return int(ts.value.timestamp() * 1000)


def _row(*, minute_offset: int, open_value: float, volume: float = 1.0) -> CandleWithMeta:
    """
    Build one canonical candle row fixture with deterministic metadata.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/market_data/application/dto/candle_with_meta.py

    Args:
        minute_offset: Candle open minute offset relative to `_START`.
        open_value: Open price used to make duplicates assertions explicit.
        volume: Base volume for the candle.
    Returns:
        CandleWithMeta: Valid canonical row fixture.
    Assumptions:
        `high/low/close` derived from `open_value` satisfy `Candle` invariants.
    Raises:
        ValueError: If candle/meta primitives violate shared-kernel invariants.
    Side Effects:
        None.
    """
    ts_open = _utc_ts(minute_offset=minute_offset)
    ts_close = UtcTimestamp(ts_open.value + timedelta(minutes=1))
    instrument_id = InstrumentId(market_id=_MARKET_ID, symbol=_SYMBOL)
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=ts_open,
        ts_close=ts_close,
        open=open_value,
        high=open_value + 2.0,
        low=open_value - 2.0,
        close=open_value + 1.0,
        volume_base=volume,
        volume_quote=None,
    )
    meta = CandleMeta(
        source="file",
        ingested_at=UtcTimestamp(ts_open.value + timedelta(minutes=5)),
        ingest_id=None,
        instrument_key=f"{_MARKET_ID.value}:{_SYMBOL}",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_market_data_acl_rejects_non_aligned_time_range() -> None:
    """
    Verify strict alignment validation for `[start, end)` against fixed `1m` grid.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/domain/errors/grid_validation_error.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        End at `+2m30s` is not divisible by one-minute frame.
    Raises:
        AssertionError: If `GridValidationError` is not raised.
    Side Effects:
        None.
    """
    feed = MarketDataCandleFeed(canonical_candle_reader=_CanonicalReaderStub(rows=()))
    with pytest.raises(GridValidationError, match=r"align to timeframe 1m"):
        feed.load_1m_dense(
            market_id=_MARKET_ID,
            symbol=_SYMBOL,
            time_range=_time_range(end_minute_offset=2, end_second=30),
        )


def test_market_data_acl_builds_dense_grid_with_nan_holes_and_contiguous_float32() -> None:
    """
    Verify dense `1m` materialization with `NaN` holes and contiguous dtype contracts.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related: src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Missing minutes remain `NaN` without interpolation.
    Raises:
        AssertionError: If timeline, dtypes, contiguity, or hole semantics are broken.
    Side Effects:
        None.
    """
    reader = _CanonicalReaderStub(
        rows=(
            _row(minute_offset=0, open_value=100.0, volume=10.0),
            _row(minute_offset=3, open_value=103.0, volume=13.0),
        )
    )
    feed = MarketDataCandleFeed(canonical_candle_reader=reader)
    candles = feed.load_1m_dense(
        market_id=_MARKET_ID,
        symbol=_SYMBOL,
        time_range=_time_range(end_minute_offset=5),
    )

    expected_ts = np.array(
        [
            _epoch_ms(_utc_ts(minute_offset=0)),
            _epoch_ms(_utc_ts(minute_offset=1)),
            _epoch_ms(_utc_ts(minute_offset=2)),
            _epoch_ms(_utc_ts(minute_offset=3)),
            _epoch_ms(_utc_ts(minute_offset=4)),
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(candles.ts_open, expected_ts)
    np.testing.assert_allclose(candles.open[[0, 3]], np.array([100.0, 103.0], dtype=np.float32))
    np.testing.assert_allclose(candles.volume[[0, 3]], np.array([10.0, 13.0], dtype=np.float32))
    assert np.isnan(candles.open[1])
    assert np.isnan(candles.high[1])
    assert np.isnan(candles.low[1])
    assert np.isnan(candles.close[1])
    assert np.isnan(candles.volume[1])

    assert candles.timeframe.code == "1m"
    assert candles.ts_open.dtype == np.int64
    assert candles.open.dtype == np.float32
    assert candles.high.dtype == np.float32
    assert candles.low.dtype == np.float32
    assert candles.close.dtype == np.float32
    assert candles.volume.dtype == np.float32
    assert candles.ts_open.flags["C_CONTIGUOUS"]
    assert candles.open.flags["C_CONTIGUOUS"]
    assert candles.high.flags["C_CONTIGUOUS"]
    assert candles.low.flags["C_CONTIGUOUS"]
    assert candles.close.flags["C_CONTIGUOUS"]
    assert candles.volume.flags["C_CONTIGUOUS"]

    assert reader.last_instrument_id == InstrumentId(market_id=_MARKET_ID, symbol=_SYMBOL)
    assert reader.last_time_range == _time_range(end_minute_offset=5)


def test_market_data_acl_is_order_independent_for_unsorted_input() -> None:
    """
    Verify unsorted sparse input yields deterministic dense output.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stable sorting by `ts_open` is applied before materialization.
    Raises:
        AssertionError: If output changes when input order changes.
    Side Effects:
        None.
    """
    rows_unsorted = (
        _row(minute_offset=2, open_value=102.0),
        _row(minute_offset=0, open_value=100.0),
        _row(minute_offset=1, open_value=101.0),
    )
    rows_sorted = (
        _row(minute_offset=0, open_value=100.0),
        _row(minute_offset=1, open_value=101.0),
        _row(minute_offset=2, open_value=102.0),
    )
    time_range = _time_range(end_minute_offset=3)

    out_unsorted = MarketDataCandleFeed(
        canonical_candle_reader=_CanonicalReaderStub(rows=rows_unsorted)
    ).load_1m_dense(
        market_id=_MARKET_ID,
        symbol=_SYMBOL,
        time_range=time_range,
    )
    out_sorted = MarketDataCandleFeed(
        canonical_candle_reader=_CanonicalReaderStub(rows=rows_sorted)
    ).load_1m_dense(
        market_id=_MARKET_ID,
        symbol=_SYMBOL,
        time_range=time_range,
    )

    np.testing.assert_array_equal(out_unsorted.ts_open, out_sorted.ts_open)
    np.testing.assert_allclose(out_unsorted.open, out_sorted.open, equal_nan=True)
    np.testing.assert_allclose(out_unsorted.high, out_sorted.high, equal_nan=True)
    np.testing.assert_allclose(out_unsorted.low, out_sorted.low, equal_nan=True)
    np.testing.assert_allclose(out_unsorted.close, out_sorted.close, equal_nan=True)
    np.testing.assert_allclose(out_unsorted.volume, out_sorted.volume, equal_nan=True)


def test_market_data_acl_applies_last_wins_for_duplicate_timestamps() -> None:
    """
    Verify duplicate timestamp policy is deterministic `last-wins`.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stable sort keeps duplicate relative order before final vectorized assignment.
    Raises:
        AssertionError: If duplicate resolution does not keep the last value.
    Side Effects:
        None.
    """
    reader = _CanonicalReaderStub(
        rows=(
            _row(minute_offset=0, open_value=100.0, volume=10.0),
            _row(minute_offset=1, open_value=101.0, volume=11.0),
            _row(minute_offset=1, open_value=111.0, volume=21.0),
        )
    )
    candles = MarketDataCandleFeed(canonical_candle_reader=reader).load_1m_dense(
        market_id=_MARKET_ID,
        symbol=_SYMBOL,
        time_range=_time_range(end_minute_offset=3),
    )

    assert candles.open[1] == np.float32(111.0)
    assert candles.high[1] == np.float32(113.0)
    assert candles.low[1] == np.float32(109.0)
    assert candles.close[1] == np.float32(112.0)
    assert candles.volume[1] == np.float32(21.0)


def test_market_data_acl_ignores_out_of_range_candles() -> None:
    """
    Verify candles outside `[start, end)` are deterministically `ignore`-ed.

    Docs: docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
    Related:
      src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Candles with `ts_open < start` or `ts_open >= end` must not affect output arrays.
    Raises:
        AssertionError: If out-of-range rows modify dense vectors.
    Side Effects:
        None.
    """
    reader = _CanonicalReaderStub(
        rows=(
            _row(minute_offset=-1, open_value=50.0),
            _row(minute_offset=0, open_value=100.0),
            _row(minute_offset=2, open_value=102.0),
            _row(minute_offset=3, open_value=999.0),
        )
    )
    candles = MarketDataCandleFeed(canonical_candle_reader=reader).load_1m_dense(
        market_id=_MARKET_ID,
        symbol=_SYMBOL,
        time_range=_time_range(end_minute_offset=3),
    )

    assert candles.open.shape[0] == 3
    assert candles.open[0] == np.float32(100.0)
    assert np.isnan(candles.open[1])
    assert candles.open[2] == np.float32(102.0)
