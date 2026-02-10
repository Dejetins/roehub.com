from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Iterator, Sequence
from uuid import UUID

import pytest

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    DailyTsOpenCount,
)
from trading.contexts.market_data.application.use_cases.rest_catchup_1m import (
    RestCatchUp1mReport,
    RestCatchUp1mUseCase,
    _missing_ranges_for_day,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class FixedClock:
    """Clock fake that always returns one predefined timestamp."""

    def __init__(self, now_ts: UtcTimestamp) -> None:
        self._now_ts = now_ts

    def now(self) -> UtcTimestamp:
        """
        Return current test timestamp.

        Parameters:
        - None.

        Returns:
        - Fixed UTC timestamp configured at initialization.

        Assumptions/Invariants:
        - Timestamp is already normalized by `UtcTimestamp`.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return self._now_ts


class FakeIndex:
    """In-memory canonical index fake with configurable distinct-ts behavior."""

    def __init__(
        self,
        *,
        last: UtcTimestamp | None,
        bounds_value: tuple[UtcTimestamp, UtcTimestamp] | None,
        daily: Sequence[DailyTsOpenCount] | None = None,
        distinct: Sequence[UtcTimestamp] | None = None,
        distinct_fn: Callable[[TimeRange], Sequence[UtcTimestamp]] | None = None,
    ) -> None:
        self._last = last
        self._bounds = bounds_value
        self._daily = list(daily or [])
        self._distinct = list(distinct or [])
        self._distinct_fn = distinct_fn
        self.distinct_calls: list[TimeRange] = []

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Return configured canonical bounds.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).

        Returns:
        - Tuple `(first_ts_open, last_ts_open)` or `None`.

        Assumptions/Invariants:
        - Bounds are preconfigured by test setup.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = instrument_id
        return self._bounds

    def max_ts_open_lt(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> UtcTimestamp | None:
        """
        Return preconfigured latest timestamp below provided bound.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - before: exclusive upper bound (unused in this fake).

        Returns:
        - Configured timestamp or `None`.

        Assumptions/Invariants:
        - Value is controlled by test fixture.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = instrument_id
        _ = before
        return self._last

    def daily_counts(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[DailyTsOpenCount]:
        """
        Return preconfigured day-level counts.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - time_range: scan range (unused in this fake).

        Returns:
        - Sequence of day/count rows.

        Assumptions/Invariants:
        - Returned data is immutable from caller perspective.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = instrument_id
        _ = time_range
        return list(self._daily)

    def distinct_ts_opens(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[UtcTimestamp]:
        """
        Return distinct timestamps for requested range.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - time_range: requested lookup range.

        Returns:
        - Sequence of distinct timestamp wrappers.

        Assumptions/Invariants:
        - If callback is provided, it defines per-range behavior.

        Errors/Exceptions:
        - None.

        Side effects:
        - Stores each call range in `distinct_calls`.
        """
        _ = instrument_id
        self.distinct_calls.append(time_range)
        if self._distinct_fn is not None:
            return list(self._distinct_fn(time_range))
        return list(self._distinct)


class FakeSource:
    """Source fake yielding rows that fall into the requested range."""

    def __init__(self, rows: Iterable[CandleWithMeta]) -> None:
        self._rows = list(rows)
        self.calls: list[TimeRange] = []

    def stream_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Yield only rows matching instrument and half-open range boundaries.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested half-open range `[start, end)`.

        Returns:
        - Iterator of filtered rows.

        Assumptions/Invariants:
        - Rows are immutable DTOs prebuilt by tests.

        Errors/Exceptions:
        - None.

        Side effects:
        - Records each call range in `calls`.
        """
        self.calls.append(time_range)
        for row in self._rows:
            if row.candle.instrument_id != instrument_id:
                continue
            ts = row.candle.ts_open.value
            if time_range.start.value <= ts < time_range.end.value:
                yield row


class RecordingWriter:
    """Writer fake recording every insert batch."""

    def __init__(self) -> None:
        self.calls: list[list[CandleWithMeta]] = []

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        """
        Capture write payload for assertions.

        Parameters:
        - rows: iterable batch to store.

        Returns:
        - None.

        Assumptions/Invariants:
        - `rows` can be materialized into list exactly once.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends one materialized batch into `calls`.
        """
        self.calls.append(list(rows))


def _ts(dt: datetime) -> UtcTimestamp:
    """
    Wrap timezone-aware datetime for concise test setup.

    Parameters:
    - dt: timezone-aware datetime.

    Returns:
    - UTC timestamp wrapper.

    Assumptions/Invariants:
    - Input datetime is timezone-aware.

    Errors/Exceptions:
    - Propagates `UtcTimestamp` validation errors.

    Side effects:
    - None.
    """
    return UtcTimestamp(dt)


def _mk_row(instrument_id: InstrumentId, ts_open: datetime) -> CandleWithMeta:
    """
    Create deterministic candle row for use-case tests.

    Parameters:
    - instrument_id: target instrument id.
    - ts_open: minute open timestamp.

    Returns:
    - Candle-with-meta DTO for test streams.

    Assumptions/Invariants:
    - `ts_open` is UTC-aware and minute-aligned in tests.

    Errors/Exceptions:
    - Propagates primitive validation errors.

    Side effects:
    - None.
    """
    ts_open_u = _ts(ts_open)
    ts_close_u = _ts(ts_open + timedelta(minutes=1))

    candle = Candle(
        instrument_id=instrument_id,
        ts_open=ts_open_u,
        ts_close=ts_close_u,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume_base=1.0,
        volume_quote=100.5,
    )
    meta = CandleMeta(
        source="rest",
        ingested_at=_ts(datetime(2026, 2, 5, 12, 5, tzinfo=timezone.utc)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=7,
        taker_buy_volume_base=0.3,
        taker_buy_volume_quote=30.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def _minute_key(dt: datetime) -> int:
    """
    Build minute key used by fake index routing helpers.

    Parameters:
    - dt: datetime to map.

    Returns:
    - Integer minute key (`epoch_seconds // 60`).

    Assumptions/Invariants:
    - Naive timestamps are not expected in tests.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return int(dt.timestamp() // 60)


def test_rest_catchup_report_to_dict_is_json_serializable() -> None:
    """
    Ensure report DTO converts timestamp fields into JSON-safe primitives.

    Parameters:
    - None.

    Returns:
    - None.
    """
    report = RestCatchUp1mReport(
        tail_start=_ts(datetime(2026, 2, 5, 12, 4, tzinfo=timezone.utc)),
        tail_end=_ts(datetime(2026, 2, 5, 12, 5, tzinfo=timezone.utc)),
        tail_rows_read=1,
        tail_rows_written=1,
        tail_batches=1,
        gap_scan_start=_ts(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)),
        gap_scan_end=_ts(datetime(2026, 2, 5, 12, 4, tzinfo=timezone.utc)),
        gap_days_scanned=4,
        gap_days_with_gaps=1,
        gap_ranges_filled=1,
        gap_rows_read=3,
        gap_rows_written=3,
        gap_rows_skipped_existing=0,
        gap_batches=1,
    )

    payload = report.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["tail_start"] == "2026-02-05T12:04:00.000Z"
    assert decoded["tail_end"] == "2026-02-05T12:05:00.000Z"
    assert decoded["gap_rows_written"] == 3
    assert decoded["gap_rows_skipped_existing"] == 0


def test_rest_catchup_run_returns_json_friendly_report() -> None:
    """
    Verify tail-only run remains correct and JSON serialization stays stable.

    Parameters:
    - None.

    Returns:
    - None.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    now = _ts(datetime(2026, 2, 5, 12, 5, 30, tzinfo=timezone.utc))

    index = FakeIndex(
        last=_ts(datetime(2026, 2, 5, 12, 3, tzinfo=timezone.utc)),
        bounds_value=None,
        daily=[DailyTsOpenCount(day=date(2026, 2, 5), count=1)],
    )
    source = FakeSource([_mk_row(instrument_id, datetime(2026, 2, 5, 12, 4, tzinfo=timezone.utc))])
    writer = RecordingWriter()

    uc = RestCatchUp1mUseCase(
        index=index,
        source=source,
        writer=writer,
        clock=FixedClock(now),
        max_days_per_insert=1,
        batch_size=100,
        ingest_id=UUID("00000000-0000-0000-0000-000000000042"),
    )

    report = uc.run(instrument_id)
    payload = report.to_dict()

    assert payload["tail_start"] == "2026-02-05T12:04:00.000Z"
    assert payload["tail_end"] == "2026-02-05T12:05:00.000Z"
    assert payload["tail_rows_written"] == 1
    assert payload["gap_rows_written"] == 0
    assert payload["gap_rows_skipped_existing"] == 0
    assert len(writer.calls) == 1
    assert writer.calls[0][0].meta.ingest_id == UUID("00000000-0000-0000-0000-000000000042")
    json.dumps(payload, ensure_ascii=False)


def test_rest_catchup_run_raises_without_canonical_seed() -> None:
    """
    Ensure catch-up fails fast when canonical history is absent.

    Parameters:
    - None.

    Returns:
    - None.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    uc = RestCatchUp1mUseCase(
        index=FakeIndex(last=None, bounds_value=None),
        source=FakeSource([]),
        writer=RecordingWriter(),
        clock=FixedClock(_ts(datetime(2026, 2, 5, 12, 5, tzinfo=timezone.utc))),
        max_days_per_insert=1,
        batch_size=100,
        ingest_id=UUID("00000000-0000-0000-0000-000000000042"),
    )

    with pytest.raises(ValueError, match="Run initial backfill first"):
        uc.run(instrument_id)


def test_gap_fill_includes_absent_days_from_daily_counts() -> None:
    """
    Ensure days absent in day-counts map are treated as zero-candle gaps and filled.

    Parameters:
    - None.

    Returns:
    - None.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    d1 = date(2026, 1, 1)
    d3 = date(2026, 1, 3)
    jan2_open = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    jan2_end = datetime(2026, 1, 3, 0, 0, tzinfo=timezone.utc)

    def distinct_for_range(time_range: TimeRange) -> Sequence[UtcTimestamp]:
        key = (_minute_key(time_range.start.value), _minute_key(time_range.end.value))
        jan2_key = (_minute_key(jan2_open), _minute_key(jan2_end))
        if key == jan2_key:
            return []
        return []

    index = FakeIndex(
        last=_ts(datetime(2026, 1, 3, 0, 0, tzinfo=timezone.utc)),
        bounds_value=(
            _ts(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
            _ts(datetime(2026, 1, 3, 0, 0, tzinfo=timezone.utc)),
        ),
        daily=[
            DailyTsOpenCount(day=d1, count=1440),
            DailyTsOpenCount(day=d3, count=1),
            # 2026-01-02 intentionally missing in daily_counts -> should be interpreted as 0.
        ],
        distinct_fn=distinct_for_range,
    )
    source = FakeSource([_mk_row(instrument_id, jan2_open)])
    writer = RecordingWriter()

    uc = RestCatchUp1mUseCase(
        index=index,
        source=source,
        writer=writer,
        clock=FixedClock(_ts(datetime(2026, 1, 3, 0, 1, 30, tzinfo=timezone.utc))),
        max_days_per_insert=1,
        batch_size=100,
        ingest_id=UUID("00000000-0000-0000-0000-000000000055"),
    )

    report = uc.run(instrument_id)

    assert report.gap_days_scanned >= 2
    assert report.gap_days_with_gaps >= 1
    assert report.gap_ranges_filled >= 1
    assert report.gap_rows_written == 1
    assert len(writer.calls) == 1
    assert writer.calls[0][0].candle.ts_open.value == jan2_open


def test_gap_fill_does_not_write_existing_minutes() -> None:
    """
    Regression: gap fill must skip rows that already exist in canonical minute index.

    Parameters:
    - None.

    Returns:
    - None.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    m00 = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    m01 = datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    m02 = datetime(2026, 1, 2, 0, 2, tzinfo=timezone.utc)
    m03 = datetime(2026, 1, 2, 0, 3, tzinfo=timezone.utc)

    day_range_key = (_minute_key(m00), _minute_key(m03))
    chunk_range_key = (_minute_key(m01), _minute_key(m03))

    def distinct_for_range(time_range: TimeRange) -> Sequence[UtcTimestamp]:
        key = (_minute_key(time_range.start.value), _minute_key(time_range.end.value))
        if key == day_range_key:
            # Imperfect day scan: thinks only 00:00 exists, so [00:01,00:03) is "missing".
            return [_ts(m00)]
        if key == chunk_range_key:
            # Defensive chunk lookup reveals all candidate minutes already exist.
            return [_ts(m01), _ts(m02)]
        return []

    index = FakeIndex(
        last=_ts(m02),
        bounds_value=(_ts(m00), _ts(m02)),
        daily=[DailyTsOpenCount(day=date(2026, 1, 2), count=1)],
        distinct_fn=distinct_for_range,
    )
    source = FakeSource([_mk_row(instrument_id, m01), _mk_row(instrument_id, m02)])
    writer = RecordingWriter()

    uc = RestCatchUp1mUseCase(
        index=index,
        source=source,
        writer=writer,
        clock=FixedClock(_ts(datetime(2026, 1, 2, 0, 3, 30, tzinfo=timezone.utc))),
        max_days_per_insert=1,
        batch_size=100,
        ingest_id=UUID("00000000-0000-0000-0000-000000000099"),
    )

    report = uc.run(instrument_id)

    assert report.gap_ranges_filled == 1
    assert report.gap_rows_read == 2
    assert report.gap_rows_written == 0
    assert report.gap_rows_skipped_existing == 2
    assert writer.calls == []


def test_gap_fill_writes_only_missing_minutes() -> None:
    """
    Gap fill must write only truly missing minutes from a reconstructed missing range.

    Parameters:
    - None.

    Returns:
    - None.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    m00 = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    m01 = datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    m02 = datetime(2026, 1, 2, 0, 2, tzinfo=timezone.utc)
    m03 = datetime(2026, 1, 2, 0, 3, tzinfo=timezone.utc)

    day_range_key = (_minute_key(m00), _minute_key(m03))
    chunk_range_key = (_minute_key(m01), _minute_key(m03))

    def distinct_for_range(time_range: TimeRange) -> Sequence[UtcTimestamp]:
        key = (_minute_key(time_range.start.value), _minute_key(time_range.end.value))
        if key == day_range_key:
            return [_ts(m00)]
        if key == chunk_range_key:
            # Only 00:01 exists; 00:02 remains truly missing.
            return [_ts(m01)]
        return []

    index = FakeIndex(
        last=_ts(m02),
        bounds_value=(_ts(m00), _ts(m02)),
        daily=[DailyTsOpenCount(day=date(2026, 1, 2), count=1)],
        distinct_fn=distinct_for_range,
    )
    source = FakeSource([_mk_row(instrument_id, m01), _mk_row(instrument_id, m02)])
    writer = RecordingWriter()

    uc = RestCatchUp1mUseCase(
        index=index,
        source=source,
        writer=writer,
        clock=FixedClock(_ts(datetime(2026, 1, 2, 0, 3, 30, tzinfo=timezone.utc))),
        max_days_per_insert=1,
        batch_size=100,
        ingest_id=UUID("00000000-0000-0000-0000-000000000099"),
    )

    report = uc.run(instrument_id)

    assert report.gap_ranges_filled == 1
    assert report.gap_rows_read == 2
    assert report.gap_rows_written == 1
    assert report.gap_rows_skipped_existing == 1
    assert len(writer.calls) == 1
    assert len(writer.calls[0]) == 1
    assert writer.calls[0][0].candle.ts_open.value == m02


def test_missing_ranges_for_day_uses_minute_buckets_for_membership() -> None:
    """
    Minute-level matching must treat `00:01:00.500` as present minute `00:01`.

    Parameters:
    - None.

    Returns:
    - None.
    """
    start = datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 0, 4, tzinfo=timezone.utc)
    existing = [_ts(datetime(2026, 1, 2, 0, 1, 0, 500000, tzinfo=timezone.utc))]

    missing = _missing_ranges_for_day(existing=existing, start=start, end=end)

    assert len(missing) == 1
    assert missing[0].start.value == datetime(2026, 1, 2, 0, 2, tzinfo=timezone.utc)
    assert missing[0].end.value == datetime(2026, 1, 2, 0, 4, tzinfo=timezone.utc)
