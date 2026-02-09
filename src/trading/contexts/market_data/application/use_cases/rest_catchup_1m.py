from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence  # noqa: F401
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
    DailyTsOpenCount,  # noqa: F401
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.use_cases.time_slicing import (
    slice_time_range_by_utc_days,
)
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class RestCatchUp1mReport:
    # tail
    tail_start: UtcTimestamp | None
    tail_end: UtcTimestamp | None
    tail_rows_read: int
    tail_rows_written: int
    tail_batches: int

    # gaps
    gap_scan_start: UtcTimestamp | None
    gap_scan_end: UtcTimestamp | None
    gap_days_scanned: int
    gap_days_with_gaps: int
    gap_ranges_filled: int
    gap_rows_read: int
    gap_rows_written: int
    gap_rows_skipped_existing: int
    gap_batches: int

    def to_dict(self) -> dict[str, object]:
        """
        Serialize report into JSON-friendly primitives for CLI/notebooks/logging.

        Parameters:
        - None.

        Returns:
        - Dictionary with only JSON-serializable primitives.

        Assumptions/Invariants:
        - Timestamp fields are `UtcTimestamp | None`.
        - Numeric counters are plain integers.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return {
            "tail_start": _ts_to_iso(self.tail_start),
            "tail_end": _ts_to_iso(self.tail_end),
            "tail_rows_read": self.tail_rows_read,
            "tail_rows_written": self.tail_rows_written,
            "tail_batches": self.tail_batches,
            "gap_scan_start": _ts_to_iso(self.gap_scan_start),
            "gap_scan_end": _ts_to_iso(self.gap_scan_end),
            "gap_days_scanned": self.gap_days_scanned,
            "gap_days_with_gaps": self.gap_days_with_gaps,
            "gap_ranges_filled": self.gap_ranges_filled,
            "gap_rows_read": self.gap_rows_read,
            "gap_rows_written": self.gap_rows_written,
            "gap_rows_skipped_existing": self.gap_rows_skipped_existing,
            "gap_batches": self.gap_batches,
        }


class RestCatchUp1mUseCase:
    """
    REST catch-up 1m:

    1) tail догоняем до "закрытой" минуты:
       start = last_closed_ts_open + 1m (из canonical index)
       end = floor(now to minute)

    2) gap-fill: ищем пропуски по всей истории, НО в одном запуске НЕ лезем в tail-range,
       чтобы не писать повторно то, что только что догнали.
       gap-scan range = [bounds.first, tail_start)

    Запись — через RawKlineWriter (в raw_*), canonical строится MV.
    """

    def __init__(
        self,
        *,
        index: CanonicalCandleIndexReader,
        source: CandleIngestSource,
        writer: RawKlineWriter,
        clock: Clock,
        max_days_per_insert: int,
        batch_size: int,
        ingest_id: UUID,
    ) -> None:
        if index is None:  # type: ignore[truthy-bool]
            raise ValueError("RestCatchUp1mUseCase requires index")
        if source is None:  # type: ignore[truthy-bool]
            raise ValueError("RestCatchUp1mUseCase requires source")
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("RestCatchUp1mUseCase requires writer")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RestCatchUp1mUseCase requires clock")
        if max_days_per_insert <= 0 or max_days_per_insert > 7:
            raise ValueError("max_days_per_insert must be in [1..7]")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self._index = index
        self._source = source
        self._writer = writer
        self._clock = clock
        self._max_days = max_days_per_insert
        self._batch_size = batch_size
        self._ingest_id = ingest_id

    def run(self, instrument_id: InstrumentId) -> RestCatchUp1mReport:
        """
        Execute tail catch-up and historical gap fill for one instrument.

        Parameters:
        - instrument_id: instrument for which catch-up and gap repair should be performed.

        Returns:
        - Aggregated report with tail and gap-fill counters.

        Assumptions/Invariants:
        - Time ranges follow half-open semantics `[start, end)`.
        - Tail and gap ranges operate on minute buckets in UTC.

        Errors/Exceptions:
        - Raises `ValueError` if canonical seed candles are absent.

        Side effects:
        - Reads canonical index and writes rows into raw storage through writer port.
        """
        end_dt = _floor_to_minute_utc(self._clock.now().value)
        end = UtcTimestamp(end_dt)

        last = self._index.max_ts_open_lt(instrument_id=instrument_id, before=end)
        if last is None:
            raise ValueError(
                f"No canonical candles for {instrument_id}. "
                "Run initial backfill first (or seed history), then rest-catchup can maintain tail/gaps."  # noqa: E501
            )

        tail_start_dt = _floor_to_minute_utc(last.value) + timedelta(minutes=1)
        tail_start = UtcTimestamp(tail_start_dt)

        tail_rows_read = tail_rows_written = tail_batches = 0
        if tail_start.value < end.value:
            tr = TimeRange(start=tail_start, end=end)
            tail_rows_read, tail_rows_written, tail_batches, _ = self._ingest_time_range(
                instrument_id=instrument_id,
                time_range=tr,
            )

        # gap-scan: [bounds.first, tail_start)  (НЕ включаем tail, чтобы не писать дубль)
        gap_scan_start = gap_scan_end = None
        gap_days_scanned = gap_days_with_gaps = gap_ranges_filled = 0
        gap_rows_read = gap_rows_written = gap_rows_skipped_existing = gap_batches = 0

        bounds = self._index.bounds(instrument_id)
        if bounds is not None:
            b_first, _b_last = bounds
            scan_start_dt = _floor_to_minute_utc(b_first.value)
            scan_end_dt = _floor_to_minute_utc(tail_start.value)

            if scan_start_dt < scan_end_dt:
                gap_scan_start = UtcTimestamp(scan_start_dt)
                gap_scan_end = UtcTimestamp(scan_end_dt)

                (
                    gap_days_scanned,
                    gap_days_with_gaps,
                    gap_ranges_filled,
                    gap_rows_read,
                    gap_rows_written,
                    gap_rows_skipped_existing,
                    gap_batches,
                ) = self._fill_gaps_in_range(
                    instrument_id=instrument_id,
                    time_range=TimeRange(start=gap_scan_start, end=gap_scan_end),
                )

        return RestCatchUp1mReport(
            tail_start=tail_start,
            tail_end=end,
            tail_rows_read=tail_rows_read,
            tail_rows_written=tail_rows_written,
            tail_batches=tail_batches,
            gap_scan_start=gap_scan_start,
            gap_scan_end=gap_scan_end,
            gap_days_scanned=gap_days_scanned,
            gap_days_with_gaps=gap_days_with_gaps,
            gap_ranges_filled=gap_ranges_filled,
            gap_rows_read=gap_rows_read,
            gap_rows_written=gap_rows_written,
            gap_rows_skipped_existing=gap_rows_skipped_existing,
            gap_batches=gap_batches,
        )

    def _ingest_time_range(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        dedup_existing: bool = False,
    ) -> tuple[int, int, int, int]:
        """
        Ingest one time range into raw storage with optional pre-write dedup filtering.

        Parameters:
        - instrument_id: instrument being ingested.
        - time_range: UTC half-open range `[start, end)` to ingest.
        - dedup_existing: when true, skip rows whose minute already exists in canonical index.

        Returns:
        - Tuple `(rows_read, rows_written, batches_written, rows_skipped_existing)`.

        Assumptions/Invariants:
        - Source emits 1m closed candles for requested range.
        - Writer performs append-only writes to raw tables.

        Errors/Exceptions:
        - Propagates source/index/writer errors.

        Side effects:
        - Reads source and canonical index, writes to raw storage.
        """
        rows_read = rows_written = rows_skipped_existing = batches = 0

        for chunk in slice_time_range_by_utc_days(time_range=time_range, max_days=self._max_days):
            existing_minute_keys: set[int] = set()
            if dedup_existing:
                existing_minute_keys = self._existing_minute_keys(
                    instrument_id=instrument_id,
                    time_range=chunk,
                )

            batch: list[CandleWithMeta] = []

            for row in self._source.stream_1m(instrument_id, chunk):
                fixed = CandleWithMeta(
                    candle=row.candle,
                    meta=row.meta.__class__(  # CandleMeta (dataclass)
                        source=row.meta.source,
                        ingested_at=row.meta.ingested_at,
                        ingest_id=self._ingest_id,
                        instrument_key=row.meta.instrument_key,
                        trades_count=row.meta.trades_count,
                        taker_buy_volume_base=row.meta.taker_buy_volume_base,
                        taker_buy_volume_quote=row.meta.taker_buy_volume_quote,
                    ),
                )

                rows_read += 1
                minute_key = _minute_key(fixed.candle.ts_open.value)

                if dedup_existing and minute_key in existing_minute_keys:
                    rows_skipped_existing += 1
                    continue

                if dedup_existing:
                    existing_minute_keys.add(minute_key)

                batch.append(fixed)

                if len(batch) >= self._batch_size:
                    self._writer.write_1m(batch)
                    rows_written += len(batch)
                    batches += 1
                    batch = []

            if batch:
                self._writer.write_1m(batch)
                rows_written += len(batch)
                batches += 1

        return rows_read, rows_written, batches, rows_skipped_existing

    def _existing_minute_keys(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> set[int]:
        """
        Load existing canonical minute keys for one instrument and one range.

        Parameters:
        - instrument_id: instrument id used in canonical index query.
        - time_range: UTC half-open range `[start, end)` for key lookup.

        Returns:
        - Set of integer minute keys (`epoch_seconds // 60`).

        Assumptions/Invariants:
        - Index returns UTC timestamps for existing candles.

        Errors/Exceptions:
        - Propagates index reader errors.

        Side effects:
        - Reads canonical index storage through port.
        """
        existing = self._index.distinct_ts_opens(
            instrument_id=instrument_id,
            time_range=time_range,
        )
        return {_minute_key(v.value) for v in existing}

    def _fill_gaps_in_range(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> tuple[int, int, int, int, int, int, int]:
        """
        Detect missing minute windows and ingest only those windows for one historical range.

        Parameters:
        - instrument_id: instrument for which gaps are repaired.
        - time_range: historical scan range in UTC half-open semantics `[start, end)`.

        Returns:
        - Tuple with counters:
          `(days_scanned, days_with_gaps, ranges_filled, rows_read, rows_written,
          rows_skipped_existing, batches)`.

        Assumptions/Invariants:
        - Day-level counts and distinct timestamps come from canonical index.
        - Missing-range reconstruction operates on minute buckets.

        Errors/Exceptions:
        - Propagates index/source/writer errors.

        Side effects:
        - Reads canonical index and writes repaired rows to raw storage.
        """
        # day -> count (IMPORTANT: day is datetime.date from ClickHouse)
        day_counts = self._index.daily_counts(instrument_id=instrument_id, time_range=time_range)
        counts_map = {r.day: int(r.count) for r in day_counts}

        scan_start = _floor_to_minute_utc(time_range.start.value)
        scan_end = _floor_to_minute_utc(time_range.end.value)

        day_cursor = scan_start.replace(hour=0, minute=0, second=0, microsecond=0)

        days_scanned = days_with_gaps = ranges_filled = 0
        rows_read = rows_written = rows_skipped_existing = batches = 0

        while day_cursor < scan_end:
            next_day = day_cursor + timedelta(days=1)

            day_start = max(day_cursor, scan_start)
            day_end = min(next_day, scan_end)

            expected = _minute_key(day_end) - _minute_key(day_start)
            if expected <= 0:
                day_cursor = next_day
                continue

            days_scanned += 1
            actual = counts_map.get(day_cursor.date(), 0)

            if actual >= expected:
                day_cursor = next_day
                continue

            days_with_gaps += 1

            existing = self._index.distinct_ts_opens(
                instrument_id=instrument_id,
                time_range=TimeRange(start=UtcTimestamp(day_start), end=UtcTimestamp(day_end)),
            )
            missing_ranges = _missing_ranges_for_day(existing=existing, start=day_start, end=day_end)  # noqa: E501

            for mr in missing_ranges:
                rr, rw, bb, rs = self._ingest_time_range(
                    instrument_id=instrument_id,
                    time_range=mr,
                    dedup_existing=True,
                )
                rows_read += rr
                rows_written += rw
                rows_skipped_existing += rs
                batches += bb
                ranges_filled += 1

            day_cursor = next_day

        return (
            days_scanned,
            days_with_gaps,
            ranges_filled,
            rows_read,
            rows_written,
            rows_skipped_existing,
            batches,
        )


def _missing_ranges_for_day(
    *,
    existing: Sequence[UtcTimestamp],
    start: datetime,
    end: datetime,
) -> list[TimeRange]:
    """
    Reconstruct missing minute ranges from existing canonical timestamps.

    Parameters:
    - existing: distinct canonical `ts_open` values observed inside `[start, end)`.
    - start: day-window start in UTC.
    - end: day-window end in UTC.

    Returns:
    - List of missing sub-ranges in half-open semantics `[start, end)`.

    Assumptions/Invariants:
    - Missing detection is minute-based and ignores second/millisecond noise.
    - Returned ranges align to exact minute boundaries.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    start_key = _minute_key(start)
    end_key = _minute_key(end)
    if end_key <= start_key:
        return []

    existing_set = {_minute_key(e.value) for e in existing}
    out: list[TimeRange] = []
    missing_start_key: int | None = None

    for minute_key in range(start_key, end_key):
        if minute_key not in existing_set:
            if missing_start_key is None:
                missing_start_key = minute_key
            continue

        if missing_start_key is not None:
            out.append(
                TimeRange(
                    start=UtcTimestamp(_minute_start_from_key(missing_start_key)),
                    end=UtcTimestamp(_minute_start_from_key(minute_key)),
                )
            )
            missing_start_key = None

    if missing_start_key is not None:
        out.append(
            TimeRange(
                start=UtcTimestamp(_minute_start_from_key(missing_start_key)),
                end=UtcTimestamp(_minute_start_from_key(end_key)),
            )
        )

    return out


def _minute_key(dt: datetime) -> int:
    """
    Convert datetime into a stable minute-level integer key in UTC.

    Parameters:
    - dt: timestamp to convert.

    Returns:
    - Integer minute key (`epoch_seconds // 60`).

    Assumptions/Invariants:
    - Input can be naive or aware; naive is treated as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    dt_utc = _ensure_tz_utc(dt)
    return int(dt_utc.timestamp() // 60)


def _minute_start_from_key(key: int) -> datetime:
    """
    Convert minute key back to UTC minute-start datetime.

    Parameters:
    - key: minute key produced by `_minute_key(...)`.

    Returns:
    - UTC datetime aligned to minute start (`...:ss=00,us=000000`).

    Assumptions/Invariants:
    - `key` represents minutes since Unix epoch.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return datetime.fromtimestamp(key * 60, tz=timezone.utc)


def _floor_to_minute_utc(dt: datetime) -> datetime:
    """
    Floor datetime to minute precision in UTC.

    Parameters:
    - dt: timestamp to normalize.

    Returns:
    - UTC datetime rounded down to the minute.

    Assumptions/Invariants:
    - Naive input is interpreted as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    dt_utc = _ensure_tz_utc(dt)
    return dt_utc.replace(second=0, microsecond=0)


def _ensure_tz_utc(dt: datetime) -> datetime:
    """
    Normalize datetime to timezone-aware UTC.

    Parameters:
    - dt: datetime value from external adapters or storage.

    Returns:
    - Timezone-aware UTC datetime.

    Assumptions/Invariants:
    - Naive datetime values are interpreted as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ts_to_iso(value: UtcTimestamp | None) -> str | None:
    """
    Convert optional UTC timestamp wrapper into ISO-8601 string.

    Parameters:
    - value: timestamp value or `None`.

    Returns:
    - ISO-8601 UTC string with `Z` suffix when value is present, otherwise `None`.

    Assumptions/Invariants:
    - `UtcTimestamp.__str__` already returns canonical UTC ISO representation.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None:
        return None
    return str(value)
