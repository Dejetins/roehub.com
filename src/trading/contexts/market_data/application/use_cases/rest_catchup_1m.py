from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol, Sequence
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.use_cases.time_slicing import (
    slice_time_range_by_utc_days,
)
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp
from trading.shared_kernel.primitives.candle_meta import CandleMeta


@dataclass(frozen=True, slots=True)
class RestCatchUp1mReport:
    instrument: str
    tail_start: str
    tail_end: str
    tail_rows_read: int
    tail_rows_written: int
    gap_rows_read: int
    gap_rows_written: int
    batches_written: int


class CanonicalCandleIndexReader(Protocol):
    """
    Минимальный контракт индекса по canonical, который нужен rest-catchup.

    Реальная реализация: ClickHouseCanonicalCandleIndexReader.
    """

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        ...

    def max_ts_open_lt(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> UtcTimestamp | None:
        ...

    def daily_counts(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[object]:
        ...

    def distinct_ts_opens(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[UtcTimestamp]:
        ...



class RestCatchUp1mUseCase:
    """
    REST догонка 1m:
    1) tail: [last_closed_ts_open + 1m, floor(now, 1m))
    2) gap-fill: только внутри уже существующей истории [min_ts_open, last_closed_ts_open]
       (то есть не "от полуночи дня", а от первого реально существующего ts_open).
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
        now = self._clock.now().value
        end_floor = _floor_to_minute(now)

        bounds = self._index.bounds(instrument_id)
        if bounds is None:
            raise ValueError(
                f"No canonical candles for {instrument_id}. "
                "Run initial history backfill first."
            )
        first_ts_open, last_ts_open = bounds

        last_closed = self._index.max_ts_open_lt(
            instrument_id=instrument_id,
            before=UtcTimestamp(end_floor),
        )
        if last_closed is None:
            # На практике это значит: bounds есть, но в окне "до end_floor" ничего нет.
            # Это странно, но лучше явно.
            raise ValueError(f"Cannot determine last closed ts_open for {instrument_id}")

        tail_start_dt = last_closed.value + timedelta(minutes=1)
        tail_end_dt = end_floor

        tail_range = TimeRange(
            start=UtcTimestamp(tail_start_dt),
            end=UtcTimestamp(tail_end_dt),
        )

        tail_read, tail_written, tail_batches = 0, 0, 0
        if tail_range.start.value < tail_range.end.value:
            tail_read, tail_written, tail_batches = self._ingest_range(
                instrument_id=instrument_id,
                time_range=tail_range,
            )

        # GAP-SCAN: только внутри существующей истории: [first_ts_open, last_closed + 1m)
        gap_scan_end = min(last_closed.value + timedelta(minutes=1), tail_range.start.value)
        gap_scan = None
        if first_ts_open.value < gap_scan_end:
            gap_scan = TimeRange(start=first_ts_open, end=UtcTimestamp(gap_scan_end))

        gap_read, gap_written, gap_batches = 0, 0, 0
        if gap_scan is not None:
            gr, gw, gb = self._fill_gaps_in_range(
                instrument_id=instrument_id,
                time_range=gap_scan,
            )
            gap_read += gr
            gap_written += gw
            gap_batches += gb

        return RestCatchUp1mReport(
            instrument=str(instrument_id),
            tail_start=str(tail_range.start),
            tail_end=str(tail_range.end),
            tail_rows_read=tail_read,
            tail_rows_written=tail_written,
            gap_rows_read=gap_read,
            gap_rows_written=gap_written,
            batches_written=tail_batches + gap_batches,
        )

    def _ingest_range(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> tuple[int, int, int]:  # noqa: E501
        # режем range на ≤N суток, а внутри пишем батчами по batch_size
        rows_read = 0
        rows_written = 0
        batches = 0

        for chunk in slice_time_range_by_utc_days(time_range, max_days=self._max_days):
            batch: list[CandleWithMeta] = []
            for row in self._source.stream_1m(instrument_id, chunk):
                rows_read += 1
                batch.append(_with_forced_ingest_id(row, ingest_id=self._ingest_id))
                if len(batch) >= self._batch_size:
                    self._writer.write_1m(batch)
                    rows_written += len(batch)
                    batches += 1
                    batch = []
            if batch:
                self._writer.write_1m(batch)
                rows_written += len(batch)
                batches += 1

        return rows_read, rows_written, batches

    def _fill_gaps_in_range(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> tuple[int, int, int]:  # noqa: E501
        """
        1) берём daily_counts по диапазону
        2) для дней, где count != expected, берём distinct_ts_opens и строим missing ranges
        3) догружаем REST-ом только missing ranges
        """
        total_read = 0
        total_written = 0
        total_batches = 0

        for day_start_dt, day_end_dt in _iterate_utc_days(time_range.start.value, time_range.end.value):  # noqa: E501
            day_start = max(day_start_dt, time_range.start.value)
            day_end = min(day_end_dt, time_range.end.value)
            if day_start >= day_end:
                continue

            expected = int((day_end - day_start).total_seconds() // 60)
            actual = _find_daily_count(
                self._index.daily_counts(
                    instrument_id=instrument_id,
                    time_range=TimeRange(start=UtcTimestamp(day_start), end=UtcTimestamp(day_end)),
                ),
                day_start_dt,
            )

            if actual == expected:
                continue

            existing = self._index.distinct_ts_opens(
                instrument_id=instrument_id,
                time_range=TimeRange(start=UtcTimestamp(day_start), end=UtcTimestamp(day_end)),
            )
            missing = _missing_ranges_for_day(existing=existing, start=day_start, end=day_end)
            for gap in missing:
                r, w, b = self._ingest_range(instrument_id=instrument_id, time_range=gap)
                total_read += r
                total_written += w
                total_batches += b

        return total_read, total_written, total_batches


def _with_forced_ingest_id(row: CandleWithMeta, *, ingest_id: UUID) -> CandleWithMeta:
    m = row.meta
    forced = CandleMeta(
        source=m.source,
        ingested_at=m.ingested_at,
        ingest_id=ingest_id,
        instrument_key=m.instrument_key,
        trades_count=m.trades_count,
        taker_buy_volume_base=m.taker_buy_volume_base,
        taker_buy_volume_quote=m.taker_buy_volume_quote,
    )
    return CandleWithMeta(candle=row.candle, meta=forced)


def _floor_to_minute(dt: datetime) -> datetime:
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.replace(second=0, microsecond=0)


def _iterate_utc_days(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    cur = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)

    day0 = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    cur_day = day0
    while cur_day < end_utc:
        nxt = cur_day + timedelta(days=1)
        yield cur_day, nxt
        cur_day = nxt


def _find_daily_count(rows: Sequence[object], day_start_utc: datetime) -> int:
    # допускаем, что адаптер вернёт dataclass/obj с .day/.count или Mapping с ключами
    for r in rows:
        if isinstance(r, dict):
            d = r.get("day")
            c = r.get("count")
        else:
            d = getattr(r, "day", None)
            c = getattr(r, "count", None)

        if d is None or c is None:
            continue

        d_utc = d.astimezone(timezone.utc) if getattr(d, "tzinfo", None) else d.replace(tzinfo=timezone.utc)  # noqa: E501
        if d_utc == day_start_utc.replace(tzinfo=timezone.utc):
            return int(c)
    return 0


def _missing_ranges_for_day(
    *,
    existing: Sequence[UtcTimestamp],
    start: datetime,
    end: datetime,
) -> list[TimeRange]:
    # existing: ts_open внутри [start, end)
    existing_set = {e.value.replace(second=0, microsecond=0, tzinfo=timezone.utc) for e in existing}

    out: list[TimeRange] = []
    cur = start.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    end_dt = end.replace(second=0, microsecond=0, tzinfo=timezone.utc)

    missing_start: datetime | None = None
    while cur < end_dt:
        if cur not in existing_set:
            if missing_start is None:
                missing_start = cur
        else:
            if missing_start is not None:
                out.append(
                    TimeRange(
                        start=UtcTimestamp(missing_start),
                        end=UtcTimestamp(cur),
                    )
                )
                missing_start = None
        cur = cur + timedelta(minutes=1)

    if missing_start is not None and missing_start < end_dt:
        out.append(TimeRange(start=UtcTimestamp(missing_start), end=UtcTimestamp(end_dt)))

    return out
