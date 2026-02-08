# src/trading/contexts/market_data/application/use_cases/rest_catchup_1m.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
from trading.shared_kernel.primitives import CandleMeta, InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class RestCatchUp1mReport:
    instrument_id: InstrumentId
    start: UtcTimestamp
    end: UtcTimestamp

    tail_rows_written: int

    gap_days_scanned: int
    gap_days_with_gaps: int
    gap_ranges_filled: int
    gap_rows_written: int

    total_rows_written: int
    duration_s: float
    lag_to_end_s: float


class RestCatchUp1mUseCase:
    """
    REST догонка 1m свечей до "последней закрытой минуты" + автозаполнение гэпов по истории.

    Семантика времени:
    - end = floor(clock.now()) до минуты (UTC); это ts_open "текущей" минуты (ещё не закрыта),
      поэтому диапазон [start, end) покрывает только закрытые свечи.
    - start = last_closed_ts_open + 1m, где last_closed_ts_open берётся из canonical индекса.

    Источник данных:
    - source.stream_1m(...) обязан соблюдать полуинтервал [start, end) и возвращать CandleWithMeta,
      где meta.source = "rest", meta.ingested_at задан, meta.instrument_key задан.
    - ingest_id унифицируется на уровне use-case (проставляется одинаковый для всего запуска).

    Запись:
    - пишем в raw через RawKlineWriter батчами batch_size
    - диапазоны режутся на чанки по max_days_per_insert (<= 7), чтобы не ловить проблемы на больших хвостах.
    """  # noqa: E501

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
        if max_days_per_insert <= 0:
            raise ValueError("max_days_per_insert must be > 0")
        if max_days_per_insert > 7:
            raise ValueError("max_days_per_insert must be <= 7")
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
        t0 = time.monotonic()

        end = self._floor_now_to_minute()
        last_closed = self._index.max_ts_open_lt(instrument_id=instrument_id, before=end)
        if last_closed is None:
            raise ValueError(
                f"No canonical candles for {instrument_id}. "
                "Run initial backfill first, then rest-catchup can maintain tail and fill gaps."
            )

        start = UtcTimestamp(_ensure_tz_utc(last_closed.value + timedelta(minutes=1)))

        gap_days_scanned = 0
        gap_days_with_gaps = 0
        gap_ranges_filled = 0
        gap_rows_written = 0

        # 1) Gap fill по истории, строго ДО start (чтобы не пересекаться с tail)
        bounds = self._index.bounds(instrument_id)
        if bounds is not None:
            hist_start, _hist_last = bounds
            gap_scan_end_dt = min(start.value, end.value)
            if hist_start.value < gap_scan_end_dt:
                scan_range = TimeRange(
                    start=hist_start,
                    end=UtcTimestamp(_ensure_tz_utc(gap_scan_end_dt)),
                )
                g = self._fill_gaps_in_range(instrument_id=instrument_id, time_range=scan_range)
                gap_days_scanned = g.days_scanned
                gap_days_with_gaps = g.days_with_gaps
                gap_ranges_filled = g.ranges_filled
                gap_rows_written = g.rows_written

        # 2) Tail догонка [start, end)
        tail_rows_written = 0
        if start.value < end.value:
            tail_rows_written = self.fill_gap(
                instrument_id=instrument_id,
                gap=TimeRange(start=start, end=end),
            )

        t1 = time.monotonic()
        total_rows = tail_rows_written + gap_rows_written

        return RestCatchUp1mReport(
            instrument_id=instrument_id,
            start=start,
            end=end,
            tail_rows_written=tail_rows_written,
            gap_days_scanned=gap_days_scanned,
            gap_days_with_gaps=gap_days_with_gaps,
            gap_ranges_filled=gap_ranges_filled,
            gap_rows_written=gap_rows_written,
            total_rows_written=total_rows,
            duration_s=(t1 - t0),
            lag_to_end_s=max(0.0, (self._clock.now().value - end.value).total_seconds()),
        )

    def fill_gap(self, *, instrument_id: InstrumentId, gap: TimeRange) -> int:
        """
        Заполнить заданный гэп (полуинтервал [gap.start, gap.end)) через REST source и raw writer.

        Возвращает: количество записанных строк (фактически отправленных в writer).
        """
        rows_written = 0
        for chunk in slice_time_range_by_utc_days(gap, max_days=self._max_days):
            rows_written += self._write_range(instrument_id=instrument_id, time_range=chunk)
        return rows_written

    def _floor_now_to_minute(self) -> UtcTimestamp:
        now = _ensure_tz_utc(self._clock.now().value)
        floored = now.replace(second=0, microsecond=0)
        return UtcTimestamp(floored)

    @dataclass(frozen=True, slots=True)
    class _GapFillStats:
        days_scanned: int
        days_with_gaps: int
        ranges_filled: int
        rows_written: int

    def _fill_gaps_in_range(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> _GapFillStats:  # noqa: E501
        # 1) Быстро получаем countDistinct(ts_open) по дням
        counts = self._index.daily_counts(instrument_id=instrument_id, time_range=time_range)
        count_by_day = {c.day: int(c.count) for c in counts}

        days_scanned = 0
        days_with_gaps = 0
        ranges_filled = 0
        rows_written = 0

        # 2) Проходим день-за-днём и сравниваем expected минут в окне vs фактический distinct
        for d_start, d_end, day_key in _iter_utc_day_windows(time_range):
            days_scanned += 1
            expected = _expected_minutes(d_start, d_end)
            actual = int(count_by_day.get(day_key, 0))

            if actual >= expected:
                continue

            days_with_gaps += 1

            existing = self._index.distinct_ts_opens(
                instrument_id=instrument_id,
                time_range=TimeRange(
                    start=UtcTimestamp(_ensure_tz_utc(d_start)),
                    end=UtcTimestamp(_ensure_tz_utc(d_end)),
                ),
            )
            missing_ranges = _missing_ranges_for_window(
                existing=existing,
                start_dt=d_start,
                end_dt=d_end,
            )
            for gap in missing_ranges:
                ranges_filled += 1
                rows_written += self.fill_gap(instrument_id=instrument_id, gap=gap)

        return RestCatchUp1mUseCase._GapFillStats(
            days_scanned=days_scanned,
            days_with_gaps=days_with_gaps,
            ranges_filled=ranges_filled,
            rows_written=rows_written,
        )

    def _write_range(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> int:
        batch: list[CandleWithMeta] = []
        written = 0

        for row in self._source.stream_1m(instrument_id, time_range):
            batch.append(_with_ingest_id(row, self._ingest_id))
            if len(batch) >= self._batch_size:
                self._writer.write_1m(batch)
                written += len(batch)
                batch.clear()

        if batch:
            self._writer.write_1m(batch)
            written += len(batch)

        return written


def _with_ingest_id(row: CandleWithMeta, ingest_id: UUID) -> CandleWithMeta:
    m = row.meta
    fixed_meta = CandleMeta(
        source=m.source,
        ingested_at=m.ingested_at,
        ingest_id=ingest_id,
        instrument_key=m.instrument_key,
        trades_count=m.trades_count,
        taker_buy_volume_base=m.taker_buy_volume_base,
        taker_buy_volume_quote=m.taker_buy_volume_quote,
    )
    return CandleWithMeta(candle=row.candle, meta=fixed_meta)


def _expected_minutes(start_dt: datetime, end_dt: datetime) -> int:
    dur_s = (end_dt - start_dt).total_seconds()
    if dur_s <= 0:
        return 0
    return int(dur_s // 60)


def _missing_ranges_for_window(
    *,
    existing: Sequence[UtcTimestamp],
    start_dt: datetime,
    end_dt: datetime,
) -> list[TimeRange]:
    """
    existing: отсортированные ts_open (distinct) внутри [start_dt, end_dt).
    Возвращает список TimeRange (missing) в тех же границах, с шагом 1m.
    """
    if start_dt >= end_dt:
        return []

    existing_set = {e.value for e in existing}

    missing: list[TimeRange] = []
    cur = start_dt
    missing_start: datetime | None = None

    while cur < end_dt:
        if cur in existing_set:
            if missing_start is not None:
                missing.append(
                    TimeRange(
                        start=UtcTimestamp(_ensure_tz_utc(missing_start)),
                        end=UtcTimestamp(_ensure_tz_utc(cur)),
                    )
                )
                missing_start = None
        else:
            if missing_start is None:
                missing_start = cur
        cur = cur + timedelta(minutes=1)

    if missing_start is not None:
        missing.append(
            TimeRange(
                start=UtcTimestamp(_ensure_tz_utc(missing_start)),
                end=UtcTimestamp(_ensure_tz_utc(end_dt)),
            )
        )

    return missing


def _iter_utc_day_windows(time_range: TimeRange) -> Iterator[tuple[datetime, datetime, date]]:
    """
    Итератор по дневным окнам (UTC), пересекающим time_range.

    Для каждого дня отдаёт:
    - window_start (datetime UTC)
    - window_end   (datetime UTC)
    - day_key      (date) — UTC дата, совпадает с toDate(ts_open) в ClickHouse
    """
    start = _ensure_tz_utc(time_range.start.value)
    end = _ensure_tz_utc(time_range.end.value)

    cursor = _floor_to_utc_day(start)
    while cursor < end:
        day_start = cursor
        day_end = cursor + timedelta(days=1)

        window_start = max(day_start, start)
        window_end = min(day_end, end)

        if window_start < window_end:
            yield (window_start, window_end, window_start.date())

        cursor = day_end


def _floor_to_utc_day(dt: datetime) -> datetime:
    dt_utc = _ensure_tz_utc(dt)
    return datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=timezone.utc)


def _ensure_tz_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
