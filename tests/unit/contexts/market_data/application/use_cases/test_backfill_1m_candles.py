from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, List, Optional

from trading.contexts.market_data.application.dto import Backfill1mCommand, CandleWithMeta
from trading.contexts.market_data.application.use_cases import Backfill1mCandlesUseCase
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class ClockSequence:
    def __init__(self, stamps: List[UtcTimestamp]) -> None:
        if not stamps:
            raise ValueError("ClockSequence requires at least one timestamp")
        self._stamps = list(stamps)
        self._i = 0

    def now(self) -> UtcTimestamp:
        if self._i >= len(self._stamps):
            return self._stamps[-1]
        ts = self._stamps[self._i]
        self._i += 1
        return ts


class FakeSource:
    def __init__(self, rows: Iterable[CandleWithMeta]) -> None:
        self._rows = list(rows)

    def stream_1m(self, instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]:  # noqa: E501
        # v1: просто отдаём заранее подготовленные строки
        return iter(self._rows)


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: List[List[CandleWithMeta]] = []

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        self.calls.append(list(rows))


def _ts(dt: datetime) -> UtcTimestamp:
    return UtcTimestamp(dt)


def _mk_row(
    instrument_id: InstrumentId,
    ts_open: datetime,
    open_: float = 10.0,
    close_: float = 11.0,
    high: float = 12.0,
    low: float = 9.0,
    volume_base: float = 1.0,
    volume_quote: Optional[float] = 2.0,
) -> CandleWithMeta:
    ts_open_u = _ts(ts_open)
    ts_close_u = _ts(ts_open + timedelta(minutes=1))

    candle = Candle(
        instrument_id=instrument_id,
        ts_open=ts_open_u,
        ts_close=ts_close_u,
        open=open_,
        high=high,
        low=low,
        close=close_,
        volume_base=volume_base,
        volume_quote=volume_quote,
    )

    meta = CandleMeta(
        source="rest",
        ingested_at=_ts(datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc)),
        ingest_id=None,  # type: ignore[arg-type]
        instrument_key="binance:spot:BTCUSDT",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )

    return CandleWithMeta(candle=candle, meta=meta)


def test_backfill_writes_in_batches_and_reports_counts() -> None:
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    start = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, 0, 3, 0, tzinfo=timezone.utc)
    cmd = Backfill1mCommand(
        instrument_id=instrument_id,
        time_range=TimeRange(_ts(start), _ts(end)),
    )

    rows = [
        _mk_row(instrument_id, datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)),
        _mk_row(instrument_id, datetime(2026, 2, 1, 0, 1, 0, tzinfo=timezone.utc)),
        _mk_row(instrument_id, datetime(2026, 2, 1, 0, 2, 0, tzinfo=timezone.utc)),
    ]

    clock = ClockSequence(
        [
            _ts(datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc)),
            _ts(datetime(2026, 2, 5, 12, 0, 1, tzinfo=timezone.utc)),
        ]
    )
    source = FakeSource(rows)
    writer = RecordingWriter()

    uc = Backfill1mCandlesUseCase(source=source, writer=writer, clock=clock, batch_size=2)
    report = uc.run(cmd)

    assert report.candles_read == 3
    assert report.rows_written == 3
    assert report.batches_written == 2
    assert len(writer.calls) == 2
    assert len(writer.calls[0]) == 2
    assert len(writer.calls[1]) == 1
    assert report.started_at.value == datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc)
    assert report.finished_at.value == datetime(2026, 2, 5, 12, 0, 1, tzinfo=timezone.utc)


def test_backfill_empty_source_writes_nothing() -> None:
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    start = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, 0, 1, 0, tzinfo=timezone.utc)
    cmd = Backfill1mCommand(
        instrument_id=instrument_id,
        time_range=TimeRange(_ts(start), _ts(end)),
    )

    clock = ClockSequence([_ts(datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc))])
    source = FakeSource([])
    writer = RecordingWriter()

    uc = Backfill1mCandlesUseCase(source=source, writer=writer, clock=clock, batch_size=10)
    report = uc.run(cmd)

    assert report.candles_read == 0
    assert report.rows_written == 0
    assert report.batches_written == 0
    assert writer.calls == []


def test_backfill_requires_positive_batch_size() -> None:
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))  # noqa: F841

    clock = ClockSequence([_ts(datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc))])
    source = FakeSource([])
    writer = RecordingWriter()

    error: Optional[Exception] = None
    try:
        Backfill1mCandlesUseCase(source=source, writer=writer, clock=clock, batch_size=0)
    except Exception as e:  # noqa: BLE001
        error = e

    assert error is not None
    assert "batch_size" in str(error)
