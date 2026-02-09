from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.services.insert_buffer import (
    AsyncRawInsertBuffer,
    InsertBufferHooks,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


class _RecordingWriter:
    """Raw writer fake recording all batch writes."""

    def __init__(self) -> None:
        """Initialize empty write call registry."""
        self.calls: list[list[CandleWithMeta]] = []

    def write_1m(self, rows) -> None:
        """Record one materialized batch write."""
        self.calls.append(list(rows))


class _MonotonicClock:
    """Clock fake returning increasing timestamps on each call."""

    def __init__(self, start: datetime) -> None:
        """Initialize monotonic clock with first timestamp."""
        self._current = start

    def now(self) -> UtcTimestamp:
        """Return current timestamp and increment by 100ms."""
        value = self._current
        self._current = self._current + timedelta(milliseconds=100)
        return UtcTimestamp(value)


def _row(ts_open: datetime, ingested_at: datetime) -> CandleWithMeta:
    """
    Build deterministic WS candle row for insert-buffer tests.

    Parameters:
    - ts_open: candle open time.
    - ingested_at: metadata receive timestamp.

    Returns:
    - CandleWithMeta row.
    """
    instrument_id = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume_base=1.0,
        volume_quote=100.0,
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=UtcTimestamp(ingested_at),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=1,
        taker_buy_volume_base=0.1,
        taker_buy_volume_quote=10.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_insert_buffer_flushes_by_size_threshold() -> None:
    """Ensure buffer flushes immediately when row count reaches max threshold."""
    async def _scenario() -> None:
        writer = _RecordingWriter()
        clock = _MonotonicClock(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))
        buffer = AsyncRawInsertBuffer(
            writer=writer,
            clock=clock,
            flush_interval_ms=500,
            max_buffer_rows=2,
        )
        await buffer.start()

        ts = datetime(2026, 2, 5, 11, 59, tzinfo=timezone.utc)
        buffer.submit(_row(ts, ts - timedelta(seconds=1)))
        buffer.submit(_row(ts + timedelta(minutes=1), ts - timedelta(seconds=1)))
        await asyncio.sleep(0.1)
        await buffer.close()

        assert len(writer.calls) == 1
        assert len(writer.calls[0]) == 2

    asyncio.run(_scenario())


def test_insert_buffer_flushes_by_timer_threshold() -> None:
    """Ensure buffer flushes by periodic timer even when size threshold is not reached."""
    async def _scenario() -> None:
        writer = _RecordingWriter()
        clock = _MonotonicClock(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))
        buffer = AsyncRawInsertBuffer(
            writer=writer,
            clock=clock,
            flush_interval_ms=20,
            max_buffer_rows=100,
        )
        await buffer.start()

        ts = datetime(2026, 2, 5, 11, 59, tzinfo=timezone.utc)
        buffer.submit(_row(ts, ts - timedelta(seconds=1)))
        await asyncio.sleep(0.12)
        await buffer.close()

        assert len(writer.calls) >= 1
        assert len(writer.calls[0]) == 1

    asyncio.run(_scenario())


def test_insert_buffer_observes_slo_callbacks() -> None:
    """Ensure insert buffer emits SLO histogram observations during flush."""
    async def _scenario() -> None:
        writer = _RecordingWriter()
        clock = _MonotonicClock(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc))
        start_observations: list[float] = []
        done_observations: list[float] = []
        batch_observations: list[tuple[int, float]] = []

        buffer = AsyncRawInsertBuffer(
            writer=writer,
            clock=clock,
            flush_interval_ms=1000,
            max_buffer_rows=1,
            hooks=InsertBufferHooks(
                on_ws_closed_to_insert_start=start_observations.append,
                on_ws_closed_to_insert_done=done_observations.append,
                on_insert_batch=lambda rows, duration: batch_observations.append((rows, duration)),
            ),
        )
        await buffer.start()

        ts = datetime(2026, 2, 5, 11, 59, tzinfo=timezone.utc)
        buffer.submit(_row(ts, ts - timedelta(seconds=1)))
        await asyncio.sleep(0.1)
        await buffer.close()

        assert start_observations
        assert done_observations
        assert batch_observations
        assert start_observations[0] >= 0.0
        assert done_observations[0] >= 0.0

    asyncio.run(_scenario())
