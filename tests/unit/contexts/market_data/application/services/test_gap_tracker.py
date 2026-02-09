from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.services.gap_tracker import WsMinuteGapTracker
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


def _row(ts_open: datetime) -> CandleWithMeta:
    """
    Build deterministic WS candle row for gap-tracker tests.

    Parameters:
    - ts_open: candle open time in UTC.

    Returns:
    - CandleWithMeta test row.
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
        ingested_at=UtcTimestamp(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=1,
        taker_buy_volume_base=0.1,
        taker_buy_volume_quote=10.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_gap_tracker_no_gap_for_consecutive_minutes() -> None:
    """Ensure consecutive minute sequence does not enqueue fill task."""
    tracker = WsMinuteGapTracker()
    base = datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)
    assert tracker.observe(_row(base)) is None
    assert tracker.observe(_row(base + timedelta(minutes=1))) is None


def test_gap_tracker_enqueues_half_open_range_for_minute_gap() -> None:
    """Ensure gap `[expected, current)` is generated when minutes are skipped."""
    tracker = WsMinuteGapTracker()
    base = datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)
    tracker.observe(_row(base))
    task = tracker.observe(_row(base + timedelta(minutes=3)))
    assert task is not None
    assert str(task.time_range.start) == str(UtcTimestamp(base + timedelta(minutes=1)))
    assert str(task.time_range.end) == str(UtcTimestamp(base + timedelta(minutes=3)))


def test_gap_tracker_out_of_order_does_not_create_gap_and_increments_metric() -> None:
    """Ensure out-of-order candle does not create gap task and increments callback counter."""
    counts = {"out_of_order": 0}

    def _on_out_of_order() -> None:
        counts["out_of_order"] += 1

    tracker = WsMinuteGapTracker(on_out_of_order=_on_out_of_order)
    base = datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)

    tracker.observe(_row(base + timedelta(minutes=5)))
    task = tracker.observe(_row(base + timedelta(minutes=4)))

    assert task is None
    assert counts["out_of_order"] == 1

