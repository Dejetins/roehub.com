from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class _FixedClock:
    """Clock fake returning deterministic UTC timestamp."""

    def __init__(self, now_value: UtcTimestamp) -> None:
        """Store fixed value returned by `now()` calls."""
        self._now_value = now_value

    def now(self) -> UtcTimestamp:
        """Return preconfigured timestamp."""
        return self._now_value


class _FakeWriter:
    """Raw writer fake recording all inserted rows."""

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        self.rows: list[CandleWithMeta] = []

    def write_1m(self, rows):  # noqa: ANN001
        """Store rows written by use-case for later assertions."""
        self.rows.extend(list(rows))


class _FakeSource:
    """REST source fake generating one candle per minute in requested range."""

    def __init__(self, instrument_id: InstrumentId) -> None:
        """Store instrument used for generated candle rows."""
        self._instrument_id = instrument_id
        self.calls: list[TimeRange] = []

    def stream_1m(self, instrument_id, time_range):  # noqa: ANN001
        """Yield deterministic 1m rows and record requested ranges."""
        _ = instrument_id
        self.calls.append(time_range)
        cursor = time_range.start.value
        while cursor < time_range.end.value:
            yield _row(self._instrument_id, cursor)
            cursor += timedelta(minutes=1)


class _FakeIndex:
    """Canonical index fake exposing configurable `bounds_1m` output."""

    def __init__(
        self,
        bounds: tuple[UtcTimestamp | None, UtcTimestamp | None],
        *,
        existing_ts_opens: list[UtcTimestamp] | None = None,
    ) -> None:
        """
        Persist tuple returned by `bounds_1m` and optional existing minute set.

        Parameters:
        - bounds: value returned by `bounds_1m`.
        - existing_ts_opens: optional existing canonical minute list for dedup checks.
        """
        self._bounds = bounds
        self._existing_ts_opens = list(existing_ts_opens or [])

    def bounds_1m(self, *, instrument_id, before):  # noqa: ANN001
        """Return preconfigured canonical bounds tuple."""
        _ = instrument_id
        _ = before
        return self._bounds

    def distinct_ts_opens(self, *, instrument_id, time_range):  # noqa: ANN001
        """
        Return existing canonical minute list filtered by requested range.

        Parameters:
        - instrument_id: requested instrument id.
        - time_range: requested UTC half-open range.

        Returns:
        - Existing timestamps inside requested range.
        """
        _ = instrument_id
        start = time_range.start.value
        end = time_range.end.value
        return [item for item in self._existing_ts_opens if start <= item.value < end]


def _row(instrument_id: InstrumentId, ts_open: datetime) -> CandleWithMeta:
    """
    Build deterministic candle row for one minute timestamp.

    Parameters:
    - instrument_id: instrument identity for the candle.
    - ts_open: minute open timestamp in UTC.

    Returns:
    - One `CandleWithMeta` row.
    """
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume_base=10.0,
        volume_quote=15.0,
    )
    meta = CandleMeta(
        source="rest",
        ingested_at=UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=1,
        taker_buy_volume_base=1.0,
        taker_buy_volume_quote=2.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_historical_task_is_clamped_to_current_canonical_min() -> None:
    """Ensure historical task end is clamped to current canonical minimum before execution."""
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    source = _FakeSource(instrument)
    writer = _FakeWriter()
    index = _FakeIndex(
        bounds=(
            UtcTimestamp(datetime(2026, 2, 9, 13, 54, tzinfo=timezone.utc)),
            UtcTimestamp(datetime(2026, 2, 9, 13, 59, tzinfo=timezone.utc)),
        )
    )

    use_case = RestFillRange1mUseCase(
        source=source,
        writer=writer,
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
        max_days_per_insert=7,
        batch_size=100,
        index_reader=index,
    )

    task = RestFillTask(
        instrument_id=instrument,
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)),
        ),
        reason="historical_backfill",
    )
    result = use_case.run(task)

    assert str(result.task.time_range.end) == str(
        UtcTimestamp(datetime(2026, 2, 9, 13, 54, tzinfo=timezone.utc))
    )
    assert result.rows_written == 4
    assert len(writer.rows) == 4


def test_historical_task_becomes_noop_when_clamp_end_is_before_start() -> None:
    """Ensure no write occurs when canonical minimum already moved before task start."""
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    source = _FakeSource(instrument)
    writer = _FakeWriter()
    index = _FakeIndex(
        bounds=(
            UtcTimestamp(datetime(2026, 2, 9, 13, 49, tzinfo=timezone.utc)),
            UtcTimestamp(datetime(2026, 2, 9, 13, 59, tzinfo=timezone.utc)),
        )
    )

    use_case = RestFillRange1mUseCase(
        source=source,
        writer=writer,
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
        max_days_per_insert=7,
        batch_size=100,
        index_reader=index,
    )

    task = RestFillTask(
        instrument_id=instrument,
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)),
        ),
        reason="historical_backfill",
    )
    result = use_case.run(task)

    assert result.rows_written == 0
    assert source.calls == []
    assert writer.rows == []


def test_chunk_ingest_skips_existing_minutes_when_index_reader_is_configured() -> None:
    """
    Ensure rest-fill does not reinsert minutes already present in canonical index.
    """
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    source = _FakeSource(instrument)
    writer = _FakeWriter()
    index = _FakeIndex(
        bounds=(None, None),
        existing_ts_opens=[
            UtcTimestamp(datetime(2026, 2, 9, 13, 51, tzinfo=timezone.utc)),
        ],
    )
    use_case = RestFillRange1mUseCase(
        source=source,
        writer=writer,
        clock=_FixedClock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
        max_days_per_insert=7,
        batch_size=100,
        index_reader=index,
    )
    task = RestFillTask(
        instrument_id=instrument,
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 9, 13, 50, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 9, 13, 53, tzinfo=timezone.utc)),
        ),
        reason="scheduler_tail",
    )

    result = use_case.run(task)

    assert result.rows_read == 3
    assert result.rows_written == 2
    assert [item.candle.ts_open.value.minute for item in writer.rows] == [50, 52]
