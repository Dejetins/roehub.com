from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence
from uuid import UUID

import pytest

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    DailyTsOpenCount,
)
from trading.contexts.market_data.application.use_cases.rest_catchup_1m import (
    RestCatchUp1mReport,
    RestCatchUp1mUseCase,
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
    """Test clock returning a fixed UTC timestamp."""

    def __init__(self, now_ts: UtcTimestamp) -> None:
        self._now_ts = now_ts

    def now(self) -> UtcTimestamp:
        """
        Return fixed current timestamp for deterministic tests.

        Parameters:
        - None.

        Returns:
        - Preconfigured `UtcTimestamp`.

        Assumptions/Invariants:
        - Timestamp is already normalized to UTC by `UtcTimestamp`.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return self._now_ts


class FakeIndex:
    """In-memory fake of canonical index reader for rest-catchup use-case tests."""

    def __init__(
        self,
        *,
        last: UtcTimestamp | None,
        bounds_value: tuple[UtcTimestamp, UtcTimestamp] | None,
        daily: Sequence[DailyTsOpenCount] | None = None,
        distinct: Sequence[UtcTimestamp] | None = None,
    ) -> None:
        self._last = last
        self._bounds = bounds_value
        self._daily = list(daily or [])
        self._distinct = list(distinct or [])

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Return configured canonical bounds.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).

        Returns:
        - Tuple of `(first_ts_open, last_ts_open)` or `None`.

        Assumptions/Invariants:
        - Fake has preconfigured bounds at initialization.

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
        Return preconfigured latest canonical ts_open before requested bound.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - before: upper bound for ts_open lookup (unused in this fake).

        Returns:
        - Preconfigured last timestamp or `None`.

        Assumptions/Invariants:
        - `last` passed to constructor represents lookup result.

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
        Return preconfigured per-day counts for gap scan.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - time_range: scan range (unused in this fake).

        Returns:
        - Sequence of day/count records.

        Assumptions/Invariants:
        - Caller may consume counts as immutable data.

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
        Return preconfigured distinct timestamps for missing-range reconstruction.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - time_range: day window (unused in this fake).

        Returns:
        - Sequence of existing distinct open timestamps.

        Assumptions/Invariants:
        - Sequence order is preserved from constructor argument.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = instrument_id
        _ = time_range
        return list(self._distinct)


class FakeSource:
    """Source fake returning preconfigured rows for every stream call."""

    def __init__(self, rows: Iterable[CandleWithMeta]) -> None:
        self._rows = list(rows)

    def stream_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Yield preconfigured source rows.

        Parameters:
        - instrument_id: requested instrument id (unused in this fake).
        - time_range: requested range (unused in this fake).

        Returns:
        - Iterator over stored rows.

        Assumptions/Invariants:
        - Rows are already valid domain DTOs.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = instrument_id
        _ = time_range
        return iter(self._rows)


class RecordingWriter:
    """Writer fake that records every batch written by use-case."""

    def __init__(self) -> None:
        self.calls: list[list[CandleWithMeta]] = []

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        """
        Capture a writer call for later assertions.

        Parameters:
        - rows: iterable batch sent by use-case.

        Returns:
        - None.

        Assumptions/Invariants:
        - Rows iterable can be consumed exactly once in this fake.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends one list entry into `calls`.
        """
        self.calls.append(list(rows))


def _ts(dt: datetime) -> UtcTimestamp:
    """Wrap timezone-aware datetime into `UtcTimestamp` for tests."""
    return UtcTimestamp(dt)


def _mk_row(instrument_id: InstrumentId, ts_open: datetime) -> CandleWithMeta:
    """
    Build one deterministic candle row used by use-case tests.

    Parameters:
    - instrument_id: instrument identifier for the candle.
    - ts_open: open timestamp for candle minute.

    Returns:
    - `CandleWithMeta` with fixed values and canonical instrument key.

    Assumptions/Invariants:
    - `ts_open` is timezone-aware UTC datetime.

    Errors/Exceptions:
    - Propagates primitive validation errors if invalid values are passed.

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


def test_rest_catchup_report_to_dict_is_json_serializable() -> None:
    """
    Ensure report DTO converts timestamp fields into JSON-safe primitive values.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - `to_dict()` is the canonical serialization contract for CLI/notebooks.

    Errors/Exceptions:
    - None.

    Side effects:
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
        gap_batches=1,
    )

    payload = report.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["tail_start"] == "2026-02-05T12:04:00.000Z"
    assert decoded["tail_end"] == "2026-02-05T12:05:00.000Z"
    assert decoded["gap_rows_written"] == 3


def test_rest_catchup_run_returns_json_friendly_report() -> None:
    """
    Verify use-case run result can be serialized through `report.to_dict()` without TypeError.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - index has a known latest canonical candle, so tail catch-up runs.

    Errors/Exceptions:
    - None.

    Side effects:
    - Writes one batch to writer fake.
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
    assert len(writer.calls) == 1

    # Regression check: must not raise TypeError for datetime-like values.
    json.dumps(payload, ensure_ascii=False)


def test_rest_catchup_run_raises_without_canonical_seed() -> None:
    """
    Ensure catch-up requires canonical seed data and fails fast otherwise.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - use-case should not infer unknown history start automatically.

    Errors/Exceptions:
    - Expects `ValueError` when index has no canonical candles.

    Side effects:
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
