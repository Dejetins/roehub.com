from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_reader import (  # noqa: E501
    ClickHouseCanonicalCandleReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class FixedClock:
    def __init__(self, ts: UtcTimestamp) -> None:
        self._ts = ts

    def now(self) -> UtcTimestamp:
        return self._ts


class StubGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.responses: list[Sequence[Mapping[str, Any]]] = []

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        self.calls.append((query, parameters))
        if not self.responses:
            return []
        return self.responses.pop(0)

    def insert_rows(self, table: str, rows):  # pragma: no cover
        raise AssertionError("insert not expected in reader test")


def _ts(dt: datetime) -> UtcTimestamp:
    return UtcTimestamp(dt)


def _canonical_row(ts_open: datetime, ingested_at: datetime) -> Mapping[str, Any]:
    return {
        "market_id": 1,
        "symbol": "BTCUSDT",
        "instrument_key": "1:BTCUSDT",
        "ts_open": ts_open,
        "ts_close": datetime(ts_open.year, ts_open.month, ts_open.day, ts_open.hour, ts_open.minute + 1, tzinfo=timezone.utc), # noqa: E501
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume_base": 1.0,
        "volume_quote": None,
        "trades_count": None,
        "taker_buy_volume_base": None,
        "taker_buy_volume_quote": None,
        "source": "file",
        "ingested_at": ingested_at,
        "ingest_id": None,
    }


def test_reader_splits_old_and_tail_ranges_and_uses_dedup_query_on_tail() -> None:
    # now = 2026-02-05 12:00 UTC => cutoff = 2026-02-04 12:00 UTC
    clock = FixedClock(_ts(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)))
    gw = StubGateway()

    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    # time_range spans before and after cutoff
    tr = TimeRange(
        _ts(datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc)),
        _ts(datetime(2026, 2, 4, 13, 0, tzinfo=timezone.utc)),
    )

    # gateway returns 1 row for old part and 1 row for tail part
    gw.responses = [
        [_canonical_row(datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc), datetime(2026, 2, 5, 10, 0, tzinfo=timezone.utc))], # noqa: E501
        [_canonical_row(datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc), datetime(2026, 2, 5, 11, 0, tzinfo=timezone.utc))], # noqa: E501
    ]

    reader = ClickHouseCanonicalCandleReader(gateway=gw, clock=clock)
    out = list(reader.read_1m(instrument, tr))

    assert len(out) == 2
    assert out[0].candle.ts_open.value == datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc)
    assert out[1].candle.ts_open.value == datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc)

    assert len(gw.calls) == 2
    q1, p1 = gw.calls[0]
    q2, p2 = gw.calls[1]

    # first query: no dedup
    assert "LIMIT 1 BY" not in q1
    # second query: tail dedup
    assert "LIMIT 1 BY" in q2
    assert p1["market_id"] == 1 and p2["market_id"] == 1
    assert p1["symbol"] == "BTCUSDT" and p2["symbol"] == "BTCUSDT"
