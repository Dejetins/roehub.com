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


def test_reader_uses_one_final_query_for_requested_range() -> None:
    # clock is retained for constructor compatibility but FINAL now handles historical duplicates
    clock = FixedClock(_ts(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)))
    gw = StubGateway()

    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    tr = TimeRange(
        _ts(datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc)),
        _ts(datetime(2026, 2, 4, 13, 0, tzinfo=timezone.utc)),
    )

    gw.responses = [
        [
            _canonical_row(
                datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc),
                datetime(2026, 2, 5, 10, 0, tzinfo=timezone.utc),
            ),
            _canonical_row(
                datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 2, 5, 11, 0, tzinfo=timezone.utc),
            ),
        ],
    ]

    reader = ClickHouseCanonicalCandleReader(gateway=gw, clock=clock)
    out = list(reader.read_1m(instrument, tr))

    assert len(out) == 2
    assert out[0].candle.ts_open.value == datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc)
    assert out[1].candle.ts_open.value == datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc)

    assert len(gw.calls) == 1
    query, parameters = gw.calls[0]
    assert " FINAL" in query
    assert "LIMIT 1 BY" not in query
    assert parameters["market_id"] == 1
    assert parameters["symbol"] == "BTCUSDT"
    assert parameters["start"] == datetime(2026, 2, 4, 11, 0, tzinfo=timezone.utc)
    assert parameters["end"] == datetime(2026, 2, 4, 13, 0, tzinfo=timezone.utc)
