from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_index_reader import (  # noqa: E501
    ClickHouseCanonicalCandleIndexReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class FakeGateway:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = list(rows)
        self.last_query: str | None = None
        self.last_params: Mapping[str, Any] | None = None

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:  # pragma: no cover  # noqa: E501
        raise AssertionError("insert_rows not expected in this test")

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        self.last_query = query
        self.last_params = parameters
        return self._rows


def test_bounds_returns_min_max():
    dt1 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc)

    gw = FakeGateway([{"first": dt1, "last": dt2}])
    r = ClickHouseCanonicalCandleIndexReader(gateway=gw, database="market_data")

    inst = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    b = r.bounds(inst)

    assert b is not None
    assert str(b[0]) == str(UtcTimestamp(dt1))
    assert str(b[1]) == str(UtcTimestamp(dt2))

    assert gw.last_query is not None
    assert "min(ts_open)" in gw.last_query

    assert gw.last_params is not None
    assert int(gw.last_params["market_id"]) == 1


def test_max_ts_open_lt():
    dt = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    gw = FakeGateway([{"last": dt}])
    r = ClickHouseCanonicalCandleIndexReader(gateway=gw, database="market_data")

    inst = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    before = UtcTimestamp(datetime(2026, 2, 1, 13, 0, tzinfo=timezone.utc))
    out = r.max_ts_open_lt(instrument_id=inst, before=before)

    assert out is not None
    assert str(out) == str(UtcTimestamp(dt))

    assert gw.last_query is not None
    assert "max(ts_open)" in gw.last_query

    assert gw.last_params is not None
    assert int(gw.last_params["market_id"]) == 1


def test_distinct_ts_opens():
    dt1 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc)
    gw = FakeGateway([{"ts_open": dt1}, {"ts_open": dt2}])
    r = ClickHouseCanonicalCandleIndexReader(gateway=gw, database="market_data")

    inst = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    tr = TimeRange(
        start=UtcTimestamp(dt1),
        end=UtcTimestamp(datetime(2026, 2, 1, 0, 2, tzinfo=timezone.utc)),
    )
    out = r.distinct_ts_opens(instrument_id=inst, time_range=tr)

    assert len(out) == 2
    assert str(out[0]) == str(UtcTimestamp(dt1))
    assert str(out[1]) == str(UtcTimestamp(dt2))

    assert gw.last_query is not None
    assert "SELECT DISTINCT" in gw.last_query
