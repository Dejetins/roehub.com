from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


class RecordingGateway:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, Sequence[Mapping[str, Any]]]] = []

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        self.inserts.append((table, rows))

    def select(self, query: str, parameters):  # pragma: no cover
        raise AssertionError("select not expected in writer test")


def _ts(dt: datetime) -> UtcTimestamp:
    return UtcTimestamp(dt)


def _row(market_id: int, symbol: str, volume_quote=None, trades_count=None) -> CandleWithMeta:
    instrument = InstrumentId(MarketId(market_id), Symbol(symbol))
    ts_open = _ts(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc))
    ts_close = _ts(datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc))

    candle = Candle(
        instrument_id=instrument,
        ts_open=ts_open,
        ts_close=ts_close,
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume_base=1.0,
        volume_quote=volume_quote,
    )

    meta = CandleMeta(
        source="file",
        ingested_at=_ts(datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)),
        ingest_id=None,
        instrument_key=f"{market_id}:{symbol}",
        trades_count=trades_count,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )

    return CandleWithMeta(candle=candle, meta=meta)


def test_writer_routes_binance_markets_to_binance_raw_and_normalizes_non_nullable() -> None:
    gw = RecordingGateway()
    writer = ClickHouseRawKlineWriter(gateway=gw)

    row = _row(1, "BTCUSDT", volume_quote=None, trades_count=None)
    writer.write_1m([row])

    assert len(gw.inserts) == 1
    table, payload = gw.inserts[0]
    assert table.endswith("market_data.raw_binance_klines_1m")
    assert payload[0]["quote_asset_volume"] == 0.0
    assert payload[0]["number_of_trades"] == 0


def test_writer_routes_bybit_markets_to_bybit_raw_and_normalizes_turnover() -> None:
    gw = RecordingGateway()
    writer = ClickHouseRawKlineWriter(gateway=gw)

    row = _row(3, "BTCUSDT", volume_quote=None)
    writer.write_1m([row])

    assert len(gw.inserts) == 1
    table, payload = gw.inserts[0]
    assert table.endswith("market_data.raw_bybit_klines_1m")
    assert payload[0]["turnover"] == 0.0
    assert payload[0]["interval_min"] == 1
    assert isinstance(payload[0]["start_time_ms"], int)
