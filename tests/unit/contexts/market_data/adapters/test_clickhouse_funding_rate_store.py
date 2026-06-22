from __future__ import annotations

from datetime import datetime, timezone

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.funding_rate_store import (  # noqa: E501
    ClickHouseFundingRateStore,
)
from trading.contexts.market_data.application.dto import FundingInstrument, FundingRateRecord
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


class _Gateway:
    def __init__(self):
        self.inserts = []
        self.select_rows = []
        self.response_rows = []

    def insert_rows(self, table, rows):
        self.inserts.append((table, list(rows)))

    def select(self, query, parameters):
        self.select_rows.append((query, dict(parameters)))
        return list(self.response_rows)


def _ts(hour: int) -> UtcTimestamp:
    return UtcTimestamp(datetime(2026, 6, 22, hour, 0, tzinfo=timezone.utc))


def test_write_funding_rates_routes_to_raw_and_canonical_tables() -> None:
    gateway = _Gateway()
    store = ClickHouseFundingRateStore(gateway=gateway, database="market_data")
    rows = [
        FundingRateRecord(
            instrument_id=InstrumentId(MarketId(2), Symbol("BTCUSDT")),
            instrument_key="binance:futures:BTCUSDT",
            funding_time=_ts(0),
            funding_rate=0.0001,
            funding_interval_minutes=480,
            funding_interval_source="binance_fundingInfo",
            source="binance_fundingRate",
            ingested_at=_ts(1),
            ingest_id="00000000-0000-0000-0000-000000000001",
        ),
        FundingRateRecord(
            instrument_id=InstrumentId(MarketId(4), Symbol("BTCUSDT")),
            instrument_key="bybit:futures:BTCUSDT",
            funding_time=_ts(0),
            funding_rate=0.0002,
            funding_interval_minutes=480,
            funding_interval_source="bybit_instruments_info",
            source="bybit_funding_history",
            ingested_at=_ts(1),
            ingest_id="00000000-0000-0000-0000-000000000002",
            bybit_category="linear",
        ),
    ]

    store.write_funding_rates(rows)

    tables = [table for table, _rows in gateway.inserts]
    assert tables == [
        "market_data.canonical_funding_rates",
        "market_data.raw_binance_funding_rates",
        "market_data.raw_bybit_funding_rates",
    ]
    assert gateway.inserts[0][1][0]["funding_interval_minutes"] == 480
    assert gateway.inserts[2][1][0]["category"] == "linear"


def test_upsert_funding_universe_keeps_interval_metadata() -> None:
    gateway = _Gateway()
    store = ClickHouseFundingRateStore(gateway=gateway, database="market_data")
    row = FundingInstrument(
        instrument_id=InstrumentId(MarketId(2), Symbol("BTCUSDT")),
        instrument_key="binance:futures:BTCUSDT",
        exchange="binance",
        market_type="futures",
        status="TRADING",
        is_tradable=1,
        base_asset="BTC",
        quote_asset="USDT",
        funding_interval_minutes=480,
        funding_interval_source="binance_standard_8h_no_adjustment_row",
        funding_cap=None,
        funding_floor=None,
        updated_at=_ts(1),
    )

    store.upsert_funding_instruments([row])

    assert gateway.inserts[0][0] == "market_data.funding_instrument_universe"
    assert gateway.inserts[0][1][0]["funding_interval_source"] == (
        "binance_standard_8h_no_adjustment_row"
    )


def test_list_tradable_funding_instruments_does_not_alias_max_updated_at() -> None:
    gateway = _Gateway()
    gateway.response_rows = [
        {
            "market_id": 2,
            "symbol": "BTCUSDT",
            "instrument_key": "binance:futures:BTCUSDT",
            "exchange": "binance",
            "market_type": "futures",
            "status": "TRADING",
            "is_tradable": 1,
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "funding_interval_minutes": 480,
            "funding_interval_source": "binance_standard_8h_no_adjustment_row",
            "funding_cap": None,
            "funding_floor": None,
            "latest_updated_at": _ts(1).value,
        }
    ]
    store = ClickHouseFundingRateStore(gateway=gateway, database="market_data")

    rows = store.list_tradable_funding_instruments(market_ids=(MarketId(2),))

    query = gateway.select_rows[0][0]
    assert "max(updated_at) AS latest_updated_at" in query
    assert "max(updated_at) AS updated_at" not in query
    assert rows[0].updated_at == _ts(1)


def test_list_tradable_funding_instruments_treats_clickhouse_naive_time_as_utc() -> None:
    gateway = _Gateway()
    gateway.response_rows = [
        {
            "market_id": 2,
            "symbol": "BTCUSDT",
            "instrument_key": "binance:futures:BTCUSDT",
            "exchange": "binance",
            "market_type": "futures",
            "status": "TRADING",
            "is_tradable": 1,
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "funding_interval_minutes": 480,
            "funding_interval_source": "binance_standard_8h_no_adjustment_row",
            "funding_cap": None,
            "funding_floor": None,
            "latest_updated_at": datetime(2026, 6, 22, 1, 0),
        }
    ]
    store = ClickHouseFundingRateStore(gateway=gateway, database="market_data")

    rows = store.list_tradable_funding_instruments(market_ids=(MarketId(2),))

    assert rows[0].updated_at == _ts(1)


def test_latest_funding_time_uses_nullable_max_for_empty_history() -> None:
    gateway = _Gateway()
    gateway.response_rows = [{"funding_time_ms": None}]
    store = ClickHouseFundingRateStore(gateway=gateway, database="market_data")

    latest = store.latest_funding_time(InstrumentId(MarketId(2), Symbol("BTCUSDT")))

    query = gateway.select_rows[0][0]
    assert "maxOrNull(toUnixTimestamp64Milli(funding_time))" in query
    assert latest is None
