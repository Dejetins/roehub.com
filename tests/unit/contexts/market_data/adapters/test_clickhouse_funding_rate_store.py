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

    def insert_rows(self, table, rows):
        self.inserts.append((table, list(rows)))

    def select(self, query, parameters):
        self.select_rows.append((query, dict(parameters)))
        return []


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
