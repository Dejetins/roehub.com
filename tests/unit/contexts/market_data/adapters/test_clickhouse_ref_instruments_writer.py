from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseInstrumentRefWriter,
)
from trading.contexts.market_data.application.dto import InstrumentRefEnrichmentUpsert
from trading.shared_kernel.primitives import MarketId, Symbol, UtcTimestamp


class _FakeGateway:
    """Gateway fake capturing insert/select calls for writer adapter tests."""

    def __init__(self) -> None:
        """Initialize empty call history."""
        self.insert_calls: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
        self.select_calls: list[tuple[str, Mapping[str, Any]]] = []
        self.select_rows: list[Mapping[str, Any]] = []

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Record insert call parameters for assertions."""
        self.insert_calls.append((table, rows))

    def select(self, query: str, parameters: Mapping[str, Any]):  # noqa: ANN001
        """Record select call parameters and return preconfigured rows."""
        self.select_calls.append((query, parameters))
        return list(self.select_rows)


def test_upsert_enrichment_writes_enrichment_columns_and_status_flags() -> None:
    """Ensure enrichment writer payload preserves status/tradable along with enrichment values."""
    gateway = _FakeGateway()
    writer = ClickHouseInstrumentRefWriter(gateway=gateway, database="market_data")

    writer.upsert_enrichment(
        [
            InstrumentRefEnrichmentUpsert(
                market_id=MarketId(1),
                symbol=Symbol("BTCUSDT"),
                status="ENABLED",
                is_tradable=1,
                base_asset="BTC",
                quote_asset="USDT",
                price_step=0.01,
                qty_step=0.0001,
                min_notional=10.0,
                updated_at=UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc)),
            )
        ]
    )

    assert len(gateway.insert_calls) == 1
    table, rows = gateway.insert_calls[0]
    assert table == "market_data.ref_instruments"
    assert rows[0]["status"] == "ENABLED"
    assert rows[0]["is_tradable"] == 1
    assert rows[0]["base_asset"] == "BTC"
    assert rows[0]["quote_asset"] == "USDT"
    assert rows[0]["price_step"] == 0.01
    assert rows[0]["qty_step"] == 0.0001
    assert rows[0]["min_notional"] == 10.0


def test_existing_latest_enrichment_reads_latest_snapshot_fields() -> None:
    """Ensure writer maps latest enrichment SELECT rows into typed snapshot mapping."""
    gateway = _FakeGateway()
    gateway.select_rows = [
        {
            "symbol": "BTCUSDT",
            "status": "ENABLED",
            "is_tradable": 1,
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "price_step": 0.01,
            "qty_step": 0.0001,
            "min_notional": 10.0,
        }
    ]
    writer = ClickHouseInstrumentRefWriter(gateway=gateway, database="market_data")

    rows = writer.existing_latest_enrichment(
        market_id=MarketId(1),
        symbols=[Symbol("BTCUSDT")],
    )

    assert len(gateway.select_calls) == 1
    query, parameters = gateway.select_calls[0]
    assert "LIMIT 1 BY symbol" in query
    assert parameters["market_id"] == 1
    assert parameters["symbols"] == ["BTCUSDT"]
    assert rows["BTCUSDT"].status == "ENABLED"
    assert rows["BTCUSDT"].is_tradable == 1
    assert rows["BTCUSDT"].base_asset == "BTC"
    assert rows["BTCUSDT"].quote_asset == "USDT"
    assert rows["BTCUSDT"].price_step == 0.01
