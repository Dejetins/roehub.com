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

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Record insert call parameters for assertions."""
        self.insert_calls.append((table, rows))

    def select(self, query: str, parameters: Mapping[str, Any]):  # noqa: ANN001
        """Return empty rows for tests that do not need selects."""
        _ = query
        _ = parameters
        return []


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
