from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseInstrumentCoverageReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


class _Gateway:
    def __init__(self, candles: int) -> None:
        self.candles = candles
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        self.calls.append((query, parameters))
        return ({"candles": self.candles},)

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        raise AssertionError(f"unexpected insert into {table}: {rows}")


def test_coverage_is_exact_distinct_closed_minutes_and_never_exceeds_one_hundred_percent() -> None:
    gateway = _Gateway(candles=65)
    reader = ClickHouseInstrumentCoverageReader(gateway=gateway, database="roehub")
    start = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)

    snapshot = reader.read(
        instrument_id=InstrumentId(MarketId(2), Symbol("BTCUSDT")),
        expected_start_at=start,
        expected_end_at=start + timedelta(minutes=60),
    )

    assert snapshot.state == "complete"
    assert snapshot.percent == 100.0
    assert "uniqExact" in gateway.calls[0][0]
    assert gateway.calls[0][1]["market_id"] == 2
