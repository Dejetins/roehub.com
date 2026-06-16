from __future__ import annotations

from typing import Any, Mapping

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseBTCUSDTMarketReadinessReader,
)


class _FakeGateway:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def select(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, parameters or {}))
        return self.rows

    def insert_rows(self, table: str, rows: list[Mapping[str, Any]]) -> None:
        _ = table
        _ = rows
        raise AssertionError("reader must not write")


def test_clickhouse_btcusdt_market_readiness_reader_maps_latest_reference_rows() -> None:
    gateway = _FakeGateway(
        rows=[
            {
                "market_id": 1,
                "exchange_name": "binance",
                "market_type": "spot",
                "market_code": "binance:spot",
                "market_enabled": 1,
                "symbol": "BTCUSDT",
                "status": "ENABLED",
                "is_tradable": 1,
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "price_step": "0.01",
                "qty_step": "0.00001",
                "min_notional": "10",
            }
        ]
    )
    reader = ClickHouseBTCUSDTMarketReadinessReader(gateway=gateway)  # type: ignore[arg-type]

    rows = reader.list_btcusdt_reference_rows()

    assert len(rows) == 1
    assert rows[0].market_id is not None
    assert rows[0].market_id.value == 1
    assert rows[0].instrument_key == "binance:spot:BTCUSDT"
    assert rows[0].market_enabled is True
    assert rows[0].price_step == 0.01
    assert rows[0].qty_step == 0.00001
    assert rows[0].min_notional == 10.0
    query, parameters = gateway.calls[0]
    assert "ref_market" in query
    assert "ref_instruments" in query
    assert "symbol = %(symbol)s" in query
    assert parameters["symbol"] == "BTCUSDT"
    assert set(parameters["market_codes"]) == {
        "binance:spot",
        "binance:futures",
        "bybit:spot",
        "bybit:futures",
    }
