from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseEnabledMarketReader,
)


class _FakeGateway:
    """
    Gateway fake capturing SQL text/params for enabled market reader tests.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/contexts/market_data/adapters/test_clickhouse_enabled_market_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_market_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Store deterministic query result rows.

        Parameters:
        - rows: sequence returned from `select`.

        Returns:
        - None.

        Assumptions/Invariants:
        - Rows follow adapter mapping schema.

        Errors/Exceptions:
        - None.

        Side effects:
        - Initializes call capture state.
        """
        self._rows = list(rows)
        self.select_calls: list[tuple[str, Mapping[str, Any]]] = []

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Fail fast when adapter unexpectedly uses insert path.

        Parameters:
        - table: destination table.
        - rows: payload rows.

        Returns:
        - None.

        Assumptions/Invariants:
        - Reader must never call inserts.

        Errors/Exceptions:
        - Raises `AssertionError` always.

        Side effects:
        - None.
        """
        _ = table, rows
        raise AssertionError("insert_rows is not expected for enabled market reader")

    def select(self, query: str, parameters: Mapping[str, Any]):  # noqa: ANN001
        """
        Capture query call and return configured rows.

        Parameters:
        - query: SQL text.
        - parameters: bind parameters mapping.

        Returns:
        - List of configured rows.

        Assumptions/Invariants:
        - Reader uses single SELECT query per call.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends one entry to `select_calls`.
        """
        self.select_calls.append((query, parameters))
        return list(self._rows)


def test_enabled_market_reader_uses_latest_state_query_and_maps_rows() -> None:
    """
    Verify query shape and row mapping for enabled markets reader.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Latest-state contract requires `ORDER BY updated_at DESC` and `LIMIT 1 BY`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    gateway = _FakeGateway(
        rows=[
            {
                "market_id": 1,
                "exchange_name": "binance",
                "market_type": "spot",
                "market_code": "binance:spot",
            },
            {
                "market_id": 3,
                "exchange_name": "bybit",
                "market_type": "spot",
                "market_code": "bybit:spot",
            },
        ]
    )
    reader = ClickHouseEnabledMarketReader(gateway=gateway, database="market_data")

    rows = reader.list_enabled_markets()

    assert [item.market_id.value for item in rows] == [1, 3]
    assert [item.market_code for item in rows] == ["binance:spot", "bybit:spot"]
    assert len(gateway.select_calls) == 1
    query, parameters = gateway.select_calls[0]
    assert parameters == {}
    assert "ORDER BY updated_at DESC" in query
    assert "LIMIT 1 BY market_id" in query
    assert "WHERE is_enabled = 1" in query
    assert "ORDER BY market_id ASC" in query
    assert "FINAL" not in query


def test_enabled_market_reader_raises_on_invalid_market_id_type() -> None:
    """
    Verify malformed `market_id` values are surfaced as ValueError.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - `MarketId` primitive validates integer-like positive value.

    Errors/Exceptions:
    - Expects `ValueError` on malformed market_id.

    Side effects:
    - None.
    """
    gateway = _FakeGateway(
        rows=[
            {
                "market_id": "bad",
                "exchange_name": "binance",
                "market_type": "spot",
                "market_code": "binance:spot",
            }
        ]
    )
    reader = ClickHouseEnabledMarketReader(gateway=gateway, database="market_data")

    with pytest.raises(ValueError):
        reader.list_enabled_markets()
