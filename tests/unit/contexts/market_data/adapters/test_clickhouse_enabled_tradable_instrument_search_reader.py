from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseEnabledTradableInstrumentSearchReader,
)
from trading.shared_kernel.primitives import MarketId


class _FakeGateway:
    """
    Gateway fake capturing SQL text and bind parameters for search reader tests.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/contexts/market_data/adapters/
        test_clickhouse_enabled_tradable_instrument_search_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_tradable_instrument_search_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Store deterministic rows returned from select.

        Parameters:
        - rows: rows returned by fake `select`.

        Returns:
        - None.

        Assumptions/Invariants:
        - Row schema matches adapter mapping requirements.

        Errors/Exceptions:
        - None.

        Side effects:
        - Initializes mutable call capture.
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
        - Search reader must not call inserts.

        Errors/Exceptions:
        - Raises `AssertionError` always.

        Side effects:
        - None.
        """
        _ = table, rows
        raise AssertionError("insert_rows is not expected for search reader")

    def select(self, query: str, parameters: Mapping[str, Any]):  # noqa: ANN001
        """
        Capture SQL call and return deterministic configured rows.

        Parameters:
        - query: SQL text.
        - parameters: bind parameters mapping.

        Returns:
        - List of configured rows.

        Assumptions/Invariants:
        - Reader issues one query per request.

        Errors/Exceptions:
        - None.

        Side effects:
        - Appends one entry to captured `select_calls`.
        """
        self.select_calls.append((query, parameters))
        return list(self._rows)


def test_search_reader_uses_latest_state_query_without_prefix_filter() -> None:
    """
    Verify SQL shape for market-scoped search without optional prefix filter.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Latest-state query must avoid `FINAL`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    gateway = _FakeGateway(
        rows=[
            {"market_id": 1, "symbol": "BTCUSDT"},
            {"market_id": 1, "symbol": "ETHUSDT"},
        ]
    )
    reader = ClickHouseEnabledTradableInstrumentSearchReader(
        gateway=gateway,
        database="market_data",
    )

    rows = reader.search_enabled_tradable_by_market(
        market_id=MarketId(1),
        symbol_prefix=None,
        limit=50,
    )

    assert [str(item.symbol) for item in rows] == ["BTCUSDT", "ETHUSDT"]
    assert len(gateway.select_calls) == 1
    query, parameters = gateway.select_calls[0]
    assert parameters == {"market_id": 1, "limit": 50}
    assert "ORDER BY updated_at DESC" in query
    assert "LIMIT 1 BY market_id, symbol" in query
    assert "LIMIT 1 BY market_id" in query
    assert "status = 'ENABLED'" in query
    assert "is_tradable = 1" in query
    assert "ORDER BY symbol ASC" in query
    assert "LIMIT %(limit)s" in query
    assert "symbol LIKE %(symbol_prefix)s" not in query
    assert "FINAL" not in query


def test_search_reader_applies_prefix_filter_and_bind_parameter() -> None:
    """
    Verify optional prefix filter produces `LIKE` clause and `%` suffix bind.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Prefix value is appended with `%` in adapter query params.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    gateway = _FakeGateway(rows=[{"market_id": 1, "symbol": "BTCUSDT"}])
    reader = ClickHouseEnabledTradableInstrumentSearchReader(
        gateway=gateway,
        database="market_data",
    )

    rows = reader.search_enabled_tradable_by_market(
        market_id=MarketId(1),
        symbol_prefix="BTC",
        limit=10,
    )

    assert len(rows) == 1
    query, parameters = gateway.select_calls[0]
    assert "symbol LIKE %(symbol_prefix)s" in query
    assert parameters == {
        "market_id": 1,
        "limit": 10,
        "symbol_prefix": "BTC%",
    }


def test_search_reader_raises_on_invalid_market_id_type() -> None:
    """
    Verify malformed row market_id is surfaced as ValueError during mapping.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - `MarketId` primitive validates mapped market id value.

    Errors/Exceptions:
    - Expects `ValueError`.

    Side effects:
    - None.
    """
    gateway = _FakeGateway(rows=[{"market_id": "bad", "symbol": "BTCUSDT"}])
    reader = ClickHouseEnabledTradableInstrumentSearchReader(
        gateway=gateway,
        database="market_data",
    )

    with pytest.raises(ValueError):
        reader.search_enabled_tradable_by_market(
            market_id=MarketId(1),
            symbol_prefix=None,
            limit=1,
        )
