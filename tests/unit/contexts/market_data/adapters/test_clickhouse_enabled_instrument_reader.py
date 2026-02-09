from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseEnabledInstrumentReader,
)


class _FakeGateway:
    """Gateway fake capturing SELECT query details and returning predefined rows."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Store deterministic rows returned by `select`.

        Parameters:
        - rows: sequence of mapping rows emitted on `select`.
        """
        self._rows = list(rows)
        self.select_calls: list[tuple[str, Mapping[str, Any]]] = []

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Fail fast when adapter unexpectedly calls insert path.

        Parameters:
        - table: destination table name.
        - rows: payload rows.
        """
        _ = table
        _ = rows
        raise AssertionError("insert_rows is not expected in enabled reader tests")

    def select(self, query: str, parameters: Mapping[str, Any]):  # noqa: ANN001
        """
        Record query and return preconfigured rows.

        Parameters:
        - query: SQL query text.
        - parameters: query bind parameters.
        """
        self.select_calls.append((query, parameters))
        return list(self._rows)


def test_enabled_reader_uses_latest_state_query_and_maps_rows() -> None:
    """Ensure enabled-reader query deduplicates by latest row and maps to InstrumentId list."""
    gateway = _FakeGateway(
        rows=[
            {"market_id": 1, "symbol": "BTCUSDT"},
            {"market_id": 3, "symbol": "ADAUSDT"},
        ]
    )
    reader = ClickHouseEnabledInstrumentReader(gateway=gateway, database="market_data")

    rows = reader.list_enabled_tradable()

    assert len(rows) == 2
    assert str(rows[0].symbol) == "BTCUSDT"
    assert str(rows[1].symbol) == "ADAUSDT"
    assert len(gateway.select_calls) == 1
    query, parameters = gateway.select_calls[0]
    assert parameters == {}
    assert "ORDER BY updated_at DESC" in query
    assert "LIMIT 1 BY market_id, symbol" in query
    assert "WHERE status = 'ENABLED'" in query
    assert "AND is_tradable = 1" in query


def test_enabled_reader_raises_on_invalid_market_id_type() -> None:
    """Ensure malformed `market_id` values are surfaced as ValueError."""
    gateway = _FakeGateway(rows=[{"market_id": "bad", "symbol": "BTCUSDT"}])
    reader = ClickHouseEnabledInstrumentReader(gateway=gateway, database="market_data")

    with pytest.raises(ValueError):
        reader.list_enabled_tradable()
