from __future__ import annotations

from typing import Any, Iterable, Mapping

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_data import RefMarketRow
from trading.contexts.market_data.application.ports.stores.market_ref_writer import MarketRefWriter
from trading.shared_kernel.primitives.market_id import MarketId


class ClickHouseMarketRefWriter(MarketRefWriter):
    def __init__(self, *, gateway: ClickHouseGateway, database: str) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseMarketRefWriter requires gateway")
        if not database.strip():
            raise ValueError("ClickHouseMarketRefWriter requires non-empty database")
        self._gw = gateway
        self._db = database

    def existing_market_ids(self, ids: Iterable[MarketId]) -> set[int]:
        ids_list = [i.value for i in ids]
        if not ids_list:
            return set()

        query = f"""
            SELECT market_id
            FROM {self._db}.ref_market
            WHERE market_id IN {{ids:Array(UInt16)}}
        """.strip()

        rows = self._gw.select(query, parameters={"ids": ids_list})
        out: set[int] = set()
        for r in rows:
            v = r.get("market_id")
            if isinstance(v, int):
                out.add(v)
        return out

    def insert(self, rows: Iterable[RefMarketRow]) -> None:
        payload: list[Mapping[str, Any]] = []
        for r in rows:
            payload.append(
                {
                    "market_id": r.market_id.value,
                    "exchange_name": r.exchange_name,
                    "market_type": r.market_type,
                    "market_code": r.market_code,
                    "is_enabled": r.is_enabled,
                    "count_symbols": r.count_symbols,
                    "updated_at": r.updated_at.value,
                }
            )

        self._gw.insert_rows(f"{self._db}.ref_market", payload)
