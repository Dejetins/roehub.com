from __future__ import annotations

from typing import Any, Iterable, Mapping

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_data import InstrumentRefUpsert
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)


class ClickHouseInstrumentRefWriter(InstrumentRefWriter):
    def __init__(self, *, gateway: ClickHouseGateway, database: str) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseInstrumentRefWriter requires gateway")
        if not database.strip():
            raise ValueError("ClickHouseInstrumentRefWriter requires non-empty database")
        self._gw = gateway
        self._db = database

    def upsert(self, rows: Iterable[InstrumentRefUpsert]) -> None:
        payload: list[Mapping[str, Any]] = []
        for r in rows:
            payload.append(
                {
                    "market_id": r.market_id.value,
                    "symbol": str(r.symbol),
                    "status": r.status,
                    "is_tradable": r.is_tradable,
                    "updated_at": r.updated_at.value,
                }
            )

        self._gw.insert_rows(f"{self._db}.ref_instruments", payload)
