from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_data import InstrumentRefUpsert
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol


class ClickHouseInstrumentRefWriter(InstrumentRefWriter):
    def __init__(self, *, gateway: ClickHouseGateway, database: str) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseInstrumentRefWriter requires gateway")
        if not database.strip():
            raise ValueError("ClickHouseInstrumentRefWriter requires non-empty database")
        self._gw = gateway
        self._db = database

    def existing_latest(
        self,
        *,
        market_id: MarketId,
        symbols: Sequence[Symbol],
    ) -> Mapping[str, tuple[str, int]]:
        if not symbols:
            return {}

        sym_list = [str(s) for s in symbols]

        query = f"""
            SELECT
                symbol,
                status,
                is_tradable
            FROM {self._db}.ref_instruments
            WHERE market_id = {{market_id:UInt16}}
              AND symbol IN {{symbols:Array(String)}}
            ORDER BY updated_at DESC
            LIMIT 1 BY symbol
        """.strip()

        rows = self._gw.select(
            query,
            parameters={
                "market_id": market_id.value,
                "symbols": sym_list,
            },
        )

        out: dict[str, tuple[str, int]] = {}
        for r in rows:
            symbol = r.get("symbol")
            status = r.get("status")
            is_tradable = r.get("is_tradable")
            if isinstance(symbol, str) and isinstance(status, str) and isinstance(is_tradable, int):
                out[symbol] = (status, is_tradable)

        return out

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
