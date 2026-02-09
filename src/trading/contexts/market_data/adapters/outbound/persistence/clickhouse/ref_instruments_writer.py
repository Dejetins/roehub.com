from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefEnrichmentSnapshot,
    InstrumentRefEnrichmentUpsert,
    InstrumentRefUpsert,
)
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol


class ClickHouseInstrumentRefWriter(InstrumentRefWriter):
    def __init__(self, *, gateway: ClickHouseGateway, database: str) -> None:
        """
        Store ClickHouse dependencies for ref_instruments writes.

        Parameters:
        - gateway: gateway abstraction used for SQL execution.
        - database: target ClickHouse database name.

        Returns:
        - None.

        Assumptions/Invariants:
        - Gateway is non-null and database is non-empty.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
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
        """
        Read latest status/tradable flags for one market-symbol set.

        Parameters:
        - market_id: market id filter.
        - symbols: symbols to fetch.

        Returns:
        - Mapping `symbol -> (status, is_tradable)`.

        Assumptions/Invariants:
        - Empty symbol input returns empty mapping without querying storage.

        Errors/Exceptions:
        - Propagates gateway query errors.

        Side effects:
        - Executes one ClickHouse SELECT query when symbols are provided.
        """
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
        """
        Insert status/tradable upsert rows into `ref_instruments`.

        Parameters:
        - rows: status and tradable updates.

        Returns:
        - None.

        Assumptions/Invariants:
        - Insert payload can be empty.

        Errors/Exceptions:
        - Propagates gateway insert errors.

        Side effects:
        - Executes ClickHouse INSERT when payload is non-empty.
        """
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

    def existing_latest_enrichment(
        self,
        *,
        market_id: MarketId,
        symbols: Sequence[Symbol],
    ) -> Mapping[str, InstrumentRefEnrichmentSnapshot]:
        """
        Read latest enrichment fields for one market-symbol set.

        Parameters:
        - market_id: market id filter.
        - symbols: symbols to fetch.

        Returns:
        - Mapping `symbol -> InstrumentRefEnrichmentSnapshot`.

        Assumptions/Invariants:
        - Empty symbol input returns empty mapping without storage queries.

        Errors/Exceptions:
        - Propagates gateway query errors.

        Side effects:
        - Executes one ClickHouse SELECT query when symbols are provided.
        """
        if not symbols:
            return {}

        sym_list = [str(s) for s in symbols]

        query = f"""
            SELECT
                symbol,
                status,
                is_tradable,
                base_asset,
                quote_asset,
                price_step,
                qty_step,
                min_notional
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

        out: dict[str, InstrumentRefEnrichmentSnapshot] = {}
        for row in rows:
            symbol = row.get("symbol")
            status = row.get("status")
            is_tradable = row.get("is_tradable")
            if not (
                isinstance(symbol, str)
                and isinstance(status, str)
                and isinstance(is_tradable, int)
            ):
                continue
            out[symbol] = InstrumentRefEnrichmentSnapshot(
                status=status,
                is_tradable=is_tradable,
                base_asset=row.get("base_asset"),
                quote_asset=row.get("quote_asset"),
                price_step=_as_optional_float(row.get("price_step")),
                qty_step=_as_optional_float(row.get("qty_step")),
                min_notional=_as_optional_float(row.get("min_notional")),
            )
        return out

    def upsert_enrichment(self, rows: Iterable[InstrumentRefEnrichmentUpsert]) -> None:
        """
        Insert enrichment rows preserving status/tradable flags.

        Parameters:
        - rows: enrichment updates containing base/quote/steps/notional fields.

        Returns:
        - None.

        Assumptions/Invariants:
        - Payload rows include status/is_tradable to avoid default-value regression.

        Errors/Exceptions:
        - Propagates gateway insert errors.

        Side effects:
        - Executes ClickHouse INSERT when payload is non-empty.
        """
        payload: list[Mapping[str, Any]] = []
        for r in rows:
            payload.append(
                {
                    "market_id": r.market_id.value,
                    "symbol": str(r.symbol),
                    "status": r.status,
                    "is_tradable": r.is_tradable,
                    "base_asset": r.base_asset,
                    "quote_asset": r.quote_asset,
                    "price_step": r.price_step,
                    "qty_step": r.qty_step,
                    "min_notional": r.min_notional,
                    "updated_at": r.updated_at.value,
                }
            )
        self._gw.insert_rows(f"{self._db}.ref_instruments", payload)


def _as_optional_float(value: Any) -> float | None:
    """
    Convert ClickHouse scalar value into optional float.

    Parameters:
    - value: raw value from gateway row mapping.

    Returns:
    - `float(value)` for numeric inputs, otherwise `None`.

    Assumptions/Invariants:
    - Non-numeric values are treated as missing optional fields.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
