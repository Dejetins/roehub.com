from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.ports.stores.enabled_tradable_instrument_search_reader import (  # noqa: E501
    EnabledTradableInstrumentSearchReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


@dataclass(frozen=True, slots=True)
class ClickHouseEnabledTradableInstrumentSearchReader(EnabledTradableInstrumentSearchReader):
    """
    ClickHouse adapter for market-scoped enabled/tradable instrument search.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/
        enabled_tradable_instrument_search_reader.py
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
      - migrations/clickhouse/market_data_ddl.sql
    """

    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self) -> None:
        """
        Validate adapter constructor invariants.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Gateway is non-null and database name is non-empty.

        Errors/Exceptions:
        - Raises `ValueError` for invalid constructor arguments.

        Side effects:
        - None.
        """
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseEnabledTradableInstrumentSearchReader requires gateway")
        if not self.database.strip():
            raise ValueError(
                "ClickHouseEnabledTradableInstrumentSearchReader requires non-empty database"
            )

    def search_enabled_tradable_by_market(
        self,
        *,
        market_id: MarketId,
        symbol_prefix: str | None,
        limit: int,
    ) -> Sequence[InstrumentId]:
        """
        Search latest enabled tradable instruments in one market with optional prefix.

        Parameters:
        - market_id: target market id.
        - symbol_prefix: optional uppercase symbol prefix, `None` means no prefix filter.
        - limit: maximum number of rows to return.

        Returns:
        - Sequence of matching instrument ids ordered by `symbol ASC`.

        Assumptions/Invariants:
        - Unknown/disabled market returns empty list via market latest-state join.
        - Latest-state reads do not use ClickHouse `FINAL`.

        Errors/Exceptions:
        - Propagates gateway query errors.
        - Raises `ValueError` on malformed rows.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        query, parameters = _build_search_query(
            database=self.database,
            market_id=market_id,
            symbol_prefix=symbol_prefix,
            limit=limit,
        )
        rows = self.gateway.select(query, parameters)
        out: list[InstrumentId] = []
        for row in rows:
            out.append(
                InstrumentId(
                    market_id=MarketId(int(row["market_id"])),
                    symbol=Symbol(str(row["symbol"])),
                )
            )
        return out


def _build_search_query(
    *,
    database: str,
    market_id: MarketId,
    symbol_prefix: str | None,
    limit: int,
) -> tuple[str, Mapping[str, Any]]:
    """
    Build SQL text and bind parameters for market-scoped instrument search.

    Parameters:
    - database: ClickHouse database name.
    - market_id: target market id.
    - symbol_prefix: optional symbol prefix.
    - limit: max rows.

    Returns:
    - Tuple of `(query, parameters)` for gateway execution.

    Assumptions/Invariants:
    - Query enforces latest-state reads with `ORDER BY updated_at DESC` and `LIMIT 1 BY`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    base_parameters: dict[str, Any] = {
        "market_id": int(market_id.value),
        "limit": int(limit),
    }
    base_query = f"""
    SELECT market_id, symbol
    FROM
    (
        SELECT
            market_id,
            symbol,
            status,
            is_tradable,
            updated_at
        FROM {database}.ref_instruments
        WHERE market_id = %(market_id)s
        ORDER BY updated_at DESC
        LIMIT 1 BY market_id, symbol
    ) AS instruments
    INNER JOIN
    (
        SELECT
            market_id,
            is_enabled,
            updated_at
        FROM {database}.ref_market
        WHERE market_id = %(market_id)s
        ORDER BY updated_at DESC
        LIMIT 1 BY market_id
    ) AS markets ON markets.market_id = instruments.market_id
    WHERE markets.is_enabled = 1
      AND status = 'ENABLED'
      AND is_tradable = 1
    """

    if symbol_prefix is not None:
        query = (
            f"{base_query}\n"
            "      AND symbol LIKE %(symbol_prefix)s\n"
            "ORDER BY symbol ASC\n"
            "LIMIT %(limit)s"
        )
        parameters = {
            **base_parameters,
            "symbol_prefix": f"{symbol_prefix}%",
        }
        return query, parameters

    query = f"{base_query}\nORDER BY symbol ASC\nLIMIT %(limit)s"
    return query, base_parameters
