from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


@dataclass(frozen=True, slots=True)
class ClickHouseEnabledInstrumentReader(EnabledInstrumentReader):
    """
    ClickHouse reader for enabled tradable instruments.

    Parameters:
    - gateway: thin query/insert gateway abstraction.
    - database: ClickHouse database name containing `ref_instruments`.
    """

    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self) -> None:
        """
        Validate reader constructor arguments.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Gateway is non-null and database name is non-empty.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseEnabledInstrumentReader requires gateway")
        if not self.database.strip():
            raise ValueError("ClickHouseEnabledInstrumentReader requires non-empty database")

    def list_enabled_tradable(self) -> Sequence[InstrumentId]:
        """
        Fetch enabled tradable instruments from latest `ref_instruments` versions.

        Parameters:
        - None.

        Returns:
        - Sequence of instrument ids.

        Assumptions/Invariants:
        - For each `(market_id, symbol)` key only the latest row by `updated_at` is considered.
        - Latest-state filtering uses `LIMIT 1 BY market_id, symbol`.

        Errors/Exceptions:
        - Propagates storage/gateway errors.
        - Raises `ValueError` on malformed market_id values.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        query = f"""
        SELECT market_id, symbol
        FROM
        (
            SELECT
                market_id,
                symbol,
                status,
                is_tradable,
                updated_at
            FROM {self.database}.ref_instruments
            ORDER BY updated_at DESC
            LIMIT 1 BY market_id, symbol
        )
        WHERE status = 'ENABLED'
          AND is_tradable = 1
        """
        rows = self.gateway.select(query, {})
        out: list[InstrumentId] = []
        for row in rows:
            out.append(
                InstrumentId(
                    market_id=MarketId(int(row["market_id"])),
                    symbol=Symbol(str(row["symbol"])),
                )
            )
        return out
