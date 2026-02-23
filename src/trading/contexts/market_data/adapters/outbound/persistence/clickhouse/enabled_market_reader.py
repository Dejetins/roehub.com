from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
from trading.contexts.market_data.application.ports.stores.enabled_market_reader import (
    EnabledMarketReader,
)
from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class ClickHouseEnabledMarketReader(EnabledMarketReader):
    """
    ClickHouse adapter reading enabled markets from latest `ref_market` state.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
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
        - Gateway is non-null.
        - Database name is non-empty.

        Errors/Exceptions:
        - Raises `ValueError` when constructor arguments are invalid.

        Side effects:
        - None.
        """
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseEnabledMarketReader requires gateway")
        if not self.database.strip():
            raise ValueError("ClickHouseEnabledMarketReader requires non-empty database")

    def list_enabled_markets(self) -> Sequence[EnabledMarketReference]:
        """
        Read enabled markets from latest rows without using ClickHouse `FINAL`.

        Parameters:
        - None.

        Returns:
        - Sequence of enabled market rows ordered by `market_id ASC`.

        Assumptions/Invariants:
        - Latest row per market is selected with `ORDER BY updated_at DESC LIMIT 1 BY market_id`.

        Errors/Exceptions:
        - Propagates gateway query errors.
        - Raises `ValueError` on malformed market_id values.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        query = f"""
        SELECT market_id, exchange_name, market_type, market_code
        FROM
        (
            SELECT
                market_id,
                exchange_name,
                market_type,
                market_code,
                is_enabled,
                updated_at
            FROM {self.database}.ref_market
            ORDER BY updated_at DESC
            LIMIT 1 BY market_id
        )
        WHERE is_enabled = 1
        ORDER BY market_id ASC
        """
        rows = self.gateway.select(query, {})
        out: list[EnabledMarketReference] = []
        for row in rows:
            out.append(
                EnabledMarketReference(
                    market_id=MarketId(int(row["market_id"])),
                    exchange_name=str(row["exchange_name"]),
                    market_type=str(row["market_type"]),
                    market_code=str(row["market_code"]),
                )
            )
        return out
