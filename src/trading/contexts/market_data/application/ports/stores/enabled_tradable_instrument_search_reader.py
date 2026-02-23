from __future__ import annotations

from typing import Protocol, Sequence

from trading.shared_kernel.primitives import InstrumentId, MarketId


class EnabledTradableInstrumentSearchReader(Protocol):
    """
    Read contract for market-scoped enabled/tradable instrument prefix search.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_tradable_instrument_search_reader.py
      - apps/api/routes/market_data_reference.py
    """

    def search_enabled_tradable_by_market(
        self,
        *,
        market_id: MarketId,
        symbol_prefix: str | None,
        limit: int,
    ) -> Sequence[InstrumentId]:
        """
        Search enabled tradable instruments inside one market.

        Parameters:
        - market_id: target market identifier.
        - symbol_prefix: optional uppercase prefix filter; `None` means no filter.
        - limit: max number of rows to return.

        Returns:
        - Sequence of instrument ids matching filters.

        Assumptions/Invariants:
        - Unknown or disabled market may return empty list.
        - Storage adapter enforces `status='ENABLED'` and `is_tradable=1`.

        Errors/Exceptions:
        - Propagates storage-specific query errors.

        Side effects:
        - Executes one storage read query.
        """
        ...
