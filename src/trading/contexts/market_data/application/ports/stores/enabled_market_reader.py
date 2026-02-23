from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference


class EnabledMarketReader(Protocol):
    """
    Read contract for enabled markets used by Market Data reference API v1.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_market_reader.py
      - apps/api/routes/market_data_reference.py
    """

    def list_enabled_markets(self) -> Sequence[EnabledMarketReference]:
        """
        Return enabled market rows from reference storage.

        Parameters:
        - None.

        Returns:
        - Sequence of enabled market reference rows.

        Assumptions/Invariants:
        - Returned rows contain only enabled markets from latest storage state.
        - Ordering may be implementation-defined and is stabilized in use-case layer.

        Errors/Exceptions:
        - Propagates storage-specific reader errors.

        Side effects:
        - May execute one storage read query.
        """
        ...
