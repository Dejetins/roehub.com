from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
from trading.contexts.market_data.application.ports.stores.enabled_market_reader import (
    EnabledMarketReader,
)


@dataclass(frozen=True, slots=True)
class ListEnabledMarketsUseCase:
    """
    Use-case returning enabled markets for Market Data reference API v1.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_market_reader.py
      - apps/api/routes/market_data_reference.py
    """

    reader: EnabledMarketReader

    def __post_init__(self) -> None:
        """
        Validate mandatory dependency.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Reader port is non-null.

        Errors/Exceptions:
        - Raises `ValueError` when dependency is missing.

        Side effects:
        - None.
        """
        if self.reader is None:  # type: ignore[truthy-bool]
            raise ValueError("ListEnabledMarketsUseCase requires reader")

    def execute(self) -> tuple[EnabledMarketReference, ...]:
        """
        List enabled markets with deterministic ordering by market_id ascending.

        Parameters:
        - None.

        Returns:
        - Tuple of enabled market rows sorted by `market_id ASC`.

        Assumptions/Invariants:
        - Reader returns only enabled markets.

        Errors/Exceptions:
        - Propagates reader/storage errors.

        Side effects:
        - Executes one read through market reader port.
        """
        markets = list(self.reader.list_enabled_markets())
        markets.sort(key=_market_sort_key)
        return tuple(markets)


def _market_sort_key(row: EnabledMarketReference) -> int:
    """
    Build deterministic sort key for enabled market rows.

    Parameters:
    - row: enabled market read-model row.

    Returns:
    - Integer market identifier used for ascending sort.

    Assumptions/Invariants:
    - `row.market_id` is validated `MarketId`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return int(row.market_id.value)
