from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import ExchangeInstrumentMetadata
from trading.shared_kernel.primitives import MarketId


class InstrumentMetadataSource(Protocol):
    """
    Source port returning exchange instrument metadata for enrichment job.

    Contract:
    - list_for_market(market_id) returns metadata rows for the requested market.
    """

    def list_for_market(self, market_id: MarketId) -> Sequence[ExchangeInstrumentMetadata]:
        """
        Fetch current exchange metadata rows for one market.

        Parameters:
        - market_id: market identity from runtime config.

        Returns:
        - Sequence of metadata rows keyed by `(market_id, symbol)`.
        """
        ...
