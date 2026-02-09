from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefEnrichmentSnapshot,
    InstrumentRefEnrichmentUpsert,
    InstrumentRefUpsert,
)
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol


class InstrumentRefWriter(Protocol):
    def existing_latest(
        self,
        *,
        market_id: MarketId,
        symbols: Sequence[Symbol],
    ) -> Mapping[str, tuple[str, int]]:
        """
        Return latest state for given (market_id, symbols).

        Mapping key: normalized symbol string.
        Mapping value: (status, is_tradable).
        """
        ...

    def upsert(self, rows: Iterable[InstrumentRefUpsert]) -> None:
        """
        Insert status/tradable updates into `ref_instruments`.

        Parameters:
        - rows: upsert rows with status and tradable flags.

        Returns:
        - None.
        """
        ...

    def existing_latest_enrichment(
        self,
        *,
        market_id: MarketId,
        symbols: Sequence[Symbol],
    ) -> Mapping[str, InstrumentRefEnrichmentSnapshot]:
        """
        Return latest enrichment state for given `(market_id, symbols)`.

        Parameters:
        - market_id: market id filter.
        - symbols: symbol set to read.

        Returns:
        - Mapping `symbol -> latest enrichment snapshot`.
        """
        ...

    def upsert_enrichment(self, rows: Iterable[InstrumentRefEnrichmentUpsert]) -> None:
        """
        Insert enrichment updates (base/quote/steps/min_notional) into `ref_instruments`.

        Parameters:
        - rows: enrichment rows; status and tradable flags must be preserved by caller.

        Returns:
        - None.
        """
        ...
