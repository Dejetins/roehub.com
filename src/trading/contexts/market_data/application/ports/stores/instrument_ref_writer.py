from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from trading.contexts.market_data.application.dto.reference_data import InstrumentRefUpsert
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
        ...
