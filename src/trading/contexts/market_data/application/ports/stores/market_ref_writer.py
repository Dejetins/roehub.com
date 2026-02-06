from __future__ import annotations

from typing import Iterable, Protocol

from trading.contexts.market_data.application.dto.reference_data import RefMarketRow
from trading.shared_kernel.primitives.market_id import MarketId


class MarketRefWriter(Protocol):
    def existing_market_ids(self, ids: Iterable[MarketId]) -> set[int]:
        ...

    def insert(self, rows: Iterable[RefMarketRow]) -> None:
        ...
