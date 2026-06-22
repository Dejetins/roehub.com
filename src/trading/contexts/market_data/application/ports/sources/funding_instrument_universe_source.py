from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import FundingInstrument
from trading.shared_kernel.primitives import MarketId


class FundingInstrumentUniverseSource(Protocol):
    def list_funding_instruments(self, market_id: MarketId) -> Sequence[FundingInstrument]:
        ...
