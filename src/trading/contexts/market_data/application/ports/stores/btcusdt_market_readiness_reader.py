from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReferenceSnapshot,
)


class BTCUSDTMarketReadinessReferenceReader(Protocol):
    def list_btcusdt_reference_rows(self) -> Sequence[BTCUSDTMarketReferenceSnapshot]: ...
