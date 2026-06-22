from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import FundingInstrument
from trading.shared_kernel.primitives import InstrumentId, MarketId


class FundingInstrumentUniverseStore(Protocol):
    def upsert_funding_instruments(self, rows: Sequence[FundingInstrument]) -> None:
        ...

    def list_tradable_funding_instruments(
        self,
        *,
        market_ids: Sequence[MarketId],
    ) -> Sequence[FundingInstrument]:
        ...

    def get_funding_instrument(self, instrument_id: InstrumentId) -> FundingInstrument | None:
        ...
