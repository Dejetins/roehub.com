from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import FundingRateRecord
from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp


class FundingRateWriter(Protocol):
    def write_funding_rates(self, rows: Sequence[FundingRateRecord]) -> None:
        ...

    def latest_funding_time(self, instrument_id: InstrumentId) -> UtcTimestamp | None:
        ...
