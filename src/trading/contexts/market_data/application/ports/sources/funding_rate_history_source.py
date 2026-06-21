from __future__ import annotations

from typing import Protocol, Sequence

from trading.contexts.market_data.application.dto import FundingRateRecord
from trading.shared_kernel.primitives import InstrumentId, TimeRange


class FundingRateHistorySource(Protocol):
    def list_funding_rates(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        funding_interval_minutes: int,
        funding_interval_source: str,
    ) -> Sequence[FundingRateRecord]:
        ...
