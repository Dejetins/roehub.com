from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import InstrumentId, TimeRange


@dataclass(frozen=True, slots=True)
class Backfill1mCommand:
    """
    Backfill1mCommand — команда на backfill 1m свечей по инструменту и диапазону.

    См. docs/architecture/market_data/market-data-use-case-backfill-1m.md
    """

    instrument_id: InstrumentId
    time_range: TimeRange

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mCommand requires instrument_id")
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mCommand requires time_range")
