from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.market_data.application.dto.reference_data import RefMarketRow
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.market_ref_writer import MarketRefWriter
from trading.shared_kernel.primitives.market_id import MarketId


@dataclass(frozen=True, slots=True)
class SeedRefMarketReport:
    inserted: int


class SeedRefMarketUseCase:
    """
    Seed 4 markets into market_data.ref_market in an idempotent way (insert-only-missing).
    """

    def __init__(self, *, writer: MarketRefWriter, clock: Clock) -> None:
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("SeedRefMarketUseCase requires writer")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("SeedRefMarketUseCase requires clock")
        self._writer = writer
        self._clock = clock

    def run(self) -> SeedRefMarketReport:
        now = self._clock.now()

        target = [
            RefMarketRow(MarketId(1), "binance", "spot", "binance:spot", 1, 0, now),
            RefMarketRow(MarketId(2), "binance", "futures", "binance:futures", 1, 0, now),
            RefMarketRow(MarketId(3), "bybit", "spot", "bybit:spot", 1, 0, now),
            RefMarketRow(MarketId(4), "bybit", "futures", "bybit:futures", 1, 0, now),
        ]

        existing = self._writer.existing_market_ids([r.market_id for r in target])
        missing = [r for r in target if r.market_id.value not in existing]

        self._writer.insert(missing)
        return SeedRefMarketReport(inserted=len(missing))
