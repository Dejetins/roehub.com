from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class Backfill1mReport:
    """
    Backfill1mReport — отчёт выполнения backfill для CLI/логов/наблюдаемости.

    См. docs/architecture/market_data/market-data-use-case-backfill-1m.md
    """

    instrument_id: InstrumentId
    time_range: TimeRange

    started_at: UtcTimestamp
    finished_at: UtcTimestamp

    candles_read: int
    rows_written: int
    batches_written: int

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mReport requires instrument_id")
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mReport requires time_range")
        if self.started_at is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mReport requires started_at")
        if self.finished_at is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mReport requires finished_at")

        if self.candles_read < 0:
            raise ValueError("Backfill1mReport requires candles_read >= 0")
        if self.rows_written < 0:
            raise ValueError("Backfill1mReport requires rows_written >= 0")
        if self.batches_written < 0:
            raise ValueError("Backfill1mReport requires batches_written >= 0")
