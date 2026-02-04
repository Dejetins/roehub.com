from __future__ import annotations

from typing import List, Protocol

from trading.contexts.market_data.application.dto import (
    Backfill1mCommand,
    Backfill1mReport,
    CandleWithMeta,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter


class Backfill1mCandles(Protocol):
    """
    Контракт use-case: backfill 1m свечей.

    См. docs/architecture/market_data/market-data-use-case-backfill-1m.md
    """

    def run(self, command: Backfill1mCommand) -> Backfill1mReport:
        ...


class Backfill1mCandlesUseCase(Backfill1mCandles):
    """
    Реализация use-case backfill 1m свечей.

    Правила v1:
    - читает поток из CandleIngestSource по TimeRange [start, end)
    - пишет в RawKlineWriter батчами
    - не делает read-back и skip-existing
    """

    def __init__(
        self,
        source: CandleIngestSource,
        writer: RawKlineWriter,
        clock: Clock,
        batch_size: int = 50_000,
    ) -> None:
        if source is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mCandlesUseCase requires source")
        if writer is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mCandlesUseCase requires writer")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("Backfill1mCandlesUseCase requires clock")
        if batch_size <= 0:
            raise ValueError("Backfill1mCandlesUseCase requires batch_size > 0")

        self._source = source
        self._writer = writer
        self._clock = clock
        self._batch_size = batch_size

    def run(self, command: Backfill1mCommand) -> Backfill1mReport:
        started_at = self._clock.now()

        candles_read = 0
        rows_written = 0
        batches_written = 0

        batch: List[CandleWithMeta] = []

        for row in self._source.stream_1m(command.instrument_id, command.time_range):
            candles_read += 1
            batch.append(row)

            if len(batch) >= self._batch_size:
                self._writer.write_1m(batch)
                rows_written += len(batch)
                batches_written += 1
                batch = []

        if batch:
            self._writer.write_1m(batch)
            rows_written += len(batch)
            batches_written += 1

        finished_at = self._clock.now()

        return Backfill1mReport(
            instrument_id=command.instrument_id,
            time_range=command.time_range,
            started_at=started_at,
            finished_at=finished_at,
            candles_read=candles_read,
            rows_written=rows_written,
            batches_written=batches_written,
        )
