from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trading.contexts.market_data.adapters.outbound.clients.files.parquet_candle_ingest_source import (  # noqa: E501
    ParquetCandleIngestSource,
    PyArrowParquetScanner,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)


@dataclass(frozen=True, slots=True)
class ParquetSourceFactory:
    clock: Clock

    def source(self, *, paths: Sequence[str], batch_size: int | None) -> CandleIngestSource:
        scanner = PyArrowParquetScanner(list(paths))

        # batch_size в parquet-сканере — это batch чтения.
        # CLI batch_size управляет batching use-case записи; чтение оставляем разумным.
        scanner_batch_size = 50_000

        return ParquetCandleIngestSource(
            scanner=scanner,
            clock=self.clock,
            batch_size=scanner_batch_size,
        )
