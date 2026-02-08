from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trading.contexts.market_data.adapters.outbound.clients.files.parquet_candle_ingest_source import (  # noqa: E501
    ParquetCandleIngestSource,
    PyArrowParquetScanner,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketDataRuntimeConfig,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)


@dataclass(frozen=True, slots=True)
class ParquetSourceFactory:
    clock: Clock
    cfg: MarketDataRuntimeConfig

    def source(self, *, paths: Sequence[str], batch_size: int | None) -> CandleIngestSource:
        """
        Build parquet ingestion source for CLI backfill wiring.

        Parameters:
        - paths: parquet file or directory paths.
        - batch_size: CLI writer batch size; kept for API compatibility, not used for scanner.

        Returns:
        - Configured `CandleIngestSource` backed by parquet scanner.

        Assumptions/Invariants:
        - runtime config is already loaded and validated.

        Errors/Exceptions:
        - Propagates scanner/source initialization errors.

        Side effects:
        - None.
        """
        scanner = PyArrowParquetScanner(list(paths))

        # batch_size в parquet-сканере — это batch чтения.
        # CLI batch_size управляет batching use-case записи; чтение оставляем разумным.
        scanner_batch_size = 50_000

        return ParquetCandleIngestSource(
            scanner=scanner,
            cfg=self.cfg,
            clock=self.clock,
            batch_size=scanner_batch_size,
        )
