from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    RestFillResult,
    RestFillTask,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.candle_ingest_source import (
    CandleIngestSource,
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.use_cases.time_slicing import (
    slice_time_range_by_utc_days,
)


@dataclass(frozen=True, slots=True)
class RestFillRange1mUseCase:
    """
    Execute one bounded REST fill range and persist only into raw tables.

    Parameters:
    - source: REST-backed candle source returning closed 1m rows.
    - writer: raw writer port used for `raw_*_klines_1m` tables.
    - clock: UTC clock for deterministic timing in reports/tests.
    - max_days_per_insert: maximum UTC-day span for one insert slice.
    - batch_size: raw write batch size.

    Assumptions/Invariants:
    - `max_days_per_insert` is in `[1, 7]` for ClickHouse partition safety.
    - `batch_size` is positive.
    """

    source: CandleIngestSource
    writer: RawKlineWriter
    clock: Clock
    max_days_per_insert: int
    batch_size: int

    def __post_init__(self) -> None:
        """
        Validate required collaborators and runtime parameters.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Collaborators are non-null object references.
        - `max_days_per_insert` and `batch_size` keep inserts bounded.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if self.source is None:  # type: ignore[truthy-bool]
            raise ValueError("RestFillRange1mUseCase requires source")
        if self.writer is None:  # type: ignore[truthy-bool]
            raise ValueError("RestFillRange1mUseCase requires writer")
        if self.clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RestFillRange1mUseCase requires clock")
        if self.max_days_per_insert <= 0 or self.max_days_per_insert > 7:
            raise ValueError("max_days_per_insert must be in [1..7]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def run(self, task: RestFillTask) -> RestFillResult:
        """
        Perform REST fill for one explicit time range.

        Parameters:
        - task: one fill command with instrument/range/reason.

        Returns:
        - Aggregated execution report with read/write counters.

        Assumptions/Invariants:
        - Input range follows half-open semantics `[start, end)`.

        Errors/Exceptions:
        - Propagates source and writer adapter errors.

        Side effects:
        - Reads external REST data and writes raw rows.
        """
        started_at = self.clock.now()
        rows_read = 0
        rows_written = 0
        batches_written = 0

        for chunk in slice_time_range_by_utc_days(
            time_range=task.time_range,
            max_days=self.max_days_per_insert,
        ):
            chunk_read, chunk_written, chunk_batches = self._ingest_chunk(task, chunk)
            rows_read += chunk_read
            rows_written += chunk_written
            batches_written += chunk_batches

        finished_at = self.clock.now()
        return RestFillResult(
            task=task,
            rows_read=rows_read,
            rows_written=rows_written,
            batches_written=batches_written,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _ingest_chunk(self, task: RestFillTask, chunk) -> tuple[int, int, int]:
        """
        Ingest one chunk and write rows in bounded batches.

        Parameters:
        - task: parent task describing instrument and reason.
        - chunk: one UTC subrange produced by time-slicing.

        Returns:
        - Tuple `(rows_read, rows_written, batches_written)` for this chunk.

        Assumptions/Invariants:
        - Source yields rows that belong to `(task.instrument_id, chunk)`.
        - Writer performs append-only insert into raw storage.

        Errors/Exceptions:
        - Propagates source and writer errors.

        Side effects:
        - Writes to raw storage through writer port.
        """
        rows_read = 0
        rows_written = 0
        batches_written = 0
        batch: list[CandleWithMeta] = []

        for row in self.source.stream_1m(task.instrument_id, chunk):
            rows_read += 1
            batch.append(row)

            if len(batch) >= self.batch_size:
                self._write_batch(batch)
                rows_written += len(batch)
                batches_written += 1
                batch = []

        if batch:
            self._write_batch(batch)
            rows_written += len(batch)
            batches_written += 1

        return rows_read, rows_written, batches_written

    def _write_batch(self, rows: Iterable[CandleWithMeta]) -> None:
        """
        Delegate one batch write to raw writer port.

        Parameters:
        - rows: iterable batch of normalized 1m candles.

        Returns:
        - None.

        Assumptions/Invariants:
        - Writer contract accepts iterable of `CandleWithMeta`.

        Errors/Exceptions:
        - Propagates writer errors.

        Side effects:
        - Writes one insert batch into raw storage.
        """
        self.writer.write_1m(rows)

