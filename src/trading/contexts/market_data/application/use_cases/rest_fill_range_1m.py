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
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.contexts.market_data.application.use_cases.time_slicing import (
    slice_time_range_by_utc_days,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


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
    index_reader: CanonicalCandleIndexReader | None = None

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
        effective_task = self._effective_task(task)
        if effective_task is None:
            finished_at = self.clock.now()
            return RestFillResult(
                task=task,
                rows_read=0,
                rows_written=0,
                batches_written=0,
                started_at=started_at,
                finished_at=finished_at,
            )

        rows_read = 0
        rows_written = 0
        batches_written = 0

        for chunk in slice_time_range_by_utc_days(
            time_range=effective_task.time_range,
            max_days=self.max_days_per_insert,
        ):
            chunk_read, chunk_written, chunk_batches = self._ingest_chunk(effective_task, chunk)
            rows_read += chunk_read
            rows_written += chunk_written
            batches_written += chunk_batches

        finished_at = self.clock.now()
        return RestFillResult(
            task=effective_task,
            rows_read=rows_read,
            rows_written=rows_written,
            batches_written=batches_written,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _effective_task(self, task: RestFillTask) -> RestFillTask | None:
        """
        Build execution task, clamping historical/bootstrap ranges to current canonical minimum.

        Parameters:
        - task: original fill task.

        Returns:
        - Clamped task to execute or `None` when range becomes empty.

        Assumptions/Invariants:
        - Clamping applies only to historical bootstrap-style reasons.
        - Missing index reader disables clamping and preserves original task.

        Errors/Exceptions:
        - Propagates index-reader errors when clamping is enabled.

        Side effects:
        - Reads canonical index bounds for selected reasons.
        """
        clamp_reasons = {"bootstrap", "scheduler_bootstrap", "historical_backfill"}
        if task.reason not in clamp_reasons or self.index_reader is None:
            return task

        now_floor = UtcTimestamp(floor_to_minute_utc(self.clock.now().value))
        canonical_min, _canonical_max = self.index_reader.bounds_1m(
            instrument_id=task.instrument_id,
            before=now_floor,
        )
        if canonical_min is None:
            return task

        clamped_end = canonical_min
        if task.time_range.end.value <= clamped_end.value:
            return task
        if task.time_range.start.value >= clamped_end.value:
            return None

        return RestFillTask(
            instrument_id=task.instrument_id,
            time_range=TimeRange(start=task.time_range.start, end=clamped_end),
            reason=task.reason,
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
        existing_minute_keys = self._existing_minute_keys(task=task, chunk=chunk)

        for row in self.source.stream_1m(task.instrument_id, chunk):
            rows_read += 1
            minute_key = _minute_key(row.candle.ts_open.value)
            if minute_key in existing_minute_keys:
                continue
            existing_minute_keys.add(minute_key)
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

    def _existing_minute_keys(self, *, task: RestFillTask, chunk) -> set[int]:
        """
        Load existing canonical minute keys for one instrument/chunk pair.

        Parameters:
        - task: parent task containing instrument id.
        - chunk: UTC half-open range for one ingest chunk.

        Returns:
        - Set of integer minute keys (`epoch_seconds // 60`) already present in canonical.

        Assumptions/Invariants:
        - Empty set is returned when canonical index reader is not configured.

        Errors/Exceptions:
        - Propagates index-reader errors when reader is configured.

        Side effects:
        - Executes canonical index query when reader is configured.
        """
        if self.index_reader is None:
            return set()
        existing = self.index_reader.distinct_ts_opens(
            instrument_id=task.instrument_id,
            time_range=chunk,
        )
        return {_minute_key(item.value) for item in existing}

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


def _minute_key(dt) -> int:
    """
    Convert timestamp to deterministic integer minute key.

    Parameters:
    - dt: UTC datetime-like value.

    Returns:
    - Integer key `floor(epoch_seconds / 60)`.

    Assumptions/Invariants:
    - Input timestamp is timezone-aware UTC in runtime pipelines.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return int(dt.timestamp() // 60)
