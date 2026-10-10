"""Bounded user requests executed inside the existing trusted Market Data scheduler."""

from __future__ import annotations

import logging
import re
from concurrent.futures import FIRST_COMPLETED, Executor, wait
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, ContextManager, Iterable, Mapping, Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.ports.provider_admission import provider_io_guard
from trading.contexts.market_data.application.ports.stores.history_coverage_reader import (
    HistoryCoverageReader,
)
from trading.contexts.market_data.application.ports.stores.raw_kline_writer import RawKlineWriter
from trading.contexts.market_data.application.services.source_retry import (
    IncompleteHistoryError,
    TemporarySourceError,
)
from trading.contexts.market_data.application.use_cases.rest_fill_range_1m import (
    RestFillRange1mUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

MAX_MINUTES = 10080
MAX_SYMBOLS = 8
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkRequest:
    kind: str
    market_id: int
    symbols: tuple[str, ...] = ()
    start_at: datetime | None = None
    end_at: datetime | None = None
    timeframe: str = "1m"

    def validate(self, *, now: datetime) -> int:
        if self.kind not in {"catalog_refresh", "candle_ingestion"} or self.market_id < 1:
            raise ValueError("invalid request kind or market")
        if self.timeframe != "1m":
            raise ValueError("only 1m source candles are supported")
        if self.kind == "catalog_refresh":
            if self.symbols or self.start_at is not None or self.end_at is not None:
                raise ValueError("catalog refresh does not accept a candle range")
            return 1
        if not 1 <= len(self.symbols) <= MAX_SYMBOLS or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("select 1 to 8 distinct instruments")
        if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{1,63}", s) for s in self.symbols):
            raise ValueError("invalid instrument symbol")
        start, end = self.start_at, self.end_at
        if start is None or end is None or start.tzinfo is None or end.tzinfo is None:
            raise ValueError("timezone-aware start and end are required")
        if any(t.second or t.microsecond for t in (start, end)):
            raise ValueError("range boundaries must align to a minute")
        if start < datetime(2017, 1, 1, tzinfo=UTC) or end > now.replace(second=0, microsecond=0):
            raise ValueError("range must contain historical closed candles")
        minutes = int((end - start).total_seconds() // 60)
        total = minutes * len(self.symbols)
        if minutes <= 0 or total > 2_147_483_647:
            raise ValueError("range must contain positive historical instrument-minutes")
        return total


class WorkRequestStore(Protocol):
    def execution(self) -> ContextManager[Callable[[], None] | None]: ...
    def recover(self, *, now: datetime) -> None: ...
    def claim(self, *, now: datetime) -> Mapping[str, Any] | None: ...
    def checkpoint(
        self,
        *,
        job: Mapping[str, Any],
        units: int,
        rows_read: int,
        rows_written: int,
        now: datetime,
    ) -> str | None: ...
    def release(self, *, job: Mapping[str, Any], now: datetime) -> None: ...
    def defer(
        self, *, job: Mapping[str, Any], error_code: str, retry_after_s: float, now: datetime
    ) -> None: ...
    def finish(
        self, *, job: Mapping[str, Any], state: str, error_code: str | None, now: datetime
    ) -> None: ...


@dataclass
class MarketDataWorkRequestRunner:
    """Bounded requests under one coordinator; existing REST fill owns all writes.

    Only classified temporary source failures retry automatically. Progress counts
    completed windows. Pause/cancel is acknowledged at a checkpoint between windows.
    """

    store: WorkRequestStore
    fill: RestFillRange1mUseCase
    refresh_catalog: Callable[[int, Callable[[], None]], None]
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    max_windows: int = 10
    window_minutes: int = 60
    retryable_storage_error: Callable[[Exception], bool | str] = lambda _: False
    repair_index: HistoryCoverageReader | None = None

    def run_once(self) -> bool:
        with self.store.execution() as check:
            if check is None:
                return False
            check()
            self.store.recover(now=self.now())
            return self._run_owned(check)

    def run_batch(
        self, *, executor: Executor, concurrency: int,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> int:
        """Refill completed slots under one coordinator, recovering only before dispatch.

        Each worker yields after its configured window budget. Repository queue order
        puts yielded work behind older eligible jobs. Ownership outlives every future,
        including submission failure and shutdown; no recovery runs beside a sibling.
        """
        if concurrency < 1:
            raise ValueError("work request concurrency must be positive")
        with self.store.execution() as check:
            if check is None:
                return 0
            check()
            self.store.recover(now=self.now())
            futures = set()
            completed = 0
            try:
                while not should_stop():
                    check()
                    while len(futures) < concurrency and not should_stop():
                        check()
                        job = self.store.claim(now=self.now())
                        if job is None:
                            break
                        futures.add(executor.submit(self._run_claimed, job, check))
                    if not futures:
                        break
                    done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.remove(future)
                        completed += bool(future.result())
            finally:
                # A cancelled await or broken coordinator is not stopped worker I/O.
                wait(futures)
            return completed + sum(bool(future.result()) for future in futures)

    def _run_owned(self, check: Callable[[], None]) -> bool:
        check()
        job = self.store.claim(now=self.now())
        if job is None:
            return False
        return self._run_claimed(job, check)

    def _run_claimed(self, job: Mapping[str, Any], check: Callable[[], None]) -> bool:
        with provider_io_guard(check):
            return self._run_claimed_guarded(job, check)

    def _run_claimed_guarded(self, job: Mapping[str, Any], check: Callable[[], None]) -> bool:
        check()
        units = int(job.get("completed_units", 0))
        rows_read = int(job.get("rows_read", 0))
        rows_written = int(job.get("rows_written", 0))
        windows = 0
        phase = "catalog_refresh"
        try:
            if job["kind"] == "catalog_refresh":
                self.refresh_catalog(int(job["market_id"]), check)
                units = 1
            else:
                per_symbol = int((job["end_at"] - job["start_at"]).total_seconds() // 60)
                for position, symbol in enumerate(job["symbols"]):
                    completed = min(per_symbol, max(0, units - position * per_symbol))
                    cursor = job["start_at"] + timedelta(minutes=completed)
                    while cursor < job["end_at"]:
                        phase = "checkpoint"
                        state = self.store.checkpoint(
                            job=job,
                            units=units,
                            rows_read=rows_read,
                            rows_written=rows_written,
                            now=self.now(),
                        )
                        if state != "running":
                            if state in {"pause_requested", "cancel_requested"}:
                                self.store.release(job=job, now=self.now())
                            return True
                        if windows >= self.max_windows:
                            phase = "release"
                            check()
                            self.store.release(job=job, now=self.now())
                            return True
                        instrument = InstrumentId(MarketId(job["market_id"]), Symbol(symbol))
                        gap_end = job["end_at"]
                        if self.repair_index is not None:
                            phase = "fill_window"
                            gap = self.repair_index.first_gap(
                                instrument_id=instrument,
                                time_range=TimeRange(UtcTimestamp(cursor), UtcTimestamp(gap_end)),
                            )
                            next_cursor = gap.start.value if gap else gap_end
                            units += int((next_cursor - cursor).total_seconds() // 60)
                            cursor = next_cursor
                            if gap is None:
                                break
                            gap_end = gap.end.value
                        end = min(cursor + timedelta(minutes=self.window_minutes), gap_end)
                        phase = "lease_check"
                        check()
                        fill = replace(
                            self.fill, writer=LeaseCheckedRawWriter(self.fill.writer, check),
                            check_lease=check,
                        )
                        phase = "fill_window"
                        result = fill.run(
                            RestFillTask(
                                instrument_id=InstrumentId(
                                    MarketId(job["market_id"]), Symbol(symbol)
                                ),
                                time_range=TimeRange(UtcTimestamp(cursor), UtcTimestamp(end)),
                                reason="user_requested_fill",
                            )
                        )
                        rows_read += result.rows_read
                        rows_written += result.rows_written
                        if self.repair_index is not None:
                            phase = "fill_window"
                            if (
                                self.repair_index.first_gap(
                                    instrument_id=instrument,
                                    time_range=TimeRange(UtcTimestamp(cursor), UtcTimestamp(end)),
                                )
                                is not None
                            ):
                                raise IncompleteHistoryError()
                        units += int((end - cursor).total_seconds() // 60)
                        cursor = end
                        windows += 1
            phase = "checkpoint"
            state = self.store.checkpoint(
                job=job, units=units, rows_read=rows_read, rows_written=rows_written, now=self.now()
            )
            if state == "pause_requested":
                self.store.release(job=job, now=self.now())
            elif state in {"running", "cancel_requested"}:
                phase = "finish"
                self.store.finish(
                    job=job,
                    state="cancelled" if state == "cancel_requested" else "succeeded",
                    error_code=None,
                    now=self.now(),
                )
        except TemporarySourceError as error:
            check()
            self.store.checkpoint(
                job=job, units=units, rows_read=rows_read, rows_written=rows_written, now=self.now()
            )
            if (
                isinstance(error, IncompleteHistoryError)
                and job.get("previous_error_code", job.get("error_code")) == error.code
                and units <= int(job.get("completed_units", 0))
                and int(job.get("retry_count", 0)) >= 3
            ):
                state = self.store.checkpoint(
                    job=job,
                    units=units,
                    rows_read=rows_read,
                    rows_written=rows_written,
                    now=self.now(),
                )
                if state in {"pause_requested", "cancel_requested"}:
                    self.store.release(job=job, now=self.now())
                elif state == "running":
                    self.store.finish(
                        job=job, state="failed", error_code=error.code, now=self.now()
                    )
                return True
            # The failed window has no checkpoint. Existing fill deduplication
            # reconciles any already stored candles when that window is replayed.
            self.store.defer(
                job={**job, "error_phase": phase},
                error_code=error.code,
                retry_after_s=error.retry_after_s,
                now=self.now(),
            )
        except Exception as error:
            check()
            # Never persist provider payloads, URLs, credentials, or exception messages.
            log.warning(
                "market data work error phase=%s error_type=%s", phase, type(error).__name__
            )
            storage_error = self.retryable_storage_error(error)
            if storage_error:
                self.store.defer(
                    job={**job, "error_phase": phase},
                    error_code=storage_error
                    if isinstance(storage_error, str)
                    else "storage_unavailable",
                    retry_after_s=0,
                    now=self.now(),
                )
            else:
                self.store.finish(
                    job={**job, "error_phase": phase},
                    state="failed",
                    error_code="source_or_storage_unavailable",
                    now=self.now(),
                )
        return True


@dataclass(frozen=True)
class LeaseCheckedRawWriter:
    delegate: RawKlineWriter
    check: Callable[[], None]

    def write_1m(self, rows: Iterable[CandleWithMeta]) -> None:
        self.check()
        self.delegate.write_1m(rows)
