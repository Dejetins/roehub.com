from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol
from uuid import uuid4

from trading.contexts.market_data.application.dto import (
    CandleRepairSourceAttempt,
    CandleRepairStatus,
    ClosedCandleTailRepairPolicy,
    ClosedCandleTailResult,
    ClosedCandleTailRow,
    MarketDataCandleRepairAuditEvent,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources import CandleIngestSource
from trading.contexts.market_data.application.ports.stores import (
    CandleRepairAuditRepository,
    CanonicalCandleReader,
)
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp

_ONE_MINUTE = timedelta(minutes=1)


class ClosedCandleHotCache(Protocol):
    """Application-facing protocol for the Redis hot-cache adapter."""

    def read_range(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start: UtcTimestamp,
        end: UtcTimestamp,
    ) -> tuple[ClosedCandleTailRow, ...]:
        ...

    def write_closed_1m(self, candle) -> bool:  # noqa: ANN001
        ...


@dataclass(frozen=True, slots=True)
class ClosedCandleTailProviderHooks:
    """Optional callbacks for bounded live-tail repair metrics."""

    on_repair_attempt: Callable[[str, str], None] | None = None
    on_repair_latency: Callable[[str, str, float], None] | None = None
    on_clickhouse_circuit_state: Callable[[int], None] | None = None


class MarketDataClosedCandleTailProvider:
    """
    Market Data-owned source chain for short closed-candle tail repair.
    """

    def __init__(
        self,
        *,
        hot_cache: ClosedCandleHotCache,
        canonical_reader: CanonicalCandleReader,
        rest_source: CandleIngestSource,
        audit_repository: CandleRepairAuditRepository,
        clock: Clock,
        policy: ClosedCandleTailRepairPolicy | None = None,
        clickhouse_circuit_open_seconds: float = 30.0,
        hooks: ClosedCandleTailProviderHooks | None = None,
    ) -> None:
        if hot_cache is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataClosedCandleTailProvider requires hot_cache")
        if canonical_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataClosedCandleTailProvider requires canonical_reader")
        if rest_source is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataClosedCandleTailProvider requires rest_source")
        if audit_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataClosedCandleTailProvider requires audit_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataClosedCandleTailProvider requires clock")
        if clickhouse_circuit_open_seconds <= 0:
            raise ValueError("clickhouse_circuit_open_seconds must be > 0")

        self._hot_cache = hot_cache
        self._canonical_reader = canonical_reader
        self._rest_source = rest_source
        self._audit_repository = audit_repository
        self._clock = clock
        self._policy = policy if policy is not None else ClosedCandleTailRepairPolicy()
        self._clickhouse_circuit_open_seconds = clickhouse_circuit_open_seconds
        self._clickhouse_circuit_open_until: datetime | None = None
        self._hooks = hooks if hooks is not None else ClosedCandleTailProviderHooks()

    def get_closed_1m_tail(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start_ts_open: UtcTimestamp,
        end_ts_open: UtcTimestamp,
        correlation_id: str,
    ) -> ClosedCandleTailResult:
        """
        Return closed 1m candles from Redis hot cache, ClickHouse, and REST tail.
        """
        started_at = time.perf_counter()
        time_range = TimeRange(start=start_ts_open, end=end_ts_open)
        now_floor = UtcTimestamp(floor_to_minute_utc(self._clock.now().value))
        if end_ts_open.value > now_floor.value:
            result = ClosedCandleTailResult(
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
                candles=(),
                sources_attempted=(
                    CandleRepairSourceAttempt(
                        source="rest",
                        status="miss",
                        error_code="non_closed_range",
                    ),
                ),
                error_code="non_closed_range",
            )
            self._record_audit(
                correlation_id=correlation_id,
                result=result,
                status="miss",
                error_code="non_closed_range",
                error_summary="requested range includes current open candle",
            )
            self._emit_result_hooks(
                result=result,
                duration_seconds=time.perf_counter() - started_at,
            )
            return result

        rows_by_ts: dict[datetime, ClosedCandleTailRow] = {}
        attempts: list[CandleRepairSourceAttempt] = []

        self._merge_rows(
            rows_by_ts,
            self._read_redis(
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
                attempts=attempts,
            ),
        )
        if _continuous(rows_by_ts, time_range):
            result = self._finalize(
                correlation_id=correlation_id,
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
                rows_by_ts=rows_by_ts,
                attempts=attempts,
            )
            self._emit_result_hooks(
                result=result,
                duration_seconds=time.perf_counter() - started_at,
            )
            return result

        self._merge_rows(
            rows_by_ts,
            self._read_clickhouse(
                instrument_id=instrument_id,
                time_range=time_range,
                attempts=attempts,
            ),
        )
        if _continuous(rows_by_ts, time_range):
            result = self._finalize(
                correlation_id=correlation_id,
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
                rows_by_ts=rows_by_ts,
                attempts=attempts,
            )
            self._emit_result_hooks(
                result=result,
                duration_seconds=time.perf_counter() - started_at,
            )
            return result

        self._merge_rows(
            rows_by_ts,
            self._read_rest_tail(
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                time_range=time_range,
                now_floor=now_floor,
                rows_by_ts=rows_by_ts,
                attempts=attempts,
            ),
        )
        result = self._finalize(
            correlation_id=correlation_id,
            instrument_id=instrument_id,
            instrument_key=instrument_key,
            time_range=time_range,
            rows_by_ts=rows_by_ts,
            attempts=attempts,
        )
        self._emit_result_hooks(
            result=result,
            duration_seconds=time.perf_counter() - started_at,
        )
        return result

    def _read_redis(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        time_range: TimeRange,
        attempts: list[CandleRepairSourceAttempt],
    ) -> tuple[ClosedCandleTailRow, ...]:
        try:
            rows = self._hot_cache.read_range(
                instrument_id=instrument_id,
                instrument_key=instrument_key,
                start=time_range.start,
                end=time_range.end,
            )
        except Exception:  # noqa: BLE001
            attempts.append(
                CandleRepairSourceAttempt(
                    source="redis_hot_cache",
                    status="failed",
                    error_code="hot_cache_exception",
                )
            )
            return ()
        attempts.append(
            CandleRepairSourceAttempt(
                source="redis_hot_cache",
                status="succeeded" if _continuous(_rows_by_ts(rows), time_range) else "miss",
            )
        )
        return rows

    def _read_clickhouse(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
        attempts: list[CandleRepairSourceAttempt],
    ) -> tuple[ClosedCandleTailRow, ...]:
        now = self._clock.now().value
        if self._clickhouse_circuit_open_until is not None:
            if now < self._clickhouse_circuit_open_until:
                attempts.append(
                    CandleRepairSourceAttempt(
                        source="clickhouse",
                        status="circuit_open",
                        error_code="clickhouse_circuit_open",
                    )
                )
                return ()
            self._clickhouse_circuit_open_until = None

        try:
            candles = tuple(self._canonical_reader.read_1m(instrument_id, time_range))
        except Exception:  # noqa: BLE001
            self._clickhouse_circuit_open_until = now + timedelta(
                seconds=self._clickhouse_circuit_open_seconds
            )
            attempts.append(
                CandleRepairSourceAttempt(
                    source="clickhouse",
                    status="failed",
                    error_code="clickhouse_exception",
                )
            )
            return ()

        rows = tuple(
            ClosedCandleTailRow(candle=candle, source="clickhouse")
            for candle in candles
        )
        attempts.append(
            CandleRepairSourceAttempt(
                source="clickhouse",
                status="succeeded" if _continuous(_rows_by_ts(rows), time_range) else "miss",
            )
        )
        return rows

    def _read_rest_tail(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        time_range: TimeRange,
        now_floor: UtcTimestamp,
        rows_by_ts: dict[datetime, ClosedCandleTailRow],
        attempts: list[CandleRepairSourceAttempt],
    ) -> tuple[ClosedCandleTailRow, ...]:
        missing_ranges = _missing_ranges(rows_by_ts=rows_by_ts, time_range=time_range)
        if not missing_ranges:
            attempts.append(CandleRepairSourceAttempt(source="rest", status="miss"))
            return ()

        oldest_allowed = now_floor.value - timedelta(
            minutes=self._policy.rest_tail_limit_minutes
        )
        if any(item.start.value < oldest_allowed for item in missing_ranges):
            attempts.append(
                CandleRepairSourceAttempt(
                    source="rest",
                    status="miss",
                    error_code="rest_tail_limit_exceeded",
                )
            )
            return ()

        restored: list[ClosedCandleTailRow] = []
        missing_ts = {
            ts_open
            for item in missing_ranges
            for ts_open in _expected_minute_values(item)
        }
        try:
            for missing_range in missing_ranges:
                for row in self._rest_source.stream_1m(instrument_id, missing_range):
                    ts_open = row.candle.ts_open.value
                    if ts_open not in missing_ts or row.candle.ts_close.value > now_floor.value:
                        continue
                    if row.meta.instrument_key != instrument_key:
                        continue
                    if not self._hot_cache.write_closed_1m(row):
                        attempts.append(
                            CandleRepairSourceAttempt(
                                source="rest",
                                status="failed",
                                error_code="hot_cache_write_failed",
                            )
                        )
                        return ()
                    restored.append(ClosedCandleTailRow(candle=row, source="rest"))
        except Exception:  # noqa: BLE001
            attempts.append(
                CandleRepairSourceAttempt(
                    source="rest",
                    status="failed",
                    error_code="rest_exception",
                )
            )
            return ()

        attempts.append(
            CandleRepairSourceAttempt(
                source="rest",
                status="succeeded" if restored else "miss",
            )
        )
        return tuple(restored)

    def _finalize(
        self,
        *,
        correlation_id: str,
        instrument_id: InstrumentId,
        instrument_key: str,
        time_range: TimeRange,
        rows_by_ts: dict[datetime, ClosedCandleTailRow],
        attempts: list[CandleRepairSourceAttempt],
    ) -> ClosedCandleTailResult:
        result = ClosedCandleTailResult(
            instrument_id=instrument_id,
            instrument_key=instrument_key,
            time_range=time_range,
            candles=tuple(sorted(rows_by_ts.values(), key=lambda row: row.ts_open.value)),
            sources_attempted=tuple(attempts),
            error_code=None if _continuous(rows_by_ts, time_range) else "missing_closed_tail",
        )
        self._record_audit(
            correlation_id=correlation_id,
            result=result,
            status="succeeded" if result.continuous else "miss",
            error_code=result.error_code,
            error_summary=None if result.continuous else "closed candle tail is incomplete",
        )
        return result

    def _record_audit(
        self,
        *,
        correlation_id: str,
        result: ClosedCandleTailResult,
        status: CandleRepairStatus,
        error_code: str | None,
        error_summary: str | None,
    ) -> None:
        self._audit_repository.record(
            event=MarketDataCandleRepairAuditEvent(
                event_id=uuid4(),
                correlation_id=correlation_id,
                instrument_id=result.instrument_id,
                instrument_key=result.instrument_key,
                time_range=result.time_range,
                status=status,
                sources_attempted=result.sources_attempted,
                restored_ts_opens=result.restored_ts_opens,
                missing_ts_opens=result.missing_ts_opens,
                created_at=self._clock.now(),
                error_code=error_code,
                error_summary=error_summary,
            )
        )

    def _merge_rows(
        self,
        rows_by_ts: dict[datetime, ClosedCandleTailRow],
        rows: tuple[ClosedCandleTailRow, ...],
    ) -> None:
        for row in rows:
            rows_by_ts.setdefault(row.ts_open.value, row)

    def _emit_result_hooks(
        self,
        *,
        result: ClosedCandleTailResult,
        duration_seconds: float,
    ) -> None:
        for attempt in result.sources_attempted:
            _emit_attempt(self._hooks.on_repair_attempt, attempt.source, attempt.status)
            _emit_latency(
                self._hooks.on_repair_latency,
                attempt.source,
                attempt.status,
                duration_seconds,
            )

        clickhouse_attempts = [
            attempt for attempt in result.sources_attempted if attempt.source == "clickhouse"
        ]
        if not clickhouse_attempts:
            return
        last_status = clickhouse_attempts[-1].status
        if last_status in {"failed", "circuit_open"}:
            _emit_circuit_state(self._hooks.on_clickhouse_circuit_state, 1)
        elif last_status in {"succeeded", "miss"}:
            _emit_circuit_state(self._hooks.on_clickhouse_circuit_state, 0)


def _continuous(rows_by_ts: dict[datetime, ClosedCandleTailRow], time_range: TimeRange) -> bool:
    expected = _expected_minute_values(time_range)
    return all(value in rows_by_ts for value in expected)


def _rows_by_ts(rows: tuple[ClosedCandleTailRow, ...]) -> dict[datetime, ClosedCandleTailRow]:
    return {row.ts_open.value: row for row in rows}


def _expected_minute_values(time_range: TimeRange) -> tuple[datetime, ...]:
    cursor = time_range.start.value
    values: list[datetime] = []
    while cursor < time_range.end.value:
        values.append(cursor)
        cursor += _ONE_MINUTE
    return tuple(values)


def _missing_ranges(
    *,
    rows_by_ts: dict[datetime, ClosedCandleTailRow],
    time_range: TimeRange,
) -> tuple[TimeRange, ...]:
    missing = [value for value in _expected_minute_values(time_range) if value not in rows_by_ts]
    if not missing:
        return ()

    ranges: list[TimeRange] = []
    start = missing[0]
    previous = missing[0]
    for value in missing[1:]:
        if value == previous + _ONE_MINUTE:
            previous = value
            continue
        ranges.append(TimeRange(UtcTimestamp(start), UtcTimestamp(previous + _ONE_MINUTE)))
        start = value
        previous = value
    ranges.append(TimeRange(UtcTimestamp(start), UtcTimestamp(previous + _ONE_MINUTE)))
    return tuple(ranges)


def _emit_attempt(
    callback: Callable[[str, str], None] | None,
    source: str,
    status: str,
) -> None:
    if callback is None:
        return
    callback(source, status)


def _emit_latency(
    callback: Callable[[str, str, float], None] | None,
    source: str,
    status: str,
    duration_seconds: float,
) -> None:
    if callback is None:
        return
    callback(source, status, max(duration_seconds, 0.0))


def _emit_circuit_state(callback: Callable[[int], None] | None, state: int) -> None:
    if callback is None:
        return
    callback(state)
