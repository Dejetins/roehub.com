from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, cast
from uuid import UUID

from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp

from .candle_with_meta import CandleWithMeta

CandleRepairSource = Literal["redis_hot_cache", "clickhouse", "rest"]
CANDLE_REPAIR_SOURCES: frozenset[str] = frozenset(
    {"redis_hot_cache", "clickhouse", "rest"}
)

CandleRepairStatus = Literal[
    "attempted",
    "succeeded",
    "miss",
    "failed",
    "circuit_open",
    "rate_limited",
]
CANDLE_REPAIR_STATUSES: frozenset[str] = frozenset(
    {"attempted", "succeeded", "miss", "failed", "circuit_open", "rate_limited"}
)

_ONE_MINUTE = timedelta(minutes=1)
_ERROR_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_:-]{0,95}$")
_FORBIDDEN_SUMMARY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|cookie|dsn|password|secret|token)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CandleRepairSourceAttempt:
    """
    Bounded summary of one source attempted during short live-tail repair.
    """

    source: CandleRepairSource
    status: CandleRepairStatus
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _validated_source(self.source))
        object.__setattr__(self, "status", _validated_status(self.status))
        object.__setattr__(
            self,
            "error_code",
            _validated_error_code(self.error_code, field="error_code"),
        )


@dataclass(frozen=True, slots=True)
class ClosedCandleTailRow:
    """
    One closed 1m candle restored from a bounded repair source.
    """

    candle: CandleWithMeta
    source: CandleRepairSource

    def __post_init__(self) -> None:
        if self.candle is None:  # type: ignore[truthy-bool]
            raise ValueError("ClosedCandleTailRow requires candle")
        object.__setattr__(self, "source", _validated_source(self.source))

    @property
    def ts_open(self) -> UtcTimestamp:
        return self.candle.candle.ts_open


@dataclass(frozen=True, slots=True)
class ClosedCandleTailRepairPolicy:
    """
    Configuration primitive for the later live-tail provider chain.
    """

    source_order: tuple[CandleRepairSource, ...] = (
        "redis_hot_cache",
        "clickhouse",
        "rest",
    )
    rest_tail_limit_minutes: int = 15
    clickhouse_timeout_seconds: float = 1.0
    fail_closed_on_audit_write: bool = True

    def __post_init__(self) -> None:
        if not self.source_order:
            raise ValueError("ClosedCandleTailRepairPolicy requires non-empty source_order")
        object.__setattr__(
            self,
            "source_order",
            tuple(_validated_source(source) for source in self.source_order),
        )
        if len(set(self.source_order)) != len(self.source_order):
            raise ValueError("ClosedCandleTailRepairPolicy.source_order must be unique")
        if self.rest_tail_limit_minutes <= 0:
            raise ValueError(
                "ClosedCandleTailRepairPolicy.rest_tail_limit_minutes must be > 0"
            )
        if self.clickhouse_timeout_seconds <= 0:
            raise ValueError(
                "ClosedCandleTailRepairPolicy.clickhouse_timeout_seconds must be > 0"
            )


@dataclass(frozen=True, slots=True)
class ClosedCandleTailResult:
    """
    Deterministic result of a Strategy request for a half-open closed-candle tail.
    """

    instrument_id: InstrumentId
    instrument_key: str
    time_range: TimeRange
    candles: tuple[ClosedCandleTailRow, ...]
    sources_attempted: tuple[CandleRepairSourceAttempt, ...]
    missing_ts_opens: tuple[UtcTimestamp, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("ClosedCandleTailResult requires instrument_id")
        instrument_key = self.instrument_key.strip()
        if not instrument_key:
            raise ValueError("ClosedCandleTailResult requires non-empty instrument_key")
        object.__setattr__(self, "instrument_key", instrument_key)
        _validate_1m_range(self.time_range)

        candles = tuple(sorted(self.candles, key=lambda row: row.ts_open.value))
        restored_ts_values: set[datetime] = set()
        for row in candles:
            if row.candle.candle.instrument_id != self.instrument_id:
                raise ValueError("ClosedCandleTailResult candle instrument_id mismatch")
            if row.candle.meta.instrument_key != instrument_key:
                raise ValueError("ClosedCandleTailResult candle instrument_key mismatch")
            ts_open = row.ts_open.value
            if not self.time_range.contains(row.ts_open):
                raise ValueError("ClosedCandleTailResult candle outside requested range")
            if ts_open in restored_ts_values:
                raise ValueError("ClosedCandleTailResult restored candles must be unique")
            restored_ts_values.add(ts_open)
        object.__setattr__(self, "candles", candles)

        sources_attempted = tuple(self.sources_attempted)
        object.__setattr__(self, "sources_attempted", sources_attempted)

        expected_ts_values = _expected_minute_opens(self.time_range)
        supplied_missing = tuple(_sorted_unique_timestamps(self.missing_ts_opens))
        computed_missing = tuple(
            UtcTimestamp(ts_open)
            for ts_open in expected_ts_values
            if ts_open not in restored_ts_values
        )
        missing = tuple(_sorted_unique_timestamps((*supplied_missing, *computed_missing)))
        for ts_open in missing:
            if not self.time_range.contains(ts_open):
                raise ValueError("ClosedCandleTailResult missing ts_open outside range")
        object.__setattr__(self, "missing_ts_opens", missing)
        object.__setattr__(
            self,
            "error_code",
            _validated_error_code(self.error_code, field="error_code"),
        )

    @property
    def continuous(self) -> bool:
        return not self.missing_ts_opens and len(self.candles) == len(
            _expected_minute_opens(self.time_range)
        )

    @property
    def restored_ts_opens(self) -> tuple[UtcTimestamp, ...]:
        return tuple(row.ts_open for row in self.candles)


@dataclass(frozen=True, slots=True)
class MarketDataCandleRepairAuditEvent:
    """
    Durable audit/outbox-equivalent record for one live-tail repair attempt.
    """

    event_id: UUID
    correlation_id: str
    instrument_id: InstrumentId
    instrument_key: str
    time_range: TimeRange
    status: CandleRepairStatus
    sources_attempted: tuple[CandleRepairSourceAttempt, ...]
    restored_ts_opens: tuple[UtcTimestamp, ...]
    missing_ts_opens: tuple[UtcTimestamp, ...]
    created_at: UtcTimestamp
    error_code: str | None = None
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if self.event_id is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataCandleRepairAuditEvent requires event_id")
        correlation_id = self.correlation_id.strip()
        if not correlation_id:
            raise ValueError(
                "MarketDataCandleRepairAuditEvent requires non-empty correlation_id"
            )
        object.__setattr__(self, "correlation_id", correlation_id)
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataCandleRepairAuditEvent requires instrument_id")
        instrument_key = self.instrument_key.strip()
        if not instrument_key:
            raise ValueError(
                "MarketDataCandleRepairAuditEvent requires non-empty instrument_key"
            )
        object.__setattr__(self, "instrument_key", instrument_key)
        _validate_1m_range(self.time_range)
        object.__setattr__(self, "status", _validated_status(self.status))
        object.__setattr__(self, "sources_attempted", tuple(self.sources_attempted))
        object.__setattr__(
            self,
            "restored_ts_opens",
            tuple(_sorted_unique_timestamps(self.restored_ts_opens)),
        )
        object.__setattr__(
            self,
            "missing_ts_opens",
            tuple(_sorted_unique_timestamps(self.missing_ts_opens)),
        )
        for ts_open in (*self.restored_ts_opens, *self.missing_ts_opens):
            _validate_minute_aligned(ts_open, field="ts_open")
            if not self.time_range.contains(ts_open):
                raise ValueError(
                    "MarketDataCandleRepairAuditEvent ts_open outside audited range"
                )
        if self.created_at is None:  # type: ignore[truthy-bool]
            raise ValueError("MarketDataCandleRepairAuditEvent requires created_at")
        object.__setattr__(
            self,
            "error_code",
            _validated_error_code(self.error_code, field="error_code"),
        )
        object.__setattr__(
            self,
            "error_summary",
            _validated_error_summary(self.error_summary),
        )


def _validated_source(source: str) -> CandleRepairSource:
    normalized = source.strip().lower()
    if normalized not in CANDLE_REPAIR_SOURCES:
        raise ValueError(f"unsupported candle repair source: {source!r}")
    return cast(CandleRepairSource, normalized)


def _validated_status(status: str) -> CandleRepairStatus:
    normalized = status.strip().lower()
    if normalized not in CANDLE_REPAIR_STATUSES:
        raise ValueError(f"unsupported candle repair status: {status!r}")
    return cast(CandleRepairStatus, normalized)


def _validated_error_code(error_code: str | None, *, field: str) -> str | None:
    if error_code is None:
        return None
    normalized = error_code.strip().lower()
    if not normalized:
        return None
    if _ERROR_CODE_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a stable redacted code")
    return normalized


def _validated_error_summary(error_summary: str | None) -> str | None:
    if error_summary is None:
        return None
    normalized = " ".join(error_summary.strip().split())
    if not normalized:
        return None
    if len(normalized) > 240:
        raise ValueError("error_summary must be <= 240 characters")
    if _FORBIDDEN_SUMMARY_RE.search(normalized):
        raise ValueError("error_summary must not contain secret-like material")
    return normalized


def _validate_1m_range(time_range: TimeRange) -> None:
    _validate_minute_aligned(time_range.start, field="time_range.start")
    _validate_minute_aligned(time_range.end, field="time_range.end")
    if time_range.duration().total_seconds() % 60 != 0:
        raise ValueError("closed candle tail range must be minute-aligned")


def _validate_minute_aligned(ts: UtcTimestamp, *, field: str) -> None:
    value = ts.value
    if value.second != 0 or value.microsecond != 0:
        raise ValueError(f"{field} must be aligned to a 1m candle open")


def _expected_minute_opens(time_range: TimeRange) -> tuple[datetime, ...]:
    cursor = time_range.start.value
    result: list[datetime] = []
    while cursor < time_range.end.value:
        result.append(cursor)
        cursor = cursor + _ONE_MINUTE
    return tuple(result)


def _sorted_unique_timestamps(values: tuple[UtcTimestamp, ...]) -> tuple[UtcTimestamp, ...]:
    unique: dict[datetime, UtcTimestamp] = {}
    for value in values:
        unique[value.value] = value
    return tuple(unique[key] for key in sorted(unique))
