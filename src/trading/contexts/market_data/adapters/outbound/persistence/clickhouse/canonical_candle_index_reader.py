from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
    DailyTsOpenCount,
)
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class ClickHouseCanonicalCandleIndexReader(CanonicalCandleIndexReader):
    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self) -> None:
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseCanonicalCandleIndexReader requires gateway")
        if not self.database.strip():
            raise ValueError("ClickHouseCanonicalCandleIndexReader requires non-empty database")

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Return first and last available minute buckets for one instrument.

        Parameters:
        - instrument_id: instrument whose canonical bounds are requested.

        Returns:
        - Tuple `(first_ts_open, last_ts_open)` in UTC or `None` when no rows exist.

        Assumptions/Invariants:
        - Bounds are computed on UTC epoch-minute keys to avoid driver/server timezone drift on
          parameterized DateTime reads.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            min(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS first_minute_key,
            max(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS last_minute_key
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
        """
        rows = self.gateway.select(
            q,
            {"market_id": int(instrument_id.market_id.value), "symbol": str(instrument_id.symbol)},
        )
        if not rows:
            return None
        first = rows[0].get("first_minute_key")
        last = rows[0].get("last_minute_key")
        if first is None or last is None:
            return None
        return (_minute_key_to_utc_timestamp(int(first)), _minute_key_to_utc_timestamp(int(last)))

    def bounds_1m(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> tuple[UtcTimestamp | None, UtcTimestamp | None]:
        """
        Return canonical min/max minute buckets for one instrument before an exclusive bound.

        Parameters:
        - instrument_id: instrument whose canonical bounds are requested.
        - before: exclusive upper bound for `ts_open`.

        Returns:
        - Tuple `(min_ts_open, max_ts_open)` where values are `None` when dataset is empty.

        Assumptions/Invariants:
        - Bounds are computed on UTC epoch-minute keys to avoid driver/server timezone drift on
          parameterized DateTime reads.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            min(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS first_minute_key,
            max(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS last_minute_key
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open < fromUnixTimestamp64Milli(%(before_ms)s, 'UTC')
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "before_ms": _dt_to_epoch_ms(before.value),
            },
        )
        if not rows:
            return (None, None)
        first = rows[0].get("first_minute_key")
        last = rows[0].get("last_minute_key")
        if first is None or last is None:
            return (None, None)
        return (_minute_key_to_utc_timestamp(int(first)), _minute_key_to_utc_timestamp(int(last)))

    def max_ts_open_lt(self, *, instrument_id: InstrumentId, before: UtcTimestamp) -> UtcTimestamp | None:  # noqa: E501
        """
        Return latest canonical minute strictly before `before`.

        Parameters:
        - instrument_id: instrument whose latest known minute is requested.
        - before: upper bound (exclusive).

        Returns:
        - Latest minute in UTC or `None` when no qualifying rows exist.

        Assumptions/Invariants:
        - Result is normalized to UTC epoch-minute key to avoid driver/server timezone drift on
          parameterized DateTime reads.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            max(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS last_minute_key
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open < fromUnixTimestamp64Milli(%(before_ms)s, 'UTC')
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "before_ms": _dt_to_epoch_ms(before.value),
            },
        )
        if not rows:
            return None
        last = rows[0].get("last_minute_key")
        if last is None:
            return None
        return _minute_key_to_utc_timestamp(int(last))

    def daily_counts(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[DailyTsOpenCount]:  # noqa: E501
        """
        Return per-day counts of distinct canonical minute buckets for one range.

        Parameters:
        - instrument_id: instrument being aggregated.
        - time_range: UTC half-open range `[start, end)` for aggregation.

        Returns:
        - Sequence of `(day, count)` rows.

        Assumptions/Invariants:
        - Distinctness is measured on UTC epoch-minute keys to avoid driver/server timezone drift on
          parameterized DateTime reads.

        Errors/Exceptions:
        - Raises `RuntimeError` on unexpected day value types from gateway.
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            formatDateTime(ts_open, '%F', 'UTC') AS day,
            uniqExact(intDiv(toUnixTimestamp64Milli(ts_open), 60000)) AS cnt
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open >= fromUnixTimestamp64Milli(%(start_ms)s, 'UTC')
          AND ts_open < fromUnixTimestamp64Milli(%(end_ms)s, 'UTC')
        GROUP BY day
        ORDER BY day
        SETTINGS max_threads = 1
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start_ms": _dt_to_epoch_ms(time_range.start.value),
                "end_ms": _dt_to_epoch_ms(time_range.end.value),
            },
        )
        out: list[DailyTsOpenCount] = []
        for r in rows:
            d = r.get("day")
            cnt = int(r.get("cnt", 0))
            if isinstance(d, date):
                out.append(DailyTsOpenCount(day=d, count=cnt))
            elif isinstance(d, str):
                # "YYYY-MM-DD"
                y, m, dd = d.split("-")
                out.append(DailyTsOpenCount(day=date(int(y), int(m), int(dd)), count=cnt))
            else:
                raise RuntimeError(f"Unexpected day type from ClickHouse: {type(d).__name__} {d!r}")
        return out

    def distinct_ts_opens(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[UtcTimestamp]:  # noqa: E501
        """
        Return distinct canonical minute starts for one instrument and one range.

        Parameters:
        - instrument_id: instrument being queried.
        - time_range: UTC half-open range `[start, end)`.

        Returns:
        - Sorted sequence of UTC minute starts.

        Assumptions/Invariants:
        - Timestamps are normalized as UTC epoch-minute keys in SQL to avoid driver/server timezone
          drift on parameterized DateTime reads.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT DISTINCT
            intDiv(toUnixTimestamp64Milli(ts_open), 60000) AS minute_key
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open >= fromUnixTimestamp64Milli(%(start_ms)s, 'UTC')
          AND ts_open < fromUnixTimestamp64Milli(%(end_ms)s, 'UTC')
        ORDER BY minute_key
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start_ms": _dt_to_epoch_ms(time_range.start.value),
                "end_ms": _dt_to_epoch_ms(time_range.end.value),
            },
        )
        return [_minute_key_to_utc_timestamp(int(r["minute_key"])) for r in rows]

    def _table(self) -> str:
        return f"{self.database.strip()}.canonical_candles_1m"


def _ensure_tz_utc(dt) -> Any:
    """
    Normalize adapter timestamp value to timezone-aware UTC.

    Parameters:
    - dt: datetime-like value returned by ClickHouse driver.

    Returns:
    - UTC-aware datetime.

    Assumptions/Invariants:
    - Naive datetimes are interpreted as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if getattr(dt, "tzinfo", None) is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _minute_key_to_utc_timestamp(minute_key: int) -> UtcTimestamp:
    """
    Convert UTC epoch-minute key to strongly-typed timestamp.

    Parameters:
    - minute_key: integer key `floor(epoch_seconds / 60)`.

    Returns:
    - `UtcTimestamp` aligned to minute boundary in UTC.
    """
    return UtcTimestamp(datetime.fromtimestamp(minute_key * 60, tz=timezone.utc))


def _dt_to_epoch_ms(dt) -> int:
    """
    Convert datetime-like value to UTC epoch milliseconds for ClickHouse-safe bind parameters.

    Parameters:
    - dt: datetime-like value, naive values are interpreted as UTC.

    Returns:
    - Integer epoch milliseconds.
    """
    return int(_ensure_tz_utc(dt).timestamp() * 1000)
