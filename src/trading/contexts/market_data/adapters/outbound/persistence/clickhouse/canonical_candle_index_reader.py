from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timezone
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
        - Bounds are computed on minute buckets via `toStartOfMinute(ts_open)`.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            min(toStartOfMinute(ts_open)) AS first,
            max(toStartOfMinute(ts_open)) AS last
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
        first = rows[0].get("first")
        last = rows[0].get("last")
        if first is None or last is None:
            return None
        return (UtcTimestamp(_ensure_tz_utc(first)), UtcTimestamp(_ensure_tz_utc(last)))

    def max_ts_open_lt(self, *, instrument_id: InstrumentId, before: UtcTimestamp) -> UtcTimestamp | None:  # noqa: E501
        """
        Return latest canonical minute strictly before `before`.

        Parameters:
        - instrument_id: instrument whose latest known minute is requested.
        - before: upper bound (exclusive).

        Returns:
        - Latest minute in UTC or `None` when no qualifying rows exist.

        Assumptions/Invariants:
        - Result is normalized to minute bucket via `toStartOfMinute(ts_open)`.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            max(toStartOfMinute(ts_open)) AS last
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open < %(before)s
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "before": _ensure_tz_utc(before.value),
            },
        )
        if not rows:
            return None
        last = rows[0].get("last")
        if last is None:
            return None
        return UtcTimestamp(_ensure_tz_utc(last))

    def daily_counts(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[DailyTsOpenCount]:  # noqa: E501
        """
        Return per-day counts of distinct canonical minute buckets for one range.

        Parameters:
        - instrument_id: instrument being aggregated.
        - time_range: UTC half-open range `[start, end)` for aggregation.

        Returns:
        - Sequence of `(day, count)` rows.

        Assumptions/Invariants:
        - Distinctness is measured on `toStartOfMinute(ts_open)` to ignore sub-minute noise.

        Errors/Exceptions:
        - Raises `RuntimeError` on unexpected day value types from gateway.
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT
            toDate(ts_open) AS day,
            uniqExact(toStartOfMinute(ts_open)) AS cnt
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open >= %(start)s
          AND ts_open < %(end)s
        GROUP BY day
        ORDER BY day
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start": _ensure_tz_utc(time_range.start.value),
                "end": _ensure_tz_utc(time_range.end.value),
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
        - Timestamps are normalized with `toStartOfMinute(ts_open)` in SQL.

        Errors/Exceptions:
        - Propagates gateway/storage errors.

        Side effects:
        - Executes one ClickHouse SELECT query.
        """
        q = f"""
        SELECT DISTINCT
            toStartOfMinute(ts_open) AS ts_open
        FROM {self._table()}
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND ts_open >= %(start)s
          AND ts_open < %(end)s
        ORDER BY ts_open
        """
        rows = self.gateway.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start": _ensure_tz_utc(time_range.start.value),
                "end": _ensure_tz_utc(time_range.end.value),
            },
        )
        return [UtcTimestamp(_ensure_tz_utc(r["ts_open"])) for r in rows]

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
