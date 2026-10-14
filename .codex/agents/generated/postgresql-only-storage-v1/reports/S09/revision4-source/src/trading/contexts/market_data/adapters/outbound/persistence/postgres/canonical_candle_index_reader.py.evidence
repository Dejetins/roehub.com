"""Exact actual-minute index. Absences never enter candle bounds or counts."""

from dataclasses import dataclass

from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    DailyTsOpenCount,
)
from trading.shared_kernel.primitives import UtcTimestamp

from .ingestion_gateway import IngestionGateway
from .query_support import (
    CoverageUnavailable,
    clipped_days,
    key,
    minute,
    observations,
    require_ready,
    timestamp,
)


@dataclass(frozen=True)
class PostgresCanonicalCandleIndexReader:
    gateway: IngestionGateway

    def bounds(self, instrument_id):
        def read(c):
            require_ready(c, instrument_id)
            row = c.execute(
                "SELECT first_ts,last_ts FROM market_data.candle_coverage_summary "
                "WHERE market_id=%s AND symbol=%s",
                key(instrument_id),
            ).fetchone()
            return (UtcTimestamp(row[0]), UtcTimestamp(row[1])) if row[0] is not None else None

        return self.gateway.run(read, read_only=True)

    def bounds_1m(self, *, instrument_id, before):
        def read(c):
            result = observations(c, before=before.value, instrument=instrument_id, limit=1)
            if not result or result[0]["clipped_count"] is None:
                raise CoverageUnavailable("canonical_coverage_unknown")
            row = result[0]

            def convert(value):
                return None if value is None else UtcTimestamp(timestamp(int(value)))

            return convert(row["first_minute"]), convert(row["last_minute"])

        return self.gateway.run(read, read_only=True)

    def max_ts_open_lt(self, *, instrument_id, before, after=None):
        _, last = self.bounds_1m(instrument_id=instrument_id, before=before)
        return last if last is not None and (after is None or last.value >= after.value) else None

    def daily_counts(self, *, instrument_id, time_range):
        def read(c):
            require_ready(c, instrument_id)
            return [
                DailyTsOpenCount(day=r["day"], count=r["actual"].count("1"))
                for r in clipped_days(
                    c, instrument_id, minute(time_range.start.value), minute(time_range.end.value)
                )
                if "1" in r["actual"]
            ]

        return self.gateway.run(read, read_only=True)

    def distinct_ts_opens(self, *, instrument_id, time_range):
        def read(c):
            require_ready(c, instrument_id)
            return [
                UtcTimestamp(timestamp(r["first_minute"] + i))
                for r in clipped_days(
                    c, instrument_id, minute(time_range.start.value), minute(time_range.end.value)
                )
                for i, bit in enumerate(r["actual"])
                if bit == "1"
            ]

        return self.gateway.run(read, read_only=True)
