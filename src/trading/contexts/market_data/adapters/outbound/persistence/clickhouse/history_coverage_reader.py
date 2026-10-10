"""Use the existing exact minute bitmap; never materialize an entire history in Python."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True)
class ClickHouseHistoryCoverageReader:
    gateway: ClickHouseGateway
    database: str = "market_data"

    def counts(
        self, *, starts: Mapping[tuple[int, str], datetime], end_at: datetime
    ) -> Mapping[tuple[int, str], int]:
        if not starts:
            return {}
        if len(starts) > 80000:
            raise ValueError("Too many history instruments")
        ranges = [(m, s, int(t.timestamp() // 60)) for (m, s), t in starts.items()]
        rows = self.gateway.select(
            f"""
            SELECT market_id, symbol,
                bitmapCardinality(bitmapSubsetInRange(minutes, first_minute,
                    toUInt32(%(end)s))) AS candles
            FROM (
                SELECT market_id, symbol, groupBitmapMergeState(minute_bitmap_state) AS minutes
                FROM {self.database}.canonical_candles_1m_stats
                WHERE (market_id, symbol) IN %(keys)s GROUP BY market_id, symbol
            ) AS stored
            INNER JOIN (
                SELECT r.1 AS market_id, r.2 AS symbol, toUInt32(r.3) AS first_minute
                FROM (SELECT arrayJoin(%(ranges)s) AS r)
            ) AS bounds USING (market_id, symbol)
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728
            """,
            {"keys": tuple(starts), "ranges": ranges, "end": int(end_at.timestamp() // 60)},
        )
        return {(int(r["market_id"]), str(r["symbol"])): int(r["candles"]) for r in rows}

    def source_absent_counts(
        self, *, starts: Mapping[tuple[int, str], datetime], end_at: datetime
    ) -> Mapping[tuple[int, str], int]:
        if not starts:
            return {}
        if len(starts) > 80000:
            raise ValueError("Too many history instruments")
        rows = self.gateway.select(
            f"""SELECT market_id, symbol,
                bitmapCardinality(bitmapSubsetInRange(bitmapAndnot(absent, actual),
                    first_minute, toUInt32(%(end)s))) AS candles
            FROM (
                SELECT market_id, symbol, groupBitmapState(minute_key) AS absent
                FROM {self.database}.source_absent_candles_1m
                WHERE (market_id, symbol) IN %(keys)s GROUP BY market_id, symbol
            ) AS source
            LEFT JOIN (
                SELECT market_id, symbol, groupBitmapMergeState(minute_bitmap_state) AS actual
                FROM {self.database}.canonical_candles_1m_stats
                WHERE (market_id, symbol) IN %(keys)s GROUP BY market_id, symbol
            ) AS stored USING (market_id, symbol)
            INNER JOIN (
                SELECT r.1 AS market_id, r.2 AS symbol, toUInt32(r.3) AS first_minute
                FROM (SELECT arrayJoin(%(ranges)s) AS r)
            ) AS bounds USING (market_id, symbol)
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728""",
            {"keys": tuple(starts), "end": int(end_at.timestamp() // 60),
             "ranges": [(m, s, int(t.timestamp() // 60)) for (m, s), t in starts.items()]},
        )
        return {(int(r["market_id"]), str(r["symbol"])): int(r["candles"]) for r in rows}

    def first_gap(self, *, instrument_id: InstrumentId, time_range: TimeRange) -> TimeRange | None:
        start = int(time_range.start.value.timestamp() // 60)
        end = int(time_range.end.value.timestamp() // 60)
        # Locate an incomplete day using bitmap cardinalities, then return at most
        # 1440 minute keys. Full days/years are skipped without provider reads.
        rows = self.gateway.select(
            f"""
            WITH toUInt32(%(start)s) AS a, toUInt32(%(end)s) AS b,
                arrayFirstIndex(d -> bitmapCardinality(bitmapSubsetInRange(minutes,
                    toUInt32(a+d*1440), toUInt32(least(b,a+(d+1)*1440)))) < least(1440,b-a-d*1440),
                    range(toUInt32(intDiv(b-a+1439,1440)))) AS i,
                a+(greatest(i,1)-1)*1440 AS day_start
            SELECT i AS block, day_start,
                bitmapToArray(bitmapSubsetInRange(minutes, toUInt32(day_start),
                    toUInt32(least(b,day_start+1440)))) AS present
            FROM (
                SELECT groupBitmapMergeState(minute_bitmap_state) AS minutes
                FROM (
                    SELECT minute_bitmap_state FROM {self.database}.canonical_candles_1m_stats
                    WHERE market_id=%(market)s AND symbol=%(symbol)s
                    UNION ALL
                    SELECT groupBitmapState(minute_key) AS minute_bitmap_state
                    FROM {self.database}.source_absent_candles_1m
                    WHERE market_id=%(market)s AND symbol=%(symbol)s
                )
            )
            SETTINGS max_threads=1, max_execution_time=10, max_memory_usage=134217728
            """,
            {
                "market": instrument_id.market_id.value,
                "symbol": str(instrument_id.symbol),
                "start": start,
                "end": end,
            },
        )
        if not rows or not int(rows[0]["block"]):
            return None
        row = rows[0]
        present = set(int(x) for x in row["present"])
        last = min(end, int(row["day_start"]) + 1440)
        first = next(x for x in range(int(row["day_start"]), last) if x not in present)
        stop = first + 1
        while stop < last and stop not in present:
            stop += 1
        return TimeRange(
            UtcTimestamp(datetime.fromtimestamp(first * 60, UTC)),
            UtcTimestamp(datetime.fromtimestamp(stop * 60, UTC)),
        )
