"""Compact range cardinalities and first gap; no canonical scan or minute expansion."""

from dataclasses import dataclass

from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

from .ingestion_gateway import IngestionGateway
from .query_support import CoverageUnavailable, key, minute, require_ready, rows, timestamp

COUNTS = """
WITH requested AS (SELECT * FROM unnest(%s::int[],%s::text[],%s::bigint[])
 AS t(market_id,symbol,a))
SELECT r.market_id,r.symbol,g.index_state,s.actual_count,
coalesce(sum(bit_count(substring(({mask})::bit varying
 FROM greatest(0,r.a-(d.day-date '1970-01-01')::bigint*1440)::int+1
 FOR greatest(0,least(1440,%s-(d.day-date '1970-01-01')::bigint*1440)
 -greatest(0,r.a-(d.day-date '1970-01-01')::bigint*1440))::int))),0) AS n
FROM requested r LEFT JOIN market_data.instrument_generations g USING(market_id,symbol)
LEFT JOIN market_data.candle_coverage_summary s USING(market_id,symbol,generation)
LEFT JOIN market_data.candle_coverage_days d ON d.market_id=r.market_id AND d.symbol=r.symbol
AND d.generation=g.generation AND d.day >= date '1970-01-01'+(r.a/1440)::int
AND d.day<=%s GROUP BY r.market_id,r.symbol,g.index_state,s.actual_count
"""
FIRST_GAP = """
SELECT day, substring(covered::text FROM greatest(0,%s-dkey)::int+1
 FOR greatest(0,least(1440,%s-dkey)-greatest(0,%s-dkey))::int) AS mask,
 greatest(dkey,%s) AS first_minute
FROM (SELECT days.day::date AS day,(days.day::date-date '1970-01-01')::bigint*1440 AS dkey,
 coalesce(d.actual | CASE WHEN d.policy_version=%s THEN d.source_absent
 ELSE 0::bit(1440) END,0::bit(1440)) AS covered
 FROM generate_series(%s::date,%s::date,interval '1 day') days(day)
 LEFT JOIN market_data.candle_coverage_days d ON d.market_id=%s AND d.symbol=%s
 AND d.day=days.day::date) q
WHERE position('0' IN substring(covered::text FROM greatest(0,%s-dkey)::int+1
 FOR greatest(0,least(1440,%s-dkey)-greatest(0,%s-dkey))::int))>0
ORDER BY day LIMIT 1
"""


@dataclass(frozen=True)
class PostgresHistoryCoverageReader:
    gateway: IngestionGateway
    policy_version: int = 1

    def _counts(self, starts, end_at, absent):
        if len(starts) > 80000:
            raise ValueError("Too many history instruments")
        if not starts:
            return {}
        a = [minute(t) for t in starts.values()]
        b = minute(end_at)
        mask = (
            (
                "CASE WHEN d.policy_version="
                + str(int(self.policy_version))
                + " THEN d.source_absent & ~d.actual ELSE 0::bit(1440) END"
            )
            if absent
            else "d.actual"
        )

        def read(c):
            result = {}
            for r in rows(
                c,
                COUNTS.format(mask=mask),
                ([k[0] for k in starts], [k[1] for k in starts], a, b, timestamp(b - 1).date()),
            ):
                if r["index_state"] != "ready" or r["actual_count"] is None:
                    raise CoverageUnavailable("canonical_coverage_unknown")
                result[(r["market_id"], r["symbol"])] = int(r["n"])
            return result

        return self.gateway.run(read)

    def counts(self, *, starts, end_at):
        return self._counts(starts, end_at, False)

    def source_absent_counts(self, *, starts, end_at):
        return self._counts(starts, end_at, True)

    def first_gap(self, *, instrument_id, time_range):
        def read(c):
            require_ready(c, instrument_id)
            a, b = minute(time_range.start.value), minute(time_range.end.value)
            frontier = c.execute(
                "SELECT origin,verified_until FROM "
                "market_data.candle_history_verification WHERE market_id=%s AND symbol=%s "
                "AND policy_version=%s",
                (*key(instrument_id), self.policy_version),
            ).fetchone()
            if frontier and frontier[0] <= timestamp(a):
                a = max(a, minute(frontier[1]))
            if a >= b:
                return None
            found = list(
                rows(
                    c,
                    FIRST_GAP,
                    (
                        a,
                        b,
                        a,
                        a,
                        self.policy_version,
                        timestamp(a).date(),
                        timestamp(b - 1).date(),
                        *key(instrument_id),
                        a,
                        b,
                        a,
                    ),
                )
            )
            if not found:
                return None
            r = found[0]
            offset = r["mask"].index("0")
            stop = r["mask"].find("1", offset)
            if stop < 0:
                stop = len(r["mask"])
            return TimeRange(
                UtcTimestamp(timestamp(r["first_minute"] + offset)),
                UtcTimestamp(timestamp(r["first_minute"] + stop)),
            )

        return self.gateway.run(read, read_only=True)
