"""Explicit generation-fenced administrative repair, never invoked by ordinary reads."""

from dataclasses import dataclass
from datetime import timedelta

from .history_range_verifier import _day, _summary
from .ingestion_gateway import IngestionGateway
from .query_support import CoverageUnavailable, key, timestamp


@dataclass(frozen=True)
class PostgresCoverageMaintenance:
    gateway: IngestionGateway

    def begin(self, *, instrument_id, expected_generation, reason, delete_range=None):
        """Invalidation requires the maintenance principal. Optional delete is <=7 days.

        Canonical/raw retention outside the explicit range is unchanged. Rebuild is
        resumable by generation and next_day, while ingestion fails closed.
        """
        if (
            delete_range
            and not 0
            < (delete_range.end.value - delete_range.start.value).total_seconds()
            <= 7 * 86400
        ):
            raise ValueError("delete requires bounded <=7 day range")

        def run(c):
            generation = c.execute(
                "SELECT market_data.invalidate_instrument(%s,%s,%s,%s)",
                (*key(instrument_id), expected_generation, reason),
            ).fetchone()[0]
            c.execute(
                "DELETE FROM market_data.verification_leases WHERE market_id=%s AND symbol=%s",
                key(instrument_id),
            )
            if delete_range:
                c.execute(
                    "DELETE FROM market_data.canonical_candles_1m WHERE market_id=%s "
                    "AND symbol=%s AND ts_open>=%s AND ts_open<%s",
                    (*key(instrument_id), delete_range.start.value, delete_range.end.value),
                )
            # Ordered index endpoints, never aggregate all canonical rows.
            bounds = []
            for direction in ("ASC", "DESC"):
                row = c.execute(
                    "SELECT ts_open FROM market_data.canonical_candles_1m "
                    "WHERE market_id=%s AND symbol=%s ORDER BY ts_open " + direction + " LIMIT 1",
                    key(instrument_id),
                ).fetchone()
                bounds.append(row[0].date() if row else timestamp(0).date())
            end = bounds[1] + timedelta(days=1)
            c.execute(
                "INSERT INTO market_data.coverage_rebuilds "
                "(market_id,symbol,generation,next_day,end_day) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT(market_id,symbol) DO UPDATE SET generation=EXCLUDED.generation,"
                "next_day=EXCLUDED.next_day,end_day=EXCLUDED.end_day",
                (*key(instrument_id), generation, bounds[0], end),
            )
            c.execute(
                "UPDATE market_data.instrument_generations SET index_state='rebuilding' "
                "WHERE market_id=%s AND symbol=%s",
                key(instrument_id),
            )
            return generation

        return self.gateway.run(run)

    def step(self, *, instrument_id, generation, days=7):
        if not 1 <= days <= 7:
            raise ValueError("rebuild step requires 1..7 days")

        def run(c):
            row = c.execute(
                "SELECT generation,index_state FROM market_data.instrument_generations "
                "WHERE market_id=%s AND symbol=%s FOR UPDATE",
                key(instrument_id),
            ).fetchone()
            if row is None or row[0] != generation or row[1] != "rebuilding":
                raise CoverageUnavailable("stale_rebuild_generation")
            start, end = c.execute(
                "SELECT next_day,end_day FROM market_data.coverage_rebuilds "
                "WHERE market_id=%s AND symbol=%s AND generation=%s",
                (*key(instrument_id), generation),
            ).fetchone()
            stop = min(end, start + timedelta(days=days))
            # At most 10080 canonical rows. No attempt to allocate the complete history.
            records = c.execute(
                "SELECT ts_open::date,bit_or(set_bit(0::bit(1440),"
                "(extract(hour FROM ts_open)*60+extract(minute FROM ts_open))::int,1))::text "
                "FROM market_data.canonical_candles_1m WHERE market_id=%s AND symbol=%s "
                "AND ts_open>=%s AND ts_open<%s GROUP BY ts_open::date ORDER BY ts_open::date",
                (*key(instrument_id), start, stop),
            ).fetchall()
            for day, mask in records:
                _day(c, instrument_id, generation, day, mask, "0" * 1440, 1)
            c.execute(
                "UPDATE market_data.coverage_rebuilds SET next_day=%s "
                "WHERE market_id=%s AND symbol=%s",
                (stop, *key(instrument_id)),
            )
            if stop == end:
                _summary(c, instrument_id, generation)
                c.execute(
                    "UPDATE market_data.instrument_generations SET index_state='ready' "
                    "WHERE market_id=%s AND symbol=%s",
                    key(instrument_id),
                )
                c.execute(
                    "DELETE FROM market_data.coverage_rebuilds WHERE market_id=%s AND symbol=%s",
                    key(instrument_id),
                )
            return stop == end

        return self.gateway.run(run)
