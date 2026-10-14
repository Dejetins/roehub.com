"""Generation-first fencing, consumed-provider receipts and contiguous verified prefixes."""

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4

from trading.contexts.market_data.application.services.source_retry import IncompleteHistoryError

from .ingestion_gateway import IngestionGateway
from .query_support import CoverageUnavailable, key, minute, timestamp


@dataclass(frozen=True)
class VerificationFence:
    generation: UUID
    token: UUID


@dataclass(frozen=True)
class PostgresHistoryRangeVerifier:
    gateway: IngestionGateway
    policy_version: int = 1

    def origin(self, *, instrument_id):
        row = self.gateway.run(
            lambda c: c.execute(
                "SELECT first_open_at FROM market_data_history_bounds "
                "WHERE market_id=%s AND symbol=%s AND state='ready'",
                key(instrument_id),
            ).fetchone()
        )
        return row[0] if row else None

    def acquire(self, *, instrument_id, ttl_seconds=120):
        """Acquire before provider I/O; never refresh a stale receipt into a new generation."""
        if not 1 <= ttl_seconds <= 120:
            raise ValueError("invalid_verification_lease_ttl")
        token = uuid4()

        def run(c):
            created = c.execute(
                "INSERT INTO market_data.instrument_generations (market_id,symbol,index_state) "
                "VALUES (%s,%s,'ready') ON CONFLICT DO NOTHING RETURNING generation",
                key(instrument_id),
            ).fetchone()
            if created:
                c.execute(
                    "INSERT INTO market_data.candle_coverage_summary "
                    "(market_id,symbol,generation) VALUES (%s,%s,%s)",
                    (*key(instrument_id), created[0]),
                )
            generation = _lock(c, instrument_id)
            row = c.execute(
                "INSERT INTO market_data.verification_leases AS l "
                "(market_id,symbol,generation,token,expires_at) "
                "VALUES (%s,%s,%s,%s,clock_timestamp()+%s*interval '1 second') "
                "ON CONFLICT(market_id,symbol) DO UPDATE SET token=EXCLUDED.token,"
                "generation=EXCLUDED.generation,expires_at=EXCLUDED.expires_at "
                "WHERE l.expires_at<=clock_timestamp() OR l.generation<>EXCLUDED.generation "
                "RETURNING token",
                (*key(instrument_id), generation, token, ttl_seconds),
            ).fetchone()
            if row is None:
                raise CoverageUnavailable("verification_lease_busy")
            return VerificationFence(generation, token)

        return self.gateway.run(run)

    def release(self, *, instrument_id, fence):
        def run(c):
            row = c.execute(
                "SELECT generation FROM market_data.instrument_generations "
                "WHERE market_id=%s AND symbol=%s FOR UPDATE",
                key(instrument_id),
            ).fetchone()
            if row is None or row[0] != fence.generation:
                return False
            return (
                c.execute(
                    "DELETE FROM market_data.verification_leases WHERE market_id=%s "
                    "AND symbol=%s AND generation=%s AND token=%s RETURNING token",
                    (*key(instrument_id), fence.generation, fence.token),
                ).fetchone()
                is not None
            )

        return self.gateway.run(run)

    def renew(self, *, instrument_id, fence, ttl_seconds=120):
        if not 1 <= ttl_seconds <= 120:
            raise ValueError("invalid_verification_lease_ttl")

        def run(c):
            if _lock(c, instrument_id) != fence.generation:
                raise CoverageUnavailable("stale_source_generation")
            _lease(c, instrument_id, fence)
            c.execute(
                "UPDATE market_data.verification_leases SET expires_at="
                "clock_timestamp()+%s*interval '1 second' WHERE market_id=%s AND symbol=%s",
                (ttl_seconds, *key(instrument_id)),
            )

        self.gateway.run(run)

    def confirm(
        self,
        *,
        instrument_id,
        time_range,
        observed_minutes,
        observed_at,
        accept_absence,
        check_lease,
        fence: VerificationFence,
        origin=None,
    ):
        a, b = minute(time_range.start.value), minute(time_range.end.value)
        if (
            not 0 < b - a <= 10080
            or timestamp(a) != time_range.start.value
            or timestamp(b) != time_range.end.value
        ):
            raise ValueError("verification requires aligned <=7 day range")
        if not observed_minutes.issubset(range(a, b)):
            raise ValueError("observations outside verified range")
        if origin is not None and (
            origin > time_range.start.value or timestamp(minute(origin)) != origin
        ):
            raise ValueError("invalid verification origin")

        def run(c):
            generation = _lock(c, instrument_id)
            if generation != fence.generation:
                raise CoverageUnavailable("stale_source_generation")
            _lease(c, instrument_id, fence)
            check_lease()
            # At most 10080 rows, direct partition bounds. This is an explicit verifier,
            # not an index-reader fallback. Returned but uncommitted rows fail closed.
            actual = {
                int(r[0])
                for r in c.execute(
                    "SELECT extract(epoch FROM ts_open)::bigint/60 "
                    "FROM market_data.canonical_candles_1m WHERE market_id=%s AND symbol=%s "
                    "AND ts_open>=%s AND ts_open<%s",
                    (*key(instrument_id), time_range.start.value, time_range.end.value),
                )
            }
            if not observed_minutes.issubset(actual):
                raise IncompleteHistoryError()
            absent = set(range(a, b)) - actual if accept_absence else set()
            observation = uuid4()
            # Replace audited absence in this receipt only; failed callbacks roll it all back.
            c.execute(
                "DELETE FROM market_data.source_absent_candles_1m "
                "WHERE market_id=%s AND symbol=%s AND minute_key>=%s AND minute_key<%s",
                (*key(instrument_id), a, b),
            )
            if absent:
                with c.cursor().copy(
                    "COPY market_data.source_absent_candles_1m "
                    "(market_id,symbol,minute_key,generation,observed_at,policy_version,"
                    "observation_id,source,range_start,range_end) FROM STDIN"
                ) as copy:
                    for m in sorted(absent):
                        copy.write_row(
                            (
                                *key(instrument_id),
                                m,
                                generation,
                                observed_at,
                                self.policy_version,
                                observation,
                                "consumed_provider_response",
                                time_range.start.value,
                                time_range.end.value,
                            )
                        )
            actual_changed = False
            coverage_changed = False
            for day_start in range(a // 1440 * 1440, b, 1440):
                day = timestamp(day_start).date()
                old = c.execute(
                    "SELECT actual::text,source_absent::text,policy_version FROM "
                    "market_data.candle_coverage_days WHERE market_id=%s AND symbol=%s AND day=%s",
                    (*key(instrument_id), day),
                ).fetchone()
                am = list(old[0] if old else "0" * 1440)
                sm = list(old[1] if old and old[2] == self.policy_version else "0" * 1440)
                for m in range(max(a, day_start), min(b, day_start + 1440)):
                    am[m - day_start] = "1" if m in actual else "0"
                    sm[m - day_start] = "1" if m in absent else "0"
                actual_changed |= "".join(am) != (old[0] if old else "0" * 1440)
                coverage_changed |= "".join(sm) != (old[1] if old else "0" * 1440)
                _day(
                    c, instrument_id, generation, day, "".join(am), "".join(sm), self.policy_version
                )
            _summary(c, instrument_id, generation)
            # A verification may repair stale masks; revisions conservatively invalidate inputs.
            if actual_changed:
                c.execute(
                    "UPDATE market_data.instrument_generations SET content_revision="
                    "content_revision+1 WHERE market_id=%s AND symbol=%s",
                    key(instrument_id),
                )
            if origin is None:
                # Absence receipt is valid, but availability discovery has not established
                # a history origin. Never invent a verified prefix from a requested suffix.
                check_lease()
                _lease(c, instrument_id, fence)
                return
            previous = c.execute(
                "SELECT origin,verified_until,policy_version FROM "
                "market_data.candle_history_verification WHERE market_id=%s AND symbol=%s",
                key(instrument_id),
            ).fetchone()
            start = origin
            if previous and previous[0] == origin and previous[2] == self.policy_version:
                start = previous[1]
                if actual_changed or coverage_changed:
                    start = min(start, time_range.start.value)
            # Advance only one bounded slice; a disjoint receipt cannot bridge the unknown gap.
            stop = start
            for offset in range(7):
                day_start = minute(stop) // 1440 * 1440
                row = c.execute(
                    "SELECT (actual | CASE WHEN policy_version=%s THEN source_absent "
                    "ELSE 0::bit(1440) END)::text FROM market_data.candle_coverage_days "
                    "WHERE market_id=%s AND symbol=%s AND day=%s",
                    (self.policy_version, *key(instrument_id), timestamp(day_start).date()),
                ).fetchone()
                if row is None:
                    break
                bits = row[0][minute(stop) - day_start :]
                gap = bits.find("0")
                stop += timedelta(minutes=len(bits) if gap < 0 else gap)
                if gap >= 0:
                    break
            c.execute(
                "INSERT INTO market_data.candle_history_verification "
                "(market_id,symbol,generation,origin,verified_until,policy_version,lease_token,"
                "verified_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(market_id,symbol) DO UPDATE SET origin=EXCLUDED.origin,"
                "verified_until=EXCLUDED.verified_until,policy_version=EXCLUDED.policy_version,"
                "lease_token=EXCLUDED.lease_token,verified_at=EXCLUDED.verified_at",
                (
                    *key(instrument_id),
                    generation,
                    origin,
                    stop,
                    self.policy_version,
                    fence.token,
                    observed_at,
                ),
            )
            check_lease()
            _lease(c, instrument_id, fence)

        self.gateway.run(run)


def _lock(c, instrument):
    row = c.execute(
        "SELECT generation,index_state FROM market_data.instrument_generations "
        "WHERE market_id=%s AND symbol=%s FOR UPDATE",
        key(instrument),
    ).fetchone()
    if row is None or row[1] != "ready":
        raise CoverageUnavailable("canonical_coverage_unknown")
    return row[0]


def _lease(c, instrument, fence):
    row = c.execute(
        "SELECT token FROM market_data.verification_leases WHERE market_id=%s "
        "AND symbol=%s AND generation=%s AND token=%s AND expires_at>clock_timestamp() "
        "FOR UPDATE",
        (*key(instrument), fence.generation, fence.token),
    ).fetchone()
    if row is None:
        raise CoverageUnavailable("stale_verification_lease")


def _day(c, instrument, generation, day, actual, absent, policy):
    c.execute(
        "INSERT INTO market_data.candle_coverage_days "
        "(market_id,symbol,day,generation,actual,source_absent,policy_version) "
        "VALUES (%s,%s,%s,%s,%s::bit(1440),%s::bit(1440),%s) "
        "ON CONFLICT(market_id,symbol,day) DO UPDATE SET actual=EXCLUDED.actual,"
        "source_absent=EXCLUDED.source_absent,policy_version=EXCLUDED.policy_version",
        (*key(instrument), day, generation, actual, absent, policy),
    )


def _summary(c, instrument, generation):
    c.execute(
        """INSERT INTO market_data.candle_coverage_summary
        (market_id,symbol,generation,actual_count,first_ts,last_ts)
        SELECT %s,%s,%s,coalesce(sum(actual_count),0),
        min(day::timestamp AT TIME ZONE 'UTC'+(position('1' IN actual::text)-1)*interval '1 minute')
            FILTER(WHERE actual_count>0),
        max(day::timestamp AT TIME ZONE 'UTC'+(1440-position('1' IN reverse(actual::text)))
            *interval '1 minute') FILTER(WHERE actual_count>0)
        FROM market_data.candle_coverage_days WHERE market_id=%s AND symbol=%s
        ON CONFLICT(market_id,symbol) DO UPDATE SET actual_count=EXCLUDED.actual_count,
        first_ts=EXCLUDED.first_ts,last_ts=EXCLUDED.last_ts""",
        (*key(instrument), generation, *key(instrument)),
    )
