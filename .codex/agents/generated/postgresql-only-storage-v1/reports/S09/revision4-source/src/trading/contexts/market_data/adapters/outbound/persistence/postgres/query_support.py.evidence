"""Shared bounded reads. Missing derived truth fails closed; never scan canonical fallback."""

from datetime import UTC, datetime, timedelta

from psycopg.rows import dict_row

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
BATCH_ROWS = 4096


class CoverageUnavailable(RuntimeError):
    pass


def minute(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC-aware timestamp required")
    delta = value - EPOCH
    # Integer microseconds avoid rounding near exact minute boundaries.
    micros = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    return -(-micros // 60000000)


def timestamp(value):
    return EPOCH + timedelta(minutes=value)


def key(instrument):
    return instrument.market_id.value, str(instrument.symbol)


def rows(conn, query, params=()):
    """Server-side transport batches, even for compact indexes/reference enumerations."""
    with conn.cursor(name="market_read", row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        while batch := cursor.fetchmany(BATCH_ROWS):
            yield from batch


def require_ready(conn, instrument):
    row = conn.execute(
        "SELECT g.generation,g.content_revision,g.funding_revision,g.index_state,s.actual_count "
        "FROM market_data.instrument_generations g LEFT JOIN market_data.candle_coverage_summary s "
        "USING(market_id,symbol,generation) WHERE g.market_id=%s AND g.symbol=%s",
        key(instrument),
    ).fetchone()
    if row is None or row[3] != "ready" or row[4] is None:
        raise CoverageUnavailable("canonical_coverage_unknown")
    return row


# Clipping is in minute-key space: a ts_open inside [start,end) uses ceil at both ends.
CLIPPED_DAYS = """
SELECT day, substring(actual::text FROM greatest(0,%s-day_key)::int+1
    FOR greatest(0,least(1440,%s-day_key)-greatest(0,%s-day_key))::int) AS actual,
    substring(CASE WHEN policy_version=%s THEN source_absent::text ELSE repeat('0',1440) END
    FROM greatest(0,%s-day_key)::int+1
    FOR greatest(0,least(1440,%s-day_key)-greatest(0,%s-day_key))::int) AS absent,
    greatest(day_key,%s) AS first_minute
FROM (SELECT *, (day-date '1970-01-01')::bigint*1440 AS day_key
      FROM market_data.candle_coverage_days
      WHERE market_id=%s AND symbol=%s AND day >= %s AND day <= %s) d ORDER BY day
"""


def clipped_days(conn, instrument, start, end, policy=1):
    return rows(
        conn,
        CLIPPED_DAYS,
        (
            start,
            end,
            start,
            policy,
            start,
            end,
            start,
            start,
            *key(instrument),
            timestamp(start).date(),
            timestamp(max(start, end - 1)).date(),
        ),
    )


# One snapshot, bounded key page; full summary fast path, partial cutoff compact days only.
OBSERVATION = """
WITH keys AS MATERIALIZED (
 SELECT g.*,s.actual_count,s.first_ts,s.last_ts FROM market_data.instrument_generations g
 LEFT JOIN market_data.candle_coverage_summary s USING(market_id,symbol,generation)
 WHERE {predicate} ORDER BY g.market_id,g.symbol LIMIT %(limit)s
)
SELECT k.*, CASE WHEN k.index_state='ready' AND k.actual_count IS NOT NULL
 THEN coalesce(p.n,k.actual_count) END AS clipped_count,
 CASE WHEN p.n IS NOT NULL THEN p.first_minute ELSE extract(epoch FROM k.first_ts)/60 END
 AS first_minute,
 CASE WHEN p.n IS NOT NULL THEN p.last_minute ELSE extract(epoch FROM k.last_ts)/60 END
 AS last_minute
FROM keys k LEFT JOIN LATERAL (
 SELECT coalesce(sum(length(replace(mask,'0',''))),0) AS n,
 min(dkey+position('1' IN mask)-1) FILTER (WHERE position('1' IN mask)>0) AS first_minute,
 max(dkey+length(mask)-position('1' IN reverse(mask)))
 FILTER (WHERE position('1' IN mask)>0) AS last_minute
 FROM (SELECT (day-date '1970-01-01')::bigint*1440 AS dkey,
 substring(actual::text FROM 1 FOR least(1440,greatest(0,%(before)s-
 (day-date '1970-01-01')::bigint*1440))::int) AS mask
 FROM market_data.candle_coverage_days d WHERE d.market_id=k.market_id AND d.symbol=k.symbol
 AND d.generation=k.generation AND day<=%(day)s AND actual_count>0) masks
 WHERE k.last_ts >= %(cutoff)s
) p ON k.last_ts >= %(cutoff)s ORDER BY k.market_id,k.symbol
"""


def observations(conn, *, before, after=(0, ""), limit=50, instrument=None):
    predicate = "(g.market_id,g.symbol)>(%(market)s,%(symbol)s) AND g.market_id IN (1,2,3,4)"
    if instrument is not None:
        predicate = "g.market_id=%(market)s AND g.symbol=%(symbol)s"
        after = key(instrument)
    return list(
        rows(
            conn,
            OBSERVATION.format(predicate=predicate),
            dict(
                market=after[0],
                symbol=after[1],
                limit=limit,
                before=minute(before),
                day=before.date(),
                cutoff=before,
            ),
        )
    )
