"""Bounded market-schema bootstrap and grants; no installation cutover."""

from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row


def provision_market_data_access(
    connection: psycopg.Connection[Any], *, application_role: str, maintenance_role: str
) -> None:
    """Grant existing non-privileged principals access without creating roles.

    Application owns no schema/tables and receives no partition/invalidation DDL.
    Maintenance may provision bounded partitions and explicitly invalidate a key.
    Installation provisioning owns role creation and membership outside this helper.
    """
    if application_role == maintenance_role:
        raise ValueError("application and maintenance roles must differ")
    membership = connection.execute(
        "SELECT pg_has_role(%s, %s, 'MEMBER')", (application_role, maintenance_role)
    ).fetchone()
    if membership is None or membership[0]:
        raise ValueError("application role must not inherit maintenance access")
    for role in (application_role, maintenance_role):
        privileged = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_roles r WHERE "
            "(r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication "
            "OR r.rolbypassrls OR r.oid = (SELECT nspowner FROM pg_namespace "
            "WHERE nspname='market_data')) AND pg_has_role(%s,r.oid,'MEMBER'))",
            (role,),
        ).fetchone()
        if privileged is None or privileged[0]:
            raise ValueError("market access roles must not inherit privileged capabilities")
        membership = connection.execute(
            "SELECT pg_has_role(%s, current_user, 'MEMBER')", (role,)
        ).fetchone()
        if membership is None or membership[0]:
            raise ValueError("market access roles must not inherit migration ownership")
        row = connection.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (role,),
        ).fetchone()
        if row is None or any(row):
            raise ValueError("market access requires existing unprivileged roles")
    for role in (application_role, maintenance_role):
        connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA market_data TO {}").format(sql.Identifier(role))
        )
    connection.execute(
        sql.SQL(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA market_data TO {}"
        ).format(sql.Identifier(application_role))
    )
    connection.execute(
        sql.SQL(
            "GRANT EXECUTE ON FUNCTION market_data.ensure_partitions(date,integer), "
            "market_data.invalidate_instrument(integer,text,uuid,text) TO {}"
        ).format(sql.Identifier(maintenance_role))
    )
    connection.execute(
        sql.SQL("GRANT SELECT, DELETE ON market_data.canonical_candles_1m TO {}").format(
            sql.Identifier(maintenance_role)
        )
    )
    connection.execute(
        sql.SQL(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON market_data.instrument_generations, "
            "market_data.verification_leases, market_data.coverage_rebuilds, "
            "market_data.candle_coverage_days, market_data.candle_coverage_summary TO {}"
        ).format(sql.Identifier(maintenance_role))
    )


def provision_current_partitions(connection: psycopg.Connection[Any]) -> None:
    """Provision current and next UTC month; historical import calls the same SQL API."""
    connection.execute("SET LOCAL statement_timeout = '60s'")
    connection.execute("SET LOCAL lock_timeout = '2s'")
    month = datetime.now(UTC).date().replace(day=1)
    connection.execute("SELECT market_data.ensure_partitions(%s, 2)", (month,))


def market_data_schema_ready(connection: psycopg.Connection[Any]) -> bool:
    """Check all required relations and current/next UTC partitions without DDL."""
    with connection.cursor(row_factory=tuple_row) as cursor:
        row = cursor.execute("""
            SELECT bool_and(to_regclass('market_data.' || name) IS NOT NULL)
            FROM unnest(ARRAY['ref_market','ref_instruments','funding_instrument_universe',
                'raw_binance_klines_1m','raw_bybit_klines_1m','canonical_candles_1m',
                'raw_binance_funding_rates','raw_bybit_funding_rates','canonical_funding_rates',
                'instrument_generations','candle_coverage_days','candle_coverage_summary',
                'source_absent_candles_1m','candle_history_verification',
                'verification_leases','coverage_rebuilds']) AS name
        """).fetchone()
        if row is None or not row[0]:
            return False
        columns = cursor.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_schema='market_data' "
            "AND table_name='ref_instruments' "
            "AND column_name IN ('enrichment_updated_at','enrichment_order')"
        ).fetchone()
        if columns is None or columns[0] != 2:
            return False
        row = cursor.execute("""
            SELECT bool_and(EXISTS (
                SELECT 1 FROM pg_inherits i
                WHERE i.inhparent = to_regclass('market_data.' || name)
                  AND i.inhrelid = to_regclass('market_data.' || name || '_' ||
                      to_char(date_trunc('month', now() AT TIME ZONE 'UTC') +
                          make_interval(months => offset_month), 'YYYYMM'))
            )) FROM unnest(ARRAY['raw_binance_klines_1m','raw_bybit_klines_1m',
                'canonical_candles_1m','raw_binance_funding_rates','raw_bybit_funding_rates',
                'canonical_funding_rates']) AS name CROSS JOIN generate_series(0,1) AS offset_month
        """).fetchone()
        return row is not None and bool(row[0])


def provision_requested_partitions(
    connection: psycopg.Connection[Any], *, after_job_id: str | None = None
) -> tuple[int, str | None]:
    """Provision at most twelve missing months per tick from at most 64 active requests.

    Keyset rotation prevents old queued requests from starving later requests.
    Requests keep their original bounds. A missing month fails writes until maintenance
    provisions it; no application role receives schema creation capability.
    """
    connection.execute("SET LOCAL statement_timeout = '60s'")
    connection.execute("SET LOCAL lock_timeout = '2s'")
    requests = connection.execute(
        "WITH demand AS ("
        "SELECT job_id,start_at,end_at FROM market_data_work_requests "
        "WHERE kind='candle_ingestion' AND state IN ('queued','running') "
        "UNION ALL SELECT token,start_at,end_at FROM market_data_stream_recovery "
        "WHERE reason='stream_history_repair' AND state IN ('pending','failed')) "
        "SELECT job_id,start_at,least(end_at,now()+interval '1 month') FROM demand "
        "WHERE (%s::uuid IS NULL OR job_id>%s::uuid) ORDER BY job_id LIMIT 64",
        (after_job_id, after_job_id),
    ).fetchall()
    if not requests and after_job_id is not None:
        return provision_requested_partitions(connection)
    cursor = str(requests[-1][0]) if requests else None
    rows = connection.execute(
        """
        WITH requests AS (
            SELECT * FROM unnest(%s::timestamptz[],%s::timestamptz[]) AS r(start_at,end_at)
        ), months AS (
            SELECT DISTINCT date_trunc('month', value AT TIME ZONE 'UTC')::date AS month
            FROM requests, LATERAL generate_series(
                date_trunc('month',start_at),
                date_trunc('month',end_at - interval '1 microsecond'), interval '1 month'
            ) AS value
        )
        SELECT month FROM months WHERE EXISTS (
            SELECT 1 FROM unnest(ARRAY['raw_binance_klines_1m','raw_bybit_klines_1m',
                'canonical_candles_1m','raw_binance_funding_rates','raw_bybit_funding_rates',
                'canonical_funding_rates']) AS name
            WHERE NOT EXISTS (
                SELECT 1 FROM pg_inherits
                WHERE inhparent=to_regclass('market_data.'||name)
                AND inhrelid=to_regclass('market_data.'||name||'_'||to_char(month,'YYYYMM'))
            )
        ) ORDER BY month LIMIT 12
    """,
        ([r[1] for r in requests], [r[2] for r in requests]),
    ).fetchall()
    for row in rows:
        connection.execute("SELECT market_data.ensure_partitions(%s, 1)", (row[0],))
    provision_current_partitions(connection)
    return len(rows), cursor
