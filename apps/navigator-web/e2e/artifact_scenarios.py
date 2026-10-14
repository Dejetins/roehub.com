"""Write only deterministic synthetic observations in S07's dedicated container."""

from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from trading.platform.postgres_budget import connect as budget_connect

DSN = "host=127.0.0.1 port=25437 dbname=postgres user=postgres"
with budget_connect(DSN) as conn:
    schemas = conn.execute(
        (
            "SELECT schema_name FROM information_schema.schemata WHERE "
            "schema_name LIKE 'artifact_management_test_%'"
        )
    ).fetchall()
    assert len(schemas) == 1, "one task-owned fixture required"
    conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schemas[0][0])))
    row = conn.execute(
        (
            "SELECT p.organization_id,o.installation_id FROM "
            "backtest_artifact_policies p JOIN identity_organizations o "
            "USING(organization_id) LIMIT 1"
        )
    ).fetchone()
    assert row is not None
    org, installation = row
    source = {
        "history": {
            "kind": "range",
            "start_utc": "2026-10-01T00:00:00Z",
            "end_utc": "2026-10-08T00:00:00Z",
        },
        "revision": "s07-fixture",
        "watermark": "2026-10-08T00:00:00Z",
        "compute_identity": "v2",
        "funding_identity": "spot",
    }
    old = {**source, "watermark": "2026-10-07T00:00:00Z"}
    published = {
        "state": "ready",
        "source": old,
        "timeframes": {
            "1m": {
                "state": "ready",
                "closed_bars": 8640,
                "start_ms": 1790812800000,
                "end_ms": 1791331200000,
                "missing_bars": 0,
            },
            "3d": {
                "state": "insufficient_history",
                "closed_bars": 0,
                "missing_bars": 0,
                "retry_after_source_end_ms": 1791590400000,
            },
        },
    }
    conn.execute(
        (
            "INSERT INTO backtest_artifact_source_state VALUES "
            "('binance','spot','BTCUSDT',%s,%s,NULL) ON CONFLICT "
            "(exchange,market_type,symbol) DO UPDATE SET desired_target=exclud"
            "ed.desired_target,published=excluded.published"
        ),
        (Jsonb({"source": source}), Jsonb(published)),
    )
    conn.execute(
        (
            "UPDATE backtest_artifact_policies SET "
            "last_success='2026-10-07T00:10:00Z' WHERE symbol='BTCUSDT'"
        )
    )
    for kind, state, phase, completed, total in [
        ("base", "error", "base", 7, 10),
        ("derivatives", "running", "derivatives", None, None),
    ]:
        rid = uuid4()
        conn.execute(
            (
                "INSERT INTO backtest_artifact_requests(request_id,organization_id"
                ",installation_id,organization_dedupe_key,exchange,market_type,sym"
                "bol,kind,trigger,snapshot,state,reason) VALUES "
                "(%s,%s,%s,%s,'binance','spot','BTCUSDT',%s,'manual','{}',%s,%s)"
            ),
            (
                rid,
                org,
                installation,
                str(rid),
                kind,
                state,
                "source_unavailable" if state == "error" else None,
            ),
        )
        progress = {
            "phase": phase,
            "timeframe": "1h" if kind == "base" else None,
            "completed_units": completed,
            "total_units": total,
            "unit": "bars",
            "progress_epoch": 2,
            "observed_at": "2026-10-08T12:00:00Z",
        }
        conn.execute(
            (
                "INSERT INTO "
                "backtest_artifact_events(request_id,state,reason,progress) VALUES"
                " (%s,%s,%s,%s)"
            ),
            (rid, state, "source_unavailable" if state == "error" else None, Jsonb(progress)),
        )
print(
    (
        "Seeded disposable source coverage, last-good failure and "
        "independent measured/unknown progress; no worker ran."
    )
)
