"""S08 simultaneous source, recovery and audit failure on one real PG16 server."""

import json
import os
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from apps.migrations.storage import apply_postgres_migrations
from tests.real_infra.market_data.test_transactional_ingestion_postgres import (
    BASE,
    candle,
    snapshot,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.candle_repair_audit_repository import (  # noqa: E501
    PostgresCandleRepairAuditRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.gateway import (
    PsycopgMarketDataPostgresGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)
from trading.contexts.market_data.application.dto import (
    MarketDataCandleRepairAuditEvent,
    RestFillTask,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROEHUB_S03_DISPOSABLE_DSN"), reason="explicit disposable PG required"
)


@pytest.fixture
def db():
    container = os.environ["ROEHUB_S03_DISPOSABLE_CONTAINER"]
    assert container == "roehub-s03-s08-functional"
    config = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
    assert config["Config"]["Labels"]["roehub.stage"] == "S08"
    mapping = config["NetworkSettings"]["Ports"]["5432/tcp"]
    assert len(mapping) == 1 and mapping[0]["HostIp"] == "127.0.0.1"
    dsn = make_conninfo(
        host="127.0.0.1", port=mapping[0]["HostPort"], user="postgres", dbname="postgres"
    )
    inside = subprocess.check_output(
        [
            "docker",
            "exec",
            container,
            "psql",
            "-U",
            "postgres",
            "-Atc",
            "SELECT system_identifier FROM pg_control_system()",
        ],
        text=True,
    ).strip()
    name = "s08_outage_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as c:
        row = c.execute("SELECT system_identifier FROM pg_control_system()").fetchone()
        assert row is not None and str(row[0]) == inside
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    target = make_conninfo(dsn, dbname=name)
    root = Path(__file__).resolve().parents[3]
    try:
        apply_postgres_migrations(
            target, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
        )
        with psycopg.connect(target) as c:
            c.execute("SELECT market_data.ensure_partitions(%s,1)", (date(2026, 2, 1),))
        yield target
    finally:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_source_recovery_and_audit_share_outage_without_false_ack(db):
    container = os.environ["ROEHUB_S03_DISPOSABLE_CONTAINER"]
    label = subprocess.check_output(
        ["docker", "inspect", "--format", '{{ index .Config.Labels "roehub.stage" }}', container],
        text=True,
    ).strip()
    assert label == "S08" and container == "roehub-s03-s08-functional"
    gateway = IngestionGateway(db)
    writer = PostgresRawKlineWriter(gateway)
    first = candle("SHAREDOUTAGE")
    writer.write_1m([first])
    before = snapshot(db, "SHAREDOUTAGE")
    window = TimeRange(UtcTimestamp(BASE), UtcTimestamp(BASE + timedelta(minutes=1)))
    recovery = PostgresStreamRecoveryStore(db)
    pending = recovery.remember(RestFillTask(first.candle.instrument_id, window, "ws_recovery"))
    audit = PostgresCandleRepairAuditRepository(gateway=PsycopgMarketDataPostgresGateway(dsn=db))
    event = MarketDataCandleRepairAuditEvent(
        event_id=uuid4(),
        correlation_id="s08-shared-outage",
        instrument_id=first.candle.instrument_id,
        instrument_key="1:SHAREDOUTAGE",
        time_range=window,
        status="succeeded",
        sources_attempted=(),
        restored_ts_opens=(UtcTimestamp(BASE),),
        missing_ts_opens=(),
        created_at=UtcTimestamp(BASE + timedelta(days=1)),
    )
    subprocess.run(["docker", "stop", "-t", "1", container], check=True, capture_output=True)
    try:
        with pytest.raises(TemporaryIngestionError):
            writer.write_1m([candle("SHAREDOUTAGE", minute=1)])
        with pytest.raises(psycopg.OperationalError):
            recovery.complete(pending)
        with pytest.raises(psycopg.OperationalError):
            audit.record(event=event)
    finally:
        subprocess.run(["docker", "start", container], check=True, capture_output=True)
        deadline = time.monotonic() + 30
        while True:
            try:
                with psycopg.connect(db):
                    break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        gateway.close()
    assert snapshot(db, "SHAREDOUTAGE") == before
    assert recovery.pending() == [pending]
    assert audit.get_by_id(event_id=event.event_id) is None
    assert audit.record(event=event) == event
    recovery.complete(pending)
    assert recovery.pending() == []
    print(
        "shared_pg_outage: source write, recovery ACK and audit all failed; "
        "durable source unchanged, recovery pending, no false audit; "
        "post-restart retry passed"
    )
