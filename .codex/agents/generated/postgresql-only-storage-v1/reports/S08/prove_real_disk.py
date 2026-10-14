"""Bounded real ENOSPC proof; operates only on an S08 labelled tmpfs PG16."""

import json
import subprocess
from datetime import date
from pathlib import Path
from uuid import uuid4

import psycopg

from apps.migrations.storage import apply_postgres_migrations
from tests.real_infra.market_data.test_transactional_ingestion_postgres import candle, snapshot
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,
)

name = "roehub-s08-disk"
config = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
assert config["Config"]["Labels"]["roehub.stage"] == "S08"
assert "size=256m" in config["HostConfig"]["Tmpfs"]["/var/lib/postgresql/data"]
mapping = config["NetworkSettings"]["Ports"]["5432/tcp"]
assert len(mapping) == 1 and mapping[0]["HostIp"] == "127.0.0.1"
dsn = f"host=127.0.0.1 port={int(mapping[0]['HostPort'])} dbname=postgres user=postgres"
inside = subprocess.check_output(
    [
        "docker",
        "exec",
        name,
        "psql",
        "-U",
        "postgres",
        "-Atc",
        "SELECT system_identifier FROM pg_control_system()",
    ],
    text=True,
).strip()
with psycopg.connect(dsn) as c:
    assert (
        str(c.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0]) == inside
    )
root = Path.cwd()
apply_postgres_migrations(
    dsn, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
)
with psycopg.connect(dsn) as c:
    c.execute("SELECT market_data.ensure_partitions(%s,1)", (date(2026, 2, 1),))
symbol = "DISKFAIL" + uuid4().hex[:8].upper()
gateway = IngestionGateway(dsn)
hints = []
writer = StoredCandleSignalWriter(
    delegate=PostgresRawKlineWriter(gateway),
    notify=lambda instrument: hints.append(str(instrument)),
)
# Seed an unrelated committed row to prove the surviving baseline.
writer.write_1m([candle("DISKBASE")])
before = snapshot(dsn, "DISKBASE")
hints.clear()
fill = subprocess.run(
    [
        "docker",
        "exec",
        name,
        "sh",
        "-c",
        "dd if=/dev/zero of=/var/lib/postgresql/data/s08-ballast bs=1M",
    ],
    capture_output=True,
    text=True,
)
assert fill.returncode != 0 and "No space left on device" in fill.stderr
failed = None
try:
    writer.write_1m([candle(symbol, minute=n) for n in range(1000)])
except Exception as error:
    failed = type(error).__name__
finally:
    subprocess.run(
        ["docker", "exec", name, "rm", "/var/lib/postgresql/data/s08-ballast"], check=True
    )
assert failed is not None, "real disk exhaustion must reject the write"
assert not hints, "no successful stored-candle hint before commit"
assert snapshot(dsn, "DISKBASE") == before
rows, summary, days = snapshot(dsn, symbol)
assert rows == [] and summary is None and days == []
with psycopg.connect(dsn) as c:
    raw = c.execute(
        "SELECT count(*) FROM market_data.raw_binance_klines_1m WHERE symbol=%s", (symbol,)
    ).fetchone()
    assert raw == (0,)
writer.write_1m([candle(symbol, minute=n) for n in range(1000)])
rows, summary, days = snapshot(dsn, symbol)
assert len(rows) == 1000 and summary[0] == 1000 and hints
print(
    json.dumps(
        {
            "result": "passed",
            "filesystem": "dedicated 256MiB tmpfs",
            "physical_ENOSPC": True,
            "write_exception": failed,
            "false_commit_hint": False,
            "partial_raw_canonical_coverage": False,
            "retry_count": len(rows),
            "baseline_preserved": True,
        }
    )
)
gateway.close()
