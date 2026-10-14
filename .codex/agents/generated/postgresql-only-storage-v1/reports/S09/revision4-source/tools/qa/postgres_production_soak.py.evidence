"""S09 production adapter proof on an explicitly owned disposable Linux PG16 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq

from apps.migrations.storage import apply_postgres_migrations, provision_runtime_roles
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.platform.postgres_budget import LIMITS, connect
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)

ROOT = Path("/workspace")
OUT = ROOT / ".codex/agents/generated/postgresql-only-storage-v1/reports/S09"
SOURCE = ROOT / ".local_artifacts/pg-comparison-20261013"
DSN = "host=roehub-s09-pg dbname=s09 user=postgres sslmode=disable"
APP_DSN = "host=roehub-s09-pg dbname=s09 user=roehub_app sslmode=disable"


def emit(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


def candle(row):
    return CandleWithMeta(
        Candle(
            InstrumentId(MarketId(row["market_id"]), Symbol(row["symbol"])),
            UtcTimestamp(row["ts_open"]),
            UtcTimestamp(row["ts_close"]),
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume_base"],
            row["volume_quote"],
        ),
        CandleMeta(
            source=row["source"],
            ingested_at=UtcTimestamp(row["ingested_at"]),
            ingest_id=UUID(row["ingest_id"]),
            instrument_key=row["instrument_key"],
            trades_count=row["trades_count"],
            taker_buy_volume_base=row["taker_buy_volume_base"],
            taker_buy_volume_quote=row["taker_buy_volume_quote"],
        ),
    )


def setup():
    assert os.environ.get("ROEHUB_PG_BUDGET_REQUIRED") == "1"
    registry = Path(os.environ["ROEHUB_PG_BUDGET_ROOT"])
    # Existing unknown records are preserved and remain quarantined.
    registry.chmod(0o755)
    (registry / "authority.json").touch(mode=0o600)
    for kind, count in LIMITS.items():
        for i in range(count):
            (registry / f"{kind}-{i}.json").touch(mode=0o600)
    apply_postgres_migrations(
        DSN, repo_root=ROOT, manifest_path=ROOT / "migrations/postgres/manifest.json"
    )
    with connect(DSN, budget_class="admin", autocommit=True) as c:
        assert 160000 <= c.info.server_version < 170000
        row = c.execute("SHOW max_connections").fetchone()
        assert row is not None and row[0] == "48"
        c.execute("CREATE ROLE roehub_app LOGIN")
        c.execute("CREATE ROLE roehub_maintenance LOGIN")
        c.execute("CREATE ROLE roehub_monitor LOGIN")
        for p in sorted((SOURCE / "candles").glob("*.parquet")):
            month = datetime.strptime(p.stem, "%Y%m").date()
            c.execute("SELECT market_data.ensure_partitions(%s,1)", (month,))
        for year in range(2030, 2034):
            c.execute("SELECT market_data.ensure_partitions(%s,12)", (f"{year}-01-01",))
    provision_runtime_roles(DSN)
    emit("setup.json", {"pg16": True, "physical_budget": LIMITS, "server_registry_mount": False})


def load():
    manifest = json.loads((SOURCE / "snapshot.json").read_text())
    expected = {r["path"]: r["sha256"] for r in manifest["parts"]}
    gw = IngestionGateway(APP_DSN, deadline_s=30)
    writer = PostgresRawKlineWriter(gw)
    n = 0
    started = time.monotonic()
    resume = OUT / "import-at-interruption.json"
    progress = json.loads(resume.read_text())["parts"] if resume.exists() else []
    n = sum(p["rows"] for p in progress)
    completed = {p["part"]: p for p in progress}
    try:
        for p in sorted((SOURCE / "candles").glob("*.parquet")):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest == expected[p.name], p.name
            if p.name in completed:
                assert completed[p.name]["sha256"] == digest
                continue
            count = 0
            for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
                data = [candle(r) for r in batch.to_pylist()]
                writer.write_1m(data)
                count += len(data)
            n += count
            progress.append({"part": p.name, "sha256": digest, "rows": count})
            emit(
                "load-progress.json",
                {"rows": n, "elapsed_s": time.monotonic() - started, "parts": progress},
            )
            print(p.name, count, n, round(time.monotonic() - started, 1), flush=True)
    finally:
        gw.close()
    with connect(DSN, budget_class="admin", autocommit=True) as c:
        c.execute("ANALYZE")
        actual = c.execute(
            (
                "SELECT count(*),count(DISTINCT (market_id,symbol)) FROM market_data.ca"
                "nonical_candles_1m WHERE ts_open < '2026-10-12'"
            )
        ).fetchone()
    assert actual == (15130115, 102), actual
    emit(
        "load.json",
        {
            "rows": n,
            "actual": actual,
            "elapsed_s": time.monotonic() - started,
            "parts": progress,
            "production_writer": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "load"])
    args = parser.parse_args()
    globals()[args.action]()
