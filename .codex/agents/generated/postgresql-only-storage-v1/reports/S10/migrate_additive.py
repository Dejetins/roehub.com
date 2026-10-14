"""Exact legacy-local additive phases, without fabricating missing old phase markers."""

import json
import os
from pathlib import Path

from apps.migrations.market_data import market_data_schema_ready, provision_current_partitions
from apps.migrations.storage import (
    POSTGRES_LOCK_KEY,
    _load_postgres_phases,
    _read_postgres_markers,
    provision_runtime_roles,
)
from trading.platform.postgres_budget import connect

root = Path("/work")
dsn = os.environ["ROEHUB_STORAGE_POSTGRES_DSN"]
phases = _load_postgres_phases(root / "migrations/postgres/manifest.json")
before = _read_postgres_markers(dsn)
for (store, version), checksum in before.items():
    if store == "postgres-sql" and version in phases:
        assert phases[version][0] == checksum, version
selected = [
    "market-data-storage-0036",
    "market-data-reference-0037",
    "market-data-verification-0038",
    "market-data-consumers-0039",
]
with connect(dsn, budget_class="admin") as db:
    assert db.execute(
        "SELECT to_regclass('public.market_data_stream_events') IS NOT NULL AND "
        "to_regclass('public.backtest_artifact_source_state') IS NOT NULL"
    ).fetchone()[0]
    assert db.execute("SELECT count(*) FROM market_data_instrument_selections").fetchone()[0] == 5
    for version in selected:
        checksum, paths = phases[version]
        db.execute("SELECT pg_advisory_xact_lock(%s)", (POSTGRES_LOCK_KEY,))
        prior = db.execute(
            "SELECT checksum FROM roehub_storage_migrations WHERE store=%s AND version=%s",
            ("postgres-sql", version),
        ).fetchone()
        if prior:
            assert prior[0] == checksum
            continue
        for path in paths:
            db.execute(path.read_text(), prepare=False)
        db.execute(
            "INSERT INTO roehub_storage_migrations(store,version,checksum) VALUES (%s,%s,%s)",
            ("postgres-sql", version, checksum),
        )
provision_runtime_roles(dsn)
with connect(dsn, budget_class="admin") as db:
    provision_current_partitions(db)
    assert market_data_schema_ready(db)
after = _read_postgres_markers(dsn)
assert all(after[k] == v for k, v in before.items())
print(
    json.dumps(
        {
            "applied": selected,
            "prior_markers_preserved": True,
            "schema_ready": True,
            "legacy_missing_sql_markers": [v for v in phases if ("postgres-sql", v) not in after],
        }
    )
)
