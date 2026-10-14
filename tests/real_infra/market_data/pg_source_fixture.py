"""Production market schema in a fresh local test database; never reuse an installation."""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)


@contextmanager
def market_source(dsn):
    name = "s07_market_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        assert admin.info.host in {"127.0.0.1", "localhost"}
        assert 160000 <= admin.info.server_version < 170000
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    target = make_conninfo(dsn, dbname=name, options="")
    gateway = IngestionGateway(target, workload="bulk", deadline_s=30)
    try:
        root = Path(__file__).resolve().parents[3]
        apply_postgres_migrations(
            target, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
        )
        with psycopg.connect(target) as connection:
            for month in (
                date(1970, 1, 1),
                date(2024, 1, 1),
                date(2026, 1, 1),
                date(2026, 2, 1),
                date(2026, 3, 1),
                date(2026, 4, 1),
                date(2026, 10, 1),
            ):
                connection.execute("SELECT market_data.ensure_partitions(%s,1)", (month,))
        yield gateway
    finally:
        gateway.close()
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
