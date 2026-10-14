"""Witness a clean reference, or explicitly adopt an exactly matching legacy schema.

DSNs come only from named private environment inputs. No SQL phase is falsely
recorded as executed: adoption uses postgres-sql-adopted provenance. Stop all
application owners before target adoption; the schema lock serializes migration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psycopg import sql

from apps.migrations.legacy_adoption import adopt, digest, schema_witness
from apps.migrations.storage import POSTGRES_LOCK_KEY, _load_postgres_phases
from trading.platform.postgres_budget import admin_connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["witness", "adopt"])
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dsn-env", default="ROEHUB_STORAGE_POSTGRES_DSN")
    args = parser.parse_args()
    phases = _load_postgres_phases(args.repo_root / "migrations/postgres/manifest.json")
    with admin_connect(os.environ[args.dsn_env]) as connection:
        connection.execute("SELECT set_config('statement_timeout','30s',true)")
        if args.action == "witness":
            rows = connection.execute(
                "SELECT store,version,checksum FROM roehub_storage_migrations"
            ).fetchall()
            actual = {(s, v): c for s, v, c in rows}
            if any(actual.get(("postgres-sql", v)) != value[0] for v, value in phases.items()):
                raise ValueError("reference requires actual complete applied SQL history")
            for table in [
                "identity_users",
                "strategy_strategies",
                "market_data_instrument_selections",
            ]:
                row = connection.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                ).fetchone()
                if row is None or row[0]:
                    raise ValueError("reference must be empty of user state")
            objects = schema_witness(connection)
            value = {
                "schema": "io.roehub.legacy-schema-witness/v1",
                "phases": {k: v[0] for k, v in phases.items()},
                "objects": objects,
                "schema_sha256": digest(objects),
            }
            with args.reference.open("x") as output:
                json.dump(value, output, indent=2)
            print(json.dumps({"schema_sha256": value["schema_sha256"], "objects": len(objects)}))
        else:
            other_row = connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE "
                "backend_type='client backend' AND datname=current_database() "
                "AND pid<>pg_backend_pid()"
            ).fetchone()
            if other_row is None or other_row[0]:
                raise ValueError("quiesced target required")
            added = adopt(
                connection, json.loads(args.reference.read_text()), phases, lock=POSTGRES_LOCK_KEY
            )
            print(json.dumps({"adopted_phases": added, "historical_execution_claimed": False}))


if __name__ == "__main__":
    main()
