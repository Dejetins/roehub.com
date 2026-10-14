"""Read-only PostgreSQL schema inventory for an explicitly selected disposable target.

The measured comparison harness is preserved as non-executable S07 evidence.
This replacement uses the current production schema; it makes no timing claims.
"""

from __future__ import annotations

import argparse
import json
import os

import psycopg


def inspect_schema(dsn: str) -> dict[str, object]:
    if not dsn:
        raise ValueError("ROEHUB_QA_POSTGRES_DSN is required")
    with psycopg.connect(
        dsn, options="-c default_transaction_read_only=on -c statement_timeout=3000"
    ) as conn:
        version = conn.info.server_version
        if not 160000 <= version < 170000:
            raise ValueError("PostgreSQL 16 is required")
        rows = conn.execute("""SELECT c.relname, c.relkind FROM pg_class c
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='market_data' AND c.relkind IN ('r','p')
            ORDER BY c.relname""").fetchall()
        if not rows:
            raise ValueError("migrated market_data schema is required")
        return {
            "server_version": version,
            "read_only": True,
            "tables": [{"name": n, "kind": k} for n, k in rows],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = inspect_schema(os.environ.get("ROEHUB_QA_POSTGRES_DSN", ""))
    except (psycopg.Error, ValueError):
        raise SystemExit(
            "PostgreSQL schema inspection failed; verify the explicit target and migrations"
        ) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
