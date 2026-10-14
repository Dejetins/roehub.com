"""Rollback-only negative adoption proofs in the disposable reference database."""

import json
from pathlib import Path

import psycopg

from apps.migrations.legacy_adoption import adopt
from apps.migrations.storage import POSTGRES_LOCK_KEY, _load_postgres_phases

root = Path("/work")
reference = json.loads(
    (
        root
        / ".codex/agents/generated/postgresql-only-storage-v1/reports/S10/adoption-reference.json"
    ).read_text()
)
phases = _load_postgres_phases(root / "migrations/postgres/manifest.json")
cases = {
    "missing_actual_storage": "DELETE FROM roehub_storage_migrations WHERE version='market-data-storage-0036'",  # noqa: E501
    "missing_actual_consumers": "DELETE FROM roehub_storage_migrations WHERE version='market-data-consumers-0039'",  # noqa: E501
    "missing_market_schema": "DROP SCHEMA market_data CASCADE",
    "missing_seed": "UPDATE notification_provider_packages SET built_in=false WHERE provider_key='fake'",  # noqa: E501
    "schema_drift": "ALTER TABLE identity_users ADD COLUMN unreviewed text",
}
with psycopg.connect("host=roehub-s10-recovery-verify dbname=reference user=postgres") as db:
    for name, sql in cases.items():
        try:
            with db.transaction(force_rollback=True):
                db.execute(sql)
                adopt(db, reference, phases, lock=POSTGRES_LOCK_KEY)
        except ValueError as error:
            print(json.dumps({"case": name, "refused": str(error)}), flush=True)
        else:
            raise AssertionError(name)
    with db.transaction(force_rollback=True):
        db.execute("DELETE FROM roehub_storage_migrations WHERE version='organization-0011'")
        added = adopt(db, reference, phases, lock=POSTGRES_LOCK_KEY)
        assert added == ["organization-0011"]
        assert adopt(db, reference, phases, lock=POSTGRES_LOCK_KEY) == []
        print(
            json.dumps(
                {
                    "compatible_legacy_adoption": added,
                    "repeat_idempotent": True,
                    "rolled_back": True,
                }
            )
        )
