"""Adoption attests current legacy compatibility, never absent PG-only execution."""

from pathlib import Path

import psycopg
import pytest

from apps.migrations.legacy_adoption import adopt, digest, schema_witness
from apps.migrations.storage import POSTGRES_LOCK_KEY, _load_postgres_phases
from tests.real_infra.market_data.test_storage_schema_postgres import db, pytestmark  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            "DELETE FROM roehub_storage_migrations WHERE version='market-data-storage-0036'",
            "actual execution",
        ),
        (
            "DELETE FROM roehub_storage_migrations WHERE version='market-data-consumers-0039'",
            "actual execution",
        ),
        ("DROP SCHEMA market_data CASCADE", "actual market schema"),
        (
            "UPDATE notification_provider_packages SET built_in=false WHERE provider_key='fake'",
            "seed postconditions",
        ),
        ("ALTER TABLE identity_users ADD COLUMN unexpected text", "schema differs"),
    ],
)
def test_adoption_refuses_missing_contract(db, mutation, reason):  # noqa: F811
    phases = _load_postgres_phases(ROOT / "migrations/postgres/manifest.json")
    with psycopg.connect(db) as connection:
        objects = schema_witness(connection)
        reference = {
            "schema": "io.roehub.legacy-schema-witness/v1",
            "phases": {k: v[0] for k, v in phases.items()},
            "objects": objects,
            "schema_sha256": digest(objects),
        }
        with connection.transaction(force_rollback=True):
            connection.execute(mutation)
            with pytest.raises(ValueError, match=reason):
                adopt(connection, reference, phases, lock=POSTGRES_LOCK_KEY)
            assert connection.execute(
                "SELECT count(*) FROM roehub_storage_migrations WHERE store='postgres-sql-adopted'"
            ).fetchone() == (0,)


def test_explicit_legacy_provenance_and_repeat(db):  # noqa: F811
    phases = _load_postgres_phases(ROOT / "migrations/postgres/manifest.json")
    with psycopg.connect(db) as connection:
        objects = schema_witness(connection)
        reference = {
            "schema": "io.roehub.legacy-schema-witness/v1",
            "phases": {k: v[0] for k, v in phases.items()},
            "objects": objects,
            "schema_sha256": digest(objects),
        }
        with connection.transaction(force_rollback=True):
            connection.execute(
                "DELETE FROM roehub_storage_migrations WHERE version='organization-0011'"
            )
            assert adopt(connection, reference, phases, lock=POSTGRES_LOCK_KEY) == [
                "organization-0011"
            ]
            assert adopt(connection, reference, phases, lock=POSTGRES_LOCK_KEY) == []
            assert connection.execute(
                "SELECT store FROM roehub_storage_migrations WHERE version='organization-0011'"
            ).fetchone() == ("postgres-sql-adopted",)
