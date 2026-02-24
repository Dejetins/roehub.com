from __future__ import annotations

from pathlib import Path
from typing import Any

import apps.migrations.bootstrap_main as bootstrap_main
from apps.migrations.bootstrap import normalize_psycopg_dsn


def test_normalize_psycopg_dsn_accepts_conninfo_with_special_password() -> None:
    """
    Verify bootstrap DSN normalizer accepts libpq conninfo with raw special characters.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        psycopg conninfo parser is authoritative syntax validator.
    Raises:
        AssertionError: If conninfo DSN is rejected or modified unexpectedly.
    Side Effects:
        None.
    """
    conninfo_dsn = (
        "host=postgres port=5432 dbname=roehub user=roehub password=M@ngala:100%"
    )

    assert normalize_psycopg_dsn(dsn=conninfo_dsn) == conninfo_dsn


def test_bootstrap_main_accepts_conninfo_dsns_from_environment(monkeypatch: Any) -> None:
    """
    Verify bootstrap CLI resolves conninfo DSNs from env and forwards them unchanged.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Runtime bootstrap call can be replaced with deterministic in-memory fake.
    Raises:
        AssertionError: If bootstrap CLI rejects conninfo values from environment.
    Side Effects:
        Monkeypatches bootstrap execution to keep test fully local and deterministic.
    """
    captured: dict[str, object] = {}
    identity_dsn = "host=postgres port=5432 dbname=roehub user=roehub password=M@ngala1906915"
    postgres_dsn = "host=postgres port=5432 dbname=roehub user=roehub password=M%ngala:1"

    def _fake_run_dev_db_bootstrap(
        *,
        identity_dsn: str,
        postgres_dsn: str,
        migrations_dir: Path,
    ) -> None:
        captured["identity_dsn"] = identity_dsn
        captured["postgres_dsn"] = postgres_dsn
        captured["migrations_dir"] = migrations_dir

    monkeypatch.setenv("IDENTITY_PG_DSN", identity_dsn)
    monkeypatch.setenv("POSTGRES_DSN", postgres_dsn)
    monkeypatch.setattr(bootstrap_main, "run_dev_db_bootstrap", _fake_run_dev_db_bootstrap)

    exit_code = bootstrap_main.main([])

    assert exit_code == 0
    assert captured["identity_dsn"] == identity_dsn
    assert captured["postgres_dsn"] == postgres_dsn
    resolved_migrations_dir = captured["migrations_dir"]
    assert isinstance(resolved_migrations_dir, Path)
    assert resolved_migrations_dir.name == "postgres"
