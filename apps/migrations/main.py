from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from alembic import command

_POSTGRES_DSN_ENV = "POSTGRES_DSN"
_DEFAULT_LOCK_KEY = 56329814721


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for fail-fast Alembic migration runner.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured command parser.
    Assumptions:
        Entry point is called from repository root or any nested path.
    Raises:
        None.
    Side Effects:
        None.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - alembic.ini
      - alembic/env.py
      - .github/workflows/ci.yml
    """
    parser = argparse.ArgumentParser(prog="roehub-migrations")
    parser.add_argument(
        "--dsn",
        default="",
        help=f"Postgres DSN. Falls back to ${_POSTGRES_DSN_ENV} when omitted.",
    )
    parser.add_argument(
        "--lock-key",
        type=int,
        default=_DEFAULT_LOCK_KEY,
        help="Advisory lock key used with pg_advisory_lock during migration upgrade.",
    )
    return parser


def _resolve_dsn(*, arg_dsn: str, environ: Mapping[str, str]) -> str:
    """
    Resolve Postgres DSN from CLI argument or environment variable.

    Args:
        arg_dsn: CLI `--dsn` value.
        environ: Environment mapping.
    Returns:
        str: Non-empty normalized DSN string.
    Assumptions:
        Environment fallback key is `POSTGRES_DSN`.
    Raises:
        ValueError: If DSN is missing.
    Side Effects:
        None.
    """
    dsn = arg_dsn.strip() if arg_dsn.strip() else environ.get(_POSTGRES_DSN_ENV, "").strip()
    if not dsn:
        raise ValueError("Migration DSN is required via --dsn or POSTGRES_DSN")
    return dsn


def _build_alembic_config(*, dsn: str, repo_root: Path) -> Config:
    """
    Build Alembic configuration for `alembic upgrade head` execution.

    Args:
        dsn: Postgres DSN.
        repo_root: Repository root path.
    Returns:
        Config: Ready-to-run Alembic configuration.
    Assumptions:
        `alembic.ini` and `alembic/` live in repository root.
    Raises:
        ValueError: If alembic.ini is missing.
    Side Effects:
        None.
    """
    alembic_ini = repo_root / "alembic.ini"
    if not alembic_ini.exists():
        raise ValueError(f"Missing Alembic config file: {alembic_ini}")

    config = Config(str(alembic_ini))
    sqlalchemy_dsn = _to_sqlalchemy_psycopg_dsn(dsn=dsn)
    config.set_main_option("sqlalchemy.url", sqlalchemy_dsn)
    config.set_main_option("script_location", str(repo_root / "alembic"))
    return config


def _upgrade_head_under_lock(*, config: Config, dsn: str, lock_key: int) -> None:
    """
    Run `alembic upgrade head` while holding Postgres advisory lock.

    Args:
        config: Prepared Alembic config.
        dsn: Postgres DSN.
        lock_key: Advisory lock key.
    Returns:
        None.
    Assumptions:
        Advisory lock must be held on the same connection used by Alembic.
    Raises:
        Exception: Any DB or Alembic failure is propagated for fail-fast startup.
    Side Effects:
        Applies DB schema migrations.
    """
    sqlalchemy_dsn = _to_sqlalchemy_psycopg_dsn(dsn=dsn)
    engine = create_engine(sqlalchemy_dsn, pool_pre_ping=True)
    with engine.connect() as connection:
        _acquire_pg_advisory_lock(connection=connection, lock_key=lock_key)
        try:
            config.attributes["connection"] = connection
            print("Running: alembic upgrade head")
            command.upgrade(config, "head")
            connection.commit()
            print("Migration success")
        except Exception:  # noqa: BLE001
            connection.rollback()
            raise
        finally:
            _release_pg_advisory_lock(connection=connection, lock_key=lock_key)
            connection.commit()


def _acquire_pg_advisory_lock(*, connection: Connection, lock_key: int) -> None:
    """
    Acquire `pg_advisory_lock` for migration critical section.

    Args:
        connection: SQLAlchemy connection.
        lock_key: Advisory lock key.
    Returns:
        None.
    Assumptions:
        Lock key is deterministic across all migration runners.
    Raises:
        Exception: Underlying DB execution errors.
    Side Effects:
        Blocks current connection until lock is obtained.
    """
    print(f"Acquiring pg_advisory_lock({lock_key})")
    connection.execute(text("SELECT pg_advisory_lock(:lock_key)"), {"lock_key": lock_key})


def _release_pg_advisory_lock(*, connection: Connection, lock_key: int) -> None:
    """
    Release `pg_advisory_lock` after migration attempt.

    Args:
        connection: SQLAlchemy connection.
        lock_key: Advisory lock key.
    Returns:
        None.
    Assumptions:
        Unlock is attempted even when upgrade fails.
    Raises:
        Exception: Underlying DB execution errors.
    Side Effects:
        Releases lock for other migration runners.
    """
    print(f"Releasing pg_advisory_lock({lock_key})")
    connection.execute(text("SELECT pg_advisory_unlock(:lock_key)"), {"lock_key": lock_key})


def _to_sqlalchemy_psycopg_dsn(*, dsn: str) -> str:
    """
    Normalize DSN to SQLAlchemy psycopg driver URL form.

    Args:
        dsn: Raw Postgres DSN.
    Returns:
        str: SQLAlchemy DSN using `postgresql+psycopg` dialect.
    Assumptions:
        Input DSN uses PostgreSQL URL scheme.
    Raises:
        ValueError: If DSN is empty or unsupported.
    Side Effects:
        None.
    """
    normalized = dsn.strip()
    if not normalized:
        raise ValueError("Postgres DSN cannot be empty")
    if normalized.startswith("postgresql+psycopg://"):
        return normalized
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    if normalized.startswith("postgres://"):
        return normalized.replace("postgres://", "postgresql+psycopg://", 1)
    raise ValueError("Postgres DSN must start with postgresql:// or postgres://")


def main(argv: list[str] | None = None) -> int:
    """
    Run fail-fast migration flow with advisory lock and `alembic upgrade head`.

    Args:
        argv: Optional CLI argument list without program name.
    Returns:
        int: Zero on success, non-zero on failure.
    Assumptions:
        Caller expects startup to fail immediately when migrations fail.
    Raises:
        None.
    Side Effects:
        Reads environment, connects to Postgres, applies migrations, prints status logs.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - alembic/env.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
      - .github/workflows/ci.yml
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        dsn = _resolve_dsn(arg_dsn=args.dsn, environ=os.environ)
        repo_root = Path(__file__).resolve().parents[2]
        config = _build_alembic_config(dsn=dsn, repo_root=repo_root)
        _upgrade_head_under_lock(config=config, dsn=dsn, lock_key=args.lock_key)
    except Exception as error:  # noqa: BLE001
        print(f"Migration failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
