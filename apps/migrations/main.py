from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.engine.url import make_url

from alembic import command
from alembic.config import Config

_POSTGRES_DSN_ENV = "POSTGRES_DSN"
_DEFAULT_LOCK_KEY = 56329814721
_POSTGRES_URL_PREFIXES: tuple[str, ...] = (
    "postgresql+psycopg://",
    "postgresql://",
    "postgres://",
)


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


def _build_alembic_config(*, repo_root: Path) -> Config:
    """
    Build Alembic configuration for `alembic upgrade head` execution.

    Args:
        repo_root: Repository root path.
    Returns:
        Config: Ready-to-run Alembic configuration.
    Assumptions:
        `alembic.ini` and `alembic/` live in repository root.
    Raises:
        ValueError: If alembic.ini is missing.
    Side Effects:
        None.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - alembic.ini
      - alembic/env.py
      - apps/migrations/main.py
    """
    alembic_ini = repo_root / "alembic.ini"
    if not alembic_ini.exists():
        raise ValueError(f"Missing Alembic config file: {alembic_ini}")

    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    return config


def _upgrade_head_under_lock(*, config: Config, sqlalchemy_url: URL, lock_key: int) -> None:
    """
    Run `alembic upgrade head` while holding Postgres advisory lock.

    Args:
        config: Prepared Alembic config.
        sqlalchemy_url: SQLAlchemy URL built from URL DSN or libpq conninfo.
        lock_key: Advisory lock key.
    Returns:
        None.
    Assumptions:
        Advisory lock must be held on the same connection used by Alembic.
    Raises:
        Exception: Any DB or Alembic failure is propagated for fail-fast startup.
    Side Effects:
        Applies DB schema migrations.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - alembic/env.py
      - apps/migrations/main.py
    """
    engine = create_engine(sqlalchemy_url, pool_pre_ping=True)
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


def _to_sqlalchemy_psycopg_url(*, dsn: str) -> URL:
    """
    Normalize DSN to SQLAlchemy psycopg URL from URL DSN or libpq conninfo.

    Args:
        dsn: Raw Postgres DSN.
    Returns:
        URL: SQLAlchemy URL using `postgresql+psycopg` dialect.
    Assumptions:
        DSN can be PostgreSQL URL or libpq conninfo keyword-value string.
    Raises:
        ValueError: If DSN is empty or unsupported.
    Side Effects:
        None.

    Docs:
      - docs/runbooks/web-ui-gateway-same-origin.md
    Related:
      - apps/migrations/bootstrap.py
      - infra/docker/docker-compose.yml
    """
    normalized = dsn.strip()
    if not normalized:
        raise ValueError("Postgres DSN cannot be empty")
    if _looks_like_postgres_url_dsn(dsn=normalized):
        return _to_sqlalchemy_url_from_postgres_url_dsn(url_dsn=normalized)
    return _to_sqlalchemy_url_from_conninfo_dsn(conninfo_dsn=normalized)


def _looks_like_postgres_url_dsn(*, dsn: str) -> bool:
    """
    Check whether DSN uses PostgreSQL URL scheme supported by migration runner.

    Args:
        dsn: Raw DSN string without trimming.
    Returns:
        bool: True when DSN starts with supported PostgreSQL URL prefix.
    Assumptions:
        URL detection is prefix-based and deterministic.
    Raises:
        None.
    Side Effects:
        None.

    Docs:
      - docs/runbooks/web-ui-gateway-same-origin.md
    Related:
      - apps/migrations/main.py
      - apps/migrations/bootstrap.py
    """
    return dsn.startswith(_POSTGRES_URL_PREFIXES)


def _to_sqlalchemy_url_from_postgres_url_dsn(*, url_dsn: str) -> URL:
    """
    Convert PostgreSQL URL DSN to SQLAlchemy URL with psycopg driver.

    Args:
        url_dsn: PostgreSQL URL DSN (`postgresql://`, `postgres://`, or `postgresql+psycopg://`).
    Returns:
        URL: SQLAlchemy URL with `postgresql+psycopg` drivername.
    Assumptions:
        URL parsing follows SQLAlchemy deterministic URL semantics.
    Raises:
        ValueError: If URL uses unsupported database driver.
    Side Effects:
        None.

    Docs:
      - docs/runbooks/web-ui-gateway-same-origin.md
    Related:
      - apps/migrations/main.py
      - alembic/env.py
    """
    parsed_url = make_url(url_dsn)
    if parsed_url.drivername not in {"postgresql", "postgres", "postgresql+psycopg"}:
        raise ValueError("Postgres URL DSN must use postgresql:// or postgres:// scheme")
    return parsed_url.set(drivername="postgresql+psycopg")


def _to_sqlalchemy_url_from_conninfo_dsn(*, conninfo_dsn: str) -> URL:
    """
    Convert libpq conninfo DSN to SQLAlchemy URL with psycopg driver.

    Args:
        conninfo_dsn: libpq DSN in keyword-value format (`host=... user=... password=...`).
    Returns:
        URL: SQLAlchemy URL with parsed auth/host/database/query components.
    Assumptions:
        `psycopg.conninfo.conninfo_to_dict` validates conninfo syntax.
    Raises:
        ValueError: If conninfo DSN is invalid or uses unsupported numeric fields.
    Side Effects:
        None.

    Docs:
      - docs/runbooks/web-ui-gateway-same-origin.md
    Related:
      - apps/migrations/bootstrap.py
      - infra/docker/docker-compose.yml
    """
    try:
        conninfo_fields = conninfo_to_dict(conninfo_dsn)
    except Exception as error:  # noqa: BLE001
        raise ValueError("Postgres DSN must be URL or libpq conninfo format") from error

    raw_port = str(conninfo_fields.pop("port", "")).strip()
    resolved_port: int | None = None
    if raw_port:
        try:
            resolved_port = int(raw_port)
        except ValueError as error:
            raise ValueError("Conninfo port must be numeric when provided") from error

    resolved_database = str(conninfo_fields.pop("dbname", "")).strip() or None
    query = {
        key: str(value)
        for key, value in sorted(conninfo_fields.items())
        if key not in {"dbname", "host", "hostaddr", "password", "port", "user"} and str(value)
    }
    resolved_host = str(
        conninfo_fields.get("host", conninfo_fields.get("hostaddr", ""))
    ).strip() or None
    resolved_user = str(conninfo_fields.get("user", "")).strip() or None
    resolved_password = str(conninfo_fields.get("password", "")).strip() or None

    return URL.create(
        "postgresql+psycopg",
        username=resolved_user,
        password=resolved_password,
        host=resolved_host,
        port=resolved_port,
        database=resolved_database,
        query=query,
    )


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
        sqlalchemy_url = _to_sqlalchemy_psycopg_url(dsn=dsn)
        repo_root = Path(__file__).resolve().parents[2]
        config = _build_alembic_config(repo_root=repo_root)
        _upgrade_head_under_lock(
            config=config,
            sqlalchemy_url=sqlalchemy_url,
            lock_key=args.lock_key,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Migration failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
