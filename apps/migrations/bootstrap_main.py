from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

from apps.migrations.bootstrap import run_dev_db_bootstrap

_IDENTITY_DSN_ENV = "IDENTITY_PG_DSN"
_POSTGRES_DSN_ENV = "POSTGRES_DSN"


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for dev DB bootstrap entrypoint.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured command-line parser.
    Assumptions:
        Identity and Alembic DSNs can be provided by CLI flags or environment.
    Raises:
        None.
    Side Effects:
        None.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - apps/migrations/main.py
      - infra/docker/docker-compose.yml
    """
    parser = argparse.ArgumentParser(prog="roehub-dev-db-bootstrap")
    parser.add_argument(
        "--identity-dsn",
        default="",
        help=f"Identity Postgres DSN. Falls back to ${_IDENTITY_DSN_ENV}.",
    )
    parser.add_argument(
        "--postgres-dsn",
        default="",
        help=f"Alembic Postgres DSN. Falls back to ${_POSTGRES_DSN_ENV}.",
    )
    parser.add_argument(
        "--identity-migrations-dir",
        default="",
        help="Optional path to identity SQL migrations directory. Defaults to migrations/postgres.",
    )
    return parser


def _resolve_required_dsn(
    *,
    arg_value: str,
    env_key: str,
    environ: Mapping[str, str],
) -> str:
    """
    Resolve required DSN from CLI argument with environment fallback.

    Args:
        arg_value: Raw DSN value from CLI parser.
        env_key: Environment variable key for fallback.
        environ: Environment mapping source.
    Returns:
        str: Non-empty DSN value.
    Assumptions:
        Empty CLI value means environment fallback must be used.
    Raises:
        ValueError: If resulting DSN is empty.
    Side Effects:
        Reads environment mapping.
    """
    normalized_arg = arg_value.strip()
    if normalized_arg:
        return normalized_arg
    normalized_env = environ.get(env_key, "").strip()
    if normalized_env:
        return normalized_env
    raise ValueError(
        "Missing required DSN. "
        f"Provide --{env_key.lower().replace('_', '-')} or {env_key}"
    )


def _resolve_identity_migrations_dir(*, arg_value: str, repo_root: Path) -> Path:
    """
    Resolve identity SQL migrations directory path.

    Args:
        arg_value: Optional CLI path override.
        repo_root: Repository root path for default resolution.
    Returns:
        Path: Existing migrations directory path.
    Assumptions:
        Default migrations directory is `<repo_root>/migrations/postgres`.
    Raises:
        ValueError: If resolved directory does not exist.
    Side Effects:
        Reads filesystem metadata.
    """
    if arg_value.strip():
        migrations_dir = Path(arg_value).expanduser().resolve()
    else:
        migrations_dir = (repo_root / "migrations" / "postgres").resolve()

    if not migrations_dir.is_dir():
        raise ValueError(f"Identity migrations directory does not exist: {migrations_dir}")
    return migrations_dir


def main(argv: list[str] | None = None) -> int:
    """
    Run dev bootstrap flow: identity SQL baseline then Alembic `upgrade head`.

    Args:
        argv: Optional CLI arguments without program name.
    Returns:
        int: Zero on success, non-zero on failure.
    Assumptions:
        Caller expects fail-fast startup on any bootstrap error.
    Raises:
        None.
    Side Effects:
        Reads environment, connects to Postgres, and mutates DB schema state.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
      - docs/architecture/roadmap/milestone-6-epics-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - apps/migrations/main.py
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        repo_root = Path(__file__).resolve().parents[2]
        identity_dsn = _resolve_required_dsn(
            arg_value=args.identity_dsn,
            env_key=_IDENTITY_DSN_ENV,
            environ=os.environ,
        )
        postgres_dsn = _resolve_required_dsn(
            arg_value=args.postgres_dsn,
            env_key=_POSTGRES_DSN_ENV,
            environ=os.environ,
        )
        migrations_dir = _resolve_identity_migrations_dir(
            arg_value=args.identity_migrations_dir,
            repo_root=repo_root,
        )
        run_dev_db_bootstrap(
            identity_dsn=identity_dsn,
            postgres_dsn=postgres_dsn,
            migrations_dir=migrations_dir,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Bootstrap failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
