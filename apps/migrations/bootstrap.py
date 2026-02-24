from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from apps.migrations.main import main as run_alembic_migrations_main

_IDENTITY_BASELINE_SQL_FILES: tuple[str, ...] = (
    "0001_identity_v1.sql",
    "0002_identity_2fa_totp_v1.sql",
    "0003_identity_exchange_keys_v1.sql",
)
_IDENTITY_EXCHANGE_KEYS_V2_SQL_FILE = "0004_identity_exchange_keys_v2.sql"


@dataclass(frozen=True, slots=True)
class IdentityExchangeKeysLayout:
    """
    Snapshot of `identity_exchange_keys` physical schema used by v2 bootstrap guard.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
      - apps/migrations/bootstrap.py
    """

    table_exists: bool
    row_count: int
    has_api_key: bool
    has_api_key_enc: bool
    has_api_key_hash: bool
    has_api_key_last4: bool


@dataclass(frozen=True, slots=True)
class IdentityExchangeKeysV2Decision:
    """
    Deterministic action for `0004_identity_exchange_keys_v2.sql`.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
      - apps/migrations/bootstrap.py
      - tests/unit/apps/migrations/test_bootstrap_decisions.py
    """

    action: Literal["apply", "skip", "fail"]
    reason: str
    hint: str | None = None


def decide_identity_exchange_keys_v2_action(
    *,
    layout: IdentityExchangeKeysLayout,
) -> IdentityExchangeKeysV2Decision:
    """
    Decide whether identity exchange keys v2 migration can be applied safely.

    Args:
        layout: Introspected table state after baseline SQL (`0001..0003`) execution.
    Returns:
        IdentityExchangeKeysV2Decision: Deterministic bootstrap decision.
    Assumptions:
        Decision is made after `0003_identity_exchange_keys_v1.sql` has already run.
    Raises:
        None.
    Side Effects:
        None.
    """
    v2_columns_present = (
        layout.has_api_key_enc and layout.has_api_key_hash and layout.has_api_key_last4
    )
    if v2_columns_present:
        return IdentityExchangeKeysV2Decision(
            action="skip",
            reason="identity_exchange_keys already has v2 columns; 0004 must be skipped",
        )

    if not layout.table_exists:
        return IdentityExchangeKeysV2Decision(
            action="fail",
            reason="identity_exchange_keys table is missing after baseline identity SQL",
            hint="Verify 0003_identity_exchange_keys_v1.sql exists and executes successfully.",
        )

    if layout.has_api_key and layout.row_count == 0:
        return IdentityExchangeKeysV2Decision(
            action="apply",
            reason="identity_exchange_keys is in v1 shape and empty; 0004 is safe to apply",
        )

    if layout.has_api_key and layout.row_count > 0:
        return IdentityExchangeKeysV2Decision(
            action="fail",
            reason=(
                "0004_identity_exchange_keys_v2.sql is unsafe: v1 schema with non-empty "
                "identity_exchange_keys table"
            ),
            hint=(
                "Run explicit re-encryption migration for existing rows, then apply v2 schema."
            ),
        )

    return IdentityExchangeKeysV2Decision(
        action="fail",
        reason="identity_exchange_keys schema shape is unknown and cannot be migrated safely",
        hint="Inspect table columns and align schema with documented v1/v2 layouts.",
    )


def run_dev_db_bootstrap(
    *,
    identity_dsn: str,
    postgres_dsn: str,
    migrations_dir: Path,
    alembic_upgrade_runner: Callable[[list[str] | None], int] = run_alembic_migrations_main,
) -> None:
    """
    Execute deterministic dev bootstrap: identity SQL baseline and Alembic `upgrade head`.

    Args:
        identity_dsn: DSN for identity baseline SQL migrations.
        postgres_dsn: DSN for Alembic migrations (`apps.migrations.main`).
        migrations_dir: Filesystem directory with identity SQL migrations.
        alembic_upgrade_runner: Callable compatible with `apps.migrations.main.main`.
    Returns:
        None.
    Assumptions:
        Identity SQL files `0001..0004` are present in `migrations_dir`.
    Raises:
        ValueError: If DSN or migrations directory values are invalid.
        RuntimeError: If identity bootstrap or Alembic upgrade fails.
    Side Effects:
        Connects to Postgres, executes SQL migrations, and mutates schema state.
    """
    apply_identity_baseline_sql(identity_dsn=identity_dsn, migrations_dir=migrations_dir)
    run_alembic_upgrade_head(
        postgres_dsn=postgres_dsn,
        alembic_upgrade_runner=alembic_upgrade_runner,
    )


def apply_identity_baseline_sql(*, identity_dsn: str, migrations_dir: Path) -> None:
    """
    Apply identity baseline SQL and guarded `0004` migration in deterministic order.

    Args:
        identity_dsn: DSN for identity Postgres schema.
        migrations_dir: Directory containing identity SQL files.
    Returns:
        None.
    Assumptions:
        `0001..0003` files are idempotent and can be re-executed.
    Raises:
        ValueError: If DSN is unsupported or migration files are missing.
        RuntimeError: If guarded decision for `0004` returns fail.
        psycopg.Error: If DB execution fails.
    Side Effects:
        Executes SQL scripts against identity Postgres schema.
    """
    normalized_identity_dsn = normalize_psycopg_dsn(dsn=identity_dsn)
    baseline_paths = _collect_sql_paths(
        migrations_dir=migrations_dir,
        filenames=_IDENTITY_BASELINE_SQL_FILES,
    )
    v2_path = _collect_sql_paths(
        migrations_dir=migrations_dir,
        filenames=(_IDENTITY_EXCHANGE_KEYS_V2_SQL_FILE,),
    )[0]

    with psycopg.connect(
        normalized_identity_dsn,
        autocommit=True,
        row_factory=cast(Any, dict_row),
    ) as connection:
        for sql_path in baseline_paths:
            print(f"Applying identity baseline SQL: {sql_path.name}")
            _execute_sql_script(connection=connection, sql_path=sql_path)

        layout = inspect_identity_exchange_keys_layout(connection=connection)
        decision = decide_identity_exchange_keys_v2_action(layout=layout)
        print(f"identity 0004 decision: {decision.action} ({decision.reason})")
        if decision.action == "apply":
            _execute_sql_script(connection=connection, sql_path=v2_path)
            print(f"Applied identity SQL: {v2_path.name}")
            return
        if decision.action == "skip":
            print(f"Skipped identity SQL: {v2_path.name}")
            return

        detail_suffix = f" Hint: {decision.hint}" if decision.hint else ""
        raise RuntimeError(f"{decision.reason}.{detail_suffix}")


def run_alembic_upgrade_head(
    *,
    postgres_dsn: str,
    alembic_upgrade_runner: Callable[[list[str] | None], int],
) -> None:
    """
    Run existing Alembic migration runner and fail fast on non-zero exit code.

    Args:
        postgres_dsn: DSN used by `apps.migrations.main`.
        alembic_upgrade_runner: Callable that executes Alembic `upgrade head`.
    Returns:
        None.
    Assumptions:
        Runner prints its own migration logs and returns process-like exit code.
    Raises:
        RuntimeError: If Alembic runner returns non-zero code.
    Side Effects:
        Executes Alembic schema upgrades for `POSTGRES_DSN`.
    """
    print("Running Alembic upgrade head for POSTGRES_DSN")
    exit_code = alembic_upgrade_runner(["--dsn", postgres_dsn])
    if exit_code != 0:
        raise RuntimeError(f"Alembic migration runner failed with exit code {exit_code}")


def inspect_identity_exchange_keys_layout(
    *,
    connection: psycopg.Connection[Any],
) -> IdentityExchangeKeysLayout:
    """
    Read `identity_exchange_keys` schema shape and row count for `0004` guard decision.

    Args:
        connection: Open psycopg connection with dict row factory.
    Returns:
        IdentityExchangeKeysLayout: Deterministic schema snapshot.
    Assumptions:
        Connection points to identity database where `0001..0003` were applied.
    Raises:
        RuntimeError: If mandatory introspection query returns no rows.
        psycopg.Error: If introspection SQL fails.
    Side Effects:
        Executes metadata and aggregate read queries.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            cast(
                Any,
                (
                    "SELECT EXISTS ("
                    " SELECT 1"
                    " FROM information_schema.tables"
                    " WHERE table_schema = 'public'"
                    "   AND table_name = 'identity_exchange_keys'"
                    ") AS table_exists"
                ),
            ),
            {},
        )
        table_state_row = cursor.fetchone()
    if table_state_row is None:
        raise RuntimeError("Failed to introspect identity_exchange_keys table presence")

    table_exists = bool(table_state_row["table_exists"])
    if not table_exists:
        return IdentityExchangeKeysLayout(
            table_exists=False,
            row_count=0,
            has_api_key=False,
            has_api_key_enc=False,
            has_api_key_hash=False,
            has_api_key_last4=False,
        )

    with connection.cursor() as cursor:
        cursor.execute(
            cast(
                Any,
                (
                    "SELECT column_name"
                    " FROM information_schema.columns"
                    " WHERE table_schema = 'public'"
                    "   AND table_name = %(table_name)s"
                    " ORDER BY column_name ASC"
                ),
            ),
            {"table_name": "identity_exchange_keys"},
        )
        column_rows = cursor.fetchall()
        cursor.execute(
            cast(Any, "SELECT COUNT(*)::bigint AS total FROM identity_exchange_keys"),
            {},
        )
        row_count_row = cursor.fetchone()

    if row_count_row is None:
        raise RuntimeError("Failed to introspect identity_exchange_keys row count")

    columns = tuple(str(row["column_name"]) for row in column_rows)
    return IdentityExchangeKeysLayout(
        table_exists=True,
        row_count=int(row_count_row["total"]),
        has_api_key="api_key" in columns,
        has_api_key_enc="api_key_enc" in columns,
        has_api_key_hash="api_key_hash" in columns,
        has_api_key_last4="api_key_last4" in columns,
    )


def normalize_psycopg_dsn(*, dsn: str) -> str:
    """
    Normalize DSN for direct psycopg connections from URL or conninfo input.

    Args:
        dsn: Raw DSN string.
    Returns:
        str: DSN accepted by psycopg3 (`postgresql://`, `postgres://`, or conninfo).
    Assumptions:
        Input DSN may use SQLAlchemy `postgresql+psycopg://` scheme or conninfo format.
    Raises:
        ValueError: If DSN is empty or has unsupported format.
    Side Effects:
        None.

    Docs:
      - docs/runbooks/web-ui-gateway-same-origin.md
    Related:
      - apps/migrations/main.py
      - infra/docker/docker-compose.yml
    """
    normalized = dsn.strip()
    if not normalized:
        raise ValueError("Postgres DSN cannot be empty")
    if normalized.startswith("postgresql+psycopg://"):
        return normalized.replace("postgresql+psycopg://", "postgresql://", 1)
    if normalized.startswith("postgresql://"):
        return normalized
    if normalized.startswith("postgres://"):
        return normalized
    try:
        conninfo_to_dict(normalized)
    except Exception as error:  # noqa: BLE001
        raise ValueError("Postgres DSN must be URL or libpq conninfo format") from error
    return normalized


def _collect_sql_paths(*, migrations_dir: Path, filenames: tuple[str, ...]) -> tuple[Path, ...]:
    """
    Resolve SQL filenames to concrete existing paths under migrations directory.

    Args:
        migrations_dir: Directory that stores identity SQL migrations.
        filenames: Ordered SQL filenames that must exist.
    Returns:
        tuple[Path, ...]: Existing file paths in the same order as `filenames`.
    Assumptions:
        Callers pass deterministic filename order.
    Raises:
        ValueError: If migrations directory or file path is missing.
    Side Effects:
        Reads filesystem metadata.
    """
    if not migrations_dir.exists():
        raise ValueError(f"Identity migrations directory does not exist: {migrations_dir}")

    resolved_paths: list[Path] = []
    for filename in filenames:
        sql_path = migrations_dir / filename
        if not sql_path.is_file():
            raise ValueError(f"Missing identity SQL migration file: {sql_path}")
        resolved_paths.append(sql_path)
    return tuple(resolved_paths)


def _execute_sql_script(
    *,
    connection: psycopg.Connection[Any],
    sql_path: Path,
) -> None:
    """
    Execute raw SQL script file via psycopg connection.

    Args:
        connection: Open psycopg connection with autocommit enabled.
        sql_path: Path to SQL script file.
    Returns:
        None.
    Assumptions:
        SQL script manages its own transaction statements if needed.
    Raises:
        OSError: If file cannot be read.
        psycopg.Error: If SQL execution fails.
    Side Effects:
        Executes SQL statements against current database connection.
    """
    sql_text = sql_path.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(cast(Any, sql_text), prepare=False)
