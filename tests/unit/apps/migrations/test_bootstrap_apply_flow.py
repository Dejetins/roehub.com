from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import apps.migrations.bootstrap as bootstrap
from apps.migrations.bootstrap import IdentityExchangeKeysLayout


def _identity_migrations_dir() -> Path:
    """
    Resolve repository identity SQL migrations directory for bootstrap flow tests.

    Args:
        None.
    Returns:
        Path: Existing `migrations/postgres` path in repository root.
    Assumptions:
        Test file location remains under `tests/unit/apps/migrations`.
    Raises:
        AssertionError: If migrations directory is unexpectedly missing.
    Side Effects:
        Reads filesystem metadata.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
    """
    migrations_dir = Path(__file__).resolve().parents[4] / "migrations" / "postgres"
    assert migrations_dir.is_dir()
    return migrations_dir


@contextmanager
def _fake_connection_context(connection: object) -> Iterator[object]:
    """
    Yield deterministic fake connection object for monkeypatched `psycopg.connect`.

    Args:
        connection: Opaque object representing opened DB connection.
    Returns:
        Iterator[object]: Context manager iterator yielding the same object once.
    Assumptions:
        Bootstrap internals are monkeypatched to avoid direct cursor usage in these tests.
    Raises:
        None.
    Side Effects:
        None.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - tests/unit/apps/migrations/test_bootstrap_apply_flow.py
    """
    yield connection


def test_apply_identity_baseline_sql_skips_0003_and_0004_when_v2_layout_is_preexisting(
    monkeypatch: Any,
) -> None:
    """
    Verify bootstrap skips `0003` and `0004` when exchange keys table is already in v2 layout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        `0001` and `0002` remain safe baseline scripts and should still be executed.
    Raises:
        AssertionError: If bootstrap executes `0003`/`0004` despite detected v2 layout.
    Side Effects:
        Monkeypatches DB access helpers for deterministic unit execution.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
    """
    executed_scripts: list[str] = []
    inspected_layouts = iter(
        [
            IdentityExchangeKeysLayout(
                table_exists=True,
                row_count=0,
                has_api_key=False,
                has_api_key_enc=True,
                has_api_key_hash=True,
                has_api_key_last4=True,
            )
        ]
    )

    monkeypatch.setattr(
        bootstrap.psycopg,
        "connect",
        lambda *args, **kwargs: _fake_connection_context(object()),
    )
    monkeypatch.setattr(
        bootstrap,
        "inspect_identity_exchange_keys_layout",
        lambda *, connection: next(inspected_layouts),
    )
    monkeypatch.setattr(
        bootstrap,
        "_execute_sql_script",
        lambda *, connection, sql_path: executed_scripts.append(sql_path.name),
    )

    bootstrap.apply_identity_baseline_sql(
        identity_dsn="postgresql://roehub:roehub@localhost:5432/roehub",
        migrations_dir=_identity_migrations_dir(),
    )

    assert executed_scripts == [
        "0001_identity_v1.sql",
        "0002_identity_2fa_totp_v1.sql",
    ]


def test_apply_identity_baseline_sql_runs_0003_and_guarded_0004_for_v1_flow(
    monkeypatch: Any,
) -> None:
    """
    Verify bootstrap keeps v1 flow: apply `0003` then guarded `0004` decision path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Clean database after `0001` and `0002` has no `identity_exchange_keys` table yet.
    Raises:
        AssertionError: If execution order diverges from deterministic v1 bootstrap contract.
    Side Effects:
        Monkeypatches DB access helpers for deterministic unit execution.

    Docs:
      - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
    Related:
      - apps/migrations/bootstrap.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
    """
    executed_scripts: list[str] = []
    inspected_layouts = iter(
        [
            IdentityExchangeKeysLayout(
                table_exists=False,
                row_count=0,
                has_api_key=False,
                has_api_key_enc=False,
                has_api_key_hash=False,
                has_api_key_last4=False,
            ),
            IdentityExchangeKeysLayout(
                table_exists=True,
                row_count=0,
                has_api_key=True,
                has_api_key_enc=False,
                has_api_key_hash=False,
                has_api_key_last4=False,
            ),
        ]
    )

    monkeypatch.setattr(
        bootstrap.psycopg,
        "connect",
        lambda *args, **kwargs: _fake_connection_context(object()),
    )
    monkeypatch.setattr(
        bootstrap,
        "inspect_identity_exchange_keys_layout",
        lambda *, connection: next(inspected_layouts),
    )
    monkeypatch.setattr(
        bootstrap,
        "_execute_sql_script",
        lambda *, connection, sql_path: executed_scripts.append(sql_path.name),
    )

    bootstrap.apply_identity_baseline_sql(
        identity_dsn="postgresql://roehub:roehub@localhost:5432/roehub",
        migrations_dir=_identity_migrations_dir(),
    )

    assert executed_scripts == [
        "0001_identity_v1.sql",
        "0002_identity_2fa_totp_v1.sql",
        "0003_identity_exchange_keys_v1.sql",
        "0004_identity_exchange_keys_v2.sql",
    ]
