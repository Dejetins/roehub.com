from __future__ import annotations

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

from alembic import context

config = context.config

target_metadata = None


def run_migrations_offline() -> None:
    """
    Run Alembic migrations in offline mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        SQL URL is provided via `alembic.ini` or runtime override.
    Raises:
        Exception: Alembic configuration/runtime errors.
    Side Effects:
        Emits SQL statements without opening DB connection.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - alembic/versions/20260215_0001_strategy_storage_v1.py
      - apps/migrations/main.py
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run Alembic migrations in online mode using injected or constructed SQLAlchemy connection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Optional injected connection comes from `apps.migrations.main` advisory-lock flow.
    Raises:
        Exception: Alembic configuration/runtime errors.
    Side Effects:
        Opens DB connection (when not injected) and applies schema changes.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - apps/migrations/main.py
      - alembic.ini
    """
    injected_connection = config.attributes.get("connection")
    if isinstance(injected_connection, Connection):
        context.configure(
            connection=injected_connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
