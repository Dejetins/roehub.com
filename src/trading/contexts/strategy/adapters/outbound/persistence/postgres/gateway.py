from __future__ import annotations

from typing import Any, Mapping, Protocol, cast

import psycopg
from psycopg.rows import dict_row


class StrategyPostgresGateway(Protocol):
    """
    StrategyPostgresGateway — minimal SQL gateway for Strategy v1 Postgres adapters.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_event_repository.py
    """

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute SQL statement and return one mapped row.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: One row or `None`.
        Assumptions:
            Query may contain `RETURNING` clause.
        Raises:
            Exception: Storage/driver errors from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Execute SQL statement and return all mapped rows.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Query rows in SQL-defined order.
        Assumptions:
            Deterministic ordering is controlled by explicit `ORDER BY` in SQL.
        Raises:
            Exception: Storage/driver errors from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute side-effecting SQL statement without returning rows.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Statement has write side effects.
        Raises:
            Exception: Storage/driver errors from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...


class PsycopgStrategyPostgresGateway(StrategyPostgresGateway):
    """
    PsycopgStrategyPostgresGateway — psycopg3 implementation for Strategy v1 SQL gateway.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/gateway.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def __init__(self, *, dsn: str) -> None:
        """
        Initialize gateway with non-empty PostgreSQL DSN.

        Args:
            dsn: PostgreSQL DSN.
        Returns:
            None.
        Assumptions:
            DSN points to Postgres instance with migrated strategy schema.
        Raises:
            ValueError: If DSN is blank.
        Side Effects:
            None.
        """
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("PsycopgStrategyPostgresGateway requires non-empty dsn")
        self._dsn = normalized_dsn

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute query and return one row mapped by column names.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: Query row or `None`.
        Assumptions:
            psycopg context manager handles transaction commit/rollback.
        Raises:
            psycopg.Error: When database operation fails.
        Side Effects:
            Opens connection and executes one query.
        """
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
                row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Execute query and return all rows mapped by column names.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Query rows.
        Assumptions:
            Row order is controlled by query `ORDER BY` clause.
        Raises:
            psycopg.Error: When database operation fails.
        Side Effects:
            Opens connection and executes one query.
        """
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
                rows = cursor.fetchall()
        return tuple(dict(row) for row in rows)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute side-effecting query without row return value.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Query semantics are validated at repository layer.
        Raises:
            psycopg.Error: When database operation fails.
        Side Effects:
            Opens connection and executes one query.
        """
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
