from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Mapping, Protocol, cast

import psycopg
from psycopg.rows import dict_row


class BacktestPostgresGateway(Protocol):
    """
    BacktestPostgresGateway — minimal SQL gateway for Backtest jobs Postgres adapters.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_lease_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    """

    def transaction(self) -> AbstractContextManager[BacktestPostgresGateway]:
        """Bind all operations in this execution context to one transaction/connection."""
        ...

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute SQL statement and return one mapped row.

        Args:
            query: SQL text.
            parameters: SQL bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: One row or `None`.
        Assumptions:
            Query may include `RETURNING` clause.
        Raises:
            Exception: Driver/storage errors from implementation.
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
            parameters: SQL bind parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Query rows in SQL-defined order.
        Assumptions:
            Deterministic ordering is controlled by explicit SQL `ORDER BY` clauses.
        Raises:
            Exception: Driver/storage errors from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute side-effecting SQL statement without returning rows.

        Args:
            query: SQL text.
            parameters: SQL bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Statement semantics are validated by repository layer.
        Raises:
            Exception: Driver/storage errors from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...


class PsycopgBacktestPostgresGateway(BacktestPostgresGateway):
    """
    PsycopgBacktestPostgresGateway — psycopg3 implementation for Backtest SQL adapters.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/milestone-5-epics-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/gateway.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - alembic/versions/20260222_0003_backtest_jobs_v1.py
    """

    def __init__(self, *, dsn: str) -> None:
        """
        Initialize gateway with non-empty PostgreSQL DSN.

        Args:
            dsn: PostgreSQL DSN.
        Returns:
            None.
        Assumptions:
            DSN points to migrated Backtest jobs schema.
        Raises:
            ValueError: If DSN is blank.
        Side Effects:
            None.
        """
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("PsycopgBacktestPostgresGateway requires non-empty dsn")
        self._dsn = normalized_dsn
        self._connection: ContextVar[tuple[object, object, psycopg.Connection[Any]] | None] = (
            ContextVar(f"backtest_transaction_{id(self)}", default=None)
        )

    @staticmethod
    def _execution_owner() -> tuple[object, object]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.current_thread(), task

    def _owned_connection(self) -> psycopg.Connection[Any] | None:
        binding = self._connection.get()
        if binding is None or binding[:2] != self._execution_owner():
            return None
        return binding[2]

    @contextmanager
    def transaction(self) -> Iterator[BacktestPostgresGateway]:
        """Commit together; nested callers use savepoints, threads have separate owners."""
        existing = self._owned_connection()
        if existing is not None:
            with existing.transaction():
                yield self
            return
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            token = self._connection.set((*self._execution_owner(), connection))
            try:
                with connection.transaction():
                    yield self
            finally:
                self._connection.reset(token)

    @contextmanager
    def _session(self) -> Iterator[psycopg.Connection[Any]]:
        existing = self._owned_connection()
        if existing is not None:
            yield existing
        else:
            with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
                yield connection

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute query and return one row mapped by column names.

        Args:
            query: SQL text.
            parameters: SQL bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: One row or `None`.
        Assumptions:
            psycopg connection context manager handles commit/rollback.
        Raises:
            psycopg.Error: On database operation failure.
        Side Effects:
            Opens one database connection and executes one query.
        """
        with self._session() as connection:
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
            parameters: SQL bind parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Query rows.
        Assumptions:
            Row order is controlled by SQL `ORDER BY` clauses.
        Raises:
            psycopg.Error: On database operation failure.
        Side Effects:
            Opens one database connection and executes one query.
        """
        with self._session() as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
                rows = cursor.fetchall()
        return tuple(dict(row) for row in rows)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute write statement without returning rows.

        Args:
            query: SQL text.
            parameters: SQL bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Query semantics are validated by repository methods.
        Raises:
            psycopg.Error: On database operation failure.
        Side Effects:
            Opens one database connection and executes one query.
        """
        with self._session() as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)


__all__ = ["BacktestPostgresGateway", "PsycopgBacktestPostgresGateway"]
