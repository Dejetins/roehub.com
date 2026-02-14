from __future__ import annotations

from typing import Any, Mapping, Protocol, cast

import psycopg
from psycopg.rows import dict_row


class IdentityPostgresGateway(Protocol):
    """
    IdentityPostgresGateway — минимальный SQL gateway для identity Postgres adapters.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
      - migrations/postgres/0001_identity_v1.sql
      - apps/api/wiring/modules/identity.py
    """

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute SQL query and return one row as mapping.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: One row or `None`.
        Assumptions:
            Query may include `RETURNING` clause.
        Raises:
            Exception: Storage/driver exceptions from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute SQL query without row return value.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Query is side-effecting write statement.
        Raises:
            Exception: Storage/driver exceptions from implementation.
        Side Effects:
            Executes one SQL statement.
        """
        ...


class PsycopgIdentityPostgresGateway(IdentityPostgresGateway):
    """
    PsycopgIdentityPostgresGateway — psycopg3 implementation of identity SQL gateway.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
      - src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
      - migrations/postgres/0001_identity_v1.sql
    """

    def __init__(self, *, dsn: str) -> None:
        """
        Initialize gateway with DSN connection string.

        Args:
            dsn: PostgreSQL DSN.
        Returns:
            None.
        Assumptions:
            DSN points to database with identity schema migrated.
        Raises:
            ValueError: If DSN is blank.
        Side Effects:
            None.
        """
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("PsycopgIdentityPostgresGateway requires non-empty dsn")
        self._dsn = normalized_dsn

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Execute query and return first row mapped by column names.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            Mapping[str, Any] | None: Query row or `None`.
        Assumptions:
            psycopg connection context handles transaction commit/rollback.
        Raises:
            psycopg.Error: When database operation fails.
        Side Effects:
            Opens one database connection and executes one query.
        """
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
                row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Execute side-effecting SQL statement without result rows.

        Args:
            query: SQL text.
            parameters: Bind parameters mapping.
        Returns:
            None.
        Assumptions:
            Statement semantics are validated by repository layer.
        Raises:
            psycopg.Error: When database operation fails.
        Side Effects:
            Opens one database connection and executes one query.
        """
        with psycopg.connect(
            self._dsn,
            row_factory=cast(Any, dict_row),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
