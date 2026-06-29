from __future__ import annotations

from typing import Any, Mapping, Protocol, cast

import psycopg
from psycopg.rows import dict_row


class NotificationPostgresGateway(Protocol):
    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Execute one SQL statement and return one mapped row."""
        ...

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """Execute one SQL statement and return all mapped rows."""
        ...

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """Execute one SQL statement without returning rows."""
        ...


class PsycopgNotificationPostgresGateway(NotificationPostgresGateway):
    def __init__(self, *, dsn: str) -> None:
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("PsycopgNotificationPostgresGateway requires non-empty dsn")
        self._dsn = normalized_dsn

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
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
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
                rows = cursor.fetchall()
        return tuple(dict(row) for row in rows)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, query), parameters)
