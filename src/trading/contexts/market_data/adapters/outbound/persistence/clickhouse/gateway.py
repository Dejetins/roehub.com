from __future__ import annotations

import threading
from typing import Any, Mapping, Protocol, Sequence


class ClickHouseGateway(Protocol):
    """
    ClickHouseGateway — тонкий шлюз над конкретным драйвером ClickHouse.

    Цель:
    - изолировать адаптеры от конкретной библиотеки
    - unit-тестировать SQL и payload без реальной БД
    """

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        ...

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        ...


class ClickHouseConnectGateway:
    """
    Опциональный gateway для библиотеки clickhouse-connect.

    НЕ является обязательным для unit-тестов.
    Используется на wiring-слое, если clickhouse-connect установлен.

    Ожидания к client:
    - client.insert(table, data, column_names=...)
    - client.query(query, parameters=...) -> имеет .column_names и .result_rows
    """

    def __init__(self, client: Any) -> None:
        if client is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseConnectGateway requires client")
        self._client = client

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return

        col_names = list(rows[0].keys())
        data = [tuple(r.get(c) for c in col_names) for r in rows]
        self._client.insert(table, data, column_names=col_names)

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        res = self._client.query(query, parameters=parameters)
        cols = list(getattr(res, "column_names"))
        out: list[dict[str, Any]] = []
        for tup in getattr(res, "result_rows"):
            out.append({cols[i]: tup[i] for i in range(len(cols))})
        return out


class ThreadLocalClickHouseConnectGateway:
    """
    ClickHouse gateway with one clickhouse-connect client per thread.

    Purpose:
    - avoid session-level concurrent query errors in multi-threaded runtimes
      (for example asyncio.to_thread worker pools).

    Expectations for each created client:
    - client.insert(table, data, column_names=...)
    - client.query(query, parameters=...) -> has .column_names and .result_rows
    """

    def __init__(self, client_factory) -> None:
        """
        Initialize gateway with a client factory used per-thread.

        Parameters:
        - client_factory: callable returning a ready clickhouse-connect client instance.

        Returns:
        - None.

        Assumptions/Invariants:
        - factory is callable and returns non-null client objects.

        Errors/Exceptions:
        - Raises `ValueError` when factory is missing or non-callable.

        Side effects:
        - None.
        """
        if client_factory is None:  # type: ignore[truthy-bool]
            raise ValueError("ThreadLocalClickHouseConnectGateway requires client_factory")
        if not callable(client_factory):
            raise ValueError("ThreadLocalClickHouseConnectGateway client_factory must be callable")
        self._factory = client_factory
        self._thread_local = threading.local()

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """
        Insert rows into ClickHouse using thread-local client.

        Parameters:
        - table: fully-qualified table name.
        - rows: row payload sequence.

        Returns:
        - None.

        Assumptions/Invariants:
        - payload rows share the same key order.

        Errors/Exceptions:
        - Propagates underlying client serialization/transport errors.

        Side effects:
        - Executes one ClickHouse INSERT request.
        """
        if not rows:
            return
        col_names = list(rows[0].keys())
        data = [tuple(r.get(c) for c in col_names) for r in rows]
        self._client().insert(table, data, column_names=col_names)

    def select(self, query: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        """
        Execute SELECT query through thread-local client and map rows to dicts.

        Parameters:
        - query: SQL text.
        - parameters: bind parameters mapping.

        Returns:
        - Sequence of dictionary rows.

        Assumptions/Invariants:
        - client result object provides `column_names` and `result_rows`.

        Errors/Exceptions:
        - Propagates underlying client query errors.

        Side effects:
        - Executes one ClickHouse query request.
        """
        res = self._client().query(query, parameters=parameters)
        cols = list(getattr(res, "column_names"))
        out: list[dict[str, Any]] = []
        for tup in getattr(res, "result_rows"):
            out.append({cols[i]: tup[i] for i in range(len(cols))})
        return out

    def _client(self) -> Any:
        """
        Return thread-local client, creating it lazily on first access.

        Parameters:
        - None.

        Returns:
        - Client object from factory tied to current thread.

        Assumptions/Invariants:
        - Thread-local storage is isolated per OS thread.

        Errors/Exceptions:
        - Raises `RuntimeError` when factory returns `None`.

        Side effects:
        - May create and cache a new client for current thread.
        """
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client
        created = self._factory()
        if created is None:  # type: ignore[truthy-bool]
            raise RuntimeError("ThreadLocalClickHouseConnectGateway factory returned None client")
        self._thread_local.client = created
        return created
