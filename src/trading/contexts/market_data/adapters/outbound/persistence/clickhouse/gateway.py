from __future__ import annotations

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
