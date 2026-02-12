from __future__ import annotations

import asyncio
import threading
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ThreadLocalClickHouseConnectGateway,
)


class _FakeQueryResult:
    """Fake clickhouse query result container with column metadata and rows."""

    def __init__(self, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
        """
        Store fake query result payload.

        Parameters:
        - columns: result column names.
        - rows: tuple-based row values.
        """
        self.column_names = columns
        self.result_rows = rows


class _FakeClient:
    """Thread-observable fake clickhouse client used by gateway tests."""

    def __init__(self, client_id: int) -> None:
        """
        Initialize fake client identity and call registries.

        Parameters:
        - client_id: numeric identity assigned by factory.
        """
        self.client_id = client_id
        self.insert_calls: list[tuple[str, list[tuple[Any, ...]], list[str]]] = []
        self.query_calls: list[tuple[str, Mapping[str, Any]]] = []
        self._query_result = _FakeQueryResult(["id", "name"], [(1, "BTCUSDT")])

    def insert(self, table: str, data: list[tuple[Any, ...]], column_names: list[str]) -> None:
        """
        Record one insert call with table/data/column_names.

        Parameters:
        - table: destination table.
        - data: tuple rows.
        - column_names: ordered column names.
        """
        self.insert_calls.append((table, data, column_names))

    def query(self, query: str, parameters: Mapping[str, Any]) -> _FakeQueryResult:
        """
        Record one query call and return preconfigured result.

        Parameters:
        - query: sql text.
        - parameters: bind parameters.
        """
        self.query_calls.append((query, parameters))
        return self._query_result


def test_thread_local_gateway_creates_distinct_clients_for_distinct_threads() -> None:
    """
    Ensure gateway allocates separate clients for different worker threads.
    """
    counter = {"created": 0}
    created_clients: list[_FakeClient] = []
    lock = threading.Lock()
    used_ids: list[int] = []

    def _factory() -> _FakeClient:
        with lock:
            counter["created"] += 1
            client = _FakeClient(counter["created"])
            created_clients.append(client)
            return client

    gateway = ThreadLocalClickHouseConnectGateway(client_factory=_factory)
    barrier = threading.Barrier(2)

    def _worker() -> tuple[int, int]:
        barrier.wait()
        rows = [{"market_id": 1, "symbol": "BTCUSDT"}]
        gateway.insert_rows("market_data.ref_instruments", rows)
        client_id = gateway._client().client_id  # type: ignore[attr-defined]
        used_ids.append(client_id)
        return (client_id, threading.get_ident())

    async def _run_concurrent() -> list[tuple[int, int]]:
        """
        Run two worker calls concurrently and return results as list payload.

        Parameters:
        - None.

        Returns:
        - List of `(client_id, thread_id)` tuples.
        """
        task1 = asyncio.create_task(asyncio.to_thread(_worker))
        task2 = asyncio.create_task(asyncio.to_thread(_worker))
        return list(await asyncio.gather(task1, task2))

    pairs = asyncio.run(_run_concurrent())
    ids = [client_id for client_id, _thread_id in pairs]
    thread_ids = [thread_id for _client_id, thread_id in pairs]
    assert len(set(thread_ids)) == 2
    assert len(set(ids)) == 2
    assert len(set(used_ids)) == 2
    assert counter["created"] == 2
    assert all(client.insert_calls for client in created_clients)


def test_thread_local_gateway_reuses_one_client_in_same_thread() -> None:
    """
    Ensure gateway keeps one cached client for repeated calls in one thread.
    """
    counter = {"created": 0}

    def _factory() -> _FakeClient:
        counter["created"] += 1
        return _FakeClient(counter["created"])

    gateway = ThreadLocalClickHouseConnectGateway(client_factory=_factory)

    def _single_thread_flow() -> int:
        gateway.select("SELECT 1", {})
        gateway.insert_rows(
            "market_data.raw_binance_klines_1m",
            [{"market_id": 1, "symbol": "BTCUSDT"}],
        )
        return gateway._client().client_id  # type: ignore[attr-defined]

    async def _run_twice_on_same_worker_thread() -> tuple[int, int]:
        first = await asyncio.to_thread(_single_thread_flow)
        second = await asyncio.to_thread(_single_thread_flow)
        return (first, second)

    first, second = asyncio.run(_run_twice_on_same_worker_thread())

    assert first == second
    assert counter["created"] == 1


def test_thread_local_gateway_select_mapping_matches_legacy_gateway_behavior() -> None:
    """
    Ensure `select` returns sequence of dict rows mapped by column order.
    """
    gateway = ThreadLocalClickHouseConnectGateway(client_factory=lambda: _FakeClient(1))
    rows = gateway.select("SELECT id, name FROM ref_instruments", {"limit": 1})

    assert isinstance(rows, Sequence)
    assert rows == [{"id": 1, "name": "BTCUSDT"}]
