"""Real PostgreSQL journal proof in an isolated disposable schema."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

DSN = os.environ.get("ROEHUB_MARKET_DATA_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL target required")


@pytest.fixture
def store():
    schema = "stream_recovery_test_" + uuid4().hex
    with psycopg.connect(DSN) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        migration = Path("migrations/postgres/0027_market_data_stream_recovery_v1.sql").read_text()
        conn.execute(sql.SQL(cast(LiteralString, migration)))
        connections = Path("migrations/postgres/0033_market_data_stream_connections_v1.sql")
        conn.execute(sql.SQL(cast(LiteralString, connections.read_text())))
    repo = PostgresStreamRecoveryStore(
        make_conninfo(DSN, options=f"-c search_path={schema}"), max_entries=2
    )
    try:
        yield repo
    finally:
        with psycopg.connect(DSN) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def item(symbol="AAVEUSDT"):
    now = datetime(2026, 10, 4, tzinfo=UTC)
    return RestFillTask(
        InstrumentId(MarketId(1), Symbol(symbol)),
        TimeRange(UtcTimestamp(now), UtcTimestamp(now + timedelta(minutes=1))),
        "ws_recovery",
    )


def test_restart_and_fenced_ack(store):
    first = store.remember(item())
    assert store.remember(item()).token == first.token
    restarted = PostgresStreamRecoveryStore(store.dsn)
    assert restarted.pending() == [first]
    store.complete(first)
    second = store.remember(item())
    assert second.token != first.token
    store.complete(first)  # late ack may not erase the new receipt
    assert store.pending() == [second]
    store.complete_many([first, second])
    assert not store.pending()


def test_capacity_terminal_state_and_explicit_rearm(store):
    first = store.remember(item())
    second = store.remember(item("ADAUSDT"))
    with pytest.raises(RuntimeError, match="capacity"):
        store.remember(item("ALGOUSDT"))
    store.fail_many([first], code="non_retryable_ingestion_error")
    assert store.pending() == [second]
    with pytest.raises(RuntimeError, match="requires_rearm"):
        store.remember(item())
    assert store.rearm_failed() == 1
    assert len(store.pending()) == 2
    store.complete(first)
    assert len(store.pending()) == 2  # pre-rearm ack is stale


def test_retry_after_persists_and_live_collector_is_exclusive(store):
    from dataclasses import replace

    first = store.remember(item())
    next_at = datetime.now(UTC) + timedelta(seconds=60)
    store.retry(replace(first, attempt=3), next_retry_at=next_at, code="source_rate_limited")
    entry = store.pending()[0]
    assert entry.attempt == 3 and entry.next_retry_at == next_at
    other = PostgresStreamRecoveryStore(store.dsn)
    with store.lease():
        with pytest.raises(RuntimeError, match="already running"):
            with other.lease():
                pass
    with other.lease():
        assert other.pending()[0] == entry


def test_window_checkpoint_keeps_identity_and_survives_reopen(store):
    from dataclasses import replace

    original = item()
    original = replace(
        original,
        time_range=TimeRange(
            original.time_range.start,
            UtcTimestamp(original.time_range.start.value + timedelta(hours=3)),
        ),
    )
    entry = store.remember(original)
    next_at = original.time_range.start.value + timedelta(hours=1)
    store.advance(entry, start_at=next_at)
    after = PostgresStreamRecoveryStore(store.dsn).pending()[0]
    assert after.task.time_range.start.value == next_at
    assert after.task_key == entry.task_key and after.token == entry.token
    assert store.remember(original) == after
    store.complete(after)
    assert not store.pending()


def test_failed_history_admission_is_durable_and_inspectable(store):
    from dataclasses import replace
    task = replace(item(), reason="stream_history_repair")
    entry = store.remember(task)
    store.fail(entry, code="source_history_incomplete")
    restarted = PostgresStreamRecoveryStore(store.dsn)
    assert not restarted.pending()
    assert restarted.failed_history_repairs()[0].task == task
    assert restarted.history_error(1, "AAVEUSDT") == "source_history_incomplete"
    assert restarted.history_error(1, "ADAUSDT") is None
    restarted.clear_history_failures(task.instrument_id, task.time_range.end.value)
    assert restarted.history_error(1, "AAVEUSDT") is None
    assert not restarted.failed_history_repairs()


def test_collector_status_contains_only_safe_repair_fields(store):
    from dataclasses import replace
    task = replace(item(), reason="stream_history_repair")
    entry = store.remember(task)
    next_at = datetime.now(UTC) + timedelta(seconds=30)
    store.retry(entry, next_retry_at=next_at, code="storage_memory_pressure")
    status = store.collector_status()
    assert isinstance(status["running"], bool)
    assert status["repairs"][(1,"AAVEUSDT")] == dict(
        state="pending", error_code="storage_memory_pressure", next_retry_at=next_at,
        reason="stream_history_repair")
    store.fail(entry, code="source_history_incomplete")
    assert store.collector_status()["repairs"][(1,"AAVEUSDT")]["state"] == "failed"
    assert "token" not in str(status) and "task_key" not in str(status)


def test_socket_status_is_fresh_scoped_and_lease_fenced(store):
    with pytest.raises(RuntimeError, match="lease required"):
        store.publish_connections({(1, "ETHUSDT"): True})
    with store.lease():
        store.publish_connections({(1, "ETHUSDT"): False, (3, "ETHUSDT"): True})
        assert store.collector_status()["connections"] == {
            (1, "ETHUSDT"): False, (3, "ETHUSDT"): True}
        with psycopg.connect(store.dsn) as conn:
            conn.execute("UPDATE market_data_stream_connections "
                         "SET observed_at=now()-interval '21 seconds'")
        assert store.collector_status()["connections"] == {}
        store.publish_connections({(1, "ETHUSDT"): True})
        assert store.collector_status()["connections"] == {(1, "ETHUSDT"): True}
    assert store.collector_status()["connections"] == {}
    with store.lease():
        # A new collector lease cannot inherit a previous collector's healthy state.
        assert store.collector_status()["connections"] == {}


def test_async_batch_reconcile_tokens_lease_and_health(store):
    import asyncio
    from dataclasses import replace

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )

    async def scenario():
        repo = AsyncPostgresStreamRecoveryStore(store.dsn, max_entries=2)
        async with repo.lease():
            entries = await repo.remember_many([item(), item(), item('ADAUSDT')])
            assert entries[0] == entries[1]
            assert len(await repo.pending()) == 2
            with pytest.raises(RuntimeError, match='capacity'):
                await repo.remember_many([item('ALGOUSDT')])
            await repo.publish_connections({(1, 'AAVEUSDT'): True})
            assert store.collector_status()['connections'] == {(1, 'AAVEUSDT'): True}
            other = AsyncPostgresStreamRecoveryStore(store.dsn)
            with pytest.raises(RuntimeError, match='already running'):
                async with other.lease():
                    pass
            await repo.fail(entries[0], code='terminal')
            with pytest.raises(RuntimeError, match='requires_rearm'):
                await repo.remember_many([item()])
        # Reopen reconciles a commit whose response the former caller never received.
        store.rearm_failed()
        async with repo.lease():
            current = (await repo.remember_many([item()]))[0]
            assert current.token != entries[0].token
            await repo.complete(entries[0])
            await repo.fail(entries[0], code='stale')
            await repo.retry(replace(entries[0], attempt=9),
                             next_retry_at=datetime.now(UTC), code='stale')
            assert current in await repo.pending()
            await repo.complete_many([current, entries[2]])
            assert not await repo.pending()

    asyncio.run(scenario())


def test_async_unknown_commit_and_session_loss_fail_closed(store):
    import asyncio
    from contextlib import asynccontextmanager

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )

    class LostAcknowledgement(AsyncPostgresStreamRecoveryStore):
        @asynccontextmanager
        async def _transaction(self):
            async with super()._transaction() as conn:
                yield conn
            self._failed = True
            raise psycopg.OperationalError('injected lost commit acknowledgement')

    async def scenario():
        repo = LostAcknowledgement(store.dsn)
        async with repo.lease():
            with pytest.raises(psycopg.OperationalError):
                await repo.remember_many([item()])
        committed = store.pending()[0]
        healthy = AsyncPostgresStreamRecoveryStore(store.dsn)
        async with healthy.lease():
            assert await healthy.remember_many([item(), item()]) == [committed, committed]
            assert healthy._conn is not None
            with psycopg.connect(store.dsn) as admin:
                admin.execute('SELECT pg_terminate_backend(%s)', (healthy._conn.info.backend_pid,))
            with pytest.raises(psycopg.OperationalError):
                await healthy.complete(committed)
            with pytest.raises(RuntimeError, match='restart required'):
                await healthy.publish_connections({(1, 'AAVEUSDT'): True})
        assert store.pending() == [committed]
        assert not store.collector_status()['connections']

    asyncio.run(scenario())


def test_async_precommit_failure_rolls_back_entire_batch(store):
    import asyncio
    from contextlib import asynccontextmanager

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )

    class BeforeCommit(AsyncPostgresStreamRecoveryStore):
        @asynccontextmanager
        async def _transaction(self):
            async with super()._transaction() as conn:
                yield conn
                raise ValueError('injected before commit')

    async def scenario():
        repo = BeforeCommit(store.dsn)
        async with repo.lease():
            with pytest.raises(ValueError, match='before commit'):
                await repo.remember_many([item(), item('ADAUSDT')])
        assert not store.pending()
    asyncio.run(scenario())


def test_async_network_reply_blackhole_is_bounded(store):
    import asyncio
    import time

    from psycopg.conninfo import conninfo_to_dict

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )

    async def scenario():
        address = conninfo_to_dict(store.dsn)
        drop = asyncio.Event()
        connections = set()
        workers = set()
        async def proxy(reader, writer):
            current = asyncio.current_task()
            workers.add(current)
            upstream_reader, upstream_writer = await asyncio.open_connection(
                str(address['host']), int(str(address['port'])))
            connections.update((writer, upstream_writer))
            async def forward(source, target, downstream):
                while data := await source.read(65536):
                    if downstream and drop.is_set():
                        continue
                    target.write(data)
                    await target.drain()
            try:
                await asyncio.gather(forward(reader, upstream_writer, False),
                                     forward(upstream_reader, writer, True))
            finally:
                writer.close()
                upstream_writer.close()
                workers.discard(current)
        server = await asyncio.start_server(proxy, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        repo = AsyncPostgresStreamRecoveryStore(
            make_conninfo(store.dsn, port=port, sslmode='disable'))
        try:
            async with repo.lease():
                drop.set()
                begin = time.monotonic()
                with pytest.raises((TimeoutError, psycopg.OperationalError)):
                    await asyncio.wait_for(repo.remember_many([item()]), 12)
                assert time.monotonic()-begin < 10
                assert repo._failed
        finally:
            server.close()
            await server.wait_closed()
            for conn in connections:
                conn.close()
            for worker in list(workers):
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('boundary', ['before_ch', 'after_ch', 'before_completion'])
def test_async_real_ch_fault_restart_requires_actual_candle(store, boundary):
    import asyncio

    import clickhouse_connect

    from apps.migrations.storage import split_clickhouse_sql
    from apps.worker.market_data_ws.wiring.io_process import verify_received_range
    from tools.qa.market_data_recovery_comparison import row
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
        ClickHouseCanonicalCandleIndexReader,
        ClickHouseRawKlineWriter,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
        ClickHouseConnectGateway,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )
    from trading.contexts.market_data.application.services.batched_recovery import (
        BatchedRecoveryAdmission,
    )
    from trading.contexts.market_data.application.services.insert_buffer import (
        AsyncRawInsertBuffer,
        UnflushedBufferError,
    )
    from trading.platform.time.system_clock import SystemClock

    password = os.environ.get('ROEHUB_MARKET_DATA_TEST_CH_PASSWORD')
    if password is None:
        pytest.skip('explicit disposable ClickHouse target required')
    client = clickhouse_connect.get_client(host='127.0.0.1',
        port=int(os.environ['ROEHUB_MARKET_DATA_TEST_CH_PORT']),
        username=os.environ['ROEHUB_MARKET_DATA_TEST_CH_USER'], password=password)
    db = 's04_fault_' + uuid4().hex
    client.command(f'CREATE DATABASE {db}')
    try:
        for statement in split_clickhouse_sql(Path(
                'migrations/clickhouse/market_data_ddl.sql'
        ).read_text().replace('market_data', db)):
            client.command(statement)
        gateway = ClickHouseConnectGateway(client)
        writer = ClickHouseRawKlineWriter(gateway, db)
        index = ClickHouseCanonicalCandleIndexReader(gateway, db)
        async def scenario():
            class FailingCompletion(AsyncPostgresStreamRecoveryStore):
                async def complete_many(self, entries):
                    raise ConnectionError('injected before completion')
            repo = (FailingCompletion if boundary == 'before_completion'
                    else AsyncPostgresStreamRecoveryStore)(store.dsn)
            async with repo.lease():
                admission = BatchedRecoveryAdmission(repo)
                async def write(rows):
                    if boundary != 'before_ch':
                        await asyncio.to_thread(writer.write_1m, rows)
                    if boundary != 'before_completion':
                        raise ConnectionError('injected CH boundary')
                buffer = AsyncRawInsertBuffer(writer=writer, clock=SystemClock(),
                    flush_interval_ms=100, max_buffer_rows=2, async_store=repo,
                    admission=admission, async_writer=write, retry_base_s=10)
                await buffer.start()
                await buffer.submit_wait(row(0, 0))
                with pytest.raises(UnflushedBufferError):
                    await buffer.close()
                await admission.close()
            pending = store.pending()
            assert len(pending) == 1
            # An empty source response/absence alone cannot satisfy received-candle repair.
            if boundary == 'before_ch':
                with pytest.raises(Exception, match='source_incomplete'):
                    verify_received_range(pending[0].task, index)
                writer.write_1m([row(0, 0)])
            verify_received_range(pending[0].task, index)
            reopened = AsyncPostgresStreamRecoveryStore(store.dsn)
            async with reopened.lease():
                assert await reopened.remember_many([pending[0].task]) == pending
                await reopened.complete_many(pending)
                assert not await reopened.pending()
            assert client.command(f'SELECT count() FROM {db}.canonical_candles_1m') == 1
        asyncio.run(scenario())
    finally:
        client.command(f'DROP DATABASE {db} SYNC')
        client.close()


def test_real_pg_shared_batch_survives_repeated_producer_cancel(store):
    import asyncio

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )
    from trading.contexts.market_data.application.services.batched_recovery import (
        BatchedRecoveryAdmission,
    )

    async def scenario():
        repo = AsyncPostgresStreamRecoveryStore(store.dsn)
        async with repo.lease():
            blocker = await psycopg.AsyncConnection.connect(store.dsn)
            await blocker.execute('SELECT pg_advisory_xact_lock(744948345627)')
            admission = BatchedRecoveryAdmission(repo)
            first = asyncio.create_task(admission.remember(item()))
            second = asyncio.create_task(admission.remember(item('ADAUSDT')))
            await asyncio.sleep(.03)
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            await blocker.commit()
            await blocker.close()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert (await second).task == item('ADAUSDT')
            await admission.close()
            assert len(await repo.pending()) == 2
    asyncio.run(scenario())
