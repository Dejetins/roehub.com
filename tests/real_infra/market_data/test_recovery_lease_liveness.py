"""Server-owned collector lease stays exclusive while its idle peer is healthy."""

import asyncio

import psycopg

from tests.real_infra.market_data.test_storage_schema_postgres import db  # noqa: F401
from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
    AsyncPostgresStreamRecoveryStore,
)


def test_idle_collector_keeps_server_lease_with_bounded_dead_peer_detection(db):  # noqa: F811
    async def exercise():
        store = AsyncPostgresStreamRecoveryStore(db)
        async with store.lease():
            assert store._conn is not None
            row = await (
                await store._conn.execute(
                    "SELECT current_setting('tcp_keepalives_idle') AS idle, "
                    "current_setting('tcp_keepalives_interval') AS interval, "
                    "current_setting('tcp_keepalives_count') AS count, "
                    "current_setting('tcp_user_timeout') AS timeout"
                )
            ).fetchone()
            assert row == {"idle": "10", "interval": "5", "count": "3", "timeout": "15000"}
            await asyncio.sleep(30)
            with psycopg.connect(db, autocommit=True) as other:
                assert other.execute("SELECT pg_try_advisory_lock(744948345628)").fetchone() == (
                    False,
                )
        with psycopg.connect(db, autocommit=True) as other:
            assert other.execute("SELECT pg_try_advisory_lock(744948345628)").fetchone() == (True,)

    asyncio.run(exercise())
