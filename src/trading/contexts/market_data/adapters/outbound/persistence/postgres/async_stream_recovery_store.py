"""One lifecycle-owned async session: journal transactions and lease cannot diverge.

A transport/unknown-commit failure poisons this lifecycle. Reopening obtains the
exclusive lease and reconciles the same stable keys; it never deletes uncertain
receipts. No schema change, connection-per-candle, or concurrent session use.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, LiteralString, cast
from uuid import uuid4

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
    _entry,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    RecoveryEntry,
    recovery_key,
)
from trading.platform.postgres_budget import async_connect


class AsyncPostgresStreamRecoveryStore:
    def __init__(self, dsn: str, *, max_entries: int = 20000) -> None:
        self._dsn = PostgresStreamRecoveryStore(dsn, max_entries=max_entries).dsn
        # The server owns the advisory lease. A client-side deadline cannot
        # release it when a broken network drops the socket's final close.
        # Bound dead-peer detection on this Linux deployment's server socket;
        # healthy idle collectors still answer TCP keepalives.
        options = str(conninfo_to_dict(self._dsn).get("options") or "")
        self._dsn = make_conninfo(
            self._dsn,
            options=options + " -c tcp_keepalives_idle=10 -c tcp_keepalives_interval=5"
            " -c tcp_keepalives_count=3 -c tcp_user_timeout=15000",
        )
        self.max_entries = max_entries
        self._conn: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._lock = asyncio.Lock()
        self._failed = False

    @asynccontextmanager
    async def lease(self):
        if self._conn is not None:
            raise RuntimeError("collector lease already entered")
        async with await async_connect(self._dsn, autocommit=True, row_factory=dict_row) as conn:
            async with self._deadline(conn):
                row = await (
                    await conn.execute("SELECT pg_try_advisory_lock(744948345628) AS locked")
                ).fetchone()
            if not row or not row["locked"]:
                raise RuntimeError("stream collector is already running")
            async with self._deadline(conn):
                slot = await (
                    await conn.execute("SELECT pg_try_advisory_lock(563298,12) AS locked")
                ).fetchone()
            if not slot or not slot["locked"]:
                raise RuntimeError("market recovery connection budget exhausted")
            self._conn, self._failed = conn, False
            try:
                yield self
            finally:
                # Caller must drain producers before exiting; lock fences remaining commands.
                async with self._lock:
                    self._conn = None

    @asynccontextmanager
    async def _deadline(self, conn):
        # Close the socket BEFORE cancelling: psycopg rollback/cancel otherwise
        # waits for a blackholed reply despite a server statement timeout.
        owner = asyncio.current_task()
        expired = False

        async def expire():
            nonlocal expired
            await asyncio.sleep(3)
            expired = True
            self._failed = True
            await conn.close()
            assert owner is not None
            owner.cancel()

        watchdog = asyncio.create_task(expire(), name="recovery-pg-deadline")
        try:
            yield
        except BaseException:
            if expired:
                raise TimeoutError("recovery PG deadline; restart required") from None
            raise
        finally:
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)

    @asynccontextmanager
    async def _transaction(self):
        async with self._lock:
            if self._conn is None or self._failed:
                raise RuntimeError("collector lease unavailable; restart required")
            try:
                async with self._deadline(self._conn):
                    async with self._conn.transaction():
                        yield self._conn
            except (psycopg.OperationalError, asyncio.CancelledError, TimeoutError):
                self._failed = True
                raise

    async def remember_many(self, tasks: list[RestFillTask]) -> list[RecoveryEntry]:
        if not tasks:
            return []
        if len(tasks) > 256:
            raise ValueError("recovery batch exceeds 256")
        keys = [recovery_key(task) for task in tasks]
        unique = dict(zip(keys, tasks))
        async with self._transaction() as conn:
            await conn.execute("SELECT pg_advisory_xact_lock(744948345627)")
            rows = await (
                await conn.execute(
                    "SELECT * FROM market_data_stream_recovery WHERE task_key=ANY(%s)",
                    (list(unique),),
                )
            ).fetchall()
            by_key = {row["task_key"]: row for row in rows}
            if any(row["state"] == "failed" for row in rows):
                raise RuntimeError("stream_recovery_requires_rearm")
            missing = [(key, task) for key, task in unique.items() if key not in by_key]
            if missing:
                count = await (
                    await conn.execute("SELECT count(*) AS n FROM market_data_stream_recovery")
                ).fetchone()
                assert count is not None
                if count["n"] + len(missing) > self.max_entries:
                    raise RuntimeError("stream_recovery_capacity_exceeded")
                rows = await (
                    await conn.execute(
                        "INSERT INTO market_data_stream_recovery "
                        "(task_key,token,market_id,symbol,start_at,end_at,reason) "
                        "SELECT * FROM unnest(%s::text[],%s::uuid[],%s::int[],%s::text[],"
                        "%s::timestamptz[],%s::timestamptz[],%s::text[]) RETURNING *",
                        (
                            [key for key, _ in missing],
                            [uuid4() for _ in missing],
                            [task.instrument_id.market_id.value for _, task in missing],
                            [str(task.instrument_id.symbol) for _, task in missing],
                            [task.time_range.start.value for _, task in missing],
                            [task.time_range.end.value for _, task in missing],
                            [task.reason for _, task in missing],
                        ),
                    )
                ).fetchall()
                by_key.update({row["task_key"]: row for row in rows})
            result = [_entry(by_key[key]) for key in keys]
        return result  # transaction exit confirmed commit

    async def pending(self) -> list[RecoveryEntry]:
        return await self._read(False)

    async def failed_history_repairs(self) -> list[RecoveryEntry]:
        return await self._read(True)

    async def _read(self, failed: bool) -> list[RecoveryEntry]:
        async with self._transaction() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM market_data_stream_recovery WHERE state=%s "
                    "AND (NOT %s OR reason='stream_history_repair') "
                    "ORDER BY start_at,task_key LIMIT %s",
                    ("failed" if failed else "pending", failed, self.max_entries),
                )
            ).fetchall()
            return [_entry(row) for row in rows]

    async def _execute(self, query: str, params: tuple) -> None:
        async with self._transaction() as conn:
            await conn.execute(cast(LiteralString, query), params)

    async def complete(self, entry: RecoveryEntry) -> None:
        await self.complete_many([entry])

    async def complete_many(self, entries: list[RecoveryEntry]) -> None:
        if entries:
            await self._execute(
                "DELETE FROM market_data_stream_recovery WHERE (task_key,token) IN "
                "(SELECT * FROM unnest(%s::text[],%s::uuid[]))",
                ([e.task_key for e in entries], [e.token for e in entries]),
            )

    async def advance(self, entry: RecoveryEntry, *, start_at: datetime) -> None:
        await self._execute(
            "UPDATE market_data_stream_recovery SET start_at=%s,attempt=0,"
            "next_retry_at=NULL,error_code=NULL,updated_at=now() "
            "WHERE task_key=%s AND token=%s AND start_at<%s AND end_at>%s",
            (start_at, entry.task_key, entry.token, start_at, start_at),
        )

    async def retry(self, entry: RecoveryEntry, *, next_retry_at: datetime, code: str) -> None:
        await self._execute(
            "UPDATE market_data_stream_recovery SET attempt=%s,next_retry_at=%s,"
            "error_code=%s,updated_at=now() WHERE task_key=%s AND token=%s",
            (entry.attempt, next_retry_at, code, entry.task_key, entry.token),
        )

    async def fail(self, entry: RecoveryEntry, *, code: str) -> None:
        await self.fail_many([entry], code=code)

    async def fail_many(self, entries: list[RecoveryEntry], *, code: str) -> None:
        if entries:
            await self._execute(
                "UPDATE market_data_stream_recovery SET state='failed',error_code=%s,"
                "updated_at=now() WHERE (task_key,token) IN "
                "(SELECT * FROM unnest(%s::text[],%s::uuid[]))",
                (code, [e.task_key for e in entries], [e.token for e in entries]),
            )

    async def publish_connections(self, snapshot: dict[tuple[int, str], bool]) -> None:
        if len(snapshot) > self.max_entries:
            raise RuntimeError("stream_connection_capacity_exceeded")
        async with self._transaction() as conn:
            await conn.execute(
                "DELETE FROM market_data_stream_connections WHERE (market_id,symbol) NOT IN "
                "(SELECT * FROM unnest(%s::int[],%s::text[]))",
                ([key[0] for key in snapshot], [key[1] for key in snapshot]),
            )
            if snapshot:
                await conn.execute(
                    "INSERT INTO market_data_stream_connections "
                    "(market_id,symbol,connected,collector_pid) "
                    "SELECT m,s,c,pg_backend_pid() FROM unnest(%s::int[],%s::text[],"
                    "%s::bool[]) AS x(m,s,c) "
                    "ON CONFLICT (market_id,symbol) DO UPDATE SET "
                    "connected=EXCLUDED.connected,collector_pid=EXCLUDED.collector_pid,"
                    "observed_at=now()",
                    (
                        [key[0] for key in snapshot],
                        [key[1] for key in snapshot],
                        list(snapshot.values()),
                    ),
                )
