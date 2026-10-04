"""Single WS collector journal with bounded admission and fenced acknowledgements."""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, LiteralString, cast
from uuid import uuid4

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.ports.stores.stream_recovery_store import (
    RecoveryEntry,
    recovery_key,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class PostgresStreamRecoveryStore:
    def __init__(self, dsn: str, *, max_entries: int = 20000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        options = str(conninfo_to_dict(dsn).get("options") or "")
        self.dsn = make_conninfo(
            dsn,
            connect_timeout=2,
            options=options + " -c statement_timeout=2000 -c lock_timeout=1000",
        )
        self._lease = None
        self.max_entries = max_entries

    @contextmanager
    def lease(self):
        """One collector; losing the session prevents acceptance/acknowledgement."""
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            locked = conn.execute("SELECT pg_try_advisory_lock(744948345628)").fetchone()
            if not locked or not locked[0]:
                raise RuntimeError("stream collector is already running")
            self._lease = conn
            try:
                yield
            finally:
                self._lease = None

    def _check_lease(self) -> None:
        if self._lease is not None:
            self._lease.execute("SELECT 1")

    def remember(self, task: RestFillTask) -> RecoveryEntry:
        self._check_lease()
        with psycopg.Connection[dict[str, Any]].connect(self.dsn, row_factory=dict_row) as conn:
            # Admission is serialized across receipt/repair producers; no unbounded journal.
            conn.execute("SELECT pg_advisory_xact_lock(744948345627)")
            key = recovery_key(task)
            row = conn.execute(
                "SELECT * FROM market_data_stream_recovery WHERE task_key=%s", (key,)
            ).fetchone()
            if row is None:
                count = conn.execute("SELECT count(*) AS n FROM market_data_stream_recovery")
                count_row = count.fetchone()
                assert count_row is not None
                if count_row["n"] >= self.max_entries:
                    raise RuntimeError("stream_recovery_capacity_exceeded")
                row = conn.execute(
                    """INSERT INTO market_data_stream_recovery
                    (task_key,token,market_id,symbol,start_at,end_at,reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        key,
                        uuid4(),
                        task.instrument_id.market_id.value,
                        str(task.instrument_id.symbol),
                        task.time_range.start.value,
                        task.time_range.end.value,
                        task.reason,
                    ),
                ).fetchone()
            assert row is not None
            if row["state"] == "failed":
                raise RuntimeError("stream_recovery_requires_rearm")
            return _entry(row)

    def pending(self) -> list[RecoveryEntry]:
        self._check_lease()
        with psycopg.Connection[dict[str, Any]].connect(self.dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                "SELECT * FROM market_data_stream_recovery WHERE state='pending' "
                "ORDER BY start_at,task_key LIMIT %s",
                (self.max_entries,),
            ).fetchall()
            return [_entry(row) for row in rows]

    def advance(self, entry: RecoveryEntry, *, start_at: datetime) -> None:
        self._execute(
            "UPDATE market_data_stream_recovery SET start_at=%s,attempt=0,"
            "next_retry_at=NULL,error_code=NULL,updated_at=now() "
            "WHERE task_key=%s AND token=%s AND start_at<%s AND end_at>%s",
            (start_at, entry.task_key, entry.token, start_at, start_at),
        )

    def complete(self, entry: RecoveryEntry) -> None:
        self._execute(
            "DELETE FROM market_data_stream_recovery WHERE task_key=%s AND token=%s",
            (entry.task_key, entry.token),
        )

    def complete_many(self, entries: list[RecoveryEntry]) -> None:
        if entries:
            self._execute(
                "DELETE FROM market_data_stream_recovery WHERE (task_key,token) IN "
                "(SELECT * FROM unnest(%s::text[],%s::uuid[]))",
                ([e.task_key for e in entries], [e.token for e in entries]),
            )

    def retry(self, entry: RecoveryEntry, *, next_retry_at: datetime, code: str) -> None:
        self._execute(
            """UPDATE market_data_stream_recovery SET attempt=%s,next_retry_at=%s,
                         error_code=%s,updated_at=now() WHERE task_key=%s AND token=%s""",
            (entry.attempt, next_retry_at, code, entry.task_key, entry.token),
        )

    def fail(self, entry: RecoveryEntry, *, code: str) -> None:
        self._execute(
            "UPDATE market_data_stream_recovery SET state='failed',error_code=%s,"
            "updated_at=now() WHERE task_key=%s AND token=%s",
            (code, entry.task_key, entry.token),
        )

    def fail_many(self, entries: list[RecoveryEntry], *, code: str) -> None:
        if entries:
            self._execute(
                "UPDATE market_data_stream_recovery SET state='failed',error_code=%s,"
                "updated_at=now() WHERE (task_key,token) IN "
                "(SELECT * FROM unnest(%s::text[],%s::uuid[]))",
                (code, [e.task_key for e in entries], [e.token for e in entries]),
            )

    def rearm_failed(self) -> int:
        """Explicit operator action after fixing the cause, never an automatic retry."""
        with psycopg.connect(self.dsn) as conn:
            return conn.execute(
                "UPDATE market_data_stream_recovery SET state='pending',"
                "next_retry_at=NULL,error_code=NULL,attempt=0,token=%s,"
                "updated_at=now() WHERE state='failed'",
                (uuid4(),),
            ).rowcount

    def _execute(self, sql: str, parameters: tuple) -> None:
        self._check_lease()
        with psycopg.connect(self.dsn) as conn:
            conn.execute(cast(LiteralString, sql), parameters)


def _entry(row: dict[str, Any]) -> RecoveryEntry:
    return RecoveryEntry(
        task=RestFillTask(
            InstrumentId(MarketId(row["market_id"]), Symbol(row["symbol"])),
            TimeRange(UtcTimestamp(row["start_at"]), UtcTimestamp(row["end_at"])),
            row["reason"],
        ),
        token=row["token"],
        attempt=row["attempt"],
        next_retry_at=row["next_retry_at"],
        error_code=row["error_code"],
        key=row["task_key"],
    )
