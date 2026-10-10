"""S04 finite synchronized bursts. Dedicated disposable PG/CH only, immutable reports.

Run via a private environment injector. --source selects the entire Python import
root before importing this harness. Baseline is final S03, candidate final S04.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import subprocess
import threading
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, LiteralString, cast
from uuid import UUID, uuid4

import clickhouse_connect
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from apps.migrations.storage import split_clickhouse_sql
from apps.worker.market_data_ws.wiring.persistent_writer import PersistentRawWriterProcess
from tools.qa.market_data_mixed_load import raw_insert
from tools.qa.market_data_resource_comparison import BASE_MINUTE, memory
from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.services.insert_buffer import AsyncRawInsertBuffer
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


def row(index, minute):
    stamp = datetime(2026, 2, 1, tzinfo=UTC) + timedelta(minutes=minute)
    instrument = InstrumentId(MarketId(1), Symbol(f"S{index:04d}USDT"))
    return CandleWithMeta(
        Candle(
            instrument,
            UtcTimestamp(stamp),
            UtcTimestamp(stamp + timedelta(minutes=1)),
            100.0,
            101.0,
            99.0,
            100.5,
            1.0,
            100.0,
        ),
        CandleMeta(
            "ws",
            UtcTimestamp(datetime.now(UTC)),
            UUID(int=1),
            f"binance:spot:{instrument.symbol}",
            1,
            0.1,
            10.0,
        ),
    )


def key(candle):
    return str(candle.instrument_id.symbol), candle.ts_open.value


def quantiles(values):
    ordered = sorted(values)
    return (
        {
            f"p{p}_ms": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p / 100))] * 1000
            for p in [50, 95, 99]
        }
        if ordered
        else {}
    )


def cgroup(container):
    result = {}
    for name in ["memory.current", "memory.events", "cpu.stat", "io.stat"]:
        result[name] = subprocess.check_output(
            ["docker", "exec", container, "cat", "/sys/fs/cgroup/" + name], text=True
        ).strip()
    return result


def child_cpu(writer):
    total = 0.0
    for pid in writer.active_pids:
        value = subprocess.check_output(["ps", "-p", str(pid), "-o", "time="], text=True).strip()
        factors = [1, 60, 3600]
        total += sum(
            float(part) * factor for part, factor in zip(reversed(value.split(":")), factors)
        )
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--streams", type=int, choices=[4, 100], required=True)
    parser.add_argument("--bursts", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ROEHUB_S04_DISPOSABLE") != "1":
        raise RuntimeError("explicit disposable opt-in required")
    # Refuse user installation even if credentials/environment were mixed up.
    for container, port, expected in [
        ("roehub-s04-pg", "5432/tcp", "36432"),
        ("roehub-s04-clickhouse-1", "8123/tcp", "39123"),
    ]:
        info = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        assert info["NetworkSettings"]["Ports"][port] == [
            {"HostIp": "127.0.0.1", "HostPort": expected}
        ]
    with args.output.open("x") as out:
        out.write("{}\n")
    db = "s04_" + uuid4().hex
    dsn = os.environ["ROEHUB_MARKET_DATA_TEST_DSN"]
    pgdsn = make_conninfo(dsn, dbname=db)
    admin = clickhouse_connect.get_client(
        host="127.0.0.1",
        port=39123,
        username="s02_admin",
        password=Path(".cache/market-data-s02/private/admin").read_text(),
    )
    fixture = dict(
        streams=args.streams,
        stored_coordinates=102,
        seed_minutes=2880,
        sparse_every=7,
        bursts=args.bursts,
        cadence_s=0.2,
        buffer_rows=1000,
        flush_ms=100,
        warmup_bursts=3,
        post_drain_s=5,
    )
    record: dict[str, Any] = dict(
        schema_version="market-data-resource-comparison/v1",
        stage="S04",
        label=args.label,
        fixture=fixture,
        fixture_sha256=hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        platform=platform.platform(),
        source=str(args.source),
        samples=[],
        failures=[],
    )
    finished = threading.Event()
    writer = None
    sampler = None
    created = False
    try:
        with psycopg.connect(dsn, autocommit=True) as pg:
            pg.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
            created = True
        with psycopg.connect(pgdsn) as pg:
            for migration in [
                "0027_market_data_stream_recovery_v1.sql",
                "0033_market_data_stream_connections_v1.sql",
            ]:
                pg.execute(
                    sql.SQL(
                        cast(
                            LiteralString,
                            (args.source / "migrations/postgres" / migration).read_text(),
                        )
                    )
                )
        admin.command("CREATE DATABASE market_data")
        for statement in split_clickhouse_sql(
            (args.source / "migrations/clickhouse/market_data_ddl.sql").read_text()
        ):
            admin.command(statement)
        for i in range(102):
            raw_insert(
                admin, "market_data", f"S{i:04d}USDT", BASE_MINUTE, 2880, "s04_seed", sparse=i >= 4
            )
        env = dict(
            CH_HOST="127.0.0.1",
            CH_PORT="39123",
            CH_DATABASE="market_data",
            ROEHUB_CH_WORKLOADS_REQUIRED="1",
            ROEHUB_CH_STREAM_PASSWORD_FILE=str(
                Path(".cache/market-data-s02/private/classes/stream.password").resolve()
            ),
        )
        writer = PersistentRawWriterProcess(environ=env)
        receipts, received, scheduled, acknowledgements = {}, {}, {}, {}
        successful = set()
        loop_delays = []
        occupancy = []

        class SyncStore(PostgresStreamRecoveryStore):
            def remember(self, task):
                entry = super().remember(task)
                receipts[(str(task.instrument_id.symbol), task.time_range.start.value)] = (
                    time.perf_counter()
                )
                return entry

        async def execute():
            nonlocal sampler
            async with AsyncExitStack() as stack:
                admission = None
                store: Any
                kwargs: dict[str, Any]
                if args.label == "candidate":
                    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
                        AsyncPostgresStreamRecoveryStore,
                    )
                    from trading.contexts.market_data.application.services.batched_recovery import (
                        BatchedRecoveryAdmission,
                    )

                    class AsyncStore(AsyncPostgresStreamRecoveryStore):
                        async def remember_many(self, tasks):
                            result = await super().remember_many(tasks)
                            stamp = time.perf_counter()
                            for task in tasks:
                                receipts[
                                    (str(task.instrument_id.symbol), task.time_range.start.value)
                                ] = stamp
                            return result

                    store = AsyncStore(pgdsn)
                    await stack.enter_async_context(store.lease())
                    admission = BatchedRecoveryAdmission(store)
                    kwargs = dict(async_store=store, admission=admission)
                else:
                    store = SyncStore(pgdsn)
                    stack.enter_context(store.lease())
                    kwargs = dict(store=store)

                async def write(rows):
                    await writer.write(rows)
                    stamp = time.perf_counter()
                    for item in rows:
                        acknowledgements[key(item.candle)] = stamp

                buffer = AsyncRawInsertBuffer(
                    writer=cast(Any, None),
                    clock=SystemClock(),
                    flush_interval_ms=100,
                    max_buffer_rows=1000,
                    async_writer=write,
                    **kwargs,
                )
                await writer.start()
                await buffer.start()
                for warm in range(3):
                    await asyncio.gather(
                        *(buffer.submit_wait(row(i, -3 + warm)) for i in range(args.streams))
                    )
                    await buffer.flush(force=True)
                record["before_memory"] = memory(admin)
                record["before_cgroups"] = {
                    c: cgroup(c) for c in ["roehub-s04-pg", "roehub-s04-clickhouse-1"]
                }
                # Force flush only before measurement; sampler uses postgres DB, not measured DB.
                if args.label == "candidate":
                    await store._conn.execute("SELECT pg_stat_force_next_flush()")
                else:
                    store._lease.execute("SELECT pg_stat_force_next_flush()")
                await asyncio.sleep(1.1)
                with psycopg.connect(dsn, autocommit=True) as pg:
                    record["pg_before"] = pg.execute(
                        "SELECT xact_commit,xact_rollback,sessions FROM pg_stat_database "
                        "WHERE datname=%s",
                        (db,),
                    ).fetchone()
                receipts.clear()
                acknowledgements.clear()
                record["started_at"] = datetime.now(UTC).isoformat()
                record["child_cpu_before"] = child_cpu(writer)
                begin = time.perf_counter()
                cpu = resource.getrusage(resource.RUSAGE_SELF)
                record["cpu_before"] = cpu.ru_utime + cpu.ru_stime

                def sample():
                    with psycopg.connect(dsn, autocommit=True) as pg:
                        while not finished.wait(1):
                            try:
                                ids = {os.getpid()} | writer.active_pids
                                rss = sum(
                                    int(line.split()[1])
                                    for line in subprocess.check_output(
                                        ["ps", "-axo", "pid=,rss="], text=True
                                    ).splitlines()
                                    if line.split() and int(line.split()[0]) in ids
                                )
                                mem = memory(admin)
                                mem["MemoryTracking"] = admin.command(
                                    "SELECT value FROM system.metrics WHERE metric='MemoryTracking'"
                                )
                                connection_row = pg.execute(
                                    "SELECT count(*) FROM pg_stat_activity " "WHERE datname=%s",
                                    (db,),
                                ).fetchone()
                                assert connection_row is not None
                                connections = connection_row[0]
                                record["samples"].append(
                                    dict(
                                        elapsed_s=time.perf_counter() - begin,
                                        worker_child_rss_bytes=rss * 1024,
                                        ch=mem,
                                        parts=admin.command(
                                            "SELECT count() FROM system.parts "
                                            "WHERE active AND database='market_data'"
                                        ),
                                        merges=admin.command(
                                            "SELECT count() FROM system.merges "
                                            "WHERE database='market_data'"
                                        ),
                                        pg_connections=connections,
                                        ch_cgroup_bytes=int(
                                            subprocess.check_output(
                                                [
                                                    "docker",
                                                    "exec",
                                                    "roehub-s04-clickhouse-1",
                                                    "cat",
                                                    "/sys/fs/cgroup/memory.current",
                                                ]
                                            )
                                        ),
                                        pg_cgroup_bytes=int(
                                            subprocess.check_output(
                                                [
                                                    "docker",
                                                    "exec",
                                                    "roehub-s04-pg",
                                                    "cat",
                                                    "/sys/fs/cgroup/memory.current",
                                                ]
                                            )
                                        ),
                                    )
                                )
                            except Exception as error:
                                record["failures"].append(type(error).__name__)

                sampler = threading.Thread(target=sample)
                sampler.start()
                running = True

                async def heartbeat():
                    target = time.perf_counter() + 0.001
                    while running:
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                        now = time.perf_counter()
                        loop_delays.append(max(0, now - target))
                        target = now + 0.001
                        occupancy.append(
                            (
                                len(buffer._buffer),
                                getattr(buffer, "_retained_rows", 0),
                                getattr(buffer, "_retained_bytes", 0),
                            )
                        )

                pulse = asyncio.create_task(heartbeat())
                producers = []

                async def send(i, minute, due):
                    item = row(i, minute)
                    k = key(item.candle)
                    scheduled[k] = due
                    received[k] = time.perf_counter()
                    await buffer.submit_wait(item)
                    successful.add(k)

                try:
                    for burst in range(args.bursts):
                        due = begin + burst * 0.2
                        await asyncio.sleep(max(0, due - time.perf_counter()))
                        producers.extend(
                            asyncio.create_task(send(i, burst, due)) for i in range(args.streams)
                        )
                    await asyncio.gather(*producers)
                    await buffer.close()
                    if admission:
                        await admission.close()
                    record["completion_s"] = time.perf_counter() - begin
                    record["occupancy_peak"] = [max(v[i] for v in occupancy) for i in range(3)]
                    record["loop"] = quantiles(loop_delays)
                    record["loop_max_ms"] = max(loop_delays) * 1000
                finally:
                    running = False
                    await pulse
                await asyncio.sleep(5)
                cpu = resource.getrusage(resource.RUSAGE_SELF)
                record["parent_cpu_s"] = cpu.ru_utime + cpu.ru_stime - record["cpu_before"]
                record["child_cpu_s"] = child_cpu(writer) - record["child_cpu_before"]
                record["family_cpu_s"] = record["parent_cpu_s"] + record["child_cpu_s"]
                record["ended_at"] = datetime.now(UTC).isoformat()
                record["post_drain_memory"] = memory(admin) if sampler is None else None
            # Closing lease flushes its session statistics; independent reads cannot add its tx.
            await asyncio.sleep(1.1)

        asyncio.run(execute())
        finished.set()
        assert sampler is not None
        sampler.join()
        with psycopg.connect(dsn, autocommit=True) as pg:
            record["pg_after"] = pg.execute(
                "SELECT xact_commit,xact_rollback,sessions FROM pg_stat_database WHERE datname=%s",
                (db,),
            ).fetchone()
            pg.execute(sql.SQL("SELECT 1"))
        assert record["pg_after"] is not None and record["pg_before"] is not None
        n = args.streams * args.bursts
        record["arrived"] = len(received)
        record["accepted"] = len(successful)
        record["acknowledged"] = len(acknowledgements)
        record["pg_transactions"] = sum(record["pg_after"][:2]) - sum(record["pg_before"][:2])
        record["pg_sessions_created"] = record["pg_after"][2] - record["pg_before"][2]
        record["pg_tx_per_candle"] = record["pg_transactions"] / n
        record["pg_sessions_per_candle"] = record["pg_sessions_created"] / n
        for name, starts in [
            ("receive_to_ack", received),
            ("receipt_to_ack", receipts),
            ("scheduled_to_ack", scheduled),
        ]:
            record[name] = quantiles([acknowledgements[k] - start for k, start in starts.items()])
        record["dispatch_lag"] = quantiles([received[k] - start for k, start in scheduled.items()])
        canonical = admin.query(
            "SELECT symbol,toUnixTimestamp64Milli(ts_open),open,high,low,close,"
            "volume_base,volume_quote "
            "FROM market_data.canonical_candles_1m "
            "WHERE ts_open>=toDateTime('2026-02-01 00:00:00') ORDER BY symbol,ts_open"
        ).result_rows
        expected = [
            (
                f"S{i:04d}USDT",
                int((datetime(2026, 2, 1, tzinfo=UTC) + timedelta(minutes=m)).timestamp()*1000),
                100.0,
                101.0,
                99.0,
                100.5,
                1.0,
                100.0,
            )
            for i in range(args.streams)
            for m in range(args.bursts)
        ]
        record["canonical_count"] = len(canonical)
        record["canonical_sha256"] = hashlib.sha256(
            json.dumps(canonical, default=str).encode()
        ).hexdigest()
        with psycopg.connect(pgdsn) as pg:
            pending_row = pg.execute("SELECT count(*) FROM market_data_stream_recovery").fetchone()
            assert pending_row is not None
            record["pending_after"] = pending_row[0]
        record["post_drain_memory"] = memory(admin)
        record["after_cgroups"] = {
            c: cgroup(c) for c in ["roehub-s04-pg", "roehub-s04-clickhouse-1"]
        }
        admin.command("SYSTEM FLUSH LOGS")
        record["query_metrics"] = admin.query(
            "SELECT type,count(),sum(read_rows),sum(read_bytes),"
            "max(memory_usage) FROM system.query_log WHERE user='roehub_stream' "
            "AND event_time_microseconds>=parseDateTime64BestEffort(%(start)s) "
            "AND event_time_microseconds<=parseDateTime64BestEffort(%(end)s) GROUP BY type",
            parameters={"start": record["started_at"], "end": record["ended_at"]},
        ).result_rows
        record["checks"] = dict(
            all_accepted=len(successful) == n,
            canonical_content=[tuple(r) for r in canonical] == expected,
            all_acknowledged=len(acknowledgements) == n,
            all_canonical=len(canonical) == n,
            empty_journal=record["pending_after"] == 0,
            bounded_rows=record["occupancy_peak"][1] <= 2000,
            bounded_bytes=record["occupancy_peak"][2] <= 64 * 1024 * 1024,
            samples_ok=not record["failures"],
        )
        assert all(record["checks"].values())
    except BaseException as error:
        record["failures"].append(type(error).__name__)
        raise
    finally:
        finished.set()
        if sampler:
            sampler.join()
        if writer:
            asyncio.run(writer.close())
        admin.command("DROP DATABASE IF EXISTS market_data SYNC")
        admin.close()
        if created:
            with psycopg.connect(dsn, autocommit=True) as pg:
                pg.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db)))
        args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
