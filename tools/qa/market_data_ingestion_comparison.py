"""S03 fixed queued-job replay: real PG/CH, deterministic loopback HTTP, two runtimes.

Extends market-data-resource-comparison/v1. Explicit disposable opt-in required.
All attempts use exclusive output names; no raw provider bodies or credentials persist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, LiteralString, cast
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import clickhouse_connect
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from apps.migrations.storage import split_clickhouse_sql
from tools.qa.market_data_mixed_load import raw_insert
from tools.qa.market_data_resource_comparison import BASE_MINUTE, memory
from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.application.ports.provider_admission import (
    ProviderRequest,
    QuotaWindow,
)
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError
from trading.contexts.market_data.application.services.work_requests import WorkRequest
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase
from trading.platform.time.system_clock import SystemClock

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "runner": "src/trading/contexts/market_data/application/services/work_requests.py",
    "repository": ("src/trading/contexts/market_data/adapters/outbound/persistence/"
                   "postgres/work_request_repository.py"),
    "http": "src/trading/contexts/market_data/adapters/outbound/clients/common_http/http_client.py",
}


def load(root, kind, label):
    path = Path(root) / PATHS[kind]
    # Candidate relative transport import needs its actual package name.
    if kind == "http" and label == "candidate":
        from trading.contexts.market_data.adapters.outbound.clients.common_http import (
            RequestsHttpClient,
        )

        return SimpleNamespace(RequestsHttpClient=RequestsHttpClient)
    spec = importlib.util.spec_from_file_location("s03_" + label + "_" + kind, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def policy():
    return ProviderRequest(
        "binance",
        "s03-emulator-egress",
        4,
        (QuotaWindow("spot_weight", 1, 100, 2), QuotaWindow("spot_weight", 60, 6000, 2)),
    )


class CountedAdmission:
    def __init__(self, delegate):
        self.delegate = delegate
        self.counts = {"attempts": 0, "admitted": 0, "source_rate_limited": 0,
                       "source_unavailable": 0}
        self.lock = threading.Lock()

    def reserve(self, request):
        with self.lock:
            self.counts["attempts"] += 1
        try:
            permit = self.delegate.reserve(request)
        except TemporarySourceError as error:
            with self.lock:
                self.counts[error.code] += 1
            raise
        with self.lock:
            self.counts["admitted"] += 1
        return permit


def http_client(root, label, dsn):
    cls = load(root, "http", label).RequestsHttpClient
    if label == "baseline":
        return cls()
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.provider_admission import (  # noqa: E501
        PostgresProviderAdmission,
    )

    return cls(
        admission=CountedAdmission(PostgresProviderAdmission(dsn)),
        policy=SimpleNamespace(request=lambda *_: policy()),
    )


def collector(root, label, dsn, url, start, count, output):
    http = http_client(root, label, dsn)
    results, retries = [], 0
    record = {}
    try:
        for index in range(count):
            offered = start + index * 3
            time.sleep(max(0, offered - time.monotonic()))
            for attempt in range(2400):
                try:
                    response = http.get_json(
                        url=url + "/api/v3/klines",
                        params={
                            "symbol": f"LIVE{index%4}USDT",
                            "startTime": BASE_MINUTE * 60000,
                            "endTime": (BASE_MINUTE + 1) * 60000 - 1,
                            "limit": 1,
                        },
                        timeout_s=10,
                        retries=0,
                        backoff_base_s=0.1,
                        backoff_max_s=0.1,
                        backoff_jitter_s=0,
                    )
                    assert len(response.body) == 1
                    results.append(time.monotonic() - offered)
                    break
                except TemporarySourceError as error:
                    retries += 1
                    time.sleep(max(0.25, min(2, error.retry_after_s)))
            else:
                raise RuntimeError("collector_drain_deadline")
        record = {
            "completed": len(results),
            "retries": retries,
            "latency_s": results,
            "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_s": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        }
    except BaseException as error:
        record = {"failure": type(error).__name__, "completed": len(results), "retries": retries}
        raise
    finally:
        if hasattr(http, "close"):
            http.close()
        record["provider_admission"] = getattr(getattr(http, "admission", None), "counts", {})
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        record["process_family_cpu_s"] = (own.ru_utime + own.ru_stime
                                           + children.ru_utime + children.ru_stime)
        Path(output).write_text(json.dumps(record, indent=2) + "\n")


class Emulator(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, fast, slow):
        super().__init__(("127.0.0.1", 0), Handler)
        self.lock = threading.Lock()
        self.fast, self.slow = fast, slow
        self.active = self.peak = self.rejected = self.requests = 0
        self.connections = set()
        self.windows = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        server = cast(Emulator, self.server)
        query = parse_qs(urlsplit(self.path).query)
        symbol = query["symbol"][0]
        start, end = int(query["startTime"][0]), int(query["endTime"][0])
        with server.lock:
            server.connections.add(self.client_address)
            server.requests += 1
            now = time.time()
            allowed = server.active < 4
            keys = [(seconds, int(now // seconds)) for seconds in (1, 60)]
            for key, limit in zip(keys, (100, 6000)):
                allowed &= server.windows.get(key, 0) + 2 <= limit
            if allowed:
                for key in keys:
                    server.windows[key] = server.windows.get(key, 0) + 2
                server.active += 1
                server.peak = max(server.peak, server.active)
            else:
                server.rejected += 1
        status = 200 if allowed else 429
        try:
            if allowed:
                delay = (
                    0.01
                    if symbol.startswith("LIVE")
                    else (server.slow if symbol == "S0000USDT" else server.fast)
                )
                time.sleep(delay)
                body = [
                    [minute, "1", "1", "1", "1", "1", minute + 59999, "1", 1, "1", "1", "0"]
                    for minute in range(start, end + 1, 60000)
                ]
            else:
                body = {}
            raw = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if not allowed:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
        finally:
            if allowed:
                with server.lock:
                    server.active -= 1


class SerializedStorage:
    """Same fixture admission on both sides: one S02 backfill query at a time.

    This isolates scheduler/provider cost from deliberate CH cap rejection. The
    separate real-CH pressure tests cover durable retry/correctness under rejection.
    """

    def __init__(self, client):
        self.gateway = ClickHouseConnectGateway(client)
        self.lock = threading.Lock()
        self.calls = self.rows = 0

    def select(self, query, parameters):
        with self.lock:
            self.calls += 1
            return self.gateway.select(query, parameters)

    def insert_rows(self, table, rows):
        with self.lock:
            self.calls += 1
            self.rows += len(rows)
            return self.gateway.insert_rows(table, rows)


@dataclass(frozen=True)
class MeasuredFill(RestFillRange1mUseCase):
    observer: Any = None

    def run(self, task):
        with self.observer.lock:
            self.observer.active += 1
        try:
            result = super().run(task)
            with self.observer.lock:
                self.observer.finished.append(time.monotonic())
            return result
        finally:
            with self.observer.lock:
                self.observer.active -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--fast", type=float, default=1.2)
    parser.add_argument("--slow", type=float, default=3)
    parser.add_argument("--collector-count", type=int, default=120)
    args = parser.parse_args()
    if os.environ.get("ROEHUB_S03_DISPOSABLE") != "1":
        raise ValueError("explicit disposable opt-in required")
    with args.output.open("x") as output:
        output.write("{}\n")
    dsn = os.environ["ROEHUB_MARKET_DATA_TEST_DSN"]
    pg_address = conninfo_to_dict(dsn)
    for container, port, expected in (
        ("roehub-s03-postgres", "5432/tcp", pg_address.get("port", "5432")),
        ("roehub-s03-clickhouse-1", "8123/tcp", os.environ["CH_PORT"]),
    ):
        ports = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container],
            text=True,
        ))
        if ports.get(port) != [{"HostIp": "127.0.0.1", "HostPort": expected}]:
            raise ValueError("target_is_not_the_named_loopback_disposable_container")
    if pg_address.get("host") != "127.0.0.1":
        raise ValueError("loopback_disposable_PG_required")
    tag = "s03_" + args.label + "_" + uuid4().hex
    schema = "market_data"  # Dedicated S03 disposable server; recreated for every run.
    pgdsn = make_conninfo(dsn, options=f"-c search_path={schema}")
    chargs: dict[str, Any] = dict(host="127.0.0.1", port=int(os.environ["CH_PORT"]))
    admin = clickhouse_connect.get_client(
        **chargs,
        username="s02_admin",
        password=Path(".cache/market-data-s02/private/admin").read_text(),
    )
    runtime = json.loads(
        subprocess.check_output(
            [
                "docker",
                "inspect",
                "--format",
                '{"image":{{json '
                '.Image}},"memory":{{.HostConfig.Memory}},"nano_cpus":{{.HostConfi'
                'g.NanoCpus}}}',
                "roehub-s03-clickhouse-1",
            ],
            text=True,
        )
    )
    record: dict[str, Any] = dict(
        schema_version="market-data-resource-comparison/v1",
        stage="S03",
        label=args.label,
        source={
            p: hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in PATHS.values()
        },
        runtime=runtime,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        hardware=platform.platform(),
        fixture=dict(
            coordinates=102,
            streams=4,
            seed_minutes=2880,
            sparse_every=7,
            jobs=102,
            windows=args.windows,
            window_minutes=1440,
            concurrency=4,
            fast_seconds=args.fast,
            slow_seconds=args.slow,
            collector=args.collector_count,
            quotas=[[1, 100, 2], [60, 6000, 2]],
            quota_concurrency=4,
        ),
        warmup_s=60,
        steady_s=300,
        samples=[],
        failures=[],
    )
    record["fixture_sha256"] = hashlib.sha256(
        json.dumps(record["fixture"], sort_keys=True).encode()
    ).hexdigest()
    server = Emulator(args.fast, args.slow)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    http = None
    child = None
    client = None
    finished = threading.Event()
    sampler = None
    created_pg = created_ch = False
    try:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            created_pg = True
        with psycopg.connect(pgdsn, autocommit=True) as connection:
            connection.execute(
                "CREATE TABLE identity_organizations (organization_id uuid PRIMARY KEY,status text)"
            )
            connection.execute(
                'CREATE TABLE identity_memberships (organization_id uuid,user_id '
                'uuid,status text,PRIMARY KEY(organization_id,user_id))'
            )
            for name in (
                "0023_market_data_work_requests_v1.sql",
                "0024_market_data_full_history_v1.sql",
                "0025_market_data_work_recovery_v1.sql",
                "0026_market_data_queue_events_v1.sql",
                "0034_market_data_provider_admission_v1.sql",
            ):
                connection.execute(
                    sql.SQL(cast(LiteralString, (ROOT / "migrations/postgres" / name).read_text()))
                )
            org, actor = uuid4(), uuid4()
            connection.execute("INSERT INTO identity_organizations VALUES (%s,'active')", (org,))
            connection.execute(
                "INSERT INTO identity_memberships VALUES (%s,%s,'active')", (org, actor)
            )
        admin.command(f"CREATE DATABASE {schema}")  # Fail if any existing data owns it.
        created_ch = True
        for statement in split_clickhouse_sql(
            (ROOT / "migrations/clickhouse/market_data_ddl.sql")
            .read_text()
            .replace("market_data", schema)
        ):
            admin.command(statement)
        for coordinate in range(102):
            raw_insert(
                admin,
                schema,
                f"S{coordinate:04d}USDT",
                BASE_MINUTE,
                2880,
                "s03_seed",
                sparse=coordinate >= 4,
            )
        client = clickhouse_connect.get_client(
            **chargs,
            username="roehub_backfill",
            password=Path(".cache/market-data-s02/private/classes/backfill.password").read_text(),
            settings={"log_comment": tag},
        )
        storage = SerializedStorage(client)
        clock = SystemClock()
        config = load_market_data_runtime_config(ROOT / "configs/dev/market_data.yaml")
        url = f"http://127.0.0.1:{server.server_port}"
        config = replace(
            config,
            markets=tuple(
                replace(m, rest=replace(m.rest, base_url=url, timeout_s=10, retries=0))
                for m in config.markets
            ),
        )
        http = http_client(args.source, args.label, pgdsn)
        observed = SimpleNamespace(lock=threading.Lock(), active=0, finished=[])
        fill = MeasuredFill(
            source=RestCandleIngestSource(config, clock, http, uuid4()),
            writer=ClickHouseRawKlineWriter(storage, schema),
            clock=clock,
            max_days_per_insert=1,
            batch_size=10000,
            index_reader=ClickHouseCanonicalCandleIndexReader(storage, schema),
            observer=observed,
        )
        repository = load(
            args.source, "repository", args.label
        ).PostgresMarketDataWorkRequestRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=pgdsn), execution_dsn=pgdsn
        )
        start_at = datetime(2026, 1, 3, tzinfo=UTC)
        end_at = start_at + timedelta(days=args.windows)
        for index in range(102):
            repository.submit(
                organization_id=org,
                actor_user_id=actor,
                key=uuid4(),
                request=WorkRequest(
                    "candle_ingestion", 1, (f"S{index:04d}USDT",), start_at, end_at
                ),
                now=datetime.now(UTC) + timedelta(microseconds=index),
            )
        runner = load(args.source, "runner", args.label).MarketDataWorkRequestRunner(
            store=repository,
            fill=fill,
            refresh_catalog=lambda *_: None,
            max_windows=1,
            window_minutes=1440,
        )
        with psycopg.connect(pgdsn) as connection:
            record["pg_stats_before"] = connection.execute(
                'SELECT xact_commit,xact_rollback,numbackends FROM pg_stat_database '
                'WHERE datname=current_database()'
            ).fetchone()
        record["pre_work_memory"] = memory(admin)
        record["container_before"] = subprocess.check_output(
            ["docker", "exec", "roehub-s03-clickhouse-1", "sh", "-c",
             "cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/io.stat"], text=True)
        begin = time.monotonic()
        child = mp.get_context("spawn").Process(
            target=collector,
            args=(
                str(args.source),
                args.label,
                pgdsn,
                url,
                begin,
                args.collector_count,
                str(args.output) + ".collector.json",
            ),
        )
        child.start()

        def sample():
            previous = begin
            with psycopg.connect(pgdsn, autocommit=True) as connection:
                while not finished.wait(1):
                    now = time.monotonic()
                    with observed.lock:
                        active = observed.active
                    row = connection.execute(
                        "SELECT count(*) FROM market_data_work_requests WHERE state='queued'"
                    ).fetchone()
                    assert row is not None
                    queued = row[0]
                    mem = memory(admin)
                    mem["MemoryTracking"] = admin.command(
                        "SELECT value FROM system.metrics WHERE metric='MemoryTracking'"
                    )
                    pg = connection.execute(
                        'SELECT xact_commit,xact_rollback,numbackends FROM pg_stat_database '
                        'WHERE datname=current_database()'
                    ).fetchone()
                    cgroup = subprocess.check_output(
                        [
                            "docker",
                            "exec",
                            "roehub-s03-clickhouse-1",
                            "cat",
                            "/sys/fs/cgroup/memory.current",
                        ],
                        text=True,
                    ).strip()
                    processes = [tuple(map(int, line.split())) for line in
                                 subprocess.check_output(
                                     ["ps", "-axo", "pid=,ppid=,rss="], text=True
                                 ).splitlines() if line.strip()]
                    family = {os.getpid()}
                    while True:
                        found = {pid for pid, parent, _ in processes if parent in family}
                        if found <= family:
                            break
                        family |= found
                    rss = sum(size for pid, _, size in processes if pid in family)
                    parts = admin.query(
                        'SELECT count(),sum(bytes_on_disk) FROM system.parts WHERE active AND '
                        'database=%(db)s',
                        parameters={"db": schema},
                    ).result_rows[0]
                    record["samples"].append(
                        dict(
                            elapsed=now - begin,
                            active=active,
                            queued=queued,
                            idle_ready_slot_seconds=(4 - active) * (now - previous)
                            if queued
                            else 0,
                            memory=mem,
                            parts=parts,
                            merges=admin.command(
                                "SELECT count() FROM system.merges WHERE database=%(db)s",
                                parameters={"db": schema}),
                            pg=pg,
                            worker_and_child_rss_kib=rss,
                            cgroup_memory_bytes=int(cgroup),
                        )
                    )
                    previous = now

        def checked_sample():
            try:
                sample()
            except BaseException as error:
                record["failures"].append("sampler:" + type(error).__name__)

        sampler = threading.Thread(target=checked_sample, daemon=True)
        sampler.start()
        with ThreadPoolExecutor(4) as pool:
            while True:
                runner.run_batch(executor=pool, concurrency=4)
                with psycopg.connect(pgdsn) as connection:
                    states = dict(
                        connection.execute(
                            "SELECT state,count(*) FROM market_data_work_requests GROUP BY state"
                        ).fetchall()
                    )
                if states == {"succeeded": 102}:
                    break
                if "failed" in states or time.monotonic() - begin > 2400:
                    raise RuntimeError("queue_failure_or_drain_deadline")
                time.sleep(2)  # Actual baseline/candidate scheduler empty-cycle cadence.
        completion = time.monotonic() - begin
        child.join(timeout=1200)
        if child.is_alive() or child.exitcode != 0:
            raise RuntimeError("collector_failed")
        finished.set()
        sampler.join()
        if hasattr(http, "close"):
            http.close()  # Reap HTTP children before measuring total family CPU.
        record["provider_admission"] = getattr(getattr(http, "admission", None), "counts", {})
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        record["process_family_cpu_s"] = (own.ru_utime + own.ru_stime
                                           + children.ru_utime + children.ru_stime)
        marks = [at - begin for at in observed.finished]
        steady = [at for at in marks if 60 <= at < 360]
        idle = sum(
            s["idle_ready_slot_seconds"] for s in record["samples"] if 60 <= s["elapsed"] < 360
        )
        with psycopg.connect(pgdsn) as connection:
            record["pg_states"] = dict(
                connection.execute(
                    "SELECT state,count(*) FROM market_data_work_requests GROUP BY state"
                ).fetchall()
            )
            record["pg_counters"] = connection.execute(
                'SELECT sum(completed_units),sum(rows_read),sum(rows_written) FROM '
                'market_data_work_requests'
            ).fetchone()
            row = connection.execute("SELECT count(*) FROM market_data_provider_permits").fetchone()
            assert row is not None
            record["permits_after_drain"] = row[0]
            record["scheduler_events"] = connection.execute(
                "SELECT event_type, error_code, count(*) FROM market_data_work_events "
                "GROUP BY event_type, error_code ORDER BY event_type, error_code"
            ).fetchall()
        record.update(
            completion_s=completion,
            windows=len(marks),
            steady_windows=len(steady),
            windows_per_second=len(steady) / 300,
            idle_ready_slot_seconds=idle,
            full_windows_per_second=len(marks) / completion,
            completion_times_s=marks,
            http_requests=server.requests,
            http_connections=len(server.connections),
            http_peak=server.peak,
            http_rejected=server.rejected,
            storage_calls=storage.calls,
            inserted_rows=storage.rows,
            maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_maxrss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            cpu_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        )
        record["canonical_new"] = admin.command(
            f"SELECT count() FROM {schema}.canonical_candles_1m "
            "WHERE ts_open >= toDateTime('2026-01-03 00:00:00')"
        )
        admin.command("SYSTEM FLUSH LOGS")
        record["query_metrics"] = admin.query(
            "SELECT type,count(),sum(read_rows),sum(read_bytes),max(memory_usage) "
            "FROM system.query_log WHERE log_comment=%(tag)s AND has(databases,%(db)s) "
            "GROUP BY type",
            parameters={"tag": tag, "db": schema},
        ).result_rows
        record["post_drain_memory"] = memory(admin)
        record["container_after"] = subprocess.check_output(
            [
                "docker",
                "exec",
                "roehub-s03-clickhouse-1",
                "sh",
                "-c",
                "cat /sys/fs/cgroup/memory.peak /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/io.stat",
            ],
            text=True,
        )
        with psycopg.connect(pgdsn) as connection:
            record["pg_stats_after"] = connection.execute(
                'SELECT xact_commit,xact_rollback,numbackends FROM pg_stat_database '
                'WHERE datname=current_database()'
            ).fetchone()
        record["checks"] = {
            "all_windows": len(marks) == 102 * args.windows,
            "canonical_count": record["canonical_new"] == 102 * args.windows * 1440,
            "required_window": completion >= 360,
            "no_permit_leak": record["permits_after_drain"] == 0,
            "sampling_complete": (not record["failures"]
                                  and len(record["samples"]) >= int(completion / 3)),
        }
        assert all(record["checks"].values()), record["checks"]
    except BaseException as error:
        record["failures"].append(type(error).__name__)
        raise
    finally:
        finished.set()
        if sampler is not None:
            sampler.join()
        if child is not None and child.is_alive():
            child.kill()
            child.join()
        if http is not None and hasattr(http, "close"):
            http.close()
        server.shutdown()
        server.server_close()
        serving.join()
        if client is not None:
            client.close()
        if created_ch:
            admin.command(f"DROP DATABASE IF EXISTS {schema} SYNC")
        admin.close()
        if created_pg:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )
        args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
