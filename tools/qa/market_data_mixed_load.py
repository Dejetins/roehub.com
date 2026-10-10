"""S02 fixed offered mixed load on an explicitly disposable ClickHouse server.

Extends market-data-resource-comparison/v1; uses the accepted S01 source reader.
Each lane owns a client in a separate process. No work is discarded: every scheduled
operation is retried or reported as failed, and finite queues drain after measurement.
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
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import Request

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from apps.migrations.storage import split_clickhouse_sql
from tools.qa.market_data_resource_comparison import BASE_MINUTE, CUTOFF, MeasuredGateway, classes
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.catalog_coverage_reader import (  # noqa: E501
    ClickHouseCatalogCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.work_errors import (
    transient_clickhouse_error,
)
from trading.shared_kernel.primitives import UtcTimestamp

LANES = (
    ("stream", 0, 1.0),
    ("interactive", 0, 0.5),
    ("backfill", 0, 1.0),
    ("backfill", 1, 1.0),
    ("artifacts", 0, 1.0),
    ("artifacts", 1, 1.0),
)


def api_surface(coverage):
    """Real catalog HTTP/ASGI route and CH reader; fixed auth/catalog/PG projections."""
    from types import SimpleNamespace as N
    from uuid import UUID

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from apps.api.routes.market_data_catalog import build_catalog_workspace_router
    from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
    from trading.shared_kernel.primitives import MarketId, OrganizationId, PaidLevel, UserId

    app = FastAPI()
    principal = CurrentUserPrincipal(UserId(UUID(int=2)), PaidLevel.free())

    def current(request: Request):
        return principal

    app.include_router(
        build_catalog_workspace_router(
            current_user_dependency=current,
            scope_resolver=cast(
                Any, N(resolve=lambda **kw: N(organization_id=OrganizationId(UUID(int=1))))
            ),
            selections=cast(
                Any, N(list_for_organization=lambda **kw: [], strategy_pins=lambda **kw: [])
            ),
            snapshots=cast(
                Any,
                N(
                    current_catalogs=lambda **kw: [
                        dict(
                            snapshot_id=UUID(int=3),
                            market_id=1,
                            refreshed_at=CUTOFF,
                            items=[
                                dict(
                                    symbol=f"S{i:04d}USDT",
                                    base_asset="S",
                                    quote_asset="USDT",
                                    price_step=0.01,
                                    qty_step=0.01,
                                    min_notional=5,
                                )
                                for i in range(102)
                            ],
                        )
                    ]
                ),
            ),
            jobs=cast(Any, N(instrument_jobs=lambda **kw: [])),
            markets=cast(
                Any,
                N(
                    execute=lambda: [
                        N(
                            market_id=MarketId(1),
                            exchange_name="binance",
                            market_type="spot",
                            market_code="binance:spot",
                        )
                    ]
                ),
            ),
            coverage=coverage,
        )
    )
    return TestClient(app)


def raw_insert(client, db, symbol, minute, count, tag, *, sparse=False):
    client.command(
        f"INSERT INTO {db}.raw_binance_klines_1m "
        "(market_id,symbol,instrument_key,open_time,close_time,open,high,low,close,volume,"
        "quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,"
        "taker_buy_quote_asset_volume,source) SELECT 1,%(symbol)s,concat('binance:spot:',"
        "%(symbol)s),toDateTime64((%(minute)s+number)*60,3,'UTC'),"
        "toDateTime64((%(minute)s+number+1)*60,3,'UTC'),1,1,1,1,1,1,1,1,1,'rest' "
        "FROM numbers(%(count)s) WHERE %(sparse)s=0 OR modulo(number,7)!=0",
        parameters=dict(symbol=symbol, minute=minute, count=count, sparse=int(sparse)),
        settings={"log_comment": tag},
    )


def lane(environ, db, label, workload, index, interval, start, duration, root, output):
    try:
        if label == "candidate":
            environ["ROEHUB_CH_WORKLOADS_REQUIRED"] = "1"
            settings = ClickHouseSettingsLoader(environ).load(workload=workload)
            client = _clickhouse_client(settings)
        else:
            spec = importlib.util.spec_from_file_location(
                "baseline_ch_wiring", Path(root) / "apps/cli/wiring/db/clickhouse.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            client = module._clickhouse_client(module.ClickHouseSettingsLoader(environ).load())
        tag = f"s02_{label}_{workload}_{index}"
        gateway = MeasuredGateway(client, tag)
        impl, _ = classes(Path(root), workload + str(index))
        stored = impl["stored"](gateway, db)
        coverage = ClickHouseCatalogCoverageReader(cast(Any, gateway), db)
        api = api_surface(coverage) if workload == "interactive" else None
        count = int(duration / interval)
        operations, failures, retries = [], [], []
        tag = f"s02_{label}_{workload}_{index}"
        for n in range(count):
            offered = start + n * interval
            time.sleep(max(0, offered - time.monotonic()))
            attempts = 0
            began = time.monotonic()
            while True:
                attempts += 1
                try:
                    if workload == "stream":
                        for s in range(4):
                            raw_insert(client, db, f"S{s:04d}USDT", BASE_MINUTE + 2880 + n, 1, tag)
                    elif workload == "interactive":
                        assert api is not None
                        response = api.get(
                            "/market-data/workspace/instruments",
                            params={
                                "exchange": "binance",
                                "start_at": (CUTOFF - timedelta(days=2)).isoformat(),
                                "end_at": CUTOFF.isoformat(),
                                "limit": 100,
                            },
                        )
                        if response.status_code != 200:
                            raise RuntimeError("api_response_" + str(response.status_code))
                    elif workload == "backfill":
                        # An exact bounded verification scan accompanies each 1,440-minute write.
                        # Amplify scan work with the same fixed 16-pass checksum in both variants.
                        client.query(
                            f"SELECT sum(cityHash64(symbol,ts_open,number)) FROM "
                            f"{db}.canonical_candles_1m CROSS JOIN numbers(16) AS passes "
                            f"WHERE ts_open < %(cutoff)s",
                            parameters={"cutoff": CUTOFF},
                            settings={"log_comment": tag},
                        )
                        raw_insert(
                            client,
                            db,
                            f"S{4+index:04d}USDT",
                            BASE_MINUTE + 2880 + n * 1440,
                            1440,
                            tag,
                        )
                    else:
                        stored.observation_page(before=UtcTimestamp(CUTOFF), limit=50, after=None)
                        client.query(
                            f"SELECT sum(close),count() FROM {db}.canonical_candles_1m "
                            "WHERE symbol=%(symbol)s AND ts_open < %(cutoff)s",
                            parameters={"symbol": f"S{4+index:04d}USDT", "cutoff": CUTOFF},
                            settings={"log_comment": tag},
                        )
                    break
                except Exception as error:
                    code = transient_clickhouse_error(error)
                    # 202 is measured on the baseline too before its classifier is extended.
                    if code is None and "Code: 202." in str(error):
                        code = "storage_concurrency_pressure"
                    retries.append(dict(operation=n, code=code or type(error).__name__))
                    if code is None or attempts >= 30:
                        failures.append(dict(operation=n, error=code or type(error).__name__))
                        break
                    time.sleep(min(2, 0.1 * attempts) + index * 0.037)
            finished = time.monotonic()
            operations.append(
                dict(
                    n=n,
                    offered_s=offered - start,
                    finished_s=finished - start,
                    latency_s=finished - offered,
                    service_s=finished - began,
                    attempts=attempts,
                )
            )
        if api is not None:
            api.close()
        client.close()
        record = dict(
            workload=workload,
            index=index,
            offered=count,
            operations=operations,
            retries=retries,
            failures=failures,
            client_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            client_cpu_s=time.process_time(),
        )
    except Exception as error:
        record = dict(workload=workload, index=index, fatal=type(error).__name__)
    Path(output).write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container", default="roehub-s02-ch")
    parser.add_argument("--duration", type=int, default=360)
    args = parser.parse_args()
    if os.environ.get("ROEHUB_S02_DISPOSABLE") != "1":
        raise ValueError("explicit disposable server opt-in required")
    client = _clickhouse_client(ClickHouseSettingsLoader(os.environ).load())
    runtime = json.loads(
        subprocess.check_output(
            [
                "docker",
                "inspect",
                "--format",
                '{"image":{{json '
                '.Image}},"memory":{{.HostConfig.Memory}},"nano_cpus":{{.HostConfig.NanoCpus}}}',
                args.container,
            ],
            text=True,
        )
    )
    db = "resource_s02_" + uuid4().hex
    record = dict(
        schema_version="market-data-resource-comparison/v1",
        stage="S02",
        label=args.label,
        runtime=runtime,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        duration_s=args.duration,
        warmup_s=60,
        steady_s=300,
        source=classes(args.source, "identity")[1],
        python=platform.python_version(),
        fixture_coordinates=102,
        live_streams=4,
        server_version=client.command("SELECT version()"),
        samples=[],
        lanes=[],
        failures=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / (args.output.stem + "-lanes")
    work.mkdir(exist_ok=False)
    try:
        for sql in split_clickhouse_sql(
            Path("migrations/clickhouse/market_data_ddl.sql").read_text().replace("market_data", db)
        ):
            client.command(sql)
        for coordinate in range(102):
            # Seed a small representative history with actual missing minutes.
            raw_insert(
                client,
                db,
                f"S{coordinate:04d}USDT",
                BASE_MINUTE,
                2880,
                "s02_seed",
                sparse=coordinate >= 4,
            )
        record["fixture_rows"] = client.command(f"SELECT count() FROM {db}.canonical_candles_1m")
        record["container_limits"] = {
            "memory_bytes": 1610612736,
            "cpus": 2,
            "server_memory_bytes": 1073741824,
        }
        record["fixture_sha256"] = hashlib.sha256(
            b"102 coordinates;4x2880 contiguous;98x2880 missing every seventh;v2"
        ).hexdigest()
        # Workload privileges are database-specific on the disposable fixture.
        if args.label == "candidate":
            from tools.installation.clickhouse_workloads import grants

            for workload in ("stream", "interactive", "backfill", "artifacts"):
                for grant in grants(workload, db):
                    client.command(grant + " TO roehub_" + workload)
        ctx = mp.get_context("spawn")
        children = []
        start = time.monotonic() + 5
        for workload, index, interval in LANES:
            path = work / f"{workload}-{index}.json"
            process = ctx.Process(
                target=lane,
                args=(
                    dict(os.environ),
                    db,
                    args.label,
                    workload,
                    index,
                    interval,
                    start,
                    args.duration,
                    str(args.source),
                    str(path),
                ),
            )
            process.start()
            children.append((process, path))
        while any(p.is_alive() for p, _ in children):
            sample: dict[str, Any] = dict(at_s=time.monotonic() - start)
            sample["memory"] = {
                str(row[0]): row[1]
                for row in client.query(
                    "SELECT metric,value FROM system.asynchronous_metrics WHERE metric IN "
                    "('jemalloc.allocated','MemoryResident')"
                ).result_rows
            }
            sample["tracked"] = client.command(
                "SELECT value FROM system.metrics WHERE metric='MemoryTracking'"
            )
            sample["active"] = client.query(
                "SELECT user,count(),sum(memory_usage) FROM system.processes WHERE "
                "user!='s02_admin' GROUP BY user"
            ).result_rows
            sample["parts"] = client.command(
                f"SELECT count() FROM system.parts WHERE active AND database='{db}'"
            )
            sample["merges"] = client.command(
                f"SELECT count() FROM system.merges WHERE database='{db}'"
            )
            sample["cgroup"] = subprocess.check_output(
                [
                    "docker",
                    "exec",
                    args.container,
                    "sh",
                    "-c",
                    "cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/cpu.stat "
                    "/sys/fs/cgroup/io.stat",
                ],
                text=True,
            ).splitlines()
            record["samples"].append(sample)
            time.sleep(1)
            if time.monotonic() - start > args.duration + 600:
                raise RuntimeError("drain deadline exceeded")
        for p, path in children:
            p.join()
            record["lanes"].append(json.loads(path.read_text()))
        record["full_completion_s"] = time.monotonic() - start
        record["background_counts"] = client.query(
            f"SELECT symbol,uniqExact(ts_open) FROM {db}.canonical_candles_1m WHERE "
            f"ts_open >= %(cutoff)s AND symbol IN ('S0004USDT','S0005USDT') GROUP BY symbol",
            parameters={"cutoff": CUTOFF},
        ).result_rows
        record["live_counts"] = client.query(
            f"SELECT symbol,uniqExact(ts_open) FROM {db}.canonical_candles_1m WHERE "
            f"ts_open >= %(cutoff)s AND symbol IN "
            f"('S0000USDT','S0001USDT','S0002USDT','S0003USDT') GROUP BY symbol",
            parameters={"cutoff": CUTOFF},
        ).result_rows
        client.command("SYSTEM FLUSH LOGS")
        record["query_metrics"] = client.query(
            "SELECT "
            "user,log_comment,type,count(),sum(read_rows),sum(read_bytes),"
            "max(memory_usage),sum(ProfileEvents['UserTimeMicroseconds']),"
            "sum(ProfileEvents['ReadBufferFromFileDescriptorReadBytes']) FROM "
            "system.query_log WHERE event_time>=now()-INTERVAL 30 MINUTE AND "
            "has(databases,%(db)s) GROUP BY user,log_comment,type",
            parameters={"db": db},
        ).result_rows
        time.sleep(5)
        record["post_drain_memory"] = {
            str(row[0]): row[1]
            for row in client.query(
                "SELECT metric,value FROM system.asynchronous_metrics WHERE metric IN "
                "('jemalloc.allocated','MemoryResident')"
            ).result_rows
        }
    except Exception as error:
        record["failures"].append(type(error).__name__)
        raise
    finally:
        for p, _ in locals().get("children", []):
            if p.is_alive():
                p.terminate()
                p.join()
        client.command(f"DROP DATABASE IF EXISTS {db}")
        client.close()
        with args.output.open("x") as f:
            json.dump(record, f, indent=2, default=str)
            f.write("\n")


if __name__ == "__main__":
    main()
