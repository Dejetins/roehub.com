"""Bounded finite source-sweep replay on a disposable ClickHouse database.

Uses current CH_* configuration; never mutates its configured application database.
The baseline is a preserved dirty-source snapshot, not HEAD. Query IDs, aggregate
metrics and hashes are retained; SQL, credentials and provider payloads are not.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client

MODULES = {
    "reconciler": (
        "src/trading/contexts/backtest_artifacts/application/services/"
        "base_source_reconciliation.py",
        "BaseSourceReconciler",
    ),
    "stored": (
        "src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/"
        "clickhouse/stored_coordinates.py",
        "ClickHouseStoredArtifactCoordinates",
    ),
    "index": (
        "src/trading/contexts/market_data/adapters/outbound/persistence/"
        "clickhouse/canonical_candle_index_reader.py",
        "ClickHouseCanonicalCandleIndexReader",
    ),
}
CUTOFF = datetime(2026, 1, 3, tzinfo=UTC)
BASE_MINUTE = 29453760  # 2026-01-01 UTC


def classes(root, label):
    result, hashes = {}, {}
    for name, (relative, symbol) in MODULES.items():
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(f"replay_{label}_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name] = getattr(module, symbol)
    return result, hashes


class MeasuredGateway:
    def __init__(self, client, tag):
        self.client, self.tag = client, tag
        self.ids = []

    def select(self, query, parameters):
        result = self.client.query(query, parameters=parameters, settings={"log_comment": self.tag})
        self.ids.append(result.query_id)
        return list(result.named_results())


class Observations:
    def __init__(self):
        self.rows = []

    def observe_source(self, *, coordinate, source, **kwargs):
        self.rows.append(
            (
                coordinate.exchange,
                coordinate.market_type,
                coordinate.symbol,
                source.history.start_utc.isoformat(),
                source.history.end_utc.isoformat(),
            )
        )


def memory(client):
    return dict(
        client.query(
            "SELECT metric,value FROM system.asynchronous_metrics WHERE metric IN "
            "('jemalloc.allocated','MemoryResident')"
        ).result_rows
    )


def sweep(client, db, impl, label, page_delay=0):
    tag = "resource_s01_" + uuid4().hex
    gateway, sink = MeasuredGateway(client, tag), Observations()
    service = impl["reconciler"](
        UUID(int=1), impl["index"](gateway, db), impl["stored"](gateway, db), sink
    )
    start_memory = memory(client)
    start, cpu = time.perf_counter(), time.process_time()
    cursor, pages = None, 0
    while True:
        cursor = service.reconcile_page(after=cursor, now=CUTOFF, limit=50)
        pages += 1
        if cursor is None:
            break
        if page_delay:
            time.sleep(page_delay)
        if pages > 100:
            raise RuntimeError("unbounded_sweep")
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    end_memory = memory(client)
    # Wait for configured ten-second query log flush; do not force live server flushes.
    time.sleep(11)
    stats = list(
        client.query(
            "SELECT count() AS queries,sum(read_rows) AS read_rows, "
            "sum(read_bytes) AS read_bytes,max(memory_usage) AS max_query_memory, "
            "sum(query_duration_ms) AS server_ms, "
            "sum(ProfileEvents['UserTimeMicroseconds']) AS cpu_us, "
            "sum(ProfileEvents['ReadBufferFromFileDescriptorReadBytes']) AS file_read_bytes, "
            "countIf(has(tables,%(canonical)s)) AS canonical_queries "
            "FROM system.query_log WHERE event_date>=today()-1 AND type='QueryFinish' "
            "AND log_comment=%(tag)s",
            parameters={"tag": tag, "canonical": db + ".canonical_candles_1m"},
        ).named_results()
    )[0]
    if stats["queries"] != len(gateway.ids):
        raise RuntimeError("incomplete_query_log")
    return dict(
        label=label,
        pages=pages,
        coordinates=len(sink.rows),
        elapsed_s=elapsed,
        client_cpu_s=cpu_elapsed,
        client_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        db_memory_before=start_memory,
        db_memory_after=end_memory,
        query_ids=gateway.ids,
        observations_sha256=hashlib.sha256(json.dumps(sink.rows).encode()).hexdigest(),
        **stats,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "compare", "index-only"], default="compare")
    parser.add_argument("--coordinates", type=int, default=102)
    parser.add_argument("--page-delay", type=float, default=0)
    args = parser.parse_args()
    if not 1 <= args.coordinates <= 1020:
        raise ValueError("bounded_fixture_required")
    base, base_hash = classes(args.baseline, "baseline")
    candidate, candidate_hash = classes(Path.cwd(), "candidate")
    client = _clickhouse_client(ClickHouseSettingsLoader(os.environ).load())
    db = "resource_s01_" + uuid4().hex
    record = dict(
        schema_version="market-data-resource-comparison/v1",
        stage="S01",
        mode="finite-replay",
        cache="warm; one complete sweep per implementation",
        clock="fixed closed cutoff",
        cutoff=CUTOFF.isoformat(),
        coordinates=args.coordinates,
        live_streams=4,
        page_limit=50,
        publisher_cadence_s=30,
        cadence_simulation="finite replay with configured page delay",
        measured_page_delay_s=args.page_delay,
        baseline=base_hash,
        candidate=candidate_hash,
        python=platform.python_version(),
        hardware=platform.platform(),
        db_version=client.query("SELECT version()").first_item,
        runs=[],
        failures=[],
    )
    client.command(f"CREATE DATABASE {db}")
    try:
        client.command(
            f"CREATE TABLE {db}.canonical_candles_1m (market_id UInt16, symbol String, "
            "ts_open DateTime64(3,'UTC')) ENGINE=MergeTree ORDER BY (market_id,symbol,ts_open)"
        )
        client.command(
            f"CREATE TABLE {db}.canonical_candles_1m_stats (market_id UInt16,symbol String, "
            "rows_state AggregateFunction(sum,UInt64), "
            "first_ts_state AggregateFunction(min,DateTime64(3,'UTC')), "
            "last_ts_state AggregateFunction(max,DateTime64(3,'UTC')), "
            "minute_bitmap_state AggregateFunction(groupBitmap,UInt32)) "
            "ENGINE=AggregatingMergeTree ORDER BY (market_id,symbol)"
        )
        client.command(f"SYSTEM STOP MERGES {db}.canonical_candles_1m")
        client.command(f"SYSTEM STOP MERGES {db}.canonical_candles_1m_stats")
        record["fixture_merges"] = "disabled only on disposable fixture tables"
        record["settings"] = {
            str(row[0]): str(row[1])
            for row in client.query(
                "SELECT name,value FROM system.settings WHERE name IN "
                "('max_threads','max_memory_usage','use_query_condition_cache')"
            ).result_rows
        }
        # Four contiguous streams, remaining histories sparse; duplicate state inserts below.
        seed = (
            f"INSERT INTO {db}.canonical_candles_1m SELECT toUInt16(1), "
            "concat('S',leftPad(toString(intDiv(number,2880)),4,'0'),'USDT'), "
            f"toDateTime64(({BASE_MINUTE}+number%2880)*60,3,'UTC') "
            f"FROM numbers({args.coordinates*2880}) "
            "WHERE intDiv(number,2880)<4 OR number%7!=0"
        )
        record["fixture_sha256"] = hashlib.sha256(seed.replace(db, "FIXTURE").encode()).hexdigest()
        client.command(seed)
        index_sql = (
            f"INSERT INTO {db}.canonical_candles_1m_stats SELECT market_id,symbol, "
            "sumState(toUInt64(1)),minState(ts_open),maxState(ts_open), "
            "groupBitmapState(toUInt32(intDiv(toUnixTimestamp(ts_open),60))) "
            f"FROM {db}.canonical_candles_1m GROUP BY market_id,symbol"
        )
        client.command(index_sql)
        client.command(index_sql)
        record["fixture_rows"] = client.query(
            f"SELECT count() FROM {db}.canonical_candles_1m"
        ).first_item
        if args.mode == "index-only":
            # Preserve the same bitmap semantics, then make canonical fallback impossible.
            client.command(f"DROP TABLE {db}.canonical_candles_1m")
            record["canonical_table"] = "absent during all measured index-only reads"
            record["warmups"] = [sweep(client, db, candidate, "candidate_warmup")]
        else:
            record["warmups"] = [sweep(client, db, base, "baseline_warmup", args.page_delay)]
        if args.mode == "compare":
            record["warmups"].append(
                sweep(client, db, candidate, "candidate_warmup", args.page_delay)
            )
        for repetition in range(3):
            order = (
                [("candidate", candidate)]
                if args.mode == "index-only"
                else [("baseline", base)]
                if args.mode == "baseline"
                else (
                    [("baseline", base), ("candidate", candidate)]
                    if repetition % 2 == 0
                    else [("candidate", candidate), ("baseline", base)]
                )
            )
            for label, impl in order:
                run = sweep(client, db, impl, label, args.page_delay)
                run["repetition"] = repetition + 1
                record["runs"].append(run)
        assert len({r["observations_sha256"] for r in record["runs"]}) == 1
        expected = []
        for coordinate in range(args.coordinates):
            minutes = [
                minute
                for minute in range(2880)
                if coordinate < 4 or (coordinate * 2880 + minute) % 7 != 0
            ]
            expected.append(
                (
                    "binance",
                    "spot",
                    f"S{coordinate:04d}USDT",
                    datetime.fromtimestamp((BASE_MINUTE + minutes[0]) * 60, UTC).isoformat(),
                    datetime.fromtimestamp((BASE_MINUTE + minutes[-1] + 1) * 60, UTC).isoformat(),
                )
            )
        record["expected_observations_sha256"] = hashlib.sha256(
            json.dumps(expected).encode()
        ).hexdigest()
        assert all(
            r["observations_sha256"] == record["expected_observations_sha256"]
            for r in record["runs"]
        )
        assert all(r["coordinates"] == args.coordinates for r in record["runs"])
    except Exception as error:
        record["failures"].append(type(error).__name__)
        raise
    finally:
        client.command(f"DROP DATABASE {db}")
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(record, f, indent=2, default=str)
            f.write("\n")


if __name__ == "__main__":
    main()
