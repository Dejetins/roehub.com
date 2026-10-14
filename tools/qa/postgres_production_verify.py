"""Independent full-value verification against immutable Parquet, outside timed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg import sql

from tools.qa.postgres_production_hash import CanonicalHasher, require_exact_millisecond_timestamps
from tools.qa.postgres_production_soak import APP_DSN, DSN, SOURCE, emit
from tools.qa.postgres_production_workload import (
    CUTOFF,
    array_hash,
    canonical_hash,
    instrument,
    span,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.application.dto import CanonicalCandleBatch1m
from trading.platform.postgres_budget import connect


def normalized(table):
    # Independent expected mapping of the accepted existing writer contract.
    values = table.to_pydict()
    values["volume_quote"] = [v if v is not None else 0.0 for v in values["volume_quote"]]
    for i, market in enumerate(values["market_id"]):
        if market <= 2:
            for field in ("trades_count", "taker_buy_volume_base", "taker_buy_volume_quote"):
                if values[field][i] is None:
                    values[field][i] = 0
        else:
            values["ts_close"][i] = values["ts_open"][i] + timedelta(minutes=1)
            for field in ("trades_count", "taker_buy_volume_base", "taker_buy_volume_quote"):
                values[field][i] = None
    return pa.Table.from_pydict(values, schema=table.schema)


def verify():
    parts = []
    total = 0
    keys = set()
    catalog = {}
    snapshot = json.loads((SOURCE / "snapshot.json").read_text())
    expected_files = {p["path"]: p["sha256"] for p in snapshot["parts"]}
    with connect(DSN, budget_class="admin") as c:
        for path in sorted((SOURCE / "candles").glob("*.parquet")):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_files[path.name]
            source = normalized(pq.read_table(path)).sort_by(
                [("market_id", "ascending"), ("symbol", "ascending"), ("ts_open", "ascending")]
            )
            for r in source.select(["market_id", "symbol", "ts_open"]).to_pylist():
                key = (r["market_id"], r["symbol"])
                catalog.setdefault(key, 0)
                if CUTOFF - timedelta(days=7) <= r["ts_open"] < CUTOFF:
                    catalog[key] += 1
            start = datetime.strptime(path.stem, "%Y%m").replace(tzinfo=UTC)
            end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
            h_in = CanonicalHasher()
            h_out = CanonicalHasher()
            n = 0
            with c.cursor(name="s09_full_parity", binary=True) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT {} FROM market_data.canonical_candles_1m "
                        "WHERE ts_open>=%s AND ts_open<%s ORDER BY market_id,symbol,ts_open"
                    ).format(sql.SQL(",").join(map(sql.Identifier, source.column_names))),
                    (start, end),
                )
                for batch in source.to_batches(max_chunksize=4096):
                    rows = cursor.fetchmany(batch.num_rows)
                    assert len(rows) == batch.num_rows
                    require_exact_millisecond_timestamps(rows, source.schema)
                    observed = pa.RecordBatch.from_arrays(
                        [
                            pa.array(
                                [str(r[j]) if isinstance(r[j], UUID) else r[j] for r in rows],
                                type=f.type,
                            )
                            for j, f in enumerate(source.schema)
                        ],
                        schema=source.schema,
                    )
                    assert batch.equals(observed), (path.name, n)
                    h_in.update_batch(batch)
                    h_out.update_batch(observed)
                    keys.update((r[0], r[1]) for r in rows)
                    n += len(rows)
                assert not cursor.fetchmany(1)
            assert h_in.hexdigest() == h_out.hexdigest()
            total += n
            parts.append(
                {
                    "part": path.name,
                    "rows": n,
                    "input_normalized_sha256": h_in.hexdigest(),
                    "output_sha256": h_out.hexdigest(),
                    "all_17_fields_equal": True,
                }
            )
            emit("parity-progress.json", {"rows": total, "parts": parts})
            print(path.name, n, total, flush=True)
    assert (total, len(keys)) == (15130115, 102)
    emit(
        "full-parity.json",
        {
            "rows": total,
            "instruments": len(keys),
            "parts": parts,
            "normalizations": (
                "existing accepted writer semantics: nullable Binance counters/quote ze"
                "ro; Bybit end=start+minute and nullable trade/taker fields"
            ),
        },
    )
    source = batch = observed = rows = None
    verify_arrays(catalog)


def verify_arrays(catalog=None):
    with connect(DSN, budget_class="admin") as c:
        target = c.execute(
            (
                "SELECT market_id,symbol,first_ts,last_ts,actual_count FROM market_data"
                ".candle_coverage_summary ORDER BY actual_count DESC LIMIT 1"
            )
        ).fetchone()
    assert target is not None
    market, symbol, first, last, count = target
    first, last = first.astimezone(UTC), last.astimezone(UTC)
    if catalog is None:
        catalog = {}
        for path in sorted((SOURCE / "candles").glob("*.parquet")):
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == {
                    p["path"]: p["sha256"]
                    for p in json.loads((SOURCE / "snapshot.json").read_text())["parts"]
                }[path.name]
            )
            keys = (
                pq.read_table(path, columns=["market_id", "symbol"])
                .group_by(["market_id", "symbol"])
                .aggregate([])
            )
            for row in keys.to_pylist():
                catalog.setdefault((row["market_id"], row["symbol"]), 0)
            recent = pq.read_table(
                path,
                columns=["market_id", "symbol", "ts_open"],
                filters=[("ts_open", ">=", CUTOFF - timedelta(days=7)), ("ts_open", "<", CUTOFF)],
            )
            for row in recent.to_pylist():
                catalog[(row["market_id"], row["symbol"])] += 1
    # Only bounded latest and seven-day timestamps need Python objects.
    latest = []
    present = set()
    for path in sorted((SOURCE / "candles").glob("*.parquet")):
        recent = pq.read_table(
            path,
            filters=[
                ("market_id", "=", market),
                ("symbol", "=", symbol),
                ("ts_open", ">=", CUTOFF - timedelta(days=7)),
                ("ts_open", "<", CUTOFF),
            ],
        )
        if recent.num_rows:
            recent = normalized(recent).sort_by([("ts_open", "ascending")])
            latest = (latest + recent.slice(max(0, recent.num_rows - 100)).to_pylist())[-100:]
            present.update(recent["ts_open"].to_pylist())
    cursor = CUTOFF - timedelta(days=7)
    gap = None
    while cursor < CUTOFF:
        if cursor not in present:
            end = cursor + timedelta(minutes=1)
            while end < CUTOFF and end.date() == cursor.date() and end not in present:
                end += timedelta(minutes=1)
            gap = [cursor.isoformat(), end.isoformat()]
            break
        cursor += timedelta(minutes=1)
    hashes = {}
    details = {}
    g = IngestionGateway(APP_DSN, workload="bulk", deadline_s=30)
    try:
        for name, start in [("year", CUTOFF - timedelta(days=365)), ("full", first)]:
            # Expected artifact arrays are spooled, not a full 17-field in-memory table.
            with (
                tempfile.TemporaryFile() as opens,
                tempfile.TemporaryFile() as closes,
                tempfile.TemporaryFile() as values,
            ):
                n = 0
                for path in sorted((SOURCE / "candles").glob("*.parquet")):
                    t = pq.read_table(
                        path,
                        columns=[
                            "ts_open",
                            "ts_close",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume_base",
                        ],
                        filters=[
                            ("market_id", "=", market),
                            ("symbol", "=", symbol),
                            ("ts_open", ">=", start),
                            ("ts_open", "<", CUTOFF),
                        ],
                    ).sort_by([("ts_open", "ascending")])
                    n += t.num_rows
                    ts = t["ts_open"].cast(pa.int64()).to_numpy()
                    ts.astype(np.int64).tofile(opens)
                    end_ts = ts + 60000 if market > 2 else t["ts_close"].cast(pa.int64()).to_numpy()
                    end_ts.astype(np.int64).tofile(closes)
                    np.column_stack(
                        [t[k].to_numpy() for k in ("open", "high", "low", "close", "volume_base")]
                    ).astype(np.float32).tofile(values)
                a = CanonicalCandleBatch1m(
                    np.memmap(opens, dtype=np.int64, mode="r", shape=(n,)),
                    np.memmap(closes, dtype=np.int64, mode="r", shape=(n,)),
                    np.memmap(values, dtype=np.float32, mode="r", shape=(n, 5)),
                )
            started = time.monotonic()
            print("array", name, "rows", a.row_count(), flush=True)
            b = PostgresCanonicalCandleReader(g).read_1m_arrays(
                instrument(symbol, market), span(start, CUTOFF)
            )
            elapsed = time.monotonic() - started
            for x, y in zip(
                (a.open_time_ms, a.close_time_ms, a.ohlcv_f32),
                (b.open_time_ms, b.close_time_ms, b.ohlcv_f32),
            ):
                np.testing.assert_array_equal(x, y)
            hashes[name] = array_hash(a)
            assert hashes[name] == array_hash(b)
            details[name] = {
                "rows": a.row_count(),
                "sha256": hashes[name],
                "every_value_equal": True,
                "adapter_seconds": elapsed,
            }
    finally:
        g.close()
    emit(
        "read-target.json",
        {
            "market": market,
            "symbol": symbol,
            "first": first.isoformat(),
            "last": last.isoformat(),
            "count": count,
            "array_hashes": hashes,
            "latest_sha256": canonical_hash(latest),
            "catalog_sha256": canonical_hash(catalog),
            "gap_expected": gap,
            "array_evidence": details,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays-only", action="store_true")
    args = parser.parse_args()
    verify_arrays() if args.arrays_only else verify()
