"""Rebuild selected historical rows from immutable S09-validated Parquet, not live proof."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
from psycopg import sql

from tools.qa.postgres_production_hash import CanonicalHasher, require_exact_millisecond_timestamps
from tools.qa.postgres_production_soak import candle
from tools.qa.postgres_production_verify import normalized
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.platform.postgres_budget import connect

source = Path("/work/.local_artifacts/pg-comparison-20261013")
path = source / "candles/202610.parquet"
snapshot = json.loads((source / "snapshot.json").read_text())
expected = next(p["sha256"] for p in snapshot["parts"] if p["path"] == path.name)
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
start = datetime(2026, 10, 10, tzinfo=UTC)
end = datetime(2026, 10, 12, tzinfo=UTC)
with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    selected = {
        (r[0], r[1])
        for r in db.execute(
            "SELECT market_id,symbol FROM market_data_instrument_selections"
        ).fetchall()
    }
    assert selected == {
        (1, "0GUSDT"),
        (1, "AAPLBUSDT"),
        (1, "AAVEUSDT"),
        (1, "ACTUSDT"),
        (2, "0GUSDT"),
    }
    rows = pq.read_table(path, filters=[("ts_open", ">=", start), ("ts_open", "<", end)])
    rows = pa.Table.from_pylist(
        [r for r in rows.to_pylist() if (r["market_id"], r["symbol"]) in selected],
        schema=rows.schema,
    )
    rows = rows.sort_by(
        [("market_id", "ascending"), ("symbol", "ascending"), ("ts_open", "ascending")]
    )
    assert rows.num_rows == 14400
    gateway = IngestionGateway(os.environ["STRATEGY_PG_DSN"])
    try:
        writer = PostgresRawKlineWriter(gateway)
        for batch in rows.to_batches(max_chunksize=1000):
            writer.write_1m(candle(r) for r in batch.to_pylist())
    finally:
        gateway.close()
    expected_rows = normalized(rows)
    hi = CanonicalHasher()
    ho = CanonicalHasher()
    query = sql.SQL(
        "SELECT {} FROM market_data.canonical_candles_1m WHERE ts_open>=%s AND ts_open<%s ORDER BY market_id,symbol,ts_open"  # noqa: E501
    ).format(sql.SQL(",").join(map(sql.Identifier, rows.column_names)))
    observed = db.execute(query, (start, end)).fetchall()
    require_exact_millisecond_timestamps(observed, rows.schema)
    actual = pa.Table.from_arrays(
        [
            pa.array([str(r[j]) if isinstance(r[j], UUID) else r[j] for r in observed], type=f.type)
            for j, f in enumerate(rows.schema)
        ],
        schema=rows.schema,
    )
    assert expected_rows.equals(actual)
    for b in expected_rows.to_batches():
        hi.update_batch(b)
    for b in actual.to_batches():
        ho.update_batch(b)
    assert hi.hexdigest() == ho.hexdigest()
    summaries = db.execute(
        "SELECT market_id,symbol,actual_count,first_ts,last_ts,generation FROM market_data.candle_coverage_summary ORDER BY market_id,symbol"  # noqa: E501
    ).fetchall()
    print(
        json.dumps(
            {
                "source_sha256": expected,
                "rows": rows.num_rows,
                "start": start,
                "end_exclusive": end,
                "value_sha256": hi.hexdigest(),
                "all_columns_exact": True,
                "coverage": summaries,
                "live_provider_proof": False,
            },
            default=str,
        )
    )
