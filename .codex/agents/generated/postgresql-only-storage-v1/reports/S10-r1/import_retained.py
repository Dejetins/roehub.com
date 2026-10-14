"""Bounded production ingestion of hash-verified selected history; not live evidence."""

import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from tools.qa.postgres_production_soak import candle
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.platform.postgres_budget import connect

source = Path("/work/.local_artifacts/pg-comparison-20261013")
snapshot = json.loads((source / "snapshot.json").read_text())
selected = {(1, "0GUSDT"), (1, "AAPLBUSDT"), (1, "AAVEUSDT"), (1, "ACTUSDT"), (2, "0GUSDT")}
gateway = IngestionGateway(os.environ["STRATEGY_PG_DSN"])
writer = PostgresRawKlineWriter(gateway)
total = 0
started = time.monotonic()
try:
    with connect(
        os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin", autocommit=True
    ) as db:
        actual = set(
            db.execute(
                "SELECT DISTINCT market_id,symbol FROM market_data_instrument_selections"
            ).fetchall()
        )
        assert actual == selected
        for part in snapshot["parts"]:
            path = source / "candles" / part["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            table = pq.read_table(
                path, filters=[("symbol", "in", ["0GUSDT", "AAPLBUSDT", "AAVEUSDT", "ACTUSDT"])]
            )
            mask = pc.or_(
                pc.equal(table["market_id"], 1),
                pc.and_(pc.equal(table["market_id"], 2), pc.equal(table["symbol"], "0GUSDT")),
            )
            table = table.filter(mask)
            if not table.num_rows:
                continue
            month = path.stem
            deadline = time.monotonic() + 600
            while (
                db.execute(
                    "SELECT to_regclass(%s)", ("market_data.raw_binance_klines_1m_" + month,)
                ).fetchone()[0]
                is None
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError("regular_maintenance_partition_not_ready:" + month)
                time.sleep(3)
            for batch in table.to_batches(max_chunksize=2000):
                writer.write_1m(candle(row) for row in batch.to_pylist())
            total += table.num_rows
            print(
                json.dumps(
                    {
                        "month": month,
                        "rows": table.num_rows,
                        "cumulative_rows": total,
                        "source_sha256": part["sha256"],
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
finally:
    gateway.close()
print(json.dumps({"completed": True, "rows": total, "live_proof": False}), flush=True)
