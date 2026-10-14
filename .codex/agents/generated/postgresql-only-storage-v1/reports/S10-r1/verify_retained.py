"""Stream independent canonical SQL and compare all retained selected provider columns."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from psycopg import sql

from tools.qa.postgres_production_hash import CanonicalHasher
from tools.qa.postgres_production_verify import normalized
from trading.platform.postgres_budget import connect

root = Path("/work/.local_artifacts/pg-comparison-20261013")
snapshot = json.loads((root / "snapshot.json").read_text())
cutoff = datetime.fromisoformat(snapshot["cutoff_exclusive_utc"].replace("Z", "+00:00"))
total = 0
with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    for part in snapshot["parts"]:
        path = root / "candles" / part["path"]
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
        table = table.sort_by(
            [("market_id", "ascending"), ("symbol", "ascending"), ("ts_open", "ascending")]
        )
        expected = normalized(table)
        a = CanonicalHasher()
        b = CanonicalHasher()
        for batch in expected.to_batches(max_chunksize=2000):
            a.update_batch(batch)
        start = datetime(int(path.stem[:4]), int(path.stem[4:]), 1, tzinfo=timezone.utc)
        end = min(
            datetime(
                start.year + int(start.month == 12),
                1 if start.month == 12 else start.month + 1,
                1,
                tzinfo=timezone.utc,
            ),
            cutoff,
        )
        query = sql.SQL(
            "SELECT {} FROM market_data.canonical_candles_1m WHERE ts_open>=%s AND ts_open<%s AND ((market_id=1 AND symbol=ANY(%s)) OR (market_id=2 AND symbol='0GUSDT')) ORDER BY market_id,symbol,ts_open"
        ).format(sql.SQL(",").join(map(sql.Identifier, table.column_names)))
        n = 0
        with db.cursor(name="retained_verify") as cursor:
            cursor.execute(query, (start, end, ["0GUSDT", "AAPLBUSDT", "AAVEUSDT", "ACTUSDT"]))
            while rows := cursor.fetchmany(2000):
                batch = pa.Table.from_arrays(
                    [
                        pa.array(
                            [str(r[j]) if isinstance(r[j], UUID) else r[j] for r in rows],
                            type=f.type,
                        )
                        for j, f in enumerate(table.schema)
                    ],
                    schema=table.schema,
                )
                for chunk in normalized(batch).to_batches():
                    b.update_batch(chunk)
                n += len(rows)
        assert n == table.num_rows, (path.stem, n, table.num_rows)
        assert a.hexdigest() == b.hexdigest(), path.stem
        total += n
        print(
            json.dumps(
                {
                    "month": path.stem,
                    "rows": n,
                    "all_columns_exact": True,
                    "canonical_sha256": a.hexdigest(),
                }
            ),
            flush=True,
        )
print(
    json.dumps(
        {"completed": True, "rows": total, "cutoff": cutoff.isoformat(), "live_proof": False}
    )
)
