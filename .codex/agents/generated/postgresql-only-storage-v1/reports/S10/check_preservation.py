"""Compare pre-cutover user/control table digests without exposing any rows."""

import json
import os
from pathlib import Path

from psycopg import sql

from trading.platform.postgres_budget import connect

baseline = json.loads(
    Path(
        "/work/.codex/agents/generated/postgresql-only-storage-v1/reports/S10/preserved-public-baseline.json"
    ).read_text()
)
results = {}
with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    for table, prior in baseline.items():
        query = sql.SQL(
            "SELECT count(*),md5(coalesce(string_agg(row_to_json(t)::text,E'\\n' ORDER BY row_to_json(t)::text COLLATE \"C\"),'')) FROM {} t"  # noqa: E501
        ).format(sql.Identifier(table))
        n, h = db.execute(query).fetchone()
        results[table] = {
            "unchanged": prior == {"count": n, "md5": h},
            "before_count": prior["count"],
            "after_count": n,
        }
print(json.dumps(results, indent=2))
