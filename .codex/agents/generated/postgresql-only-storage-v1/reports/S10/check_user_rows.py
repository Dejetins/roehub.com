"""Check preserved old rows despite authorized login and additive proof commands."""

import json
import os

import psycopg
from psycopg import sql

from trading.platform.postgres_budget import connect

names = [
    "identity_users",
    "backtest_artifact_policies",
    "backtest_artifact_profiles",
    "backtest_jobs",
    "backtest_job_variants",
    "strategy_strategies",
]
result = {}
with (
    psycopg.connect("host=roehub-s10-recovery-verify dbname=restore user=postgres") as old,
    connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as current,
):
    for name in names:
        columns = old.execute(
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey) WHERE i.indrelid=to_regclass(%s) AND i.indisprimary ORDER BY a.attnum",  # noqa: E501
            (name,),
        ).fetchall()
        if not columns:
            continue
        keys = [r[0] for r in columns]
        query = sql.SQL("SELECT row_to_json(t) FROM {} t").format(sql.Identifier(name))
        before = old.execute(query).fetchall()
        after = current.execute(query).fetchall()
        index = {tuple(str(row[k]) for k in keys): row for (row,) in after}
        changed = {}
        missing = 0
        for (row,) in before:
            other = index.get(tuple(str(row[k]) for k in keys))
            if other is None:
                missing += 1
                continue
            for key, value in row.items():
                if value != other.get(key):
                    changed[key] = changed.get(key, 0) + 1
        result[name] = {
            "old_rows": len(before),
            "old_rows_missing": missing,
            "changed_columns": changed,
            "new_rows": len(after) - len(before),
        }
print(json.dumps(result, indent=2))
