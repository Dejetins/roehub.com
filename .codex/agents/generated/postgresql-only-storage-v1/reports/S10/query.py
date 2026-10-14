"""Counted operator SQL with explicitly selected, non-secret result columns."""

import json
import os
import sys

from trading.platform.postgres_budget import connect

with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    cur = db.execute(sys.stdin.read())
    if cur.description:
        print(json.dumps(cur.fetchall(), default=str))
