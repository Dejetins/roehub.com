"""Apply current immutable phases and role grants with stopped old owners."""

import json
import os
from pathlib import Path

from apps.migrations.market_data import market_data_schema_ready, provision_current_partitions
from apps.migrations.storage import (
    _read_postgres_markers,
    apply_postgres_migrations,
    provision_runtime_roles,
)
from trading.platform.postgres_budget import connect

root = Path("/work")
dsn = os.environ["ROEHUB_STORAGE_POSTGRES_DSN"]
before = _read_postgres_markers(dsn)
apply_postgres_migrations(
    dsn, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
)
provision_runtime_roles(dsn)
with connect(dsn, budget_class="admin") as db:
    provision_current_partitions(db)
    assert market_data_schema_ready(db)
after = _read_postgres_markers(dsn)
assert all(after[k] == v for k, v in before.items())
print(
    json.dumps(
        {
            "before": len(before),
            "after": len(after),
            "new": [list(k) for k in after if k not in before],
            "prior_checksums_preserved": True,
            "schema_ready": True,
        }
    )
)
