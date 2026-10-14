"""Disposable real backend death preserves unknown physical reservations."""

import json
import os
import subprocess
import sys
from pathlib import Path

from trading.platform.postgres_budget import connect, reservation_counts

root = Path("/tmp/s10r1-crash-budget")
root.mkdir(mode=0o755)
root.chmod(0o755)
for name in ["authority.json"] + [
    f"{k}-{i}.json" for k, n in [("market", 16), ("control", 24), ("admin", 8)] for i in range(n)
]:
    (root / name).touch(mode=0o600)
os.environ["ROEHUB_PG_BUDGET_ROOT"] = str(root)
os.environ["ROEHUB_PG_BUDGET_REQUIRED"] = "1"
dsn = "host=roehub-s10r1-test-pg dbname=postgres user=postgres sslmode=disable"
code = (
    "import os;from trading.platform.postgres_budget import connect;c=connect("
    + repr(dsn)
    + ',budget_class="market");c.execute("SELECT 1");os._exit(7)'
)
r = subprocess.run([sys.executable, "-c", code])
assert r.returncode == 7
counts = reservation_counts()
assert counts["market"]["quarantined"] == 1
with connect(dsn, budget_class="admin") as db:
    n = db.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE 'rh-market-%'"
    ).fetchone()[0]
assert n == 0
assert reservation_counts()["market"]["quarantined"] == 1
print(
    json.dumps(
        {
            "child_exit": 7,
            "remote_backend_count_after_exit": n,
            "unknown_reservation_retained": True,
            "counts": counts,
            "target": "roehub-s10r1-test-pg",
            "installed_registry_used": False,
        }
    )
)
