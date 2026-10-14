"""Explicit expected-fault phase on the same full S09 PG; never installed runtime."""

import json
import threading
import time
from pathlib import Path

from tools.qa.postgres_production_soak import APP_DSN, DSN, OUT
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)
from trading.platform.postgres_budget import BudgetExhausted, connect

root = Path("/run/roehub-pg-budget")
assert all(
    not p.stat().st_size
    for kind in ("market", "control", "admin")
    for p in root.glob(kind + "-*.json")
)
proof = {}
connections = []
try:
    for kind, count in [("market", 16), ("control", 24), ("admin", 8)]:
        for _ in range(count):
            connections.append(
                connect(DSN if kind == "admin" else APP_DSN, budget_class=kind, autocommit=True)
            )
    observer = connections[-1]
    actual = observer.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend'"
    ).fetchone()[0]
    assert actual == 48
    waits = {}
    for kind in ("market", "control", "admin"):
        start = time.monotonic()
        try:
            connect(DSN, budget_class=kind)
        except BudgetExhausted:
            pass
        else:
            raise AssertionError("physical overflow accepted")
        waits[kind] = time.monotonic() - start
        assert waits[kind] < 3
    observer.execute("CREATE TEMP TABLE s09_admin_headroom(n int)")
    observer.execute("INSERT INTO s09_admin_headroom VALUES(1)")
    assert observer.execute("SELECT count(*) FROM s09_admin_headroom").fetchone()[0] == 1
    proof["physical"] = {
        "connections": actual,
        "classes": {"market": 16, "control": 24, "admin": 8},
        "overflow_wait_s": waits,
        "admin_dml_at_saturation": True,
    }
finally:
    for c in reversed(connections):
        c.close()
g = IngestionGateway(APP_DSN, size=1, max_waiters=0)
entered = threading.Event()
release = threading.Event()
result = []


def hold(c):
    entered.set()
    assert release.wait(2)
    return c.execute("SELECT count(*) FROM market_data.ref_market").fetchone()[0]


def worker():
    result.append(g.run(hold))


t = threading.Thread(target=worker)
t.start()
assert entered.wait(2)
try:
    try:
        g.run(lambda c: c.execute("SELECT count(*) FROM market_data.ref_market"))
    except TemporaryIngestionError as e:
        assert e.code == "postgres_admission_full"
    else:
        raise AssertionError("admission bypass")
finally:
    release.set()
    t.join(3)
    g.close()
assert len(result) == 1
proof["admission"] = {"bounded_rejection": True, "active_request_completed": True}
# Actual PG backend cancellation, then a fresh production transaction.
g = IngestionGateway(APP_DSN, deadline_s=5)
entered = threading.Event()
pids = []
errors = []


def slow(c):
    pids.append(c.info.backend_pid)
    entered.set()
    c.execute("SELECT pg_sleep(10)")


def cancelled():
    try:
        g.run(slow)
    except Exception as e:
        errors.append({"type": type(e).__name__, "sqlstate": getattr(e, "sqlstate", None)})


t = threading.Thread(target=cancelled)
t.start()
assert entered.wait(2)
with connect(DSN, budget_class="admin", autocommit=True) as admin:
    assert admin.execute("SELECT pg_cancel_backend(%s)", (pids[0],)).fetchone()[0]
t.join(6)
assert not t.is_alive() and errors and errors[0]["sqlstate"] == "57014"
assert (
    g.run(
        lambda c: c.execute("SELECT count(*) FROM market_data.candle_coverage_summary").fetchone()[
            0
        ]
    )
    >= 102
)
g.close()
proof["cancellation"] = {"expected_errors": errors, "fresh_transaction_succeeded": True}
proof["quarantine"] = {
    k: sum(p.stat().st_size > 0 for p in root.glob(k + "-*.json"))
    for k in ("market", "control", "admin")
}
(OUT / "faults.json").write_text(json.dumps(proof, indent=2))
print(json.dumps(proof))
