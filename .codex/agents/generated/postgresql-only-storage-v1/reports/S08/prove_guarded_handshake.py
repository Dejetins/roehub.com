"""Linux production reservation + actual blackhole, no PostgreSQL authentication mock."""

import importlib.util
import json
import os
import socket
import threading
import time
from pathlib import Path

from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)
from trading.platform.postgres_budget import LIMITS, connect

root = Path(os.environ["ROEHUB_PG_BUDGET_ROOT"])
assert not list(root.iterdir()), "fresh owned registry required"
root.chmod(0o755)
(root / "authority.json").touch(mode=0o600)
for kind, count in LIMITS.items():
    for index in range(count):
        (root / f"{kind}-{index}.json").touch(mode=0o600)
assert importlib.util.find_spec("clickhouse_connect") is None
# Warm and close real admitted PG connections, proving unchanged ordinary transport.
for _ in range(5):
    c = connect("host=127.0.0.1 port=5432 dbname=postgres user=postgres", budget_class="admin")
    assert c.execute("SHOW max_connections").fetchone()[0] == "48"
    c.close()
assert all(not p.stat().st_size for p in root.glob("admin-*.json"))
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(1)
release = threading.Event()


def blackhole():
    c, _ = listener.accept()
    try:
        release.wait(4)
    finally:
        c.close()


thread = threading.Thread(target=blackhole)
thread.start()
g = IngestionGateway(
    f"host=127.0.0.1 port={listener.getsockname()[1]} user=fixture dbname=fixture sslmode=disable",
    deadline_s=0.2,
)
started = time.monotonic()
try:
    try:
        g.run(lambda c: c.execute("SELECT 1"))
    except TemporaryIngestionError as e:
        assert e.code == "postgres_connect_timeout"
    else:
        raise AssertionError("handshake unexpectedly succeeded")
    elapsed = time.monotonic() - started
    # 0.2s connect deadline plus the existing bounded 1s FIN-witness cleanup.
    assert elapsed < 1.6
finally:
    g.close()
    release.set()
    listener.close()
    thread.join(5)
assert not thread.is_alive()
quarantined = sum(p.stat().st_size > 0 for p in root.glob("market-*.json"))
assert quarantined == 1  # Unknown peer result MUST NOT reclaim the slot.
print(
    json.dumps(
        {
            "status": "passed",
            "driver_absent": True,
            "classes": LIMITS,
            "deadline_s": 0.2,
            "elapsed_including_fin_witness_cleanup_s": elapsed,
            "cleanup_allowance_s": 1,
            "error": "postgres_connect_timeout",
            "unknown_transport_quarantined": quarantined,
            "normal_pg_closes": 5,
            "server_max_connections": 48,
        }
    )
)
