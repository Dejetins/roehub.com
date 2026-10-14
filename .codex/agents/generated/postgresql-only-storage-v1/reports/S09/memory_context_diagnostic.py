"""Post-soak causal diagnostic; privileged own-session inspection, not timed proof."""

import json
from datetime import UTC, datetime

from tools.qa.postgres_production_soak import DSN, OUT
from tools.qa.postgres_production_workload import PostgresRawKlineWriter, replay
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (  # noqa: E501
    IngestionGateway,
)

result = {}
for mode in ("same_partition", "expanding_partitions"):
    gateway = IngestionGateway(DSN, deadline_s=30)
    writer = PostgresRawKlineWriter(gateway)
    samples = []
    try:
        for i in range(8000):
            month = i % 42 if mode == "expanding_partitions" else 0
            date = datetime(2030 + month // 12, month % 12 + 1, 1, tzinfo=UTC)
            offset = int((date - datetime(2030, 1, 1, tzinfo=UTC)).total_seconds() / 60)
            writer.write_1m(replay("S09MEM5" + mode.upper().replace("_", ""), offset + i, 1))
            if i % 100 != 0 and i != 7999:
                continue
            contexts = gateway.run(
                lambda c: c.execute(
                    "SELECT name,sum(total_bytes),sum(used_bytes),count(*) "
                    "FROM pg_backend_memory_contexts GROUP BY name ORDER BY sum(total_bytes) DESC"
                ).fetchall()
            )
            samples.append(
                {
                    "iteration": i,
                    "month": month,
                    "pid": gateway.run(
                        lambda c: c.execute("SELECT pg_backend_pid()").fetchone()[0]
                    ),
                    "contexts": contexts,
                }
            )
    finally:
        gateway.close()
    result[mode] = samples
(OUT / "memory-context-diagnostic-production.json").write_text(
    json.dumps(result, indent=2, default=int) + "\n"
)
