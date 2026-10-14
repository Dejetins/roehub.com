"""Plans for the actual production read SQL and immutable S09 read target."""

import json
from datetime import datetime, timedelta

from tools.qa.postgres_production_soak import APP_DSN, OUT
from tools.qa.postgres_production_workload import CUTOFF
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    ARRAY_RANGE,
    CANDLE_RANGE,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
    COUNTS,
    FIRST_GAP,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
    minute,
    timestamp,
)
from trading.platform.postgres_budget import connect

x = json.loads((OUT / "read-target.json").read_text())
k = (x["market"], x["symbol"])
a, b = minute(CUTOFF - timedelta(days=7)), minute(CUTOFF)
with connect(APP_DSN, budget_class="market") as c:
    keys = c.execute(
        "SELECT market_id,symbol FROM market_data.instrument_generations ORDER BY market_id,symbol"
    ).fetchall()
    queries = {
        "year": (ARRAY_RANGE, (*k, CUTOFF - timedelta(days=365), CUTOFF)),
        "full": (ARRAY_RANGE, (*k, datetime.fromisoformat(x["first"]), CUTOFF)),
        "latest": (
            CANDLE_RANGE.replace("ORDER BY ts_open", "ORDER BY ts_open DESC LIMIT %s"),
            (*k, datetime.fromisoformat(x["first"]), CUTOFF, 100),
        ),
        "catalog": (
            COUNTS.format(mask="d.actual"),
            (
                [k[0] for k in keys],
                [k[1] for k in keys],
                [a] * len(keys),
                b,
                timestamp(b - 1).date(),
            ),
        ),
        "gap": (
            FIRST_GAP,
            (a, b, a, a, 1, timestamp(a).date(), timestamp(b - 1).date(), *k, a, b, a),
        ),
    }
    result = {
        name: {
            "sql": q,
            "params": p,
            "plan": c.execute("EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) " + q, p).fetchone()[0],
        }
        for name, (q, p) in queries.items()
    }
(OUT / "query-plans.json").write_text(json.dumps(result, default=str, indent=2))
print({k: v["plan"][0]["Execution Time"] for k, v in result.items()})
