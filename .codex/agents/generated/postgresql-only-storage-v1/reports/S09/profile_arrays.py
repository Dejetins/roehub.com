import cProfile
import json
import pstats
import time
from datetime import datetime

from tools.qa.postgres_production_soak import APP_DSN, OUT
from tools.qa.postgres_production_workload import CUTOFF, array_hash, instrument, span
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)

x = json.loads((OUT / "read-target.json").read_text())
g = IngestionGateway(APP_DSN, workload="bulk", deadline_s=30)
p = cProfile.Profile()
start = time.monotonic()
try:
    p.enable()
    b = PostgresCanonicalCandleReader(g).read_1m_arrays(
        instrument(x["symbol"], x["market"]), span(datetime.fromisoformat(x["first"]), CUTOFF)
    )
    p.disable()
    assert array_hash(b) == x["array_hashes"]["full"]
finally:
    g.close()
with (OUT / "array-profile-before.txt").open("w") as f:
    f.write(
        "Profiling overhead included; not guardrail timing. Wall "
        + str(time.monotonic() - start)
        + "\n"
    )
    pstats.Stats(p, stream=f).sort_stats("tottime").print_stats(35)
