import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from apps.cli.wiring.db.postgres import close_market_gateways, market_gateway
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (
    PostgresHistoryRangeVerifier,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

cfg = load_market_data_runtime_config(Path("/work/.local_artifacts/platform-local/config/market-data.yaml"))
gw = market_gateway(os.environ, workload="stream")
clock = SystemClock()
http = RequestsHttpClient.from_config(cfg, os.environ)
end = datetime(2020, 10, 15, 4, tzinfo=timezone.utc)
task = RestFillTask(
    InstrumentId(MarketId(1), Symbol("AAVEUSDT")),
    TimeRange(UtcTimestamp(end - timedelta(hours=1)), UtcTimestamp(end)),
    "stream_history_repair",
)
try:
    r = RestFillRange1mUseCase(
        source=RestCandleIngestSource(cfg=cfg, clock=clock, http=http, ingest_id=uuid4()),
        writer=PostgresRawKlineWriter(gateway=gw),
        clock=clock,
        index_reader=PostgresCanonicalCandleIndexReader(gateway=gw),
        history_verifier=PostgresHistoryRangeVerifier(gw),
        max_days_per_insert=cfg.backfill.max_days_per_insert,
        batch_size=500,
    ).run(task)
    print(json.dumps({"success": True, "result": str(r)}))
except Exception as e:
    print(
        json.dumps(
            {
                "type": type(e).__name__,
                "safe_message": str(e)[:200],
                "frames": [
                    {"file": f.filename, "line": f.lineno, "name": f.name}
                    for f in traceback.extract_tb(e.__traceback__)
                ],
            }
        )
    )
finally:
    http.close()
    close_market_gateways()
