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

cfg = load_market_data_runtime_config(Path("/etc/roehub/market-data.yaml"))
gw = market_gateway(os.environ, workload="bulk")
clock = SystemClock()
http = RequestsHttpClient.from_config(cfg, os.environ)
end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
verifier = PostgresHistoryRangeVerifier(gw)
uc = RestFillRange1mUseCase(
    source=RestCandleIngestSource(cfg=cfg, clock=clock, http=http, ingest_id=uuid4()),
    writer=PostgresRawKlineWriter(gateway=gw), clock=clock,
    index_reader=PostgresCanonicalCandleIndexReader(gateway=gw),
    history_verifier=verifier, max_days_per_insert=7, batch_size=500,
)
def snapshot(label):
    rows=gw.run(lambda c:c.execute("SELECT * FROM market_data.candle_history_verification ORDER BY market_id,symbol").fetchall())
    print(json.dumps({"kind":label,"at":datetime.now(timezone.utc),"frontier":rows},default=str),flush=True)
snapshot("before")
try:
    for mid,symbol in [(1,"0GUSDT"),(1,"AAPLBUSDT"),(1,"AAVEUSDT"),(1,"ACTUSDT"),(2,"0GUSDT")]:
        instrument=InstrumentId(MarketId(mid),Symbol(symbol))
        origin=verifier.origin(instrument_id=instrument)
        if origin is None: raise RuntimeError("missing_original_origin")
        cursor=origin
        print(json.dumps({"kind":"start","market_id":mid,"symbol":symbol,"origin":origin,"end":end},default=str),flush=True)
        while cursor<end:
            stop=min(cursor+timedelta(days=7),end)
            result=uc.run(RestFillTask(instrument,TimeRange(UtcTimestamp(cursor),UtcTimestamp(stop)),"stream_history_repair"))
            print(json.dumps({"kind":"window","market_id":mid,"symbol":symbol,"start":cursor,"end":stop,"result":str(result)},default=str),flush=True)
            cursor=stop
        snapshot("verified_"+str(mid)+"_"+symbol)
    snapshot("after")
finally:
    http.close()
    close_market_gateways()
