"""Run configured real funding-universe sync per existing futures source."""

import dataclasses
import json
import os

from apps.cli.wiring.db.postgres import close_market_gateways
from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
    build_market_data_scheduler_app,
)
from trading.contexts.market_data.application.use_cases.sync_futures_funding_universe import (
    SyncFuturesFundingUniverseUseCase,
)
from trading.shared_kernel.primitives import MarketId

app = build_market_data_scheduler_app(
    config_path="/etc/roehub/market-data.yaml",
    environ=os.environ,
    metrics_port=9296,
)
original = app._funding_sync_use_case
for market in (2, 4):
    try:
        report = SyncFuturesFundingUniverseUseCase(
            source=original._source,
            store=original._store,
            clock=original._clock,
            market_ids=(MarketId(market),),
        ).run()
        print(
            json.dumps({"market_id": market, "result": dataclasses.asdict(report)}, default=str),
            flush=True,
        )
    except Exception as error:
        print(
            json.dumps(
                {"market_id": market, "failed": type(error).__name__, "reason": str(error)[:150]}
            ),
            flush=True,
        )
app._provider_http.close()
close_market_gateways()
