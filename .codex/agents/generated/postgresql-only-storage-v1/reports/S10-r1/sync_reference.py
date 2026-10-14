"""Real configured exchange catalog refresh, retaining per-provider failures."""

import dataclasses
import json
import os

from apps.cli.wiring.db.postgres import close_market_gateways
from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
    build_market_data_scheduler_app,
)
from trading.shared_kernel.primitives import MarketId

app = build_market_data_scheduler_app(
    config_path="/etc/roehub/market-data.yaml",
    environ=os.environ,
    metrics_port=9295,
)
print(json.dumps({"seed": dataclasses.asdict(app._seed_use_case.run())}), flush=True)
for market in [1, 2, 3, 4]:
    try:
        report = dataclasses.replace(
            app._catalog_refresh_use_case, market_ids=(MarketId(market),)
        ).run()
        print(
            json.dumps(
                {"market_id": market, "status": "synced", "report": dataclasses.asdict(report)}
            ),
            flush=True,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "market_id": market,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error)[:180],
                }
            ),
            flush=True,
        )
app._provider_http.close()

close_market_gateways()
