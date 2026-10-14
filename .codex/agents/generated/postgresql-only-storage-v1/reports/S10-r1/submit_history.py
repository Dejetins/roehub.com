import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.services.work_requests import WorkRequest
from trading.platform.postgres_budget import connect

now = datetime.now(timezone.utc)
end = now.replace(second=0, microsecond=0)
with connect(os.environ["STRATEGY_PG_DSN"], budget_class="admin") as db:
    members = db.execute(
        "SELECT organization_id,user_id FROM identity_memberships WHERE status='active'"
    ).fetchall()
    assert len(members) == 1
    selected = db.execute(
        "SELECT DISTINCT s.market_id,s.symbol,b.first_open_at FROM market_data_instrument_selections s JOIN market_data_history_bounds b USING(market_id,symbol) WHERE b.state='ready' ORDER BY s.market_id,s.symbol"
    ).fetchall()
    assert len(selected) == 5
repo = PostgresMarketDataWorkRequestRepository(
    gateway=PsycopgBacktestPostgresGateway(dsn=os.environ["STRATEGY_PG_DSN"])
)
for market, symbol, start in selected:
    row = repo.submit(
        organization_id=members[0][0],
        actor_user_id=members[0][1],
        key=uuid4(),
        request=WorkRequest("candle_ingestion", market, (symbol,), start, end),
        now=now,
    )
    print(
        json.dumps(
            {k: row[k] for k in ["job_id", "market_id", "symbols", "start_at", "end_at", "state"]},
            default=str,
        ),
        flush=True,
    )
