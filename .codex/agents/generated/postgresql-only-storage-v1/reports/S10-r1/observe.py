import json
import os
from datetime import datetime, timezone

from trading.platform.postgres_budget import connect, reservation_counts

queries = {
    "coverage": "SELECT market_id,symbol,actual_count,first_ts,last_ts,generation FROM market_data.candle_coverage_summary ORDER BY market_id,symbol",
    "connections": "SELECT market_id,symbol,connected,observed_at FROM market_data_stream_connections ORDER BY market_id,symbol",
    "bounds": "SELECT market_id,symbol,state,first_open_at,error_code FROM market_data_history_bounds ORDER BY market_id,symbol",
    "funding": "SELECT market_id,symbol,count(*) FROM market_data.canonical_funding_rates GROUP BY market_id,symbol",
    "jobs": "SELECT kind,state,error_code,count(*) FROM market_data_work_requests GROUP BY kind,state,error_code",
}
result = {"observed_at": datetime.now(timezone.utc).isoformat(), "budget": reservation_counts()}
with connect(os.environ["STRATEGY_PG_DSN"], budget_class="admin") as db:
    for k, q in queries.items():
        try:
            with db.transaction():
                result[k] = db.execute(q).fetchall()
        except Exception as e:
            result[k] = {"error": type(e).__name__, "message": str(e)[:180]}
print(json.dumps(result, default=str, indent=2))
