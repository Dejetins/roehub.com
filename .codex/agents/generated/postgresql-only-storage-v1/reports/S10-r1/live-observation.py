"""Independent installed SQL evidence, no state changes."""
import json,os
from datetime import datetime,timezone
from trading.platform.postgres_budget import connect,reservation_counts
q={
'closed_minutes':"SELECT market_id,symbol,count(*),min(ts_open),max(ts_open),count(DISTINCT ts_open),array_agg(DISTINCT source) FROM market_data.canonical_candles_1m WHERE ts_open>='2026-10-14 08:43+00' AND ts_open<'2026-10-14 08:53+00' GROUP BY market_id,symbol ORDER BY market_id,symbol",
'connections':"SELECT market_id,symbol,connected,observed_at,collector_pid FROM market_data_stream_connections ORDER BY market_id,symbol",
'events':"SELECT market_id,symbol,event_type,occurred_at,reason,error_code,start_at,end_at FROM market_data_stream_events WHERE occurred_at>='2026-10-14 08:40+00' ORDER BY event_id",
'frontier':"SELECT * FROM market_data.candle_history_verification",
'old_sessions':"SELECT pid,state,backend_start FROM pg_stat_activity WHERE client_addr IN ('172.23.0.6','172.23.0.8') AND backend_start<'2026-10-14 08:41+00'",
'recovery':"SELECT market_id,symbol,state,reason,error_code,start_at,end_at FROM market_data_stream_recovery ORDER BY market_id,symbol,start_at",
}
r={'observed_at':datetime.now(timezone.utc).isoformat(),'budget':reservation_counts()}
with connect(os.environ['STRATEGY_PG_DSN'],budget_class='admin') as c:
 for k,sql in q.items():
  try:
   with c.transaction():r[k]=c.execute(sql).fetchall()
  except Exception as e:r[k]={'error':type(e).__name__}
print(json.dumps(r,default=str,indent=2))
