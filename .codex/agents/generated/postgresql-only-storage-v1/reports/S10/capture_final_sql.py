"""Independent counted SQL observations; no credentials or private row payloads."""

import json
import os

from trading.platform.postgres_budget import connect

queries = {
    "coverage": "SELECT market_id,symbol,actual_count,first_ts,last_ts,generation FROM market_data.candle_coverage_summary ORDER BY market_id,symbol",  # noqa: E501
    "canonical": "SELECT market_id,symbol,count(*),min(ts_open),max(ts_open) FROM market_data.canonical_candles_1m GROUP BY market_id,symbol ORDER BY market_id,symbol",  # noqa: E501
    "selected": "SELECT DISTINCT market_id,symbol FROM market_data_instrument_selections ORDER BY market_id,symbol",  # noqa: E501
    "stream_connections": "SELECT market_id,symbol,connected,observed_at FROM market_data_stream_connections ORDER BY market_id,symbol",  # noqa: E501
    "funding": "SELECT market_id,count(*) FROM market_data.canonical_funding_rates GROUP BY market_id",  # noqa: E501
    "artifact_requests": "SELECT request_id,exchange,market_type,symbol,state,reason FROM backtest_artifact_requests WHERE created_at>'2026-10-14 07:30:00Z' ORDER BY created_at",  # noqa: E501
    "artifact_events": "SELECT state,reason,count(*) FROM backtest_artifact_events WHERE observed_at>'2026-10-14 07:30:00Z' GROUP BY state,reason ORDER BY state,reason",  # noqa: E501
    "catalog_job": "SELECT job_id,kind,state,error_code,rows_written FROM market_data_work_requests WHERE job_id='81415c14-943f-4e46-8eb1-40c070e7e91a'",  # noqa: E501
    "catalog_events": "SELECT event_type,state,error_code FROM market_data_work_events WHERE job_id='81415c14-943f-4e46-8eb1-40c070e7e91a' ORDER BY event_id",  # noqa: E501
    "slot_safety": "SELECT state,count(*) FROM backtest_artifact_slot_ownership GROUP BY state",
    "slot_readers": "SELECT count(*) FROM backtest_artifact_slot_readers",
    "postgres_parameters": "SELECT name,setting,unit FROM pg_settings WHERE name IN ('max_connections','shared_buffers','work_mem') ORDER BY name",  # noqa: E501
    "adoption": "SELECT store,count(*) FROM roehub_storage_migrations GROUP BY store ORDER BY store",  # noqa: E501
}
with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    results = {name: db.execute(query).fetchall() for name, query in queries.items()}
print(json.dumps(results, indent=2, default=str))
