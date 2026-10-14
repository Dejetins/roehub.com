"""Apply the captured market-only reset manifest with old owners already stopped."""

import json
import os
from pathlib import Path

from trading.platform.postgres_budget import connect

out = Path("/work/.codex/agents/generated/postgresql-only-storage-v1/reports/S10")
manifest = json.loads((out / "reset-manifest.json").read_text())
counts = {}
with connect(os.environ["ROEHUB_STORAGE_POSTGRES_DSN"], budget_class="admin") as db:
    assert (
        db.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()"  # noqa: E501
        ).fetchone()[0]
        == 0
    )
    assert db.execute("SELECT count(*) FROM market_data.canonical_candles_1m").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM backtest_artifact_slot_readers").fetchone()[0] == 0
    assert (
        db.execute(
            "SELECT count(*) FROM backtest_artifact_slot_ownership WHERE state<>'available'"
        ).fetchone()[0]
        == 0
    )
    for table in manifest["clear_derived_tables"]:
        assert table.replace("_", "").isalnum()
        counts[table] = db.execute("DELETE FROM " + table).rowcount
    counts["market_data_work_requests"] = db.execute("""UPDATE market_data_work_requests SET
        state=CASE WHEN state='cancelled' THEN 'cancelled' ELSE 'failed' END,
        completed_units=0,rows_read=0,rows_written=0,error_code='storage_generation_reset',
        error_phase=NULL,worker_token=NULL,next_retry_at=NULL,retry_count=0,
        control_version=control_version+1,progress_epoch=progress_epoch+1,
        updated_at=now(),finished_at=now()""").rowcount
    counts["backtest_artifact_requests"] = db.execute("""UPDATE backtest_artifact_requests
        SET state='error',reason='storage_generation_reset',control_version=control_version+1
        WHERE state<>'error'""").rowcount
    db.execute("""INSERT INTO backtest_artifact_events(request_id,state,reason,progress)
        SELECT request_id,state,reason,'{}'::jsonb FROM backtest_artifact_requests
        WHERE reason='storage_generation_reset'""")
    counts["backtest_artifact_builds"] = db.execute("""UPDATE backtest_artifact_builds
        SET state='error',next_retry_at=NULL,reservation_bytes=0 WHERE state<>'error'""").rowcount
    db.execute("UPDATE backtest_artifact_attempts SET state='error' WHERE state<>'error'")
    counts["backtest_artifact_source_state"] = db.execute("""UPDATE backtest_artifact_source_state
        SET desired_target='{"source":null,"state":"storage_generation_reset"}',published='{}',
        reconciliation_cursor=NULL,source_generation=NULL,content_revision=NULL,
        funding_revision=NULL,source_contract_version=NULL""").rowcount
    db.execute("""UPDATE backtest_artifact_dispatch_state SET source_cursor=NULL,
        schedule_cursor=NULL,updated_at=now()""")
    db.execute("""INSERT INTO market_data.instrument_generations
        (market_id,symbol,index_state,invalidation_reason)
        SELECT market_id,symbol,'ready','s10_fresh_empty' FROM market_data_instrument_selections
        ON CONFLICT DO NOTHING""")
    db.execute("""INSERT INTO market_data.candle_coverage_summary(market_id,symbol,generation)
        SELECT market_id,symbol,generation FROM market_data.instrument_generations""")
    generations = db.execute(
        "SELECT market_id,symbol,generation FROM market_data.instrument_generations ORDER BY 1,2"
    ).fetchall()
print(
    json.dumps(
        {
            "counts": counts,
            "fresh_generations": generations,
            "canonical_rows": 0,
            "verified_frontiers": 0,
        },
        default=str,
    )
)
