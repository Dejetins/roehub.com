"""Installed-budget spawned production fill loop; replies require clean PG ownership."""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

from apps.worker.market_data_ws.wiring.io_process import MarketDataIoProcess
from trading.contexts.market_data.application.dto import RestFillTask
from trading.platform.postgres_budget import connect, reservation_counts
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


async def main():
    io = MarketDataIoProcess(
        config_path="/work/.local_artifacts/platform-local/config/market-data.yaml",
        environ=os.environ,
    )
    before = reservation_counts()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    for i in range(18):
        stop = end - timedelta(minutes=i)
        task = RestFillTask(
            InstrumentId(MarketId(1), Symbol("ACTUSDT")),
            TimeRange(UtcTimestamp(stop - timedelta(minutes=1)), UtcTimestamp(stop)),
            "ws_recovery",
        )
        for attempt in range(5):
            try:
                result = await io.fill(task)
                break
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "attempt_error": type(error).__name__,
                            "code": str(error),
                            "quarantined": reservation_counts()["market"]["quarantined"],
                        }
                    ),
                    flush=True,
                )
                if attempt == 4:
                    raise
                await asyncio.sleep(2)
        print(
            json.dumps(
                {
                    "call": i + 1,
                    "rows_read": result.rows_read,
                    "rows_written": result.rows_written,
                    "active_children": len(io.active_pids),
                    "quarantined": reservation_counts()["market"]["quarantined"],
                }
            ),
            flush=True,
        )
    after = reservation_counts()
    assert after["market"]["quarantined"] == before["market"]["quarantined"]
    with connect(os.environ["STRATEGY_PG_DSN"], budget_class="admin") as db:
        count = db.execute(
            "SELECT count(*) FROM market_data.canonical_candles_1m WHERE market_id=1 AND symbol='ACTUSDT' AND ts_open>=%s AND ts_open<%s",
            (end - timedelta(minutes=18), end),
        ).fetchone()[0]
    assert count == 18
    print(
        json.dumps(
            {
                "completed": True,
                "real_spawn_calls": 18,
                "canonical_minutes": count,
                "before": before,
                "after": after,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
