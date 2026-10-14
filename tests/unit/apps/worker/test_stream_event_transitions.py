import asyncio
from collections import deque
from types import SimpleNamespace
from typing import cast

from apps.worker.market_data_ws.wiring.modules.market_data_ws import (
    MarketDataWsApp,
    WorkerConnectionPlan,
)


def test_short_disconnect_is_published_in_order_before_heartbeat():
    async def run():
        app = object.__new__(MarketDataWsApp)
        app._connections = {}
        app._connection_transitions = deque()
        app._reported_connections = {}
        app._connection_changed = asyncio.Event()
        app._publish_connections = None
        app._ingestion_failure = None
        snapshots = []
        stop = asyncio.Event()

        async def publish(snapshot):
            snapshots.append(snapshot)
            if len(snapshots) == 3:
                stop.set()

        app._async_publish_connections = publish
        plan = SimpleNamespace(
            market=SimpleNamespace(market_id=SimpleNamespace(value=1)), symbols=("BTCUSDT",)
        )
        for connected in [True, False, True]:
            app._record_connection(cast(WorkerConnectionPlan, plan), connected)
        await asyncio.wait_for(app._report_connections(stop), 1)
        assert [x[(1, "BTCUSDT")] for x in snapshots] == [True, False, True]
        assert not app._connection_transitions

    asyncio.run(run())
