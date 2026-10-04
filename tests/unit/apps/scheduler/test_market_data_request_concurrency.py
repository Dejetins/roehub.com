"""Configured explicit-request capacity reaches the actual scheduler thread pool."""
import asyncio
from pathlib import Path
from threading import Barrier
from typing import Any, cast

from apps.scheduler.market_data_scheduler.wiring.modules.market_data_scheduler import (
    MarketDataSchedulerApp,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)


def test_requests_only_uses_configured_four_worker_pool_and_drains_before_stop():
    async def exercise():
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        config = load_market_data_runtime_config(Path('configs/dev/market_data.yaml'))
        assert config.ingestion.rest_concurrency_instruments == 4
        entered = Barrier(4)
        finished = []

        class Runner:
            def run_batch(self, *, executor, concurrency):
                assert concurrency == 4

                def work(index):
                    entered.wait(timeout=5)
                    finished.append(index)

                list(executor.map(work, range(concurrency)))
                loop.call_soon_threadsafe(stop.set)
                return concurrency

        app = object.__new__(MarketDataSchedulerApp)
        app._config = config
        app.work_request_runner = cast(Any, Runner())
        app.history_bounds_runner = None
        await asyncio.wait_for(app.run_work_requests(stop), timeout=10)
        assert sorted(finished) == [0, 1, 2, 3]

    asyncio.run(exercise())
