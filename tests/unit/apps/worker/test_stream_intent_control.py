"""Intent changes win over in-flight preparation and recovery cooldown."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from apps.worker.market_data_ws.wiring.modules import market_data_ws as module
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


def test_off_during_preparation_never_opens_stream(monkeypatch):
    async def scenario():
        instrument = InstrumentId(MarketId(1), Symbol('ADAUSDT'))
        desired = {instrument}
        stop = asyncio.Event()
        app = object.__new__(module.MarketDataWsApp)
        app._connections = {}
        app._metrics_port = 0
        app._ingestion_failure = None
        app._subscription_refresh_seconds = .001
        monkeypatch.setattr(app, "_insert_buffer",
            SimpleNamespace(start=AsyncMock(), close=AsyncMock(), failure=None), raising=False)
        enqueue = AsyncMock()
        monkeypatch.setattr(app, "_rest_fill_queue", SimpleNamespace(
            start=AsyncMock(), close=AsyncMock(), enqueue=enqueue, failure=None), raising=False)
        async def read():
            return tuple(desired)
        async def prepare(pending):
            desired.clear()  # Off arrives while the history query is in flight.
            return pending, ()
        seen = []
        def plans(instruments):
            seen.extend(instruments)
            stop.set()
            return []
        app._read_desired = read
        app._prepare_history = prepare
        app._build_connection_plans = plans
        monkeypatch.setattr(module, 'start_http_server', lambda _: None)
        await asyncio.wait_for(app._run(stop), 1)
        assert not seen
        enqueue.assert_not_called()
    asyncio.run(scenario())


def test_off_is_polled_during_preparation_cooldown(monkeypatch):
    async def scenario():
        from trading.contexts.market_data.application.services.ingestion_retry import (
            TemporaryIngestionError,
        )
        instrument = InstrumentId(MarketId(1), Symbol('ADAUSDT'))
        stop = asyncio.Event()
        app = object.__new__(module.MarketDataWsApp)
        app._connections = {}
        app._metrics_port = 0
        app._ingestion_failure = None
        app._subscription_refresh_seconds = .001
        monkeypatch.setattr(app, "_insert_buffer",
            SimpleNamespace(start=AsyncMock(), close=AsyncMock(), failure=None), raising=False)
        enqueue = AsyncMock()
        monkeypatch.setattr(app, "_rest_fill_queue", SimpleNamespace(
            start=AsyncMock(), close=AsyncMock(), enqueue=enqueue, failure=None), raising=False)
        reads = 0
        async def read():
            nonlocal reads
            reads += 1
            if reads > 1:
                stop.set()
                return ()
            return (instrument,)
        app._read_desired = read
        app._prepare_history = AsyncMock(side_effect=TemporaryIngestionError('storage_unavailable'))
        app._build_connection_plans = lambda instruments: []
        monkeypatch.setattr(module, 'start_http_server', lambda _: None)
        await asyncio.wait_for(app._run(stop), .5)
        assert reads == 2
        app._prepare_history.assert_awaited_once()
    asyncio.run(scenario())
