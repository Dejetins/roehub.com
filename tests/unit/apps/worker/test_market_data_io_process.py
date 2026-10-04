"""Real spawned processes: cancellation, timeout and successful result ownership."""

import asyncio
import multiprocessing
import time

import pytest

from apps.worker.market_data_ws.wiring.io_process import MarketDataIoProcess
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)


def hanging_child(connection, mode, payload, config, environ):
    time.sleep(60)


def successful_child(connection, mode, payload, config, environ):
    connection.send({"ok": True, "result": 42})
    connection.close()


def test_timeout_reaps_real_process_without_default_executor_threads():
    async def scenario():
        io = MarketDataIoProcess(
            config_path="unused", environ={}, timeout_s=0.1, child_target=hanging_child
        )
        with pytest.raises(TemporaryIngestionError, match="ingestion_io_timeout"):
            await io.run("fill", None)
        assert not io.active_pids
        assert getattr(asyncio.get_running_loop(), "_default_executor") is None

    asyncio.run(scenario())
    assert not multiprocessing.active_children()


def test_cancel_reaps_child_and_success_returns_result():
    async def scenario():
        io = MarketDataIoProcess(config_path="unused", environ={}, child_target=hanging_child)
        call = asyncio.create_task(io.run("fill", None))
        await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 1.5)
        assert not io.active_pids
        healthy = MarketDataIoProcess(
            config_path="unused", environ={}, child_target=successful_child
        )
        assert await healthy.run("fill", None) == 42
        assert not healthy.active_pids

    asyncio.run(scenario())
    assert not multiprocessing.active_children()


def watching_child(connection, mode, payload, config, environ):
    import threading

    from apps.worker.market_data_ws.wiring.io_process import watch_parent

    threading.Thread(target=watch_parent, args=(connection,), daemon=True).start()
    time.sleep(60)


def parent_for_watch(report):
    async def scenario():
        io = MarketDataIoProcess(config_path="unused", environ={}, child_target=watching_child)
        pending = asyncio.create_task(io.run("fill", None))
        while not io.active_pids:
            await asyncio.sleep(0.01)
        report.send(next(iter(io.active_pids)))
        report.close()
        await pending

    asyncio.run(scenario())


def test_parent_death_stops_owned_io_process():
    import os
    import signal
    import subprocess

    ctx = multiprocessing.get_context("spawn")
    report, sender = ctx.Pipe(duplex=False)
    parent = ctx.Process(target=parent_for_watch, args=(sender,))
    child_pid = None
    try:
        parent.start()
        sender.close()
        assert report.poll(5)
        child_pid = report.recv()
        parent.kill()
        parent.join(2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = subprocess.run(
                ["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("orphan child kept running after parent death")
    finally:
        report.close()
        if parent.is_alive():
            parent.kill()
            parent.join(2)
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.close()
