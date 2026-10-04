"""Actual processes/transport; no database needed for ownership failure cases."""

import asyncio
import multiprocessing
import os
import queue
import signal
import subprocess
import threading
import time

import pytest

from apps.worker.market_data_ws.wiring.persistent_writer import (
    PersistentRawWriterProcess,
    _encode,
    _read_commands,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)


def fake_writer(sock, environ):
    commands = queue.Queue(maxsize=1)
    threading.Thread(target=_read_commands, args=(sock, commands), daemon=True).start()
    if environ.get("behavior") == "startup_hang":
        time.sleep(60)
    sock.sendall(_encode({"ok": True}))
    while True:
        commands.get()
        behavior = environ.get("behavior")
        if behavior == "hang":
            time.sleep(60)
        if behavior == "permanent":
            sock.sendall(_encode({"ok": False, "temporary": False, "code": "bad_schema"}))
        elif behavior == "temporary":
            sock.sendall(
                _encode(
                    {
                        "ok": False,
                        "temporary": True,
                        "code": "storage_memory_pressure",
                        "retry_after_s": 42,
                    }
                )
            )
        else:
            sock.sendall(_encode({"ok": True}))


def test_warmed_writer_reuses_process_and_serializes_calls():
    async def scenario():
        writer = PersistentRawWriterProcess(environ={}, child_target=fake_writer)
        try:
            await writer.start()
            pid = writer.active_pids.copy()
            await asyncio.gather(*(writer.write([]) for _ in range(10)))
            assert writer.active_pids == pid
            assert getattr(asyncio.get_running_loop(), "_default_executor") is None
        finally:
            await writer.close()
        assert not writer.active_pids
        with pytest.raises(RuntimeError, match="closed"):
            await writer.write([])

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["cancel", "close", "timeout"])
def test_blocked_write_is_reaped_and_next_generation_is_clean(action):
    async def scenario():
        writer = PersistentRawWriterProcess(environ={"behavior": "hang"}, child_target=fake_writer)
        await writer.start()
        old_pid = next(iter(writer.active_pids))
        writer._timeout_s = 0.15
        call = asyncio.create_task(writer.write([]))
        await asyncio.sleep(0.05)
        if action == "cancel":
            call.cancel()
        elif action == "close":
            await asyncio.wait_for(writer.close(), 1.5)
        with pytest.raises(
            TemporaryIngestionError if action == "timeout" else asyncio.CancelledError
        ):
            await asyncio.wait_for(call, 1.5)
        assert not writer.active_pids
        if action != "close":
            writer._timeout_s = 3
            writer._environ = {}
            try:
                await writer.write([])
                assert old_pid not in writer.active_pids
            finally:
                await writer.close()

    asyncio.run(scenario())


def test_writer_initialization_timeout_and_idle_death_are_recoverable():
    async def scenario():
        writer = PersistentRawWriterProcess(
            environ={"behavior": "startup_hang"}, timeout_s=0.1, child_target=fake_writer
        )
        with pytest.raises(TemporaryIngestionError, match="timeout"):
            await writer.start()
        assert not writer.active_pids
        writer._environ = {}
        writer._timeout_s = 3
        await writer.start()
        old = next(iter(writer.active_pids))
        os.kill(old, signal.SIGKILL)
        await asyncio.sleep(0.1)
        try:
            await writer.write([])
            assert old not in writer.active_pids
        finally:
            await writer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("behavior", ["temporary", "permanent"])
def test_errors_keep_classification_and_discard_failed_client(behavior):
    async def scenario():
        writer = PersistentRawWriterProcess(
            environ={"behavior": behavior}, child_target=fake_writer
        )
        await writer.start()
        with pytest.raises(
            TemporaryIngestionError if behavior == "temporary" else RuntimeError
        ) as e:
            await writer.write([])
        if behavior == "temporary":
            assert isinstance(e.value, TemporaryIngestionError)
            assert e.value.retry_after_s == 42
        assert not writer.active_pids
        await writer.close()

    asyncio.run(scenario())


def persistent_parent(report):
    async def scenario():
        writer = PersistentRawWriterProcess(environ={"behavior": "hang"}, child_target=fake_writer)
        await writer.start()
        report.send(next(iter(writer.active_pids)))
        report.close()
        await writer.write([])

    asyncio.run(scenario())


def test_parent_death_reaps_persistent_writer_during_blocked_write():
    ctx = multiprocessing.get_context("spawn")
    report, sender = ctx.Pipe(duplex=False)
    parent = ctx.Process(target=persistent_parent, args=(sender,))
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
            pytest.fail("orphan writer survived parent death")
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


def unread_writer(sock, environ):
    sock.sendall(_encode({"ok": True}))
    time.sleep(60)


def test_cancel_interrupts_backpressured_transport_without_a_thread():
    from typing import Any, cast

    async def scenario():
        writer = PersistentRawWriterProcess(environ={}, child_target=unread_writer)
        await writer.start()
        call = asyncio.create_task(writer.write(cast(Any, [b"x" * 2_000_000])))
        await asyncio.sleep(0.05)
        assert not call.done()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 1.5)
        assert not writer.active_pids
        assert getattr(asyncio.get_running_loop(), "_default_executor") is None
        await writer.close()

    asyncio.run(scenario())


def test_app_stop_interrupts_writer_warmup_and_reaps_child():
    from contextlib import nullcontext
    from unittest.mock import AsyncMock

    from apps.worker.market_data_ws.wiring.modules.market_data_ws import MarketDataWsApp

    async def scenario():
        writer = PersistentRawWriterProcess(
            environ={"behavior": "startup_hang"}, child_target=fake_writer
        )
        app = object.__new__(MarketDataWsApp)
        app._recovery_lease = nullcontext
        app._start_io = writer.start
        app._close_io = writer.close
        app._run = AsyncMock()
        stop = asyncio.Event()
        running = asyncio.create_task(app.run(stop))
        while not writer.active_pids:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(running, 1.5)
        assert not writer.active_pids
        app._run.assert_not_called()

    asyncio.run(scenario())
