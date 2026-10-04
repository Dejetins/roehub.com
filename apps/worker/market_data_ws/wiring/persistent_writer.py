"""Single warmed, killable raw writer with bounded private IPC.

Only this process owns its ClickHouse client. One request is in flight. A timeout,
transport failure or cancellation destroys the generation before another call;
the caller's durable receipts retain any uncertain INSERT for safe recovery.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import pickle
import queue
import socket
import struct
import threading
from typing import Mapping, Sequence

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from apps.worker.market_data_ws.wiring.io_process import classify_io_error
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseRawKlineWriter,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)

_MAX_FRAME_BYTES = 8 * 1024 * 1024


def _encode(value) -> bytes:
    body = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError("writer IPC frame exceeds bound")
    return struct.pack("!I", len(body)) + body


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise EOFError
        result.extend(chunk)
    return bytes(result)


def _read_commands(sock: socket.socket, commands: queue.Queue) -> None:
    # Remains active during a blocked INSERT/client initialization, so parent death
    # cannot leave an orphan writer. The parent serializes all request/response pairs.
    try:
        while True:
            size = struct.unpack("!I", _receive_exact(sock, 4))[0]
            if size > _MAX_FRAME_BYTES:
                raise ValueError("writer IPC frame exceeds bound")
            commands.put(pickle.loads(_receive_exact(sock, size)))
    except (EOFError, OSError, ValueError, pickle.UnpicklingError):
        os._exit(1)


def _writer_child(sock: socket.socket, environ: dict[str, str]) -> None:
    logging.disable(logging.CRITICAL)
    commands: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=_read_commands, args=(sock, commands), daemon=True).start()
    client = None
    try:
        settings = ClickHouseSettingsLoader(environ).load()
        client = _clickhouse_client(settings)
        gateway = ThreadLocalClickHouseConnectGateway(client_factory=lambda: client)
        writer = ClickHouseRawKlineWriter(gateway=gateway, database=settings.database)
        sock.sendall(_encode({"ok": True}))  # Ready includes client initialization.
        while True:
            rows = commands.get()
            writer.write_1m(rows)
            sock.sendall(_encode({"ok": True}))
    except Exception as exc:
        code = classify_io_error(exc)
        try:
            sock.sendall(
                _encode(
                    {
                        "ok": False,
                        "temporary": code is not None,
                        "code": code or "non_retryable_ingestion_error",
                        "retry_after_s": float(getattr(exc, "retry_after_s", 0)),
                    }
                )
            )
        except OSError:
            pass
    finally:
        if client is not None:
            client.close()
        sock.close()


class PersistentRawWriterProcess:
    """Reuse one warmed writer; kill/reap before reusing after an uncertain outcome."""

    def __init__(
        self, *, environ: Mapping[str, str], timeout_s: float = 60, child_target=None
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("writer timeout must be positive")
        self._environ = dict(environ)
        self._timeout_s = timeout_s
        self._target = child_target or _writer_child
        self._lock = asyncio.Lock()
        self._child = None
        self._socket: socket.socket | None = None
        self._closed = False
        self._active_call: asyncio.Task | None = None

    @property
    def active_pids(self) -> set[int]:
        pid = self._child.pid if self._child is not None else None
        return {pid} if pid is not None else set()

    async def _receive_exact(self, size: int) -> bytes:
        sock = self._socket
        assert sock is not None
        result = bytearray()
        loop = asyncio.get_running_loop()
        while len(result) < size:
            chunk = await loop.sock_recv(sock, size - len(result))
            if not chunk:
                raise TemporaryIngestionError("ingestion_process_exited")
            result.extend(chunk)
        return bytes(result)

    async def _reply(self) -> None:
        size = struct.unpack("!I", await self._receive_exact(4))[0]
        if size > _MAX_FRAME_BYTES:
            raise RuntimeError("writer IPC frame exceeds bound")
        reply = pickle.loads(await self._receive_exact(size))
        if not reply["ok"]:
            if reply["temporary"]:
                raise TemporaryIngestionError(reply["code"], reply["retry_after_s"])
            raise RuntimeError(reply["code"])

    async def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._child is not None:
            if self._child.is_alive():
                return
            self._discard()
        parent, child_socket = socket.socketpair()
        parent.setblocking(False)
        child = multiprocessing.get_context("spawn").Process(
            target=self._target, args=(child_socket, self._environ)
        )
        self._socket, self._child = parent, child
        try:
            child.start()
        finally:
            child_socket.close()
        await self._reply()

    def _discard(self) -> None:
        child, sock = self._child, self._socket
        if sock is not None:
            sock.close()
        if child is not None:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=0.5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=0.5)
                if child.is_alive():
                    raise RuntimeError("writer child could not be stopped")
            child.close()
        self._child, self._socket = None, None

    async def _call(self, rows: Sequence[CandleWithMeta] | None) -> None:
        async with self._lock:
            self._active_call = asyncio.current_task()
            try:
                # Includes startup and transport, not only server response time.
                async with asyncio.timeout(self._timeout_s):
                    await self._ensure_started()
                    if rows is not None:
                        assert self._socket is not None
                        await asyncio.get_running_loop().sock_sendall(self._socket, _encode(rows))
                        await self._reply()
            except TimeoutError as exc:
                self._discard()
                raise TemporaryIngestionError("ingestion_io_timeout") from exc
            except OSError as exc:
                self._discard()
                raise TemporaryIngestionError("ingestion_process_exited") from exc
            except BaseException:
                self._discard()
                raise
            finally:
                self._active_call = None

    async def start(self) -> None:
        await self._call(None)

    async def write(self, rows: Sequence[CandleWithMeta]) -> None:
        await self._call(rows)

    async def close(self) -> None:
        # Normal lifecycle drains first; concurrent close also interrupts blocked IPC.
        self._closed = True
        if self._active_call is not None and self._active_call is not asyncio.current_task():
            self._active_call.cancel()
            await asyncio.gather(self._active_call, return_exceptions=True)
        self._discard()
