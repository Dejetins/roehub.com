"""Killable I/O boundary around the existing Market Data adapters/use case.

Spawn never inherits database clients or threads. Only normalized DTOs and bounded
safe results cross the private pipe; credentials stay in process memory.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import psycopg

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader, _clickhouse_client
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseRawKlineWriter,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.work_errors import (
    transient_clickhouse_error,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    RestFillResult,
    RestFillTask,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
    transient_ingestion_error,
)
from trading.contexts.market_data.application.services.reconnect_tail_fill import (
    ReconnectTailFillPlanner,
)
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase
from trading.platform.time.system_clock import SystemClock


def classify_io_error(error: Exception) -> str | None:
    if isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError)):
        return "recovery_store_unavailable"
    return transient_ingestion_error(error) or transient_clickhouse_error(error)


def verify_received_range(task: RestFillTask, index) -> None:
    """A successful empty REST response cannot acknowledge a received WS minute."""
    if task.reason == "ws_recovery":
        keys = index.distinct_ts_opens(instrument_id=task.instrument_id,
                                      time_range=task.time_range)
        expected = int((task.time_range.end.value - task.time_range.start.value)
                       / timedelta(minutes=1))
        if len(set(keys)) != expected:
            raise TemporaryIngestionError("source_incomplete")


def watch_parent(connection) -> None:
    """Terminate orphan I/O if the owning process disappears."""
    try:
        connection.recv()
    except EOFError:
        os._exit(1)
    except OSError:
        return


def _child(connection, mode: str, payload, config_path: str, environ: dict[str, str]) -> None:
    # Never forward a driver's exception traceback/payload or credentials over stdout/stderr.
    logging.disable(logging.CRITICAL)

    threading.Thread(target=watch_parent, args=(connection,), daemon=True).start()
    try:
        cfg = load_market_data_runtime_config(Path(config_path))
        settings = ClickHouseSettingsLoader(environ).load()
        gateway = ThreadLocalClickHouseConnectGateway(
            client_factory=lambda: _clickhouse_client(settings)
        )
        writer = ClickHouseRawKlineWriter(gateway=gateway, database=settings.database)
        index = ClickHouseCanonicalCandleIndexReader(gateway=gateway, database=settings.database)
        clock = SystemClock()
        if mode == "write":
            writer.write_1m(payload)
            result = None
        elif mode == "plan":
            result = ReconnectTailFillPlanner(
                index_reader=index,
                clock=clock,
                bootstrap_start_by_market={
                    int(m.market_id.value): m.rest.earliest_available_ts_utc for m in cfg.markets
                },
                tail_lookback_minutes=cfg.ingestion.tail_lookback_minutes,
            ).plan(payload)
        else:
            source = RestCandleIngestSource(
                cfg=cfg, clock=clock, http=RequestsHttpClient(), ingest_id=uuid4()
            )
            result = RestFillRange1mUseCase(
                source=source,
                writer=writer,
                clock=clock,
                index_reader=index,
                max_days_per_insert=cfg.backfill.max_days_per_insert,
                batch_size=500,
            ).run(payload)
            verify_received_range(payload, index)
        connection.send({"ok": True, "result": result})
    except Exception as exc:
        code = classify_io_error(exc)
        connection.send(
            {
                "ok": False,
                "temporary": code is not None,
                "code": code or "non_retryable_ingestion_error",
                "retry_after_s": float(getattr(exc, "retry_after_s", 0)),
            }
        )
    finally:
        connection.close()


class MarketDataIoProcess:
    """One bounded child per call; caller owns concurrency (one raw writer + REST workers)."""

    def __init__(
        self,
        *,
        config_path: str,
        environ: Mapping[str, str],
        timeout_s: float = 60,
        child_target=None,
    ) -> None:
        self._config_path = str(Path(config_path).resolve())
        self._environ = dict(environ)
        self._timeout_s = timeout_s
        self._target = child_target or _child
        self.active_pids: set[int] = set()
        self._plan_lock = asyncio.Lock()

    async def run(self, mode: str, payload) -> Any:
        ctx = multiprocessing.get_context("spawn")
        receiver, sender = ctx.Pipe(duplex=True)
        child = ctx.Process(
            target=self._target, args=(sender, mode, payload, self._config_path, self._environ)
        )
        try:
            child.start()
            sender.close()
            assert child.pid is not None
            self.active_pids.add(child.pid)
            deadline = asyncio.get_running_loop().time() + self._timeout_s
            while not receiver.poll():
                if not child.is_alive():
                    raise TemporaryIngestionError("ingestion_process_exited")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TemporaryIngestionError("ingestion_io_timeout")
                await asyncio.sleep(0.025)
            try:
                reply = receiver.recv()
            except EOFError as exc:
                raise TemporaryIngestionError("ingestion_process_exited") from exc
            if not reply["ok"]:
                if reply["temporary"]:
                    raise TemporaryIngestionError(reply["code"], reply["retry_after_s"])
                raise RuntimeError(reply["code"])
            return reply["result"]
        finally:
            sender.close()
            receiver.close()
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=0.5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=0.5)
                self.active_pids.discard(child.pid)
                if child.is_alive():
                    raise RuntimeError("ingestion child could not be stopped")
                child.close()

    async def fill(self, task: RestFillTask) -> RestFillResult:
        return await self.run("fill", task)

    async def write(self, rows: Sequence[CandleWithMeta]) -> None:
        await self.run("write", rows)

    async def plan(self, instruments) -> list[RestFillTask]:
        # Reconnecting many market groups must not spawn an unbounded planner fan-out.
        async with self._plan_lock:
            return await self.run("plan", instruments)
