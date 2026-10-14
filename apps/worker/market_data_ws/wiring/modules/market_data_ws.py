from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import AbstractContextManager, AsyncExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence
from uuid import UUID, uuid4

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from psycopg.conninfo import make_conninfo

from apps.cli.wiring.db.postgres import market_gateway
from apps.worker.market_data_ws.wiring.io_process import MarketDataIoProcess, classify_io_error
from apps.worker.market_data_ws.wiring.persistent_writer import PersistentRawWriterProcess
from trading.contexts.backtest.adapters.outbound import PsycopgBacktestPostgresGateway
from trading.contexts.market_data.adapters.outbound.clients.binance import (
    BinanceWsClosedCandleStream,
    BinanceWsHooks,
)
from trading.contexts.market_data.adapters.outbound.clients.bybit import (
    BybitWsClosedCandleStream,
    BybitWsHooks,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketConfig,
    MarketDataRuntimeConfig,
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.messaging.redis import (
    FanoutLiveCandlePublisher,
    NoopLiveCandlePublisher,
    RedisCandleHotCache,
    RedisCandleHotCacheHooks,
    RedisHotCacheLiveCandlePublisher,
    RedisLiveCandlePublisherHooks,
    RedisStreamsLiveCandlePublisher,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
    AsyncPostgresStreamRecoveryStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_bounds_repository import (  # noqa: E501
    PostgresHistoryBoundsRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
    PostgresHistoryCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)
from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.ports.feeds import LiveCandlePublisher
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.services import (
    AsyncRawInsertBuffer,
    AsyncRestFillQueue,
    InsertBufferHooks,
    ReconnectTailFillPlanner,
    RestFillQueueHooks,
    WsMinuteGapTracker,
)
from trading.contexts.market_data.application.services.batched_recovery import (
    BatchedRecoveryAdmission,
)
from trading.contexts.market_data.application.services.history_readiness import HistoryReadiness
from trading.contexts.market_data.application.services.history_stream_gate import (
    HistoryReadyInstrumentReader,
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,
)
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import InstrumentId, MarketId

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerConnectionPlan:
    """
    One websocket connection plan for a specific market and symbol subset.

    Parameters:
    - market: market runtime config entry.
    - symbols: symbol batch for one websocket connection.
    - instruments: corresponding instrument ids for reconnect tail planning.
    """

    market: MarketConfig
    symbols: tuple[str, ...]
    instruments: tuple[InstrumentId, ...]


async def _control_read_until_stopped(operation, stop_event: asyncio.Event):
    """Cancel/join managed control I/O promptly when the collector stops."""
    task = asyncio.create_task(operation)
    stopped = asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait((task, stopped), return_when=asyncio.FIRST_COMPLETED)
        if stop_event.is_set():
            raise asyncio.CancelledError
        return await task
    finally:
        for pending in (task, stopped):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, stopped, return_exceptions=True)


class MarketDataWsMetrics:
    """
    Prometheus metrics bundle for live WS worker.

    Assumptions/Invariants:
    - Metric names are stable and aligned with EPIC 3 requirements.
    """

    def __init__(self) -> None:
        """
        Create Prometheus metric objects for worker runtime.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Metrics are instantiated once per worker process.

        Errors/Exceptions:
        - May raise prometheus-client registration errors on duplicate names.

        Side effects:
        - Registers metrics in default Prometheus registry.
        """
        self.ws_connected = Gauge("ws_connected", "Current number of active websocket connections")
        self.ws_reconnects_total = Counter("ws_reconnects_total", "Websocket reconnect count")
        self.ws_messages_total = Counter("ws_messages_total", "Received websocket messages")
        self.ws_errors_total = Counter("ws_errors_total", "Websocket processing errors")
        self.ignored_non_closed_total = Counter(
            "ignored_non_closed_total",
            "Ignored non-closed 1m updates from websocket streams",
        )

        self.insert_rows_total = Counter("insert_rows_total", "Rows inserted into raw tables")
        self.insert_batches_total = Counter("insert_batches_total", "Insert batch count")
        self.insert_duration_seconds = Histogram(
            "insert_duration_seconds",
            "Raw insert duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.insert_errors_total = Counter("insert_errors_total", "Raw insert errors")

        self.ws_closed_to_insert_start_seconds = Histogram(
            "ws_closed_to_insert_start_seconds",
            "Latency from closed WS receive to raw insert start",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0),
        )
        self.ws_closed_to_insert_done_seconds = Histogram(
            "ws_closed_to_insert_done_seconds",
            "Latency from closed WS receive to raw insert complete",
            buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0),
        )

        self.ws_out_of_order_total = Counter(
            "ws_out_of_order_total",
            "Out-of-order websocket candle events",
        )
        self.ws_duplicates_total = Counter(
            "ws_duplicates_total",
            "Duplicate websocket candle events",
        )
        self.redis_publish_total = Counter(
            "redis_publish_total",
            "Successful Redis stream publish operations",
        )
        self.redis_publish_errors_total = Counter(
            "redis_publish_errors_total",
            "Redis stream publish failures",
        )
        self.redis_publish_duplicates_total = Counter(
            "redis_publish_duplicates_total",
            "Redis publish duplicate or out-of-order stream id events",
        )
        self.redis_publish_duration_seconds = Histogram(
            "redis_publish_duration_seconds",
            "Redis publish call duration in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        self.redis_hot_cache_writes_total = Counter(
            "redis_hot_cache_writes_total",
            "Successful Redis hot-cache write operations",
        )
        self.redis_hot_cache_write_errors_total = Counter(
            "redis_hot_cache_write_errors_total",
            "Redis hot-cache write failures",
        )
        self.redis_hot_cache_read_hits_total = Counter(
            "redis_hot_cache_read_hits_total",
            "Redis hot-cache range reads that returned at least one candle",
        )
        self.redis_hot_cache_read_misses_total = Counter(
            "redis_hot_cache_read_misses_total",
            "Redis hot-cache range reads that returned no candles",
        )
        self.redis_hot_cache_read_errors_total = Counter(
            "redis_hot_cache_read_errors_total",
            "Redis hot-cache range read failures",
        )
        self.redis_hot_cache_write_duration_seconds = Histogram(
            "redis_hot_cache_write_duration_seconds",
            "Redis hot-cache write call duration in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        self.redis_hot_cache_read_duration_seconds = Histogram(
            "redis_hot_cache_read_duration_seconds",
            "Redis hot-cache range read call duration in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        self.market_data_hot_cache_write_total = Counter(
            "market_data_hot_cache_write_total",
            "Successful Market Data live-tail hot-cache writes",
        )
        self.market_data_hot_cache_error_total = Counter(
            "market_data_hot_cache_error_total",
            "Market Data live-tail hot-cache read/write errors",
        )
        self.market_data_hot_cache_hit_total = Counter(
            "market_data_hot_cache_hit_total",
            "Market Data live-tail hot-cache reads with at least one candle",
        )
        self.market_data_hot_cache_miss_total = Counter(
            "market_data_hot_cache_miss_total",
            "Market Data live-tail hot-cache reads without candles",
        )

        self.rest_fill_tasks_total = Counter("rest_fill_tasks_total", "Scheduled rest fill tasks")
        self.rest_fill_active = Gauge("rest_fill_active", "Number of active rest fill tasks")
        self.rest_fill_errors_total = Counter("rest_fill_errors_total", "Rest fill task errors")
        self.rest_fill_duration_seconds = Histogram(
            "rest_fill_duration_seconds",
            "Rest fill task duration in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        self._ws_connection_count = 0

    def ws_connection_state(self, state: int) -> None:
        """
        Update active websocket connection gauge from connect/disconnect events.

        Parameters:
        - state: `1` for connected, `0` for disconnected.

        Returns:
        - None.

        Assumptions/Invariants:
        - Events are emitted in best-effort order by stream adapters.

        Errors/Exceptions:
        - None.

        Side effects:
        - Updates internal counter and gauge value.
        """
        if state == 1:
            self._ws_connection_count += 1
        else:
            self._ws_connection_count = max(self._ws_connection_count - 1, 0)
        self.ws_connected.set(self._ws_connection_count)

    def on_insert_batch(self, rows: int, duration_seconds: float) -> None:
        """
        Record raw insert batch metrics.

        Parameters:
        - rows: inserted row count.
        - duration_seconds: insert duration.

        Returns:
        - None.

        Assumptions/Invariants:
        - Counters are non-negative.

        Errors/Exceptions:
        - None.

        Side effects:
        - Updates Prometheus counters and histogram.
        """
        self.insert_rows_total.inc(rows)
        self.insert_batches_total.inc()
        self.insert_duration_seconds.observe(duration_seconds)

    def on_rest_fill_started(self, task: RestFillTask) -> None:
        """
        Record start of one rest fill task.

        Parameters:
        - task: started task metadata.

        Returns:
        - None.

        Assumptions/Invariants:
        - Task has already been enqueued.

        Errors/Exceptions:
        - None.

        Side effects:
        - Increments active-task gauge.
        """
        _ = task
        self.rest_fill_active.inc()

    def on_rest_fill_succeeded(self, task: RestFillTask, result, duration_seconds: float) -> None:
        """
        Record successful rest fill completion.

        Parameters:
        - task: completed task.
        - result: task execution report.
        - duration_seconds: task duration.

        Returns:
        - None.

        Assumptions/Invariants:
        - Active gauge was incremented at task start.

        Errors/Exceptions:
        - None.

        Side effects:
        - Decrements active-task gauge and observes duration histogram.
        """
        _ = task
        _ = result
        self.rest_fill_active.dec()
        self.rest_fill_duration_seconds.observe(duration_seconds)

    def on_rest_fill_failed(
        self,
        task: RestFillTask,
        exc: Exception,
        duration_seconds: float,
    ) -> None:
        """
        Record failed rest fill completion.

        Parameters:
        - task: failed task.
        - exc: captured exception.
        - duration_seconds: task duration.

        Returns:
        - None.

        Assumptions/Invariants:
        - Active gauge was incremented at task start.

        Errors/Exceptions:
        - None.

        Side effects:
        - Decrements active-task gauge and updates error/duration metrics.
        """
        _ = task
        _ = exc
        self.rest_fill_active.dec()
        self.rest_fill_errors_total.inc()
        self.rest_fill_duration_seconds.observe(duration_seconds)

    def on_hot_cache_write_success(self) -> None:
        self.redis_hot_cache_writes_total.inc()
        self.market_data_hot_cache_write_total.inc()

    def on_hot_cache_write_error(self) -> None:
        self.redis_hot_cache_write_errors_total.inc()
        self.market_data_hot_cache_error_total.inc()

    def on_hot_cache_read_hit(self) -> None:
        self.redis_hot_cache_read_hits_total.inc()
        self.market_data_hot_cache_hit_total.inc()

    def on_hot_cache_read_miss(self) -> None:
        self.redis_hot_cache_read_misses_total.inc()
        self.market_data_hot_cache_miss_total.inc()

    def on_hot_cache_read_error(self) -> None:
        self.redis_hot_cache_read_errors_total.inc()
        self.market_data_hot_cache_error_total.inc()


class MarketDataWsApp:
    """
    Runtime orchestrator for EPIC 3 market-data websocket worker.

    Parameters:
    - config: market-data runtime config.
    - instrument_reader: enabled-instruments reader.
    - index_reader: canonical index reader for reconnect tail planning.
    - insert_buffer: async raw insert buffer.
    - rest_fill_queue: background rest fill queue.
    - live_candle_publisher: best-effort live feed publisher for WS closed candles.
    - gap_tracker: in-memory minute gap tracker.
    - reconnect_planner: reconnect/restart tail planner.
    - ingest_id: process-level ingest session id.
    - metrics: worker metrics bundle.
    - metrics_port: HTTP port for `/metrics` endpoint.
    """

    def __init__(
        self,
        *,
        config: MarketDataRuntimeConfig,
        instrument_reader: EnabledInstrumentReader,
        index_reader: PostgresCanonicalCandleIndexReader,
        insert_buffer: AsyncRawInsertBuffer,
        rest_fill_queue: AsyncRestFillQueue,
        live_candle_publisher: LiveCandlePublisher,
        gap_tracker: WsMinuteGapTracker,
        reconnect_planner: ReconnectTailFillPlanner,
        ingest_id: UUID,
        metrics: MarketDataWsMetrics,
        metrics_port: int,
        subscription_refresh_seconds: float = 15.0,
        plan_recovery: Callable[..., Awaitable[list[RestFillTask]]] | None = None,
        recovery_lease: Callable[[], AbstractContextManager] = nullcontext,
        async_recovery_lease=None,
        async_publish_connections=None,
        start_io: Callable[[], Awaitable[None]] | None = None,
        close_io: Callable[[], Awaitable[None]] | None = None,
        read_desired=None,
        prepare_history=None,
        publish_connections: Callable[[dict[tuple[int, str], bool]], None] | None = None,
    ) -> None:
        """
        Validate and store worker runtime dependencies.

        Parameters:
        - See class-level documentation.

        Returns:
        - None.

        Assumptions/Invariants:
        - All collaborators are pre-built and ready for use.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if metrics_port <= 0:
            raise ValueError("metrics_port must be > 0")
        if subscription_refresh_seconds <= 0:
            raise ValueError("subscription_refresh_seconds must be > 0")
        if live_candle_publisher is None:  # type: ignore[truthy-bool]
            raise ValueError("live_candle_publisher must be provided")
        self._config = config
        self._instrument_reader = instrument_reader
        self._index_reader = index_reader
        self._insert_buffer = insert_buffer
        self._rest_fill_queue = rest_fill_queue
        self._live_candle_publisher = live_candle_publisher
        self._gap_tracker = gap_tracker
        self._reconnect_planner = reconnect_planner
        self._ingest_id = ingest_id
        self._read_desired = read_desired
        self._prepare_history = prepare_history
        self._publish_connections = publish_connections
        self._connections: dict[tuple[int, str], bool] = {}
        self._connection_transitions: deque[dict[tuple[int, str], bool]] = deque()
        self._reported_connections: dict[tuple[int, str], bool] = {}
        self._connection_changed = asyncio.Event()
        self._metrics = metrics
        self._metrics_port = metrics_port
        self._subscription_refresh_seconds = subscription_refresh_seconds
        self._plan_recovery = plan_recovery
        self._recovery_lease = recovery_lease
        self._async_recovery_lease = async_recovery_lease
        self._async_publish_connections = async_publish_connections
        self._ingestion_failure: Exception | None = None
        self._start_io = start_io
        self._close_io = close_io

    async def run(self, stop_event: asyncio.Event) -> None:
        async with AsyncExitStack() as stack:
            if self._async_recovery_lease is not None:
                await stack.enter_async_context(self._async_recovery_lease())
            else:
                stack.enter_context(self._recovery_lease())
            status_task = asyncio.create_task(self._report_connections(stop_event))
            try:
                if self._start_io is not None:
                    warmup = asyncio.ensure_future(self._start_io())
                    stopped = asyncio.create_task(stop_event.wait())
                    try:
                        await asyncio.wait((warmup, stopped), return_when=asyncio.FIRST_COMPLETED)
                        if stop_event.is_set():
                            return
                        try:
                            await warmup
                        except Exception as exc:
                            code = classify_io_error(exc)
                            if code is None:
                                raise
                            # A temporary startup outage is retried by the durable buffer.
                            log.warning("writer warmup deferred code=%s", code)
                    finally:
                        for task in (warmup, stopped):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(warmup, stopped, return_exceptions=True)
                await self._run(stop_event)
            finally:
                status_task.cancel()
                await asyncio.gather(status_task, return_exceptions=True)
                if self._close_io is not None:
                    await self._close_io()

    async def _report_connections(self, stop_event: asyncio.Event) -> None:
        """Publish bounded snapshots outside the event loop, including initial failures."""
        if self._publish_connections is None and self._async_publish_connections is None:
            return
        while not stop_event.is_set():
            self._connection_changed.clear()
            transition = self._connection_transitions[0] if self._connection_transitions else None
            snapshot = (
                {**self._reported_connections, **transition}
                if transition
                else dict(self._connections)
            )
            published = False
            try:
                if self._async_publish_connections is not None:
                    publication = asyncio.create_task(self._async_publish_connections(snapshot))
                else:
                    assert self._publish_connections is not None
                    publication = asyncio.create_task(
                        asyncio.to_thread(self._publish_connections, snapshot)
                    )
                try:
                    await asyncio.shield(publication)
                    published = True
                    self._reported_connections = snapshot
                    if transition is not None:
                        self._connection_transitions.popleft()
                except asyncio.CancelledError:
                    await publication  # Finish before releasing the collector session.
                    raise
            except Exception:
                # Expired observations become unknown; never retain a false healthy state.
                log.error("stream connection status publication failed")
            if self._connection_transitions:
                # Retry failed publication on the next heartbeat, never busy-loop.
                if published:
                    continue
            if stop_event.is_set():
                break
            try:
                await asyncio.wait_for(self._connection_changed.wait(), timeout=5)
            except TimeoutError:
                pass

    def _record_connection(self, plan: WorkerConnectionPlan, connected: bool) -> None:
        changes = {
            (plan.market.market_id.value, symbol): connected
            for symbol in plan.symbols
            if self._connections.get((plan.market.market_id.value, symbol)) != connected
        }
        self._connections.update(changes)
        if changes:
            if len(self._connection_transitions) >= 256:
                # Fail visibly instead of silently dropping events during a PG outage.
                self._ingestion_failure = RuntimeError("stream_event_buffer_capacity_exceeded")
                return
            self._connection_transitions.append(changes)
            self._connection_changed.set()

    def _connection_state(self, plan: WorkerConnectionPlan, connected: int) -> None:
        self._metrics.ws_connection_state(connected)
        self._record_connection(plan, bool(connected))

    def _connection_error(self, plan: WorkerConnectionPlan) -> None:
        self._metrics.ws_errors_total.inc()
        self._record_connection(plan, False)

    async def _run(self, stop_event: asyncio.Event) -> None:
        """
        Start worker runtime and serve until stop event is set.

        Parameters:
        - stop_event: cooperative shutdown signal.

        Returns:
        - None.

        Assumptions/Invariants:
        - Stop event is controlled by process signal handlers in entrypoint.

        Errors/Exceptions:
        - Propagates fatal initialization exceptions.

        Side effects:
        - Opens websocket connections.
        - Starts metrics server and background tasks.
        - Writes raw candles and triggers REST fills.
        """
        start_http_server(self._metrics_port)
        log.info("metrics server started on port %s", self._metrics_port)

        ws_tasks: dict[tuple[int, tuple[str, ...]], asyncio.Task[None]] = {}
        readiness_failures = 0
        next_preparation_at = 0.0
        admitted: set[InstrumentId] = set()
        try:
            await self._insert_buffer.start()
            await self._rest_fill_queue.start()
            while not stop_event.is_set():
                if self._ingestion_failure is not None:
                    raise RuntimeError("stream ingestion failed") from self._ingestion_failure
                for component in (self._insert_buffer, self._rest_fill_queue):
                    if getattr(component, "failure", None) is not None:
                        raise RuntimeError("stream ingestion failed; recovery receipts retained")
                try:
                    if self._read_desired is not None:
                        assert self._prepare_history is not None
                        requested = set(
                            await _control_read_until_stopped(self._read_desired(), stop_event)
                        )
                        removed = admitted - requested
                        admitted.intersection_update(requested)
                        if removed:
                            for instrument in removed:
                                log.info(
                                    "stream_disabled market_id=%s symbol=%s",
                                    instrument.market_id.value,
                                    str(instrument.symbol),
                                )
                            # Apply disable before any unrelated history/storage check.
                            kept = {
                                self._plan_key(p): p
                                for p in self._build_connection_plans(
                                    sorted(
                                        admitted, key=lambda i: (i.market_id.value, str(i.symbol))
                                    )
                                )
                            }
                            await self._replace_subscription_tasks(
                                ws_tasks=ws_tasks, plans=kept, stop_event=stop_event
                            )
                            ws_tasks = {
                                key: asyncio.create_task(self._run_plan(plan, stop_event))
                                for key, plan in kept.items()
                            }
                        pending = requested - admitted
                        ready, repairs = (), ()
                        if pending and asyncio.get_running_loop().time() >= next_preparation_at:
                            ready, repairs = await _control_read_until_stopped(
                                self._prepare_history(tuple(pending)), stop_event
                            )
                            # Intent may change while storage checks are in flight.
                            requested = set(
                                await _control_read_until_stopped(self._read_desired(), stop_event)
                            )
                            admitted.intersection_update(requested)
                            ready = tuple(i for i in ready if i in requested)
                            repairs = tuple(r for r in repairs if r.instrument_id in requested)
                            readiness_failures = 0
                            next_preparation_at = 0.0
                        for instrument in ready:
                            log.info(
                                "stream_history_ready market_id=%s symbol=%s",
                                instrument.market_id.value,
                                str(instrument.symbol),
                            )
                        admitted.update(ready)
                        instruments = tuple(
                            sorted(admitted, key=lambda i: (i.market_id.value, str(i.symbol)))
                        )
                        for repair in repairs:
                            if await self._rest_fill_queue.enqueue(repair):
                                log.info(
                                    "stream_preparing market_id=%s symbol=%s",
                                    repair.instrument_id.market_id.value,
                                    str(repair.instrument_id.symbol),
                                )
                    else:
                        instruments = await asyncio.to_thread(
                            self._instrument_reader.list_enabled_tradable
                        )
                except asyncio.CancelledError:
                    if stop_event.is_set():
                        break
                    raise
                except Exception as exc:
                    code = classify_io_error(exc)
                    if code is None:
                        raise
                    readiness_failures += 1
                    delay = min(300, 15 * 2 ** min(readiness_failures - 1, 5))
                    next_preparation_at = asyncio.get_running_loop().time() + delay
                    log.warning("stream_preparation_wait code=%s retry_seconds=%s", code, delay)
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=min(delay, self._subscription_refresh_seconds),
                        )
                    except TimeoutError:
                        pass
                    continue
                if self._read_desired is None and isinstance(
                    self._instrument_reader, HistoryReadyInstrumentReader
                ):
                    for repair in self._instrument_reader.repairs:
                        await self._rest_fill_queue.enqueue(repair)
                plans = self._build_connection_plans(instruments)
                desired = {self._plan_key(plan): plan for plan in plans}
                needs_refresh = set(ws_tasks) != set(desired) or any(
                    task.done() for task in ws_tasks.values()
                )
                if needs_refresh:
                    await self._replace_subscription_tasks(
                        ws_tasks=ws_tasks,
                        plans=desired,
                        stop_event=stop_event,
                    )
                    ws_tasks = {
                        key: asyncio.create_task(
                            self._run_plan(plan, stop_event),
                            name=f"ws-{plan.market.market_id.value}-{'-'.join(plan.symbols)}",
                        )
                        for key, plan in desired.items()
                    }
                    if not ws_tasks:
                        log.info("no enabled instruments yet; waiting for subscription refresh")
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=self._subscription_refresh_seconds
                    )
                except TimeoutError:
                    continue
        finally:
            log.info("worker shutdown requested")
            await self._replace_subscription_tasks(
                ws_tasks=ws_tasks,
                plans={},
                stop_event=stop_event,
            )
            results = await asyncio.gather(
                self._insert_buffer.close(), self._rest_fill_queue.close(), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    @staticmethod
    def _plan_key(plan: WorkerConnectionPlan) -> tuple[int, tuple[str, ...]]:
        """Return the stable identity of one websocket subscription plan."""

        return (plan.market.market_id.value, plan.symbols)

    async def _replace_subscription_tasks(
        self,
        *,
        ws_tasks: Mapping[tuple[int, tuple[str, ...]], asyncio.Task[None]],
        plans: Mapping[tuple[int, tuple[str, ...]], WorkerConnectionPlan],
        stop_event: asyncio.Event,
    ) -> None:
        """Cancel obsolete stream tasks before a refreshed subscription set starts."""

        _ = plans
        _ = stop_event
        for task in ws_tasks.values():
            if not task.done():
                task.cancel()
        if ws_tasks:
            await asyncio.gather(*ws_tasks.values(), return_exceptions=True)
        self._connections.clear()

    def _build_connection_plans(
        self,
        instruments: Sequence[InstrumentId],
    ) -> list[WorkerConnectionPlan]:
        """
        Group instruments by market and split symbol batches per connection.

        Parameters:
        - instruments: enabled tradable instruments from reference storage.

        Returns:
        - List of websocket connection plans.

        Assumptions/Invariants:
        - Per-market split size is defined by `market.ws.max_symbols_per_connection`.

        Errors/Exceptions:
        - Raises `KeyError` when instrument market_id is absent in runtime config.

        Side effects:
        - None.
        """
        by_market: dict[int, list[InstrumentId]] = {}
        for instrument in instruments:
            by_market.setdefault(instrument.market_id.value, []).append(instrument)

        plans: list[WorkerConnectionPlan] = []
        for market_id_value, rows in by_market.items():
            market = self._config.market_by_id(MarketId(market_id_value))
            chunk_size = market.ws.max_symbols_per_connection
            for offset in range(0, len(rows), chunk_size):
                chunk = tuple(rows[offset : offset + chunk_size])
                plans.append(
                    WorkerConnectionPlan(
                        market=market,
                        symbols=tuple(str(item.symbol) for item in chunk),
                        instruments=chunk,
                    )
                )
        return plans

    async def _run_plan(self, plan: WorkerConnectionPlan, stop_event: asyncio.Event) -> None:
        """
        Run one websocket connection plan until shutdown.

        Parameters:
        - plan: one market/symbol split for websocket connection.
        - stop_event: cooperative shutdown signal.

        Returns:
        - None.

        Assumptions/Invariants:
        - Plan symbols/instruments are aligned by position.

        Errors/Exceptions:
        - Runtime exceptions are handled by stream adapters and surfaced via metrics/logs.

        Side effects:
        - Maintains one websocket connection lifecycle.
        """
        market = plan.market
        if market.exchange == "binance":
            stream = BinanceWsClosedCandleStream(
                market_id=market.market_id,
                market_type=market.market_type,
                ws_url=market.ws.url,
                symbols=plan.symbols,
                ping_interval_s=market.ws.ping_interval_s,
                pong_timeout_s=market.ws.pong_timeout_s,
                reconnect=market.ws.reconnect,
                clock=SystemClock(),
                ingest_id=self._ingest_id,
                on_closed_candle=self._on_closed_candle,
                on_connected=lambda: self._on_connection_ready(plan.instruments),
                hooks=BinanceWsHooks(
                    on_connected=lambda connected: self._connection_state(plan, connected),
                    on_reconnect=self._metrics.ws_reconnects_total.inc,
                    on_message=self._metrics.ws_messages_total.inc,
                    on_error=lambda: self._connection_error(plan),
                    on_ignored_non_closed=self._metrics.ignored_non_closed_total.inc,
                ),
            )
            await stream.run(stop_event)
            return

        if market.exchange == "bybit":
            stream = BybitWsClosedCandleStream(
                market_id=market.market_id,
                market_type=market.market_type,
                ws_url=market.ws.url,
                symbols=plan.symbols,
                ping_interval_s=market.ws.ping_interval_s,
                pong_timeout_s=market.ws.pong_timeout_s,
                reconnect=market.ws.reconnect,
                clock=SystemClock(),
                ingest_id=self._ingest_id,
                on_closed_candle=self._on_closed_candle,
                on_connected=lambda: self._on_connection_ready(plan.instruments),
                hooks=BybitWsHooks(
                    on_connected=lambda connected: self._connection_state(plan, connected),
                    on_reconnect=self._metrics.ws_reconnects_total.inc,
                    on_message=self._metrics.ws_messages_total.inc,
                    on_error=lambda: self._connection_error(plan),
                    on_ignored_non_closed=self._metrics.ignored_non_closed_total.inc,
                ),
            )
            await stream.run(stop_event)
            return

        raise ValueError(f"unsupported exchange for ws worker: {market.exchange!r}")

    async def _on_closed_candle(self, row: CandleWithMeta) -> None:
        """
        Handle one normalized closed candle from websocket stream.

        Parameters:
        - row: normalized closed 1m candle with metadata.

        Returns:
        - None.

        Assumptions/Invariants:
        - Insert buffer and rest queue are already started.

        Errors/Exceptions:
        - None. Buffer enqueue errors during shutdown are ignored.

        Side effects:
        - Enqueues row into insert buffer.
        - Live Redis publication belongs to the committed writer callback.
        - Schedules gap-based rest fill task when minute sequence has gaps.
        """
        try:
            submit_wait = getattr(self._insert_buffer, "submit_wait", None)
            if submit_wait is not None:
                await submit_wait(row)
            else:
                self._insert_buffer.submit(row)
        except RuntimeError as exc:
            # Only a stopping buffer may reject intake quietly. Persistence/capacity
            # failures must break the connection and leave the receipt recoverable.
            if getattr(self._insert_buffer, "_stopping", False):
                return
            if classify_io_error(exc) is None:
                self._ingestion_failure = exc
            raise

        task = self._gap_tracker.observe(row)
        if task is not None:
            await self._rest_fill_queue.enqueue(task)

    async def _on_connection_ready(self, instruments: tuple[InstrumentId, ...]) -> None:
        """
        Schedule reconnect/restart tail tasks when connection is (re)established.

        Parameters:
        - instruments: instrument ids bound to this websocket connection.

        Returns:
        - None.

        Assumptions/Invariants:
        - Planner uses canonical index and current UTC minute floor.

        Errors/Exceptions:
        - Propagates planner/index errors to stream runtime.

        Side effects:
        - Enqueues reconnect/bootstrap rest fill tasks.
        """
        try:
            tasks = (
                await self._plan_recovery(instruments)
                if self._plan_recovery is not None
                else self._reconnect_planner.plan(instruments)
            )
            for task in tasks:
                await self._rest_fill_queue.enqueue(task)
        except Exception as exc:
            if classify_io_error(exc) is None:
                self._ingestion_failure = exc
            raise


def build_market_data_ws_app(
    *,
    config_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
    stored_candle_observer: Callable[[InstrumentId], None] | None = None,
) -> MarketDataWsApp:
    """
    Build fully wired market-data websocket worker app.

    Parameters:
    - config_path: path to `market_data.yaml`.
    - environ: environment mapping with PostgreSQL settings.
    - metrics_port: Prometheus HTTP port.

    Returns:
    - Ready-to-run worker app instance.

    Assumptions/Invariants:
    - PostgreSQL credentials are available in environment mapping.

    Errors/Exceptions:
    - Propagates config parsing and infrastructure wiring errors.

    Side effects:
    - Creates PostgreSQL client and Prometheus metric objects.
    """
    config = load_market_data_runtime_config(Path(config_path))
    gateway = market_gateway(environ, workload="stream")

    raw_writer = PostgresRawKlineWriter(gateway=gateway)
    index_reader = PostgresCanonicalCandleIndexReader(
        gateway=gateway,
    )
    recovery_store = PostgresStreamRecoveryStore(environ.get("STRATEGY_PG_DSN", ""))
    async_recovery = AsyncPostgresStreamRecoveryStore(environ.get("STRATEGY_PG_DSN", ""))
    admission = BatchedRecoveryAdmission(async_recovery)
    instrument_reader = HistoryReadyInstrumentReader(
        on_ready=recovery_store.clear_history_failures,
        source=_effective_instrument_reader(environ=environ),
        readiness=HistoryReadiness(
            origins=PostgresHistoryBoundsRepository(
                gateway=PsycopgBacktestPostgresGateway(dsn=environ["STRATEGY_PG_DSN"])
            ),
            coverage=PostgresHistoryCoverageReader(gateway=gateway),
        ),
    )
    metrics = MarketDataWsMetrics()
    clock = SystemClock()
    ingest_id = uuid4()
    redis_cfg = config.live_feed.redis_streams
    publishers: list[LiveCandlePublisher] = []

    hot_cache_cfg = config.live_feed.redis_hot_cache
    if hot_cache_cfg.enabled:
        hot_cache = RedisCandleHotCache(
            connection_config=redis_cfg,
            config=hot_cache_cfg,
            environ=environ,
            hooks=RedisCandleHotCacheHooks(
                on_write_success=metrics.on_hot_cache_write_success,
                on_write_error=metrics.on_hot_cache_write_error,
                on_write_duration=metrics.redis_hot_cache_write_duration_seconds.observe,
                on_read_hit=metrics.on_hot_cache_read_hit,
                on_read_miss=metrics.on_hot_cache_read_miss,
                on_read_error=metrics.on_hot_cache_read_error,
                on_read_duration=metrics.redis_hot_cache_read_duration_seconds.observe,
            ),
        )
        publishers.append(RedisHotCacheLiveCandlePublisher(hot_cache))

    if redis_cfg.enabled:
        publishers.append(
            RedisStreamsLiveCandlePublisher(
                config=redis_cfg,
                runtime_config=config,
                process_ingest_id=ingest_id,
                environ=environ,
                hooks=RedisLiveCandlePublisherHooks(
                    on_publish_success=metrics.redis_publish_total.inc,
                    on_publish_error=metrics.redis_publish_errors_total.inc,
                    on_publish_duplicate=metrics.redis_publish_duplicates_total.inc,
                    on_publish_duration=metrics.redis_publish_duration_seconds.observe,
                ),
            ),
        )
    if not publishers:
        live_candle_publisher = NoopLiveCandlePublisher()
    elif len(publishers) == 1:
        live_candle_publisher = publishers[0]
    else:
        live_candle_publisher = FanoutLiveCandlePublisher(publishers)

    io = MarketDataIoProcess(config_path=config_path, environ=environ)
    live_writer = PersistentRawWriterProcess(environ=environ)

    async def observed_live_write(rows) -> None:
        from trading.contexts.market_data.application.services.stored_candle_signal import (
            write_then_publish,
        )

        await write_then_publish(
            rows, live_writer.write, live_candle_publisher.publish_1m_closed, stored_candle_observer
        )

    async def defer_received(entry) -> None:
        await rest_queue.enqueue(entry.task)

    insert_buffer = AsyncRawInsertBuffer(
        writer=(
            raw_writer
            if stored_candle_observer is None
            else StoredCandleSignalWriter(raw_writer, stored_candle_observer)
        ),
        async_writer=observed_live_write,
        on_deferred=defer_received,
        async_store=async_recovery,
        admission=admission,
        classify_error=classify_io_error,
        clock=clock,
        flush_interval_ms=config.raw_write.flush_interval_ms,
        max_buffer_rows=config.raw_write.max_buffer_rows,
        hooks=InsertBufferHooks(
            on_ws_closed_to_insert_start=metrics.ws_closed_to_insert_start_seconds.observe,
            on_ws_closed_to_insert_done=metrics.ws_closed_to_insert_done_seconds.observe,
            on_insert_batch=metrics.on_insert_batch,
            on_insert_error=metrics.insert_errors_total.inc,
        ),
    )

    async def preparation_still_requested(task):
        if task.reason != "stream_history_repair":
            return True
        desired = await io.run("selected", None)
        return task.instrument_id in desired

    rest_queue = AsyncRestFillQueue(
        async_executor=io.fill,
        should_run=preparation_still_requested,
        window_minutes=60,
        async_store=async_recovery,
        admission=admission,
        classify_error=classify_io_error,
        worker_count=config.ingestion.rest_concurrency_instruments,
        hooks=RestFillQueueHooks(
            on_task_enqueued=lambda task: _on_rest_task_enqueued(task, metrics),
            on_task_started=metrics.on_rest_fill_started,
            on_task_succeeded=metrics.on_rest_fill_succeeded,
            on_task_failed=metrics.on_rest_fill_failed,
            on_task_cancelled=lambda _: metrics.rest_fill_active.dec(),
        ),
    )
    gap_tracker = WsMinuteGapTracker(
        on_duplicate=metrics.ws_duplicates_total.inc,
        on_out_of_order=metrics.ws_out_of_order_total.inc,
    )
    reconnect_planner = ReconnectTailFillPlanner(
        index_reader=index_reader,
        clock=clock,
        bootstrap_start_by_market={
            int(market.market_id.value): market.rest.earliest_available_ts_utc
            for market in config.markets
        },
        tail_lookback_minutes=config.ingestion.tail_lookback_minutes,
    )

    async def close_io() -> None:
        try:
            await admission.close()
        finally:
            await live_writer.close()

    return MarketDataWsApp(
        config=config,
        instrument_reader=instrument_reader,
        index_reader=index_reader,
        insert_buffer=insert_buffer,
        rest_fill_queue=rest_queue,
        live_candle_publisher=live_candle_publisher,
        gap_tracker=gap_tracker,
        reconnect_planner=reconnect_planner,
        ingest_id=ingest_id,
        metrics=metrics,
        metrics_port=metrics_port,
        plan_recovery=io.plan,
        async_recovery_lease=async_recovery.lease,
        async_publish_connections=async_recovery.publish_connections,
        start_io=live_writer.start,
        close_io=close_io,
        read_desired=lambda: io.run("selected", None),
        prepare_history=lambda pending: io.run("prepare", pending),
    )


def _effective_instrument_reader(
    *, environ: Mapping[str, str]
) -> PostgresInstrumentSelectionRepository:
    dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not dsn:
        raise ValueError("STRATEGY_PG_DSN is required for effective instrument selections")
    return PostgresInstrumentSelectionRepository(
        reference_fallback=False,
        gateway=PsycopgBacktestPostgresGateway(
            dsn=make_conninfo(
                dsn, connect_timeout=2, options="-c statement_timeout=2000 -c lock_timeout=1000"
            )
        ),
    )


def _on_rest_task_enqueued(task: RestFillTask, metrics: MarketDataWsMetrics) -> None:
    """
    Record rest task enqueue metric.

    Parameters:
    - task: accepted fill task.
    - metrics: worker metrics bundle.

    Returns:
    - None.

    Assumptions/Invariants:
    - Task enqueue metric does not require labels.

    Errors/Exceptions:
    - None.

    Side effects:
    - Increments `rest_fill_tasks_total`.
    """
    _ = task
    metrics.rest_fill_tasks_total.inc()
