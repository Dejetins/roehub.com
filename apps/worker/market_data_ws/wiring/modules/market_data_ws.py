from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import UUID, uuid4

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.binance import (
    BinanceWsClosedCandleStream,
    BinanceWsHooks,
)
from trading.contexts.market_data.adapters.outbound.clients.bybit import (
    BybitWsClosedCandleStream,
    BybitWsHooks,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketConfig,
    MarketDataRuntimeConfig,
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseConnectGateway,
    ClickHouseEnabledInstrumentReader,
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.application.dto import CandleWithMeta, RestFillTask
from trading.contexts.market_data.application.services import (
    AsyncRawInsertBuffer,
    AsyncRestFillQueue,
    InsertBufferHooks,
    ReconnectTailFillPlanner,
    RestFillQueueHooks,
    WsMinuteGapTracker,
)
from trading.contexts.market_data.application.use_cases import RestFillRange1mUseCase
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


class MarketDataWsApp:
    """
    Runtime orchestrator for EPIC 3 market-data websocket worker.

    Parameters:
    - config: market-data runtime config.
    - instrument_reader: enabled-instruments reader.
    - index_reader: canonical index reader for reconnect tail planning.
    - insert_buffer: async raw insert buffer.
    - rest_fill_queue: background rest fill queue.
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
        instrument_reader: ClickHouseEnabledInstrumentReader,
        index_reader: ClickHouseCanonicalCandleIndexReader,
        insert_buffer: AsyncRawInsertBuffer,
        rest_fill_queue: AsyncRestFillQueue,
        gap_tracker: WsMinuteGapTracker,
        reconnect_planner: ReconnectTailFillPlanner,
        ingest_id: UUID,
        metrics: MarketDataWsMetrics,
        metrics_port: int,
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
        self._config = config
        self._instrument_reader = instrument_reader
        self._index_reader = index_reader
        self._insert_buffer = insert_buffer
        self._rest_fill_queue = rest_fill_queue
        self._gap_tracker = gap_tracker
        self._reconnect_planner = reconnect_planner
        self._ingest_id = ingest_id
        self._metrics = metrics
        self._metrics_port = metrics_port

    async def run(self, stop_event: asyncio.Event) -> None:
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

        await self._insert_buffer.start()
        await self._rest_fill_queue.start()

        plans = self._build_connection_plans(self._instrument_reader.list_enabled_tradable())
        ws_tasks = [
            asyncio.create_task(
                self._run_plan(plan, stop_event),
                name=f"ws-{plan.market.market_id.value}",
            )
            for plan in plans
        ]

        if not ws_tasks:
            log.warning("no enabled instruments found; worker will stay idle until shutdown")

        await stop_event.wait()
        log.info("worker shutdown requested")

        await asyncio.gather(*ws_tasks, return_exceptions=True)
        await self._insert_buffer.close()
        await self._rest_fill_queue.close()

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
                    on_connected=self._metrics.ws_connection_state,
                    on_reconnect=self._metrics.ws_reconnects_total.inc,
                    on_message=self._metrics.ws_messages_total.inc,
                    on_error=self._metrics.ws_errors_total.inc,
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
                    on_connected=self._metrics.ws_connection_state,
                    on_reconnect=self._metrics.ws_reconnects_total.inc,
                    on_message=self._metrics.ws_messages_total.inc,
                    on_error=self._metrics.ws_errors_total.inc,
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
        - Schedules gap-based rest fill task when minute sequence has gaps.
        """
        try:
            self._insert_buffer.submit(row)
        except RuntimeError:
            return

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
        tasks = self._reconnect_planner.plan(instruments)
        for task in tasks:
            await self._rest_fill_queue.enqueue(task)


def build_market_data_ws_app(
    *,
    config_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
) -> MarketDataWsApp:
    """
    Build fully wired market-data websocket worker app.

    Parameters:
    - config_path: path to `market_data.yaml`.
    - environ: environment mapping with ClickHouse settings.
    - metrics_port: Prometheus HTTP port.

    Returns:
    - Ready-to-run worker app instance.

    Assumptions/Invariants:
    - ClickHouse credentials are available in environment mapping.

    Errors/Exceptions:
    - Propagates config parsing and infrastructure wiring errors.

    Side effects:
    - Creates ClickHouse client and Prometheus metric objects.
    """
    config = load_market_data_runtime_config(Path(config_path))
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_client = _clickhouse_client(clickhouse_settings)
    gateway = ClickHouseConnectGateway(clickhouse_client)

    raw_writer = ClickHouseRawKlineWriter(gateway=gateway, database=clickhouse_settings.database)
    index_reader = ClickHouseCanonicalCandleIndexReader(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    instrument_reader = ClickHouseEnabledInstrumentReader(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    metrics = MarketDataWsMetrics()
    clock = SystemClock()
    ingest_id = uuid4()

    rest_source = RestCandleIngestSource(
        cfg=config,
        clock=clock,
        http=RequestsHttpClient(),
        ingest_id=ingest_id,
    )
    rest_fill_use_case = RestFillRange1mUseCase(
        source=rest_source,
        writer=raw_writer,
        clock=clock,
        max_days_per_insert=config.backfill.max_days_per_insert,
        batch_size=10_000,
    )

    insert_buffer = AsyncRawInsertBuffer(
        writer=raw_writer,
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
    rest_queue = AsyncRestFillQueue(
        executor=rest_fill_use_case.run,
        worker_count=config.ingestion.rest_concurrency_instruments,
        hooks=RestFillQueueHooks(
            on_task_enqueued=lambda task: _on_rest_task_enqueued(task, metrics),
            on_task_started=metrics.on_rest_fill_started,
            on_task_succeeded=metrics.on_rest_fill_succeeded,
            on_task_failed=metrics.on_rest_fill_failed,
        ),
    )
    gap_tracker = WsMinuteGapTracker(
        on_duplicate=metrics.ws_duplicates_total.inc,
        on_out_of_order=metrics.ws_out_of_order_total.inc,
    )
    reconnect_planner = ReconnectTailFillPlanner(index_reader=index_reader, clock=clock)

    return MarketDataWsApp(
        config=config,
        instrument_reader=instrument_reader,
        index_reader=index_reader,
        insert_buffer=insert_buffer,
        rest_fill_queue=rest_queue,
        gap_tracker=gap_tracker,
        reconnect_planner=reconnect_planner,
        ingest_id=ingest_id,
        metrics=metrics,
        metrics_port=metrics_port,
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
