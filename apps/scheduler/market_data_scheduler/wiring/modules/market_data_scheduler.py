from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram, start_http_server

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
)
from trading.contexts.market_data.adapters.outbound.clients.rest_instrument_metadata_source import (
    RestInstrumentMetadataSource,
)
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    MarketDataRuntimeConfig,
    load_market_data_runtime_config,
)
from trading.contexts.market_data.adapters.outbound.config.whitelist import (
    load_whitelist_rows_from_csv,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleIndexReader,
    ClickHouseEnabledInstrumentReader,
    ClickHouseInstrumentRefWriter,
    ClickHouseMarketRefWriter,
    ClickHouseRawKlineWriter,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.application.dto import RestFillTask, WhitelistInstrumentRow
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
)
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.services import (
    AsyncRestFillQueue,
    SchedulerBackfillPlanner,
)
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.contexts.market_data.application.use_cases import (
    EnrichRefInstrumentsFromExchangeUseCase,
    RestCatchUp1mUseCase,
    RestFillRange1mUseCase,
    SeedRefMarketUseCase,
    SyncWhitelistToRefInstrumentsUseCase,
)
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerJob:
    """
    Scheduler job descriptor with fixed name and interval.

    Parameters:
    - name: job name used in logs/metrics.
    - interval_seconds: periodic interval between runs.
    """

    name: str
    interval_seconds: int


class MarketDataSchedulerMetrics:
    """
    Prometheus metrics bundle for maintenance scheduler jobs.
    """

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        """
        Create scheduler metric objects.

        Parameters:
        - registry: optional explicit Prometheus registry (tests can pass isolated one).

        Returns:
        - None.

        Assumptions/Invariants:
        - Metrics are instantiated once per scheduler process.

        Errors/Exceptions:
        - May raise registration errors on duplicate metric names.

        Side effects:
        - Registers metrics in default Prometheus registry.
        """
        effective_registry = registry if registry is not None else REGISTRY

        self.scheduler_job_runs_total = Counter(
            "scheduler_job_runs_total",
            "Scheduler job run count",
            labelnames=("job",),
            registry=effective_registry,
        )
        self.scheduler_job_errors_total = Counter(
            "scheduler_job_errors_total",
            "Scheduler job error count",
            labelnames=("job",),
            registry=effective_registry,
        )
        self.scheduler_job_duration_seconds = Histogram(
            "scheduler_job_duration_seconds",
            "Scheduler job duration in seconds",
            labelnames=("job",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0, 180.0),
            registry=effective_registry,
        )
        self.scheduler_tasks_planned_total = Counter(
            "scheduler_tasks_planned_total",
            "Planned maintenance tasks grouped by reason",
            labelnames=("reason",),
            registry=effective_registry,
        )
        self.scheduler_tasks_enqueued_total = Counter(
            "scheduler_tasks_enqueued_total",
            "Enqueued maintenance tasks grouped by reason",
            labelnames=("reason",),
            registry=effective_registry,
        )
        self.scheduler_startup_scan_instruments_total = Counter(
            "scheduler_startup_scan_instruments_total",
            "How many instruments were scanned in startup scan runs",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_instruments_total = Counter(
            "scheduler_rest_catchup_instruments_total",
            "How many instruments were processed by periodic rest insurance catchup",
            labelnames=("status",),
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_tail_minutes_total = Counter(
            "scheduler_rest_catchup_tail_minutes_total",
            "Total tail minutes requested by periodic rest catchup",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_tail_rows_written_total = Counter(
            "scheduler_rest_catchup_tail_rows_written_total",
            "Total tail rows written by periodic rest catchup",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_gap_days_scanned_total = Counter(
            "scheduler_rest_catchup_gap_days_scanned_total",
            "Total historical UTC days scanned for gaps by periodic rest catchup",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_gap_days_with_gaps_total = Counter(
            "scheduler_rest_catchup_gap_days_with_gaps_total",
            "Total historical UTC days where gaps were detected",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_gap_ranges_filled_total = Counter(
            "scheduler_rest_catchup_gap_ranges_filled_total",
            "Total missing ranges filled by periodic rest catchup",
            registry=effective_registry,
        )
        self.scheduler_rest_catchup_gap_rows_written_total = Counter(
            "scheduler_rest_catchup_gap_rows_written_total",
            "Total historical gap rows written by periodic rest catchup",
            registry=effective_registry,
        )


class MarketDataSchedulerApp:
    """
    Runtime orchestrator for market-data maintenance scheduler.

    Parameters:
    - config: runtime market-data config.
    - whitelist_path: path to whitelist CSV used by sync job.
    - seed_use_case: seed ref_market use-case.
    - sync_use_case: whitelist sync use-case.
    - enrich_use_case: instrument enrichment use-case.
    - instrument_reader: enabled-instruments reader.
    - index_reader: canonical index reader.
    - rest_fill_queue: async queue executing rest fill tasks in background.
    - backfill_planner: deterministic planner of bootstrap/historical/tail ranges.
    - rest_catchup_use_case: periodic full-history gap/tail catchup executor.
    - metrics: scheduler metrics bundle.
    - metrics_port: HTTP port for `/metrics`.
    """

    def __init__(
        self,
        *,
        config: MarketDataRuntimeConfig,
        whitelist_path: str,
        seed_use_case: SeedRefMarketUseCase,
        sync_use_case: SyncWhitelistToRefInstrumentsUseCase,
        enrich_use_case: EnrichRefInstrumentsFromExchangeUseCase,
        instrument_reader: EnabledInstrumentReader,
        index_reader: CanonicalCandleIndexReader,
        rest_fill_queue: AsyncRestFillQueue,
        backfill_planner: SchedulerBackfillPlanner,
        rest_catchup_use_case: RestCatchUp1mUseCase,
        metrics: MarketDataSchedulerMetrics,
        metrics_port: int,
    ) -> None:
        """
        Validate and store scheduler runtime dependencies.

        Parameters:
        - See class-level documentation.

        Returns:
        - None.

        Assumptions/Invariants:
        - Job dependencies are ready for immediate invocation.

        Errors/Exceptions:
        - Raises `ValueError` on invalid constructor arguments.

        Side effects:
        - None.
        """
        if metrics_port <= 0:
            raise ValueError("metrics_port must be > 0")
        self._config = config
        self._whitelist_path = whitelist_path
        self._seed_use_case = seed_use_case
        self._sync_use_case = sync_use_case
        self._enrich_use_case = enrich_use_case
        self._instrument_reader = instrument_reader
        self._index_reader = index_reader
        self._rest_fill_queue = rest_fill_queue
        self._backfill_planner = backfill_planner
        self._rest_catchup_use_case = rest_catchup_use_case
        self._metrics = metrics
        self._metrics_port = metrics_port
        self._clock: Clock = SystemClock()

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Start scheduler jobs and run until stop event is set.

        Parameters:
        - stop_event: cooperative shutdown event.

        Returns:
        - None.

        Assumptions/Invariants:
        - Stop event is controlled by process signal handlers.

        Errors/Exceptions:
        - Propagates fatal initialization/runtime exceptions.

        Side effects:
        - Starts Prometheus endpoint.
        - Runs startup sequence (S1, S2, startup scan).
        - Runs periodic scheduler jobs.
        """
        start_http_server(self._metrics_port)
        log.info("scheduler metrics server started on port %s", self._metrics_port)
        await self._rest_fill_queue.start()

        startup_jobs = [
            SchedulerJob(
                name="sync_whitelist",
                interval_seconds=self._config.scheduler.jobs.sync_whitelist.interval_seconds,
            ),
            SchedulerJob(
                name="enrich",
                interval_seconds=self._config.scheduler.jobs.enrich.interval_seconds,
            ),
            SchedulerJob(
                name="startup_scan",
                interval_seconds=1,
            ),
        ]
        for job in startup_jobs:
            await self._run_once(job)

        jobs = [
            SchedulerJob(
                name="sync_whitelist",
                interval_seconds=self._config.scheduler.jobs.sync_whitelist.interval_seconds,
            ),
            SchedulerJob(
                name="enrich",
                interval_seconds=self._config.scheduler.jobs.enrich.interval_seconds,
            ),
            SchedulerJob(
                name="rest_insurance_catchup",
                interval_seconds=self._config.scheduler.jobs.rest_insurance_catchup.interval_seconds,
            ),
        ]

        periodic_tasks = [
            asyncio.create_task(
                self._run_periodic_job(job, stop_event),
                name=f"scheduler-{job.name}",
            )
            for job in jobs
        ]
        await stop_event.wait()
        await asyncio.gather(*periodic_tasks, return_exceptions=True)
        await self._rest_fill_queue.close()

    async def _run_periodic_job(self, job: SchedulerJob, stop_event: asyncio.Event) -> None:
        """
        Execute one scheduler job periodically until shutdown.

        Parameters:
        - job: job descriptor.
        - stop_event: cooperative shutdown event.

        Returns:
        - None.

        Assumptions/Invariants:
        - Job handlers are idempotent and safe for repeated execution.

        Errors/Exceptions:
        - None. Handler exceptions are captured into metrics/logs.

        Side effects:
        - Runs job-specific side effects according to handler logic.
        """
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=job.interval_seconds)
            except TimeoutError:
                await self._run_once(job)

    async def _run_once(self, job: SchedulerJob) -> None:
        """
        Execute one job run with metrics and error handling.

        Parameters:
        - job: job descriptor.

        Returns:
        - None.

        Assumptions/Invariants:
        - Job name maps to one known handler.

        Errors/Exceptions:
        - None. Exceptions are logged and converted into error metrics.

        Side effects:
        - Updates job metrics and invokes job handler.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        self._metrics.scheduler_job_runs_total.labels(job=job.name).inc()

        try:
            if job.name == "sync_whitelist":
                await self._run_sync_whitelist_job()
            elif job.name == "enrich":
                await self._run_enrich_job()
            elif job.name == "startup_scan":
                await self._run_startup_scan_job()
            elif job.name == "rest_insurance_catchup":
                await self._run_rest_insurance_job()
            else:
                raise RuntimeError(f"unknown scheduler job: {job.name}")
        except Exception:  # noqa: BLE001
            self._metrics.scheduler_job_errors_total.labels(job=job.name).inc()
            log.exception("scheduler job failed: %s", job.name)
        finally:
            duration = max(loop.time() - started, 0.0)
            self._metrics.scheduler_job_duration_seconds.labels(job=job.name).observe(duration)

    async def _run_sync_whitelist_job(self) -> None:
        """
        Run S1: seed markets and sync whitelist into `ref_instruments`.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Whitelist CSV schema follows existing project contract.

        Errors/Exceptions:
        - Propagates whitelist parsing and use-case errors.

        Side effects:
        - Writes into `ref_market` and `ref_instruments` tables.
        """
        await asyncio.to_thread(self._seed_use_case.run)
        whitelist_rows = await asyncio.to_thread(
            load_whitelist_rows_from_csv,
            Path(self._whitelist_path),
        )
        dto_rows = [
            WhitelistInstrumentRow(instrument_id=row.instrument_id, is_enabled=row.is_enabled)
            for row in whitelist_rows
        ]
        await asyncio.to_thread(self._sync_use_case.run, dto_rows)

    async def _run_enrich_job(self) -> None:
        """
        Run S2: enrich reference instruments from exchange instrument metadata endpoints.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Only enabled tradable symbols are enriched.

        Errors/Exceptions:
        - Propagates enrich use-case errors.

        Side effects:
        - Writes enrichment rows into `ref_instruments`.
        """
        report = await asyncio.to_thread(self._enrich_use_case.run)
        log.info(
            "enrich_ref_instruments completed: instruments_total=%s rows_upserted=%s missing_metadata=%s",  # noqa: E501
            report.instruments_total,
            report.rows_upserted,
            report.symbols_missing_metadata,
        )

    async def _run_startup_scan_job(self) -> None:
        """
        Run startup maintenance scan once and enqueue bootstrap/historical/tail tasks.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Startup scan should execute exactly once per process start.

        Errors/Exceptions:
        - Propagates planning or queue errors.

        Side effects:
        - Reads canonical bounds for enabled instruments.
        - Enqueues background REST fill tasks.
        """
        await self._scan_and_enqueue(scan_reason="startup_scan")

    async def _run_rest_insurance_job(self) -> None:
        """
        Run S3: periodic maintenance scan and enqueue insurance/background fill tasks.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Fill ranges use half-open semantics `[start, end)`.
        - Current in-flight minute is excluded via `now_floor`.

        Errors/Exceptions:
        - Propagates per-instrument errors to caller (job wrapper handles them).

        Side effects:
        - Reads canonical bounds for enabled instruments.
        - Enqueues background REST fill tasks.
        """
        await self._scan_and_enqueue(
            scan_reason="rest_insurance_catchup",
            allowed_reasons={"scheduler_bootstrap", "historical_backfill"},
        )
        await self._run_periodic_gap_catchup()

    async def _run_periodic_gap_catchup(self) -> None:
        """
        Run full-history REST catchup for all enabled instruments with bounded concurrency.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Use-case performs tail catchup and historical gap scan against canonical index.
        - Instruments without canonical seed are skipped (bootstrap is handled by planner queue).

        Errors/Exceptions:
        - None. Per-instrument failures are logged and reflected in metrics.

        Side effects:
        - Executes REST catchup use-case and writes rows into raw storage.
        """
        instruments = await asyncio.to_thread(self._instrument_reader.list_enabled_tradable)
        if not instruments:
            log.info("rest_insurance_catchup: no enabled instruments for periodic gap catchup")
            return

        semaphore = asyncio.Semaphore(self._config.ingestion.rest_concurrency_instruments)
        inter_instrument_delay_s = self._config.ingestion.rest_inter_instrument_delay_s
        remaining_instruments = len(instruments)
        remaining_instruments_lock = asyncio.Lock()
        now_floor = UtcTimestamp(floor_to_minute_utc(self._clock.now().value))

        async def _sleep_between_instruments_if_needed() -> None:
            """
            Apply configured pause only when there are pending instruments in this run.

            Parameters:
            - None.

            Returns:
            - None.

            Assumptions/Invariants:
            - `remaining_instruments` is decremented exactly once per instrument task.
            - Delay value is already validated as non-negative by runtime config parser.

            Errors/Exceptions:
            - Propagates asyncio cancellation errors from `asyncio.sleep`.

            Side effects:
            - Suspends current coroutine for configured delay between instrument runs.
            """
            nonlocal remaining_instruments
            async with remaining_instruments_lock:
                remaining_instruments -= 1
                has_next = remaining_instruments > 0
            if inter_instrument_delay_s <= 0 or not has_next:
                return
            await asyncio.sleep(inter_instrument_delay_s)

        async def _run_one(instrument: InstrumentId) -> None:
            """
            Execute one periodic REST catchup run under shared concurrency semaphore.

            Parameters:
            - instrument: enabled tradable instrument id.

            Returns:
            - None.

            Assumptions/Invariants:
            - Per-instrument failures do not abort other scheduled instruments.

            Errors/Exceptions:
            - None. Errors are converted into metrics/log records.

            Side effects:
            - Triggers REST catchup execution and updates scheduler metrics.
            - Applies configured delay before releasing execution slot when next instrument exists.
            """
            async with semaphore:
                try:
                    report = await asyncio.to_thread(self._rest_catchup_use_case.run, instrument)
                except ValueError as exc:
                    if "Run initial backfill first" in str(exc):
                        self._metrics.scheduler_rest_catchup_instruments_total.labels(
                            status="skipped_no_seed"
                        ).inc()
                        log.info(
                            "rest_insurance_catchup skipped instrument without canonical seed: %s",
                            instrument,
                        )
                        return
                    self._metrics.scheduler_rest_catchup_instruments_total.labels(
                        status="failed"
                    ).inc()
                    log.exception("rest_insurance_catchup failed for %s", instrument)
                    return
                except Exception:  # noqa: BLE001
                    self._metrics.scheduler_rest_catchup_instruments_total.labels(
                        status="failed"
                    ).inc()
                    log.exception("rest_insurance_catchup failed for %s", instrument)
                    return
                else:
                    self._metrics.scheduler_rest_catchup_instruments_total.labels(status="ok").inc()
                    tail_minutes = _minutes_between(report.tail_start, report.tail_end)
                    lag_to_now_minutes = _minutes_between(report.tail_end, now_floor)
                    self._metrics.scheduler_rest_catchup_tail_minutes_total.inc(tail_minutes)
                    self._metrics.scheduler_rest_catchup_tail_rows_written_total.inc(
                        report.tail_rows_written
                    )
                    self._metrics.scheduler_rest_catchup_gap_days_scanned_total.inc(
                        report.gap_days_scanned
                    )
                    self._metrics.scheduler_rest_catchup_gap_days_with_gaps_total.inc(
                        report.gap_days_with_gaps
                    )
                    self._metrics.scheduler_rest_catchup_gap_ranges_filled_total.inc(
                        report.gap_ranges_filled
                    )
                    self._metrics.scheduler_rest_catchup_gap_rows_written_total.inc(
                        report.gap_rows_written
                    )
                    log.info(
                        "rest_insurance_catchup instrument=%s tail_minutes=%s tail_rows_written=%s "
                        "gap_days_scanned=%s gap_days_with_gaps=%s gap_ranges_filled=%s "
                        "gap_rows_written=%s lag_to_now_minutes=%s",
                        instrument,
                        tail_minutes,
                        report.tail_rows_written,
                        report.gap_days_scanned,
                        report.gap_days_with_gaps,
                        report.gap_ranges_filled,
                        report.gap_rows_written,
                        lag_to_now_minutes,
                    )
                finally:
                    await _sleep_between_instruments_if_needed()

        await asyncio.gather(
            *[asyncio.create_task(_run_one(instrument)) for instrument in instruments]
        )

    async def _scan_and_enqueue(
        self,
        *,
        scan_reason: str,
        allowed_reasons: set[str] | None = None,
    ) -> None:
        """
        Build and enqueue maintenance tasks for all enabled instruments.

        Parameters:
        - scan_reason: textual reason used in logs.
        - allowed_reasons: optional whitelist of task reasons to enqueue.

        Returns:
        - None.

        Assumptions/Invariants:
        - Instrument list source is `ref_instruments` with enabled+tradable filter.
        - Task ranges keep half-open semantics `[start, end)`.

        Errors/Exceptions:
        - Propagates storage and queue errors.

        Side effects:
        - Executes canonical bounds reads.
        - Enqueues REST fill tasks into background queue.
        """
        instruments = await asyncio.to_thread(self._instrument_reader.list_enabled_tradable)
        if not instruments:
            log.info("%s: no enabled instruments for maintenance scan", scan_reason)
            return

        if scan_reason == "startup_scan":
            self._metrics.scheduler_startup_scan_instruments_total.inc(len(instruments))

        now_floor = UtcTimestamp(floor_to_minute_utc(self._clock.now().value))
        semaphore = asyncio.Semaphore(self._config.ingestion.rest_concurrency_instruments)
        planned_tasks: list[RestFillTask] = []

        async def _plan_one(instrument: InstrumentId) -> list[RestFillTask]:
            """
            Plan maintenance tasks for one instrument under bounded planner concurrency.

            Parameters:
            - instrument: enabled tradable instrument id.

            Returns:
            - Planned task list for that instrument.

            Assumptions/Invariants:
            - Canonical bounds are read with `before=now_floor` to exclude in-flight minute.

            Errors/Exceptions:
            - Propagates index reader and planner errors.

            Side effects:
            - Reads canonical bounds from storage.
            """
            async with semaphore:
                bounds = await asyncio.to_thread(
                    self._index_reader.bounds_1m,
                    instrument_id=instrument,
                    before=now_floor,
                )
                earliest = self._config.market_by_id(
                    instrument.market_id
                ).rest.earliest_available_ts_utc
                return self._backfill_planner.plan_for_instrument(
                    instrument_id=instrument,
                    earliest_market_ts=earliest,
                    bounds_1m=bounds,
                    now_floor=now_floor,
                )

        grouped = await asyncio.gather(
            *[asyncio.create_task(_plan_one(instrument)) for instrument in instruments]
        )
        for tasks in grouped:
            if allowed_reasons is None:
                planned_tasks.extend(tasks)
                continue
            planned_tasks.extend([task for task in tasks if task.reason in allowed_reasons])

        planned_by_reason: dict[str, int] = {}
        for task in planned_tasks:
            planned_by_reason[task.reason] = planned_by_reason.get(task.reason, 0) + 1
            self._metrics.scheduler_tasks_planned_total.labels(reason=task.reason).inc()

        _log_first_planned_tasks(
            scan_reason=scan_reason,
            tasks=planned_tasks,
            config=self._config,
            limit=10,
        )

        enqueued = 0
        enqueued_by_reason: dict[str, int] = {}
        for task in planned_tasks:
            accepted = await self._rest_fill_queue.enqueue(task)
            if accepted:
                enqueued += 1
                enqueued_by_reason[task.reason] = enqueued_by_reason.get(task.reason, 0) + 1
                self._metrics.scheduler_tasks_enqueued_total.labels(reason=task.reason).inc()

        log.info(
            "%s: instruments_scanned=%s planned_tasks=%s planned_by_reason=%s enqueued_tasks=%s enqueued_by_reason=%s",  # noqa: E501
            scan_reason,
            len(instruments),
            len(planned_tasks),
            planned_by_reason,
            enqueued,
            enqueued_by_reason,
        )


def _log_first_planned_tasks(
    *,
    scan_reason: str,
    tasks: list[RestFillTask],
    config: MarketDataRuntimeConfig,
    limit: int,
) -> None:
    """
    Emit debug-friendly preview logs for the first planned maintenance tasks.

    Parameters:
    - scan_reason: current scan type (`startup_scan` or periodic scan).
    - tasks: planned task list.
    - config: runtime config used to derive instrument key prefixes.
    - limit: maximum number of tasks to log.

    Returns:
    - None.

    Assumptions/Invariants:
    - `limit` can be zero or positive; negative values are treated as zero.

    Errors/Exceptions:
    - None.

    Side effects:
    - Writes INFO logs through scheduler logger.
    """
    if not tasks:
        return
    size = max(limit, 0)
    for index, task in enumerate(tasks[:size], start=1):
        market = config.market_by_id(task.instrument_id.market_id)
        instrument_key = f"{market.market_code}:{task.instrument_id.symbol}"
        log.info(
            "%s planned_task[%s]: instrument_key=%s reason=%s range=[%s, %s)",
            scan_reason,
            index,
            instrument_key,
            task.reason,
            task.time_range.start,
            task.time_range.end,
        )


def _minutes_between(start: UtcTimestamp | None, end: UtcTimestamp | None) -> int:
    """
    Compute non-negative difference in whole minutes between optional UTC timestamps.

    Parameters:
    - start: range start timestamp or `None`.
    - end: range end timestamp or `None`.

    Returns:
    - Non-negative minute difference rounded down to whole minutes.

    Assumptions/Invariants:
    - Both timestamps are UTC-aware when present.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if start is None or end is None:
        return 0
    delta_seconds = (end.value - start.value).total_seconds()
    if delta_seconds <= 0:
        return 0
    return int(delta_seconds // 60)


def build_market_data_scheduler_app(
    *,
    config_path: str,
    whitelist_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
) -> MarketDataSchedulerApp:
    """
    Build fully wired market-data maintenance scheduler app.

    Parameters:
    - config_path: path to `market_data.yaml`.
    - whitelist_path: path to `whitelist.csv`.
    - environ: environment mapping with ClickHouse settings.
    - metrics_port: Prometheus HTTP port.

    Returns:
    - Ready-to-run scheduler app instance.

    Assumptions/Invariants:
    - ClickHouse credentials are available in environment mapping.

    Errors/Exceptions:
    - Propagates config parsing and infrastructure wiring errors.

    Side effects:
    - Creates ClickHouse client and Prometheus metric objects.
    """
    config = load_market_data_runtime_config(Path(config_path))
    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )

    market_writer = ClickHouseMarketRefWriter(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    instrument_writer = ClickHouseInstrumentRefWriter(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    seed_use_case = SeedRefMarketUseCase(writer=market_writer, clock=SystemClock())
    sync_use_case = SyncWhitelistToRefInstrumentsUseCase(
        writer=instrument_writer,
        clock=SystemClock(),
        known_market_ids=set(config.market_ids()),
    )

    index_reader = ClickHouseCanonicalCandleIndexReader(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    instrument_reader = ClickHouseEnabledInstrumentReader(
        gateway=gateway,
        database=clickhouse_settings.database,
    )
    rest_source = RestCandleIngestSource(
        cfg=config,
        clock=SystemClock(),
        http=RequestsHttpClient(),
        ingest_id=uuid4(),
    )
    rest_fill_use_case = RestFillRange1mUseCase(
        source=rest_source,
        writer=ClickHouseRawKlineWriter(gateway=gateway, database=clickhouse_settings.database),
        clock=SystemClock(),
        max_days_per_insert=config.backfill.max_days_per_insert,
        batch_size=10_000,
        index_reader=index_reader,
    )
    rest_catchup_use_case = RestCatchUp1mUseCase(
        index=index_reader,
        source=rest_source,
        writer=ClickHouseRawKlineWriter(gateway=gateway, database=clickhouse_settings.database),
        clock=SystemClock(),
        max_days_per_insert=config.backfill.max_days_per_insert,
        batch_size=10_000,
        ingest_id=uuid4(),
    )
    metadata_source = RestInstrumentMetadataSource(cfg=config, http=RequestsHttpClient())
    enrich_use_case = EnrichRefInstrumentsFromExchangeUseCase(
        instrument_reader=instrument_reader,
        metadata_source=metadata_source,
        writer=instrument_writer,
        clock=SystemClock(),
    )
    rest_fill_queue = AsyncRestFillQueue(
        executor=rest_fill_use_case.run,
        worker_count=config.ingestion.rest_concurrency_instruments,
    )
    backfill_planner = SchedulerBackfillPlanner(
        tail_lookback_minutes=config.ingestion.tail_lookback_minutes,
    )

    return MarketDataSchedulerApp(
        config=config,
        whitelist_path=whitelist_path,
        seed_use_case=seed_use_case,
        sync_use_case=sync_use_case,
        enrich_use_case=enrich_use_case,
        instrument_reader=instrument_reader,
        index_reader=index_reader,
        rest_fill_queue=rest_fill_queue,
        backfill_planner=backfill_planner,
        rest_catchup_use_case=rest_catchup_use_case,
        metrics=MarketDataSchedulerMetrics(),
        metrics_port=metrics_port,
    )
