from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from prometheus_client import Counter, Histogram, start_http_server

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.clients.rest_candle_ingest_source import (
    RestCandleIngestSource,
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
    ClickHouseConnectGateway,
    ClickHouseEnabledInstrumentReader,
    ClickHouseInstrumentRefWriter,
    ClickHouseMarketRefWriter,
    ClickHouseRawKlineWriter,
)
from trading.contexts.market_data.application.dto import RestFillTask, WhitelistInstrumentRow
from trading.contexts.market_data.application.services.minute_utils import floor_to_minute_utc
from trading.contexts.market_data.application.use_cases import (
    RestFillRange1mUseCase,
    SeedRefMarketUseCase,
    SyncWhitelistToRefInstrumentsUseCase,
)
from trading.platform.time.system_clock import SystemClock
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

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

    def __init__(self) -> None:
        """
        Create scheduler metric objects.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Metrics are instantiated once per scheduler process.

        Errors/Exceptions:
        - May raise registration errors on duplicate metric names.

        Side effects:
        - Registers metrics in default Prometheus registry.
        """
        self.scheduler_job_runs_total = Counter(
            "scheduler_job_runs_total",
            "Scheduler job run count",
            labelnames=("job",),
        )
        self.scheduler_job_errors_total = Counter(
            "scheduler_job_errors_total",
            "Scheduler job error count",
            labelnames=("job",),
        )
        self.scheduler_job_duration_seconds = Histogram(
            "scheduler_job_duration_seconds",
            "Scheduler job duration in seconds",
            labelnames=("job",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0, 180.0),
        )


class MarketDataSchedulerApp:
    """
    Runtime orchestrator for market-data maintenance scheduler.

    Parameters:
    - config: runtime market-data config.
    - whitelist_path: path to whitelist CSV used by sync job.
    - seed_use_case: seed ref_market use-case.
    - sync_use_case: whitelist sync use-case.
    - instrument_reader: enabled-instruments reader.
    - index_reader: canonical index reader.
    - rest_fill_use_case: bounded rest range fill use-case.
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
        instrument_reader: ClickHouseEnabledInstrumentReader,
        index_reader: ClickHouseCanonicalCandleIndexReader,
        rest_fill_use_case: RestFillRange1mUseCase,
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
        self._instrument_reader = instrument_reader
        self._index_reader = index_reader
        self._rest_fill_use_case = rest_fill_use_case
        self._metrics = metrics
        self._metrics_port = metrics_port
        self._bootstrap_start = UtcTimestamp(datetime(2017, 1, 1, tzinfo=timezone.utc))
        self._clock = SystemClock()

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
        - Runs periodic scheduler jobs.
        """
        start_http_server(self._metrics_port)
        log.info("scheduler metrics server started on port %s", self._metrics_port)

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

        tasks = [
            asyncio.create_task(
                self._run_periodic_job(job, stop_event),
                name=f"scheduler-{job.name}",
            )
            for job in jobs
        ]
        await stop_event.wait()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_periodic_job(self, job: SchedulerJob, stop_event: asyncio.Event) -> None:
        """
        Execute one scheduler job on start and then periodically.

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
        await self._run_once(job)

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
        Run S2 placeholder: enrich reference instruments.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Enrich use-case is not implemented in current codebase.

        Errors/Exceptions:
        - None.

        Side effects:
        - Emits informational log line.
        """
        log.info("enrich_ref_instruments job is not implemented in current repository state")

    async def _run_rest_insurance_job(self) -> None:
        """
        Run S3: periodic insurance rest catchup and bootstrap for empty instruments.

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
        - Executes REST reads and raw writes for enabled instruments.
        """
        instruments = await asyncio.to_thread(self._instrument_reader.list_enabled_tradable)
        if not instruments:
            return

        now_floor = UtcTimestamp(floor_to_minute_utc(self._clock.now().value))
        lookback_start = UtcTimestamp(
            now_floor.value - timedelta(minutes=self._config.ingestion.tail_lookback_minutes)
        )
        semaphore = asyncio.Semaphore(self._config.ingestion.rest_concurrency_instruments)

        async def _fill_one(instrument) -> None:
            async with semaphore:
                bounds = await asyncio.to_thread(self._index_reader.bounds, instrument)
                if bounds is None:
                    if self._bootstrap_start.value >= now_floor.value:
                        return
                    task = RestFillTask(
                        instrument_id=instrument,
                        time_range=TimeRange(start=self._bootstrap_start, end=now_floor),
                        reason="scheduler_bootstrap",
                    )
                else:
                    if lookback_start.value >= now_floor.value:
                        return
                    task = RestFillTask(
                        instrument_id=instrument,
                        time_range=TimeRange(start=lookback_start, end=now_floor),
                        reason="scheduler_tail",
                    )
                await asyncio.to_thread(self._rest_fill_use_case.run, task)

        await asyncio.gather(*[asyncio.create_task(_fill_one(inst)) for inst in instruments])


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
    clickhouse_client = _clickhouse_client(clickhouse_settings)
    gateway = ClickHouseConnectGateway(clickhouse_client)

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
    )

    return MarketDataSchedulerApp(
        config=config,
        whitelist_path=whitelist_path,
        seed_use_case=seed_use_case,
        sync_use_case=sync_use_case,
        instrument_reader=instrument_reader,
        index_reader=index_reader,
        rest_fill_use_case=rest_fill_use_case,
        metrics=MarketDataSchedulerMetrics(),
        metrics_port=metrics_port,
    )
