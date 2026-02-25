from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from apps.api.dto.backtests import BacktestsPostRequest, build_backtest_run_request
from apps.api.wiring.modules.indicators import (
    build_indicators_candle_feed,
    build_indicators_compute,
)
from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobLeaseRepository,
    PostgresBacktestJobRepository,
    PostgresBacktestJobResultsRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    load_backtest_runtime_config,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobLeaseRepository,
    BacktestJobRequestDecoder,
)
from trading.contexts.backtest.application.services import BacktestCandleTimelineBuilder
from trading.contexts.backtest.application.use_cases import (
    BacktestJobRunReportV1,
    RunBacktestJobRunnerV1,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.platform.time.system_clock import SystemClock

_LOG = logging.getLogger(__name__)
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


@dataclass(frozen=True, slots=True)
class _ApiBacktestJobRequestDecoderV1(BacktestJobRequestDecoder):
    """
    Decode persisted Backtest job payload using canonical API request DTO contract.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/ports/backtest_job_request_decoder.py
      - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
    """

    def decode(self, *, payload: Mapping[str, Any]):
        """
        Convert persisted `request_json` mapping into `RunBacktestRequest`.

        Args:
            payload: Persisted request payload mapping.
        Returns:
            RunBacktestRequest: Decoded application request DTO.
        Assumptions:
            Payload follows strict `BacktestsPostRequest` shape.
        Raises:
            ValueError: If payload validation or conversion fails.
        Side Effects:
            None.
        """
        request = BacktestsPostRequest.model_validate(payload)
        return build_backtest_run_request(request=request)


class BacktestJobRunnerMetrics:
    """
    Prometheus metrics bundle for Backtest job-runner worker process.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
      - apps/worker/backtest_job_runner/main/main.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        """
        Register worker metrics in provided or default Prometheus registry.

        Args:
            registry: Optional registry for tests or custom process setups.
        Returns:
            None.
        Assumptions:
            Metrics names are stable for EPIC-10 observability contract.
        Raises:
            ValueError: Propagated by Prometheus client on duplicate metric names.
        Side Effects:
            Registers metrics in target registry.
        """
        self.registry = registry or REGISTRY
        self.claim_total = Counter(
            "backtest_job_runner_claim_total",
            "Backtest job-runner claimed jobs total",
            registry=self.registry,
        )
        self.succeeded_total = Counter(
            "backtest_job_runner_succeeded_total",
            "Backtest job-runner succeeded jobs total",
            registry=self.registry,
        )
        self.failed_total = Counter(
            "backtest_job_runner_failed_total",
            "Backtest job-runner failed jobs total",
            registry=self.registry,
        )
        self.cancelled_total = Counter(
            "backtest_job_runner_cancelled_total",
            "Backtest job-runner cancelled jobs total",
            registry=self.registry,
        )
        self.lease_lost_total = Counter(
            "backtest_job_runner_lease_lost_total",
            "Backtest job-runner lease-lost events total",
            registry=self.registry,
        )
        self.job_duration_seconds = Histogram(
            "backtest_job_runner_job_duration_seconds",
            "Backtest job-runner claimed job duration in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 300.0, 900.0),
            registry=self.registry,
        )
        self.stage_duration_seconds = Histogram(
            "backtest_job_runner_stage_duration_seconds",
            "Backtest job-runner stage duration in seconds",
            labelnames=("stage",),
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 300.0, 900.0),
            registry=self.registry,
        )
        self.active_claimed_jobs = Gauge(
            "backtest_job_runner_active_claimed_jobs",
            "Active claimed backtest jobs in current worker process",
            registry=self.registry,
        )

    def observe_job(self, *, report: BacktestJobRunReportV1, duration_seconds: float) -> None:
        """
        Observe one claimed job processing result and update counters/histograms.

        Args:
            report: Job processing report from orchestrator.
            duration_seconds: Total claimed job processing duration.
        Returns:
            None.
        Assumptions:
            Report status is one of `succeeded|failed|cancelled|lease_lost`.
        Raises:
            None.
        Side Effects:
            Updates Prometheus counters and histograms.
        """
        self.job_duration_seconds.observe(max(duration_seconds, 0.0))
        self.stage_duration_seconds.labels(stage="stage_a").observe(
            report.stage_a_duration_seconds
        )
        self.stage_duration_seconds.labels(stage="stage_b").observe(
            report.stage_b_duration_seconds
        )
        self.stage_duration_seconds.labels(stage="finalizing").observe(
            report.finalizing_duration_seconds
        )

        if report.status == "succeeded":
            self.succeeded_total.inc()
            return
        if report.status == "failed":
            self.failed_total.inc()
            return
        if report.status == "cancelled":
            self.cancelled_total.inc()
            return
        self.lease_lost_total.inc()


@dataclass(frozen=True, slots=True)
class BacktestJobRunnerApp:
    """
    Runtime claim/poll loop wrapper for EPIC-10 Backtest job-runner worker process.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - apps/worker/backtest_job_runner/main/main.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    claim_poll_seconds: float
    lease_seconds: int
    locked_by: str
    lease_repository: BacktestJobLeaseRepository
    runner_use_case: RunBacktestJobRunnerV1
    metrics: BacktestJobRunnerMetrics
    metrics_port: int

    def __post_init__(self) -> None:
        """
        Validate runtime loop invariants and scalar configuration bounds.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Single app instance handles one sequential claim-processing loop.
        Raises:
            ValueError: If one scalar or dependency is invalid.
        Side Effects:
            None.
        """
        if self.claim_poll_seconds <= 0:
            raise ValueError("BacktestJobRunnerApp.claim_poll_seconds must be > 0")
        if self.lease_seconds <= 0:
            raise ValueError("BacktestJobRunnerApp.lease_seconds must be > 0")
        if not self.locked_by.strip():
            raise ValueError("BacktestJobRunnerApp.locked_by must be non-empty")
        if self.metrics_port <= 0:
            raise ValueError("BacktestJobRunnerApp.metrics_port must be > 0")
        if self.lease_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestJobRunnerApp.lease_repository is required")
        if self.runner_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestJobRunnerApp.runner_use_case is required")
        if self.metrics is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestJobRunnerApp.metrics is required")

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Run claim loop until stop event is set.

        Args:
            stop_event: Cooperative shutdown signal from process entrypoint.
        Returns:
            None.
        Assumptions:
            Worker processes one claimed job at a time.
        Raises:
            Exception: Unexpected claim/process errors are logged and loop continues.
        Side Effects:
            Starts metrics HTTP server and performs storage/compute IO in loop.
        """
        start_http_server(self.metrics_port, registry=self.metrics.registry)
        _LOG.info(
            "event=metrics_started component=backtest-job-runner metrics_port=%s",
            self.metrics_port,
        )

        while not stop_event.is_set():
            claimed = None
            try:
                claimed = self.lease_repository.claim_next(
                    now=_utc_now(),
                    locked_by=self.locked_by,
                    lease_seconds=self.lease_seconds,
                )
            except Exception:  # noqa: BLE001
                _LOG.exception(
                    "event=claim_failed component=backtest-job-runner locked_by=%s",
                    self.locked_by,
                )
                await _wait_with_stop(
                    stop_event=stop_event,
                    timeout_seconds=self.claim_poll_seconds,
                )
                continue

            if claimed is None:
                await _wait_with_stop(
                    stop_event=stop_event,
                    timeout_seconds=self.claim_poll_seconds,
                )
                continue

            self.metrics.claim_total.inc()
            self.metrics.active_claimed_jobs.inc()
            started = perf_counter()
            try:
                report = self.runner_use_case.process_claimed_job(
                    job=claimed,
                    locked_by=self.locked_by,
                )
                self.metrics.observe_job(
                    report=report,
                    duration_seconds=max(perf_counter() - started, 0.0),
                )
                _LOG.info(
                    (
                        "event=job_processed component=backtest-job-runner job_id=%s "
                        "attempt=%s status=%s locked_by=%s"
                    ),
                    report.job_id,
                    report.attempt,
                    report.status,
                    self.locked_by,
                )
            except Exception:  # noqa: BLE001
                _LOG.exception(
                    (
                        "event=job_processing_crashed component=backtest-job-runner "
                        "job_id=%s attempt=%s locked_by=%s"
                    ),
                    claimed.job_id,
                    claimed.attempt,
                    self.locked_by,
                )
            finally:
                self.metrics.active_claimed_jobs.dec()


def build_backtest_job_runner_app(
    *,
    config_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
) -> BacktestJobRunnerApp:
    """
    Build fully wired Backtest job-runner worker app with fail-fast dependencies.

    Args:
        config_path: Path to `backtest.yaml` runtime config.
        environ: Process environment mapping.
        metrics_port: Prometheus metrics HTTP server port.
    Returns:
        BacktestJobRunnerApp: Ready-to-run worker app.
    Assumptions:
        Runtime environment includes `STRATEGY_PG_DSN` and valid ClickHouse settings.
    Raises:
        ValueError: If required environment/settings are missing or invalid.
        FileNotFoundError: If runtime configs cannot be loaded.
    Side Effects:
        Initializes compute/storage adapters and runtime dependencies.
    """
    if metrics_port <= 0:
        raise ValueError("build_backtest_job_runner_app metrics_port must be > 0")

    runtime_config = load_backtest_runtime_config(Path(config_path))
    strategy_pg_dsn = environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    if not strategy_pg_dsn:
        raise ValueError(f"{_STRATEGY_PG_DSN_KEY} is required for backtest job-runner worker")

    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=strategy_pg_dsn)
    job_repository = PostgresBacktestJobRepository(gateway=postgres_gateway)
    lease_repository = PostgresBacktestJobLeaseRepository(gateway=postgres_gateway)
    results_repository = PostgresBacktestJobResultsRepository(gateway=postgres_gateway)

    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )
    canonical_reader = ClickHouseCanonicalCandleReader(
        gateway=clickhouse_gateway,
        clock=SystemClock(),
        database=clickhouse_settings.database,
    )
    candle_feed = build_indicators_candle_feed(canonical_candle_reader=canonical_reader)
    candle_timeline_builder = BacktestCandleTimelineBuilder(candle_feed=candle_feed)
    indicator_compute = build_indicators_compute(environ=environ)
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(environ=environ)
    request_decoder = _ApiBacktestJobRequestDecoderV1()

    runner_use_case = RunBacktestJobRunnerV1(
        job_repository=job_repository,
        lease_repository=lease_repository,
        results_repository=results_repository,
        request_decoder=request_decoder,
        candle_timeline_builder=candle_timeline_builder,
        indicator_compute=indicator_compute,
        defaults_provider=defaults_provider,
        warmup_bars_default=runtime_config.warmup_bars_default,
        top_k_default=runtime_config.top_k_default,
        preselect_default=runtime_config.preselect_default,
        top_trades_n_default=runtime_config.reporting.top_trades_n_default,
        top_k_persisted_default=runtime_config.jobs.top_k_persisted_default,
        init_cash_quote_default=runtime_config.execution.init_cash_quote_default,
        fixed_quote_default=runtime_config.execution.fixed_quote_default,
        safe_profit_percent_default=runtime_config.execution.safe_profit_percent_default,
        slippage_pct_default=runtime_config.execution.slippage_pct_default,
        fee_pct_default_by_market_id=runtime_config.execution.fee_pct_default_by_market_id,
        max_variants_per_compute=runtime_config.guards.max_variants_per_compute,
        max_compute_bytes_total=runtime_config.guards.max_compute_bytes_total,
        lease_seconds=runtime_config.jobs.lease_seconds,
        heartbeat_seconds=runtime_config.jobs.heartbeat_seconds,
        snapshot_seconds=runtime_config.jobs.snapshot_seconds,
        snapshot_variants_step=runtime_config.jobs.snapshot_variants_step,
        max_numba_threads=runtime_config.cpu.max_numba_threads,
    )
    return BacktestJobRunnerApp(
        claim_poll_seconds=runtime_config.jobs.claim_poll_seconds,
        lease_seconds=runtime_config.jobs.lease_seconds,
        locked_by=_build_locked_by(),
        lease_repository=lease_repository,
        runner_use_case=runner_use_case,
        metrics=BacktestJobRunnerMetrics(),
        metrics_port=metrics_port,
    )


async def _wait_with_stop(*, stop_event: asyncio.Event, timeout_seconds: float) -> None:
    """
    Wait for stop event with timeout, returning early on cooperative shutdown.

    Args:
        stop_event: Shared shutdown event.
        timeout_seconds: Wait timeout in seconds.
    Returns:
        None.
    Assumptions:
        Timeout is positive and pre-validated by caller.
    Raises:
        None.
    Side Effects:
        Suspends current coroutine for up to timeout duration.
    """
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
    except TimeoutError:
        return


def _build_locked_by() -> str:
    """
    Build deterministic lease owner identifier in `<hostname>-<pid>` format.

    Args:
        None.
    Returns:
        str: Lease owner literal.
    Assumptions:
        Hostname and process id are stable for process lifetime.
    Raises:
        None.
    Side Effects:
        Reads OS hostname and process id.
    """
    hostname = socket.gethostname().strip() or "unknown-host"
    return f"{hostname}-{os.getpid()}"


def _utc_now() -> datetime:
    """
    Return timezone-aware UTC `now` timestamp for lease/progress operations.

    Args:
        None.
    Returns:
        datetime: Current UTC timestamp.
    Assumptions:
        Worker persists all lifecycle timestamps in UTC.
    Raises:
        None.
    Side Effects:
        None.
    """
    from datetime import timezone

    return datetime.now(timezone.utc)


__all__ = [
    "BacktestJobRunnerApp",
    "BacktestJobRunnerMetrics",
    "build_backtest_job_runner_app",
]
