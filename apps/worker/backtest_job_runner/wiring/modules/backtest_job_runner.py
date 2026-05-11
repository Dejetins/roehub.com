from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from trading.contexts.backtest.adapters.outbound import (
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    PostgresBacktestJobLeaseRepository,
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestComboPlanningService,
    BacktestNoRiskExactScoringService,
    BacktestPreflightService,
    BacktestPreparePoolsService,
    BacktestRuntimeConfig,
    BacktestRuntimeJobOrchestrationService,
    BacktestTpSlExactScoringService,
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobWorkerResult,
    BacktestJobWorkerUseCase,
)
from trading.contexts.backtest.domain.entities import BacktestJob
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)

log = logging.getLogger(__name__)

_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_TASK_KIND_FULL_JOB = "full_job"


@dataclass(frozen=True, slots=True)
class BacktestJobRunnerRuntimeConfig:
    enabled: bool
    concurrency: int
    poll_interval_seconds: float
    empty_backoff_seconds: float
    lease_seconds: int
    heartbeat_interval_seconds: float
    max_jobs_per_process: int
    metrics_port: int

    def __post_init__(self) -> None:
        if self.concurrency != 1:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_CONCURRENCY=1 is required for R2")
        if self.poll_interval_seconds <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_POLL_INTERVAL_SECONDS must be > 0")
        if self.empty_backoff_seconds <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_EMPTY_BACKOFF_SECONDS must be > 0")
        if self.lease_seconds <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_LEASE_SECONDS must be > 0")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_HEARTBEAT_INTERVAL_SECONDS must be > 0")
        if self.heartbeat_interval_seconds >= self.lease_seconds:
            raise ValueError(
                "ROEHUB_BACKTEST_RUNNER_HEARTBEAT_INTERVAL_SECONDS must be less than "
                "ROEHUB_BACKTEST_RUNNER_LEASE_SECONDS"
            )
        if self.max_jobs_per_process <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_MAX_JOBS_PER_PROCESS must be > 0")
        if self.metrics_port <= 0:
            raise ValueError("ROEHUB_BACKTEST_RUNNER_METRICS_PORT must be > 0")


class BacktestJobRunnerMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.tasks_claimed_total = Counter(
            "backtest_runner_tasks_claimed_total",
            "Backtest runner claimed tasks count",
            ("task_kind", "paid_level"),
            registry=self.registry,
        )
        self.tasks_finished_total = Counter(
            "backtest_runner_tasks_finished_total",
            "Backtest runner finished tasks count",
            ("task_kind", "status"),
            registry=self.registry,
        )
        self.task_duration_seconds = Histogram(
            "backtest_runner_task_duration_seconds",
            "Backtest runner task duration in seconds",
            ("task_kind", "status"),
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 3600, 7200, 21600),
            registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "backtest_runner_queue_wait_seconds",
            "Backtest runner queue wait in seconds",
            ("task_kind", "paid_level"),
            buckets=(0.1, 1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
            registry=self.registry,
        )
        self.active = Gauge(
            "backtest_runner_active",
            "Backtest runner active task gauge",
            ("task_kind",),
            registry=self.registry,
        )
        self.lease_lost_total = Counter(
            "backtest_runner_lease_lost_total",
            "Backtest runner lease lost count",
            ("task_kind",),
            registry=self.registry,
        )
        self.last_success_unixtime = Gauge(
            "backtest_runner_last_success_unixtime",
            "Backtest runner last successful task unix timestamp",
            ("task_kind",),
            registry=self.registry,
        )
        self.active.labels(task_kind=_TASK_KIND_FULL_JOB).set(0)
        self.last_success_unixtime.labels(task_kind=_TASK_KIND_FULL_JOB).set(0)

    def observe_result(
        self,
        *,
        result: BacktestJobWorkerResult,
        duration_seconds: float,
    ) -> None:
        if not result.claimed:
            return
        job = result.job
        paid_level = _paid_level_from_job(job=job)
        status = "lease_lost" if result.lease_lost or job is None else job.state
        self.tasks_claimed_total.labels(
            task_kind=_TASK_KIND_FULL_JOB,
            paid_level=paid_level,
        ).inc()
        self.tasks_finished_total.labels(task_kind=_TASK_KIND_FULL_JOB, status=status).inc()
        self.task_duration_seconds.labels(
            task_kind=_TASK_KIND_FULL_JOB,
            status=status,
        ).observe(max(duration_seconds, 0.0))
        if job is not None and job.started_at is not None:
            queue_wait_seconds = (job.started_at - job.created_at).total_seconds()
            self.queue_wait_seconds.labels(
                task_kind=_TASK_KIND_FULL_JOB,
                paid_level=paid_level,
            ).observe(max(queue_wait_seconds, 0.0))
        if result.lease_lost:
            self.lease_lost_total.labels(task_kind=_TASK_KIND_FULL_JOB).inc()
        if job is not None and job.state == "succeeded":
            self.last_success_unixtime.labels(task_kind=_TASK_KIND_FULL_JOB).set(
                datetime.now(UTC).timestamp()
            )


@dataclass(frozen=True, slots=True)
class BacktestJobRunnerApp:
    runtime_config: BacktestJobRunnerRuntimeConfig
    worker: BacktestJobWorkerUseCase
    metrics: BacktestJobRunnerMetrics
    metrics_port: int

    async def run(self, stop_event: asyncio.Event) -> None:
        start_http_server(self.metrics_port, registry=self.metrics.registry)
        log.info("backtest-job-runner metrics server started on port %s", self.metrics_port)
        processed_jobs = 0
        while not stop_event.is_set():
            self.metrics.active.labels(task_kind=_TASK_KIND_FULL_JOB).set(1)
            started = time.perf_counter()
            try:
                result = self.worker.run_next()
            except Exception:  # noqa: BLE001
                self.metrics.tasks_finished_total.labels(
                    task_kind=_TASK_KIND_FULL_JOB,
                    status="runner_error",
                ).inc()
                log.exception("backtest-job-runner iteration failed")
                result = BacktestJobWorkerResult(job=None, claimed=False)
            finally:
                self.metrics.active.labels(task_kind=_TASK_KIND_FULL_JOB).set(0)

            duration_seconds = time.perf_counter() - started
            self.metrics.observe_result(result=result, duration_seconds=duration_seconds)
            if result.claimed:
                processed_jobs += 1
                log.info(
                    "backtest job processed: claimed=%s state=%s lease_lost=%s",
                    result.claimed,
                    None if result.job is None else result.job.state,
                    result.lease_lost,
                )
                if processed_jobs >= self.runtime_config.max_jobs_per_process:
                    log.info(
                        "backtest-job-runner reached max jobs per process: %s",
                        self.runtime_config.max_jobs_per_process,
                    )
                    return
                wait_seconds = self.runtime_config.poll_interval_seconds
            else:
                wait_seconds = self.runtime_config.empty_backoff_seconds

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            except TimeoutError:
                continue


def load_backtest_job_runner_runtime_config(
    *,
    environ: Mapping[str, str],
) -> BacktestJobRunnerRuntimeConfig:
    return BacktestJobRunnerRuntimeConfig(
        enabled=_env_bool(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_ENABLED",
            default=True,
        ),
        concurrency=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_CONCURRENCY",
            default=1,
        ),
        poll_interval_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_POLL_INTERVAL_SECONDS",
            default=2.0,
        ),
        empty_backoff_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_EMPTY_BACKOFF_SECONDS",
            default=5.0,
        ),
        lease_seconds=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_LEASE_SECONDS",
            default=120,
        ),
        heartbeat_interval_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_HEARTBEAT_INTERVAL_SECONDS",
            default=30.0,
        ),
        max_jobs_per_process=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_MAX_JOBS_PER_PROCESS",
            default=10,
        ),
        metrics_port=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_METRICS_PORT",
            default=9204,
        ),
    )


def build_backtest_job_runner_app(
    *,
    environ: Mapping[str, str],
    runtime_config: BacktestJobRunnerRuntimeConfig | None = None,
    metrics_port: int | None = None,
) -> BacktestJobRunnerApp:
    effective_runtime_config = runtime_config or load_backtest_job_runner_runtime_config(
        environ=environ
    )
    effective_environ = _with_local_dev_default(environ=environ)
    postgres_dsn = effective_environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    if not postgres_dsn:
        raise ValueError(f"{_STRATEGY_PG_DSN_KEY} is required for backtest-job-runner")

    artifact_config_path = resolve_backtest_artifacts_config_path(environ=effective_environ)
    artifact_config = load_backtest_artifacts_runtime_config(artifact_config_path)
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=effective_environ,
        artifact_config_path=artifact_config_path,
    )
    artifact_path_builder = BacktestArtifactPathBuilderV2(
        root=artifact_config.artifact_root_path()
    )
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=artifact_path_builder)
    artifact_context_resolver = FilesystemBacktestArtifactContextResolver(
        artifact_loader=artifact_loader
    )
    artifact_array_loader = FilesystemBacktestArtifactArrayLoader(
        artifact_loader=artifact_loader
    )
    backtest_runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=artifact_config.hit_times_grid.tp_levels_pct,
        hit_times_sl_levels_pct=artifact_config.hit_times_grid.sl_levels_pct,
        artifact_config_hash=build_backtest_artifacts_runtime_config_hash(
            config=artifact_config
        ),
    )
    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    job_repository = PostgresBacktestJobRepository(gateway=postgres_gateway)
    lease_repository = PostgresBacktestJobLeaseRepository(gateway=postgres_gateway)
    preflight_service = BacktestPreflightService(
        defaults_provider=defaults_provider,
        artifact_context_resolver=artifact_context_resolver,
        runtime_config=backtest_runtime_config,
    )
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
    )
    executor = BacktestRuntimeJobOrchestrationService(
        prepare_pools=prepare_pools,
        combo_planning=BacktestComboPlanningService(),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=BacktestTpSlHitTimesService(
            artifact_array_loader=artifact_array_loader
        ),
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=artifact_array_loader,
    )
    worker = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=job_repository,
        preflight_service=preflight_service,
        executor=executor,
        lease_seconds=effective_runtime_config.lease_seconds,
        heartbeat_interval_seconds=effective_runtime_config.heartbeat_interval_seconds,
        locked_by=_build_locked_by(),
    )
    return BacktestJobRunnerApp(
        runtime_config=effective_runtime_config,
        worker=worker,
        metrics=BacktestJobRunnerMetrics(),
        metrics_port=metrics_port or effective_runtime_config.metrics_port,
    )


def _with_local_dev_default(*, environ: Mapping[str, str]) -> Mapping[str, str]:
    if environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG", "").strip():
        return environ
    if environ.get("ROEHUB_ENV", "").strip():
        return environ
    return {**environ, "ROEHUB_ENV": "dev"}


def _build_locked_by() -> str:
    hostname = socket.gethostname().strip() or "unknown-host"
    return f"backtest-job-runner:{hostname}-{os.getpid()}"


def _paid_level_from_job(*, job: BacktestJob | None) -> str:
    if job is None:
        return "unknown"
    admission = job.request_json.get("admission")
    if isinstance(admission, Mapping):
        paid_level = admission.get("paid_level")
        if isinstance(paid_level, str) and paid_level.strip():
            return paid_level.strip()
    paid_level = job.request_json.get("paid_level")
    if isinstance(paid_level, str) and paid_level.strip():
        return paid_level.strip()
    return "unknown"


def _env_bool(*, environ: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = environ.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean literal")


def _env_int(*, environ: Mapping[str, str], key: str, default: int) -> int:
    raw_value = environ.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value.strip())
    except ValueError as error:
        raise ValueError(f"{key} must be integer") from error
    return value


def _env_float(*, environ: Mapping[str, str], key: str, default: float) -> float:
    raw_value = environ.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value.strip())
    except ValueError as error:
        raise ValueError(f"{key} must be numeric") from error
    return value


__all__ = [
    "BacktestJobRunnerApp",
    "BacktestJobRunnerMetrics",
    "BacktestJobRunnerRuntimeConfig",
    "build_backtest_job_runner_app",
    "load_backtest_job_runner_runtime_config",
]
