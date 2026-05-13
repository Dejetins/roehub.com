from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from trading.contexts.backtest.adapters.outbound import (
    DEFAULT_LAZY_TRADES_CACHE_ROOT,
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    LocalFileBacktestLazyTradesCache,
    PostgresBacktestJobLeaseRepository,
    PostgresBacktestJobRepository,
    PostgresBacktestLazyTradesMaterializationRepository,
    PsycopgBacktestPostgresGateway,
    YamlBacktestGridDefaultsProvider,
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
    DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
    BacktestSchedulingClass,
)
from trading.contexts.backtest.application.services.v2.lazy_trades_detail import (
    BacktestLazyTradesDetailService,
)
from trading.contexts.backtest.application.services.v2.preflight import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest.application.use_cases import (
    BacktestJobWorkerResult,
    BacktestJobWorkerUseCase,
    BacktestLazyTradesMaterializationWorkerResult,
    BacktestLazyTradesMaterializationWorkerUseCase,
)
from trading.contexts.backtest.domain.entities import BacktestJob
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)

from .child_process import BacktestChildProcessExecutor

log = logging.getLogger(__name__)

_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_TASK_KIND_FULL_JOB = "full_job"
_TASK_KIND_LAZY_DETAIL = "lazy_detail"
_SCHEDULING_CLASS_LIGHT_CANDIDATE: BacktestSchedulingClass = "light_candidate"
_SCHEDULING_CLASS_HEAVY: BacktestSchedulingClass = "heavy"


@dataclass(frozen=True, slots=True)
class BacktestJobRunnerRuntimeConfig:
    enabled: bool
    concurrency: int
    light_concurrency: int
    heavy_concurrency: int
    poll_interval_seconds: float
    empty_backoff_seconds: float
    lease_seconds: int
    heartbeat_interval_seconds: float
    max_jobs_per_process: int
    metrics_port: int
    child_timeout_seconds: float
    light_max_estimated_combinations: int
    light_max_actual_combinations: int
    lazy_detail_anti_starvation_limit: int = 5
    full_job_anti_starvation_limit: int = 4

    def __post_init__(self) -> None:
        if self.concurrency != 1:
            raise ValueError(
                "ROEHUB_BACKTEST_RUNNER_CONCURRENCY=1 is required; use "
                "ROEHUB_BACKTEST_LIGHT_CONCURRENCY for full-job child lanes"
            )
        if self.light_concurrency <= 0:
            raise ValueError("ROEHUB_BACKTEST_LIGHT_CONCURRENCY must be > 0")
        if self.light_concurrency > 3:
            raise ValueError(
                "ROEHUB_BACKTEST_LIGHT_CONCURRENCY > 3 requires separate benchmark evidence"
            )
        if self.heavy_concurrency != 1:
            raise ValueError("ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1 is required for v1")
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
        if self.child_timeout_seconds <= 0:
            raise ValueError("ROEHUB_BACKTEST_CHILD_TIMEOUT_SECONDS must be > 0")
        if self.light_max_estimated_combinations <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_LIGHT_MAX_ESTIMATED_COMBINATIONS must be > 0"
            )
        if self.light_max_actual_combinations <= 0:
            raise ValueError("ROEHUB_BACKTEST_LIGHT_MAX_ACTUAL_COMBINATIONS must be > 0")
        if self.lazy_detail_anti_starvation_limit <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_RUNNER_LAZY_DETAIL_ANTI_STARVATION_LIMIT must be > 0"
            )
        if self.full_job_anti_starvation_limit <= 0:
            raise ValueError("ROEHUB_BACKTEST_FULL_JOB_ANTI_STARVATION_LIMIT must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestRunnerTaskResult:
    task_kind: str
    claimed: bool
    status: str = "empty"
    paid_level: str = "unknown"
    scheduling_class: str = "none"
    created_at: datetime | None = None
    started_at: datetime | None = None
    lease_lost: bool = False
    cache_status: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestRunnerTaskLaunch:
    task_kind: str
    scheduling_class: str
    run: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class BacktestRunnerActiveTask:
    task_kind: str
    scheduling_class: str
    started_at_monotonic: float


@dataclass(slots=True)
class BacktestRunnerTaskScheduler:
    light_full_job_worker: BacktestJobWorkerUseCase
    heavy_full_job_worker: BacktestJobWorkerUseCase
    lazy_detail_worker: BacktestLazyTradesMaterializationWorkerUseCase
    light_concurrency: int = 2
    heavy_concurrency: int = 1
    lazy_detail_anti_starvation_limit: int = 5
    full_job_anti_starvation_limit: int = 4
    _lazy_detail_streak: int = 0
    _try_heavy_next: bool = True
    _light_batch_launched: int = 0
    _consecutive_light_claims: int = 0
    _full_empty_rounds: int = 0

    def next_launch(
        self,
        *,
        active_light: int,
        active_heavy: int,
        active_lazy: int,
    ) -> BacktestRunnerTaskLaunch | None:
        if active_heavy > 0:
            return None
        if active_light > 0:
            if (
                active_light < self.light_concurrency
                and self._light_batch_launched < self.light_concurrency
                and self._consecutive_light_claims < self.full_job_anti_starvation_limit
            ):
                self._light_batch_launched += 1
                return BacktestRunnerTaskLaunch(
                    task_kind=_TASK_KIND_FULL_JOB,
                    scheduling_class=_SCHEDULING_CLASS_LIGHT_CANDIDATE,
                    run=self.light_full_job_worker.run_next,
                )
            return None
        if active_lazy > 0:
            return None

        self._light_batch_launched = 0
        if (
            self._try_heavy_next
            or self._consecutive_light_claims >= self.full_job_anti_starvation_limit
        ):
            self._try_heavy_next = False
            return BacktestRunnerTaskLaunch(
                task_kind=_TASK_KIND_FULL_JOB,
                scheduling_class=_SCHEDULING_CLASS_HEAVY,
                run=self.heavy_full_job_worker.run_next,
            )
        if self._full_empty_rounds >= 2:
            return BacktestRunnerTaskLaunch(
                task_kind=_TASK_KIND_LAZY_DETAIL,
                scheduling_class="none",
                run=self.lazy_detail_worker.run_next,
            )
        self._light_batch_launched = 1
        return BacktestRunnerTaskLaunch(
            task_kind=_TASK_KIND_FULL_JOB,
            scheduling_class=_SCHEDULING_CLASS_LIGHT_CANDIDATE,
            run=self.light_full_job_worker.run_next,
        )

    def record_result(
        self,
        *,
        scheduling_class: str,
        result: BacktestRunnerTaskResult,
    ) -> None:
        if result.task_kind == _TASK_KIND_LAZY_DETAIL:
            if result.claimed:
                self._lazy_detail_streak += 1
            return
        self._lazy_detail_streak = 0
        if scheduling_class == _SCHEDULING_CLASS_HEAVY:
            if result.claimed:
                self._try_heavy_next = True
                self._consecutive_light_claims = 0
                self._full_empty_rounds = 0
            else:
                self._full_empty_rounds += 1
            return
        if result.claimed:
            self._consecutive_light_claims += 1
            self._full_empty_rounds = 0
            if self._consecutive_light_claims >= self.full_job_anti_starvation_limit:
                self._try_heavy_next = True
        else:
            self._try_heavy_next = True
            self._full_empty_rounds += 1


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
        self.active_children = Gauge(
            "backtest_runner_active_children",
            "Backtest runner active child process count",
            ("scheduling_class",),
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
        self.lazy_trades_cache_total = Counter(
            "backtest_lazy_trades_cache_total",
            "Backtest lazy trades cache outcomes observed by runner",
            ("status",),
            registry=self.registry,
        )
        self.active.labels(task_kind=_TASK_KIND_FULL_JOB).set(0)
        self.active.labels(task_kind=_TASK_KIND_LAZY_DETAIL).set(0)
        self.active_children.labels(scheduling_class="light").set(0)
        self.active_children.labels(scheduling_class="heavy").set(0)
        self.last_success_unixtime.labels(task_kind=_TASK_KIND_FULL_JOB).set(0)
        self.last_success_unixtime.labels(task_kind=_TASK_KIND_LAZY_DETAIL).set(0)

    def observe_result(
        self,
        *,
        result: (
            BacktestRunnerTaskResult
            | BacktestJobWorkerResult
            | BacktestLazyTradesMaterializationWorkerResult
        ),
        duration_seconds: float,
    ) -> None:
        normalized = _coerce_task_result(result=result)
        if not normalized.claimed:
            return
        status = "lease_lost" if normalized.lease_lost else normalized.status
        self.tasks_claimed_total.labels(
            task_kind=normalized.task_kind,
            paid_level=normalized.paid_level,
        ).inc()
        self.tasks_finished_total.labels(
            task_kind=normalized.task_kind,
            status=status,
        ).inc()
        self.task_duration_seconds.labels(
            task_kind=normalized.task_kind,
            status=status,
        ).observe(max(duration_seconds, 0.0))
        if normalized.created_at is not None and normalized.started_at is not None:
            queue_wait_seconds = (
                normalized.started_at - normalized.created_at
            ).total_seconds()
            self.queue_wait_seconds.labels(
                task_kind=normalized.task_kind,
                paid_level=normalized.paid_level,
            ).observe(max(queue_wait_seconds, 0.0))
        if normalized.lease_lost:
            self.lease_lost_total.labels(task_kind=normalized.task_kind).inc()
        if normalized.cache_status is not None:
            self.lazy_trades_cache_total.labels(status=normalized.cache_status).inc()
        if normalized.status in {"succeeded", "completed"}:
            self.last_success_unixtime.labels(task_kind=normalized.task_kind).set(
                datetime.now(UTC).timestamp()
            )

    def set_active_children(self, *, light: int, heavy: int) -> None:
        self.active.labels(task_kind=_TASK_KIND_FULL_JOB).set(light + heavy)
        self.active_children.labels(scheduling_class="light").set(light)
        self.active_children.labels(scheduling_class="heavy").set(heavy)


@dataclass(frozen=True, slots=True)
class BacktestJobRunnerApp:
    runtime_config: BacktestJobRunnerRuntimeConfig
    worker: BacktestRunnerTaskScheduler
    metrics: BacktestJobRunnerMetrics
    metrics_port: int

    async def run(self, stop_event: asyncio.Event) -> None:
        start_http_server(self.metrics_port, registry=self.metrics.registry)
        log.info("backtest-job-runner metrics server started on port %s", self.metrics_port)
        processed_jobs = 0
        loop = asyncio.get_running_loop()
        active: dict[asyncio.Future[Any], BacktestRunnerActiveTask] = {}
        max_workers = (
            self.runtime_config.light_concurrency
            + self.runtime_config.heavy_concurrency
            + 1
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while not stop_event.is_set():
                processed_jobs += self._reap_finished_tasks(active=active)
                light_active, heavy_active, lazy_active = _active_counts(active=active)
                self.metrics.set_active_children(light=light_active, heavy=heavy_active)
                if (
                    processed_jobs >= self.runtime_config.max_jobs_per_process
                    and not active
                ):
                    log.info(
                        "backtest-job-runner reached parent max task accounting: %s",
                        self.runtime_config.max_jobs_per_process,
                    )
                    return

                launched = self._launch_available_tasks(
                    loop=loop,
                    pool=pool,
                    active=active,
                )
                if active:
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=self.runtime_config.poll_interval_seconds,
                        )
                    except TimeoutError:
                        continue
                else:
                    wait_seconds = (
                        self.runtime_config.poll_interval_seconds
                        if launched
                        else self.runtime_config.empty_backoff_seconds
                    )
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
                    except TimeoutError:
                        continue

    def _launch_available_tasks(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        pool: ThreadPoolExecutor,
        active: dict[asyncio.Future[Any], BacktestRunnerActiveTask],
    ) -> bool:
        launched = False
        while True:
            light_active, heavy_active, lazy_active = _active_counts(active=active)
            launch = self.worker.next_launch(
                active_light=light_active,
                active_heavy=heavy_active,
                active_lazy=lazy_active,
            )
            if launch is None:
                return launched
            future = loop.run_in_executor(pool, launch.run)
            active[future] = BacktestRunnerActiveTask(
                task_kind=launch.task_kind,
                scheduling_class=launch.scheduling_class,
                started_at_monotonic=time.perf_counter(),
            )
            launched = True
            log.info(
                "backtest runner launched task: kind=%s scheduling_class=%s",
                launch.task_kind,
                launch.scheduling_class,
            )
            if launch.scheduling_class == _SCHEDULING_CLASS_HEAVY:
                return launched

    def _reap_finished_tasks(
        self,
        *,
        active: dict[asyncio.Future[Any], BacktestRunnerActiveTask],
    ) -> int:
        processed = 0
        for future, meta in list(active.items()):
            if not future.done():
                continue
            active.pop(future)
            duration_seconds = time.perf_counter() - meta.started_at_monotonic
            try:
                raw_result = future.result()
                normalized_result = _coerce_task_result(
                    result=raw_result,
                    scheduling_class=meta.scheduling_class,
                )
            except Exception:  # noqa: BLE001
                self.metrics.tasks_finished_total.labels(
                    task_kind=meta.task_kind,
                    status="runner_error",
                ).inc()
                log.exception(
                    "backtest runner task failed before worker result: kind=%s "
                    "scheduling_class=%s",
                    meta.task_kind,
                    meta.scheduling_class,
                )
                normalized_result = BacktestRunnerTaskResult(
                    task_kind=meta.task_kind,
                    scheduling_class=meta.scheduling_class,
                    claimed=False,
                    status="runner_error",
                )
            self.metrics.observe_result(
                result=normalized_result,
                duration_seconds=duration_seconds,
            )
            self.worker.record_result(
                scheduling_class=meta.scheduling_class,
                result=normalized_result,
            )
            if normalized_result.claimed:
                processed += 1
                log.info(
                    "backtest runner task processed: kind=%s scheduling_class=%s "
                    "claimed=%s status=%s lease_lost=%s",
                    normalized_result.task_kind,
                    normalized_result.scheduling_class,
                    normalized_result.claimed,
                    normalized_result.status,
                    normalized_result.lease_lost,
                )
        return processed


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
        light_concurrency=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_LIGHT_CONCURRENCY",
            default=2,
        ),
        heavy_concurrency=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_HEAVY_CONCURRENCY",
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
        child_timeout_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_CHILD_TIMEOUT_SECONDS",
            default=21_600.0,
        ),
        light_max_estimated_combinations=_env_int_with_aliases(
            environ=environ,
            key="ROEHUB_BACKTEST_LIGHT_MAX_ESTIMATED_COMBINATIONS",
            aliases=("ROEHUB_BACKTEST_LIGHT_MAX_COMBINATIONS",),
            default=DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
        ),
        light_max_actual_combinations=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_LIGHT_MAX_ACTUAL_COMBINATIONS",
            default=DEFAULT_LIGHT_ACTUAL_COMBINATIONS,
        ),
        lazy_detail_anti_starvation_limit=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_RUNNER_LAZY_DETAIL_ANTI_STARVATION_LIMIT",
            default=5,
        ),
        full_job_anti_starvation_limit=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_FULL_JOB_ANTI_STARVATION_LIMIT",
            default=4,
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
    materialization_repository = PostgresBacktestLazyTradesMaterializationRepository(
        gateway=postgres_gateway
    )
    preflight_service = BacktestPreflightService(
        defaults_provider=defaults_provider,
        artifact_context_resolver=artifact_context_resolver,
        runtime_config=backtest_runtime_config,
    )
    light_executor = BacktestChildProcessExecutor(
        environ=effective_environ,
        scheduling_class=_SCHEDULING_CLASS_LIGHT_CANDIDATE,
        light_max_actual_combinations=(
            effective_runtime_config.light_max_actual_combinations
        ),
        timeout_seconds=effective_runtime_config.child_timeout_seconds,
    )
    heavy_executor = BacktestChildProcessExecutor(
        environ=effective_environ,
        scheduling_class=_SCHEDULING_CLASS_HEAVY,
        light_max_actual_combinations=(
            effective_runtime_config.light_max_actual_combinations
        ),
        timeout_seconds=effective_runtime_config.child_timeout_seconds,
    )
    light_full_job_worker = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=job_repository,
        preflight_service=preflight_service,
        executor=light_executor,
        lease_seconds=effective_runtime_config.lease_seconds,
        heartbeat_interval_seconds=effective_runtime_config.heartbeat_interval_seconds,
        locked_by=_build_locked_by(),
        scheduling_classes=("light_candidate", "light"),
    )
    heavy_full_job_worker = BacktestJobWorkerUseCase(
        lease_repository=lease_repository,
        job_repository=job_repository,
        preflight_service=preflight_service,
        executor=heavy_executor,
        lease_seconds=effective_runtime_config.lease_seconds,
        heartbeat_interval_seconds=effective_runtime_config.heartbeat_interval_seconds,
        locked_by=_build_locked_by(),
        scheduling_classes=("heavy",),
    )
    prepare_pools = BacktestPreparePoolsService(
        artifact_array_loader=artifact_array_loader,
        defaults_provider=defaults_provider,
    )
    lazy_trades_service = BacktestLazyTradesDetailService(
        prepare_pools=prepare_pools,
        tp_sl_hit_times=BacktestTpSlHitTimesService(
            artifact_array_loader=artifact_array_loader
        ),
        cache=LocalFileBacktestLazyTradesCache(
            root=Path(
                effective_environ.get(
                    "ROEHUB_BACKTEST_TRADES_CACHE_ROOT",
                    str(DEFAULT_LAZY_TRADES_CACHE_ROOT),
                )
            )
        ),
    )
    lazy_detail_worker = BacktestLazyTradesMaterializationWorkerUseCase(
        materialization_repository=materialization_repository,
        job_repository=job_repository,
        lazy_trades_service=lazy_trades_service,
        lease_seconds=effective_runtime_config.lease_seconds,
        heartbeat_interval_seconds=effective_runtime_config.heartbeat_interval_seconds,
        locked_by=_build_locked_by(),
    )
    scheduler = BacktestRunnerTaskScheduler(
        light_full_job_worker=light_full_job_worker,
        heavy_full_job_worker=heavy_full_job_worker,
        lazy_detail_worker=lazy_detail_worker,
        light_concurrency=effective_runtime_config.light_concurrency,
        heavy_concurrency=effective_runtime_config.heavy_concurrency,
        lazy_detail_anti_starvation_limit=(
            effective_runtime_config.lazy_detail_anti_starvation_limit
        ),
        full_job_anti_starvation_limit=(
            effective_runtime_config.full_job_anti_starvation_limit
        ),
    )
    return BacktestJobRunnerApp(
        runtime_config=effective_runtime_config,
        worker=scheduler,
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


def _coerce_task_result(
    *,
    result: (
        BacktestRunnerTaskResult
        | BacktestJobWorkerResult
        | BacktestLazyTradesMaterializationWorkerResult
    ),
    scheduling_class: str = "none",
) -> BacktestRunnerTaskResult:
    if isinstance(result, BacktestRunnerTaskResult):
        return result
    if isinstance(result, BacktestLazyTradesMaterializationWorkerResult):
        return _lazy_detail_task_result(result)
    return _full_job_task_result(result, scheduling_class=scheduling_class)


def _full_job_task_result(
    result: BacktestJobWorkerResult,
    *,
    scheduling_class: str = "none",
) -> BacktestRunnerTaskResult:
    job = result.job
    return BacktestRunnerTaskResult(
        task_kind=_TASK_KIND_FULL_JOB,
        claimed=result.claimed,
        status=result.status
        or ("lease_lost" if result.lease_lost or job is None else job.state),
        scheduling_class=scheduling_class,
        paid_level=_paid_level_from_job(job=job),
        created_at=None if job is None else job.created_at,
        started_at=None if job is None else job.started_at,
        lease_lost=result.lease_lost,
    )


def _lazy_detail_task_result(
    result: BacktestLazyTradesMaterializationWorkerResult,
) -> BacktestRunnerTaskResult:
    task = result.task
    return BacktestRunnerTaskResult(
        task_kind=_TASK_KIND_LAZY_DETAIL,
        claimed=result.claimed,
        status="lease_lost" if result.lease_lost or task is None else task.status,
        scheduling_class="none",
        paid_level="unknown",
        created_at=None if task is None else task.created_at,
        started_at=None if task is None else task.started_at,
        lease_lost=result.lease_lost,
        cache_status=result.cache_status,
    )


def _active_counts(
    *,
    active: Mapping[asyncio.Future[Any], BacktestRunnerActiveTask],
) -> tuple[int, int, int]:
    light = 0
    heavy = 0
    lazy = 0
    for meta in active.values():
        if meta.task_kind == _TASK_KIND_LAZY_DETAIL:
            lazy += 1
        elif meta.scheduling_class == _SCHEDULING_CLASS_HEAVY:
            heavy += 1
        else:
            light += 1
    return light, heavy, lazy


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


def _env_int_with_aliases(
    *,
    environ: Mapping[str, str],
    key: str,
    aliases: tuple[str, ...],
    default: int,
) -> int:
    for candidate in (key, *aliases):
        raw_value = environ.get(candidate)
        if raw_value is not None and raw_value.strip():
            return _env_int(environ=environ, key=candidate, default=default)
    return default


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
    "BacktestRunnerTaskResult",
    "BacktestRunnerTaskScheduler",
    "build_backtest_job_runner_app",
    "load_backtest_job_runner_runtime_config",
]
