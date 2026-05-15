from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast
from urllib.parse import urljoin

import httpx

from apps.api.wiring.modules.backtest import build_backtest_ai_configurator_use_cases
from apps.worker.backtest_ai_configurator.wiring.observability import (
    BacktestAiConfiguratorHealthState,
    BacktestAiConfiguratorMetrics,
    start_backtest_ai_configurator_http_server,
)
from trading.contexts.backtest.adapters.outbound import (
    LMStudioOpenAICompatibleAdapter,
    PsycopgBacktestPostgresGateway,
    load_backtest_ai_configurator_runtime_config,
    resolve_backtest_ai_configurator_config_path,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigGenerationLimiter,
    BacktestAiConfigWorkerUseCase,
)

log = logging.getLogger(__name__)
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorWorkerRuntimeConfig:
    enabled: bool
    poll_interval_seconds: float
    empty_backoff_seconds: float
    heartbeat_interval_seconds: float
    max_jobs_per_process: int
    metrics_port: int
    drain_mode: bool = False

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_AI_CONFIG_WORKER_POLL_INTERVAL_SECONDS must be > 0"
            )
        if self.empty_backoff_seconds <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_AI_CONFIG_WORKER_EMPTY_BACKOFF_SECONDS must be > 0"
            )
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_AI_CONFIG_WORKER_HEARTBEAT_INTERVAL_SECONDS must be > 0"
            )
        if self.max_jobs_per_process <= 0:
            raise ValueError(
                "ROEHUB_BACKTEST_AI_CONFIG_WORKER_MAX_JOBS_PER_PROCESS must be > 0"
            )
        if self.metrics_port <= 0:
            raise ValueError("ROEHUB_BACKTEST_AI_CONFIG_WORKER_METRICS_PORT must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorWorkerApp:
    runtime_config: BacktestAiConfiguratorWorkerRuntimeConfig
    worker: BacktestAiConfigWorkerUseCase
    metrics: BacktestAiConfiguratorMetrics
    health: BacktestAiConfiguratorHealthState
    metrics_port: int
    queue_snapshot: Callable[[], None]

    async def run(self, stop_event: asyncio.Event) -> None:
        self.health.mark_loop_started()
        server = start_backtest_ai_configurator_http_server(
            host="127.0.0.1",
            port=self.metrics_port,
            metrics=self.metrics,
            health=self.health,
            before_metrics=self.queue_snapshot,
        )
        processed_jobs = 0
        try:
            while not stop_event.is_set():
                self.health.mark_loop_tick()
                started = time.perf_counter()
                try:
                    result = self.worker.run_next()
                    duration_seconds = max(time.perf_counter() - started, 0.0)
                except Exception:  # noqa: BLE001
                    self.metrics.jobs_total.labels(
                        status="worker_error",
                        mode="unknown",
                        tier="unknown",
                        model_id=self.health.model_id,
                    ).inc()
                    log.exception("event=backtest_ai_config_worker_iteration_failed")
                    raise
                if result.claimed:
                    processed_jobs += 1
                    job = result.job
                    self.metrics.observe_result(
                        result=result,
                        duration_seconds=duration_seconds,
                    )
                    self.metrics.observe_attempts(
                        attempts=result.llm_attempts,
                        model_id=(job.model_id if job is not None else None)
                        or self.health.model_id,
                    )
                    log.info(
                        "event=backtest_ai_config_job_processed job_id=%s status=%s "
                        "lease_lost=%s skipped_source_page=%s duration_sec=%.3f",
                        None if job is None else job.job_id,
                        "lease_lost" if job is None else job.state,
                        result.lease_lost,
                        result.skipped_source_page,
                        duration_seconds,
                    )
                    if processed_jobs >= self.runtime_config.max_jobs_per_process:
                        log.info(
                            "backtest-ai-configurator-worker reached max jobs per process: %s",
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
        finally:
            self.health.mark_stopping()
            server.shutdown()


def load_backtest_ai_configurator_worker_runtime_config(
    *,
    environ: Mapping[str, str],
) -> BacktestAiConfiguratorWorkerRuntimeConfig:
    return BacktestAiConfiguratorWorkerRuntimeConfig(
        enabled=_env_bool(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_ENABLED",
            default=True,
        ),
        poll_interval_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_POLL_INTERVAL_SECONDS",
            default=2.0,
        ),
        empty_backoff_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_EMPTY_BACKOFF_SECONDS",
            default=5.0,
        ),
        heartbeat_interval_seconds=_env_float(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_HEARTBEAT_INTERVAL_SECONDS",
            default=15.0,
        ),
        max_jobs_per_process=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_MAX_JOBS_PER_PROCESS",
            default=10,
        ),
        metrics_port=_env_int(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_METRICS_PORT",
            default=9205,
        ),
        drain_mode=_env_bool(
            environ=environ,
            key="ROEHUB_BACKTEST_AI_CONFIG_WORKER_DRAIN_MODE",
            default=False,
        ),
    )


def build_backtest_ai_configurator_worker_app(
    *,
    environ: Mapping[str, str],
    runtime_config: BacktestAiConfiguratorWorkerRuntimeConfig | None = None,
    metrics_port: int | None = None,
) -> BacktestAiConfiguratorWorkerApp:
    if not environ.get(_STRATEGY_PG_DSN_KEY, "").strip():
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required for backtest-ai-configurator-worker"
        )
    effective_worker_config = (
        runtime_config
        or load_backtest_ai_configurator_worker_runtime_config(environ=environ)
    )
    ai_config_path = resolve_backtest_ai_configurator_config_path(environ=environ)
    ai_runtime_config = load_backtest_ai_configurator_runtime_config(ai_config_path)
    adapter = LMStudioOpenAICompatibleAdapter(config=ai_runtime_config.model)
    use_cases = build_backtest_ai_configurator_use_cases(
        environ=environ,
        llm_gateway=adapter,
    )
    if use_cases is None:
        raise ValueError("backtest AI configurator use cases are unavailable")
    metrics = BacktestAiConfiguratorMetrics()
    metrics.set_model_metadata(
        model_id=ai_runtime_config.model.model_id,
        loaded=True,
        runtime=ai_runtime_config.model.runtime,
        quantization=_quantization_from_model_path(ai_runtime_config.model.model_path),
    )
    postgres_gateway = PsycopgBacktestPostgresGateway(
        dsn=environ[_STRATEGY_PG_DSN_KEY].strip()
    )
    health = BacktestAiConfiguratorHealthState(
        config_loaded=True,
        model_id=ai_runtime_config.model.model_id,
        model_path=str(ai_runtime_config.model.model_path),
        drain_mode=effective_worker_config.drain_mode,
        readiness_checks=(
            _model_path_check(path=ai_runtime_config.model.model_path),
            _runtime_connection_check(base_url=ai_runtime_config.model.base_url),
            _postgres_queue_audit_check(gateway=postgres_gateway),
        ),
    )
    return BacktestAiConfiguratorWorkerApp(
        runtime_config=effective_worker_config,
        worker=BacktestAiConfigWorkerUseCase(
            job_repository=use_cases.jobs.repository,
            lease_repository=use_cases.lease_repository,
            pipeline=use_cases.pipeline,
            lease_seconds=use_cases.runtime_config.queue.lease_seconds,
            max_attempts=use_cases.runtime_config.queue.repair_attempts + 1,
            heartbeat_interval_seconds=(
                effective_worker_config.heartbeat_interval_seconds
            ),
            locked_by=_build_locked_by(),
            generation_limiter=BacktestAiConfigGenerationLimiter(
                active_generations=use_cases.runtime_config.model.active_generations,
                active_callback=lambda active: metrics.set_active_generation(
                    model_id=ai_runtime_config.model.model_id,
                    active=active,
                ),
            ),
        ),
        metrics=metrics,
        health=health,
        metrics_port=metrics_port or effective_worker_config.metrics_port,
        queue_snapshot=_queue_snapshot_callback(
            metrics=metrics,
            repository=use_cases.jobs.repository,
        ),
    )


def _queue_snapshot_callback(
    *,
    metrics: BacktestAiConfiguratorMetrics,
    repository: object,
) -> Callable[[], None]:
    def _snapshot() -> None:
        counter = getattr(repository, "count_jobs_by_state", None)
        if not callable(counter):
            return
        counter_func = cast(Callable[..., int], counter)
        metrics.set_queue_depth(
            queued=counter_func(state="queued"),
            running=counter_func(state="running"),
            repairing=counter_func(state="repairing"),
        )

    return _snapshot


def _model_path_check(*, path: Path) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        return path.exists(), "model_path"

    return _check


def _runtime_connection_check(*, base_url: str) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(urljoin(base_url.rstrip("/") + "/", "v1/models"))
            return response.status_code < 500, "runtime_connection"
        except httpx.HTTPError:
            return False, "runtime_connection"

    return _check


def _postgres_queue_audit_check(
    *,
    gateway: PsycopgBacktestPostgresGateway,
) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        try:
            gateway.fetch_one(
                query=(
                    "SELECT "
                    "(SELECT count(*) FROM backtest_ai_config_jobs LIMIT 1) AS jobs_count, "
                    "(SELECT count(*) FROM backtest_ai_config_llm_attempts LIMIT 1) "
                    "AS attempts_count"
                ),
                parameters={},
            )
        except Exception:  # noqa: BLE001
            return False, "postgres_queue_audit"
        return True, "postgres_queue_audit"

    return _check


def _quantization_from_model_path(path: Path) -> str:
    normalized = str(path).lower()
    if "4bit" in normalized or "4-bit" in normalized:
        return "4bit"
    if "8bit" in normalized or "8-bit" in normalized:
        return "8bit"
    return "unknown"


def _build_locked_by() -> str:
    hostname = socket.gethostname().strip() or "unknown-host"
    return f"backtest-ai-configurator-worker:{hostname}-{os.getpid()}"


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
        return int(raw_value.strip())
    except ValueError as error:
        raise ValueError(f"{key} must be integer") from error


def _env_float(*, environ: Mapping[str, str], key: str, default: float) -> float:
    raw_value = environ.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError as error:
        raise ValueError(f"{key} must be numeric") from error


__all__ = [
    "BacktestAiConfiguratorWorkerApp",
    "BacktestAiConfiguratorWorkerRuntimeConfig",
    "BacktestAiConfiguratorMetrics",
    "build_backtest_ai_configurator_worker_app",
    "load_backtest_ai_configurator_worker_runtime_config",
]
