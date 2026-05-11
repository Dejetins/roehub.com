from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Mapping

from apps.api.wiring.modules.backtest import build_backtest_ai_configurator_use_cases
from trading.contexts.backtest.adapters.outbound import (
    MLXOpenAICompatibleAdapter,
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


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorWorkerApp:
    runtime_config: BacktestAiConfiguratorWorkerRuntimeConfig
    worker: BacktestAiConfigWorkerUseCase

    async def run(self, stop_event: asyncio.Event) -> None:
        processed_jobs = 0
        while not stop_event.is_set():
            started = time.perf_counter()
            result = self.worker.run_next()
            duration_seconds = max(time.perf_counter() - started, 0.0)
            if result.claimed:
                processed_jobs += 1
                job = result.job
                log.info(
                    "backtest AI configurator job processed: job_id=%s status=%s "
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
    )


def build_backtest_ai_configurator_worker_app(
    *,
    environ: Mapping[str, str],
    runtime_config: BacktestAiConfiguratorWorkerRuntimeConfig | None = None,
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
    adapter = MLXOpenAICompatibleAdapter(config=ai_runtime_config.model)
    use_cases = build_backtest_ai_configurator_use_cases(
        environ=environ,
        llm_gateway=adapter,
    )
    if use_cases is None:
        raise ValueError("backtest AI configurator use cases are unavailable")
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
                active_generations=use_cases.runtime_config.model.active_generations
            ),
        ),
    )


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
    "build_backtest_ai_configurator_worker_app",
    "load_backtest_ai_configurator_worker_runtime_config",
]
