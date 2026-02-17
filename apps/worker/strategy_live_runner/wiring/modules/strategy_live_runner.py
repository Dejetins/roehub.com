from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from prometheus_client import Counter, Histogram, start_http_server

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.strategy.adapters.outbound import (
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PsycopgStrategyPostgresGateway,
    RedisStrategyLiveCandleStream,
    RedisStrategyLiveCandleStreamConfig,
    SystemRunnerSleeper,
    SystemStrategyClock,
    load_strategy_live_runner_runtime_config,
)
from trading.contexts.strategy.application import (
    StrategyLiveRunner,
    StrategyLiveRunnerIterationReport,
)
from trading.platform.time.system_clock import SystemClock

log = logging.getLogger(__name__)

_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


class StrategyLiveRunnerMetrics:
    """
    StrategyLiveRunnerMetrics — Prometheus metrics bundle for strategy live-runner worker.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - apps/worker/strategy_live_runner/main/main.py
      - docs/runbooks/market-data-redis-streams.md
    """

    def __init__(self) -> None:
        """
        Register Prometheus metrics used by strategy live-runner runtime.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metrics are created once per worker process.
        Raises:
            ValueError: Propagated by prometheus client on duplicate metric names.
        Side Effects:
            Registers metrics in default Prometheus registry.
        """
        self.iterations_total = Counter(
            "strategy_live_runner_iterations_total",
            "Strategy live-runner successful iterations count",
        )
        self.iteration_errors_total = Counter(
            "strategy_live_runner_iteration_errors_total",
            "Strategy live-runner iteration failures count",
        )
        self.messages_read_total = Counter(
            "strategy_live_runner_messages_read_total",
            "Strategy live-runner read messages count",
        )
        self.messages_acked_total = Counter(
            "strategy_live_runner_messages_acked_total",
            "Strategy live-runner acked messages count",
        )
        self.failed_runs_total = Counter(
            "strategy_live_runner_failed_runs_total",
            "Strategy live-runner failed runs count",
        )
        self.iteration_duration_seconds = Histogram(
            "strategy_live_runner_iteration_duration_seconds",
            "Strategy live-runner iteration duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0),
        )

    def observe_iteration(
        self,
        *,
        report: StrategyLiveRunnerIterationReport,
        duration_seconds: float,
    ) -> None:
        """
        Observe one successful live-runner iteration report.

        Args:
            report: Iteration counters produced by runner service.
            duration_seconds: Measured iteration duration.
        Returns:
            None.
        Assumptions:
            Report values are non-negative deterministic counters.
        Raises:
            None.
        Side Effects:
            Updates counters/histogram in Prometheus registry.
        """
        self.iterations_total.inc()
        self.messages_read_total.inc(report.read_messages)
        self.messages_acked_total.inc(report.acked_messages)
        self.failed_runs_total.inc(report.failed_runs)
        self.iteration_duration_seconds.observe(max(duration_seconds, 0.0))


@dataclass(frozen=True, slots=True)
class StrategyLiveRunnerApp:
    """
    StrategyLiveRunnerApp — runtime loop wrapper over `StrategyLiveRunner` service.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/main/main.py
      - configs/dev/strategy_live_runner.yaml
    """

    poll_interval_seconds: int
    runner: StrategyLiveRunner
    metrics: StrategyLiveRunnerMetrics
    metrics_port: int

    def __post_init__(self) -> None:
        """
        Validate runtime invariants for strategy live-runner app wrapper.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Poll interval and metrics port are positive integers.
        Raises:
            ValueError: If one of numeric parameters is invalid.
        Side Effects:
            None.
        """
        if self.poll_interval_seconds <= 0:
            raise ValueError("StrategyLiveRunnerApp.poll_interval_seconds must be > 0")
        if self.metrics_port <= 0:
            raise ValueError("StrategyLiveRunnerApp.metrics_port must be > 0")

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Run strategy live-runner loop until stop event is set.

        Args:
            stop_event: Cooperative shutdown signal shared with process entrypoint.
        Returns:
            None.
        Assumptions:
            Runner service performs one deterministic poll iteration per call.
        Raises:
            Exception: Unexpected runtime errors are logged and loop continues.
        Side Effects:
            Starts Prometheus HTTP endpoint and performs storage/network IO each iteration.
        """
        start_http_server(self.metrics_port)
        log.info("strategy live-runner metrics server started on port %s", self.metrics_port)
        while not stop_event.is_set():
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                report = self.runner.run_once()
                duration_seconds = loop.time() - started
                self.metrics.observe_iteration(
                    report=report,
                    duration_seconds=duration_seconds,
                )
            except Exception:  # noqa: BLE001
                self.metrics.iteration_errors_total.inc()
                log.exception("strategy live-runner iteration failed")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_seconds)
            except TimeoutError:
                continue


def build_strategy_live_runner_app(
    *,
    config_path: str,
    environ: Mapping[str, str],
    metrics_port: int,
) -> StrategyLiveRunnerApp:
    """
    Build fully wired strategy live-runner worker app.

    Args:
        config_path: Path to `strategy_live_runner.yaml`.
        environ: Runtime environment mapping.
        metrics_port: Prometheus HTTP endpoint port.
    Returns:
        StrategyLiveRunnerApp: Ready-to-run app instance.
    Assumptions:
        Postgres DSN and ClickHouse settings are provided via environment.
    Raises:
        ValueError: If required runtime configuration/env variables are missing.
    Side Effects:
        Creates storage clients and runtime adapters.
    """
    runtime_config = load_strategy_live_runner_runtime_config(Path(config_path))
    if not runtime_config.redis_streams.enabled:
        raise ValueError(
            "strategy_live_runner.redis_streams.enabled must be true for live-runner worker"
        )

    strategy_pg_dsn = environ.get(_STRATEGY_PG_DSN_KEY, "").strip()
    if not strategy_pg_dsn:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required for strategy live-runner worker"
        )

    postgres_gateway = PsycopgStrategyPostgresGateway(dsn=strategy_pg_dsn)
    strategy_repository = PostgresStrategyRepository(gateway=postgres_gateway)
    run_repository = PostgresStrategyRunRepository(gateway=postgres_gateway)

    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )
    canonical_reader = ClickHouseCanonicalCandleReader(
        gateway=clickhouse_gateway,
        clock=SystemClock(),
        database=clickhouse_settings.database,
    )

    redis_config = runtime_config.redis_streams
    live_candle_stream = RedisStrategyLiveCandleStream(
        config=RedisStrategyLiveCandleStreamConfig(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password_env=redis_config.password_env,
            socket_timeout_s=redis_config.socket_timeout_s,
            connect_timeout_s=redis_config.connect_timeout_s,
            stream_prefix=redis_config.stream_prefix,
            consumer_group=redis_config.consumer_group,
            consumer_name=_build_consumer_name(),
            read_count=redis_config.read_count,
            block_ms=redis_config.block_ms,
        ),
        environ=environ,
    )

    runner = StrategyLiveRunner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        live_candle_stream=live_candle_stream,
        canonical_candle_reader=canonical_reader,
        clock=SystemStrategyClock(),
        sleeper=SystemRunnerSleeper(),
        repair_retry_attempts=runtime_config.repair.retry_attempts,
        repair_backoff_seconds=runtime_config.repair.retry_backoff_seconds,
    )
    return StrategyLiveRunnerApp(
        poll_interval_seconds=runtime_config.poll_interval_seconds,
        runner=runner,
        metrics=StrategyLiveRunnerMetrics(),
        metrics_port=metrics_port,
    )


def _build_consumer_name() -> str:
    """
    Build deterministic strategy live-runner consumer name `<hostname>-<pid>`.

    Args:
        None.
    Returns:
        str: Deterministic consumer name for Redis Streams group membership.
    Assumptions:
        Hostname and pid are stable for process lifetime.
    Raises:
        None.
    Side Effects:
        Reads hostname and process id from OS.
    """
    hostname = socket.gethostname().strip() or "unknown-host"
    return f"{hostname}-{os.getpid()}"
