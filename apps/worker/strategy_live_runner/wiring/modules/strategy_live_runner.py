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
    LogOnlyTelegramNotifier,
    PostgresConfirmedTelegramChatBindingResolver,
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PsycopgStrategyPostgresGateway,
    RedisStrategyLiveCandleStream,
    RedisStrategyLiveCandleStreamConfig,
    RedisStrategyRealtimeOutputPublisher,
    RedisStrategyRealtimeOutputPublisherConfig,
    RedisStrategyRealtimeOutputPublisherHooks,
    SystemRunnerSleeper,
    SystemStrategyClock,
    TelegramBotApiNotifier,
    TelegramBotApiNotifierConfig,
    TelegramNotifierHooks,
    load_strategy_live_runner_runtime_config,
)
from trading.contexts.strategy.application import (
    NoOpStrategyRealtimeOutputPublisher,
    NoOpTelegramNotifier,
    StrategyLiveRunner,
    StrategyLiveRunnerIterationReport,
    TelegramNotificationPolicy,
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
        self.realtime_output_publish_total = Counter(
            "strategy_realtime_output_publish_total",
            "Strategy realtime output successful publish count",
        )
        self.realtime_output_publish_errors_total = Counter(
            "strategy_realtime_output_publish_errors_total",
            "Strategy realtime output publish failures count",
        )
        self.realtime_output_publish_duplicates_total = Counter(
            "strategy_realtime_output_publish_duplicates_total",
            "Strategy realtime output duplicate/out-of-order publish count",
        )
        self.telegram_notify_total = Counter(
            "strategy_telegram_notify_total",
            "Strategy telegram notifications successfully sent count",
        )
        self.telegram_notify_errors_total = Counter(
            "strategy_telegram_notify_errors_total",
            "Strategy telegram notifications failed send count",
        )
        self.telegram_notify_skipped_total = Counter(
            "strategy_telegram_notify_skipped_total",
            "Strategy telegram notifications skipped due to missing confirmed chat binding",
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

    def realtime_output_hooks(self) -> RedisStrategyRealtimeOutputPublisherHooks:
        """
        Build metrics callbacks bundle for realtime output publish adapter.

        Args:
            None.
        Returns:
            RedisStrategyRealtimeOutputPublisherHooks: Hook callbacks bound to Prometheus counters.
        Assumptions:
            Hook callbacks are lightweight and thread-safe in single worker process.
        Raises:
            None.
        Side Effects:
            None.
        """
        return RedisStrategyRealtimeOutputPublisherHooks(
            on_publish_success=self.realtime_output_publish_total.inc,
            on_publish_error=self.realtime_output_publish_errors_total.inc,
            on_publish_duplicate=self.realtime_output_publish_duplicates_total.inc,
        )

    def telegram_notifier_hooks(self) -> TelegramNotifierHooks:
        """
        Build metrics callbacks bundle for Telegram notifier adapters.

        Args:
            None.
        Returns:
            TelegramNotifierHooks: Hook callbacks bound to Prometheus counters.
        Assumptions:
            Hook callbacks are lightweight and thread-safe in single worker process.
        Raises:
            None.
        Side Effects:
            None.
        """
        return TelegramNotifierHooks(
            on_notify_sent=self.telegram_notify_total.inc,
            on_notify_error=self.telegram_notify_errors_total.inc,
            on_notify_skipped=self.telegram_notify_skipped_total.inc,
        )


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
        config_path: Path to `strategy.yaml` or `strategy_live_runner.yaml`.
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
    runtime_config = load_strategy_live_runner_runtime_config(
        Path(config_path),
        environ=environ,
    )
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
    metrics = StrategyLiveRunnerMetrics()

    realtime_output_config = runtime_config.realtime_output
    realtime_output_publisher = NoOpStrategyRealtimeOutputPublisher()
    if realtime_output_config.enabled:
        realtime_output_publisher = RedisStrategyRealtimeOutputPublisher(
            config=RedisStrategyRealtimeOutputPublisherConfig(
                host=realtime_output_config.host,
                port=realtime_output_config.port,
                db=realtime_output_config.db,
                password_env=realtime_output_config.password_env,
                socket_timeout_s=realtime_output_config.socket_timeout_s,
                connect_timeout_s=realtime_output_config.connect_timeout_s,
                metrics_stream_prefix=realtime_output_config.metrics_stream_prefix,
                events_stream_prefix=realtime_output_config.events_stream_prefix,
            ),
            environ=environ,
            hooks=metrics.realtime_output_hooks(),
        )

    telegram_config = runtime_config.telegram
    telegram_notifier = NoOpTelegramNotifier()
    telegram_notification_policy = TelegramNotificationPolicy(
        failed_debounce_seconds=telegram_config.debounce_failed_seconds
    )
    if telegram_config.enabled:
        chat_binding_resolver = PostgresConfirmedTelegramChatBindingResolver(
            gateway=postgres_gateway
        )
        if telegram_config.mode == "log_only":
            telegram_notifier = LogOnlyTelegramNotifier(
                chat_binding_resolver=chat_binding_resolver,
                hooks=metrics.telegram_notifier_hooks(),
            )
        elif telegram_config.mode == "telegram":
            bot_token = _require_non_empty_env_value(
                environ=environ,
                key=telegram_config.bot_token_env,
                setting_name="strategy_live_runner.telegram.bot_token_env",
            )
            telegram_notifier = TelegramBotApiNotifier(
                config=TelegramBotApiNotifierConfig(
                    bot_token=bot_token,
                    api_base_url=telegram_config.api_base_url,
                    send_timeout_s=telegram_config.send_timeout_s,
                ),
                chat_binding_resolver=chat_binding_resolver,
                hooks=metrics.telegram_notifier_hooks(),
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
        realtime_output_publisher=realtime_output_publisher,
        telegram_notifier=telegram_notifier,
        telegram_notification_policy=telegram_notification_policy,
    )
    return StrategyLiveRunnerApp(
        poll_interval_seconds=runtime_config.poll_interval_seconds,
        runner=runner,
        metrics=metrics,
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


def _require_non_empty_env_value(
    *,
    environ: Mapping[str, str],
    key: str | None,
    setting_name: str,
) -> str:
    """
    Resolve required environment variable and fail fast when missing or blank.

    Args:
        environ: Runtime environment mapping.
        key: Environment variable name.
        setting_name: Config setting path used in deterministic error messages.
    Returns:
        str: Non-empty environment variable value.
    Assumptions:
        Function is used for required secrets like `TELEGRAM_BOT_TOKEN`.
    Raises:
        ValueError: If environment variable name or value is missing.
    Side Effects:
        None.
    """
    if key is None or not key.strip():
        raise ValueError(f"{setting_name} must be non-empty")
    raw_value = environ.get(key, "")
    value = raw_value.strip()
    if not value:
        raise ValueError(
            (
                f"{setting_name} requires environment variable {key} "
                "with non-empty value"
            )
        )
    return value
