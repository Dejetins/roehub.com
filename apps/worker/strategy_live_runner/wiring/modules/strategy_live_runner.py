from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from apps.cli.wiring.db.clickhouse import (  # noqa: PLC2701
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.live_execution.adapters.outbound import (
    PostgresExchangeAccountProjectionRepository,
    PostgresExecutionIntentRepository,
    PostgresPaperAccountingRepository,
    PostgresStrategyPositionOwnershipRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    ExecutionIngressService,
    StrategyPositionOwnershipService,
)
from trading.contexts.market_data.adapters.outbound.clients import RestCandleIngestSource
from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.config import load_market_data_runtime_config
from trading.contexts.market_data.adapters.outbound.messaging.redis import RedisCandleHotCache
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ThreadLocalClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresCandleRepairAuditRepository,
    PsycopgMarketDataPostgresGateway,
)
from trading.contexts.market_data.application.dto import ClosedCandleTailRepairPolicy
from trading.contexts.market_data.application.services import MarketDataClosedCandleTailProvider
from trading.contexts.notifications.adapters import (
    PostgresNotificationRepository,
    PsycopgNotificationPostgresGateway,
)
from trading.contexts.notifications.application import NotificationSourceRouter
from trading.contexts.strategy.adapters.outbound import (
    LogOnlyTelegramNotifier,
    NotificationsTelegramNotifier,
    PostgresConfirmedTelegramChatBindingResolver,
    PostgresLiveStrategyProfileRepository,
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PostgresStrategySignalRepository,
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
from trading.contexts.strategy.adapters.outbound.acl.live_execution_producer import (
    LiveExecutionStrategySignalProducer,
)
from trading.contexts.strategy.adapters.outbound.config import StrategyProducerRuntimeConfig
from trading.contexts.strategy.application import (
    NoOpStrategyRealtimeOutputPublisher,
    NoOpTelegramNotifier,
    StrategyLiveRunner,
    StrategyLiveRunnerIterationReport,
    TelegramNotificationPolicy,
)
from trading.contexts.strategy.application.ports import (
    NoOpStrategyExecutionProducer,
    StrategyExecutionProducer,
)
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.platform.time.system_clock import SystemClock

log = logging.getLogger(__name__)

_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_NOTIFICATIONS_PG_DSN_KEY = "NOTIFICATIONS_PG_DSN"
_POSTGRES_DSN_KEY = "POSTGRES_DSN"
_PRODUCER_ALLOWED_MODES = ("paper", "testnet")
_PRODUCER_BLOCKED_REASONS = (
    "producer_disabled",
    "producer_mode_not_allowed",
    "producer_allowlist_missing",
)


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

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
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
        self.registry = registry or REGISTRY
        self._lock = threading.Lock()
        self._started_unixtime = 0.0
        self._last_cycle_start_unixtime = 0.0
        self._last_cycle_end_unixtime = 0.0
        self._last_success_unixtime = 0.0
        self._last_error_reason = "none"
        self._producer_enabled = False
        self._producer_allow_all = False
        self._producer_allowed_modes: tuple[str, ...] = ()
        self._producer_allowed_users = 0
        self._producer_allowed_strategies = 0

        self.iterations_total = Counter(
            "strategy_live_runner_iterations_total",
            "Strategy live-runner successful iterations count",
            registry=self.registry,
        )
        self.iteration_errors_total = Counter(
            "strategy_live_runner_iteration_errors_total",
            "Strategy live-runner iteration failures count",
            registry=self.registry,
        )
        self.messages_read_total = Counter(
            "strategy_live_runner_messages_read_total",
            "Strategy live-runner read messages count",
            registry=self.registry,
        )
        self.messages_acked_total = Counter(
            "strategy_live_runner_messages_acked_total",
            "Strategy live-runner acked messages count",
            registry=self.registry,
        )
        self.failed_runs_total = Counter(
            "strategy_live_runner_failed_runs_total",
            "Strategy live-runner failed runs count",
            registry=self.registry,
        )
        self.iteration_duration_seconds = Histogram(
            "strategy_live_runner_iteration_duration_seconds",
            "Strategy live-runner iteration duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0),
            registry=self.registry,
        )
        self.producer_admin_enabled = Gauge(
            "strategy_producer_admin_enabled",
            "Strategy producer admin switch, 1 when enabled",
            registry=self.registry,
        )
        self.producer_allow_all = Gauge(
            "strategy_producer_allow_all",
            "Strategy producer allow-all switch, 1 when enabled",
            registry=self.registry,
        )
        self.producer_allowed_mode = Gauge(
            "strategy_producer_allowed_mode",
            "Strategy producer allowed mode flags",
            ("mode",),
            registry=self.registry,
        )
        self.producer_allowlist_entries = Gauge(
            "strategy_producer_allowlist_entries",
            "Strategy producer allowlist entry counts",
            ("scope",),
            registry=self.registry,
        )
        self.producer_ready = Gauge(
            "strategy_producer_ready",
            "Strategy producer process readiness, 1 when loop is healthy",
            registry=self.registry,
        )
        self.producer_last_cycle_start_unixtime = Gauge(
            "strategy_producer_last_cycle_start_unixtime",
            "Unix timestamp of the latest producer cycle start",
            registry=self.registry,
        )
        self.producer_last_cycle_end_unixtime = Gauge(
            "strategy_producer_last_cycle_end_unixtime",
            "Unix timestamp of the latest producer cycle end",
            registry=self.registry,
        )
        self.producer_last_success_unixtime = Gauge(
            "strategy_producer_last_success_unixtime",
            "Unix timestamp of the latest successful producer cycle",
            registry=self.registry,
        )
        self.producer_polled_runs = Gauge(
            "strategy_producer_polled_runs",
            "Active strategy runs polled during the latest cycle",
            registry=self.registry,
        )
        self.producer_active_instruments = Gauge(
            "strategy_producer_active_instruments",
            "Active instruments processed during the latest cycle",
            registry=self.registry,
        )
        self.producer_source_events_total = Counter(
            "strategy_producer_source_events_total",
            "Strategy producer source event records created through live_execution",
            ("mode", "outcome"),
            registry=self.registry,
        )
        self.producer_skipped_strategies_total = Counter(
            "strategy_producer_skipped_strategies_total",
            "Strategy producer signals skipped before source-event creation",
            ("reason",),
            registry=self.registry,
        )
        self.producer_last_source_event_unixtime = Gauge(
            "strategy_producer_last_source_event_unixtime",
            "Unix timestamp of latest strategy producer source event by bounded outcome",
            ("mode", "outcome"),
            registry=self.registry,
        )
        self.producer_source_event_latency_seconds = Histogram(
            "strategy_producer_source_event_latency_seconds",
            "Seconds between signal candle close and source-event creation",
            ("mode", "outcome"),
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
            registry=self.registry,
        )
        self.producer_signal_lag_seconds = Gauge(
            "strategy_producer_signal_lag_seconds",
            "Seconds between latest signal candle close and producer observation",
            registry=self.registry,
        )
        self.realtime_output_publish_total = Counter(
            "strategy_realtime_output_publish_total",
            "Strategy realtime output successful publish count",
            registry=self.registry,
        )
        self.realtime_output_publish_errors_total = Counter(
            "strategy_realtime_output_publish_errors_total",
            "Strategy realtime output publish failures count",
            registry=self.registry,
        )
        self.realtime_output_publish_duplicates_total = Counter(
            "strategy_realtime_output_publish_duplicates_total",
            "Strategy realtime output duplicate/out-of-order publish count",
            registry=self.registry,
        )
        self.telegram_notify_total = Counter(
            "strategy_telegram_notify_total",
            "Strategy telegram notifications successfully sent count",
            registry=self.registry,
        )
        self.telegram_notify_errors_total = Counter(
            "strategy_telegram_notify_errors_total",
            "Strategy telegram notifications failed send count",
            registry=self.registry,
        )
        self.telegram_notify_skipped_total = Counter(
            "strategy_telegram_notify_skipped_total",
            "Strategy telegram notifications skipped due to missing confirmed chat binding",
            registry=self.registry,
        )
        self.strategy_signal_total = Counter(
            "strategy_signal_total",
            "Strategy live evaluator journal outcomes",
            ("mode", "action", "outcome"),
            registry=self.registry,
        )
        self.strategy_position_ownership_total = Counter(
            "strategy_position_ownership_total",
            "Strategy position ownership reserve/release/conflict outcomes.",
            ("result", "reason"),
            registry=self.registry,
        )
        self.strategy_capital_reservation_total = Counter(
            "strategy_capital_reservation_total",
            "Strategy capital reservation outcomes.",
            ("result", "reason"),
            registry=self.registry,
        )
        self.strategy_paper_accounting_total = Counter(
            "strategy_paper_accounting_total",
            "Strategy paper order/fill/accounting outcomes.",
            ("result", "reason"),
            registry=self.registry,
        )

    def mark_started(self, *, producer_config: StrategyProducerRuntimeConfig) -> None:
        now = time.time()
        with self._lock:
            self._started_unixtime = now
            self._last_error_reason = "none"
            self._producer_enabled = producer_config.enabled
            self._producer_allow_all = producer_config.allow_all
            self._producer_allowed_modes = producer_config.allowed_modes
            self._producer_allowed_users = len(producer_config.allowed_user_ids)
            self._producer_allowed_strategies = len(producer_config.allowed_strategy_ids)
        self.producer_admin_enabled.set(1 if producer_config.enabled else 0)
        self.producer_allow_all.set(1 if producer_config.allow_all else 0)
        for mode in _PRODUCER_ALLOWED_MODES:
            self.producer_allowed_mode.labels(mode=mode).set(
                1 if mode in producer_config.allowed_modes else 0
            )
        self.producer_allowlist_entries.labels(scope="user").set(
            len(producer_config.allowed_user_ids)
        )
        self.producer_allowlist_entries.labels(scope="strategy").set(
            len(producer_config.allowed_strategy_ids)
        )
        self.producer_ready.set(1)

    def mark_iteration_started(self) -> None:
        now = time.time()
        with self._lock:
            self._last_cycle_start_unixtime = now
        self.producer_last_cycle_start_unixtime.set(now)

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
        now = time.time()
        self.iterations_total.inc()
        self.messages_read_total.inc(report.read_messages)
        self.messages_acked_total.inc(report.acked_messages)
        self.failed_runs_total.inc(report.failed_runs)
        self.iteration_duration_seconds.observe(max(duration_seconds, 0.0))
        self.producer_polled_runs.set(report.polled_runs)
        self.producer_active_instruments.set(report.active_instruments)
        self.producer_last_cycle_end_unixtime.set(now)
        self.producer_last_success_unixtime.set(now)
        self.producer_ready.set(1)
        with self._lock:
            self._last_cycle_end_unixtime = now
            self._last_success_unixtime = now
            self._last_error_reason = "none"

    def observe_iteration_error(self) -> None:
        now = time.time()
        self.iteration_errors_total.inc()
        self.producer_last_cycle_end_unixtime.set(now)
        self.producer_ready.set(0)
        with self._lock:
            self._last_cycle_end_unixtime = now
            self._last_error_reason = "iteration_error"

    def observe_source_event_created(self, signal: StrategySignal) -> None:
        now = time.time()
        mode = _bounded_signal_mode(signal.mode)
        outcome = _bounded_signal_outcome(signal.outcome)
        lag_seconds = max(0.0, now - signal.bar_ts_close.astimezone(timezone.utc).timestamp())
        self.producer_source_events_total.labels(mode=mode, outcome=outcome).inc()
        self.producer_last_source_event_unixtime.labels(mode=mode, outcome=outcome).set(now)
        self.producer_source_event_latency_seconds.labels(
            mode=mode,
            outcome=outcome,
        ).observe(lag_seconds)
        self.producer_signal_lag_seconds.set(lag_seconds)

    def observe_source_event_blocked(self, *, reason: str) -> None:
        self.producer_skipped_strategies_total.labels(
            reason=_bounded_producer_block_reason(reason)
        ).inc()

    def health_payload(self) -> dict[str, object]:
        with self._lock:
            return {
                "service": "strategy_producer",
                "status": "live" if self._started_unixtime > 0 else "starting",
                "started_unixtime": self._started_unixtime,
                "producer": self._producer_payload_locked(),
            }

    def readiness_payload(self) -> tuple[int, dict[str, object]]:
        with self._lock:
            ready = self._started_unixtime > 0 and self._last_error_reason == "none"
            payload = {
                "service": "strategy_producer",
                "ready": ready,
                "reason": "ready" if ready else self._last_error_reason,
                "last_cycle_start_unixtime": self._last_cycle_start_unixtime,
                "last_cycle_end_unixtime": self._last_cycle_end_unixtime,
                "last_success_unixtime": self._last_success_unixtime,
                "producer": self._producer_payload_locked(),
            }
        return (HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, payload)

    def _producer_payload_locked(self) -> dict[str, object]:
        return {
            "enabled": self._producer_enabled,
            "allow_all": self._producer_allow_all,
            "allowed_modes": list(self._producer_allowed_modes),
            "allowed_user_count": self._producer_allowed_users,
            "allowed_strategy_count": self._producer_allowed_strategies,
        }

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

    def observe_strategy_signal(self, signal: StrategySignal) -> None:
        self.strategy_signal_total.labels(
            mode=signal.mode,
            action=signal.signal_action,
            outcome=signal.outcome,
        ).inc()

    def observe_strategy_position_ownership(self, result: str, reason: str) -> None:
        self.strategy_position_ownership_total.labels(
            result=result,
            reason=(reason or "unknown")[:80],
        ).inc()

    def observe_strategy_capital_reservation(self, result: str, reason: str) -> None:
        self.strategy_capital_reservation_total.labels(
            result=result,
            reason=(reason or "unknown")[:80],
        ).inc()

    def observe_strategy_paper_accounting(self, result: str, reason: str) -> None:
        self.strategy_paper_accounting_total.labels(
            result=result,
            reason=(reason or "unknown")[:80],
        ).inc()


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
    producer_config: StrategyProducerRuntimeConfig

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
        self.metrics.mark_started(producer_config=self.producer_config)
        http_server = StrategyLiveRunnerHttpServer(
            metrics_port=self.metrics_port,
            metrics=self.metrics,
        )
        http_server.start()
        log.info("strategy live-runner health/metrics server started on port %s", self.metrics_port)
        try:
            while not stop_event.is_set():
                loop = asyncio.get_running_loop()
                started = loop.time()
                self.metrics.mark_iteration_started()
                try:
                    report = self.runner.run_once()
                    duration_seconds = loop.time() - started
                    self.metrics.observe_iteration(
                        report=report,
                        duration_seconds=duration_seconds,
                    )
                except Exception:  # noqa: BLE001
                    self.metrics.observe_iteration_error()
                    log.exception("strategy live-runner iteration failed")

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_seconds)
                except TimeoutError:
                    continue
        finally:
            http_server.stop()


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
    signal_repository = PostgresStrategySignalRepository(gateway=postgres_gateway)
    live_profile_repository = PostgresLiveStrategyProfileRepository(gateway=postgres_gateway)
    metrics = StrategyLiveRunnerMetrics()
    position_ownership_coordinator = StrategyPositionOwnershipService(
        repository=PostgresStrategyPositionOwnershipRepository(gateway=postgres_gateway),
        on_transition=metrics.observe_strategy_position_ownership,
    )
    paper_accounting_service = CapitalReservationPaperAccountingService(
        repository=PostgresPaperAccountingRepository(gateway=postgres_gateway),
        account_projection_repository=PostgresExchangeAccountProjectionRepository(
            gateway=postgres_gateway
        ),
        clock=SystemStrategyClock(),
        on_capital_reservation=metrics.observe_strategy_capital_reservation,
        on_paper_accounting=metrics.observe_strategy_paper_accounting,
    )

    clickhouse_settings = ClickHouseSettingsLoader(environ).load()
    clickhouse_gateway = ThreadLocalClickHouseConnectGateway(
        client_factory=lambda: _clickhouse_client(clickhouse_settings)
    )
    canonical_reader = ClickHouseCanonicalCandleReader(
        gateway=clickhouse_gateway,
        database=clickhouse_settings.database,
    )
    market_data_config = load_market_data_runtime_config(
        _resolve_market_data_config_path(strategy_config_path=Path(config_path))
    )
    market_data_clock = SystemClock()
    hot_cache_config = market_data_config.live_feed.redis_hot_cache
    if not hot_cache_config.enabled:
        raise ValueError(
            "market_data.live_feed.redis_hot_cache.enabled must be true for live-tail repair"
        )
    hot_cache = RedisCandleHotCache(
        connection_config=market_data_config.live_feed.redis_streams,
        config=hot_cache_config,
        environ=environ,
    )
    rest_source = RestCandleIngestSource(
        cfg=market_data_config,
        clock=market_data_clock,
        http=RequestsHttpClient(),
        ingest_id=uuid4(),
    )
    candle_repair_audit_repository = PostgresCandleRepairAuditRepository(
        gateway=PsycopgMarketDataPostgresGateway(dsn=strategy_pg_dsn)
    )
    closed_candle_tail_provider = MarketDataClosedCandleTailProvider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=candle_repair_audit_repository,
        clock=market_data_clock,
        policy=ClosedCandleTailRepairPolicy(),
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
            pending_claim_min_idle_ms=redis_config.pending_claim_min_idle_ms,
        ),
        environ=environ,
    )
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
        if telegram_config.mode == "notifications":
            notifications_gateway = PsycopgNotificationPostgresGateway(
                dsn=_resolve_notification_postgres_dsn(environ=environ)
            )
            telegram_notifier = NotificationsTelegramNotifier(
                repository=PostgresNotificationRepository(gateway=notifications_gateway),
                router=NotificationSourceRouter(),
                hooks=metrics.telegram_notifier_hooks(),
            )
        elif telegram_config.mode == "log_only":
            chat_binding_resolver = PostgresConfirmedTelegramChatBindingResolver(
                gateway=postgres_gateway
            )
            telegram_notifier = LogOnlyTelegramNotifier(
                chat_binding_resolver=chat_binding_resolver,
                hooks=metrics.telegram_notifier_hooks(),
            )
        elif telegram_config.mode == "telegram":
            chat_binding_resolver = PostgresConfirmedTelegramChatBindingResolver(
                gateway=postgres_gateway
            )
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

    producer_config = runtime_config.producer
    execution_producer_delegate: StrategyExecutionProducer = NoOpStrategyExecutionProducer()
    if producer_config.enabled:
        execution_repository = PostgresExecutionIntentRepository(gateway=postgres_gateway)
        execution_producer_delegate = LiveExecutionStrategySignalProducer(
            ingress_service=ExecutionIngressService(
                repository=execution_repository,
                clock=SystemLiveExecutionClock(),
            ),
            repository=execution_repository,
        )
    execution_producer = GuardedStrategyExecutionProducer(
        delegate=execution_producer_delegate,
        producer_config=producer_config,
        on_source_event_created=metrics.observe_source_event_created,
        on_source_event_blocked=metrics.observe_source_event_blocked,
    )

    runner = StrategyLiveRunner(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        live_candle_stream=live_candle_stream,
        closed_candle_tail_provider=closed_candle_tail_provider,
        signal_repository=signal_repository,
        clock=SystemStrategyClock(),
        sleeper=SystemRunnerSleeper(),
        repair_retry_attempts=runtime_config.repair.retry_attempts,
        repair_backoff_seconds=runtime_config.repair.retry_backoff_seconds,
        realtime_output_publisher=realtime_output_publisher,
        live_profile_repository=live_profile_repository,
        position_ownership_coordinator=position_ownership_coordinator,
        capital_reservation_coordinator=paper_accounting_service,
        paper_accounting_recorder=paper_accounting_service,
        execution_producer=execution_producer,
        on_signal_recorded=metrics.observe_strategy_signal,
        telegram_notifier=telegram_notifier,
        telegram_notification_policy=telegram_notification_policy,
    )
    return StrategyLiveRunnerApp(
        poll_interval_seconds=runtime_config.poll_interval_seconds,
        runner=runner,
        metrics=metrics,
        metrics_port=metrics_port,
        producer_config=producer_config,
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


def _resolve_market_data_config_path(*, strategy_config_path: Path) -> Path:
    return strategy_config_path.with_name("market_data.yaml")


def _resolve_notification_postgres_dsn(*, environ: Mapping[str, str]) -> str:
    for key in (_NOTIFICATIONS_PG_DSN_KEY, _STRATEGY_PG_DSN_KEY, _POSTGRES_DSN_KEY):
        value = environ.get(key, "").strip()
        if value:
            return value
    raise ValueError("strategy notifications mode requires NOTIFICATIONS_PG_DSN or fallback DSN")


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


@dataclass(frozen=True, slots=True)
class StrategyProducerGateDecision:
    allowed: bool
    reason: str


class GuardedStrategyExecutionProducer(StrategyExecutionProducer):
    def __init__(
        self,
        *,
        delegate: StrategyExecutionProducer,
        producer_config: StrategyProducerRuntimeConfig,
        on_source_event_created: object | None = None,
        on_source_event_blocked: object | None = None,
    ) -> None:
        self._delegate = delegate
        self._producer_config = producer_config
        self._on_source_event_created = on_source_event_created
        self._on_source_event_blocked = on_source_event_blocked

    def record_signal(self, *, signal: StrategySignal) -> None:
        decision = self.evaluate(signal=signal)
        if not decision.allowed:
            if callable(self._on_source_event_blocked):
                self._on_source_event_blocked(reason=decision.reason)
            log.info(
                "strategy producer skipped signal reason=%s mode=%s outcome=%s",
                decision.reason,
                signal.mode,
                signal.outcome,
            )
            return

        self._delegate.record_signal(signal=signal)
        if callable(self._on_source_event_created):
            self._on_source_event_created(signal)

    def evaluate(self, *, signal: StrategySignal) -> StrategyProducerGateDecision:
        if not self._producer_config.enabled:
            return StrategyProducerGateDecision(False, "producer_disabled")
        if signal.mode not in self._producer_config.allowed_modes:
            return StrategyProducerGateDecision(False, "producer_mode_not_allowed")
        if self._producer_config.allow_all:
            return StrategyProducerGateDecision(True, "allowed")
        if str(signal.owner_user_id) in self._producer_config.allowed_user_ids:
            return StrategyProducerGateDecision(True, "allowed_user")
        if str(signal.strategy_id) in self._producer_config.allowed_strategy_ids:
            return StrategyProducerGateDecision(True, "allowed_strategy")
        return StrategyProducerGateDecision(False, "producer_allowlist_missing")


class StrategyLiveRunnerHttpServer:
    def __init__(self, *, metrics_port: int, metrics: StrategyLiveRunnerMetrics) -> None:
        self._metrics_port = metrics_port
        self._metrics = metrics
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        metrics = self._metrics

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health/live":
                    _write_json(self, HTTPStatus.OK, metrics.health_payload())
                    return
                if self.path == "/health/ready":
                    status, payload = metrics.readiness_payload()
                    _write_json(self, status, payload)
                    return
                if self.path == "/metrics":
                    payload = generate_latest(metrics.registry)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                _write_json(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"status": "not_found", "service": "strategy_producer"},
                )

            def log_message(self, format: str, *args: object) -> None:
                log.debug("strategy producer http: " + format, *args)

        self._server = ThreadingHTTPServer(("127.0.0.1", self._metrics_port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="strategy-producer-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


def _write_json(
    handler: BaseHTTPRequestHandler,
    status: int | HTTPStatus,
    payload: Mapping[str, object],
) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _bounded_signal_mode(mode: str) -> str:
    if mode in {"monitor_only", "paper", "live", "testnet"}:
        return mode
    return "unknown"


def _bounded_signal_outcome(outcome: str) -> str:
    if outcome in {"warmup", "no_signal", "signal", "blocked"}:
        return outcome
    return "unknown"


def _bounded_producer_block_reason(reason: str) -> str:
    if reason in _PRODUCER_BLOCKED_REASONS:
        return reason
    return "unknown"
