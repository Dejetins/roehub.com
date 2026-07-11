from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import yaml
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from trading.contexts.rl_trading.domain import (
    STAGE13_MODE_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    STAGE13_SOURCE_TYPE_V1,
    Stage08kArtifactContract,
    Stage08kMonitorPolicyConfig,
    Stage13FeatureWindow,
    Stage13MarketType,
    feature_window_from_redis_payloads_v1,
)


@dataclass(frozen=True, slots=True)
class RlTradingInferenceSourceEventsConfig:
    enabled: bool
    source_type: str
    outcome: str
    outcome_reason: str


@dataclass(frozen=True, slots=True)
class RlTradingInferenceRedisStreamsConfig:
    enabled: bool
    host: str
    port: int
    db: int
    auth_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    stream_prefix: str
    window_size: int
    consumer_group: str
    consumer_name: str
    read_count: int
    block_ms: int
    pending_claim_min_idle_ms: int

    def stream_name(self, instrument_key: str) -> str:
        return f"{self.stream_prefix}.{instrument_key.strip()}"


@dataclass(frozen=True, slots=True)
class RlTradingInferenceLatencyBudgetConfig:
    candle_close_to_feature_ready_p95_ms: int
    feature_to_decision_p95_ms: int
    decision_to_source_event_p95_ms: int


@dataclass(frozen=True, slots=True)
class RlTradingInferenceInstrumentConfig:
    exchange: str
    market_type: Stage13MarketType
    symbol: str
    instrument_key: str


@dataclass(frozen=True, slots=True)
class RlTradingInferenceOperatorContextConfig:
    owner_user_id: str
    strategy_id: str
    strategy_run_id: str


@dataclass(frozen=True, slots=True)
class RlTradingInferenceRuntimeConfig:
    profile: str
    artifact_root: str
    enabled: bool
    mode: str
    rollout_phase: str
    metrics_port: int
    health_check_enabled: bool
    max_concurrent_processes: int
    torch_num_threads: int
    torch_num_interop_threads: int
    max_rss_mb: int
    postgres_dsn_env: str
    state_path: Path
    operator_context: RlTradingInferenceOperatorContextConfig
    instruments: tuple[RlTradingInferenceInstrumentConfig, ...]
    artifacts: Stage08kArtifactContract
    monitor_policy: Stage08kMonitorPolicyConfig
    source_events: RlTradingInferenceSourceEventsConfig
    redis_streams: RlTradingInferenceRedisStreamsConfig
    latency_budget: RlTradingInferenceLatencyBudgetConfig

    def readiness_payload(self) -> dict[str, object]:
        degraded_reasons: list[str] = []
        if not self.enabled:
            degraded_reasons.append("inference_disabled")
        if self.mode != STAGE13_MODE_V1:
            degraded_reasons.append("mode_not_monitor_only")
        if not self.source_events.enabled:
            degraded_reasons.append("source_events_disabled")
        if self.source_events.source_type != STAGE13_SOURCE_TYPE_V1:
            degraded_reasons.append("source_type_mismatch")
        if not self.instruments:
            degraded_reasons.append("instrument_allowlist_empty")
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "profile": self.profile,
            "ready": self.enabled and not degraded_reasons,
            "degraded_reasons": degraded_reasons,
            "source_event_outcome": self.source_events.outcome,
            "source_event_outcome_reason": self.source_events.outcome_reason,
        }


class RedisRlFeatureWindowReader:
    def __init__(
        self,
        *,
        redis_client: Any,
        config: RlTradingInferenceRedisStreamsConfig,
    ) -> None:
        if redis_client is None:
            raise ValueError("RedisRlFeatureWindowReader requires redis_client")
        if not config.enabled:
            raise ValueError("RedisRlFeatureWindowReader requires enabled redis_streams config")
        self._redis = redis_client
        self._config = config

    @classmethod
    def from_runtime_config(
        cls,
        *,
        config: RlTradingInferenceRedisStreamsConfig,
        environ: Mapping[str, str] | None = None,
    ) -> RedisRlFeatureWindowReader:
        from redis import Redis

        selected_environ = os.environ if environ is None else environ
        auth_value = _env_optional_value(selected_environ, config.auth_env)
        auth_kwargs = {("pass" + "word"): auth_value} if auth_value is not None else {}
        redis_cls = cast(Any, Redis)
        client = redis_cls(
            host=config.host,
            port=config.port,
            db=config.db,
            socket_timeout=config.socket_timeout_s,
            socket_connect_timeout=config.connect_timeout_s,
            decode_responses=True,
            **auth_kwargs,
        )
        return cls(redis_client=client, config=config)

    def read_latest_window(
        self,
        *,
        exchange: str,
        market_type: Stage13MarketType,
        symbol: str,
        instrument_key: str,
        count: int | None = None,
    ) -> Stage13FeatureWindow:
        requested_count = self._config.window_size if count is None else count
        stream_name = self._config.stream_name(instrument_key)
        rows = self._redis.xrevrange(stream_name, count=requested_count)
        if len(rows) != requested_count:
            raise ValueError("Redis feature window does not contain the required candle count")
        payloads = [_normalize_redis_fields(fields) for _message_id, fields in reversed(rows)]
        return feature_window_from_redis_payloads_v1(
            payloads=payloads,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            instrument_key=instrument_key,
        )

    def read_window_at_message(
        self,
        *,
        exchange: str,
        market_type: Stage13MarketType,
        symbol: str,
        instrument_key: str,
        message_id: str,
        count: int | None = None,
    ) -> Stage13FeatureWindow:
        requested_count = self._config.window_size if count is None else count
        stream_name = self._config.stream_name(instrument_key)
        rows = self._redis.xrevrange(
            stream_name,
            max=message_id,
            count=requested_count,
        )
        if len(rows) != requested_count:
            raise ValueError("Redis feature window does not contain the required candle count")
        newest_message_id = rows[0][0]
        if isinstance(newest_message_id, bytes):
            newest_message_id = newest_message_id.decode("utf-8")
        if str(newest_message_id) != message_id:
            raise ValueError("Redis feature window does not end at the consumed message")
        payloads = [_normalize_redis_fields(fields) for _message_id, fields in reversed(rows)]
        expected_message_id = (
            f"{int(_parse_utc_timestamp(payloads[-1]['ts_open']).timestamp() * 1_000)}-0"
        )
        if message_id != expected_message_id:
            raise ValueError("Redis message id does not match the final candle open time")
        return feature_window_from_redis_payloads_v1(
            payloads=payloads,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            instrument_key=instrument_key,
        )


@dataclass(frozen=True, slots=True)
class RlTradingRedisCandleMessage:
    instrument_key: str
    message_id: str


class RedisRlClosedCandleStream:
    def __init__(
        self,
        *,
        redis_client: Any,
        config: RlTradingInferenceRedisStreamsConfig,
        instrument_keys: Sequence[str],
    ) -> None:
        if redis_client is None:
            raise ValueError("RedisRlClosedCandleStream requires redis_client")
        if not config.enabled:
            raise ValueError("RedisRlClosedCandleStream requires enabled redis_streams config")
        normalized = tuple(sorted({value.strip() for value in instrument_keys if value.strip()}))
        if not normalized:
            raise ValueError("RedisRlClosedCandleStream requires instruments")
        self._redis = redis_client
        self._config = config
        self._instrument_keys = normalized
        self._created_groups: set[str] = set()
        self._consumer_name = (
            f"{socket.gethostname()}-{os.getpid()}"
            if config.consumer_name == "auto"
            else config.consumer_name
        )

    @classmethod
    def from_runtime_config(
        cls,
        *,
        config: RlTradingInferenceRedisStreamsConfig,
        instrument_keys: Sequence[str],
        environ: Mapping[str, str] | None = None,
    ) -> RedisRlClosedCandleStream:
        from redis import Redis

        selected_environ = os.environ if environ is None else environ
        auth_value = _env_optional_value(selected_environ, config.auth_env)
        auth_kwargs = {("pass" + "word"): auth_value} if auth_value is not None else {}
        redis_cls = cast(Any, Redis)
        client = redis_cls(
            host=config.host,
            port=config.port,
            db=config.db,
            socket_timeout=config.socket_timeout_s,
            socket_connect_timeout=config.connect_timeout_s,
            decode_responses=True,
            **auth_kwargs,
        )
        return cls(
            redis_client=client,
            config=config,
            instrument_keys=instrument_keys,
        )

    def read(self) -> tuple[RlTradingRedisCandleMessage, ...]:
        for instrument_key in self._instrument_keys:
            self._ensure_group(instrument_key=instrument_key)
            claimed = self._redis.xautoclaim(
                self._config.stream_name(instrument_key),
                self._config.consumer_group,
                self._consumer_name,
                self._config.pending_claim_min_idle_ms,
                start_id="0-0",
                count=self._config.read_count,
            )
            claimed_rows = _claimed_entries(claimed)
            if claimed_rows:
                return tuple(
                    RlTradingRedisCandleMessage(
                        instrument_key=instrument_key,
                        message_id=str(message_id),
                    )
                    for message_id, _fields in claimed_rows
                )
        streams = {
            self._config.stream_name(instrument_key): ">"
            for instrument_key in self._instrument_keys
        }
        raw = self._redis.xreadgroup(
            groupname=self._config.consumer_group,
            consumername=self._consumer_name,
            streams=streams,
            count=self._config.read_count,
            block=self._config.block_ms,
        )
        stream_to_instrument = {
            self._config.stream_name(instrument_key): instrument_key
            for instrument_key in self._instrument_keys
        }
        messages: list[RlTradingRedisCandleMessage] = []
        for stream_name, entries in raw:
            normalized_stream = (
                stream_name.decode("utf-8") if isinstance(stream_name, bytes) else str(stream_name)
            )
            instrument_key = stream_to_instrument.get(normalized_stream)
            if instrument_key is None:
                continue
            messages.extend(
                RlTradingRedisCandleMessage(
                    instrument_key=instrument_key,
                    message_id=(
                        message_id.decode("utf-8")
                        if isinstance(message_id, bytes)
                        else str(message_id)
                    ),
                )
                for message_id, _fields in entries
            )
        return tuple(messages)

    def ack(self, *, message: RlTradingRedisCandleMessage) -> None:
        self._redis.xack(
            self._config.stream_name(message.instrument_key),
            self._config.consumer_group,
            message.message_id,
        )

    def _ensure_group(self, *, instrument_key: str) -> None:
        from redis.exceptions import ResponseError

        stream_name = self._config.stream_name(instrument_key)
        if stream_name in self._created_groups:
            return
        try:
            self._redis.xgroup_create(
                name=stream_name,
                groupname=self._config.consumer_group,
                id="$",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
        self._created_groups.add(stream_name)


class RlTradingInferenceMetrics:
    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()
        self._ready = False
        self._degraded_reasons: set[str] = set()
        self._last_decision_unixtime = 0.0
        self.ready = Gauge(
            "rl_trading_inference_ready",
            "RL monitor-only inference worker readiness, 1 when ready",
            registry=self.registry,
        )
        self.decisions_total = Counter(
            "rl_trading_inference_decisions_total",
            "RL monitor-only inference decisions recorded by bounded outcome",
            ("mode", "outcome", "reason"),
            registry=self.registry,
        )
        self.segment_latency_seconds = Histogram(
            "rl_trading_inference_segment_latency_seconds",
            "RL monitor-only inference segment latency in seconds",
            ("segment",),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.feature_parity_total = Counter(
            "rl_trading_inference_feature_parity_total",
            "RL monitor-only train/live feature parity checks by result",
            ("result",),
            registry=self.registry,
        )
        self.degraded_state = Gauge(
            "rl_trading_inference_degraded_state",
            "RL monitor-only inference degraded state by bounded reason",
            ("reason",),
            registry=self.registry,
        )
        self.last_decision_unixtime = Gauge(
            "rl_trading_inference_last_decision_unixtime",
            "Unix timestamp of latest monitor-only inference decision",
            registry=self.registry,
        )
        self.candles_total = Counter(
            "rl_trading_inference_candles_total",
            "Closed 1m candles handled by bounded result",
            ("result",),
            registry=self.registry,
        )
        self.close_boundary_retries_total = Counter(
            "rl_trading_inference_close_boundary_retries_total",
            "Closed-candle reads deferred until the declared candle close boundary",
            registry=self.registry,
        )
        self.sessions_total = Counter(
            "rl_trading_inference_sessions_total",
            "Article session eligibility checks by bounded eligibility and reason",
            ("eligibility", "reason"),
            registry=self.registry,
        )
        self.actions_total = Counter(
            "rl_trading_inference_actions_total",
            "DQN monitor actions by requested action, effective action and result",
            ("requested_action", "effective_action", "result"),
            registry=self.registry,
        )
        self.virtual_exits_total = Counter(
            "rl_trading_inference_virtual_exits_total",
            "Monitor-only virtual exits by bounded validity and reason",
            ("valid", "reason"),
            registry=self.registry,
        )
        self.errors_total = Counter(
            "rl_trading_inference_errors_total",
            "Monitor-only runtime errors by bounded operation and reason",
            ("operation", "reason"),
            registry=self.registry,
        )
        self.safety_breaches_total = Counter(
            "rl_trading_inference_safety_breaches_total",
            "Monitor-only contract breaches by bounded reason",
            ("reason",),
            registry=self.registry,
        )
        self.feed_lag_seconds = Gauge(
            "rl_trading_inference_feed_lag_seconds",
            "Latest observed closed candle lag in seconds",
            registry=self.registry,
        )
        self.last_candle_unixtime = Gauge(
            "rl_trading_inference_last_candle_unixtime",
            "Unix timestamp of latest successfully acknowledged candle",
            registry=self.registry,
        )
        self.virtual_realized_pnl_quote = Gauge(
            "rl_trading_inference_virtual_realized_pnl_quote",
            "Cumulative monitor-only valid virtual PnL in quote currency since process start",
            registry=self.registry,
        )
        self.pending_virtual_positions = Gauge(
            "rl_trading_inference_pending_virtual_positions",
            "Current count of monitor-only pending virtual positions",
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "rl_trading_inference_model_loaded",
            "Trusted Stage 08K checkpoint loaded, 1 when loaded",
            registry=self.registry,
        )
        self._virtual_pnl_quote = 0.0

    def set_readiness(self, *, ready: bool, degraded_reasons: Sequence[str]) -> None:
        bounded_reasons = {_bounded_reason(reason) for reason in degraded_reasons}
        with self._lock:
            stale_reasons = self._degraded_reasons - bounded_reasons
            self._degraded_reasons = bounded_reasons
            self._ready = ready
            self.ready.set(1 if ready else 0)
            for reason in stale_reasons:
                self.degraded_state.labels(reason=reason).set(0)
            for reason in bounded_reasons:
                self.degraded_state.labels(reason=reason).set(1)

    def observe_decision(self, *, outcome: str, reason: str) -> None:
        bounded_outcome = _bounded_reason(outcome)
        bounded_reason = _bounded_reason(reason)
        with self._lock:
            self._last_decision_unixtime = time.time()
            self.decisions_total.labels(
                mode=STAGE13_MODE_V1,
                outcome=bounded_outcome,
                reason=bounded_reason,
            ).inc()
            self.last_decision_unixtime.set(self._last_decision_unixtime)

    def observe_segment_latency(self, *, segment: str, seconds: float) -> None:
        self.segment_latency_seconds.labels(segment=_bounded_reason(segment)).observe(seconds)

    def observe_feature_parity(self, *, result: str) -> None:
        self.feature_parity_total.labels(result=_bounded_reason(result)).inc()

    def observe_candle(self, *, result: str, candle_close_unixtime: float) -> None:
        now = time.time()
        self.candles_total.labels(result=_bounded_reason(result)).inc()
        self.last_candle_unixtime.set(candle_close_unixtime)
        self.feed_lag_seconds.set(max(0.0, now - candle_close_unixtime))

    def observe_close_boundary_retry(self) -> None:
        self.close_boundary_retries_total.inc()

    def observe_session(self, *, eligible: bool, reason: str) -> None:
        self.sessions_total.labels(
            eligibility="eligible" if eligible else "ineligible",
            reason=_bounded_reason(reason),
        ).inc()

    def observe_action(
        self, *, requested_action: str, effective_action: str, result: str
    ) -> None:
        self.actions_total.labels(
            requested_action=_bounded_reason(requested_action),
            effective_action=_bounded_reason(effective_action),
            result=_bounded_reason(result),
        ).inc()

    def observe_virtual_exit(self, *, valid: bool, reason: str, pnl_quote: float) -> None:
        self.virtual_exits_total.labels(
            valid="true" if valid else "false",
            reason=_bounded_reason(reason),
        ).inc()
        if valid:
            self._virtual_pnl_quote += float(pnl_quote)
            self.virtual_realized_pnl_quote.set(self._virtual_pnl_quote)

    def observe_error(self, *, operation: str, reason: str) -> None:
        self.errors_total.labels(
            operation=_bounded_reason(operation),
            reason=_bounded_reason(reason),
        ).inc()

    def observe_safety_breach(self, *, reason: str) -> None:
        self.safety_breaches_total.labels(reason=_bounded_reason(reason)).inc()

    def set_pending_virtual_positions(self, count: int) -> None:
        self.pending_virtual_positions.set(max(0, int(count)))

    def set_model_loaded(self, *, loaded: bool) -> None:
        self.model_loaded.set(1 if loaded else 0)

    def health_payload(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self._ready,
                "degraded_reasons": sorted(self._degraded_reasons),
                "last_decision_unixtime": self._last_decision_unixtime,
            }

    def render_latest(self) -> bytes:
        return generate_latest(self.registry)


class RlTradingInferenceHttpServer:
    def __init__(
        self,
        *,
        metrics: RlTradingInferenceMetrics,
        host: str = "127.0.0.1",
        port: int,
    ) -> None:
        self._server = _MetricsHttpServer((host, port), _InferenceRequestHandler)
        self._server.metrics = metrics
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def server_address(self) -> tuple[str, int]:
        address = cast(tuple[Any, ...], self._server.server_address)
        host, port = address[0], address[1]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class _MetricsHttpServer(ThreadingHTTPServer):
    metrics: RlTradingInferenceMetrics


class _InferenceRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        metrics = cast(_MetricsHttpServer, self.server).metrics
        if self.path == "/health/live":
            self._write_json(HTTPStatus.OK, {"live": True})
            return
        if self.path == "/health/ready":
            payload = metrics.health_payload()
            status = HTTPStatus.OK if payload["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_json(status, payload)
            return
        if self.path == "/metrics":
            body = metrics.render_latest()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        del format, args
        return

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def load_rl_trading_inference_runtime_config(path: str | Path) -> RlTradingInferenceRuntimeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("rl_trading_ml_runtime.yaml must contain a mapping")
    inference = _mapping(payload.get("inference"), "inference")
    source_events = _mapping(inference.get("source_events"), "inference.source_events")
    redis_streams = _mapping(inference.get("redis_streams"), "inference.redis_streams")
    latency_budget = _mapping(inference.get("latency_budget_ms"), "inference.latency_budget_ms")
    operator_context = _mapping(
        inference.get("operator_context"),
        "inference.operator_context",
    )
    monitor_policy = _mapping(
        inference.get("monitor_policy"),
        "inference.monitor_policy",
    )
    artifacts = _mapping(monitor_policy.get("artifacts"), "inference.monitor_policy.artifacts")
    instrument_rows = _sequence(inference.get("instruments"), "inference.instruments")
    artifact_root = Path(str(payload.get("artifact_root", "")))
    config = RlTradingInferenceRuntimeConfig(
        profile=str(payload.get("profile", "")),
        artifact_root=str(artifact_root),
        enabled=bool(inference.get("enabled", False)),
        mode=str(inference.get("mode", STAGE13_MODE_V1)),
        rollout_phase=str(inference.get("rollout_phase", "disabled")),
        metrics_port=int(inference.get("metrics_port", 9213)),
        health_check_enabled=bool(inference.get("health_check_enabled", True)),
        max_concurrent_processes=int(inference.get("max_concurrent_processes", 1)),
        torch_num_threads=int(inference.get("torch_num_threads", 1)),
        torch_num_interop_threads=int(inference.get("torch_num_interop_threads", 1)),
        max_rss_mb=int(inference.get("max_rss_mb", 0)),
        postgres_dsn_env=str(inference.get("postgres_dsn_env", "STRATEGY_PG_DSN")),
        state_path=Path(str(inference.get("state_path", ""))),
        operator_context=RlTradingInferenceOperatorContextConfig(
            owner_user_id=str(operator_context.get("owner_user_id", "")),
            strategy_id=str(operator_context.get("strategy_id", "")),
            strategy_run_id=str(operator_context.get("strategy_run_id", "")),
        ),
        instruments=tuple(
            RlTradingInferenceInstrumentConfig(
                exchange=str(row.get("exchange", "")),
                market_type=cast(Stage13MarketType, str(row.get("market_type", ""))),
                symbol=str(row.get("symbol", "")),
                instrument_key=str(row.get("instrument_key", "")),
            )
            for row in (_mapping(item, "inference.instruments[]") for item in instrument_rows)
        ),
        artifacts=Stage08kArtifactContract(
            artifact_root=artifact_root,
            candidate_manifest_path=Path(str(artifacts.get("candidate_manifest_path", ""))),
            candidate_manifest_sha256=str(artifacts.get("candidate_manifest_sha256", "")),
            evaluation_manifest_path=Path(
                str(artifacts.get("evaluation_manifest_path", ""))
            ),
            evaluation_manifest_sha256=str(
                artifacts.get("evaluation_manifest_sha256", "")
            ),
            checkpoint_path=Path(str(artifacts.get("checkpoint_path", ""))),
            checkpoint_sha256=str(artifacts.get("checkpoint_sha256", "")),
            normalization_stats_path=Path(
                str(artifacts.get("normalization_stats_path", ""))
            ),
            normalization_stats_file_sha256=str(
                artifacts.get("normalization_stats_file_sha256", "")
            ),
        ),
        monitor_policy=Stage08kMonitorPolicyConfig(
            policy_id=str(monitor_policy.get("policy_id", "")),
            direction_policy=str(monitor_policy.get("direction_policy", "")),
            virtual_hold_minutes=int(monitor_policy.get("virtual_hold_minutes", 0)),
            taker_fee_rate=float(monitor_policy.get("taker_fee_rate", 0.0)),
            slippage_rate=float(monitor_policy.get("slippage_rate", 0.0)),
            virtual_notional_quote=float(
                monitor_policy.get("virtual_notional_quote", 0.0)
            ),
            funding_model=str(monitor_policy.get("funding_model", "")),
        ),
        source_events=RlTradingInferenceSourceEventsConfig(
            enabled=bool(source_events.get("enabled", False)),
            source_type=str(source_events.get("source_type", "")),
            outcome=str(source_events.get("outcome", "")),
            outcome_reason=str(source_events.get("outcome_reason", "")),
        ),
        redis_streams=RlTradingInferenceRedisStreamsConfig(
            enabled=bool(redis_streams.get("enabled", False)),
            host=str(redis_streams.get("host", "127.0.0.1")),
            port=int(redis_streams.get("port", 6379)),
            db=int(redis_streams.get("db", 0)),
            auth_env=_optional_text(redis_streams.get("auth_env")),
            socket_timeout_s=float(redis_streams.get("socket_timeout_s", 2.0)),
            connect_timeout_s=float(redis_streams.get("connect_timeout_s", 2.0)),
            stream_prefix=str(redis_streams.get("stream_prefix", "md.candles.1m")),
            window_size=int(redis_streams.get("window_size", 30)),
            consumer_group=str(
                redis_streams.get("consumer_group", "rl.inference.monitor.v1")
            ),
            consumer_name=str(redis_streams.get("consumer_name", "auto")),
            read_count=int(redis_streams.get("read_count", 20)),
            block_ms=int(redis_streams.get("block_ms", 1_000)),
            pending_claim_min_idle_ms=int(
                redis_streams.get("pending_claim_min_idle_ms", 60_000)
            ),
        ),
        latency_budget=RlTradingInferenceLatencyBudgetConfig(
            candle_close_to_feature_ready_p95_ms=int(
                latency_budget.get("candle_close_to_feature_ready_p95", 250)
            ),
            feature_to_decision_p95_ms=int(latency_budget.get("feature_to_decision_p95", 100)),
            decision_to_source_event_p95_ms=int(
                latency_budget.get("decision_to_source_event_p95", 50)
            ),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: RlTradingInferenceRuntimeConfig) -> None:
    if config.mode != STAGE13_MODE_V1:
        raise ValueError("inference.mode must be monitor_only")
    if config.source_events.source_type != STAGE13_SOURCE_TYPE_V1:
        raise ValueError("inference.source_events.source_type must be ml_agent_decision")
    if config.source_events.outcome != STAGE13_SOURCE_EVENT_OUTCOME_V1:
        raise ValueError("inference.source_events.outcome must be no_intent")
    if config.source_events.outcome_reason != STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1:
        raise ValueError(
            "inference.source_events.outcome_reason must be monitor_only_no_intent"
        )
    if config.max_concurrent_processes != 1:
        raise ValueError("inference.max_concurrent_processes must remain 1")
    if config.metrics_port <= 0:
        raise ValueError("inference.metrics_port must be positive")
    if config.redis_streams.window_size <= 0:
        raise ValueError("inference.redis_streams.window_size must be positive")
    if config.redis_streams.window_size < 90:
        raise ValueError("inference.redis_streams.window_size must be at least 90 for Stage 08K")
    if config.enabled and not config.source_events.enabled:
        raise ValueError("enabled inference requires monitor-only source events")
    if not 1 <= len(config.instruments) <= 20:
        raise ValueError("inference.instruments must contain between 1 and 20 rows")
    phase_limits = {
        "disabled": None,
        "one_ticker_1h": 1,
        "five_ticker_24h": 5,
        "twenty_ticker_7d": 20,
    }
    if config.rollout_phase not in phase_limits:
        raise ValueError("unsupported inference.rollout_phase")
    expected_count = phase_limits[config.rollout_phase]
    if config.enabled and expected_count != len(config.instruments):
        raise ValueError("inference rollout phase instrument count mismatch")
    seen: set[str] = set()
    for instrument in config.instruments:
        if instrument.exchange != "binance" or instrument.market_type != "futures":
            raise ValueError("Stage 08K monitor instruments must be binance futures")
        expected_key = f"binance:futures:{instrument.symbol}"
        if instrument.instrument_key != expected_key:
            raise ValueError("Stage 08K monitor instrument_key mismatch")
        if instrument.instrument_key in seen:
            raise ValueError("Stage 08K monitor instrument duplicate")
        seen.add(instrument.instrument_key)
    for field_name in ("owner_user_id", "strategy_id", "strategy_run_id"):
        value = str(getattr(config.operator_context, field_name)).strip()
        if not value:
            raise ValueError(f"inference.operator_context.{field_name} is required")
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError(
                f"inference.operator_context.{field_name} must be a UUID"
            ) from exc
    if not config.postgres_dsn_env.strip():
        raise ValueError("inference.postgres_dsn_env is required")
    state_path = config.state_path.expanduser().resolve(strict=False)
    artifact_root = Path(config.artifact_root).expanduser().resolve(strict=False)
    try:
        state_path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("inference.state_path must be under artifact_root") from exc
    if config.redis_streams.read_count <= 0 or config.redis_streams.block_ms < 0:
        raise ValueError("invalid Redis consumer read settings")
    if config.redis_streams.pending_claim_min_idle_ms < 0:
        raise ValueError("invalid Redis pending claim setting")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return cast(Sequence[object], value)


def _claimed_entries(payload: object) -> list[tuple[object, Mapping[object, object]]]:
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        rows = payload[1]
        if isinstance(rows, list):
            return [
                (row[0], cast(Mapping[object, object], row[1]))
                for row in rows
                if isinstance(row, (tuple, list))
                and len(row) == 2
                and isinstance(row[1], Mapping)
            ]
    return []


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _env_optional_value(environ: Mapping[str, str], key: str | None) -> str | None:
    if key is None:
        return None
    value = environ.get(key, "").strip()
    return value or None


def _normalize_redis_fields(fields: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in fields.items():
        rendered_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        if isinstance(value, bytes):
            normalized[rendered_key] = value.decode("utf-8")
        else:
            normalized[rendered_key] = value
    return normalized


def _parse_utc_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Redis candle timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _bounded_reason(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized else "unknown"
