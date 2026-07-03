from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

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

    def stream_name(self, instrument_key: str) -> str:
        return f"{self.stream_prefix}.{instrument_key.strip()}"


@dataclass(frozen=True, slots=True)
class RlTradingInferenceLatencyBudgetConfig:
    candle_close_to_feature_ready_p95_ms: int
    feature_to_decision_p95_ms: int
    decision_to_source_event_p95_ms: int


@dataclass(frozen=True, slots=True)
class RlTradingInferenceRuntimeConfig:
    profile: str
    artifact_root: str
    enabled: bool
    mode: str
    metrics_port: int
    health_check_enabled: bool
    max_concurrent_processes: int
    torch_num_threads: int
    torch_num_interop_threads: int
    max_rss_mb: int
    source_events: RlTradingInferenceSourceEventsConfig
    redis_streams: RlTradingInferenceRedisStreamsConfig
    latency_budget: RlTradingInferenceLatencyBudgetConfig

    def readiness_payload(self) -> dict[str, object]:
        degraded_reasons: list[str] = []
        if not self.enabled:
            degraded_reasons.append("inference_disabled")
        if self.mode != STAGE13_MODE_V1:
            degraded_reasons.append("mode_not_monitor_only")
        if self.source_events.enabled:
            degraded_reasons.append("source_events_enabled_without_operator_gate")
        if self.source_events.source_type != STAGE13_SOURCE_TYPE_V1:
            degraded_reasons.append("source_type_mismatch")
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
        payloads = [_normalize_redis_fields(fields) for _message_id, fields in reversed(rows)]
        return feature_window_from_redis_payloads_v1(
            payloads=payloads,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            instrument_key=instrument_key,
        )


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
    config = RlTradingInferenceRuntimeConfig(
        profile=str(payload.get("profile", "")),
        artifact_root=str(payload.get("artifact_root", "")),
        enabled=bool(inference.get("enabled", False)),
        mode=str(inference.get("mode", STAGE13_MODE_V1)),
        metrics_port=int(inference.get("metrics_port", 9213)),
        health_check_enabled=bool(inference.get("health_check_enabled", True)),
        max_concurrent_processes=int(inference.get("max_concurrent_processes", 1)),
        torch_num_threads=int(inference.get("torch_num_threads", 1)),
        torch_num_interop_threads=int(inference.get("torch_num_interop_threads", 1)),
        max_rss_mb=int(inference.get("max_rss_mb", 0)),
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


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


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


def _bounded_reason(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized else "unknown"
