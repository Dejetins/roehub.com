from __future__ import annotations

import json
import logging
import resource
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigLlmAttempt,
    BacktestAiConfigWorkerResult,
)

log = logging.getLogger(__name__)

_HISTOGRAM_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    20,
    30,
    60,
    120,
)
_TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


class BacktestAiConfiguratorMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        PlatformCollector(registry=self.registry)
        self.process_resident_memory_bytes = Gauge(
            "process_resident_memory_bytes",
            "Resident memory size in bytes.",
            registry=self.registry,
        )
        self.process_cpu_seconds_total = Gauge(
            "process_cpu_seconds_total",
            "Total user and system CPU time spent in seconds.",
            registry=self.registry,
        )
        self.jobs_total = Counter(
            "backtest_ai_config_jobs_total",
            "Backtest AI configurator jobs completed by terminal status.",
            ("status", "mode", "tier", "model_id"),
            registry=self.registry,
        )
        self.jobs_inflight = Gauge(
            "backtest_ai_config_jobs_inflight",
            "Backtest AI configurator jobs currently running in this worker.",
            ("mode", "model_id"),
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "backtest_ai_config_queue_depth",
            "Backtest AI configurator queue depth by bounded priority bucket.",
            ("priority",),
            registry=self.registry,
        )
        self.active_generations = Gauge(
            "backtest_ai_config_active_generations",
            "Backtest AI configurator active MLX generations.",
            ("model_id",),
            registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "backtest_ai_config_queue_wait_seconds",
            "Backtest AI configurator queue wait in seconds.",
            ("mode", "tier", "model_id"),
            buckets=_HISTOGRAM_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.stage_duration_seconds = Histogram(
            "backtest_ai_config_stage_duration_seconds",
            "Backtest AI configurator worker stage duration in seconds.",
            ("stage", "mode", "model_id"),
            buckets=_HISTOGRAM_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.llm_latency_seconds = Histogram(
            "backtest_ai_config_llm_latency_seconds",
            "Backtest AI configurator LLM request latency in seconds.",
            ("model_id",),
            buckets=_HISTOGRAM_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.total_latency_seconds = Histogram(
            "backtest_ai_config_total_latency_seconds",
            "Backtest AI configurator total job latency in seconds.",
            ("mode", "tier", "model_id"),
            buckets=_HISTOGRAM_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.prompt_tokens_estimated = Histogram(
            "backtest_ai_config_prompt_tokens_estimated",
            "Backtest AI configurator estimated prompt tokens.",
            ("model_id",),
            buckets=_TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.completion_tokens_estimated = Histogram(
            "backtest_ai_config_completion_tokens_estimated",
            "Backtest AI configurator estimated completion tokens.",
            ("model_id",),
            buckets=_TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.validation_failures_total = Counter(
            "backtest_ai_config_validation_failures_total",
            "Backtest AI configurator validation failures by bounded code.",
            ("code",),
            registry=self.registry,
        )
        self.repair_attempts_total = Counter(
            "backtest_ai_config_repair_attempts_total",
            "Backtest AI configurator repair attempts by result.",
            ("result", "model_id"),
            registry=self.registry,
        )
        self.security_decisions_total = Counter(
            "backtest_ai_config_security_decisions_total",
            "Backtest AI configurator security decisions.",
            ("decision", "flag"),
            registry=self.registry,
        )
        self.output_gate_failures_total = Counter(
            "backtest_ai_config_output_gate_failures_total",
            "Backtest AI configurator output gate failures by bounded code.",
            ("code",),
            registry=self.registry,
        )
        self.quota_rejections_total = Counter(
            "backtest_ai_config_quota_rejections_total",
            "Backtest AI configurator quota rejections.",
            ("tier", "window"),
            registry=self.registry,
        )
        self.capacity_rejections_total = Counter(
            "backtest_ai_config_capacity_rejections_total",
            "Backtest AI configurator capacity rejections.",
            ("reason",),
            registry=self.registry,
        )
        self.applied_total = Counter(
            "backtest_ai_config_applied_total",
            "Backtest AI configurator applied feedback observed by worker.",
            ("mode", "tier"),
            registry=self.registry,
        )
        self.model_reload_total = Counter(
            "backtest_ai_config_model_reload_total",
            "Backtest AI configurator model reload outcomes.",
            ("result", "model_id"),
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "backtest_ai_config_model_loaded",
            "Backtest AI configurator active model readiness.",
            ("model_id",),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "backtest_ai_config_model_info",
            "Backtest AI configurator active model metadata.",
            ("model_id", "runtime", "quantization"),
            registry=self.registry,
        )
        self._initialize_bounded_labels()
        self.refresh_process_metrics()

    def observe_result(
        self,
        *,
        result: BacktestAiConfigWorkerResult,
        duration_seconds: float,
    ) -> None:
        if not result.claimed:
            return
        job = result.job
        status = "lease_lost" if result.lease_lost or job is None else job.state
        mode = "unknown" if job is None else job.mode
        tier = _tier_from_job(job=job)
        model_id = _model_id_from_job(job=job)
        self.jobs_total.labels(
            status=_bounded(status),
            mode=_bounded(mode),
            tier=tier,
            model_id=model_id,
        ).inc()
        self.stage_duration_seconds.labels(
            stage="worker_iteration",
            mode=_bounded(mode),
            model_id=model_id,
        ).observe(max(duration_seconds, 0.0))
        if job is not None and job.started_at is not None:
            self.queue_wait_seconds.labels(
                mode=job.mode,
                tier=tier,
                model_id=model_id,
            ).observe(max((job.started_at - job.queued_at).total_seconds(), 0.0))
        if job is not None and job.finished_at is not None:
            self.total_latency_seconds.labels(
                mode=job.mode,
                tier=tier,
                model_id=model_id,
            ).observe(max((job.finished_at - job.queued_at).total_seconds(), 0.0))
        else:
            self.total_latency_seconds.labels(
                mode=_bounded(mode),
                tier=tier,
                model_id=model_id,
            ).observe(max(duration_seconds, 0.0))
        if job is not None and job.applied_at is not None:
            self.applied_total.labels(mode=job.mode, tier=tier).inc()
        if job is not None:
            for item in job.validation_errors_json:
                code = item.get("code")
                if isinstance(code, str) and code.strip():
                    bounded_code = _bounded(code)
                    self.validation_failures_total.labels(code=bounded_code).inc()
                    self.output_gate_failures_total.labels(code=bounded_code).inc()
            self._observe_security(job.last_error_json)

    def observe_attempts(
        self,
        *,
        attempts: tuple[BacktestAiConfigLlmAttempt, ...],
        model_id: str,
    ) -> None:
        for attempt in attempts:
            if attempt.latency_ms is not None:
                self.llm_latency_seconds.labels(model_id=model_id).observe(
                    max(attempt.latency_ms / 1000, 0.0)
                )
            if attempt.input_tokens_estimate is not None:
                self.prompt_tokens_estimated.labels(model_id=model_id).observe(
                    max(attempt.input_tokens_estimate, 0)
                )
            if attempt.output_tokens_estimate is not None:
                self.completion_tokens_estimated.labels(model_id=model_id).observe(
                    max(attempt.output_tokens_estimate, 0)
                )
            if attempt.attempt_kind == "repair":
                result = "success" if attempt.success else "failed"
                self.repair_attempts_total.labels(
                    result=result,
                    model_id=model_id,
                ).inc()

    def set_queue_depth(self, *, queued: int, running: int, repairing: int) -> None:
        self.queue_depth.labels(priority="queued").set(max(queued, 0))
        self.queue_depth.labels(priority="running").set(max(running, 0))
        self.queue_depth.labels(priority="repairing").set(max(repairing, 0))

    def set_active_generation(self, *, model_id: str, active: bool) -> None:
        self.active_generations.labels(model_id=model_id).set(1 if active else 0)
        self.jobs_inflight.labels(mode="unknown", model_id=model_id).set(
            1 if active else 0
        )

    def set_model_metadata(
        self,
        *,
        model_id: str,
        loaded: bool,
        runtime: str = "mlx",
        quantization: str = "unknown",
    ) -> None:
        self.model_loaded.labels(model_id=model_id).set(1 if loaded else 0)
        self.model_info.labels(
            model_id=model_id,
            runtime=runtime,
            quantization=quantization,
        ).set(1)

    def refresh_process_metrics(self) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        self.process_resident_memory_bytes.set(float(rss_bytes))
        self.process_cpu_seconds_total.set(float(usage.ru_utime + usage.ru_stime))

    def _observe_security(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            self.security_decisions_total.labels(decision="allow", flag="none").inc()
            return
        raw_decision = payload.get("security_decision")
        decision = _bounded(raw_decision if isinstance(raw_decision, str) else "allow")
        raw_flags = payload.get("security_flags")
        if isinstance(raw_flags, list) and raw_flags:
            for flag in raw_flags:
                if isinstance(flag, str) and flag.strip():
                    self.security_decisions_total.labels(
                        decision=decision,
                        flag=_bounded(flag),
                    ).inc()
            return
        self.security_decisions_total.labels(decision=decision, flag="none").inc()

    def _initialize_bounded_labels(self) -> None:
        for status in (
            "ready",
            "needs_clarification",
            "blocked_by_policy",
            "input_too_large",
            "security_review",
            "failed",
            "cancelled",
            "lease_lost",
        ):
            self.jobs_total.labels(
                status=status,
                mode="unknown",
                tier="unknown",
                model_id="unknown",
            ).inc(0)
        for priority in ("queued", "running", "repairing"):
            self.queue_depth.labels(priority=priority).set(0)
        for decision in ("allow", "allow_with_audit", "block", "security_review"):
            self.security_decisions_total.labels(decision=decision, flag="none").inc(0)
        for result in ("success", "failed"):
            self.model_reload_total.labels(result=result, model_id="unknown").inc(0)
            self.repair_attempts_total.labels(result=result, model_id="unknown").inc(0)
        self.quota_rejections_total.labels(tier="unknown", window="unknown").inc(0)
        self.capacity_rejections_total.labels(reason="unknown").inc(0)
        self.applied_total.labels(mode="unknown", tier="unknown").inc(0)
        self.jobs_inflight.labels(mode="unknown", model_id="unknown").set(0)
        self.active_generations.labels(model_id="unknown").set(0)
        self.queue_wait_seconds.labels(
            mode="unknown",
            tier="unknown",
            model_id="unknown",
        ).observe(0)
        self.stage_duration_seconds.labels(
            stage="worker_iteration",
            mode="unknown",
            model_id="unknown",
        ).observe(0)
        self.llm_latency_seconds.labels(model_id="unknown").observe(0)
        self.total_latency_seconds.labels(
            mode="unknown",
            tier="unknown",
            model_id="unknown",
        ).observe(0)
        self.prompt_tokens_estimated.labels(model_id="unknown").observe(0)
        self.completion_tokens_estimated.labels(model_id="unknown").observe(0)


@dataclass(slots=True)
class BacktestAiConfiguratorHealthState:
    config_loaded: bool
    model_id: str
    model_path: str
    drain_mode: bool
    readiness_checks: tuple[Callable[[], tuple[bool, str]], ...] = ()
    loop_started: bool = False
    stopping: bool = False
    last_loop_unixtime: float = field(default_factory=time.time)

    def mark_loop_started(self) -> None:
        self.loop_started = True
        self.last_loop_unixtime = time.time()

    def mark_loop_tick(self) -> None:
        self.last_loop_unixtime = time.time()

    def mark_stopping(self) -> None:
        self.stopping = True

    def live_payload(self) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "status": "live",
                "loop_started": self.loop_started,
                "stopping": self.stopping,
                "last_loop_unixtime": self.last_loop_unixtime,
            },
        )

    def ready_payload(self) -> tuple[int, dict[str, Any]]:
        checks: dict[str, bool] = {
            "config_loaded": self.config_loaded,
            "model_registry": bool(self.model_id.strip()),
            "queue_loop": self.loop_started and not self.stopping,
            "drain_mode": not self.drain_mode,
        }
        for check in self.readiness_checks:
            ok, name = check()
            checks[name] = ok
        ready = all(checks.values())
        return (
            200 if ready else 503,
            {
                "status": "ready" if ready else "not_ready",
                "checks": checks,
                "model_id": self.model_id,
                "model_path_configured": bool(self.model_path.strip()),
            },
        )


def start_backtest_ai_configurator_http_server(
    *,
    host: str,
    port: int,
    metrics: BacktestAiConfiguratorMetrics,
    health: BacktestAiConfiguratorHealthState,
    before_metrics: Callable[[], None] | None = None,
) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health/live":
                status, payload = health.live_payload()
                self._send_json(status=status, payload=payload)
                return
            if self.path == "/health/ready":
                status, payload = health.ready_payload()
                self._send_json(status=status, payload=payload)
                return
            if self.path == "/metrics":
                if before_metrics is not None:
                    before_metrics()
                metrics.refresh_process_metrics()
                body = generate_latest(metrics.registry)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            log.debug("backtest-ai-configurator ops http: " + format, *args)

        def _send_json(self, *, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"backtest-ai-configurator-ops-http-{port}",
        daemon=True,
    )
    thread.start()
    log.info("event=ops_http_started service=backtest-ai-configurator-worker port=%s", port)
    return server


def _model_id_from_job(*, job: Any | None) -> str:
    if job is None:
        return "unknown"
    value = getattr(job, "model_id", None)
    if isinstance(value, str) and value.strip():
        return _bounded(value)
    return "unknown"


def _tier_from_job(*, job: Any | None) -> str:
    if job is None:
        return "unknown"
    feedback = getattr(job, "user_feedback_json", None)
    if isinstance(feedback, Mapping):
        tier = feedback.get("tier")
        if isinstance(tier, str) and tier.strip():
            return _bounded(tier)
    return "unknown"


def _bounded(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        return "unknown"
    return "".join(ch for ch in normalized[:80] if ch.isalnum() or ch == "_") or "unknown"
