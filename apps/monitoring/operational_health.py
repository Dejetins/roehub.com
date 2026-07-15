"""Independent, redacted operational health service for generated Roehub topology."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

OperationalState = Literal["ready", "degraded", "stopped", "unknown"]
ProbeFunction = Callable[["OperationalProbe"], tuple[OperationalState, str]]
LogSink = Callable[["OperationalStatus"], bool]


class OperationalProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    capability: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,63}$")
    kind: Literal["http_json", "http_reachability", "tcp_reachability", "openbao"]
    target: str = Field(min_length=3, max_length=512)
    runbook_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{1,127}$")
    action_ref: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,127}$")
    required: bool = True
    timeout_seconds: float = Field(default=1.0, gt=0, le=10)


class OperationalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.operational-manifest/v1alpha1"] = Field(
        default="io.roehub.operational-manifest/v1alpha1",
        alias="schema",
    )
    profile: Literal["base", "trading", "ml"]
    services: tuple[OperationalProbe, ...]


class OperationalStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service_id: str
    capability: str
    state: OperationalState
    detail_code: str
    runbook_id: str
    action_ref: str
    required: bool
    observed_at: datetime


class OperationalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.operational-health/v1alpha1"] = Field(
        default="io.roehub.operational-health/v1alpha1",
        alias="schema",
    )
    profile: Literal["base", "trading", "ml"]
    generated_at: datetime
    overall_state: OperationalState
    services: tuple[OperationalStatus, ...]


def load_operational_manifest(path: Path) -> OperationalManifest:
    return OperationalManifest.model_validate_json(path.read_text(encoding="utf-8"))


class OperationalHealthService:
    """Probe only generated endpoints and expose bounded state without raw payloads."""

    def __init__(
        self,
        *,
        manifest: OperationalManifest,
        probe: ProbeFunction | None = None,
        log_sink: LogSink | None = None,
        interval_seconds: float = 5.0,
        freshness_sla_seconds: float | None = None,
    ) -> None:
        if not 0.5 <= interval_seconds <= 300:
            raise ValueError("interval_seconds must be between 0.5 and 300")
        service_ids = tuple(item.service_id for item in manifest.services)
        if len(service_ids) != len(set(service_ids)):
            raise ValueError("operational service ids must be unique")
        self._manifest = manifest
        self._probe = probe or probe_operational_target
        self._log_sink = log_sink or (lambda _status: True)
        self._interval_seconds = interval_seconds
        self._freshness_sla_seconds = (
            freshness_sla_seconds
            if freshness_sla_seconds is not None
            else max(5.0, interval_seconds * 3)
        )
        if not 1.0 <= self._freshness_sla_seconds <= 900:
            raise ValueError("freshness_sla_seconds must be between 1 and 900")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._statuses: dict[str, OperationalStatus] = {}
        self._log_push_success = True
        self._last_refresh_completed_at: datetime | None = None
        self._last_refresh_monotonic: float | None = None
        self._worker_failed = False

    @property
    def manifest(self) -> OperationalManifest:
        return self._manifest

    def start(self) -> None:
        if self._thread is not None:
            return
        self.refresh()
        self._thread = threading.Thread(
            target=self._run,
            name="roehub-operational-health",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 2))
        self._thread = None

    def refresh(self) -> OperationalSnapshot:
        observed_at = datetime.now(UTC)
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(self._manifest.services)))) as pool:
            outcomes = tuple(pool.map(self._safe_probe, self._manifest.services))
        changed: list[OperationalStatus] = []
        next_statuses: dict[str, OperationalStatus] = {}
        with self._lock:
            previous = self._statuses
            for spec, (state, detail_code) in zip(self._manifest.services, outcomes, strict=True):
                status = OperationalStatus(
                    service_id=spec.service_id,
                    capability=spec.capability,
                    state=state,
                    detail_code=detail_code,
                    runbook_id=spec.runbook_id,
                    action_ref=(
                        spec.action_ref
                        if state == "stopped" and spec.action_ref == "restart_service"
                        else "diagnostics"
                    ),
                    required=spec.required,
                    observed_at=observed_at,
                )
                next_statuses[spec.service_id] = status
                if (
                    previous.get(spec.service_id) is None
                    or previous[spec.service_id].state != state
                ):
                    changed.append(status)
            self._statuses = next_statuses
            self._last_refresh_completed_at = observed_at
            self._last_refresh_monotonic = time.monotonic()
            self._worker_failed = False
        if changed:
            batch_success = True
            for status in changed:
                try:
                    batch_success = self._log_sink(status) and batch_success
                except Exception:  # pragma: no cover - defensive sink isolation
                    batch_success = False
            self._log_push_success = batch_success
        return self.snapshot()

    def snapshot(self) -> OperationalSnapshot:
        with self._lock:
            services = tuple(self._statuses[key] for key in sorted(self._statuses))
            generated_at = self._last_refresh_completed_at
            stale = self._is_stale_locked()
        if stale and services:
            services = tuple(
                item.model_copy(
                    update={
                        "state": "unknown",
                        "detail_code": "probe.snapshot_stale",
                        "action_ref": "diagnostics",
                    }
                )
                for item in services
            )
        states = {item.state for item in services if item.required}
        if not services or stale:
            overall: OperationalState = "unknown"
        elif "stopped" in states:
            overall = "stopped"
        elif "degraded" in states:
            overall = "degraded"
        elif "unknown" in states:
            overall = "unknown"
        else:
            overall = "ready"
        return OperationalSnapshot(
            profile=self._manifest.profile,
            generated_at=generated_at or datetime.now(UTC),
            overall_state=overall,
            services=services,
        )

    def readiness(self) -> tuple[bool, str]:
        with self._lock:
            if self._worker_failed:
                return False, "refresh_worker_failed"
            if self._last_refresh_completed_at is None:
                return False, "refresh_not_completed"
            if self._is_stale_locked():
                return False, "snapshot_stale"
        if self._thread is not None and not self._thread.is_alive():
            return False, "refresh_worker_stopped"
        return True, "ready"

    def metrics_text(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP roehub_operational_service_state Current bounded service state.",
            "# TYPE roehub_operational_service_state gauge",
        ]
        for item in snapshot.services:
            labels = {
                "action_ref": item.action_ref,
                "capability": item.capability,
                "profile": snapshot.profile,
                "runbook_id": item.runbook_id,
                "service": item.service_id,
                "state": item.state,
            }
            rendered = ",".join(
                f'{key}="{_prometheus_escape(value)}"' for key, value in sorted(labels.items())
            )
            lines.append(f"roehub_operational_service_state{{{rendered}}} 1")
        lines.extend(
            [
                "# HELP roehub_operational_log_push_success "
                "Whether the last bounded log push succeeded.",
                "# TYPE roehub_operational_log_push_success gauge",
                f"roehub_operational_log_push_success {1 if self._log_push_success else 0}",
                "# HELP roehub_operational_snapshot_fresh Whether the latest refresh is fresh.",
                "# TYPE roehub_operational_snapshot_fresh gauge",
                f"roehub_operational_snapshot_fresh {1 if self.readiness()[0] else 0}",
                "# HELP roehub_operational_snapshot_timestamp_seconds Latest snapshot time.",
                "# TYPE roehub_operational_snapshot_timestamp_seconds gauge",
                "roehub_operational_snapshot_timestamp_seconds "
                f"{snapshot.generated_at.timestamp():.3f}",
                "",
            ]
        )
        return "\n".join(lines)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self.refresh()
            except Exception:  # pragma: no cover - defensive worker isolation
                with self._lock:
                    self._worker_failed = True

    def _safe_probe(self, spec: OperationalProbe) -> tuple[OperationalState, str]:
        try:
            return self._probe(spec)
        except Exception:  # pragma: no cover - defensive probe isolation
            return "unknown", "probe.internal_error"

    def _is_stale_locked(self) -> bool:
        if self._last_refresh_monotonic is None:
            return True
        return time.monotonic() - self._last_refresh_monotonic > self._freshness_sla_seconds


def probe_operational_target(spec: OperationalProbe) -> tuple[OperationalState, str]:
    try:
        if spec.kind == "tcp_reachability":
            host, port_text = spec.target.rsplit(":", 1)
            with socket.create_connection((host, int(port_text)), timeout=spec.timeout_seconds):
                return "unknown", "probe.reachable_no_readiness"
        request = urllib.request.Request(
            spec.target,
            method="GET",
            headers={"User-Agent": "roehub-operational-health/v1"},
        )
        with urllib.request.urlopen(request, timeout=spec.timeout_seconds) as response:  # noqa: S310
            body = response.read(16_384)
            if spec.kind == "openbao":
                payload = json.loads(body)
                if not payload.get("initialized") or payload.get("sealed"):
                    return "degraded", "probe.openbao_not_ready"
                return "ready", "probe.openbao_ready"
            if spec.kind == "http_reachability":
                return "unknown", "probe.reachable_no_readiness"
            return _json_health_state(body)
    except urllib.error.HTTPError as error:
        return "degraded", f"probe.http_{error.code}"
    except (ConnectionRefusedError, ConnectionResetError):
        return "stopped", "probe.connection_refused"
    except (TimeoutError, socket.timeout):
        return "unknown", "probe.timeout"
    except (OSError, urllib.error.URLError):
        return "stopped", "probe.unreachable"
    except (ValueError, json.JSONDecodeError):
        return "degraded", "probe.response_invalid"


def _json_health_state(body: bytes) -> tuple[OperationalState, str]:
    payload = json.loads(body)
    if not isinstance(payload, dict):
        return "degraded", "probe.response_invalid"
    status = payload.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"ready", "ok"}:
            return "ready", "probe.domain_ready"
        if normalized in {"degraded", "not_ready", "unhealthy"}:
            return "degraded", "probe.domain_degraded"
        if normalized == "stopped":
            return "stopped", "probe.domain_stopped"
        if normalized in {"unknown", "live", "alive"}:
            return "unknown", "probe.domain_unknown"
    ready = payload.get("ready")
    if ready is True:
        return "ready", "probe.domain_ready"
    if ready is False:
        return "degraded", "probe.domain_degraded"
    return "unknown", "probe.domain_unknown"


def build_loki_log_sink(*, url: str, profile: str) -> LogSink:
    endpoint = url.rstrip("/") + "/loki/api/v1/push"

    def push(status: OperationalStatus) -> bool:
        line = json.dumps(
            {
                "action_ref": status.action_ref,
                "detail_code": status.detail_code,
                "event": "operational_state_transition",
                "service": status.service_id,
                "state": status.state,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        payload = {
            "streams": [
                {
                    "stream": {
                        "profile": profile,
                        "service": status.service_id,
                        "source": "roehub-operational-health",
                    },
                    "values": [[str(time.time_ns()), line]],
                }
            ]
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:  # noqa: S310
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError):
            return False

    return push


def create_operational_health_app(*, service: OperationalHealthService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(
        title="Roehub operational health",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        is_ready, reason = service.readiness()
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "degraded", "reason": reason},
        )

    @app.get("/api/v1/operational-health", response_model=OperationalSnapshot)
    def operational_health() -> OperationalSnapshot:
        return service.snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(service.metrics_text(), media_type="text/plain; version=0.0.4")

    @app.exception_handler(Exception)
    def unexpected_error(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "operational_health.unavailable"}},
        )

    return app


def _prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve generated Roehub operational health")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--freshness-sla-seconds", type=float, default=15.0)
    parser.add_argument("--loki-url", default="")
    args = parser.parse_args(argv)
    manifest = load_operational_manifest(args.manifest)
    sink = (
        build_loki_log_sink(url=args.loki_url, profile=manifest.profile)
        if args.loki_url
        else None
    )
    service = OperationalHealthService(
        manifest=manifest,
        interval_seconds=args.interval_seconds,
        freshness_sla_seconds=args.freshness_sla_seconds,
        log_sink=sink,
    )
    uvicorn.run(
        create_operational_health_app(service=service),
        host=args.host,
        port=args.port,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
