#!/usr/bin/env python3
"""Prove Stage 20 generated observability with controlled local failures."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "configs/installation/generated/trading/compose.yaml"
OVERRIDE = ROOT / "tests/fixtures/observability-runtime-override.yaml"
STALE_OVERRIDE = ROOT / "tests/fixtures/observability-stale-override.yaml"
DEFAULT_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/20-observability-runtime-proof.json"
)
PROJECT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
MONITORING_SERVICES = (
    "alertmanager",
    "blackbox",
    "grafana",
    "loki",
    "operational-health",
    "prometheus",
)
STARTUP_EXCLUDED_SERVICES = frozenset({"exchange-execution"})
INJECTIONS = (
    ("web", "ready"),
    ("api", "ready"),
    ("strategy-live-runner", "ready"),
    ("postgresql", "unknown"),
    ("clickhouse", "unknown"),
    ("redis", "unknown"),
    ("openbao", "degraded"),
    ("plugin-gateway", "degraded"),
)


class ObservabilityProofError(RuntimeError):
    """Raised when a real Stage 20 boundary cannot be demonstrated."""


def _run(command: list[str], *, timeout: float = 300) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        raise ObservabilityProofError(
            f"command failed ({error.returncode}): {' '.join(command)}"
        ) from error
    return completed.stdout.strip()


def _compose(project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(COMPOSE),
        "-f",
        str(OVERRIDE),
    ]


def _stale_compose(project: str) -> list[str]:
    return [*_compose(project), "-f", str(STALE_OVERRIDE)]


def _port(project: str, service: str, container_port: int) -> int:
    value = _run([*_compose(project), "port", service, str(container_port)])
    try:
        return int(value.rsplit(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise ObservabilityProofError(f"missing host port for {service}") from error


def _get_json(
    port: int,
    path: str,
    *,
    params: dict[str, str] | None = None,
    expected_status: int = 200,
) -> Any:
    response = httpx.get(
        f"http://127.0.0.1:{port}{path}",
        params=params,
        timeout=5,
        follow_redirects=False,
    )
    if response.status_code != expected_status:
        raise ObservabilityProofError(
            f"unexpected bounded HTTP status: path={path}, status={response.status_code}"
        )
    return response.json()


def _wait_http(port: int, path: str, *, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}{path}",
                timeout=3,
                follow_redirects=False,
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise ObservabilityProofError(f"HTTP boundary did not recover: {path}")


def _wait_http_status(
    port: int,
    path: str,
    expected_status: int,
    *,
    timeout: float = 90,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status = 0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}{path}",
                timeout=3,
                follow_redirects=False,
            )
            last_status = response.status_code
            if response.status_code == expected_status:
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise ObservabilityProofError(
        f"HTTP boundary did not reach status {expected_status}: "
        f"path={path}, status={last_status}"
    )


def _snapshot(operational_port: int) -> dict[str, Any]:
    payload = _get_json(operational_port, "/api/v1/operational-health")
    if payload.get("schema") != "io.roehub.operational-health/v1alpha1":
        raise ObservabilityProofError("operational snapshot contract mismatch")
    return payload


def _service_state(snapshot: dict[str, Any], service: str) -> str:
    for item in snapshot["services"]:
        if item["service_id"] == service:
            return str(item["state"])
    raise ObservabilityProofError(f"service absent from operational manifest: {service}")


def _wait_state(
    operational_port: int,
    service: str,
    expected: set[str],
    *,
    timeout: float = 90,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout
    last_state = "missing"
    while time.monotonic() < deadline:
        snapshot = _snapshot(operational_port)
        last_state = _service_state(snapshot, service)
        if last_state in expected:
            detail = next(
                str(item["detail_code"])
                for item in snapshot["services"]
                if item["service_id"] == service
            )
            return last_state, detail
        time.sleep(1)
    raise ObservabilityProofError(
        f"operational state did not converge: service={service}, state={last_state}"
    )


def _prometheus_alert_firing(prometheus_port: int, service: str) -> bool:
    payload = _get_json(prometheus_port, "/api/v1/alerts")
    return any(
        item.get("labels", {}).get("alertname")
        == "RoehubOperationalServiceDegraded"
        and item.get("labels", {}).get("service") == service
        and item.get("state") == "firing"
        for item in payload.get("data", {}).get("alerts", [])
    )


def _prometheus_named_alert_firing(prometheus_port: int, alert_name: str) -> bool:
    payload = _get_json(prometheus_port, "/api/v1/alerts")
    return any(
        item.get("labels", {}).get("alertname") == alert_name
        and item.get("state") == "firing"
        for item in payload.get("data", {}).get("alerts", [])
    )


def _wait_alert(
    prometheus_port: int,
    service: str,
    *,
    firing: bool,
    timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _prometheus_alert_firing(prometheus_port, service) is firing:
            return
        time.sleep(1)
    raise ObservabilityProofError(
        f"Prometheus alert did not {'fire' if firing else 'resolve'} for {service}"
    )


def _wait_named_alert(
    prometheus_port: int,
    alert_name: str,
    *,
    firing: bool,
    timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _prometheus_named_alert_firing(prometheus_port, alert_name) is firing:
            return
        time.sleep(1)
    raise ObservabilityProofError(
        f"Prometheus alert did not {'fire' if firing else 'resolve'}: {alert_name}"
    )


def _alertmanager_has_service(alertmanager_port: int, service: str) -> bool:
    payload = _get_json(alertmanager_port, "/api/v2/alerts")
    return any(
        item.get("labels", {}).get("alertname")
        == "RoehubOperationalServiceDegraded"
        and item.get("labels", {}).get("service") == service
        for item in payload
    )


def _wait_alertmanager(
    alertmanager_port: int,
    service: str,
    *,
    active: bool,
    timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _alertmanager_has_service(alertmanager_port, service) is active:
            return
        time.sleep(1)
    raise ObservabilityProofError(
        f"Alertmanager alert did not {'appear' if active else 'resolve'} for {service}"
    )


def _loki_transitions(loki_port: int, service: str) -> list[dict[str, Any]]:
    payload = _get_json(
        loki_port,
        "/loki/api/v1/query_range",
        params={
            "query": (
                '{source="roehub-operational-health",'
                f'profile="trading",service="{service}"}}'
            ),
            "limit": "500",
            "since": "2h",
        },
    )
    transitions: list[dict[str, Any]] = []
    for stream in payload.get("data", {}).get("result", []):
        for _timestamp, line in stream.get("values", []):
            value = json.loads(line)
            if value.get("event") == "operational_state_transition":
                transitions.append(value)
    return transitions


def _wait_loki_state(
    loki_port: int,
    service: str,
    state: str,
    *,
    timeout: float = 60,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        transitions = _loki_transitions(loki_port, service)
        if any(item.get("state") == state for item in transitions):
            return len(transitions)
        time.sleep(1)
    raise ObservabilityProofError(
        f"Loki transition did not appear: service={service}, state={state}"
    )


def _prometheus_history(prometheus_port: int, service: str) -> list[dict[str, Any]]:
    now = time.time()
    payload = _get_json(
        prometheus_port,
        "/api/v1/query_range",
        params={
            "query": (
                "roehub_operational_service_state"
                f'{{service="{service}",state="stopped"}}'
            ),
            "start": str(now - 3600),
            "end": str(now),
            "step": "1",
        },
    )
    return list(payload.get("data", {}).get("result", []))


def _monitoring_ready(ports: dict[str, int]) -> dict[str, str]:
    paths = {
        "alertmanager": "/-/ready",
        "blackbox": "/-/ready",
        "grafana": "/api/health",
        "loki": "/ready",
        "operational-health": "/health/ready",
        "prometheus": "/-/ready",
    }
    for service, path in paths.items():
        _wait_http(ports[service], path)
    return {service: "ready" for service in paths}


def _running_monitoring(project: str) -> list[str]:
    running = set(_run([*_compose(project), "ps", "--services", "--status", "running"]).split())
    expected: set[str] = set(MONITORING_SERVICES)
    missing = expected - running
    if missing:
        raise ObservabilityProofError(
            f"independent monitoring stopped with application service: {sorted(missing)}"
        )
    return sorted(expected & running)


def _volume_evidence(project: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for owner in ("alertmanager", "grafana", "loki", "prometheus"):
        name = f"{project}_{owner}-data"
        payload = json.loads(_run(["docker", "volume", "inspect", name]))[0]
        if payload.get("Labels", {}).get("io.roehub.state-owner") != owner:
            raise ObservabilityProofError(f"observability volume owner mismatch: {owner}")
        result[owner] = "persistent-volume-present"
    return result


def _residual_resources(project: str) -> dict[str, list[str]]:
    label = f"label=com.docker.compose.project={project}"
    commands = {
        "containers": ["docker", "ps", "-a", "--filter", label, "--format", "{{.ID}}"],
        "volumes": ["docker", "volume", "ls", "--filter", label, "--format", "{{.Name}}"],
        "networks": ["docker", "network", "ls", "--filter", label, "--format", "{{.Name}}"],
    }
    return {
        resource: sorted(value for value in _run(command).splitlines() if value)
        for resource, command in commands.items()
    }


def _cleanup_project(project: str, compose: list[str]) -> dict[str, Any]:
    _run([*compose, "down", "-v", "--remove-orphans"], timeout=180)
    residual = _residual_resources(project)
    if any(residual.values()):
        raise ObservabilityProofError(
            f"Compose cleanup left residual resources: {residual}"
        )
    return {
        "status": "completed",
        "project": project,
        "down_exit_status": 0,
        "residual_resources": residual,
    }


def verify_observability_runtime(
    *,
    output: Path,
    project: str,
    reuse: bool,
    keep: bool,
) -> dict[str, Any]:
    if PROJECT_PATTERN.fullmatch(project) is None:
        raise ObservabilityProofError("unsafe Docker Compose project name")
    compose = _compose(project)
    started_here = not reuse
    if started_here:
        try:
            _run([*compose, "config", "--quiet"])
            _run([*compose, "build", "operational-health"], timeout=360)
            startup_services = [
                service
                for service in _run([*compose, "config", "--services"]).splitlines()
                if service not in STARTUP_EXCLUDED_SERVICES
            ]
            _run([*compose, "up", "-d", *startup_services], timeout=360)
        except Exception:
            _cleanup_project(project, compose)
            raise
    ports = {
        "alertmanager": _port(project, "alertmanager", 9093),
        "blackbox": _port(project, "blackbox", 9115),
        "grafana": _port(project, "grafana", 3000),
        "loki": _port(project, "loki", 3100),
        "operational-health": _port(project, "operational-health", 9300),
        "prometheus": _port(project, "prometheus", 9090),
    }
    payload: dict[str, Any] | None = None
    try:
        readiness = _monitoring_ready(ports)
        baseline = _snapshot(ports["operational-health"])
        if _service_state(baseline, "openbao") != "degraded":
            raise ObservabilityProofError("fresh sealed OpenBao must map to degraded")
        if _service_state(baseline, "plugin-gateway") != "degraded":
            raise ObservabilityProofError(
                "plugin HTTP 200 domain-degraded response must not map to ready"
            )
        baseline_states = {
            service: _service_state(baseline, service) for service, _state in INJECTIONS
        }
        expected_baseline = dict(INJECTIONS)
        if baseline_states != expected_baseline:
            raise ObservabilityProofError(
                f"typed baseline mismatch: expected={expected_baseline}, actual={baseline_states}"
            )
        injections: list[dict[str, Any]] = []
        for service, recovery_state in INJECTIONS:
            _run([*compose, "stop", "-t", "10", service], timeout=60)
            stopped_state, stopped_detail = _wait_state(
                ports["operational-health"],
                service,
                {"stopped"},
            )
            running_monitoring = _running_monitoring(project)
            _monitoring_ready(ports)
            _wait_alert(ports["prometheus"], service, firing=True)
            if service == "api":
                _wait_alertmanager(ports["alertmanager"], service, active=True)
            stopped_log_count = _wait_loki_state(
                ports["loki"],
                service,
                "stopped",
            )
            _run([*compose, "start", service], timeout=60)
            recovered_state, recovered_detail = _wait_state(
                ports["operational-health"],
                service,
                {recovery_state},
            )
            if recovery_state == "ready":
                _wait_alert(ports["prometheus"], service, firing=False)
            if service == "api":
                _wait_alertmanager(ports["alertmanager"], service, active=False)
            recovered_log_count = _wait_loki_state(
                ports["loki"],
                service,
                recovered_state,
            )
            injections.append(
                {
                    "service": service,
                    "failure_state": stopped_state,
                    "failure_detail": stopped_detail,
                    "recovery_state": recovered_state,
                    "recovery_detail": recovered_detail,
                    "monitoring_running": running_monitoring,
                    "bounded_log_transitions": recovered_log_count,
                    "new_failure_transitions": recovered_log_count >= stopped_log_count,
                }
            )

        _run([*compose, "pause", "api"], timeout=60)
        paused_state, paused_detail = _wait_state(
            ports["operational-health"],
            "api",
            {"unknown"},
        )
        _wait_alert(ports["prometheus"], "api", firing=True)
        _run([*compose, "unpause", "api"], timeout=60)
        unpaused_state, unpaused_detail = _wait_state(
            ports["operational-health"],
            "api",
            {"ready"},
        )
        _wait_alert(ports["prometheus"], "api", firing=False)
        timeout_injection = {
            "service": "api",
            "failure_state": paused_state,
            "failure_detail": paused_detail,
            "recovery_state": unpaused_state,
            "recovery_detail": unpaused_detail,
        }

        api_stopped_history = _prometheus_history(ports["prometheus"], "api")
        api_logs_before_restart = _loki_transitions(ports["loki"], "api")
        if not api_stopped_history:
            raise ObservabilityProofError("Prometheus did not retain the API failure series")
        if not api_logs_before_restart:
            raise ObservabilityProofError("Loki did not retain the API transition log")

        _run([*compose, "restart", "prometheus", "loki"], timeout=90)
        ports["prometheus"] = _port(project, "prometheus", 9090)
        ports["loki"] = _port(project, "loki", 3100)
        _wait_http(ports["prometheus"], "/-/ready")
        _wait_http(ports["loki"], "/ready")
        api_logs_after_restart = _loki_transitions(ports["loki"], "api")
        if len(api_logs_after_restart) < len(api_logs_before_restart):
            raise ObservabilityProofError("Loki transitions did not survive restart")
        persisted_history = _prometheus_history(ports["prometheus"], "api")
        if not persisted_history:
            raise ObservabilityProofError("Prometheus failure series did not survive restart")

        _run(
            [
                *_stale_compose(project),
                "up",
                "-d",
                "--no-deps",
                "--force-recreate",
                "operational-health",
            ],
            timeout=120,
        )
        ports["operational-health"] = _port(project, "operational-health", 9300)
        stale_readiness = _wait_http_status(
            ports["operational-health"],
            "/health/ready",
            503,
            timeout=45,
        )
        stale_snapshot = _snapshot(ports["operational-health"])
        if stale_snapshot.get("overall_state") != "unknown" or any(
            item.get("state") != "unknown" or item.get("action_ref") != "diagnostics"
            for item in stale_snapshot.get("services", [])
        ):
            raise ObservabilityProofError(
                "stale operational snapshot did not fail closed to unknown diagnostics"
            )
        _wait_named_alert(
            ports["prometheus"],
            "RoehubOperationalSnapshotStale",
            firing=True,
        )
        _run(
            [
                *compose,
                "up",
                "-d",
                "--no-deps",
                "--force-recreate",
                "operational-health",
            ],
            timeout=120,
        )
        ports["operational-health"] = _port(project, "operational-health", 9300)
        _wait_http(ports["operational-health"], "/health/ready")
        _wait_named_alert(
            ports["prometheus"],
            "RoehubOperationalSnapshotStale",
            firing=False,
        )

        grafana_search = httpx.get(
            f"http://127.0.0.1:{ports['grafana']}/api/search",
            timeout=5,
            follow_redirects=False,
        )
        if grafana_search.status_code != 401:
            raise ObservabilityProofError("Grafana anonymous API access is not disabled")

        observed_states = {
            item["failure_state"] for item in injections
        } | {item["recovery_state"] for item in injections} | {
            timeout_injection["failure_state"],
            timeout_injection["recovery_state"],
        }
        if observed_states != {"ready", "degraded", "stopped", "unknown"}:
            raise ObservabilityProofError(
                f"four-state runtime proof incomplete: {sorted(observed_states)}"
            )
        payload = {
            "schema": "io.roehub.observability-runtime-proof/v1alpha1",
            "status": "passed",
            "compose_project": project,
            "profile": "trading",
            "baseline_overall_state": baseline["overall_state"],
            "baseline_service_states": baseline_states,
            "fresh_openbao_state": "degraded",
            "plugin_http_200_degraded_state": "degraded",
            "observed_operational_states": sorted(observed_states),
            "monitoring_readiness": readiness,
            "failure_injections": injections,
            "timeout_unknown_injection": timeout_injection,
            "stale_snapshot_fault_injection": {
                "readiness_status": 503,
                "readiness_reason": stale_readiness.get("reason"),
                "overall_state": "unknown",
                "actions": "diagnostics-only",
                "alert": "fired-and-resolved",
            },
            "api_alert_lifecycle": "fired-and-resolved",
            "alert_receiver": "local-audit-only",
            "prometheus_failure_series_persisted": True,
            "loki_transition_count_before_restart": len(api_logs_before_restart),
            "loki_transition_count_after_restart": len(api_logs_after_restart),
            "observability_volumes": _volume_evidence(project),
            "grafana_anonymous_api": "denied",
            "external_notifications": False,
            "external_provider_writes": False,
            "real_order_effects": False,
            "production_mutation": False,
        }
    finally:
        if keep:
            if payload is not None:
                payload["cleanup"] = {
                    "status": "resources_retained_by_request",
                    "project": project,
                }
        elif started_here or reuse:
            if payload is not None:
                payload["cleanup"] = _cleanup_project(project, compose)
            else:
                _cleanup_project(project, compose)
    if payload is None:
        raise ObservabilityProofError("runtime proof payload was not produced")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--project",
        default=f"roehub-stage20-{uuid4().hex[:8]}",
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = verify_observability_runtime(
            output=args.output,
            project=args.project,
            reuse=args.reuse,
            keep=args.keep,
        )
    except (ObservabilityProofError, OSError, ValueError, httpx.HTTPError) as error:
        print(f"observability runtime verification failed: {error}")
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
