from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from apps.monitoring.operational_health import (
    OperationalHealthService,
    OperationalManifest,
    OperationalProbe,
    OperationalStatus,
    build_loki_log_sink,
    create_operational_health_app,
    probe_operational_target,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _manifest() -> OperationalManifest:
    return OperationalManifest.model_validate(
        {
            "schema": "io.roehub.operational-manifest/v1alpha1",
            "profile": "base",
            "services": [
                {
                    "service_id": "api",
                    "capability": "product.web_api",
                    "kind": "http_json",
                    "target": "http://api:8000/health",
                    "runbook_id": "web.api-health-degraded",
                    "action_ref": "restart_service",
                },
                {
                    "service_id": "postgresql",
                    "capability": "storage.postgresql",
                    "kind": "tcp_reachability",
                    "target": "postgresql:5432",
                    "runbook_id": "runtime.service-degraded",
                    "action_ref": "diagnostics",
                    "required": False,
                },
                {
                    "service_id": "redis",
                    "capability": "transport.redis",
                    "kind": "tcp_reachability",
                    "target": "redis:6379",
                    "runbook_id": "runtime.service-degraded",
                    "action_ref": "diagnostics",
                    "required": False,
                },
            ],
        }
    )


def test_operational_health_maps_states_and_emits_only_bounded_labels() -> None:
    outcomes = {
        "api": ("stopped", "probe.connection_refused"),
        "postgresql": ("degraded", "probe.http_503"),
        "redis": ("unknown", "probe.timeout"),
    }
    transitions: list[OperationalStatus] = []
    service = OperationalHealthService(
        manifest=_manifest(),
        probe=lambda spec: outcomes[spec.service_id],  # type: ignore[return-value]
        log_sink=lambda status: transitions.append(status) is None,
    )

    snapshot = service.refresh()

    assert snapshot.overall_state == "stopped"
    assert {item.state for item in snapshot.services} == {
        "degraded",
        "stopped",
        "unknown",
    }
    assert len(transitions) == 3
    metrics = service.metrics_text()
    assert 'service="api"' in metrics
    assert 'state="stopped"' in metrics
    for forbidden in ("organization", "account", "target=", "password", "token"):
        assert forbidden not in metrics.lower()


def test_operational_health_api_and_json_schema_match() -> None:
    service = OperationalHealthService(
        manifest=_manifest(),
        probe=lambda _spec: ("ready", "probe.http_ready"),
    )
    with TestClient(create_operational_health_app(service=service)) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "reason": "ready",
        }
        response = client.get("/api/v1/operational-health")
        assert response.status_code == 200
        assert response.json()["overall_state"] == "ready"
    schema = json.loads(
        (REPO_ROOT / "schemas/ops/operational-health.schema.json").read_text()
    )
    jsonschema.validate(response.json(), schema, format_checker=jsonschema.FormatChecker())


def test_loki_sink_sends_only_sanitized_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status = 204

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "apps.monitoring.operational_health.urllib.request.urlopen",
        fake_urlopen,
    )
    sink = build_loki_log_sink(url="http://loki:3100", profile="base")
    status = OperationalStatus(
        service_id="api",
        capability="product.web_api",
        state="stopped",
        detail_code="probe.connection_refused",
        runbook_id="web.api-health-degraded",
        action_ref="restart_service",
        required=True,
        observed_at=datetime.now(UTC),
    )

    assert sink(status) is True
    serialized = json.dumps(captured, sort_keys=True).lower()
    assert "operational_state_transition" in serialized
    for forbidden in ("organization", "account", "authorization", "password", "token"):
        assert forbidden not in serialized


def test_duplicate_operational_service_ids_are_rejected() -> None:
    probe = OperationalProbe(
        service_id="api",
        capability="product.web_api",
        kind="http_json",
        target="http://api:8000/health",
        runbook_id="web.api-health-degraded",
        action_ref="restart_service",
    )
    manifest = OperationalManifest(profile="base", services=(probe, probe))

    try:
        OperationalHealthService(manifest=manifest)
    except ValueError as error:
        assert str(error) == "operational service ids must be unique"
    else:
        raise AssertionError("duplicate service ids must be rejected")


def test_http_200_domain_degraded_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status = 200

        def read(self, _limit: int) -> bytes:
            return b'{"status":"degraded"}'

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "apps.monitoring.operational_health.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )
    spec = _manifest().services[0]

    assert probe_operational_target(spec) == ("degraded", "probe.domain_degraded")


def test_reachability_is_unknown_without_domain_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Socket:
        def __enter__(self) -> "_Socket":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "apps.monitoring.operational_health.socket.create_connection",
        lambda *_args, **_kwargs: _Socket(),
    )

    assert probe_operational_target(_manifest().services[1]) == (
        "unknown",
        "probe.reachable_no_readiness",
    )


def test_stale_snapshot_fails_readiness_and_suppresses_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic = {"value": 0.0}
    monkeypatch.setattr(
        "apps.monitoring.operational_health.time.monotonic",
        lambda: monotonic["value"],
    )
    service = OperationalHealthService(
        manifest=_manifest(),
        probe=lambda _spec: ("stopped", "probe.connection_refused"),
        freshness_sla_seconds=1.0,
    )
    service.refresh()
    monotonic["value"] = 2.0

    snapshot = service.snapshot()

    assert snapshot.overall_state == "unknown"
    assert {status.action_ref for status in snapshot.services} == {"diagnostics"}
    assert service.readiness() == (False, "snapshot_stale")


def test_probe_exception_is_bounded_unknown() -> None:
    def broken_probe(_spec: OperationalProbe) -> tuple[str, str]:
        raise RuntimeError("unbounded internal failure")

    service = OperationalHealthService(
        manifest=_manifest(),
        probe=broken_probe,  # type: ignore[arg-type]
    )

    snapshot = service.refresh()

    assert {status.state for status in snapshot.services} == {"unknown"}
    assert {status.detail_code for status in snapshot.services} == {
        "probe.internal_error"
    }
