from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from apps.api.operational_health_client import (
    HttpOperationalHealthClient,
    OperationalHealthClientError,
    build_operational_health_client_from_environ,
)


def _payload() -> dict[str, object]:
    observed_at = datetime.now(UTC).isoformat()
    return {
        "schema": "io.roehub.operational-health/v1alpha1",
        "profile": "base",
        "generated_at": observed_at,
        "overall_state": "degraded",
        "services": [
            {
                "service_id": "api",
                "capability": "product.web_api",
                "state": "degraded",
                "detail_code": "probe.http_503",
                "runbook_id": "web.api-health-degraded",
                "action_ref": "restart_service",
                "required": True,
                "observed_at": observed_at,
            }
        ],
    }


def test_http_operational_health_client_validates_typed_snapshot() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_payload(), request=request)
    )
    client = HttpOperationalHealthClient(
        base_url="http://operational-health:9300",
        transport=transport,
    )

    snapshot = client.snapshot()

    assert snapshot.profile == "base"
    assert snapshot.services[0].service_id == "api"


def test_http_operational_health_client_fails_closed_with_sanitized_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="sensitive upstream", request=request)
    )
    client = HttpOperationalHealthClient(
        base_url="http://operational-health:9300",
        transport=transport,
    )

    with pytest.raises(
        OperationalHealthClientError,
        match="operational health snapshot is unavailable",
    ):
        client.snapshot()


def test_operational_health_client_environment_is_optional_and_typed() -> None:
    assert build_operational_health_client_from_environ(environ={}) is None
    client = build_operational_health_client_from_environ(
        environ={
            "ROEHUB_OPERATIONAL_HEALTH_URL": "http://operational-health:9300",
            "ROEHUB_OPERATIONAL_HEALTH_TIMEOUT_SECONDS": "1.5",
        }
    )
    assert isinstance(client, HttpOperationalHealthClient)
    assert client.timeout_seconds == 1.5
