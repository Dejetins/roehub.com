from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trading.contexts.exchange_control.adapters.inbound.http.app import (
    EXCHANGE_CONTROL_METRICS_PORT,
    ExchangeControlRuntimeConfig,
    create_exchange_control_app,
)
from trading.contexts.exchange_control.application.service_identity import (
    EXCHANGE_CONTROL_SERVICE_IDENTITY,
    ExchangeControlServiceIdentity,
)


def _build_client() -> TestClient:
    config = ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "dev"})
    return TestClient(create_exchange_control_app(config=config))


def test_service_identity_is_mandatory_exchange_control() -> None:
    identity = ExchangeControlServiceIdentity(name=EXCHANGE_CONTROL_SERVICE_IDENTITY)

    assert identity.name == "exchange-control"

    with pytest.raises(ValueError, match="service identity"):
        ExchangeControlServiceIdentity(name="apps-api")


def test_prod_runtime_requires_localhost_port_9205_and_disabled_validation() -> None:
    config = ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "prod"})

    assert config.service_identity_name == "exchange-control"
    assert config.bind_host == "127.0.0.1"
    assert config.metrics_port == EXCHANGE_CONTROL_METRICS_PORT
    assert not config.real_exchange_validation_enabled

    with pytest.raises(ValueError, match="port 9205"):
        ExchangeControlRuntimeConfig.from_environ(
            environ={"ROEHUB_ENV": "prod"},
            metrics_port=9206,
        )

    with pytest.raises(ValueError, match="real exchange validation"):
        ExchangeControlRuntimeConfig.from_environ(
            environ={
                "ROEHUB_ENV": "prod",
                "ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED": "true",
            }
        )


def test_health_ready_exposes_service_identity_and_disabled_external_validation() -> None:
    client = _build_client()

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "exchange-control",
        "service_identity": "exchange-control",
        "checks": [
            {"name": "service_identity", "status": "ready"},
            {"name": "external_exchange_validation", "status": "ready"},
        ],
    }


def test_metrics_expose_secret_safe_exchange_control_series() -> None:
    client = _build_client()

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "exchange_control_active 1.0" in response.text
    assert "exchange_connection_validation_total" in response.text
    assert 'exchange="none"' in response.text
    assert "api_key" not in response.text
    assert "connection_id" not in response.text
