from __future__ import annotations

from fastapi.testclient import TestClient

from apps.exchange_execution.main.app import create_app


def test_exchange_execution_health_ready_and_metrics_with_disabled_redis() -> None:
    app = create_app(
        environ={
            "ROEHUB_ENV": "dev",
            "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "configs/dev/exchange_execution.yaml",
            "STRATEGY_FAIL_FAST": "false",
        }
    )

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        metrics = client.get("/metrics")

    assert ready.status_code == 200
    payload = ready.json()
    assert payload["service"] == "exchange-execution"
    assert payload["status"] == "degraded"
    assert payload["adapter_mode"] == "disabled"
    assert {item["name"] for item in payload["dependencies"]} >= {
        "config",
        "adapter",
        "postgres",
        "redis",
        "backpressure",
        "dlq",
        "clock_drift",
    }
    assert metrics.status_code == 200
    assert "exchange_execution_adapter_disabled 1.0" in metrics.text


def test_exchange_execution_run_once_reports_redis_unavailable() -> None:
    app = create_app(
        environ={
            "ROEHUB_ENV": "prod",
            "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "configs/prod/exchange_execution.yaml",
            "STRATEGY_FAIL_FAST": "false",
        }
    )

    with TestClient(app) as client:
        response = client.post("/internal/v1/run-once")

    assert response.status_code == 503
    assert response.json()["reason"] == "ConnectionError"
