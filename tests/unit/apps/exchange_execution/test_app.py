from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from apps.exchange_execution.main.app import ExchangeExecutionMetrics, create_app


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


def test_exchange_execution_metrics_record_rate_limit_waits() -> None:
    metrics = ExchangeExecutionMetrics()

    metrics.record_rate_limit_wait("bybit", "submit", 0.125)

    payload = generate_latest(metrics.registry).decode("utf-8")
    assert (
        'exchange_execution_rate_limit_wait_total{exchange="bybit",operation="submit"} 1.0'
        in payload
    )
    assert (
        'exchange_execution_rate_limit_wait_seconds_sum{exchange="bybit",operation="submit"} '
        "0.125"
    ) in payload


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


def test_exchange_execution_prod_readiness_requires_pitr_verification() -> None:
    app = create_app(
        environ={
            "ROEHUB_ENV": "prod",
            "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "configs/prod/exchange_execution.yaml",
            "STRATEGY_FAIL_FAST": "false",
        }
    )

    with TestClient(app) as client:
        ready = client.get("/health/ready")

    assert ready.status_code == 200
    dependency = next(
        item for item in ready.json()["dependencies"] if item["name"] == "ledger_pitr"
    )
    assert dependency["status"] == "degraded"
    assert dependency["reason"] == "pitr_restore_not_verified"


def test_exchange_execution_prod_readiness_accepts_pitr_verification_marker() -> None:
    app = create_app(
        environ={
            "ROEHUB_ENV": "prod",
            "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "configs/prod/exchange_execution.yaml",
            "STRATEGY_FAIL_FAST": "false",
            "ROEHUB_EXECUTION_PITR_VERIFIED": "true",
        }
    )

    with TestClient(app) as client:
        ready = client.get("/health/ready")

    dependency = next(
        item for item in ready.json()["dependencies"] if item["name"] == "ledger_pitr"
    )
    assert dependency["status"] == "ready"
    assert dependency["reason"] == "pitr_restore_verified"


def test_exchange_execution_cancel_after_submit_env_override_disables_canary_cancel() -> None:
    app = create_app(
        environ={
            "ROEHUB_ENV": "prod",
            "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "configs/prod/exchange_execution.yaml",
            "ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT": "false",
            "STRATEGY_FAIL_FAST": "false",
        }
    )

    assert app.state.exchange_execution_service._config.cancel_after_submit is False
