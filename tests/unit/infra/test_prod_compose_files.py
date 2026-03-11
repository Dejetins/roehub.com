from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _load_yaml(*, relative_path: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    payload = yaml.safe_load((repo_root / relative_path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"{relative_path} is empty")
    return cast(dict[str, Any], payload)


def test_backend_prod_compose_contains_private_backend_services_only() -> None:
    payload = _load_yaml(relative_path="infra/docker/docker-compose.backend.yml")
    services = payload["services"]

    for service_name in (
        "postgres",
        "clickhouse",
        "redis",
        "db-bootstrap",
        "api",
        "market-data-ws-worker",
        "market-data-scheduler",
        "grafana",
        "prometheus",
        "blackbox",
    ):
        assert service_name in services

    assert "web" not in services
    assert "gateway" not in services


def test_backend_prod_compose_publishes_api_on_localhost_only() -> None:
    payload = _load_yaml(relative_path="infra/docker/docker-compose.backend.yml")
    api_service = payload["services"]["api"]

    assert api_service["ports"] == ["${API_HOST_BIND:-127.0.0.1}:${API_HOST_PORT:-8000}:8000"]
    assert api_service["image"] == "${ROEHUB_APP_IMAGE:?ROEHUB_APP_IMAGE is required}"


def test_web_prod_compose_contains_only_web_service() -> None:
    payload = _load_yaml(relative_path="infra/docker/docker-compose.web.prod.yml")
    services = payload["services"]

    assert list(services.keys()) == ["web"]
    assert services["web"]["image"] == "${ROEHUB_APP_IMAGE:?ROEHUB_APP_IMAGE is required}"
    assert services["web"]["ports"] == ["127.0.0.1:${WEB_HOST_PORT:-8010}:8010"]
