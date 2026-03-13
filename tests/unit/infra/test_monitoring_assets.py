from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(*, relative_path: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((_repo_root() / relative_path).read_text(encoding="utf-8")),
    )


def _load_yaml(*, relative_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((_repo_root() / relative_path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"{relative_path} is empty")
    return cast(dict[str, Any], payload)


def test_prometheus_monitoring_contract_includes_required_jobs_and_targets() -> None:
    payload = _load_yaml(relative_path="infra/monitoring/monitoring/prometheus/prometheus.yml")

    assert payload["rule_files"] == ["/etc/prometheus/rules/mac-studio-monitoring.rules.yml"]
    scrape_job_names = [job["job_name"] for job in payload["scrape_configs"]]
    assert scrape_job_names == [
        "api",
        "blackbox",
        "blackbox_http",
        "blackbox_tcp",
        "cadvisor",
        "clickhouse_exporter",
        "prometheus",
        "market-data-ws-worker",
        "market-data-scheduler",
        "node_exporter",
        "postgres_exporter",
        "redis_exporter",
    ]

    jobs_by_name = {job["job_name"]: job for job in payload["scrape_configs"]}
    assert jobs_by_name["api"]["static_configs"] == [{"targets": ["api:8000"]}]
    assert jobs_by_name["node_exporter"]["static_configs"] == [
        {"targets": ["host.lima.internal:9100"]}
    ]
    http_targets = jobs_by_name["blackbox_http"]["static_configs"]
    assert http_targets == [
        {"labels": {"service": "api_health"}, "targets": ["http://api:8000/health"]},
        {
            "labels": {"service": "clickhouse_http_ping"},
            "targets": ["http://clickhouse:8123/ping"],
        },
        {"labels": {"service": "grafana_health"}, "targets": ["http://grafana:3000/api/health"]},
        {
            "labels": {"service": "prometheus_health"},
            "targets": ["http://prometheus:9090/-/healthy"],
        },
    ]


def test_blackbox_and_grafana_provisioning_assets_are_repo_managed() -> None:
    blackbox_payload = _load_yaml(relative_path="infra/monitoring/monitoring/blackbox/blackbox.yml")
    datasource_payload = _load_yaml(
        relative_path="infra/monitoring/monitoring/grafana/provisioning/datasources/roehub-prometheus.yml"
    )
    dashboard_provider_payload = _load_yaml(
        relative_path="infra/monitoring/monitoring/grafana/provisioning/dashboards/roehub-dashboards.yml"
    )

    assert blackbox_payload["modules"]["http_2xx"]["http"]["valid_status_codes"] == [200]
    assert datasource_payload["datasources"][0]["uid"] == "roehub-prometheus"
    assert datasource_payload["datasources"][0]["url"] == "http://prometheus:9090"
    assert dashboard_provider_payload["providers"][0]["options"]["path"] == (
        "/var/lib/grafana/dashboards/roehub"
    )


def test_repo_managed_dashboards_cover_required_monitoring_views() -> None:
    dashboard_dir = _repo_root() / "infra/monitoring/monitoring/grafana/dashboards/roehub"
    dashboard_files = sorted(path.name for path in dashboard_dir.glob("*.json"))

    assert dashboard_files == [
        "01-platform-overview.json",
        "02-mac-studio-host.json",
        "03-containers.json",
        "04-datastores.json",
        "05-api-market-data.json",
    ]

    titles = [
        _load_json(relative_path=f"infra/monitoring/monitoring/grafana/dashboards/roehub/{name}")[
            "title"
        ]
        for name in dashboard_files
    ]
    assert titles == [
        "Platform Overview",
        "Mac Studio Host",
        "Containers",
        "Datastores",
        "API and Market Data",
    ]
