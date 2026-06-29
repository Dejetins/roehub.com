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
        "backtest-artifact-publisher",
        "node_exporter",
        "postgres_exporter",
        "redis_exporter",
    ]

    jobs_by_name = {job["job_name"]: job for job in payload["scrape_configs"]}
    assert jobs_by_name["api"]["static_configs"] == [{"targets": ["api:8000"]}]
    assert jobs_by_name["backtest-artifact-publisher"]["static_configs"] == [
        {"targets": ["backtest-artifact-publisher:9203"]}
    ]
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


def test_macos_prometheus_stage17_rules_are_repo_managed() -> None:
    payload = _load_yaml(relative_path="infra/macos/prometheus/prometheus.prod.yml")
    assert payload["rule_files"] == [
        "/opt/roehub/config/prometheus.rules/live-execution-stage17.rules.yml",
        "/opt/roehub/config/prometheus.rules/strategy-producer.rules.yml",
        "/opt/roehub/config/prometheus.rules/market-data-funding.rules.yml",
        "/opt/roehub/config/prometheus.rules/notifications-admin.rules.yml",
    ]
    jobs_by_name = {job["job_name"]: job for job in payload["scrape_configs"]}
    assert jobs_by_name["exchange-execution"]["static_configs"] == [
        {"targets": ["127.0.0.1:9206"]}
    ]
    assert jobs_by_name["strategy-producer"]["static_configs"] == [
        {"targets": ["127.0.0.1:9207"]}
    ]

    rules_payload = _load_yaml(
        relative_path="infra/macos/prometheus/rules/live-execution-stage17.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["live-execution-production-readiness"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "LiveExecutionDlqGrowing",
        "LiveExecutionClockDriftUnsafe",
        "LiveExecutionPrivateStreamMissingForSubmit",
        "LiveExecutionLimiterWaitHigh",
        "LiveExecutionSubmitLatencyHigh",
        "LiveExecutionDispatchBackpressure",
        "LiveExecutionReconciliationPending",
        "LiveExecutionPitrNotVerified",
        "LiveExecutionUnknownState",
    }
    for rule in alerts.values():
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["labels"]["owner"] == "live-execution"
        assert rule["annotations"]["runbook"] == (
            "docs/runbooks/exchange-execution.md#stage-17-alert-actions"
        )
        assert rule["annotations"]["escalation"]
        assert rule["annotations"]["action"]


def test_macos_bootstrap_installs_stage17_prometheus_rules() -> None:
    script = (_repo_root() / "scripts/macos/bootstrap_native_prod.sh").read_text(
        encoding="utf-8"
    )
    assert "/opt/roehub/config/prometheus.rules" in script
    assert "live-execution-stage17.rules.yml" in script
    assert "strategy-producer.rules.yml" in script
    assert "market-data-funding.rules.yml" in script
    assert "notifications-admin.rules.yml" in script


def test_macos_prometheus_notifications_admin_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(
        relative_path="infra/macos/prometheus/rules/notifications-admin.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["notifications-admin"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "NotificationsCriticalUnknownDelivery",
        "NotificationsDispatcherPendingOld",
        "NotificationsWorkerDown",
        "NotificationsRetry429High",
        "NotificationsMissedReportSchedule",
    }
    for rule in alerts.values():
        assert rule["labels"]["owner"] == "notifications"
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["runbook"].startswith(
            "docs/runbooks/notifications-admin-alerts.md#"
        )
        assert rule["annotations"]["escalation"]
        assert rule["annotations"]["action"]


def test_macos_prometheus_funding_rules_are_repo_managed() -> None:
    rules_payload = _load_yaml(
        relative_path="infra/macos/prometheus/rules/market-data-funding.rules.yml"
    )
    groups = rules_payload["groups"]
    assert [group["name"] for group in groups] == ["market-data-funding"]
    rules = groups[0]["rules"]
    alerts = {rule["alert"]: rule for rule in rules}
    assert set(alerts) == {
        "MarketDataFundingCatchupErrors",
        "MarketDataFundingNoRecentSuccess",
        "MarketDataFundingLagHigh",
        "MarketDataFundingMissingIntervals",
    }
    for rule in rules:
        assert "symbol" not in rule.get("labels", {})
    assert "scheduler_funding_catchup_" in json.dumps(rules_payload)


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
