from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from trading.platform.config.operational_topology import (
    OBSERVABILITY_IMAGES,
    build_operational_manifest,
    observability_compose_services,
    observability_volumes,
    render_observability_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _topology(profile: str = "trading") -> dict[str, Any]:
    return json.loads(
        (
            REPO_ROOT
            / f"configs/installation/generated/{profile}/runtime-topology.json"
        ).read_text()
    )


def test_generated_operational_manifest_is_complete_redacted_and_linked() -> None:
    manifest = build_operational_manifest(topology=_topology())
    schema = json.loads(
        (REPO_ROOT / "schemas/ops/operational-manifest.schema.json").read_text()
    )
    jsonschema.validate(manifest, schema)
    service_ids = [item["service_id"] for item in manifest["services"]]
    assert len(service_ids) == len(set(service_ids))
    assert {
        "api",
        "web",
        "plugin-gateway",
        "postgresql",
        "clickhouse",
        "redis",
        "openbao",
    } <= set(service_ids)
    forbidden = {"organization", "account", "user", "secret", "token"}
    for item in manifest["services"]:
        assert not forbidden.intersection(item)
        runbook = yaml.safe_load(
            (
                REPO_ROOT
                / f"docs/runbooks/ops/{item['runbook_id']}.yaml"
            ).read_text()
        )
        assert item["action_ref"] in {
            action["id"] for action in runbook["spec"]["allowed_actions"]
        }


def test_observability_compose_is_pinned_non_root_and_persistent() -> None:
    services = observability_compose_services()
    assert set(services) == {
        "alertmanager",
        "blackbox",
        "grafana",
        "loki",
        "prometheus",
    }
    assert set(OBSERVABILITY_IMAGES.values()) == {
        service["image"] for service in services.values()
    }
    for service in services.values():
        assert "@sha256:" in service["image"]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
    assert services["grafana"]["environment"]["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert services["grafana"]["environment"] | {
        "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",
        "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
        "GF_ANALYTICS_REPORTING_ENABLED": "false",
        "GF_PLUGINS_PREINSTALL_DISABLED": "true",
        "GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED": "true",
    } == services["grafana"]["environment"]
    assert services["grafana"]["depends_on"] == {
        "secret-init": {"condition": "service_completed_successfully"}
    }
    assert set(observability_volumes()) == {
        "alertmanager-data",
        "grafana-data",
        "loki-data",
        "prometheus-data",
    }


def test_generated_alerts_dashboards_and_scrapes_follow_topology() -> None:
    outputs = render_observability_outputs(topology=_topology())
    prometheus = yaml.safe_load(outputs["prometheus.yml"])
    rules = yaml.safe_load(outputs["observability/alerts.yml"])
    dashboard = json.loads(outputs["observability/grafana-dashboard.json"])
    jobs = {item["job_name"] for item in prometheus["scrape_configs"]}
    alerts = {
        rule["alert"]
        for group in rules["groups"]
        for rule in group["rules"]
    }
    assert {
        "operational-health",
        "prometheus",
        "alertmanager",
        "loki",
        "blackbox-http",
        "blackbox-tcp",
    } <= jobs
    assert alerts == {
        "RoehubOperationalHealthUnavailable",
        "RoehubOperationalLogPushFailed",
        "RoehubOperationalServiceDegraded",
        "RoehubOperationalSnapshotStale",
    }
    assert dashboard["uid"] == "roehub-trading-operational"
    assert dashboard["panels"][0]["targets"][0]["expr"] == (
        "roehub_operational_service_state"
    )
    loki = yaml.safe_load(outputs["observability/loki.yml"])
    assert loki["analytics"] == {"reporting_enabled": False}
