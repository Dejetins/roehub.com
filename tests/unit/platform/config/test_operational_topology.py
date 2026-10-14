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
        (REPO_ROOT / f"configs/installation/generated/{profile}/runtime-topology.json").read_text()
    )


def test_generated_operational_manifest_is_complete_redacted_and_linked() -> None:
    manifest = build_operational_manifest(topology=_topology())
    schema = json.loads((REPO_ROOT / "schemas/ops/operational-manifest.schema.json").read_text())
    jsonschema.validate(manifest, schema)
    service_ids = [item["service_id"] for item in manifest["services"]]
    assert len(service_ids) == len(set(service_ids))
    assert {
        "api",
        "web",
        "postgresql",
        "redis",
        "openbao",
    } <= set(service_ids)
    forbidden = {"organization", "account", "user", "secret", "token"}
    for item in manifest["services"]:
        assert not forbidden.intersection(item)
        runbook = yaml.safe_load(
            (REPO_ROOT / f"docs/runbooks/ops/{item['runbook_id']}.yaml").read_text()
        )
        assert item["action_ref"] in {action["id"] for action in runbook["spec"]["allowed_actions"]}


def test_observability_compose_is_pinned_non_root_and_persistent() -> None:
    services = observability_compose_services()
    assert set(services) == {
        "alertmanager",
        "blackbox",
        "prometheus",
    }
    assert set(OBSERVABILITY_IMAGES.values()) == {service["image"] for service in services.values()}
    for service in services.values():
        assert "@sha256:" in service["image"]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
    assert set(observability_volumes()) == {
        "alertmanager-data",
        "prometheus-data",
    }


def test_generated_alerts_and_scrapes_follow_topology() -> None:
    outputs = render_observability_outputs(topology=_topology())
    prometheus = yaml.safe_load(outputs["prometheus.yml"])
    rules = yaml.safe_load(outputs["observability/alerts.yml"])
    jobs = {item["job_name"] for item in prometheus["scrape_configs"]}
    alerts = {rule["alert"] for group in rules["groups"] for rule in group["rules"]}
    assert {
        "operational-health",
        "prometheus",
        "alertmanager",
        "blackbox-http",
        "blackbox-tcp",
    } <= jobs
    assert alerts == {
        "RoehubOperationalHealthUnavailable",
        "RoehubOperationalServiceDegraded",
        "RoehubOperationalSnapshotStale",
    }
    assert not any("grafana" in path for path in outputs)
