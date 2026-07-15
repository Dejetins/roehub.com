"""Deterministic observability assets derived from the generated runtime topology."""

from __future__ import annotations

from typing import Any

from trading.platform.config.installation import json_bytes, yaml_bytes

OBSERVABILITY_IMAGES = {
    "prometheus": (
        "prom/prometheus:v3.5.0@sha256:"
        "63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996"
    ),
    "alertmanager": (
        "prom/alertmanager:v0.28.1@sha256:"
        "27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba"
    ),
    "blackbox": (
        "prom/blackbox-exporter:v0.27.0@sha256:"
        "a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d"
    ),
    "grafana": (
        "grafana/grafana:12.0.2@sha256:"
        "b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa"
    ),
    "loki": (
        "grafana/loki:3.5.1@sha256:"
        "a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745"
    ),
}

SAFE_RESTART_SERVICES = frozenset(
    {
        "api",
        "exchange-control",
        "exchange-execution",
        "notification-report-scheduler",
        "plugin-gateway",
        "rl-inference",
        "strategy-live-runner",
        "telegram-bot-worker",
        "web",
    }
)


def render_observability_outputs(*, topology: dict[str, Any]) -> dict[str, bytes]:
    manifest = build_operational_manifest(topology=topology)
    return {
        "prometheus.yml": yaml_bytes(_prometheus_config(manifest=manifest)),
        "observability/alertmanager.yml": yaml_bytes(_alertmanager_config()),
        "observability/alerts.yml": yaml_bytes(_alert_rules()),
        "observability/blackbox.yml": yaml_bytes(_blackbox_config()),
        "observability/grafana-dashboard.json": json_bytes(
            _grafana_dashboard(profile=str(topology["profile"]))
        ),
        "observability/grafana-dashboards.yml": yaml_bytes(
            _grafana_dashboard_provisioning()
        ),
        "observability/grafana-datasources.yml": yaml_bytes(
            _grafana_datasources()
        ),
        "observability/loki.yml": yaml_bytes(_loki_config()),
        "observability/operational-manifest.json": json_bytes(manifest),
    }


def build_operational_manifest(*, topology: dict[str, Any]) -> dict[str, Any]:
    profile = str(topology["profile"])
    probes: list[dict[str, Any]] = [
        _probe(
            service_id="postgresql",
            capability="storage.postgresql",
            kind="tcp_reachability",
            target="postgresql:5432",
            action_ref="diagnostics",
            required=False,
        ),
        _probe(
            service_id="redis",
            capability="transport.redis",
            kind="tcp_reachability",
            target="redis:6379",
            action_ref="diagnostics",
            required=False,
        ),
        _probe(
            service_id="openbao",
            capability="secrets.openbao",
            kind="openbao",
            target=(
                "http://openbao:8200/v1/sys/health?standbyok=true&"
                "sealedcode=200&uninitcode=200"
            ),
            runbook_id="auth.openbao-unavailable",
            action_ref="diagnostics",
        ),
    ]
    if profile != "base":
        probes.append(
            _probe(
                service_id="clickhouse",
                capability="storage.clickhouse",
                kind="http_reachability",
                target="http://clickhouse:8123/ping",
                runbook_id="database.clickhouse-degraded",
                action_ref="diagnostics",
                required=False,
            )
        )
    for role in topology["roles"]:
        if role.get("lifecycle") != "service" or role["name"] == "operational-health":
            continue
        service_id = str(role["name"])
        runbook_id = (
            "web.api-health-degraded"
            if service_id in {"api", "web"}
            else "runtime.service-degraded"
        )
        probes.append(
            _probe(
                service_id=service_id,
                capability=_capability(service_id),
                kind=(
                    "http_json"
                    if str(role["health_path"]) in {"/health", "/health/ready"}
                    else "http_reachability"
                ),
                target=(
                    f"http://{service_id}:{int(role['internal_port'])}"
                    f"{role['health_path']}"
                ),
                runbook_id=runbook_id,
                action_ref=(
                    "restart_service"
                    if service_id in SAFE_RESTART_SERVICES
                    else "diagnostics"
                ),
                required=str(role["health_path"]) in {"/health", "/health/ready"},
            )
        )
    return {
        "schema": "io.roehub.operational-manifest/v1alpha1",
        "profile": profile,
        "services": sorted(probes, key=lambda item: item["service_id"]),
    }


def observability_compose_services() -> dict[str, Any]:
    common = {
        "cap_drop": ["ALL"],
        "networks": ["roehub"],
        "read_only": True,
        "restart": "unless-stopped",
        "security_opt": ["no-new-privileges:true"],
    }
    return {
        "prometheus": {
            **common,
            "image": OBSERVABILITY_IMAGES["prometheus"],
            "user": "65534:65534",
            "command": [
                "--config.file=/etc/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
                "--storage.tsdb.retention.time=15d",
                "--web.enable-lifecycle",
            ],
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=16m"],
            "volumes": [
                "./prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                "./observability/alerts.yml:/etc/prometheus/alerts.yml:ro",
                "prometheus-data:/prometheus",
            ],
            "deploy": {"resources": {"limits": {"cpus": "0.5", "memory": "512M"}}},
        },
        "alertmanager": {
            **common,
            "image": OBSERVABILITY_IMAGES["alertmanager"],
            "user": "65534:65534",
            "command": [
                "--config.file=/etc/alertmanager/alertmanager.yml",
                "--storage.path=/alertmanager",
            ],
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=16m"],
            "volumes": [
                "./observability/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro",
                "alertmanager-data:/alertmanager",
            ],
            "deploy": {"resources": {"limits": {"cpus": "0.25", "memory": "192M"}}},
        },
        "blackbox": {
            **common,
            "image": OBSERVABILITY_IMAGES["blackbox"],
            "user": "65534:65534",
            "command": ["--config.file=/etc/blackbox_exporter/config.yml"],
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=16m"],
            "volumes": [
                "./observability/blackbox.yml:/etc/blackbox_exporter/config.yml:ro"
            ],
            "deploy": {"resources": {"limits": {"cpus": "0.25", "memory": "128M"}}},
        },
        "loki": {
            **common,
            "image": OBSERVABILITY_IMAGES["loki"],
            "user": "10001:10001",
            "command": ["-config.file=/etc/loki/local-config.yaml"],
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=16m"],
            "volumes": [
                "./observability/loki.yml:/etc/loki/local-config.yaml:ro",
                "loki-data:/loki",
            ],
            "deploy": {"resources": {"limits": {"cpus": "0.5", "memory": "512M"}}},
        },
        "grafana": {
            **common,
            "image": OBSERVABILITY_IMAGES["grafana"],
            "user": "472:472",
            "depends_on": {
                "secret-init": {"condition": "service_completed_successfully"}
            },
            "environment": {
                "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",
                "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
                "GF_ANALYTICS_REPORTING_ENABLED": "false",
                "GF_AUTH_ANONYMOUS_ENABLED": "false",
                "GF_PLUGINS_PREINSTALL_DISABLED": "true",
                "GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED": "true",
                "GF_SECURITY_ADMIN_PASSWORD__FILE": (
                    "/run/grafana-secrets/admin-password"
                ),
                "GF_USERS_ALLOW_SIGN_UP": "false",
            },
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=32m"],
            "volumes": [
                "./observability/grafana-datasources.yml:"
                "/etc/grafana/provisioning/datasources/roehub.yml:ro",
                "./observability/grafana-dashboards.yml:"
                "/etc/grafana/provisioning/dashboards/roehub.yml:ro",
                "./observability/grafana-dashboard.json:"
                "/var/lib/grafana/dashboards/roehub/operational.json:ro",
                "grafana-data:/var/lib/grafana",
                "grafana-secrets:/run/grafana-secrets:ro",
            ],
            "deploy": {"resources": {"limits": {"cpus": "0.5", "memory": "512M"}}},
        },
    }


def observability_volumes() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "alertmanager-data": {"labels": {"io.roehub.state-owner": "alertmanager"}},
        "grafana-data": {"labels": {"io.roehub.state-owner": "grafana"}},
        "loki-data": {"labels": {"io.roehub.state-owner": "loki"}},
        "prometheus-data": {"labels": {"io.roehub.state-owner": "prometheus"}},
    }


def _probe(
    *,
    service_id: str,
    capability: str,
    kind: str,
    target: str,
    runbook_id: str = "runtime.service-degraded",
    action_ref: str = "restart_service",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "action_ref": action_ref,
        "capability": capability,
        "kind": kind,
        "required": required,
        "runbook_id": runbook_id,
        "service_id": service_id,
        "target": target,
        "timeout_seconds": 1.0,
    }


def _capability(service_id: str) -> str:
    if service_id in {"api", "web"}:
        return "product.web_api"
    if service_id.startswith("plugin"):
        return "extensions.plugins"
    if service_id.startswith("notification") or service_id == "telegram-bot-worker":
        return "notifications.delivery"
    if service_id.startswith("market-data"):
        return "market_data.ingestion"
    if service_id.startswith("exchange"):
        return "trading.execution"
    if service_id.startswith("backtest"):
        return "research.backtests"
    if service_id.startswith("strategy"):
        return "trading.strategy"
    if service_id.startswith("rl-"):
        return "ml.inference"
    return "runtime.service"


def _prometheus_config(*, manifest: dict[str, Any]) -> dict[str, Any]:
    http_targets = [
        {
            "labels": {
                "capability": item["capability"],
                "service": item["service_id"],
            },
            "targets": [item["target"]],
        }
        for item in manifest["services"]
        if item["kind"] in {"http_json", "http_reachability", "openbao"}
    ]
    tcp_targets = [
        {
            "labels": {
                "capability": item["capability"],
                "service": item["service_id"],
            },
            "targets": [item["target"]],
        }
        for item in manifest["services"]
        if item["kind"] == "tcp_reachability"
    ]
    return {
        "alerting": {
            "alertmanagers": [
                {"static_configs": [{"targets": ["alertmanager:9093"]}]}
            ]
        },
        "global": {"evaluation_interval": "2s", "scrape_interval": "2s"},
        "rule_files": ["/etc/prometheus/alerts.yml"],
        "scrape_configs": [
            {
                "job_name": "operational-health",
                "static_configs": [
                    {
                        "labels": {"profile": manifest["profile"]},
                        "targets": ["operational-health:9300"],
                    }
                ],
            },
            {
                "job_name": "prometheus",
                "static_configs": [{"targets": ["prometheus:9090"]}],
            },
            {
                "job_name": "alertmanager",
                "static_configs": [{"targets": ["alertmanager:9093"]}],
            },
            {
                "job_name": "loki",
                "static_configs": [{"targets": ["loki:3100"]}],
            },
            {
                "job_name": "blackbox-http",
                "metrics_path": "/probe",
                "params": {"module": ["http_reachable"]},
                "relabel_configs": _blackbox_relabel(),
                "static_configs": http_targets,
            },
            {
                "job_name": "blackbox-tcp",
                "metrics_path": "/probe",
                "params": {"module": ["tcp_connect"]},
                "relabel_configs": _blackbox_relabel(),
                "static_configs": tcp_targets,
            },
        ],
    }


def _blackbox_relabel() -> list[dict[str, Any]]:
    return [
        {"source_labels": ["__address__"], "target_label": "__param_target"},
        {"source_labels": ["__param_target"], "target_label": "instance"},
        {"replacement": "blackbox:9115", "target_label": "__address__"},
    ]


def _alert_rules() -> dict[str, Any]:
    return {
        "groups": [
            {
                "name": "roehub-operational-health",
                "rules": [
                    {
                        "alert": "RoehubOperationalServiceDegraded",
                        "annotations": {
                            "action_ref": "{{ $labels.action_ref }}",
                            "description": (
                                "Generated domain health is not ready; use only the linked "
                                "allowlisted action."
                            ),
                            "runbook": (
                                "/runbooks/{{ $labels.runbook_id }}"
                            ),
                            "summary": (
                                "Roehub service {{ $labels.service }} is "
                                "{{ $labels.state }}"
                            ),
                        },
                        "expr": "roehub_operational_service_state{state!=\"ready\"} == 1",
                        "for": "0s",
                        "labels": {"severity": "warning"},
                    },
                    {
                        "alert": "RoehubOperationalHealthUnavailable",
                        "annotations": {
                            "action_ref": "diagnostics",
                            "description": (
                                "Prometheus cannot scrape the independent health service."
                            ),
                            "runbook": "/runbooks/runtime.observability-unavailable",
                            "summary": "Roehub operational health service is unavailable",
                        },
                        "expr": "up{job=\"operational-health\"} == 0",
                        "for": "0s",
                        "labels": {
                            "runbook_id": "runtime.observability-unavailable",
                            "severity": "critical",
                        },
                    },
                    {
                        "alert": "RoehubOperationalLogPushFailed",
                        "annotations": {
                            "action_ref": "diagnostics",
                            "description": (
                                "The bounded transition sink is unavailable; do not treat "
                                "missing logs as healthy state."
                            ),
                            "runbook": "/runbooks/runtime.observability-unavailable",
                            "summary": "Roehub operational transitions are not reaching Loki",
                        },
                        "expr": "roehub_operational_log_push_success == 0",
                        "for": "0s",
                        "labels": {
                            "runbook_id": "runtime.observability-unavailable",
                            "severity": "warning",
                        },
                    },
                    {
                        "alert": "RoehubOperationalSnapshotStale",
                        "annotations": {
                            "action_ref": "diagnostics",
                            "description": (
                                "The health worker has not completed a fresh bounded refresh."
                            ),
                            "runbook": "/runbooks/runtime.observability-unavailable",
                            "summary": "Roehub operational snapshot is stale",
                        },
                        "expr": "roehub_operational_snapshot_fresh == 0",
                        "for": "0s",
                        "labels": {
                            "runbook_id": "runtime.observability-unavailable",
                            "severity": "critical",
                        },
                    },
                ],
            }
        ]
    }


def _alertmanager_config() -> dict[str, Any]:
    return {
        "receivers": [{"name": "local-audit-only"}],
        "route": {"group_wait": "0s", "receiver": "local-audit-only"},
    }


def _blackbox_config() -> dict[str, Any]:
    return {
        "modules": {
            "http_reachable": {
                "http": {
                    "follow_redirects": False,
                    "preferred_ip_protocol": "ip4",
                    "valid_status_codes": [200, 204, 503],
                },
                "prober": "http",
                "timeout": "2s",
            },
            "tcp_connect": {"prober": "tcp", "timeout": "2s"},
        }
    }


def _loki_config() -> dict[str, Any]:
    return {
        "analytics": {"reporting_enabled": False},
        "auth_enabled": False,
        "common": {
            "instance_addr": "127.0.0.1",
            "path_prefix": "/loki",
            "replication_factor": 1,
            "ring": {"kvstore": {"store": "inmemory"}},
            "storage": {
                "filesystem": {
                    "chunks_directory": "/loki/chunks",
                    "rules_directory": "/loki/rules",
                }
            },
        },
        "schema_config": {
            "configs": [
                {
                    "from": "2024-01-01",
                    "index": {"period": "24h", "prefix": "index_"},
                    "object_store": "filesystem",
                    "schema": "v13",
                    "store": "tsdb",
                }
            ]
        },
        "server": {"http_listen_port": 3100},
    }


def _grafana_datasources() -> dict[str, Any]:
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "access": "proxy",
                "isDefault": True,
                "name": "Roehub Prometheus",
                "type": "prometheus",
                "uid": "roehub-prometheus",
                "url": "http://prometheus:9090",
            },
            {
                "access": "proxy",
                "name": "Roehub Loki",
                "type": "loki",
                "uid": "roehub-loki",
                "url": "http://loki:3100",
            },
        ],
    }


def _grafana_dashboard_provisioning() -> dict[str, Any]:
    return {
        "apiVersion": 1,
        "providers": [
            {
                "disableDeletion": True,
                "editable": False,
                "name": "Roehub generated",
                "options": {"path": "/var/lib/grafana/dashboards/roehub"},
                "type": "file",
            }
        ],
    }


def _grafana_dashboard(*, profile: str) -> dict[str, Any]:
    return {
        "annotations": {"list": []},
        "editable": False,
        "panels": [
            {
                "datasource": {"type": "prometheus", "uid": "roehub-prometheus"},
                "fieldConfig": {"defaults": {}, "overrides": []},
                "gridPos": {"h": 12, "w": 24, "x": 0, "y": 0},
                "id": 1,
                "options": {"showHeader": True},
                "targets": [
                    {
                        "expr": "roehub_operational_service_state",
                        "format": "table",
                        "refId": "A",
                    }
                ],
                "title": "Generated operational state",
                "type": "table",
            }
        ],
        "schemaVersion": 41,
        "tags": ["roehub", "generated", profile],
        "templating": {"list": []},
        "time": {"from": "now-1h", "to": "now"},
        "title": f"Roehub {profile} operational health",
        "uid": f"roehub-{profile}-operational",
        "version": 1,
    }


__all__ = [
    "OBSERVABILITY_IMAGES",
    "build_operational_manifest",
    "observability_compose_services",
    "observability_volumes",
    "render_observability_outputs",
]
