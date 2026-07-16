from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import yaml

from tools.release.generate_runtime_topology import (
    DEFAULT_SERVICE_MANIFEST,
    PROJECT_MAP,
    ROOT,
    SERVICE_SCHEMA,
    run,
)
from trading.platform.config.installation import InstallationConfigError
from trading.platform.config.runtime_topology import (
    load_json_object,
    validate_runtime_service_manifest,
)


def test_runtime_topology_generation_is_deterministic_and_complete(tmp_path: Path) -> None:
    assert run(["--output", str(tmp_path), "--write"]) == 0
    assert run(["--output", str(tmp_path), "--check"]) == 0

    expected = {
            "base": {
                "alertmanager",
                "api",
                "blackbox",
                "grafana",
                "loki",
                "notification-dispatcher",
                "notification-report-scheduler",
                "openbao",
                "operational-health",
                "plugin-gateway",
                "postgresql",
                "prometheus",
            "redis",
            "secret-init",
            "storage-migrations",
            "telegram-bot-worker",
            "web",
        },
            "trading": {
                "alertmanager",
                "api",
                "backtest-artifact-publisher",
                "backtest-job-runner",
                "blackbox",
            "clickhouse",
            "clickhouse-exporter",
            "exchange-control",
                "exchange-execution",
                "grafana",
                "loki",
            "market-data-scheduler",
            "market-data-ws",
            "notification-dispatcher",
            "notification-report-scheduler",
                "openbao",
                "operational-health",
            "plugin-gateway",
                "postgresql",
                "prometheus",
            "redis",
            "secret-init",
            "storage-migrations",
            "strategy-live-runner",
            "telegram-bot-worker",
            "web",
        },
            "ml": {
                "alertmanager",
                "api",
            "backtest-artifact-publisher",
                "backtest-job-runner",
                "blackbox",
            "clickhouse",
            "clickhouse-exporter",
            "exchange-control",
                "exchange-execution",
                "grafana",
                "loki",
            "market-data-scheduler",
            "market-data-ws",
            "notification-dispatcher",
            "notification-report-scheduler",
                "openbao",
                "operational-health",
            "plugin-gateway",
                "postgresql",
                "prometheus",
            "redis",
            "rl-inference",
            "secret-init",
            "storage-migrations",
            "strategy-live-runner",
            "telegram-bot-worker",
            "web",
        },
    }
    for profile, default_services in expected.items():
        compose = yaml.safe_load((tmp_path / profile / "compose.yaml").read_text())
        control_policy = json.loads(
            (tmp_path / profile / "control-policy.json").read_text(encoding="utf-8")
        )
        control_schema = json.loads(
            (ROOT / "schemas/operations/control-policy.v1alpha1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(control_schema).validate(control_policy)
        services = compose["services"]
        assert compose["networks"]["roehub"]["internal"] is True
        assert compose["networks"]["web-ingress"] == {
            "internal": False,
            "labels": {
                "io.roehub.profile": profile,
                "io.roehub.trust-boundary": "web-ingress",
                "io.roehub.ingress-purpose": "host-web-ui-only",
            },
        }
        assert services["web"]["networks"] == ["roehub", "web-ingress"]
        market_data_egress = compose["networks"].get("market-data-egress")
        if profile == "base":
            assert market_data_egress is None
        else:
            assert market_data_egress == {
                "internal": False,
                "labels": {
                    "io.roehub.profile": profile,
                    "io.roehub.trust-boundary": "market-data-egress",
                    "io.roehub.egress-purpose": "public-market-data-only",
                },
            }
            assert services["market-data-ws"]["networks"] == [
                "roehub",
                "market-data-egress",
            ]
            assert services["market-data-scheduler"]["networks"] == [
                "roehub",
                "market-data-egress",
            ]
            assert all(
                service["networks"] == ["roehub"]
                for name, service in services.items()
                if name
                not in {"market-data-ws", "market-data-scheduler", "web", "secret-init"}
                and "networks" in service
            )
        assert {
            name for name, row in services.items() if not row.get("profiles")
        } == default_services
        rendered = json.dumps(compose, sort_keys=True)
        assert ":latest" not in rendered
        assert '"privileged": true' not in rendered
        for name, service in services.items():
            if "build" not in service:
                continue
            assert service["user"] == "65532:65532", name
            assert service["read_only"] is True, name
            assert service["cap_drop"] == ["ALL"], name
            assert service["deploy"]["resources"]["limits"], name
        assert set(control_policy["allowed_services"]) == set(services)
        assert set(control_policy["default_services"]) == {
            name
            for name, row in services.items()
            if row.get("restart") != "no" and not row.get("profiles")
        }
        assert all(
            "@sha256:" in row["release_reference"]
            for row in control_policy["services"].values()
        )
        operational_health = services["operational-health"]
        assert operational_health["environment"] == {
            "PYTHONUNBUFFERED": "1",
            "ROEHUB_ENV": "prod",
            "ROEHUB_PROFILE": profile,
        }
        assert operational_health["volumes"] == [
            "./observability/operational-manifest.json:"
            "/etc/roehub/operational-manifest.json:ro"
        ]
        assert services["web"]["environment"]["WEB_API_BASE_URL"] == "http://web:8010"
        assert (
            services["web"]["environment"]["WEB_API_UPSTREAM_URL"]
            == "http://api:8000"
        )
        if profile != "base":
            exchange_control_probe = services["exchange-control"]["healthcheck"]["test"][-1]
            assert "/health/live" in exchange_control_probe
            exchange_execution_probe = services["exchange-execution"]["healthcheck"]["test"][-1]
            assert "/health/live" in exchange_execution_probe
            if profile == "ml":
                rl_inference_probe = services["rl-inference"]["healthcheck"]["test"][-1]
                assert "/health/live" in rl_inference_probe
            topology = json.loads((tmp_path / profile / "runtime-topology.json").read_text())
            exchange_control = next(
                role for role in topology["roles"] if role["name"] == "exchange-control"
            )
            assert exchange_control["health_path"] == "/health/ready"
        operational_serialized = json.dumps(operational_health, sort_keys=True).lower()
        for forbidden in (
            "database_url",
            "dsn",
            "password",
            "runtime-secrets",
            "secret",
            "token",
        ):
            assert forbidden not in operational_serialized
        for service_name in (
            "alertmanager",
            "blackbox",
            "grafana",
            "loki",
            "openbao",
            "operational-health",
            "postgresql",
            "prometheus",
            "redis",
            "secret-init",
        ):
            assert control_policy["services"][service_name]["restart_allowed"] is False
        for service_name in ("api", "plugin-gateway", "web"):
            assert control_policy["services"][service_name]["restart_allowed"] is True


def test_generated_service_configs_use_dns_and_safe_modes(tmp_path: Path) -> None:
    assert run(["--output", str(tmp_path), "--write"]) == 0
    trading = tmp_path / "trading" / "service-configs"
    market = yaml.safe_load((trading / "market-data.yaml").read_text())
    strategy = yaml.safe_load((trading / "strategy.yaml").read_text())
    execution = yaml.safe_load((trading / "exchange-execution.yaml").read_text())
    notifications = yaml.safe_load((trading / "notifications.yaml").read_text())
    backtest_artifacts = yaml.safe_load(
        (trading / "backtest-artifacts.yaml").read_text()
    )

    assert market["market_data"]["live_feed"]["redis_streams"]["host"] == "redis"
    assert strategy["strategy"]["live_worker"]["redis_streams"]["host"] == "redis"
    assert strategy["strategy"]["producer"]["enabled"] is False
    assert strategy["strategy"]["telegram"]["enabled"] is False
    assert execution["exchange_execution"]["process"]["adapter_mode"] == "disabled"
    assert execution["exchange_execution"]["process"]["consumer_enabled"] is False
    assert notifications["notifications"]["dispatcher"]["provider_mode"] == "log_only"
    assert notifications["notifications"]["telegram_bot"]["enabled"] is False
    assert notifications["notifications"]["report_scheduler"]["enabled"] is False
    assert (
        backtest_artifacts["backtest_artifacts"]["artifact_root"]
        == "/var/lib/roehub/artifacts/backtest/v2"
    )


def test_runtime_manifest_fails_closed_when_current_component_is_uncovered() -> None:
    manifest = load_json_object(DEFAULT_SERVICE_MANIFEST)
    candidate = copy.deepcopy(manifest)
    candidate["roles"] = [
        role for role in candidate["roles"] if role["name"] != "exchange-execution"
    ]
    try:
        validate_runtime_service_manifest(
            candidate,
            load_json_object(SERVICE_SCHEMA),
            load_json_object(PROJECT_MAP),
            repo_root=ROOT,
        )
    except InstallationConfigError as error:
        assert "app:exchange_execution" in str(error)
    else:
        raise AssertionError("uncovered current component must fail closed")


def test_runtime_topology_check_detects_stale_output(tmp_path: Path) -> None:
    assert run(["--output", str(tmp_path), "--write"]) == 0
    (tmp_path / "base" / "compose.yaml").write_text("services: {}\n")
    assert run(["--output", str(tmp_path), "--check"]) == 1
