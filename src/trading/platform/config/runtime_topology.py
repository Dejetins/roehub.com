"""Deterministic container topology generated from installation and service manifests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml

from trading.platform.config.installation import (
    InstallationConfigError,
    json_bytes,
    sha256_bytes,
    yaml_bytes,
)
from trading.platform.config.installation import (
    render_profile as render_installation_profile,
)
from trading.platform.config.operational_topology import (
    SAFE_RESTART_SERVICES,
    observability_compose_services,
    observability_volumes,
    render_observability_outputs,
)

RUNTIME_SERVICE_MANIFEST_SCHEMA = "io.roehub.runtime-service-manifest/v1alpha1"
RUNTIME_TOPOLOGY_SCHEMA = "io.roehub.runtime-topology/v1alpha1"
PROFILE_ORDER = {"base": 0, "trading": 1, "ml": 2}
PINNED_INFRASTRUCTURE_IMAGES = {
    "postgresql": (
        "postgres:16@sha256:" "be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c"
    ),
    "clickhouse": (
        "clickhouse/clickhouse-server:24.8@sha256:"
        "1ffa82edee000a42c09313bd9f1293d94c570aee74babc1b3ca9983a35fa597b"
    ),
    "redis": (
        "redis:7.2-bookworm@sha256:"
        "e51cbc16f94b2426e80b9516db174a07d55e882217a1ec1d729b137b32e24e42"
    ),
    "openbao": (
        "ghcr.io/dejetins/roehub-openbao:2.5.4-roehub-licensed-qr.1@sha256:"
        "8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a"
    ),
    "secret_init": (
        "alpine:3.22@sha256:" "14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
    ),
}
_COVERAGE_WITHOUT_NAMED_MAIN: frozenset[str] = frozenset()
_NON_PRODUCT_RUNTIME_COMPONENTS = frozenset({"app:runtime_probe"})


def validate_runtime_service_manifest(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    project_map: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Validate manifest shape, current-component coverage and safe topology invariants."""

    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.path),
    )
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(item) for item in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise InstallationConfigError(f"runtime service manifest schema failed: {rendered}")
    if manifest.get("schema") != RUNTIME_SERVICE_MANIFEST_SCHEMA:
        raise InstallationConfigError("runtime service manifest schema id mismatch")

    roles = manifest["roles"]
    names = [str(role["name"]) for role in roles]
    if len(names) != len(set(names)):
        raise InstallationConfigError("runtime role names must be unique")

    components = {
        str(component["id"]): component
        for component in project_map.get("components", [])
        if component.get("kind") in {"app", "worker"}
    }
    required_components = {
        component_id
        for component_id, component in components.items()
        if component_id not in _NON_PRODUCT_RUNTIME_COMPONENTS
        and (component.get("entrypoints") or component_id in _COVERAGE_WITHOUT_NAMED_MAIN)
    }
    covered_components: set[str] = set()
    covered_entrypoints: set[str] = set()
    for role in roles:
        role_name = str(role["name"])
        command_text = " ".join(str(item) for item in role["command"]).lower()
        for forbidden in ("127.0.0.1", "localhost", "mainnet", ":latest"):
            if forbidden in command_text:
                raise InstallationConfigError(
                    f"runtime role {role_name} command contains forbidden coupling: {forbidden}"
                )
        for component_id in role["component_ids"]:
            if component_id not in components:
                raise InstallationConfigError(
                    f"runtime role {role_name} references unknown current component: {component_id}"
                )
            covered_components.add(str(component_id))
        for entrypoint in role["entrypoints"]:
            entrypoint_path = repo_root / str(entrypoint)
            if not entrypoint_path.is_file():
                raise InstallationConfigError(
                    f"runtime role {role_name} entrypoint is missing: {entrypoint}"
                )
            covered_entrypoints.add(str(entrypoint))
        for dependency in role["depends_on"]:
            if dependency not in names and dependency not in {
                "postgresql",
                "clickhouse",
                "redis",
                "openbao",
            }:
                raise InstallationConfigError(
                    f"runtime role {role_name} has unknown dependency: {dependency}"
                )

    missing_components = sorted(required_components - covered_components)
    if missing_components:
        raise InstallationConfigError(
            f"current runtime components are absent from manifest: {missing_components}"
        )
    current_entrypoints = {
        str(entrypoint)
        for component_id in required_components
        for entrypoint in components[component_id].get("entrypoints", [])
    }
    missing_entrypoints = sorted(current_entrypoints - covered_entrypoints)
    if missing_entrypoints:
        raise InstallationConfigError(
            f"current runtime entrypoints are absent from manifest: {missing_entrypoints}"
        )


def render_runtime_profile(
    *,
    config: dict[str, Any],
    release_manifest: dict[str, Any],
    service_manifest: dict[str, Any],
    profile: str,
    config_source: bytes,
    release_source: bytes,
    service_manifest_source: bytes,
    config_templates: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Render deterministic full Compose, service configs and topology evidence."""

    if profile not in PROFILE_ORDER:
        raise InstallationConfigError(f"unsupported runtime profile: {profile}")
    installation_outputs = render_installation_profile(
        config,
        release_manifest,
        profile,
        config_source=config_source,
        manifest_source=release_source,
    )
    service_configs = _render_service_configs(
        profile=profile,
        installation_config=config,
        templates=config_templates,
    )
    roles = [
        copy.deepcopy(role) for role in service_manifest["roles"] if profile in role["profiles"]
    ]
    if profile != "base":
        storage_migrations = next(role for role in roles if role["name"] == "storage-migrations")
        if "clickhouse" not in storage_migrations["depends_on"]:
            storage_migrations["depends_on"].append("clickhouse")
    topology = {
        "schema": RUNTIME_TOPOLOGY_SCHEMA,
        "profile": profile,
        "safe_defaults": {
            "mainnet": False,
            "notification_provider_mode": "log_only",
            "rl_inference_enabled": False,
            "trading_mode": config["trading"]["mode"],
            "update_checks": False,
        },
        "infrastructure": [
            "postgresql",
            "redis",
            "openbao",
            *(["clickhouse"] if profile != "base" else []),
        ],
        "roles": roles,
        "observability": [
            "operational-health",
            "prometheus",
            "alertmanager",
            "blackbox",
            "grafana",
            "loki",
        ],
    }
    compose = _render_compose(
        config=config,
        service_manifest=service_manifest,
        profile=profile,
        roles=roles,
    )
    control_policy = _render_control_policy(
        compose=compose,
        profile=profile,
        release_manifest=release_manifest,
        service_manifest=service_manifest,
    )
    observability_outputs = render_observability_outputs(topology=topology)
    outputs = {
        "compose.yaml": yaml_bytes(compose),
        "control-policy.json": json_bytes(control_policy),
        "runtime-topology.json": json_bytes(topology),
        "service-config.json": installation_outputs["service-config.json"],
        "oidc.json": installation_outputs["oidc.json"],
        "openbao.json": installation_outputs["openbao.json"],
        "prometheus.yml": observability_outputs.pop("prometheus.yml"),
        "effective-config.redacted.json": installation_outputs["effective-config.redacted.json"],
        **{f"service-configs/{name}": content for name, content in service_configs.items()},
        **observability_outputs,
    }
    generation_manifest = {
        "schema": "io.roehub.runtime-topology-generation/v1alpha1",
        "profile": profile,
        "inputs": {
            "release_manifest_sha256": sha256_bytes(release_source),
            "roehub_yaml_sha256": sha256_bytes(config_source),
            "runtime_service_manifest_sha256": sha256_bytes(service_manifest_source),
        },
        "outputs": {
            name: {"sha256": sha256_bytes(content)} for name, content in sorted(outputs.items())
        },
    }
    outputs["generation-manifest.json"] = json_bytes(generation_manifest)
    return dict(sorted(outputs.items()))


def _render_service_configs(
    *, profile: str, installation_config: dict[str, Any], templates: Mapping[str, bytes]
) -> dict[str, bytes]:
    parsed = {name: yaml.safe_load(content.decode("utf-8")) for name, content in templates.items()}
    notifications = copy.deepcopy(parsed["notifications.yaml"])
    dispatcher = notifications["notifications"]["dispatcher"]
    dispatcher["enabled"] = True
    dispatcher["provider_mode"] = "log_only"
    notifications_root = notifications["notifications"]
    notifications_root["telegram_bot"]["enabled"] = False
    notifications_root["telegram_bot"]["metrics_port"] = 9212
    notifications_root["report_scheduler"] = {
        "enabled": False,
        "poll_interval_seconds": 60,
        "metrics_port": 9211,
    }

    market_data = copy.deepcopy(parsed["market-data.yaml"])
    redis_streams = market_data["market_data"]["live_feed"]["redis_streams"]
    redis_streams["host"] = "redis"
    redis_streams["port"] = 6379
    if "redis_hot_cache" in market_data["market_data"]["live_feed"]:
        market_data["market_data"]["live_feed"]["redis_hot_cache"]["host"] = "redis"

    strategy = copy.deepcopy(parsed["strategy.yaml"])
    strategy_root = strategy["strategy"]
    strategy_root["telegram"]["enabled"] = False
    strategy_root["producer"]["enabled"] = False
    for section in ("live_worker", "realtime_output"):
        strategy_root[section]["redis_streams"]["host"] = "redis"
        strategy_root[section]["redis_streams"]["port"] = 6379

    exchange_execution = copy.deepcopy(parsed["exchange-execution.yaml"])
    execution_root = exchange_execution["exchange_execution"]
    execution_root["http"]["host"] = "0.0.0.0"
    execution_root["redis_streams"]["host"] = "redis"
    execution_root["redis_streams"]["port"] = 6379
    execution_root["process"]["adapter_mode"] = "disabled"
    execution_root["process"]["consumer_enabled"] = False

    rl_runtime = copy.deepcopy(parsed["rl-runtime.yaml"])
    rl_runtime["profile"] = profile
    rl_runtime["artifact_root"] = "/var/lib/roehub/ml"
    rl_runtime["device_policy"] = "cpu_only_for_container_default"
    rl_runtime["inference"]["enabled"] = False
    rl_runtime["inference"]["mode"] = "monitor_only"
    rl_runtime["inference"]["rollout_phase"] = "disabled"
    rl_runtime["inference"]["state_path"] = "/var/lib/roehub/ml/inference/monitor-state.json"
    rl_runtime["inference"]["instruments"] = [
        {
            "exchange": "binance",
            "instrument_key": "binance:futures:BTCUSDT",
            "market_type": "futures",
            "symbol": "BTCUSDT",
        }
    ]
    rl_runtime["inference"]["operator_context"] = {
        "organization_id": "00000000-0000-4000-8000-000000000001",
        "owner_user_id": "00000000-0000-4000-8000-000000000002",
        "strategy_id": "00000000-0000-4000-8000-000000000003",
        "strategy_run_id": "00000000-0000-4000-8000-000000000004",
    }
    artifacts = rl_runtime["inference"]["monitor_policy"]["artifacts"]
    artifacts.update(
        {
            "candidate_manifest_path": (
                "/var/lib/roehub/ml/candidates/not-configured/candidate-manifest.json"
            ),
            "candidate_manifest_sha256": "0" * 64,
            "checkpoint_path": ("/var/lib/roehub/ml/candidates/not-configured/checkpoint.pth"),
            "checkpoint_sha256": "0" * 64,
            "evaluation_manifest_path": (
                "/var/lib/roehub/ml/candidates/not-configured/evaluation-manifest.json"
            ),
            "evaluation_manifest_sha256": "0" * 64,
            "normalization_stats_path": (
                "/var/lib/roehub/ml/candidates/not-configured/normalization-stats.json"
            ),
            "normalization_stats_file_sha256": "0" * 64,
        }
    )
    rl_runtime["inference"]["redis_streams"]["host"] = "redis"
    rl_runtime["inference"]["redis_streams"]["port"] = 6379
    rl_runtime["inference"]["redis_streams"]["enabled"] = False
    rl_runtime["runtime_artifacts"]["allowed_root"] = "/var/lib/roehub/ml"

    backtest_artifacts = copy.deepcopy(parsed["backtest-artifacts.yaml"])
    backtest_artifacts["backtest_artifacts"]["artifact_root"] = (
        "/var/lib/roehub/artifacts/backtest/v2"
    )
    indicators = copy.deepcopy(parsed["indicators.yaml"])
    return {
        "notifications.yaml": yaml_bytes(notifications),
        "market-data.yaml": yaml_bytes(market_data),
        "market_data.yaml": yaml_bytes(market_data),
        "strategy.yaml": yaml_bytes(strategy),
        "exchange-execution.yaml": yaml_bytes(exchange_execution),
        "rl-runtime.yaml": yaml_bytes(rl_runtime),
        "backtest-artifacts.yaml": yaml_bytes(backtest_artifacts),
        "indicators.yaml": yaml_bytes(indicators),
        "plugin-publisher-keys.json": json_bytes(
            {"contract": "PluginPublisherKeys/v1alpha1", "keys": {}}
        ),
    }


def _render_compose(
    *,
    config: dict[str, Any],
    service_manifest: dict[str, Any],
    profile: str,
    roles: list[dict[str, Any]],
) -> dict[str, Any]:
    services: dict[str, Any] = {
        "secret-init": _secret_init_service(),
        "postgresql": _postgres_service(),
        "redis": _redis_service(),
        "openbao": _openbao_service(),
    }
    if profile != "base":
        services["clickhouse"] = _clickhouse_service()
    for role in roles:
        if role["lifecycle"] in {"host-service", "host-tool"}:
            continue
        services[role["name"]] = _runtime_role_service(
            role=role,
            service_manifest=service_manifest,
            profile=profile,
            config=config,
        )
    services.update(observability_compose_services())
    volumes = {
        "postgres-data": {"labels": {"io.roehub.state-owner": "postgresql"}},
        "redis-data": {"labels": {"io.roehub.state-owner": "redis"}},
        "openbao-data": {"labels": {"io.roehub.state-owner": "openbao"}},
        "openbao-audit": {"labels": {"io.roehub.state-owner": "openbao-audit"}},
        "runtime-secrets": {"labels": {"io.roehub.state-owner": "installation"}},
        "grafana-secrets": {"labels": {"io.roehub.state-owner": "grafana-auth"}},
        "artifacts": {"labels": {"io.roehub.state-owner": "artifact-store"}},
    }
    if profile != "base":
        volumes["clickhouse-data"] = {"labels": {"io.roehub.state-owner": "clickhouse"}}
        volumes["clickhouse-logs"] = {"labels": {"io.roehub.state-owner": "clickhouse-logs"}}
    if profile == "ml":
        volumes["ml-state"] = {"labels": {"io.roehub.state-owner": "ml-runtime"}}
    volumes.update(observability_volumes())
    networks: dict[str, Any] = {
        "roehub": {
            "internal": True,
            "labels": {
                "io.roehub.profile": profile,
                "io.roehub.trust-boundary": "installation-internal",
            },
        }
    }
    if any(role.get("network_access") == "market_data_egress" for role in roles):
        networks["market-data-egress"] = {
            "internal": False,
            "labels": {
                "io.roehub.profile": profile,
                "io.roehub.trust-boundary": "market-data-egress",
                "io.roehub.egress-purpose": "public-market-data-only",
            },
        }
    if any(role.get("network_access") == "web_ingress" for role in roles):
        networks["web-ingress"] = {
            "internal": False,
            "labels": {
                "io.roehub.profile": profile,
                "io.roehub.trust-boundary": "web-ingress",
                "io.roehub.ingress-purpose": "host-web-ui-only",
            },
        }
    return {
        "name": f"roehub-{profile}",
        "services": services,
        "networks": networks,
        "volumes": volumes,
        "x-roehub": {
            "generated": True,
            "profile": profile,
            "schema": RUNTIME_TOPOLOGY_SCHEMA,
        },
    }


def _render_control_policy(
    *,
    compose: dict[str, Any],
    profile: str,
    release_manifest: dict[str, Any],
    service_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind every Compose image and mutable service field to the release manifest."""

    release_images = release_manifest.get("images")
    services = compose.get("services")
    if not isinstance(release_images, dict) or not isinstance(services, dict):
        raise InstallationConfigError("control policy inputs are invalid")
    runtime_images = {
        str(service_manifest["images"]["runtime"]): "runtime",
        str(service_manifest["images"]["ml_runtime"]): "ml_runtime",
    }
    release_references = {
        str(name): str(value["reference"])
        for name, value in release_images.items()
        if isinstance(value, dict) and isinstance(value.get("reference"), str)
    }
    service_specs: dict[str, Any] = {}
    default_services: list[str] = []
    for service_name, service in sorted(services.items()):
        if not isinstance(service, dict):
            raise InstallationConfigError(f"invalid Compose service: {service_name}")
        image = service.get("image")
        if not isinstance(image, str):
            raise InstallationConfigError(f"Compose service image is missing: {service_name}")
        release_key = runtime_images.get(image)
        if release_key is None:
            preferred_key = str(service_name).replace("-", "_")
            if release_references.get(preferred_key) == image:
                release_key = preferred_key
            matches = [name for name, reference in release_references.items() if reference == image]
            if release_key is None and len(matches) == 1:
                release_key = matches[0]
            if release_key is None:
                raise InstallationConfigError(
                    f"Compose image is absent or ambiguous in release manifest: {service_name}"
                )
        release_reference = release_references.get(release_key)
        if release_reference is None or "@sha256:" not in release_reference:
            raise InstallationConfigError(
                f"release image is not digest-pinned for control policy: {release_key}"
            )
        service_specs[str(service_name)] = {
            "image": image,
            "mounts": service.get("volumes", []),
            "environment_names": sorted((service.get("environment") or {}).keys()),
            "resources": service.get("deploy", {}).get("resources", {}).get("limits", {}),
            "command_sha256": sha256_bytes(
                json.dumps(
                    service.get("command", []), separators=(",", ":")
                ).encode("utf-8")
            ),
            "release_reference": release_reference,
            "restart_allowed": str(service_name) in SAFE_RESTART_SERVICES,
        }
        if service.get("restart") != "no" and not service.get("profiles"):
            default_services.append(str(service_name))
    return {
        "schema": "io.roehub.control-policy/v1alpha1",
        "profile": profile,
        "release_version": release_manifest["version"],
        "allowed_services": sorted(service_specs),
        "default_services": sorted(default_services),
        "services": service_specs,
    }


def _secret_init_service() -> dict[str, Any]:
    script = (
        "umask 077; "
        "if [ ! -s /run/roehub-secrets/postgresql-password ]; then "
        "head -c 48 /dev/urandom | base64 > /run/roehub-secrets/postgresql-password; "
        "chmod 0400 /run/roehub-secrets/postgresql-password; fi; "
        "if [ ! -s /run/roehub-secrets/clickhouse-password ]; then "
        "head -c 48 /dev/urandom | base64 > /run/roehub-secrets/clickhouse-password; "
        "chmod 0400 /run/roehub-secrets/clickhouse-password; "
        "chown 65532:65532 /run/roehub-secrets/clickhouse-password; fi; "
        "if [ ! -s /run/roehub-secrets/redis-password ]; then "
        "head -c 48 /dev/urandom | base64 > /run/roehub-secrets/redis-password; "
        "chmod 0400 /run/roehub-secrets/redis-password; fi; "
        "chown 0:0 /run/roehub-secrets/redis-password; "
        "if [ -e /run/roehub-secrets/redis-password-server ]; then "
        "chown 0:0 /run/roehub-secrets/redis-password-server; fi; "
        "if [ -e /run/roehub-secrets/redis.conf ]; then "
        "chown 0:0 /run/roehub-secrets/redis.conf; fi; "
        "cp /run/roehub-secrets/redis-password "
        "/run/roehub-secrets/redis-password-server; "
        "chmod 0400 /run/roehub-secrets/redis-password-server; "
        "chown 999:999 /run/roehub-secrets/redis-password-server; "
        "{ printf 'bind 0.0.0.0\\nprotected-mode yes\\nport 6379\\n' ; "
        "printf 'appendonly yes\\nmaxmemory-policy noeviction\\ndir /data\\nrequirepass '; "
        "cat /run/roehub-secrets/redis-password; } "
        "> /run/roehub-secrets/redis.conf; "
        "chmod 0400 /run/roehub-secrets/redis.conf; "
        "chown 999:999 /run/roehub-secrets/redis.conf; "
        "chown 65532:65532 /run/roehub-secrets/redis-password; "
        "if [ ! -s /run/roehub-secrets/identity-exchange-kek ]; then "
        "head -c 32 /dev/urandom | base64 > /run/roehub-secrets/identity-exchange-kek; "
        "chmod 0400 /run/roehub-secrets/identity-exchange-kek; "
        "chown 65532:65532 /run/roehub-secrets/identity-exchange-kek; fi; "
        "if [ ! -s /run/roehub-secrets/pgpass ]; then "
        "printf 'postgresql:5432:roehub:roehub:' > /run/roehub-secrets/pgpass; "
        "cat /run/roehub-secrets/postgresql-password >> /run/roehub-secrets/pgpass; "
        "chmod 0400 /run/roehub-secrets/pgpass; "
        "chown 65532:65532 /run/roehub-secrets/pgpass; fi; "
        "if [ ! -s /run/roehub-secrets/exchange-control-internal-api-token ]; then "
        "head -c 48 /dev/urandom | base64 > "
        "/run/roehub-secrets/exchange-control-internal-api-token; "
        "chmod 0400 /run/roehub-secrets/exchange-control-internal-api-token; "
        "chown 65532:65532 "
        "/run/roehub-secrets/exchange-control-internal-api-token; fi; "
        "if [ ! -s /run/roehub-secrets/exchange-control-transit-token ]; then "
        "head -c 48 /dev/urandom | base64 > "
        "/run/roehub-secrets/exchange-control-transit-token; "
        "chmod 0400 /run/roehub-secrets/exchange-control-transit-token; "
        "chown 65532:65532 /run/roehub-secrets/exchange-control-transit-token; fi; "
        "if [ ! -s /run/grafana-secrets/admin-password ]; then "
        "head -c 48 /dev/urandom | base64 > "
        "/run/grafana-secrets/admin-password; "
        "chmod 0400 /run/grafana-secrets/admin-password; "
        "chown 472:472 /run/grafana-secrets/admin-password; fi"
    )
    return {
        "image": PINNED_INFRASTRUCTURE_IMAGES["secret_init"],
        "command": ["/bin/sh", "-eu", "-c", script],
        "network_mode": "none",
        "read_only": True,
        "cap_drop": ["ALL"],
        "cap_add": ["CHOWN"],
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": ["/tmp:rw,noexec,nosuid,size=8m"],
        "volumes": [
            "runtime-secrets:/run/roehub-secrets",
            "grafana-secrets:/run/grafana-secrets",
        ],
        "labels": {
            "io.roehub.root-justification": (
                "one-shot ownership handoff for generated runtime secret files"
            )
        },
        "deploy": {"resources": {"limits": {"cpus": "0.25", "memory": "64M"}}},
        "restart": "no",
    }


def _postgres_service() -> dict[str, Any]:
    return {
        "image": PINNED_INFRASTRUCTURE_IMAGES["postgresql"],
        "environment": {
            "POSTGRES_DB": "roehub",
            "POSTGRES_USER": "roehub",
            "POSTGRES_PASSWORD_FILE": "/run/roehub-secrets/postgresql-password",
        },
        "depends_on": {"secret-init": {"condition": "service_completed_successfully"}},
        "healthcheck": {
            "test": ["CMD-SHELL", "pg_isready -U roehub -d roehub"],
            "interval": "2s",
            "timeout": "2s",
            "retries": 30,
        },
        "restart": "unless-stopped",
        "deploy": {"resources": {"limits": {"cpus": "1.0", "memory": "1024M"}}},
        "labels": {
            "io.roehub.root-justification": (
                "official init entrypoint owns a fresh volume then execs PostgreSQL non-root"
            )
        },
        "networks": ["roehub"],
        "volumes": [
            "postgres-data:/var/lib/postgresql/data",
            "runtime-secrets:/run/roehub-secrets:ro",
        ],
    }


def _redis_service() -> dict[str, Any]:
    return {
        "image": PINNED_INFRASTRUCTURE_IMAGES["redis"],
        "command": ["redis-server", "/run/roehub-secrets/redis.conf"],
        "depends_on": {"secret-init": {"condition": "service_completed_successfully"}},
        "healthcheck": {
            "test": [
                "CMD-SHELL",
                "redis-cli --no-auth-warning -a "
                "$$(cat /run/roehub-secrets/redis-password-server) ping",
            ],
            "interval": "2s",
            "timeout": "2s",
            "retries": 30,
        },
        "restart": "unless-stopped",
        "deploy": {"resources": {"limits": {"cpus": "0.5", "memory": "512M"}}},
        "labels": {
            "io.roehub.root-justification": (
                "official init entrypoint verifies a fresh volume then execs Redis non-root"
            )
        },
        "networks": ["roehub"],
        "volumes": [
            "redis-data:/data",
            "runtime-secrets:/run/roehub-secrets:ro",
        ],
    }


def _clickhouse_service() -> dict[str, Any]:
    return {
        "image": PINNED_INFRASTRUCTURE_IMAGES["clickhouse"],
        "environment": {
            "CLICKHOUSE_DB": "roehub",
            "CLICKHOUSE_USER": "default",
            "CLICKHOUSE_PASSWORD_FILE": "/run/roehub-secrets/clickhouse-password",
        },
        "depends_on": {"secret-init": {"condition": "service_completed_successfully"}},
        "healthcheck": {
            "test": [
                "CMD-SHELL",
                "clickhouse-client --user default --password "
                "$$(cat /run/roehub-secrets/clickhouse-password) --query 'SELECT 1'",
            ],
            "interval": "3s",
            "timeout": "3s",
            "retries": 40,
        },
        "restart": "unless-stopped",
        "deploy": {"resources": {"limits": {"cpus": "2.0", "memory": "2048M"}}},
        "labels": {
            "io.roehub.root-justification": (
                "official init entrypoint owns fresh data/log volumes then execs "
                "ClickHouse non-root"
            )
        },
        "networks": ["roehub"],
        "volumes": [
            "clickhouse-data:/var/lib/clickhouse",
            "clickhouse-logs:/var/log/clickhouse-server",
            "runtime-secrets:/run/roehub-secrets:ro",
        ],
    }


def _openbao_service() -> dict[str, Any]:
    health = (
        "bao status -address=http://127.0.0.1:8200 >/dev/null 2>&1; "
        'code=$$?; [ "$$code" -eq 0 ] || [ "$$code" -eq 2 ]'
    )
    return {
        "image": PINNED_INFRASTRUCTURE_IMAGES["openbao"],
        "entrypoint": ["bao"],
        "command": ["server", "-config=/openbao/config/openbao.hcl"],
        "user": "100:1000",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "healthcheck": {
            "test": ["CMD-SHELL", health],
            "interval": "3s",
            "timeout": "3s",
            "retries": 30,
        },
        "restart": "unless-stopped",
        "deploy": {"resources": {"limits": {"cpus": "0.5", "memory": "512M"}}},
        "networks": ["roehub"],
        "volumes": [
            "../../../../infra/openbao/config/openbao.hcl:/openbao/config/openbao.hcl:ro",
            "openbao-data:/openbao/file",
            "openbao-audit:/openbao/logs",
        ],
    }


def _runtime_role_service(
    *, role: dict[str, Any], service_manifest: dict[str, Any], profile: str, config: dict[str, Any]
) -> dict[str, Any]:
    image_key = str(role["image"])
    lifecycle = str(role["lifecycle"])
    service: dict[str, Any] = {
        "image": service_manifest["images"][image_key],
        "command": list(role["command"]),
        "user": "65532:65532",
        "read_only": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m", "/run:rw,noexec,nosuid,size=16m"],
        "networks": _role_networks(role),
        "environment": _runtime_environment(profile=profile, config=config),
        "volumes": [
            "./service-config.json:/etc/roehub/service-config.json:ro",
            "./service-configs/indicators.yaml:/etc/roehub/indicators.yaml:ro",
            "./service-configs/market_data.yaml:/etc/roehub/market_data.yaml:ro",
            "runtime-secrets:/run/roehub-secrets:ro",
        ],
        "deploy": {
            "resources": {
                "limits": {
                    "cpus": str(role["resources"]["cpus"]),
                    "memory": f"{role['resources']['memory_mb']}M",
                }
            }
        },
        "labels": {
            "io.roehub.lifecycle": lifecycle,
            "io.roehub.profile": profile,
            "io.roehub.role": role["name"],
        },
    }
    if role.get("config_mount"):
        service["volumes"].append(
            f"./service-configs/{role['config_mount']}:/etc/roehub/{role['config_mount']}:ro"
        )
    if role["name"] == "api":
        service["volumes"].extend(
            [
                "./service-configs/plugin-publisher-keys.json:"
                "/etc/roehub/plugin-publisher-keys.json:ro",
                "./service-configs/strategy.yaml:/etc/roehub/strategy.yaml:ro",
            ]
        )
        service["environment"]["ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED"] = (
            "false" if profile == "base" else "true"
        )
        service["environment"]["ROEHUB_OPERATIONAL_HEALTH_URL"] = (
            "http://operational-health:9300"
        )
    if role["name"] == "operational-health":
        service["environment"] = {
            "PYTHONUNBUFFERED": "1",
            "ROEHUB_ENV": "prod",
            "ROEHUB_PROFILE": profile,
        }
        service["volumes"] = [
            "./observability/operational-manifest.json:"
            "/etc/roehub/operational-manifest.json:ro"
        ]
    if role["name"] == "strategy-live-runner":
        service["environment"]["ROEHUB_METRICS_BIND_HOST"] = "0.0.0.0"
    if role["name"] == "exchange-control":
        service["environment"].update(
            {
                "OPENBAO_ADDR": "http://openbao:8200",
                "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER": "openbao_transit_v1",
                "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN_FILE": (
                    "/run/roehub-secrets/exchange-control-transit-token"
                ),
                "ROEHUB_EXCHANGE_CONTROL_CONTAINER_BIND": "true",
            }
        )
    if role["name"] == "exchange-execution":
        service["environment"].update(
            {
                "OPENBAO_ADDR": "http://openbao:8200",
                "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN_FILE": (
                    "/run/roehub-secrets/exchange-control-transit-token"
                ),
                "ROEHUB_EXCHANGE_EXECUTION_CONTAINER_BIND": "true",
            }
        )
    for path in role.get("persistent_paths", []):
        volume = "ml-state" if path == "/var/lib/roehub/ml" else "artifacts"
        service["volumes"].append(f"{volume}:{path}")
    service["depends_on"] = {
        dependency: {
            "condition": (
                "service_completed_successfully"
                if dependency == "storage-migrations"
                else "service_healthy"
            )
        }
        for dependency in role["depends_on"]
    }
    if lifecycle == "service":
        service["restart"] = "unless-stopped"
        service["init"] = True
        port = int(role["internal_port"])
        path = str(role.get("container_health_path", role["health_path"]))
        probe = (
            "import urllib.request; "
            f"urllib.request.urlopen('http://127.0.0.1:{port}{path}', timeout=2).read(1)"
        )
        service["healthcheck"] = {
            "test": ["CMD", "python", "-c", probe],
            "interval": "5s",
            "timeout": "3s",
            "retries": 20,
            "start_period": "5s",
        }
    else:
        service["restart"] = "no"
        if lifecycle == "operator-tool":
            service["profiles"] = ["tools"]
        elif lifecycle == "isolated-job":
            service["profiles"] = ["jobs"]
    if role["name"] == "web":
        service["ports"] = [f"127.0.0.1:{config['ports']['http']}:8010"]
    return service


def _role_networks(role: Mapping[str, Any]) -> list[str]:
    """Return the exact declared networks for one runtime role."""

    access = str(role.get("network_access", "internal_only"))
    if access == "internal_only":
        return ["roehub"]
    if access == "market_data_egress":
        return ["roehub", "market-data-egress"]
    if access == "web_ingress":
        return ["roehub", "web-ingress"]
    raise InstallationConfigError(
        f"runtime role {role['name']} has unsupported network_access: {access}"
    )


def _runtime_environment(*, profile: str, config: dict[str, Any]) -> dict[str, str]:
    postgres_dsn = "host=postgresql port=5432 dbname=roehub user=roehub"
    insecure_localhost = config["tls"]["mode"] == "disabled" and config["domain"] == "localhost"
    origin_scheme = "http" if config["tls"]["mode"] == "disabled" else "https"
    origin_port = (
        config["ports"]["http"] if config["tls"]["mode"] == "disabled" else config["ports"]["https"]
    )
    environment = {
        "PYTHONUNBUFFERED": "1",
        "ROEHUB_ENV": "prod",
        "ROEHUB_PROFILE": profile,
        "NUMBA_CACHE_DIR": "/tmp/roehub-numba-cache",
        "ROEHUB_NUMBA_CACHE_DIR": "/tmp/roehub-numba-cache",
        "PGPASSFILE": "/run/roehub-secrets/pgpass",
        "POSTGRES_DSN": postgres_dsn,
        "ROEHUB_STORAGE_POSTGRES_DSN": postgres_dsn,
        "ROEHUB_STORAGE_REDIS_URL": "redis://redis:6379/0",
        "ROEHUB_REDIS_PASSWORD_FILE": "/run/roehub-secrets/redis-password",
        "IDENTITY_PG_DSN": postgres_dsn,
        "EXTENSIONS_PG_DSN": postgres_dsn,
        "IDENTITY_FAIL_FAST": "true",
        "IDENTITY_SESSION_IDLE_TTL_SECONDS": "1800",
        "IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS": "43200",
        "IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE": ("/run/roehub-secrets/identity-exchange-kek"),
        "IDENTITY_LOCAL_RP_ID": str(config["domain"]),
        "IDENTITY_LOCAL_RP_NAME": "Roehub",
        "IDENTITY_LOCAL_ORIGIN": (f"{origin_scheme}://{config['domain']}:{origin_port}"),
        "IDENTITY_LOCAL_ALLOW_INSECURE_LOCALHOST": ("true" if insecure_localhost else "false"),
        "STRATEGY_PG_DSN": postgres_dsn,
        "NOTIFICATIONS_PG_DSN": postgres_dsn,
        "ROEHUB_REDIS_HOST": "redis",
        "ROEHUB_REDIS_PORT": "6379",
        "ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE": ("/etc/roehub/plugin-publisher-keys.json"),
        "ROEHUB_PLUGIN_UNSIGNED_DEVELOPMENT": "false",
        "ROEHUB_PLUGIN_GATEWAY_URL": "http://plugin-gateway:9209",
        "ROEHUB_PLUGIN_BUNDLE_SPOOL_ROOT": ("/var/lib/roehub/artifacts/plugin-bundles"),
        "WEB_API_BASE_URL": "http://web:8010",
        "WEB_API_UPSTREAM_URL": "http://api:8000",
        "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL": "http://exchange-control:9205",
        "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN_FILE": (
            "/run/roehub-secrets/exchange-control-internal-api-token"
        ),
        "ROEHUB_EXCHANGE_EXECUTION_CONFIG": "/etc/roehub/exchange-execution.yaml",
        "ROEHUB_STRATEGY_CONFIG": "/etc/roehub/strategy.yaml",
        "ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED": "false",
        "ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED": "false",
        "ROEHUB_EXCHANGE_ACCOUNT_STATE_SYNC_ENABLED": "false",
    }
    if profile != "base":
        environment.update(
            {
                "CH_HOST": "clickhouse",
                "CH_PORT": "8123",
                "CH_DATABASE": "roehub",
                "CH_USER": "default",
                "ROEHUB_STORAGE_CLICKHOUSE_DSN": "http://clickhouse:8123",
                "ROEHUB_CLICKHOUSE_PASSWORD_FILE": ("/run/roehub-secrets/clickhouse-password"),
            }
        )
    return environment


def write_runtime_outputs(output_root: Path, profile: str, outputs: Mapping[str, bytes]) -> None:
    profile_root = output_root / profile
    expected = set(outputs)
    for relative, content in outputs.items():
        path = profile_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
    if profile_root.exists():
        for path in sorted(profile_root.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(profile_root).as_posix() not in expected:
                raise InstallationConfigError(f"unexpected generated runtime file: {path}")


def check_runtime_outputs(output_root: Path, profile: str, outputs: Mapping[str, bytes]) -> None:
    profile_root = output_root / profile
    stale = [
        relative
        for relative, content in outputs.items()
        if not (profile_root / relative).is_file()
        or (profile_root / relative).read_bytes() != content
    ]
    actual = (
        {
            path.relative_to(profile_root).as_posix()
            for path in profile_root.rglob("*")
            if path.is_file()
        }
        if profile_root.exists()
        else set()
    )
    extra = sorted(actual - set(outputs))
    if stale or extra:
        raise InstallationConfigError(
            f"generated runtime topology is stale: profile={profile}, "
            f"stale={sorted(stale)}, extra={extra}"
        )


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InstallationConfigError(f"JSON root must be an object: {path}")
    return value
