"""Render an explicit, data-preserving local installation for the selected Navigator client.

Input contains paths and image identities, never credentials. Existing external volumes
are adopted; this renderer never creates users, seeds databases or enables artifact jobs.
The independent controller is bootstrapped separately so stopping the platform remains
recoverable. All workload mutations use DockerComposeControlBackend's bound policy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from trading.platform.config.installation import json_bytes, sha256_bytes, yaml_bytes
from trading.platform.config.runtime_topology import _render_control_policy

WORKERS = {
    "market-data-ws": ("apps.worker.market_data_ws.main.main", 9201),
    "market-data-scheduler": ("apps.scheduler.market_data_scheduler.main.main", 9202),
    "backtest-artifact-publisher": ("apps.scheduler.backtest_artifact_publisher.main.main", 9203),
}


def _budget_volume(config: dict) -> str:
    volume = config.get("postgres_budget_volume", "")
    if not isinstance(volume, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", volume):
        raise ValueError("one pre-provisioned shared PostgreSQL budget volume is required")
    if config.get("postgres_role_contract") != "separated-v1":
        raise ValueError(
            "application/monitor/maintenance roles must be provisioned before activation"
        )
    return volume


def render(
    config: dict, release: dict, *, source: bytes, release_source: bytes
) -> dict[str, bytes]:
    if config.get("schema") != "io.roehub.local-installation/v2alpha1":
        raise ValueError("local installation requires PG-only v2 configuration")
    if any("clickhouse" in key.lower() for key in config):
        raise ValueError("legacy storage keys are incompatible with PG-only v2")
    budget_volume = _budget_volume(config)
    root = Path(config["root"])
    if not root.is_absolute():
        raise ValueError("installation root must be absolute")
    runtime = config["runtime_image"]
    artifact_root = config["artifact_root"]
    mounts = [
        f"{root}/config:/etc/roehub:ro",
        f"{artifact_root}:{artifact_root}",
        "pg-budget:/run/roehub-pg-budget",
    ]
    common = dict(
        image=runtime,
        environment={
            "ROEHUB_PG_BUDGET_REQUIRED": "1",
            "ROEHUB_PG_BUDGET_ROOT": "/run/roehub-pg-budget",
        },
        user=config["runtime_user"],
        init=True,
        restart="unless-stopped",
        networks=["platform"],
        security_opt=["no-new-privileges:true"],
        cap_drop=["ALL"],
        stop_grace_period="180s",
        logging={"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}},
    )

    def app(command, *, volumes=(), env="worker", port=None, health=None):
        value = {
            **common,
            "command": command,
            "volumes": [*mounts, *volumes],
            "env_file": [f"{root}/private/{env}.env"],
            "deploy": {"resources": {"limits": {"cpus": "2", "memory": "2G"}}},
        }
        if port:
            value["ports"] = [port]
        if health:
            value["healthcheck"] = {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    f"import urllib.request; urllib.request.urlopen('{health}', timeout=2)",
                ],
                "interval": "5s",
                "timeout": "3s",
                "retries": 12,
                "start_period": "30s",
            }
        return value

    services = {
        "postgresql": {
            "image": release["images"]["postgresql"]["reference"],
            "restart": "unless-stopped",
            "networks": ["platform"],
            "volumes": [
                "postgres-data:/var/lib/postgresql/data",
            ],
            "env_file": [f"{root}/private/postgresql.env"],
            "healthcheck": {
                "test": [
                    "CMD-SHELL",
                    "test \"$$(sed -n 8p /var/lib/postgresql/data/postmaster.pid "
                    "| tr -d ' ')\" = ready",
                ],
                "interval": "5s",
                "timeout": "3s",
                "retries": 12,
            },
            "command": [
                "postgres",
                "-c",
                "max_connections=48",
                "-c",
                "shared_buffers=256MB",
                "-c",
                "work_mem=4MB",
                "-c",
                "reserved_connections=0",
                "-c",
                "superuser_reserved_connections=0",
            ],
            "stop_grace_period": "60s",
        },
    }
    services["api"] = app(
        [
            "python",
            "-m",
            "uvicorn",
            "tools.qa.backtests_client_fixture:create_api",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-access-log",
        ],
        env="api",
        volumes=[
            "control-socket:/var/run/roehub:ro",
            f"{root}/private/api:/etc/roehub-api-identity:ro",
            "trades-cache:/var/lib/roehub/trades-cache",
        ],
        health="http://127.0.0.1:8000/health",
    )
    services["web"] = app(
        [
            "python",
            "-m",
            "uvicorn",
            "tools.qa.navigator_preview:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8010",
            "--no-access-log",
        ],
        env="web",
        port="127.0.0.1:20120:8010",
        health="http://127.0.0.1:8010/health/ready",
    )
    # Web needs neither storage paths nor database credentials.
    services["web"]["volumes"] = []
    services["web"]["depends_on"] = {"api": {"condition": "service_healthy"}}
    for name, (module, port) in WORKERS.items():
        filename = (
            "backtest-artifacts.yaml"
            if name == "backtest-artifact-publisher"
            else "market-data.yaml"
        )
        command = [
            "python",
            "-m",
            module,
            "--config",
            f"/etc/roehub/{filename}",
            "--metrics-port",
            str(port),
        ]
        if name == "market-data-scheduler":
            command.append("--requests-only")
        services[name] = app(command, health=f"http://127.0.0.1:{port}/metrics")
    for name in ["api", *WORKERS]:
        services[name]["depends_on"] = {
            db: {"condition": "service_healthy"} for db in ("postgresql",)
        }
    services["operational-health"] = app(
        [
            "python",
            "-m",
            "apps.monitoring.operational_health",
            "--manifest",
            "/etc/roehub/operational-manifest.json",
            "--host",
            "0.0.0.0",
            "--port",
            "9300",
            "--interval-seconds",
            "2",
        ],
        env="observer",
        health="http://127.0.0.1:9300/health/live",
    )
    services["operational-health"]["volumes"] = [
        f"{root}/config:/etc/roehub:ro",
        "pg-budget:/run/roehub-pg-budget",
    ]
    compose = {
        "services": services,
        "networks": {"platform": {"name": config["network"], "external": True}},
        "volumes": {
            "pg-budget": {"external": True, "name": budget_volume},
            "postgres-data": {"external": True, "name": config["postgres_volume"]},
            "control-socket": {"external": True, "name": config["socket_volume"]},
            "trades-cache": {},
        },
    }
    limits = {
        "postgresql": ("2", "4G"),
        "api": ("1", "512M"),
        "web": ("0.5", "256M"),
        "operational-health": ("0.5", "128M"),
        "market-data-ws": ("1", "1G"),
        "market-data-scheduler": ("1", "384M"),
        "backtest-artifact-publisher": ("1", "768M"),
    }
    for name, (cpus, memory) in limits.items():
        services[name]["deploy"] = {"resources": {"limits": {"cpus": cpus, "memory": memory}}}
    manifest = {"images": {"runtime": runtime, "ml_runtime": "__unused__"}}
    policy = _render_control_policy(
        compose=compose, profile="trading", release_manifest=release, service_manifest=manifest
    )
    probes = []
    for name in services:
        if name == "operational-health":
            continue
        port, path, capability = {
            "api": (8000, "/health", "platform.api"),
            "web": (8010, "/health/ready", "platform.web"),
            "postgresql": (5432, "", "storage.postgresql"),
            "market-data-ws": (9201, "/metrics", "market_data.streaming"),
            "market-data-scheduler": (9202, "/metrics", "market_data.scheduler"),
            "backtest-artifact-publisher": (9203, "/metrics", "research.artifacts"),
        }[name]
        probes.append(
            dict(
                service_id=name,
                capability=capability,
                kind="postgresql"
                if name == "postgresql"
                else "http_json"
                if name in {"api", "web"}
                else "http_reachability",
                target=f"{name}:{port}" if not path else f"http://{name}:{port}{path}",
                runbook_id="runtime.service-degraded",
                action_ref="manage_worker" if name in WORKERS else "diagnostics",
                required=name in {"api", "web"},
            )
        )
    outputs = {
        "compose.yaml": yaml_bytes(compose),
        "control-policy.json": json_bytes(policy),
        "config/operational-manifest.json": json_bytes(
            {
                "schema": "io.roehub.operational-manifest/v1alpha1",
                "profile": "trading",
                "services": probes,
            }
        ),
    }
    if "control_image" in config:
        outputs["control-compose.yaml"] = yaml_bytes(render_controller(config))
    outputs["generation-manifest.json"] = json_bytes(
        {
            "schema": "io.roehub.runtime-topology-generation/v1alpha1",
            "profile": "trading",
            "inputs": {
                "release_manifest_sha256": sha256_bytes(release_source),
                "local_installation_sha256": sha256_bytes(source),
            },
            "outputs": {name: {"sha256": sha256_bytes(value)} for name, value in outputs.items()},
        }
    )
    return outputs


def render_controller(config: dict) -> dict:
    """Independent owner control plane; never part of its own workload allowlist."""
    if config.get("schema") != "io.roehub.local-installation/v2alpha1":
        raise ValueError("local installation requires PG-only v2 configuration")
    if any("clickhouse" in key.lower() for key in config):
        raise ValueError("legacy storage keys are incompatible with PG-only v2")
    budget_volume = _budget_volume(config)
    root = Path(config["root"])
    image = config["control_image"]
    if not image.startswith("sha256:") or len(image) != 71:
        raise ValueError("local controller image must be pinned by image ID")
    command = [
        "python",
        "-m",
        "apps.control_agent.main.main",
        "--profile-root",
        str(root),
        "--trusted-release-manifest",
        str(root / "release.json"),
        "--profile",
        "trading",
        "--project",
        "roehub-local",
        "--socket",
        "/var/run/roehub/control.sock",
        "--job-socket",
        "/var/run/roehub/job.sock",
        "--journal",
        str(root / "control-state/operations.jsonl"),
    ]
    for name in ("api", "owner", "job"):
        command.extend([f"--{name}-token-file", str(root / "private" / name)])
    return {
        "services": {
            "control-agent": {
                "image": image,
                "user": config["runtime_user"],
                "group_add": ["0"],
                "init": True,
                "restart": "unless-stopped",
                "networks": ["platform"],
                "security_opt": ["no-new-privileges:true"],
                "cap_drop": ["ALL"],
                "env_file": [str(root / "private/controller.env")],
                "environment": {
                    "ROEHUB_PG_BUDGET_REQUIRED": "1",
                    "ROEHUB_PG_BUDGET_ROOT": "/run/roehub-pg-budget",
                },
                "command": command,
                "volumes": [
                    "/var/run/docker.sock:/var/run/docker.sock",
                    f"{root}:{root}:ro",
                    f"{root}/control-state:{root}/control-state",
                    f"{root}/config:/etc/roehub:ro",
                    f"{config['artifact_root']}:{config['artifact_root']}",
                    "control-socket:/var/run/roehub",
                    "pg-budget:/run/roehub-pg-budget",
                ],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import socket; s=socket.socket(socket.AF_UNIX); "
                        "s.settimeout(2); s.connect('/var/run/roehub/control.sock'); s.close()",
                    ],
                    "interval": "5s",
                    "timeout": "3s",
                    "retries": 6,
                    "start_period": "5s",
                },
                "logging": {"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}},
            }
        },
        "networks": {"platform": {"external": True, "name": config["network"]}},
        "volumes": {
            "control-socket": {"external": True, "name": config["socket_volume"]},
            "pg-budget": {"external": True, "name": budget_volume},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    args = parser.parse_args()
    source, release_source = args.config.read_bytes(), args.release_manifest.read_bytes()
    config, release = json.loads(source), json.loads(release_source)
    for name, value in render(
        config, release, source=source, release_source=release_source
    ).items():
        path = Path(config["root"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    print("Local installation topology rendered; no services started")


if __name__ == "__main__":
    main()
