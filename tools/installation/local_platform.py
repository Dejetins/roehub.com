"""Render an explicit, data-preserving local installation for the selected Navigator client.

Input contains paths and image identities, never credentials. Existing external volumes
are adopted; this renderer never creates users, seeds databases or enables artifact jobs.
The independent controller is bootstrapped separately so stopping the platform remains
recoverable. All workload mutations use DockerComposeControlBackend's bound policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading.platform.config.installation import json_bytes, sha256_bytes, yaml_bytes
from trading.platform.config.runtime_topology import _render_control_policy

WORKERS = {
    "market-data-ws": ("apps.worker.market_data_ws.main.main", 9201),
    "market-data-scheduler": ("apps.scheduler.market_data_scheduler.main.main", 9202),
    "backtest-artifact-publisher": ("apps.scheduler.backtest_artifact_publisher.main.main", 9203),
}


def render(
    config: dict, release: dict, *, source: bytes, release_source: bytes
) -> dict[str, bytes]:
    root = Path(config["root"])
    if not root.is_absolute():
        raise ValueError("installation root must be absolute")
    runtime = config["runtime_image"]
    artifact_root = config["artifact_root"]
    mounts = [f"{root}/config:/etc/roehub:ro", f"{artifact_root}:{artifact_root}"]
    common = dict(
        image=runtime,
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
            "volumes": ["postgres-data:/var/lib/postgresql/data"],
            "env_file": [f"{root}/private/postgresql.env"],
            "healthcheck": {
                "test": ["CMD-SHELL", 'pg_isready -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'],
                "interval": "5s",
                "timeout": "3s",
                "retries": 12,
            },
            "stop_grace_period": "60s",
        },
        "clickhouse": {
            "image": release["images"]["clickhouse"]["reference"],
            "restart": "unless-stopped",
            "networks": ["platform"],
            "volumes": [
                "clickhouse-data:/var/lib/clickhouse",
                f"{root}/config/clickhouse-observability.xml:/etc/clickhouse-server/config.d/local.xml:ro",
                f"{root}/config/clickhouse-profiles.xml:/etc/clickhouse-server/users.d/local.xml:ro",
                f"{root}/config/clickhouse-capacity.xml:/etc/clickhouse-server/config.d/capacity.xml:ro",
                f"{root}/config/clickhouse-query-limits.xml:/etc/clickhouse-server/users.d/capacity.xml:ro",
            ],
            "env_file": [f"{root}/private/clickhouse.env"],
            "stop_grace_period": "60s",
            "healthcheck": {
                # Adopted installations have this table. Exercise the storage read path;
                # /ping remains successful even when memory pressure rejects all reads.
                "test": ["CMD", "clickhouse-client", "--query",
                         "SELECT count() FROM (SELECT market_id, symbol, "
                         "groupBitmapMerge(minute_bitmap_state) "
                         "FROM market_data.canonical_candles_1m_stats "
                         "GROUP BY market_id, symbol) SETTINGS max_threads=1, "
                         "max_execution_time=2, max_memory_usage=33554432"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 12,
            },
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
            db: {"condition": "service_healthy"} for db in ("postgresql", "clickhouse")
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
    services["operational-health"]["volumes"] = [f"{root}/config:/etc/roehub:ro"]
    workload_mode = config.get("clickhouse_workload_budgets", "legacy")
    if workload_mode not in {"legacy", "provisioned", "active"}:
        raise ValueError("invalid clickhouse_workload_budgets lifecycle")
    if workload_mode in {"provisioned", "active"}:
        # Provision profiles/users first, then switch application clients. Class files
        # contain no admin credential, and each service receives only its own identity.
        services["clickhouse"]["volumes"].append(
            f"{root}/private/clickhouse-workloads/users.xml:"
            "/run/roehub-private/workloads.xml:ro"
        )
        services["clickhouse"]["volumes"].append(
            f"{root}/config/clickhouse-workloads-entrypoint.sh:/workloads-entrypoint.sh:ro"
        )
        services["clickhouse"]["entrypoint"] = ["/bin/sh", "/workloads-entrypoint.sh"]
    if workload_mode == "active":
        for service, workload in {
            "api": "interactive", "market-data-ws": "stream",
            "market-data-scheduler": "backfill", "backtest-artifact-publisher": "artifacts",
        }.items():
            services[service]["environment"] = {
                "ROEHUB_CH_WORKLOADS_REQUIRED": "1",
                f"ROEHUB_CH_{workload.upper()}_PASSWORD_FILE": "/run/roehub-ch/password",
                # Compose environment replaces legacy shared credentials from env_file.
                "CH_USER": "", "CLICKHOUSE_USER": "", "CH_PASSWORD": "",
                "CLICKHOUSE_PASSWORD": "", "ROEHUB_CLICKHOUSE_PASSWORD_FILE": "",
                "CH_PASSWORD_FILE": "", "CLICKHOUSE_PASSWORD_FILE": "",
            }
            services[service]["volumes"].append(
                f"{root}/private/clickhouse-workloads/{workload}.password:"
                "/run/roehub-ch/password:ro"
            )
    compose = {
        "services": services,
        "networks": {"platform": {"name": config["network"], "external": True}},
        "volumes": {
            "postgres-data": {"external": True, "name": config["postgres_volume"]},
            "clickhouse-data": {"external": True, "name": config["clickhouse_volume"]},
            "control-socket": {"external": True, "name": config["socket_volume"]},
            "trades-cache": {},
        },
    }
    limits = {
        "postgresql": ("1", "256M"),
        "clickhouse": ("2", "1536M"),
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
            "clickhouse": (8123, "/ping", "storage.clickhouse"),
            "market-data-ws": (9201, "/metrics", "market_data.streaming"),
            "market-data-scheduler": (9202, "/metrics", "market_data.scheduler"),
            "backtest-artifact-publisher": (9203, "/metrics", "research.artifacts"),
        }[name]
        probes.append(
            dict(
                service_id=name,
                capability=capability,
                kind="tcp_reachability"
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
    local_profiles = Path(__file__).resolve().parents[2] / "configs" / "dev"
    outputs = {
        "config/clickhouse-observability.xml": (
            local_profiles / "clickhouse-local-observability.xml").read_bytes(),
        "config/clickhouse-profiles.xml": (
            local_profiles / "clickhouse-local-profiles.xml").read_bytes(),
        "config/clickhouse-capacity.xml": (
            b"<clickhouse>"
            b"<max_server_memory_usage>1073741824</max_server_memory_usage>"
            # Bound overhead even if a diagnostic query explicitly opts into this cache.
            b"<query_condition_cache_size>1048576</query_condition_cache_size>"
            b"<mark_cache_size>67108864</mark_cache_size>"
            b"<uncompressed_cache_size>16777216</uncompressed_cache_size>"
            # Bound background work too: query limits alone cannot constrain log merges.
            b"<background_pool_size>4</background_pool_size>"
            b"<background_merges_mutations_concurrency_ratio>1</background_merges_mutations_concurrency_ratio>"
            b"<background_schedule_pool_size>16</background_schedule_pool_size>"
            b"<background_move_pool_size>2</background_move_pool_size>"
            b"<background_fetches_pool_size>2</background_fetches_pool_size>"
            b"<background_common_pool_size>2</background_common_pool_size>"
            b"<background_buffer_flush_schedule_pool_size>2</background_buffer_flush_schedule_pool_size>"
            b"<background_message_broker_schedule_pool_size>2</background_message_broker_schedule_pool_size>"
            b"<background_distributed_schedule_pool_size>2</background_distributed_schedule_pool_size>"
            b"<merges_mutations_memory_usage_soft_limit>268435456</merges_mutations_memory_usage_soft_limit>"
            b"<memory_worker_period_ms>100</memory_worker_period_ms>"
            b"<memory_worker_purge_total_memory_threshold_ratio>0.7</memory_worker_purge_total_memory_threshold_ratio>"
            b"<merge_tree>"
            # The 1,000+ column metric log otherwise merges all column buffers together.
            b"<vertical_merge_algorithm_min_rows_to_activate>1</vertical_merge_algorithm_min_rows_to_activate>"
            b"<merge_max_block_size>512</merge_max_block_size>"
            b"<merge_max_block_size_bytes>1048576</merge_max_block_size_bytes>"
            b"<max_bytes_to_merge_at_max_space_in_pool>134217728</max_bytes_to_merge_at_max_space_in_pool>"
            b"<number_of_free_entries_in_pool_to_execute_mutation>2</number_of_free_entries_in_pool_to_execute_mutation>"
            b"<number_of_free_entries_in_pool_to_lower_max_size_of_merge>2</number_of_free_entries_in_pool_to_lower_max_size_of_merge>"
            b"<number_of_free_entries_in_pool_to_execute_optimize_entire_partition>2</number_of_free_entries_in_pool_to_execute_optimize_entire_partition>"
            b"</merge_tree>"
            b"</clickhouse>\n"
        ),
        "config/clickhouse-query-limits.xml": (
            b"<clickhouse>"
            b"<profiles>"
            b"<default>"
            b"<use_query_condition_cache>0</use_query_condition_cache>"
            b"<max_threads>2</max_threads>"
            b"<max_memory_usage>268435456</max_memory_usage>"
            b"</default>"
            b"</profiles>"
            b"</clickhouse>\n"
        ),
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
    if workload_mode in {"provisioned", "active"}:
        # Root bootstrap copies only the password-hash config into the container with
        # the image's database UID. Host secrets stay 0600; the official entrypoint
        # still drops privileges, and recreation repeats this deterministic step.
        outputs["config/clickhouse-workloads-entrypoint.sh"] = (
            b"#!/bin/sh\nset -eu\n"
            b"test -s /run/roehub-private/workloads.xml\n"
            b"install -m 600 -o clickhouse -g clickhouse "
            b"/run/roehub-private/workloads.xml /etc/clickhouse-server/users.d/workloads.xml\n"
            b'exec /entrypoint.sh "$@"\n'
        )
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
                "command": command,
                "volumes": [
                    "/var/run/docker.sock:/var/run/docker.sock",
                    f"{root}:{root}:ro",
                    f"{root}/control-state:{root}/control-state",
                    f"{root}/config:/etc/roehub:ro",
                    f"{config['artifact_root']}:{config['artifact_root']}",
                    "control-socket:/var/run/roehub",
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
        "volumes": {"control-socket": {"external": True, "name": config["socket_volume"]}},
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
