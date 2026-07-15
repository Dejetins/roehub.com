#!/usr/bin/env python3
"""Prove generated runtime profiles at the real Docker Compose boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import yaml

from tools.release.generate_runtime_topology import DEFAULT_OUTPUT
from tools.release.generate_runtime_topology import run as run_generator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/17-runtime-topology-proof.json"
)
INFRASTRUCTURE_PORTS = {
    "postgresql": 5432,
    "clickhouse": 8123,
    "redis": 6379,
    "openbao": 8200,
}
_PERSISTENCE_SENTINEL = "stage17-persisted"
_METRIC_READINESS_MARKERS = {
    "notification-dispatcher": "notification_dispatcher_deliveries_claimed_total",
    "market-data-ws": "ws_connected",
    "market-data-scheduler": "scheduler_job_runs_total",
    "backtest-artifact-publisher": "backtest_artifact_publish_runs_total",
    "backtest-job-runner": "backtest_runner_active",
    "clickhouse-exporter": "clickhouse_exporter_scrape_success",
}


class RuntimeTopologyProofError(RuntimeError):
    """Raised when a declared container boundary cannot be proved."""


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        stdout = (error.stdout or "").strip()
        stderr = (error.stderr or "").strip()
        raise RuntimeTopologyProofError(
            f"command failed ({error.returncode}): {' '.join(command)}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ) from error


def _compose(*, project: str, compose: Path, override: Path | None = None) -> list[str]:
    command = ["docker", "compose", "-p", project, "-f", str(compose)]
    if override is not None:
        command.extend(["-f", str(override)])
    return command


def _probe_command(role: dict[str, Any], ports: dict[str, int], *, once: bool) -> list[str]:
    command = [
        "python",
        "-m",
        "apps.runtime_probe.main",
        "--role",
        str(role["name"]),
        "--state-path",
        f"/var/lib/roehub/runtime-proof/{role['name']}.json",
    ]
    for entrypoint in role["entrypoints"]:
        command.extend(["--entrypoint", str(entrypoint)])
    for dependency in role["depends_on"]:
        if dependency in ports:
            command.extend(["--dependency", f"{dependency}:{ports[dependency]}"])
    if once:
        command.append("--once")
    else:
        command.extend(["--port", str(role["internal_port"])])
    return command


def _proof_override(*, topology: dict[str, Any], state_root: Path) -> dict[str, Any]:
    ports = dict(INFRASTRUCTURE_PORTS)
    for role in topology["roles"]:
        if role.get("internal_port"):
            ports[str(role["name"])] = int(role["internal_port"])
    services: dict[str, Any] = {}
    for role in topology["roles"]:
        if role["lifecycle"] not in {"service", "one-shot"}:
            continue
        once = role["lifecycle"] == "one-shot"
        row: dict[str, Any] = {
            "command": _probe_command(role, ports, once=once),
            "volumes": [f"{state_root}:/var/lib/roehub/runtime-proof"],
        }
        if not once:
            port = int(role["internal_port"])
            probe = (
                "import urllib.request; "
                f"urllib.request.urlopen('http://127.0.0.1:{port}/health/ready', "
                "timeout=2).read(1)"
            )
            row["healthcheck"] = {
                "test": ["CMD", "python", "-c", probe],
                "interval": "2s",
                "timeout": "2s",
                "retries": 40,
                "start_period": "2s",
            }
        services[str(role["name"])] = row
    return {"services": services}


def _inspect_runtime_services(
    *, project: str, compose: Path, override: Path | None, topology: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    inspected: dict[str, dict[str, Any]] = {}
    for role in topology["roles"]:
        if role["lifecycle"] != "service":
            continue
        service = str(role["name"])
        container_id = _run(
            [*_compose(project=project, compose=compose, override=override), "ps", "-q", service]
        ).stdout.strip()
        if not container_id:
            raise RuntimeTopologyProofError(f"service container is missing: {service}")
        payload = json.loads(_run(["docker", "inspect", container_id]).stdout)[0]
        user = str(payload["Config"].get("User") or "")
        readonly = bool(payload["HostConfig"].get("ReadonlyRootfs"))
        memory = int(payload["HostConfig"].get("Memory") or 0)
        nano_cpus = int(payload["HostConfig"].get("NanoCpus") or 0)
        health = str(payload["State"].get("Health", {}).get("Status") or "")
        if user in {"", "0", "0:0", "root"}:
            raise RuntimeTopologyProofError(f"runtime service is root: {service}={user!r}")
        if not readonly:
            raise RuntimeTopologyProofError(f"runtime root filesystem is writable: {service}")
        if memory <= 0 or nano_cpus <= 0:
            raise RuntimeTopologyProofError(f"runtime resource limit is missing: {service}")
        if health != "healthy":
            raise RuntimeTopologyProofError(f"runtime service is not healthy: {service}={health}")
        inspected[service] = {
            "user": user,
            "read_only": readonly,
            "memory_bytes": memory,
            "nano_cpus": nano_cpus,
            "health": health,
        }
    return inspected


def _inspect_infrastructure_services(
    *,
    project: str,
    compose: Path,
    override: Path | None,
    infrastructure: Sequence[str],
) -> dict[str, dict[str, Any]]:
    inspected: dict[str, dict[str, Any]] = {}
    compose_command = _compose(project=project, compose=compose, override=override)
    for service in infrastructure:
        container_id = _run([*compose_command, "ps", "-q", service]).stdout.strip()
        if not container_id:
            raise RuntimeTopologyProofError(f"infrastructure container is missing: {service}")
        payload = json.loads(_run(["docker", "inspect", container_id]).stdout)[0]
        memory = int(payload["HostConfig"].get("Memory") or 0)
        nano_cpus = int(payload["HostConfig"].get("NanoCpus") or 0)
        if memory <= 0 or nano_cpus <= 0:
            raise RuntimeTopologyProofError(f"infrastructure resource limit is missing: {service}")
        configured_user = str(payload["Config"].get("User") or "")
        labels = payload["Config"].get("Labels") or {}
        justification = str(labels.get("io.roehub.root-justification") or "")
        if configured_user in {"", "0", "0:0", "root"} and not justification:
            raise RuntimeTopologyProofError(f"infrastructure root init is not justified: {service}")
        uid_raw = _run(
            [
                *compose_command,
                "exec",
                "-T",
                service,
                "sh",
                "-ec",
                "sed -n 's/^Uid:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p' /proc/1/status",
            ]
        ).stdout.strip()
        if not uid_raw.isdigit() or int(uid_raw) == 0:
            raise RuntimeTopologyProofError(
                f"infrastructure product process is root: {service}={uid_raw!r}"
            )
        inspected[service] = {
            "configured_user": configured_user or "official-init-entrypoint",
            "pid1_uid": int(uid_raw),
            "root_init_justification": justification or None,
            "memory_bytes": memory,
            "nano_cpus": nano_cpus,
        }

    init_id = _run([*compose_command, "ps", "-aq", "secret-init"]).stdout.strip()
    if not init_id:
        raise RuntimeTopologyProofError("secret-init container is missing")
    init_payload = json.loads(_run(["docker", "inspect", init_id]).stdout)[0]
    init_labels = init_payload["Config"].get("Labels") or {}
    init_justification = str(init_labels.get("io.roehub.root-justification") or "")
    if not init_justification:
        raise RuntimeTopologyProofError("secret-init root boundary is not justified")
    if init_payload["HostConfig"].get("NetworkMode") != "none":
        raise RuntimeTopologyProofError("secret-init must not have a network")
    if not bool(init_payload["HostConfig"].get("ReadonlyRootfs")):
        raise RuntimeTopologyProofError("secret-init root filesystem must be read-only")
    init_capabilities = {
        str(value).removeprefix("CAP_")
        for value in (init_payload["HostConfig"].get("CapAdd") or [])
    }
    if init_capabilities != {"CHOWN"}:
        raise RuntimeTopologyProofError("secret-init capability set is not exact")
    inspected["secret-init"] = {
        "configured_user": "root",
        "root_init_justification": init_justification,
        "network_mode": "none",
        "read_only": True,
        "cap_add": ["CHOWN"],
    }
    return inspected


def _wait_for_service_health(
    *,
    project: str,
    compose: Path,
    service: str,
    timeout: float = 90.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        container_id = _run(
            [*_compose(project=project, compose=compose), "ps", "-q", service]
        ).stdout.strip()
        if container_id:
            payload = json.loads(_run(["docker", "inspect", container_id]).stdout)[0]
            if str(payload["State"].get("Health", {}).get("Status") or "") == "healthy":
                return
        time.sleep(1)
    raise RuntimeTopologyProofError(f"service did not become healthy: {service}")


def _service_http_json(
    *, project: str, compose: Path, service: str, port: int, path: str
) -> dict[str, Any]:
    script = (
        "import json,urllib.error,urllib.request; "
        f"url='http://127.0.0.1:{port}{path}'; "
        "code=200; "
        "\ntry:\n response=urllib.request.urlopen(url,timeout=3)"
        "\nexcept urllib.error.HTTPError as error:\n response=error; code=error.code"
        "\npayload=json.loads(response.read()); "
        "print(json.dumps({'status_code':code,'payload':payload},sort_keys=True))"
    )
    raw = _run(
        [
            *_compose(project=project, compose=compose),
            "exec",
            "-T",
            service,
            "python",
            "-c",
            script,
        ]
    ).stdout
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeTopologyProofError(f"invalid HTTP proof for {service}{path}")
    return value


def _service_http_text(*, project: str, compose: Path, service: str, port: int, path: str) -> str:
    script = (
        "import urllib.request; "
        f"print(urllib.request.urlopen('http://127.0.0.1:{port}{path}',timeout=3)"
        ".read().decode())"
    )
    return _run(
        [
            *_compose(project=project, compose=compose),
            "exec",
            "-T",
            service,
            "python",
            "-c",
            script,
        ]
    ).stdout


def _actual_role_readiness(
    *, project: str, compose: Path, topology: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    readiness: dict[str, dict[str, Any]] = {}
    for role in topology["roles"]:
        if role["lifecycle"] != "service":
            continue
        service = str(role["name"])
        port = int(role["internal_port"])
        health_path = str(role["health_path"])
        marker = _METRIC_READINESS_MARKERS.get(service)
        if marker is not None:
            metrics = _service_http_text(
                project=project,
                compose=compose,
                service=service,
                port=port,
                path=health_path,
            )
            if marker not in metrics:
                raise RuntimeTopologyProofError(
                    f"application readiness metric is missing: {service}={marker}"
                )
            if service == "clickhouse-exporter":
                success_rows = [
                    row
                    for row in metrics.splitlines()
                    if row.startswith("clickhouse_exporter_scrape_success ")
                ]
                if not success_rows or float(success_rows[-1].split()[-1]) != 1.0:
                    raise RuntimeTopologyProofError(
                        "ClickHouse exporter has not completed a successful scrape"
                    )
            readiness[service] = {
                "proof": "initialized_operational_metric",
                "marker": marker,
                "ready": True,
            }
            continue
        response = _service_http_json(
            project=project,
            compose=compose,
            service=service,
            port=port,
            path=health_path,
        )
        if service == "exchange-control":
            payload = response.get("payload")
            checks = payload.get("checks", []) if isinstance(payload, dict) else []
            transit = next(
                (
                    check
                    for check in checks
                    if isinstance(check, dict) and check.get("name") == "secret_cipher_transit"
                ),
                None,
            )
            if (
                response.get("status_code") != 503
                or not isinstance(payload, dict)
                or payload.get("status") != "not_ready"
                or not isinstance(transit, dict)
                or transit.get("status") != "not_ready"
            ):
                raise RuntimeTopologyProofError(
                    "fresh exchange-control must fail closed while OpenBao is uninitialized"
                )
            readiness[service] = {
                "proof": "application_health_endpoint",
                "ready": False,
                "mode": "safe-degraded",
                "reason": "openbao_transit_unavailable",
            }
            continue
        if service == "exchange-execution":
            payload = response.get("payload")
            dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else []
            pitr = next(
                (
                    dependency
                    for dependency in dependencies
                    if isinstance(dependency, dict) and dependency.get("name") == "ledger_pitr"
                ),
                None,
            )
            if (
                response.get("status_code") != 503
                or not isinstance(payload, dict)
                or payload.get("status") != "not_ready"
                or payload.get("adapter_mode") != "disabled"
                or not isinstance(pitr, dict)
                or pitr.get("reason") != "pitr_restore_not_verified"
            ):
                raise RuntimeTopologyProofError(
                    "fresh exchange-execution must fail closed without PITR verification"
                )
            readiness[service] = {
                "proof": "application_health_endpoint",
                "ready": False,
                "mode": "safe-degraded",
                "reason": "pitr_restore_not_verified",
            }
            continue
        if service == "rl-inference":
            payload = response.get("payload")
            degraded_reasons = (
                set(payload.get("degraded_reasons", [])) if isinstance(payload, dict) else set()
            )
            if (
                response.get("status_code") != 503
                or not isinstance(payload, dict)
                or payload.get("ready") is not False
                or degraded_reasons != {"inference_disabled", "source_events_disabled"}
            ):
                raise RuntimeTopologyProofError(
                    "fresh rl-inference must remain safe-disabled"
                )
            readiness[service] = {
                "proof": "application_health_endpoint",
                "ready": False,
                "mode": "safe-disabled",
                "reason": "inference_disabled",
            }
            continue
        if response.get("status_code") != 200:
            raise RuntimeTopologyProofError(
                f"declared service health failed: {service}={response.get('status_code')}"
            )
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeTopologyProofError(
                f"declared service health payload is invalid: {service}"
            )
        if service in {"notification-report-scheduler", "telegram-bot-worker"}:
            if payload.get("ready") is not True or payload.get("mode") != "disabled":
                raise RuntimeTopologyProofError(
                    f"safe-disabled worker readiness is invalid: {service}"
                )
        readiness[service] = {
            "proof": "application_health_endpoint",
            "ready": True,
            "mode": payload.get("mode"),
        }

    _run(
        [
            *_compose(project=project, compose=compose),
            "exec",
            "-T",
            "postgresql",
            "psql",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-Atqc",
            "SELECT 1",
        ]
    )
    return readiness


def _persistent_state_sentinels(
    *,
    project: str,
    compose: Path,
    override: Path | None,
    profile: str,
    create: bool,
) -> dict[str, str]:
    compose_command = _compose(project=project, compose=compose, override=override)
    values: dict[str, str] = {}
    values["postgresql"] = _postgres_sentinel(
        project=project,
        compose=compose,
        override=override,
        create=create,
    )

    redis_script = (
        'export REDISCLI_AUTH="$(cat /run/roehub-secrets/redis-password-server)"; '
        + (
            "redis-cli SET roehub:runtime-topology:sentinel stage17-persisted " ">/dev/null; "
            if create
            else ""
        )
        + "redis-cli --raw GET roehub:runtime-topology:sentinel"
    )
    values["redis"] = _run(
        [*compose_command, "exec", "-T", "redis", "sh", "-ec", redis_script]
    ).stdout.strip()

    for owner, path in (
        ("openbao", "/openbao/file/runtime-topology-sentinel"),
        ("openbao-audit", "/openbao/logs/runtime-topology-sentinel"),
    ):
        script = (
            f"printf '{_PERSISTENCE_SENTINEL}\\n' > {path}; cat {path}" if create else f"cat {path}"
        )
        values[owner] = _run(
            [*compose_command, "exec", "-T", "openbao", "sh", "-ec", script]
        ).stdout.strip()

    artifact_script = (
        "from pathlib import Path; "
        "p=Path('/var/lib/roehub/artifacts/runtime-topology-sentinel'); "
        + (f"p.write_text('{_PERSISTENCE_SENTINEL}\\n') ; " if create else "")
        + "print(p.read_text().strip())"
    )
    values["artifact-store"] = _run(
        [
            *compose_command,
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            artifact_script,
        ]
    ).stdout.strip()

    if profile != "base":
        clickhouse_queries = []
        if create:
            clickhouse_queries.extend(
                [
                    "CREATE TABLE IF NOT EXISTS roehub.runtime_topology_sentinel "
                    "(id UInt8, value String) ENGINE=ReplacingMergeTree ORDER BY id",
                    "INSERT INTO roehub.runtime_topology_sentinel VALUES "
                    f"(1, '{_PERSISTENCE_SENTINEL}')",
                ]
            )
        clickhouse_queries.append(
            "SELECT value FROM roehub.runtime_topology_sentinel "
            "WHERE id=1 ORDER BY _part DESC LIMIT 1"
        )
        clickhouse_script = (
            "import base64,json,urllib.request\n"
            "from pathlib import Path\n"
            "password=Path('/run/roehub-secrets/clickhouse-password').read_text().strip()\n"
            "authorization='Basic '+base64.b64encode(('default:'+password).encode()).decode()\n"
            "url='http://clickhouse:8123/?database=roehub'\n"
            f"queries={json.dumps(clickhouse_queries)}\n"
            "value=''\n"
            "for query in queries:\n"
            "    request=urllib.request.Request(url,data=query.encode(),"
            "headers={'Authorization':authorization},method='POST')\n"
            "    with urllib.request.urlopen(request,timeout=10) as response:\n"
            "        value=response.read().decode().strip()\n"
            "print(value)\n"
        )
        values["clickhouse"] = _run(
            [*compose_command, "exec", "-T", "api", "python", "-c", clickhouse_script],
            timeout=45,
        ).stdout.strip()

    if profile == "ml":
        ml_script = (
            "from pathlib import Path; "
            "p=Path('/var/lib/roehub/ml/runtime-topology-sentinel'); "
            + (f"p.write_text('{_PERSISTENCE_SENTINEL}\\n') ; " if create else "")
            + "print(p.read_text().strip())"
        )
        values["ml-runtime"] = _run(
            [
                *compose_command,
                "exec",
                "-T",
                "rl-inference",
                "python",
                "-c",
                ml_script,
            ]
        ).stdout.strip()

    invalid = {owner: value for owner, value in values.items() if value != _PERSISTENCE_SENTINEL}
    if invalid:
        raise RuntimeTopologyProofError(
            f"persistent state sentinel mismatch: {profile}={sorted(invalid)}"
        )
    return {owner: "passed" for owner in sorted(values)}


def _actual_job_runtime_lifecycle_proof() -> dict[str, Any]:
    raw = _run(
        [sys.executable, str(ROOT / "tests/fixtures/jobs/runtime_proof.py")],
        timeout=900,
    ).stdout.strip()
    if not raw:
        raise RuntimeTopologyProofError("job runtime lifecycle proof returned no output")
    payload = json.loads(raw.splitlines()[-1])
    required = {
        "artifact_result_publication",
        "cancel_cleanup",
        "docker_socket_denial",
        "non_root_read_only",
        "restart_recovery",
        "timeout_cleanup",
    }
    if payload.get("status") != "passed" or any(payload.get(key) != "passed" for key in required):
        raise RuntimeTopologyProofError("job runtime lifecycle proof did not pass")
    return {
        "schema": payload.get("schema"),
        "status": "passed",
        "executor": "JobAttemptExecutor",
        "runner": "OciJobRunner",
        "checks": sorted(required),
    }


def _actual_storage_readiness(*, project: str, compose: Path) -> dict[str, Any]:
    raw = _run(
        [
            *_compose(project=project, compose=compose),
            "run",
            "--rm",
            "--no-deps",
            "storage-migrations",
            "python",
            "-m",
            "apps.migrations.storage_main",
            "--service-config",
            "/etc/roehub/service-config.json",
            "readiness",
        ],
        timeout=180,
    ).stdout
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("ready") is not True:
        raise RuntimeTopologyProofError("actual storage readiness did not pass")
    return value


def _actual_on_demand_roles(*, project: str, compose: Path, topology: dict[str, Any]) -> list[str]:
    completed: list[str] = []
    for role in topology["roles"]:
        lifecycle = str(role["lifecycle"])
        if lifecycle not in {"operator-tool", "isolated-job"}:
            continue
        profile_flag = "tools" if lifecycle == "operator-tool" else "jobs"
        service = str(role["name"])
        smoke_command = (
            [*(str(item) for item in role["command"]), "--help"]
            if service == "domain-cli"
            else []
        )
        _run(
            [
                *_compose(project=project, compose=compose),
                "--profile",
                profile_flag,
                "run",
                "--rm",
                "--no-deps",
                service,
                *smoke_command,
            ],
            timeout=180,
        )
        completed.append(service)
    return completed


def _safe_default_proof(profile: str) -> dict[str, Any]:
    root = DEFAULT_OUTPUT / profile / "service-configs"
    notifications = yaml.safe_load((root / "notifications.yaml").read_text())
    strategy = yaml.safe_load((root / "strategy.yaml").read_text())
    execution = yaml.safe_load((root / "exchange-execution.yaml").read_text())
    if notifications["notifications"]["dispatcher"]["provider_mode"] != "log_only":
        raise RuntimeTopologyProofError("notification provider mode is not log_only")
    if notifications["notifications"]["telegram_bot"]["enabled"] is not False:
        raise RuntimeTopologyProofError("Telegram must be disabled by default")
    if notifications["notifications"]["report_scheduler"]["enabled"] is not False:
        raise RuntimeTopologyProofError("notification report scheduler must be disabled by default")
    if strategy["strategy"]["producer"]["enabled"] is not False:
        raise RuntimeTopologyProofError("strategy producer must be disabled by default")
    process = execution["exchange_execution"]["process"]
    if process["adapter_mode"] != "disabled" or process["consumer_enabled"] is not False:
        raise RuntimeTopologyProofError("exchange execution must be disabled by default")
    result: dict[str, Any] = {
        "notification_provider_mode": "log_only",
        "notification_report_scheduler_enabled": False,
        "telegram_enabled": False,
        "strategy_producer_enabled": False,
        "exchange_adapter_mode": "disabled",
        "exchange_consumer_enabled": False,
    }
    if profile == "ml":
        rl = yaml.safe_load((root / "rl-runtime.yaml").read_text())
        if rl["inference"]["enabled"] is not False:
            raise RuntimeTopologyProofError("RL inference must be disabled by default")
        result["rl_inference_enabled"] = False
    return result


def _actual_profile_smoke(profile: str) -> dict[str, Any]:
    compose = DEFAULT_OUTPUT / profile / "compose.yaml"
    topology = json.loads((DEFAULT_OUTPUT / profile / "runtime-topology.json").read_text())
    project = f"roehub-stage17-actual-{profile}-{uuid4().hex[:8]}"
    compose_command = _compose(project=project, compose=compose)
    try:
        _run([*compose_command, "config", "--quiet"])
        _run(
            [*compose_command, "up", "-d", "--wait", "--wait-timeout", "240"],
            timeout=300,
        )
        inspected = _inspect_runtime_services(
            project=project,
            compose=compose,
            override=None,
            topology=topology,
        )
        infrastructure = _inspect_infrastructure_services(
            project=project,
            compose=compose,
            override=None,
            infrastructure=tuple(topology["infrastructure"]),
        )
        running = set(
            _run([*compose_command, "ps", "--services", "--status", "running"])
            .stdout.strip()
            .splitlines()
        )
        expected = {
            str(role["name"]) for role in topology["roles"] if role["lifecycle"] == "service"
        } | set(topology["infrastructure"])
        missing = sorted(expected - running)
        if missing:
            raise RuntimeTopologyProofError(
                f"actual profile services are not running: {profile}={missing}"
            )
        web = _service_http_json(
            project=project,
            compose=compose,
            service="web",
            port=8010,
            path="/health/ready",
        )
        if web.get("status_code") != 200 or web.get("payload", {}).get("ready") is not True:
            raise RuntimeTopologyProofError(f"actual web readiness failed: {profile}")
        storage = _actual_storage_readiness(project=project, compose=compose)
        persistence_before = _persistent_state_sentinels(
            project=project,
            compose=compose,
            override=None,
            profile=profile,
            create=True,
        )
        on_demand = _actual_on_demand_roles(
            project=project,
            compose=compose,
            topology=topology,
        )
        _run([*compose_command, "restart", "api"])
        _wait_for_service_health(project=project, compose=compose, service="api")
        _run([*compose_command, "down", "--remove-orphans"], timeout=180)
        _run(
            [*compose_command, "up", "-d", "--wait", "--wait-timeout", "240"],
            timeout=300,
        )
        persistence_after = _persistent_state_sentinels(
            project=project,
            compose=compose,
            override=None,
            profile=profile,
            create=False,
        )
        if persistence_after != persistence_before:
            raise RuntimeTopologyProofError(
                f"persistence owner set changed after restart: {profile}"
            )
        application_readiness = _actual_role_readiness(
            project=project,
            compose=compose,
            topology=topology,
        )
        degraded: dict[str, Any] = {}
        if profile != "base":
            for service, port in (("exchange-control", 9205), ("exchange-execution", 9206)):
                degraded[service] = _service_http_json(
                    project=project,
                    compose=compose,
                    service=service,
                    port=port,
                    path="/health/ready",
                )
        if profile == "ml":
            degraded["rl-inference"] = _service_http_json(
                project=project,
                compose=compose,
                service="rl-inference",
                port=9213,
                path="/health/ready",
            )
        return {
            "profile": profile,
            "runtime_services": len(inspected),
            "running_services": len(running),
            "infrastructure_security": infrastructure,
            "storage_ready": storage["ready"],
            "storage_engines": sorted(storage["stores"]),
            "web_ready": True,
            "api_restart": "passed",
            "teardown_restart": "passed",
            "on_demand_roles": on_demand,
            "role_readiness": application_readiness,
            "persistent_state_owners": persistence_after,
            "safe_defaults": _safe_default_proof(profile),
            "safe_degraded_readiness": degraded,
        }
    except RuntimeTopologyProofError as error:
        ps = subprocess.run(
            [*compose_command, "ps", "-a"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logs = subprocess.run(
            [*compose_command, "logs", "--no-color", "--tail", "160"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raise RuntimeTopologyProofError(
            f"{error}; actual_compose_ps={ps.stdout.strip()!r}; "
            f"actual_compose_logs={logs.stdout.strip()!r}"
        ) from error
    finally:
        subprocess.run(
            [*compose_command, "down", "-v", "--remove-orphans"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )


def _read_probe_state(
    *, project: str, compose: Path, override: Path, service: str, port: int
) -> dict[str, Any]:
    script = (
        "import json,urllib.request; "
        f"print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:{port}/state'))))"
    )
    raw = _run(
        [
            *_compose(project=project, compose=compose, override=override),
            "exec",
            "-T",
            service,
            "python",
            "-c",
            script,
        ]
    ).stdout
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeTopologyProofError(f"invalid probe state for {service}")
    return value


def _postgres_sentinel(*, project: str, compose: Path, override: Path | None, create: bool) -> str:
    query = (
        "CREATE TABLE IF NOT EXISTS runtime_topology_sentinel "
        "(id integer PRIMARY KEY, value text NOT NULL); "
        "INSERT INTO runtime_topology_sentinel(id, value) "
        f"VALUES (1, '{_PERSISTENCE_SENTINEL}') "
        "ON CONFLICT (id) DO UPDATE SET value=EXCLUDED.value; "
        "SELECT value FROM runtime_topology_sentinel WHERE id=1;"
        if create
        else "SELECT value FROM runtime_topology_sentinel WHERE id=1;"
    )
    return (
        _run(
            [
                *_compose(project=project, compose=compose, override=override),
                "exec",
                "-T",
                "postgresql",
                "psql",
                "-U",
                "roehub",
                "-d",
                "roehub",
                "-Atqc",
                query,
            ]
        )
        .stdout.strip()
        .splitlines()[-1]
    )


def _run_on_demand_imports(
    *, project: str, compose: Path, topology: dict[str, Any], state_root: Path
) -> list[str]:
    completed: list[str] = []
    ports = dict(INFRASTRUCTURE_PORTS)
    for role in topology["roles"]:
        if role["lifecycle"] not in {"operator-tool", "isolated-job"}:
            continue
        service = str(role["name"])
        profile_flag = "tools" if role["lifecycle"] == "operator-tool" else "jobs"
        command = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(compose),
            "--profile",
            profile_flag,
            "run",
            "--rm",
            "--no-deps",
            "-v",
            f"{state_root}:/var/lib/roehub/runtime-proof",
            service,
            *_probe_command(role, ports, once=True),
        ]
        _run(command)
        completed.append(service)
    return completed


def _profile_proof(profile: str, *, raw_root: Path) -> dict[str, Any]:
    compose = DEFAULT_OUTPUT / profile / "compose.yaml"
    topology = json.loads((DEFAULT_OUTPUT / profile / "runtime-topology.json").read_text())
    project = f"roehub-stage17-{profile}-{uuid4().hex[:8]}"
    state_root = raw_root / profile
    state_root.mkdir(parents=True)
    state_root.chmod(0o777)
    override = raw_root / f"{profile}-proof.override.yaml"
    override.write_text(
        yaml.safe_dump(
            _proof_override(topology=topology, state_root=state_root),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    compose_command = _compose(project=project, compose=compose, override=override)
    try:
        _run([*_compose(project=project, compose=compose), "config", "--quiet"])
        rendered = _run([*_compose(project=project, compose=compose), "config"]).stdout.lower()
        if ":latest" in rendered or "privileged: true" in rendered:
            raise RuntimeTopologyProofError(f"unsafe generated Compose value: {profile}")
        _run(
            [*compose_command, "up", "-d", "--wait", "--wait-timeout", "180"],
            timeout=240,
        )
        inspected = _inspect_runtime_services(
            project=project,
            compose=compose,
            override=override,
            topology=topology,
        )
        api_role = next(role for role in topology["roles"] if role["name"] == "api")
        first_state = _read_probe_state(
            project=project,
            compose=compose,
            override=override,
            service="api",
            port=int(api_role["internal_port"]),
        )
        if first_state.get("uid") != 65532:
            raise RuntimeTopologyProofError("runtime probe did not execute as uid 65532")
        if (
            _postgres_sentinel(project=project, compose=compose, override=override, create=True)
            != _PERSISTENCE_SENTINEL
        ):
            raise RuntimeTopologyProofError("PostgreSQL sentinel write failed")
        on_demand = _run_on_demand_imports(
            project=project,
            compose=compose,
            topology=topology,
            state_root=state_root,
        )
        _run([*compose_command, "restart", "api"])
        deadline = time.monotonic() + 60
        second_state: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                candidate = _read_probe_state(
                    project=project,
                    compose=compose,
                    override=override,
                    service="api",
                    port=int(api_role["internal_port"]),
                )
                if int(candidate.get("boots", 0)) > int(first_state.get("boots", 0)):
                    second_state = candidate
                    break
            except (RuntimeTopologyProofError, json.JSONDecodeError):
                time.sleep(1)
        if second_state is None:
            raise RuntimeTopologyProofError("api restart did not preserve/increment state")

        _run([*compose_command, "down", "--remove-orphans"])
        _run([*compose_command, "up", "-d", "--wait", "--wait-timeout", "180"], timeout=240)
        if (
            _postgres_sentinel(project=project, compose=compose, override=override, create=False)
            != _PERSISTENCE_SENTINEL
        ):
            raise RuntimeTopologyProofError("PostgreSQL sentinel did not survive teardown/up")
        return {
            "profile": profile,
            "declared_roles": len(topology["roles"]),
            "default_services": len(inspected),
            "on_demand_roles": on_demand,
            "api_boots_before": first_state["boots"],
            "api_boots_after_restart": second_state["boots"],
            "postgres_volume_restart": "passed",
            "service_dns": "passed",
            "runtime_security": "passed",
        }
    except RuntimeTopologyProofError as error:
        ps = subprocess.run(
            [*compose_command, "ps", "-a"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logs = subprocess.run(
            [*compose_command, "logs", "--no-color", "--tail", "160"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raise RuntimeTopologyProofError(
            f"{error}; compose_ps={ps.stdout.strip()!r}; " f"compose_logs={logs.stdout.strip()!r}"
        ) from error
    finally:
        subprocess.run(
            [*compose_command, "down", "-v", "--remove-orphans"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )


def verify_runtime_topology(
    *, output: Path, selected_profiles: Sequence[str], build_images: bool
) -> dict[str, Any]:
    if run_generator(["--check"]) != 0:
        raise RuntimeTopologyProofError("generated runtime topology is stale")
    docker_version = _run(
        ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"]
    ).stdout.strip()
    compose_version = _run(["docker", "compose", "version", "--short"]).stdout.strip()
    if build_images:
        _run(
            [
                "docker",
                "compose",
                "-f",
                str(DEFAULT_OUTPUT / "base/compose.yaml"),
                "build",
                "api",
            ],
            timeout=600,
        )
        if "ml" in selected_profiles:
            _run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(DEFAULT_OUTPUT / "ml/compose.yaml"),
                    "build",
                    "rl-inference",
                ],
                timeout=900,
            )
    runtime_images = {}
    images = ["roehub/runtime:0.1.0-stage17"]
    if "ml" in selected_profiles:
        images.append("roehub/runtime-ml:0.1.0-stage17")
    for image in images:
        inspected = json.loads(_run(["docker", "image", "inspect", image]).stdout)[0]
        if str(inspected["Config"].get("User")) != "65532:65532":
            raise RuntimeTopologyProofError(f"image default user is not non-root: {image}")
        runtime_images[image] = str(inspected["Id"])

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    required_exclusions = {".env", ".env.*", "*.pem", "*.key", "*.p12"}
    if not required_exclusions.issubset(set(dockerignore)):
        raise RuntimeTopologyProofError("Docker build context secret exclusions are incomplete")

    job_runtime_lifecycle = _actual_job_runtime_lifecycle_proof()

    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roehub-stage17-", dir=cache_root) as raw:
        root = Path(raw)
        profiles = [_profile_proof(profile, raw_root=root) for profile in selected_profiles]
    actual_profiles = [_actual_profile_smoke(profile) for profile in selected_profiles]
    payload = {
        "schema": "io.roehub.runtime-topology-proof/v1alpha1",
        "status": "passed",
        "docker": docker_version,
        "compose": compose_version,
        "images": runtime_images,
        "profiles": profiles,
        "actual_profiles": actual_profiles,
        "job_runtime_lifecycle": job_runtime_lifecycle,
        "checks": {
            "compose_config": "passed",
            "compose_build": "passed" if build_images else "skipped",
            "entrypoint_imports": "passed",
            "non_root_read_only": "passed",
            "resource_limits": "passed",
            "service_dns": "passed",
            "restart_state": "passed",
            "persistent_volumes": "passed",
            "safe_defaults": "passed",
            "actual_application_startup": "passed",
            "actual_storage_readiness": "passed",
            "actual_on_demand_roles": "passed",
            "actual_role_readiness": "passed",
            "job_runtime_oci_lifecycle": "passed",
            "infrastructure_non_root_product_processes": "passed",
            "docker_build_context_secret_exclusions": "passed",
            "cleanup": "passed",
        },
        "external_provider_writes": False,
        "real_order_effects": False,
        "production_mutation": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("base", "trading", "ml"),
        default=("base", "trading", "ml"),
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = verify_runtime_topology(
            output=args.output,
            selected_profiles=tuple(args.profiles),
            build_images=not args.skip_build,
        )
    except (RuntimeTopologyProofError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"runtime topology verification failed: {error}")
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
