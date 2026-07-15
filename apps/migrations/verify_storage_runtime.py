"""Real Docker proof for greenfield storage bootstrap, recovery, and readiness."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, cast


class RuntimeProofError(RuntimeError):
    """Raised when disposable storage evidence is incomplete."""


_SENSITIVE_ENV_KEYS = (
    "ROEHUB_STORAGE_POSTGRES_PASSWORD",
    "ROEHUB_STORAGE_CLICKHOUSE_PASSWORD",
    "ROEHUB_STORAGE_REDIS_PASSWORD",
    "ROEHUB_STORAGE_POSTGRES_DSN",
    "ROEHUB_STORAGE_CLICKHOUSE_DSN",
    "ROEHUB_STORAGE_REDIS_URL",
)


def _redact(value: str, *, environ: Mapping[str, str]) -> str:
    redacted = value
    for key in _SENSITIVE_ENV_KEYS:
        material = environ.get(key, "")
        if material:
            redacted = redacted.replace(material, "<redacted>")
    return redacted


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environ: Mapping[str, str],
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environ),
        text=True,
        capture_output=True,
        check=False,
    )
    if (result.returncode == 0) != expect_success:
        stdout = _redact(result.stdout[-4000:], environ=environ)
        stderr = _redact(result.stderr[-4000:], environ=environ)
        raise RuntimeProofError(
            f"unexpected exit={result.returncode}; stdout={stdout!r}; stderr={stderr!r}"
        )
    return result


def _compose(project: str, path: Path, *args: str) -> list[str]:
    return ["docker", "compose", "-p", project, "-f", str(path), *args]


def _status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("storage command returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("storage command returned invalid JSON status") from error
    if not isinstance(payload, dict) or payload.get("ready") is not True:
        raise RuntimeProofError("storage status is not ready")
    return payload


def _organization_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("organization proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("organization proof returned invalid JSON status") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.organization-runtime-proof/v1alpha1"
        or payload.get("role_matrix") != "passed"
        or payload.get("last_owner") != "passed"
        or payload.get("audit_redaction") != "passed"
    ):
        raise RuntimeProofError("organization runtime proof is incomplete")
    return payload


def _local_auth_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("local auth proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("local auth proof returned invalid JSON status") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.local-auth-runtime-proof/v1alpha1"
        or payload.get("bootstrap_hash_only") != "passed"
        or payload.get("single_active_bootstrap") != "passed"
        or payload.get("passkey_counter") != "passed"
        or payload.get("recovery_replay") != "rejected"
        or payload.get("rate_limit") != "passed"
        or payload.get("audit_immutable") != "passed"
    ):
        raise RuntimeProofError("local auth runtime proof is incomplete")
    return payload


def _oidc_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("OIDC proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("OIDC proof returned invalid JSON status") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.oidc-runtime-proof/v1alpha1"
        or payload.get("invitation_provisioning") != "passed"
        or payload.get("uninvited_provisioning") != "rejected"
        or payload.get("authenticated_linking") != "passed"
        or payload.get("subject_takeover") != "rejected"
        or payload.get("audit_immutable") != "passed"
    ):
        raise RuntimeProofError("OIDC runtime proof is incomplete")
    return payload


def _research_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("research tenancy proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("research tenancy proof returned invalid JSON status") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.research-tenancy-runtime-proof/v1alpha1"
        or payload.get("server_derived_scope") != "passed"
        or payload.get("cross_organization_repository_read") != "rejected"
        or payload.get("shared_candle_parity") != "passed"
        or payload.get("authorization_overhead") != "passed"
    ):
        raise RuntimeProofError("research tenancy runtime proof is incomplete")
    return payload


def _trading_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("trading tenancy proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError("trading tenancy proof returned invalid JSON status") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.trading-tenancy-runtime-proof/v1alpha1"
        or payload.get("two_organization_paper") != "passed"
        or payload.get("cross_organization_repository_read") != "rejected"
        or payload.get("negative_authorization") != "rejected_422_no_write"
        or payload.get("client_risk_spoof") != "rejected_422_no_write"
        or payload.get("account_ownership_mismatch") != "rejected_by_server_resolver"
        or payload.get("risk_denial") != "kill_switch_closed"
        or payload.get("duplicate_intent") != "deduplicated"
        or payload.get("position_ownership") != "passed"
        or payload.get("unknown_state_reconciliation") != "matched_without_resubmit"
        or payload.get("private_stream_session") != "persisted_ready"
        or payload.get("request_observation") != "persisted_testnet_submitted"
        or payload.get("controlled_testnet_submits") != 1
        or payload.get("mainnet_attempt") != "guard_rejected_before_submit"
        or payload.get("mainnet_submits") != 0
    ):
        raise RuntimeProofError("trading tenancy runtime proof is incomplete")
    constraints = payload.get("database_constraints")
    if not isinstance(constraints, dict) or constraints != {
        "intent_connection": "execution_intents_org_connection_fk",
        "strategy_membership": "rejected",
    }:
        raise RuntimeProofError("trading tenancy database constraints are incomplete")
    return payload


def _notification_provider_status(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeProofError("notification provider proof returned no JSON status")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise RuntimeProofError(
            "notification provider proof returned invalid JSON status"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "io.roehub.notification-provider-runtime-proof/v1alpha1"
        or payload.get("two_organizations_two_instances") != "passed"
        or payload.get("delivery_idempotency") != "passed"
        or payload.get("duplicate_update") != "worker_recovered_idempotently"
        or payload.get("per_organization_secret_refs") != "passed"
        or payload.get("pre_acceptance_failure") != "retry"
        or payload.get("post_acceptance_timeout") != "unknown"
        or payload.get("cancellation") != "unknown_persisted_before_propagation"
        or payload.get("retry_after") != "persisted"
        or payload.get("bounded_backoff_jitter") != "passed"
        or payload.get("shutdown_recovery") != "unknown_without_resubmit"
        or payload.get("provider_health") != "ready_and_degraded"
        or payload.get("cross_organization_write") != "rejected"
        or payload.get("provider_secret_scope") != "rejected"
        or payload.get("critical_fallback") != "not_used"
        or payload.get("durable_cursor") != "advanced_by_worker"
        or payload.get("telegram_command_transaction") != "atomic_worker_recovery"
        or payload.get("explicit_replay") != "linked_new_delivery"
        or payload.get("command_registry_entries") != 18
    ):
        raise RuntimeProofError("notification provider runtime proof is incomplete")
    return payload


def _assert_versions(payload: dict[str, object], *, mode: str) -> None:
    expected = {
        "postgresql": [
            "identity-0001-0009",
            "alembic:20260711_0043",
            "strategy-0010",
            "organization-0011",
            "local-auth-0012",
            "oidc-provider-0013",
            "research-tenancy-0014",
            "trading-tenancy-0015",
            "notification-providers-0016",
            "extensions-plugin-platform-0017",
            "artifact-store-0018",
            "isolated-job-runtime-0019",
            "execution-gateway-safety-0020",
        ],
        "clickhouse": ["0001", "0002"],
        "redis": [],
    }
    stores = payload.get("stores")
    if not isinstance(stores, dict):
        raise RuntimeProofError("storage status has no stores")
    for name, versions in expected.items():
        store = stores.get(name)
        if not isinstance(store, dict):
            raise RuntimeProofError(f"storage status is missing {name}")
        if store.get("mode") != mode or store.get("schema_versions") != versions:
            raise RuntimeProofError(
                f"storage status mismatch for {name}: "
                f"mode={store.get('mode')!r}, schema_versions={store.get('schema_versions')!r}"
            )


def _generate_config(repo_root: Path, output: Path, *, config: Path | None) -> Path:
    command = [
        "uv",
        "run",
        "python",
        "tools/release/generate_installation_config.py",
        "--output",
        str(output),
        "--profile",
        "trading",
        "--write",
    ]
    if config is not None:
        command[4:4] = ["--config", str(config)]
    _run(command, cwd=repo_root, environ=os.environ)
    service_config = output / "trading/service-config.json"
    if not service_config.is_file():
        raise RuntimeProofError("installation generator produced no service config")
    return service_config


def _proof_environment(service_config: Path) -> dict[str, str]:
    environ = dict(os.environ)
    credential_keys = _SENSITIVE_ENV_KEYS[:3]
    credential_values = tuple(secrets.token_hex(24) for _key in credential_keys)
    environ.update(dict(zip(credential_keys, credential_values, strict=True)))
    environ["ROEHUB_STORAGE_POSTGRES_DSN"] = (
        "host=postgresql port=5432 dbname=roehub user=roehub password="
        + environ[credential_keys[0]]
    )
    environ["ROEHUB_STORAGE_CLICKHOUSE_DSN"] = (
        "http://roehub:" + environ[credential_keys[1]] + "@clickhouse:8123/default"
    )
    environ["ROEHUB_STORAGE_REDIS_URL"] = (
        "redis://:" + environ[credential_keys[2]] + "@redis:6379/0"
    )
    environ["ROEHUB_STORAGE_SERVICE_CONFIG"] = str(service_config)
    environ["ROEHUB_DISPOSABLE_STORAGE_PROOF"] = "1"
    return environ


def _run_storage(
    project: str,
    compose_path: Path,
    repo_root: Path,
    environ: Mapping[str, str],
    *command: str,
) -> dict[str, object]:
    result = _run(
        _compose(project, compose_path, "run", "--rm", "storage-migrations", *command),
        cwd=repo_root,
        environ=environ,
    )
    return _status(result.stdout)


def run_runtime_proof(repo_root: Path) -> dict[str, object]:
    """Execute the full disposable proof and remove all containers and volumes."""

    if shutil.which("docker") is None:
        raise RuntimeProofError("docker executable is unavailable")
    project = f"roehub-stage04-{secrets.token_hex(4)}"
    embedded_compose = repo_root / "infra/docker/storage-embedded.compose.yml"
    external_compose = repo_root / "infra/docker/storage-external.compose.yml"
    fault_fixture = repo_root / "tests/fixtures/storage/clickhouse-interrupted"
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cleanup_complete = False

    with tempfile.TemporaryDirectory(prefix="roehub-stage04-", dir=cache_root) as temp:
        temp_root = Path(temp)
        embedded_config = _generate_config(repo_root, temp_root / "embedded", config=None)
        external_config = _generate_config(
            repo_root,
            temp_root / "external",
            config=repo_root / "tests/fixtures/installation/roehub-external.yaml",
        )
        environ = _proof_environment(embedded_config)
        try:
            _run(
                _compose(project, embedded_compose, "config", "--quiet"),
                cwd=repo_root,
                environ=environ,
            )
            external_environment = dict(environ)
            external_environment["ROEHUB_STORAGE_SERVICE_CONFIG"] = str(external_config)
            _run(
                _compose(project, external_compose, "config", "--quiet"),
                cwd=repo_root,
                environ=external_environment,
            )
            _run(
                _compose(project, embedded_compose, "build", "storage-migrations"),
                cwd=repo_root,
                environ=environ,
            )
            _run(
                _compose(
                    project,
                    embedded_compose,
                    "up",
                    "-d",
                    "postgresql",
                    "clickhouse",
                    "redis",
                ),
                cwd=repo_root,
                environ=environ,
            )

            _run(
                _compose(
                    project,
                    embedded_compose,
                    "run",
                    "--rm",
                    "-v",
                    f"{fault_fixture}:/proof:ro",
                    "storage-migrations",
                    "bootstrap",
                    "--service-config",
                    "/etc/roehub/service-config.json",
                    "--clickhouse-manifest",
                    "/proof/manifest.json",
                ),
                cwd=repo_root,
                environ=environ,
                expect_success=False,
            )
            postgres_partial = _run(
                _compose(
                    project,
                    embedded_compose,
                    "exec",
                    "-T",
                    "postgresql",
                    "psql",
                    "-U",
                    "roehub",
                    "-d",
                    "roehub",
                    "-Atc",
                    (
                        "SELECT (SELECT version_num FROM alembic_version), "
                        "count(*) FROM roehub_storage_migrations"
                    ),
                ),
                cwd=repo_root,
                environ=environ,
            ).stdout.strip()
            if not postgres_partial.startswith("20260711_0043|"):
                raise RuntimeProofError("interrupted run lost PostgreSQL migration state")
            clickhouse_partial = _run(
                _compose(
                    project,
                    embedded_compose,
                    "exec",
                    "-T",
                    "clickhouse",
                    "sh",
                    "-ec",
                    (
                        'clickhouse-client --user roehub --password "$CLICKHOUSE_PASSWORD" '
                        '--query "SELECT '
                        "(SELECT count() FROM system.tables WHERE database = 'roehub' "
                        "AND name = 'interrupted_migration_probe'), "
                        "(SELECT count() FROM roehub.roehub_schema_migrations) "
                        'FORMAT TabSeparated"'
                    ),
                ),
                cwd=repo_root,
                environ=environ,
            ).stdout.strip()
            if clickhouse_partial != "1\t0":
                raise RuntimeProofError("interrupted ClickHouse state was not observable")

            recovered = _run_storage(project, embedded_compose, repo_root, environ)
            _assert_versions(recovered, mode="embedded")
            idempotent = _run_storage(project, embedded_compose, repo_root, environ)
            _assert_versions(idempotent, mode="embedded")
            local_auth_environment = dict(environ)
            local_auth_environment["IDENTITY_PG_DSN"] = environ["ROEHUB_STORAGE_POSTGRES_DSN"]
            bootstrap_file = temp_root / "local-auth-bootstrap"
            _run(
                _compose(
                    project,
                    embedded_compose,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python",
                    "-e",
                    "IDENTITY_PG_DSN",
                    "-v",
                    f"{temp_root}:/proof",
                    "storage-migrations",
                    "-m",
                    "apps.cli.commands.local_auth_bootstrap",
                    "--output-file",
                    "/proof/local-auth-bootstrap",
                ),
                cwd=repo_root,
                environ=local_auth_environment,
            )
            if not bootstrap_file.is_file() or bootstrap_file.stat().st_mode & 0o077:
                raise RuntimeProofError("local auth bootstrap file is missing or too permissive")
            local_auth_proof = _local_auth_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "-e",
                        "IDENTITY_PG_DSN",
                        "-v",
                        f"{temp_root}:/proof:ro",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.local_auth_runtime_probe",
                        "--bootstrap-file",
                        "/proof/local-auth-bootstrap",
                    ),
                    cwd=repo_root,
                    environ=local_auth_environment,
                ).stdout
            )
            organization_proof = _organization_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.organization_runtime_probe",
                    ),
                    cwd=repo_root,
                    environ=environ,
                ).stdout
            )
            oidc_proof = _oidc_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "-e",
                        "IDENTITY_PG_DSN",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.oidc_runtime_probe",
                    ),
                    cwd=repo_root,
                    environ=local_auth_environment,
                ).stdout
            )
            research_proof = _research_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "-e",
                        "ROEHUB_DISPOSABLE_STORAGE_PROOF",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.research_runtime_probe",
                    ),
                    cwd=repo_root,
                    environ=environ,
                ).stdout
            )
            trading_proof = _trading_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "-e",
                        "ROEHUB_DISPOSABLE_STORAGE_PROOF",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.trading_runtime_probe",
                    ),
                    cwd=repo_root,
                    environ=environ,
                ).stdout
            )
            notification_provider_proof = _notification_provider_status(
                _run(
                    _compose(
                        project,
                        embedded_compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "python",
                        "-e",
                        "ROEHUB_DISPOSABLE_STORAGE_PROOF",
                        "storage-migrations",
                        "-m",
                        "apps.migrations.notification_provider_runtime_probe",
                    ),
                    cwd=repo_root,
                    environ=environ,
                ).stdout
            )

            _run(
                _compose(
                    project,
                    embedded_compose,
                    "stop",
                    "postgresql",
                    "clickhouse",
                    "redis",
                ),
                cwd=repo_root,
                environ=environ,
            )
            _run(
                _compose(
                    project,
                    embedded_compose,
                    "up",
                    "-d",
                    "postgresql",
                    "clickhouse",
                    "redis",
                ),
                cwd=repo_root,
                environ=environ,
            )
            restarted = _run_storage(
                project,
                embedded_compose,
                repo_root,
                environ,
                "status",
                "--service-config",
                "/etc/roehub/service-config.json",
            )
            _assert_versions(restarted, mode="embedded")

            image_id = f"{project}-storage-migrations"
            _run(
                ["docker", "image", "inspect", image_id],
                cwd=repo_root,
                environ=environ,
            )
            external = _status(
                _run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        f"{project}_storage",
                        "--read-only",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--tmpfs",
                        "/tmp:rw,noexec,nosuid,size=32m",
                        "--user",
                        "65534:65534",
                        "-e",
                        "ROEHUB_STORAGE_POSTGRES_DSN",
                        "-e",
                        "ROEHUB_STORAGE_CLICKHOUSE_DSN",
                        "-e",
                        "ROEHUB_STORAGE_REDIS_URL",
                        "-v",
                        f"{external_config}:/etc/roehub/service-config.json:ro",
                        image_id,
                        "readiness",
                        "--service-config",
                        "/etc/roehub/service-config.json",
                    ],
                    cwd=repo_root,
                    environ=environ,
                ).stdout
            )
            _assert_versions(external, mode="external")
        finally:
            cleanup = _run(
                _compose(project, embedded_compose, "down", "-v", "--remove-orphans"),
                cwd=repo_root,
                environ=environ,
            )
            cleanup_complete = cleanup.returncode == 0

    docker_version = _run(
        ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"],
        cwd=repo_root,
        environ=os.environ,
    ).stdout.strip()
    return {
        "schema": "io.roehub.storage-runtime-proof/v1alpha1",
        "docker": docker_version,
        "compose": "passed",
        "fresh_bootstrap": "passed",
        "interrupted_recovery": "passed",
        "idempotent_rerun": "passed",
        "persistent_volume_restart": "passed",
        "external_readiness": "passed",
        "organization_isolation": "passed"
        if organization_proof.get("database_constraints")
        else "failed",
        "organization_constraints": sorted(
            cast(dict[str, object], organization_proof["database_constraints"])
        ),
        "organization_audit_events": organization_proof.get("audit_events", 0),
        "local_auth": local_auth_proof,
        "oidc": oidc_proof,
        "research_tenancy": research_proof,
        "trading_tenancy": trading_proof,
        "notification_providers": notification_provider_proof,
        "cleanup": "passed" if cleanup_complete else "failed",
    }


def main() -> int:
    try:
        result = run_runtime_proof(Path(__file__).resolve().parents[2])
    except RuntimeProofError as error:
        print(f"storage runtime proof failed: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
