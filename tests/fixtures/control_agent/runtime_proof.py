from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from apps.api.control_agent_client import ApiControlAgentClient
from apps.control_agent.auth import read_private_credential
from apps.control_agent.docker_backend import DockerComposeControlBackend
from apps.worker.job_runtime.control_agent_client import ControlAgentJobUnixClient
from tests.fixtures.control_agent.socket_bridge import (
    start_unix_to_tcp_bridge,
    stop_bridge,
)
from trading.contexts.operations import (
    ControlOperationError,
    ControlOperationService,
    OperationAction,
    OperationRequest,
    OperationState,
)
from trading.contexts.operations.adapters import (
    AppendOnlyOperationJournal,
    ControlAgentUnixClient,
)

ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = ROOT / "configs/installation/generated/base"
TRUSTED_RELEASE = ROOT / "tools/release/release-metadata.json"
EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/18-control-agent-runtime-proof.json"
)


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    timeout: float = 360.0,
    input_text: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        input=input_text,
        env=dict(environ) if environ is not None else None,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with code {completed.returncode}: {command[0]}"
        )
    return completed


def _compose_command(project: str, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(PROFILE_ROOT / "compose.yaml"),
        *arguments,
    ]


def _prove_postgres_api_audit(
    *,
    project: str,
    socket_path: Path,
    api_identity_file: Path,
) -> dict[str, object]:
    migration_sql = (
        ROOT / "migrations/postgres/0021_control_operation_audit_v1.sql"
    ).read_text(encoding="utf-8")
    _run(
        _compose_command(
            project,
            "exec",
            "-T",
            "postgresql",
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-f",
            "/dev/stdin",
        ),
        input_text=migration_sql,
    )
    role = "control_audit_probe"
    role_sql = f"""
        DROP ROLE IF EXISTS {role};
        CREATE ROLE {role} LOGIN;
        GRANT CONNECT ON DATABASE roehub TO {role};
        GRANT USAGE ON SCHEMA public TO {role};
        GRANT SELECT, INSERT ON control_operation_audit_events TO {role};
        GRANT SELECT, UPDATE ON control_operation_audit_cursor TO {role};
    """
    _run(
        _compose_command(
            project,
            "exec",
            "-T",
            "postgresql",
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-f",
            "/dev/stdin",
        ),
        input_text=role_sql,
    )
    hba_marker = "roehub-stage18-control-audit"
    _run(
        _compose_command(
            project,
            "exec",
            "-T",
            "--user",
            "postgres",
            "postgresql",
            "sh",
            "-ec",
            (
                "sed -i '1i host roehub control_audit_probe 0.0.0.0/0 trust "
                f"# {hba_marker}' $PGDATA/pg_hba.conf; "
                "pg_ctl reload -D $PGDATA"
            ),
        )
    )
    bridge_server, bridge_thread, bridge_port = start_unix_to_tcp_bridge(socket_path)
    try:
        completed = _run(
            _compose_command(
                project,
                "run",
                "--rm",
                "--no-deps",
                "--user",
                "0",
                "-v",
                f"{ROOT}:/workspace:ro",
                "-v",
                f"{api_identity_file}:/run/roehub-control-agent-api.identity:ro",
                "-e",
                "PYTHONPATH=/workspace/src:/workspace",
                "-e",
                "IDENTITY_FAIL_FAST=false",
                "-e",
                "IDENTITY_EXCHANGE_KEYS_KEK_B64_FILE=",
                "-e",
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL=",
                "-e",
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN_FILE=",
                "-e",
                f"ROEHUB_CONTROL_AGENT_BRIDGE=host.docker.internal:{bridge_port}",
                "-e",
                "ROEHUB_CONTROL_AGENT_SOCKET=/tmp/roehub-control-agent-bridge.sock",
                "-e",
                (
                    "ROEHUB_CONTROL_AGENT_API_IDENTITY_FILE="
                    "/run/roehub-control-agent-api.identity"
                ),
                "-e",
                (
                    "ROEHUB_STORAGE_POSTGRES_DSN=host=postgresql port=5432 "
                    f"dbname=roehub user={role}"
                ),
                "--entrypoint",
                "python",
                "api",
                "/workspace/tests/fixtures/control_agent/api_audit_probe.py",
            ),
            timeout=120,
            check=False,
        )
    finally:
        stop_bridge(bridge_server, bridge_thread)
        _run(
            _compose_command(
                project,
                "exec",
                "-T",
                "--user",
                "postgres",
                "postgresql",
                "sh",
                "-ec",
                (
                    f"sed -i '/{hba_marker}/d' $PGDATA/pg_hba.conf; "
                    "pg_ctl reload -D $PGDATA"
                ),
            ),
            check=False,
            timeout=30,
        )
        _run(
            _compose_command(
                project,
                "exec",
                "-T",
                "postgresql",
                "psql",
                "-U",
                "roehub",
                "-d",
                "roehub",
                "-c",
                f"DROP ROLE IF EXISTS {role}",
            ),
            check=False,
            timeout=30,
        )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout)[-1200:]
        raise RuntimeError(f"API PostgreSQL audit probe failed: {diagnostic}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError("API PostgreSQL audit probe returned invalid evidence")
    return payload


def _write_private(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.write("\n")


def _cli(
    *,
    socket_path: Path,
    identity_file: Path,
    args: Sequence[str],
) -> dict[str, object]:
    completed = _run(
        [
            "uv",
            "run",
            "roehubctl",
            "--socket",
            str(socket_path),
            "--identity-file",
            str(identity_file),
            *args,
        ]
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("roehubctl returned an invalid response")
    return payload


def _wait_for_socket(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if path.is_socket():
            return
        if process.poll() is not None:
            raise RuntimeError("control-agent exited before opening its socket")
        time.sleep(0.05)
    raise RuntimeError("control-agent socket did not become ready")


def _rebind_compose_hash(profile_root: Path) -> None:
    generation_path = profile_root / "generation-manifest.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["outputs"]["compose.yaml"]["sha256"] = hashlib.sha256(
        (profile_root / "compose.yaml").read_bytes()
    ).hexdigest()
    generation_path.write_text(json.dumps(generation), encoding="utf-8")


def _prove_policy_rejections(root: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for mutation in ("image", "mount", "environment"):
        profile = root / f"tamper-{mutation}"
        shutil.copytree(PROFILE_ROOT, profile)
        compose_path = profile / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        api = compose["services"]["api"]
        if mutation == "image":
            api["image"] = "invalid.local/runtime:latest"
        elif mutation == "mount":
            api["volumes"].append("/tmp:/host")
        else:
            api["environment"]["UNDECLARED_OVERRIDE"] = "enabled"
        compose_path.write_text(yaml.safe_dump(compose, sort_keys=True), encoding="utf-8")
        _rebind_compose_hash(profile)
        try:
            DockerComposeControlBackend(
                profile_root=profile,
                project="roehub-rejected",
                trusted_release_manifest=TRUSTED_RELEASE,
            )
        except ControlOperationError as error:
            outcomes[mutation] = error.code
        else:
            raise RuntimeError(f"control policy accepted {mutation} override")
    return outcomes


class _AuditSink:
    def __init__(self) -> None:
        self.events: dict[str, Mapping[str, object]] = {}

    def append_control_event(
        self, *, entry_hash: str, payload: Mapping[str, object]
    ) -> None:
        self.events.setdefault(entry_hash, payload)

    def current_sequence(self) -> int:
        return 0


def _crash_after_effect(_request: OperationRequest, _result: object) -> NoReturn:
    raise SystemExit(70)


def run() -> dict[str, object]:
    project = f"roehub-stage18-{uuid4().hex[:10]}"
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roehub-stage18-", dir=cache_root) as raw:
        root = Path(raw)
        socket_path = root / "control-agent.sock"
        job_socket_path = root / "job-control.sock"
        journal_path = root / "operations.jsonl"
        api_identity_file = root / "api.identity"
        owner_identity_file = root / "owner.identity"
        job_identity_file = root / "job.identity"
        _write_private(api_identity_file, secrets.token_urlsafe(48))
        _write_private(owner_identity_file, secrets.token_urlsafe(48))
        _write_private(job_identity_file, secrets.token_urlsafe(48))
        release_state_path = journal_path.with_suffix(
            journal_path.suffix + ".release-state"
        )
        release_state_path.write_text(
            json.dumps(
                {
                    "schema": "io.roehub.installed-release/v1alpha1",
                    "version": "0.1.1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        release_state_path.chmod(0o600)
        agent = subprocess.Popen(
            [
                "uv",
                "run",
                "roehub-control-agent",
                "--profile-root",
                str(PROFILE_ROOT),
                "--trusted-release-manifest",
                str(TRUSTED_RELEASE),
                "--profile",
                "base",
                "--project",
                project,
                "--socket",
                str(socket_path),
                "--job-socket",
                str(job_socket_path),
                "--journal",
                str(journal_path),
                "--api-token-file",
                str(api_identity_file),
                "--owner-token-file",
                str(owner_identity_file),
                "--job-token-file",
                str(job_identity_file),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for_socket(socket_path, agent)
            _wait_for_socket(job_socket_path, agent)
            job_client = ControlAgentJobUnixClient(
                socket_path=job_socket_path,
                identity_key=read_private_credential(job_identity_file),
            )
            job_client.ping()
            job_probe = job_client.run(
                (
                    "docker",
                    "container",
                    "inspect",
                    "roehub-job-" + "f" * 32,
                ),
                environ={},
                timeout_seconds=10,
            )
            start = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=[
                    "start",
                    "--profile",
                    "base",
                    "--operation-id",
                    "00000000-0000-4000-8000-000000000180",
                ],
            )
            stop_args = [
                "stop",
                "--profile",
                "base",
                "--service",
                "web",
                "--service",
                "api",
                "--service",
                "postgresql",
                "--operation-id",
                "00000000-0000-4000-8000-000000000181",
            ]
            stopped = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=stop_args,
            )
            replayed = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=stop_args,
            )
            doctor = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=["doctor", "--profile", "base"],
            )
            rollback = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=[
                    "rollback",
                    "--profile",
                    "base",
                    "--release-version",
                    "0.1.0",
                    "--operation-id",
                    "00000000-0000-4000-8000-000000000182",
                ],
            )
            _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=[
                    "stop",
                    "--profile",
                    "base",
                    "--service",
                    "web",
                    "--service",
                    "api",
                    "--service",
                    "postgresql",
                    "--operation-id",
                    "00000000-0000-4000-8000-000000000183",
                ],
            )
            recovered = _cli(
                socket_path=socket_path,
                identity_file=owner_identity_file,
                args=[
                    "recover",
                    "--profile",
                    "base",
                    "--operation-id",
                    "00000000-0000-4000-8000-000000000184",
                ],
            )

            crash_journal = AppendOnlyOperationJournal(path=root / "crash.jsonl")
            backend = DockerComposeControlBackend(
                profile_root=PROFILE_ROOT,
                project=project,
                trusted_release_manifest=TRUSTED_RELEASE,
                effect_receipt_dir=root / "crash-effects",
            )
            crash_request = OperationRequest(
                operation_id=UUID("00000000-0000-4000-8000-000000000185"),
                action=OperationAction.RESTART,
                profile="base",
                services=("redis",),
            )
            try:
                crash_submit_result = ControlOperationService(
                    backend=backend,
                    journal=crash_journal,
                    after_effect=_crash_after_effect,
                ).submit(crash_request)
            except SystemExit:
                pass
            else:
                raise RuntimeError(
                    "crash injection did not interrupt operation finalization: "
                    f"{crash_submit_result.detail_code}"
                )
            crash_recovery = ControlOperationService(
                backend=backend,
                journal=crash_journal,
            ).reconcile(crash_request.operation_id)

            invalid = OperationRequest(
                operation_id=uuid4(), action=OperationAction.INSPECT
            ).model_dump(mode="json", by_alias=True)
            try:
                OperationRequest.model_validate(
                    {**invalid, "command": ["sh", "-c", "id"]}
                )
            except ValidationError:
                shell_rejection = "passed"
            else:
                raise RuntimeError("arbitrary shell field was accepted")

            api_client = ApiControlAgentClient(
                transport=ControlAgentUnixClient(
                    socket_path=socket_path,
                    identity="api",
                    identity_key=read_private_credential(api_identity_file),
                )
            )
            sink = _AuditSink()
            audit_cursor = api_client.reconcile_audit(sink=sink)
            postgres_audit = _prove_postgres_api_audit(
                project=project,
                socket_path=socket_path,
                api_identity_file=api_identity_file,
            )
            policy_rejections = _prove_policy_rejections(root)
            journal_entries = AppendOnlyOperationJournal(path=journal_path).entries()
            if doctor.get("detail_code") != "topology.degraded":
                raise RuntimeError("doctor did not report the degraded topology")
            doctor_services = doctor.get("active_services")
            if not isinstance(doctor_services, list):
                raise RuntimeError("doctor active service list is invalid")
            if "keycloak" in doctor_services:
                raise RuntimeError("unexpected Keycloak service was reported")
            if stopped.get("journal_sequence") != replayed.get("journal_sequence"):
                raise RuntimeError("idempotent replay appended a second effect")
            if crash_recovery.state != OperationState.SUCCEEDED:
                raise RuntimeError("crash recovery remained unknown")
            if not sink.events or audit_cursor != len(journal_entries):
                raise RuntimeError("API audit reconciliation missed local journal events")
            if postgres_audit.get("events") != len(journal_entries):
                raise RuntimeError("PostgreSQL API audit reconciliation missed journal events")
            payload: dict[str, object] = {
                "schema": "io.roehub.control-agent-runtime-proof/v1alpha1",
                "status": "passed",
                "project": project,
                "checks": {
                    "typed_start": start.get("state"),
                    "degraded_doctor": doctor.get("detail_code"),
                    "rollback": rollback.get("detail_code"),
                    "recover": recovered.get("detail_code"),
                    "operation_id_idempotency": "passed",
                    "crash_unknown_reconciliation": crash_recovery.detail_code,
                    "image_mount_environment_rejection": policy_rejections,
                    "arbitrary_shell_rejection": shell_rejection,
                    "local_journal_without_postgresql": "passed",
                    "api_audit_reconciliation": "passed",
                    "api_postgresql_audit_reconciliation": postgres_audit,
                    "keycloak_absence_typed": "not_installed",
                    "job_control_rpc": (
                        "passed" if job_probe.returncode != 0 else "unexpected_container"
                    ),
                },
                "journal_entries": len(journal_entries),
                "audit_events": len(sink.events),
                "production_mutation": False,
                "external_provider_writes": False,
                "real_order_effects": False,
                "secrets_recorded": False,
            }
            return payload
        finally:
            agent.terminate()
            try:
                agent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                agent.kill()
                agent.wait(timeout=5)
            _run(
                [
                    "docker",
                    "compose",
                    "-p",
                    project,
                    "-f",
                    str(PROFILE_ROOT / "compose.yaml"),
                    "down",
                    "-v",
                    "--remove-orphans",
                ],
                check=False,
                timeout=180,
            )


def main() -> int:
    payload = run()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
