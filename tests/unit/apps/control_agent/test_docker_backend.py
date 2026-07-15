from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from apps.control_agent.docker_backend import (
    DockerComposeControlBackend,
    _ReleaseStateStore,
)
from trading.contexts.operations import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
    OperationState,
)

ROOT = Path(__file__).resolve().parents[4]
BASE_PROFILE = ROOT / "configs/installation/generated/base"
TRUSTED_RELEASE = ROOT / "tools/release/release-metadata.json"


def _copy_profile(tmp_path: Path) -> Path:
    destination = tmp_path / "base"
    shutil.copytree(BASE_PROFILE, destination)
    return destination


def _rebind_compose_hash(profile_root: Path) -> None:
    generation_path = profile_root / "generation-manifest.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["outputs"]["compose.yaml"]["sha256"] = hashlib.sha256(
        (profile_root / "compose.yaml").read_bytes()
    ).hexdigest()
    generation_path.write_text(json.dumps(generation), encoding="utf-8")


@pytest.mark.parametrize("mutation", ["image", "mount", "environment"])
def test_policy_rejects_compose_runtime_override(tmp_path: Path, mutation: str) -> None:
    profile_root = _copy_profile(tmp_path)
    compose_path = profile_root / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    if mutation == "image":
        api["image"] = "attacker.invalid/runtime:latest"
    elif mutation == "mount":
        api["volumes"].append("/tmp:/host")
    else:
        api["environment"]["UNDECLARED_RUNTIME_OVERRIDE"] = "enabled"
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=True), encoding="utf-8")
    _rebind_compose_hash(profile_root)

    with pytest.raises(ControlOperationError, match="control_agent"):
        DockerComposeControlBackend(
            profile_root=profile_root,
            project="roehub-test",
            trusted_release_manifest=TRUSTED_RELEASE,
        )


def test_backend_generates_only_fixed_compose_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    active = "api\nopenbao\nplugin-gateway\npostgresql\nredis\nweb\n"
    policy = json.loads((BASE_PROFILE / "control-policy.json").read_text(encoding="utf-8"))
    image_ids = {
        row["image"]: row["release_reference"].rsplit("@", 1)[1]
        for row in policy["services"].values()
    }

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            output = (
                json.dumps([command[3]])
                if "@sha256:" in command[3]
                else image_ids[command[3]]
            )
            return subprocess.CompletedProcess(command, 0, output + "\n", "")
        if command[1:3] == ["container", "inspect"]:
            payload = [
                {
                    "Id": identifier,
                    "Image": "sha256:" + "1" * 64,
                    "Config": {"Labels": {"com.docker.compose.service": identifier}},
                    "State": {"StartedAt": "2026-07-13T00:00:00Z", "Running": True},
                }
                for identifier in command[3:]
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "ps" in command and "-q" in command:
            return subprocess.CompletedProcess(command, 0, active, "")
        if "ps" in command and "--services" in command:
            return subprocess.CompletedProcess(command, 0, active, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("apps.control_agent.docker_backend.subprocess.run", fake_run)
    backend = DockerComposeControlBackend(
        profile_root=BASE_PROFILE,
        project="roehub-stage18-unit",
        trusted_release_manifest=TRUSTED_RELEASE,
        effect_receipt_dir=tmp_path / "effects",
    )
    result = backend.execute(
        OperationRequest(
            operation_id=UUID("00000000-0000-4000-8000-000000000018"),
            action=OperationAction.RECOVER,
            profile="base",
        )
    )

    assert result.state == OperationState.SUCCEEDED
    effect = next(command for command in commands if "up" in command)
    assert effect == [
        "docker",
        "compose",
        "-p",
        "roehub-stage18-unit",
        "--project-directory",
        str(BASE_PROFILE),
        "-f",
        "-",
        "up",
        "-d",
        "--no-build",
        "--pull",
        "never",
        "--wait",
        "--wait-timeout",
        "180",
    ]


def test_backend_reloads_bundle_and_rejects_post_start_tamper(tmp_path: Path) -> None:
    profile_root = _copy_profile(tmp_path)
    backend = DockerComposeControlBackend(
        profile_root=profile_root,
        project="roehub-stage18-toctou",
        trusted_release_manifest=TRUSTED_RELEASE,
    )
    compose_path = profile_root / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    compose["services"]["api"]["command"] = ["sh", "-c", "id"]
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=True), encoding="utf-8")

    with pytest.raises(ControlOperationError, match="control_agent.policy_hash_mismatch"):
        backend.execute(
            OperationRequest(
                operation_id=UUID("00000000-0000-4000-8000-000000000019"),
                action=OperationAction.INSPECT,
                profile="base",
            )
        )


def test_reconcile_stays_unknown_without_effect_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "ps" in command and "--services" in command:
            return subprocess.CompletedProcess(command, 0, "api\n", "")
        if "ps" in command and "-q" in command:
            return subprocess.CompletedProcess(command, 0, "api-id\n", "")
        if command[1:3] == ["container", "inspect"]:
            payload = [
                {
                    "Id": "api-id",
                    "Image": "sha256:" + "1" * 64,
                    "Config": {"Labels": {"com.docker.compose.service": "api"}},
                    "State": {
                        "StartedAt": "2026-07-13T00:00:00Z",
                        "Running": True,
                        "Health": {"Status": "healthy"},
                    },
                }
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("apps.control_agent.docker_backend.subprocess.run", fake_run)
    backend = DockerComposeControlBackend(
        profile_root=BASE_PROFILE,
        project="roehub-stage18-reconcile",
        trusted_release_manifest=TRUSTED_RELEASE,
        effect_receipt_dir=tmp_path / "effects",
    )
    request = OperationRequest(
        operation_id=UUID("00000000-0000-4000-8000-000000000020"),
        action=OperationAction.RESTART,
        profile="base",
        services=("api",),
    )

    result = backend.reconcile(request)

    assert result.state == OperationState.UNKNOWN
    assert result.detail_code == "operation.effect_unknown"


def test_restart_rejects_stateful_service_before_docker_effect(tmp_path: Path) -> None:
    backend = DockerComposeControlBackend(
        profile_root=BASE_PROFILE,
        project="roehub-stage20-restart-policy",
        trusted_release_manifest=TRUSTED_RELEASE,
        effect_receipt_dir=tmp_path / "effects",
    )

    with pytest.raises(ControlOperationError, match="control_agent.restart_rejected"):
        backend.execute(
            OperationRequest(
                operation_id=UUID("00000000-0000-4000-8000-000000000021"),
                action=OperationAction.RESTART,
                profile="base",
                services=("postgresql",),
            )
        )


def test_restart_waits_for_compose_health_before_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    policy = json.loads((BASE_PROFILE / "control-policy.json").read_text(encoding="utf-8"))
    image_ids = {
        row["image"]: row["release_reference"].rsplit("@", 1)[1]
        for row in policy["services"].values()
    }

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            output = (
                json.dumps([command[3]])
                if "@sha256:" in command[3]
                else image_ids[command[3]]
            )
            return subprocess.CompletedProcess(command, 0, output + "\n", "")
        if "ps" in command and "--services" in command:
            return subprocess.CompletedProcess(command, 0, "api\n", "")
        if "ps" in command and "-q" in command:
            return subprocess.CompletedProcess(command, 0, "api-id\n", "")
        if command[1:3] == ["container", "inspect"]:
            payload = [
                {
                    "Id": "api-id",
                    "Image": "sha256:" + "1" * 64,
                    "Config": {"Labels": {"com.docker.compose.service": "api"}},
                    "State": {
                        "StartedAt": "2026-07-13T00:00:00Z",
                        "Running": True,
                        "Health": {"Status": "healthy"},
                    },
                }
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("apps.control_agent.docker_backend.subprocess.run", fake_run)
    backend = DockerComposeControlBackend(
        profile_root=BASE_PROFILE,
        project="roehub-stage20-restart-wait",
        trusted_release_manifest=TRUSTED_RELEASE,
        effect_receipt_dir=tmp_path / "effects",
    )

    result = backend.execute(
        OperationRequest(
            operation_id=UUID("00000000-0000-4000-8000-000000000022"),
            action=OperationAction.RESTART,
            profile="base",
            services=("api",),
        )
    )

    assert result.state == OperationState.SUCCEEDED
    restart_index = next(index for index, command in enumerate(commands) if "restart" in command)
    wait_index = next(
        index
        for index, command in enumerate(commands)
        if "up" in command and "--wait" in command
    )
    assert restart_index < wait_index
    assert commands[wait_index][-3:] == ["--wait-timeout", "180", "api"]


def test_release_state_enforces_install_update_and_rollback_direction(
    tmp_path: Path,
) -> None:
    store = _ReleaseStateStore(tmp_path / "release-state.json")

    assert store.validate_transition(action=OperationAction.INSTALL, target="0.1.0") is None
    store.write("0.1.0")
    assert (
        store.validate_transition(action=OperationAction.UPDATE, target="0.2.0")
        == "0.1.0"
    )
    assert (
        store.validate_transition(action=OperationAction.ROLLBACK, target="0.0.9")
        == "0.1.0"
    )
    with pytest.raises(ControlOperationError, match="release_transition_invalid"):
        store.validate_transition(action=OperationAction.UPDATE, target="0.0.9")


def test_trusted_release_manifest_is_an_external_hash_anchor(tmp_path: Path) -> None:
    profile_root = _copy_profile(tmp_path)
    trusted = tmp_path / "release-metadata.json"
    trusted.write_bytes(TRUSTED_RELEASE.read_bytes() + b"\n")

    with pytest.raises(ControlOperationError, match="release_manifest_unbound"):
        DockerComposeControlBackend(
            profile_root=profile_root,
            project="roehub-stage18-anchor",
            trusted_release_manifest=trusted,
        )
