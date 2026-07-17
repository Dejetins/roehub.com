from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from infra.openbao import verify_runtime

ROOT = Path(__file__).resolve().parents[3]


def test_runtime_image_includes_the_owner_bootstrap_module_and_policies() -> None:
    dockerfile = (ROOT / "infra" / "docker" / "Dockerfile.runtime").read_text(encoding="utf-8")

    assert "COPY infra/openbao infra/openbao" in dockerfile
    assert (
        "COPY --from=builder --chown=65532:65532 /build/infra/openbao infra/openbao"
        in dockerfile
    )


def test_embedded_openbao_is_digest_pinned_hardened_and_persistent() -> None:
    compose_path = ROOT / "infra" / "docker" / "openbao-embedded.compose.yml"
    payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    service = payload["services"]["openbao"]

    assert service["image"].endswith(
        "@sha256:8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a"
    )
    assert "-dev" not in " ".join(service["command"])
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "openbao-data:/openbao/file" in service["volumes"]
    assert "openbao-audit:/openbao/logs" in service["volumes"]
    assert payload["services"]["openbao"]["ports"] == [
        "127.0.0.1:${ROEHUB_OPENBAO_PORT:-18200}:8200"
    ]


def test_openbao_uses_raft_and_policy_boundaries_are_disjoint() -> None:
    config = (ROOT / "infra" / "openbao" / "config" / "openbao.hcl").read_text()
    policies = ROOT / "infra" / "openbao" / "policies"
    api = (policies / "roehub-api.hcl").read_text()
    identity = (policies / "roehub-identity.hcl").read_text()
    notifications = (policies / "roehub-notification-dispatcher.hcl").read_text()
    telegram_worker = (policies / "roehub-telegram-bot-worker.hcl").read_text()
    exchange = (policies / "roehub-exchange-execution.hcl").read_text()

    assert 'storage "raft"' in config
    assert "/openbao/file" in config
    assert 'path "kv/data/roehub/*"' in api and '["deny"]' in api
    assert "transit/decrypt/*" in api
    assert "roehub/oidc/*" in identity and "roehub/telegram/*" not in identity
    assert "roehub/telegram/providers/*" in notifications
    assert "roehub/telegram/recipients/*" in notifications
    assert '["read"]' in notifications and "roehub/oidc/*" not in notifications
    assert "roehub/telegram/providers/*" in telegram_worker
    assert "roehub/telegram/recipients/*" in telegram_worker
    assert '["create", "update", "read"]' in telegram_worker
    assert "transit/decrypt/roehub-exchange-credentials" in exchange


def test_every_service_policy_allows_only_self_renewal_for_token_lifecycle() -> None:
    policies = ROOT / "infra" / "openbao" / "policies"

    for policy_path in sorted(policies.glob("*.hcl")):
        policy = policy_path.read_text(encoding="utf-8")
        assert 'path "auth/token/renew-self"' in policy, policy_path.name
        assert 'capabilities = ["update"]' in policy, policy_path.name
        assert 'path "auth/token/create"' not in policy, policy_path.name


def test_bootstrap_and_verifier_encode_owner_custody_without_assert_gates() -> None:
    bootstrap = yaml.safe_load(
        (ROOT / "configs" / "openbao" / "bootstrap.yaml").read_text(encoding="utf-8")
    )
    verifier_source = (ROOT / "infra" / "openbao" / "verify_runtime.py").read_text(encoding="utf-8")

    assert bootstrap["initialization"] == {
        "unseal_shares": 3,
        "unseal_threshold": 2,
        "share_delivery": "owner-pgp-encrypted-files",
        "initial_admin_delivery": "owner-pgp-encrypted-file",
        "revoke_initial_admin_after_bootstrap": True,
    }
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(verifier_source)))


def test_backup_policy_requires_encryption_separate_volume_and_custody() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "openbao" / "backup.yaml").read_text(encoding="utf-8")
    )

    assert payload["encryption"]["scheme"] == "age-x25519"
    assert payload["destination"]["require_separate_volume"] is True
    assert payload["ownership"]["recovery_custodian"] == "installation_owner"
    assert payload["ownership"]["unseal_custodian"] == "installation_owner"
    assert "secret" not in json.dumps(payload).lower()


def test_runtime_verifier_fails_closed_when_compose_teardown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verify_runtime,
        "_compose",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        verify_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(verify_runtime.RuntimeProofError, match="cleanup failed"):
        verify_runtime._cleanup_project("roehub-stage08-test", 18200)


def test_runtime_verifier_applies_offline_image_override_after_base_compose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    override = tmp_path / "openbao-image-override.json"
    override.write_text('{"services":{"openbao":{"image":"sha256:fixture"}}}')

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(verify_runtime.subprocess, "run", fake_run)

    verify_runtime._compose(
        "roehub-stage08-test",
        18200,
        "config",
        "--quiet",
        compose_override=override,
    )

    assert captured["command"] == [
        "docker",
        "compose",
        "--project-name",
        "roehub-stage08-test",
        "--file",
        str(verify_runtime.COMPOSE),
        "--file",
        str(override),
        "config",
        "--quiet",
    ]


def test_runtime_verifier_cli_passes_compose_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("services: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_verify(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"schema": "fixture", "status": "passed"}

    monkeypatch.setattr(verify_runtime, "verify", fake_verify)

    assert verify_runtime.main(["--compose-override", str(override)]) == 0
    assert captured["compose_override"] == override
