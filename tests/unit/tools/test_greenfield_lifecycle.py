from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.release import verify_greenfield_lifecycle as lifecycle
from tools.release.greenfield_oidc_fixture import _DisposableProvider
from tools.release.verify_greenfield_lifecycle import (
    GreenfieldLifecycleError,
    Installation,
    _json_output,
    _sha256_bytes,
)


def test_installation_command_is_scoped_to_explicit_project_and_signed_profile(
    tmp_path: Path,
) -> None:
    installation = Installation(
        project="roehub-stage23-unit",
        bundle=tmp_path / "bundle",
        state=tmp_path / "state",
    )

    assert installation.command() == [
        "docker",
        "compose",
        "-p",
        "roehub-stage23-unit",
        "-f",
        str(tmp_path / "bundle/configs/installation/generated/trading/compose.yaml"),
        "-f",
        str(tmp_path / "state/compose.trading.offline.yaml"),
    ]


def test_identity_reconciliation_uses_canonical_administrative_audit_table() -> None:
    assert "identity_administrative_audit_events" in lifecycle._CORE_IDENTITY_TABLES
    assert "identity_admin_audit_events" not in lifecycle._CORE_IDENTITY_TABLES


def test_stage21_monitoring_override_keeps_internal_services_unpublished() -> None:
    override = yaml.safe_load(
        (
            lifecycle.ROOT
            / "tests/fixtures/observability-runtime-override.yaml"
        ).read_text()
    )
    services = override["services"]
    for service in (
        "alertmanager",
        "blackbox",
        "grafana",
        "loki",
        "operational-health",
        "prometheus",
        "web",
    ):
        assert "ports" not in services[service]
    assert services["grafana"]["environment"] == {
        "GF_SECURITY_ADMIN_PASSWORD": "stage21-disposable-only",
        "GF_SECURITY_ADMIN_PASSWORD__FILE": "",
    }


def test_json_output_requires_object() -> None:
    result = subprocess.CompletedProcess(args=["fixture"], returncode=0, stdout=b"[]", stderr=b"")

    with pytest.raises(GreenfieldLifecycleError, match="non-object"):
        _json_output(result, label="fixture")


def test_json_output_accepts_sanitized_object() -> None:
    payload = {"schema": "io.roehub.test/v1", "status": "passed"}
    result = subprocess.CompletedProcess(
        args=["fixture"],
        returncode=0,
        stdout=json.dumps(payload).encode(),
        stderr=b"",
    )

    assert _json_output(result, label="fixture") == payload


def test_openbao_metadata_uses_explicit_internal_http_address(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_compose_exec(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["service"] = args[1]
        captured["command"] = args[2]
        captured["allowed_codes"] = kwargs["allowed_codes"]
        return subprocess.CompletedProcess(
            args=["bao", "status"],
            returncode=2,
            stdout=json.dumps(
                {
                    "initialized": False,
                    "sealed": True,
                    "storage_type": "file",
                    "version": "2.5.4",
                }
            ).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(lifecycle, "_compose_exec", fake_compose_exec)
    installation = Installation(
        project="roehub-stage23-unit",
        bundle=tmp_path / "bundle",
        state=tmp_path / "state",
    )

    assert lifecycle._openbao_metadata(installation) == {
        "initialized": False,
        "sealed": True,
        "storage_type": "file",
        "version": "2.5.4",
    }
    assert captured == {
        "service": "openbao",
        "command": [
            "bao",
            "status",
            "-address=http://127.0.0.1:8200",
            "-format=json",
        ],
        "allowed_codes": frozenset({0, 2}),
    }


def test_pw_json_labels_invalid_machine_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(lifecycle, "_pw", lambda *args, **kwargs: "undefined")

    with pytest.raises(
        GreenfieldLifecycleError,
        match="browser fixture returned invalid JSON",
    ):
        lifecycle._pw_json(
            session="fixture",
            command=["run-code", "fixture"],
            cwd=tmp_path,
            label="browser fixture",
        )


def test_pw_json_accepts_object_machine_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(lifecycle, "_pw", lambda *args, **kwargs: '{"status":"passed"}')

    assert lifecycle._pw_json(
        session="fixture",
        command=["run-code", "fixture"],
        cwd=tmp_path,
        label="browser fixture",
    ) == {"status": "passed"}


def test_browser_boundary_reads_csrf_inside_page_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_pw_json(**kwargs: object) -> dict[str, object]:
        command = kwargs["command"]
        assert isinstance(command, list)
        captured["code"] = str(command[1])
        return {
            "csrf_present": True,
            "organization_count": 2,
            "invitation_count": 2,
        }

    monkeypatch.setattr(lifecycle, "_pw_json", fake_pw_json)

    lifecycle._browser_create_boundaries(session="fixture", cwd=tmp_path)

    assert "const csrfPresent = await page.evaluate" in captured["code"]
    assert "csrf_present: csrfPresent" in captured["code"]
    assert "const csrf = () =>" not in captured["code"]


def test_browser_isolation_uses_bounded_cdp_screenshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}
    screenshot = tmp_path / "admin.png"
    capture_calls: list[Path] = []

    def fake_pw_json(**kwargs: object) -> dict[str, object]:
        command = kwargs["command"]
        assert isinstance(command, list)
        captured["code"] = str(command[1])
        return {"admin_visible": True}

    def fake_pw(*args: object, **kwargs: object) -> str:
        command = args[1]
        assert isinstance(command, list)
        if command[0] == "requests":
            return "/api/v1/organizations /api/v1/admin/organizations/fixture"
        return "Total messages: 0 (Errors: 0, Warnings: 0)"

    monkeypatch.setattr(lifecycle, "_pw_json", fake_pw_json)
    monkeypatch.setattr(lifecycle, "_pw", fake_pw)
    monkeypatch.setattr(
        lifecycle,
        "_capture_browser_screenshot",
        lambda **kwargs: (
            capture_calls.append(kwargs["screenshot"]),
            kwargs["screenshot"].write_bytes(b"png"),
            "sha256:fixture",
        )[-1],
    )

    lifecycle._browser_validate_isolation(
        session="fixture",
        cwd=tmp_path,
        organizations={
            "primary_organization_id": "primary",
            "secondary_organization_id": "secondary",
        },
        users={"a": {"user_id": "user-a"}, "b": {"user_id": "user-b"}},
        screenshot=screenshot,
    )

    assert "page.screenshot" not in captured["code"]
    assert capture_calls == [screenshot]


def test_console_error_count_uses_playwright_summary() -> None:
    assert (
        lifecycle._console_error_count(
            "Total messages: 0 (Errors: 0, Warnings: 0)"
        )
        == 0
    )
    assert (
        lifecycle._console_error_count(
            "Total messages: 2 (Errors: 1, Warnings: 1)"
        )
        == 1
    )


def test_console_error_count_fails_closed_on_unknown_format() -> None:
    with pytest.raises(
        lifecycle.GreenfieldLifecycleError,
        match="unknown result format",
    ):
        lifecycle._console_error_count("no console summary")


def test_restored_passkey_login_forces_only_the_animated_button_click(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_pw_json(**kwargs: object) -> dict[str, object]:
        command = kwargs["command"]
        assert isinstance(command, list)
        captured["code"] = str(command[1])
        return {"admin_visible": True, "restored_passkey_login": True}

    monkeypatch.setattr(lifecycle, "_pw_json", fake_pw_json)

    lifecycle._browser_restore_login(session="fixture", cwd=tmp_path)

    assert "[data-passkey-login]').click({force: true})" in captured["code"]
    assert "[data-admin-presence].is-ready" in captured["code"]


def test_recovery_lifecycle_retries_only_transient_host_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefixes: list[str] = []
    overrides: list[Path] = []

    def fake_verify(
        *,
        project_prefix: str,
        image_override: Path,
    ) -> dict[str, object]:
        prefixes.append(project_prefix)
        overrides.append(image_override)
        if len(prefixes) == 1:
            raise lifecycle.RecoveryRuntimeProofError(
                "operational-health readiness failed"
            )
        return {"status": "passed"}

    monkeypatch.setattr(lifecycle, "verify_recovery_runtime", fake_verify)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    payload, attempts = lifecycle._verify_recovery_lifecycle(
        project_prefix="roehub-stage23-recovery-unit",
        image_override=tmp_path / "compose.base.offline.yaml",
    )

    assert payload == {"status": "passed"}
    assert attempts == 2
    assert prefixes == [
        "roehub-stage23-recovery-unit-attempt-1",
        "roehub-stage23-recovery-unit-attempt-2",
    ]
    assert overrides == [
        tmp_path / "compose.base.offline.yaml",
        tmp_path / "compose.base.offline.yaml",
    ]


def test_recovery_lifecycle_does_not_retry_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_verify(
        *,
        project_prefix: str,
        image_override: Path,
    ) -> dict[str, object]:
        assert image_override == tmp_path / "compose.base.offline.yaml"
        nonlocal calls
        calls += 1
        raise lifecycle.RecoveryRuntimeProofError("restore comparison failed")

    monkeypatch.setattr(lifecycle, "verify_recovery_runtime", fake_verify)

    with pytest.raises(
        lifecycle.GreenfieldLifecycleError,
        match="restore comparison failed",
    ):
        lifecycle._verify_recovery_lifecycle(
            project_prefix="fixture",
            image_override=tmp_path / "compose.base.offline.yaml",
        )
    assert calls == 1


def test_cdp_screenshot_requires_and_writes_bounded_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"\x89PNG\r\n\x1a\nfixture"
    captured: dict[str, str] = {}

    def fake_pw_json(**kwargs: object) -> dict[str, object]:
        command = kwargs["command"]
        assert isinstance(command, list)
        captured["code"] = str(command[1])
        return {"data": base64.b64encode(content).decode(), "dom_frozen": True}

    monkeypatch.setattr(
        lifecycle,
        "_pw_json",
        fake_pw_json,
    )
    screenshot = tmp_path / "admin.png"

    digest = lifecycle._capture_browser_screenshot(
        session="fixture",
        cwd=tmp_path,
        screenshot=screenshot,
    )

    assert screenshot.read_bytes() == content
    assert digest == lifecycle._sha256_bytes(content)
    assert "source.innerText" in captured["code"]
    assert ".slice(0, 4000)" in captured["code"]
    assert "page.context().newPage()" in captured["code"]
    assert "evidencePage.setContent" in captured["code"]
    assert "document.body.replaceChildren(evidence)" in captured["code"]
    assert "width: 1280, height: 720" in captured["code"]
    assert "Page.captureScreenshot" in captured["code"]


def test_sha256_bytes_uses_explicit_algorithm_prefix() -> None:
    assert _sha256_bytes(b"stage23") == (
        "sha256:cce661c1b393cf8d2c1423340dbcc841698d7bd8faec9b7a159a5d1ffd34322b"
    )


def test_disposable_oidc_provider_returns_only_fixture_identity() -> None:
    provider = _DisposableProvider(
        email="viewer-a@stage23.invalid.example",
        subject="stage23-disposable-subject-a",
    )

    identity = provider.exchange_code(
        code="fixture",
        code_verifier="fixture",
        expected_nonce_sha256="fixture",
    )

    assert provider.provider_id == "stage23-disposable-oidc"
    assert identity.email == "viewer-a@stage23.invalid.example"
    assert identity.email_verified is True
