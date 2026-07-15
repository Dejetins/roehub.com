from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.roehubctl.main.main import _redact, main


def test_effective_config_redacts_nested_sensitive_values() -> None:
    payload = {
        "database_dsn": "postgresql://private",
        "nested": {"access_token": "private", "enabled": True},
        "items": [{"password_file": "/run/private", "name": "safe"}],
    }

    redacted = _redact(payload)

    assert redacted == {
        "database_dsn": "[redacted]",
        "nested": {"access_token": "[redacted]", "enabled": True},
        "items": [{"password_file": "[redacted]", "name": "safe"}],
    }


def test_effective_rejects_noncanonical_path_and_residual_secret(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrong = tmp_path / "effective-config.json"
    wrong.write_text("{}", encoding="utf-8")

    assert main(["effective", "--path", str(wrong)], environ={}) == 2
    wrong_payload = json.loads(capsys.readouterr().out)
    assert wrong_payload["code"] == "roehubctl.effective_config_path_rejected"

    canonical = tmp_path / "effective-config.redacted.json"
    canonical.write_text(
        json.dumps({"innocent": "Bearer " + "a" * 32}),
        encoding="utf-8",
    )

    assert main(["effective", "--path", str(canonical)], environ={}) == 2
    residual_payload = json.loads(capsys.readouterr().out)
    assert residual_payload["code"] == "roehubctl.effective_config_redaction_failed"
