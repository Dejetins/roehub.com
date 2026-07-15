from __future__ import annotations

import json
from pathlib import Path

from apps.common.runtime_health import RuntimeHealthState
from apps.worker.job_runtime.main import main as job_runtime_main
from apps.worker.notification_report_scheduler.main.main import _config


def test_safe_disabled_runtime_role_contract(tmp_path: Path) -> None:
    config = tmp_path / "notifications.yaml"
    config.write_text(
        "notifications:\n"
        "  report_scheduler:\n"
        "    enabled: false\n"
        "    poll_interval_seconds: 60\n",
        encoding="utf-8",
    )

    assert _config(config) == (False, 60)
    assert RuntimeHealthState(
        service="notification-report-scheduler",
        ready=True,
        mode="disabled",
        reason="disabled_by_safe_default",
    ).payload() == {
        "service": "notification-report-scheduler",
        "ready": True,
        "mode": "disabled",
        "reason": "disabled_by_safe_default",
    }


def test_job_runtime_doctor_uses_mounted_artifact_root(
    tmp_path: Path, capsys
) -> None:
    artifact_root = tmp_path / "artifacts"

    assert (
        job_runtime_main(["doctor", "--artifact-root", str(artifact_root)]) == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "artifact_volume": "writable",
        "docker_control": "typed-unix-socket-not-mounted",
        "executor": "JobAttemptExecutor",
        "oci_runner": "OciJobRunner",
        "status": "ready",
    }
    assert not (artifact_root / ".job-runtime-doctor").exists()
