from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.rl_trading import stage10a_retraining_promotion_lifecycle as stage10a
from trading.contexts.rl_trading.domain import retraining_promotion_lifecycle as rpl
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage10a_cli_plan_retrain_manual_writes_candidate_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    exit_code = stage10a.main(
        [
            "plan-retrain",
            "--artifact-root",
            str(root),
            "--output-root",
            str(root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1),
            "--run-id",
            "stage10a_cli_manual",
            "--generated-at-utc",
            "2026-07-02T22:30:00Z",
            "--mode",
            "full_retrain",
            "--trigger",
            "manual",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "accepted"
    task = _read_json(Path(payload["retrain_task_path"]))
    assert task["status"] == "planned_candidate"
    assert task["auto_promote"] is False


def test_stage10a_cli_scheduled_trigger_is_blocked_until_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    exit_code = stage10a.main(
        [
            "plan-retrain",
            "--artifact-root",
            str(root),
            "--output-root",
            str(root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1),
            "--run-id",
            "stage10a_cli_scheduled",
            "--generated-at-utc",
            "2026-07-02T22:31:00Z",
            "--mode",
            "fine_tune",
            "--trigger",
            "scheduled",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    task = _read_json(Path(payload["retrain_task_path"]))
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert task["blockers"] == ["schedule_disabled_by_default"]


def test_stage10a_cli_promotion_check_and_rollback_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_promotion_fixture(root)
    common_args = [
        "promotion-check",
        "--artifact-root",
        str(root),
        "--output-root",
        str(root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1),
        "--run-id",
        "stage10a_cli_promotion",
        "--generated-at-utc",
        "2026-07-02T22:32:00Z",
        "--candidate-model-version-id",
        "stage10a_candidate",
        "--candidate-manifest-path",
        str(fixture["candidate_manifest_path"]),
        "--expected-candidate-manifest-sha256",
        str(fixture["candidate_manifest_sha256"]),
        "--calibration-pack-path",
        str(fixture["calibration_pack_path"]),
        "--expected-calibration-pack-sha256",
        str(fixture["calibration_pack_sha256"]),
        "--pnl-after-fees-funding-slippage-quote",
        "1000",
        "--max-drawdown-quote",
        "100",
        "--trades-count",
        "250",
        "--ticker-positive-group-ratio",
        "0.60",
        "--out-of-sample-days",
        "45",
        "--overfit-ratio",
        "1.10",
        "--latency-p95-ms",
        "125",
        "--resource-rss-mb",
        "2048",
        "--artifact-integrity-ok",
        "--registry-integrity-ok",
    ]

    blocked_exit = stage10a.main(common_args)
    blocked = json.loads(capsys.readouterr().out)
    assert blocked_exit == 2
    assert blocked["status"] == "blocked"

    accepted_exit = stage10a.main(
        [
            *common_args,
            "--operator-ref-hash",
            "2" * 64,
            "--admin-ref-hash",
            "3" * 64,
        ]
    )
    accepted = json.loads(capsys.readouterr().out)
    assert accepted_exit == 0
    assert accepted["status"] == "accepted"
    check = _read_json(Path(accepted["check_path"]))
    assert check["status"] == "promotion_ready"
    assert check["registry_write_performed"] is False

    rollback_exit = stage10a.main(
        [
            "rollback-dry-run",
            "--artifact-root",
            str(root),
            "--output-root",
            str(root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1),
            "--run-id",
            "stage10a_cli_rollback",
            "--generated-at-utc",
            "2026-07-02T22:33:00Z",
            "--to-model-version-id",
            "stage09b_previous_accepted_champion_restore_drill",
            "--to-calibration-pack-id",
            "stage09b_previous_calibration_pack_restore_drill",
            "--current-registry-metadata-sha256",
            "4" * 64,
            "--previous-champion-manifest-sha256",
            "5" * 64,
            "--previous-calibration-pack-sha256",
            "6" * 64,
            "--operator-ref-hash",
            "7" * 64,
        ]
    )
    rollback = json.loads(capsys.readouterr().out)
    assert rollback_exit == 0
    assert rollback["status"] == "accepted"
    manifest = _read_json(Path(rollback["rollback_manifest_path"]))
    assert manifest["no_artifact_deletion"] is True


def _patch_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "rl_trading"
    root.mkdir()
    monkeypatch.setattr(rpl, "RL_TRADING_ARTIFACT_ROOT_V1", str(root))
    return root


def _write_promotion_fixture(root: Path) -> dict[str, Any]:
    candidate_manifest_path = _write_json(
        root / "models" / "stage10a_candidate" / "candidate_manifest.json",
        {"candidate_id": "stage10a_candidate", "status": "candidate"},
    )
    calibration_pack_path = _write_json(
        root / "calibration_packs" / "stage10" / "calibration_pack.json",
        {"calibration_pack_id": rpl.DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1, "status": "accepted"},
    )
    return {
        "calibration_pack_path": calibration_pack_path,
        "calibration_pack_sha256": compute_file_sha256(calibration_pack_path),
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": compute_file_sha256(candidate_manifest_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
