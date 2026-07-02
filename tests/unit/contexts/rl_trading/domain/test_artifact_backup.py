from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading.contexts.rl_trading.domain import artifact_backup as ab
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage09b_backup_restore_drill_writes_and_validates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    candidate_manifest = _write_json(
        root / "evaluation_runs" / "stage08m" / "candidate_manifest.json",
        {"candidate_id": "stage08m_a3823cbd01143878_fd7c614b", "model_state": "redacted"},
    )
    candidate_scorecard = _write_json(
        root / "evaluation_runs" / "stage08m" / "scorecard_summary.json",
        {"stage09_allowed": True, "status": "accepted"},
    )
    source_manifest = _write_json(
        root / "datasets" / "stage08j" / "manifest.json",
        {"stage": "08J", "status": "accepted"},
    )
    research_summary = _write_json(
        root / "evaluation_runs" / "stage08l" / "summary.json",
        {"stage": "08L", "status": "accepted"},
    )

    result = ab.run_stage09b_backup_restore_drill_v1(
        ab.Stage09BDrillConfig(
            artifact_root=root,
            backup_root=root / "backups" / ab.STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1,
            restore_root=root / "restore_drills" / ab.STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage09b_test",
            generated_at_utc=datetime(2026, 7, 2, 20, 15, tzinfo=UTC),
            current_champion_manifest_path=candidate_manifest,
            expected_current_champion_manifest_sha256=compute_file_sha256(candidate_manifest),
            current_champion_scorecard_path=candidate_scorecard,
            expected_current_champion_scorecard_sha256=compute_file_sha256(candidate_scorecard),
            source_manifest_path=source_manifest,
            expected_source_manifest_sha256=compute_file_sha256(source_manifest),
            research_source_summary_path=research_summary,
            expected_research_source_summary_sha256=compute_file_sha256(research_summary),
        )
    )

    assert result["status"] == "accepted"
    backup_manifest = Path(str(result["backup_manifest_path"]))
    registry_dump = Path(str(result["registry_metadata_dump_path"]))
    rollback_manifest = Path(str(result["rollback_manifest_path"]))
    restore_report = Path(str(result["restore_report_path"]))
    assert backup_manifest.is_file()
    assert registry_dump.is_file()
    assert rollback_manifest.is_file()
    assert restore_report.is_file()

    backup_payload = _read_json(backup_manifest)
    restore_payload = _read_json(restore_report)
    registry_payload = _read_json(registry_dump)
    assert backup_payload["artifact_count"] == 8
    assert restore_payload["restored_artifact_count"] == 8
    assert registry_payload["calibration_pack"]["status"] == "not_created_pre_stage10"  # type: ignore[index]
    assert registry_payload["retention_policy"]["retain_forever"]  # type: ignore[index]
    assert registry_payload["retention_policy"]["retain_days"] == [  # type: ignore[index]
        {
            "days": 30,
            "name": "stage09b_restore_drill_copies",
            "reason": "operator replay window after accepted drill",
        }
    ]
    rendered_dump = json.dumps(registry_payload, sort_keys=True)
    assert "weights" not in rendered_dump
    assert registry_payload["safety"]["raw_provider_payloads_in_dump"] is False  # type: ignore[index]


def test_stage09b_restore_detects_tampered_backup_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    result = _run_fixture_drill(root)
    backup_manifest = _read_json(Path(str(result["backup_manifest_path"])))
    first_entry = backup_manifest["backup_entries"][0]  # type: ignore[index]
    backup_file = Path(str(backup_manifest["backup_root"])) / str(  # type: ignore[index]
        first_entry["backup_relative_path"]  # type: ignore[index]
    )
    backup_file.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ab.ArtifactBackupError, match="backup_artifact_sha256_mismatch"):
        ab.restore_stage09b_backup_v1(
            backup_manifest_path=Path(str(result["backup_manifest_path"])),
            restore_run_root=root / "restore_drills" / "tampered",
            generated_at_utc=datetime(2026, 7, 2, 20, 30, tzinfo=UTC),
        )


def test_stage09b_artifact_spec_blocks_paths_outside_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_artifact_root(tmp_path, monkeypatch)

    with pytest.raises(ab.ArtifactBackupError, match="path_outside_artifact_root"):
        ab.Stage09BArtifactSpec(
            artifact_id="outside",
            role="source_manifest",
            source_path=str(tmp_path / "outside.json"),
            expected_sha256="1" * 64,
            retention_class="retain_forever",
        )


def _run_fixture_drill(root: Path) -> dict[str, object]:
    candidate_manifest = _write_json(root / "evaluation_runs" / "stage08m" / "m.json", {"m": 1})
    candidate_scorecard = _write_json(root / "evaluation_runs" / "stage08m" / "s.json", {"s": 1})
    source_manifest = _write_json(root / "datasets" / "stage08j" / "manifest.json", {"d": 1})
    research_summary = _write_json(root / "evaluation_runs" / "stage08l" / "summary.json", {"r": 1})
    return ab.run_stage09b_backup_restore_drill_v1(
        ab.Stage09BDrillConfig(
            artifact_root=root,
            backup_root=root / "backups" / ab.STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1,
            restore_root=root / "restore_drills" / ab.STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage09b_tamper",
            generated_at_utc=datetime(2026, 7, 2, 20, 20, tzinfo=UTC),
            current_champion_manifest_path=candidate_manifest,
            expected_current_champion_manifest_sha256=compute_file_sha256(candidate_manifest),
            current_champion_scorecard_path=candidate_scorecard,
            expected_current_champion_scorecard_sha256=compute_file_sha256(candidate_scorecard),
            source_manifest_path=source_manifest,
            expected_source_manifest_sha256=compute_file_sha256(source_manifest),
            research_source_summary_path=research_summary,
            expected_research_source_summary_sha256=compute_file_sha256(research_summary),
        )
    )


def _patch_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "rl_trading"
    root.mkdir()
    monkeypatch.setattr(ab, "RL_TRADING_ARTIFACT_ROOT_V1", str(root))
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
