from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading.contexts.rl_trading.domain import retraining_promotion_lifecycle as rpl
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage10a_retrain_plan_is_deterministic_and_never_auto_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    config = rpl.Stage10ARetrainTaskConfig(
        artifact_root=root,
        output_root=root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1,
        run_id="stage10a_manual_full",
        generated_at_utc=datetime(2026, 7, 2, 22, 0, tzinfo=UTC),
        retrain_mode="full_retrain",
        trigger="manual",
        requested_by_ref_hash="1" * 64,
    )

    first = rpl.run_stage10a_retrain_task_plan_v1(config)
    second = rpl.run_stage10a_retrain_task_plan_v1(config)

    assert first["status"] == "accepted"
    assert first["summary_hash"] == second["summary_hash"]
    assert first["summary_sha256"] == second["summary_sha256"]
    task = _read_json(Path(str(first["retrain_task_path"])))
    assert task["status"] == "planned_candidate"
    assert task["retrain_mode"] == "full_retrain"
    assert task["manual_trigger_supported"] is True
    assert task["auto_promote"] is False
    assert task["registry_write_performed"] is False
    assert task["candidate_output_state"] == "candidate_pending_promotion_scorecard"


def test_stage10a_schedule_is_disabled_by_default_and_drift_creates_candidate_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    scheduled = rpl.run_stage10a_retrain_task_plan_v1(
        rpl.Stage10ARetrainTaskConfig(
            artifact_root=root,
            output_root=root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage10a_scheduled_disabled",
            generated_at_utc=datetime(2026, 7, 2, 22, 5, tzinfo=UTC),
            retrain_mode="fine_tune",
            trigger="scheduled",
        )
    )
    scheduled_task = _read_json(Path(str(scheduled["retrain_task_path"])))

    assert scheduled["status"] == "blocked"
    assert scheduled_task["schedule"]["status"] == "disabled_by_default"
    assert scheduled_task["blockers"] == ["schedule_disabled_by_default"]
    assert scheduled_task["auto_promote"] is False

    drift = rpl.run_stage10a_retrain_task_plan_v1(
        rpl.Stage10ARetrainTaskConfig(
            artifact_root=root,
            output_root=root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage10a_drift_fine_tune",
            generated_at_utc=datetime(2026, 7, 2, 22, 6, tzinfo=UTC),
            retrain_mode="fine_tune",
            trigger="drift",
            drift_signal_id="feature_drift:btc_20260702",
        )
    )
    drift_task = _read_json(Path(str(drift["retrain_task_path"])))

    assert drift["status"] == "accepted"
    assert drift_task["drift"]["creates_candidate_task"] is True
    assert drift_task["drift"]["mutates_champion"] is False
    assert drift_task["retrain_mode"] == "fine_tune"


def test_stage10a_promotion_requires_hard_gates_and_operator_admin_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    fixture = _write_promotion_fixture(root)
    common = _promotion_config_kwargs(root, fixture)

    blocked = rpl.run_stage10a_promotion_check_v1(
        rpl.Stage10APromotionCheckConfig(
            **common,
            operator_ref_hash=None,
            admin_ref_hash=None,
            auto_promote_requested=False,
        )
    )
    blocked_check = _read_json(Path(str(blocked["check_path"])))

    assert blocked["status"] == "blocked"
    assert blocked_check["status"] == "blocked"
    assert blocked_check["registry_write_performed"] is False
    assert blocked_check["auto_promote"] is False
    assert blocked_check["blockers"] == [
        "admin_approval_present",
        "operator_approval_present",
    ]

    accepted = rpl.run_stage10a_promotion_check_v1(
        rpl.Stage10APromotionCheckConfig(
            **common,
            operator_ref_hash="2" * 64,
            admin_ref_hash="3" * 64,
            auto_promote_requested=False,
        )
    )
    accepted_check = _read_json(Path(str(accepted["check_path"])))

    assert accepted["status"] == "accepted"
    assert accepted_check["status"] == "promotion_ready"
    assert accepted_check["activation_mutation"] is False
    assert accepted_check["registry_write_performed"] is False
    assert accepted_check["warnings"] == []

    auto_promote = rpl.run_stage10a_promotion_check_v1(
        rpl.Stage10APromotionCheckConfig(
            **common,
            operator_ref_hash="2" * 64,
            admin_ref_hash="3" * 64,
            auto_promote_requested=True,
        )
    )
    auto_promote_check = _read_json(Path(str(auto_promote["check_path"])))
    assert auto_promote["status"] == "blocked"
    assert auto_promote_check["blockers"] == ["auto_promotion_not_requested"]


def test_stage10a_rollback_manifest_is_host_local_and_non_destructive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    result = rpl.run_stage10a_rollback_dry_run_v1(
        rpl.Stage10ARollbackConfig(
            artifact_root=root,
            output_root=root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1,
            run_id="stage10a_rollback",
            generated_at_utc=datetime(2026, 7, 2, 22, 20, tzinfo=UTC),
            current_champion_model_version_id=rpl.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
            previous_champion_model_version_id="stage09b_previous_accepted_champion_restore_drill",
            current_calibration_pack_id=rpl.DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
            previous_calibration_pack_id="stage09b_previous_calibration_pack_restore_drill",
            current_registry_metadata_sha256="4" * 64,
            previous_champion_manifest_sha256="5" * 64,
            previous_calibration_pack_sha256="6" * 64,
            operator_ref_hash="7" * 64,
            reason="stage10a_operator_rollback",
        )
    )

    manifest = _read_json(Path(str(result["rollback_manifest_path"])))
    assert result["status"] == "accepted"
    assert manifest["status"] == "rollback_dry_run_ready"
    assert manifest["no_artifact_deletion"] is True
    assert manifest["registry_write_performed"] is False
    assert "stage10a_retraining_promotion_lifecycle.py rollback-dry-run" in manifest["command"]


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


def _promotion_config_kwargs(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_root": root,
        "output_root": root / "lifecycle_runs" / rpl.STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1,
        "run_id": "stage10a_promotion",
        "generated_at_utc": datetime(2026, 7, 2, 22, 10, tzinfo=UTC),
        "candidate_model_version_id": "stage10a_candidate",
        "current_champion_model_version_id": rpl.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
        "candidate_manifest_path": fixture["candidate_manifest_path"],
        "expected_candidate_manifest_sha256": fixture["candidate_manifest_sha256"],
        "calibration_pack_path": fixture["calibration_pack_path"],
        "expected_calibration_pack_sha256": fixture["calibration_pack_sha256"],
        "calibration_pack_id": rpl.DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
        "calibration_pack_hash": rpl.DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1,
        "scorecard": rpl.Stage10APromotionScorecard(
            pnl_after_fees_funding_slippage_quote=1000.0,
            max_drawdown_quote=100.0,
            trades_count=250,
            ticker_positive_group_ratio=0.60,
            out_of_sample_days=45,
            overfit_ratio=1.10,
            latency_p95_ms=125.0,
            resource_rss_mb=2048.0,
            artifact_integrity_ok=True,
            registry_integrity_ok=True,
        ),
        "approval_reason": "stage10a_promotion_check",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
