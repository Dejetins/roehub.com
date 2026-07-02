from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from trading.contexts.rl_trading.domain import model_registry as mr
from trading.contexts.rl_trading.domain.hf_reproducibility import compute_file_sha256


def test_stage09_candidate_gate_accepts_only_08m_candidate() -> None:
    payload = mr.validate_stage09_accepted_candidate_input_v1(
        candidate_id=mr.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
        candidate_manifest_sha256=mr.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
    )

    assert payload["stage09_allowed"] is True
    assert payload["policy_id"] == "supervised_oracle_label_warm_start_contextual_bandit"

    with pytest.raises(mr.ModelRegistryError, match="unexpected_stage09_candidate_id"):
        mr.validate_stage09_accepted_candidate_input_v1(
            candidate_id="stage08k_blocked_candidate",
            candidate_manifest_sha256=mr.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
        )


def test_registry_state_machine_blocks_invalid_model_and_activation_transitions() -> None:
    mr.validate_registry_transition_v1(
        entity="model_version",
        current_status="candidate",
        next_status="accepted_champion",
    )
    with pytest.raises(mr.ModelRegistryError, match="invalid_registry_transition"):
        mr.validate_registry_transition_v1(
            entity="model_version",
            current_status="candidate",
            next_status="rollback_candidate",
        )
    with pytest.raises(mr.ModelRegistryError, match="invalid_registry_transition"):
        mr.validate_registry_transition_v1(
            entity="activation",
            current_status="inactive",
            next_status="live",
        )


def test_activation_requires_accepted_matching_hashes_and_writes_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    dataset = _dataset(root)
    model = _model(root, status="accepted_champion")
    calibration = _calibration(root, model_version_id=model.model_version_id)
    scope = _scope()
    activation = mr.build_activation_record_v1(
        activation_id="11111111-1111-1111-1111-111111111111",
        scope=scope,
        next_state="shadow",
        dataset=dataset,
        model=model,
        calibration=calibration,
        activation_matrix_hash="5" * 64,
    )
    audit = mr.build_activation_audit_event_v1(
        event_type="activate",
        scope=scope,
        next_activation=activation,
        reason_code="stage09_shadow_activation_gate",
        operator_ref_hash="6" * 64,
        generated_at_utc=datetime(2026, 7, 2, 18, 30, tzinfo=UTC),
    )

    assert activation.model_checkpoint_sha256 == model.checkpoint_sha256
    assert activation.calibration_pack_hash == calibration.calibration_pack_hash
    assert audit["event_payload_hash"]
    assert audit["operator_ref_hash"] == "6" * 64

    mismatched_calibration = _calibration(
        root,
        model_version_id=model.model_version_id,
        dataset_hash="7" * 64,
    )
    with pytest.raises(mr.ModelRegistryError, match="calibration_dataset_hash_mismatch"):
        mr.assert_activation_ready_v1(
            dataset=dataset,
            model=model,
            calibration=mismatched_calibration,
            scope=scope,
            activation_matrix_hash="5" * 64,
        )


def test_checkpoint_loader_validates_path_and_sha_before_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    checkpoint = root / "models" / "candidate" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted local checkpoint bytes")
    checkpoint_sha = compute_file_sha256(checkpoint)
    model = _model(root, status="accepted_champion", checkpoint_sha256=checkpoint_sha)
    torch = _FakeTorch()

    payload = mr.load_trusted_checkpoint_weights_v1(
        model=model,
        torch_module=torch,
        artifact_root=root,
    )

    assert payload["loaded_path"] == str(checkpoint.resolve())
    assert torch.calls == [
        {
            "map_location": "cpu",
            "path": str(checkpoint.resolve()),
            "weights_only": True,
        }
    ]

    checkpoint.write_bytes(b"corrupt")
    failing_torch = _FakeTorch()
    with pytest.raises(mr.ModelRegistryError, match="checkpoint_sha256_mismatch"):
        mr.load_trusted_checkpoint_weights_v1(
            model=model,
            torch_module=failing_torch,
            artifact_root=root,
        )
    assert failing_torch.calls == []


def test_checkpoint_loader_blocks_candidate_and_requires_explicit_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    checkpoint = root / "models" / "candidate" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted")
    checkpoint_sha = compute_file_sha256(checkpoint)

    with pytest.raises(mr.ModelRegistryError, match="model_status_not_runtime_loadable"):
        mr.validate_checkpoint_before_load_v1(
            model=_model(root, status="candidate", checkpoint_sha256=checkpoint_sha),
            artifact_root=root,
        )

    rollback_model = _model(
        root,
        status="rollback_candidate",
        checkpoint_sha256=checkpoint_sha,
    )
    with pytest.raises(mr.ModelRegistryError, match="rollback_candidate_requires_explicit"):
        mr.validate_checkpoint_before_load_v1(model=rollback_model, artifact_root=root)

    assert (
        mr.validate_checkpoint_before_load_v1(
            model=rollback_model,
            artifact_root=root,
            explicit_rollback_selected=True,
        )
        == checkpoint.resolve()
    )


def test_cleanup_plan_retains_accepted_source_and_active_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    now = datetime(2026, 7, 2, 18, 30, tzinfo=UTC)
    policy = mr.ArtifactLifecyclePolicy(
        artifact_root=str(root),
        rejected_run_retention_days=7,
        disk_quota_bytes=10_000_000,
        disk_watermark_pct=85.0,
    )
    cleanup = mr.plan_registry_cleanup_v1(
        artifacts=[
            _artifact(root, "old-rejected", "training_run", "rejected", now - timedelta(days=8)),
            _artifact(root, "new-rejected", "training_run", "rejected", now - timedelta(days=2)),
            _artifact(
                root,
                "champion",
                "model_version",
                "accepted_champion",
                now - timedelta(days=30),
                referenced_by_active_metadata=True,
            ),
            _artifact(
                root,
                "source-manifest",
                "source_manifest",
                "superseded",
                now - timedelta(days=30),
                source_manifest=True,
            ),
        ],
        policy=policy,
        now_utc=now,
    )

    delete_candidates = cast(list[dict[str, object]], cleanup["delete_candidates"])
    retained_artifacts = cast(list[dict[str, object]], cleanup["retain"])

    assert [item["artifact_id"] for item in delete_candidates] == ["old-rejected"]
    retain_reasons = {
        item["artifact_id"]: item["retention_reason"] for item in retained_artifacts
    }
    assert retain_reasons["new-rejected"] == "retention_window_not_elapsed"
    assert retain_reasons["champion"] == "active_metadata_reference_retained"
    assert retain_reasons["source-manifest"] == "source_manifest_retained"


def test_mark_missing_artifact_is_explicit_state_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_artifact_root(tmp_path, monkeypatch)
    model = _model(root, status="accepted_champion")

    missing = mr.mark_model_artifact_missing_v1(model)

    assert missing.status == "missing_artifact"
    with pytest.raises(mr.ModelRegistryError, match="model_status_not_runtime_loadable"):
        mr.assert_model_runtime_load_allowed_v1(missing)


class _FakeTorch:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def load(self, path: str, **kwargs: Any) -> dict[str, object]:
        self.calls.append({"path": path, **kwargs})
        return {"loaded_path": path, "kwargs": dict(kwargs)}


def _patch_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "rl_trading"
    root.mkdir()
    monkeypatch.setattr(mr, "RL_TRADING_ARTIFACT_ROOT_V1", str(root))
    return root


def _scope() -> mr.ActivationScope:
    return mr.ActivationScope(
        model_family="rl_platform_warm_start_v1",
        feature_contract_hash="1" * 64,
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
    )


def _dataset(root: Path, *, dataset_hash: str = "2" * 64) -> mr.DatasetVersionRecord:
    return mr.DatasetVersionRecord(
        dataset_version_id="article_future_10m_5pct_contrast_v1",
        dataset_hash=dataset_hash,
        manifest_path=str(root / "datasets" / "stage08j" / "manifest.json"),
        manifest_sha256="3" * 64,
        feature_contract_hash="1" * 64,
        status="accepted",
    )


def _model(
    root: Path,
    *,
    status: mr.ModelVersionStatus,
    checkpoint_sha256: str = "4" * 64,
) -> mr.ModelVersionRecord:
    return mr.ModelVersionRecord(
        model_version_id=mr.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
        model_family="rl_platform_warm_start_v1",
        feature_contract_hash="1" * 64,
        dataset_version_id="article_future_10m_5pct_contrast_v1",
        dataset_hash="2" * 64,
        checkpoint_path=str(root / "models" / "candidate" / "checkpoint.pt"),
        checkpoint_sha256=checkpoint_sha256,
        model_state_hash="8" * 64,
        status=status,
    )


def _calibration(
    root: Path,
    *,
    model_version_id: str,
    dataset_hash: str = "2" * 64,
) -> mr.CalibrationPackRecord:
    return mr.CalibrationPackRecord(
        calibration_pack_id="calibration-stage09-a",
        model_version_id=model_version_id,
        feature_contract_hash="1" * 64,
        dataset_hash=dataset_hash,
        calibration_pack_hash="9" * 64,
        calibration_path=str(root / "models" / "candidate" / "calibration.json"),
        calibration_sha256="a" * 64,
        status="accepted",
    )


def _artifact(
    root: Path,
    artifact_id: str,
    entity: mr.ArtifactEntity,
    status: str,
    updated_at_utc: datetime,
    *,
    referenced_by_active_metadata: bool = False,
    source_manifest: bool = False,
) -> mr.RegistryArtifactRecord:
    return mr.RegistryArtifactRecord(
        artifact_id=artifact_id,
        entity=entity,
        status=status,
        artifact_path=str(root / "artifacts" / artifact_id),
        artifact_sha256="b" * 64,
        updated_at_utc=updated_at_utc,
        referenced_by_active_metadata=referenced_by_active_metadata,
        source_manifest=source_manifest,
    )
