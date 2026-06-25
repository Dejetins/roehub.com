from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading.stage08e_roehub_native_full_training_run import main
from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    SESSIONIZED_DATASET_MANIFEST_KIND_V1,
)


def test_stage08e_roehub_native_training_cli_writes_candidate_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    manifest_path = _write_stage06_manifest_fixture(tmp_path / "stage06")

    result = main(
        [
            "run",
            "--stage06-manifest-path",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--episodes",
            "2",
            "--batch-size",
            "2",
            "--train-start",
            "2",
            "--replay-capacity",
            "32",
            "--validation-every-episodes",
            "1",
            "--checkpoint-every-episodes",
            "1",
            "--progress-emit-every-episodes",
            "1",
            "--progress-emit-every-sec",
            "3600",
            "--validation-max-sessions",
            "2",
            "--device-policy",
            "cpu_only_deterministic",
            "--torch-num-threads",
            "1",
            "--torch-num-interop-threads",
            "1",
            "--generated-at-utc",
            "2026-06-24T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["candidate_manifest_path"]).read_text(encoding="utf-8"))

    assert result == 0
    assert payload["status"] == "completed"
    assert manifest["candidate_level"] == "roehub_native_candidate"
    assert manifest["stage"] == "08E"
    assert manifest["dataset_dependency"]["stage"] == "06"
    assert manifest["dataset_dependency"]["splits"]["train"]["full_split_selected"] is True
    assert manifest["metrics_summary"]["completed_episodes"] == 2
    assert manifest["metrics_summary"]["scripted_transition_sequence_used"] is False
    assert manifest["next_stage_handoff"]["stage08f_allowed"] is True


def _write_stage06_manifest_fixture(root: Path) -> Path:
    train_features_path = root / "hf_period_rebuild_current_trading/train/BTCUSDT/sessions.f32.npy"
    validation_features_path = (
        root / "hf_period_rebuild_current_trading/validation/BTCUSDT/sessions.f32.npy"
    )
    _write_features(train_features_path, session_count=3)
    _write_features(validation_features_path, session_count=2)
    train_sha = _file_sha256_hex(train_features_path)
    validation_sha = _file_sha256_hex(validation_features_path)
    manifest = {
        "deterministic_rebuild_hash": "c" * 64,
        "leakage_report": {"status": "accepted"},
        "manifest_kind": SESSIONIZED_DATASET_MANIFEST_KIND_V1,
        "market": "binance:futures",
        "split_artifacts": [
            _split_entry(
                path=train_features_path,
                sha256=train_sha,
                split="train",
                session_count=3,
            ),
            _split_entry(
                path=validation_features_path,
                sha256=validation_sha,
                split="validation",
                session_count=2,
            ),
        ],
        "stage": "06",
        "status": "accepted",
        "total_sessions": 5,
    }
    manifest_path = root / "stage06_sessionized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def _split_entry(*, path: Path, sha256: str, split: str, session_count: int) -> dict[str, object]:
    return {
        "candidate_count": session_count,
        "dataset_version": "hf_period_rebuild_current_trading",
        "deterministic_rebuild_hash": f"{split}-hash",
        "files": {
            "features": {
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": sha256,
            }
        },
        "split": split,
        "symbol": "BTCUSDT",
    }


def _write_features(path: Path, *, session_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.zeros((session_count, 150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    minute = np.arange(150, dtype=np.float32)
    for session_idx in range(session_count):
        close = 100.0 + float(session_idx) + minute * np.float32(0.02)
        values = {
            "close": close,
            "high": close + np.float32(0.05),
            "low": close - np.float32(0.05),
            "num_trades": np.full_like(close, 15.0 + float(session_idx)),
            "open": close - np.float32(0.01),
            "volume": np.full_like(close, 30.0 + float(session_idx)),
            "volume_weighted_average": close,
        }
        for name, value in values.items():
            features[session_idx, :, FEATURE_NAMES_V1.index(name)] = value
    np.save(path, features)


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
