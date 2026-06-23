from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading.stage07b_full_candidate_training_run import main
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1


def test_stage07b_candidate_training_writes_progress_and_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    manifest_path = _write_stage06_fixture(tmp_path)
    result = main(
        [
            "run",
            "--sessionized-manifest",
            str(manifest_path),
            "--expected-sessionized-manifest-sha256",
            _file_sha256_hex(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--planned-training-steps",
            "3",
            "--progress-emit-every-steps",
            "1",
            "--progress-emit-every-sec",
            "3600",
            "--checkpoint-every-steps",
            "2",
            "--validation-every-steps",
            "1",
            "--validation-max-transitions",
            "8",
            "--batch-size",
            "4",
            "--replay-capacity",
            "64",
            "--hidden-dim",
            "32",
            "--hidden-dim",
            "32",
            "--torch-num-threads",
            "1",
            "--torch-num-interop-threads",
            "1",
            "--generated-at-utc",
            "2026-06-23T12:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["status"] == "completed"
    manifest = json.loads(Path(payload["candidate_manifest_path"]).read_text(encoding="utf-8"))
    progress_path = Path(manifest["artifact_hashes"]["progress_jsonl"]["path"])
    progress_events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["stage"] == "07B"
    assert manifest["status"] == "completed"
    assert manifest["dataset_dependency"]["manifest_sha256"] == _file_sha256_hex(manifest_path)
    assert progress_events[-1]["status"] == "completed"
    assert progress_events[-1]["completed_training_steps"] == 3
    assert progress_events[-1]["progress_pct"] == 100.0
    assert manifest["next_stage_handoff"]["stage08_allowed"] is True


def _write_stage06_fixture(tmp_path: Path) -> Path:
    split_artifacts = []
    for split in ("train", "validation"):
        artifact_root = tmp_path / "hf_period_rebuild_current_trading" / split / "BTCUSDT"
        artifact_root.mkdir(parents=True)
        features_path = artifact_root / "sessions.f32.npy"
        np.save(features_path, _session_features())
        split_artifacts.append(
            {
                "dataset_version": "hf_period_rebuild_current_trading",
                "files": {
                    "features": {
                        "bytes": features_path.stat().st_size,
                        "path": str(features_path),
                        "sha256": _file_sha256_hex(features_path),
                    }
                },
                "split": split,
                "symbol": "BTCUSDT",
            }
        )
    manifest = {
        "market": "binance:futures",
        "split_artifacts": split_artifacts,
        "stage": "06",
        "status": "accepted",
    }
    manifest_path = tmp_path / "stage06_sessionized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _session_features() -> np.ndarray:
    features = np.zeros((2, 150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    minute = np.arange(150, dtype=np.float32)
    for session_idx in range(2):
        close = 100.0 + session_idx + minute * 0.01
        values = {
            "close": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "num_trades": np.full_like(close, 11.0),
            "open": close - 0.01,
            "volume": np.full_like(close, 25.0),
            "volume_weighted_average": close,
        }
        for name, value in values.items():
            features[session_idx, :, FEATURE_NAMES_V1.index(name)] = value
    return features


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
