from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading.stage07a_training_runner_smoke import run_stage07a_training_smoke
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1, TrainingSmokeConfig


def test_stage07a_training_smoke_loads_sessionized_artifact_and_runs_torch_update(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    manifest_path = _write_stage06_fixture(tmp_path)
    record = run_stage07a_training_smoke(
        sessionized_manifest_path=manifest_path,
        expected_sessionized_manifest_sha256=_file_sha256_hex(manifest_path),
        output_root=tmp_path / "smoke",
        dataset_version="post_hf_extension_current_trading",
        split="post_hf_extension",
        symbols=("BTCUSDT",),
        max_session_artifacts=1,
        config=TrainingSmokeConfig(
            seed=13,
            max_sessions=2,
            batch_size=4,
            update_steps=2,
            torch_num_threads=1,
        ),
        generated_at_utc=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    )

    assert record["status"] == "accepted"
    assert record["dataset_dependency"]["manifest_sha256"] == _file_sha256_hex(manifest_path)
    assert record["metrics"]["batch_shapes"]["q_values_shape"] == [4, 4]
    assert record["safety"]["candidate_model_claim"] is False


def _write_stage06_fixture(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "post_hf_extension_current_trading" / "post_hf_extension" / "BTCUSDT"
    artifact_root.mkdir(parents=True)
    features_path = artifact_root / "sessions.f32.npy"
    np.save(features_path, _session_features())
    manifest = {
        "market": "binance:futures",
        "split_artifacts": [
            {
                "dataset_version": "post_hf_extension_current_trading",
                "files": {
                    "features": {
                        "bytes": features_path.stat().st_size,
                        "path": str(features_path),
                        "sha256": _file_sha256_hex(features_path),
                    }
                },
                "split": "post_hf_extension",
                "symbol": "BTCUSDT",
            }
        ],
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
