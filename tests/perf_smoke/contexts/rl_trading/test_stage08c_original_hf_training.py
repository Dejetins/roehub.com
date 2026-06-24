from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.rl_trading.stage08c_original_hf_full_training_run import main
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1


def test_stage08c_original_hf_training_cli_writes_candidate_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "hf_dataset"
    _write_split_npz(dataset_dir / "train_data.npz", session_count=3)
    _write_split_npz(dataset_dir / "val_data.npz", session_count=2)

    result = main(
        [
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--allow-fixture-hashes",
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
    assert manifest["candidate_level"] == "hf_original_candidate"
    assert manifest["dataset_dependency"]["splits"]["train"]["hash_matches_expected"] is False
    assert manifest["metrics_summary"]["completed_episodes"] == 2
    assert manifest["metrics_summary"]["scripted_transition_sequence_used"] is False


def _write_split_npz(path: Path, *, session_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "_keys_map_": {f"fetcher_{idx}": ("BTCUSDT", idx) for idx in range(session_count)}
    }
    for idx in range(session_count):
        payload[f"fetcher_{idx}"] = _session_features(idx)
    np.savez(path, **payload)


def _session_features(seed: int) -> np.ndarray:
    features = np.zeros((150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    minute = np.arange(150, dtype=np.float32)
    close = 100.0 + float(seed) + minute * np.float32(0.02)
    values = {
        "close": close,
        "high": close + np.float32(0.05),
        "low": close - np.float32(0.05),
        "num_trades": np.full_like(close, 15.0 + float(seed)),
        "open": close - np.float32(0.01),
        "volume": np.full_like(close, 30.0 + float(seed)),
        "volume_weighted_average": close,
    }
    for name, value in values.items():
        features[:, FEATURE_NAMES_V1.index(name)] = value
    return features
