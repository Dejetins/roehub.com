from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading.stage07b_full_candidate_training_run import main as stage07b_main
from scripts.rl_trading.stage08_roehub_backtest_evaluation import main as stage08_main
from trading.contexts.rl_trading.domain import FEATURE_NAMES_V1


def test_stage08_cli_evaluates_tiny_stage07b_candidate(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    manifest_path = _write_stage06_fixture(tmp_path)
    stage07b_result = stage07b_main(
        [
            "run",
            "--sessionized-manifest",
            str(manifest_path),
            "--expected-sessionized-manifest-sha256",
            _file_sha256_hex(manifest_path),
            "--output-root",
            str(tmp_path / "stage07b_runs"),
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
            "2026-06-24T12:00:00Z",
        ]
    )
    stage07b_payload = json.loads(capsys.readouterr().out)
    candidate_manifest_path = Path(stage07b_payload["candidate_manifest_path"])

    stage08_result = stage08_main(
        [
            "--candidate-manifest",
            str(candidate_manifest_path),
            "--expected-candidate-manifest-sha256",
            _file_sha256_hex(candidate_manifest_path),
            "--sessionized-manifest",
            str(manifest_path),
            "--expected-sessionized-manifest-sha256",
            _file_sha256_hex(manifest_path),
            "--output-root",
            str(tmp_path / "stage08_runs"),
            "--split",
            "test",
            "--max-sessions",
            "2",
            "--torch-num-threads",
            "1",
            "--torch-num-interop-threads",
            "1",
            "--generated-at-utc",
            "2026-06-24T12:01:00Z",
        ]
    )
    stage08_payload = json.loads(capsys.readouterr().out)
    evaluation_manifest = json.loads(
        Path(stage08_payload["evaluation_manifest_path"]).read_text(encoding="utf-8")
    )

    assert stage07b_result == 0
    assert stage08_result in {0, 2}
    assert stage08_payload["selected_session_count"] == 2
    assert stage08_payload["selected_symbol_count"] == 1
    assert evaluation_manifest["stage"] == "08"
    assert evaluation_manifest["dataset_dependency"]["manifest_sha256"] == _file_sha256_hex(
        manifest_path
    )
    assert len(evaluation_manifest["scorecards"]) == 5
    assert evaluation_manifest["parity_fixture"]["passed"] is True
    assert evaluation_manifest["safety"]["model_registry_write"] is False


def _write_stage06_fixture(tmp_path: Path) -> Path:
    split_artifacts = []
    for split in ("train", "validation", "test"):
        artifact_root = tmp_path / "hf_period_rebuild_current_trading" / split / "BTCUSDT"
        artifact_root.mkdir(parents=True)
        features_path = artifact_root / "sessions.f32.npy"
        metadata_path = artifact_root / "metadata.json"
        np.save(features_path, _session_features())
        metadata_path.write_text(
            json.dumps(
                {
                    "sessions": [
                        {"signal_ts_open": "2026-06-01T00:00:00Z"},
                        {"signal_ts_open": "2026-06-01T01:00:00Z"},
                    ]
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        split_artifacts.append(
            {
                "dataset_version": "hf_period_rebuild_current_trading",
                "files": {
                    "features": {
                        "bytes": features_path.stat().st_size,
                        "path": str(features_path),
                        "sha256": _file_sha256_hex(features_path),
                    },
                    "metadata": {
                        "bytes": metadata_path.stat().st_size,
                        "path": str(metadata_path),
                        "sha256": _file_sha256_hex(metadata_path),
                    },
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
        close = 100.0 + session_idx + minute * 0.08
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
