from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.rl_trading.stage08f_roehub_native_backtest_evaluation import main
from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    SESSIONIZED_DATASET_MANIFEST_KIND_V1,
    SESSIONIZED_SPLIT_ARTIFACT_KIND_V1,
    RoehubNativeTrainingConfig,
    UpstreamAlphaConfig,
    hash_json_payload_v1,
    run_stage08e_roehub_native_training_v1,
)


def test_stage08f_roehub_native_evaluation_cli_writes_sanitized_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    alpha = UpstreamAlphaConfig(
        seed=59,
        batch_size=2,
        train_start=2,
        replay_capacity=32,
        target_update_freq=1,
        eps_start=1.0,
        eps_end=0.0,
        torch_num_threads=1,
    )
    candidate = run_stage08e_roehub_native_training_v1(
        train_sequences=_feature_cube(3),
        validation_sequences=_feature_cube(2),
        dataset_dependency={
            "dataset_version": "hf_period_rebuild_current_trading",
            "sessionized_manifest_path": str(tmp_path / "stage06_sessionized_manifest.json"),
            "sessionized_manifest_sha256": "a" * 64,
            "sessionized_manifest_status": "accepted",
            "source_market": "binance:futures",
            "splits": {
                "train": {"selected_session_count": 3, "split": "train"},
                "validation": {"selected_session_count": 2, "split": "validation"},
            },
            "stage": "06",
        },
        output_root=tmp_path / "stage08e",
        run_id="stage08e_fixture",
        config=RoehubNativeTrainingConfig(
            alpha=alpha,
            stage="08E",
            planned_episodes=2,
            validation_every_episodes=1,
            checkpoint_every_episodes=1,
            progress_emit_every_episodes=1,
            progress_emit_every_sec=3600,
            validation_max_sessions=2,
            device_policy="cpu_only_deterministic",
        ),
    )
    stage06_manifest = _write_stage06_manifest(tmp_path)

    result = main(
        [
            "--candidate-manifest",
            str(candidate["candidate_manifest_path"]),
            "--expected-candidate-manifest-sha256",
            "",
            "--stage06-manifest-path",
            str(stage06_manifest),
            "--allow-fixture-hashes",
            "--output-root",
            str(tmp_path / "stage08f"),
            "--max-test-sessions",
            "3",
            "--max-backtest-sessions",
            "3",
            "--device-policy",
            "cpu_only_deterministic",
            "--torch-num-threads",
            "1",
            "--torch-num-interop-threads",
            "1",
            "--generated-at-utc",
            "2026-06-25T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["evaluation_manifest_path"]).read_text(encoding="utf-8"))

    assert result in {0, 2}
    assert payload["status"] in {"accepted_for_research", "blocked"}
    assert manifest["stage"] == "08F"
    assert manifest["safety"]["contains_raw_checkpoint_tensors"] is False
    assert manifest["safety"]["exchange_side_effects"] is False
    assert manifest["methodology"]["raw_argmax_acceptance"] is False
    assert manifest["dataset_dependency"]["test_split"]["selected_session_count"] == 3
    assert "roehub_native_candidate_filtered_backtest" in {
        row["policy_name"] for row in manifest["scorecards"]
    }
    assert "deterministic_random_valid_action" in {
        row["policy_name"] for row in manifest["scorecards"]
    }


def _write_stage06_manifest(tmp_path: Path) -> Path:
    output_root = tmp_path / "stage06"
    entries = []
    for split in ("test", "backtest"):
        artifact_root = output_root / "hf_period_rebuild_current_trading" / split / "BTCUSDT"
        artifact_root.mkdir(parents=True, exist_ok=True)
        features_path = artifact_root / "sessions.f32.npy"
        signal_path = artifact_root / "signal_time_ms.i64.npy"
        metadata_path = artifact_root / "metadata.json"
        features = np.stack([_session_features(idx) for idx in range(4)])
        np.save(features_path, features)
        np.save(
            signal_path,
            np.asarray(
                [
                    1_782_345_600_000,
                    1_782_345_660_000,
                    1_782_345_720_000,
                    1_782_345_780_000,
                ],
                dtype=np.int64,
            ),
        )
        metadata = {
            "sessions": [
                {
                    "signal_ts_open": f"2026-06-25T00:0{idx}:00Z",
                    "split": split,
                    "symbol": "BTCUSDT",
                    "volatility_score": 0.1 + float(idx) * 0.1,
                }
                for idx in range(4)
            ]
        }
        metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        files = {
            "features": _file_payload(features_path),
            "metadata": _file_payload(metadata_path),
            "signal_time_ms": _file_payload(signal_path),
        }
        entries.append(
            {
                "artifact_kind": SESSIONIZED_SPLIT_ARTIFACT_KIND_V1,
                "candidate_count": 4,
                "dataset_version": "hf_period_rebuild_current_trading",
                "deterministic_rebuild_hash": hash_json_payload_v1(
                    {"files": files, "split": split}
                ),
                "files": files,
                "schema_version": 1,
                "split": split,
                "symbol": "BTCUSDT",
            }
        )
    manifest = {
        "leakage_report": {"status": "accepted"},
        "manifest_kind": SESSIONIZED_DATASET_MANIFEST_KIND_V1,
        "market": "binance:futures",
        "split_artifacts": entries,
        "stage": "06",
        "status": "accepted",
        "total_sessions": 8,
    }
    path = tmp_path / "stage06_sessionized_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _file_payload(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    import hashlib

    return {"bytes": len(data), "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


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


def _feature_cube(session_count: int) -> np.ndarray:
    return np.stack([_session_features(idx) for idx in range(session_count)]).astype(np.float32)
