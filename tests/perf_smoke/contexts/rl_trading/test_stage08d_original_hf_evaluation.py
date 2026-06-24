from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.rl_trading.stage08d_original_hf_backtest_evaluation import main
from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    HfOriginalTrainingConfig,
    UpstreamAlphaConfig,
    run_stage08c_hf_original_training_v1,
)


def test_stage08d_original_hf_evaluation_cli_writes_sanitized_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("torch")
    alpha = UpstreamAlphaConfig(
        seed=47,
        batch_size=2,
        train_start=2,
        replay_capacity=32,
        target_update_freq=1,
        eps_start=1.0,
        eps_end=0.0,
        torch_num_threads=1,
    )
    training_config = HfOriginalTrainingConfig(
        alpha=alpha,
        planned_episodes=2,
        validation_every_episodes=1,
        checkpoint_every_episodes=1,
        progress_emit_every_episodes=1,
        progress_emit_every_sec=3600,
        validation_max_sessions=2,
        device_policy="cpu_only_deterministic",
    )
    candidate = run_stage08c_hf_original_training_v1(
        train_sequences=np.stack([_session_features(idx) for idx in range(3)]),
        validation_sequences=np.stack([_session_features(idx) for idx in range(2)]),
        dataset_dependency={
            "dataset_dir": str(tmp_path / "hf_dataset"),
            "source_market": "binance:futures",
            "splits": {
                "train": {"sha256": "a" * 64, "selected_session_count": 3},
                "validation": {"sha256": "b" * 64, "selected_session_count": 2},
            },
            "stage": "04",
        },
        output_root=tmp_path / "stage08c",
        run_id="stage08c_fixture",
        config=training_config,
    )
    dataset_dir = tmp_path / "hf_dataset"
    _write_split_npz(dataset_dir / "test_data.npz", session_count=3)
    _write_split_npz(dataset_dir / "backtest_data.npz", session_count=3)

    result = main(
        [
            "--candidate-manifest",
            str(candidate["candidate_manifest_path"]),
            "--expected-candidate-manifest-sha256",
            "",
            "--dataset-dir",
            str(dataset_dir),
            "--allow-fixture-hashes",
            "--output-root",
            str(tmp_path / "stage08d"),
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
            "2026-06-24T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(
        Path(payload["evaluation_manifest_path"]).read_text(encoding="utf-8")
    )

    assert result in {0, 2}
    assert payload["status"] in {"accepted", "blocked"}
    assert manifest["stage"] == "08D"
    assert manifest["safety"]["contains_raw_checkpoint_tensors"] is False
    assert manifest["safety"]["exchange_side_effects"] is False
    assert manifest["methodology"]["raw_argmax_acceptance"] is False
    assert "hf_original_candidate_filtered_backtest" in {
        row["policy_name"] for row in manifest["scorecards"]
    }


def _write_split_npz(path: Path, *, session_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "_keys_map_": {
            f"fetcher_{idx}": ("BTCUSDT", np.datetime64(f"2026-06-24T00:0{idx}:00"))
            for idx in range(session_count)
        }
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
