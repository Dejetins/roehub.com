from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    STAGE08C_CANDIDATE_LEVEL_V1,
    HfOriginalTrainingConfig,
    UpstreamAlphaConfig,
    run_stage08c_hf_original_training_v1,
)


def test_stage08c_hf_original_training_writes_best_final_and_manifest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    alpha = UpstreamAlphaConfig(
        seed=31,
        batch_size=2,
        train_start=2,
        replay_capacity=32,
        target_update_freq=1,
        eps_start=1.0,
        eps_end=0.0,
        torch_num_threads=1,
    )
    config = HfOriginalTrainingConfig(
        alpha=alpha,
        planned_episodes=3,
        validation_every_episodes=1,
        checkpoint_every_episodes=2,
        progress_emit_every_episodes=1,
        progress_emit_every_sec=3600,
        validation_max_sessions=2,
        device_policy="cpu_only_deterministic",
    )
    dataset_dependency = {
        "dataset_dir": "/opt/roehub/state/rl_trading/hf_reproducibility/dataset",
        "source_market": "binance:futures",
        "splits": {
            "train": {"sha256": "a" * 64, "selected_session_count": 3},
            "validation": {"sha256": "b" * 64, "selected_session_count": 2},
        },
        "stage": "04",
    }

    manifest = run_stage08c_hf_original_training_v1(
        train_sequences=_session_features(3, alpha=alpha),
        validation_sequences=_session_features(2, alpha=alpha),
        dataset_dependency=dataset_dependency,
        output_root=tmp_path,
        run_id="stage08c_fixture",
        config=config,
        generated_at_utc=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
        code_version={"git_head": "fixture"},
    )

    progress_path = Path(manifest["artifact_hashes"]["progress_jsonl"]["path"])
    events = [line for line in progress_path.read_text(encoding="utf-8").splitlines() if line]

    assert manifest["candidate_level"] == STAGE08C_CANDIDATE_LEVEL_V1
    assert manifest["candidate_level"] == "hf_original_candidate"
    assert manifest["status"] == "completed"
    assert manifest["metrics_summary"]["completed_episodes"] == 3
    assert manifest["metrics_summary"]["completed_env_steps"] == 30
    assert manifest["metrics_summary"]["scripted_transition_sequence_used"] is False
    assert manifest["metrics_summary"]["training_used_environment_rollout"] is True
    assert manifest["checkpoint_policy"]["best_checkpoint"] == "best.pth"
    assert manifest["checkpoint_policy"]["default_evaluation_checkpoint"] == "best"
    assert manifest["next_stage_handoff"]["stage08d_allowed"] is True
    assert Path(manifest["artifact_hashes"]["best_checkpoint"]["path"]).exists()
    assert Path(manifest["artifact_hashes"]["final_checkpoint"]["path"]).exists()
    assert events[-1].find('"status":"completed"') != -1


def _session_features(session_count: int, *, alpha: UpstreamAlphaConfig) -> np.ndarray:
    features = np.zeros(
        (session_count, alpha.full_seq_len, len(FEATURE_NAMES_V1)),
        dtype=np.float32,
    )
    minute = np.arange(alpha.full_seq_len, dtype=np.float32)
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
    return features
