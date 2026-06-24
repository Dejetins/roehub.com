from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    HfOriginalEvaluationConfig,
    HfOriginalSplitData,
    HfOriginalTrainingConfig,
    UpstreamAlphaConfig,
    run_stage08c_hf_original_training_v1,
    run_stage08d_hf_original_evaluation_v1,
)


def test_stage08d_hf_original_evaluation_separates_raw_and_filtered_surfaces(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    alpha = UpstreamAlphaConfig(
        seed=43,
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
        train_sequences=_session_features(3, alpha=alpha),
        validation_sequences=_session_features(2, alpha=alpha),
        dataset_dependency={
            "dataset_dir": "/tmp/hf_fixture",
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
        generated_at_utc=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
        code_version={"git_head": "fixture"},
    )
    split = HfOriginalSplitData(
        split_name="test",
        sequences=_session_features(4, alpha=alpha),
        symbols=("BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT"),
        signal_times_utc=(
            "2026-06-24T00:00:00Z",
            "2026-06-24T00:00:00Z",
            "2026-06-24T00:01:00Z",
            "2026-06-24T00:02:00Z",
        ),
        source_payload={"split_name": "test", "sha256": "c" * 64},
    )

    manifest = run_stage08d_hf_original_evaluation_v1(
        candidate_manifest=candidate,
        candidate_manifest_path=Path(candidate["candidate_manifest_path"]),
        candidate_manifest_sha256="d" * 64,
        test_split=split,
        backtest_split=split,
        output_root=tmp_path / "stage08d",
        run_id="stage08d_fixture",
        config=HfOriginalEvaluationConfig(alpha=alpha, device_policy="cpu_only_deterministic"),
        generated_at_utc=datetime(2026, 6, 24, 13, 0, tzinfo=UTC),
        code_version={"git_head": "fixture"},
    )

    scorecards = {row["policy_name"]: row for row in manifest["scorecards"]}
    filtered = scorecards["hf_original_candidate_filtered_backtest"]
    raw = scorecards["hf_original_candidate_raw_argmax_test_diagnostic"]

    assert manifest["stage"] == "08D"
    assert manifest["candidate_dependency"]["checkpoint_name"] == "best"
    assert manifest["methodology"]["raw_argmax_acceptance"] is False
    assert manifest["status"] in {"accepted", "blocked"}
    assert raw["raw_argmax_only"] is True
    assert raw["acceptance_backtest"] is False
    assert filtered["acceptance_backtest"] is True
    assert filtered["grouping"]["max_parallel_sessions"] == alpha.max_parallel_sessions
    assert filtered["grouping"]["skipped_sessions_due_parallel_cap"] == 0
    assert filtered["filter_policy"]["selection_strategy"] == "advantage_based_filter"
    assert filtered["q_value_cache"]["misses"] > 0
    assert Path(manifest["artifact_hashes"]["scorecards"]["path"]).exists()
    assert Path(manifest["artifact_hashes"]["balance_curve"]["path"]).exists()


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
