from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    RoehubNativeEvaluationConfig,
    RoehubNativeSplitData,
    RoehubNativeTrainingConfig,
    UpstreamAlphaConfig,
    run_stage08e_roehub_native_training_v1,
    run_stage08f_roehub_native_evaluation_v1,
)


def test_stage08f_roehub_native_evaluation_records_research_verdict(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    alpha = UpstreamAlphaConfig(
        seed=53,
        batch_size=2,
        train_start=2,
        replay_capacity=32,
        target_update_freq=1,
        eps_start=1.0,
        eps_end=0.0,
        torch_num_threads=1,
    )
    training_config = RoehubNativeTrainingConfig(
        alpha=alpha,
        stage="08E",
        planned_episodes=3,
        validation_every_episodes=1,
        checkpoint_every_episodes=1,
        progress_emit_every_episodes=1,
        progress_emit_every_sec=3600,
        validation_max_sessions=2,
        device_policy="cpu_only_deterministic",
    )
    candidate = run_stage08e_roehub_native_training_v1(
        train_sequences=_session_features(3, alpha=alpha),
        validation_sequences=_session_features(2, alpha=alpha),
        dataset_dependency={
            "dataset_version": "hf_period_rebuild_current_trading",
            "sessionized_manifest_path": "/tmp/stage06_sessionized_manifest.json",
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
        config=training_config,
        generated_at_utc=datetime(2026, 6, 25, 9, 0, tzinfo=UTC),
        code_version={"git_head": "fixture"},
    )
    split = RoehubNativeSplitData(
        split_name="backtest",
        sequences=_session_features(4, alpha=alpha),
        symbols=("BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT"),
        signal_times_utc=(
            "2026-06-25T00:00:00Z",
            "2026-06-25T00:00:00Z",
            "2026-06-25T00:01:00Z",
            "2026-06-25T00:02:00Z",
        ),
        source_payload={"split_name": "backtest", "stage": "06"},
        volatility_scores=(0.1, 0.2, 0.3, 0.4),
    )

    manifest = run_stage08f_roehub_native_evaluation_v1(
        candidate_manifest=candidate,
        candidate_manifest_path=Path(candidate["candidate_manifest_path"]),
        candidate_manifest_sha256="b" * 64,
        test_split=split,
        backtest_split=split,
        output_root=tmp_path / "stage08f",
        run_id="stage08f_fixture",
        config=RoehubNativeEvaluationConfig(
            alpha=alpha,
            device_policy="cpu_only_deterministic",
            deterministic_random_seed=17,
        ),
        generated_at_utc=datetime(2026, 6, 25, 10, 0, tzinfo=UTC),
        code_version={"git_head": "fixture"},
    )

    scorecards = {row["policy_name"]: row for row in manifest["scorecards"]}
    filtered = scorecards["roehub_native_candidate_filtered_backtest"]
    raw = scorecards["roehub_native_candidate_raw_argmax_test_diagnostic"]

    assert manifest["stage"] == "08F"
    assert manifest["candidate_dependency"]["candidate_level"] == "roehub_native_candidate"
    assert manifest["candidate_dependency"]["checkpoint_name"] == "best"
    assert manifest["methodology"]["raw_argmax_acceptance"] is False
    assert manifest["research_candidate_save_allowed"] in {True, False}
    assert manifest["status"] in {"accepted_for_research", "blocked"}
    assert manifest["simulator_accounting_parity_fixture"]["passed"] is True
    assert manifest["safety"]["model_registry_write"] is False
    assert raw["raw_argmax_only"] is True
    assert raw["acceptance_backtest"] is False
    assert filtered["acceptance_backtest"] is True
    assert filtered["grouping"]["max_parallel_sessions"] == alpha.max_parallel_sessions
    assert filtered["metrics_by_volatility_bucket"]
    assert "deterministic_random_valid_action" in scorecards
    assert Path(manifest["artifact_hashes"]["scorecards"]["path"]).exists()
    assert Path(manifest["artifact_hashes"]["simulator_accounting_parity_fixture"]["path"]).exists()


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
