from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    HfOriginalEvaluationConfig,
    HfOriginalSplitData,
    HfOriginalTrainingConfig,
    NormalizationStats,
    QValueCache,
    UpstreamAlphaConfig,
    alpha_with_evaluation_overrides_v1,
    compute_train_only_normalization_stats_v1,
    evaluate_stage08d_baseline_backtest_v1,
    evaluate_stage08d_grouped_backtest_v1,
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
    assert filtered["grouping"]["scheduling_rule"] == "rolling_open_sessions"
    assert filtered["grouping"]["skipped_sessions_due_parallel_cap"] == 2
    assert filtered["filter_policy"]["selection_strategy"] == "advantage_based_filter"
    assert filtered["q_value_cache"]["misses"] > 0
    assert Path(manifest["artifact_hashes"]["scorecards"]["path"]).exists()
    assert Path(manifest["artifact_hashes"]["balance_curve"]["path"]).exists()


def test_alpha_evaluation_overrides_do_not_change_training_hyperparameters() -> None:
    base = UpstreamAlphaConfig(learning_rate=0.0003, batch_size=32, replay_capacity=1000)
    overrides = UpstreamAlphaConfig(
        learning_rate=0.99,
        batch_size=2,
        replay_capacity=8,
        long_action_threshold=0.02,
        short_action_threshold=0.003,
        close_action_threshold=0.004,
        use_risk_management=True,
        stop_loss=0.015,
        take_profit=0.035,
        trailing_stop=0.012,
        max_parallel_sessions=3,
        position_fraction=0.4,
    )

    merged = alpha_with_evaluation_overrides_v1(base, overrides)

    assert merged.learning_rate == base.learning_rate
    assert merged.batch_size == base.batch_size
    assert merged.replay_capacity == base.replay_capacity
    assert merged.long_action_threshold == 0.02
    assert merged.use_risk_management is True
    assert merged.max_parallel_sessions == 3
    assert merged.position_fraction == 0.4


def test_baseline_backtest_applies_risk_management_forced_close() -> None:
    alpha = UpstreamAlphaConfig(
        initial_balance=100.0,
        use_risk_management=True,
        stop_loss=0.5,
        take_profit=0.0001,
        trailing_stop=0.5,
        position_fraction=1.0,
    )
    split = HfOriginalSplitData(
        split_name="backtest",
        sequences=_session_features(1, alpha=alpha),
        symbols=("BTCUSDT",),
        signal_times_utc=("2026-06-26T00:00:00Z",),
        source_payload={"split_name": "backtest", "sha256": "e" * 64},
    )

    scorecard = evaluate_stage08d_baseline_backtest_v1(
        split=split,
        config=HfOriginalEvaluationConfig(alpha=alpha),
        policy_name="always_long_fixture",
        fixed_action_id=1,
    )

    risk_management = scorecard["risk_management"]
    assert risk_management["use_risk_management"] is True
    assert risk_management["reason_counts"]["risk_management_take_profit_forced_close"] > 0
    assert scorecard["closed_trades"] > 0


def test_grouped_backtest_uses_rolling_open_sessions_and_shared_balance() -> None:
    alpha = UpstreamAlphaConfig(
        initial_balance=100.0,
        position_fraction=0.5,
        max_parallel_sessions=2,
        agent_session_len=10,
    )
    split = HfOriginalSplitData(
        split_name="backtest",
        sequences=_session_features(4, alpha=alpha),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"),
        signal_times_utc=(
            "2026-06-26T00:00:00Z",
            "2026-06-26T00:00:00Z",
            "2026-06-26T00:01:00Z",
            "2026-06-26T00:10:00Z",
        ),
        source_payload={"split_name": "backtest", "sha256": "f" * 64},
    )

    scorecard, _ = evaluate_stage08d_grouped_backtest_v1(
        split=split,
        normalization_stats=_fixture_normalization_stats(alpha),
        agent=cast(Any, _AlwaysLongAgent()),
        config=HfOriginalEvaluationConfig(alpha=alpha),
    )

    grouping = scorecard["grouping"]
    assert grouping["scheduling_rule"] == "rolling_open_sessions"
    assert grouping["selected_session_indices"] == [0, 1, 3]
    assert grouping["skipped_sessions_due_parallel_cap"] == 1
    assert scorecard["starting_equity_quote"] == 100.0
    assert scorecard["position_fraction_application"] == "shared_balance_position_fraction"
    sizing = scorecard["position_sizing_samples"]
    assert sizing[0]["position_size_quote"] == 50.0
    assert sizing[1]["position_size_quote"] > 50.0
    assert scorecard["shared_balance_final_quote"] > scorecard["shared_balance_initial_quote"]


def test_grouped_backtest_filters_unmasked_q_before_environment_action_mask() -> None:
    alpha = UpstreamAlphaConfig(
        initial_balance=100.0,
        position_fraction=1.0,
        max_parallel_sessions=1,
        agent_session_len=10,
    )
    split = HfOriginalSplitData(
        split_name="backtest",
        sequences=_session_features(1, alpha=alpha),
        symbols=("BTCUSDT",),
        signal_times_utc=("2026-06-26T00:00:00Z",),
        source_payload={"split_name": "backtest", "sha256": "0" * 64},
    )

    scorecard, balance_curve = evaluate_stage08d_grouped_backtest_v1(
        split=split,
        normalization_stats=_fixture_normalization_stats(alpha),
        agent=cast(Any, _AlwaysLongAgent()),
        config=HfOriginalEvaluationConfig(alpha=alpha),
    )

    assert scorecard["requested_action_counts"]["open_long"] == scorecard["decisions_count"]
    assert scorecard["action_counts"]["open_long"] == 1
    assert scorecard["action_counts"]["close"] == 1
    assert scorecard["action_counts"]["hold"] > 0
    assert scorecard["reward_sum"] == 0.0
    assert scorecard["backtest_reporting_reward_sum"] == 0.0
    assert scorecard["training_reward_sum"] != 0.0
    assert balance_curve[1]["unmasked_filter_action_id"] == 1
    assert balance_curve[1]["filter_selected_action_id"] == 1
    assert balance_curve[1]["effective_action_id"] == 0
    assert balance_curve[1]["backtest_reporting_reward"] == 0.0
    assert "training_reward" in balance_curve[1]


class _AlwaysLongAgent:
    def __init__(self) -> None:
        self.q_value_cache = QValueCache()

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def predict_ensemble(
        self,
        state: np.ndarray,
        *,
        n_samples: int,
        cache_key: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_q_values(state), np.zeros((4,), dtype=np.float32)


def _fixture_normalization_stats(alpha: UpstreamAlphaConfig) -> NormalizationStats:
    return compute_train_only_normalization_stats_v1(
        _session_features(3, alpha=alpha),
        config=alpha,
    )


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
