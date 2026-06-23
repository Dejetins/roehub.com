from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    FEATURE_NAMES_V1,
    STAGE07A_REQUIRED_ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    CandidateTrainingConfig,
    D3qnArchitectureConfig,
    PrioritizedReplayBuffer,
    PrioritizedReplayConfig,
    TrainingRunnerError,
    TrainingSmokeConfig,
    assert_stage02c_action_state_reward_compatibility_v1,
    assert_stage07a_trainable_source_v1,
    build_stage07a_transition_set_v1,
    build_stage07b_transition_set_v1,
    build_training_run_record_v1,
    d3qn_architecture_config_for_stage07b_v1,
    d3qn_architecture_config_for_transition_set_v1,
    run_d3qn_per_training_smoke_v1,
)


def test_stage02c_action_reward_state_contract_is_required() -> None:
    assert_stage02c_action_state_reward_compatibility_v1()

    assert ACTION_STATE_REWARD_CONTRACT_HASH_V1 == (
        STAGE07A_REQUIRED_ACTION_STATE_REWARD_CONTRACT_HASH_V1
    )


def test_stage07a_trainable_source_fails_closed_for_spot_and_bybit() -> None:
    with pytest.raises(TrainingRunnerError, match="blocked_not_training_source_v1"):
        assert_stage07a_trainable_source_v1(exchange="bybit", market_type="spot")


def test_prioritized_replay_sampling_and_priority_update_are_deterministic() -> None:
    buffer = PrioritizedReplayBuffer(
        observation_dim=3,
        config=PrioritizedReplayConfig(capacity=8),
        seed=11,
    )
    for idx in range(6):
        observation = np.asarray([idx, idx + 1, idx + 2], dtype=np.float32)
        buffer.add(
            observation=observation,
            action=idx % 4,
            reward=float(idx) / 10.0,
            next_observation=observation + 1.0,
            done=idx == 5,
            priority=float(idx + 1),
        )

    sample = buffer.sample(batch_size=3)
    assert sample.observations.shape == (3, 3)
    assert sample.actions.shape == (3,)
    assert sample.weights.shape == (3,)
    assert np.all(sample.weights <= 1.0)

    buffer.update_priorities(indices=sample.indices, td_errors=np.asarray([0.2, -0.4, 0.8]))
    repeated = PrioritizedReplayBuffer(
        observation_dim=3,
        config=PrioritizedReplayConfig(capacity=8),
        seed=11,
    )
    for idx in range(6):
        observation = np.asarray([idx, idx + 1, idx + 2], dtype=np.float32)
        repeated.add(
            observation=observation,
            action=idx % 4,
            reward=float(idx) / 10.0,
            next_observation=observation + 1.0,
            done=idx == 5,
            priority=float(idx + 1),
        )
    repeated_sample = repeated.sample(batch_size=3)

    assert sample.indices.tolist() == repeated_sample.indices.tolist()


def test_transition_set_is_deterministic_and_uses_action_reward_contract() -> None:
    config = TrainingSmokeConfig(seed=7, max_sessions=2, batch_size=4, update_steps=2)
    transitions = build_stage07a_transition_set_v1(
        session_features=_session_features(session_count=3),
        config=config,
    )
    repeated = build_stage07a_transition_set_v1(
        session_features=_session_features(session_count=3),
        config=config,
    )

    assert transitions.transition_count == 20
    assert transitions.episode_count == 2
    assert transitions.state_dim == (30 * len(FEATURE_NAMES_V1)) + (30 * 4) + 4
    assert transitions.action_counts == repeated.action_counts
    assert transitions.transition_set_hash() == repeated.transition_set_hash()
    assert sum(transitions.action_counts) == transitions.transition_count
    assert transitions.as_payload()["action_state_reward_contract_hash"] == (
        ACTION_STATE_REWARD_CONTRACT_HASH_V1
    )


def test_d3qn_architecture_and_run_record_hash_are_deterministic() -> None:
    config = TrainingSmokeConfig(seed=7, max_sessions=1, batch_size=4, update_steps=2)
    transitions = build_stage07a_transition_set_v1(
        session_features=_session_features(session_count=1),
        config=config,
    )
    architecture = d3qn_architecture_config_for_transition_set_v1(
        transitions=transitions,
        config=config,
    )
    record = build_training_run_record_v1(
        generated_at_utc=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
        config=config,
        architecture=architecture,
        dataset_manifest_path="/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json",
        dataset_manifest_sha256="61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08",
        transitions=transitions,
        device_payload={"selected_device": "cpu", "mps_available": True, "mps_built": True},
        metrics={"final_loss": 1.0},
        resource_usage={"rss_mb_after": 10.0, "wall_seconds": 1.0},
        artifact_hashes={"model_state_dict": {"sha256": "abc", "bytes": 1}},
    )
    repeated = build_training_run_record_v1(
        generated_at_utc=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
        config=config,
        architecture=D3qnArchitectureConfig(input_dim=transitions.state_dim),
        dataset_manifest_path="/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json",
        dataset_manifest_sha256="61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08",
        transitions=transitions,
        device_payload={"selected_device": "cpu", "mps_available": True, "mps_built": True},
        metrics={"final_loss": 1.0},
        resource_usage={"rss_mb_after": 10.0, "wall_seconds": 1.0},
        artifact_hashes={"model_state_dict": {"sha256": "abc", "bytes": 1}},
    )

    assert record["run_record_hash"] == repeated["run_record_hash"]
    assert record["config_hash"] == config.config_hash()
    assert record["architecture_hash"] == architecture.architecture_hash()
    assert record["safety"]["candidate_model_claim"] is False


def test_stage07b_candidate_config_and_transition_contract_are_deterministic() -> None:
    config = CandidateTrainingConfig(
        seed=17,
        batch_size=4,
        planned_training_steps=3,
        progress_emit_every_steps=1,
        checkpoint_every_steps=2,
        validation_every_steps=1,
        replay=PrioritizedReplayConfig(capacity=64),
        hidden_dims=(32, 32),
    )
    transitions = build_stage07b_transition_set_v1(
        session_features=_session_features(session_count=3),
        config=config,
    )
    repeated = build_stage07b_transition_set_v1(
        session_features=_session_features(session_count=3),
        config=config,
    )
    architecture = d3qn_architecture_config_for_stage07b_v1(
        transitions=transitions,
        config=config,
    )

    assert transitions.transition_count == 30
    assert transitions.transition_set_hash() == repeated.transition_set_hash()
    assert config.as_payload()["config_id"] == "roehub_stage07b_candidate_training_config_v1"
    assert config.config_hash() == CandidateTrainingConfig(
        seed=17,
        batch_size=4,
        planned_training_steps=3,
        progress_emit_every_steps=1,
        checkpoint_every_steps=2,
        validation_every_steps=1,
        replay=PrioritizedReplayConfig(capacity=64),
        hidden_dims=(32, 32),
    ).config_hash()
    assert architecture.as_payload()["hidden_dims"] == [32, 32]


def test_torch_d3qn_per_update_shapes_when_optional_extra_is_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = TrainingSmokeConfig(
        seed=7,
        max_sessions=2,
        batch_size=4,
        update_steps=2,
        torch_num_threads=1,
    )
    transitions = build_stage07a_transition_set_v1(
        session_features=_session_features(session_count=2),
        config=config,
    )

    record = run_d3qn_per_training_smoke_v1(
        transitions=transitions,
        dataset_manifest_path="/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json",
        dataset_manifest_sha256="61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08",
        output_root=tmp_path,
        config=config,
        generated_at_utc=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    )

    shapes = record["metrics"]["batch_shapes"]
    assert shapes["observations_shape"] == [4, transitions.state_dim]
    assert shapes["q_values_shape"] == [4, 4]
    assert shapes["target_q_shape"] == [4]
    assert shapes["td_errors_shape"] == [4]
    assert record["artifact_hashes"]["model_state_dict"]["bytes"] > 0
    assert record["safety"]["smoke_model_state_is_candidate"] is False


def _session_features(*, session_count: int) -> np.ndarray:
    features = np.zeros((session_count, 150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    minute = np.arange(150, dtype=np.float32)
    for session_idx in range(session_count):
        base = 100.0 + float(session_idx)
        close = base + (minute * 0.01) + np.sin(minute / 9.0).astype(np.float32) * 0.05
        open_ = close - 0.01
        high = close + 0.05
        low = close - 0.05
        values = {
            "close": close,
            "high": high,
            "low": low,
            "num_trades": np.full_like(close, 10.0 + session_idx),
            "open": open_,
            "volume": np.full_like(close, 25.0),
            "volume_weighted_average": close,
        }
        for name, value in values.items():
            features[session_idx, :, FEATURE_NAMES_V1.index(name)] = value
    return features
