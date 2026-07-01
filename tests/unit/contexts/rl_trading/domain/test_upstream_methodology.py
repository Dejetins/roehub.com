from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    FEATURE_NAMES_V1,
    RL_ACTION_COUNT_V1,
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    UPSTREAM_METHODOLOGY_PARITY_ID_V1,
    FilteredBacktestPolicy,
    QValueCache,
    RlTrainingState,
    SumTreePrioritizedReplayBuffer,
    TorchD3qnPerAgent,
    UpstreamAlphaConfig,
    UpstreamTradingEnvironment,
    apply_upstream_normalization_v1,
    build_tiny_stage08b_session_features_v1,
    build_torch_cnn_dueling_q_network_v1,
    build_upstream_state_v1,
    compute_train_only_normalization_stats_v1,
    copy_torch_cnn_dueling_state_v1,
    run_stage08b_core_smoke_v1,
    select_checkpoint_policy_v1,
    torch_cnn_dueling_forward_v1,
    valid_upstream_training_actions_v1,
)
from trading.contexts.rl_trading.domain.upstream_methodology import (
    _release_device_cache_if_needed,
)


def test_upstream_alpha_config_exposes_architecture_and_parity_literals() -> None:
    config = UpstreamAlphaConfig()
    payload = config.as_payload()

    assert payload["architecture_id"] == UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1
    assert payload["architecture_id"] == "roehub_d3qn_cnn_dueling_v1"
    assert payload["methodology_parity_id"] == UPSTREAM_METHODOLOGY_PARITY_ID_V1
    assert payload["methodology_parity_id"] == "upstream_methodology_parity"
    assert payload["cnn_maps"] == [32, 64, 128]
    assert payload["full_seq_len"] == 150
    assert payload["pre_signal_len"] == 90
    assert payload["agent_history_len"] == 30
    assert payload["agent_session_len"] == 10
    assert payload["input_history_len"] == 29
    assert payload["batch_size"] == 16
    assert payload["train_start"] == 10_000
    assert payload["long_action_threshold"] == 0.012695
    assert payload["short_action_threshold"] == 0.009902
    assert payload["close_action_threshold"] == 0.001141
    assert payload["max_parallel_sessions"] == 2
    assert payload["position_fraction"] == 0.5
    assert payload["use_risk_management"] is False
    assert config.state_dim == (29 * len(FEATURE_NAMES_V1)) + 4 + (3 * 4)


def test_upstream_alpha_config_hash_changes_for_backtest_overrides() -> None:
    base = UpstreamAlphaConfig()
    tuned = UpstreamAlphaConfig(
        long_action_threshold=0.02,
        short_action_threshold=0.003,
        close_action_threshold=0.004,
        use_risk_management=True,
        stop_loss=0.015,
        take_profit=0.035,
        trailing_stop=0.012,
    )

    assert base.config_hash() != tuned.config_hash()
    assert tuned.as_payload()["use_risk_management"] is True
    assert tuned.as_payload()["stop_loss"] == 0.015


def test_normalization_stats_are_train_only_without_validation_leakage() -> None:
    config = UpstreamAlphaConfig()
    train = build_tiny_stage08b_session_features_v1(session_count=2, config=config)
    validation = train.copy()
    validation[:, :, FEATURE_NAMES_V1.index("volume")] *= 1000.0
    validation[:, :, FEATURE_NAMES_V1.index("num_trades")] *= 500.0

    train_only = compute_train_only_normalization_stats_v1(train, config=config)
    leaked = compute_train_only_normalization_stats_v1(
        np.concatenate([train, validation], axis=0),
        config=config,
    )
    validation_window = validation[0, : config.agent_history_len]
    normalized = apply_upstream_normalization_v1(
        validation_window,
        train_only,
        config=config,
    )

    assert train_only.source_split == "train"
    assert train_only.stats_hash() != leaked.stats_hash()
    assert train_only.means["volume"] != leaked.means["volume"]
    assert normalized.shape == (config.input_history_len, len(FEATURE_NAMES_V1))


def test_state_builder_uses_normalized_history_extras_and_short_action_history() -> None:
    config = UpstreamAlphaConfig(initial_balance=100.0)
    features = build_tiny_stage08b_session_features_v1(session_count=1, config=config)
    stats = compute_train_only_normalization_stats_v1(features, config=config)
    state = build_upstream_state_v1(
        session=features[0],
        step_idx=2,
        action_history=[1, 3],
        training_state=RlTrainingState(balance=100.0, position_side="long", entry_price=100.0),
        normalization_stats=stats,
        config=config,
    )

    history_end = config.flat_history_size
    extras = state[history_end : history_end + 4]
    encoded_actions = state[history_end + 4 :]

    assert state.shape == (config.state_dim,)
    assert extras[0] == 1.0
    assert encoded_actions.shape == (config.action_history_len * RL_ACTION_COUNT_V1,)
    assert encoded_actions[0:4].sum() == 0.0
    assert encoded_actions[4 + 1] == 1.0
    assert encoded_actions[8 + 3] == 1.0


def test_environment_masks_no_pyramiding_and_forces_last_step_close() -> None:
    config = UpstreamAlphaConfig(initial_balance=100.0)
    features = build_tiny_stage08b_session_features_v1(session_count=1, config=config)
    stats = compute_train_only_normalization_stats_v1(features, config=config)
    env = UpstreamTradingEnvironment(sequences=features, normalization_stats=stats, config=config)
    env.reset(forced_index=0)

    _, _, done, _, first = env.step(1)
    _, _, done, _, second = env.step(2)
    while env.step_idx < config.agent_session_len - 1:
        env.step(0)
    _, _, done, _, last = env.step(0)

    assert done is True
    assert first["effective_action_id"] == 1
    assert second["masked_action_id"] == 0
    assert last["effective_action_id"] == 3
    assert last["audit_reason"] == "last_step_forced_close"
    assert valid_upstream_training_actions_v1(position_side=None, is_last_step=True) == (0,)


def test_sumtree_per_sampling_and_priority_updates_are_deterministic() -> None:
    buffer = SumTreePrioritizedReplayBuffer(
        capacity=8,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100,
        epsilon=1e-6,
        seed=17,
    )
    for idx in range(6):
        state = np.asarray([idx, idx + 1], dtype=np.float32)
        buffer.add(
            state=state,
            action=idx % 4,
            reward=float(idx) / 10.0,
            next_state=state + 1.0,
            done=idx == 5,
        )

    sample = buffer.sample(batch_size=3)
    before_total = float(buffer.tree[0])
    buffer.update_priorities(sample.tree_indices, np.asarray([0.1, 0.5, 1.0]))
    payload = buffer.state_payload()
    restored = SumTreePrioritizedReplayBuffer(
        capacity=8,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100,
        epsilon=1e-6,
        seed=17,
    )
    restored.restore_state_payload(payload)

    assert sample.states.shape == (3, 2)
    assert sample.weights.shape == (3,)
    assert float(buffer.tree[0]) != before_total
    assert len(restored) == len(buffer)
    assert np.array_equal(restored.tree, buffer.tree)


def test_release_device_cache_only_flushes_mps_cache() -> None:
    calls: list[str] = []

    class FakeMps:
        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    class FakeTorch:
        mps = FakeMps()

    _release_device_cache_if_needed(torch=FakeTorch(), device_type="cpu")
    assert calls == []

    _release_device_cache_if_needed(torch=FakeTorch(), device_type="mps")
    assert calls == ["synchronize", "empty_cache"]


def test_filtered_policy_rejects_weak_actions_and_records_cache_stats() -> None:
    config = UpstreamAlphaConfig()
    policy = FilteredBacktestPolicy.from_config(config)
    cache = QValueCache()

    first = cache.get_or_compute(
        ("BTCUSDT", "2026-06-24T00:00:00Z"),
        lambda: np.asarray([1.0, 1.001, 1.002, 1.0005], dtype=np.float32),
    )
    repeated = cache.get_or_compute(
        ("BTCUSDT", "2026-06-24T00:00:00Z"),
        lambda: np.asarray([99.0, 99.0, 99.0, 99.0], dtype=np.float32),
    )
    decision = policy.select_from_q_values(first)

    assert np.array_equal(first, repeated)
    assert decision.requested_action_id == 2
    assert decision.effective_action_id == 0
    assert decision.rejected is True
    assert decision.rejection_reason == "weak_advantage_threshold"
    assert policy.stats_payload()["rejection_counts"] == {"weak_advantage_threshold": 1}
    assert cache.stats_payload() == {"cache_entries": 1, "hits": 1, "misses": 1}


def test_ensemble_policy_rejects_only_weak_advantage_with_high_uncertainty() -> None:
    config = UpstreamAlphaConfig()
    policy = FilteredBacktestPolicy.from_config(config, selection_strategy="ensemble_q_filter")

    confident_decision = policy.select_from_q_values(
        np.asarray([1.0, 1.2, 1.0, 1.0], dtype=np.float32),
        q_std=np.asarray([0.0, 0.2, 0.0, 0.0], dtype=np.float32),
    )
    weak_uncertain_decision = policy.select_from_q_values(
        np.asarray([1.0, 1.001, 1.0, 1.0], dtype=np.float32),
        q_std=np.asarray([0.0, 0.2, 0.0, 0.0], dtype=np.float32),
    )

    assert confident_decision.requested_action_id == 1
    assert confident_decision.effective_action_id == 1
    assert confident_decision.rejection_reason is None
    assert weak_uncertain_decision.requested_action_id == 1
    assert weak_uncertain_decision.effective_action_id == 0
    assert (
        weak_uncertain_decision.rejection_reason
        == "weak_advantage_threshold+high_ensemble_uncertainty"
    )


def test_checkpoint_policy_selects_best_and_keeps_final_diagnostic() -> None:
    selected = select_checkpoint_policy_v1(
        [
            {"Validation_mean_pnl": -1.0, "completed_training_steps": 10},
            {"Validation_mean_pnl": 2.5, "completed_training_steps": 20},
            {"Validation_mean_pnl": 1.5, "completed_training_steps": 30},
        ]
    )

    assert selected["best_checkpoint"] == "best.pth"
    assert selected["final_checkpoint"] == "final.pth"
    assert selected["best_step"] == 20
    assert selected["default_evaluation_checkpoint"] == "best"
    assert selected["final_is_diagnostic_unless_selected"] is True


def test_torch_cnn_dueling_network_forward_and_target_sync_when_available() -> None:
    torch = pytest.importorskip("torch")
    config = UpstreamAlphaConfig(torch_num_threads=1)
    device = torch.device("cpu")
    policy = build_torch_cnn_dueling_q_network_v1(torch=torch, config=config, device=device)
    target = build_torch_cnn_dueling_q_network_v1(torch=torch, config=config, device=device)
    copy_torch_cnn_dueling_state_v1(target=target, source=policy)

    states = torch.zeros(2, config.state_dim, dtype=torch.float32, device=device)
    q_values = torch_cnn_dueling_forward_v1(network=policy, states=states, config=config)
    dropout_count = sum(1 for module in policy.modules() if isinstance(module, torch.nn.Dropout))

    assert q_values.shape == (2, 4)
    assert dropout_count >= 4
    assert target.state_dict().keys() == policy.state_dict().keys()


def test_stage08b_rollout_smoke_uses_epsilon_agent_not_scripted_transitions(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config = UpstreamAlphaConfig(
        seed=19,
        initial_balance=100.0,
        batch_size=2,
        train_start=2,
        target_update_freq=1,
        replay_capacity=64,
        eps_start=1.0,
        eps_end=1.0,
        torch_num_threads=1,
    )
    features = build_tiny_stage08b_session_features_v1(session_count=3, config=config)
    report = run_stage08b_core_smoke_v1(
        session_features=features,
        output_root=tmp_path,
        config=config,
        episodes=2,
        generated_at_utc=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
    )

    metrics = report["metrics"]
    assert report["architecture_id"] == "roehub_d3qn_cnn_dueling_v1"
    assert report["action_state_reward_contract_hash"] == ACTION_STATE_REWARD_CONTRACT_HASH_V1
    assert metrics["scripted_transition_sequence_used"] is False
    assert metrics["selection_mode_counts"]["epsilon_random"] > 0
    assert metrics["learn_update_count"] > 0
    assert metrics["target_sync_count"] > 0
    assert Path(str(report["report_path"])).exists()


def test_torch_agent_train_start_and_gradient_clip_when_available() -> None:
    pytest.importorskip("torch")
    config = UpstreamAlphaConfig(
        seed=23,
        initial_balance=100.0,
        batch_size=2,
        train_start=3,
        target_update_freq=1,
        replay_capacity=16,
        eps_start=0.0,
        eps_end=0.0,
        torch_num_threads=1,
    )
    features = build_tiny_stage08b_session_features_v1(session_count=1, config=config)
    stats = compute_train_only_normalization_stats_v1(features, config=config)
    env = UpstreamTradingEnvironment(sequences=features, normalization_stats=stats, config=config)
    agent = TorchD3qnPerAgent(config=config)
    state, _ = env.reset(forced_index=0)

    for _ in range(2):
        action = agent.select_action(state, training=True, valid_actions=env.valid_actions())
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state
    assert agent.learn() is None

    action = agent.select_action(state, training=True, valid_actions=env.valid_actions())
    next_state, reward, done, _, _ = env.step(action)
    agent.store_experience(state, action, reward, next_state, done)
    result = agent.learn()

    assert result is not None
    assert result.gradient_norm_before_clip >= 0.0
    assert result.target_synced is True
