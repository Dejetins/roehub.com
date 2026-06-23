from __future__ import annotations

import importlib
import math
import os
import platform
import random
import resource
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .action_state_reward_contract import (
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RL_ACTION_COUNT_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
    build_state_extras_v1,
    encode_action_history_v1,
)
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_DTYPE_V1, FEATURE_NAMES_V1
from .raw_feature_dataset import (
    BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
    hash_json_payload_v1,
    hash_ndarray_payload_v1,
    render_raw_feature_json_payload_v1,
    training_source_gate_payload_v1,
)
from .sessionized_dataset import (
    SESSIONIZED_AGENT_HISTORY_LEN_V1,
    SESSIONIZED_AGENT_SESSION_LEN_V1,
    SESSIONIZED_FULL_SEQ_LEN_V1,
    SESSIONIZED_PRE_SIGNAL_LEN_V1,
)

STAGE07A_TRAINING_RUN_RECORD_SCHEMA_VERSION_V1 = 1
STAGE07A_TRAINING_RUN_KIND_V1 = "rl_trading_stage07a_training_runner_smoke"
STAGE07A_RUNTIME_ARTIFACT_ROOT_V1 = "/opt/roehub/state/rl_trading/"
STAGE07A_REQUIRED_ACTION_STATE_REWARD_CONTRACT_HASH_V1 = (
    "255d765b9474620671167412465fc55a058c0233d5da242a276143fb6816b557"
)
D3QN_ARCHITECTURE_ID_V1 = "roehub_d3qn_mlp_v1"
PER_REPLAY_BUFFER_ID_V1 = "roehub_per_replay_v1"
TRAINING_SMOKE_CONFIG_ID_V1 = "roehub_stage07a_training_smoke_config_v1"
STAGE07A_TRANSITION_SET_KIND_V1 = "rl_trading_stage07a_transition_set"

DevicePolicy = Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"]
RunStatus = Literal["accepted", "blocked"]


class TrainingRunnerError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class D3qnArchitectureConfig:
    input_dim: int
    action_count: int = RL_ACTION_COUNT_V1
    hidden_dims: tuple[int, ...] = (64, 64)
    value_hidden_dim: int = 64
    advantage_hidden_dim: int = 64
    activation: str = "relu"
    dtype: str = FEATURE_DTYPE_V1

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, "input_dim")
        if self.action_count != RL_ACTION_COUNT_V1:
            raise TrainingRunnerError(reason="unexpected_action_count", field="action_count")
        if not self.hidden_dims:
            raise TrainingRunnerError(reason="hidden_dims_required", field="hidden_dims")
        for index, dim in enumerate(self.hidden_dims):
            _positive_int(dim, f"hidden_dims[{index}]")
        _positive_int(self.value_hidden_dim, "value_hidden_dim")
        _positive_int(self.advantage_hidden_dim, "advantage_hidden_dim")
        if self.activation != "relu":
            raise TrainingRunnerError(reason="unsupported_activation", field="activation")
        if self.dtype != FEATURE_DTYPE_V1:
            raise TrainingRunnerError(reason="unsupported_dtype", field="dtype")

    def as_payload(self) -> dict[str, object]:
        return {
            "action_count": self.action_count,
            "activation": self.activation,
            "advantage_hidden_dim": self.advantage_hidden_dim,
            "architecture_id": D3QN_ARCHITECTURE_ID_V1,
            "dueling": True,
            "double_dqn_target": True,
            "dtype": self.dtype,
            "hidden_dims": list(self.hidden_dims),
            "input_dim": self.input_dim,
            "value_hidden_dim": self.value_hidden_dim,
        }

    def architecture_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


@dataclass(frozen=True, slots=True)
class PrioritizedReplayConfig:
    capacity: int = 512
    alpha: float = 0.6
    beta: float = 0.4
    epsilon: float = 1e-5
    min_priority: float = 1e-5

    def __post_init__(self) -> None:
        _positive_int(self.capacity, "capacity")
        _bounded_float(self.alpha, "alpha", low=0.0, high=1.0)
        _bounded_float(self.beta, "beta", low=0.0, high=1.0)
        _positive_float(self.epsilon, "epsilon")
        _positive_float(self.min_priority, "min_priority")

    def as_payload(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "capacity": self.capacity,
            "epsilon": self.epsilon,
            "min_priority": self.min_priority,
            "replay_id": PER_REPLAY_BUFFER_ID_V1,
        }


@dataclass(frozen=True, slots=True)
class TrainingSmokeConfig:
    seed: int = 240723
    max_sessions: int = 4
    agent_history_len: int = SESSIONIZED_AGENT_HISTORY_LEN_V1
    agent_session_len: int = SESSIONIZED_AGENT_SESSION_LEN_V1
    initial_balance: float = 100.0
    slippage: float = 0.0
    transaction_fee: float = 0.001
    inaction_penalty_ratio: float = 0.0001
    batch_size: int = 8
    update_steps: int = 8
    gamma: float = 0.99
    learning_rate: float = 0.001
    target_sync_interval: int = 4
    torch_num_threads: int = 2
    torch_num_interop_threads: int = 1
    device_policy: DevicePolicy = "cpu_only_deterministic"
    replay: PrioritizedReplayConfig = field(default_factory=PrioritizedReplayConfig)
    hidden_dims: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise TrainingRunnerError(reason="invalid_seed", field="seed")
        _positive_int(self.max_sessions, "max_sessions")
        _positive_int(self.agent_history_len, "agent_history_len")
        _positive_int(self.agent_session_len, "agent_session_len")
        _positive_float(self.initial_balance, "initial_balance")
        _non_negative_float(self.slippage, "slippage")
        _non_negative_float(self.transaction_fee, "transaction_fee")
        _non_negative_float(self.inaction_penalty_ratio, "inaction_penalty_ratio")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.update_steps, "update_steps")
        _bounded_float(self.gamma, "gamma", low=0.0, high=1.0)
        _positive_float(self.learning_rate, "learning_rate")
        _positive_int(self.target_sync_interval, "target_sync_interval")
        _positive_int(self.torch_num_threads, "torch_num_threads")
        _positive_int(self.torch_num_interop_threads, "torch_num_interop_threads")
        if self.device_policy not in {"cpu_only_deterministic", "mps_preferred_cpu_fallback"}:
            raise TrainingRunnerError(reason="unsupported_device_policy", field="device_policy")
        if not self.hidden_dims:
            raise TrainingRunnerError(reason="hidden_dims_required", field="hidden_dims")
        for index, dim in enumerate(self.hidden_dims):
            _positive_int(dim, f"hidden_dims[{index}]")

    def as_payload(self) -> dict[str, object]:
        return {
            "agent_history_len": self.agent_history_len,
            "agent_session_len": self.agent_session_len,
            "batch_size": self.batch_size,
            "config_id": TRAINING_SMOKE_CONFIG_ID_V1,
            "device_policy": self.device_policy,
            "gamma": self.gamma,
            "hidden_dims": list(self.hidden_dims),
            "inaction_penalty_ratio": self.inaction_penalty_ratio,
            "initial_balance": self.initial_balance,
            "learning_rate": self.learning_rate,
            "max_sessions": self.max_sessions,
            "replay": self.replay.as_payload(),
            "seed": self.seed,
            "slippage": self.slippage,
            "target_sync_interval": self.target_sync_interval,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "torch_num_threads": self.torch_num_threads,
            "transaction_fee": self.transaction_fee,
            "update_steps": self.update_steps,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


@dataclass(frozen=True, slots=True)
class PrioritizedReplaySample:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainingTransitionSet:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episode_count: int
    source_session_count: int
    action_counts: tuple[int, ...]

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[1])

    def as_payload(self) -> dict[str, object]:
        return {
            "action_counts": list(self.action_counts),
            "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
            "artifact_kind": STAGE07A_TRANSITION_SET_KIND_V1,
            "dones_sha256": hash_ndarray_payload_v1(self.dones.astype(np.int8)),
            "episode_count": self.episode_count,
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "next_observations_sha256": hash_ndarray_payload_v1(self.next_observations),
            "observations_sha256": hash_ndarray_payload_v1(self.observations),
            "rewards_sha256": hash_ndarray_payload_v1(self.rewards),
            "schema_version": STAGE07A_TRAINING_RUN_RECORD_SCHEMA_VERSION_V1,
            "source_session_count": self.source_session_count,
            "state_dim": self.state_dim,
            "transition_count": self.transition_count,
        }

    def transition_set_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


class PrioritizedReplayBuffer:
    def __init__(
        self,
        *,
        observation_dim: int,
        config: PrioritizedReplayConfig | None = None,
        seed: int = 0,
    ) -> None:
        selected_config = PrioritizedReplayConfig() if config is None else config
        _positive_int(observation_dim, "observation_dim")
        if isinstance(seed, bool) or seed < 0:
            raise TrainingRunnerError(reason="invalid_seed", field="seed")
        self.config = selected_config
        self.observation_dim = observation_dim
        self._rng = np.random.default_rng(seed)
        self._observations = np.zeros(
            (selected_config.capacity, observation_dim),
            dtype=np.float32,
        )
        self._next_observations = np.zeros_like(self._observations)
        self._actions = np.zeros(selected_config.capacity, dtype=np.int64)
        self._rewards = np.zeros(selected_config.capacity, dtype=np.float32)
        self._dones = np.zeros(selected_config.capacity, dtype=np.bool_)
        self._priorities = np.zeros(selected_config.capacity, dtype=np.float64)
        self._next_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        priority: float | None = None,
    ) -> None:
        observation_f32 = _one_dim_float32(observation, "observation")
        next_observation_f32 = _one_dim_float32(next_observation, "next_observation")
        if observation_f32.shape[0] != self.observation_dim:
            raise TrainingRunnerError(reason="observation_dim_mismatch", field="observation")
        if next_observation_f32.shape[0] != self.observation_dim:
            raise TrainingRunnerError(
                reason="observation_dim_mismatch",
                field="next_observation",
            )
        if isinstance(action, bool) or action < 0 or action >= RL_ACTION_COUNT_V1:
            raise TrainingRunnerError(reason="invalid_action", field="action")
        reward_value = _finite_float(reward, "reward")
        priority_value = 1.0
        if priority is None and self._size:
            priority_value = float(np.max(self._priorities[: self._size]))
        if priority is not None:
            priority_value = max(abs(_finite_float(priority, "priority")), self.config.min_priority)

        idx = self._next_idx
        self._observations[idx] = observation_f32
        self._next_observations[idx] = next_observation_f32
        self._actions[idx] = int(action)
        self._rewards[idx] = np.float32(reward_value)
        self._dones[idx] = bool(done)
        self._priorities[idx] = max(priority_value, self.config.min_priority)
        self._next_idx = (self._next_idx + 1) % self.config.capacity
        self._size = min(self._size + 1, self.config.capacity)

    def sample(self, *, batch_size: int, beta: float | None = None) -> PrioritizedReplaySample:
        _positive_int(batch_size, "batch_size")
        if self._size <= 0:
            raise TrainingRunnerError(reason="replay_buffer_empty")
        if batch_size > self._size:
            raise TrainingRunnerError(reason="batch_size_exceeds_replay_size", field="batch_size")
        selected_beta = self.config.beta if beta is None else beta
        _bounded_float(selected_beta, "beta", low=0.0, high=1.0)
        priorities = self._priorities[: self._size]
        powered = np.power(np.maximum(priorities, self.config.min_priority), self.config.alpha)
        total = float(np.sum(powered))
        if total <= 0.0 or not math.isfinite(total):
            probabilities = np.full(self._size, 1.0 / float(self._size), dtype=np.float64)
        else:
            probabilities = powered / total
        indices = self._rng.choice(self._size, size=batch_size, replace=False, p=probabilities)
        selected_probs = probabilities[indices]
        weights = np.power(float(self._size) * selected_probs, -selected_beta)
        max_weight = float(np.max(weights))
        if max_weight > 0.0 and math.isfinite(max_weight):
            weights = weights / max_weight
        return PrioritizedReplaySample(
            observations=np.ascontiguousarray(self._observations[indices], dtype=np.float32),
            actions=np.ascontiguousarray(self._actions[indices], dtype=np.int64),
            rewards=np.ascontiguousarray(self._rewards[indices], dtype=np.float32),
            next_observations=np.ascontiguousarray(
                self._next_observations[indices],
                dtype=np.float32,
            ),
            dones=np.ascontiguousarray(self._dones[indices], dtype=np.bool_),
            indices=np.ascontiguousarray(indices, dtype=np.int64),
            weights=np.ascontiguousarray(weights, dtype=np.float32),
            probabilities=np.ascontiguousarray(selected_probs, dtype=np.float32),
        )

    def update_priorities(self, *, indices: np.ndarray, td_errors: np.ndarray) -> None:
        index_values = np.asarray(indices, dtype=np.int64)
        td_error_values = np.asarray(td_errors, dtype=np.float64)
        if index_values.ndim != 1 or td_error_values.ndim != 1:
            raise TrainingRunnerError(reason="priority_update_arrays_must_be_1d")
        if index_values.shape[0] != td_error_values.shape[0]:
            raise TrainingRunnerError(reason="priority_update_length_mismatch")
        for idx, error in zip(index_values, td_error_values, strict=True):
            if idx < 0 or idx >= self._size:
                raise TrainingRunnerError(reason="priority_update_index_out_of_range")
            self._priorities[int(idx)] = max(
                abs(_finite_float(float(error), "td_error")) + self.config.epsilon,
                self.config.min_priority,
            )


def assert_stage02c_action_state_reward_compatibility_v1() -> None:
    expected_hash = STAGE07A_REQUIRED_ACTION_STATE_REWARD_CONTRACT_HASH_V1
    if ACTION_STATE_REWARD_CONTRACT_HASH_V1 != expected_hash:
        raise TrainingRunnerError(
            reason="action_state_reward_contract_hash_mismatch",
            field="ACTION_STATE_REWARD_CONTRACT_HASH_V1",
        )


def assert_stage07a_trainable_source_v1(*, exchange: str, market_type: str) -> None:
    gate = training_source_gate_payload_v1(exchange=exchange, market_type=market_type)
    if gate["status"] != "trainable":
        raise TrainingRunnerError(
            reason=BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
            field=f"{gate['exchange']}:{gate['market_type']}",
        )


def default_training_smoke_config_v1() -> TrainingSmokeConfig:
    return TrainingSmokeConfig()


def d3qn_architecture_config_for_transition_set_v1(
    *,
    transitions: TrainingTransitionSet,
    config: TrainingSmokeConfig | None = None,
) -> D3qnArchitectureConfig:
    selected_config = default_training_smoke_config_v1() if config is None else config
    return D3qnArchitectureConfig(
        input_dim=transitions.state_dim,
        hidden_dims=selected_config.hidden_dims,
    )


def build_stage07a_transition_set_v1(
    *,
    session_features: np.ndarray,
    config: TrainingSmokeConfig | None = None,
) -> TrainingTransitionSet:
    assert_stage02c_action_state_reward_compatibility_v1()
    assert_stage07a_trainable_source_v1(exchange="binance", market_type="futures")
    selected_config = default_training_smoke_config_v1() if config is None else config
    features = _validate_session_features(session_features)
    source_session_count = int(features.shape[0])
    selected_session_count = min(source_session_count, selected_config.max_sessions)
    if selected_session_count <= 0:
        raise TrainingRunnerError(reason="no_sessions_available")

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    action_counts = [0] * RL_ACTION_COUNT_V1
    scripted_actions = (1, 0, 3, 2, 0, 3, 0, 1, 0, 3)
    action_offset = selected_config.seed % len(scripted_actions)

    for session_idx in range(selected_session_count):
        session = features[session_idx]
        training_state = RlTrainingState(balance=selected_config.initial_balance)
        action_history: list[int | None] = [None] * selected_config.agent_history_len
        for step_idx in range(selected_config.agent_session_len):
            price = _session_close_price(session, step_idx=step_idx)
            observation = _build_agent_observation(
                session=session,
                step_idx=step_idx,
                action_history=action_history,
                training_state=training_state,
                price=price,
                config=selected_config,
            )
            script_index = (action_offset + session_idx + step_idx) % len(scripted_actions)
            action_id = scripted_actions[script_index]
            result = apply_training_reward_step_v1(
                state=training_state,
                action_id=action_id,
                price=price,
                initial_balance=selected_config.initial_balance,
                slippage=selected_config.slippage,
                transaction_fee=selected_config.transaction_fee,
                inaction_penalty_ratio=selected_config.inaction_penalty_ratio,
                is_last_step=(step_idx == selected_config.agent_session_len - 1),
            )
            next_action_history = [*action_history[1:], result.effective_action_id]
            next_price = _session_close_price(session, step_idx=step_idx + 1)
            next_observation = _build_agent_observation(
                session=session,
                step_idx=step_idx + 1,
                action_history=next_action_history,
                training_state=result.state,
                price=next_price,
                config=selected_config,
            )
            observations.append(observation)
            actions.append(result.effective_action_id)
            rewards.append(result.reward)
            next_observations.append(next_observation)
            dones.append(step_idx == selected_config.agent_session_len - 1)
            action_counts[result.effective_action_id] += 1
            training_state = result.state
            action_history = next_action_history

    return TrainingTransitionSet(
        observations=np.ascontiguousarray(np.vstack(observations), dtype=np.float32),
        actions=np.ascontiguousarray(np.asarray(actions, dtype=np.int64)),
        rewards=np.ascontiguousarray(np.asarray(rewards, dtype=np.float32)),
        next_observations=np.ascontiguousarray(np.vstack(next_observations), dtype=np.float32),
        dones=np.ascontiguousarray(np.asarray(dones, dtype=np.bool_)),
        episode_count=selected_session_count,
        source_session_count=source_session_count,
        action_counts=tuple(action_counts),
    )


def run_d3qn_per_training_smoke_v1(
    *,
    transitions: TrainingTransitionSet,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    output_root: Path,
    config: TrainingSmokeConfig | None = None,
    generated_at_utc: datetime | None = None,
) -> dict[str, Any]:
    assert_stage02c_action_state_reward_compatibility_v1()
    selected_config = default_training_smoke_config_v1() if config is None else config
    if transitions.transition_count < selected_config.batch_size:
        raise TrainingRunnerError(reason="not_enough_transitions_for_batch")
    torch = _import_torch()
    _seed_training_libraries(torch=torch, seed=selected_config.seed)
    _configure_torch_threads(torch=torch, config=selected_config)
    device, device_payload = _select_torch_device(torch=torch, config=selected_config)
    architecture = d3qn_architecture_config_for_transition_set_v1(
        transitions=transitions,
        config=selected_config,
    )
    model = _build_torch_d3qn_modules(torch=torch, architecture=architecture, device=device)
    target_model = _build_torch_d3qn_modules(torch=torch, architecture=architecture, device=device)
    _copy_torch_modules_state(target=target_model, source=model)
    optimizer = torch.optim.Adam(
        _torch_module_parameters(model),
        lr=selected_config.learning_rate,
    )
    replay = PrioritizedReplayBuffer(
        observation_dim=transitions.state_dim,
        config=selected_config.replay,
        seed=selected_config.seed,
    )
    for idx in range(transitions.transition_count):
        replay.add(
            observation=transitions.observations[idx],
            action=int(transitions.actions[idx]),
            reward=float(transitions.rewards[idx]),
            next_observation=transitions.next_observations[idx],
            done=bool(transitions.dones[idx]),
        )

    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.perf_counter()
    losses: list[float] = []
    td_abs_means: list[float] = []
    batch_shape_payload: dict[str, object] = {}
    for update_idx in range(selected_config.update_steps):
        sample = replay.sample(batch_size=selected_config.batch_size)
        observations = torch.as_tensor(sample.observations, dtype=torch.float32, device=device)
        actions = torch.as_tensor(sample.actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(sample.rewards, dtype=torch.float32, device=device)
        next_observations = torch.as_tensor(
            sample.next_observations,
            dtype=torch.float32,
            device=device,
        )
        dones = torch.as_tensor(sample.dones.astype(np.float32), dtype=torch.float32, device=device)
        weights = torch.as_tensor(sample.weights, dtype=torch.float32, device=device)

        q_values = _torch_d3qn_forward(model, observations)
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = _torch_d3qn_forward(model, next_observations).argmax(dim=1)
            next_target_q = _torch_d3qn_forward(target_model, next_observations).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_q = rewards + (1.0 - dones) * selected_config.gamma * next_target_q
        td_errors = target_q - selected_q_values
        loss = torch.mean(weights * torch.square(td_errors))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (update_idx + 1) % selected_config.target_sync_interval == 0:
            _copy_torch_modules_state(target=target_model, source=model)
        td_np = np.asarray(td_errors.detach().cpu().numpy(), dtype=np.float64)
        replay.update_priorities(indices=sample.indices, td_errors=td_np)
        losses.append(float(loss.detach().cpu()))
        td_abs_means.append(float(np.mean(np.abs(td_np), dtype=np.float64)))
        if not batch_shape_payload:
            batch_shape_payload = {
                "actions_shape": list(sample.actions.shape),
                "next_observations_shape": list(sample.next_observations.shape),
                "observations_shape": list(sample.observations.shape),
                "q_values_shape": list(q_values.detach().cpu().shape),
                "target_q_shape": list(target_q.detach().cpu().shape),
                "td_errors_shape": list(td_errors.detach().cpu().shape),
                "weights_shape": list(sample.weights.shape),
            }

    _synchronize_if_needed(torch=torch, device_type=str(device.type))
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    wall_seconds = time.perf_counter() - start_wall
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "stage07a_smoke_model_state.pt"
    torch.save(_cpu_state_dict_payload(model), model_path)
    artifact_hashes = {
        "model_state_dict": _file_payload(model_path),
        "transition_set": {
            "sha256": transitions.transition_set_hash(),
            "transitions": transitions.transition_count,
        },
    }
    metrics = {
        "batch_shapes": batch_shape_payload,
        "closed_episode_count": int(np.count_nonzero(transitions.dones)),
        "final_loss": _round_float(losses[-1]),
        "first_loss": _round_float(losses[0]),
        "loss_count": len(losses),
        "loss_delta": _round_float(losses[-1] - losses[0]),
        "mean_abs_td_error_last": _round_float(td_abs_means[-1]),
        "reward_sum": _round_float(float(np.sum(transitions.rewards, dtype=np.float64))),
        "transition_count": transitions.transition_count,
        "update_steps": selected_config.update_steps,
    }
    resource_usage = {
        "cpu_system_seconds_delta": _round_float(end_usage.ru_stime - start_usage.ru_stime),
        "cpu_user_seconds_delta": _round_float(end_usage.ru_utime - start_usage.ru_utime),
        "mps_available": bool(device_payload["mps_available"]),
        "mps_built": bool(device_payload["mps_built"]),
        "process_threads_observed": _process_thread_count(),
        "rss_mb_after": _rss_mb(),
        "selected_device": device_payload["selected_device"],
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_num_threads": int(torch.get_num_threads()),
        "wall_seconds": _round_float(wall_seconds),
    }
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    record = build_training_run_record_v1(
        generated_at_utc=generated,
        config=selected_config,
        architecture=architecture,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        transitions=transitions,
        device_payload=device_payload,
        metrics=metrics,
        resource_usage=resource_usage,
        artifact_hashes=artifact_hashes,
    )
    record_path = output_root / "stage07a_training_run_record.json"
    record = finalize_training_run_record_v1(
        {
            **record,
            "run_record_path": str(record_path),
        }
    )
    _atomic_write_json(record_path, record)
    return record


def build_training_run_record_v1(
    *,
    generated_at_utc: datetime,
    config: TrainingSmokeConfig,
    architecture: D3qnArchitectureConfig,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    transitions: TrainingTransitionSet,
    device_payload: dict[str, object],
    metrics: dict[str, Any],
    resource_usage: dict[str, Any],
    artifact_hashes: dict[str, Any],
) -> dict[str, Any]:
    return finalize_training_run_record_v1(
        {
            "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
            "architecture": architecture.as_payload(),
            "architecture_hash": architecture.architecture_hash(),
            "artifact_hashes": artifact_hashes,
            "artifact_kind": STAGE07A_TRAINING_RUN_KIND_V1,
            "config": config.as_payload(),
            "config_hash": config.config_hash(),
            "dataset_dependency": {
                "manifest_path": dataset_manifest_path,
                "manifest_sha256": dataset_manifest_sha256,
                "stage": "06",
                "training_source": "binance:futures",
            },
            "dependency_isolation": {
                "default_api_runtime_requires_torch": False,
                "torch_extra": "rl-ml",
            },
            "device": device_payload,
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "generated_at_utc": _format_utc(generated_at_utc),
            "metrics": metrics,
            "resource_usage": resource_usage,
            "safety": {
                "candidate_model_claim": False,
                "contains_raw_provider_payloads": False,
                "contains_secrets": False,
                "exchange_side_effects": False,
                "mainnet_submit": False,
                "model_registry_write": False,
                "paper_testnet_live_enabled": False,
                "runtime_artifact_root": STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
                "smoke_model_state_is_candidate": False,
            },
            "schema_version": STAGE07A_TRAINING_RUN_RECORD_SCHEMA_VERSION_V1,
            "stage": "07A",
            "status": "accepted",
            "transition_set": transitions.as_payload(),
        }
    )


def finalize_training_run_record_v1(record: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in record.items() if key != "run_record_hash"}
    return {**payload, "run_record_hash": hash_json_payload_v1(payload)}


def render_training_run_record_v1(record: dict[str, Any]) -> str:
    return render_raw_feature_json_payload_v1(record)


def _validate_session_features(session_features: np.ndarray) -> np.ndarray:
    features = np.asarray(session_features, dtype=np.float32)
    if features.ndim != 3:
        raise TrainingRunnerError(reason="session_features_must_be_3d")
    expected_shape = (SESSIONIZED_FULL_SEQ_LEN_V1, len(FEATURE_NAMES_V1))
    if tuple(features.shape[1:]) != expected_shape:
        raise TrainingRunnerError(reason="unexpected_session_shape", field=str(features.shape))
    if features.shape[0] <= 0:
        raise TrainingRunnerError(reason="empty_session_features")
    if not np.all(np.isfinite(features)):
        raise TrainingRunnerError(reason="non_finite_session_features")
    close_idx = FEATURE_NAMES_V1.index("close")
    if np.any(features[:, :, close_idx] <= 0.0):
        raise TrainingRunnerError(reason="non_positive_close_price")
    return np.ascontiguousarray(features, dtype=np.float32)


def _build_agent_observation(
    *,
    session: np.ndarray,
    step_idx: int,
    action_history: Sequence[int | None],
    training_state: RlTrainingState,
    price: float,
    config: TrainingSmokeConfig,
) -> np.ndarray:
    history_start = SESSIONIZED_PRE_SIGNAL_LEN_V1 - config.agent_history_len + step_idx
    history_end = history_start + config.agent_history_len
    if history_start < 0 or history_end > session.shape[0]:
        raise TrainingRunnerError(reason="agent_history_window_out_of_bounds")
    features = np.ascontiguousarray(session[history_start:history_end, :], dtype=np.float32)
    state_extras = build_state_extras_v1(
        position_side=training_state.position_side,
        entry_price=training_state.entry_price,
        current_price=price,
        step_idx=min(step_idx, config.agent_session_len),
        session_len=config.agent_session_len,
    )
    action_history_values = list(action_history)
    if len(action_history_values) != config.agent_history_len:
        raise TrainingRunnerError(reason="action_history_len_mismatch")
    state = np.concatenate(
        (
            features.reshape(-1),
            np.asarray(encode_action_history_v1(action_history_values), dtype=np.float32),
            np.asarray(state_extras, dtype=np.float32),
        )
    )
    return np.ascontiguousarray(state, dtype=np.float32)


def _session_close_price(session: np.ndarray, *, step_idx: int) -> float:
    close_idx = FEATURE_NAMES_V1.index("close")
    price_idx = min(SESSIONIZED_PRE_SIGNAL_LEN_V1 + step_idx, session.shape[0] - 1)
    return _positive_float(float(session[price_idx, close_idx]), "close")


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise TrainingRunnerError(reason="torch_import_failed", field=str(exc)) from exc


def _seed_training_libraries(*, torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def _configure_torch_threads(*, torch: Any, config: TrainingSmokeConfig) -> None:
    torch.set_num_threads(config.torch_num_threads)
    current_interop_threads = int(torch.get_num_interop_threads())
    if current_interop_threads != config.torch_num_interop_threads:
        torch.set_num_interop_threads(config.torch_num_interop_threads)


def _select_torch_device(
    *,
    torch: Any,
    config: TrainingSmokeConfig,
) -> tuple[Any, dict[str, object]]:
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    selected_device = "cpu"
    if config.device_policy == "mps_preferred_cpu_fallback" and mps_available:
        selected_device = "mps"
    device = torch.device(selected_device)
    return device, {
        "device_policy": config.device_policy,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "selected_device": selected_device,
        "torch_version": str(torch.__version__),
    }


def _build_torch_d3qn_modules(
    *,
    torch: Any,
    architecture: D3qnArchitectureConfig,
    device: Any,
) -> dict[str, Any]:
    dims = [architecture.input_dim, *architecture.hidden_dims]
    shared_layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
        shared_layers.append(torch.nn.Linear(in_dim, out_dim))
        shared_layers.append(torch.nn.ReLU())
    trunk_dim = architecture.hidden_dims[-1]
    modules = {
        "advantage": torch.nn.Sequential(
            torch.nn.Linear(trunk_dim, architecture.advantage_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture.advantage_hidden_dim, architecture.action_count),
        ).to(device),
        "shared": torch.nn.Sequential(*shared_layers).to(device),
        "value": torch.nn.Sequential(
            torch.nn.Linear(trunk_dim, architecture.value_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture.value_hidden_dim, 1),
        ).to(device),
    }
    return modules


def _torch_module_parameters(modules: dict[str, Any]) -> list[Any]:
    params: list[Any] = []
    for name in ("shared", "value", "advantage"):
        params.extend(list(modules[name].parameters()))
    return params


def _torch_d3qn_forward(modules: dict[str, Any], observations: Any) -> Any:
    features = modules["shared"](observations)
    value = modules["value"](features)
    advantage = modules["advantage"](features)
    return value + advantage - advantage.mean(dim=1, keepdim=True)


def _copy_torch_modules_state(*, target: dict[str, Any], source: dict[str, Any]) -> None:
    for name in ("shared", "value", "advantage"):
        target[name].load_state_dict(source[name].state_dict())


def _cpu_state_dict_payload(modules: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name in ("shared", "value", "advantage"):
        payload[name] = {
            key: value.detach().cpu()
            for key, value in modules[name].state_dict().items()
        }
    return payload


def _synchronize_if_needed(*, torch: Any, device_type: str) -> None:
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_training_run_record_v1(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_payload(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": _file_sha256_hex(path),
    }


def _file_sha256_hex(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return _round_float(int(output.strip()) / 1024.0)
    except Exception:
        return None


def _process_thread_count() -> int | None:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["/bin/ps", "-M", str(os.getpid())],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return max(len([line for line in output.splitlines() if line.strip()]) - 1, 0)
        except Exception:
            return None
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/task"))
    except Exception:
        return None


def _one_dim_float32(value: np.ndarray, field: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 1:
        raise TrainingRunnerError(reason="array_must_be_1d", field=field)
    if not np.all(np.isfinite(out)):
        raise TrainingRunnerError(reason="array_contains_non_finite_values", field=field)
    return np.ascontiguousarray(out, dtype=np.float32)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_float(value: float) -> float:
    return round(float(value), 8)


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise TrainingRunnerError(reason="invalid_positive_int", field=field)
    return value


def _finite_float(value: float, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise TrainingRunnerError(reason="non_finite_float", field=field)
    return out


def _positive_float(value: float, field: str) -> float:
    out = _finite_float(value, field)
    if out <= 0.0:
        raise TrainingRunnerError(reason="non_positive_float", field=field)
    return out


def _non_negative_float(value: float, field: str) -> float:
    out = _finite_float(value, field)
    if out < 0.0:
        raise TrainingRunnerError(reason="negative_float", field=field)
    return out


def _bounded_float(value: float, field: str, *, low: float, high: float) -> float:
    out = _finite_float(value, field)
    if out < low or out > high:
        raise TrainingRunnerError(reason="float_out_of_bounds", field=field)
    return out
