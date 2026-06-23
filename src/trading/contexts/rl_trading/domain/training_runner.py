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
from typing import Any, Literal, Sequence, cast

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
STAGE07B_CANDIDATE_RUN_SCHEMA_VERSION_V1 = 1
STAGE07B_CANDIDATE_RUN_KIND_V1 = "rl_trading_stage07b_full_candidate_training_run"
STAGE07B_CANDIDATE_CONFIG_ID_V1 = "roehub_stage07b_candidate_training_config_v1"
STAGE07B_PROGRESS_KIND_V1 = "rl_trading_stage07b_training_progress"
STAGE07B_CANDIDATE_MANIFEST_KIND_V1 = "rl_trading_stage07b_candidate_manifest"
D3QN_ARCHITECTURE_ID_V1 = "roehub_d3qn_mlp_v1"
PER_REPLAY_BUFFER_ID_V1 = "roehub_per_replay_v1"
TRAINING_SMOKE_CONFIG_ID_V1 = "roehub_stage07a_training_smoke_config_v1"
STAGE07A_TRANSITION_SET_KIND_V1 = "rl_trading_stage07a_transition_set"

DevicePolicy = Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"]
RunStatus = Literal["accepted", "blocked"]
CandidateTrainingStatus = Literal["starting", "running", "completed", "failed", "interrupted"]


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
class CandidateTrainingConfig:
    seed: int = 240723
    train_dataset_version: str = "hf_period_rebuild_current_trading"
    train_split: str = "train"
    validation_dataset_version: str = "hf_period_rebuild_current_trading"
    validation_split: str = "validation"
    agent_history_len: int = SESSIONIZED_AGENT_HISTORY_LEN_V1
    agent_session_len: int = SESSIONIZED_AGENT_SESSION_LEN_V1
    initial_balance: float = 100.0
    slippage: float = 0.0
    transaction_fee: float = 0.001
    inaction_penalty_ratio: float = 0.0001
    batch_size: int = 256
    planned_training_steps: int = 100_000
    progress_emit_every_steps: int = 10_000
    progress_emit_every_sec: int = 300
    checkpoint_every_steps: int = 10_000
    validation_every_steps: int = 10_000
    validation_max_transitions: int = 4_096
    gamma: float = 0.99
    learning_rate: float = 0.0005
    target_sync_interval: int = 1_000
    torch_num_threads: int = 4
    torch_num_interop_threads: int = 1
    device_policy: DevicePolicy = "cpu_only_deterministic"
    replay: PrioritizedReplayConfig = field(
        default_factory=lambda: PrioritizedReplayConfig(capacity=200_000)
    )
    hidden_dims: tuple[int, ...] = (128, 128)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise TrainingRunnerError(reason="invalid_seed", field="seed")
        _non_empty_text(self.train_dataset_version, "train_dataset_version")
        _non_empty_text(self.train_split, "train_split")
        _non_empty_text(self.validation_dataset_version, "validation_dataset_version")
        _non_empty_text(self.validation_split, "validation_split")
        _positive_int(self.agent_history_len, "agent_history_len")
        _positive_int(self.agent_session_len, "agent_session_len")
        _positive_float(self.initial_balance, "initial_balance")
        _non_negative_float(self.slippage, "slippage")
        _non_negative_float(self.transaction_fee, "transaction_fee")
        _non_negative_float(self.inaction_penalty_ratio, "inaction_penalty_ratio")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.planned_training_steps, "planned_training_steps")
        _positive_int(self.progress_emit_every_steps, "progress_emit_every_steps")
        _positive_int(self.progress_emit_every_sec, "progress_emit_every_sec")
        _positive_int(self.checkpoint_every_steps, "checkpoint_every_steps")
        _positive_int(self.validation_every_steps, "validation_every_steps")
        _positive_int(self.validation_max_transitions, "validation_max_transitions")
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
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "config_id": STAGE07B_CANDIDATE_CONFIG_ID_V1,
            "device_policy": self.device_policy,
            "gamma": self.gamma,
            "hidden_dims": list(self.hidden_dims),
            "inaction_penalty_ratio": self.inaction_penalty_ratio,
            "initial_balance": self.initial_balance,
            "learning_rate": self.learning_rate,
            "planned_training_steps": self.planned_training_steps,
            "progress_emit_every_sec": self.progress_emit_every_sec,
            "progress_emit_every_steps": self.progress_emit_every_steps,
            "replay": self.replay.as_payload(),
            "seed": self.seed,
            "slippage": self.slippage,
            "target_sync_interval": self.target_sync_interval,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "torch_num_threads": self.torch_num_threads,
            "train_dataset_version": self.train_dataset_version,
            "train_split": self.train_split,
            "transaction_fee": self.transaction_fee,
            "validation_dataset_version": self.validation_dataset_version,
            "validation_every_steps": self.validation_every_steps,
            "validation_max_transitions": self.validation_max_transitions,
            "validation_split": self.validation_split,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())

    def as_smoke_compatible_config(self, *, max_sessions: int) -> TrainingSmokeConfig:
        return TrainingSmokeConfig(
            seed=self.seed,
            max_sessions=max_sessions,
            agent_history_len=self.agent_history_len,
            agent_session_len=self.agent_session_len,
            initial_balance=self.initial_balance,
            slippage=self.slippage,
            transaction_fee=self.transaction_fee,
            inaction_penalty_ratio=self.inaction_penalty_ratio,
            batch_size=self.batch_size,
            update_steps=min(self.planned_training_steps, 1),
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            target_sync_interval=self.target_sync_interval,
            torch_num_threads=self.torch_num_threads,
            torch_num_interop_threads=self.torch_num_interop_threads,
            device_policy=self.device_policy,
            replay=self.replay,
            hidden_dims=self.hidden_dims,
        )


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

    def priority_state_payload(self) -> dict[str, object]:
        return {
            "next_idx": self._next_idx,
            "priorities": np.ascontiguousarray(
                self._priorities[: self._size],
                dtype=np.float64,
            ),
            "rng_state": self._rng.bit_generator.state,
            "size": self._size,
        }

    def restore_priority_state(self, payload: dict[str, object]) -> None:
        size = payload["size"]
        next_idx = payload["next_idx"]
        if not isinstance(size, int) or isinstance(size, bool):
            raise TrainingRunnerError(reason="replay_priority_state_size_invalid")
        if not isinstance(next_idx, int) or isinstance(next_idx, bool):
            raise TrainingRunnerError(reason="replay_priority_state_next_idx_invalid")
        if size < 0 or size > self._size:
            raise TrainingRunnerError(reason="replay_priority_state_size_mismatch")
        if next_idx < 0 or next_idx >= self.config.capacity:
            raise TrainingRunnerError(reason="replay_priority_state_next_idx_mismatch")
        priorities = np.asarray(payload["priorities"], dtype=np.float64)
        if priorities.ndim != 1 or priorities.shape[0] != size:
            raise TrainingRunnerError(reason="replay_priority_state_priorities_mismatch")
        self._priorities[:size] = priorities
        if size < self._priorities.shape[0]:
            self._priorities[size:] = 0.0
        self._next_idx = next_idx
        self._size = size
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            self._rng.bit_generator.state = rng_state


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


def default_stage07b_candidate_training_config_v1() -> CandidateTrainingConfig:
    return CandidateTrainingConfig()


def build_stage07b_transition_set_v1(
    *,
    session_features: np.ndarray,
    config: CandidateTrainingConfig | None = None,
) -> TrainingTransitionSet:
    selected_config = (
        default_stage07b_candidate_training_config_v1() if config is None else config
    )
    features = _validate_session_features(session_features)
    smoke_config = selected_config.as_smoke_compatible_config(
        max_sessions=int(features.shape[0]),
    )
    return build_stage07a_transition_set_v1(
        session_features=features,
        config=smoke_config,
    )


def d3qn_architecture_config_for_stage07b_v1(
    *,
    transitions: TrainingTransitionSet,
    config: CandidateTrainingConfig | None = None,
) -> D3qnArchitectureConfig:
    selected_config = (
        default_stage07b_candidate_training_config_v1() if config is None else config
    )
    return D3qnArchitectureConfig(
        input_dim=transitions.state_dim,
        hidden_dims=selected_config.hidden_dims,
    )


def run_stage07b_candidate_training_v1(
    *,
    train_transitions: TrainingTransitionSet,
    validation_transitions: TrainingTransitionSet,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    output_root: Path,
    run_id: str,
    config: CandidateTrainingConfig | None = None,
    generated_at_utc: datetime | None = None,
    code_version: dict[str, object] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    assert_stage02c_action_state_reward_compatibility_v1()
    selected_config = (
        default_stage07b_candidate_training_config_v1() if config is None else config
    )
    if train_transitions.transition_count < selected_config.batch_size:
        raise TrainingRunnerError(reason="not_enough_train_transitions_for_batch")
    if validation_transitions.transition_count <= 0:
        raise TrainingRunnerError(reason="validation_transitions_required")
    if selected_config.replay.capacity < train_transitions.transition_count:
        raise TrainingRunnerError(
            reason="replay_capacity_below_train_transition_count",
            field="replay.capacity",
        )

    torch = _import_torch()
    _seed_training_libraries(torch=torch, seed=selected_config.seed)
    _configure_torch_threads_for_values(
        torch=torch,
        torch_num_threads=selected_config.torch_num_threads,
        torch_num_interop_threads=selected_config.torch_num_interop_threads,
    )
    device, device_payload = _select_torch_device_for_policy(
        torch=torch,
        device_policy=selected_config.device_policy,
    )
    architecture = d3qn_architecture_config_for_stage07b_v1(
        transitions=train_transitions,
        config=selected_config,
    )
    model = _build_torch_d3qn_modules(torch=torch, architecture=architecture, device=device)
    target_model = _build_torch_d3qn_modules(torch=torch, architecture=architecture, device=device)
    _copy_torch_modules_state(target=target_model, source=model)
    optimizer = torch.optim.Adam(
        _torch_module_parameters(model),
        lr=selected_config.learning_rate,
    )
    replay = _build_replay_from_transitions_v1(
        transitions=train_transitions,
        config=selected_config.replay,
        seed=selected_config.seed,
    )

    run_dir = output_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    paths = {
        "candidate_manifest": run_dir / "candidate_manifest.json",
        "candidate_report": run_dir / "candidate_training_report.json",
        "latest_checkpoint": run_dir / "latest_checkpoint.json",
        "latest_status": run_dir / "latest_status.json",
        "progress": run_dir / "progress.jsonl",
        "training_config": run_dir / "training_config.json",
    }
    source_payload = {} if code_version is None else code_version
    training_config_payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture": architecture.as_payload(),
        "architecture_hash": architecture.architecture_hash(),
        "code_version": source_payload,
        "config": selected_config.as_payload(),
        "config_hash": selected_config.config_hash(),
        "dataset_dependency": {
            "manifest_path": dataset_manifest_path,
            "manifest_sha256": dataset_manifest_sha256,
            "stage": "06",
            "training_source": "binance:futures",
        },
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated),
        "run_id": run_id,
        "schema_version": STAGE07B_CANDIDATE_RUN_SCHEMA_VERSION_V1,
        "stage": "07B",
        "training_plan": _stage07b_training_plan_payload(selected_config),
    }
    training_config_payload = {
        **training_config_payload,
        "training_config_hash": hash_json_payload_v1(training_config_payload),
    }
    _atomic_write_json(paths["training_config"], training_config_payload)

    writer = _Stage07bProgressWriter(
        run_id=run_id,
        progress_path=paths["progress"],
        latest_status_path=paths["latest_status"],
        planned_training_steps=selected_config.planned_training_steps,
        device=str(device_payload["selected_device"]),
    )
    completed_steps = 0
    train_curve: list[dict[str, object]] = []
    validation_curve: list[dict[str, object]] = []
    loss_window: list[float] = []
    checkpoint_payload: dict[str, object] | None = None
    if resume:
        checkpoint_payload = _load_latest_stage07b_checkpoint_payload(
            torch=torch,
            latest_checkpoint_path=paths["latest_checkpoint"],
            device=device,
        )
        if checkpoint_payload is not None:
            completed_steps_payload = checkpoint_payload["completed_training_steps"]
            if not isinstance(completed_steps_payload, int) or isinstance(
                completed_steps_payload,
                bool,
            ):
                raise TrainingRunnerError(reason="checkpoint_completed_steps_invalid")
            completed_steps = completed_steps_payload
            _load_torch_modules_state(target=model, state=checkpoint_payload["model_state"])
            _load_torch_modules_state(
                target=target_model,
                state=checkpoint_payload["target_model_state"],
            )
            optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
            replay_priority_state = checkpoint_payload["replay_priority_state"]
            if not isinstance(replay_priority_state, dict):
                raise TrainingRunnerError(reason="replay_priority_state_invalid")
            replay.restore_priority_state(replay_priority_state)
            train_curve_payload = checkpoint_payload.get("train_curve", [])
            validation_curve_payload = checkpoint_payload.get("validation_curve", [])
            if not isinstance(train_curve_payload, list):
                raise TrainingRunnerError(reason="checkpoint_train_curve_invalid")
            if not isinstance(validation_curve_payload, list):
                raise TrainingRunnerError(reason="checkpoint_validation_curve_invalid")
            train_curve = cast(list[dict[str, object]], train_curve_payload)
            validation_curve = cast(list[dict[str, object]], validation_curve_payload)

    writer.emit(
        status="starting" if completed_steps == 0 else "running",
        completed_training_steps=completed_steps,
        details={
            "checkpoint_resume": checkpoint_payload is not None,
            "config_hash": selected_config.config_hash(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_transition_count": train_transitions.transition_count,
            "validation_transition_count": validation_transitions.transition_count,
        },
    )
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    last_emit_step = completed_steps
    last_emit_wall = time.perf_counter()
    status: CandidateTrainingStatus = "running"
    try:
        for step in range(completed_steps + 1, selected_config.planned_training_steps + 1):
            loss_value, td_abs_mean = _run_d3qn_per_update_step_v1(
                torch=torch,
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                replay=replay,
                batch_size=selected_config.batch_size,
                gamma=selected_config.gamma,
                device=device,
            )
            loss_window.append(loss_value)
            if step % selected_config.target_sync_interval == 0:
                _copy_torch_modules_state(target=target_model, source=model)
            should_validate = (
                step == 1
                or step == selected_config.planned_training_steps
                or step % selected_config.validation_every_steps == 0
            )
            if should_validate:
                train_summary = {
                    "completed_training_steps": step,
                    "loss_window_mean": _round_float(
                        float(np.mean(np.asarray(loss_window, dtype=np.float64)))
                    ),
                    "loss_window_size": len(loss_window),
                    "mean_abs_td_error_last": _round_float(td_abs_mean),
                }
                train_curve.append(train_summary)
                validation_curve.append(
                    {
                        "completed_training_steps": step,
                        **_compute_validation_curve_point_v1(
                            torch=torch,
                            model=model,
                            target_model=target_model,
                            transitions=validation_transitions,
                            batch_size=selected_config.batch_size,
                            gamma=selected_config.gamma,
                            max_transitions=selected_config.validation_max_transitions,
                            device=device,
                        ),
                    }
                )
                loss_window = []
            should_checkpoint = (
                step == selected_config.planned_training_steps
                or step % selected_config.checkpoint_every_steps == 0
            )
            if should_checkpoint:
                _save_stage07b_checkpoint_v1(
                    torch=torch,
                    path=checkpoints_dir / f"checkpoint_step_{step:08d}.pt",
                    latest_checkpoint_path=paths["latest_checkpoint"],
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay=replay,
                    completed_training_steps=step,
                    run_id=run_id,
                    config_hash=selected_config.config_hash(),
                    architecture_hash=architecture.architecture_hash(),
                    dataset_manifest_sha256=dataset_manifest_sha256,
                    train_curve=train_curve,
                    validation_curve=validation_curve,
                )
            wall_now = time.perf_counter()
            should_emit = (
                step == 1
                or step == selected_config.planned_training_steps
                or (step - last_emit_step) >= selected_config.progress_emit_every_steps
                or (wall_now - last_emit_wall) >= selected_config.progress_emit_every_sec
            )
            if should_emit:
                writer.emit(
                    status="running",
                    completed_training_steps=step,
                    details={
                        "last_loss": _round_float(loss_value),
                        "train_curve_points": len(train_curve),
                        "validation_curve_points": len(validation_curve),
                    },
                )
                last_emit_step = step
                last_emit_wall = wall_now
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        writer.emit(
            status="interrupted",
            completed_training_steps=max(last_emit_step, completed_steps),
            details={"reason": "keyboard_interrupt"},
        )
        raise
    except Exception as exc:
        status = "failed"
        writer.emit(
            status="failed",
            completed_training_steps=max(last_emit_step, completed_steps),
            details={"reason": type(exc).__name__},
        )
        raise

    _synchronize_if_needed(torch=torch, device_type=str(device.type))
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    resource_usage = _stage07b_resource_usage_payload(
        torch=torch,
        start_usage=start_usage,
        end_usage=end_usage,
        device_payload=device_payload,
        progress_writer=writer,
        completed_training_steps=selected_config.planned_training_steps,
    )
    final_checkpoint_path = checkpoints_dir / (
        f"checkpoint_step_{selected_config.planned_training_steps:08d}.pt"
    )
    writer.emit(
        status=status,
        completed_training_steps=selected_config.planned_training_steps,
        details={
            "final_checkpoint_path": str(final_checkpoint_path),
            "train_curve_points": len(train_curve),
            "validation_curve_points": len(validation_curve),
        },
    )
    artifact_hashes = {
        "candidate_checkpoint": _file_payload(final_checkpoint_path),
        "latest_status": _file_payload(paths["latest_status"]),
        "latest_checkpoint": _file_payload(paths["latest_checkpoint"]),
        "progress_jsonl": _file_payload(paths["progress"]),
        "training_config": _file_payload(paths["training_config"]),
        "train_transition_set": {
            "sha256": train_transitions.transition_set_hash(),
            "transitions": train_transitions.transition_count,
        },
        "validation_transition_set": {
            "sha256": validation_transitions.transition_set_hash(),
            "transitions": validation_transitions.transition_count,
        },
    }
    metrics = {
        "completed_training_steps": selected_config.planned_training_steps,
        "planned_training_steps": selected_config.planned_training_steps,
        "progress_pct": 100.0,
        "throughput_steps_per_sec": _round_float(
            selected_config.planned_training_steps / max(writer.elapsed_sec(), 1e-9)
        ),
        "train_curve": train_curve,
        "train_transition_count": train_transitions.transition_count,
        "validation_curve": validation_curve,
        "validation_transition_count": validation_transitions.transition_count,
    }
    report = build_stage07b_candidate_report_v1(
        generated_at_utc=generated,
        finished_at_utc=datetime.now(UTC).replace(microsecond=0),
        run_id=run_id,
        run_dir=run_dir,
        config=selected_config,
        architecture=architecture,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        code_version=source_payload,
        metrics=metrics,
        resource_usage=resource_usage,
        artifact_hashes=artifact_hashes,
    )
    _atomic_write_json(paths["candidate_report"], report)
    artifact_hashes = {
        **artifact_hashes,
        "candidate_report": _file_payload(paths["candidate_report"]),
    }
    manifest = build_stage07b_candidate_manifest_v1(
        generated_at_utc=generated,
        run_id=run_id,
        run_dir=run_dir,
        config=selected_config,
        architecture=architecture,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        code_version=source_payload,
        metrics=metrics,
        resource_usage=resource_usage,
        artifact_hashes=artifact_hashes,
    )
    _atomic_write_json(paths["candidate_manifest"], manifest)
    manifest = {
        **manifest,
        "artifact_hashes": artifact_hashes,
        "candidate_manifest_path": str(paths["candidate_manifest"]),
    }
    manifest = finalize_stage07b_candidate_manifest_v1(manifest)
    _atomic_write_json(paths["candidate_manifest"], manifest)
    return manifest


def build_stage07b_candidate_report_v1(
    *,
    generated_at_utc: datetime,
    finished_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: CandidateTrainingConfig,
    architecture: D3qnArchitectureConfig,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    code_version: dict[str, object],
    metrics: dict[str, Any],
    resource_usage: dict[str, Any],
    artifact_hashes: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture": architecture.as_payload(),
        "architecture_hash": architecture.architecture_hash(),
        "artifact_hashes": artifact_hashes,
        "artifact_kind": STAGE07B_CANDIDATE_RUN_KIND_V1,
        "code_version": code_version,
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
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "finished_at_utc": _format_utc(finished_at_utc),
        "generated_at_utc": _format_utc(generated_at_utc),
        "metrics": metrics,
        "resource_usage": resource_usage,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": _stage07b_safety_payload(),
        "schema_version": STAGE07B_CANDIDATE_RUN_SCHEMA_VERSION_V1,
        "stage": "07B",
        "status": "completed",
        "training_plan": _stage07b_training_plan_payload(config),
    }
    return {**payload, "candidate_report_hash": hash_json_payload_v1(payload)}


def build_stage07b_candidate_manifest_v1(
    *,
    generated_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: CandidateTrainingConfig,
    architecture: D3qnArchitectureConfig,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    code_version: dict[str, object],
    metrics: dict[str, Any],
    resource_usage: dict[str, Any],
    artifact_hashes: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_hash": architecture.architecture_hash(),
        "artifact_hashes": artifact_hashes,
        "artifact_kind": STAGE07B_CANDIDATE_MANIFEST_KIND_V1,
        "candidate_level": "candidate_for_stage08_evaluation_only",
        "code_version": code_version,
        "config_hash": config.config_hash(),
        "dataset_dependency": {
            "manifest_path": dataset_manifest_path,
            "manifest_sha256": dataset_manifest_sha256,
            "stage": "06",
            "training_source": "binance:futures",
        },
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated_at_utc),
        "metrics_summary": {
            "completed_training_steps": metrics["completed_training_steps"],
            "planned_training_steps": metrics["planned_training_steps"],
            "progress_pct": metrics["progress_pct"],
            "throughput_steps_per_sec": metrics["throughput_steps_per_sec"],
            "train_curve_points": len(metrics["train_curve"]),
            "train_transition_count": metrics["train_transition_count"],
            "validation_curve_points": len(metrics["validation_curve"]),
            "validation_transition_count": metrics["validation_transition_count"],
        },
        "next_stage_handoff": {
            "stage08_allowed": True,
            "stage08_input": "candidate_manifest_path",
        },
        "resource_summary": resource_usage,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": _stage07b_safety_payload(),
        "schema_version": STAGE07B_CANDIDATE_RUN_SCHEMA_VERSION_V1,
        "stage": "07B",
        "status": "completed",
        "training_plan": _stage07b_training_plan_payload(config),
    }
    return finalize_stage07b_candidate_manifest_v1(payload)


def finalize_stage07b_candidate_manifest_v1(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in manifest.items() if key != "candidate_manifest_hash"}
    return {**payload, "candidate_manifest_hash": hash_json_payload_v1(payload)}


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


class _Stage07bProgressWriter:
    def __init__(
        self,
        *,
        run_id: str,
        progress_path: Path,
        latest_status_path: Path,
        planned_training_steps: int,
        device: str,
    ) -> None:
        self.run_id = run_id
        self.progress_path = progress_path
        self.latest_status_path = latest_status_path
        self.planned_training_steps = planned_training_steps
        self.device = device
        self.started_at_wall = time.perf_counter()
        self.started_at_utc = datetime.now(UTC).replace(microsecond=0)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def elapsed_sec(self) -> float:
        return _round_float(time.perf_counter() - self.started_at_wall)

    def emit(
        self,
        *,
        status: CandidateTrainingStatus,
        completed_training_steps: int,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        elapsed = self.elapsed_sec()
        progress_pct = _round_float(
            (completed_training_steps / self.planned_training_steps) * 100.0
        )
        eta_sec: float | None = None
        if status in {"starting", "running"} and completed_training_steps > 0:
            remaining = self.planned_training_steps - completed_training_steps
            eta_sec = _round_float((elapsed / completed_training_steps) * remaining)
        elif status == "completed":
            eta_sec = 0.0
        event = {
            "artifact_kind": STAGE07B_PROGRESS_KIND_V1,
            "completed_training_steps": completed_training_steps,
            "details": {} if details is None else details,
            "device": self.device,
            "elapsed_sec": elapsed,
            "eta_sec": eta_sec,
            "planned_training_steps": self.planned_training_steps,
            "progress_pct": progress_pct,
            "resource_snapshot": _resource_snapshot_payload(),
            "run_id": self.run_id,
            "stage": "07B",
            "started_at_utc": _format_utc(self.started_at_utc),
            "status": status,
            "timestamp": _format_utc(datetime.now(UTC).replace(microsecond=0)),
        }
        rendered = _render_json_line_payload(event)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        latest = {
            **event,
            "latest_status_path": str(self.latest_status_path),
            "progress_path": str(self.progress_path),
        }
        _atomic_write_json(self.latest_status_path, latest)
        return event


def _build_replay_from_transitions_v1(
    *,
    transitions: TrainingTransitionSet,
    config: PrioritizedReplayConfig,
    seed: int,
) -> PrioritizedReplayBuffer:
    if config.capacity < transitions.transition_count:
        raise TrainingRunnerError(
            reason="replay_capacity_below_train_transition_count",
            field="replay.capacity",
        )
    replay = PrioritizedReplayBuffer(
        observation_dim=transitions.state_dim,
        config=config,
        seed=seed,
    )
    for idx in range(transitions.transition_count):
        replay.add(
            observation=transitions.observations[idx],
            action=int(transitions.actions[idx]),
            reward=float(transitions.rewards[idx]),
            next_observation=transitions.next_observations[idx],
            done=bool(transitions.dones[idx]),
        )
    return replay


def _run_d3qn_per_update_step_v1(
    *,
    torch: Any,
    model: dict[str, Any],
    target_model: dict[str, Any],
    optimizer: Any,
    replay: PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    device: Any,
) -> tuple[float, float]:
    sample = replay.sample(batch_size=batch_size)
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
        target_q = rewards + (1.0 - dones) * gamma * next_target_q
    td_errors = target_q - selected_q_values
    loss = torch.mean(weights * torch.square(td_errors))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    td_np = np.asarray(td_errors.detach().cpu().numpy(), dtype=np.float64)
    replay.update_priorities(indices=sample.indices, td_errors=td_np)
    return float(loss.detach().cpu()), float(np.mean(np.abs(td_np), dtype=np.float64))


def _compute_validation_curve_point_v1(
    *,
    torch: Any,
    model: dict[str, Any],
    target_model: dict[str, Any],
    transitions: TrainingTransitionSet,
    batch_size: int,
    gamma: float,
    max_transitions: int,
    device: Any,
) -> dict[str, object]:
    limit = min(transitions.transition_count, max_transitions)
    if limit <= 0:
        raise TrainingRunnerError(reason="validation_transitions_required")
    losses: list[float] = []
    td_means: list[float] = []
    with torch.no_grad():
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            observations = torch.as_tensor(
                transitions.observations[start:end],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                transitions.actions[start:end],
                dtype=torch.long,
                device=device,
            )
            rewards = torch.as_tensor(
                transitions.rewards[start:end],
                dtype=torch.float32,
                device=device,
            )
            next_observations = torch.as_tensor(
                transitions.next_observations[start:end],
                dtype=torch.float32,
                device=device,
            )
            dones = torch.as_tensor(
                transitions.dones[start:end].astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            q_values = _torch_d3qn_forward(model, observations)
            selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_actions = _torch_d3qn_forward(model, next_observations).argmax(dim=1)
            next_target_q = _torch_d3qn_forward(target_model, next_observations).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_q = rewards + (1.0 - dones) * gamma * next_target_q
            td_errors = target_q - selected_q_values
            losses.append(float(torch.mean(torch.square(td_errors)).detach().cpu()))
            td_np = np.asarray(td_errors.detach().cpu().numpy(), dtype=np.float64)
            td_means.append(float(np.mean(np.abs(td_np), dtype=np.float64)))
    return {
        "batch_count": len(losses),
        "max_transitions": max_transitions,
        "mean_abs_td_error": _round_float(float(np.mean(np.asarray(td_means)))),
        "sampled_transition_count": limit,
        "td_mse": _round_float(float(np.mean(np.asarray(losses, dtype=np.float64)))),
    }


def _save_stage07b_checkpoint_v1(
    *,
    torch: Any,
    path: Path,
    latest_checkpoint_path: Path,
    model: dict[str, Any],
    target_model: dict[str, Any],
    optimizer: Any,
    replay: PrioritizedReplayBuffer,
    completed_training_steps: int,
    run_id: str,
    config_hash: str,
    architecture_hash: str,
    dataset_manifest_sha256: str,
    train_curve: list[dict[str, object]],
    validation_curve: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture_hash": architecture_hash,
        "completed_training_steps": completed_training_steps,
        "config_hash": config_hash,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_state": _cpu_state_dict_payload(model),
        "optimizer_state": optimizer.state_dict(),
        "replay_priority_state": replay.priority_state_payload(),
        "run_id": run_id,
        "schema_version": STAGE07B_CANDIDATE_RUN_SCHEMA_VERSION_V1,
        "stage": "07B",
        "target_model_state": _cpu_state_dict_payload(target_model),
        "train_curve": train_curve,
        "validation_curve": validation_curve,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    _atomic_write_json(
        latest_checkpoint_path,
        {
            "checkpoint": _file_payload(path),
            "completed_training_steps": completed_training_steps,
            "run_id": run_id,
            "stage": "07B",
        },
    )


def _load_latest_stage07b_checkpoint_payload(
    *,
    torch: Any,
    latest_checkpoint_path: Path,
    device: Any,
) -> dict[str, object] | None:
    if not latest_checkpoint_path.exists():
        return None
    latest = _read_json_payload(latest_checkpoint_path)
    checkpoint = latest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TrainingRunnerError(reason="latest_checkpoint_payload_invalid")
    path_value = checkpoint.get("path")
    if not isinstance(path_value, str):
        raise TrainingRunnerError(reason="latest_checkpoint_path_missing")
    checkpoint_path = Path(path_value)
    if not checkpoint_path.exists():
        raise TrainingRunnerError(reason="latest_checkpoint_file_missing", field=path_value)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TrainingRunnerError(reason="stage07b_checkpoint_payload_invalid")
    return payload


def _load_torch_modules_state(*, target: dict[str, Any], state: object) -> None:
    if not isinstance(state, dict):
        raise TrainingRunnerError(reason="torch_module_state_invalid")
    for name in ("shared", "value", "advantage"):
        if name not in state:
            raise TrainingRunnerError(reason="torch_module_state_missing", field=name)
        target[name].load_state_dict(state[name])


def _stage07b_training_plan_payload(config: CandidateTrainingConfig) -> dict[str, object]:
    return {
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "device_policy": config.device_policy,
        "planned_training_steps": config.planned_training_steps,
        "progress_emit_every_sec": config.progress_emit_every_sec,
        "progress_emit_every_steps": config.progress_emit_every_steps,
        "progress_pct_rule": "completed_training_steps / planned_training_steps * 100",
        "resume_behavior": (
            "rebuild deterministic replay transitions from the frozen Stage 06 manifest, "
            "then restore model, target model, optimizer, replay priorities and replay RNG "
            "from latest_checkpoint.json"
        ),
        "train_dataset_version": config.train_dataset_version,
        "train_split": config.train_split,
        "validation_dataset_version": config.validation_dataset_version,
        "validation_every_steps": config.validation_every_steps,
        "validation_split": config.validation_split,
    }


def _stage07b_safety_payload() -> dict[str, object]:
    return {
        "candidate_for_stage08_evaluation_only": True,
        "contains_raw_provider_payloads": False,
        "contains_secrets": False,
        "exchange_side_effects": False,
        "mainnet_submit": False,
        "model_registry_write": False,
        "paper_testnet_live_enabled": False,
        "promotion_or_activation": False,
        "runtime_artifact_root": STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    }


def _stage07b_resource_usage_payload(
    *,
    torch: Any,
    start_usage: Any,
    end_usage: Any,
    device_payload: dict[str, object],
    progress_writer: _Stage07bProgressWriter,
    completed_training_steps: int,
) -> dict[str, object]:
    elapsed = progress_writer.elapsed_sec()
    return {
        "completed_training_steps": completed_training_steps,
        "cpu_system_seconds_delta": _round_float(end_usage.ru_stime - start_usage.ru_stime),
        "cpu_user_seconds_delta": _round_float(end_usage.ru_utime - start_usage.ru_utime),
        "mps_available": bool(device_payload["mps_available"]),
        "mps_built": bool(device_payload["mps_built"]),
        "process_threads_observed": _process_thread_count(),
        "rss_mb_after": _rss_mb(),
        "selected_device": device_payload["selected_device"],
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_num_threads": int(torch.get_num_threads()),
        "wall_seconds": elapsed,
    }


def _resource_snapshot_payload() -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_system_seconds": _round_float(usage.ru_stime),
        "cpu_user_seconds": _round_float(usage.ru_utime),
        "process_threads_observed": _process_thread_count(),
        "rss_mb": _rss_mb(),
    }


def _read_json_payload(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _render_json_line_payload(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
    _configure_torch_threads_for_values(
        torch=torch,
        torch_num_threads=config.torch_num_threads,
        torch_num_interop_threads=config.torch_num_interop_threads,
    )


def _configure_torch_threads_for_values(
    *,
    torch: Any,
    torch_num_threads: int,
    torch_num_interop_threads: int,
) -> None:
    torch.set_num_threads(torch_num_threads)
    current_interop_threads = int(torch.get_num_interop_threads())
    if current_interop_threads != torch_num_interop_threads:
        torch.set_num_interop_threads(torch_num_interop_threads)


def _select_torch_device(
    *,
    torch: Any,
    config: TrainingSmokeConfig,
) -> tuple[Any, dict[str, object]]:
    return _select_torch_device_for_policy(torch=torch, device_policy=config.device_policy)


def _select_torch_device_for_policy(
    *,
    torch: Any,
    device_policy: DevicePolicy,
) -> tuple[Any, dict[str, object]]:
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    selected_device = "cpu"
    if device_policy == "mps_preferred_cpu_fallback" and mps_available:
        selected_device = "mps"
    device = torch.device(selected_device)
    return device, {
        "device_policy": device_policy,
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


def _non_empty_text(value: str, field: str) -> str:
    if not value.strip():
        raise TrainingRunnerError(reason="empty_text", field=field)
    return value
