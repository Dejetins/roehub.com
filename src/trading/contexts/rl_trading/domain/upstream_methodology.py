from __future__ import annotations

import importlib
import math
import random
import resource
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from .action_state_reward_contract import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RL_ACTION_COUNT_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
    build_state_extras_v1,
    encode_action_history_v1,
    normalize_rl_action_id_v1,
)
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_DTYPE_V1, FEATURE_NAMES_V1
from .raw_feature_dataset import (
    hash_json_payload_v1,
    render_raw_feature_json_payload_v1,
    training_source_gate_payload_v1,
)
from .sessionized_dataset import (
    SESSIONIZED_AGENT_HISTORY_LEN_V1,
    SESSIONIZED_AGENT_SESSION_LEN_V1,
    SESSIONIZED_FULL_SEQ_LEN_V1,
    SESSIONIZED_PRE_SIGNAL_LEN_V1,
)

UPSTREAM_METHODOLOGY_PARITY_ID_V1 = "upstream_methodology_parity"
UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1 = "roehub_d3qn_cnn_dueling_v1"
UPSTREAM_SOURCE_SHA_STAGE08A_V1 = "f71130903f8237351164f4b875494185465bf1ea"
STAGE08B_CORE_SMOKE_KIND_V1 = "rl_trading_stage08b_upstream_methodology_core_smoke"
STAGE08B_CORE_SMOKE_SCHEMA_VERSION_V1 = 1
STAGE08B_CORE_SMOKE_CONFIG_ID_V1 = "roehub_stage08b_upstream_methodology_core_smoke_v1"

UPSTREAM_PRICE_CHANNELS_V1: tuple[str, ...] = (
    "open",
    "high",
    "volume_weighted_average",
    "low",
    "close",
)
UPSTREAM_VOLUME_CHANNELS_V1: tuple[str, ...] = ("volume", "num_trades")

SelectionStrategy = Literal["advantage_based_filter", "ensemble_q_filter"]
SelectionMode = Literal["epsilon_random", "greedy", "cache_hit"]


class UpstreamMethodologyError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class UpstreamAlphaConfig:
    seed: int = 25
    full_seq_len: int = SESSIONIZED_FULL_SEQ_LEN_V1
    pre_signal_len: int = SESSIONIZED_PRE_SIGNAL_LEN_V1
    agent_history_len: int = SESSIONIZED_AGENT_HISTORY_LEN_V1
    agent_session_len: int = SESSIONIZED_AGENT_SESSION_LEN_V1
    action_history_len: int = 3
    initial_balance: float = 10_000.0
    transaction_fee: float = 0.0004
    slippage: float = 0.00025
    inaction_penalty_ratio: float = 0.001
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 16
    target_update_freq: int = 100
    train_start: int = 10_000
    max_gradient_norm: float = 1.0
    replay_capacity: int = 230_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 20_000
    per_epsilon: float = 1e-6
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frames: int = 50_000
    cnn_maps: tuple[int, ...] = (32, 64, 128)
    cnn_kernels: tuple[int, ...] = (7, 5, 3)
    cnn_strides: tuple[int, ...] = (2, 1, 1)
    dense_val: tuple[int, ...] = (128, 64)
    dense_adv: tuple[int, ...] = (128, 64)
    dropout_p: float = 0.1
    long_action_threshold: float = 0.012695
    short_action_threshold: float = 0.009902
    close_action_threshold: float = 0.001141
    ensemble_n_samples: int = 5
    ensemble_max_sigma: float = 0.01
    max_parallel_sessions: int = 2
    position_fraction: float = 0.5
    torch_num_threads: int = 1
    torch_num_interop_threads: int = 1
    dtype: str = FEATURE_DTYPE_V1

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise UpstreamMethodologyError(reason="invalid_seed", field="seed")
        for field_name in (
            "full_seq_len",
            "pre_signal_len",
            "agent_history_len",
            "agent_session_len",
            "action_history_len",
            "batch_size",
            "target_update_freq",
            "train_start",
            "replay_capacity",
            "per_beta_frames",
            "eps_decay_frames",
            "ensemble_n_samples",
            "max_parallel_sessions",
            "torch_num_threads",
            "torch_num_interop_threads",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.full_seq_len != SESSIONIZED_FULL_SEQ_LEN_V1:
            raise UpstreamMethodologyError(reason="unexpected_full_seq_len", field="full_seq_len")
        if self.pre_signal_len >= self.full_seq_len:
            raise UpstreamMethodologyError(reason="pre_signal_len_out_of_range")
        if self.agent_history_len < 2:
            raise UpstreamMethodologyError(
                reason="agent_history_len_too_short",
                field="agent_history_len",
            )
        if self.action_history_len > self.agent_session_len:
            raise UpstreamMethodologyError(
                reason="action_history_len_exceeds_session_len",
                field="action_history_len",
            )
        for field_name in (
            "initial_balance",
            "gamma",
            "learning_rate",
            "max_gradient_norm",
            "per_alpha",
            "per_beta_start",
            "per_epsilon",
            "ensemble_max_sigma",
            "position_fraction",
        ):
            _positive_float(getattr(self, field_name), field_name)
        for field_name in ("transaction_fee", "slippage", "inaction_penalty_ratio"):
            _non_negative_float(getattr(self, field_name), field_name)
        for field_name in (
            "gamma",
            "per_alpha",
            "per_beta_start",
            "eps_start",
            "eps_end",
            "dropout_p",
        ):
            _bounded_float(getattr(self, field_name), field_name, low=0.0, high=1.0)
        if self.eps_start < self.eps_end:
            raise UpstreamMethodologyError(reason="eps_start_below_eps_end")
        if len(self.cnn_maps) != len(self.cnn_kernels) or len(self.cnn_maps) != len(
            self.cnn_strides
        ):
            raise UpstreamMethodologyError(reason="cnn_layer_config_length_mismatch")
        for field_name in ("cnn_maps", "cnn_kernels", "cnn_strides", "dense_val", "dense_adv"):
            values = getattr(self, field_name)
            if not values:
                raise UpstreamMethodologyError(reason="empty_config_sequence", field=field_name)
            for index, value in enumerate(values):
                _positive_int(value, f"{field_name}[{index}]")
        if self.dtype != FEATURE_DTYPE_V1:
            raise UpstreamMethodologyError(reason="unsupported_dtype", field="dtype")

    @property
    def input_history_len(self) -> int:
        return self.agent_history_len - 1

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES_V1)

    @property
    def flat_history_size(self) -> int:
        return self.input_history_len * self.feature_count

    @property
    def additional_feature_count(self) -> int:
        return len(build_state_extras_v1(
            position_side=None,
            entry_price=None,
            current_price=1.0,
            step_idx=0,
            session_len=self.agent_session_len,
        )) + (self.action_history_len * RL_ACTION_COUNT_V1)

    @property
    def state_dim(self) -> int:
        return self.flat_history_size + self.additional_feature_count

    @property
    def cnn_input_shape(self) -> tuple[int, int, int]:
        return (self.feature_count, self.input_history_len, 1)

    def as_payload(self) -> dict[str, object]:
        return {
            "action_history_len": self.action_history_len,
            "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
            "batch_size": self.batch_size,
            "cnn_input_shape": list(self.cnn_input_shape),
            "cnn_kernels": list(self.cnn_kernels),
            "cnn_maps": list(self.cnn_maps),
            "cnn_strides": list(self.cnn_strides),
            "config_id": STAGE08B_CORE_SMOKE_CONFIG_ID_V1,
            "dense_adv": list(self.dense_adv),
            "dense_val": list(self.dense_val),
            "dropout_p": self.dropout_p,
            "eps_decay_frames": self.eps_decay_frames,
            "eps_end": self.eps_end,
            "eps_start": self.eps_start,
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "feature_names": list(FEATURE_NAMES_V1),
            "gamma": self.gamma,
            "initial_balance": self.initial_balance,
            "input_history_len": self.input_history_len,
            "learning_rate": self.learning_rate,
            "max_gradient_norm": self.max_gradient_norm,
            "methodology_parity_id": UPSTREAM_METHODOLOGY_PARITY_ID_V1,
            "per_alpha": self.per_alpha,
            "per_beta_frames": self.per_beta_frames,
            "per_beta_start": self.per_beta_start,
            "per_epsilon": self.per_epsilon,
            "replay_capacity": self.replay_capacity,
            "seed": self.seed,
            "state_dim": self.state_dim,
            "target_update_freq": self.target_update_freq,
            "train_start": self.train_start,
            "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    means: Mapping[str, float]
    stds: Mapping[str, float]
    source_split: str
    sequence_count: int
    feature_names: tuple[str, ...] = FEATURE_NAMES_V1

    def as_payload(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "means": {key: float(self.means[key]) for key in sorted(self.means)},
            "source_split": self.source_split,
            "stds": {key: float(self.stds[key]) for key in sorted(self.stds)},
            "train_only": self.source_split == "train",
            "sequence_count": self.sequence_count,
            "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
        }

    def stats_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


@dataclass(frozen=True, slots=True)
class UpstreamReplaySample:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    tree_indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class ActionSelection:
    action_id: int
    epsilon: float
    mode: SelectionMode
    q_values: tuple[float, ...] | None
    valid_actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LearningStepResult:
    loss: float
    mean_abs_td_error: float
    gradient_norm_before_clip: float
    target_synced: bool


@dataclass(frozen=True, slots=True)
class FilteredDecision:
    requested_action_id: int
    effective_action_id: int
    rejected: bool
    rejection_reason: str | None
    confidence: float
    uncertainty: float | None
    q_values: tuple[float, ...]


@dataclass(slots=True)
class QValueCache:
    _values: dict[Hashable, np.ndarray] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_compute(
        self,
        key: Hashable,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        if key in self._values:
            self.hits += 1
            return np.asarray(self._values[key], dtype=np.float32)
        self.misses += 1
        value = np.ascontiguousarray(compute(), dtype=np.float32)
        if value.shape != (RL_ACTION_COUNT_V1,):
            raise UpstreamMethodologyError(reason="q_value_shape_mismatch")
        self._values[key] = value
        return value

    def stats_payload(self) -> dict[str, object]:
        return {
            "cache_entries": len(self._values),
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass(slots=True)
class FilteredBacktestPolicy:
    long_action_threshold: float
    short_action_threshold: float
    close_action_threshold: float
    ensemble_max_sigma: float
    selection_strategy: SelectionStrategy = "advantage_based_filter"
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: UpstreamAlphaConfig | None = None,
        *,
        selection_strategy: SelectionStrategy = "advantage_based_filter",
    ) -> FilteredBacktestPolicy:
        selected_config = default_upstream_alpha_config_v1() if config is None else config
        return cls(
            long_action_threshold=selected_config.long_action_threshold,
            short_action_threshold=selected_config.short_action_threshold,
            close_action_threshold=selected_config.close_action_threshold,
            ensemble_max_sigma=selected_config.ensemble_max_sigma,
            selection_strategy=selection_strategy,
        )

    def select_from_q_values(
        self,
        q_values: np.ndarray,
        *,
        q_std: np.ndarray | None = None,
    ) -> FilteredDecision:
        q = _q_values_1d(q_values)
        std = None if q_std is None else _q_values_1d(q_std)
        advantage = q - q[0]
        requested_action = int(np.argmax(advantage))
        confidence = float(advantage[requested_action])
        uncertainty = None if std is None else float(std[requested_action])
        rejection_reason: str | None = None
        if requested_action != 0:
            threshold = self._threshold_for_action(requested_action)
            if confidence <= threshold:
                rejection_reason = "weak_advantage_threshold"
            if (
                self.selection_strategy == "ensemble_q_filter"
                and uncertainty is not None
                and uncertainty >= self.ensemble_max_sigma
            ):
                rejection_reason = (
                    "high_ensemble_uncertainty"
                    if rejection_reason is None
                    else f"{rejection_reason}+high_ensemble_uncertainty"
                )
        effective_action = 0 if rejection_reason is not None else requested_action
        if rejection_reason is not None:
            self.rejection_counts[rejection_reason] = (
                self.rejection_counts.get(rejection_reason, 0) + 1
            )
        return FilteredDecision(
            requested_action_id=requested_action,
            effective_action_id=effective_action,
            rejected=rejection_reason is not None,
            rejection_reason=rejection_reason,
            confidence=confidence,
            uncertainty=uncertainty,
            q_values=tuple(float(value) for value in q),
        )

    def _threshold_for_action(self, action_id: int) -> float:
        if action_id == 1:
            return self.long_action_threshold
        if action_id == 2:
            return self.short_action_threshold
        if action_id == 3:
            return self.close_action_threshold
        return 0.0

    def stats_payload(self) -> dict[str, object]:
        return {
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "selection_strategy": self.selection_strategy,
            "thresholds": {
                "close": self.close_action_threshold,
                "ensemble_max_sigma": self.ensemble_max_sigma,
                "long": self.long_action_threshold,
                "short": self.short_action_threshold,
            },
        }


class QValueProvider(Protocol):
    def __call__(self, state: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class CheckpointSelectionPolicy:
    validation_metric_name: str = "Validation_mean_pnl"
    higher_is_better: bool = True
    default_evaluation_checkpoint: Literal["best", "final"] = "best"


def default_upstream_alpha_config_v1() -> UpstreamAlphaConfig:
    return UpstreamAlphaConfig()


def compute_train_only_normalization_stats_v1(
    train_sequences: np.ndarray | Sequence[np.ndarray],
    *,
    config: UpstreamAlphaConfig | None = None,
) -> NormalizationStats:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    sequences = _as_sequence_list(train_sequences, config=selected_config)
    if not sequences:
        raise UpstreamMethodologyError(reason="train_sequences_required")

    values_by_channel: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES_V1}
    for sequence in sequences:
        for channel_idx, channel_name in enumerate(FEATURE_NAMES_V1):
            arr = np.asarray(sequence[:, channel_idx], dtype=np.float64)
            if channel_name in UPSTREAM_PRICE_CHANNELS_V1:
                transformed = np.log(np.maximum(arr[1:] / (arr[:-1] + 1e-9), 1e-9))
            elif channel_name in UPSTREAM_VOLUME_CHANNELS_V1:
                transformed = np.log(np.maximum(arr, 0.0) + 1.0)
            else:
                transformed = arr
            finite = transformed[np.isfinite(transformed)]
            values_by_channel[channel_name].extend(float(value) for value in finite)

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for channel_name, values in values_by_channel.items():
        if not values:
            means[channel_name] = 0.0
            stds[channel_name] = 1.0
            continue
        arr = np.asarray(values, dtype=np.float64)
        means[channel_name] = float(np.mean(arr))
        std = float(np.std(arr))
        stds[channel_name] = 1.0 if std < 1e-7 else std
    return NormalizationStats(
        means=means,
        stds=stds,
        source_split="train",
        sequence_count=len(sequences),
    )


def apply_upstream_normalization_v1(
    window: np.ndarray,
    stats: NormalizationStats,
    *,
    config: UpstreamAlphaConfig | None = None,
) -> np.ndarray:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    values = np.asarray(window, dtype=np.float32)
    expected_shape = (selected_config.agent_history_len, len(FEATURE_NAMES_V1))
    if values.shape != expected_shape:
        raise UpstreamMethodologyError(
            reason="normalization_window_shape_mismatch",
            field=str(values.shape),
        )
    out = np.zeros(
        (selected_config.input_history_len, len(FEATURE_NAMES_V1)),
        dtype=np.float32,
    )
    for channel_idx, channel_name in enumerate(FEATURE_NAMES_V1):
        arr = values[:, channel_idx].astype(np.float64)
        mean = float(stats.means.get(channel_name, 0.0))
        std = float(stats.stds.get(channel_name, 1.0))
        if std <= 0.0 or not math.isfinite(std):
            std = 1.0
        if channel_name in UPSTREAM_PRICE_CHANNELS_V1:
            transformed = np.log(np.maximum(arr[1:] / (arr[:-1] + 1e-9), 1e-9))
        elif channel_name in UPSTREAM_VOLUME_CHANNELS_V1:
            transformed = np.log(np.maximum(arr, 0.0) + 1.0)[-selected_config.input_history_len :]
        else:
            transformed = arr[-selected_config.input_history_len :]
        normalized = (transformed - mean) / std
        out[:, channel_idx] = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def build_upstream_state_v1(
    *,
    session: np.ndarray,
    step_idx: int,
    action_history: Sequence[int | None],
    training_state: RlTrainingState,
    normalization_stats: NormalizationStats,
    config: UpstreamAlphaConfig | None = None,
) -> np.ndarray:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    sequence = _validate_session(session, config=selected_config)
    if step_idx < 0 or step_idx > selected_config.agent_session_len:
        raise UpstreamMethodologyError(reason="invalid_step_idx", field="step_idx")
    end = selected_config.pre_signal_len + step_idx
    start = end - selected_config.agent_history_len
    if start < 0 or end > sequence.shape[0]:
        raise UpstreamMethodologyError(reason="state_window_out_of_range")
    window = sequence[start:end]
    normalized = apply_upstream_normalization_v1(
        window,
        normalization_stats,
        config=selected_config,
    )
    price = session_close_price_v1(sequence, step_idx=step_idx, config=selected_config)
    extras = build_state_extras_v1(
        position_side=training_state.position_side,
        entry_price=training_state.entry_price,
        current_price=price,
        step_idx=step_idx,
        session_len=selected_config.agent_session_len,
    )
    history = _trim_action_history(
        action_history,
        action_history_len=selected_config.action_history_len,
    )
    encoded_history = encode_action_history_v1(history)
    state = np.concatenate(
        [
            normalized.reshape(-1),
            np.asarray(extras, dtype=np.float32),
            np.asarray(encoded_history, dtype=np.float32),
        ],
    )
    if state.shape != (selected_config.state_dim,):
        raise UpstreamMethodologyError(reason="state_dim_mismatch", field=str(state.shape))
    return np.ascontiguousarray(state, dtype=np.float32)


def session_close_price_v1(
    session: np.ndarray,
    *,
    step_idx: int,
    config: UpstreamAlphaConfig | None = None,
) -> float:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    sequence = _validate_session(session, config=selected_config)
    price_idx = min(
        selected_config.pre_signal_len - 1 + max(step_idx, 0),
        sequence.shape[0] - 1,
    )
    return _positive_float(
        float(sequence[price_idx, FEATURE_NAMES_V1.index("close")]),
        "close_price",
    )


class UpstreamTradingEnvironment:
    def __init__(
        self,
        *,
        sequences: np.ndarray | Sequence[np.ndarray],
        normalization_stats: NormalizationStats,
        config: UpstreamAlphaConfig | None = None,
    ) -> None:
        self.config = default_upstream_alpha_config_v1() if config is None else config
        self.sequences = _as_sequence_list(sequences, config=self.config)
        if not self.sequences:
            raise UpstreamMethodologyError(reason="sequences_required")
        self.normalization_stats = normalization_stats
        self._rng = np.random.default_rng(self.config.seed)
        self.current_seq: np.ndarray | None = None
        self.current_index = 0
        self.step_idx = 0
        self.state = RlTrainingState(balance=self.config.initial_balance)
        self.action_history: list[int | None] = [None] * self.config.action_history_len

    def reset(self, *, forced_index: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if forced_index is None:
            selected_index = int(self._rng.integers(0, len(self.sequences)))
        else:
            if forced_index < 0 or forced_index >= len(self.sequences):
                raise UpstreamMethodologyError(reason="forced_index_out_of_range")
            selected_index = forced_index
        self.current_index = selected_index
        self.current_seq = self.sequences[selected_index]
        self.step_idx = 0
        self.state = RlTrainingState(balance=self.config.initial_balance)
        self.action_history = [None] * self.config.action_history_len
        return self.observation(), self.info_payload()

    def observation(self) -> np.ndarray:
        if self.current_seq is None:
            raise UpstreamMethodologyError(reason="environment_not_reset")
        return build_upstream_state_v1(
            session=self.current_seq,
            step_idx=self.step_idx,
            action_history=self.action_history,
            training_state=self.state,
            normalization_stats=self.normalization_stats,
            config=self.config,
        )

    def valid_actions(self) -> tuple[int, ...]:
        return valid_upstream_training_actions_v1(
            position_side=self.state.position_side,
            is_last_step=self.step_idx == self.config.agent_session_len - 1,
        )

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.current_seq is None:
            raise UpstreamMethodologyError(reason="environment_not_reset")
        requested_action = normalize_rl_action_id_v1(action_id)
        masked_action = mask_upstream_training_action_v1(
            action_id=requested_action,
            position_side=self.state.position_side,
            is_last_step=self.step_idx == self.config.agent_session_len - 1,
        )
        action_for_reward = (
            requested_action
            if self.step_idx == self.config.agent_session_len - 1
            else masked_action
        )
        price = session_close_price_v1(self.current_seq, step_idx=self.step_idx, config=self.config)
        result = apply_training_reward_step_v1(
            state=self.state,
            action_id=action_for_reward,
            price=price,
            initial_balance=self.config.initial_balance,
            slippage=self.config.slippage,
            transaction_fee=self.config.transaction_fee,
            inaction_penalty_ratio=self.config.inaction_penalty_ratio,
            is_last_step=self.step_idx == self.config.agent_session_len - 1,
        )
        self.state = result.state
        self.action_history = [*self.action_history[1:], result.effective_action_id]
        self.step_idx += 1
        terminated = self.step_idx >= self.config.agent_session_len
        observation = (
            np.zeros((self.config.state_dim,), dtype=np.float32)
            if terminated
            else self.observation()
        )
        info = self.info_payload()
        info.update(
            {
                "audit_reason": result.audit_reason,
                "closed_position": result.closed_position,
                "effective_action_id": result.effective_action_id,
                "effective_action_name": result.effective_action_name,
                "masked_action_id": masked_action,
                "pnl_change": _round_float(result.pnl_change),
                "requested_action_id": requested_action,
            }
        )
        if terminated:
            info.update(
                {
                    "episode_closed_trades": result.state.closed_trades,
                    "episode_profitable_trades": result.state.profitable_trades,
                    "episode_realized_pnl": _round_float(result.state.realized_pnl),
                }
            )
        return observation, float(result.reward), terminated, False, info

    def info_payload(self) -> dict[str, object]:
        return {
            "balance": _round_float(self.state.balance),
            "current_index": self.current_index,
            "position_side": self.state.position_side,
            "realized_pnl": _round_float(self.state.realized_pnl),
            "step": self.step_idx,
        }


def valid_upstream_training_actions_v1(
    *,
    position_side: str | None,
    is_last_step: bool,
) -> tuple[int, ...]:
    if is_last_step:
        return (0,) if position_side is None else (3,)
    return (0, 1, 2) if position_side is None else (0, 3)


def mask_upstream_training_action_v1(
    *,
    action_id: int,
    position_side: str | None,
    is_last_step: bool,
) -> int:
    normalized = normalize_rl_action_id_v1(action_id)
    if normalized in valid_upstream_training_actions_v1(
        position_side=position_side,
        is_last_step=is_last_step,
    ):
        return normalized
    if is_last_step and position_side is not None:
        return 3
    return 0


class SumTreePrioritizedReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_frames: int,
        epsilon: float,
        seed: int,
    ) -> None:
        _positive_int(capacity, "capacity")
        _positive_int(beta_frames, "beta_frames")
        _bounded_float(alpha, "alpha", low=0.0, high=1.0)
        _bounded_float(beta_start, "beta_start", low=0.0, high=1.0)
        _positive_float(epsilon, "epsilon")
        if isinstance(seed, bool) or seed < 0:
            raise UpstreamMethodologyError(reason="invalid_seed", field="seed")
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame_idx = 0
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity <<= 1
        self.tree = np.zeros((2 * self.tree_capacity) - 1, dtype=np.float64)
        self.data: list[tuple[np.ndarray, int, float, np.ndarray, bool] | None] = (
            [None] * capacity
        )
        self.idx = 0
        self.size = 0
        self.max_priority = 1.0
        self.state_dim: int | None = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_value = _state_1d(state, field="state", expected_dim=self.state_dim)
        if self.state_dim is None:
            self.state_dim = int(state_value.shape[0])
        next_state_value = _state_1d(
            next_state,
            field="next_state",
            expected_dim=self.state_dim,
        )
        normalized_action = normalize_rl_action_id_v1(action)
        reward_value = _finite_float(reward, "reward")
        self.data[self.idx] = (
            state_value,
            normalized_action,
            reward_value,
            next_state_value,
            bool(done),
        )
        tree_idx = self.idx + self.tree_capacity - 1
        self._update_tree(tree_idx, self.max_priority**self.alpha)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame_idx += 1

    def sample(self, batch_size: int) -> UpstreamReplaySample:
        _positive_int(batch_size, "batch_size")
        if self.size < batch_size:
            raise UpstreamMethodologyError(reason="not_enough_samples_for_batch")
        total_priority = float(self.tree[0])
        if total_priority <= 0.0 or not math.isfinite(total_priority):
            raise UpstreamMethodologyError(reason="invalid_total_priority")
        segment = total_priority / float(batch_size)
        beta = min(
            1.0,
            self.beta_start
            + (float(self.frame_idx) * (1.0 - self.beta_start) / float(self.beta_frames)),
        )
        leaves = self.tree[self.tree_capacity - 1 : self.tree_capacity - 1 + self.size]
        positive_leaf = leaves[leaves > 0.0]
        min_prob = (
            float(np.min(positive_leaf)) / total_priority
            if positive_leaf.size
            else 1.0 / float(self.size)
        )
        max_weight = (min_prob * float(self.size)) ** (-beta)
        rows: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        tree_indices: list[int] = []
        weights: list[float] = []
        for batch_idx in range(batch_size):
            left = segment * float(batch_idx)
            right = segment * float(batch_idx + 1)
            value = float(self._rng.uniform(left, right))
            tree_idx = self._retrieve(0, value)
            data_idx = tree_idx - (self.tree_capacity - 1)
            if data_idx < 0 or data_idx >= self.size:
                data_idx = min(max(data_idx, 0), self.size - 1)
                tree_idx = data_idx + self.tree_capacity - 1
            row = self.data[data_idx]
            if row is None:
                raise UpstreamMethodologyError(reason="sampled_empty_replay_row")
            sample_prob = float(self.tree[tree_idx]) / total_priority
            weight = ((sample_prob * float(self.size)) ** (-beta)) / max(max_weight, 1e-12)
            rows.append(row)
            tree_indices.append(tree_idx)
            weights.append(weight)
        states, actions, rewards, next_states, dones = zip(*rows, strict=True)
        return UpstreamReplaySample(
            states=np.ascontiguousarray(np.vstack(states), dtype=np.float32),
            actions=np.ascontiguousarray(np.asarray(actions, dtype=np.int64)),
            rewards=np.ascontiguousarray(np.asarray(rewards, dtype=np.float32)),
            next_states=np.ascontiguousarray(np.vstack(next_states), dtype=np.float32),
            dones=np.ascontiguousarray(np.asarray(dones, dtype=np.bool_)),
            tree_indices=np.ascontiguousarray(np.asarray(tree_indices, dtype=np.int64)),
            weights=np.ascontiguousarray(np.asarray(weights, dtype=np.float32)),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        indices = np.asarray(tree_indices, dtype=np.int64)
        errors = np.asarray(td_errors, dtype=np.float64)
        if indices.ndim != 1 or errors.ndim != 1 or indices.shape[0] != errors.shape[0]:
            raise UpstreamMethodologyError(reason="priority_update_shape_mismatch")
        for tree_idx, error in zip(indices, errors, strict=True):
            if tree_idx < self.tree_capacity - 1 or tree_idx >= len(self.tree):
                raise UpstreamMethodologyError(reason="priority_tree_index_out_of_range")
            priority = (abs(_finite_float(float(error), "td_error")) + self.epsilon) ** self.alpha
            self._update_tree(int(tree_idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def state_payload(self) -> dict[str, object]:
        state_dim = self.state_dim
        if state_dim is None:
            state_dim = 0
        states = np.zeros((self.size, state_dim), dtype=np.float32)
        next_states = np.zeros((self.size, state_dim), dtype=np.float32)
        actions = np.zeros((self.size,), dtype=np.int64)
        rewards = np.zeros((self.size,), dtype=np.float32)
        dones = np.zeros((self.size,), dtype=np.bool_)
        for idx in range(self.size):
            row = self.data[idx]
            if row is None:
                raise UpstreamMethodologyError(reason="replay_state_missing_row")
            state, action, reward, next_state, done = row
            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward
            next_states[idx] = next_state
            dones[idx] = done
        return {
            "actions": actions,
            "alpha": self.alpha,
            "beta_frames": self.beta_frames,
            "beta_start": self.beta_start,
            "capacity": self.capacity,
            "dones": dones,
            "epsilon": self.epsilon,
            "frame_idx": self.frame_idx,
            "idx": self.idx,
            "max_priority": self.max_priority,
            "next_states": next_states,
            "rewards": rewards,
            "rng_state": self._rng.bit_generator.state,
            "size": self.size,
            "state_dim": state_dim,
            "states": states,
            "tree": np.ascontiguousarray(self.tree, dtype=np.float64),
            "tree_capacity": self.tree_capacity,
        }

    def restore_state_payload(self, payload: Mapping[str, object]) -> None:
        capacity = _payload_int(payload.get("capacity"), "capacity")
        tree_capacity = _payload_int(payload.get("tree_capacity"), "tree_capacity")
        size = _payload_int(payload.get("size"), "size")
        idx = _payload_int(payload.get("idx"), "idx")
        state_dim = _payload_int(payload.get("state_dim"), "state_dim")
        if capacity != self.capacity:
            raise UpstreamMethodologyError(reason="replay_capacity_mismatch")
        if tree_capacity != self.tree_capacity:
            raise UpstreamMethodologyError(reason="replay_tree_capacity_mismatch")
        if size < 0 or size > self.capacity:
            raise UpstreamMethodologyError(reason="replay_size_out_of_range")
        if idx < 0 or idx >= self.capacity:
            raise UpstreamMethodologyError(reason="replay_idx_out_of_range")
        tree = np.asarray(payload.get("tree"), dtype=np.float64)
        if tree.shape != self.tree.shape:
            raise UpstreamMethodologyError(reason="replay_tree_shape_mismatch")
        states = np.asarray(payload.get("states"), dtype=np.float32)
        next_states = np.asarray(payload.get("next_states"), dtype=np.float32)
        actions = np.asarray(payload.get("actions"), dtype=np.int64)
        rewards = np.asarray(payload.get("rewards"), dtype=np.float32)
        dones = np.asarray(payload.get("dones"), dtype=np.bool_)
        expected_2d = (size, state_dim)
        if states.shape != expected_2d or next_states.shape != expected_2d:
            raise UpstreamMethodologyError(reason="replay_state_array_shape_mismatch")
        if actions.shape != (size,) or rewards.shape != (size,) or dones.shape != (size,):
            raise UpstreamMethodologyError(reason="replay_vector_shape_mismatch")
        self.tree = np.ascontiguousarray(tree, dtype=np.float64)
        self.data = [None] * self.capacity
        for row_idx in range(size):
            self.data[row_idx] = (
                np.ascontiguousarray(states[row_idx], dtype=np.float32),
                normalize_rl_action_id_v1(int(actions[row_idx])),
                float(rewards[row_idx]),
                np.ascontiguousarray(next_states[row_idx], dtype=np.float32),
                bool(dones[row_idx]),
            )
        self.idx = idx
        self.size = size
        self.max_priority = _payload_float(payload.get("max_priority"), "max_priority")
        self.frame_idx = _payload_int(payload.get("frame_idx"), "frame_idx")
        self.state_dim = None if state_dim == 0 else state_dim
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            self._rng.bit_generator.state = rng_state

    def _retrieve(self, idx: int, value: float) -> int:
        left = (2 * idx) + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    def _update_tree(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2


class TorchD3qnPerAgent:
    def __init__(
        self,
        *,
        config: UpstreamAlphaConfig | None = None,
        device_policy: Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"] = (
            "cpu_only_deterministic"
        ),
    ) -> None:
        self.config = default_upstream_alpha_config_v1() if config is None else config
        self.torch = _import_torch()
        _seed_torch_and_numpy(torch=self.torch, seed=self.config.seed)
        _configure_torch_threads(
            torch=self.torch,
            torch_num_threads=self.config.torch_num_threads,
            torch_num_interop_threads=self.config.torch_num_interop_threads,
        )
        self.device, self.device_payload = _select_device(
            torch=self.torch,
            device_policy=device_policy,
        )
        self.policy_net = build_torch_cnn_dueling_q_network_v1(
            torch=self.torch,
            config=self.config,
            device=self.device,
        )
        self.target_net = build_torch_cnn_dueling_q_network_v1(
            torch=self.torch,
            config=self.config,
            device=self.device,
        )
        copy_torch_cnn_dueling_state_v1(target=self.target_net, source=self.policy_net)
        self.target_net.eval()
        self.optimizer = self.torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay_buffer = SumTreePrioritizedReplayBuffer(
            capacity=self.config.replay_capacity,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_frames=self.config.per_beta_frames,
            epsilon=self.config.per_epsilon,
            seed=self.config.seed,
        )
        self.q_value_cache = QValueCache()
        self._rng = np.random.default_rng(self.config.seed)
        self.total_steps = 0
        self.learn_steps = 0
        self.target_sync_count = 0

    def epsilon(self) -> float:
        return float(
            self.config.eps_end
            + (self.config.eps_start - self.config.eps_end)
            * np.exp(-float(self.total_steps) / float(self.config.eps_decay_frames))
        )

    def select_action_with_details(
        self,
        state: np.ndarray,
        *,
        training: bool,
        valid_actions: Sequence[int] | None = None,
        use_cache: bool = False,
        cache_key: Hashable | None = None,
    ) -> ActionSelection:
        state_value = _state_1d(state, field="state", expected_dim=self.config.state_dim)
        valid = _normalize_valid_actions(valid_actions)
        epsilon = self.epsilon()
        if training and self._rng.random() < epsilon:
            action = int(self._rng.choice(np.asarray(valid, dtype=np.int64)))
            return ActionSelection(
                action_id=action,
                epsilon=epsilon,
                mode="epsilon_random",
                q_values=None,
                valid_actions=valid,
            )
        if use_cache and cache_key is not None:
            cache_hit_before = self.q_value_cache.hits
            q_values = self.q_value_cache.get_or_compute(
                cache_key,
                lambda: self.predict_q_values(state_value),
            )
            mode: SelectionMode = (
                "cache_hit" if self.q_value_cache.hits > cache_hit_before else "greedy"
            )
        else:
            q_values = self.predict_q_values(state_value)
            mode = "greedy"
        masked_q = q_values.copy()
        invalid = set(range(RL_ACTION_COUNT_V1)).difference(valid)
        for action_id in invalid:
            masked_q[action_id] = -np.inf
        action = int(np.argmax(masked_q))
        return ActionSelection(
            action_id=action,
            epsilon=epsilon,
            mode=mode,
            q_values=tuple(float(value) for value in q_values),
            valid_actions=valid,
        )

    def select_action(
        self,
        state: np.ndarray,
        *,
        training: bool,
        valid_actions: Sequence[int] | None = None,
    ) -> int:
        return self.select_action_with_details(
            state,
            training=training,
            valid_actions=valid_actions,
        ).action_id

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        state_value = _state_1d(state, field="state", expected_dim=self.config.state_dim)
        self.policy_net.eval()
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(
                state_value.reshape(1, -1),
                dtype=self.torch.float32,
                device=self.device,
            )
            q_values = torch_cnn_dueling_forward_v1(
                network=self.policy_net,
                states=tensor,
                config=self.config,
            )
        _synchronize_if_needed(torch=self.torch, device_type=str(self.device.type))
        return np.ascontiguousarray(q_values.detach().cpu().numpy().reshape(-1), dtype=np.float32)

    def predict_ensemble(
        self,
        state: np.ndarray,
        *,
        n_samples: int,
        use_cache: bool = True,
        cache_key: Hashable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _positive_int(n_samples, "n_samples")
        state_value = _state_1d(state, field="state", expected_dim=self.config.state_dim)
        if use_cache and cache_key is not None:
            cached = self.q_value_cache.get_or_compute(
                ("ensemble_mean", cache_key),
                lambda: self._ensemble_q_values(state_value, n_samples=n_samples)[0],
            )
            _, std = self._ensemble_q_values(state_value, n_samples=n_samples)
            return cached, std
        return self._ensemble_q_values(state_value, n_samples=n_samples)

    def _ensemble_q_values(
        self,
        state: np.ndarray,
        *,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.policy_net.train()
        tensor = self.torch.as_tensor(
            state.reshape(1, -1),
            dtype=self.torch.float32,
            device=self.device,
        )
        samples = []
        with self.torch.no_grad():
            for _ in range(n_samples):
                q_values = torch_cnn_dueling_forward_v1(
                    network=self.policy_net,
                    states=tensor,
                    config=self.config,
                )
                samples.append(q_values.detach().cpu().numpy().reshape(-1))
        self.policy_net.eval()
        arr = np.asarray(samples, dtype=np.float32)
        return (
            np.ascontiguousarray(np.mean(arr, axis=0), dtype=np.float32),
            np.ascontiguousarray(np.std(arr, axis=0), dtype=np.float32),
        )

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> LearningStepResult | None:
        if len(self.replay_buffer) < self.config.train_start:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        self.policy_net.train()
        self.target_net.eval()
        sample = self.replay_buffer.sample(self.config.batch_size)
        states = self.torch.as_tensor(sample.states, dtype=self.torch.float32, device=self.device)
        actions = self.torch.as_tensor(sample.actions, dtype=self.torch.long, device=self.device)
        rewards = self.torch.as_tensor(sample.rewards, dtype=self.torch.float32, device=self.device)
        next_states = self.torch.as_tensor(
            sample.next_states,
            dtype=self.torch.float32,
            device=self.device,
        )
        dones = self.torch.as_tensor(
            sample.dones.astype(np.float32),
            dtype=self.torch.float32,
            device=self.device,
        )
        weights = self.torch.as_tensor(sample.weights, dtype=self.torch.float32, device=self.device)

        current_q = torch_cnn_dueling_forward_v1(
            network=self.policy_net,
            states=states,
            config=self.config,
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self.torch.no_grad():
            next_actions = torch_cnn_dueling_forward_v1(
                network=self.policy_net,
                states=next_states,
                config=self.config,
            ).argmax(dim=1)
            next_target_q = torch_cnn_dueling_forward_v1(
                network=self.target_net,
                states=next_states,
                config=self.config,
            ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_target_q
        td_errors = target_q - current_q
        loss = self.torch.nn.functional.smooth_l1_loss(
            current_q,
            target_q,
            reduction="none",
        )
        weighted_loss = (weights * loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        gradient_norm = self.torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.config.max_gradient_norm,
        )
        self.optimizer.step()
        td_errors_np = np.asarray(td_errors.detach().cpu().numpy(), dtype=np.float64)
        self.replay_buffer.update_priorities(sample.tree_indices, td_errors_np)
        self.learn_steps += 1
        target_synced = False
        if self.learn_steps % self.config.target_update_freq == 0:
            copy_torch_cnn_dueling_state_v1(target=self.target_net, source=self.policy_net)
            self.target_sync_count += 1
            target_synced = True
        return LearningStepResult(
            loss=float(weighted_loss.detach().cpu()),
            mean_abs_td_error=float(np.mean(np.abs(td_errors_np), dtype=np.float64)),
            gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
            target_synced=target_synced,
        )

    def increment_step(self) -> None:
        self.total_steps += 1

    def release_device_cache(self) -> None:
        _release_device_cache_if_needed(torch=self.torch, device_type=str(self.device.type))


def build_torch_cnn_dueling_q_network_v1(
    *,
    torch: Any,
    config: UpstreamAlphaConfig,
    device: Any,
) -> Any:
    conv_layers = []
    in_channels = config.feature_count
    for out_channels, kernel, stride in zip(
        config.cnn_maps,
        config.cnn_kernels,
        config.cnn_strides,
        strict=True,
    ):
        conv_layers.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel, 1),
                stride=(stride, 1),
            )
        )
        conv_layers.append(torch.nn.ReLU(inplace=True))
        conv_layers.append(torch.nn.Dropout(p=config.dropout_p))
        in_channels = out_channels
    feature_extractor = torch.nn.Sequential(*conv_layers)
    with torch.no_grad():
        dummy = torch.zeros(1, *config.cnn_input_shape)
        flat_cnn_size = int(feature_extractor(dummy).reshape(1, -1).shape[1])
    value_layers = _dense_stream_layers(
        torch=torch,
        input_dim=flat_cnn_size + config.additional_feature_count,
        hidden_dims=config.dense_val,
        output_dim=1,
        dropout_p=config.dropout_p,
    )
    advantage_layers = _dense_stream_layers(
        torch=torch,
        input_dim=flat_cnn_size + config.additional_feature_count,
        hidden_dims=config.dense_adv,
        output_dim=RL_ACTION_COUNT_V1,
        dropout_p=config.dropout_p,
    )
    return torch.nn.ModuleDict(
        {
            "advantage": torch.nn.Sequential(*advantage_layers),
            "feature_extractor": feature_extractor,
            "value": torch.nn.Sequential(*value_layers),
        }
    ).to(device)


def torch_cnn_dueling_forward_v1(
    *,
    network: Any,
    states: Any,
    config: UpstreamAlphaConfig,
) -> Any:
    batch_size = states.size(0)
    history_part = states[:, : config.flat_history_size]
    extra_part = states[:, config.flat_history_size :]
    history_tensor = history_part.reshape(batch_size, *config.cnn_input_shape)
    features = network["feature_extractor"](history_tensor).reshape(batch_size, -1)
    combined = states.new_empty((batch_size, features.size(1) + extra_part.size(1)))
    combined[:, : features.size(1)] = features
    combined[:, features.size(1) :] = extra_part
    value = network["value"](combined)
    advantage = network["advantage"](combined)
    return value + advantage - advantage.mean(dim=1, keepdim=True)


def copy_torch_cnn_dueling_state_v1(*, target: Any, source: Any) -> None:
    target.load_state_dict(source.state_dict())


def run_upstream_environment_rollout_v1(
    *,
    environment: UpstreamTradingEnvironment,
    agent: TorchD3qnPerAgent,
    episodes: int,
) -> dict[str, object]:
    _positive_int(episodes, "episodes")
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.perf_counter()
    action_counts = {ACTION_NAMES_BY_ID_V1[action_id]: 0 for action_id in ACTION_NAMES_BY_ID_V1}
    selection_mode_counts = {"cache_hit": 0, "epsilon_random": 0, "greedy": 0}
    audit_reason_counts: dict[str, int] = {}
    learning_results: list[LearningStepResult] = []
    episode_pnls: list[float] = []
    transitions = 0
    for episode_idx in range(episodes):
        state, _ = environment.reset(forced_index=episode_idx % len(environment.sequences))
        done = False
        latest_info: Mapping[str, object] = {}
        while not done:
            selection = agent.select_action_with_details(
                state,
                training=True,
                valid_actions=environment.valid_actions(),
            )
            next_state, reward, done, _, info = environment.step(selection.action_id)
            agent.store_experience(
                state,
                selection.action_id,
                reward,
                next_state,
                done,
            )
            learning_result = agent.learn()
            if learning_result is not None:
                learning_results.append(learning_result)
            agent.increment_step()
            effective_action_id = _payload_int(info["effective_action_id"], "effective_action_id")
            action_counts[ACTION_NAMES_BY_ID_V1[effective_action_id]] += 1
            selection_mode_counts[selection.mode] += 1
            audit_reason = str(info["audit_reason"])
            audit_reason_counts[audit_reason] = audit_reason_counts.get(audit_reason, 0) + 1
            transitions += 1
            state = next_state
            latest_info = info
        episode_pnls.append(
            _payload_float(latest_info.get("episode_realized_pnl", 0.0), "episode_realized_pnl")
        )
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    wall_seconds = time.perf_counter() - start_wall
    return {
        "action_counts": action_counts,
        "audit_reason_counts": dict(sorted(audit_reason_counts.items())),
        "episode_count": episodes,
        "episode_pnls": [_round_float(value) for value in episode_pnls],
        "learn_update_count": len(learning_results),
        "mean_abs_td_error_last": (
            None
            if not learning_results
            else _round_float(learning_results[-1].mean_abs_td_error)
        ),
        "replay_size": len(agent.replay_buffer),
        "resource_usage": {
            "cpu_system_seconds_delta": _round_float(
                end_usage.ru_stime - start_usage.ru_stime
            ),
            "cpu_user_seconds_delta": _round_float(end_usage.ru_utime - start_usage.ru_utime),
            "rss_mb_after": _rss_mb(),
            "selected_device": str(agent.device_payload["selected_device"]),
            "wall_seconds": _round_float(wall_seconds),
        },
        "scripted_transition_sequence_used": False,
        "selection_mode_counts": selection_mode_counts,
        "target_sync_count": agent.target_sync_count,
        "total_agent_steps": agent.total_steps,
        "transition_count": transitions,
    }


def select_checkpoint_policy_v1(
    validation_curve: Sequence[Mapping[str, object]],
    *,
    policy: CheckpointSelectionPolicy | None = None,
) -> dict[str, object]:
    selected_policy = CheckpointSelectionPolicy() if policy is None else policy
    if not validation_curve:
        return {
            "best_checkpoint": "final.pth",
            "default_evaluation_checkpoint": "final",
            "final_checkpoint": "final.pth",
            "reason": "validation_curve_empty_final_fallback",
            "validation_metric_name": selected_policy.validation_metric_name,
        }
    best_row: Mapping[str, object] | None = None
    best_value: float | None = None
    for row in validation_curve:
        if selected_policy.validation_metric_name not in row:
            continue
        value = _payload_float(
            row[selected_policy.validation_metric_name],
            selected_policy.validation_metric_name,
        )
        if best_value is None:
            best_value = value
            best_row = row
            continue
        if selected_policy.higher_is_better and value > best_value:
            best_value = value
            best_row = row
        if not selected_policy.higher_is_better and value < best_value:
            best_value = value
            best_row = row
    if best_row is None or best_value is None:
        raise UpstreamMethodologyError(reason="validation_metric_missing")
    return {
        "best_checkpoint": "best.pth",
        "best_metric_value": _round_float(best_value),
        "best_step": _payload_int(best_row.get("completed_training_steps", 0), "best_step"),
        "default_evaluation_checkpoint": selected_policy.default_evaluation_checkpoint,
        "final_checkpoint": "final.pth",
        "final_is_diagnostic_unless_selected": (
            selected_policy.default_evaluation_checkpoint == "best"
        ),
        "validation_metric_name": selected_policy.validation_metric_name,
    }


def run_stage08b_core_smoke_v1(
    *,
    session_features: np.ndarray,
    output_root: Path,
    config: UpstreamAlphaConfig | None = None,
    episodes: int = 3,
    generated_at_utc: datetime | None = None,
) -> dict[str, Any]:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    features = _validate_session_features(session_features, config=selected_config)
    gate = training_source_gate_payload_v1(exchange="binance", market_type="futures")
    if gate["status"] != "trainable":
        raise UpstreamMethodologyError(reason="blocked_not_training_source_v1")
    stats = compute_train_only_normalization_stats_v1(features, config=selected_config)
    environment = UpstreamTradingEnvironment(
        sequences=features,
        normalization_stats=stats,
        config=selected_config,
    )
    agent = TorchD3qnPerAgent(config=selected_config)
    rollout = run_upstream_environment_rollout_v1(
        environment=environment,
        agent=agent,
        episodes=episodes,
    )
    checkpoint_policy = select_checkpoint_policy_v1(
        [
            {
                "Validation_mean_pnl": value,
                "completed_training_steps": (idx + 1) * selected_config.agent_session_len,
            }
            for idx, value in enumerate(cast(Sequence[float], rollout["episode_pnls"]))
        ]
    )
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "stage08b_core_smoke_report.json"
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "artifact_kind": STAGE08B_CORE_SMOKE_KIND_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "checkpoint_policy": checkpoint_policy,
        "config": selected_config.as_payload(),
        "config_hash": selected_config.config_hash(),
        "contains_raw_checkpoint_tensors": False,
        "contains_raw_provider_payloads": False,
        "contains_secrets": False,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated),
        "metrics": rollout,
        "normalization_stats_hash": stats.stats_hash(),
        "prompt_stage": "08B",
        "report_path": str(report_path),
        "schema_version": STAGE08B_CORE_SMOKE_SCHEMA_VERSION_V1,
        "source_gate": gate,
        "status": "accepted_smoke",
        "upstream_methodology_parity": True,
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    }
    payload = {**payload, "smoke_report_hash": hash_json_payload_v1(payload)}
    _atomic_write_json(report_path, payload)
    return {**payload, "artifact_hashes": {"smoke_report": _file_payload(report_path)}}


def build_tiny_stage08b_session_features_v1(
    *,
    session_count: int,
    config: UpstreamAlphaConfig | None = None,
) -> np.ndarray:
    selected_config = default_upstream_alpha_config_v1() if config is None else config
    _positive_int(session_count, "session_count")
    features = np.zeros(
        (session_count, selected_config.full_seq_len, len(FEATURE_NAMES_V1)),
        dtype=np.float32,
    )
    minute = np.arange(selected_config.full_seq_len, dtype=np.float32)
    for session_idx in range(session_count):
        base = 100.0 + float(session_idx)
        wave = np.sin((minute + session_idx) / 8.0).astype(np.float32) * 0.1
        trend = minute * np.float32(0.015 + (session_idx % 3) * 0.002)
        close = base + trend + wave
        values = {
            "close": close,
            "high": close + np.float32(0.08),
            "low": close - np.float32(0.08),
            "num_trades": np.full_like(close, 20.0 + float(session_idx)),
            "open": close - np.float32(0.02),
            "volume": np.full_like(close, 50.0 + float(session_idx)),
            "volume_weighted_average": close + np.float32(0.01),
        }
        for name, value in values.items():
            features[session_idx, :, FEATURE_NAMES_V1.index(name)] = value
    return np.ascontiguousarray(features, dtype=np.float32)


def _dense_stream_layers(
    *,
    torch: Any,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout_p: float,
) -> list[Any]:
    layers = []
    previous = input_dim
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(previous, hidden_dim))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Dropout(p=dropout_p))
        previous = hidden_dim
    layers.append(torch.nn.Linear(previous, output_dim))
    return layers


def _as_sequence_list(
    sequences: np.ndarray | Sequence[np.ndarray],
    *,
    config: UpstreamAlphaConfig,
) -> list[np.ndarray]:
    if isinstance(sequences, np.ndarray):
        values = _validate_session_features(sequences, config=config)
        return [
            np.ascontiguousarray(values[idx], dtype=np.float32)
            for idx in range(values.shape[0])
        ]
    out = [_validate_session(item, config=config) for item in sequences]
    return [np.ascontiguousarray(item, dtype=np.float32) for item in out]


def _validate_session_features(
    session_features: np.ndarray,
    *,
    config: UpstreamAlphaConfig,
) -> np.ndarray:
    features = np.asarray(session_features, dtype=np.float32)
    if features.ndim != 3:
        raise UpstreamMethodologyError(reason="session_features_must_be_3d")
    expected_shape = (config.full_seq_len, len(FEATURE_NAMES_V1))
    if tuple(features.shape[1:]) != expected_shape:
        raise UpstreamMethodologyError(
            reason="session_features_shape_mismatch",
            field=str(features.shape),
        )
    if features.shape[0] <= 0:
        raise UpstreamMethodologyError(reason="empty_session_features")
    if not np.all(np.isfinite(features)):
        raise UpstreamMethodologyError(reason="non_finite_session_features")
    close_idx = FEATURE_NAMES_V1.index("close")
    if np.any(features[:, :, close_idx] <= 0.0):
        raise UpstreamMethodologyError(reason="non_positive_close")
    return np.ascontiguousarray(features, dtype=np.float32)


def _validate_session(session: np.ndarray, *, config: UpstreamAlphaConfig) -> np.ndarray:
    value = np.asarray(session, dtype=np.float32)
    expected_shape = (config.full_seq_len, len(FEATURE_NAMES_V1))
    if value.shape != expected_shape:
        raise UpstreamMethodologyError(
            reason="session_shape_mismatch",
            field=str(value.shape),
        )
    if not np.all(np.isfinite(value)):
        raise UpstreamMethodologyError(reason="non_finite_session")
    if np.any(value[:, FEATURE_NAMES_V1.index("close")] <= 0.0):
        raise UpstreamMethodologyError(reason="non_positive_close")
    return np.ascontiguousarray(value, dtype=np.float32)


def _trim_action_history(
    action_history: Sequence[int | None],
    *,
    action_history_len: int,
) -> tuple[int | None, ...]:
    values = list(action_history)
    if len(values) < action_history_len:
        values = ([None] * (action_history_len - len(values))) + values
    values = values[-action_history_len:]
    for action_id in values:
        if action_id is not None:
            normalize_rl_action_id_v1(action_id)
    return tuple(values)


def _normalize_valid_actions(valid_actions: Sequence[int] | None) -> tuple[int, ...]:
    if valid_actions is None:
        return tuple(range(RL_ACTION_COUNT_V1))
    normalized = tuple(sorted({normalize_rl_action_id_v1(int(item)) for item in valid_actions}))
    if not normalized:
        raise UpstreamMethodologyError(reason="valid_actions_required")
    return normalized


def _q_values_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if out.shape != (RL_ACTION_COUNT_V1,):
        raise UpstreamMethodologyError(reason="q_values_shape_mismatch", field=str(out.shape))
    if not np.all(np.isfinite(out)):
        raise UpstreamMethodologyError(reason="non_finite_q_values")
    return np.ascontiguousarray(out, dtype=np.float32)


def _state_1d(
    value: np.ndarray,
    *,
    field: str,
    expected_dim: int | None,
) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 1:
        raise UpstreamMethodologyError(reason="state_must_be_1d", field=field)
    if expected_dim is not None and out.shape[0] != expected_dim:
        raise UpstreamMethodologyError(reason="state_dim_mismatch", field=field)
    if not np.all(np.isfinite(out)):
        raise UpstreamMethodologyError(reason="non_finite_state", field=field)
    return np.ascontiguousarray(out, dtype=np.float32)


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise UpstreamMethodologyError(reason="torch_import_failed", field=str(exc)) from exc


def _seed_torch_and_numpy(*, torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def _configure_torch_threads(
    *,
    torch: Any,
    torch_num_threads: int,
    torch_num_interop_threads: int,
) -> None:
    torch.set_num_threads(torch_num_threads)
    current_interop_threads = int(torch.get_num_interop_threads())
    if current_interop_threads != torch_num_interop_threads:
        torch.set_num_interop_threads(torch_num_interop_threads)


def _select_device(
    *,
    torch: Any,
    device_policy: Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"],
) -> tuple[Any, dict[str, object]]:
    if device_policy not in {"cpu_only_deterministic", "mps_preferred_cpu_fallback"}:
        raise UpstreamMethodologyError(reason="unsupported_device_policy")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    selected_device = "cpu"
    if device_policy == "mps_preferred_cpu_fallback" and mps_available:
        selected_device = "mps"
    return torch.device(selected_device), {
        "device_policy": device_policy,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "selected_device": selected_device,
        "torch_version": str(torch.__version__),
    }


def _synchronize_if_needed(*, torch: Any, device_type: str) -> None:
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _release_device_cache_if_needed(*, torch: Any, device_type: str) -> None:
    if device_type != "mps" or not hasattr(torch, "mps"):
        return
    torch.mps.synchronize()
    empty_cache = getattr(torch.mps, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_raw_feature_json_payload_v1(dict(payload)) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_payload(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    import hashlib

    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _format_utc(value: datetime) -> str:
    selected = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return selected.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _payload_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpstreamMethodologyError(reason="payload_int_invalid", field=field)
    return value


def _payload_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamMethodologyError(reason="payload_float_invalid", field=field)
    return _finite_float(float(value), field)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    if value > 10_000_000:
        value = value / (1024.0 * 1024.0)
    else:
        value = value / 1024.0
    return _round_float(value)


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise UpstreamMethodologyError(reason="non_positive_int", field=field)
    return int(value)


def _finite_float(value: float, field: str) -> float:
    if isinstance(value, bool):
        raise UpstreamMethodologyError(reason="invalid_float", field=field)
    out = float(value)
    if not math.isfinite(out):
        raise UpstreamMethodologyError(reason="non_finite_float", field=field)
    return out


def _positive_float(value: float, field: str) -> float:
    out = _finite_float(value, field)
    if out <= 0.0:
        raise UpstreamMethodologyError(reason="non_positive_float", field=field)
    return out


def _non_negative_float(value: float, field: str) -> float:
    out = _finite_float(value, field)
    if out < 0.0:
        raise UpstreamMethodologyError(reason="negative_float", field=field)
    return out


def _bounded_float(value: float, field: str, *, low: float, high: float) -> float:
    out = _finite_float(value, field)
    if out < low or out > high:
        raise UpstreamMethodologyError(reason="float_out_of_bounds", field=field)
    return out


def _round_float(value: float) -> float:
    return round(float(value), 10)
