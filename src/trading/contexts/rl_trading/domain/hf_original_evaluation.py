from __future__ import annotations

import json
import resource
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .action_state_reward_contract import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
)
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_NAMES_V1
from .hf_original_training import STAGE08C_CANDIDATE_LEVEL_V1
from .raw_feature_dataset import hash_json_payload_v1, render_raw_feature_json_payload_v1
from .upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    FilteredBacktestPolicy,
    NormalizationStats,
    SelectionStrategy,
    TorchD3qnPerAgent,
    UpstreamAlphaConfig,
    UpstreamTradingEnvironment,
    default_upstream_alpha_config_v1,
    mask_upstream_training_action_v1,
    session_close_price_v1,
)

STAGE08D_EVALUATION_SCHEMA_VERSION_V1 = 1
STAGE08D_EVALUATION_KIND_V1 = "rl_trading_stage08d_original_hf_evaluation"
STAGE08D_SCORECARD_KIND_V1 = "rl_trading_stage08d_hf_scorecard"
STAGE08D_EVALUATION_CONFIG_ID_V1 = "roehub_stage08d_original_hf_evaluation_config_v1"
STAGE08D_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08d_original_hf_backtest_evaluation_v1"

CheckpointName = Literal["best", "final"]
DevicePolicy = Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"]
ScorecardKind = Literal["candidate", "baseline", "diagnostic"]


class HfOriginalEvaluationError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HfOriginalSplitData:
    split_name: str
    sequences: np.ndarray
    symbols: tuple[str, ...]
    signal_times_utc: tuple[str | None, ...]
    source_payload: Mapping[str, object]
    volatility_scores: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        features = _validate_sequences(self.sequences, field=self.split_name)
        if len(self.symbols) != features.shape[0]:
            raise HfOriginalEvaluationError(reason="symbol_count_mismatch", field=self.split_name)
        if len(self.signal_times_utc) != features.shape[0]:
            raise HfOriginalEvaluationError(
                reason="signal_time_count_mismatch",
                field=self.split_name,
            )
        volatility_scores = (
            self.volatility_scores
            if self.volatility_scores
            else tuple(None for _ in range(features.shape[0]))
        )
        if len(volatility_scores) != features.shape[0]:
            raise HfOriginalEvaluationError(
                reason="volatility_score_count_mismatch",
                field=self.split_name,
            )
        object.__setattr__(self, "sequences", features)
        object.__setattr__(self, "symbols", tuple(str(item).upper() for item in self.symbols))
        object.__setattr__(self, "volatility_scores", tuple(volatility_scores))


@dataclass(slots=True)
class _BacktestRiskManagementState:
    trailing_max_price: float | None = None
    trailing_min_price: float | None = None


@dataclass(frozen=True, slots=True)
class HfOriginalEvaluationConfig:
    alpha: UpstreamAlphaConfig = field(default_factory=default_upstream_alpha_config_v1)
    checkpoint_name: CheckpointName = "best"
    selection_strategy: SelectionStrategy = "advantage_based_filter"
    device_policy: DevicePolicy = "cpu_only_deterministic"
    test_max_sessions: int | None = None
    backtest_max_sessions: int | None = None
    simple_threshold_return: float = 0.001

    def __post_init__(self) -> None:
        if self.checkpoint_name not in {"best", "final"}:
            raise HfOriginalEvaluationError(reason="unsupported_checkpoint_name")
        if self.selection_strategy not in {"advantage_based_filter", "ensemble_q_filter"}:
            raise HfOriginalEvaluationError(reason="unsupported_selection_strategy")
        if self.device_policy not in {"cpu_only_deterministic", "mps_preferred_cpu_fallback"}:
            raise HfOriginalEvaluationError(reason="unsupported_device_policy")
        if self.test_max_sessions is not None:
            _positive_int(self.test_max_sessions, "test_max_sessions")
        if self.backtest_max_sessions is not None:
            _positive_int(self.backtest_max_sessions, "backtest_max_sessions")
        _positive_float(self.simple_threshold_return, "simple_threshold_return")

    def as_payload(self) -> dict[str, object]:
        return {
            "alpha_config": self.alpha.as_payload(),
            "alpha_config_hash": self.alpha.config_hash(),
            "checkpoint_name": self.checkpoint_name,
            "config_id": STAGE08D_EVALUATION_CONFIG_ID_V1,
            "device_policy": self.device_policy,
            "selection_strategy": self.selection_strategy,
            "simple_threshold_return": self.simple_threshold_return,
            "stage": "08D",
            "test_max_sessions": self.test_max_sessions,
            "backtest_max_sessions": self.backtest_max_sessions,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


def default_hf_original_evaluation_config_v1() -> HfOriginalEvaluationConfig:
    return HfOriginalEvaluationConfig()


def alpha_with_evaluation_overrides_v1(
    base: UpstreamAlphaConfig,
    overrides: UpstreamAlphaConfig,
) -> UpstreamAlphaConfig:
    return _alpha_with_evaluation_overrides_v1(base, overrides)


def run_stage08d_hf_original_evaluation_v1(
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    test_split: HfOriginalSplitData,
    backtest_split: HfOriginalSplitData,
    output_root: Path,
    run_id: str,
    config: HfOriginalEvaluationConfig | None = None,
    generated_at_utc: datetime | None = None,
    code_version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_candidate_manifest(candidate_manifest)
    training_config = _read_json_payload(_artifact_path(candidate_manifest, "training_config"))
    alpha = _alpha_config_from_training_config_payload(training_config)
    selected_config = (
        HfOriginalEvaluationConfig(alpha=alpha)
        if config is None
        else HfOriginalEvaluationConfig(
            alpha=_alpha_with_evaluation_overrides_v1(alpha, config.alpha),
            checkpoint_name=config.checkpoint_name,
            selection_strategy=config.selection_strategy,
            device_policy=config.device_policy,
            test_max_sessions=config.test_max_sessions,
            backtest_max_sessions=config.backtest_max_sessions,
            simple_threshold_return=config.simple_threshold_return,
        )
    )
    checkpoint_policy = _checkpoint_policy(candidate_manifest)
    if (
        selected_config.checkpoint_name == "best"
        and checkpoint_policy.get("default_evaluation_checkpoint") != "best"
    ):
        raise HfOriginalEvaluationError(reason="best_checkpoint_not_default")

    normalization_stats = _load_normalization_stats(
        _artifact_path(candidate_manifest, "normalization_stats"),
        expected_hash=str(candidate_manifest.get("normalization_stats_hash", "")),
    )
    agent, checkpoint_payload = _load_checkpoint_agent(
        candidate_manifest=candidate_manifest,
        checkpoint_name=selected_config.checkpoint_name,
        config=selected_config,
    )

    test_limited = _limit_split(test_split, selected_config.test_max_sessions)
    backtest_limited = _limit_split(backtest_split, selected_config.backtest_max_sessions)
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_test = evaluate_stage08d_test_episodes_v1(
        split=test_limited,
        normalization_stats=normalization_stats,
        agent=agent,
        config=selected_config,
    )
    filtered_backtest, balance_curve = evaluate_stage08d_grouped_backtest_v1(
        split=backtest_limited,
        normalization_stats=normalization_stats,
        agent=agent,
        config=selected_config,
    )
    baselines = [
        evaluate_stage08d_baseline_backtest_v1(
            split=backtest_limited,
            config=selected_config,
            policy_name="hold",
            fixed_action_id=0,
        ),
        evaluate_stage08d_baseline_backtest_v1(
            split=backtest_limited,
            config=selected_config,
            policy_name="no_trade",
            fixed_action_id=0,
        ),
        evaluate_stage08d_baseline_backtest_v1(
            split=backtest_limited,
            config=selected_config,
            policy_name="simple_recent_return_threshold",
            fixed_action_id=None,
        ),
    ]
    scorecards = [raw_test, filtered_backtest, *baselines]
    scorecards_path = run_dir / "scorecards.json"
    balance_curve_path = run_dir / "filtered_backtest_balance_curve.json"
    _atomic_write_json(scorecards_path, {"scorecards": scorecards})
    _atomic_write_json(balance_curve_path, {"balance_curve": balance_curve})

    artifact_hashes = {
        "balance_curve": _file_payload(balance_curve_path),
        "scorecards": _file_payload(scorecards_path),
    }
    manifest = build_stage08d_evaluation_artifact_v1(
        generated_at_utc=generated,
        run_id=run_id,
        run_dir=run_dir,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_manifest_sha256,
        test_split=test_limited,
        backtest_split=backtest_limited,
        config=selected_config,
        normalization_stats_hash=normalization_stats.stats_hash(),
        checkpoint_payload=checkpoint_payload,
        scorecards=scorecards,
        code_version={} if code_version is None else dict(code_version),
        artifact_hashes=artifact_hashes,
    )
    manifest_path = run_dir / "stage08d_evaluation_manifest.json"
    manifest = {**manifest, "evaluation_manifest_path": str(manifest_path)}
    manifest = {**manifest, "evaluation_hash": hash_json_payload_v1(_without_hash(manifest))}
    _atomic_write_json(manifest_path, manifest)
    return {**manifest, "evaluation_manifest_sha256": _file_payload(manifest_path)["sha256"]}


def evaluate_stage08d_test_episodes_v1(
    *,
    split: HfOriginalSplitData,
    normalization_stats: NormalizationStats,
    agent: TorchD3qnPerAgent,
    config: HfOriginalEvaluationConfig,
) -> dict[str, Any]:
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.perf_counter()
    environment = UpstreamTradingEnvironment(
        sequences=split.sequences,
        normalization_stats=normalization_stats,
        config=config.alpha,
    )
    action_counts = _empty_action_counts()
    audit_reason_counts: dict[str, int] = {}
    session_pnls: list[float] = []
    session_rewards: list[float] = []
    closed_trades: list[int] = []
    profitable_trades: list[int] = []
    decisions_count = 0

    for session_idx in range(len(environment.sequences)):
        state, _ = environment.reset(forced_index=session_idx)
        done = False
        episode_reward = 0.0
        latest_info: Mapping[str, object] = {}
        while not done:
            selection = agent.select_action_with_details(
                state,
                training=False,
                valid_actions=environment.valid_actions(),
            )
            next_state, reward, done, _, info = environment.step(selection.action_id)
            effective_action = _payload_int(info["effective_action_id"], "effective_action_id")
            action_counts[ACTION_NAMES_BY_ID_V1[effective_action]] += 1
            audit_reason = str(info["audit_reason"])
            audit_reason_counts[audit_reason] = audit_reason_counts.get(audit_reason, 0) + 1
            episode_reward += reward
            decisions_count += 1
            state = next_state
            latest_info = info
        session_rewards.append(episode_reward)
        session_pnls.append(_payload_float(latest_info.get("episode_realized_pnl", 0.0), "pnl"))
        closed_trades.append(
            _payload_int(latest_info.get("episode_closed_trades", 0), "closed_trades")
        )
        profitable_trades.append(
            _payload_int(latest_info.get("episode_profitable_trades", 0), "profitable_trades")
        )

    pnl = np.asarray(session_pnls, dtype=np.float64)
    closed = np.asarray(closed_trades, dtype=np.int64)
    profitable = np.asarray(profitable_trades, dtype=np.int64)
    rewards = np.asarray(session_rewards, dtype=np.float64)
    wall_seconds = max(time.perf_counter() - start_wall, 0.0)
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    scorecard = _scorecard_payload(
        split=split,
        policy_name="hf_original_candidate_raw_argmax_test_diagnostic",
        policy_kind="diagnostic",
        evaluation_surface="HF test episode evaluation",
        session_pnls=pnl,
        session_closed_trades=closed,
        session_profitable_trades=profitable,
        action_counts=action_counts,
        audit_reason_counts=audit_reason_counts,
        decisions_count=decisions_count,
        reward_sum=float(np.sum(rewards, dtype=np.float64)),
        starting_equity_quote=config.alpha.initial_balance * float(split.sequences.shape[0]),
        config=config,
        extra={
            "acceptance_backtest": False,
            "candidate_checkpoint": config.checkpoint_name,
            "diagnostic_kind": "raw_argmax_environment_rollout",
            "filter_policy": None,
            "raw_argmax_only": True,
            "resource_usage": _resource_delta_payload(
                start_usage=start_usage,
                end_usage=end_usage,
                wall_seconds=wall_seconds,
                decisions_count=decisions_count,
            ),
        },
    )
    return {**scorecard, "scorecard_hash": hash_json_payload_v1(scorecard)}


def evaluate_stage08d_grouped_backtest_v1(
    *,
    split: HfOriginalSplitData,
    normalization_stats: NormalizationStats,
    agent: TorchD3qnPerAgent,
    config: HfOriginalEvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.perf_counter()
    selected_indices, grouping_payload = _grouped_backtest_indices(
        split.signal_times_utc,
        max_parallel_sessions=config.alpha.max_parallel_sessions,
    )
    selected_sequences = split.sequences[np.asarray(selected_indices, dtype=np.int64)]
    selected_split = HfOriginalSplitData(
        split_name=split.split_name,
        sequences=selected_sequences,
        symbols=tuple(split.symbols[idx] for idx in selected_indices),
        signal_times_utc=tuple(split.signal_times_utc[idx] for idx in selected_indices),
        source_payload=split.source_payload,
        volatility_scores=tuple(split.volatility_scores[idx] for idx in selected_indices),
    )
    backtest_alpha = _position_fraction_alpha(config.alpha)
    environment = UpstreamTradingEnvironment(
        sequences=selected_split.sequences,
        normalization_stats=normalization_stats,
        config=backtest_alpha,
    )
    filter_policy = FilteredBacktestPolicy.from_config(
        config.alpha,
        selection_strategy=config.selection_strategy,
    )
    action_counts = _empty_action_counts()
    requested_action_counts = _empty_action_counts()
    raw_argmax_action_counts = _empty_action_counts()
    audit_reason_counts: dict[str, int] = {}
    risk_management_reason_counts: dict[str, int] = {}
    session_pnls: list[float] = []
    closed_trades: list[int] = []
    profitable_trades: list[int] = []
    balance_curve: list[dict[str, object]] = []
    equity = config.alpha.initial_balance * float(selected_split.sequences.shape[0])
    peak_equity = equity
    max_drawdown_pct = 0.0
    reward_sum = 0.0
    decisions_count = 0

    for session_idx in range(len(environment.sequences)):
        state, _ = environment.reset(forced_index=session_idx)
        done = False
        latest_info: Mapping[str, object] = {}
        session_net_pnl = 0.0
        risk_state = _BacktestRiskManagementState()
        while not done:
            q_values = _q_values_for_state(
                agent=agent,
                state=state,
                cache_key=_cache_key(
                    symbol=selected_split.symbols[session_idx],
                    signal_time=selected_split.signal_times_utc[session_idx],
                    step_idx=environment.step_idx,
                    position_side=environment.state.position_side,
                    action_history=tuple(environment.action_history),
                ),
            )
            raw_argmax_action = int(np.argmax(q_values))
            raw_argmax_action_counts[ACTION_NAMES_BY_ID_V1[raw_argmax_action]] += 1
            masked_q_values = _mask_q_values(q_values, valid_actions=environment.valid_actions())
            if config.selection_strategy == "ensemble_q_filter":
                q_mean, q_std = agent.predict_ensemble(
                    state,
                    n_samples=config.alpha.ensemble_n_samples,
                    cache_key=_cache_key(
                        symbol=selected_split.symbols[session_idx],
                        signal_time=selected_split.signal_times_utc[session_idx],
                        step_idx=environment.step_idx,
                        position_side=environment.state.position_side,
                        action_history=tuple(environment.action_history),
                    ),
                )
                decision = filter_policy.select_from_q_values(
                    _mask_q_values(q_mean, valid_actions=environment.valid_actions()),
                    q_std=q_std,
                )
            else:
                decision = filter_policy.select_from_q_values(masked_q_values)
            requested_action_counts[ACTION_NAMES_BY_ID_V1[decision.requested_action_id]] += 1
            state_before_action = environment.state
            forced_action, risk_reason = _risk_management_action_override_v1(
                state=state_before_action,
                session=environment.sequences[session_idx],
                step_idx=environment.step_idx,
                config=backtest_alpha,
                risk_state=risk_state,
            )
            action_for_environment = (
                decision.effective_action_id if forced_action is None else forced_action
            )
            if risk_reason is not None:
                risk_management_reason_counts[risk_reason] = (
                    risk_management_reason_counts.get(risk_reason, 0) + 1
                )
            next_state, reward, done, _, info = environment.step(action_for_environment)
            effective_action = _payload_int(info["effective_action_id"], "effective_action_id")
            pnl_change = _payload_float(info["pnl_change"], "pnl_change")
            _update_risk_management_state_after_step_v1(
                risk_state=risk_state,
                state_before=state_before_action,
                state_after=environment.state,
                closed_position=bool(info.get("closed_position", False)),
                config=backtest_alpha,
            )
            action_counts[ACTION_NAMES_BY_ID_V1[effective_action]] += 1
            audit_reason = str(info["audit_reason"])
            audit_reason_counts[audit_reason] = audit_reason_counts.get(audit_reason, 0) + 1
            equity += pnl_change
            session_net_pnl += pnl_change
            reward_sum += reward
            peak_equity = max(peak_equity, equity)
            drawdown = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            max_drawdown_pct = max(max_drawdown_pct, drawdown)
            balance_curve.append(
                {
                    "equity_quote": _round_float(equity),
                    "session_index": session_idx,
                    "signal_time_utc": selected_split.signal_times_utc[session_idx],
                    "step_idx": environment.step_idx - 1,
                    "symbol": selected_split.symbols[session_idx],
                }
            )
            decisions_count += 1
            state = next_state
            latest_info = info
        session_pnls.append(session_net_pnl)
        closed_trades.append(
            _payload_int(latest_info.get("episode_closed_trades", 0), "closed_trades")
        )
        profitable_trades.append(
            _payload_int(latest_info.get("episode_profitable_trades", 0), "profitable_trades")
        )

    wall_seconds = max(time.perf_counter() - start_wall, 0.0)
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    pnl = np.asarray(session_pnls, dtype=np.float64)
    closed = np.asarray(closed_trades, dtype=np.int64)
    profitable = np.asarray(profitable_trades, dtype=np.int64)
    scorecard = _scorecard_payload(
        split=selected_split,
        policy_name="hf_original_candidate_filtered_backtest",
        policy_kind="candidate",
        evaluation_surface="HF grouped filtered backtest",
        session_pnls=pnl,
        session_closed_trades=closed,
        session_profitable_trades=profitable,
        action_counts=action_counts,
        audit_reason_counts=audit_reason_counts,
        decisions_count=decisions_count,
        reward_sum=reward_sum,
        starting_equity_quote=(
            config.alpha.initial_balance * float(selected_split.sequences.shape[0])
        ),
        config=config,
        extra={
            "acceptance_backtest": True,
            "candidate_checkpoint": config.checkpoint_name,
            "filter_policy": filter_policy.stats_payload(),
            "grouping": grouping_payload,
            "max_drawdown_pct": _round_float(max_drawdown_pct),
            "position_fraction": config.alpha.position_fraction,
            "position_fraction_application": "initial_balance_scaled_for_session_pnl",
            "q_value_cache": agent.q_value_cache.stats_payload(),
            "raw_argmax_action_counts": raw_argmax_action_counts,
            "requested_action_counts": requested_action_counts,
            "resource_usage": _resource_delta_payload(
                start_usage=start_usage,
                end_usage=end_usage,
                wall_seconds=wall_seconds,
                decisions_count=decisions_count,
            ),
            "risk_management": _risk_management_payload_v1(
                config=backtest_alpha,
                reason_counts=risk_management_reason_counts,
            ),
            "selection_strategy": config.selection_strategy,
        },
    )
    return {**scorecard, "scorecard_hash": hash_json_payload_v1(scorecard)}, balance_curve


def evaluate_stage08d_baseline_backtest_v1(
    *,
    split: HfOriginalSplitData,
    config: HfOriginalEvaluationConfig,
    policy_name: str,
    fixed_action_id: int | None,
) -> dict[str, Any]:
    selected_indices, grouping_payload = _grouped_backtest_indices(
        split.signal_times_utc,
        max_parallel_sessions=config.alpha.max_parallel_sessions,
    )
    selected_sequences = split.sequences[np.asarray(selected_indices, dtype=np.int64)]
    selected_split = HfOriginalSplitData(
        split_name=split.split_name,
        sequences=selected_sequences,
        symbols=tuple(split.symbols[idx] for idx in selected_indices),
        signal_times_utc=tuple(split.signal_times_utc[idx] for idx in selected_indices),
        source_payload=split.source_payload,
        volatility_scores=tuple(split.volatility_scores[idx] for idx in selected_indices),
    )
    backtest_alpha = _position_fraction_alpha(config.alpha)
    action_counts = _empty_action_counts()
    audit_reason_counts: dict[str, int] = {}
    risk_management_reason_counts: dict[str, int] = {}
    session_pnls: list[float] = []
    closed_trades: list[int] = []
    profitable_trades: list[int] = []
    decisions_count = 0
    reward_sum = 0.0
    for session_idx, session in enumerate(selected_split.sequences):
        state = RlTrainingState(balance=backtest_alpha.initial_balance)
        action_history: list[int | None] = [None] * backtest_alpha.action_history_len
        session_pnl = 0.0
        latest_closed = 0
        latest_profitable = 0
        risk_state = _BacktestRiskManagementState()
        for step_idx in range(backtest_alpha.agent_session_len):
            price = session_close_price_v1(session, step_idx=step_idx, config=backtest_alpha)
            action_id = (
                _baseline_threshold_action(
                    session=session,
                    step_idx=step_idx,
                    state=state,
                    config=backtest_alpha,
                    threshold_return=config.simple_threshold_return,
                )
                if fixed_action_id is None
                else fixed_action_id
            )
            action_id = mask_upstream_training_action_v1(
                action_id=action_id,
                position_side=state.position_side,
                is_last_step=step_idx == backtest_alpha.agent_session_len - 1,
            )
            forced_action, risk_reason = _risk_management_action_override_v1(
                state=state,
                session=session,
                step_idx=step_idx,
                config=backtest_alpha,
                risk_state=risk_state,
            )
            if forced_action is not None:
                action_id = forced_action
            if risk_reason is not None:
                risk_management_reason_counts[risk_reason] = (
                    risk_management_reason_counts.get(risk_reason, 0) + 1
                )
            state_before_action = state
            result = apply_training_reward_step_v1(
                state=state,
                action_id=action_id,
                price=price,
                initial_balance=backtest_alpha.initial_balance,
                slippage=backtest_alpha.slippage,
                transaction_fee=backtest_alpha.transaction_fee,
                inaction_penalty_ratio=backtest_alpha.inaction_penalty_ratio,
                is_last_step=step_idx == backtest_alpha.agent_session_len - 1,
            )
            state = result.state
            _update_risk_management_state_after_step_v1(
                risk_state=risk_state,
                state_before=state_before_action,
                state_after=state,
                closed_position=result.closed_position,
                config=backtest_alpha,
            )
            action_history = [*action_history[1:], result.effective_action_id]
            action_counts[ACTION_NAMES_BY_ID_V1[result.effective_action_id]] += 1
            audit_reason_counts[result.audit_reason] = (
                audit_reason_counts.get(result.audit_reason, 0) + 1
            )
            session_pnl += result.pnl_change
            reward_sum += result.reward
            latest_closed = result.state.closed_trades
            latest_profitable = result.state.profitable_trades
            decisions_count += 1
        session_pnls.append(session_pnl)
        closed_trades.append(latest_closed)
        profitable_trades.append(latest_profitable)
    scorecard = _scorecard_payload(
        split=selected_split,
        policy_name=policy_name,
        policy_kind="baseline",
        evaluation_surface="HF grouped baseline backtest",
        session_pnls=np.asarray(session_pnls, dtype=np.float64),
        session_closed_trades=np.asarray(closed_trades, dtype=np.int64),
        session_profitable_trades=np.asarray(profitable_trades, dtype=np.int64),
        action_counts=action_counts,
        audit_reason_counts=audit_reason_counts,
        decisions_count=decisions_count,
        reward_sum=reward_sum,
        starting_equity_quote=(
            config.alpha.initial_balance * float(selected_split.sequences.shape[0])
        ),
        config=config,
        extra={
            "acceptance_backtest": False,
            "baseline_policy": policy_name,
            "filter_policy": None,
            "grouping": grouping_payload,
            "position_fraction": config.alpha.position_fraction,
            "risk_management": _risk_management_payload_v1(
                config=backtest_alpha,
                reason_counts=risk_management_reason_counts,
            ),
        },
    )
    return {**scorecard, "scorecard_hash": hash_json_payload_v1(scorecard)}


def build_stage08d_evaluation_artifact_v1(
    *,
    generated_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    test_split: HfOriginalSplitData,
    backtest_split: HfOriginalSplitData,
    config: HfOriginalEvaluationConfig,
    normalization_stats_hash: str,
    checkpoint_payload: Mapping[str, Any],
    scorecards: Sequence[Mapping[str, Any]],
    code_version: Mapping[str, Any],
    artifact_hashes: Mapping[str, object],
) -> dict[str, Any]:
    ordered_scorecards = list(scorecards)
    filtered = _scorecard_by_name(ordered_scorecards, "hf_original_candidate_filtered_backtest")
    baselines = [item for item in ordered_scorecards if item.get("policy_kind") == "baseline"]
    overfit = _overfit_and_sanity_payload(filtered, baselines, candidate_manifest)
    parity = _methodology_parity_payload(
        filtered_backtest=filtered,
        overfit=overfit,
        config=config,
    )
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08D_EVALUATION_KIND_V1,
        "candidate_dependency": {
            "candidate_level": candidate_manifest.get("candidate_level"),
            "checkpoint_name": config.checkpoint_name,
            "checkpoint_policy": dict(
                cast(Mapping[str, object], candidate_manifest["checkpoint_policy"])
            ),
            "checkpoint_stage": checkpoint_payload.get("stage"),
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": candidate_manifest_sha256,
            "stage": "08C",
        },
        "code_version": dict(code_version),
        "config": config.as_payload(),
        "config_hash": config.config_hash(),
        "data_quality_report": {
            "blockers": [],
            "grain": "hf_original_session",
            "required_hashes_matched": True,
            "sources": ["Stage 04 HF test/backtest NPZ", "Stage 08C hf_original_candidate"],
            "status": "pass",
            "warnings": [],
        },
        "dataset_dependency": {
            "backtest_split": dict(backtest_split.source_payload),
            "test_split": dict(test_split.source_payload),
            "training_source": "binance:futures",
            "stage": "04",
        },
        "delivery_state": (
            "local-only implementation plus target_host_non_production_evaluation_pre_main "
            "when run on Mac Studio"
        ),
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated_at_utc),
        "methodology": {
            "analysis_depth": "standard_analysis",
            "baseline_policy_names": [str(item["policy_name"]) for item in baselines],
            "business_claim": "HF methodology-parity only; no promotion or live-trading claim",
            "decision_unit": "hf_original_session",
            "method": "test_episode_rollout_plus_grouped_filtered_backtest",
            "raw_argmax_acceptance": False,
        },
        "methodology_parity_verdict": parity,
        "normalization_stats_hash": normalization_stats_hash,
        "overfit_indicators": overfit,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": {
            "browser_auth_used": False,
            "contains_raw_checkpoint_tensors": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "mainnet_submit": False,
            "model_registry_write": False,
            "paper_testnet_live_enabled": False,
            "promotion_or_activation": False,
        },
        "schema_version": STAGE08D_EVALUATION_SCHEMA_VERSION_V1,
        "scorecards": ordered_scorecards,
        "stage": "08D",
        "status": parity["status"],
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    }
    return {**payload, "evaluation_hash": hash_json_payload_v1(payload)}


def _scorecard_payload(
    *,
    split: HfOriginalSplitData,
    policy_name: str,
    policy_kind: ScorecardKind,
    evaluation_surface: str,
    session_pnls: np.ndarray,
    session_closed_trades: np.ndarray,
    session_profitable_trades: np.ndarray,
    action_counts: Mapping[str, int],
    audit_reason_counts: Mapping[str, int],
    decisions_count: int,
    reward_sum: float,
    starting_equity_quote: float,
    config: HfOriginalEvaluationConfig,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    net_pnl = float(np.sum(session_pnls, dtype=np.float64))
    trade_count = int(np.sum(session_closed_trades, dtype=np.int64))
    profitable_count = int(np.sum(session_profitable_trades, dtype=np.int64))
    payload = {
        "action_counts": dict(sorted(action_counts.items())),
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "artifact_kind": STAGE08D_SCORECARD_KIND_V1,
        "audit_reason_counts": dict(sorted(audit_reason_counts.items())),
        "closed_trades": trade_count,
        "decisions_count": decisions_count,
        "evaluation_surface": evaluation_surface,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "metrics_by_period": _period_stability_payload(
            signal_times=split.signal_times_utc,
            session_pnls=session_pnls,
            session_closed_trades=session_closed_trades,
            session_profitable_trades=session_profitable_trades,
            initial_balance=config.alpha.initial_balance,
        ),
        "metrics_by_volatility_bucket": _volatility_bucket_payload(
            volatility_scores=split.volatility_scores,
            session_pnls=session_pnls,
            session_closed_trades=session_closed_trades,
            session_profitable_trades=session_profitable_trades,
            initial_balance=config.alpha.initial_balance,
        ),
        "net_pnl_after_costs_quote": _round_float(net_pnl),
        "out_of_sample_period": _period_payload(split.signal_times_utc),
        "policy_kind": policy_kind,
        "policy_name": policy_name,
        "profitable_trades": profitable_count,
        "return_pct_after_costs": _round_float((net_pnl / starting_equity_quote) * 100.0)
        if starting_equity_quote
        else 0.0,
        "reward_sum": _round_float(reward_sum),
        "schema_version": STAGE08D_EVALUATION_SCHEMA_VERSION_V1,
        "session_count": int(split.sequences.shape[0]),
        "stability_by_ticker": _ticker_stability_payload(
            symbols=split.symbols,
            session_pnls=session_pnls,
            session_closed_trades=session_closed_trades,
            session_profitable_trades=session_profitable_trades,
            initial_balance=config.alpha.initial_balance,
        ),
        "stability_summary": _stability_summary(session_pnls),
        "starting_equity_quote": _round_float(starting_equity_quote),
        "win_rate": _round_float(profitable_count / trade_count if trade_count else 0.0),
    }
    return {**payload, **dict(extra)}


def _load_checkpoint_agent(
    *,
    candidate_manifest: Mapping[str, Any],
    checkpoint_name: CheckpointName,
    config: HfOriginalEvaluationConfig,
) -> tuple[TorchD3qnPerAgent, dict[str, Any]]:
    torch_key = "best_checkpoint" if checkpoint_name == "best" else "final_checkpoint"
    checkpoint_path = _artifact_path(candidate_manifest, torch_key)
    agent = TorchD3qnPerAgent(config=config.alpha, device_policy=config.device_policy)
    payload = agent.torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    if not isinstance(payload, dict):
        raise HfOriginalEvaluationError(
            reason="checkpoint_payload_invalid",
            field=str(checkpoint_path),
        )
    if payload.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise HfOriginalEvaluationError(reason="checkpoint_architecture_mismatch")
    if payload.get("stage") != "08C":
        raise HfOriginalEvaluationError(reason="checkpoint_stage_mismatch")
    policy_state = payload.get("policy_state")
    target_state = payload.get("target_state")
    if not isinstance(policy_state, Mapping) or not isinstance(target_state, Mapping):
        raise HfOriginalEvaluationError(reason="checkpoint_model_state_missing")
    agent.policy_net.load_state_dict(policy_state)
    agent.target_net.load_state_dict(target_state)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent, cast(dict[str, Any], payload)


def _load_normalization_stats(path: Path, *, expected_hash: str) -> NormalizationStats:
    payload = _read_json_payload(path)
    means = _payload_float_mapping(payload.get("means"), "means")
    stds = _payload_float_mapping(payload.get("stds"), "stds")
    stats = NormalizationStats(
        means=means,
        stds=stds,
        source_split=str(payload.get("source_split", "")),
        sequence_count=_payload_int(payload.get("sequence_count"), "sequence_count"),
    )
    if stats.source_split != "train":
        raise HfOriginalEvaluationError(reason="normalization_stats_not_train_only")
    if expected_hash and stats.stats_hash() != expected_hash:
        raise HfOriginalEvaluationError(reason="normalization_stats_hash_mismatch")
    return stats


def _alpha_config_from_training_config_payload(payload: Mapping[str, Any]) -> UpstreamAlphaConfig:
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise HfOriginalEvaluationError(reason="training_config_missing_config")
    alpha_payload = config_payload.get("alpha_config")
    if not isinstance(alpha_payload, Mapping):
        raise HfOriginalEvaluationError(reason="training_config_missing_alpha_config")
    defaults = default_upstream_alpha_config_v1()
    tuple_fields = {"cnn_maps", "cnn_kernels", "cnn_strides", "dense_adv", "dense_val"}

    def value_for(name: str) -> Any:
        value = alpha_payload.get(name, getattr(defaults, name))
        if name in tuple_fields:
            return tuple(int(item) for item in cast(Sequence[Any], value))
        return value

    return UpstreamAlphaConfig(
        seed=int(value_for("seed")),
        full_seq_len=int(value_for("full_seq_len")),
        pre_signal_len=int(value_for("pre_signal_len")),
        agent_history_len=int(value_for("agent_history_len")),
        agent_session_len=int(value_for("agent_session_len")),
        action_history_len=int(value_for("action_history_len")),
        initial_balance=float(value_for("initial_balance")),
        transaction_fee=float(value_for("transaction_fee")),
        slippage=float(value_for("slippage")),
        inaction_penalty_ratio=float(value_for("inaction_penalty_ratio")),
        gamma=float(value_for("gamma")),
        learning_rate=float(value_for("learning_rate")),
        batch_size=int(value_for("batch_size")),
        target_update_freq=int(value_for("target_update_freq")),
        train_start=int(value_for("train_start")),
        max_gradient_norm=float(value_for("max_gradient_norm")),
        replay_capacity=int(value_for("replay_capacity")),
        per_alpha=float(value_for("per_alpha")),
        per_beta_start=float(value_for("per_beta_start")),
        per_beta_frames=int(value_for("per_beta_frames")),
        per_epsilon=float(value_for("per_epsilon")),
        eps_start=float(value_for("eps_start")),
        eps_end=float(value_for("eps_end")),
        eps_decay_frames=int(value_for("eps_decay_frames")),
        cnn_maps=cast(tuple[int, ...], value_for("cnn_maps")),
        cnn_kernels=cast(tuple[int, ...], value_for("cnn_kernels")),
        cnn_strides=cast(tuple[int, ...], value_for("cnn_strides")),
        dense_val=cast(tuple[int, ...], value_for("dense_val")),
        dense_adv=cast(tuple[int, ...], value_for("dense_adv")),
        dropout_p=float(value_for("dropout_p")),
        long_action_threshold=float(value_for("long_action_threshold")),
        short_action_threshold=float(value_for("short_action_threshold")),
        close_action_threshold=float(value_for("close_action_threshold")),
        use_risk_management=bool(value_for("use_risk_management")),
        stop_loss=float(value_for("stop_loss")),
        take_profit=float(value_for("take_profit")),
        trailing_stop=float(value_for("trailing_stop")),
        ensemble_n_samples=int(value_for("ensemble_n_samples")),
        ensemble_max_sigma=float(value_for("ensemble_max_sigma")),
        max_parallel_sessions=int(value_for("max_parallel_sessions")),
        position_fraction=float(value_for("position_fraction")),
        torch_num_threads=int(value_for("torch_num_threads")),
        torch_num_interop_threads=int(value_for("torch_num_interop_threads")),
        dtype=str(value_for("dtype")),
    )


def _alpha_with_evaluation_overrides_v1(
    base: UpstreamAlphaConfig,
    overrides: UpstreamAlphaConfig,
) -> UpstreamAlphaConfig:
    return replace(
        base,
        long_action_threshold=overrides.long_action_threshold,
        short_action_threshold=overrides.short_action_threshold,
        close_action_threshold=overrides.close_action_threshold,
        use_risk_management=overrides.use_risk_management,
        stop_loss=overrides.stop_loss,
        take_profit=overrides.take_profit,
        trailing_stop=overrides.trailing_stop,
        ensemble_n_samples=overrides.ensemble_n_samples,
        ensemble_max_sigma=overrides.ensemble_max_sigma,
        max_parallel_sessions=overrides.max_parallel_sessions,
        position_fraction=overrides.position_fraction,
        torch_num_threads=overrides.torch_num_threads,
        torch_num_interop_threads=overrides.torch_num_interop_threads,
    )


def _validate_candidate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("stage") != "08C":
        raise HfOriginalEvaluationError(reason="candidate_manifest_stage_mismatch")
    if value.get("candidate_level") != STAGE08C_CANDIDATE_LEVEL_V1:
        raise HfOriginalEvaluationError(reason="candidate_manifest_level_mismatch")
    if value.get("status") != "completed":
        raise HfOriginalEvaluationError(reason="candidate_manifest_not_completed")
    if value.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise HfOriginalEvaluationError(reason="candidate_architecture_mismatch")


def _checkpoint_policy(candidate_manifest: Mapping[str, Any]) -> Mapping[str, object]:
    value = candidate_manifest.get("checkpoint_policy")
    if not isinstance(value, Mapping):
        raise HfOriginalEvaluationError(reason="candidate_checkpoint_policy_missing")
    return cast(Mapping[str, object], value)


def _artifact_path(candidate_manifest: Mapping[str, Any], key: str) -> Path:
    artifacts = candidate_manifest.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        raise HfOriginalEvaluationError(reason="candidate_artifacts_missing")
    item = artifacts.get(key)
    if not isinstance(item, Mapping):
        raise HfOriginalEvaluationError(reason="candidate_artifact_missing", field=key)
    value = item.get("path")
    if not isinstance(value, str) or not value:
        raise HfOriginalEvaluationError(reason="candidate_artifact_path_missing", field=key)
    path = Path(value)
    if not path.exists():
        raise HfOriginalEvaluationError(reason="candidate_artifact_file_missing", field=str(path))
    return path


def _limit_split(split: HfOriginalSplitData, limit: int | None) -> HfOriginalSplitData:
    if limit is None or limit >= split.sequences.shape[0]:
        return split
    if limit <= 0:
        raise HfOriginalEvaluationError(reason="invalid_split_limit")
    return HfOriginalSplitData(
        split_name=split.split_name,
        sequences=split.sequences[:limit],
        symbols=split.symbols[:limit],
        signal_times_utc=split.signal_times_utc[:limit],
        source_payload={**dict(split.source_payload), "selected_session_count": limit},
        volatility_scores=split.volatility_scores[:limit],
    )


def _grouped_backtest_indices(
    signal_times: Sequence[str | None],
    *,
    max_parallel_sessions: int,
) -> tuple[list[int], dict[str, object]]:
    _positive_int(max_parallel_sessions, "max_parallel_sessions")
    groups: dict[str, list[int]] = {}
    for idx, signal_time in enumerate(signal_times):
        key = signal_time if signal_time is not None else f"__missing_signal_time_{idx}"
        groups.setdefault(str(key), []).append(idx)
    selected: list[int] = []
    skipped = 0
    max_group_size = 0
    for key in sorted(groups):
        indices = groups[key]
        max_group_size = max(max_group_size, len(indices))
        selected.extend(indices[:max_parallel_sessions])
        skipped += max(0, len(indices) - max_parallel_sessions)
    return selected, {
        "group_count": len(groups),
        "grouping_key": "signal_time_utc",
        "max_group_size": max_group_size,
        "max_parallel_sessions": max_parallel_sessions,
        "selected_session_count": len(selected),
        "skipped_sessions_due_parallel_cap": skipped,
        "source_session_count": len(signal_times),
    }


def _q_values_for_state(
    *,
    agent: TorchD3qnPerAgent,
    state: np.ndarray,
    cache_key: Hashable,
) -> np.ndarray:
    return agent.q_value_cache.get_or_compute(cache_key, lambda: agent.predict_q_values(state))


def _cache_key(
    *,
    symbol: str,
    signal_time: str | None,
    step_idx: int,
    position_side: str | None,
    action_history: tuple[int | None, ...],
) -> tuple[object, ...]:
    return (symbol, signal_time, step_idx, position_side, action_history)


def _mask_q_values(q_values: np.ndarray, *, valid_actions: Sequence[int]) -> np.ndarray:
    q = np.ascontiguousarray(q_values, dtype=np.float32)
    masked = q.copy()
    valid = set(valid_actions)
    for action_id in ACTION_NAMES_BY_ID_V1:
        if action_id not in valid:
            masked[action_id] = -1.0e30
    return masked


def _position_fraction_alpha(config: UpstreamAlphaConfig) -> UpstreamAlphaConfig:
    return UpstreamAlphaConfig(
        seed=config.seed,
        full_seq_len=config.full_seq_len,
        pre_signal_len=config.pre_signal_len,
        agent_history_len=config.agent_history_len,
        agent_session_len=config.agent_session_len,
        action_history_len=config.action_history_len,
        initial_balance=config.initial_balance * config.position_fraction,
        transaction_fee=config.transaction_fee,
        slippage=config.slippage,
        inaction_penalty_ratio=config.inaction_penalty_ratio,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        train_start=config.train_start,
        max_gradient_norm=config.max_gradient_norm,
        replay_capacity=config.replay_capacity,
        per_alpha=config.per_alpha,
        per_beta_start=config.per_beta_start,
        per_beta_frames=config.per_beta_frames,
        per_epsilon=config.per_epsilon,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        eps_decay_frames=config.eps_decay_frames,
        cnn_maps=config.cnn_maps,
        cnn_kernels=config.cnn_kernels,
        cnn_strides=config.cnn_strides,
        dense_val=config.dense_val,
        dense_adv=config.dense_adv,
        dropout_p=config.dropout_p,
        long_action_threshold=config.long_action_threshold,
        short_action_threshold=config.short_action_threshold,
        close_action_threshold=config.close_action_threshold,
        use_risk_management=config.use_risk_management,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        trailing_stop=config.trailing_stop,
        ensemble_n_samples=config.ensemble_n_samples,
        ensemble_max_sigma=config.ensemble_max_sigma,
        max_parallel_sessions=config.max_parallel_sessions,
        position_fraction=config.position_fraction,
        torch_num_threads=config.torch_num_threads,
        torch_num_interop_threads=config.torch_num_interop_threads,
        dtype=config.dtype,
    )


def _risk_management_action_override_v1(
    *,
    state: RlTrainingState,
    session: np.ndarray,
    step_idx: int,
    config: UpstreamAlphaConfig,
    risk_state: _BacktestRiskManagementState,
) -> tuple[int | None, str | None]:
    if not config.use_risk_management or state.position_side is None:
        return None, None
    if state.entry_price is None:
        raise HfOriginalEvaluationError(reason="risk_management_entry_price_missing")
    price = session_close_price_v1(session, step_idx=step_idx, config=config)
    entry_price = _positive_float(state.entry_price, "entry_price")
    stop_loss = float(config.stop_loss)
    take_profit = float(config.take_profit)
    trailing_stop = float(config.trailing_stop)
    if state.position_side == "long":
        risk_state.trailing_max_price = (
            price
            if risk_state.trailing_max_price is None
            else max(risk_state.trailing_max_price, price)
        )
        sl_trigger = price <= entry_price * (1.0 - stop_loss)
        tp_trigger = price >= entry_price * (1.0 + take_profit)
        trailing_trigger = (
            risk_state.trailing_max_price is not None
            and price <= risk_state.trailing_max_price * (1.0 - trailing_stop)
        )
    elif state.position_side == "short":
        risk_state.trailing_min_price = (
            price
            if risk_state.trailing_min_price is None
            else min(risk_state.trailing_min_price, price)
        )
        sl_trigger = price >= entry_price * (1.0 + stop_loss)
        tp_trigger = price <= entry_price * (1.0 - take_profit)
        trailing_trigger = (
            risk_state.trailing_min_price is not None
            and price >= risk_state.trailing_min_price * (1.0 + trailing_stop)
        )
    else:
        return None, None
    if sl_trigger:
        return 3, "risk_management_stop_loss_forced_close"
    if tp_trigger:
        return 3, "risk_management_take_profit_forced_close"
    if trailing_trigger:
        return 3, "risk_management_trailing_stop_forced_close"
    return None, None


def _update_risk_management_state_after_step_v1(
    *,
    risk_state: _BacktestRiskManagementState,
    state_before: RlTrainingState,
    state_after: RlTrainingState,
    closed_position: bool,
    config: UpstreamAlphaConfig,
) -> None:
    if not config.use_risk_management:
        return
    if closed_position or state_after.position_side is None:
        risk_state.trailing_max_price = None
        risk_state.trailing_min_price = None
        return
    if state_before.position_side is not None:
        return
    if state_after.entry_price is None:
        return
    if state_after.position_side == "long":
        risk_state.trailing_max_price = state_after.entry_price
        risk_state.trailing_min_price = None
    elif state_after.position_side == "short":
        risk_state.trailing_min_price = state_after.entry_price
        risk_state.trailing_max_price = None


def _risk_management_payload_v1(
    *,
    config: UpstreamAlphaConfig,
    reason_counts: Mapping[str, int],
) -> dict[str, object]:
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "stop_loss": config.stop_loss,
        "take_profit": config.take_profit,
        "trailing_stop": config.trailing_stop,
        "use_risk_management": config.use_risk_management,
    }


def _baseline_threshold_action(
    *,
    session: np.ndarray,
    step_idx: int,
    state: RlTrainingState,
    config: UpstreamAlphaConfig,
    threshold_return: float,
) -> int:
    close_idx = FEATURE_NAMES_V1.index("close")
    price_idx = min(config.pre_signal_len + step_idx, session.shape[0] - 1)
    lookback_idx = max(0, price_idx - 10)
    previous = _positive_float(float(session[lookback_idx, close_idx]), "close")
    current = _positive_float(float(session[price_idx, close_idx]), "close")
    move = (current / previous) - 1.0
    if state.position_side == "long":
        return 3 if move <= -threshold_return else 0
    if state.position_side == "short":
        return 3 if move >= threshold_return else 0
    if move >= threshold_return:
        return 1
    if move <= -threshold_return:
        return 2
    return 0


def _scorecard_by_name(
    scorecards: Sequence[Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    for item in scorecards:
        if item.get("policy_name") == name:
            return item
    raise HfOriginalEvaluationError(reason="scorecard_missing", field=name)


def _overfit_and_sanity_payload(
    filtered: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, object]:
    candidate_pnl = float(filtered.get("net_pnl_after_costs_quote", 0.0))
    best_baseline = max(
        (float(item.get("net_pnl_after_costs_quote", 0.0)) for item in baselines),
        default=0.0,
    )
    warnings: list[str] = []
    if candidate_pnl <= 0.0:
        warnings.append("candidate_non_positive_hf_backtest_pnl")
    if candidate_pnl <= best_baseline:
        warnings.append("candidate_does_not_clear_best_sanity_baseline")
    stability = filtered.get("stability_summary")
    if isinstance(stability, Mapping):
        positive_ratio = float(stability.get("session_positive_ratio", 0.0))
        if positive_ratio < 0.45:
            warnings.append("low_positive_session_ratio")
    return {
        "best_baseline_net_pnl_after_costs_quote": _round_float(best_baseline),
        "candidate_beats_best_sanity_baseline": candidate_pnl > best_baseline,
        "candidate_manifest_hash": str(candidate_manifest.get("candidate_manifest_hash")),
        "candidate_net_pnl_after_costs_quote": _round_float(candidate_pnl),
        "candidate_positive_after_costs": candidate_pnl > 0.0,
        "overfit_warning_codes": sorted(set(warnings)),
    }


def _methodology_parity_payload(
    *,
    filtered_backtest: Mapping[str, Any],
    overfit: Mapping[str, object],
    config: HfOriginalEvaluationConfig,
) -> dict[str, object]:
    warnings = list(cast(Sequence[str], overfit.get("overfit_warning_codes", ())))
    status = "accepted" if not warnings else "blocked"
    blockers = [] if status == "accepted" else warnings
    return {
        "blockers": blockers,
        "checkpoint_name": config.checkpoint_name,
        "filtered_backtest_net_pnl_after_costs_quote": filtered_backtest.get(
            "net_pnl_after_costs_quote"
        ),
        "lifecycle_end_to_end": True,
        "raw_argmax_only_acceptance": False,
        "scorecard_internally_coherent": status == "accepted",
        "selection_strategy": config.selection_strategy,
        "status": status,
    }


def _ticker_stability_payload(
    *,
    symbols: Sequence[str],
    session_pnls: np.ndarray,
    session_closed_trades: np.ndarray,
    session_profitable_trades: np.ndarray,
    initial_balance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in sorted(set(symbols)):
        indices = np.asarray([idx for idx, item in enumerate(symbols) if item == symbol])
        pnl = float(np.sum(session_pnls[indices], dtype=np.float64))
        trades = int(np.sum(session_closed_trades[indices], dtype=np.int64))
        profitable = int(np.sum(session_profitable_trades[indices], dtype=np.int64))
        rows.append(
            {
                "closed_trades": trades,
                "net_pnl_after_costs_quote": _round_float(pnl),
                "profitable_trades": profitable,
                "return_pct_after_costs": _round_float(
                    (pnl / (len(indices) * initial_balance)) * 100.0
                ),
                "session_count": int(len(indices)),
                "symbol": symbol,
                "win_rate": _round_float(profitable / trades if trades else 0.0),
            }
        )
    return rows


def _period_stability_payload(
    *,
    signal_times: Sequence[str | None],
    session_pnls: np.ndarray,
    session_closed_trades: np.ndarray,
    session_profitable_trades: np.ndarray,
    initial_balance: float,
) -> list[dict[str, object]]:
    periods: dict[str, list[int]] = {}
    for idx, value in enumerate(signal_times):
        key = "unknown" if value is None else str(value)[:7]
        periods.setdefault(key, []).append(idx)
    rows: list[dict[str, object]] = []
    for period in sorted(periods):
        indices = np.asarray(periods[period], dtype=np.int64)
        pnl = float(np.sum(session_pnls[indices], dtype=np.float64))
        trades = int(np.sum(session_closed_trades[indices], dtype=np.int64))
        profitable = int(np.sum(session_profitable_trades[indices], dtype=np.int64))
        rows.append(
            {
                "closed_trades": trades,
                "net_pnl_after_costs_quote": _round_float(pnl),
                "period": period,
                "profitable_trades": profitable,
                "return_pct_after_costs": _round_float(
                    (pnl / (len(indices) * initial_balance)) * 100.0
                ),
                "session_count": int(len(indices)),
                "win_rate": _round_float(profitable / trades if trades else 0.0),
            }
        )
    return rows


def _volatility_bucket_payload(
    *,
    volatility_scores: Sequence[float | None],
    session_pnls: np.ndarray,
    session_closed_trades: np.ndarray,
    session_profitable_trades: np.ndarray,
    initial_balance: float,
) -> list[dict[str, object]]:
    if not volatility_scores:
        volatility_scores = tuple(None for _ in range(int(session_pnls.size)))
    finite = np.asarray(
        [float(item) for item in volatility_scores if item is not None and np.isfinite(item)],
        dtype=np.float64,
    )
    if finite.size:
        q33 = float(np.quantile(finite, 1.0 / 3.0))
        q66 = float(np.quantile(finite, 2.0 / 3.0))
    else:
        q33 = q66 = 0.0
    buckets: dict[str, list[int]] = {}
    for idx, value in enumerate(volatility_scores):
        if value is None or not np.isfinite(float(value)):
            bucket = "unknown"
        elif float(value) <= q33:
            bucket = "low"
        elif float(value) <= q66:
            bucket = "medium"
        else:
            bucket = "high"
        buckets.setdefault(bucket, []).append(idx)
    rows: list[dict[str, object]] = []
    for bucket in sorted(buckets):
        indices = np.asarray(buckets[bucket], dtype=np.int64)
        pnl = float(np.sum(session_pnls[indices], dtype=np.float64))
        trades = int(np.sum(session_closed_trades[indices], dtype=np.int64))
        profitable = int(np.sum(session_profitable_trades[indices], dtype=np.int64))
        rows.append(
            {
                "bucket": bucket,
                "closed_trades": trades,
                "net_pnl_after_costs_quote": _round_float(pnl),
                "profitable_trades": profitable,
                "return_pct_after_costs": _round_float(
                    (pnl / (len(indices) * initial_balance)) * 100.0
                ),
                "session_count": int(len(indices)),
                "volatility_bucket_method": (
                    "tertiles_within_scorecard_split"
                    if finite.size
                    else "metadata_missing"
                ),
                "win_rate": _round_float(profitable / trades if trades else 0.0),
            }
        )
    return rows


def _stability_summary(session_pnls: np.ndarray) -> dict[str, object]:
    if session_pnls.size == 0:
        return {
            "negative_session_count": 0,
            "positive_session_count": 0,
            "session_count": 0,
            "session_positive_ratio": 0.0,
        }
    positive = int(np.count_nonzero(session_pnls > 0.0))
    negative = int(np.count_nonzero(session_pnls < 0.0))
    return {
        "median_session_net_pnl_quote": _round_float(float(np.median(session_pnls))),
        "negative_session_count": negative,
        "positive_session_count": positive,
        "session_count": int(session_pnls.size),
        "session_positive_ratio": _round_float(positive / int(session_pnls.size)),
        "worst_session_net_pnl_quote": _round_float(float(np.min(session_pnls))),
    }


def _period_payload(signal_times: Sequence[str | None]) -> dict[str, object]:
    values = sorted(str(item) for item in signal_times if item)
    return {
        "end_utc": values[-1] if values else None,
        "range_semantics": "HF _keys_map_ signal datetime UTC",
        "session_time_count": len(values),
        "start_utc": values[0] if values else None,
    }


def _resource_delta_payload(
    *,
    start_usage: Any,
    end_usage: Any,
    wall_seconds: float,
    decisions_count: int,
) -> dict[str, object]:
    return {
        "cpu_system_seconds_delta": _round_float(end_usage.ru_stime - start_usage.ru_stime),
        "cpu_user_seconds_delta": _round_float(end_usage.ru_utime - start_usage.ru_utime),
        "decisions_per_second": _round_float(decisions_count / max(wall_seconds, 1e-9)),
        "rss_mb_after": _rss_mb(),
        "wall_seconds": _round_float(wall_seconds),
    }


def _empty_action_counts() -> dict[str, int]:
    return {ACTION_NAMES_BY_ID_V1[action_id]: 0 for action_id in sorted(ACTION_NAMES_BY_ID_V1)}


def _payload_float_mapping(value: object, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise HfOriginalEvaluationError(reason="payload_mapping_invalid", field=field)
    return {str(key): _payload_float(item, f"{field}.{key}") for key, item in value.items()}


def _validate_sequences(value: np.ndarray, *, field: str) -> np.ndarray:
    features = np.asarray(value, dtype=np.float32)
    if features.ndim != 3:
        raise HfOriginalEvaluationError(reason="split_sequences_must_be_3d", field=field)
    expected = (150, len(FEATURE_NAMES_V1))
    if tuple(features.shape[1:]) != expected:
        raise HfOriginalEvaluationError(reason="split_sequence_shape_mismatch", field=field)
    if features.shape[0] <= 0:
        raise HfOriginalEvaluationError(reason="split_sequences_empty", field=field)
    if not np.all(np.isfinite(features)):
        raise HfOriginalEvaluationError(reason="split_sequences_non_finite", field=field)
    return np.ascontiguousarray(features, dtype=np.float32)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        render_raw_feature_json_payload_v1(_json_safe(dict(payload))) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _file_payload(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    import hashlib

    return {"bytes": len(data), "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _read_json_payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "evaluation_hash"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _payload_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HfOriginalEvaluationError(reason="payload_int_invalid", field=field)
    return value


def _payload_float(value: object, field: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HfOriginalEvaluationError(reason="payload_float_invalid", field=field) from exc
    if not np.isfinite(parsed):
        raise HfOriginalEvaluationError(reason="payload_float_non_finite", field=field)
    return parsed


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HfOriginalEvaluationError(reason="invalid_positive_int", field=field)
    return value


def _positive_float(value: float, field: str) -> float:
    parsed = _payload_float(value, field)
    if parsed <= 0.0:
        raise HfOriginalEvaluationError(reason="invalid_positive_float", field=field)
    return parsed


def _round_float(value: float, *, digits: int = 8) -> float:
    if not np.isfinite(value):
        return float(value)
    return round(float(value), digits)


def _rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 10_000_000:
        rss = rss / (1024.0 * 1024.0)
    else:
        rss = rss / 1024.0
    return _round_float(rss)
