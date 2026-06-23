from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

import numpy as np

from . import training_runner as _training_runner
from .action_state_reward_contract import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RL_ACTION_COUNT_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
)
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_NAMES_V1
from .raw_feature_dataset import hash_json_payload_v1
from .sessionized_dataset import (
    SESSIONIZED_AGENT_HISTORY_LEN_V1,
    SESSIONIZED_AGENT_SESSION_LEN_V1,
    SESSIONIZED_FULL_SEQ_LEN_V1,
    SESSIONIZED_PRE_SIGNAL_LEN_V1,
)
from .training_runner import (
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    CandidateTrainingConfig,
    D3qnArchitectureConfig,
    PrioritizedReplayConfig,
    TrainingSmokeConfig,
)

STAGE08_EVALUATION_SCHEMA_VERSION_V1 = 1
STAGE08_EVALUATION_KIND_V1 = "rl_trading_stage08_backtest_evaluation"
STAGE08_SCORECARD_KIND_V1 = "rl_trading_stage08_scorecard"
STAGE08_EVALUATION_CONFIG_ID_V1 = "roehub_stage08_backtest_evaluation_config_v1"
STAGE08_RESEARCH_FUNDING_MODEL_V1 = "research_zero_funding_no_point_in_time_arrays"
STAGE08_RUNTIME_ARTIFACT_ROOT_V1 = STAGE07A_RUNTIME_ARTIFACT_ROOT_V1

PolicyKind = Literal["candidate", "baseline"]


class Stage08EvaluationError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage08EvaluationConfig:
    initial_balance: float = 100.0
    slippage: float = 0.0
    transaction_fee: float = 0.001
    inaction_penalty_ratio: float = 0.0001
    funding_model: str = STAGE08_RESEARCH_FUNDING_MODEL_V1
    funding_pnl_per_step: float = 0.0
    random_seed: int = 240824
    simple_threshold_return: float = 0.001
    agent_history_len: int = SESSIONIZED_AGENT_HISTORY_LEN_V1
    agent_session_len: int = SESSIONIZED_AGENT_SESSION_LEN_V1

    def __post_init__(self) -> None:
        _positive_float(self.initial_balance, "initial_balance")
        _non_negative_float(self.slippage, "slippage")
        _non_negative_float(self.transaction_fee, "transaction_fee")
        _non_negative_float(self.inaction_penalty_ratio, "inaction_penalty_ratio")
        _finite_float(self.funding_pnl_per_step, "funding_pnl_per_step")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise Stage08EvaluationError(reason="invalid_random_seed", field="random_seed")
        if self.random_seed < 0:
            raise Stage08EvaluationError(reason="invalid_random_seed", field="random_seed")
        _positive_float(self.simple_threshold_return, "simple_threshold_return")
        _positive_int(self.agent_history_len, "agent_history_len")
        _positive_int(self.agent_session_len, "agent_session_len")

    def as_payload(self) -> dict[str, object]:
        return {
            "agent_history_len": self.agent_history_len,
            "agent_session_len": self.agent_session_len,
            "config_id": STAGE08_EVALUATION_CONFIG_ID_V1,
            "funding_model": self.funding_model,
            "funding_pnl_per_step": self.funding_pnl_per_step,
            "inaction_penalty_ratio": self.inaction_penalty_ratio,
            "initial_balance": self.initial_balance,
            "random_seed": self.random_seed,
            "simple_threshold_return": self.simple_threshold_return,
            "slippage": self.slippage,
            "transaction_fee": self.transaction_fee,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())

    def as_observation_config(self) -> TrainingSmokeConfig:
        return TrainingSmokeConfig(
            max_sessions=1,
            agent_history_len=self.agent_history_len,
            agent_session_len=self.agent_session_len,
            initial_balance=self.initial_balance,
            slippage=self.slippage,
            transaction_fee=self.transaction_fee,
            inaction_penalty_ratio=self.inaction_penalty_ratio,
            update_steps=1,
            batch_size=1,
        )


@dataclass(frozen=True, slots=True)
class Stage08StepContext:
    session_index: int
    step_idx: int
    session: np.ndarray
    symbol: str
    signal_time_utc: str | None
    price: float
    state: RlTrainingState
    action_history: tuple[int | None, ...]


class Stage08ActionPolicy(Protocol):
    @property
    def policy_name(self) -> str:
        ...

    @property
    def policy_kind(self) -> PolicyKind:
        ...

    def select_actions(self, contexts: Sequence[Stage08StepContext]) -> tuple[int, ...]:
        ...

    def diagnostics(self) -> dict[str, object]:
        ...


@dataclass(slots=True)
class Stage08FixedActionPolicy:
    policy_name: str
    action_id: int
    policy_kind: PolicyKind = "baseline"

    def __post_init__(self) -> None:
        _normalize_action(self.action_id)

    def select_actions(self, contexts: Sequence[Stage08StepContext]) -> tuple[int, ...]:
        return tuple(self.action_id for _ in contexts)

    def diagnostics(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_name": ACTION_NAMES_BY_ID_V1[self.action_id],
            "policy": "fixed_action",
        }


@dataclass(slots=True)
class Stage08RandomActionPolicy:
    seed: int
    policy_name: str = "random"
    policy_kind: PolicyKind = "baseline"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise Stage08EvaluationError(reason="invalid_random_seed", field="seed")
        self._rng = np.random.default_rng(self.seed)

    def select_actions(self, contexts: Sequence[Stage08StepContext]) -> tuple[int, ...]:
        values = self._rng.integers(0, RL_ACTION_COUNT_V1, size=len(contexts), endpoint=False)
        return tuple(int(value) for value in values)

    def diagnostics(self) -> dict[str, object]:
        return {
            "action_count": RL_ACTION_COUNT_V1,
            "policy": "deterministic_random",
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class Stage08SimpleThresholdPolicy:
    threshold_return: float
    policy_name: str = "simple_threshold"
    policy_kind: PolicyKind = "baseline"

    def __post_init__(self) -> None:
        _positive_float(self.threshold_return, "threshold_return")

    def select_actions(self, contexts: Sequence[Stage08StepContext]) -> tuple[int, ...]:
        actions: list[int] = []
        close_idx = FEATURE_NAMES_V1.index("close")
        for context in contexts:
            price_idx = min(
                SESSIONIZED_PRE_SIGNAL_LEN_V1 + context.step_idx,
                context.session.shape[0] - 1,
            )
            lookback_idx = max(0, price_idx - 10)
            previous = _positive_float(float(context.session[lookback_idx, close_idx]), "close")
            move = (context.price / previous) - 1.0
            if context.state.position_side == "long":
                actions.append(3 if move <= -self.threshold_return else 0)
            elif context.state.position_side == "short":
                actions.append(3 if move >= self.threshold_return else 0)
            elif move >= self.threshold_return:
                actions.append(1)
            elif move <= -self.threshold_return:
                actions.append(2)
            else:
                actions.append(0)
        return tuple(actions)

    def diagnostics(self) -> dict[str, object]:
        return {
            "policy": "simple_recent_return_threshold",
            "threshold_return": self.threshold_return,
        }


class Stage08TorchD3qnPolicy:
    policy_name = "stage07b_candidate"
    policy_kind: PolicyKind = "candidate"

    def __init__(
        self,
        *,
        candidate_manifest: Mapping[str, Any],
        device_policy: str = "cpu_only_deterministic",
        torch_num_threads: int = 1,
        torch_num_interop_threads: int = 1,
    ) -> None:
        self._candidate_manifest = dict(candidate_manifest)
        self._torch = _training_runner._import_torch()
        _training_runner._configure_torch_threads_for_values(
            torch=self._torch,
            torch_num_threads=torch_num_threads,
            torch_num_interop_threads=torch_num_interop_threads,
        )
        self._device, self._device_payload = _training_runner._select_torch_device_for_policy(
            torch=self._torch,
            device_policy=cast(Any, device_policy),
        )
        training_config_path = _artifact_path(candidate_manifest, "training_config")
        training_config = _read_json_payload(training_config_path)
        architecture = _architecture_from_training_config(training_config)
        self._architecture = architecture
        self._model = _training_runner._build_torch_d3qn_modules(
            torch=self._torch,
            architecture=architecture,
            device=self._device,
        )
        checkpoint_path = _artifact_path(candidate_manifest, "candidate_checkpoint")
        checkpoint_payload = self._torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise Stage08EvaluationError(
                reason="candidate_checkpoint_payload_invalid",
                field=str(checkpoint_path),
            )
        _training_runner._load_torch_modules_state(
            target=self._model,
            state=checkpoint_payload.get("model_state"),
        )
        for module in self._model.values():
            module.eval()
        self._checkpoint_path = checkpoint_path
        self._training_config_path = training_config_path

    def select_actions(self, contexts: Sequence[Stage08StepContext]) -> tuple[int, ...]:
        if not contexts:
            return ()
        observation_config = Stage08EvaluationConfig().as_observation_config()
        observations = np.vstack(
            [
                _training_runner._build_agent_observation(
                    session=context.session,
                    step_idx=context.step_idx,
                    action_history=context.action_history,
                    training_state=context.state,
                    price=context.price,
                    config=observation_config,
                )
                for context in contexts
            ]
        )
        with self._torch.no_grad():
            tensor = self._torch.as_tensor(
                observations,
                dtype=self._torch.float32,
                device=self._device,
            )
            q_values = _training_runner._torch_d3qn_forward(self._model, tensor)
            actions = q_values.argmax(dim=1).detach().cpu().numpy()
        _training_runner._synchronize_if_needed(
            torch=self._torch,
            device_type=str(self._device.type),
        )
        return tuple(int(action) for action in actions)

    def diagnostics(self) -> dict[str, object]:
        return {
            "architecture": self._architecture.as_payload(),
            "candidate_manifest_hash": str(self._candidate_manifest.get("candidate_manifest_hash")),
            "checkpoint_path": str(self._checkpoint_path),
            "device": self._device_payload,
            "policy": "stage07b_d3qn_argmax",
            "training_config_path": str(self._training_config_path),
        }


def evaluate_stage08_policy_v1(
    *,
    session_features: np.ndarray,
    symbols: Sequence[str],
    signal_times_utc: Sequence[str | None] | None,
    policy: Stage08ActionPolicy,
    config: Stage08EvaluationConfig | None = None,
) -> dict[str, Any]:
    selected_config = Stage08EvaluationConfig() if config is None else config
    features = _validate_session_features(session_features)
    session_count = int(features.shape[0])
    normalized_symbols = _normalize_symbols(symbols, session_count=session_count)
    normalized_signal_times = _normalize_signal_times(
        signal_times_utc,
        session_count=session_count,
    )

    start_wall = time.perf_counter()
    states = [
        RlTrainingState(balance=selected_config.initial_balance)
        for _ in range(session_count)
    ]
    histories: list[list[int | None]] = [
        [None] * selected_config.agent_history_len for _ in range(session_count)
    ]
    session_pnls = np.zeros(session_count, dtype=np.float64)
    session_closed_trades = np.zeros(session_count, dtype=np.int64)
    session_profitable_trades = np.zeros(session_count, dtype=np.int64)
    action_counts = np.zeros(RL_ACTION_COUNT_V1, dtype=np.int64)
    audit_reason_counts: dict[str, int] = {}
    starting_equity = selected_config.initial_balance * float(session_count)
    equity = starting_equity
    peak_equity = starting_equity
    max_drawdown_pct = 0.0
    total_reward = 0.0
    total_funding_pnl = 0.0
    decisions_count = 0

    for step_idx in range(selected_config.agent_session_len):
        contexts = tuple(
            Stage08StepContext(
                session_index=session_idx,
                step_idx=step_idx,
                session=features[session_idx],
                symbol=normalized_symbols[session_idx],
                signal_time_utc=normalized_signal_times[session_idx],
                price=_training_runner._session_close_price(
                    features[session_idx],
                    step_idx=step_idx,
                ),
                state=states[session_idx],
                action_history=tuple(histories[session_idx]),
            )
            for session_idx in range(session_count)
        )
        requested_actions = policy.select_actions(contexts)
        if len(requested_actions) != len(contexts):
            raise Stage08EvaluationError(reason="policy_action_count_mismatch")
        for context, requested_action in zip(contexts, requested_actions, strict=True):
            action_id = _normalize_action(requested_action)
            result = apply_training_reward_step_v1(
                state=states[context.session_index],
                action_id=action_id,
                price=context.price,
                initial_balance=selected_config.initial_balance,
                slippage=selected_config.slippage,
                transaction_fee=selected_config.transaction_fee,
                inaction_penalty_ratio=selected_config.inaction_penalty_ratio,
                is_last_step=(step_idx == selected_config.agent_session_len - 1),
            )
            funding_pnl = _funding_adjustment_for_step(
                state_before=states[context.session_index],
                config=selected_config,
            )
            states[context.session_index] = result.state
            histories[context.session_index] = [
                *histories[context.session_index][1:],
                result.effective_action_id,
            ]
            pnl_after_costs = result.pnl_change + funding_pnl
            session_pnls[context.session_index] += pnl_after_costs
            if result.closed_position:
                session_closed_trades[context.session_index] += 1
            if result.closed_position and pnl_after_costs > 0.0:
                session_profitable_trades[context.session_index] += 1
            action_counts[result.effective_action_id] += 1
            audit_reason_counts[result.audit_reason] = (
                audit_reason_counts.get(result.audit_reason, 0) + 1
            )
            total_reward += result.reward
            total_funding_pnl += funding_pnl
            equity += pnl_after_costs
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0.0:
                drawdown = ((peak_equity - equity) / peak_equity) * 100.0
                max_drawdown_pct = max(max_drawdown_pct, drawdown)
            decisions_count += 1

    wall_seconds = max(time.perf_counter() - start_wall, 0.0)
    net_pnl = float(np.sum(session_pnls, dtype=np.float64))
    trade_count = int(np.sum(session_closed_trades, dtype=np.int64))
    profitable_trades = int(np.sum(session_profitable_trades, dtype=np.int64))
    win_rate = profitable_trades / trade_count if trade_count else 0.0
    ticker_rows = _ticker_stability_payload(
        symbols=normalized_symbols,
        session_pnls=session_pnls,
        session_closed_trades=session_closed_trades,
        session_profitable_trades=session_profitable_trades,
        initial_balance=selected_config.initial_balance,
    )
    scorecard = {
        "action_counts": {
            ACTION_NAMES_BY_ID_V1[action_id]: int(action_counts[action_id])
            for action_id in sorted(ACTION_NAMES_BY_ID_V1)
        },
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "artifact_kind": STAGE08_SCORECARD_KIND_V1,
        "audit_reason_counts": dict(sorted(audit_reason_counts.items())),
        "closed_trades": trade_count,
        "costs": {
            "funding_model": selected_config.funding_model,
            "funding_pnl_quote": _round_float(total_funding_pnl),
            "funding_policy_status": (
                "research_only_approximation"
                if selected_config.funding_model == STAGE08_RESEARCH_FUNDING_MODEL_V1
                else "provided"
            ),
            "slippage_rate": selected_config.slippage,
            "transaction_fee_rate": selected_config.transaction_fee,
        },
        "decisions_count": decisions_count,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "latency_resource_notes": {
            "decisions_per_second": _round_float(decisions_count / max(wall_seconds, 1e-9)),
            "rss_mb_after": _training_runner._rss_mb(),
            "wall_seconds": _round_float(wall_seconds),
        },
        "max_drawdown_pct": _round_float(max_drawdown_pct),
        "net_pnl_after_costs_quote": _round_float(net_pnl),
        "out_of_sample_period": _period_payload(normalized_signal_times),
        "policy_diagnostics": policy.diagnostics(),
        "policy_kind": policy.policy_kind,
        "policy_name": policy.policy_name,
        "profitable_trades": profitable_trades,
        "return_pct_after_costs": _round_float((net_pnl / starting_equity) * 100.0),
        "reward_sum": _round_float(total_reward),
        "schema_version": STAGE08_EVALUATION_SCHEMA_VERSION_V1,
        "session_count": session_count,
        "starting_equity_quote": _round_float(starting_equity),
        "stability_by_ticker": ticker_rows,
        "stability_summary": _stability_summary(ticker_rows),
        "win_rate": _round_float(win_rate),
    }
    return {**scorecard, "scorecard_hash": hash_json_payload_v1(scorecard)}


def build_stage08_evaluation_artifact_v1(
    *,
    generated_at_utc: datetime,
    candidate_manifest_path: str,
    candidate_manifest_sha256: str,
    sessionized_manifest_path: str,
    sessionized_manifest_sha256: str,
    selection: Mapping[str, object],
    scorecards: Sequence[Mapping[str, Any]],
    candidate_report: Mapping[str, Any] | None,
    parity_fixture: Mapping[str, Any],
    config: Stage08EvaluationConfig,
    code_version: Mapping[str, object],
    artifact_hashes: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    ordered_scorecards = sorted(
        (dict(item) for item in scorecards),
        key=lambda row: str(row["policy_name"]),
    )
    candidate = next(
        (item for item in ordered_scorecards if item.get("policy_kind") == "candidate"),
        None,
    )
    baselines = [item for item in ordered_scorecards if item.get("policy_kind") == "baseline"]
    overfit_indicators = _overfit_indicators_payload(
        candidate_scorecard=candidate,
        baseline_scorecards=baselines,
        candidate_report=candidate_report,
    )
    research_candidate_save_allowed = bool(
        candidate is not None
        and float(candidate.get("net_pnl_after_costs_quote", 0.0)) > 0.0
        and bool(parity_fixture.get("passed"))
    )
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "artifact_hashes": dict(artifact_hashes or {}),
        "artifact_kind": STAGE08_EVALUATION_KIND_V1,
        "candidate_dependency": {
            "manifest_path": candidate_manifest_path,
            "manifest_sha256": candidate_manifest_sha256,
            "stage": "07B",
        },
        "code_version": dict(code_version),
        "config": config.as_payload(),
        "config_hash": config.config_hash(),
        "data_quality_report": {
            "blockers": [],
            "grain": "session",
            "limitations": [
                (
                    "funding uses research-only zero approximation because point-in-time "
                    "funding arrays are not consumed by this offline Stage 08 evaluator"
                )
            ],
            "required_hashes_matched": True,
            "sources": ["Stage 06 sessionized manifest", "Stage 07B candidate manifest"],
            "status": "pass_with_warnings",
            "warnings": ["research_only_funding_approximation"],
        },
        "dataset_dependency": {
            "manifest_path": sessionized_manifest_path,
            "manifest_sha256": sessionized_manifest_sha256,
            "stage": "06",
            "training_source": "binance:futures",
        },
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated_at_utc),
        "methodology": {
            "depth": "standard_analysis",
            "decision_unit": "session",
            "method": "out_of_sample_deterministic_simulator_evaluation",
            "promotion_claim": False,
            "sanity_baselines": [str(item["policy_name"]) for item in baselines],
        },
        "next_stage_handoff": {
            "stage09_allowed": research_candidate_save_allowed,
            "stage10a_promotion_profile_required": True,
        },
        "overfit_indicators": overfit_indicators,
        "parity_fixture": dict(parity_fixture),
        "research_candidate_save_allowed": research_candidate_save_allowed,
        "safety": {
            "candidate_for_research_only": True,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "mainnet_submit": False,
            "model_registry_write": False,
            "paper_testnet_live_enabled": False,
            "promotion_or_activation": False,
            "runtime_artifact_root": STAGE08_RUNTIME_ARTIFACT_ROOT_V1,
        },
        "schema_version": STAGE08_EVALUATION_SCHEMA_VERSION_V1,
        "scorecards": ordered_scorecards,
        "selection": dict(selection),
        "stage": "08",
        "status": "accepted_for_research" if research_candidate_save_allowed else "blocked",
    }
    return {**payload, "evaluation_hash": hash_json_payload_v1(payload)}


def stage08_accounting_parity_fixture_v1(
    *,
    config: Stage08EvaluationConfig | None = None,
) -> dict[str, object]:
    selected_config = Stage08EvaluationConfig() if config is None else config
    state = RlTrainingState(balance=selected_config.initial_balance)
    opened = apply_training_reward_step_v1(
        state=state,
        action_id=1,
        price=100.0,
        initial_balance=selected_config.initial_balance,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=selected_config.inaction_penalty_ratio,
    )
    closed = apply_training_reward_step_v1(
        state=opened.state,
        action_id=3,
        price=110.0,
        initial_balance=selected_config.initial_balance,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=selected_config.inaction_penalty_ratio,
    )
    expected_open_fee = -0.1
    expected_close_net_pnl = 9.88011
    expected_total_net_pnl = 9.78011
    observed_total = opened.pnl_change + closed.pnl_change
    passed = (
        abs(opened.pnl_change - expected_open_fee) <= 1e-9
        and abs(closed.pnl_change - expected_close_net_pnl) <= 1e-9
        and abs(observed_total - expected_total_net_pnl) <= 1e-9
    )
    return {
        "closed_pnl_change": _round_float(closed.pnl_change),
        "expected_close_net_pnl": expected_close_net_pnl,
        "expected_open_fee": expected_open_fee,
        "expected_total_net_pnl": expected_total_net_pnl,
        "observed_total_net_pnl": _round_float(observed_total),
        "open_pnl_change": _round_float(opened.pnl_change),
        "passed": passed,
        "source": "apply_training_reward_step_v1",
    }


def default_stage08_evaluation_policies_v1(
    *,
    random_seed: int,
    simple_threshold_return: float,
) -> tuple[Stage08ActionPolicy, ...]:
    return (
        Stage08FixedActionPolicy(policy_name="hold", action_id=0),
        Stage08FixedActionPolicy(policy_name="no_trade", action_id=0),
        Stage08RandomActionPolicy(seed=random_seed),
        Stage08SimpleThresholdPolicy(threshold_return=simple_threshold_return),
    )


def candidate_training_config_from_payload_v1(
    payload: Mapping[str, Any],
) -> CandidateTrainingConfig:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise Stage08EvaluationError(reason="candidate_training_config_missing")
    replay_payload = config.get("replay")
    replay = (
        PrioritizedReplayConfig()
        if not isinstance(replay_payload, Mapping)
        else PrioritizedReplayConfig(
            capacity=int(replay_payload.get("capacity", 512)),
            alpha=float(replay_payload.get("alpha", 0.6)),
            beta=float(replay_payload.get("beta", 0.4)),
            epsilon=float(replay_payload.get("epsilon", 1e-5)),
            min_priority=float(replay_payload.get("min_priority", 1e-5)),
        )
    )
    hidden_dims = tuple(
        int(value) for value in cast(Sequence[Any], config.get("hidden_dims", (128, 128)))
    )
    return CandidateTrainingConfig(
        seed=int(config.get("seed", 240723)),
        train_dataset_version=str(
            config.get("train_dataset_version", "hf_period_rebuild_current_trading")
        ),
        train_split=str(config.get("train_split", "train")),
        validation_dataset_version=str(
            config.get("validation_dataset_version", "hf_period_rebuild_current_trading")
        ),
        validation_split=str(config.get("validation_split", "validation")),
        initial_balance=float(config.get("initial_balance", 100.0)),
        slippage=float(config.get("slippage", 0.0)),
        transaction_fee=float(config.get("transaction_fee", 0.001)),
        inaction_penalty_ratio=float(config.get("inaction_penalty_ratio", 0.0001)),
        batch_size=int(config.get("batch_size", 256)),
        planned_training_steps=int(config.get("planned_training_steps", 100_000)),
        progress_emit_every_steps=int(config.get("progress_emit_every_steps", 10_000)),
        progress_emit_every_sec=int(config.get("progress_emit_every_sec", 300)),
        checkpoint_every_steps=int(config.get("checkpoint_every_steps", 10_000)),
        validation_every_steps=int(config.get("validation_every_steps", 10_000)),
        validation_max_transitions=int(config.get("validation_max_transitions", 4_096)),
        gamma=float(config.get("gamma", 0.99)),
        learning_rate=float(config.get("learning_rate", 0.0005)),
        target_sync_interval=int(config.get("target_sync_interval", 1_000)),
        torch_num_threads=int(config.get("torch_num_threads", 4)),
        torch_num_interop_threads=int(config.get("torch_num_interop_threads", 1)),
        device_policy=cast(Any, str(config.get("device_policy", "cpu_only_deterministic"))),
        replay=replay,
        hidden_dims=hidden_dims,
    )


def _architecture_from_training_config(payload: Mapping[str, Any]) -> D3qnArchitectureConfig:
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise Stage08EvaluationError(reason="candidate_architecture_missing")
    return D3qnArchitectureConfig(
        input_dim=int(architecture["input_dim"]),
        hidden_dims=tuple(int(value) for value in cast(Sequence[Any], architecture["hidden_dims"])),
        value_hidden_dim=int(architecture.get("value_hidden_dim", 64)),
        advantage_hidden_dim=int(architecture.get("advantage_hidden_dim", 64)),
    )


def _validate_session_features(session_features: np.ndarray) -> np.ndarray:
    features = np.asarray(session_features, dtype=np.float32)
    if features.ndim != 3:
        raise Stage08EvaluationError(reason="session_features_must_be_3d")
    expected_shape = (SESSIONIZED_FULL_SEQ_LEN_V1, len(FEATURE_NAMES_V1))
    if tuple(features.shape[1:]) != expected_shape:
        raise Stage08EvaluationError(reason="unexpected_session_shape", field=str(features.shape))
    if features.shape[0] <= 0:
        raise Stage08EvaluationError(reason="empty_session_features")
    if not np.all(np.isfinite(features)):
        raise Stage08EvaluationError(reason="non_finite_session_features")
    close_idx = FEATURE_NAMES_V1.index("close")
    if np.any(features[:, :, close_idx] <= 0.0):
        raise Stage08EvaluationError(reason="non_positive_close_price")
    return np.ascontiguousarray(features, dtype=np.float32)


def _normalize_symbols(symbols: Sequence[str], *, session_count: int) -> tuple[str, ...]:
    if len(symbols) != session_count:
        raise Stage08EvaluationError(reason="symbol_count_mismatch", field="symbols")
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols)
    if any(not symbol for symbol in normalized):
        raise Stage08EvaluationError(reason="empty_symbol", field="symbols")
    return normalized


def _normalize_signal_times(
    signal_times_utc: Sequence[str | None] | None,
    *,
    session_count: int,
) -> tuple[str | None, ...]:
    if signal_times_utc is None:
        return tuple(None for _ in range(session_count))
    if len(signal_times_utc) != session_count:
        raise Stage08EvaluationError(reason="signal_time_count_mismatch")
    return tuple(None if item is None else str(item) for item in signal_times_utc)


def _funding_adjustment_for_step(
    *,
    state_before: RlTrainingState,
    config: Stage08EvaluationConfig,
) -> float:
    if state_before.position_side is None:
        return 0.0
    if config.funding_model == STAGE08_RESEARCH_FUNDING_MODEL_V1:
        return 0.0
    return config.funding_pnl_per_step


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
        indices = np.asarray(
            [idx for idx, item in enumerate(symbols) if item == symbol],
            dtype=np.int64,
        )
        symbol_pnl = float(np.sum(session_pnls[indices], dtype=np.float64))
        symbol_trades = int(np.sum(session_closed_trades[indices], dtype=np.int64))
        symbol_profitable = int(np.sum(session_profitable_trades[indices], dtype=np.int64))
        rows.append(
            {
                "closed_trades": symbol_trades,
                "net_pnl_after_costs_quote": _round_float(symbol_pnl),
                "profitable_trades": symbol_profitable,
                "return_pct_after_costs": _round_float(
                    (symbol_pnl / (float(indices.size) * initial_balance)) * 100.0
                ),
                "session_count": int(indices.size),
                "symbol": symbol,
                "win_rate": _round_float(
                    symbol_profitable / symbol_trades if symbol_trades else 0.0
                ),
            }
        )
    return rows


def _stability_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "negative_ticker_count": 0,
            "positive_ticker_count": 0,
            "ticker_count": 0,
            "ticker_positive_ratio": 0.0,
        }
    pnl_values = np.asarray(
        [float(cast(Any, row["net_pnl_after_costs_quote"])) for row in rows],
        dtype=np.float64,
    )
    positive = int(np.count_nonzero(pnl_values > 0.0))
    negative = int(np.count_nonzero(pnl_values < 0.0))
    return {
        "median_ticker_net_pnl_quote": _round_float(float(np.median(pnl_values))),
        "negative_ticker_count": negative,
        "positive_ticker_count": positive,
        "ticker_count": len(rows),
        "ticker_positive_ratio": _round_float(positive / len(rows)),
        "worst_ticker_net_pnl_quote": _round_float(float(np.min(pnl_values))),
    }


def _period_payload(signal_times: Sequence[str | None]) -> dict[str, object]:
    values = sorted(str(item) for item in signal_times if item)
    return {
        "end_utc": values[-1] if values else None,
        "range_semantics": "session signal timestamp UTC",
        "session_time_count": len(values),
        "start_utc": values[0] if values else None,
    }


def _overfit_indicators_payload(
    *,
    candidate_scorecard: Mapping[str, Any] | None,
    baseline_scorecards: Sequence[Mapping[str, Any]],
    candidate_report: Mapping[str, Any] | None,
) -> dict[str, object]:
    best_baseline_pnl = max(
        (float(row.get("net_pnl_after_costs_quote", 0.0)) for row in baseline_scorecards),
        default=0.0,
    )
    candidate_pnl = (
        float(candidate_scorecard.get("net_pnl_after_costs_quote", 0.0))
        if candidate_scorecard is not None
        else 0.0
    )
    indicators: dict[str, object] = {
        "best_baseline_net_pnl_after_costs_quote": _round_float(best_baseline_pnl),
        "candidate_beats_best_sanity_baseline": candidate_pnl > best_baseline_pnl,
        "candidate_net_pnl_after_costs_quote": _round_float(candidate_pnl),
        "candidate_positive_after_costs": candidate_pnl > 0.0,
        "overfit_warning_codes": [],
    }
    warnings: list[str] = []
    if candidate_pnl <= 0.0:
        warnings.append("candidate_non_positive_out_of_sample_pnl")
    if candidate_pnl <= best_baseline_pnl:
        warnings.append("candidate_does_not_clear_best_sanity_baseline")
    if candidate_report is not None:
        metrics = candidate_report.get("metrics")
        if isinstance(metrics, Mapping):
            train_curve = metrics.get("train_curve")
            validation_curve = metrics.get("validation_curve")
            if isinstance(train_curve, Sequence) and isinstance(validation_curve, Sequence):
                indicators["train_curve_points"] = len(train_curve)
                indicators["validation_curve_points"] = len(validation_curve)
                if train_curve and validation_curve:
                    last_train = train_curve[-1]
                    last_validation = validation_curve[-1]
                    if isinstance(last_train, Mapping) and isinstance(last_validation, Mapping):
                        train_loss = float(last_train.get("loss_window_mean", 0.0))
                        validation_loss = float(last_validation.get("td_mse", 0.0))
                        indicators["last_train_loss_window_mean"] = _round_float(train_loss)
                        indicators["last_validation_td_mse"] = _round_float(validation_loss)
                        if train_loss > 0.0 and validation_loss > train_loss * 5.0:
                            warnings.append("validation_td_mse_much_higher_than_train_loss")
    indicators["overfit_warning_codes"] = sorted(set(warnings))
    return indicators


def _artifact_path(candidate_manifest: Mapping[str, Any], key: str) -> Path:
    artifacts = candidate_manifest.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        raise Stage08EvaluationError(reason="candidate_artifact_hashes_missing")
    item = artifacts.get(key)
    if not isinstance(item, Mapping):
        raise Stage08EvaluationError(reason="candidate_artifact_missing", field=key)
    value = item.get("path")
    if not isinstance(value, str) or not value:
        raise Stage08EvaluationError(reason="candidate_artifact_path_missing", field=key)
    path = Path(value)
    if not path.exists():
        raise Stage08EvaluationError(reason="candidate_artifact_file_missing", field=str(path))
    return path


def _read_json_payload(path: Path) -> dict[str, Any]:
    import json

    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _normalize_action(action_id: int) -> int:
    if isinstance(action_id, bool) or action_id not in ACTION_NAMES_BY_ID_V1:
        raise Stage08EvaluationError(reason="unsupported_action_id", field="action_id")
    return int(action_id)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_float(value: float, *, digits: int = 8) -> float:
    if not np.isfinite(value):
        return float(value)
    return round(float(value), digits)


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage08EvaluationError(reason="invalid_positive_int", field=field)
    return value


def _finite_float(value: float, field: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise Stage08EvaluationError(reason="invalid_finite_float", field=field)
    return parsed


def _positive_float(value: float, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed <= 0.0:
        raise Stage08EvaluationError(reason="invalid_positive_float", field=field)
    return parsed


def _non_negative_float(value: float, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed < 0.0:
        raise Stage08EvaluationError(reason="invalid_non_negative_float", field=field)
    return parsed
