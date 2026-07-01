from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from .action_state_reward_contract import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
)
from .backtest_evaluation import stage08_accounting_parity_fixture_v1
from .feature_contract import FEATURE_CONTRACT_HASH_V1
from .hf_original_evaluation import (
    CheckpointName,
    DevicePolicy,
    HfOriginalEvaluationConfig,
    HfOriginalSplitData,
    ScorecardKind,
    _alpha_config_from_training_config_payload,
    _artifact_path,
    _atomic_write_json,
    _BacktestRiskManagementState,
    _empty_action_counts,
    _file_payload,
    _format_utc,
    _grouped_backtest_indices,
    _limit_split,
    _load_normalization_stats,
    _read_json_payload,
    _risk_management_action_override_v1,
    _risk_management_payload_v1,
    _round_float,
    _scorecard_by_name,
    _scorecard_payload,
    _update_risk_management_state_after_step_v1,
    _without_hash,
    alpha_with_evaluation_overrides_v1,
    default_hf_original_evaluation_config_v1,
    evaluate_stage08d_baseline_backtest_v1,
    evaluate_stage08d_grouped_backtest_v1,
    evaluate_stage08d_test_episodes_v1,
)
from .raw_feature_dataset import hash_json_payload_v1
from .roehub_native_training import STAGE08E_CANDIDATE_LEVEL_V1
from .upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    SelectionStrategy,
    TorchD3qnPerAgent,
    UpstreamAlphaConfig,
    default_upstream_alpha_config_v1,
    mask_upstream_training_action_v1,
    session_close_price_v1,
)

STAGE08F_EVALUATION_SCHEMA_VERSION_V1 = 1
STAGE08F_EVALUATION_KIND_V1 = "rl_trading_stage08f_roehub_native_evaluation"
STAGE08F_SCORECARD_KIND_V1 = "rl_trading_stage08f_native_scorecard"
STAGE08F_EVALUATION_CONFIG_ID_V1 = "roehub_stage08f_native_evaluation_config_v1"
STAGE08F_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08f_roehub_native_backtest_evaluation_v1"

RoehubNativeSplitData = HfOriginalSplitData


class RoehubNativeEvaluationError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RoehubNativeEvaluationConfig:
    alpha: UpstreamAlphaConfig = field(default_factory=default_upstream_alpha_config_v1)
    checkpoint_name: CheckpointName = "best"
    selection_strategy: SelectionStrategy = "advantage_based_filter"
    device_policy: DevicePolicy = "cpu_only_deterministic"
    test_max_sessions: int | None = None
    backtest_max_sessions: int | None = None
    simple_threshold_return: float = 0.001
    deterministic_random_seed: int = 806

    def __post_init__(self) -> None:
        default_hf_original_evaluation_config_v1().__class__(
            alpha=self.alpha,
            checkpoint_name=self.checkpoint_name,
            selection_strategy=self.selection_strategy,
            device_policy=self.device_policy,
            test_max_sessions=self.test_max_sessions,
            backtest_max_sessions=self.backtest_max_sessions,
            simple_threshold_return=self.simple_threshold_return,
        )
        if self.deterministic_random_seed < 0:
            raise RoehubNativeEvaluationError(reason="invalid_deterministic_random_seed")

    def with_alpha(self, alpha: UpstreamAlphaConfig) -> RoehubNativeEvaluationConfig:
        return replace(self, alpha=alpha)

    def as_hf_config(self) -> HfOriginalEvaluationConfig:
        return HfOriginalEvaluationConfig(
            alpha=self.alpha,
            checkpoint_name=self.checkpoint_name,
            selection_strategy=self.selection_strategy,
            device_policy=self.device_policy,
            test_max_sessions=self.test_max_sessions,
            backtest_max_sessions=self.backtest_max_sessions,
            simple_threshold_return=self.simple_threshold_return,
        )

    def as_payload(self) -> dict[str, object]:
        payload = self.as_hf_config().as_payload()
        payload["config_id"] = STAGE08F_EVALUATION_CONFIG_ID_V1
        payload["deterministic_random_seed"] = self.deterministic_random_seed
        payload["stage"] = "08F"
        return payload

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


def default_roehub_native_evaluation_config_v1() -> RoehubNativeEvaluationConfig:
    return RoehubNativeEvaluationConfig()


def run_stage08f_roehub_native_evaluation_v1(
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    test_split: RoehubNativeSplitData,
    backtest_split: RoehubNativeSplitData,
    output_root: Path,
    run_id: str,
    config: RoehubNativeEvaluationConfig | None = None,
    generated_at_utc: datetime | None = None,
    code_version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_candidate_manifest(candidate_manifest)
    training_config = _read_json_payload(_artifact_path(candidate_manifest, "training_config"))
    alpha = _alpha_config_from_training_config_payload(training_config)
    selected_config = (
        default_roehub_native_evaluation_config_v1().with_alpha(alpha)
        if config is None
        else config.with_alpha(alpha_with_evaluation_overrides_v1(alpha, config.alpha))
    )
    if (
        selected_config.checkpoint_name == "best"
        and _checkpoint_policy(candidate_manifest).get("default_evaluation_checkpoint") != "best"
    ):
        raise RoehubNativeEvaluationError(reason="best_checkpoint_not_default")

    normalization_stats = _load_normalization_stats(
        _artifact_path(candidate_manifest, "normalization_stats"),
        expected_hash=str(candidate_manifest.get("normalization_stats_hash", "")),
    )
    agent, checkpoint_payload = _load_stage08e_checkpoint_agent(
        candidate_manifest=candidate_manifest,
        checkpoint_name=selected_config.checkpoint_name,
        config=selected_config,
    )
    hf_config = selected_config.as_hf_config()
    test_limited = _limit_split(test_split, selected_config.test_max_sessions)
    backtest_limited = _limit_split(backtest_split, selected_config.backtest_max_sessions)
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_test = _native_scorecard(
        evaluate_stage08d_test_episodes_v1(
            split=test_limited,
            normalization_stats=normalization_stats,
            agent=agent,
            config=hf_config,
        )
    )
    filtered_backtest, balance_curve = evaluate_stage08d_grouped_backtest_v1(
        split=backtest_limited,
        normalization_stats=normalization_stats,
        agent=agent,
        config=hf_config,
    )
    filtered_backtest = _native_scorecard(filtered_backtest)
    baselines = [
        _native_scorecard(
            evaluate_stage08d_baseline_backtest_v1(
                split=backtest_limited,
                config=hf_config,
                policy_name="hold",
                fixed_action_id=0,
            )
        ),
        _native_scorecard(
            evaluate_stage08d_baseline_backtest_v1(
                split=backtest_limited,
                config=hf_config,
                policy_name="no_trade",
                fixed_action_id=0,
            )
        ),
        _native_scorecard(
            evaluate_stage08f_random_baseline_backtest_v1(
                split=backtest_limited,
                config=selected_config,
            )
        ),
        _native_scorecard(
            evaluate_stage08d_baseline_backtest_v1(
                split=backtest_limited,
                config=hf_config,
                policy_name="simple_recent_return_threshold",
                fixed_action_id=None,
            )
        ),
    ]
    scorecards = [raw_test, filtered_backtest, *baselines]
    scorecards_path = run_dir / "scorecards.json"
    balance_curve_path = run_dir / "filtered_backtest_balance_curve.json"
    parity_fixture_path = run_dir / "simulator_accounting_parity_fixture.json"
    parity_fixture = stage08_accounting_parity_fixture_v1()
    _atomic_write_json(scorecards_path, {"scorecards": scorecards})
    _atomic_write_json(balance_curve_path, {"balance_curve": balance_curve})
    _atomic_write_json(parity_fixture_path, parity_fixture)

    artifact_hashes = {
        "balance_curve": _file_payload(balance_curve_path),
        "scorecards": _file_payload(scorecards_path),
        "simulator_accounting_parity_fixture": _file_payload(parity_fixture_path),
    }
    manifest = build_stage08f_evaluation_artifact_v1(
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
        simulator_accounting_parity_fixture=parity_fixture,
        code_version={} if code_version is None else dict(code_version),
        artifact_hashes=artifact_hashes,
    )
    manifest_path = run_dir / "stage08f_evaluation_manifest.json"
    manifest = {**manifest, "evaluation_manifest_path": str(manifest_path)}
    manifest = {**manifest, "evaluation_hash": hash_json_payload_v1(_without_hash(manifest))}
    _atomic_write_json(manifest_path, manifest)
    return {**manifest, "evaluation_manifest_sha256": _file_payload(manifest_path)["sha256"]}


def evaluate_stage08f_random_baseline_backtest_v1(
    *,
    split: RoehubNativeSplitData,
    config: RoehubNativeEvaluationConfig,
) -> dict[str, Any]:
    selected_indices, grouping_payload = _grouped_backtest_indices(
        split.signal_times_utc,
        max_parallel_sessions=config.alpha.max_parallel_sessions,
        agent_session_len=config.alpha.agent_session_len,
    )
    selected_sequences = split.sequences[np.asarray(selected_indices, dtype=np.int64)]
    selected_split = RoehubNativeSplitData(
        split_name=split.split_name,
        sequences=selected_sequences,
        symbols=tuple(split.symbols[idx] for idx in selected_indices),
        signal_times_utc=tuple(split.signal_times_utc[idx] for idx in selected_indices),
        source_payload=split.source_payload,
        volatility_scores=tuple(split.volatility_scores[idx] for idx in selected_indices),
    )
    rng = np.random.default_rng(config.deterministic_random_seed)
    action_counts = _empty_action_counts()
    audit_reason_counts: dict[str, int] = {}
    risk_management_reason_counts: dict[str, int] = {}
    session_pnls: list[float] = []
    closed_trades: list[int] = []
    profitable_trades: list[int] = []
    decisions_count = 0
    training_reward_sum = 0.0
    backtest_reporting_reward_sum = 0.0
    shared_balance = float(config.alpha.initial_balance)
    risk_management_config = config.alpha
    for session in selected_split.sequences:
        position_size = shared_balance * config.alpha.position_fraction
        backtest_alpha = replace(config.alpha, initial_balance=position_size)
        risk_management_config = backtest_alpha
        state = RlTrainingState(balance=position_size)
        session_pnl = 0.0
        trade_realized_since_open = 0.0
        latest_closed = 0
        latest_profitable = 0
        risk_state = _BacktestRiskManagementState()
        for step_idx in range(backtest_alpha.agent_session_len):
            price = session_close_price_v1(session, step_idx=step_idx, config=backtest_alpha)
            requested_action_id = int(rng.integers(0, len(ACTION_NAMES_BY_ID_V1)))
            action_id = mask_upstream_training_action_v1(
                action_id=requested_action_id,
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
            action_counts[ACTION_NAMES_BY_ID_V1[result.effective_action_id]] += 1
            audit_reason_counts[result.audit_reason] = (
                audit_reason_counts.get(result.audit_reason, 0) + 1
            )
            session_pnl += result.pnl_change
            trade_realized_since_open += result.pnl_change
            training_reward_sum += result.reward
            if result.closed_position:
                shared_balance += trade_realized_since_open
                trade_realized_since_open = 0.0
            latest_closed = result.state.closed_trades
            latest_profitable = result.state.profitable_trades
            decisions_count += 1
        session_pnls.append(session_pnl)
        closed_trades.append(latest_closed)
        profitable_trades.append(latest_profitable)
    scorecard = _scorecard_payload(
        split=selected_split,
        policy_name="deterministic_random_valid_action",
        policy_kind=cast(ScorecardKind, "baseline"),
        evaluation_surface="Roehub-native grouped baseline backtest",
        session_pnls=np.asarray(session_pnls, dtype=np.float64),
        session_closed_trades=np.asarray(closed_trades, dtype=np.int64),
        session_profitable_trades=np.asarray(profitable_trades, dtype=np.int64),
        action_counts=action_counts,
        audit_reason_counts=audit_reason_counts,
        decisions_count=decisions_count,
        reward_sum=backtest_reporting_reward_sum,
        starting_equity_quote=config.alpha.initial_balance,
        config=config.as_hf_config(),
        extra={
            "acceptance_backtest": False,
            "backtest_reporting_reward_sum": _round_float(backtest_reporting_reward_sum),
            "baseline_policy": "deterministic_random_valid_action",
            "deterministic_random_seed": config.deterministic_random_seed,
            "filter_policy": None,
            "grouping": grouping_payload,
            "position_fraction": config.alpha.position_fraction,
            "position_fraction_application": "shared_balance_position_fraction",
            "risk_management": _risk_management_payload_v1(
                config=risk_management_config,
                reason_counts=risk_management_reason_counts,
            ),
            "shared_balance_final_quote": _round_float(shared_balance),
            "shared_balance_initial_quote": _round_float(config.alpha.initial_balance),
            "training_reward_sum": _round_float(training_reward_sum),
        },
    )
    return {**scorecard, "scorecard_hash": hash_json_payload_v1(scorecard)}


def build_stage08f_evaluation_artifact_v1(
    *,
    generated_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    test_split: RoehubNativeSplitData,
    backtest_split: RoehubNativeSplitData,
    config: RoehubNativeEvaluationConfig,
    normalization_stats_hash: str,
    checkpoint_payload: Mapping[str, Any],
    scorecards: Sequence[Mapping[str, Any]],
    simulator_accounting_parity_fixture: Mapping[str, object],
    code_version: Mapping[str, Any],
    artifact_hashes: Mapping[str, object],
) -> dict[str, Any]:
    ordered_scorecards = list(scorecards)
    filtered = _scorecard_by_name(ordered_scorecards, "roehub_native_candidate_filtered_backtest")
    baselines = [item for item in ordered_scorecards if item.get("policy_kind") == "baseline"]
    verdict = _research_candidate_verdict_payload(
        filtered_backtest=filtered,
        baselines=baselines,
        candidate_manifest=candidate_manifest,
        simulator_accounting_parity_fixture=simulator_accounting_parity_fixture,
    )
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08F_EVALUATION_KIND_V1,
        "candidate_dependency": {
            "candidate_level": candidate_manifest.get("candidate_level"),
            "checkpoint_name": config.checkpoint_name,
            "checkpoint_policy": dict(
                cast(Mapping[str, object], candidate_manifest["checkpoint_policy"])
            ),
            "checkpoint_stage": checkpoint_payload.get("stage"),
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": candidate_manifest_sha256,
            "stage": "08E",
        },
        "code_version": dict(code_version),
        "config": config.as_payload(),
        "config_hash": config.config_hash(),
        "data_quality_report": {
            "blockers": [],
            "grain": "roehub_native_stage06_session",
            "required_hashes_matched": True,
            "sources": [
                "Stage 06 sessionized test/backtest artifacts",
                "Stage 08E roehub_native_candidate",
            ],
            "status": "pass",
            "warnings": [],
        },
        "dataset_dependency": {
            "backtest_split": dict(backtest_split.source_payload),
            "test_split": dict(test_split.source_payload),
            "training_source": "binance:futures",
            "stage": "06",
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
            "business_claim": (
                "Roehub-native research-candidate decision only; no promotion or "
                "runtime activation claim"
            ),
            "decision_unit": "roehub_native_stage06_session",
            "method": "test_episode_rollout_plus_grouped_filtered_backtest",
            "raw_argmax_acceptance": False,
        },
        "native_research_verdict": verdict,
        "normalization_stats_hash": normalization_stats_hash,
        "overfit_indicators": verdict["warning_indicators"],
        "research_candidate_save_allowed": verdict["research_candidate_save_allowed"],
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
        "schema_version": STAGE08F_EVALUATION_SCHEMA_VERSION_V1,
        "scorecards": ordered_scorecards,
        "simulator_accounting_parity_fixture": dict(simulator_accounting_parity_fixture),
        "stage": "08F",
        "stage09_handoff": {
            "allowed": verdict["research_candidate_save_allowed"],
            "next_stage": "09" if verdict["research_candidate_save_allowed"] else None,
            "reason": verdict["decision_reason"],
        },
        "status": "accepted_for_research"
        if verdict["research_candidate_save_allowed"]
        else "blocked",
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
        "warning_register_from_stage08d": candidate_manifest.get(
            "warning_register_from_stage08d",
            [],
        ),
    }
    return {**payload, "evaluation_hash": hash_json_payload_v1(payload)}


def _load_stage08e_checkpoint_agent(
    *,
    candidate_manifest: Mapping[str, Any],
    checkpoint_name: CheckpointName,
    config: RoehubNativeEvaluationConfig,
) -> tuple[TorchD3qnPerAgent, dict[str, Any]]:
    torch_key = "best_checkpoint" if checkpoint_name == "best" else "final_checkpoint"
    checkpoint_path = _artifact_path(candidate_manifest, torch_key)
    agent = TorchD3qnPerAgent(config=config.alpha, device_policy=config.device_policy)
    payload = agent.torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    if not isinstance(payload, dict):
        raise RoehubNativeEvaluationError(
            reason="checkpoint_payload_invalid",
            field=str(checkpoint_path),
        )
    if payload.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise RoehubNativeEvaluationError(reason="checkpoint_architecture_mismatch")
    if payload.get("stage") != "08E":
        raise RoehubNativeEvaluationError(reason="checkpoint_stage_mismatch")
    policy_state = payload.get("policy_state")
    target_state = payload.get("target_state")
    if not isinstance(policy_state, Mapping) or not isinstance(target_state, Mapping):
        raise RoehubNativeEvaluationError(reason="checkpoint_model_state_missing")
    agent.policy_net.load_state_dict(policy_state)
    agent.target_net.load_state_dict(target_state)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent, cast(dict[str, Any], payload)


def _native_scorecard(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(scorecard)
    name = str(out.get("policy_name", ""))
    name_map = {
        "hf_original_candidate_filtered_backtest": "roehub_native_candidate_filtered_backtest",
        "hf_original_candidate_raw_argmax_test_diagnostic": (
            "roehub_native_candidate_raw_argmax_test_diagnostic"
        ),
    }
    out["artifact_kind"] = STAGE08F_SCORECARD_KIND_V1
    out["policy_name"] = name_map.get(name, name)
    out["schema_version"] = STAGE08F_EVALUATION_SCHEMA_VERSION_V1
    surface = str(out.get("evaluation_surface", ""))
    out["evaluation_surface"] = surface.replace("HF", "Roehub-native Stage 06")
    period = out.get("out_of_sample_period")
    if isinstance(period, Mapping):
        out["out_of_sample_period"] = {
            **dict(period),
            "range_semantics": "Stage 06 signal_time_ms UTC",
        }
    out["scorecard_hash"] = hash_json_payload_v1(
        {key: value for key, value in out.items() if key != "scorecard_hash"}
    )
    return out


def _research_candidate_verdict_payload(
    *,
    filtered_backtest: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    simulator_accounting_parity_fixture: Mapping[str, object],
) -> dict[str, Any]:
    candidate_pnl = float(filtered_backtest.get("net_pnl_after_costs_quote", 0.0))
    candidate_closed_trades = int(filtered_backtest.get("closed_trades", 0))
    best_baseline = max(
        (float(item.get("net_pnl_after_costs_quote", 0.0)) for item in baselines),
        default=0.0,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(simulator_accounting_parity_fixture.get("passed")):
        blockers.append("simulator_accounting_parity_failed")
    if candidate_pnl <= 0.0:
        blockers.append("candidate_non_positive_native_backtest_pnl")
    if candidate_closed_trades <= 0:
        blockers.append("no_actionable_native_research_trades")
    if candidate_pnl <= best_baseline:
        warnings.append("candidate_does_not_clear_best_sanity_baseline")
    stability = filtered_backtest.get("stability_summary")
    if isinstance(stability, Mapping):
        positive_ratio = float(stability.get("session_positive_ratio", 0.0))
        if positive_ratio < 0.45:
            warnings.append("low_positive_session_ratio")
    if candidate_manifest.get("warning_register_from_stage08d"):
        warnings.append("stage08d_warning_register_carried_forward")
    allowed = not blockers
    return {
        "best_baseline_net_pnl_after_costs_quote": _round_float(best_baseline),
        "blockers": sorted(set(blockers)),
        "candidate_beats_best_sanity_baseline": candidate_pnl > best_baseline,
        "candidate_manifest_hash": str(candidate_manifest.get("candidate_manifest_hash")),
        "candidate_net_pnl_after_costs_quote": _round_float(candidate_pnl),
        "candidate_positive_after_costs": candidate_pnl > 0.0,
        "decision_reason": "research_candidate_saved" if allowed else "blocked",
        "research_candidate_save_allowed": allowed,
        "simulator_accounting_parity_passed": bool(
            simulator_accounting_parity_fixture.get("passed")
        ),
        "status": "accepted_for_research" if allowed else "blocked",
        "warning_indicators": sorted(set(warnings)),
    }


def _validate_candidate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("stage") != "08E":
        raise RoehubNativeEvaluationError(reason="candidate_manifest_stage_mismatch")
    if value.get("candidate_level") != STAGE08E_CANDIDATE_LEVEL_V1:
        raise RoehubNativeEvaluationError(reason="candidate_manifest_level_mismatch")
    if value.get("status") != "completed":
        raise RoehubNativeEvaluationError(reason="candidate_manifest_not_completed")
    if value.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise RoehubNativeEvaluationError(reason="candidate_architecture_mismatch")


def _checkpoint_policy(candidate_manifest: Mapping[str, Any]) -> Mapping[str, object]:
    value = candidate_manifest.get("checkpoint_policy")
    if not isinstance(value, Mapping):
        raise RoehubNativeEvaluationError(reason="candidate_checkpoint_policy_missing")
    return cast(Mapping[str, object], value)
