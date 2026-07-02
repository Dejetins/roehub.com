from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.rl_trading import (  # noqa: E402
    stage08f_roehub_native_backtest_evaluation as native_eval_cli,
)
from scripts.rl_trading import (  # noqa: E402
    stage08g_cpu_optuna_calibration as optuna_cli,
)
from scripts.rl_trading import (  # noqa: E402
    stage08h_oracle_supervised_dataset_diagnostics as diagnostics_cli,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    SESSIONIZED_PRE_SIGNAL_LEN_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    UpstreamAlphaConfig,
    hash_json_payload_v1,
)

STAGE08L_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08l_reward_warm_start_research_v1"
STAGE08L_SCHEMA_VERSION_V1 = 1
STAGE08L_ARTIFACT_KIND_V1 = "rl_trading_stage08l_reward_warm_start_research"
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08L_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_STAGE08I2_MATRIX_PATH = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08i2_exhaustive_methodology_discrepancy_audit_v1/"
    "stage08i2_methodology_discrepancy_matrix.json"
)
DEFAULT_STAGE08I4_MATRIX_PATH = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08i4_post_repair_methodology_recheck_v1/"
    "stage08i4_methodology_recheck_matrix.json"
)
DEFAULT_STAGE08K_SUMMARY_PATH = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/"
    "stage08k_dual_branch_cpu_76f51186c00ecb54255e/"
    "stage08k_dual_branch_cpu_run_summary.json"
)
DEFAULT_PROMPT_PATH = (
    REPO_ROOT
    / ".codex/agents/generated/rl-trading-agent-platform-v1/08l-reward-warm-start-research.md"
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_PROFILE = "30/10"
MANDATORY_08I2_SURFACES = {
    "action_q_policy_distribution",
    "dataset_geometry_and_distribution",
    "full_evaluator_backtest_parity",
    "optuna_and_calibration_overfit",
    "past_only_signal_strength",
    "reward_sparsity_and_semantics",
    "sanity_baselines",
    "session_extractor_policy",
}
STAGE08L_DOMINANCE_SHARE_LIMIT = 0.80
STAGE08L_MIN_GROUP_POSITIVE_RATIO = 0.25
STAGE08L_MAX_OPEN_SIDE_SHARE = 0.95


class Stage08LResearchError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except Stage08LResearchError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] == "accepted" else 2


def _run(args: argparse.Namespace) -> dict[str, Any]:
    generated = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    alpha = UpstreamAlphaConfig()
    profile = diagnostics_cli._parse_profile(args.profile)  # noqa: SLF001
    i2_matrix = _load_and_validate_i2_matrix(args.stage08i2_matrix_path)
    i4_matrix = _load_and_validate_i4_matrix(args.stage08i4_matrix_path)
    stage08k_summary = _load_and_validate_stage08k_summary(args.stage08k_summary_path)
    stage08k_final_manifest_path = _stage08k_native_final_manifest_path(stage08k_summary)
    stage08k_final_manifest = _read_json(stage08k_final_manifest_path)
    stage08k_final_scorecard = optuna_cli._candidate_scorecard(  # noqa: SLF001
        stage08k_final_manifest,
        branch="roehub_native",
    )
    stage08k_gate = _strict_gate_snapshot(stage08k_summary=stage08k_summary)

    manifest = native_eval_cli._read_json(args.stage08j_manifest_path)  # noqa: SLF001
    manifest_sha256 = native_eval_cli.compute_file_sha256(args.stage08j_manifest_path)
    train_split = native_eval_cli._load_stage06_split(  # noqa: SLF001
        manifest=manifest,
        manifest_path=args.stage08j_manifest_path,
        manifest_sha256=manifest_sha256,
        dataset_version=args.dataset_version,
        split="train",
        max_sessions=args.max_train_sessions,
        max_artifacts=args.max_train_artifacts,
        allow_fixture_hashes=args.allow_fixture_hashes,
        accepted_stages=("08J",),
    )
    test_split = native_eval_cli._load_stage06_split(  # noqa: SLF001
        manifest=manifest,
        manifest_path=args.stage08j_manifest_path,
        manifest_sha256=manifest_sha256,
        dataset_version=args.dataset_version,
        split="test",
        max_sessions=args.max_eval_sessions,
        max_artifacts=args.max_eval_artifacts,
        allow_fixture_hashes=args.allow_fixture_hashes,
        accepted_stages=("08J",),
    )
    backtest_split = native_eval_cli._load_stage06_split(  # noqa: SLF001
        manifest=manifest,
        manifest_path=args.stage08j_manifest_path,
        manifest_sha256=manifest_sha256,
        dataset_version=args.dataset_version,
        split="backtest",
        max_sessions=args.max_eval_sessions,
        max_artifacts=args.max_eval_artifacts,
        allow_fixture_hashes=args.allow_fixture_hashes,
        accepted_stages=("08J",),
    )

    cost_ratio = 2.0 * (alpha.transaction_fee + alpha.slippage)
    supervised = _supervised_warm_start_payload(
        train_sequences=train_split.sequences,
        eval_splits={
            "train": train_split.sequences,
            "test": test_split.sequences,
            "backtest": backtest_split.sequences,
        },
        profile=profile,
        cost_ratio=cost_ratio,
    )
    backtest_predictions = cast(
        np.ndarray,
        supervised["split_predictions"]["backtest"]["ridge_past_window_model"],
    )
    simple_predictions = diagnostics_cli._recent_return_rule_labels(  # noqa: SLF001
        sequences=backtest_split.sequences,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    random_predictions = _deterministic_random_labels(
        count=backtest_split.sequences.shape[0],
        seed=args.deterministic_random_seed,
    )
    oracle = diagnostics_cli._oracle_payload(  # noqa: SLF001
        sequences=backtest_split.sequences,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    oracle_labels = cast(np.ndarray, oracle["labels"])
    scorecards = [
        _fixed_horizon_bandit_scorecard(
            split=backtest_split,
            labels=np.zeros(backtest_split.sequences.shape[0], dtype=np.int8),
            profile=profile,
            policy_name="hold_no_trade",
            cost_ratio=cost_ratio,
            alpha=alpha,
        ),
        _fixed_horizon_bandit_scorecard(
            split=backtest_split,
            labels=random_predictions,
            profile=profile,
            policy_name="deterministic_random_contextual_bandit",
            cost_ratio=cost_ratio,
            alpha=alpha,
        ),
        _fixed_horizon_bandit_scorecard(
            split=backtest_split,
            labels=simple_predictions,
            profile=profile,
            policy_name="simple_recent_return_threshold_contextual_bandit",
            cost_ratio=cost_ratio,
            alpha=alpha,
        ),
        _fixed_horizon_bandit_scorecard(
            split=backtest_split,
            labels=backtest_predictions,
            profile=profile,
            policy_name="supervised_oracle_label_warm_start_contextual_bandit",
            cost_ratio=cost_ratio,
            alpha=alpha,
        ),
        _fixed_horizon_bandit_scorecard(
            split=backtest_split,
            labels=oracle_labels,
            profile=profile,
            policy_name="oracle_label_upper_bound_not_candidate",
            cost_ratio=cost_ratio,
            alpha=alpha,
        ),
    ]
    reward_proxy = diagnostics_cli._reward_sparsity_payload(  # noqa: SLF001
        oracle=oracle,
        session_len=profile[1],
        flat_hold_penalty=alpha.inaction_penalty_ratio,
    )
    decision = _candidate_path_decision(
        scorecards=scorecards,
        supervised=supervised,
        min_trades=args.min_closed_trades,
    )
    run_id = args.run_id or _default_run_id(
        args=args,
        backtest_split=backtest_split,
        decision=decision,
        manifest_sha256=manifest_sha256,
        profile=profile,
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bounded_matrix = _bounded_experiment_matrix(
        args=args,
        profile=profile,
        output_root=args.output_root,
        run_id=run_id,
    )
    prompt_sha256 = _file_sha256_hex(args.prompt_path)
    payload = {
        "artifact_kind": STAGE08L_ARTIFACT_KIND_V1,
        "bounded_experiment_matrix": bounded_matrix,
        "candidate_path_decision": decision,
        "code_version": _source_state_payload(),
        "comparison": {
            "contextual_bandit_scorecards": scorecards,
            "reward_proxy": reward_proxy,
            "stage08k_native_final_scorecard": {
                "closed_trades": stage08k_final_scorecard.get("closed_trades"),
                "net_pnl_after_costs_quote": stage08k_final_scorecard.get(
                    "net_pnl_after_costs_quote"
                ),
                "policy_name": stage08k_final_scorecard.get("policy_name"),
                "return_pct_after_costs": stage08k_final_scorecard.get(
                    "return_pct_after_costs"
                ),
                "win_rate": stage08k_final_scorecard.get("win_rate"),
            },
            "stage08k_strict_gate": stage08k_gate,
            "supervised_warm_start": _without_predictions(supervised),
        },
        "contract_marker": "reward_research_not_contract_replacement",
        "cost_model": {
            "position_fraction": alpha.position_fraction,
            "round_trip_cost_ratio": cost_ratio,
            "slippage": alpha.slippage,
            "transaction_fee": alpha.transaction_fee,
        },
        "data_quality": {
            "article_manifest_path": str(args.stage08j_manifest_path),
            "article_manifest_sha256": manifest_sha256,
            "backtest_split": dict(backtest_split.source_payload),
            "input_matrices": {
                "stage08i2": i2_matrix,
                "stage08i4": i4_matrix,
            },
            "stage08k_summary_path": str(args.stage08k_summary_path),
            "stage08k_summary_sha256": _file_sha256_hex(args.stage08k_summary_path),
        },
        "delivery_state": (
            "target_host_non_production_research_pre_main when run on Mac Studio; "
            "no registry promotion, paper/testnet/live/mainnet, browser/auth, "
            "or exchange side effect"
        ),
        "generated_at_utc": _format_utc(generated),
        "methodology": {
            "analysis_depth": "research",
            "baseline_reward_contract": (
                "Stage 02C realized PnL / initial balance minus flat-hold penalty"
            ),
            "decision_unit": "roehub_native_stage08j_article_selector_session",
            "profile": {"agent_history_len": profile[0], "agent_session_len": profile[1]},
            "research_only_reward_variants": [
                "dense_mark_to_market_proxy",
                "realized_plus_unrealized_delta_proxy",
                "transaction_cost_aware_fixed_horizon_proxy",
            ],
            "stage": "08L",
        },
        "prompt": {
            "path": str(args.prompt_path.relative_to(REPO_ROOT)),
            "sha256": prompt_sha256,
        },
        "proof_boundary": "target_host_non_production_research_pre_main",
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": {
            "browser_auth_used": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "mainnet_submit": False,
            "model_registry_write": False,
            "paper_testnet_live_enabled": False,
            "promotion_or_activation": False,
            "stage02c_reward_contract_replaced": False,
        },
        "schema_version": STAGE08L_SCHEMA_VERSION_V1,
        "stage": "08L",
        "stage09_allowed": False,
        "status": "accepted" if decision["candidate_path_justified"] else "blocked",
    }
    summary = {**payload, "summary_hash": hash_json_payload_v1(payload)}
    summary_path = run_dir / "stage08l_reward_warm_start_research_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "candidate_path_justified": decision["candidate_path_justified"],
        "run_dir": str(run_dir),
        "run_id": run_id,
        "stage09_allowed": False,
        "status": summary["status"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256_hex(summary_path),
    }


def _supervised_warm_start_payload(
    *,
    train_sequences: np.ndarray,
    eval_splits: Mapping[str, np.ndarray],
    profile: tuple[int, int],
    cost_ratio: float,
) -> dict[str, Any]:
    train_features = diagnostics_cli._feature_matrix(  # noqa: SLF001
        sequences=train_sequences,
        profile=profile,
    )
    train_oracle = diagnostics_cli._oracle_payload(  # noqa: SLF001
        sequences=train_sequences,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    train_labels = cast(np.ndarray, train_oracle["labels"])
    scaler_mean = np.mean(train_features, axis=0)
    scaler_std = np.std(train_features, axis=0)
    scaler_std = np.where(scaler_std == 0.0, 1.0, scaler_std)
    train_x = (train_features - scaler_mean) / scaler_std
    weights = diagnostics_cli._fit_ridge_classifier(train_x, train_labels)  # noqa: SLF001
    majority_label = int(np.bincount(train_labels, minlength=3).argmax())
    split_payloads: dict[str, Any] = {}
    split_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name, sequences in eval_splits.items():
        features = diagnostics_cli._feature_matrix(sequences=sequences, profile=profile)  # noqa: SLF001
        oracle = diagnostics_cli._oracle_payload(  # noqa: SLF001
            sequences=sequences,
            profile=profile,
            cost_ratio=cost_ratio,
        )
        labels = cast(np.ndarray, oracle["labels"])
        x = (features - scaler_mean) / scaler_std
        predicted = diagnostics_cli._predict_ridge_classifier(x, weights)  # noqa: SLF001
        recent = diagnostics_cli._recent_return_rule_labels(  # noqa: SLF001
            sequences=sequences,
            profile=profile,
            cost_ratio=cost_ratio,
        )
        majority = np.full(labels.shape, majority_label, dtype=np.int8)
        split_payloads[split_name] = {
            "label_counts": diagnostics_cli._label_counts(labels),  # noqa: SLF001
            "majority_baseline": diagnostics_cli._classification_metrics(  # noqa: SLF001
                labels,
                majority,
            ),
            "recent_return_baseline": diagnostics_cli._classification_metrics(  # noqa: SLF001
                labels,
                recent,
            ),
            "ridge_past_window_model": {
                **diagnostics_cli._classification_metrics(labels, predicted),  # noqa: SLF001
                "prediction_counts": diagnostics_cli._label_counts(predicted),  # noqa: SLF001
            },
        }
        split_predictions[split_name] = {
            "majority": majority,
            "recent_return_baseline": recent,
            "ridge_past_window_model": predicted,
        }
    return {
        "feature_count": int(train_features.shape[1]),
        "model": "closed_form_ridge_classifier_numpy",
        "split_predictions": split_predictions,
        "splits": split_payloads,
        "status": "completed",
        "train_label_counts": diagnostics_cli._label_counts(train_labels),  # noqa: SLF001
    }


def _fixed_horizon_bandit_scorecard(
    *,
    split: Any,
    labels: np.ndarray,
    profile: tuple[int, int],
    policy_name: str,
    cost_ratio: float,
    alpha: UpstreamAlphaConfig,
) -> dict[str, Any]:
    returns = _fixed_horizon_returns(
        sequences=split.sequences,
        labels=labels,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    trade_mask = labels != 0
    position_notional = alpha.initial_balance * alpha.position_fraction
    pnl = returns * position_notional
    action_counts = {
        "close": int(np.sum(trade_mask)),
        "hold": int(np.sum(labels == 0)),
        "open_long": int(np.sum(labels == 1)),
        "open_short": int(np.sum(labels == 2)),
    }
    by_month = _group_metric_rows(
        labels=[_month_key(value) for value in split.signal_times_utc],
        pnls=pnl,
        label_field="period",
    )
    by_ticker = _group_metric_rows(
        labels=[str(value) for value in split.symbols],
        pnls=pnl,
        label_field="symbol",
    )
    by_volatility = _group_metric_rows(
        labels=_volatility_buckets(split.volatility_scores),
        pnls=pnl,
        label_field="bucket",
    )
    closed_trades = int(np.sum(trade_mask))
    win_rate = 0.0 if closed_trades == 0 else float(np.mean(returns[trade_mask] > 0.0))
    return {
        "action_balance": optuna_cli._action_balance_payload(action_counts),  # noqa: SLF001
        "action_counts": action_counts,
        "closed_trades": closed_trades,
        "mean_net_return_after_costs": _round_float(float(np.mean(returns))),
        "metrics_by_period": by_month,
        "metrics_by_volatility_bucket": by_volatility,
        "net_pnl_after_costs_quote": _round_float(float(np.sum(pnl))),
        "policy_kind": "oracle_upper_bound"
        if policy_name == "oracle_label_upper_bound_not_candidate"
        else "candidate_proxy"
        if policy_name == "supervised_oracle_label_warm_start_contextual_bandit"
        else "baseline",
        "policy_name": policy_name,
        "positive_trade_ratio": _round_float(win_rate),
        "proxy_surface": "fixed_horizon_contextual_bandit_research_only",
        "return_pct_after_costs": _round_float(float(np.sum(pnl) / alpha.initial_balance)),
        "stability_by_ticker": by_ticker,
        "stability_summary": {
            "monthly_dominance": optuna_cli._dominance_payload(  # noqa: SLF001
                rows=by_month,
                label_key="period",
            ),
            "monthly_positive_group_ratio": _round_float(
                optuna_cli._positive_group_ratio(by_month)  # noqa: SLF001
            ),
            "ticker_dominance": optuna_cli._dominance_payload(  # noqa: SLF001
                rows=by_ticker,
                label_key="symbol",
            ),
            "ticker_positive_group_ratio": _round_float(
                optuna_cli._positive_group_ratio(by_ticker)  # noqa: SLF001
            ),
            "volatility_bucket_dominance": optuna_cli._dominance_payload(  # noqa: SLF001
                rows=by_volatility,
                label_key="bucket",
            ),
        },
        "trade_count": closed_trades,
    }


def _fixed_horizon_returns(
    *,
    sequences: np.ndarray,
    labels: np.ndarray,
    profile: tuple[int, int],
    cost_ratio: float,
) -> np.ndarray:
    _, session_len = profile
    close_idx = diagnostics_cli._feature_index("close")  # noqa: SLF001
    start = SESSIONIZED_PRE_SIGNAL_LEN_V1 - 1
    stop = start + session_len
    if stop > sequences.shape[1]:
        raise Stage08LResearchError(reason="profile_session_len_exceeds_sequence")
    entry = np.maximum(sequences[:, start, close_idx].astype(np.float64), 1e-12)
    exit_ = np.maximum(sequences[:, stop - 1, close_idx].astype(np.float64), 1e-12)
    long_return = (exit_ / entry) - 1.0 - cost_ratio
    short_return = (entry / exit_) - 1.0 - cost_ratio
    out = np.zeros(labels.shape[0], dtype=np.float64)
    out[labels == 1] = long_return[labels == 1]
    out[labels == 2] = short_return[labels == 2]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _candidate_path_decision(
    *,
    scorecards: Sequence[Mapping[str, Any]],
    supervised: Mapping[str, Any],
    min_trades: int,
) -> dict[str, Any]:
    scorecard_by_name = {str(item["policy_name"]): item for item in scorecards}
    candidate = scorecard_by_name["supervised_oracle_label_warm_start_contextual_bandit"]
    baselines = [
        item
        for item in scorecards
        if item.get("policy_kind") == "baseline"
    ]
    best_baseline = max(
        float(item.get("net_pnl_after_costs_quote", 0.0)) for item in baselines
    )
    candidate_pnl = float(candidate.get("net_pnl_after_costs_quote", 0.0))
    stability = cast(Mapping[str, Any], candidate["stability_summary"])
    action_balance = cast(Mapping[str, Any], candidate["action_balance"])
    backtest_metrics = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], supervised["splits"])["backtest"],
    )
    ridge_metrics = cast(Mapping[str, Any], backtest_metrics["ridge_past_window_model"])
    recent_metrics = cast(Mapping[str, Any], backtest_metrics["recent_return_baseline"])
    blockers: list[str] = []
    if candidate_pnl <= best_baseline:
        blockers.append("warm_start_bandit_does_not_clear_best_technical_baseline")
    if int(candidate.get("closed_trades", 0)) < min_trades:
        blockers.append("warm_start_bandit_trade_count_below_minimum")
    if _dominance_share(stability, "monthly_dominance") > STAGE08L_DOMINANCE_SHARE_LIMIT:
        blockers.append("warm_start_monthly_dominance_too_high")
    if _dominance_share(stability, "ticker_dominance") > STAGE08L_DOMINANCE_SHARE_LIMIT:
        blockers.append("warm_start_ticker_dominance_too_high")
    if _dominance_share(stability, "volatility_bucket_dominance") > STAGE08L_DOMINANCE_SHARE_LIMIT:
        blockers.append("warm_start_volatility_bucket_dominance_too_high")
    if (
        float(stability.get("monthly_positive_group_ratio", 0.0))
        < STAGE08L_MIN_GROUP_POSITIVE_RATIO
    ):
        blockers.append("warm_start_monthly_stability_broken")
    if float(stability.get("ticker_positive_group_ratio", 0.0)) < STAGE08L_MIN_GROUP_POSITIVE_RATIO:
        blockers.append("warm_start_ticker_stability_broken")
    if float(action_balance.get("open_side_dominance_share", 1.0)) > STAGE08L_MAX_OPEN_SIDE_SHARE:
        blockers.append("warm_start_action_distribution_pathologically_one_sided")
    ridge_balanced = float(ridge_metrics.get("balanced_accuracy") or 0.0)
    recent_balanced = float(recent_metrics.get("balanced_accuracy") or 0.0)
    if ridge_balanced <= recent_balanced:
        blockers.append("warm_start_classifier_does_not_beat_recent_return_baseline")
    allowed = not blockers
    return {
        "best_technical_baseline_net_pnl_after_costs_quote": _round_float(best_baseline),
        "blockers": sorted(set(blockers)),
        "candidate_beats_best_technical_baseline": candidate_pnl > best_baseline,
        "candidate_net_pnl_after_costs_quote": _round_float(candidate_pnl),
        "candidate_path_justified": allowed,
        "decision_reason": "next_corrective_warm_start_candidate_stage_required"
        if allowed
        else "no_reward_or_warm_start_path_accepted",
        "next_stage": "08M-supervised-warm-start-candidate-scorecard" if allowed else None,
        "stage09_allowed": False,
        "status": "accepted" if allowed else "blocked",
    }


def _load_and_validate_i2_matrix(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _matrix_rows(payload, "methodology_discrepancy_matrix")
    surfaces = {str(row.get("surface")) for row in rows}
    missing = sorted(MANDATORY_08I2_SURFACES - surfaces)
    if missing:
        raise Stage08LResearchError(reason="stage08i2_matrix_missing_surface", field=missing[0])
    if payload.get("status") != "blocked" or payload.get("stage09_allowed") is not False:
        raise Stage08LResearchError(reason="stage08i2_matrix_unexpected_status", field=str(path))
    return {
        "path": str(path),
        "sha256": _file_sha256_hex(path),
        "status": payload.get("status"),
        "stage09_allowed": payload.get("stage09_allowed"),
        "surface_count": len(rows),
    }


def _load_and_validate_i4_matrix(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _matrix_rows(payload, "methodology_recheck_matrix")
    surfaces = {str(row.get("surface")) for row in rows}
    missing = sorted(MANDATORY_08I2_SURFACES - surfaces)
    if missing:
        raise Stage08LResearchError(reason="stage08i4_matrix_missing_surface", field=missing[0])
    if payload.get("status") != "accepted" or payload.get("stage09_allowed") is not False:
        raise Stage08LResearchError(reason="stage08i4_matrix_unexpected_status", field=str(path))
    if payload.get("08j_allowed") is not True:
        raise Stage08LResearchError(reason="stage08i4_did_not_allow_08j", field=str(path))
    by_surface = {str(row.get("surface")): row for row in rows}
    if by_surface["full_evaluator_backtest_parity"].get("recheck_disposition") != "closed_by_08i3":
        raise Stage08LResearchError(reason="stage08i4_evaluator_parity_not_closed")
    if by_surface["reward_sparsity_and_semantics"].get("owner_next_stage") != "08K":
        raise Stage08LResearchError(reason="stage08i4_reward_row_not_assigned_to_08k")
    if by_surface["action_q_policy_distribution"].get("owner_next_stage") != "08K":
        raise Stage08LResearchError(reason="stage08i4_action_row_not_assigned_to_08k")
    return {
        "08j_allowed": payload.get("08j_allowed"),
        "08k_allowed": payload.get("08k_allowed"),
        "path": str(path),
        "sha256": _file_sha256_hex(path),
        "stage09_allowed": payload.get("stage09_allowed"),
        "status": payload.get("status"),
        "surface_count": len(rows),
    }


def _load_and_validate_stage08k_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("stage") != "08K":
        raise Stage08LResearchError(reason="stage08k_summary_stage_mismatch", field=str(path))
    if payload.get("stage09_allowed") is not False:
        raise Stage08LResearchError(reason="stage08k_unexpectedly_allowed_stage09", field=str(path))
    branch = _native_stage08k_branch(payload)
    if branch.get("stage09_allowed") is not False:
        raise Stage08LResearchError(reason="stage08k_native_branch_unexpectedly_allowed_stage09")
    return payload


def _native_stage08k_branch(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = payload.get("branches")
    if not isinstance(branches, Mapping):
        raise Stage08LResearchError(reason="stage08k_branches_missing")
    branch = branches.get("roehub_native_article_selector_30_10")
    if not isinstance(branch, Mapping):
        raise Stage08LResearchError(reason="stage08k_native_branch_missing")
    return cast(Mapping[str, Any], branch)


def _stage08k_native_final_manifest_path(payload: Mapping[str, Any]) -> Path:
    branch = _native_stage08k_branch(payload)
    optuna_summary_path = branch.get("optuna_summary_path")
    if not isinstance(optuna_summary_path, str) or not optuna_summary_path:
        raise Stage08LResearchError(reason="stage08k_native_optuna_summary_missing")
    optuna_summary = _read_json(Path(optuna_summary_path))
    final_path = optuna_summary.get("final_evaluation_manifest_path")
    if not isinstance(final_path, str) or not final_path:
        raise Stage08LResearchError(reason="stage08k_final_manifest_missing")
    return Path(final_path)


def _strict_gate_snapshot(*, stage08k_summary: Mapping[str, Any]) -> dict[str, Any]:
    final_manifest = _read_json(_stage08k_native_final_manifest_path(stage08k_summary))
    optuna_summary = _read_json(
        Path(str(_native_stage08k_branch(stage08k_summary)["optuna_summary_path"]))
    )
    gate = optuna_summary.get("final_strict_research_gate")
    return {
        "final_evaluation_manifest_path": optuna_summary.get("final_evaluation_manifest_path"),
        "final_evaluation_manifest_sha256": _file_sha256_hex(
            Path(str(optuna_summary.get("final_evaluation_manifest_path")))
        ),
        "final_strict_research_gate": gate if isinstance(gate, Mapping) else {},
        "scorecard_count": len(final_manifest.get("scorecards", []))
        if isinstance(final_manifest.get("scorecards"), list)
        else None,
    }


def _matrix_rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise Stage08LResearchError(reason="matrix_rows_missing", field=key)
    parsed = [cast(Mapping[str, Any], item) for item in rows if isinstance(item, Mapping)]
    if len(parsed) != 8:
        raise Stage08LResearchError(reason="matrix_row_count_mismatch", field=key)
    return parsed


def _bounded_experiment_matrix(
    *,
    args: argparse.Namespace,
    profile: tuple[int, int],
    output_root: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    expected_path = output_root / run_id / "stage08l_reward_warm_start_research_summary.json"
    return [
        {
            "dataset_branch": "roehub_native_article_selector_30_10",
            "expected_artifact_path": str(expected_path),
            "hypothesis": (
                "Stage 02C realized-PnL reward is the baseline; dense/shaped reward "
                "is research-only."
            ),
            "max_runtime": "bounded numpy diagnostics only; no Torch training and no Optuna",
            "metrics": [
                "current_reward_non_zero_trade_step_ratio_proxy",
                "dense_mark_to_market_non_zero_step_ratio_proxy",
            ],
            "profile": f"{profile[0]}/{profile[1]}",
            "stop_conditions": ["input_matrix_missing", "stage08k_not_blocked"],
        },
        {
            "dataset_branch": "roehub_native_article_selector_30_10",
            "expected_artifact_path": str(expected_path),
            "hypothesis": (
                "A past-window supervised oracle-label warm start should beat simple "
                "technical baselines before a new candidate stage is justified."
            ),
            "max_runtime": "closed-form ridge classifier on loaded Stage 08J arrays",
            "metrics": [
                "balanced_accuracy",
                "prediction_counts",
                "fixed_horizon_proxy_pnl",
            ],
            "profile": f"{profile[0]}/{profile[1]}",
            "stop_conditions": [
                "classifier_does_not_beat_recent_return_baseline",
                "proxy_pnl_does_not_clear_best_baseline",
            ],
        },
        {
            "dataset_branch": "roehub_native_article_selector_30_10",
            "expected_artifact_path": str(expected_path),
            "hypothesis": (
                "A contextual-bandit sanity proxy must be stable across month/ticker/"
                "volatility groups before it can justify a bounded candidate prompt."
            ),
            "max_runtime": "single fixed-horizon proxy pass on untouched backtest split",
            "metrics": [
                "monthly_dominance",
                "ticker_positive_group_ratio",
                "volatility_bucket_dominance",
                "action_balance",
            ],
            "profile": f"{profile[0]}/{profile[1]}",
            "stop_conditions": [
                "single_group_dominates_proxy_result",
                "ticker_stability_broken",
                "action_distribution_pathologically_one_sided",
            ],
        },
    ]


def _group_metric_rows(
    *,
    labels: Sequence[str],
    pnls: np.ndarray,
    label_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for label, pnl in zip(labels, pnls, strict=False):
        grouped.setdefault(label, []).append(float(pnl))
    rows = []
    for label in sorted(grouped):
        values = np.asarray(grouped[label], dtype=np.float64)
        rows.append(
            {
                label_field: label,
                "net_pnl_after_costs_quote": _round_float(float(np.sum(values))),
                "positive_ratio": _round_float(float(np.mean(values > 0.0))),
                "session_count": int(values.shape[0]),
            }
        )
    return rows


def _volatility_buckets(values: Sequence[float | None]) -> list[str]:
    parsed = np.asarray([0.0 if value is None else float(value) for value in values])
    if parsed.size == 0:
        return []
    q33, q66 = np.quantile(parsed, [0.3333333333, 0.6666666667])
    out = []
    for value in parsed:
        if value <= q33:
            out.append("low")
        elif value <= q66:
            out.append("medium")
        else:
            out.append("high")
    return out


def _deterministic_random_labels(*, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=count, dtype=np.int8)


def _month_key(value: object) -> str:
    text = "" if value is None else str(value)
    return text[:7] if len(text) >= 7 else "unknown"


def _dominance_share(payload: Mapping[str, Any], key: str) -> float:
    item = payload.get(key)
    if not isinstance(item, Mapping):
        return 1.0
    return float(item.get("dominance_share", 1.0))


def _without_predictions(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "split_predictions"
    }


def _default_run_id(
    *,
    args: argparse.Namespace,
    backtest_split: Any,
    decision: Mapping[str, Any],
    manifest_sha256: str,
    profile: tuple[int, int],
) -> str:
    digest = hash_json_payload_v1(
        {
            "backtest_split": dict(backtest_split.source_payload),
            "candidate_path_justified": decision["candidate_path_justified"],
            "dataset_version": args.dataset_version,
            "manifest_sha256": manifest_sha256,
            "profile": list(profile),
            "stage": "08L",
        }
    )
    return f"stage08l_reward_warm_start_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    source_paths = (
        "scripts/rl_trading/stage08l_reward_warm_start_research.py",
        "scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py",
        "scripts/rl_trading/stage08g_cpu_optuna_calibration.py",
        "scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py",
        "src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py",
        "src/trading/contexts/rl_trading/domain/sessionized_dataset.py",
    )
    files = []
    for relative in source_paths:
        path = REPO_ROOT / relative
        if path.exists():
            files.append({"path": relative, "sha256": _file_sha256_hex(path)})
    payload: dict[str, object] = {"source_file_hashes": files, "source_paths": list(source_paths)}
    if (REPO_ROOT / ".git").exists():
        try:
            payload["git_head"] = _git_output("rev-parse", "HEAD")
            payload["git_status_short"] = _git_output(
                "status",
                "--short",
                "--",
                *source_paths,
            ).splitlines()
        except Exception as exc:
            payload["git_unavailable_reason"] = type(exc).__name__
    return payload


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _round_float(value: float) -> float:
    return float(round(value, 10))


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 08L reward/warm-start research.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument("--stage08i2-matrix-path", type=Path, default=DEFAULT_STAGE08I2_MATRIX_PATH)
    parser.add_argument("--stage08i4-matrix-path", type=Path, default=DEFAULT_STAGE08I4_MATRIX_PATH)
    parser.add_argument("--stage08k-summary-path", type=Path, default=DEFAULT_STAGE08K_SUMMARY_PATH)
    parser.add_argument(
        "--stage08j-manifest-path",
        type=Path,
        default=optuna_cli.DEFAULT_STAGE08J_MANIFEST_PATH,
    )
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--max-train-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-eval-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-train-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--max-eval-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--min-closed-trades", type=int, default=100)
    parser.add_argument("--deterministic-random-seed", type=int, default=812)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
