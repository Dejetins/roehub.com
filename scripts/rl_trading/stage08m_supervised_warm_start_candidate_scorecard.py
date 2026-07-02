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


def _resolve_repo_root() -> Path:
    candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / ".codex").exists() and (candidate / "src").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
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
from scripts.rl_trading import (  # noqa: E402
    stage08l_reward_warm_start_research as stage08l_cli,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    UpstreamAlphaConfig,
    hash_json_payload_v1,
)

STAGE08M_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08m_supervised_warm_start_candidate_scorecard_v1"
STAGE08M_SCHEMA_VERSION_V1 = 1
STAGE08M_ARTIFACT_KIND_V1 = "rl_trading_stage08m_supervised_warm_start_candidate_scorecard"
STAGE08M_CANDIDATE_ARTIFACT_KIND_V1 = "rl_trading_stage08m_supervised_warm_start_candidate"
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08M_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_STAGE08L_SUMMARY_PATH = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08l_reward_warm_start_research_v1/"
    "stage08l_reward_warm_start_99a00ffa43c83b9ac553/"
    "stage08l_reward_warm_start_research_summary.json"
)
DEFAULT_STAGE08L_SUMMARY_SHA256 = (
    "5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1"
)
DEFAULT_STAGE08L_SUMMARY_HASH = (
    "59bdb534baa97bd172266edb4405774ecc12e2005900386ce4d4bae479f28216"
)
DEFAULT_PROMPT_PATH = (
    REPO_ROOT
    / ".codex/agents/generated/rl-trading-agent-platform-v1/"
    "08m-supervised-warm-start-candidate-scorecard.md"
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_PROFILE = "30/10"
STAGE08M_CANDIDATE_POLICY_NAME = "supervised_oracle_label_warm_start_contextual_bandit"


class Stage08MScorecardError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except Stage08MScorecardError as exc:
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
    profile = diagnostics_cli._parse_profile(args.profile)  # noqa: SLF001
    bounded_matrix = _bounded_candidate_matrix(args=args, profile=profile)
    stage08l_summary = _load_and_validate_stage08l_summary(
        path=args.stage08l_summary_path,
        expected_file_sha256=args.expected_stage08l_summary_sha256,
        expected_summary_hash=args.expected_stage08l_summary_hash,
    )
    stage08k_summary = stage08l_cli._load_and_validate_stage08k_summary(  # noqa: SLF001
        args.stage08k_summary_path
    )
    stage08k_final_manifest_path = stage08l_cli._stage08k_native_final_manifest_path(  # noqa: SLF001
        stage08k_summary
    )
    stage08k_final_manifest = _read_json(stage08k_final_manifest_path)
    stage08k_final_scorecard = optuna_cli._candidate_scorecard(  # noqa: SLF001
        stage08k_final_manifest,
        branch="roehub_native",
    )
    stage08k_gate = stage08l_cli._strict_gate_snapshot(  # noqa: SLF001
        stage08k_summary=stage08k_summary
    )

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

    alpha = UpstreamAlphaConfig()
    cost_ratio = 2.0 * (alpha.transaction_fee + alpha.slippage)
    supervised = _fit_supervised_candidate(
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
    random_predictions = stage08l_cli._deterministic_random_labels(  # noqa: SLF001
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
        _stage08m_scorecard(
            split=backtest_split,
            labels=np.zeros(backtest_split.sequences.shape[0], dtype=np.int8),
            profile=profile,
            policy_name="hold_no_trade",
            cost_ratio=cost_ratio,
            alpha=alpha,
            policy_kind="baseline",
        ),
        _stage08m_scorecard(
            split=backtest_split,
            labels=random_predictions,
            profile=profile,
            policy_name="deterministic_random_contextual_bandit",
            cost_ratio=cost_ratio,
            alpha=alpha,
            policy_kind="baseline",
        ),
        _stage08m_scorecard(
            split=backtest_split,
            labels=simple_predictions,
            profile=profile,
            policy_name="simple_recent_return_threshold_contextual_bandit",
            cost_ratio=cost_ratio,
            alpha=alpha,
            policy_kind="baseline",
        ),
        _stage08m_scorecard(
            split=backtest_split,
            labels=backtest_predictions,
            profile=profile,
            policy_name=STAGE08M_CANDIDATE_POLICY_NAME,
            cost_ratio=cost_ratio,
            alpha=alpha,
            policy_kind="candidate",
        ),
        _stage08m_scorecard(
            split=backtest_split,
            labels=oracle_labels,
            profile=profile,
            policy_name="oracle_label_upper_bound_not_candidate",
            cost_ratio=cost_ratio,
            alpha=alpha,
            policy_kind="diagnostic",
        ),
    ]
    final_scorecard = _scorecard_by_name(scorecards, STAGE08M_CANDIDATE_POLICY_NAME)
    final_manifest = {
        "artifact_kind": "rl_trading_stage08m_final_holdout_scorecards",
        "scorecards": scorecards,
        "stage": "08M",
    }
    gate_args = argparse.Namespace(
        stage_label="08K",
        min_calibration_closed_trades=args.min_closed_trades,
    )
    strict_gate = optuna_cli._final_holdout_gate(  # noqa: SLF001
        args=gate_args,
        branch="roehub_native",
        final_scorecard=final_scorecard,
        final_manifest=final_manifest,
    )
    stage08k_reference = _stage08k_reference_payload(stage08k_final_scorecard, stage08k_gate)
    run_id = args.run_id or _default_run_id(
        args=args,
        backtest_split=backtest_split,
        final_scorecard=final_scorecard,
        manifest_sha256=manifest_sha256,
        model_state_hash=str(supervised["model_state_hash"]),
        profile=profile,
        strict_gate=strict_gate,
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = _file_sha256_hex(args.prompt_path)
    candidate_manifest = _candidate_manifest_payload(
        args=args,
        backtest_split=backtest_split,
        generated=generated,
        manifest_sha256=manifest_sha256,
        profile=profile,
        run_dir=run_dir,
        run_id=run_id,
        stage08l_summary=stage08l_summary,
        supervised=supervised,
        strict_gate=strict_gate,
    )
    candidate_manifest_path = run_dir / "stage08m_supervised_warm_start_candidate_manifest.json"
    _atomic_write_json(candidate_manifest_path, candidate_manifest)
    candidate_manifest_sha256 = _file_sha256_hex(candidate_manifest_path)
    summary_payload = {
        "artifact_kind": STAGE08M_ARTIFACT_KIND_V1,
        "bounded_candidate_matrix": bounded_matrix,
        "candidate_artifact": {
            "candidate_id": candidate_manifest["candidate_id"],
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": candidate_manifest_sha256,
            "model_state_hash": supervised["model_state_hash"],
            "policy_name": STAGE08M_CANDIDATE_POLICY_NAME,
        },
        "code_version": _source_state_payload(),
        "comparison": {
            "final_holdout_scorecards": scorecards,
            "stage08k_native_candidate_reference": stage08k_reference,
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
            "stage08k_summary_path": str(args.stage08k_summary_path),
            "stage08k_summary_sha256": _file_sha256_hex(args.stage08k_summary_path),
            "stage08l_summary": stage08l_summary,
            "test_split": dict(test_split.source_payload),
            "train_split": dict(train_split.source_payload),
        },
        "delivery_state": (
            "target_host_readiness_pre_main with Mac Studio non-production "
            "candidate scorecard artifact; no registry promotion, paper/testnet/live/"
            "mainnet, browser/auth, or exchange side effect"
        ),
        "final_holdout_gate": strict_gate,
        "generated_at_utc": _format_utc(generated),
        "methodology": {
            "analysis_depth": "research_candidate_scorecard",
            "baseline_reward_contract": (
                "Stage 02C realized PnL / initial balance minus flat-hold penalty"
            ),
            "calibration_or_test_split_optimized_final": False,
            "decision_unit": "roehub_native_stage08j_article_selector_session",
            "final_holdout_optimized": False,
            "model": "closed_form_ridge_classifier_numpy",
            "profile": {"agent_history_len": profile[0], "agent_session_len": profile[1]},
            "stage": "08M",
            "training_split_only_for_model_fit": True,
        },
        "next_stage": "09-model-registry-activation" if strict_gate["stage09_allowed"] else None,
        "prompt": {"path": str(args.prompt_path.relative_to(REPO_ROOT)), "sha256": prompt_sha256},
        "proof_boundary": "target_host_readiness_pre_main",
        "proof_subtype": "target_host_non_production_candidate_scorecard_pre_main",
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
        "schema_version": STAGE08M_SCHEMA_VERSION_V1,
        "stage": "08M",
        "stage09_allowed": strict_gate["stage09_allowed"],
        "status": "accepted" if strict_gate["stage09_allowed"] else "blocked",
    }
    summary = {**summary_payload, "summary_hash": hash_json_payload_v1(summary_payload)}
    summary_path = run_dir / "stage08m_supervised_warm_start_candidate_scorecard_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "candidate_id": candidate_manifest["candidate_id"],
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "stage09_allowed": strict_gate["stage09_allowed"],
        "status": summary["status"],
        "strict_gate_blockers": strict_gate["blockers"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256_hex(summary_path),
    }


def _fit_supervised_candidate(
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
    model_state = {
        "algorithm": "closed_form_ridge_classifier_numpy",
        "feature_count": int(train_features.shape[1]),
        "label_order": {"0": "hold", "1": "open_long", "2": "open_short"},
        "scaler_mean": _float_list(scaler_mean),
        "scaler_std": _float_list(scaler_std),
        "weights": _float_matrix(weights),
    }
    model_state_hash = hash_json_payload_v1(model_state)
    return {
        "feature_count": int(train_features.shape[1]),
        "model": "closed_form_ridge_classifier_numpy",
        "model_state": model_state,
        "model_state_hash": model_state_hash,
        "split_predictions": split_predictions,
        "splits": split_payloads,
        "status": "completed",
        "train_label_counts": diagnostics_cli._label_counts(train_labels),  # noqa: SLF001
    }


def _stage08m_scorecard(
    *,
    split: Any,
    labels: np.ndarray,
    profile: tuple[int, int],
    policy_name: str,
    cost_ratio: float,
    alpha: UpstreamAlphaConfig,
    policy_kind: str,
) -> dict[str, Any]:
    scorecard = stage08l_cli._fixed_horizon_bandit_scorecard(  # noqa: SLF001
        split=split,
        labels=labels,
        profile=profile,
        policy_name=policy_name,
        cost_ratio=cost_ratio,
        alpha=alpha,
    )
    scorecard["policy_kind"] = policy_kind
    scorecard.pop("proxy_surface", None)
    scorecard["scorecard_surface"] = "final_holdout_contextual_bandit_candidate_scorecard"
    scorecard["split"] = split.split_name
    return scorecard


def _load_and_validate_stage08l_summary(
    *,
    path: Path,
    expected_file_sha256: str | None,
    expected_summary_hash: str | None,
) -> dict[str, Any]:
    payload = _read_json(path)
    actual_sha256 = _file_sha256_hex(path)
    if expected_file_sha256 and actual_sha256 != expected_file_sha256:
        raise Stage08MScorecardError(reason="stage08l_summary_sha256_mismatch", field=str(path))
    summary_hash = payload.get("summary_hash")
    if expected_summary_hash and summary_hash != expected_summary_hash:
        raise Stage08MScorecardError(reason="stage08l_summary_hash_mismatch", field=str(path))
    if payload.get("stage") != "08L" or payload.get("status") != "accepted":
        raise Stage08MScorecardError(reason="stage08l_summary_not_accepted", field=str(path))
    if payload.get("stage09_allowed") is not False:
        raise Stage08MScorecardError(reason="stage08l_unexpected_stage09_allowed", field=str(path))
    if payload.get("contract_marker") != "reward_research_not_contract_replacement":
        raise Stage08MScorecardError(reason="stage08l_contract_marker_missing", field=str(path))
    decision = payload.get("candidate_path_decision")
    if not isinstance(decision, Mapping) or decision.get("candidate_path_justified") is not True:
        raise Stage08MScorecardError(
            reason="stage08l_candidate_path_not_justified",
            field=str(path),
        )
    return {
        "candidate_path_justified": decision.get("candidate_path_justified"),
        "path": str(path),
        "sha256": actual_sha256,
        "stage09_allowed": payload.get("stage09_allowed"),
        "status": payload.get("status"),
        "summary_hash": summary_hash,
    }


def _candidate_manifest_payload(
    *,
    args: argparse.Namespace,
    backtest_split: Any,
    generated: datetime,
    manifest_sha256: str,
    profile: tuple[int, int],
    run_dir: Path,
    run_id: str,
    stage08l_summary: Mapping[str, Any],
    supervised: Mapping[str, Any],
    strict_gate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = f"stage08m_{str(supervised['model_state_hash'])[:16]}_{manifest_sha256[:8]}"
    return {
        "artifact_kind": STAGE08M_CANDIDATE_ARTIFACT_KIND_V1,
        "candidate_id": candidate_id,
        "contract_marker": "reward_research_not_contract_replacement",
        "data_lineage": {
            "article_manifest_path": str(args.stage08j_manifest_path),
            "article_manifest_sha256": manifest_sha256,
            "backtest_split": dict(backtest_split.source_payload),
            "dataset_version": args.dataset_version,
            "stage08l_summary": dict(stage08l_summary),
        },
        "final_holdout_gate": strict_gate,
        "generated_at_utc": _format_utc(generated),
        "model_state": supervised["model_state"],
        "model_state_hash": supervised["model_state_hash"],
        "policy_name": STAGE08M_CANDIDATE_POLICY_NAME,
        "profile": {"agent_history_len": profile[0], "agent_session_len": profile[1]},
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": {
            "exchange_side_effects": False,
            "model_registry_write": False,
            "paper_testnet_live_enabled": False,
            "promotion_or_activation": False,
            "stage02c_reward_contract_replaced": False,
        },
        "stage": "08M",
        "stage09_allowed": strict_gate["stage09_allowed"],
        "status": "accepted_candidate" if strict_gate["stage09_allowed"] else "blocked_candidate",
    }


def _stage08k_reference_payload(
    final_scorecard: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    strict_gate = gate.get("final_strict_research_gate")
    return {
        "closed_trades": final_scorecard.get("closed_trades"),
        "final_evaluation_manifest_path": gate.get("final_evaluation_manifest_path"),
        "final_evaluation_manifest_sha256": gate.get("final_evaluation_manifest_sha256"),
        "net_pnl_after_costs_quote": final_scorecard.get("net_pnl_after_costs_quote"),
        "policy_name": final_scorecard.get("policy_name"),
        "return_pct_after_costs": final_scorecard.get("return_pct_after_costs"),
        "strict_gate": strict_gate if isinstance(strict_gate, Mapping) else {},
        "win_rate": final_scorecard.get("win_rate"),
    }


def _bounded_candidate_matrix(
    *,
    args: argparse.Namespace,
    profile: tuple[int, int],
) -> list[dict[str, Any]]:
    expected_path_template = str(
        args.output_root
        / "<run_id>"
        / "stage08m_supervised_warm_start_candidate_scorecard_summary.json"
    )
    return [
        {
            "dataset_branch": "roehub_native_article_selector_30_10",
            "expected_artifact_path": expected_path_template,
            "final_holdout_gate": [
                "positive final PnL after costs",
                "beats best sanity baseline",
                "closed_trades >= min_closed_trades",
                "dominance shares <= 0.8",
                "monthly/ticker positive group ratio >= 0.25",
                "open-side dominance <= 0.95",
            ],
            "implementation_path": (
                "closed-form ridge classifier warm-start over past-window features; "
                "no DQN retraining, no Optuna, no registry write"
            ),
            "max_runtime": "bounded NumPy fit plus one final holdout scorecard pass",
            "metrics": [
                "balanced_accuracy on train/test/backtest",
                "final_holdout_net_pnl_after_costs",
                "best_sanity_baseline_delta",
                "closed_trades",
                "monthly/ticker/volatility dominance",
                "action balance",
            ],
            "profile": f"{profile[0]}/{profile[1]}",
            "stop_conditions": [
                "stage08l_summary_hash_mismatch",
                "article_dataset_missing_or_not_accepted",
                "candidate_does_not_clear_strict_native_gate",
            ],
        }
    ]


def _default_run_id(
    *,
    args: argparse.Namespace,
    backtest_split: Any,
    final_scorecard: Mapping[str, Any],
    manifest_sha256: str,
    model_state_hash: str,
    profile: tuple[int, int],
    strict_gate: Mapping[str, Any],
) -> str:
    digest = hash_json_payload_v1(
        {
            "backtest_split": dict(backtest_split.source_payload),
            "candidate_pnl": final_scorecard.get("net_pnl_after_costs_quote"),
            "dataset_version": args.dataset_version,
            "manifest_sha256": manifest_sha256,
            "model_state_hash": model_state_hash,
            "profile": list(profile),
            "stage": "08M",
            "stage09_allowed": strict_gate.get("stage09_allowed"),
        }
    )
    return f"stage08m_supervised_warm_start_{digest[:20]}"


def _scorecard_by_name(
    scorecards: Sequence[Mapping[str, Any]],
    policy_name: str,
) -> Mapping[str, Any]:
    for scorecard in scorecards:
        if scorecard.get("policy_name") == policy_name:
            return scorecard
    raise Stage08MScorecardError(reason="scorecard_missing", field=policy_name)


def _without_predictions(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "split_predictions"}
    }


def _source_state_payload() -> dict[str, object]:
    source_paths = (
        "scripts/rl_trading/stage08m_supervised_warm_start_candidate_scorecard.py",
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


def _float_list(values: np.ndarray) -> list[float]:
    return [float(round(float(value), 12)) for value in values.reshape(-1)]


def _float_matrix(values: np.ndarray) -> list[list[float]]:
    return [
        [float(round(float(value), 12)) for value in row]
        for row in values.reshape(values.shape[0], -1)
    ]


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


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 08M supervised warm-start candidate scorecard."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument("--stage08l-summary-path", type=Path, default=DEFAULT_STAGE08L_SUMMARY_PATH)
    parser.add_argument(
        "--expected-stage08l-summary-sha256",
        type=str,
        default=DEFAULT_STAGE08L_SUMMARY_SHA256,
    )
    parser.add_argument(
        "--expected-stage08l-summary-hash",
        type=str,
        default=DEFAULT_STAGE08L_SUMMARY_HASH,
    )
    parser.add_argument(
        "--stage08k-summary-path",
        type=Path,
        default=stage08l_cli.DEFAULT_STAGE08K_SUMMARY_PATH,
    )
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
