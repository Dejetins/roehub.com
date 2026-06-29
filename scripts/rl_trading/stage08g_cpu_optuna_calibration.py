from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.rl_trading import (  # noqa: E402
    stage08d_original_hf_backtest_evaluation as hf_eval_cli,
)
from scripts.rl_trading import (  # noqa: E402
    stage08f_roehub_native_backtest_evaluation as native_eval_cli,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    HfOriginalEvaluationConfig,
    HfOriginalEvaluationError,
    HfOriginalSplitData,
    RoehubNativeEvaluationConfig,
    RoehubNativeEvaluationError,
    RoehubNativeSplitData,
    UpstreamAlphaConfig,
    hash_json_payload_v1,
    run_stage08d_hf_original_evaluation_v1,
    run_stage08f_roehub_native_evaluation_v1,
)

STAGE08G_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08g_dual_branch_cpu_optuna_training_evaluation_v1"

DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08G_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_TORCH_NUM_THREADS = max(1, os.cpu_count() or 1)
DEFAULT_MIN_CALIBRATION_CLOSED_TRADES = 100

BranchName = str


class Stage08GOptunaError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except (Stage08GOptunaError, HfOriginalEvaluationError, RoehubNativeEvaluationError) as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] in {"completed", "accepted_for_research"} else 2


def _run(args: argparse.Namespace) -> dict[str, Any]:
    optuna = _load_optuna()
    stage_label = str(args.stage_label)
    stage_label_lower = stage_label.lower()
    generated = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    candidate_manifest_path = _candidate_manifest_path(args)
    candidate_manifest_sha256 = _file_sha256_hex(candidate_manifest_path)
    expected_sha = _expected_candidate_sha(args)
    if expected_sha and candidate_manifest_sha256 != expected_sha:
        raise Stage08GOptunaError(
            reason="candidate_manifest_sha256_mismatch",
            field=str(candidate_manifest_path),
        )
    candidate_manifest = _read_json(candidate_manifest_path)
    branch = str(args.branch)
    calibration_split, final_split = _load_branch_splits(args)
    run_id = args.run_id or _default_run_id(
        branch=branch,
        candidate_manifest_sha256=candidate_manifest_sha256,
        calibration_split=calibration_split,
        final_split=final_split,
        args=args,
    )
    run_dir = args.output_root / run_id
    trials_root = run_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    study_storage = f"sqlite:///{run_dir / 'optuna.db'}"
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            warn_independent_sampling=False,
            seed=args.optuna_seed,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=2),
        study_name=f"stage{stage_label_lower}_{branch}_{generated.strftime('%Y%m%dT%H%M%SZ')}",
        storage=study_storage,
        load_if_exists=False,
    )
    trial_records: list[dict[str, Any]] = []

    def objective(trial: Any) -> tuple[float, float]:
        alpha = _trial_alpha(args=args, trial=trial)
        trial_run_id = f"trial_{trial.number:05d}_{alpha.config_hash()[:12]}"
        manifest = _run_branch_evaluation(
            branch=branch,
            candidate_manifest=candidate_manifest,
            candidate_manifest_path=candidate_manifest_path,
            candidate_manifest_sha256=candidate_manifest_sha256,
            test_split=calibration_split,
            backtest_split=calibration_split,
            output_root=trials_root,
            run_id=trial_run_id,
            alpha=alpha,
            args=args,
            generated_at_utc=generated,
        )
        scorecard = _candidate_scorecard(manifest, branch=branch)
        values = _objective_values(scorecard)
        best_baseline_pnl = _best_baseline_net_pnl(manifest)
        candidate_pnl = float(scorecard.get("net_pnl_after_costs_quote", 0.0))
        record = {
            "alpha_config_hash": alpha.config_hash(),
            "evaluation_hash": manifest["evaluation_hash"],
            "evaluation_manifest_path": manifest["evaluation_manifest_path"],
            "evaluation_manifest_sha256": _file_sha256_hex(
                Path(str(manifest["evaluation_manifest_path"]))
            ),
            "objective_values": values,
            "params": dict(trial.params),
            "run_id": trial_run_id,
            "scorecard": {
                "closed_trades": scorecard.get("closed_trades"),
                "candidate_beats_best_sanity_baseline": candidate_pnl > best_baseline_pnl,
                "max_drawdown_pct": scorecard.get("max_drawdown_pct"),
                "net_pnl_after_costs_quote": scorecard.get("net_pnl_after_costs_quote"),
                "return_pct_after_costs": scorecard.get("return_pct_after_costs"),
                "best_baseline_net_pnl_after_costs_quote": best_baseline_pnl,
                "baseline_delta_net_pnl_after_costs_quote": candidate_pnl
                - best_baseline_pnl,
                "win_rate": scorecard.get("win_rate"),
            },
            "trial_number": trial.number,
        }
        trial_records.append(record)
        for key, value in record["scorecard"].items():
            trial.set_user_attr(str(key), value)
        trial.set_user_attr("evaluation_manifest_path", record["evaluation_manifest_path"])
        return values

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=False)
    completed = [
        trial
        for trial in study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        if trial.values is not None
    ]
    if not completed:
        raise Stage08GOptunaError(reason="optuna_no_completed_trials")
    best_trial = _select_best_trial(
        completed_trials=completed,
        trial_records=trial_records,
        min_closed_trades=args.min_calibration_closed_trades,
    )
    best_alpha = _alpha_from_params(args=args, params=dict(best_trial.params))
    final_manifest = _run_branch_evaluation(
        branch=branch,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_manifest_sha256,
        test_split=calibration_split,
        backtest_split=final_split,
        output_root=run_dir,
        run_id=f"final_holdout_{best_alpha.config_hash()[:12]}",
        alpha=best_alpha,
        args=args,
        generated_at_utc=generated,
    )
    summary = _summary_payload(
        args=args,
        branch=branch,
        run_id=run_id,
        run_dir=run_dir,
        generated_at_utc=generated,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_manifest_sha256,
        calibration_split=calibration_split,
        final_split=final_split,
        study_storage=study_storage,
        best_trial=best_trial,
        best_alpha=best_alpha,
        trial_records=trial_records,
        final_manifest=final_manifest,
    )
    summary_path = run_dir / f"stage{stage_label_lower}_optuna_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "best_trial_number": best_trial.number,
        "branch": branch,
        "final_evaluation_manifest_path": final_manifest["evaluation_manifest_path"],
        "run_dir": str(run_dir),
        "run_id": run_id,
        "status": summary["status"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256_hex(summary_path),
    }


def _load_optuna() -> Any:
    try:
        return importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise Stage08GOptunaError(reason="optuna_dependency_missing", field="optuna") from exc


def _candidate_manifest_path(args: argparse.Namespace) -> Path:
    if args.candidate_manifest is not None:
        return cast(Path, args.candidate_manifest)
    if args.branch == "hf_original":
        return hf_eval_cli.DEFAULT_CANDIDATE_MANIFEST
    if args.branch == "roehub_native":
        return native_eval_cli.DEFAULT_CANDIDATE_MANIFEST
    raise Stage08GOptunaError(reason="unsupported_branch", field=str(args.branch))


def _expected_candidate_sha(args: argparse.Namespace) -> str | None:
    if args.expected_candidate_manifest_sha256:
        return str(args.expected_candidate_manifest_sha256)
    if args.candidate_manifest is not None:
        return None
    if args.branch == "hf_original":
        return hf_eval_cli.DEFAULT_CANDIDATE_MANIFEST_SHA256
    if args.branch == "roehub_native":
        return native_eval_cli.DEFAULT_CANDIDATE_MANIFEST_SHA256
    return None


def _load_branch_splits(
    args: argparse.Namespace,
) -> tuple[
    HfOriginalSplitData | RoehubNativeSplitData,
    HfOriginalSplitData | RoehubNativeSplitData,
]:
    if args.branch == "hf_original":
        specs = {spec.split_name: spec for spec in hf_eval_cli.expected_hf_split_specs_v1()}
        calibration = hf_eval_cli._load_hf_split(
            dataset_dir=args.hf_dataset_dir,
            split_spec=specs[args.hf_calibration_split],
            max_sessions=args.max_calibration_sessions,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        final = hf_eval_cli._load_hf_split(
            dataset_dir=args.hf_dataset_dir,
            split_spec=specs[args.hf_final_split],
            max_sessions=args.max_final_sessions,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        return calibration, final
    if args.branch == "roehub_native":
        manifest = native_eval_cli._read_json(args.stage06_manifest_path)
        manifest_sha256 = native_eval_cli.compute_file_sha256(args.stage06_manifest_path)
        calibration = native_eval_cli._load_stage06_split(
            manifest=manifest,
            manifest_path=args.stage06_manifest_path,
            manifest_sha256=manifest_sha256,
            dataset_version=args.dataset_version,
            split=args.native_calibration_split,
            max_sessions=args.max_calibration_sessions,
            max_artifacts=args.max_calibration_artifacts,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        final = native_eval_cli._load_stage06_split(
            manifest=manifest,
            manifest_path=args.stage06_manifest_path,
            manifest_sha256=manifest_sha256,
            dataset_version=args.dataset_version,
            split=args.native_final_split,
            max_sessions=args.max_final_sessions,
            max_artifacts=args.max_final_artifacts,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        return calibration, final
    raise Stage08GOptunaError(reason="unsupported_branch", field=str(args.branch))


def _trial_alpha(*, args: argparse.Namespace, trial: Any) -> UpstreamAlphaConfig:
    use_risk_management = bool(trial.suggest_categorical("use_rm", [True, False]))
    stop_loss = 0.0
    take_profit = 0.0
    trailing_stop = 0.0
    if use_risk_management:
        stop_loss = float(trial.suggest_float("stop_loss", 0.005, 0.03))
        take_profit = float(trial.suggest_float("take_profit", 0.01, 0.05))
        trailing_stop = float(trial.suggest_float("trail", 0.001, 0.02))
    ensemble_max_sigma = args.ensemble_max_sigma
    if args.selection_strategy == "ensemble_q_filter":
        ensemble_max_sigma = float(trial.suggest_float("max_sigma", 0.001, 0.015, log=True))
    return UpstreamAlphaConfig(
        agent_history_len=args.agent_history_len,
        agent_session_len=args.agent_session_len,
        long_action_threshold=float(trial.suggest_float("long_thr", 0.001, 0.03, log=True)),
        short_action_threshold=float(trial.suggest_float("short_thr", 0.001, 0.03, log=True)),
        close_action_threshold=float(trial.suggest_float("close_thr", 0.001, 0.03, log=True)),
        use_risk_management=use_risk_management,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        ensemble_n_samples=args.ensemble_n_samples,
        ensemble_max_sigma=ensemble_max_sigma,
        max_parallel_sessions=args.max_parallel_sessions,
        position_fraction=args.position_fraction,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )


def _alpha_from_params(
    *,
    args: argparse.Namespace,
    params: Mapping[str, Any],
) -> UpstreamAlphaConfig:
    use_risk_management = bool(params.get("use_rm", False))
    ensemble_max_sigma = args.ensemble_max_sigma
    if args.selection_strategy == "ensemble_q_filter" and "max_sigma" in params:
        ensemble_max_sigma = float(params["max_sigma"])
    return UpstreamAlphaConfig(
        agent_history_len=args.agent_history_len,
        agent_session_len=args.agent_session_len,
        long_action_threshold=float(params["long_thr"]),
        short_action_threshold=float(params["short_thr"]),
        close_action_threshold=float(params["close_thr"]),
        use_risk_management=use_risk_management,
        stop_loss=float(params.get("stop_loss", 0.0 if not use_risk_management else 0.01)),
        take_profit=float(params.get("take_profit", 0.0 if not use_risk_management else 0.02)),
        trailing_stop=float(params.get("trail", 0.0 if not use_risk_management else 0.005)),
        ensemble_n_samples=args.ensemble_n_samples,
        ensemble_max_sigma=ensemble_max_sigma,
        max_parallel_sessions=args.max_parallel_sessions,
        position_fraction=args.position_fraction,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )


def _run_branch_evaluation(
    *,
    branch: BranchName,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    test_split: HfOriginalSplitData | RoehubNativeSplitData,
    backtest_split: HfOriginalSplitData | RoehubNativeSplitData,
    output_root: Path,
    run_id: str,
    alpha: UpstreamAlphaConfig,
    args: argparse.Namespace,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    if branch == "hf_original":
        return run_stage08d_hf_original_evaluation_v1(
            candidate_manifest=candidate_manifest,
            candidate_manifest_path=candidate_manifest_path,
            candidate_manifest_sha256=candidate_manifest_sha256,
            test_split=cast(HfOriginalSplitData, test_split),
            backtest_split=cast(HfOriginalSplitData, backtest_split),
            output_root=output_root,
            run_id=run_id,
            config=HfOriginalEvaluationConfig(
                alpha=alpha,
                checkpoint_name=args.checkpoint_name,
                selection_strategy=args.selection_strategy,
                device_policy="cpu_only_deterministic",
                test_max_sessions=None,
                backtest_max_sessions=None,
                simple_threshold_return=args.simple_threshold_return,
            ),
            generated_at_utc=generated_at_utc,
            code_version=_source_state_payload(),
        )
    if branch == "roehub_native":
        return run_stage08f_roehub_native_evaluation_v1(
            candidate_manifest=candidate_manifest,
            candidate_manifest_path=candidate_manifest_path,
            candidate_manifest_sha256=candidate_manifest_sha256,
            test_split=cast(RoehubNativeSplitData, test_split),
            backtest_split=cast(RoehubNativeSplitData, backtest_split),
            output_root=output_root,
            run_id=run_id,
            config=RoehubNativeEvaluationConfig(
                alpha=alpha,
                checkpoint_name=args.checkpoint_name,
                selection_strategy=args.selection_strategy,
                device_policy="cpu_only_deterministic",
                test_max_sessions=None,
                backtest_max_sessions=None,
                simple_threshold_return=args.simple_threshold_return,
                deterministic_random_seed=args.deterministic_random_seed,
            ),
            generated_at_utc=generated_at_utc,
            code_version=_source_state_payload(),
        )
    raise Stage08GOptunaError(reason="unsupported_branch", field=branch)


def _candidate_scorecard(manifest: Mapping[str, Any], *, branch: BranchName) -> Mapping[str, Any]:
    expected = (
        "hf_original_candidate_filtered_backtest"
        if branch == "hf_original"
        else "roehub_native_candidate_filtered_backtest"
    )
    scorecards = manifest.get("scorecards")
    if not isinstance(scorecards, list):
        raise Stage08GOptunaError(reason="scorecards_missing")
    for item in scorecards:
        if isinstance(item, Mapping) and item.get("policy_name") == expected:
            return cast(Mapping[str, Any], item)
    raise Stage08GOptunaError(reason="candidate_scorecard_missing", field=expected)


def _objective_values(scorecard: Mapping[str, Any]) -> tuple[float, float]:
    return_pct = float(scorecard.get("return_pct_after_costs", 0.0))
    win_rate = float(scorecard.get("win_rate", 0.0))
    return return_pct, win_rate


def _best_baseline_net_pnl(manifest: Mapping[str, Any]) -> float:
    best = 0.0
    scorecards = manifest.get("scorecards")
    if not isinstance(scorecards, list):
        return best
    for item in scorecards:
        if not isinstance(item, Mapping):
            continue
        if item.get("policy_kind") != "baseline":
            continue
        best = max(best, float(item.get("net_pnl_after_costs_quote", 0.0)))
    return best


def _select_best_trial(
    *,
    completed_trials: list[Any],
    trial_records: list[dict[str, Any]],
    min_closed_trades: int,
) -> Any:
    records_by_number = {
        int(record["trial_number"]): record for record in trial_records if "trial_number" in record
    }
    candidates: list[tuple[tuple[float, float, float, float, float, float], Any]] = []
    for trial in completed_trials:
        record = records_by_number.get(int(trial.number))
        if record is None:
            continue
        scorecard = record.get("scorecard")
        if not isinstance(scorecard, Mapping):
            continue
        closed_trades = float(scorecard.get("closed_trades", 0.0))
        if closed_trades < float(min_closed_trades):
            continue
        return_pct = float(scorecard.get("return_pct_after_costs", 0.0))
        win_rate = float(scorecard.get("win_rate", 0.0))
        max_drawdown_pct = float(scorecard.get("max_drawdown_pct", 0.0))
        baseline_delta = float(
            scorecard.get("baseline_delta_net_pnl_after_costs_quote", 0.0)
        )
        candidate_beats_baseline = (
            1.0 if bool(scorecard.get("candidate_beats_best_sanity_baseline")) else 0.0
        )
        key = (
            return_pct,
            win_rate,
            -max_drawdown_pct,
            candidate_beats_baseline,
            baseline_delta,
            closed_trades,
        )
        candidates.append((key, trial))
    if not candidates:
        raise Stage08GOptunaError(
            reason="optuna_no_trade_sufficient_trials",
            field=f"min_closed_trades={min_closed_trades}",
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _summary_payload(
    *,
    args: argparse.Namespace,
    branch: BranchName,
    run_id: str,
    run_dir: Path,
    generated_at_utc: datetime,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    calibration_split: HfOriginalSplitData | RoehubNativeSplitData,
    final_split: HfOriginalSplitData | RoehubNativeSplitData,
    study_storage: str,
    best_trial: Any,
    best_alpha: UpstreamAlphaConfig,
    trial_records: list[dict[str, Any]],
    final_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    final_scorecard = _candidate_scorecard(final_manifest, branch=branch)
    status = (
        "accepted_for_research"
        if _final_holdout_allows_stage09(final_scorecard)
        else "completed"
    )
    payload = {
        "artifact_kind": f"rl_trading_stage{str(args.stage_label).lower()}_cpu_optuna_summary",
        "best_alpha_config": best_alpha.as_payload(),
        "best_alpha_config_hash": best_alpha.config_hash(),
        "best_trial_number": int(best_trial.number),
        "best_trial_params": dict(best_trial.params),
        "best_trial_values": list(best_trial.values or []),
        "branch": branch,
        "calibration_split": dict(calibration_split.source_payload),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "final_evaluation_hash": final_manifest.get("evaluation_hash"),
        "final_evaluation_manifest_path": final_manifest.get("evaluation_manifest_path"),
        "final_scorecard": {
            "closed_trades": final_scorecard.get("closed_trades"),
            "net_pnl_after_costs_quote": final_scorecard.get("net_pnl_after_costs_quote"),
            "return_pct_after_costs": final_scorecard.get("return_pct_after_costs"),
            "win_rate": final_scorecard.get("win_rate"),
        },
        "final_split": dict(final_split.source_payload),
        "generated_at_utc": _format_utc(generated_at_utc),
        "max_parallel_sessions_decision": {
            "source_default": 2,
            "used_value": args.max_parallel_sessions,
            "optimized_in_this_stage": False,
        },
        "methodology": {
            "calibration_split_used_for_optuna": True,
            "device_policy": "cpu_only_deterministic",
            "final_split_optimized_by_optuna": False,
            "min_calibration_closed_trades": args.min_calibration_closed_trades,
            "position_fraction_optimized_in_this_stage": False,
            "selection_rule": [
                "closed_trades >= min_calibration_closed_trades",
                "max return_pct_after_costs",
                "max win_rate",
                "min max_drawdown_pct",
                "candidate beats best sanity baseline",
                "max baseline_delta_net_pnl_after_costs_quote",
            ],
            "search_space": [
                "long_thr",
                "short_thr",
                "close_thr",
                "use_rm",
                "stop_loss",
                "take_profit",
                "trail",
                "max_sigma only for ensemble_q_filter",
            ],
            "upstream_source_sha": "f71130903f8237351164f4b875494185465bf1ea",
        },
        "run_dir": str(run_dir),
        "run_id": run_id,
        "schema_version": 1,
        "stage": str(args.stage_label),
        "status": status,
        "study_storage": study_storage,
        "trial_count_requested": args.trials,
        "trial_records": trial_records,
    }
    return {**payload, "summary_hash": hash_json_payload_v1(payload)}


def _final_holdout_allows_stage09(scorecard: Mapping[str, Any]) -> bool:
    return float(scorecard.get("net_pnl_after_costs_quote", 0.0)) > 0.0


def _default_run_id(
    *,
    branch: BranchName,
    candidate_manifest_sha256: str,
    calibration_split: HfOriginalSplitData | RoehubNativeSplitData,
    final_split: HfOriginalSplitData | RoehubNativeSplitData,
    args: argparse.Namespace,
) -> str:
    digest = hash_json_payload_v1(
        {
            "branch": branch,
            "calibration_split": dict(calibration_split.source_payload),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "final_split": dict(final_split.source_payload),
            "max_parallel_sessions": args.max_parallel_sessions,
            "position_fraction": args.position_fraction,
            "selection_strategy": args.selection_strategy,
            "stage": args.stage_label,
            "trials": args.trials,
        }
    )
    stage_label = str(args.stage_label).lower()
    candidate_prefix = candidate_manifest_sha256[:8]
    return f"stage{stage_label}_{branch}_{candidate_prefix}_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    source_paths = (
        "src/trading/contexts/rl_trading/domain/hf_original_evaluation.py",
        "src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py",
        "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
        "scripts/rl_trading/stage08g_cpu_optuna_calibration.py",
    )
    files = []
    for relative in source_paths:
        path = REPO_ROOT / relative
        if path.exists():
            files.append({"path": relative, "sha256": _file_sha256_hex(path)})
    return {"source_file_hashes": files, "source_paths": list(source_paths)}


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


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 08G CPU Optuna calibration.")
    parser.add_argument("--branch", choices=("hf_original", "roehub_native"), required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--expected-candidate-manifest-sha256", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--stage-label", choices=("08G", "08H"), default="08G")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--optuna-seed", type=int, default=1708)
    parser.add_argument(
        "--min-calibration-closed-trades",
        type=_non_negative_int,
        default=DEFAULT_MIN_CALIBRATION_CLOSED_TRADES,
    )
    parser.add_argument("--agent-history-len", type=int, default=30)
    parser.add_argument("--agent-session-len", type=int, default=10)
    parser.add_argument("--checkpoint-name", choices=("best", "final"), default="best")
    parser.add_argument(
        "--selection-strategy",
        choices=("advantage_based_filter", "ensemble_q_filter"),
        default="advantage_based_filter",
    )
    parser.add_argument("--simple-threshold-return", type=float, default=0.001)
    parser.add_argument("--ensemble-n-samples", type=int, default=5)
    parser.add_argument("--ensemble-max-sigma", type=float, default=0.01)
    parser.add_argument("--max-parallel-sessions", type=int, default=2)
    parser.add_argument("--position-fraction", type=float, default=0.5)
    parser.add_argument("--torch-num-threads", type=int, default=DEFAULT_TORCH_NUM_THREADS)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--deterministic-random-seed", type=int, default=806)
    parser.add_argument("--generated-at-utc", type=str, default=None)

    parser.add_argument("--hf-dataset-dir", type=Path, default=hf_eval_cli.DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--hf-calibration-split", type=str, default="test")
    parser.add_argument("--hf-final-split", type=str, default="backtest")

    parser.add_argument(
        "--stage06-manifest-path",
        type=Path,
        default=native_eval_cli.DEFAULT_STAGE06_MANIFEST_PATH,
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=native_eval_cli.DEFAULT_DATASET_VERSION,
    )
    parser.add_argument("--native-calibration-split", type=str, default="test")
    parser.add_argument("--native-final-split", type=str, default="backtest")

    parser.add_argument("--max-calibration-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-final-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-calibration-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--max-final-artifacts", type=_optional_positive_int, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
