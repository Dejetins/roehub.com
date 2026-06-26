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
    stage08d_original_hf_backtest_evaluation as hf_eval_cli,
)
from scripts.rl_trading import (  # noqa: E402
    stage08f_roehub_native_backtest_evaluation as native_eval_cli,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    FEATURE_NAMES_V1,
    SESSIONIZED_FULL_SEQ_LEN_V1,
    SESSIONIZED_PRE_SIGNAL_LEN_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    UpstreamAlphaConfig,
    hash_json_payload_v1,
)

STAGE08H_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08h_oracle_supervised_selector_reward_90_60_v1"
STAGE08H_SCHEMA_VERSION_V1 = 1
STAGE08H_ARTIFACT_KIND_V1 = "rl_trading_stage08h_dataset_diagnostics"
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08H_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_BRANCHES = ("hf_original", "roehub_native")
DEFAULT_SPLITS = ("train", "validation", "test", "backtest")
DEFAULT_PROFILES = ("30/10", "90/60")
LABEL_NAMES = ("hold", "long", "short")


class Stage08HDiagnosticsError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except (
        Stage08HDiagnosticsError,
        hf_eval_cli.HfOriginalEvaluationError,
        native_eval_cli.RoehubNativeEvaluationError,
    ) as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    generated = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    branches = _selected_branches(args.branches)
    split_names = _selected_splits(args.splits)
    profiles = [_parse_profile(value) for value in args.profiles]
    run_id = args.run_id or _default_run_id(args=args, branches=branches, profiles=profiles)
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    alpha = UpstreamAlphaConfig()
    cost_model = {
        "round_trip_cost_ratio": 2.0 * (alpha.transaction_fee + alpha.slippage),
        "slippage": alpha.slippage,
        "transaction_fee": alpha.transaction_fee,
    }
    datasets = {
        branch: _load_branch_splits(branch=branch, split_names=split_names, args=args)
        for branch in branches
    }
    branch_results: dict[str, Any] = {}
    for branch, splits in datasets.items():
        branch_results[branch] = _branch_diagnostics(
            branch=branch,
            splits=splits,
            profiles=profiles,
            cost_ratio=float(cost_model["round_trip_cost_ratio"]),
            flat_hold_penalty=alpha.inaction_penalty_ratio,
        )

    payload = {
        "artifact_kind": STAGE08H_ARTIFACT_KIND_V1,
        "branches": branch_results,
        "cost_model": cost_model,
        "generated_at_utc": _format_utc(generated),
        "methodology": {
            "baseline_definition": (
                "Baseline is a simple control policy measured on the same branch, split, "
                "profile and cost model as the candidate diagnostic."
            ),
            "lookahead_policy": (
                "Oracle uses future prices only for diagnostics/labels. Supervised features "
                "use past-only windows ending at pre_signal_len."
            ),
            "profiles": [
                {"agent_history_len": profile[0], "agent_session_len": profile[1]}
                for profile in profiles
            ],
            "stage": "08H",
        },
        "run_dir": str(run_dir),
        "run_id": run_id,
        "schema_version": STAGE08H_SCHEMA_VERSION_V1,
        "source_state": _source_state_payload(),
        "split_names": list(split_names),
        "status": "completed",
    }
    summary = {**payload, "summary_hash": hash_json_payload_v1(payload)}
    summary_path = run_dir / "stage08h_dataset_diagnostics_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "status": summary["status"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256_hex(summary_path),
    }


def _branch_diagnostics(
    *,
    branch: str,
    splits: Mapping[str, Any],
    profiles: Sequence[tuple[int, int]],
    cost_ratio: float,
    flat_hold_penalty: float,
) -> dict[str, Any]:
    profile_results: dict[str, Any] = {}
    for history_len, session_len in profiles:
        profile_key = f"{history_len}/{session_len}"
        split_results = {
            split_name: _split_diagnostics(
                sequences=split.sequences,
                profile=(history_len, session_len),
                cost_ratio=cost_ratio,
                flat_hold_penalty=flat_hold_penalty,
            )
            for split_name, split in splits.items()
        }
        supervised = (
            _supervised_sanity(
                train_sequences=splits["train"].sequences,
                eval_splits={name: split.sequences for name, split in splits.items()},
                profile=(history_len, session_len),
                cost_ratio=cost_ratio,
            )
            if "train" in splits
            else {"status": "skipped", "reason": "train_split_missing"}
        )
        profile_results[profile_key] = {
            "split_diagnostics": split_results,
            "supervised_sanity": supervised,
            "training_profile": {
                "agent_history_len": history_len,
                "agent_session_len": session_len,
                "expected_env_steps_for_55000_episodes": 55_000 * session_len,
            },
        }
    return {
        "profile_results": profile_results,
        "source_splits": {
            name: dict(split.source_payload) for name, split in splits.items()
        },
    }


def _split_diagnostics(
    *,
    sequences: np.ndarray,
    profile: tuple[int, int],
    cost_ratio: float,
    flat_hold_penalty: float,
) -> dict[str, Any]:
    oracle = _oracle_payload(sequences=sequences, profile=profile, cost_ratio=cost_ratio)
    selectors = _selector_payloads(
        sequences=sequences,
        labels=oracle["labels"],
        best_any_return=oracle["best_any_return"],
        profile=profile,
    )
    return {
        "oracle": _oracle_summary_payload(oracle),
        "reward_sparsity_proxy": _reward_sparsity_payload(
            oracle=oracle,
            session_len=profile[1],
            flat_hold_penalty=flat_hold_penalty,
        ),
        "selectors": selectors,
        "session_count": int(sequences.shape[0]),
    }


def _oracle_payload(
    *,
    sequences: np.ndarray,
    profile: tuple[int, int],
    cost_ratio: float,
) -> dict[str, Any]:
    _validate_profile(profile)
    _, session_len = profile
    close = sequences[:, :, _feature_index("close")].astype(np.float64, copy=False)
    start = SESSIONIZED_PRE_SIGNAL_LEN_V1 - 1
    stop = start + session_len
    if stop > close.shape[1]:
        raise Stage08HDiagnosticsError(
            reason="profile_session_len_exceeds_sequence",
            field=f"{profile[0]}/{profile[1]}",
        )
    prices = np.maximum(close[:, start:stop], 1e-12)
    future_max = np.maximum.accumulate(prices[:, ::-1], axis=1)[:, ::-1]
    future_min = np.minimum.accumulate(prices[:, ::-1], axis=1)[:, ::-1]
    future_max_excl = np.concatenate(
        [future_max[:, 1:], np.full((prices.shape[0], 1), np.nan)],
        axis=1,
    )
    future_min_excl = np.concatenate(
        [future_min[:, 1:], np.full((prices.shape[0], 1), np.nan)],
        axis=1,
    )
    long_returns = (future_max_excl / prices) - 1.0 - cost_ratio
    short_returns = (prices / future_min_excl) - 1.0 - cost_ratio
    long_returns[:, -1] = np.nan
    short_returns[:, -1] = np.nan
    best_long_return = np.nanmax(long_returns, axis=1)
    best_short_return = np.nanmax(short_returns, axis=1)
    long_entry = np.nanargmax(long_returns, axis=1)
    short_entry = np.nanargmax(short_returns, axis=1)
    best_side = np.where(best_long_return >= best_short_return, 1, 2)
    best_any_return = np.maximum(best_long_return, best_short_return)
    labels = np.where(best_any_return > 0.0, best_side, 0).astype(np.int8)
    best_entry = np.where(best_side == 1, long_entry, short_entry).astype(np.int32)
    best_exit = _best_exit_offsets(prices=prices, labels=labels, entry_offsets=best_entry)
    return {
        "best_any_return": best_any_return,
        "best_entry_offset": best_entry,
        "best_exit_offset": best_exit,
        "best_long_return": best_long_return,
        "best_short_return": best_short_return,
        "labels": labels,
    }


def _best_exit_offsets(
    *,
    prices: np.ndarray,
    labels: np.ndarray,
    entry_offsets: np.ndarray,
) -> np.ndarray:
    exits = np.zeros(labels.shape[0], dtype=np.int32)
    for idx, label in enumerate(labels):
        entry = int(entry_offsets[idx])
        if label == 0 or entry >= prices.shape[1] - 1:
            exits[idx] = entry
            continue
        future = prices[idx, entry + 1 :]
        offset = int(np.argmax(future) if label == 1 else np.argmin(future))
        exits[idx] = entry + 1 + offset
    return exits


def _oracle_summary_payload(oracle: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = cast(np.ndarray, oracle["labels"])
    best_any = cast(np.ndarray, oracle["best_any_return"])
    best_long = cast(np.ndarray, oracle["best_long_return"])
    best_short = cast(np.ndarray, oracle["best_short_return"])
    entry = cast(np.ndarray, oracle["best_entry_offset"])
    exit_ = cast(np.ndarray, oracle["best_exit_offset"])
    positive = best_any > 0.0
    return {
        "best_any_net_return_after_costs": _distribution_payload(best_any),
        "best_long_net_return_after_costs": _distribution_payload(best_long),
        "best_short_net_return_after_costs": _distribution_payload(best_short),
        "label_counts": _label_counts(labels),
        "mean_best_entry_offset": _round_float(float(np.mean(entry))),
        "mean_best_exit_offset": _round_float(float(np.mean(exit_))),
        "mean_positive_holding_steps": (
            None
            if not np.any(positive)
            else _round_float(float(np.mean(np.maximum(exit_[positive] - entry[positive], 0))))
        ),
        "positive_opportunity_ratio_after_costs": _round_float(float(np.mean(positive))),
    }


def _reward_sparsity_payload(
    *,
    oracle: Mapping[str, np.ndarray],
    session_len: int,
    flat_hold_penalty: float,
) -> dict[str, Any]:
    labels = cast(np.ndarray, oracle["labels"])
    entry = cast(np.ndarray, oracle["best_entry_offset"])
    exit_ = cast(np.ndarray, oracle["best_exit_offset"])
    positive = labels != 0
    if not np.any(positive):
        return {
            "status": "no_positive_oracle_trades",
            "current_reward_non_zero_trade_step_ratio_proxy": 0.0,
            "dense_mark_to_market_non_zero_step_ratio_proxy": 0.0,
        }
    duration = np.maximum(exit_[positive] - entry[positive], 1)
    current_non_zero_steps = 2.0
    current_ratio = current_non_zero_steps / float(session_len)
    dense_ratio = float(np.mean(duration)) / float(session_len)
    return {
        "current_reward_description": (
            "PnL signal is mostly paid on entry/close; flat waiting before entry receives "
            "flat-hold penalty."
        ),
        "current_reward_non_zero_trade_step_ratio_proxy": _round_float(current_ratio),
        "dense_mark_to_market_non_zero_step_ratio_proxy": _round_float(dense_ratio),
        "mean_flat_wait_penalty_before_oracle_entry": _round_float(
            float(np.mean(entry[positive]) * flat_hold_penalty)
        ),
        "positive_oracle_trade_count": int(np.sum(positive)),
    }


def _selector_payloads(
    *,
    sequences: np.ndarray,
    labels: np.ndarray,
    best_any_return: np.ndarray,
    profile: tuple[int, int],
) -> dict[str, Any]:
    scores = _selector_scores(sequences=sequences, profile=profile)
    payload: dict[str, Any] = {}
    for score_name, values in scores.items():
        payload[score_name] = {}
        for fraction in (0.1, 0.2):
            count = max(1, int(np.ceil(values.shape[0] * fraction)))
            order = np.argsort(values)
            if score_name == "mean_reversion_proxy":
                selected = order[-count:]
            else:
                selected = order[-count:]
            selected_labels = labels[selected]
            selected_returns = best_any_return[selected]
            payload[score_name][f"top_{int(fraction * 100)}pct"] = {
                "label_counts": _label_counts(selected_labels),
                "mean_best_any_return": _round_float(float(np.mean(selected_returns))),
                "positive_opportunity_ratio": _round_float(float(np.mean(selected_returns > 0.0))),
                "selected_session_count": int(count),
            }
    return payload


def _selector_scores(*, sequences: np.ndarray, profile: tuple[int, int]) -> dict[str, np.ndarray]:
    history_len, _ = profile
    window = _history_window(sequences=sequences, history_len=history_len)
    close = np.maximum(window[:, :, _feature_index("close")].astype(np.float64), 1e-12)
    high = window[:, :, _feature_index("high")].astype(np.float64)
    low = window[:, :, _feature_index("low")].astype(np.float64)
    volume = window[:, :, _feature_index("volume")].astype(np.float64)
    num_trades = window[:, :, _feature_index("num_trades")].astype(np.float64)
    returns = np.diff(close, axis=1) / close[:, :-1]
    realized_volatility = np.std(returns, axis=1)
    signed_trend = (close[:, -1] / close[:, 0]) - 1.0
    abs_trend = np.abs(signed_trend)
    range_ratio = (np.max(high, axis=1) - np.min(low, axis=1)) / close[:, -1]
    liquidity_proxy = (
        np.log1p(np.mean(volume, axis=1)) + np.log1p(np.mean(num_trades, axis=1))
    )
    return {
        "abs_trend": abs_trend,
        "high_volatility_proxy": realized_volatility + range_ratio,
        "liquidity_proxy": liquidity_proxy,
        "mean_reversion_proxy": realized_volatility / np.maximum(abs_trend, 1e-9),
        "signed_uptrend": signed_trend,
    }


def _supervised_sanity(
    *,
    train_sequences: np.ndarray,
    eval_splits: Mapping[str, np.ndarray],
    profile: tuple[int, int],
    cost_ratio: float,
) -> dict[str, Any]:
    train_features = _feature_matrix(sequences=train_sequences, profile=profile)
    train_oracle = _oracle_payload(
        sequences=train_sequences,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    train_labels = cast(np.ndarray, train_oracle["labels"])
    scaler_mean = np.mean(train_features, axis=0)
    scaler_std = np.std(train_features, axis=0)
    scaler_std = np.where(scaler_std == 0.0, 1.0, scaler_std)
    train_x = (train_features - scaler_mean) / scaler_std
    weights = _fit_ridge_classifier(train_x, train_labels)
    majority_label = int(np.bincount(train_labels, minlength=len(LABEL_NAMES)).argmax())
    split_payloads: dict[str, Any] = {}
    for split_name, sequences in eval_splits.items():
        features = _feature_matrix(sequences=sequences, profile=profile)
        oracle = _oracle_payload(sequences=sequences, profile=profile, cost_ratio=cost_ratio)
        labels = cast(np.ndarray, oracle["labels"])
        x = (features - scaler_mean) / scaler_std
        predicted = _predict_ridge_classifier(x, weights)
        recent = _recent_return_rule_labels(
            sequences=sequences,
            profile=profile,
            cost_ratio=cost_ratio,
        )
        majority = np.full(labels.shape, majority_label, dtype=np.int8)
        split_payloads[split_name] = {
            "label_counts": _label_counts(labels),
            "majority_baseline": _classification_metrics(labels, majority),
            "recent_return_baseline": _classification_metrics(labels, recent),
            "ridge_past_window_model": {
                **_classification_metrics(labels, predicted),
                "prediction_counts": _label_counts(predicted),
            },
        }
    return {
        "feature_count": int(train_features.shape[1]),
        "model": "closed_form_ridge_classifier_numpy",
        "splits": split_payloads,
        "status": "completed",
        "train_label_counts": _label_counts(train_labels),
    }


def _feature_matrix(*, sequences: np.ndarray, profile: tuple[int, int]) -> np.ndarray:
    history_len, _ = profile
    window = _history_window(sequences=sequences, history_len=history_len)
    close = np.maximum(window[:, :, _feature_index("close")].astype(np.float64), 1e-12)
    high = window[:, :, _feature_index("high")].astype(np.float64)
    low = window[:, :, _feature_index("low")].astype(np.float64)
    vwap = np.maximum(
        window[:, :, _feature_index("volume_weighted_average")].astype(np.float64),
        1e-12,
    )
    volume = window[:, :, _feature_index("volume")].astype(np.float64)
    num_trades = window[:, :, _feature_index("num_trades")].astype(np.float64)
    returns = np.diff(close, axis=1) / close[:, :-1]
    tail = min(10, returns.shape[1])
    features = np.column_stack(
        [
            (close[:, -1] / close[:, 0]) - 1.0,
            np.mean(returns, axis=1),
            np.std(returns, axis=1),
            np.min(returns, axis=1),
            np.max(returns, axis=1),
            np.sum(returns[:, -tail:], axis=1),
            (np.max(high, axis=1) - np.min(low, axis=1)) / close[:, -1],
            (close[:, -1] / vwap[:, -1]) - 1.0,
            np.log1p(np.mean(volume, axis=1)),
            np.log1p(np.std(volume, axis=1)),
            np.log1p(np.mean(num_trades, axis=1)),
            np.log1p(np.std(num_trades, axis=1)),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_ridge_classifier(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(features.shape[0]), features])
    y = np.zeros((labels.shape[0], len(LABEL_NAMES)), dtype=np.float64)
    y[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _predict_ridge_classifier(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(features.shape[0]), features])
    return np.argmax(x @ weights, axis=1).astype(np.int8)


def _recent_return_rule_labels(
    *,
    sequences: np.ndarray,
    profile: tuple[int, int],
    cost_ratio: float,
) -> np.ndarray:
    history_len, _ = profile
    window = _history_window(sequences=sequences, history_len=history_len)
    close = np.maximum(window[:, :, _feature_index("close")].astype(np.float64), 1e-12)
    recent_return = (close[:, -1] / close[:, 0]) - 1.0
    labels = np.zeros(recent_return.shape[0], dtype=np.int8)
    labels[recent_return > cost_ratio] = 1
    labels[recent_return < -cost_ratio] = 2
    return labels


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    accuracy = float(np.mean(labels == predictions))
    recalls = []
    for label_id in range(len(LABEL_NAMES)):
        mask = labels == label_id
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == label_id)))
    return {
        "accuracy": _round_float(accuracy),
        "balanced_accuracy": _round_float(float(np.mean(recalls))) if recalls else None,
    }


def _history_window(*, sequences: np.ndarray, history_len: int) -> np.ndarray:
    if history_len < 2 or history_len > SESSIONIZED_PRE_SIGNAL_LEN_V1:
        raise Stage08HDiagnosticsError(reason="invalid_history_len", field=str(history_len))
    start = SESSIONIZED_PRE_SIGNAL_LEN_V1 - history_len
    stop = SESSIONIZED_PRE_SIGNAL_LEN_V1
    return sequences[:, start:stop, :]


def _load_branch_splits(
    *,
    branch: str,
    split_names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if branch == "hf_original":
        specs = {spec.split_name: spec for spec in hf_eval_cli.expected_hf_split_specs_v1()}
        return {
            split_name: hf_eval_cli._load_hf_split(  # noqa: SLF001
                dataset_dir=args.hf_dataset_dir,
                split_spec=specs[split_name],
                max_sessions=args.max_sessions_per_split,
                allow_fixture_hashes=args.allow_fixture_hashes,
            )
            for split_name in split_names
        }
    if branch == "roehub_native":
        manifest = native_eval_cli._read_json(args.stage06_manifest_path)  # noqa: SLF001
        manifest_sha256 = native_eval_cli.compute_file_sha256(args.stage06_manifest_path)
        return {
            split_name: native_eval_cli._load_stage06_split(  # noqa: SLF001
                manifest=manifest,
                manifest_path=args.stage06_manifest_path,
                manifest_sha256=manifest_sha256,
                dataset_version=args.dataset_version,
                split=split_name,
                max_sessions=args.max_sessions_per_split,
                max_artifacts=args.max_artifacts_per_split,
                allow_fixture_hashes=args.allow_fixture_hashes,
            )
            for split_name in split_names
        }
    raise Stage08HDiagnosticsError(reason="unsupported_branch", field=branch)


def _selected_branches(values: Sequence[str]) -> tuple[str, ...]:
    if "all" in values:
        return DEFAULT_BRANCHES
    selected = tuple(dict.fromkeys(values))
    unsupported = [value for value in selected if value not in DEFAULT_BRANCHES]
    if unsupported:
        raise Stage08HDiagnosticsError(reason="unsupported_branch", field=",".join(unsupported))
    return selected


def _selected_splits(values: Sequence[str]) -> tuple[str, ...]:
    if "all" in values:
        return DEFAULT_SPLITS
    selected = tuple(dict.fromkeys(values))
    unsupported = [value for value in selected if value not in DEFAULT_SPLITS]
    if unsupported:
        raise Stage08HDiagnosticsError(reason="unsupported_split", field=",".join(unsupported))
    return selected


def _parse_profile(value: str) -> tuple[int, int]:
    text = value.strip().lower().replace("x", "/")
    parts = text.split("/")
    if len(parts) != 2:
        raise Stage08HDiagnosticsError(reason="invalid_profile", field=value)
    try:
        profile = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise Stage08HDiagnosticsError(reason="invalid_profile", field=value) from exc
    _validate_profile(profile)
    return profile


def _validate_profile(profile: tuple[int, int]) -> None:
    history_len, session_len = profile
    if history_len < 2 or history_len > SESSIONIZED_PRE_SIGNAL_LEN_V1:
        raise Stage08HDiagnosticsError(reason="invalid_history_len", field=str(history_len))
    if (
        session_len < 2
        or (SESSIONIZED_PRE_SIGNAL_LEN_V1 - 1 + session_len) > SESSIONIZED_FULL_SEQ_LEN_V1
    ):
        raise Stage08HDiagnosticsError(reason="invalid_session_len", field=str(session_len))


def _feature_index(name: str) -> int:
    try:
        return FEATURE_NAMES_V1.index(name)
    except ValueError as exc:
        raise Stage08HDiagnosticsError(reason="feature_missing", field=name) from exc


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(LABEL_NAMES))
    return {LABEL_NAMES[index]: int(counts[index]) for index in range(len(LABEL_NAMES))}


def _distribution_payload(values: np.ndarray) -> dict[str, float]:
    return {
        "max": _round_float(float(np.max(values))),
        "mean": _round_float(float(np.mean(values))),
        "median": _round_float(float(np.median(values))),
        "p10": _round_float(float(np.quantile(values, 0.10))),
        "p90": _round_float(float(np.quantile(values, 0.90))),
    }


def _round_float(value: float) -> float:
    return float(round(value, 10))


def _default_run_id(
    *,
    args: argparse.Namespace,
    branches: Sequence[str],
    profiles: Sequence[tuple[int, int]],
) -> str:
    digest = hash_json_payload_v1(
        {
            "branches": list(branches),
            "dataset_version": args.dataset_version,
            "hf_dataset_dir": str(args.hf_dataset_dir),
            "max_artifacts_per_split": args.max_artifacts_per_split,
            "max_sessions_per_split": args.max_sessions_per_split,
            "profiles": [list(profile) for profile in profiles],
            "splits": list(_selected_splits(args.splits)),
            "stage": "08H",
            "stage06_manifest_path": str(args.stage06_manifest_path),
        }
    )
    return f"stage08h_dataset_diagnostics_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    source_paths = (
        "scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py",
        "scripts/rl_trading/stage08c_original_hf_full_training_run.py",
        "scripts/rl_trading/stage08e_roehub_native_full_training_run.py",
        "scripts/rl_trading/stage08g_cpu_optuna_calibration.py",
        "scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py",
        "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
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


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 08H oracle/supervised/session-selector diagnostics."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=("all", "hf_original", "roehub_native"),
        default=["all"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("all", "train", "validation", "test", "backtest"),
        default=["all"],
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--max-sessions-per-split", type=_optional_positive_int, default=None)
    parser.add_argument("--max-artifacts-per-split", type=_optional_positive_int, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--hf-dataset-dir", type=Path, default=hf_eval_cli.DEFAULT_HF_DATASET_DIR)
    parser.add_argument(
        "--stage06-manifest-path",
        type=Path,
        default=native_eval_cli.DEFAULT_STAGE06_MANIFEST_PATH,
    )
    parser.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
