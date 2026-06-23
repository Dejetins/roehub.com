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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE08_RUNTIME_ARTIFACT_ROOT_V1,
    Stage08EvaluationConfig,
    Stage08EvaluationError,
    Stage08TorchD3qnPolicy,
    build_stage08_evaluation_artifact_v1,
    candidate_training_config_from_payload_v1,
    default_stage08_evaluation_policies_v1,
    evaluate_stage08_policy_v1,
    render_raw_feature_json_payload_v1,
    stage08_accounting_parity_fixture_v1,
)

DEFAULT_CANDIDATE_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1")
    / "stage07b_candidate_b43be9c1_61995c61_c5fbee2b"
    / "candidate_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST_SHA256 = (
    "709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158"
)
DEFAULT_STAGE06_SESSIONIZED_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1")
    / "stage06_sessionized_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE08_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / "stage08_roehub_backtest_evaluation_v1"
)
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/backtest_evaluation.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08_roehub_backtest_evaluation.py",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run_command(args)
    except Stage08EvaluationError as exc:
        print(
            _render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"})
        )
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] in {"accepted_for_research", "completed"} else 2


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    candidate_manifest_sha256 = _file_sha256_hex(args.candidate_manifest)
    if args.expected_candidate_manifest_sha256 and (
        candidate_manifest_sha256 != args.expected_candidate_manifest_sha256
    ):
        raise Stage08EvaluationError(
            reason="candidate_manifest_sha256_mismatch",
            field=str(args.candidate_manifest),
        )
    candidate_manifest = _read_json(args.candidate_manifest)
    _validate_candidate_manifest(candidate_manifest)

    sessionized_manifest = args.sessionized_manifest
    expected_sessionized_sha256 = (
        args.expected_sessionized_manifest_sha256
        or str(candidate_manifest["dataset_dependency"]["manifest_sha256"])
    )
    sessionized_manifest_sha256 = _file_sha256_hex(sessionized_manifest)
    if sessionized_manifest_sha256 != expected_sessionized_sha256:
        raise Stage08EvaluationError(
            reason="sessionized_manifest_sha256_mismatch",
            field=str(sessionized_manifest),
        )
    manifest = _read_json(sessionized_manifest)
    _validate_stage06_manifest(manifest)

    candidate_report = _read_optional_artifact(candidate_manifest, "candidate_report")
    training_config = _read_optional_artifact(candidate_manifest, "training_config")
    candidate_config = (
        candidate_training_config_from_payload_v1(training_config)
        if training_config is not None
        else None
    )
    evaluation_config = Stage08EvaluationConfig(
        initial_balance=(
            candidate_config.initial_balance
            if candidate_config is not None
            else args.initial_balance
        ),
        slippage=candidate_config.slippage if candidate_config is not None else args.slippage,
        transaction_fee=(
            candidate_config.transaction_fee
            if candidate_config is not None
            else args.transaction_fee
        ),
        inaction_penalty_ratio=(
            candidate_config.inaction_penalty_ratio
            if candidate_config is not None
            else args.inaction_penalty_ratio
        ),
        random_seed=args.random_seed,
        simple_threshold_return=args.simple_threshold_return,
    )
    session_features, symbols, signal_times, selection = _load_session_features(
        manifest=manifest,
        dataset_version=args.dataset_version,
        split=args.split,
        symbols=args.symbol,
        max_session_artifacts=args.max_session_artifacts,
        max_sessions=args.max_sessions,
    )
    candidate_policy = Stage08TorchD3qnPolicy(
        candidate_manifest=candidate_manifest,
        device_policy=args.device_policy,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )
    policies = (
        candidate_policy,
        *default_stage08_evaluation_policies_v1(
            random_seed=evaluation_config.random_seed,
            simple_threshold_return=evaluation_config.simple_threshold_return,
        ),
    )
    scorecards = [
        evaluate_stage08_policy_v1(
            session_features=session_features,
            symbols=symbols,
            signal_times_utc=signal_times,
            policy=policy,
            config=evaluation_config,
        )
        for policy in policies
    ]
    generated_at = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    run_id = args.run_id or _default_run_id(
        candidate_manifest_sha256=candidate_manifest_sha256,
        sessionized_manifest_sha256=sessionized_manifest_sha256,
        selection=selection,
        config=evaluation_config,
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scorecards_path = run_dir / "scorecards.json"
    _atomic_write_json(scorecards_path, {"scorecards": scorecards})
    parity_fixture = stage08_accounting_parity_fixture_v1(config=evaluation_config)
    artifact = build_stage08_evaluation_artifact_v1(
        generated_at_utc=generated_at,
        candidate_manifest_path=str(args.candidate_manifest),
        candidate_manifest_sha256=candidate_manifest_sha256,
        sessionized_manifest_path=str(sessionized_manifest),
        sessionized_manifest_sha256=sessionized_manifest_sha256,
        selection=selection,
        scorecards=scorecards,
        candidate_report=candidate_report,
        parity_fixture=parity_fixture,
        config=evaluation_config,
        code_version=_source_state_payload(),
        artifact_hashes={"scorecards": _file_payload(scorecards_path)},
    )
    evaluation_manifest_path = run_dir / "stage08_evaluation_manifest.json"
    _atomic_write_json(evaluation_manifest_path, artifact)
    selected_symbols = cast(Sequence[str], selection["selected_symbols"])
    return {
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "evaluation_hash": artifact["evaluation_hash"],
        "evaluation_manifest_path": str(evaluation_manifest_path),
        "evaluation_manifest_sha256": _file_sha256_hex(evaluation_manifest_path),
        "research_candidate_save_allowed": artifact["research_candidate_save_allowed"],
        "run_dir": str(run_dir),
        "run_id": run_id,
        "scorecards_path": str(scorecards_path),
        "scorecards_sha256": _file_sha256_hex(scorecards_path),
        "selected_session_count": selection["selected_session_count"],
        "selected_symbol_count": len(selected_symbols),
        "status": artifact["status"],
    }


def _load_session_features(
    *,
    manifest: Mapping[str, Any],
    dataset_version: str,
    split: str,
    symbols: Sequence[str] | None,
    max_session_artifacts: int | None,
    max_sessions: int | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str | None, ...], dict[str, object]]:
    selected_symbols = None if not symbols else {symbol.upper() for symbol in symbols}
    entries = []
    for entry in _split_artifact_entries(manifest):
        if entry.get("dataset_version") != dataset_version:
            continue
        if entry.get("split") != split:
            continue
        symbol = str(entry.get("symbol", "")).upper()
        if selected_symbols is not None and symbol not in selected_symbols:
            continue
        entries.append(entry)
    entries.sort(key=lambda item: str(item["symbol"]))
    if max_session_artifacts is not None:
        entries = entries[:max_session_artifacts]
    if not entries:
        raise Stage08EvaluationError(reason="no_sessionized_split_artifact_selected")

    chunks: list[np.ndarray] = []
    session_symbols: list[str] = []
    signal_times: list[str | None] = []
    remaining = max_sessions
    selected_artifact_count = 0
    for entry in entries:
        symbol = str(entry["symbol"]).upper()
        features_path = _artifact_file_path(entry, "features")
        features = np.load(features_path, mmap_mode="r")
        if features.ndim != 3 or features.shape[0] == 0:
            continue
        take = (
            int(features.shape[0])
            if remaining is None
            else min(int(features.shape[0]), remaining)
        )
        if take <= 0:
            break
        chunks.append(np.array(features[:take], dtype=np.float32, copy=True))
        del features
        metadata_rows = _metadata_rows(entry)
        session_symbols.extend([symbol] * take)
        signal_times.extend(_signal_times_from_metadata(metadata_rows, take=take))
        selected_artifact_count += 1
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break
    if not chunks:
        raise Stage08EvaluationError(reason="selected_session_features_empty")
    selected_session_count = len(session_symbols)
    selection = {
        "dataset_version": dataset_version,
        "max_session_artifacts": max_session_artifacts,
        "max_sessions": max_sessions,
        "selected_artifact_count": selected_artifact_count,
        "selected_session_count": selected_session_count,
        "selected_symbols": sorted(set(session_symbols)),
        "selection_limited": max_session_artifacts is not None or max_sessions is not None,
        "split": split,
    }
    features = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    return features, tuple(session_symbols), tuple(signal_times), selection


def _validate_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "07B":
        raise Stage08EvaluationError(reason="unexpected_candidate_manifest_stage", field="stage")
    if manifest.get("status") != "completed":
        raise Stage08EvaluationError(reason="candidate_manifest_not_completed", field="status")
    handoff = manifest.get("next_stage_handoff")
    if not isinstance(handoff, Mapping) or handoff.get("stage08_allowed") is not True:
        raise Stage08EvaluationError(reason="candidate_manifest_stage08_not_allowed")
    dependency = manifest.get("dataset_dependency")
    if not isinstance(dependency, Mapping):
        raise Stage08EvaluationError(reason="candidate_dataset_dependency_missing")
    if dependency.get("training_source") != "binance:futures":
        raise Stage08EvaluationError(reason="unexpected_candidate_training_source")


def _validate_stage06_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "06":
        raise Stage08EvaluationError(reason="unexpected_sessionized_manifest_stage", field="stage")
    if manifest.get("status") != "accepted":
        raise Stage08EvaluationError(reason="sessionized_manifest_not_accepted", field="status")
    if manifest.get("market") != "binance:futures":
        raise Stage08EvaluationError(
            reason="unexpected_sessionized_manifest_market",
            field="market",
        )


def _split_artifact_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = manifest.get("split_artifacts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise Stage08EvaluationError(reason="split_artifacts_not_sequence")
    return tuple(cast(Mapping[str, Any], item) for item in entries if isinstance(item, Mapping))


def _artifact_file_path(entry: Mapping[str, Any], key: str) -> Path:
    files = entry.get("files")
    if not isinstance(files, Mapping):
        raise Stage08EvaluationError(reason="split_artifact_files_not_mapping")
    item = files.get(key)
    if not isinstance(item, Mapping):
        raise Stage08EvaluationError(reason="split_artifact_file_missing", field=key)
    return Path(str(item["path"]))


def _metadata_rows(entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    try:
        metadata_path = _artifact_file_path(entry, "metadata")
    except Stage08EvaluationError:
        return ()
    if not metadata_path.exists():
        return ()
    payload = _read_json(metadata_path)
    sessions = payload.get("sessions")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        return ()
    return tuple(cast(Mapping[str, Any], item) for item in sessions if isinstance(item, Mapping))


def _signal_times_from_metadata(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    take: int,
) -> list[str | None]:
    values: list[str | None] = []
    for index in range(take):
        if index >= len(metadata_rows):
            values.append(None)
            continue
        value = metadata_rows[index].get("signal_ts_open")
        values.append(str(value) if isinstance(value, str) and value else None)
    return values


def _read_optional_artifact(
    candidate_manifest: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    artifacts = candidate_manifest.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        return None
    item = artifacts.get(key)
    if not isinstance(item, Mapping):
        return None
    path_value = item.get("path")
    if not isinstance(path_value, str):
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return _read_json(path)


def _default_run_id(
    *,
    candidate_manifest_sha256: str,
    sessionized_manifest_sha256: str,
    selection: Mapping[str, object],
    config: Stage08EvaluationConfig,
) -> str:
    payload = {
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "config_hash": config.config_hash(),
        "selection": dict(selection),
        "sessionized_manifest_sha256": sessionized_manifest_sha256,
        "stage": "08",
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"stage08_eval_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    head = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--short", "--", *SOURCE_STATE_PATHS)
    diff = _git_output("diff", "--", *SOURCE_STATE_PATHS)
    return {
        "git_head": head,
        "git_status_short": status.splitlines(),
        "source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "source_paths": list(SOURCE_STATE_PATHS),
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_payload(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": _file_sha256_hex(path)}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_raw_feature_json_payload_v1(dict(payload)) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 08 RL backtest evaluation.")
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        type=str,
        default=DEFAULT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--sessionized-manifest",
        type=Path,
        default=DEFAULT_STAGE06_SESSIONIZED_MANIFEST,
    )
    parser.add_argument("--expected-sessionized-manifest-sha256", type=str, default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="hf_period_rebuild_current_trading")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--max-session-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--max-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--inaction-penalty-ratio", type=float, default=0.0001)
    parser.add_argument("--random-seed", type=int, default=240824)
    parser.add_argument("--simple-threshold-return", type=float, default=0.001)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="cpu_only_deterministic",
    )
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
