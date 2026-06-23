from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    CandidateTrainingConfig,
    PrioritizedReplayConfig,
    TrainingRunnerError,
    build_stage07b_transition_set_v1,
    hash_json_payload_v1,
    run_stage07b_candidate_training_v1,
    training_source_gate_payload_v1,
)

DEFAULT_STAGE06_SESSIONIZED_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1")
    / "stage06_sessionized_manifest.json"
)
DEFAULT_STAGE06_SESSIONIZED_MANIFEST_SHA256 = (
    "61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "training_runs"
    / "stage07b_full_candidate_training_run_v1"
)
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/training_runner.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage07b_full_candidate_training_run.py",
    "apps/worker/rl_trading_trainer/main/main.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status_command(run_dir=args.run_dir)
    if args.command == "run":
        return _run_command(args)
    parser.error("unknown command")
    return 2


def _run_command(args: argparse.Namespace) -> int:
    gate = training_source_gate_payload_v1(exchange=args.exchange, market_type=args.market_type)
    if gate["status"] != "trainable":
        print(
            json.dumps(
                {
                    "exchange": gate["exchange"],
                    "market_type": gate["market_type"],
                    "reason": "blocked_not_training_source_v1",
                    "status": "blocked",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2

    try:
        manifest_sha256 = _file_sha256_hex(args.sessionized_manifest)
        if args.expected_sessionized_manifest_sha256 and (
            manifest_sha256 != args.expected_sessionized_manifest_sha256
        ):
            raise TrainingRunnerError(
                reason="sessionized_manifest_sha256_mismatch",
                field=str(args.sessionized_manifest),
            )
        manifest = _read_json(args.sessionized_manifest)
        _validate_stage06_manifest(manifest)
        replay = PrioritizedReplayConfig(
            capacity=args.replay_capacity,
            alpha=args.replay_alpha,
            beta=args.replay_beta,
            epsilon=args.replay_epsilon,
            min_priority=args.replay_min_priority,
        )
        config = CandidateTrainingConfig(
            seed=args.seed,
            train_dataset_version=args.train_dataset_version,
            train_split=args.train_split,
            validation_dataset_version=args.validation_dataset_version,
            validation_split=args.validation_split,
            batch_size=args.batch_size,
            planned_training_steps=args.planned_training_steps,
            progress_emit_every_steps=args.progress_emit_every_steps,
            progress_emit_every_sec=args.progress_emit_every_sec,
            checkpoint_every_steps=args.checkpoint_every_steps,
            validation_every_steps=args.validation_every_steps,
            validation_max_transitions=args.validation_max_transitions,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            target_sync_interval=args.target_sync_interval,
            torch_num_threads=args.torch_num_threads,
            torch_num_interop_threads=args.torch_num_interop_threads,
            device_policy=args.device_policy,
            replay=replay,
            hidden_dims=tuple(args.hidden_dim or [128, 128]),
        )
        train_features, train_selection = _load_session_features(
            manifest=manifest,
            dataset_version=config.train_dataset_version,
            split=config.train_split,
            symbols=args.symbol,
            max_session_artifacts=args.max_train_session_artifacts,
            max_sessions=args.max_train_sessions,
        )
        validation_features, validation_selection = _load_session_features(
            manifest=manifest,
            dataset_version=config.validation_dataset_version,
            split=config.validation_split,
            symbols=args.symbol,
            max_session_artifacts=args.max_validation_session_artifacts,
            max_sessions=args.max_validation_sessions,
        )
        train_transitions = build_stage07b_transition_set_v1(
            session_features=train_features,
            config=config,
        )
        validation_transitions = build_stage07b_transition_set_v1(
            session_features=validation_features,
            config=config,
        )
        source_state = _source_state_payload()
        run_id = args.run_id or _default_run_id(
            config_hash=config.config_hash(),
            manifest_sha256=manifest_sha256,
            source_state=source_state,
            train_selection=train_selection,
            validation_selection=validation_selection,
        )
        run_dir = args.output_root / run_id
        source_state = {
            **source_state,
            "operator_commands": {
                "resume": _resume_command_for_args(args=args, run_id=run_id),
                "status": (
                    "uv run --extra rl-ml python "
                    "scripts/rl_trading/stage07b_full_candidate_training_run.py "
                    f"status --run-dir {run_dir}"
                ),
            },
            "selection": {
                "train": train_selection,
                "validation": validation_selection,
            },
        }
        manifest_payload = run_stage07b_candidate_training_v1(
            train_transitions=train_transitions,
            validation_transitions=validation_transitions,
            dataset_manifest_path=str(args.sessionized_manifest),
            dataset_manifest_sha256=manifest_sha256,
            output_root=args.output_root,
            run_id=run_id,
            config=config,
            generated_at_utc=(
                _parse_utc(args.generated_at_utc)
                if args.generated_at_utc is not None
                else datetime.now(UTC).replace(microsecond=0)
            ),
            code_version=source_state,
            resume=bool(args.resume),
        )
    except TrainingRunnerError as exc:
        print(
            json.dumps(
                {"field": exc.field, "reason": exc.reason, "status": "blocked"},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "candidate_manifest_hash": manifest_payload["candidate_manifest_hash"],
                "candidate_manifest_path": manifest_payload["candidate_manifest_path"],
                "config_hash": manifest_payload["config_hash"],
                "dataset_manifest_sha256": manifest_payload["dataset_dependency"][
                    "manifest_sha256"
                ],
                "run_dir": manifest_payload["run_dir"],
                "run_id": manifest_payload["run_id"],
                "status": manifest_payload["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _status_command(*, run_dir: Path) -> int:
    latest_status_path = run_dir / "latest_status.json"
    candidate_manifest_path = run_dir / "candidate_manifest.json"
    if not latest_status_path.exists():
        print(
            json.dumps(
                {
                    "latest_status_path": str(latest_status_path),
                    "reason": "latest_status_missing",
                    "run_dir": str(run_dir),
                    "status": "blocked",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2
    latest = _read_json(latest_status_path)
    payload: dict[str, Any] = {
        "candidate_manifest_path": (
            str(candidate_manifest_path) if candidate_manifest_path.exists() else None
        ),
        "candidate_manifest_sha256": (
            _file_sha256_hex(candidate_manifest_path) if candidate_manifest_path.exists() else None
        ),
        "latest_status": latest,
        "latest_status_path": str(latest_status_path),
        "latest_status_sha256": _file_sha256_hex(latest_status_path),
        "progress_path": str(run_dir / "progress.jsonl"),
        "run_dir": str(run_dir),
        "status": latest.get("status", "unknown"),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


def _load_session_features(
    *,
    manifest: Mapping[str, Any],
    dataset_version: str,
    split: str,
    symbols: Sequence[str] | None,
    max_session_artifacts: int | None,
    max_sessions: int | None,
) -> tuple[np.ndarray, dict[str, object]]:
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
        raise TrainingRunnerError(reason="no_sessionized_split_artifact_selected")

    chunks: list[np.ndarray] = []
    remaining = max_sessions
    selected_session_count = 0
    for entry in entries:
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
        chunks.append(np.asarray(features[:take], dtype=np.float32))
        selected_session_count += take
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break
    if not chunks:
        raise TrainingRunnerError(reason="selected_session_features_empty")
    payload = {
        "dataset_version": dataset_version,
        "max_session_artifacts": max_session_artifacts,
        "max_sessions": max_sessions,
        "selected_artifact_count": len(entries),
        "selected_session_count": selected_session_count,
        "selection_limited": max_session_artifacts is not None or max_sessions is not None,
        "split": split,
        "symbols": None if selected_symbols is None else sorted(selected_symbols),
    }
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32), payload


def _validate_stage06_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "06":
        raise TrainingRunnerError(reason="unexpected_sessionized_manifest_stage", field="stage")
    if manifest.get("status") != "accepted":
        raise TrainingRunnerError(reason="sessionized_manifest_not_accepted", field="status")
    if manifest.get("market") != "binance:futures":
        raise TrainingRunnerError(reason="unexpected_sessionized_manifest_market", field="market")


def _split_artifact_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = manifest.get("split_artifacts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TrainingRunnerError(reason="split_artifacts_not_sequence")
    return tuple(cast(Mapping[str, Any], item) for item in entries if isinstance(item, Mapping))


def _artifact_file_path(entry: Mapping[str, Any], key: str) -> Path:
    files = entry.get("files")
    if not isinstance(files, Mapping):
        raise TrainingRunnerError(reason="split_artifact_files_not_mapping")
    item = files.get(key)
    if not isinstance(item, Mapping):
        raise TrainingRunnerError(reason="split_artifact_file_missing", field=key)
    return Path(str(item["path"]))


def _default_run_id(
    *,
    config_hash: str,
    manifest_sha256: str,
    source_state: Mapping[str, Any],
    train_selection: Mapping[str, object],
    validation_selection: Mapping[str, object],
) -> str:
    digest = hash_json_payload_v1(
        {
            "config_hash": config_hash,
            "manifest_sha256": manifest_sha256,
            "source_state": source_state,
            "stage": "07B",
            "train_selection": train_selection,
            "validation_selection": validation_selection,
        }
    )
    return f"stage07b_candidate_{digest[:20]}"


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
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
    ).strip()


def _resume_command_for_args(*, args: argparse.Namespace, run_id: str) -> str:
    parts = [
        "uv run --extra rl-ml python",
        "scripts/rl_trading/stage07b_full_candidate_training_run.py",
        "run",
        "--resume",
        f"--run-id {run_id}",
        f"--planned-training-steps {args.planned_training_steps}",
        f"--batch-size {args.batch_size}",
        f"--replay-capacity {args.replay_capacity}",
        f"--torch-num-threads {args.torch_num_threads}",
        f"--torch-num-interop-threads {args.torch_num_interop_threads}",
        f"--device-policy {args.device_policy}",
    ]
    return " ".join(parts)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


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


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect Stage 07B candidate training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run or resume Stage 07B candidate training.")
    run.add_argument(
        "--sessionized-manifest",
        type=Path,
        default=DEFAULT_STAGE06_SESSIONIZED_MANIFEST,
    )
    run.add_argument(
        "--expected-sessionized-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE06_SESSIONIZED_MANIFEST_SHA256,
    )
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id", type=str, default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--exchange", type=str, default="binance")
    run.add_argument("--market-type", type=str, default="futures")
    run.add_argument(
        "--train-dataset-version",
        type=str,
        default="hf_period_rebuild_current_trading",
    )
    run.add_argument("--train-split", type=str, default="train")
    run.add_argument(
        "--validation-dataset-version",
        type=str,
        default="hf_period_rebuild_current_trading",
    )
    run.add_argument("--validation-split", type=str, default="validation")
    run.add_argument("--symbol", action="append", default=None)
    run.add_argument("--max-train-session-artifacts", type=_optional_positive_int, default=None)
    run.add_argument(
        "--max-validation-session-artifacts",
        type=_optional_positive_int,
        default=None,
    )
    run.add_argument("--max-train-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--max-validation-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--seed", type=int, default=240723)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--planned-training-steps", type=int, default=100_000)
    run.add_argument("--progress-emit-every-steps", type=int, default=10_000)
    run.add_argument("--progress-emit-every-sec", type=int, default=300)
    run.add_argument("--checkpoint-every-steps", type=int, default=10_000)
    run.add_argument("--validation-every-steps", type=int, default=10_000)
    run.add_argument("--validation-max-transitions", type=int, default=4_096)
    run.add_argument("--gamma", type=float, default=0.99)
    run.add_argument("--learning-rate", type=float, default=0.0005)
    run.add_argument("--target-sync-interval", type=int, default=1_000)
    run.add_argument("--torch-num-threads", type=int, default=4)
    run.add_argument("--torch-num-interop-threads", type=int, default=1)
    run.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="cpu_only_deterministic",
    )
    run.add_argument("--replay-capacity", type=int, default=200_000)
    run.add_argument("--replay-alpha", type=float, default=0.6)
    run.add_argument("--replay-beta", type=float, default=0.4)
    run.add_argument("--replay-epsilon", type=float, default=1e-5)
    run.add_argument("--replay-min-priority", type=float, default=1e-5)
    run.add_argument("--hidden-dim", action="append", type=int, default=None)
    run.add_argument("--generated-at-utc", type=str, default=None)

    status = subparsers.add_parser("status", help="Print latest Stage 07B run status.")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
