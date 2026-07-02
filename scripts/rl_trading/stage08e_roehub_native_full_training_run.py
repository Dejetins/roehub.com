from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    FEATURE_NAMES_V1,
    SESSIONIZED_DATASET_MANIFEST_KIND_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    STAGE08E_RUNTIME_ARTIFACT_SUBDIR_V1,
    RoehubNativeTrainingConfig,
    RoehubNativeTrainingError,
    UpstreamAlphaConfig,
    compute_file_sha256,
    hash_json_payload_v1,
    run_stage08e_roehub_native_training_v1,
)
from trading.contexts.rl_trading.domain.hf_original_training import (  # noqa: E402
    HfOriginalTrainingError,
)

DEFAULT_STAGE06_MANIFEST_PATH = Path(
    "/opt/roehub/state/rl_trading/datasets/"
    "stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "training_runs"
    / STAGE08E_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_TORCH_NUM_THREADS = max(1, os.cpu_count() or 1)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VALIDATION_SPLIT = "validation"
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/hf_original_training.py",
    "src/trading/contexts/rl_trading/domain/roehub_native_training.py",
    "src/trading/contexts/rl_trading/domain/sessionized_dataset.py",
    "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08e_roehub_native_full_training_run.py",
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
    try:
        alpha = UpstreamAlphaConfig(
            seed=args.seed,
            agent_history_len=args.agent_history_len,
            agent_session_len=args.agent_session_len,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_start=args.train_start,
            replay_capacity=args.replay_capacity,
            target_update_freq=args.target_update_freq,
            eps_decay_frames=args.eps_decay_frames,
            torch_num_threads=args.torch_num_threads,
            torch_num_interop_threads=args.torch_num_interop_threads,
        )
        config = RoehubNativeTrainingConfig(
            alpha=alpha,
            stage="08E",
            planned_episodes=args.episodes,
            validation_every_episodes=args.validation_every_episodes,
            checkpoint_every_episodes=args.checkpoint_every_episodes,
            progress_emit_every_episodes=args.progress_emit_every_episodes,
            progress_emit_every_sec=args.progress_emit_every_sec,
            validation_max_sessions=args.validation_max_sessions,
            device_policy=args.device_policy,
        )
        manifest = _read_json(args.stage06_manifest_path)
        manifest_sha256 = compute_file_sha256(args.stage06_manifest_path)
        train, train_payload = _load_stage06_split_features(
            manifest=manifest,
            manifest_path=args.stage06_manifest_path,
            manifest_sha256=manifest_sha256,
            dataset_version=args.dataset_version,
            split=args.train_split,
            max_sessions=args.max_train_sessions,
            max_artifacts=args.max_train_artifacts,
            allow_fixture_hashes=args.allow_fixture_hashes,
            accepted_stages=(args.sessionized_manifest_stage,),
        )
        validation, validation_payload = _load_stage06_split_features(
            manifest=manifest,
            manifest_path=args.stage06_manifest_path,
            manifest_sha256=manifest_sha256,
            dataset_version=args.dataset_version,
            split=args.validation_split,
            max_sessions=args.max_validation_sessions,
            max_artifacts=args.max_validation_artifacts,
            allow_fixture_hashes=args.allow_fixture_hashes,
            accepted_stages=(args.sessionized_manifest_stage,),
        )
        dataset_dependency = {
            "allow_fixture_hashes": bool(args.allow_fixture_hashes),
            "dataset_version": args.dataset_version,
            "leakage_report_status": _mapping_field(manifest, "leakage_report").get("status"),
            "selector_id": _policy_id_from_manifest(manifest),
            "sessionized_manifest_stage": args.sessionized_manifest_stage,
            "sessionized_manifest_path": str(args.stage06_manifest_path),
            "sessionized_manifest_rebuild_hash": manifest.get("deterministic_rebuild_hash"),
            "sessionized_manifest_sha256": manifest_sha256,
            "sessionized_manifest_status": manifest.get("status"),
            "source_market": "binance:futures",
            "splits": {
                "train": train_payload,
                "validation": validation_payload,
            },
            "stage": args.sessionized_manifest_stage,
            "stage06_report": (
                "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/"
                "06-dataset-qa-session-extractor.md"
            ),
            "stage08j_report": (
                "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/"
                "08j-article-session-extractor-dataset.md"
                if args.sessionized_manifest_stage == "08J"
                else None
            ),
            "total_stage06_sessions": manifest.get("total_sessions"),
        }
        source_state = _source_state_payload()
        run_id = args.run_id or _default_run_id(
            config_hash=config.config_hash(),
            dataset_dependency=dataset_dependency,
            source_state=source_state,
        )
        run_dir = args.output_root / run_id
        source_state = {
            **source_state,
            "operator_commands": {
                "resume": _resume_command_for_args(args=args, run_id=run_id),
                "status": (
                    "uv run --extra rl-ml python "
                    "scripts/rl_trading/stage08e_roehub_native_full_training_run.py "
                    f"status --run-dir {run_dir}"
                ),
            },
        }
        candidate_manifest = run_stage08e_roehub_native_training_v1(
            train_sequences=train,
            validation_sequences=validation,
            dataset_dependency=dataset_dependency,
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
    except (RoehubNativeTrainingError, HfOriginalTrainingError) as exc:
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
                "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
                "candidate_manifest_path": candidate_manifest["candidate_manifest_path"],
                "config_hash": candidate_manifest["config_hash"],
                "run_dir": candidate_manifest["run_dir"],
                "run_id": candidate_manifest["run_id"],
                "status": candidate_manifest["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _status_command(*, run_dir: Path) -> int:
    latest_status_path = run_dir / "latest_status.json"
    manifest_path = run_dir / "roehub_native_candidate_manifest.json"
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
        "candidate_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "candidate_manifest_sha256": _file_sha256_hex(manifest_path)
        if manifest_path.exists()
        else None,
        "latest_status": latest,
        "latest_status_path": str(latest_status_path),
        "latest_status_sha256": _file_sha256_hex(latest_status_path),
        "progress_path": str(run_dir / "progress.jsonl"),
        "run_dir": str(run_dir),
        "status": latest.get("status", "unknown"),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


def _load_stage06_split_features(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    dataset_version: str,
    split: str,
    max_sessions: int | None,
    max_artifacts: int | None,
    allow_fixture_hashes: bool,
    accepted_stages: Sequence[str] = ("06",),
) -> tuple[np.ndarray, dict[str, object]]:
    _validate_stage06_manifest(manifest, accepted_stages=accepted_stages)
    entries = [
        entry
        for entry in _split_artifact_entries(manifest)
        if entry.get("dataset_version") == dataset_version and entry.get("split") == split
    ]
    if not entries:
        raise RoehubNativeTrainingError(
            reason="stage06_split_artifacts_not_found",
            field=f"{dataset_version}:{split}",
        )
    entries.sort(key=lambda item: str(item.get("symbol", "")))
    total_entry_sessions = sum(_int_field(entry, "candidate_count") for entry in entries)
    selected_entries = entries if max_artifacts is None else entries[:max_artifacts]
    arrays: list[np.ndarray] = []
    artifact_summary: list[dict[str, object]] = []
    remaining = max_sessions
    selected_session_count = 0
    for entry in selected_entries:
        files = _mapping_field(entry, "files")
        feature_payload = _mapping_field(files, "features")
        feature_path = Path(_string_field(feature_payload, "path"))
        if not feature_path.exists():
            raise RoehubNativeTrainingError(
                reason="stage06_features_file_missing",
                field=str(feature_path),
            )
        sha256 = compute_file_sha256(feature_path)
        expected_sha256 = _string_field(feature_payload, "sha256")
        hash_matches = sha256 == expected_sha256
        if not hash_matches and not allow_fixture_hashes:
            raise RoehubNativeTrainingError(
                reason="stage06_features_hash_mismatch",
                field=str(feature_path),
            )
        features = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
        if features.ndim != 3 or tuple(features.shape[1:]) != (150, len(FEATURE_NAMES_V1)):
            raise RoehubNativeTrainingError(
                reason="stage06_features_shape_mismatch",
                field=str(feature_path),
            )
        candidate_count = _int_field(entry, "candidate_count")
        if features.shape[0] != candidate_count and not allow_fixture_hashes:
            raise RoehubNativeTrainingError(
                reason="stage06_features_candidate_count_mismatch",
                field=str(feature_path),
            )
        selected_count = int(features.shape[0])
        if remaining is not None:
            if remaining <= 0:
                break
            selected_count = min(selected_count, remaining)
            remaining -= selected_count
        if selected_count <= 0:
            continue
        arrays.append(np.ascontiguousarray(features[:selected_count], dtype=np.float32))
        selected_session_count += selected_count
        artifact_summary.append(
            {
                "candidate_count": candidate_count,
                "deterministic_rebuild_hash": entry.get("deterministic_rebuild_hash"),
                "features_sha256": sha256,
                "hash_matches_manifest": hash_matches,
                "selected_session_count": selected_count,
                "symbol": entry.get("symbol"),
            }
        )
    if not arrays:
        raise RoehubNativeTrainingError(
            reason="stage06_split_empty_selection",
            field=f"{dataset_version}:{split}",
        )
    features_out = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    split_artifact_summary_hash = hash_json_payload_v1(
        {
            "artifact_summary": artifact_summary,
            "dataset_version": dataset_version,
            "manifest_sha256": manifest_sha256,
            "split": split,
        }
    )
    return features_out, {
        "dataset_version": dataset_version,
        "full_split_selected": (
            max_artifacts is None
            and max_sessions is None
            and selected_session_count == total_entry_sessions
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_stage": manifest.get("stage"),
        "max_artifacts": max_artifacts,
        "max_sessions": max_sessions,
        "selector_id": _policy_id_from_manifest(manifest),
        "selected_session_count": selected_session_count,
        "split": split,
        "split_artifact_count_selected": len(artifact_summary),
        "split_artifact_count_total": len(entries),
        "split_artifact_summary_hash": split_artifact_summary_hash,
        "total_session_count": total_entry_sessions,
    }


def _validate_stage06_manifest(
    manifest: Mapping[str, Any],
    *,
    accepted_stages: Sequence[str] = ("06",),
) -> None:
    if manifest.get("stage") not in set(accepted_stages):
        raise RoehubNativeTrainingError(reason="unexpected_stage06_manifest_stage")
    if manifest.get("status") != "accepted":
        raise RoehubNativeTrainingError(reason="stage06_manifest_not_accepted")
    if manifest.get("manifest_kind") != SESSIONIZED_DATASET_MANIFEST_KIND_V1:
        raise RoehubNativeTrainingError(reason="unexpected_stage06_manifest_kind")
    if manifest.get("market") != "binance:futures":
        raise RoehubNativeTrainingError(reason="unexpected_stage06_manifest_market")


def _policy_id_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        return None
    value = policy.get("policy_id")
    return str(value) if isinstance(value, str) and value else None


def _split_artifact_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = manifest.get("split_artifacts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RoehubNativeTrainingError(reason="stage06_split_artifacts_not_sequence")
    return tuple(cast(Mapping[str, Any], entry) for entry in entries if isinstance(entry, Mapping))


def _mapping_field(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise RoehubNativeTrainingError(reason="mapping_field_required", field=field)
    return cast(Mapping[str, Any], value)


def _string_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RoehubNativeTrainingError(reason="string_field_required", field=field)
    return value


def _int_field(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoehubNativeTrainingError(reason="int_field_required", field=field)
    return int(value)


def _default_run_id(
    *,
    config_hash: str,
    dataset_dependency: Mapping[str, Any],
    source_state: Mapping[str, Any],
) -> str:
    digest = hash_json_payload_v1(
        {
            "config_hash": config_hash,
            "dataset_dependency": dataset_dependency,
            "source_state": source_state,
            "stage": "08E",
        }
    )
    stage06_hash = str(dataset_dependency["sessionized_manifest_sha256"])[:8]
    return f"stage08e_roehub_native_{stage06_hash}_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    source_files = []
    for relative in SOURCE_STATE_PATHS:
        path = REPO_ROOT / relative
        if path.exists():
            source_files.append(
                {
                    "path": relative,
                    "sha256": _file_sha256_hex(path),
                }
            )
    if not (REPO_ROOT / ".git").exists():
        return {
            "git_unavailable_reason": "snapshot_without_git_directory",
            "source_diff_sha256": None,
            "source_file_hashes": source_files,
            "source_paths": list(SOURCE_STATE_PATHS),
        }
    try:
        head = _git_output("rev-parse", "HEAD")
        status = _git_output("status", "--short", "--", *SOURCE_STATE_PATHS)
        diff = _git_output("diff", "--", *SOURCE_STATE_PATHS)
        git_payload: dict[str, object] = {
            "git_head": head,
            "git_status_short": status.splitlines(),
            "source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        }
    except Exception as exc:
        git_payload = {
            "git_unavailable_reason": type(exc).__name__,
            "source_diff_sha256": None,
        }
    return {
        **git_payload,
        "source_file_hashes": source_files,
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
        "scripts/rl_trading/stage08e_roehub_native_full_training_run.py",
        "run",
        "--resume",
        f"--run-id {run_id}",
        f"--stage06-manifest-path {args.stage06_manifest_path}",
        f"--output-root {args.output_root}",
        f"--dataset-version {args.dataset_version}",
        f"--train-split {args.train_split}",
        f"--validation-split {args.validation_split}",
        f"--episodes {args.episodes}",
        f"--agent-history-len {args.agent_history_len}",
        f"--agent-session-len {args.agent_session_len}",
        f"--validation-every-episodes {args.validation_every_episodes}",
        f"--checkpoint-every-episodes {args.checkpoint_every_episodes}",
        f"--progress-emit-every-episodes {args.progress_emit_every_episodes}",
        f"--progress-emit-every-sec {args.progress_emit_every_sec}",
        f"--device-policy {args.device_policy}",
        f"--torch-num-threads {args.torch_num_threads}",
        f"--torch-num-interop-threads {args.torch_num_interop_threads}",
    ]
    if args.allow_fixture_hashes:
        parts.append("--allow-fixture-hashes")
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
    parser = argparse.ArgumentParser(description="Run or inspect Stage 08E Roehub-native training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run or resume Stage 08E Roehub-native training.")
    run.add_argument("--stage06-manifest-path", type=Path, default=DEFAULT_STAGE06_MANIFEST_PATH)
    run.add_argument(
        "--sessionized-manifest-stage",
        choices=("06", "08J"),
        default="06",
    )
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id", type=str, default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-fixture-hashes", action="store_true")
    run.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    run.add_argument("--train-split", type=str, default=DEFAULT_TRAIN_SPLIT)
    run.add_argument("--validation-split", type=str, default=DEFAULT_VALIDATION_SPLIT)
    run.add_argument("--max-train-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--max-validation-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--max-train-artifacts", type=_optional_positive_int, default=None)
    run.add_argument("--max-validation-artifacts", type=_optional_positive_int, default=None)
    run.add_argument("--validation-max-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--seed", type=int, default=25)
    run.add_argument("--agent-history-len", type=int, default=30)
    run.add_argument("--agent-session-len", type=int, default=10)
    run.add_argument("--episodes", type=int, default=55_000)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--learning-rate", type=float, default=1e-4)
    run.add_argument("--train-start", type=int, default=10_000)
    run.add_argument("--replay-capacity", type=int, default=230_000)
    run.add_argument("--target-update-freq", type=int, default=100)
    run.add_argument("--eps-decay-frames", type=int, default=50_000)
    run.add_argument("--validation-every-episodes", type=int, default=1_000)
    run.add_argument("--checkpoint-every-episodes", type=int, default=1_000)
    run.add_argument("--progress-emit-every-episodes", type=int, default=100)
    run.add_argument("--progress-emit-every-sec", type=int, default=300)
    run.add_argument("--torch-num-threads", type=int, default=DEFAULT_TORCH_NUM_THREADS)
    run.add_argument("--torch-num-interop-threads", type=int, default=1)
    run.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="mps_preferred_cpu_fallback",
    )
    run.add_argument("--generated-at-utc", type=str, default=None)

    status = subparsers.add_parser("status", help="Print latest Stage 08E run status.")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
