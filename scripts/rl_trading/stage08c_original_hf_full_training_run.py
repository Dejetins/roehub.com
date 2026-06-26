from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    FEATURE_NAMES_V1,
    HF_DATASET_REPO_ID_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    STAGE08C_RUNTIME_ARTIFACT_SUBDIR_V1,
    HfDatasetSplitSpec,
    HfOriginalTrainingConfig,
    HfOriginalTrainingError,
    UpstreamAlphaConfig,
    compute_file_sha256,
    expected_hf_dataset_manifest_hash_v1,
    expected_hf_split_specs_v1,
    hash_json_payload_v1,
    run_stage08c_hf_original_training_v1,
)

DEFAULT_HF_DATASET_DIR = (
    Path("/opt/roehub/state/rl_trading/hf_reproducibility/dataset")
    / "ResearchRL"
    / "open-rl-trading-binance-dataset"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "training_runs"
    / STAGE08C_RUNTIME_ARTIFACT_SUBDIR_V1
)
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/hf_original_training.py",
    "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08c_original_hf_full_training_run.py",
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
        config = HfOriginalTrainingConfig(
            alpha=alpha,
            planned_episodes=args.episodes,
            validation_every_episodes=args.validation_every_episodes,
            checkpoint_every_episodes=args.checkpoint_every_episodes,
            progress_emit_every_episodes=args.progress_emit_every_episodes,
            progress_emit_every_sec=args.progress_emit_every_sec,
            validation_max_sessions=args.validation_max_sessions,
            device_policy=args.device_policy,
        )
        specs = {spec.split_name: spec for spec in expected_hf_split_specs_v1()}
        train, train_payload = _load_hf_split_features(
            dataset_dir=args.dataset_dir,
            split_spec=specs["train"],
            max_sessions=args.max_train_sessions,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        validation, validation_payload = _load_hf_split_features(
            dataset_dir=args.dataset_dir,
            split_spec=specs["validation"],
            max_sessions=args.max_validation_sessions,
            allow_fixture_hashes=args.allow_fixture_hashes,
        )
        dataset_dependency = {
            "allow_fixture_hashes": bool(args.allow_fixture_hashes),
            "dataset_dir": str(args.dataset_dir),
            "dataset_manifest_hash": expected_hf_dataset_manifest_hash_v1(),
            "dataset_repo_id": HF_DATASET_REPO_ID_V1,
            "source_market": "binance:futures",
            "splits": {
                "train": train_payload,
                "validation": validation_payload,
            },
            "stage": "04",
            "stage04_report": (
                "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/"
                "04-hf-reproducibility.md"
            ),
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
                    "scripts/rl_trading/stage08c_original_hf_full_training_run.py "
                    f"status --run-dir {run_dir}"
                ),
            },
        }
        manifest = run_stage08c_hf_original_training_v1(
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
    except HfOriginalTrainingError as exc:
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
                "candidate_manifest_hash": manifest["candidate_manifest_hash"],
                "candidate_manifest_path": manifest["candidate_manifest_path"],
                "config_hash": manifest["config_hash"],
                "run_dir": manifest["run_dir"],
                "run_id": manifest["run_id"],
                "status": manifest["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _status_command(*, run_dir: Path) -> int:
    latest_status_path = run_dir / "latest_status.json"
    manifest_path = run_dir / "hf_original_candidate_manifest.json"
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


def _load_hf_split_features(
    *,
    dataset_dir: Path,
    split_spec: HfDatasetSplitSpec,
    max_sessions: int | None,
    allow_fixture_hashes: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    file_path = dataset_dir / split_spec.file_name
    if not file_path.exists():
        raise HfOriginalTrainingError(reason="missing_hf_split_file", field=str(file_path))
    sha256 = compute_file_sha256(file_path)
    if sha256 != split_spec.expected_sha256 and not allow_fixture_hashes:
        raise HfOriginalTrainingError(reason="hf_split_hash_mismatch", field=split_spec.file_name)
    with np.load(file_path, allow_pickle=True) as archive:
        keys = sorted(
            (key for key in archive.files if key.startswith("fetcher_")),
            key=_fetcher_key_sort_value,
        )
        total_sessions = len(keys)
        if sha256 == split_spec.expected_sha256 and total_sessions != split_spec.observed_sessions:
            raise HfOriginalTrainingError(
                reason="hf_split_session_count_mismatch",
                field=split_spec.file_name,
            )
        selected_keys = keys if max_sessions is None else keys[:max_sessions]
        if not selected_keys:
            raise HfOriginalTrainingError(
                reason="hf_split_empty_selection",
                field=split_spec.file_name,
            )
        features = np.empty(
            (len(selected_keys), 150, len(FEATURE_NAMES_V1)),
            dtype=np.float32,
        )
        for row_idx, key in enumerate(selected_keys):
            arr = np.asarray(archive[key], dtype=np.float32)
            if arr.shape != (150, len(FEATURE_NAMES_V1)):
                raise HfOriginalTrainingError(reason="hf_session_shape_mismatch", field=key)
            features[row_idx] = arr
        keys_map_count = _keys_map_count(archive)
    return np.ascontiguousarray(features, dtype=np.float32), {
        "expected_sha256": split_spec.expected_sha256,
        "file_name": split_spec.file_name,
        "file_path": str(file_path),
        "hash_matches_expected": sha256 == split_spec.expected_sha256,
        "keys_map_count": keys_map_count,
        "selected_session_count": len(selected_keys),
        "sha256": sha256,
        "split_name": split_spec.split_name,
        "total_session_count": total_sessions,
    }


def _keys_map_count(archive: Any) -> int | None:
    if "_keys_map_" not in archive.files:
        return None
    value = archive["_keys_map_"]
    try:
        item = value.item()
    except Exception:
        return None
    if isinstance(item, Mapping):
        return len(item)
    return None


def _fetcher_key_sort_value(value: str) -> tuple[int, str]:
    try:
        return int(value.split("_", 1)[1]), value
    except Exception:
        return sys.maxsize, value


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
            "stage": "08C",
        }
    )
    return f"stage08c_hf_original_{digest[:20]}"


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
        "scripts/rl_trading/stage08c_original_hf_full_training_run.py",
        "run",
        "--resume",
        f"--run-id {run_id}",
        f"--dataset-dir {args.dataset_dir}",
        f"--output-root {args.output_root}",
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
    parser = argparse.ArgumentParser(description="Run or inspect Stage 08C original HF training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run or resume Stage 08C original HF training.")
    run.add_argument("--dataset-dir", type=Path, default=DEFAULT_HF_DATASET_DIR)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id", type=str, default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-fixture-hashes", action="store_true")
    run.add_argument("--max-train-sessions", type=_optional_positive_int, default=None)
    run.add_argument("--max-validation-sessions", type=_optional_positive_int, default=None)
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
    run.add_argument("--torch-num-threads", type=int, default=1)
    run.add_argument("--torch-num-interop-threads", type=int, default=1)
    run.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="mps_preferred_cpu_fallback",
    )
    run.add_argument("--generated-at-utc", type=str, default=None)

    status = subparsers.add_parser("status", help="Print latest Stage 08C run status.")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
