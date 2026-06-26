from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.rl_trading import stage08c_original_hf_full_training_run as hf_train_cli  # noqa: E402
from scripts.rl_trading import stage08e_roehub_native_full_training_run as native_train_cli  # noqa: E402
from scripts.rl_trading import stage08g_cpu_optuna_calibration as optuna_cli  # noqa: E402
from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    hash_json_payload_v1,
)

STAGE08G_DUAL_BRANCH_RUN_KIND_V1 = "rl_trading_stage08g_dual_branch_cpu_run"
STAGE08G_DUAL_BRANCH_SCHEMA_VERSION_V1 = 1
DEFAULT_TRAINING_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "training_runs"
    / optuna_cli.STAGE08G_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_ORCHESTRATOR_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / optuna_cli.STAGE08G_RUNTIME_ARTIFACT_SUBDIR_V1
    / "dual_branch_runs"
)


class Stage08GDualBranchRunError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except Stage08GDualBranchRunError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] in {"completed", "accepted_for_research", "planned"} else 2


def _run(args: argparse.Namespace) -> dict[str, Any]:
    generated = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    run_id = args.run_id or _default_run_id(args=args)
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_training_output_root = args.hf_training_output_root or (
        DEFAULT_TRAINING_OUTPUT_ROOT / "hf_original"
    )
    native_training_output_root = args.native_training_output_root or (
        DEFAULT_TRAINING_OUTPUT_ROOT / "roehub_native"
    )
    evaluation_output_root = args.evaluation_output_root or optuna_cli.DEFAULT_OUTPUT_ROOT

    steps: list[dict[str, Any]] = []
    try:
        hf_train_payload = _run_step(
            name="hf_original_cpu_training",
            command=_hf_training_command(
                args=args,
                generated_at_utc=generated,
                output_root=hf_training_output_root,
            ),
            steps=steps,
            dry_run=args.dry_run,
        )
        hf_candidate_path = _path_from_payload(
            hf_train_payload,
            key="candidate_manifest_path",
            step="hf_original_cpu_training",
            dry_run=args.dry_run,
        )
        hf_candidate_sha256 = None if args.dry_run else _file_sha256_hex(hf_candidate_path)

        hf_optuna_payload = _run_step(
            name="hf_original_cpu_optuna",
            command=_optuna_command(
                args=args,
                branch="hf_original",
                candidate_manifest_path=hf_candidate_path,
                candidate_manifest_sha256=hf_candidate_sha256,
                generated_at_utc=generated,
                output_root=evaluation_output_root,
            ),
            steps=steps,
            dry_run=args.dry_run,
        )

        native_train_payload = _run_step(
            name="roehub_native_cpu_training",
            command=_native_training_command(
                args=args,
                generated_at_utc=generated,
                output_root=native_training_output_root,
            ),
            steps=steps,
            dry_run=args.dry_run,
        )
        native_candidate_path = _path_from_payload(
            native_train_payload,
            key="candidate_manifest_path",
            step="roehub_native_cpu_training",
            dry_run=args.dry_run,
        )
        native_candidate_sha256 = None if args.dry_run else _file_sha256_hex(native_candidate_path)

        native_optuna_payload = _run_step(
            name="roehub_native_cpu_optuna",
            command=_optuna_command(
                args=args,
                branch="roehub_native",
                candidate_manifest_path=native_candidate_path,
                candidate_manifest_sha256=native_candidate_sha256,
                generated_at_utc=generated,
                output_root=evaluation_output_root,
            ),
            steps=steps,
            dry_run=args.dry_run,
        )
        summary = _summary_payload(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            generated_at_utc=generated,
            steps=steps,
            hf_training=hf_train_payload,
            hf_optuna=hf_optuna_payload,
            native_training=native_train_payload,
            native_optuna=native_optuna_payload,
        )
    except Stage08GDualBranchRunError as exc:
        summary = _blocked_summary_payload(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            generated_at_utc=generated,
            steps=steps,
            reason=exc.reason,
            field=exc.field,
        )
    summary_path = run_dir / "stage08g_dual_branch_cpu_run_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "status": summary["status"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256_hex(summary_path),
    }


def _run_step(
    *,
    name: str,
    command: Sequence[str],
    steps: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    step_record: dict[str, Any] = {
        "command": _display_command(command),
        "name": name,
        "status": "planned" if dry_run else "running",
    }
    steps.append(step_record)
    if dry_run:
        payload = {"status": "planned"}
        step_record["payload"] = payload
        return payload
    completed = _run_command_capture(command)
    step_record["returncode"] = completed.returncode
    step_record["stderr_tail"] = _tail_text(completed.stderr)
    step_record["stdout_tail"] = _tail_text(completed.stdout)
    payload = _json_from_stdout(completed.stdout, step=name)
    step_record["payload"] = payload
    if completed.returncode != 0:
        step_record["status"] = "blocked"
        raise Stage08GDualBranchRunError(
            reason="stage08g_step_failed",
            field=f"{name}: returncode={completed.returncode}",
        )
    step_record["status"] = str(payload.get("status", "completed"))
    return payload


def _run_command_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(command),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _hf_training_command(
    *,
    args: argparse.Namespace,
    generated_at_utc: datetime,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/rl_trading/stage08c_original_hf_full_training_run.py",
        "run",
        "--dataset-dir",
        str(args.hf_dataset_dir),
        "--output-root",
        str(output_root),
        "--device-policy",
        "cpu_only_deterministic",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--train-start",
        str(args.train_start),
        "--replay-capacity",
        str(args.replay_capacity),
        "--target-update-freq",
        str(args.target_update_freq),
        "--eps-decay-frames",
        str(args.eps_decay_frames),
        "--validation-every-episodes",
        str(args.validation_every_episodes),
        "--checkpoint-every-episodes",
        str(args.checkpoint_every_episodes),
        "--progress-emit-every-episodes",
        str(args.progress_emit_every_episodes),
        "--progress-emit-every-sec",
        str(args.progress_emit_every_sec),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--torch-num-interop-threads",
        str(args.torch_num_interop_threads),
        "--generated-at-utc",
        _format_utc(generated_at_utc),
    ]
    _append_optional(command, "--run-id", args.hf_training_run_id)
    _append_flag(command, "--resume", args.resume_training)
    _append_flag(command, "--allow-fixture-hashes", args.allow_fixture_hashes)
    _append_optional(command, "--max-train-sessions", args.hf_max_train_sessions)
    _append_optional(command, "--max-validation-sessions", args.hf_max_validation_sessions)
    _append_optional(command, "--validation-max-sessions", args.validation_max_sessions)
    return command


def _native_training_command(
    *,
    args: argparse.Namespace,
    generated_at_utc: datetime,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/rl_trading/stage08e_roehub_native_full_training_run.py",
        "run",
        "--stage06-manifest-path",
        str(args.stage06_manifest_path),
        "--dataset-version",
        args.dataset_version,
        "--train-split",
        args.native_train_split,
        "--validation-split",
        args.native_validation_split,
        "--output-root",
        str(output_root),
        "--device-policy",
        "cpu_only_deterministic",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--train-start",
        str(args.train_start),
        "--replay-capacity",
        str(args.replay_capacity),
        "--target-update-freq",
        str(args.target_update_freq),
        "--eps-decay-frames",
        str(args.eps_decay_frames),
        "--validation-every-episodes",
        str(args.validation_every_episodes),
        "--checkpoint-every-episodes",
        str(args.checkpoint_every_episodes),
        "--progress-emit-every-episodes",
        str(args.progress_emit_every_episodes),
        "--progress-emit-every-sec",
        str(args.progress_emit_every_sec),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--torch-num-interop-threads",
        str(args.torch_num_interop_threads),
        "--generated-at-utc",
        _format_utc(generated_at_utc),
    ]
    _append_optional(command, "--run-id", args.native_training_run_id)
    _append_flag(command, "--resume", args.resume_training)
    _append_flag(command, "--allow-fixture-hashes", args.allow_fixture_hashes)
    _append_optional(command, "--max-train-sessions", args.native_max_train_sessions)
    _append_optional(command, "--max-validation-sessions", args.native_max_validation_sessions)
    _append_optional(command, "--max-train-artifacts", args.native_max_train_artifacts)
    _append_optional(command, "--max-validation-artifacts", args.native_max_validation_artifacts)
    _append_optional(command, "--validation-max-sessions", args.validation_max_sessions)
    return command


def _optuna_command(
    *,
    args: argparse.Namespace,
    branch: str,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str | None,
    generated_at_utc: datetime,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/rl_trading/stage08g_cpu_optuna_calibration.py",
        "--branch",
        branch,
        "--candidate-manifest",
        str(candidate_manifest_path),
        "--output-root",
        str(output_root),
        "--trials",
        str(args.trials),
        "--jobs",
        str(args.jobs),
        "--optuna-seed",
        str(args.optuna_seed),
        "--checkpoint-name",
        args.checkpoint_name,
        "--selection-strategy",
        args.selection_strategy,
        "--simple-threshold-return",
        str(args.simple_threshold_return),
        "--ensemble-n-samples",
        str(args.ensemble_n_samples),
        "--ensemble-max-sigma",
        str(args.ensemble_max_sigma),
        "--max-parallel-sessions",
        str(args.max_parallel_sessions),
        "--position-fraction",
        str(args.position_fraction),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--torch-num-interop-threads",
        str(args.torch_num_interop_threads),
        "--deterministic-random-seed",
        str(args.deterministic_random_seed),
        "--generated-at-utc",
        _format_utc(generated_at_utc),
        "--hf-dataset-dir",
        str(args.hf_dataset_dir),
        "--hf-calibration-split",
        args.hf_calibration_split,
        "--hf-final-split",
        args.hf_final_split,
        "--stage06-manifest-path",
        str(args.stage06_manifest_path),
        "--dataset-version",
        args.dataset_version,
        "--native-calibration-split",
        args.native_calibration_split,
        "--native-final-split",
        args.native_final_split,
    ]
    if candidate_manifest_sha256 is not None:
        command.extend(["--expected-candidate-manifest-sha256", candidate_manifest_sha256])
    _append_flag(command, "--allow-fixture-hashes", args.allow_fixture_hashes)
    _append_optional(command, "--max-calibration-sessions", args.max_calibration_sessions)
    _append_optional(command, "--max-final-sessions", args.max_final_sessions)
    _append_optional(command, "--max-calibration-artifacts", args.max_calibration_artifacts)
    _append_optional(command, "--max-final-artifacts", args.max_final_artifacts)
    return command


def _summary_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    generated_at_utc: datetime,
    steps: Sequence[Mapping[str, Any]],
    hf_training: Mapping[str, Any],
    hf_optuna: Mapping[str, Any],
    native_training: Mapping[str, Any],
    native_optuna: Mapping[str, Any],
) -> dict[str, Any]:
    hf_stage09 = hf_optuna.get("status") == "accepted_for_research"
    native_stage09 = native_optuna.get("status") == "accepted_for_research"
    status = (
        "planned"
        if args.dry_run
        else "accepted_for_research"
        if hf_stage09 and native_stage09
        else "completed"
    )
    payload = {
        "artifact_kind": STAGE08G_DUAL_BRANCH_RUN_KIND_V1,
        "branch_order": ["hf_original", "roehub_native"],
        "branches": {
            "hf_original": {
                "candidate_manifest_path": hf_training.get("candidate_manifest_path"),
                "optuna_summary_path": hf_optuna.get("summary_path"),
                "stage09_allowed": hf_stage09,
                "training_run_dir": hf_training.get("run_dir"),
            },
            "roehub_native": {
                "candidate_manifest_path": native_training.get("candidate_manifest_path"),
                "optuna_summary_path": native_optuna.get("summary_path"),
                "stage09_allowed": native_stage09,
                "training_run_dir": native_training.get("run_dir"),
            },
        },
        "execution_mode": args.execution_mode,
        "generated_at_utc": _format_utc(generated_at_utc),
        "methodology": {
            "device_policy": "cpu_only_deterministic",
            "hf_training_dataset": "original_hf_stage04_dataset",
            "native_training_dataset": "accepted_stage06_roehub_sessionized_dataset",
            "optuna_trials_default_source": "Habr workflow command uses --trials 100 --jobs 1",
            "parallel_training": False,
            "parallel_training_reason": (
                "single host CPU reproducibility; parallel training would compete for CPU/RAM/IO"
            ),
            "stage06_training_split": args.native_train_split,
            "stage06_validation_split": args.native_validation_split,
            "upstream_source_sha": "f71130903f8237351164f4b875494185465bf1ea",
        },
        "run_dir": str(run_dir),
        "run_id": run_id,
        "schema_version": STAGE08G_DUAL_BRANCH_SCHEMA_VERSION_V1,
        "stage": "08G",
        "stage09_allowed": bool(hf_stage09 and native_stage09),
        "status": status,
        "steps": list(steps),
    }
    return {**payload, "summary_hash": hash_json_payload_v1(payload)}


def _blocked_summary_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    generated_at_utc: datetime,
    steps: Sequence[Mapping[str, Any]],
    reason: str,
    field: str | None,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": STAGE08G_DUAL_BRANCH_RUN_KIND_V1,
        "blocked_reason": reason,
        "blocked_field": field,
        "execution_mode": args.execution_mode,
        "generated_at_utc": _format_utc(generated_at_utc),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "schema_version": STAGE08G_DUAL_BRANCH_SCHEMA_VERSION_V1,
        "stage": "08G",
        "stage09_allowed": False,
        "status": "blocked",
        "steps": list(steps),
    }
    return {**payload, "summary_hash": hash_json_payload_v1(payload)}


def _path_from_payload(
    payload: Mapping[str, Any],
    *,
    key: str,
    step: str,
    dry_run: bool,
) -> Path:
    if dry_run:
        return Path(f"/planned/{step}/{key}.json")
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise Stage08GDualBranchRunError(reason="step_payload_path_missing", field=f"{step}:{key}")
    path = Path(value)
    if not path.exists():
        raise Stage08GDualBranchRunError(reason="step_payload_path_not_found", field=str(path))
    return path


def _json_from_stdout(stdout: str, *, step: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise Stage08GDualBranchRunError(reason="step_json_stdout_missing", field=step)


def _default_run_id(*, args: argparse.Namespace) -> str:
    digest = hash_json_payload_v1(
        {
            "dataset_version": args.dataset_version,
            "episodes": args.episodes,
            "execution_mode": args.execution_mode,
            "hf_dataset_dir": str(args.hf_dataset_dir),
            "max_parallel_sessions": args.max_parallel_sessions,
            "native_final_split": args.native_final_split,
            "native_train_split": args.native_train_split,
            "native_validation_split": args.native_validation_split,
            "selection_strategy": args.selection_strategy,
            "stage": "08G",
            "stage06_manifest_path": str(args.stage06_manifest_path),
            "trials": args.trials,
        }
    )
    return f"stage08g_dual_branch_cpu_{digest[:20]}"


def _append_optional(command: list[str], option: str, value: object | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def _append_flag(command: list[str], option: str, value: bool) -> None:
    if value:
        command.append(option)


def _display_command(command: Sequence[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/.:=+")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _tail_text(value: str, *, max_chars: int = 4000) -> str:
    return value[-max_chars:] if len(value) > max_chars else value


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
        description="Run Stage 08G sequential CPU training, Optuna and final holdout."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ORCHESTRATOR_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument(
        "--execution-mode",
        choices=("sequential_cpu",),
        default="sequential_cpu",
    )

    parser.add_argument("--hf-dataset-dir", type=Path, default=hf_train_cli.DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--hf-training-output-root", type=Path, default=None)
    parser.add_argument("--hf-training-run-id", type=str, default=None)
    parser.add_argument("--hf-max-train-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--hf-max-validation-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--hf-calibration-split", type=str, default="test")
    parser.add_argument("--hf-final-split", type=str, default="backtest")

    parser.add_argument(
        "--stage06-manifest-path",
        type=Path,
        default=native_train_cli.DEFAULT_STAGE06_MANIFEST_PATH,
    )
    parser.add_argument("--native-training-output-root", type=Path, default=None)
    parser.add_argument("--native-training-run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=native_train_cli.DEFAULT_DATASET_VERSION,
    )
    parser.add_argument(
        "--native-train-split",
        type=str,
        default=native_train_cli.DEFAULT_TRAIN_SPLIT,
    )
    parser.add_argument(
        "--native-validation-split",
        type=str,
        default=native_train_cli.DEFAULT_VALIDATION_SPLIT,
    )
    parser.add_argument("--native-calibration-split", type=str, default="test")
    parser.add_argument("--native-final-split", type=str, default="backtest")
    parser.add_argument("--native-max-train-sessions", type=_optional_positive_int, default=None)
    parser.add_argument(
        "--native-max-validation-sessions",
        type=_optional_positive_int,
        default=None,
    )
    parser.add_argument("--native-max-train-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument(
        "--native-max-validation-artifacts",
        type=_optional_positive_int,
        default=None,
    )

    parser.add_argument("--evaluation-output-root", type=Path, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--resume-training", action="store_true")

    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=55_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-start", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=230_000)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--eps-decay-frames", type=int, default=50_000)
    parser.add_argument("--validation-every-episodes", type=int, default=1_000)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=1_000)
    parser.add_argument("--progress-emit-every-episodes", type=int, default=100)
    parser.add_argument("--progress-emit-every-sec", type=int, default=300)
    parser.add_argument("--validation-max-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)

    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--optuna-seed", type=int, default=1708)
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
    parser.add_argument("--deterministic-random-seed", type=int, default=806)
    parser.add_argument("--max-calibration-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-final-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-calibration-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--max-final-artifacts", type=_optional_positive_int, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
