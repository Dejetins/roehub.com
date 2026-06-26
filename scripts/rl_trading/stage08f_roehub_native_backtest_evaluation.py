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
    FEATURE_NAMES_V1,
    SESSIONIZED_DATASET_MANIFEST_KIND_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    STAGE08F_RUNTIME_ARTIFACT_SUBDIR_V1,
    RoehubNativeEvaluationConfig,
    RoehubNativeEvaluationError,
    RoehubNativeSplitData,
    UpstreamAlphaConfig,
    compute_file_sha256,
    hash_json_payload_v1,
    run_stage08f_roehub_native_evaluation_v1,
)
from trading.contexts.rl_trading.domain.hf_original_evaluation import (  # noqa: E402
    HfOriginalEvaluationError,
)

DEFAULT_STAGE06_MANIFEST_PATH = Path(
    "/opt/roehub/state/rl_trading/datasets/"
    "stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/training_runs")
    / "stage08e_roehub_native_full_training_run_v1"
    / "full"
    / "stage08e_roehub_native_full"
    / "roehub_native_candidate_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST_SHA256 = (
    "c130ca5ede6f0e6f1d57e7940b385a52dbfab616bca0b01b2771f6de46613cdc"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08F_RUNTIME_ARTIFACT_SUBDIR_V1
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_BACKTEST_SPLIT = "backtest"
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/hf_original_evaluation.py",
    "src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py",
    "src/trading/contexts/rl_trading/domain/roehub_native_training.py",
    "src/trading/contexts/rl_trading/domain/sessionized_dataset.py",
    "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run_command(args)
    except (RoehubNativeEvaluationError, HfOriginalEvaluationError) as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] == "accepted_for_research" else 2


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    candidate_manifest_sha256 = _file_sha256_hex(args.candidate_manifest)
    if (
        args.expected_candidate_manifest_sha256
        and candidate_manifest_sha256 != args.expected_candidate_manifest_sha256
    ):
        raise RoehubNativeEvaluationError(
            reason="candidate_manifest_sha256_mismatch",
            field=str(args.candidate_manifest),
        )
    candidate_manifest = _read_json(args.candidate_manifest)
    stage06_manifest = _read_json(args.stage06_manifest_path)
    stage06_manifest_sha256 = compute_file_sha256(args.stage06_manifest_path)
    test_split = _load_stage06_split(
        manifest=stage06_manifest,
        manifest_path=args.stage06_manifest_path,
        manifest_sha256=stage06_manifest_sha256,
        dataset_version=args.dataset_version,
        split=args.test_split,
        max_sessions=args.max_test_sessions,
        max_artifacts=args.max_test_artifacts,
        allow_fixture_hashes=args.allow_fixture_hashes,
    )
    backtest_split = _load_stage06_split(
        manifest=stage06_manifest,
        manifest_path=args.stage06_manifest_path,
        manifest_sha256=stage06_manifest_sha256,
        dataset_version=args.dataset_version,
        split=args.backtest_split,
        max_sessions=args.max_backtest_sessions,
        max_artifacts=args.max_backtest_artifacts,
        allow_fixture_hashes=args.allow_fixture_hashes,
    )
    alpha = UpstreamAlphaConfig(
        long_action_threshold=args.long_action_threshold,
        short_action_threshold=args.short_action_threshold,
        close_action_threshold=args.close_action_threshold,
        use_risk_management=args.use_risk_management,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        trailing_stop=args.trailing_stop,
        ensemble_n_samples=args.ensemble_n_samples,
        ensemble_max_sigma=args.ensemble_max_sigma,
        max_parallel_sessions=args.max_parallel_sessions,
        position_fraction=args.position_fraction,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )
    config = RoehubNativeEvaluationConfig(
        alpha=alpha,
        checkpoint_name=args.checkpoint_name,
        selection_strategy=args.selection_strategy,
        device_policy=args.device_policy,
        test_max_sessions=args.max_test_sessions,
        backtest_max_sessions=args.max_backtest_sessions,
        simple_threshold_return=args.simple_threshold_return,
        deterministic_random_seed=args.deterministic_random_seed,
    )
    source_state = _source_state_payload()
    run_id = args.run_id or _default_run_id(
        candidate_manifest_sha256=candidate_manifest_sha256,
        test_split=test_split,
        backtest_split=backtest_split,
        config=config,
        source_state=source_state,
    )
    manifest = run_stage08f_roehub_native_evaluation_v1(
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=candidate_manifest_sha256,
        test_split=test_split,
        backtest_split=backtest_split,
        output_root=args.output_root,
        run_id=run_id,
        config=config,
        generated_at_utc=(
            _parse_utc(args.generated_at_utc)
            if args.generated_at_utc is not None
            else datetime.now(UTC).replace(microsecond=0)
        ),
        code_version=source_state,
    )
    return {
        "evaluation_hash": manifest["evaluation_hash"],
        "evaluation_manifest_path": manifest["evaluation_manifest_path"],
        "evaluation_manifest_sha256": _file_sha256_hex(Path(manifest["evaluation_manifest_path"])),
        "native_research_verdict": manifest["native_research_verdict"],
        "research_candidate_save_allowed": manifest["research_candidate_save_allowed"],
        "run_dir": manifest["run_dir"],
        "run_id": manifest["run_id"],
        "stage09_allowed": manifest["stage09_handoff"]["allowed"],
        "status": manifest["status"],
    }


def _load_stage06_split(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    dataset_version: str,
    split: str,
    max_sessions: int | None,
    max_artifacts: int | None,
    allow_fixture_hashes: bool,
) -> RoehubNativeSplitData:
    _validate_stage06_manifest(manifest)
    entries = [
        entry
        for entry in _split_artifact_entries(manifest)
        if entry.get("dataset_version") == dataset_version and entry.get("split") == split
    ]
    if not entries:
        raise RoehubNativeEvaluationError(
            reason="stage06_split_artifacts_not_found",
            field=f"{dataset_version}:{split}",
        )
    entries.sort(key=lambda item: str(item.get("symbol", "")))
    total_entry_sessions = sum(_int_field(entry, "candidate_count") for entry in entries)
    selected_entries = entries if max_artifacts is None else entries[:max_artifacts]
    arrays: list[np.ndarray] = []
    symbols: list[str] = []
    signal_times: list[str | None] = []
    volatility_scores: list[float | None] = []
    artifact_summary: list[dict[str, object]] = []
    remaining = max_sessions
    selected_session_count = 0
    for entry in selected_entries:
        files = _mapping_field(entry, "files")
        feature_payload = _mapping_field(files, "features")
        signal_payload = _mapping_field(files, "signal_time_ms")
        metadata_payload = _mapping_field(files, "metadata")
        feature_path = Path(_string_field(feature_payload, "path"))
        signal_path = Path(_string_field(signal_payload, "path"))
        metadata_path = Path(_string_field(metadata_payload, "path"))
        for path in (feature_path, signal_path, metadata_path):
            if not path.exists():
                raise RoehubNativeEvaluationError(
                    reason="stage06_artifact_file_missing",
                    field=str(path),
                )
        feature_sha256 = compute_file_sha256(feature_path)
        signal_sha256 = compute_file_sha256(signal_path)
        metadata_sha256 = compute_file_sha256(metadata_path)
        hash_matches = (
            feature_sha256 == _string_field(feature_payload, "sha256")
            and signal_sha256 == _string_field(signal_payload, "sha256")
            and metadata_sha256 == _string_field(metadata_payload, "sha256")
        )
        if not hash_matches and not allow_fixture_hashes:
            raise RoehubNativeEvaluationError(
                reason="stage06_artifact_hash_mismatch",
                field=str(feature_path),
            )
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        signal_time_ms = np.asarray(np.load(signal_path), dtype=np.int64)
        sessions = _session_metadata(metadata_path)
        if features.ndim != 3 or tuple(features.shape[1:]) != (150, len(FEATURE_NAMES_V1)):
            raise RoehubNativeEvaluationError(
                reason="stage06_features_shape_mismatch",
                field=str(feature_path),
            )
        candidate_count = _int_field(entry, "candidate_count")
        if (
            features.shape[0] != candidate_count
            or signal_time_ms.shape[0] != candidate_count
            or len(sessions) != candidate_count
        ) and not allow_fixture_hashes:
            raise RoehubNativeEvaluationError(
                reason="stage06_artifact_candidate_count_mismatch",
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
        symbol = str(entry.get("symbol", "UNKNOWN")).upper()
        symbols.extend(symbol for _ in range(selected_count))
        signal_times.extend(_format_ms_utc(int(value)) for value in signal_time_ms[:selected_count])
        volatility_scores.extend(
            _optional_float(item.get("volatility_score"))
            for item in sessions[:selected_count]
        )
        selected_session_count += selected_count
        artifact_summary.append(
            {
                "candidate_count": candidate_count,
                "deterministic_rebuild_hash": entry.get("deterministic_rebuild_hash"),
                "features_sha256": feature_sha256,
                "hash_matches_manifest": hash_matches,
                "metadata_sha256": metadata_sha256,
                "selected_session_count": selected_count,
                "signal_time_ms_sha256": signal_sha256,
                "symbol": symbol,
            }
        )
    if not arrays:
        raise RoehubNativeEvaluationError(
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
    source_payload = {
        "allow_fixture_hashes": bool(allow_fixture_hashes),
        "dataset_version": dataset_version,
        "full_split_selected": (
            max_artifacts is None
            and max_sessions is None
            and selected_session_count == total_entry_sessions
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "max_artifacts": max_artifacts,
        "max_sessions": max_sessions,
        "selected_session_count": selected_session_count,
        "split_artifact_count_selected": len(artifact_summary),
        "split_artifact_count_total": len(entries),
        "split_artifact_summary_hash": split_artifact_summary_hash,
        "split_name": split,
        "total_session_count": total_entry_sessions,
    }
    return RoehubNativeSplitData(
        split_name=split,
        sequences=features_out,
        symbols=tuple(symbols),
        signal_times_utc=tuple(signal_times),
        source_payload=source_payload,
        volatility_scores=tuple(volatility_scores),
    )


def _validate_stage06_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "06":
        raise RoehubNativeEvaluationError(reason="unexpected_stage06_manifest_stage")
    if manifest.get("status") != "accepted":
        raise RoehubNativeEvaluationError(reason="stage06_manifest_not_accepted")
    if manifest.get("manifest_kind") != SESSIONIZED_DATASET_MANIFEST_KIND_V1:
        raise RoehubNativeEvaluationError(reason="unexpected_stage06_manifest_kind")
    if manifest.get("market") != "binance:futures":
        raise RoehubNativeEvaluationError(reason="unexpected_stage06_manifest_market")


def _split_artifact_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = manifest.get("split_artifacts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RoehubNativeEvaluationError(reason="stage06_split_artifacts_not_sequence")
    return tuple(cast(Mapping[str, Any], entry) for entry in entries if isinstance(entry, Mapping))


def _mapping_field(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise RoehubNativeEvaluationError(reason="mapping_field_required", field=field)
    return cast(Mapping[str, Any], value)


def _string_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RoehubNativeEvaluationError(reason="string_field_required", field=field)
    return value


def _int_field(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoehubNativeEvaluationError(reason="int_field_required", field=field)
    return int(value)


def _session_metadata(path: Path) -> list[Mapping[str, Any]]:
    payload = _read_json(path)
    sessions = payload.get("sessions")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise RoehubNativeEvaluationError(reason="stage06_metadata_sessions_invalid")
    return [cast(Mapping[str, Any], item) for item in sessions if isinstance(item, Mapping)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)  # type: ignore[arg-type]
    return parsed if np.isfinite(parsed) else None


def _default_run_id(
    *,
    candidate_manifest_sha256: str,
    test_split: RoehubNativeSplitData,
    backtest_split: RoehubNativeSplitData,
    config: RoehubNativeEvaluationConfig,
    source_state: Mapping[str, Any],
) -> str:
    digest = hash_json_payload_v1(
        {
            "backtest_split": dict(backtest_split.source_payload),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "config_hash": config.config_hash(),
            "source_state": dict(source_state),
            "stage": "08F",
            "test_split": dict(test_split.source_payload),
        }
    )
    return f"stage08f_roehub_native_{candidate_manifest_sha256[:8]}_{digest[:20]}"


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


def _format_ms_utc(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 08F Roehub-native evaluation/backtest.")
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        type=str,
        default=DEFAULT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument("--stage06-manifest-path", type=Path, default=DEFAULT_STAGE06_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--test-split", type=str, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--backtest-split", type=str, default=DEFAULT_BACKTEST_SPLIT)
    parser.add_argument("--max-test-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-backtest-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-test-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--max-backtest-artifacts", type=_optional_positive_int, default=None)
    parser.add_argument("--checkpoint-name", choices=("best", "final"), default="best")
    parser.add_argument(
        "--selection-strategy",
        choices=("advantage_based_filter", "ensemble_q_filter"),
        default="advantage_based_filter",
    )
    parser.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="cpu_only_deterministic",
    )
    parser.add_argument("--simple-threshold-return", type=float, default=0.001)
    parser.add_argument("--long-action-threshold", type=float, default=0.012695)
    parser.add_argument("--short-action-threshold", type=float, default=0.009902)
    parser.add_argument("--close-action-threshold", type=float, default=0.001141)
    parser.add_argument("--use-risk-management", action="store_true")
    parser.add_argument("--stop-loss", type=float, default=0.01)
    parser.add_argument("--take-profit", type=float, default=0.02)
    parser.add_argument("--trailing-stop", type=float, default=0.005)
    parser.add_argument("--ensemble-n-samples", type=int, default=5)
    parser.add_argument("--ensemble-max-sigma", type=float, default=0.01)
    parser.add_argument("--max-parallel-sessions", type=int, default=2)
    parser.add_argument("--position-fraction", type=float, default=0.5)
    parser.add_argument("--deterministic-random-seed", type=int, default=806)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
