from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
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
    HF_DATASET_REPO_ID_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    STAGE08D_RUNTIME_ARTIFACT_SUBDIR_V1,
    HfDatasetSplitSpec,
    HfOriginalEvaluationConfig,
    HfOriginalEvaluationError,
    HfOriginalSplitData,
    UpstreamAlphaConfig,
    compute_file_sha256,
    expected_hf_dataset_manifest_hash_v1,
    expected_hf_split_specs_v1,
    hash_json_payload_v1,
    run_stage08d_hf_original_evaluation_v1,
)

DEFAULT_HF_DATASET_DIR = (
    Path("/opt/roehub/state/rl_trading/hf_reproducibility/dataset")
    / "ResearchRL"
    / "open-rl-trading-binance-dataset"
)
DEFAULT_CANDIDATE_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/training_runs")
    / "stage08c_original_hf_full_training_run_v1"
    / "full"
    / "stage08c_hf_original_full"
    / "hf_original_candidate_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST_SHA256 = (
    "189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08D_RUNTIME_ARTIFACT_SUBDIR_V1
)
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/hf_original_evaluation.py",
    "src/trading/contexts/rl_trading/domain/upstream_methodology.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08d_original_hf_backtest_evaluation.py",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run_command(args)
    except HfOriginalEvaluationError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] == "accepted" else 2


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    candidate_manifest_sha256 = _file_sha256_hex(args.candidate_manifest)
    if (
        args.expected_candidate_manifest_sha256
        and candidate_manifest_sha256 != args.expected_candidate_manifest_sha256
    ):
        raise HfOriginalEvaluationError(
            reason="candidate_manifest_sha256_mismatch",
            field=str(args.candidate_manifest),
        )
    candidate_manifest = _read_json(args.candidate_manifest)
    specs = {spec.split_name: spec for spec in expected_hf_split_specs_v1()}
    test_split = _load_hf_split(
        dataset_dir=args.dataset_dir,
        split_spec=specs["test"],
        max_sessions=args.max_test_sessions,
        allow_fixture_hashes=args.allow_fixture_hashes,
    )
    backtest_split = _load_hf_split(
        dataset_dir=args.dataset_dir,
        split_spec=specs["backtest"],
        max_sessions=args.max_backtest_sessions,
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
    config = HfOriginalEvaluationConfig(
        alpha=alpha,
        checkpoint_name=args.checkpoint_name,
        selection_strategy=args.selection_strategy,
        device_policy=args.device_policy,
        test_max_sessions=args.max_test_sessions,
        backtest_max_sessions=args.max_backtest_sessions,
        simple_threshold_return=args.simple_threshold_return,
    )
    source_state = _source_state_payload()
    run_id = args.run_id or _default_run_id(
        candidate_manifest_sha256=candidate_manifest_sha256,
        test_split=test_split,
        backtest_split=backtest_split,
        config=config,
        source_state=source_state,
    )
    manifest = run_stage08d_hf_original_evaluation_v1(
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
        "methodology_parity_verdict": manifest["methodology_parity_verdict"],
        "run_dir": manifest["run_dir"],
        "run_id": manifest["run_id"],
        "status": manifest["status"],
    }


def _load_hf_split(
    *,
    dataset_dir: Path,
    split_spec: HfDatasetSplitSpec,
    max_sessions: int | None,
    allow_fixture_hashes: bool,
) -> HfOriginalSplitData:
    file_path = dataset_dir / split_spec.file_name
    if not file_path.exists():
        raise HfOriginalEvaluationError(reason="missing_hf_split_file", field=str(file_path))
    sha256 = compute_file_sha256(file_path)
    if sha256 != split_spec.expected_sha256 and not allow_fixture_hashes:
        raise HfOriginalEvaluationError(reason="hf_split_hash_mismatch", field=split_spec.file_name)
    with np.load(file_path, allow_pickle=True) as archive:
        keys = sorted(
            (key for key in archive.files if key.startswith("fetcher_")),
            key=_fetcher_key_sort_value,
        )
        total_sessions = len(keys)
        if sha256 == split_spec.expected_sha256 and total_sessions != split_spec.observed_sessions:
            raise HfOriginalEvaluationError(
                reason="hf_split_session_count_mismatch",
                field=split_spec.file_name,
            )
        selected_keys = keys if max_sessions is None else keys[:max_sessions]
        if not selected_keys:
            raise HfOriginalEvaluationError(
                reason="hf_split_empty_selection",
                field=split_spec.file_name,
            )
        key_map = _keys_map(archive)
        features = np.empty(
            (len(selected_keys), 150, len(FEATURE_NAMES_V1)),
            dtype=np.float32,
        )
        symbols: list[str] = []
        signal_times: list[str | None] = []
        for row_idx, key in enumerate(selected_keys):
            arr = np.asarray(archive[key], dtype=np.float32)
            if arr.shape != (150, len(FEATURE_NAMES_V1)):
                raise HfOriginalEvaluationError(reason="hf_session_shape_mismatch", field=key)
            features[row_idx] = arr
            symbol, signal_time = _metadata_for_key(key_map, key)
            symbols.append(symbol)
            signal_times.append(signal_time)
    source_payload = {
        "allow_fixture_hashes": bool(allow_fixture_hashes),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_hash": expected_hf_dataset_manifest_hash_v1(),
        "dataset_repo_id": HF_DATASET_REPO_ID_V1,
        "expected_sha256": split_spec.expected_sha256,
        "file_name": split_spec.file_name,
        "file_path": str(file_path),
        "hash_matches_expected": sha256 == split_spec.expected_sha256,
        "selected_session_count": len(selected_keys),
        "sha256": sha256,
        "split_name": split_spec.split_name,
        "total_session_count": total_sessions,
    }
    return HfOriginalSplitData(
        split_name=split_spec.split_name,
        sequences=np.ascontiguousarray(features, dtype=np.float32),
        symbols=tuple(symbols),
        signal_times_utc=tuple(signal_times),
        source_payload=source_payload,
    )


def _keys_map(archive: Any) -> Mapping[str, Any]:
    if "_keys_map_" not in archive.files:
        return {}
    value = archive["_keys_map_"]
    try:
        item = value.item()
    except Exception:
        return {}
    return item if isinstance(item, Mapping) else {}


def _metadata_for_key(key_map: Mapping[str, Any], key: str) -> tuple[str, str | None]:
    value = key_map.get(key)
    if isinstance(value, tuple) and len(value) >= 2:
        symbol = str(value[0]).upper()
        signal_time = value[1]
        if hasattr(signal_time, "astimezone"):
            text = signal_time.astimezone(UTC).replace(microsecond=0).isoformat()
            return symbol, text.replace("+00:00", "Z")
        return symbol, str(signal_time)
    return "UNKNOWN", None


def _fetcher_key_sort_value(value: str) -> tuple[int, str]:
    try:
        return int(value.split("_", 1)[1]), value
    except Exception:
        return sys.maxsize, value


def _default_run_id(
    *,
    candidate_manifest_sha256: str,
    test_split: HfOriginalSplitData,
    backtest_split: HfOriginalSplitData,
    config: HfOriginalEvaluationConfig,
    source_state: Mapping[str, Any],
) -> str:
    digest = hash_json_payload_v1(
        {
            "backtest_split": dict(backtest_split.source_payload),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "config_hash": config.config_hash(),
            "source_state": dict(source_state),
            "stage": "08D",
            "test_split": dict(test_split.source_payload),
        }
    )
    return f"stage08d_hf_original_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    source_files = []
    for relative in SOURCE_STATE_PATHS:
        path = REPO_ROOT / relative
        if path.exists():
            source_files.append({"path": relative, "sha256": _file_sha256_hex(path)})
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
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


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


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 08D original HF evaluation/backtest.")
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        type=str,
        default=DEFAULT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--max-test-sessions", type=_optional_positive_int, default=None)
    parser.add_argument("--max-backtest-sessions", type=_optional_positive_int, default=None)
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
    parser.add_argument(
        "--checkpoint-name",
        choices=("best", "final"),
        default="best",
    )
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
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
