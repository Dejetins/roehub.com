from __future__ import annotations

import argparse
import hashlib
import json
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
    TrainingRunnerError,
    TrainingSmokeConfig,
    build_stage07a_transition_set_v1,
    run_d3qn_per_training_smoke_v1,
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
    / "training_smokes"
    / "stage07a_training_runner_smoke_v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
        record = run_stage07a_training_smoke(
            sessionized_manifest_path=args.sessionized_manifest,
            expected_sessionized_manifest_sha256=args.expected_sessionized_manifest_sha256,
            output_root=args.output_root,
            dataset_version=args.dataset_version,
            split=args.split,
            symbols=args.symbol,
            max_session_artifacts=args.max_session_artifacts,
            config=TrainingSmokeConfig(
                seed=args.seed,
                max_sessions=args.max_sessions,
                batch_size=args.batch_size,
                update_steps=args.update_steps,
                torch_num_threads=args.torch_num_threads,
                torch_num_interop_threads=args.torch_num_interop_threads,
                device_policy=args.device_policy,
            ),
            generated_at_utc=(
                _parse_utc(args.generated_at_utc)
                if args.generated_at_utc is not None
                else datetime.now(UTC).replace(microsecond=0)
            ),
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
                "config_hash": record["config_hash"],
                "dataset_manifest_sha256": record["dataset_dependency"]["manifest_sha256"],
                "metrics": record["metrics"],
                "run_record_hash": record["run_record_hash"],
                "run_record_path": record["run_record_path"],
                "selected_device": record["resource_usage"]["selected_device"],
                "status": record["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def run_stage07a_training_smoke(
    *,
    sessionized_manifest_path: Path,
    expected_sessionized_manifest_sha256: str,
    output_root: Path,
    dataset_version: str | None,
    split: str | None,
    symbols: Sequence[str] | None,
    max_session_artifacts: int,
    config: TrainingSmokeConfig,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    manifest_sha256 = _file_sha256_hex(sessionized_manifest_path)
    if (
        expected_sessionized_manifest_sha256
        and manifest_sha256 != expected_sessionized_manifest_sha256
    ):
        raise TrainingRunnerError(
            reason="sessionized_manifest_sha256_mismatch",
            field=str(sessionized_manifest_path),
        )
    manifest = _read_json(sessionized_manifest_path)
    _validate_stage06_manifest(manifest)
    features = _load_selected_session_features(
        manifest=manifest,
        dataset_version=dataset_version,
        split=split,
        symbols=symbols,
        max_session_artifacts=max_session_artifacts,
        max_sessions=config.max_sessions,
    )
    transitions = build_stage07a_transition_set_v1(session_features=features, config=config)
    return run_d3qn_per_training_smoke_v1(
        transitions=transitions,
        dataset_manifest_path=str(sessionized_manifest_path),
        dataset_manifest_sha256=manifest_sha256,
        output_root=output_root,
        config=config,
        generated_at_utc=generated_at_utc,
    )


def _load_selected_session_features(
    *,
    manifest: Mapping[str, Any],
    dataset_version: str | None,
    split: str | None,
    symbols: Sequence[str] | None,
    max_session_artifacts: int,
    max_sessions: int,
) -> np.ndarray:
    if max_session_artifacts <= 0:
        raise TrainingRunnerError(
            reason="invalid_max_session_artifacts",
            field="max_session_artifacts",
        )
    selected_symbols = None if not symbols else {symbol.upper() for symbol in symbols}
    entries = []
    for entry in _split_artifact_entries(manifest):
        if dataset_version is not None and entry.get("dataset_version") != dataset_version:
            continue
        if split is not None and entry.get("split") != split:
            continue
        symbol = str(entry.get("symbol", "")).upper()
        if selected_symbols is not None and symbol not in selected_symbols:
            continue
        entries.append(entry)
    entries.sort(
        key=lambda item: (
            str(item["dataset_version"]),
            str(item["split"]),
            str(item["symbol"]),
        )
    )
    if not entries:
        raise TrainingRunnerError(reason="no_sessionized_split_artifact_selected")

    chunks: list[np.ndarray] = []
    remaining = max_sessions
    for entry in entries[:max_session_artifacts]:
        features_path = _artifact_file_path(entry, "features")
        features = np.load(features_path, mmap_mode="r")
        if features.ndim != 3 or features.shape[0] == 0:
            continue
        take = min(int(features.shape[0]), remaining)
        chunks.append(np.asarray(features[:take], dtype=np.float32))
        remaining -= take
        if remaining <= 0:
            break
    if not chunks:
        raise TrainingRunnerError(reason="selected_session_features_empty")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 07A D3QN/PER trainer smoke.")
    parser.add_argument(
        "--sessionized-manifest",
        type=Path,
        default=DEFAULT_STAGE06_SESSIONIZED_MANIFEST,
    )
    parser.add_argument(
        "--expected-sessionized-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE06_SESSIONIZED_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--dataset-version", type=str, default="post_hf_extension_current_trading")
    parser.add_argument("--split", type=str, default="post_hf_extension")
    parser.add_argument("--symbol", action="append", default=["BTCUSDT"])
    parser.add_argument("--max-session-artifacts", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=240723)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--update-steps", type=int, default=8)
    parser.add_argument("--torch-num-threads", type=int, default=2)
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
