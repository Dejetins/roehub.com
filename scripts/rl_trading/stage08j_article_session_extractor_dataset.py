from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
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

from scripts.rl_trading import stage06_dataset_qa_session_extractor as stage06  # noqa: E402
from scripts.rl_trading import stage08d_original_hf_backtest_evaluation as hf_eval_cli  # noqa: E402
from scripts.rl_trading import (  # noqa: E402
    stage08h_oracle_supervised_dataset_diagnostics as diagnostics,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    FEATURE_NAMES_V1,
    SESSIONIZED_ARTICLE_POLICY_ID_V1,
    SESSIONIZED_DATASET_MANIFEST_KIND_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    HfOriginalEvaluationError,
    HfOriginalSplitData,
    RoehubNativeEvaluationError,
    RoehubNativeSplitData,
    SessionExtractionPolicy,
    SessionizedDatasetError,
    UpstreamAlphaConfig,
    apply_session_split_embargo_v1,
    article_session_extraction_policy_v1,
    build_leakage_report_v1,
    build_sessionized_dataset_manifest_v1,
    compute_file_sha256,
    hash_json_payload_v1,
    raw_feature_source_windows_from_stage04c_v1,
    select_article_future_impulse_session_candidates_v1,
    session_split_windows_from_stage04c_v1,
)

DEFAULT_STAGE04C_MANIFEST = stage06.DEFAULT_STAGE04C_MANIFEST
DEFAULT_STAGE04C_MANIFEST_SHA256 = stage06.DEFAULT_STAGE04C_MANIFEST_SHA256
DEFAULT_STAGE05_RAW_FEATURE_MANIFEST = stage06.DEFAULT_STAGE05_RAW_FEATURE_MANIFEST
DEFAULT_STAGE06_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1")
    / "stage06_sessionized_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1"
)
DEFAULT_COMPARISON_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / "stage08j_article_session_extractor_dataset_v1"
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_COMPARISON_SPLITS = ("train", "validation", "test", "backtest")
SOURCE_STATE_PATHS = (
    "src/trading/contexts/rl_trading/domain/sessionized_dataset.py",
    "src/trading/contexts/rl_trading/domain/__init__.py",
    "scripts/rl_trading/stage08j_article_session_extractor_dataset.py",
    "scripts/rl_trading/stage06_dataset_qa_session_extractor.py",
    "scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except (
        SessionizedDatasetError,
        HfOriginalEvaluationError,
        RoehubNativeEvaluationError,
    ) as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] == "accepted" else 2


def _run(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    policy = article_session_extraction_policy_v1(
        max_sessions_per_symbol_split=args.max_sessions_per_symbol_split
    )
    manifest = build_article_sessionized_dataset(
        refresh_manifest_path=args.refresh_manifest,
        expected_refresh_manifest_sha256=args.expected_refresh_manifest_sha256,
        raw_feature_manifest_path=args.raw_feature_manifest,
        output_root=args.output_root,
        from_clickhouse=bool(args.from_clickhouse),
        symbols=args.symbol,
        all_symbols=bool(args.all_symbols),
        dataset_versions=args.dataset_version,
        splits=args.split,
        max_symbols=args.max_symbols,
        max_minutes_per_source_window=args.max_minutes_per_source_window,
        policy=policy,
        generated_at_utc=generated_at,
    )
    manifest_path = args.output_root / "stage08j_article_sessionized_manifest.json"
    comparison = build_selector_comparison(
        hf_dataset_dir=args.hf_dataset_dir,
        stage06_manifest_path=args.stage06_manifest_path,
        article_manifest_path=manifest_path,
        output_root=args.comparison_output_root,
        run_id=args.run_id,
        generated_at_utc=generated_at,
        dataset_version=args.comparison_dataset_version,
        splits=args.comparison_split or list(DEFAULT_COMPARISON_SPLITS),
        max_sessions_per_split=args.max_comparison_sessions_per_split,
        max_artifacts_per_split=args.max_comparison_artifacts_per_split,
        allow_fixture_hashes=bool(args.allow_fixture_hashes),
    )
    manifest_sha256 = _file_sha256_hex(manifest_path)
    comparison_path = Path(str(comparison["comparison_path"]))
    return {
        "article_manifest_path": str(manifest_path),
        "article_manifest_sha256": manifest_sha256,
        "comparison_path": str(comparison_path),
        "comparison_sha256": _file_sha256_hex(comparison_path),
        "deterministic_rebuild_hash": manifest["deterministic_rebuild_hash"],
        "proof_boundary": "target_host_non_production_dataset_pre_main",
        "selector_id": SESSIONIZED_ARTICLE_POLICY_ID_V1,
        "split_artifact_count": manifest["split_artifact_count"],
        "status": manifest["status"],
        "total_sessions": manifest["total_sessions"],
    }


def build_article_sessionized_dataset(
    *,
    refresh_manifest_path: Path,
    expected_refresh_manifest_sha256: str,
    raw_feature_manifest_path: Path,
    output_root: Path,
    from_clickhouse: bool,
    symbols: Sequence[str] | None,
    all_symbols: bool,
    dataset_versions: Sequence[str] | None,
    splits: Sequence[str] | None,
    max_symbols: int | None,
    max_minutes_per_source_window: int | None,
    policy: SessionExtractionPolicy,
    generated_at_utc: datetime,
) -> dict[str, object]:
    if policy.policy_id != SESSIONIZED_ARTICLE_POLICY_ID_V1:
        raise SessionizedDatasetError(reason="article_selector_policy_required", field="policy_id")
    if max_symbols is not None and max_symbols <= 0:
        raise SessionizedDatasetError(reason="invalid_max_symbols", field="max_symbols")
    if max_minutes_per_source_window is not None and max_minutes_per_source_window <= 0:
        raise SessionizedDatasetError(
            reason="invalid_max_minutes_per_source_window",
            field="max_minutes_per_source_window",
        )
    if not all_symbols and not symbols:
        raise SessionizedDatasetError(reason="symbol_selection_required", field="symbol")
    if not from_clickhouse and not raw_feature_manifest_path.exists():
        raise SessionizedDatasetError(
            reason="raw_feature_manifest_not_found",
            field=str(raw_feature_manifest_path),
        )

    refresh_manifest_sha256 = _file_sha256_hex(refresh_manifest_path)
    if (
        expected_refresh_manifest_sha256
        and refresh_manifest_sha256 != expected_refresh_manifest_sha256
    ):
        raise SessionizedDatasetError(
            reason="refresh_manifest_sha256_mismatch",
            field="expected_refresh_manifest_sha256",
        )
    refresh_manifest = _read_json(refresh_manifest_path)
    split_windows = session_split_windows_from_stage04c_v1(
        manifest=refresh_manifest,
        dataset_versions=dataset_versions,
        splits=splits,
    )
    split_windows = apply_session_split_embargo_v1(split_windows, policy=policy)
    source_windows = list(
        raw_feature_source_windows_from_stage04c_v1(
            manifest=refresh_manifest,
            dataset_versions=dataset_versions,
            symbols=None if all_symbols else symbols,
        )
    )
    if max_symbols is not None:
        selected_symbols = sorted({window.symbol for window in source_windows})[:max_symbols]
        source_windows = [window for window in source_windows if window.symbol in selected_symbols]
    if not source_windows:
        raise SessionizedDatasetError(reason="no_source_windows_selected")

    output_root.mkdir(parents=True, exist_ok=True)
    gateway = None
    database: str | None = None
    raw_entries: Mapping[tuple[str, str], Mapping[str, Any]] = {}
    if from_clickhouse:
        settings = stage06.ClickHouseSettingsLoader(stage06.os.environ).load()
        gateway = stage06.ClickHouseConnectGateway(stage06._clickhouse_client(settings))
        database = settings.database
    else:
        raw_entries = stage06._raw_slab_entries_by_key(_read_json(raw_feature_manifest_path))

    split_entries: list[dict[str, object]] = []
    all_candidates = []
    rejected: list[dict[str, object]] = []
    for source_window in source_windows:
        for split_window in split_windows:
            if split_window.dataset_version != source_window.dataset_version:
                continue
            effective_split_window = stage06._effective_split_window(
                source_window=source_window,
                split_window=split_window,
                max_minutes_per_source_window=max_minutes_per_source_window,
                policy=policy,
            )
            if effective_split_window is None:
                rejected.append(
                    stage06._rejected_payload(
                        source_window=source_window,
                        split_window=split_window,
                        reason="lifecycle_no_signal_overlap_for_split",
                    )
                )
                continue

            if from_clickhouse:
                if gateway is None or database is None:
                    raise SessionizedDatasetError(reason="clickhouse_gateway_not_initialized")
                slab = stage06._read_slab_from_clickhouse(
                    gateway=gateway,
                    database=database,
                    market_id=source_window.market_id,
                    symbol=source_window.symbol,
                    start=stage06._parse_utc(effective_split_window.source_start_utc),
                    end=stage06._parse_utc(effective_split_window.source_end_utc),
                )
            else:
                slab = stage06._load_raw_slab_from_manifest(
                    raw_entries=raw_entries,
                    dataset_version=source_window.dataset_version,
                    symbol=source_window.symbol,
                )

            candidates = select_article_future_impulse_session_candidates_v1(
                slab=slab,
                split_window=effective_split_window,
                symbol=source_window.symbol,
                policy=policy,
            )
            if not candidates:
                rejected.append(
                    stage06._rejected_payload(
                        source_window=source_window,
                        split_window=effective_split_window,
                        reason="no_article_future_impulse_candidates",
                    )
                )
                continue

            entry = stage06._write_split_artifact(
                output_root=output_root,
                slab=slab,
                source_window=source_window,
                split_window=effective_split_window,
                candidates=candidates,
                policy=policy,
            )
            split_entries.append(entry)
            all_candidates.extend(candidates)

    leakage_report = build_leakage_report_v1(
        candidates=all_candidates,
        split_windows=split_windows,
        policy=policy,
    )
    leakage_report["rejected_windows"] = rejected
    leakage_report["rejected_windows_count"] = len(rejected)
    leakage_report["rejected_reason_counts"] = dict(
        sorted(Counter(str(row["reason"]) for row in rejected).items())
    )
    _atomic_write_json(output_root / "stage08j_leakage_report.json", leakage_report)

    build_scope = {
        "all_symbols": all_symbols,
        "dataset_versions": sorted({window.dataset_version for window in source_windows}),
        "from_clickhouse": from_clickhouse,
        "max_minutes_per_source_window": max_minutes_per_source_window,
        "max_symbols": max_symbols,
        "raw_feature_manifest": None if from_clickhouse else str(raw_feature_manifest_path),
        "selected_symbols": sorted({window.symbol for window in source_windows}),
        "effective_split_windows": [window.as_payload() for window in split_windows],
        "selector_id": SESSIONIZED_ARTICLE_POLICY_ID_V1,
        "splits": sorted({window.split for window in split_windows}),
        "scope": (
            "bounded_sample"
            if max_minutes_per_source_window is not None or max_symbols is not None
            else "full_selected_windows"
        ),
    }
    manifest = build_sessionized_dataset_manifest_v1(
        generated_at_utc=generated_at_utc,
        stage04c_manifest_path=str(refresh_manifest_path),
        stage04c_manifest_sha256=refresh_manifest_sha256,
        output_root=str(output_root),
        split_entries=split_entries,
        leakage_report=leakage_report,
        build_scope=build_scope,
        policy=policy,
        stage="08J",
    )
    manifest["proof_boundary"] = "target_host_non_production_dataset_pre_main"
    manifest["stage08j_handoff"] = {
        "08k_allowed": manifest["status"] == "accepted",
        "stage09_allowed": False,
    }
    manifest["source_state"] = _source_state_payload()
    _atomic_write_json(output_root / "stage08j_article_sessionized_manifest.json", manifest)
    return manifest


def build_selector_comparison(
    *,
    hf_dataset_dir: Path,
    stage06_manifest_path: Path,
    article_manifest_path: Path,
    output_root: Path,
    run_id: str | None,
    generated_at_utc: datetime,
    dataset_version: str,
    splits: Sequence[str],
    max_sessions_per_split: int | None,
    max_artifacts_per_split: int | None,
    allow_fixture_hashes: bool,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_id = run_id or _default_comparison_id(
        stage06_manifest_path=stage06_manifest_path,
        article_manifest_path=article_manifest_path,
        dataset_version=dataset_version,
        splits=splits,
        max_sessions_per_split=max_sessions_per_split,
        max_artifacts_per_split=max_artifacts_per_split,
    )
    run_dir = output_root / comparison_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_specs = {spec.split_name: spec for spec in hf_eval_cli.expected_hf_split_specs_v1()}
    hf_splits = {
        split: hf_eval_cli._load_hf_split(
            dataset_dir=hf_dataset_dir,
            split_spec=hf_specs[split],
            max_sessions=max_sessions_per_split,
            allow_fixture_hashes=allow_fixture_hashes,
        )
        for split in splits
        if split in hf_specs
    }
    stage06_manifest = _read_json(stage06_manifest_path)
    stage06_sha = compute_file_sha256(stage06_manifest_path)
    article_manifest = _read_json(article_manifest_path)
    article_sha = compute_file_sha256(article_manifest_path)
    stage06_splits = {
        split: _load_sessionized_split(
            manifest=stage06_manifest,
            manifest_path=stage06_manifest_path,
            manifest_sha256=stage06_sha,
            dataset_version=dataset_version,
            split=split,
            max_sessions=max_sessions_per_split,
            max_artifacts=max_artifacts_per_split,
            allow_fixture_hashes=allow_fixture_hashes,
            accepted_stages=("06",),
        )
        for split in splits
    }
    article_splits = {
        split: _load_sessionized_split(
            manifest=article_manifest,
            manifest_path=article_manifest_path,
            manifest_sha256=article_sha,
            dataset_version=dataset_version,
            split=split,
            max_sessions=max_sessions_per_split,
            max_artifacts=max_artifacts_per_split,
            allow_fixture_hashes=allow_fixture_hashes,
            accepted_stages=("08J",),
        )
        for split in splits
    }
    alpha = UpstreamAlphaConfig()
    cost_ratio = 2.0 * (alpha.transaction_fee + alpha.slippage)
    branches = {
        "hf_original": _branch_payload(
            splits=hf_splits,
            profile=(30, 10),
            cost_ratio=cost_ratio,
        ),
        "stage06_current_selector": _branch_payload(
            splits=stage06_splits,
            profile=(30, 10),
            cost_ratio=cost_ratio,
        ),
        "article_selector": _branch_payload(
            splits=article_splits,
            profile=(30, 10),
            cost_ratio=cost_ratio,
        ),
    }
    payload = {
        "artifact_kind": "rl_trading_stage08j_selector_distribution_comparison",
        "branches": branches,
        "cost_model": {
            "round_trip_cost_ratio": _round_float(cost_ratio),
            "slippage": alpha.slippage,
            "transaction_fee": alpha.transaction_fee,
        },
        "dataset_version": dataset_version,
        "generated_at_utc": _format_utc(generated_at_utc),
        "methodology": {
            "oracle_profile": "30/10",
            "selector_id": SESSIONIZED_ARTICLE_POLICY_ID_V1,
            "supervised_sanity": "closed_form_ridge_classifier_numpy_past_only",
        },
        "run_dir": str(run_dir),
        "run_id": comparison_id,
        "source_state": _source_state_payload(),
        "splits": list(splits),
        "status": "accepted",
    }
    payload["comparison_hash"] = hash_json_payload_v1(payload)
    comparison_path = run_dir / "stage08j_selector_distribution_comparison.json"
    _atomic_write_json(comparison_path, payload)
    return {
        "comparison_path": str(comparison_path),
        "run_dir": str(run_dir),
        "run_id": comparison_id,
        "status": payload["status"],
    }


def _branch_payload(
    *,
    splits: Mapping[str, HfOriginalSplitData | RoehubNativeSplitData],
    profile: tuple[int, int],
    cost_ratio: float,
) -> dict[str, Any]:
    split_payloads = {
        split_name: _split_payload(split=split, profile=profile, cost_ratio=cost_ratio)
        for split_name, split in splits.items()
    }
    supervised = (
        diagnostics._supervised_sanity(
            train_sequences=splits["train"].sequences,
            eval_splits={name: split.sequences for name, split in splits.items()},
            profile=profile,
            cost_ratio=cost_ratio,
        )
        if "train" in splits
        else {"reason": "train_split_missing", "status": "skipped"}
    )
    return {
        "session_count": int(sum(split.sequences.shape[0] for split in splits.values())),
        "source_splits": {name: dict(split.source_payload) for name, split in splits.items()},
        "split_diagnostics": split_payloads,
        "supervised_sanity": supervised,
        "symbol_count": len({symbol for split in splits.values() for symbol in split.symbols}),
    }


def _split_payload(
    *,
    split: HfOriginalSplitData | RoehubNativeSplitData,
    profile: tuple[int, int],
    cost_ratio: float,
) -> dict[str, Any]:
    oracle = diagnostics._oracle_payload(
        sequences=split.sequences,
        profile=profile,
        cost_ratio=cost_ratio,
    )
    return {
        "oracle": diagnostics._oracle_summary_payload(oracle),
        "range_and_volatility": _range_volatility_payload(split.sequences),
        "session_count": int(split.sequences.shape[0]),
        "symbol_count": len(set(split.symbols)),
        "symbol_counts_top20": _top_counts(split.symbols, limit=20),
        "symbol_month_counts": _symbol_month_counts(
            symbols=split.symbols,
            signal_times=split.signal_times_utc,
        ),
        "time_month_counts": _month_counts(split.signal_times_utc),
    }


def _range_volatility_payload(sequences: np.ndarray) -> dict[str, object]:
    close = np.maximum(sequences[:, :, _feature_index("close")].astype(np.float64), 1e-12)
    high = sequences[:, :, _feature_index("high")].astype(np.float64)
    low = sequences[:, :, _feature_index("low")].astype(np.float64)
    pre = close[:, :90]
    returns = np.diff(pre, axis=1) / pre[:, :-1]
    realized = np.std(returns, axis=1)
    range_ratio = (np.max(high[:, :90], axis=1) - np.min(low[:, :90], axis=1)) / pre[:, -1]
    return {
        "pre_signal_range_ratio": _distribution_payload(range_ratio),
        "pre_signal_realized_volatility": _distribution_payload(realized),
    }


def _load_sessionized_split(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    dataset_version: str,
    split: str,
    max_sessions: int | None,
    max_artifacts: int | None,
    allow_fixture_hashes: bool,
    accepted_stages: Sequence[str],
) -> RoehubNativeSplitData:
    _validate_sessionized_manifest(manifest, accepted_stages=accepted_stages)
    entries = [
        entry
        for entry in _split_artifact_entries(manifest)
        if entry.get("dataset_version") == dataset_version and entry.get("split") == split
    ]
    if not entries:
        raise RoehubNativeEvaluationError(
            reason="sessionized_split_artifacts_not_found",
            field=f"{dataset_version}:{split}",
        )
    entries.sort(key=lambda item: str(item.get("symbol", "")))
    total_entry_sessions = sum(_int_field(entry, "candidate_count") for entry in entries)
    selected_entries = entries if max_artifacts is None else entries[:max_artifacts]
    arrays: list[np.ndarray] = []
    symbols: list[str] = []
    signal_times: list[str | None] = []
    scores: list[float | None] = []
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
                    reason="sessionized_artifact_file_missing",
                    field=str(path),
                )
        hash_matches = (
            compute_file_sha256(feature_path) == _string_field(feature_payload, "sha256")
            and compute_file_sha256(signal_path) == _string_field(signal_payload, "sha256")
            and compute_file_sha256(metadata_path) == _string_field(metadata_payload, "sha256")
        )
        if not hash_matches and not allow_fixture_hashes:
            raise RoehubNativeEvaluationError(
                reason="sessionized_artifact_hash_mismatch",
                field=str(feature_path),
            )
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        signal_time_ms = np.asarray(np.load(signal_path), dtype=np.int64)
        sessions = _session_metadata(metadata_path)
        candidate_count = _int_field(entry, "candidate_count")
        if (
            features.ndim != 3
            or tuple(features.shape[1:]) != (150, len(FEATURE_NAMES_V1))
            or features.shape[0] != candidate_count
            or signal_time_ms.shape[0] != candidate_count
            or len(sessions) != candidate_count
        ) and not allow_fixture_hashes:
            raise RoehubNativeEvaluationError(
                reason="sessionized_artifact_candidate_count_mismatch",
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
        signal_times.extend(
            _format_ms_utc(int(value)) for value in signal_time_ms[:selected_count]
        )
        scores.extend(
            _optional_float(item.get("volatility_score")) for item in sessions[:selected_count]
        )
        selected_session_count += selected_count
    if not arrays:
        raise RoehubNativeEvaluationError(
            reason="sessionized_split_empty_selection",
            field=f"{dataset_version}:{split}",
        )
    features_out = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    return RoehubNativeSplitData(
        split_name=split,
        sequences=features_out,
        symbols=tuple(symbols),
        signal_times_utc=tuple(signal_times),
        source_payload={
            "dataset_version": dataset_version,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_stage": manifest.get("stage"),
            "selector_id": _policy_id_from_manifest(manifest),
            "selected_session_count": selected_session_count,
            "split_name": split,
            "total_session_count": total_entry_sessions,
        },
        volatility_scores=tuple(scores[:selected_session_count]),
    )


def _validate_sessionized_manifest(
    manifest: Mapping[str, Any],
    *,
    accepted_stages: Sequence[str],
) -> None:
    if manifest.get("stage") not in set(accepted_stages):
        raise RoehubNativeEvaluationError(reason="unexpected_sessionized_manifest_stage")
    if manifest.get("status") != "accepted":
        raise RoehubNativeEvaluationError(reason="sessionized_manifest_not_accepted")
    if manifest.get("manifest_kind") != SESSIONIZED_DATASET_MANIFEST_KIND_V1:
        raise RoehubNativeEvaluationError(reason="unexpected_sessionized_manifest_kind")
    if manifest.get("market") != "binance:futures":
        raise RoehubNativeEvaluationError(reason="unexpected_sessionized_manifest_market")


def _policy_id_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        return None
    value = policy.get("policy_id")
    return str(value) if isinstance(value, str) and value else None


def _split_artifact_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = manifest.get("split_artifacts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RoehubNativeEvaluationError(reason="sessionized_split_artifacts_not_sequence")
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
        raise RoehubNativeEvaluationError(reason="sessionized_metadata_sessions_invalid")
    return [cast(Mapping[str, Any], item) for item in sessions if isinstance(item, Mapping)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)  # type: ignore[arg-type]
    return parsed if np.isfinite(parsed) else None


def _top_counts(values: Sequence[str], *, limit: int) -> dict[str, int]:
    return dict(Counter(values).most_common(limit))


def _month_counts(signal_times: Sequence[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(_month_key(value) for value in signal_times).items()))


def _symbol_month_counts(
    *,
    symbols: Sequence[str],
    signal_times: Sequence[str | None],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{symbol}|{_month_key(signal_time)}"
                for symbol, signal_time in zip(symbols, signal_times, strict=True)
            ).items()
        )
    )


def _month_key(value: str | None) -> str:
    if not value:
        return "unknown"
    return value[:7]


def _distribution_payload(values: np.ndarray) -> dict[str, float]:
    return {
        "max": _round_float(float(np.max(values))),
        "mean": _round_float(float(np.mean(values))),
        "median": _round_float(float(np.median(values))),
        "p10": _round_float(float(np.quantile(values, 0.10))),
        "p90": _round_float(float(np.quantile(values, 0.90))),
    }


def _feature_index(name: str) -> int:
    return FEATURE_NAMES_V1.index(name)


def _default_comparison_id(
    *,
    stage06_manifest_path: Path,
    article_manifest_path: Path,
    dataset_version: str,
    splits: Sequence[str],
    max_sessions_per_split: int | None,
    max_artifacts_per_split: int | None,
) -> str:
    digest = hash_json_payload_v1(
        {
            "article_manifest_path": str(article_manifest_path),
            "dataset_version": dataset_version,
            "max_artifacts_per_split": max_artifacts_per_split,
            "max_sessions_per_split": max_sessions_per_split,
            "splits": list(splits),
            "stage": "08J",
            "stage06_manifest_path": str(stage06_manifest_path),
        }
    )
    return f"stage08j_selector_comparison_{digest[:20]}"


def _source_state_payload() -> dict[str, object]:
    files = []
    for relative in SOURCE_STATE_PATHS:
        path = REPO_ROOT / relative
        if path.exists():
            files.append({"path": relative, "sha256": _file_sha256_hex(path)})
    payload: dict[str, object] = {
        "source_file_hashes": files,
        "source_paths": list(SOURCE_STATE_PATHS),
    }
    if (REPO_ROOT / ".git").exists():
        try:
            payload["git_head"] = _git_output("rev-parse", "HEAD")
            payload["git_status_short"] = _git_output(
                "status",
                "--short",
                "--",
                *SOURCE_STATE_PATHS,
            ).splitlines()
        except Exception as exc:
            payload["git_unavailable_reason"] = type(exc).__name__
    return payload


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _format_ms_utc(value: int) -> str:
    return _format_utc(datetime.fromtimestamp(value / 1000, tz=UTC))


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _round_float(value: float) -> float:
    return float(round(value, 10))


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return parsed if parsed > 0 else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 08J article-selector dataset.")
    parser.add_argument("--refresh-manifest", type=Path, default=DEFAULT_STAGE04C_MANIFEST)
    parser.add_argument(
        "--expected-refresh-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE04C_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--raw-feature-manifest",
        type=Path,
        default=DEFAULT_STAGE05_RAW_FEATURE_MANIFEST,
    )
    parser.add_argument("--stage06-manifest-path", type=Path, default=DEFAULT_STAGE06_MANIFEST)
    parser.add_argument("--hf-dataset-dir", type=Path, default=hf_eval_cli.DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--comparison-output-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--from-clickhouse", action="store_true")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--dataset-version", action="append", default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--max-minutes-per-source-window", type=int, default=None)
    parser.add_argument(
        "--max-sessions-per-symbol-split",
        type=_optional_positive_int,
        default=None,
    )
    parser.add_argument("--comparison-dataset-version", type=str, default=DEFAULT_DATASET_VERSION)
    parser.add_argument(
        "--comparison-split",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--max-comparison-sessions-per-split",
        type=_optional_positive_int,
        default=None,
    )
    parser.add_argument(
        "--max-comparison-artifacts-per-split",
        type=_optional_positive_int,
        default=None,
    )
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
