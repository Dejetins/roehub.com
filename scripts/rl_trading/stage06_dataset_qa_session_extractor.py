from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.cli.wiring.db.clickhouse import (  # noqa: E402
    ClickHouseSettingsLoader,
    _clickhouse_client,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (  # noqa: E402,E501
    ClickHouseConnectGateway,
    ClickHouseGateway,
)
from trading.contexts.rl_trading.domain import (  # noqa: E402
    RawFeatureCandleBatch,
    RawFeatureSlab,
    RawFeatureSourceWindow,
    SessionCandidate,
    SessionExtractionPolicy,
    SessionizedDatasetError,
    SessionSplitWindow,
    assert_sessionized_trainable_source_v1,
    build_gap_report_v1,
    build_leakage_report_v1,
    build_raw_feature_slab_v1,
    build_sessionized_dataset_manifest_v1,
    build_split_artifact_entry_v1,
    materialize_session_features_v1,
    raw_feature_source_windows_from_stage04c_v1,
    render_raw_feature_json_payload_v1,
    select_high_volatility_session_candidates_v1,
    session_metadata_payload_v1,
    session_signal_time_array_v1,
    session_split_windows_from_stage04c_v1,
    training_source_gate_payload_v1,
)

DEFAULT_STAGE04C_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest")
    / "stage04c_dataset_refresh_manifest.json"
)
DEFAULT_STAGE04C_MANIFEST_SHA256 = (
    "9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344"
)
DEFAULT_STAGE05_RAW_FEATURE_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1")
    / "stage05_raw_feature_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gate = training_source_gate_payload_v1(exchange=args.exchange, market_type=args.market_type)
    if gate["status"] != "trainable":
        print(
            _render_status(
                {
                    "exchange": gate["exchange"],
                    "market_type": gate["market_type"],
                    "reason": "blocked_not_training_source_v1",
                    "status": "blocked",
                }
            )
        )
        return 2

    try:
        manifest = build_sessionized_dataset(
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
            policy=SessionExtractionPolicy(
                signal_stride_minutes=int(args.signal_stride_minutes),
                high_volatility_top_fraction=float(args.high_volatility_top_fraction),
                max_sessions_per_symbol_split=int(args.max_sessions_per_symbol_split),
            ),
            generated_at_utc=(
                _parse_utc(args.generated_at_utc)
                if args.generated_at_utc is not None
                else _now_utc()
            ),
        )
    except SessionizedDatasetError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2

    manifest_path = args.output_root / "stage06_sessionized_manifest.json"
    print(
        _render_status(
            {
                "dataset_manifest": str(manifest_path),
                "deterministic_rebuild_hash": manifest["deterministic_rebuild_hash"],
                "manifest_file_sha256": _file_sha256_hex(manifest_path),
                "split_artifact_count": manifest["split_artifact_count"],
                "status": manifest["status"],
                "total_sessions": manifest["total_sessions"],
            }
        )
    )
    return 0 if manifest["status"] == "accepted" else 2


def build_sessionized_dataset(
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

    assert_sessionized_trainable_source_v1(exchange="binance", market_type="futures")
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
    gateway: ClickHouseGateway | None = None
    database: str | None = None
    raw_entries: Mapping[tuple[str, str], Mapping[str, Any]] = {}
    if from_clickhouse:
        settings = ClickHouseSettingsLoader(os.environ).load()
        gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
        database = settings.database
    else:
        raw_entries = _raw_slab_entries_by_key(_read_json(raw_feature_manifest_path))

    split_entries: list[dict[str, object]] = []
    all_candidates = []
    rejected: list[dict[str, object]] = []
    for source_window in source_windows:
        for split_window in split_windows:
            if split_window.dataset_version != source_window.dataset_version:
                continue
            effective_split_window = _effective_split_window(
                source_window=source_window,
                split_window=split_window,
                max_minutes_per_source_window=max_minutes_per_source_window,
                policy=policy,
            )
            if effective_split_window is None:
                rejected.append(
                    _rejected_payload(
                        source_window=source_window,
                        split_window=split_window,
                        reason="lifecycle_no_signal_overlap_for_split",
                    )
                )
                continue

            if from_clickhouse:
                if gateway is None or database is None:
                    raise SessionizedDatasetError(reason="clickhouse_gateway_not_initialized")
                slab = _read_slab_from_clickhouse(
                    gateway=gateway,
                    database=database,
                    market_id=source_window.market_id,
                    symbol=source_window.symbol,
                    start=_parse_utc(effective_split_window.source_start_utc),
                    end=_parse_utc(effective_split_window.source_end_utc),
                )
            else:
                slab = _load_raw_slab_from_manifest(
                    raw_entries=raw_entries,
                    dataset_version=source_window.dataset_version,
                    symbol=source_window.symbol,
                )

            candidates = select_high_volatility_session_candidates_v1(
                slab=slab,
                split_window=effective_split_window,
                symbol=source_window.symbol,
                policy=policy,
            )
            if not candidates:
                rejected.append(
                    _rejected_payload(
                        source_window=source_window,
                        split_window=effective_split_window,
                        reason="no_high_volatility_candidates",
                    )
                )
                continue

            entry = _write_split_artifact(
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
    _atomic_write_json(output_root / "stage06_leakage_report.json", leakage_report)

    build_scope = {
        "all_symbols": all_symbols,
        "dataset_versions": sorted({window.dataset_version for window in source_windows}),
        "from_clickhouse": from_clickhouse,
        "max_minutes_per_source_window": max_minutes_per_source_window,
        "max_symbols": max_symbols,
        "raw_feature_manifest": None if from_clickhouse else str(raw_feature_manifest_path),
        "selected_symbols": sorted({window.symbol for window in source_windows}),
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
    )
    _atomic_write_json(output_root / "stage06_sessionized_manifest.json", manifest)
    return manifest


def _write_split_artifact(
    *,
    output_root: Path,
    slab: RawFeatureSlab,
    source_window: RawFeatureSourceWindow,
    split_window: SessionSplitWindow,
    candidates: Sequence[SessionCandidate],
    policy: SessionExtractionPolicy,
) -> dict[str, object]:
    artifact_root = (
        output_root / source_window.dataset_version / split_window.split / source_window.symbol
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    features_path = artifact_root / "sessions.f32.npy"
    signal_time_path = artifact_root / "signal_time_ms.i64.npy"
    metadata_path = artifact_root / "metadata.json"
    split_manifest_path = artifact_root / "manifest.json"

    features = materialize_session_features_v1(slab=slab, candidates=candidates, policy=policy)
    np.save(features_path, features)
    np.save(signal_time_path, session_signal_time_array_v1(candidates))
    _atomic_write_json(metadata_path, {"sessions": session_metadata_payload_v1(candidates)})

    artifact_files = {
        "features": _file_payload(features_path),
        "metadata": _file_payload(metadata_path),
        "signal_time_ms": _file_payload(signal_time_path),
    }
    entry = build_split_artifact_entry_v1(
        dataset_version=source_window.dataset_version,
        split=split_window.split,
        symbol=source_window.symbol,
        candidates=candidates,
        artifact_files=artifact_files,
        gap_report=build_gap_report_v1(slab=slab),
        policy=policy,
    )
    entry["manifest_path"] = str(split_manifest_path)
    _atomic_write_json(split_manifest_path, entry)
    return entry


def _raw_slab_entries_by_key(
    manifest: Mapping[str, Any],
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    if manifest.get("stage") != "05" or manifest.get("status") != "accepted":
        raise SessionizedDatasetError(reason="raw_feature_manifest_not_accepted")
    entries = manifest.get("slabs")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise SessionizedDatasetError(reason="raw_feature_manifest_slabs_not_sequence")
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        key = (str(entry["dataset_version"]), str(entry["symbol"]).upper())
        out[key] = entry
    return out


def _load_raw_slab_from_manifest(
    *,
    raw_entries: Mapping[tuple[str, str], Mapping[str, Any]],
    dataset_version: str,
    symbol: str,
) -> RawFeatureSlab:
    entry = raw_entries.get((dataset_version, symbol.upper()))
    if entry is None:
        raise SessionizedDatasetError(
            reason="raw_feature_slab_not_found",
            field=f"{dataset_version}:{symbol}",
        )
    files = entry.get("files")
    if not isinstance(files, Mapping):
        raise SessionizedDatasetError(reason="raw_feature_slab_files_not_mapping")
    return RawFeatureSlab(
        open_time_ms=np.load(_file_path(files, "open_time_ms"), mmap_mode="r"),
        close_time_ms=np.load(_file_path(files, "close_time_ms"), mmap_mode="r"),
        features_f32=np.load(_file_path(files, "features"), mmap_mode="r"),
    )


def _file_path(files: Mapping[str, Any], key: str) -> Path:
    item = files.get(key)
    if not isinstance(item, Mapping):
        raise SessionizedDatasetError(reason="raw_feature_file_missing", field=key)
    return Path(str(item["path"]))


def _effective_split_window(
    *,
    source_window: RawFeatureSourceWindow,
    split_window: SessionSplitWindow,
    max_minutes_per_source_window: int | None,
    policy: SessionExtractionPolicy,
) -> SessionSplitWindow | None:
    source_start = max(
        _parse_utc(source_window.source_start_utc),
        _parse_utc(split_window.source_start_utc),
    )
    source_end = min(
        _parse_utc(source_window.source_end_utc),
        _parse_utc(split_window.source_end_utc),
    )
    if max_minutes_per_source_window is not None:
        source_end = min(
            source_end,
            source_start + timedelta(minutes=max_minutes_per_source_window),
        )
    if source_end <= source_start:
        return None

    signal_start = max(
        _parse_utc(split_window.signal_start_utc),
        source_start + timedelta(minutes=policy.pre_signal_len),
    )
    signal_end = min(
        _parse_utc(split_window.signal_end_utc),
        source_end - timedelta(minutes=policy.post_signal_len),
    )
    if signal_end <= signal_start:
        return None
    return replace(
        split_window,
        signal_start_utc=_format_utc(signal_start),
        signal_end_utc=_format_utc(signal_end),
        source_start_utc=_format_utc(source_start),
        source_end_utc=_format_utc(source_end),
    )


def _read_slab_from_clickhouse(
    *,
    gateway: ClickHouseGateway,
    database: str,
    market_id: int,
    symbol: str,
    start: datetime,
    end: datetime,
) -> RawFeatureSlab:
    expected_rows = _minute_count(start, end)
    q = f"""
    SELECT
        toUnixTimestamp64Milli(ts_open) AS open_time_ms,
        toUnixTimestamp64Milli(ts_close) AS close_time_ms,
        toFloat32(open) AS open_f32,
        toFloat32(high) AS high_f32,
        toFloat32(low) AS low_f32,
        toFloat32(close) AS close_f32,
        toFloat32(volume_base) AS volume_base_f32,
        toFloat32(volume_quote) AS volume_quote_f32,
        toInt64(trades_count) AS trades_count_i64
    FROM {database}.canonical_candles_1m FINAL
    WHERE market_id = %(market_id)s
      AND symbol = %(symbol)s
      AND toUnixTimestamp64Milli(ts_open) >= %(start_ms)s
      AND toUnixTimestamp64Milli(ts_open) < %(end_ms)s
    ORDER BY ts_open
    """
    rows = gateway.select(
        q,
        {
            "end_ms": int(end.timestamp() * 1000),
            "market_id": market_id,
            "start_ms": int(start.timestamp() * 1000),
            "symbol": symbol,
        },
    )
    if len(rows) != expected_rows:
        raise SessionizedDatasetError(
            reason="candle_row_count_mismatch",
            field=f"{symbol}:{_format_utc(start)}:{_format_utc(end)}",
        )
    return build_raw_feature_slab_v1(_rows_to_batch(rows))


def _rows_to_batch(rows: Sequence[Mapping[str, Any]]) -> RawFeatureCandleBatch:
    row_count = len(rows)
    return RawFeatureCandleBatch(
        open_time_ms=np.fromiter(
            (int(_required(row, "open_time_ms")) for row in rows),
            dtype=np.int64,
            count=row_count,
        ),
        close_time_ms=np.fromiter(
            (int(_required(row, "close_time_ms")) for row in rows),
            dtype=np.int64,
            count=row_count,
        ),
        open_f32=_float32_column(rows, "open_f32"),
        high_f32=_float32_column(rows, "high_f32"),
        low_f32=_float32_column(rows, "low_f32"),
        close_f32=_float32_column(rows, "close_f32"),
        volume_base_f32=_float32_column(rows, "volume_base_f32"),
        volume_quote_f32=_float32_column(rows, "volume_quote_f32"),
        trades_count_i64=np.fromiter(
            (int(_required(row, "trades_count_i64")) for row in rows),
            dtype=np.int64,
            count=row_count,
        ),
    )


def _float32_column(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.fromiter(
        (float(_required(row, field)) for row in rows),
        dtype=np.float32,
        count=len(rows),
    )


def _required(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None:
        raise SessionizedDatasetError(reason="missing_required_feature_field", field=field)
    return value


def _rejected_payload(
    *,
    source_window: RawFeatureSourceWindow,
    split_window: SessionSplitWindow,
    reason: str,
) -> dict[str, object]:
    return {
        "dataset_version": source_window.dataset_version,
        "reason": reason,
        "signal_end_utc": split_window.signal_end_utc,
        "signal_start_utc": split_window.signal_start_utc,
        "split": split_window.split,
        "symbol": source_window.symbol,
    }


def _file_payload(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": _file_sha256_hex(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_raw_feature_json_payload_v1(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minute_count(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() // 60)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 06 sessionized dataset artifacts.")
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--from-clickhouse", action="store_true")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--dataset-version", action="append", default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--max-minutes-per-source-window", type=int, default=None)
    parser.add_argument("--signal-stride-minutes", type=int, default=30)
    parser.add_argument("--high-volatility-top-fraction", type=float, default=0.01)
    parser.add_argument("--max-sessions-per-symbol-split", type=int, default=64)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


if __name__ == "__main__":
    raise SystemExit(main())
