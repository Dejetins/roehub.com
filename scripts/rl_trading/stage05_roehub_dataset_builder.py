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
from trading.contexts.rl_trading.domain.feature_contract import (  # noqa: E402
    FEATURE_CONTRACT_HASH_V1,
)
from trading.contexts.rl_trading.domain.raw_feature_dataset import (  # noqa: E402
    BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
    RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
    RAW_FEATURE_SLAB_KIND_V1,
    RawFeatureCandleBatch,
    RawFeatureDatasetError,
    RawFeatureSlab,
    RawFeatureSourceWindow,
    assert_trainable_source_v1,
    build_golden_feature_parity_fixture_v1,
    build_raw_feature_dataset_manifest_v1,
    build_raw_feature_slab_manifest_entry_v1,
    build_raw_feature_slab_v1,
    feature_stats_payload_v1,
    raw_feature_source_windows_from_stage04c_v1,
    render_raw_feature_json_payload_v1,
    training_source_gate_payload_v1,
)

DEFAULT_STAGE04C_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest")
    / "stage04c_dataset_refresh_manifest.json"
)
DEFAULT_STAGE04C_MANIFEST_SHA256 = (
    "9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1"
)
DEFAULT_CHUNK_MINUTES = 7 * 24 * 60


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gate = training_source_gate_payload_v1(exchange=args.exchange, market_type=args.market_type)
    if gate["status"] != "trainable":
        print(
            _render_status(
                {
                    "exchange": gate["exchange"],
                    "market_type": gate["market_type"],
                    "reason": BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
                    "status": "blocked",
                }
            )
        )
        return 2

    try:
        manifest = build_raw_feature_dataset_from_clickhouse(
            refresh_manifest_path=args.refresh_manifest,
            expected_refresh_manifest_sha256=args.expected_refresh_manifest_sha256,
            output_root=args.output_root,
            symbols=args.symbol,
            all_symbols=bool(args.all_symbols),
            dataset_versions=args.dataset_version,
            max_symbols=args.max_symbols,
            max_minutes_per_symbol=args.max_minutes_per_symbol,
            chunk_minutes=int(args.chunk_minutes),
            resume_existing_slabs=bool(args.resume_existing_slabs),
            generated_at_utc=(
                _parse_utc(args.generated_at_utc)
                if args.generated_at_utc is not None
                else _now_utc()
            ),
        )
    except RawFeatureDatasetError as exc:
        print(
            _render_status(
                {
                    "field": exc.field,
                    "reason": exc.reason,
                    "status": "blocked",
                }
            )
        )
        return 2

    print(
        _render_status(
            {
                "dataset_manifest": str(args.output_root / "stage05_raw_feature_manifest.json"),
                "deterministic_rebuild_hash": manifest["deterministic_rebuild_hash"],
                "manifest_file_sha256": _file_sha256_hex(
                    args.output_root / "stage05_raw_feature_manifest.json"
                ),
                "slab_count": manifest["slab_count"],
                "status": manifest["status"],
                "total_rows": manifest["total_rows"],
            }
        )
    )
    return 0


def build_raw_feature_dataset_from_clickhouse(
    *,
    refresh_manifest_path: Path,
    expected_refresh_manifest_sha256: str,
    output_root: Path,
    symbols: Sequence[str] | None,
    all_symbols: bool,
    dataset_versions: Sequence[str] | None,
    max_symbols: int | None,
    max_minutes_per_symbol: int | None,
    chunk_minutes: int,
    resume_existing_slabs: bool,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    if chunk_minutes <= 0:
        raise RawFeatureDatasetError(reason="invalid_chunk_minutes", field="chunk_minutes")
    if max_symbols is not None and max_symbols <= 0:
        raise RawFeatureDatasetError(reason="invalid_max_symbols", field="max_symbols")
    if max_minutes_per_symbol is not None and max_minutes_per_symbol <= 0:
        raise RawFeatureDatasetError(
            reason="invalid_max_minutes_per_symbol",
            field="max_minutes_per_symbol",
        )
    if not all_symbols and not symbols:
        raise RawFeatureDatasetError(reason="symbol_selection_required", field="symbol")

    assert_trainable_source_v1(exchange="binance", market_type="futures")
    refresh_manifest_sha256 = _file_sha256_hex(refresh_manifest_path)
    if (
        expected_refresh_manifest_sha256
        and refresh_manifest_sha256 != expected_refresh_manifest_sha256
    ):
        raise RawFeatureDatasetError(
            reason="refresh_manifest_sha256_mismatch",
            field="expected_refresh_manifest_sha256",
        )
    refresh_manifest = _read_json(refresh_manifest_path)
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
        raise RawFeatureDatasetError(reason="no_source_windows_selected")

    settings = ClickHouseSettingsLoader(os.environ).load()
    gateway = ClickHouseConnectGateway(_clickhouse_client(settings))
    output_root.mkdir(parents=True, exist_ok=True)

    slab_entries: list[dict[str, Any]] = []
    parity_fixture: dict[str, Any] | None = None
    for original_window in source_windows:
        window = _bounded_source_window(
            original_window,
            max_minutes_per_symbol=max_minutes_per_symbol,
        )
        entry, candidate_fixture = _materialize_slab(
            gateway=gateway,
            database=settings.database,
            source_window=window,
            output_root=output_root,
            chunk_minutes=chunk_minutes,
            build_parity_fixture=parity_fixture is None,
            resume_existing_slabs=resume_existing_slabs,
        )
        slab_entries.append(entry)
        if parity_fixture is None:
            parity_fixture = candidate_fixture

    build_scope = {
        "all_symbols": all_symbols,
        "dataset_versions": sorted({window.dataset_version for window in source_windows}),
        "max_minutes_per_symbol": max_minutes_per_symbol,
        "max_symbols": max_symbols,
        "selected_symbols": sorted({window.symbol for window in source_windows}),
        "scope": (
            "bounded_sample" if max_minutes_per_symbol is not None else "full_selected_windows"
        ),
    }
    manifest = build_raw_feature_dataset_manifest_v1(
        generated_at_utc=generated_at_utc,
        stage04c_manifest_path=str(refresh_manifest_path),
        stage04c_manifest_sha256=refresh_manifest_sha256,
        output_root=str(output_root),
        slab_entries=slab_entries,
        parity_fixture=parity_fixture,
        build_scope=build_scope,
    )
    _atomic_write_json(output_root / "stage05_raw_feature_manifest.json", manifest)
    if parity_fixture is not None:
        _atomic_write_json(output_root / "stage05_feature_parity_fixture.json", parity_fixture)
    return manifest


def _materialize_slab(
    *,
    gateway: ClickHouseGateway,
    database: str,
    source_window: RawFeatureSourceWindow,
    output_root: Path,
    chunk_minutes: int,
    build_parity_fixture: bool,
    resume_existing_slabs: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if source_window.expected_minutes <= 0:
        raise RawFeatureDatasetError(reason="empty_source_window", field=source_window.symbol)

    slab_root = output_root / source_window.dataset_version / source_window.symbol
    slab_root.mkdir(parents=True, exist_ok=True)
    features_path = slab_root / "features.f32.npy"
    open_time_path = slab_root / "open_time_ms.i64.npy"
    close_time_path = slab_root / "close_time_ms.i64.npy"
    slab_manifest_path = slab_root / "manifest.json"

    if resume_existing_slabs and not build_parity_fixture:
        existing_entry = _existing_slab_entry_if_resumable(
            slab_manifest_path=slab_manifest_path,
            source_window=source_window,
        )
        if existing_entry is not None:
            return existing_entry, None

    features_mm = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(source_window.expected_minutes, 7),
    )
    open_time_mm = np.lib.format.open_memmap(
        open_time_path,
        mode="w+",
        dtype=np.int64,
        shape=(source_window.expected_minutes,),
    )
    close_time_mm = np.lib.format.open_memmap(
        close_time_path,
        mode="w+",
        dtype=np.int64,
        shape=(source_window.expected_minutes,),
    )

    start = _parse_utc(source_window.source_start_utc)
    end = _parse_utc(source_window.source_end_utc)
    offset = 0
    fixture_batch: RawFeatureCandleBatch | None = None
    fixture_features: np.ndarray | None = None
    for chunk_start, chunk_end in _iter_chunks(start=start, end=end, chunk_minutes=chunk_minutes):
        expected_rows = _minute_count(chunk_start, chunk_end)
        batch = _read_candle_batch(
            gateway=gateway,
            database=database,
            market_id=source_window.market_id,
            symbol=source_window.symbol,
            start=chunk_start,
            end=chunk_end,
            expected_rows=expected_rows,
        )
        slab = build_raw_feature_slab_v1(batch)
        rows = slab.row_count()
        open_time_mm[offset : offset + rows] = slab.open_time_ms
        close_time_mm[offset : offset + rows] = slab.close_time_ms
        features_mm[offset : offset + rows, :] = slab.features_f32
        if build_parity_fixture and fixture_batch is None:
            fixture_rows = min(3, rows)
            fixture_batch = _head_batch(batch, fixture_rows)
            fixture_features = np.ascontiguousarray(
                slab.features_f32[:fixture_rows],
                dtype=np.float32,
            )
        offset += rows

    if offset != source_window.expected_minutes:
        raise RawFeatureDatasetError(
            reason="materialized_row_count_mismatch",
            field=source_window.symbol,
        )
    features_mm.flush()
    open_time_mm.flush()
    close_time_mm.flush()
    del features_mm
    del open_time_mm
    del close_time_mm

    full_slab = RawFeatureSlab(
        open_time_ms=np.load(open_time_path, mmap_mode="r"),
        close_time_ms=np.load(close_time_path, mmap_mode="r"),
        features_f32=np.load(features_path, mmap_mode="r"),
    )
    feature_stats = feature_stats_payload_v1(full_slab.features_f32)
    artifact_files = {
        "close_time_ms": _array_file_payload(close_time_path, full_slab.close_time_ms),
        "features": _array_file_payload(features_path, full_slab.features_f32),
        "open_time_ms": _array_file_payload(open_time_path, full_slab.open_time_ms),
    }
    entry = build_raw_feature_slab_manifest_entry_v1(
        source_window=source_window,
        slab=full_slab,
        feature_stats=feature_stats,
        artifact_files=artifact_files,
    )
    entry["manifest_path"] = str(slab_manifest_path)
    _atomic_write_json(slab_manifest_path, entry)

    parity_fixture = None
    if build_parity_fixture and fixture_batch is not None and fixture_features is not None:
        parity_fixture = build_golden_feature_parity_fixture_v1(
            source_window=source_window,
            batch=fixture_batch,
            offline_features_f32=fixture_features,
            row_indices=tuple(range(fixture_batch.row_count())),
        )
    return entry, parity_fixture


def _read_candle_batch(
    *,
    gateway: ClickHouseGateway,
    database: str,
    market_id: int,
    symbol: str,
    start: datetime,
    end: datetime,
    expected_rows: int,
) -> RawFeatureCandleBatch:
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
      AND ts_open >= %(start)s
      AND ts_open < %(end)s
    ORDER BY ts_open
    """
    rows = gateway.select(
        q,
        {
            "end": end,
            "market_id": market_id,
            "start": start,
            "symbol": symbol,
        },
    )
    if len(rows) != expected_rows:
        raise RawFeatureDatasetError(
            reason="candle_row_count_mismatch",
            field=f"{symbol}:{_format_utc(start)}:{_format_utc(end)}",
        )
    return _rows_to_batch(rows)


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
        raise RawFeatureDatasetError(reason="missing_required_feature_field", field=field)
    return value


def _head_batch(batch: RawFeatureCandleBatch, row_count: int) -> RawFeatureCandleBatch:
    return RawFeatureCandleBatch(
        open_time_ms=np.ascontiguousarray(batch.open_time_ms[:row_count], dtype=np.int64),
        close_time_ms=np.ascontiguousarray(batch.close_time_ms[:row_count], dtype=np.int64),
        open_f32=np.ascontiguousarray(batch.open_f32[:row_count], dtype=np.float32),
        high_f32=np.ascontiguousarray(batch.high_f32[:row_count], dtype=np.float32),
        low_f32=np.ascontiguousarray(batch.low_f32[:row_count], dtype=np.float32),
        close_f32=np.ascontiguousarray(batch.close_f32[:row_count], dtype=np.float32),
        volume_base_f32=np.ascontiguousarray(batch.volume_base_f32[:row_count], dtype=np.float32),
        volume_quote_f32=np.ascontiguousarray(batch.volume_quote_f32[:row_count], dtype=np.float32),
        trades_count_i64=np.ascontiguousarray(batch.trades_count_i64[:row_count], dtype=np.int64),
    )


def _array_file_payload(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "path": str(path),
        "sha256": _file_sha256_hex(path),
        "shape": [int(value) for value in array.shape],
    }


def _bounded_source_window(
    source_window: RawFeatureSourceWindow,
    *,
    max_minutes_per_symbol: int | None,
) -> RawFeatureSourceWindow:
    if max_minutes_per_symbol is None or max_minutes_per_symbol >= source_window.expected_minutes:
        return source_window
    start = _parse_utc(source_window.source_start_utc)
    end = start + timedelta(minutes=max_minutes_per_symbol)
    return replace(
        source_window,
        source_end_utc=_format_utc(end),
        expected_minutes=max_minutes_per_symbol,
    )


def _iter_chunks(
    *,
    start: datetime,
    end: datetime,
    chunk_minutes: int,
):
    cursor = start
    delta = timedelta(minutes=chunk_minutes)
    while cursor < end:
        next_end = min(cursor + delta, end)
        yield cursor, next_end
        cursor = next_end


def _minute_count(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() // 60)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 05 Roehub raw feature slabs.")
    parser.add_argument("--refresh-manifest", type=Path, default=DEFAULT_STAGE04C_MANIFEST)
    parser.add_argument(
        "--expected-refresh-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE04C_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--dataset-version", action="append", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--max-minutes-per-symbol", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES)
    parser.add_argument(
        "--resume-existing-slabs",
        action="store_true",
        help="Reuse completed per-slab manifests and arrays from output-root.",
    )
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


def _existing_slab_entry_if_resumable(
    *,
    slab_manifest_path: Path,
    source_window: RawFeatureSourceWindow,
) -> dict[str, Any] | None:
    if not slab_manifest_path.exists():
        return None

    entry = _read_json(slab_manifest_path)
    expected_fields: Mapping[str, object] = {
        "artifact_kind": RAW_FEATURE_SLAB_KIND_V1,
        "dataset_version": source_window.dataset_version,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "market": "binance:futures",
        "market_id": source_window.market_id,
        "row_count": source_window.expected_minutes,
        "schema_version": RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
        "source_end_utc": source_window.source_end_utc,
        "source_start_utc": source_window.source_start_utc,
        "symbol": source_window.symbol,
    }
    for field, expected in expected_fields.items():
        if entry.get(field) != expected:
            raise RawFeatureDatasetError(
                reason="resume_slab_manifest_mismatch",
                field=f"{slab_manifest_path}:{field}",
            )

    files = entry.get("files")
    if not isinstance(files, Mapping):
        raise RawFeatureDatasetError(
            reason="resume_slab_manifest_mismatch",
            field=f"{slab_manifest_path}:files",
        )
    for file_key in ("features", "open_time_ms", "close_time_ms"):
        file_payload = files.get(file_key)
        if not isinstance(file_payload, Mapping):
            raise RawFeatureDatasetError(
                reason="resume_slab_manifest_mismatch",
                field=f"{slab_manifest_path}:files.{file_key}",
            )
        file_path = Path(str(file_payload.get("path") or ""))
        if not file_path.exists():
            raise RawFeatureDatasetError(
                reason="resume_slab_file_missing",
                field=str(file_path),
            )
    entry["manifest_path"] = str(slab_manifest_path)
    return entry


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
