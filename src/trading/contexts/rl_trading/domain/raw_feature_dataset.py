from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import numpy as np

from .feature_contract import (
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_DTYPE_V1,
    FEATURE_NAMES_V1,
    FeatureContractViolation,
    RlFeatureCandle,
    build_article_feature_vector_v1,
    training_source_matrix_payload_v1,
)

RAW_FEATURE_DATASET_SCHEMA_VERSION_V1 = 1
RAW_FEATURE_DATASET_MANIFEST_KIND_V1 = "rl_trading_raw_feature_dataset_manifest"
RAW_FEATURE_SLAB_KIND_V1 = "rl_trading_raw_feature_slab"
RAW_FEATURE_PARITY_FIXTURE_KIND_V1 = "rl_trading_feature_parity_fixture"
RAW_FEATURE_RANGE_SEMANTICS_V1 = "half-open UTC [start, end)"
BLOCKED_NOT_TRAINING_SOURCE_REASON_V1 = "blocked_not_training_source_v1"

RawFeatureDatasetStatus = Literal["accepted", "blocked", "partial"]


class RawFeatureDatasetError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RawFeatureSourceWindow:
    dataset_version: str
    symbol: str
    market_id: int
    source_start_utc: str
    source_end_utc: str
    expected_minutes: int

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "expected_minutes": self.expected_minutes,
            "market_id": self.market_id,
            "range_semantics": RAW_FEATURE_RANGE_SEMANTICS_V1,
            "source_end_utc": self.source_end_utc,
            "source_start_utc": self.source_start_utc,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class RawFeatureCandleBatch:
    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    open_f32: np.ndarray
    high_f32: np.ndarray
    low_f32: np.ndarray
    close_f32: np.ndarray
    volume_base_f32: np.ndarray
    volume_quote_f32: np.ndarray
    trades_count_i64: np.ndarray

    def __post_init__(self) -> None:
        _validate_i64_vector("open_time_ms", self.open_time_ms)
        _validate_i64_vector("close_time_ms", self.close_time_ms)
        for field in (
            "open_f32",
            "high_f32",
            "low_f32",
            "close_f32",
            "volume_base_f32",
            "volume_quote_f32",
        ):
            _validate_f32_vector(field, cast(np.ndarray, getattr(self, field)))
        _validate_i64_vector("trades_count_i64", self.trades_count_i64)

        row_count = self.row_count()
        for field in (
            "close_time_ms",
            "open_f32",
            "high_f32",
            "low_f32",
            "close_f32",
            "volume_base_f32",
            "volume_quote_f32",
            "trades_count_i64",
        ):
            if int(cast(np.ndarray, getattr(self, field)).shape[0]) != row_count:
                raise RawFeatureDatasetError(reason="unaligned_candle_batch", field=field)

        if row_count == 0:
            return
        if np.any(self.close_time_ms <= self.open_time_ms):
            raise RawFeatureDatasetError(reason="invalid_time_range", field="close_time_ms")
        if np.any(np.diff(self.open_time_ms) <= 0):
            raise RawFeatureDatasetError(reason="non_monotonic_open_time", field="open_time_ms")
        if np.any(self.high_f32 < np.maximum(self.open_f32, self.close_f32)):
            raise RawFeatureDatasetError(reason="invalid_ohlc_high", field="high_f32")
        if np.any(self.low_f32 > np.minimum(self.open_f32, self.close_f32)):
            raise RawFeatureDatasetError(reason="invalid_ohlc_low", field="low_f32")
        if np.any(self.volume_base_f32 < 0.0):
            raise RawFeatureDatasetError(reason="negative_volume_base", field="volume_base_f32")
        if np.any(self.volume_quote_f32 < 0.0):
            raise RawFeatureDatasetError(reason="negative_volume_quote", field="volume_quote_f32")
        if np.any(self.trades_count_i64 < 0):
            raise RawFeatureDatasetError(reason="negative_trades_count", field="trades_count_i64")

    def row_count(self) -> int:
        return int(self.open_time_ms.shape[0])

    def feature_candle_at(self, index: int) -> RlFeatureCandle:
        return RlFeatureCandle(
            open=float(self.open_f32[index]),
            high=float(self.high_f32[index]),
            low=float(self.low_f32[index]),
            close=float(self.close_f32[index]),
            volume_base=float(self.volume_base_f32[index]),
            volume_quote=float(self.volume_quote_f32[index]),
            trades_count=int(self.trades_count_i64[index]),
        )


@dataclass(frozen=True, slots=True)
class RawFeatureSlab:
    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    features_f32: np.ndarray

    def __post_init__(self) -> None:
        _validate_i64_vector("open_time_ms", self.open_time_ms)
        _validate_i64_vector("close_time_ms", self.close_time_ms)
        if self.features_f32.dtype != np.dtype(np.float32):
            raise RawFeatureDatasetError(reason="invalid_features_dtype", field="features_f32")
        if self.features_f32.ndim != 2 or self.features_f32.shape[1] != len(FEATURE_NAMES_V1):
            raise RawFeatureDatasetError(reason="invalid_features_shape", field="features_f32")
        if int(self.close_time_ms.shape[0]) != self.row_count():
            raise RawFeatureDatasetError(reason="unaligned_slab", field="close_time_ms")
        if int(self.features_f32.shape[0]) != self.row_count():
            raise RawFeatureDatasetError(reason="unaligned_slab", field="features_f32")
        if self.row_count() and not np.all(np.isfinite(self.features_f32)):
            raise RawFeatureDatasetError(reason="non_finite_features", field="features_f32")

    def row_count(self) -> int:
        return int(self.open_time_ms.shape[0])


def training_source_gate_payload_v1(*, exchange: str, market_type: str) -> dict[str, str]:
    normalized = (exchange.strip().lower(), market_type.strip().lower())
    for row in training_source_matrix_payload_v1():
        if (row["exchange"], row["market_type"]) == normalized:
            return dict(row)
    return {
        "exchange": normalized[0],
        "market_type": normalized[1],
        "reason": "unknown exchange/market branch is not a v1 training source",
        "status": BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
    }


def assert_trainable_source_v1(*, exchange: str, market_type: str) -> None:
    gate = training_source_gate_payload_v1(exchange=exchange, market_type=market_type)
    if gate["status"] != "trainable":
        raise RawFeatureDatasetError(
            reason=BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
            field=f"{gate['exchange']}:{gate['market_type']}",
        )


def raw_feature_source_windows_from_stage04c_v1(
    *,
    manifest: Mapping[str, Any],
    dataset_versions: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> tuple[RawFeatureSourceWindow, ...]:
    _validate_stage04c_manifest(manifest)
    allowed_versions = None if dataset_versions is None else {str(v) for v in dataset_versions}
    allowed_symbols = None if symbols is None else {str(s).upper() for s in symbols}
    market_id = int(manifest["market_id"])

    windows: list[RawFeatureSourceWindow] = []
    for version in _dataset_version_rows(manifest):
        dataset_version = str(version["dataset_version"])
        if allowed_versions is not None and dataset_version not in allowed_versions:
            continue
        if version.get("status") != "accepted":
            raise RawFeatureDatasetError(
                reason="dataset_version_not_accepted",
                field=dataset_version,
            )
        included_symbols = {str(symbol).upper() for symbol in version["included_symbols"]}
        for row in _symbol_source_windows(version):
            symbol = str(row["symbol"]).upper()
            if symbol not in included_symbols:
                continue
            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue
            if row.get("status") != "accepted":
                raise RawFeatureDatasetError(
                    reason="symbol_window_not_accepted",
                    field=f"{dataset_version}:{symbol}",
                )
            expected_minutes = int(row.get("expected_minutes") or 0)
            if expected_minutes <= 0:
                raise RawFeatureDatasetError(
                    reason="empty_symbol_source_window",
                    field=f"{dataset_version}:{symbol}",
                )
            windows.append(
                RawFeatureSourceWindow(
                    dataset_version=dataset_version,
                    symbol=symbol,
                    market_id=market_id,
                    source_start_utc=str(row["safe_source_start_utc"]),
                    source_end_utc=str(row["safe_source_end_utc"]),
                    expected_minutes=expected_minutes,
                )
            )

    windows.sort(key=lambda item: (item.dataset_version, item.symbol))
    if allowed_versions is not None:
        found_versions = {item.dataset_version for item in windows}
        missing = sorted(allowed_versions - found_versions)
        if missing:
            raise RawFeatureDatasetError(
                reason="requested_dataset_version_not_found",
                field=missing[0],
            )
    if allowed_symbols is not None:
        found_symbols = {item.symbol for item in windows}
        missing = sorted(allowed_symbols - found_symbols)
        if missing:
            raise RawFeatureDatasetError(reason="requested_symbol_not_found", field=missing[0])
    return tuple(windows)


def build_raw_feature_slab_v1(batch: RawFeatureCandleBatch) -> RawFeatureSlab:
    row_count = batch.row_count()
    features = np.empty((row_count, len(FEATURE_NAMES_V1)), dtype=np.float32)
    if row_count == 0:
        return RawFeatureSlab(
            open_time_ms=np.ascontiguousarray(batch.open_time_ms, dtype=np.int64),
            close_time_ms=np.ascontiguousarray(batch.close_time_ms, dtype=np.int64),
            features_f32=features,
        )

    vwap = _derive_vwap_vector_v1(batch)
    features[:, 0] = batch.open_f32
    features[:, 1] = batch.high_f32
    features[:, 2] = vwap
    features[:, 3] = batch.low_f32
    features[:, 4] = batch.close_f32
    features[:, 5] = batch.volume_base_f32
    features[:, 6] = batch.trades_count_i64.astype(np.float32)

    return RawFeatureSlab(
        open_time_ms=np.ascontiguousarray(batch.open_time_ms, dtype=np.int64),
        close_time_ms=np.ascontiguousarray(batch.close_time_ms, dtype=np.int64),
        features_f32=np.ascontiguousarray(features, dtype=np.float32),
    )


def feature_stats_payload_v1(features_f32: np.ndarray) -> tuple[dict[str, object], ...]:
    if features_f32.dtype != np.dtype(np.float32):
        raise RawFeatureDatasetError(reason="invalid_features_dtype", field="features_f32")
    if features_f32.ndim != 2 or features_f32.shape[1] != len(FEATURE_NAMES_V1):
        raise RawFeatureDatasetError(reason="invalid_features_shape", field="features_f32")
    if int(features_f32.shape[0]) == 0:
        raise RawFeatureDatasetError(reason="empty_features", field="features_f32")

    stats: list[dict[str, object]] = []
    for idx, name in enumerate(FEATURE_NAMES_V1):
        column = features_f32[:, idx]
        nonfinite_count = int(np.count_nonzero(~np.isfinite(column)))
        if nonfinite_count:
            raise RawFeatureDatasetError(reason="non_finite_features", field=name)
        stats.append(
            {
                "feature": name,
                "max": float(np.max(column)),
                "mean": float(np.mean(column, dtype=np.float64)),
                "min": float(np.min(column)),
                "nonfinite_count": nonfinite_count,
                "std": float(np.std(column, dtype=np.float64)),
            }
        )
    return tuple(stats)


def hash_ndarray_payload_v1(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def hash_raw_feature_slab_payload_v1(slab: RawFeatureSlab) -> str:
    payload = {
        "close_time_ms_sha256": hash_ndarray_payload_v1(slab.close_time_ms),
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "features_f32_sha256": hash_ndarray_payload_v1(slab.features_f32),
        "open_time_ms_sha256": hash_ndarray_payload_v1(slab.open_time_ms),
        "schema_version": RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
        "slab_kind": RAW_FEATURE_SLAB_KIND_V1,
    }
    return hash_json_payload_v1(payload)


def hash_json_payload_v1(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def render_raw_feature_json_payload_v1(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def build_raw_feature_slab_manifest_entry_v1(
    *,
    source_window: RawFeatureSourceWindow,
    slab: RawFeatureSlab,
    feature_stats: Sequence[Mapping[str, object]],
    artifact_files: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    rebuild_hash = hash_raw_feature_slab_payload_v1(slab)
    return {
        "schema_version": RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
        "artifact_kind": RAW_FEATURE_SLAB_KIND_V1,
        "dataset_version": source_window.dataset_version,
        "deterministic_rebuild_hash": rebuild_hash,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "feature_stats": list(feature_stats),
        "files": dict(artifact_files),
        "market": "binance:futures",
        "market_id": source_window.market_id,
        "range_semantics": RAW_FEATURE_RANGE_SEMANTICS_V1,
        "row_count": slab.row_count(),
        "source_end_utc": source_window.source_end_utc,
        "source_start_utc": source_window.source_start_utc,
        "symbol": source_window.symbol,
    }


def build_raw_feature_dataset_manifest_v1(
    *,
    generated_at_utc: datetime,
    stage04c_manifest_path: str,
    stage04c_manifest_sha256: str,
    output_root: str,
    slab_entries: Sequence[Mapping[str, Any]],
    parity_fixture: Mapping[str, Any] | None,
    build_scope: Mapping[str, Any],
) -> dict[str, Any]:
    entries = sorted(
        (dict(entry) for entry in slab_entries),
        key=lambda item: (str(item["dataset_version"]), str(item["symbol"])),
    )
    status: RawFeatureDatasetStatus = "accepted" if entries else "blocked"
    return {
        "schema_version": RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
        "manifest_kind": RAW_FEATURE_DATASET_MANIFEST_KIND_V1,
        "stage": "05",
        "status": status,
        "generated_at_utc": _format_utc(generated_at_utc),
        "market": "binance:futures",
        "stage04c_manifest": {
            "path": stage04c_manifest_path,
            "sha256": stage04c_manifest_sha256,
        },
        "feature_contract_dependency": {
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "feature_dtype": FEATURE_DTYPE_V1,
            "feature_names": list(FEATURE_NAMES_V1),
            "source": "Stage 02B accepted feature/live-feed contract",
        },
        "build_scope": dict(build_scope),
        "output_root": output_root,
        "slabs": entries,
        "slab_count": len(entries),
        "total_rows": sum(int(entry["row_count"]) for entry in entries),
        "deterministic_rebuild_hash": hash_json_payload_v1(
            {
                "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
                "slabs": [
                    {
                        "dataset_version": entry["dataset_version"],
                        "deterministic_rebuild_hash": entry["deterministic_rebuild_hash"],
                        "symbol": entry["symbol"],
                    }
                    for entry in entries
                ],
                "stage04c_manifest_sha256": stage04c_manifest_sha256,
            }
        ),
        "blocked_training_sources": _blocked_training_sources_payload(),
        "golden_feature_parity_fixture": None if parity_fixture is None else dict(parity_fixture),
        "safety": {
            "contains_sessionized_training_dataset": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "market_data_writes": False,
            "runtime_artifact_root": "/opt/roehub/state/rl_trading/",
        },
    }


def build_golden_feature_parity_fixture_v1(
    *,
    source_window: RawFeatureSourceWindow,
    batch: RawFeatureCandleBatch,
    offline_features_f32: np.ndarray,
    row_indices: Sequence[int],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if offline_features_f32.dtype != np.dtype(np.float32):
        raise RawFeatureDatasetError(reason="invalid_features_dtype", field="offline_features_f32")
    if offline_features_f32.ndim != 2 or offline_features_f32.shape[1] != len(FEATURE_NAMES_V1):
        raise RawFeatureDatasetError(reason="invalid_features_shape", field="offline_features_f32")
    if int(offline_features_f32.shape[0]) != batch.row_count():
        raise RawFeatureDatasetError(
            reason="unaligned_parity_fixture",
            field="offline_features_f32",
        )
    if not row_indices:
        raise RawFeatureDatasetError(reason="empty_parity_fixture", field="row_indices")

    samples: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for index in row_indices:
        if index < 0 or index >= batch.row_count():
            raise RawFeatureDatasetError(reason="parity_index_out_of_range", field=str(index))
        live_vector = _feature_vector_tuple(batch.feature_candle_at(index))
        offline_vector = tuple(float(v) for v in offline_features_f32[index].tolist())
        diffs = [abs(left - right) for left, right in zip(offline_vector, live_vector)]
        sample_max_abs_diff = max(diffs) if diffs else 0.0
        max_abs_diff = max(max_abs_diff, sample_max_abs_diff)
        samples.append(
            {
                "close_time_ms": int(batch.close_time_ms[index]),
                "live_equivalent_vector": list(live_vector),
                "max_abs_diff": sample_max_abs_diff,
                "offline_vector": list(offline_vector),
                "open_time_ms": int(batch.open_time_ms[index]),
                "row_index": int(index),
            }
        )

    if max_abs_diff > tolerance:
        raise RawFeatureDatasetError(reason="feature_parity_mismatch", field="max_abs_diff")

    return {
        "schema_version": RAW_FEATURE_DATASET_SCHEMA_VERSION_V1,
        "artifact_kind": RAW_FEATURE_PARITY_FIXTURE_KIND_V1,
        "dataset_version": source_window.dataset_version,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "market": "binance:futures",
        "max_abs_diff": max_abs_diff,
        "sample_count": len(samples),
        "samples": samples,
        "source": "offline raw slab vs live-equivalent RlFeatureCandle builder",
        "symbol": source_window.symbol,
        "tolerance": tolerance,
    }


def _derive_vwap_vector_v1(batch: RawFeatureCandleBatch) -> np.ndarray:
    volume = batch.volume_base_f32
    quote = batch.volume_quote_f32
    vwap = np.empty(batch.row_count(), dtype=np.float32)
    positive_volume = volume > 0.0
    zero_volume = volume == 0.0
    inconsistent_zero = zero_volume & (quote != 0.0)
    if np.any(inconsistent_zero):
        raise RawFeatureDatasetError(
            reason="inconsistent_zero_base_positive_quote_volume",
            field="volume_quote_f32",
        )
    vwap[positive_volume] = (
        quote[positive_volume].astype(np.float64) / volume[positive_volume].astype(np.float64)
    ).astype(np.float32)
    vwap[zero_volume] = batch.close_f32[zero_volume]
    if not np.all(np.isfinite(vwap)):
        raise RawFeatureDatasetError(reason="non_finite_vwap", field="volume_weighted_average")
    return vwap


def _feature_vector_tuple(candle: RlFeatureCandle) -> tuple[float, ...]:
    try:
        return tuple(float(np.float32(value)) for value in build_article_feature_vector_v1(candle))
    except FeatureContractViolation as exc:
        raise RawFeatureDatasetError(reason=exc.reason, field=exc.field) from exc


def _validate_stage04c_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage") != "04C":
        raise RawFeatureDatasetError(reason="unexpected_refresh_manifest_stage", field="stage")
    if manifest.get("acceptance_status") != "accepted":
        raise RawFeatureDatasetError(
            reason="refresh_manifest_not_accepted",
            field="acceptance_status",
        )
    if manifest.get("market") != "binance:futures":
        raise RawFeatureDatasetError(reason="unexpected_refresh_manifest_market", field="market")
    dependency = manifest.get("feature_contract_dependency")
    if not isinstance(dependency, Mapping):
        raise RawFeatureDatasetError(reason="missing_feature_contract_dependency")
    if dependency.get("feature_contract_hash") != FEATURE_CONTRACT_HASH_V1:
        raise RawFeatureDatasetError(reason="feature_contract_hash_mismatch")


def _dataset_version_rows(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = manifest.get("dataset_versions")
    if not isinstance(rows, Sequence):
        raise RawFeatureDatasetError(reason="dataset_versions_not_sequence")
    return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))


def _symbol_source_windows(version: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = version.get("symbol_source_windows")
    if not isinstance(rows, Sequence):
        raise RawFeatureDatasetError(reason="symbol_source_windows_not_sequence")
    return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))


def _blocked_training_sources_payload() -> list[dict[str, str]]:
    return [
        dict(row)
        for row in training_source_matrix_payload_v1()
        if row["status"] == BLOCKED_NOT_TRAINING_SOURCE_REASON_V1
    ]


def _validate_i64_vector(field: str, array: np.ndarray) -> None:
    if array.dtype != np.dtype(np.int64):
        raise RawFeatureDatasetError(reason="invalid_array_dtype", field=field)
    if array.ndim != 1:
        raise RawFeatureDatasetError(reason="invalid_array_shape", field=field)


def _validate_f32_vector(field: str, array: np.ndarray) -> None:
    if array.dtype != np.dtype(np.float32):
        raise RawFeatureDatasetError(reason="invalid_array_dtype", field=field)
    if array.ndim != 1:
        raise RawFeatureDatasetError(reason="invalid_array_shape", field=field)
    if array.size and not np.all(np.isfinite(array)):
        raise RawFeatureDatasetError(reason="non_finite_array", field=field)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
