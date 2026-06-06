from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trading.contexts.backtest.application.dto import PreparedIndicatorPool

BITS_PER_WORD = 64
SIDECAR_SCHEMA_VERSION = 1
SIDECAR_MANIFEST_FILENAME = "matrix_sidecar_manifest.json"
SIDECAR_POS_BITS_FILENAME = "signals_pos_bits.u64.npy"
SIDECAR_NEG_BITS_FILENAME = "signals_neg_bits.u64.npy"
SIDECAR_ROW_HASHES_FILENAME = "signal_row_hashes.u64.npy"
SIDECAR_UNIQUE_ROW_IDS_FILENAME = "unique_signal_row_ids.u32.npy"
SIDECAR_DUPLICATE_ROW_IDS_FILENAME = "duplicate_signal_row_ids.u32.npy"
SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME = "duplicate_unique_signal_row_ids.u32.npy"
SIDECAR_PADDING_POLICY = "zeroed_trailing_bits"
SIDECAR_BIT_ORDER = "little_endian_lsb_first_per_uint64_word"


@dataclass(frozen=True, slots=True)
class PackedSignalBitsets:
    """
    Runtime bitset representation for one prepared signal matrix.

    Bit order is little-endian inside each uint64 word: bar `t` maps to
    `word = t // 64`, `bit = t % 64`. Padding bits after `signal_length` are zero.
    """

    pos_bits: np.ndarray
    neg_bits: np.ndarray
    signal_length: int
    word_count: int

    @property
    def packed_bytes(self) -> int:
        return int(self.pos_bits.nbytes + self.neg_bits.nbytes)


@dataclass(frozen=True, slots=True)
class MatrixSidecarArtifacts:
    pos_bits: np.ndarray
    neg_bits: np.ndarray
    signal_row_hashes: np.ndarray
    unique_signal_row_ids: np.ndarray
    duplicate_signal_row_ids: np.ndarray
    duplicate_unique_signal_row_ids: np.ndarray
    manifest: dict[str, Any]

    @property
    def packed(self) -> PackedSignalBitsets:
        return PackedSignalBitsets(
            pos_bits=np.ascontiguousarray(np.asarray(self.pos_bits, dtype=np.uint64)),
            neg_bits=np.ascontiguousarray(np.asarray(self.neg_bits, dtype=np.uint64)),
            signal_length=int(self.manifest["source_signal_shape"][1]),
            word_count=int(self.manifest["word_count"]),
        )


@dataclass(frozen=True, slots=True)
class SidecarBitsetLoadResult:
    packed_by_indicator: tuple[PackedSignalBitsets, ...]
    sidecar_used: bool
    sidecar_available: bool
    sidecar_load_ms: float | None
    sidecar_fallback_reason: str | None
    sidecar_dir: str | None


@dataclass(frozen=True, slots=True)
class IndicatorBitsetPackTelemetry:
    indicator_id: str
    rows_after_prefilter: int
    signal_length: int
    word_count: int
    padding_bits: int
    positive_signal_count: int
    negative_signal_count: int
    neutral_signal_count: int
    packed_bytes: int
    padding_valid: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "rows_after_prefilter": self.rows_after_prefilter,
            "signal_length": self.signal_length,
            "word_count": self.word_count,
            "padding_bits": self.padding_bits,
            "positive_signal_count": self.positive_signal_count,
            "negative_signal_count": self.negative_signal_count,
            "neutral_signal_count": self.neutral_signal_count,
            "packed_bytes": self.packed_bytes,
            "padding_valid": self.padding_valid,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBitsetPackTelemetry:
    rows_after_prefilter: int
    signal_length: int
    word_count: int
    padding_bits: int
    packed_bytes: int
    estimated_peak_bytes: int
    padding_valid: bool
    consensus_sample_count: int
    consensus_sample_mismatches: int
    consensus_sample_parity: bool
    arrays_released_before_return: bool
    indicators: tuple[IndicatorBitsetPackTelemetry, ...]
    source: str = "runtime_pack"
    sidecar_used: bool = False
    sidecar_available: bool = False
    sidecar_load_ms: float | None = None
    sidecar_fallback_reason: str | None = None
    sidecar_dir: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_runtime_bitset_pack_shadow_v1",
            "source": self.source,
            "sidecar_used": self.sidecar_used,
            "sidecar_available": self.sidecar_available,
            "sidecar_load_ms": self.sidecar_load_ms,
            "sidecar_fallback_reason": self.sidecar_fallback_reason,
            "sidecar_dir": self.sidecar_dir,
            "rows_after_prefilter": self.rows_after_prefilter,
            "signal_length": self.signal_length,
            "word_count": self.word_count,
            "word_count_formula": "W = ceil(T / 64)",
            "padding_bits": self.padding_bits,
            "packed_bytes": self.packed_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "padding_valid": self.padding_valid,
            "consensus_sample_count": self.consensus_sample_count,
            "consensus_sample_mismatches": self.consensus_sample_mismatches,
            "consensus_sample_parity": self.consensus_sample_parity,
            "arrays_released_before_return": self.arrays_released_before_return,
            "bit_order": "little_endian_lsb_first_per_uint64_word",
            "arrays": {
                "pos_bits": {"dtype": "uint64", "shape": ["rows", "W"]},
                "neg_bits": {"dtype": "uint64", "shape": ["rows", "W"]},
            },
            "indicators": [indicator.as_mapping() for indicator in self.indicators],
        }


def word_count_for_signal_length(signal_length: int) -> int:
    if signal_length <= 0:
        raise ValueError("signal_length must be > 0")
    return (int(signal_length) + BITS_PER_WORD - 1) // BITS_PER_WORD


def pack_signal_matrix(trade_t: np.ndarray) -> PackedSignalBitsets:
    """
    Pack one current `+1/0/-1` signal matrix into positive and negative bitsets.
    """

    signal_matrix = np.ascontiguousarray(np.asarray(trade_t, dtype=np.int8))
    if signal_matrix.ndim != 2:
        raise ValueError("trade_t must be a 2D matrix")
    rows, signal_length = (int(signal_matrix.shape[0]), int(signal_matrix.shape[1]))
    if rows <= 0 or signal_length <= 0:
        raise ValueError("trade_t must be non-empty")
    invalid = (signal_matrix != np.int8(1)) & (signal_matrix != np.int8(0))
    invalid &= signal_matrix != np.int8(-1)
    if bool(np.any(invalid)):
        raise ValueError("trade_t contains values outside +1/0/-1")

    word_count = word_count_for_signal_length(signal_length)
    return PackedSignalBitsets(
        pos_bits=_pack_bool_matrix(signal_matrix == np.int8(1), word_count=word_count),
        neg_bits=_pack_bool_matrix(signal_matrix == np.int8(-1), word_count=word_count),
        signal_length=signal_length,
        word_count=word_count,
    )


def build_matrix_sidecar_artifacts(
    *,
    signal_matrix: np.ndarray,
    source_manifest_path: Path,
    source_signals_path: Path,
    output_dir: Path,
    identity: dict[str, Any],
) -> MatrixSidecarArtifacts:
    """
    Generate deterministic test/benchmark sidecar bitsets from one canonical signal matrix.
    """

    matrix = np.ascontiguousarray(np.asarray(signal_matrix, dtype=np.int8))
    if matrix.ndim != 2:
        raise ValueError("signal_matrix must be a 2D matrix")
    rows, signal_length = (int(matrix.shape[0]), int(matrix.shape[1]))
    if rows <= 0 or signal_length <= 0:
        raise ValueError("signal_matrix must be non-empty")
    packed = pack_signal_matrix(matrix)
    row_hashes = signal_row_hashes_u64(matrix)
    unique_row_ids, duplicate_row_ids, duplicate_unique_row_ids = _duplicate_maps(row_hashes)
    manifest = {
        "schema": "backtest_matrix_sidecar_bitsets_v1",
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": file_sha256_hex(source_manifest_path),
        "source_signals_path": str(source_signals_path),
        "source_signals_sha256": file_sha256_hex(source_signals_path),
        "source_signal_shape": [rows, signal_length],
        "source_signal_dtype": "int8",
        "word_count": packed.word_count,
        "word_count_formula": "W = ceil(T / 64)",
        "padding_bits": (packed.word_count * BITS_PER_WORD) - signal_length,
        "padding_policy": SIDECAR_PADDING_POLICY,
        "bit_order": SIDECAR_BIT_ORDER,
        "identity": dict(identity),
        "artifacts": {
            SIDECAR_POS_BITS_FILENAME: {"dtype": "uint64", "shape": [rows, packed.word_count]},
            SIDECAR_NEG_BITS_FILENAME: {"dtype": "uint64", "shape": [rows, packed.word_count]},
            SIDECAR_ROW_HASHES_FILENAME: {"dtype": "uint64", "shape": [rows]},
            SIDECAR_UNIQUE_ROW_IDS_FILENAME: {
                "dtype": "uint32",
                "shape": [int(unique_row_ids.shape[0])],
            },
            SIDECAR_DUPLICATE_ROW_IDS_FILENAME: {
                "dtype": "uint32",
                "shape": [int(duplicate_row_ids.shape[0])],
            },
            SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME: {
                "dtype": "uint32",
                "shape": [int(duplicate_unique_row_ids.shape[0])],
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / SIDECAR_POS_BITS_FILENAME, packed.pos_bits)
    np.save(output_dir / SIDECAR_NEG_BITS_FILENAME, packed.neg_bits)
    np.save(output_dir / SIDECAR_ROW_HASHES_FILENAME, row_hashes)
    np.save(output_dir / SIDECAR_UNIQUE_ROW_IDS_FILENAME, unique_row_ids)
    np.save(output_dir / SIDECAR_DUPLICATE_ROW_IDS_FILENAME, duplicate_row_ids)
    np.save(output_dir / SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME, duplicate_unique_row_ids)
    (output_dir / SIDECAR_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MatrixSidecarArtifacts(
        pos_bits=packed.pos_bits,
        neg_bits=packed.neg_bits,
        signal_row_hashes=row_hashes,
        unique_signal_row_ids=unique_row_ids,
        duplicate_signal_row_ids=duplicate_row_ids,
        duplicate_unique_signal_row_ids=duplicate_unique_row_ids,
        manifest=manifest,
    )


def load_matrix_sidecar_artifacts(
    sidecar_dir: Path,
    *,
    expected_source_manifest_sha256: str | None = None,
    expected_source_signals_sha256: str | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> MatrixSidecarArtifacts:
    manifest_path = sidecar_dir / SIDECAR_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("matrix sidecar manifest must be a JSON object")
    _validate_sidecar_manifest(
        manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_source_signals_sha256=expected_source_signals_sha256,
        expected_identity=expected_identity,
    )
    _validate_manifest_source_files(manifest)
    artifacts = MatrixSidecarArtifacts(
        pos_bits=np.load(sidecar_dir / SIDECAR_POS_BITS_FILENAME, mmap_mode="r"),
        neg_bits=np.load(sidecar_dir / SIDECAR_NEG_BITS_FILENAME, mmap_mode="r"),
        signal_row_hashes=np.load(sidecar_dir / SIDECAR_ROW_HASHES_FILENAME, mmap_mode="r"),
        unique_signal_row_ids=np.load(sidecar_dir / SIDECAR_UNIQUE_ROW_IDS_FILENAME, mmap_mode="r"),
        duplicate_signal_row_ids=np.load(
            sidecar_dir / SIDECAR_DUPLICATE_ROW_IDS_FILENAME,
            mmap_mode="r",
        ),
        duplicate_unique_signal_row_ids=np.load(
            sidecar_dir / SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME,
            mmap_mode="r",
        ),
        manifest=manifest,
    )
    validate_matrix_sidecar_artifacts(artifacts)
    return artifacts


def load_or_pack_signal_bitsets(
    indicator_pools: Sequence[PreparedIndicatorPool],
    *,
    sidecar_artifact_dir: Path | None = None,
    time_slice_start: int = 0,
    time_slice_stop: int | None = None,
) -> SidecarBitsetLoadResult:
    if sidecar_artifact_dir is None:
        return SidecarBitsetLoadResult(
            packed_by_indicator=tuple(pack_signal_matrix(pool.trade_T) for pool in indicator_pools),
            sidecar_used=False,
            sidecar_available=False,
            sidecar_load_ms=None,
            sidecar_fallback_reason=None,
            sidecar_dir=None,
        )

    started = _perf_counter()
    try:
        packed = tuple(
            _load_prepared_pool_from_sidecar(
                sidecar_artifact_dir,
                pool=pool,
                time_slice_start=time_slice_start,
                time_slice_stop=time_slice_stop,
            )
            for pool in indicator_pools
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        return SidecarBitsetLoadResult(
            packed_by_indicator=tuple(pack_signal_matrix(pool.trade_T) for pool in indicator_pools),
            sidecar_used=False,
            sidecar_available=True,
            sidecar_load_ms=(_perf_counter() - started) * 1000.0,
            sidecar_fallback_reason=str(error),
            sidecar_dir=str(sidecar_artifact_dir),
        )
    return SidecarBitsetLoadResult(
        packed_by_indicator=packed,
        sidecar_used=True,
        sidecar_available=True,
        sidecar_load_ms=(_perf_counter() - started) * 1000.0,
        sidecar_fallback_reason=None,
        sidecar_dir=str(sidecar_artifact_dir),
    )


def signal_row_hashes_u64(signal_matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(signal_matrix, dtype=np.int8))
    if matrix.ndim != 2:
        raise ValueError("signal_matrix must be a 2D matrix")
    hashes = np.empty(int(matrix.shape[0]), dtype=np.uint64)
    for row_index, row in enumerate(matrix):
        digest = hashlib.blake2b(
            np.ascontiguousarray(row).view(np.uint8),
            digest_size=8,
            person=b"rhbtrow1",
        ).digest()
        hashes[row_index] = np.frombuffer(digest, dtype="<u8")[0]
    return hashes


def validate_matrix_sidecar_artifacts(artifacts: MatrixSidecarArtifacts) -> None:
    manifest = artifacts.manifest
    rows, signal_length = _manifest_source_shape(manifest)
    word_count = int(manifest.get("word_count", -1))
    expected_word_count = word_count_for_signal_length(signal_length)
    if word_count != expected_word_count:
        raise ValueError(
            f"matrix sidecar word_count mismatch; got {word_count}, expected "
            f"{expected_word_count}"
        )
    _validate_array(
        artifacts.pos_bits,
        name=SIDECAR_POS_BITS_FILENAME,
        dtype=np.dtype(np.uint64),
        shape=(rows, word_count),
    )
    _validate_array(
        artifacts.neg_bits,
        name=SIDECAR_NEG_BITS_FILENAME,
        dtype=np.dtype(np.uint64),
        shape=(rows, word_count),
    )
    _validate_array(
        artifacts.signal_row_hashes,
        name=SIDECAR_ROW_HASHES_FILENAME,
        dtype=np.dtype(np.uint64),
        shape=(rows,),
    )
    _validate_array(
        artifacts.unique_signal_row_ids,
        name=SIDECAR_UNIQUE_ROW_IDS_FILENAME,
        dtype=np.dtype(np.uint32),
        ndim=1,
    )
    _validate_array(
        artifacts.duplicate_signal_row_ids,
        name=SIDECAR_DUPLICATE_ROW_IDS_FILENAME,
        dtype=np.dtype(np.uint32),
        ndim=1,
    )
    _validate_array(
        artifacts.duplicate_unique_signal_row_ids,
        name=SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME,
        dtype=np.dtype(np.uint32),
        shape=artifacts.duplicate_signal_row_ids.shape,
    )
    if not _padding_valid(artifacts.packed):
        raise ValueError("matrix sidecar padding bits must be zero")
    _validate_duplicate_maps(artifacts, rows=rows)


def unpack_signal_bitsets(packed: PackedSignalBitsets) -> np.ndarray:
    positive = _unpack_word_matrix(
        packed.pos_bits,
        signal_length=packed.signal_length,
    )
    negative = _unpack_word_matrix(
        packed.neg_bits,
        signal_length=packed.signal_length,
    )
    signal = np.zeros(positive.shape, dtype=np.int8)
    signal[positive] = np.int8(1)
    signal[negative] = np.int8(-1)
    return signal


def bitset_consensus_row(
    packed_by_indicator: Sequence[PackedSignalBitsets],
    row_positions: Sequence[int],
) -> np.ndarray:
    if len(packed_by_indicator) != len(row_positions):
        raise ValueError("packed_by_indicator and row_positions length mismatch")
    if not packed_by_indicator:
        return np.empty(0, dtype=np.int8)

    signal_length = packed_by_indicator[0].signal_length
    word_count = packed_by_indicator[0].word_count
    pos_words = np.full(word_count, np.uint64((1 << BITS_PER_WORD) - 1), dtype=np.uint64)
    neg_words = np.full(word_count, np.uint64((1 << BITS_PER_WORD) - 1), dtype=np.uint64)

    for packed, row_position in zip(packed_by_indicator, row_positions, strict=True):
        if packed.signal_length != signal_length or packed.word_count != word_count:
            raise ValueError("all packed indicators must share the same signal length")
        row_idx = int(row_position)
        pos_words &= packed.pos_bits[row_idx]
        neg_words &= packed.neg_bits[row_idx]

    positive = _unpack_words(pos_words, signal_length=signal_length)
    negative = _unpack_words(neg_words, signal_length=signal_length)
    consensus = np.zeros(signal_length, dtype=np.int8)
    consensus[positive] = np.int8(1)
    consensus[negative] = np.int8(-1)
    return consensus


def build_runtime_bitset_pack_telemetry(
    indicator_pools: Sequence[PreparedIndicatorPool],
    *,
    consensus_sample_limit: int = 16,
    sidecar_artifact_dir: Path | None = None,
    time_slice_start: int = 0,
    time_slice_stop: int | None = None,
) -> RuntimeBitsetPackTelemetry:
    """
    Build shadow-only runtime bitsets and validate sampled consensus parity.

    The packed arrays are intentionally not exposed through the returned telemetry.
    Current scoring and top-N continue to consume the existing prepared pools.
    """

    if consensus_sample_limit <= 0:
        raise ValueError("consensus_sample_limit must be > 0")
    packed_by_indicator: tuple[PackedSignalBitsets, ...] | None = None
    try:
        load_result = load_or_pack_signal_bitsets(
            indicator_pools,
            sidecar_artifact_dir=sidecar_artifact_dir,
            time_slice_start=time_slice_start,
            time_slice_stop=time_slice_stop,
        )
        packed_by_indicator = load_result.packed_by_indicator
        indicators = tuple(
            _indicator_telemetry(pool=pool, packed=packed)
            for pool, packed in zip(indicator_pools, packed_by_indicator, strict=True)
        )
        _validate_aligned_word_counts(indicators)
        sample_count, mismatches = _consensus_sample_mismatches(
            indicator_pools=indicator_pools,
            packed_by_indicator=packed_by_indicator,
            sample_limit=consensus_sample_limit,
        )
        packed_bytes = sum(indicator.packed_bytes for indicator in indicators)
        input_bytes = sum(int(np.asarray(pool.trade_T).nbytes) for pool in indicator_pools)
        return RuntimeBitsetPackTelemetry(
            rows_after_prefilter=sum(
                indicator.rows_after_prefilter for indicator in indicators
            ),
            signal_length=0 if not indicators else indicators[0].signal_length,
            word_count=0 if not indicators else indicators[0].word_count,
            padding_bits=0 if not indicators else indicators[0].padding_bits,
            packed_bytes=packed_bytes,
            estimated_peak_bytes=packed_bytes + input_bytes,
            padding_valid=all(indicator.padding_valid for indicator in indicators),
            consensus_sample_count=sample_count,
            consensus_sample_mismatches=mismatches,
            consensus_sample_parity=mismatches == 0,
            arrays_released_before_return=True,
            indicators=indicators,
            source="sidecar" if load_result.sidecar_used else "runtime_pack",
            sidecar_used=load_result.sidecar_used,
            sidecar_available=load_result.sidecar_available,
            sidecar_load_ms=load_result.sidecar_load_ms,
            sidecar_fallback_reason=load_result.sidecar_fallback_reason,
            sidecar_dir=load_result.sidecar_dir,
        )
    finally:
        del packed_by_indicator
        gc.collect()


def _pack_bool_matrix(mask: np.ndarray, *, word_count: int) -> np.ndarray:
    rows = int(mask.shape[0])
    byte_count = word_count * 8
    packed_bytes = np.zeros((rows, byte_count), dtype=np.uint8)
    source_bytes = np.packbits(
        np.ascontiguousarray(mask, dtype=np.bool_),
        axis=1,
        bitorder="little",
    )
    packed_bytes[:, : int(source_bytes.shape[1])] = source_bytes
    return np.ascontiguousarray(packed_bytes.view(np.uint64).reshape(rows, word_count))


def _load_prepared_pool_from_sidecar(
    sidecar_artifact_dir: Path,
    *,
    pool: PreparedIndicatorPool,
    time_slice_start: int,
    time_slice_stop: int | None,
) -> PackedSignalBitsets:
    sidecar = load_matrix_sidecar_artifacts(
        _indicator_sidecar_dir(sidecar_artifact_dir, str(pool.indicator_id))
    )
    row_ids = np.asarray(pool.row_ids, dtype=np.int64)
    if row_ids.ndim != 1 or int(row_ids.size) == 0:
        raise ValueError(f"row_ids for {pool.indicator_id!r} must be non-empty")
    source_rows, source_signal_length = _manifest_source_shape(sidecar.manifest)
    if int(row_ids.min()) < 0 or int(row_ids.max()) >= source_rows:
        raise ValueError(f"row_ids for {pool.indicator_id!r} exceed sidecar row count")
    start = int(time_slice_start)
    stop = source_signal_length if time_slice_stop is None else int(time_slice_stop)
    if start < 0 or stop <= start or stop > source_signal_length:
        raise ValueError(
            f"time slice [{start}, {stop}) is outside sidecar source length "
            f"{source_signal_length}"
        )
    if int(pool.trade_T.shape[1]) != stop - start:
        raise ValueError(
            f"prepared signal length for {pool.indicator_id!r} does not match sidecar slice"
        )
    if start == 0 and stop == source_signal_length:
        return PackedSignalBitsets(
            pos_bits=np.ascontiguousarray(np.asarray(sidecar.pos_bits[row_ids], dtype=np.uint64)),
            neg_bits=np.ascontiguousarray(np.asarray(sidecar.neg_bits[row_ids], dtype=np.uint64)),
            signal_length=source_signal_length,
            word_count=int(sidecar.manifest["word_count"]),
        )
    selected_full_rows = unpack_signal_bitsets(
        PackedSignalBitsets(
            pos_bits=np.ascontiguousarray(
                np.asarray(sidecar.pos_bits[row_ids], dtype=np.uint64)
            ),
            neg_bits=np.ascontiguousarray(
                np.asarray(sidecar.neg_bits[row_ids], dtype=np.uint64)
            ),
            signal_length=source_signal_length,
            word_count=int(sidecar.manifest["word_count"]),
        )
    )
    selected = np.ascontiguousarray(selected_full_rows[:, start:stop])
    reference = np.ascontiguousarray(np.asarray(pool.trade_T, dtype=np.int8))
    if not np.array_equal(selected, reference):
        raise ValueError(
            f"sidecar signal bits do not match prepared rows for {pool.indicator_id!r}"
        )
    return pack_signal_matrix(selected)


def _indicator_sidecar_dir(sidecar_artifact_dir: Path, indicator_id: str) -> Path:
    return sidecar_artifact_dir / _safe_indicator_path(indicator_id)


def _safe_indicator_path(indicator_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in indicator_id)
    return safe or "indicator"


def _unpack_word_matrix(words: np.ndarray, *, signal_length: int) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(words, dtype=np.uint64))
    rows = int(matrix.shape[0])
    byte_matrix = matrix.reshape(rows, -1).view(np.uint8)
    bits = np.unpackbits(byte_matrix, axis=1, bitorder="little")
    return np.ascontiguousarray(bits[:, :signal_length].astype(np.bool_, copy=False))


def _unpack_words(words: np.ndarray, *, signal_length: int) -> np.ndarray:
    row = np.ascontiguousarray(np.asarray(words, dtype=np.uint64)).reshape(1, -1)
    return _unpack_word_matrix(row, signal_length=signal_length)[0]


def _indicator_telemetry(
    *,
    pool: PreparedIndicatorPool,
    packed: PackedSignalBitsets,
) -> IndicatorBitsetPackTelemetry:
    trade_t = np.asarray(pool.trade_T, dtype=np.int8)
    positive = int(np.count_nonzero(trade_t == np.int8(1)))
    negative = int(np.count_nonzero(trade_t == np.int8(-1)))
    total = int(trade_t.size)
    padding_bits = (packed.word_count * BITS_PER_WORD) - packed.signal_length
    return IndicatorBitsetPackTelemetry(
        indicator_id=str(pool.indicator_id),
        rows_after_prefilter=int(trade_t.shape[0]),
        signal_length=packed.signal_length,
        word_count=packed.word_count,
        padding_bits=padding_bits,
        positive_signal_count=positive,
        negative_signal_count=negative,
        neutral_signal_count=total - positive - negative,
        packed_bytes=packed.packed_bytes,
        padding_valid=_padding_valid(packed),
    )


def _padding_valid(packed: PackedSignalBitsets) -> bool:
    padding_bits = (packed.word_count * BITS_PER_WORD) - packed.signal_length
    if padding_bits <= 0:
        return True
    valid_bits_in_last_word = packed.signal_length % BITS_PER_WORD
    if valid_bits_in_last_word == 0:
        return True
    valid_mask = np.uint64((1 << valid_bits_in_last_word) - 1)
    padding_mask = np.uint64((1 << BITS_PER_WORD) - 1) ^ valid_mask
    return bool(
        np.all((packed.pos_bits[:, -1] & padding_mask) == 0)
        and np.all((packed.neg_bits[:, -1] & padding_mask) == 0)
    )


def _validate_sidecar_manifest(
    manifest: dict[str, Any],
    *,
    expected_source_manifest_sha256: str | None,
    expected_source_signals_sha256: str | None,
    expected_identity: dict[str, Any] | None,
) -> None:
    if manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported matrix sidecar schema_version")
    if manifest.get("schema") != "backtest_matrix_sidecar_bitsets_v1":
        raise ValueError("unsupported matrix sidecar schema")
    if manifest.get("source_signal_dtype") != "int8":
        raise ValueError("matrix sidecar source_signal_dtype must be int8")
    if manifest.get("padding_policy") != SIDECAR_PADDING_POLICY:
        raise ValueError("matrix sidecar padding_policy mismatch")
    if manifest.get("bit_order") != SIDECAR_BIT_ORDER:
        raise ValueError("matrix sidecar bit_order mismatch")
    if (
        expected_source_manifest_sha256 is not None
        and manifest.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        raise ValueError("matrix sidecar source manifest hash mismatch")
    if (
        expected_source_signals_sha256 is not None
        and manifest.get("source_signals_sha256") != expected_source_signals_sha256
    ):
        raise ValueError("matrix sidecar source signals hash mismatch")
    if expected_identity is not None:
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("matrix sidecar identity must be an object")
        for key, expected in expected_identity.items():
            if identity.get(key) != expected:
                raise ValueError(f"matrix sidecar identity mismatch for {key!r}")
    _manifest_source_shape(manifest)


def _validate_manifest_source_files(manifest: dict[str, Any]) -> None:
    source_manifest_path = manifest.get("source_manifest_path")
    source_signals_path = manifest.get("source_signals_path")
    if not isinstance(source_manifest_path, str) or not source_manifest_path:
        raise ValueError("matrix sidecar source_manifest_path must be a non-empty string")
    if not isinstance(source_signals_path, str) or not source_signals_path:
        raise ValueError("matrix sidecar source_signals_path must be a non-empty string")
    source_manifest_sha256 = str(manifest.get("source_manifest_sha256", ""))
    source_signals_sha256 = str(manifest.get("source_signals_sha256", ""))
    if file_sha256_hex(Path(source_manifest_path)) != source_manifest_sha256:
        raise ValueError("matrix sidecar source manifest hash mismatch")
    if _signal_manifest_signals_sha256(Path(source_manifest_path)) != source_signals_sha256:
        raise ValueError("matrix sidecar source signals hash mismatch")


def _signal_manifest_signals_sha256(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix sidecar source manifest must be a YAML object")
    signals = payload.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("matrix sidecar source manifest signals must be an object")
    sha256 = signals.get("sha256")
    if not isinstance(sha256, str) or not sha256:
        raise ValueError("matrix sidecar source manifest signals.sha256 is required")
    return sha256


def _manifest_source_shape(manifest: dict[str, Any]) -> tuple[int, int]:
    shape = manifest.get("source_signal_shape")
    if (
        not isinstance(shape, list | tuple)
        or len(shape) != 2
        or int(shape[0]) <= 0
        or int(shape[1]) <= 0
    ):
        raise ValueError("matrix sidecar source_signal_shape must be [rows, T]")
    return int(shape[0]), int(shape[1])


def _validate_array(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> None:
    if np.dtype(array.dtype) != dtype:
        raise ValueError(f"{name} dtype mismatch; got {array.dtype}, expected {dtype}")
    if shape is not None and tuple(int(value) for value in array.shape) != shape:
        raise ValueError(f"{name} shape mismatch; got {array.shape}, expected {shape}")
    if ndim is not None and int(array.ndim) != ndim:
        raise ValueError(f"{name} ndim mismatch; got {array.ndim}, expected {ndim}")


def _duplicate_maps(row_hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_by_hash: dict[int, int] = {}
    unique_ids: list[int] = []
    duplicate_ids: list[int] = []
    duplicate_unique_ids: list[int] = []
    for row_id, value in enumerate(np.asarray(row_hashes, dtype=np.uint64)):
        key = int(value)
        first = first_by_hash.get(key)
        if first is None:
            first_by_hash[key] = row_id
            unique_ids.append(row_id)
        else:
            duplicate_ids.append(row_id)
            duplicate_unique_ids.append(first)
    return (
        np.asarray(unique_ids, dtype=np.uint32),
        np.asarray(duplicate_ids, dtype=np.uint32),
        np.asarray(duplicate_unique_ids, dtype=np.uint32),
    )


def _validate_duplicate_maps(artifacts: MatrixSidecarArtifacts, *, rows: int) -> None:
    unique = np.asarray(artifacts.unique_signal_row_ids, dtype=np.uint32)
    duplicate = np.asarray(artifacts.duplicate_signal_row_ids, dtype=np.uint32)
    duplicate_unique = np.asarray(artifacts.duplicate_unique_signal_row_ids, dtype=np.uint32)
    if int(unique.size) == 0:
        raise ValueError("matrix sidecar unique_signal_row_ids must not be empty")
    if bool(np.any(unique >= rows)) or bool(np.any(duplicate >= rows)):
        raise ValueError("matrix sidecar row id map contains row ids outside source shape")
    if bool(np.any(duplicate_unique >= rows)):
        raise ValueError("matrix sidecar duplicate map contains unique ids outside source shape")
    covered = sorted(int(value) for value in np.concatenate([unique, duplicate]))
    if covered != list(range(rows)):
        raise ValueError("matrix sidecar unique/duplicate maps must cover every source row once")
    row_hashes = np.asarray(artifacts.signal_row_hashes, dtype=np.uint64)
    for duplicate_id, unique_id in zip(duplicate, duplicate_unique, strict=True):
        if row_hashes[int(duplicate_id)] != row_hashes[int(unique_id)]:
            raise ValueError("matrix sidecar duplicate map hash mismatch")


def file_sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _perf_counter() -> float:
    import time

    return time.perf_counter()


def _validate_aligned_word_counts(
    indicators: Sequence[IndicatorBitsetPackTelemetry],
) -> None:
    if not indicators:
        return
    signal_length = indicators[0].signal_length
    word_count = indicators[0].word_count
    for indicator in indicators:
        if indicator.signal_length != signal_length or indicator.word_count != word_count:
            raise ValueError("all indicator pools must share the same T and W")


def _consensus_sample_mismatches(
    *,
    indicator_pools: Sequence[PreparedIndicatorPool],
    packed_by_indicator: Sequence[PackedSignalBitsets],
    sample_limit: int,
) -> tuple[int, int]:
    samples = _sample_row_positions(
        row_counts=tuple(int(pool.trade_T.shape[0]) for pool in indicator_pools),
        sample_limit=sample_limit,
    )
    mismatches = 0
    for row_positions in samples:
        reference = _reference_consensus_row(
            tuple(
                np.asarray(pool.trade_T, dtype=np.int8)[row_position]
                for pool, row_position in zip(indicator_pools, row_positions, strict=True)
            )
        )
        bitset = bitset_consensus_row(packed_by_indicator, row_positions)
        if not np.array_equal(reference, bitset):
            mismatches += 1
    return len(samples), mismatches


def _sample_row_positions(
    *,
    row_counts: Sequence[int],
    sample_limit: int,
) -> tuple[tuple[int, ...], ...]:
    if not row_counts:
        return ()
    if any(count <= 0 for count in row_counts):
        raise ValueError("row counts must be positive")
    samples: list[tuple[int, ...]] = []

    def append_once(sample: tuple[int, ...]) -> None:
        if sample not in samples and len(samples) < sample_limit:
            samples.append(sample)

    append_once(tuple(0 for _ in row_counts))
    append_once(tuple(count - 1 for count in row_counts))
    for offset in range(sample_limit):
        append_once(tuple((offset + pos) % count for pos, count in enumerate(row_counts)))
        if len(samples) >= sample_limit:
            break
    return tuple(samples)


def _reference_consensus_row(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.int8)
    stacked = np.vstack([np.asarray(row, dtype=np.int8) for row in rows])
    positive = np.all(stacked == np.int8(1), axis=0)
    negative = np.all(stacked == np.int8(-1), axis=0)
    consensus = np.zeros(stacked.shape[1], dtype=np.int8)
    consensus[positive] = np.int8(1)
    consensus[negative] = np.int8(-1)
    return np.ascontiguousarray(consensus)


__all__ = [
    "BITS_PER_WORD",
    "IndicatorBitsetPackTelemetry",
    "MatrixSidecarArtifacts",
    "PackedSignalBitsets",
    "RuntimeBitsetPackTelemetry",
    "SIDECAR_DUPLICATE_ROW_IDS_FILENAME",
    "SIDECAR_DUPLICATE_UNIQUE_ROW_IDS_FILENAME",
    "SIDECAR_MANIFEST_FILENAME",
    "SIDECAR_NEG_BITS_FILENAME",
    "SIDECAR_POS_BITS_FILENAME",
    "SIDECAR_ROW_HASHES_FILENAME",
    "SIDECAR_UNIQUE_ROW_IDS_FILENAME",
    "SidecarBitsetLoadResult",
    "bitset_consensus_row",
    "build_matrix_sidecar_artifacts",
    "build_runtime_bitset_pack_telemetry",
    "file_sha256_hex",
    "load_matrix_sidecar_artifacts",
    "load_or_pack_signal_bitsets",
    "pack_signal_matrix",
    "signal_row_hashes_u64",
    "unpack_signal_bitsets",
    "validate_matrix_sidecar_artifacts",
    "word_count_for_signal_length",
]
