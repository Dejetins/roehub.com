from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from trading.contexts.backtest.application.dto import PreparedIndicatorPool

BITS_PER_WORD = 64


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

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": "backtest_runtime_bitset_pack_shadow_v1",
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
        packed_by_indicator = tuple(
            pack_signal_matrix(pool.trade_T) for pool in indicator_pools
        )
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
    "PackedSignalBitsets",
    "RuntimeBitsetPackTelemetry",
    "bitset_consensus_row",
    "build_runtime_bitset_pack_telemetry",
    "pack_signal_matrix",
    "unpack_signal_bitsets",
    "word_count_for_signal_length",
]
