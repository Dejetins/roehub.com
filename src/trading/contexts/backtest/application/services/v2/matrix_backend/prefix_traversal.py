from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numba as nb
import numpy as np

from trading.contexts.backtest.application.services.v2.matrix_backend.bitsets import (
    BITS_PER_WORD,
    PackedSignalBitsets,
)

ALL_BITS = np.uint64(18446744073709551615)
MAX_PREFIX_MATERIALIZED_CANDIDATES = 2_000_000


@dataclass(frozen=True, slots=True)
class CompiledPrefixTraversalResult:
    rows_by_indicator: Mapping[str, np.ndarray]
    telemetry: Mapping[str, Any]

    @property
    def candidate_count(self) -> int:
        if not self.rows_by_indicator:
            return 0
        first = next(iter(self.rows_by_indicator.values()))
        return int(first.shape[0])


def collect_compiled_prefix_candidates(
    *,
    indicator_ids: Sequence[str],
    packed_by_indicator: Sequence[PackedSignalBitsets],
    min_closed_trades: int,
    direction_mode: str,
    max_materialized_candidates: int = MAX_PREFIX_MATERIALIZED_CANDIDATES,
) -> CompiledPrefixTraversalResult:
    ids = tuple(str(indicator_id) for indicator_id in indicator_ids)
    arity = len(ids)
    if arity not in (6, 7):
        raise ValueError("compiled prefix traversal supports arity 6 and 7 only")
    if len(packed_by_indicator) != arity:
        raise ValueError("packed bitset arity mismatch")
    if direction_mode not in {"long_only", "long_short_reversal"}:
        raise ValueError(f"unsupported direction_mode={direction_mode!r}")

    pos_stack, neg_stack, row_counts, signal_length, word_count = _bitset_stacks(
        packed_by_indicator
    )
    total_count = int(np.prod(row_counts, dtype=np.int64))
    if total_count > max_materialized_candidates:
        raise ValueError(
            "compiled prefix traversal materialization limit exceeded: "
            f"{total_count} > {max_materialized_candidates}"
        )

    selectivity_order = _selectivity_order(
        pos_stack=pos_stack,
        neg_stack=neg_stack,
        row_counts=row_counts,
        direction_mode=direction_mode,
    )
    out_rows_by_pos = np.empty((arity, total_count), dtype=np.int32)
    out_ordinals = np.empty(total_count, dtype=np.int64)
    counters = np.zeros(4, dtype=np.int64)
    started = time.perf_counter()
    selected_count = _collect_prefix_candidates(
        pos_stack,
        neg_stack,
        row_counts,
        selectivity_order,
        np.int32(max(int(min_closed_trades), 1)),
        np.int8(1 if direction_mode == "long_only" else 2),
        np.int32(signal_length),
        np.int32(word_count),
        _last_word_mask(signal_length),
        out_rows_by_pos,
        out_ordinals,
        counters,
    )
    elapsed_s = time.perf_counter() - started
    selected = int(selected_count)
    rows = out_rows_by_pos[:, :selected]
    ordinals = out_ordinals[:selected]
    if selected > 1:
        canonical_order = np.argsort(ordinals, kind="stable")
        rows = np.ascontiguousarray(rows[:, canonical_order])
        ordinals = np.ascontiguousarray(ordinals[canonical_order])
    else:
        rows = np.ascontiguousarray(rows)
        ordinals = np.ascontiguousarray(ordinals)

    rows_by_indicator = {
        indicator_id: np.ascontiguousarray(rows[pos], dtype=np.int32)
        for pos, indicator_id in enumerate(ids)
    }
    traversal_candidates_per_sec = (
        None if elapsed_s <= 0.0 else float(total_count) / elapsed_s
    )
    telemetry = {
        "schema": "backtest_compiled_prefix_product_traversal_v1",
        "backend_id": "compiled_prefix_product_traversal_v1",
        "combo_iteration_mode": "compiled_prefix_product_traversal",
        "combo_count_planned": total_count,
        "prefix_candidates_selected": selected,
        "prefix_candidates_pruned": total_count - selected,
        "prefix_nodes_visited": int(counters[0]),
        "prefix_nodes_reused": int(counters[1]),
        "prefix_pruned_subtrees": int(counters[2]),
        "prefix_pruned_candidate_upper_bound": int(counters[3]),
        "prefix_min_closed_trades": int(max(int(min_closed_trades), 1)),
        "selectivity_order": [int(item) for item in selectivity_order.tolist()],
        "signal_length": int(signal_length),
        "word_count": int(word_count),
        "compiled_loop_elapsed_s": elapsed_s,
        "combo_iteration_candidates_per_sec": traversal_candidates_per_sec,
        "first_canonical_ordinal": None if selected == 0 else int(ordinals[0]),
        "last_canonical_ordinal": None if selected == 0 else int(ordinals[-1]),
    }
    return CompiledPrefixTraversalResult(
        rows_by_indicator=rows_by_indicator,
        telemetry=telemetry,
    )


def _bitset_stacks(
    packed_by_indicator: Sequence[PackedSignalBitsets],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    signal_lengths = {int(packed.signal_length) for packed in packed_by_indicator}
    word_counts = {int(packed.word_count) for packed in packed_by_indicator}
    if len(signal_lengths) != 1 or len(word_counts) != 1:
        raise ValueError("compiled prefix traversal requires aligned bitset shapes")
    signal_length = signal_lengths.pop()
    word_count = word_counts.pop()
    arity = len(packed_by_indicator)
    row_counts = np.asarray(
        [int(packed.pos_bits.shape[0]) for packed in packed_by_indicator],
        dtype=np.int32,
    )
    max_rows = int(np.max(row_counts))
    pos_stack = np.zeros((arity, max_rows, word_count), dtype=np.uint64)
    neg_stack = np.zeros((arity, max_rows, word_count), dtype=np.uint64)
    for pos, packed in enumerate(packed_by_indicator):
        rows = int(row_counts[pos])
        pos_stack[pos, :rows, :] = np.ascontiguousarray(packed.pos_bits, dtype=np.uint64)
        neg_stack[pos, :rows, :] = np.ascontiguousarray(packed.neg_bits, dtype=np.uint64)
    return (
        np.ascontiguousarray(pos_stack),
        np.ascontiguousarray(neg_stack),
        row_counts,
        signal_length,
        word_count,
    )


def _selectivity_order(
    *,
    pos_stack: np.ndarray,
    neg_stack: np.ndarray,
    row_counts: np.ndarray,
    direction_mode: str,
) -> np.ndarray:
    active_counts: list[tuple[int, int]] = []
    for pos in range(int(row_counts.shape[0])):
        rows = int(row_counts[pos])
        active = pos_stack[pos, :rows, :]
        if direction_mode != "long_only":
            active = active | neg_stack[pos, :rows, :]
        byte_view = np.ascontiguousarray(active).view(np.uint8)
        active_counts.append((int(np.sum(np.unpackbits(byte_view))), pos))
    return np.asarray([pos for _, pos in sorted(active_counts)], dtype=np.int32)


def _last_word_mask(signal_length: int) -> np.uint64:
    remainder = int(signal_length) % BITS_PER_WORD
    if remainder == 0:
        return ALL_BITS
    return np.uint64((1 << remainder) - 1)


@nb.njit(cache=True)
def _collect_prefix_candidates(
    pos_stack: np.ndarray,
    neg_stack: np.ndarray,
    row_counts: np.ndarray,
    selectivity_order: np.ndarray,
    min_closed_trades: np.int32,
    direction_mode: np.int8,
    signal_length: np.int32,
    word_count: np.int32,
    last_word_mask: np.uint64,
    out_rows_by_pos: np.ndarray,
    out_ordinals: np.ndarray,
    counters: np.ndarray,
) -> np.int64:
    arity = int(row_counts.shape[0])
    prefix_pos = np.empty((arity, int(word_count)), dtype=np.uint64)
    prefix_neg = np.empty((arity, int(word_count)), dtype=np.uint64)
    selected_rows = np.zeros(arity, dtype=np.int32)
    row_choice_by_level = np.zeros(arity, dtype=np.int32)

    level = 0
    out_count = np.int64(0)
    while level >= 0:
        original_pos = int(selectivity_order[level])
        if row_choice_by_level[level] >= row_counts[original_pos]:
            row_choice_by_level[level] = np.int32(0)
            level -= 1
            if level >= 0:
                row_choice_by_level[level] += np.int32(1)
            continue

        row_idx = row_choice_by_level[level]
        selected_rows[original_pos] = row_idx
        counters[0] += np.int64(1)
        if level > 0:
            counters[1] += np.int64(1)

        active_count = np.int32(0)
        for word_idx in range(int(word_count)):
            word_mask = ALL_BITS
            if word_idx == int(word_count) - 1:
                word_mask = last_word_mask
            pos_bits = pos_stack[original_pos, row_idx, word_idx] & word_mask
            neg_bits = np.uint64(0)
            if direction_mode != np.int8(1):
                neg_bits = neg_stack[original_pos, row_idx, word_idx] & word_mask
            if level > 0:
                pos_bits &= prefix_pos[level - 1, word_idx]
                if direction_mode != np.int8(1):
                    neg_bits &= prefix_neg[level - 1, word_idx]
            prefix_pos[level, word_idx] = pos_bits
            prefix_neg[level, word_idx] = neg_bits
            active_count += np.int32(_popcount_u64(pos_bits))
            if direction_mode != np.int8(1):
                active_count += np.int32(_popcount_u64(neg_bits))

        if active_count < min_closed_trades:
            counters[2] += np.int64(1)
            counters[3] += _remaining_product(row_counts, selectivity_order, level + 1)
            row_choice_by_level[level] += np.int32(1)
            continue

        if level == arity - 1:
            for pos in range(arity):
                out_rows_by_pos[pos, out_count] = selected_rows[pos]
            out_ordinals[out_count] = _canonical_ordinal(selected_rows, row_counts)
            out_count += np.int64(1)
            row_choice_by_level[level] += np.int32(1)
        else:
            level += 1
            row_choice_by_level[level] = np.int32(0)

    _ = signal_length
    return out_count


@nb.njit(cache=True, inline="always")
def _remaining_product(
    row_counts: np.ndarray,
    selectivity_order: np.ndarray,
    start_level: int,
) -> np.int64:
    total = np.int64(1)
    for level in range(start_level, int(selectivity_order.shape[0])):
        total *= np.int64(row_counts[int(selectivity_order[level])])
    return total


@nb.njit(cache=True, inline="always")
def _canonical_ordinal(selected_rows: np.ndarray, row_counts: np.ndarray) -> np.int64:
    ordinal = np.int64(0)
    for pos in range(int(row_counts.shape[0])):
        ordinal *= np.int64(row_counts[pos])
        ordinal += np.int64(selected_rows[pos])
    return ordinal


@nb.njit(cache=True, inline="always")
def _popcount_u64(value: np.uint64) -> int:
    count = 0
    while value != np.uint64(0):
        value &= value - np.uint64(1)
        count += 1
    return count


__all__ = [
    "CompiledPrefixTraversalResult",
    "MAX_PREFIX_MATERIALIZED_CANDIDATES",
    "collect_compiled_prefix_candidates",
]
