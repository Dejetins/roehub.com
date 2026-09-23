"""Bounded exact local admission; financial buffers are never changed.

For a total primary order, each excluded eligible row has K predecessors in
its own prefix and cannot mutate global top-K. Keep EVERY prefix entrant,
including subsequently evicted rows, so the unchanged global heap performs
exactly its original mutation sequence. Retained indices may exceed K.
NaNs use the original admission instead.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=False)
def _less(a, b, scores, rows, multiplier):
    left = float(scores[a]) * multiplier
    right = float(scores[b]) * multiplier
    if left != right:
        return left < right
    for j in range(rows.shape[0]):
        if rows[j, a] != rows[j, b]:
            return rows[j, a] < rows[j, b]
    return False  # Primary-key equality never admits a later row.


@nb.njit(cache=True, fastmath=False)
def select_indices(scores, trades, rows, multiplier, minimum, k):
    """Return every local prefix top-K entrant in input order, or None for NaN."""
    n = len(scores)
    chosen = np.empty(min(k, n), dtype=np.int64)
    admitted = np.empty(n, dtype=np.int64)
    count = 0
    size = 0
    for i in range(n):
        # Check before eligibility: preserve even an ineligible NaN distinction.
        if np.isnan(float(scores[i])):
            return None
        if int(trades[i]) < minimum:
            continue
        if size < k:
            admitted[count] = i
            count += 1
            pos = size
            size += 1
            while pos > 0:
                parent = (pos - 1) // 2
                if not _less(i, chosen[parent], scores, rows, multiplier):
                    break
                chosen[pos] = chosen[parent]
                pos = parent
            chosen[pos] = i
        elif _less(chosen[0], i, scores, rows, multiplier):
            admitted[count] = i
            count += 1
            pos = 0
            while 2 * pos + 1 < size:
                child = 2 * pos + 1
                if child + 1 < size and _less(
                    chosen[child + 1], chosen[child], scores, rows, multiplier
                ):
                    child += 1
                if not _less(chosen[child], i, scores, rows, multiplier):
                    break
                chosen[pos] = chosen[child]
                pos = child
            chosen[pos] = i
    return admitted[:count]


def local_admission_indices(
    *, scores, trades, selected, row_ids, multiplier, minimum, k, max_bytes
):
    """Allocate only bounded integer identity scratch; unsupported shapes fall back."""
    n = len(scores)
    if k < 1 or n <= k or multiplier not in (-1.0, 1.0) or not -(2**63) <= minimum < 2**63:
        return None
    if scores.ndim != 1 or scores.dtype not in (
        np.dtype("float64"),
        np.dtype("int64"),
        np.dtype("int32"),
    ):
        return None
    if trades.shape != (n,) or trades.dtype.kind not in "iu":
        return None
    if not selected or len(selected) != len(row_ids):
        return None
    if (len(selected) * n + 2 * n + min(k, n)) * 8 > max_bytes:
        return None
    for indices, ids in zip(selected, row_ids, strict=True):
        if (
            indices.shape != (n,)
            or indices.dtype.kind not in "iu"
            or ids.ndim != 1
            or ids.dtype.kind != "i"
        ):
            return None
        if len(indices) and (indices.min() < 0 or indices.max() >= len(ids)):
            return None
    try:
        rows = np.empty((len(selected), n), dtype=np.int64)
        for j, (indices, ids) in enumerate(zip(selected, row_ids, strict=True)):
            rows[j] = ids[indices]
        return select_indices(scores, trades, rows, multiplier, minimum, k)
    except MemoryError:
        return None
