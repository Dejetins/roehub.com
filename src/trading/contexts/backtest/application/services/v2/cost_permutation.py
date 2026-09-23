"""Bounded job-owned scheduling around unchanged independent-row native kernels."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any

import numba as nb
import numpy as np
from numba.np.ufunc.parallel import get_thread_id

from .compute_policy import BacktestComputePolicy
from .job_scratch import BacktestJobScratch

_KERNELS = frozenset(
    {
        "event_segments_2_no_risk",
        "streaming_2_no_risk",
        "event_segments_n_no_risk",
        "matrix_bitset_no_risk",
        "matrix_bitset_no_risk_long_only",
        "event_segments_n_tp_sl_15m_grid",
        "event_segments_n_tp_sl_15m_grid_cell_blocks",
        "event_segments_n_tp_sl_15m_grid_execution_sizing",
    }
)


@dataclass(slots=True)
class CostPermutationState:
    # Strong input references: identity never outlives its owner. Bounded by byte budget.
    costs: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    verified_shapes: set[tuple[int, int]] = field(default_factory=set)
    telemetry: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def cache_bytes(self) -> int:
        return sum(values.nbytes for _, values in self.costs)

    def record(self, kernel: str, reason: str, **values: float) -> None:
        row = self.telemetry.setdefault(kernel + ":" + reason, {"calls": 0.0})
        row["calls"] += 1
        for key, value in values.items():
            row[key] = row.get(key, 0.0) + value


def state_for(scratch: BacktestJobScratch) -> CostPermutationState:
    state = scratch.get("cost_permutation")
    if state is None:
        state = CostPermutationState()
        scratch.retain("cost_permutation", state)
    if not isinstance(state, CostPermutationState):
        raise TypeError("invalid job cost state")
    return state


@nb.njit(cache=True, parallel=True, fastmath=False)
def native_owners(n: int) -> np.ndarray:
    result = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        result[i] = get_thread_id()
    return result


@nb.njit(cache=True, fastmath=False)
def balanced_order(cost: np.ndarray, p: int) -> np.ndarray:
    n = len(cost)
    order = np.argsort(cost)[::-1]
    capacity = np.empty(p, dtype=np.int64)
    start = np.empty(p, dtype=np.int64)
    fill = np.zeros(p, dtype=np.int64)
    load = np.zeros(p, dtype=np.int64)
    result = np.empty(n, dtype=np.int64)
    offset = 0
    for j in range(p):
        capacity[j] = n // p + (1 if j < n % p else 0)
        start[j] = offset
        offset += capacity[j]
    for i in order:
        best = -1
        for j in range(p):
            if fill[j] < capacity[j] and (best < 0 or load[j] < load[best]):
                best = j
        result[start[best] + fill[best]] = i
        fill[best] += 1
        load[best] += cost[i]
    return result


@nb.njit(cache=True, fastmath=False)
def row_activity(array: np.ndarray, packed: bool) -> np.ndarray:
    result = np.zeros(array.shape[0], dtype=np.int64)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if packed:
                value = np.uint64(array[row, col])
                while value != 0:
                    result[row] += 1
                    value &= value - np.uint64(1)
            elif array[row, col] != 0:
                result[row] += 1
    return result


def _activity(array: np.ndarray, state: CostPermutationState, packed: bool) -> np.ndarray:
    for owner, counts in state.costs:
        if owner is array:
            return counts
    counts = row_activity(array, packed)
    state.costs.append((array, counts))
    return counts


def _cost(bound: dict[str, Any], n: int, state: CostPermutationState) -> np.ndarray:
    result = np.ones(n, dtype=np.int64)
    combo = bound.get("combo_idx_by_indicator")
    if combo is not None:
        counts = bound.get("segment_counts", bound.get("counts"))
        for j in range(combo.shape[0]):
            if counts is not None:
                result += counts[j, combo[j]]
            else:
                for prefix in ("pos_bits_", "neg_bits_"):
                    bits = bound.get(prefix + str(j))
                    if bits is not None and bits.size:
                        result += _activity(bits, state, True)[combo[j]]
    else:
        for side in ("left", "right"):
            rows = bound["combo_" + side + "_idx"]
            counts = bound.get(side + "_segment_counts")
            if counts is None:
                counts = _activity(bound[side + "_trade_t"], state, False)
            result += counts[rows]
    return result


def score_with_cost_permutation(
    kernel: Any,
    policy: BacktestComputePolicy,
    scratch: BacktestJobScratch | None,
    *args: Any,
) -> None:
    """Fallback only before scoring; a native failure never publishes temporary output."""
    if scratch is None:
        kernel(*args)
        return
    state = state_for(scratch)
    name = getattr(getattr(kernel, "py_func", None), "__name__", "unsupported")
    reason = "unsupported_kernel"
    bound: dict[str, Any] = {}
    n = 0
    extra_bytes = 0
    p = nb.get_num_threads()
    if p != policy.threads.num_threads:
        raise RuntimeError("cost permutation effective thread budget mismatch")
    if name in _KERNELS:
        bound = dict(zip(inspect.signature(kernel.py_func).parameters, args, strict=True))
        combo = bound.get("combo_idx_by_indicator")
        inputs = (
            [combo] if combo is not None else [bound["combo_left_idx"], bound["combo_right_idx"]]
        )
        outputs = [value for key, value in bound.items() if key.startswith("out_")]
        n = int(inputs[0].shape[-1]) if inputs[0].ndim else 0
        reason = "unsupported_shape"
        if (
            all(
                isinstance(a, np.ndarray)
                and a.dtype == np.int32
                and a.ndim == (2 if combo is not None else 1)
                and a.shape[-1] == n
                and a.flags.c_contiguous
                for a in inputs
            )
            and all(
                isinstance(a, np.ndarray)
                and a.ndim == 1
                and a.size in (0, n)
                and a.flags.c_contiguous
                and a.flags.writeable
                for a in outputs
            )
            and (combo is None or 1 <= combo.shape[0] <= 7)
        ):
            reason = "insufficient_work"
            if n >= policy.cost_permutation_min_rows and n >= p and p > 1:
                reason = "unsupported_scheduler"
                if (
                    getattr(nb, "threading_layer")() == "workqueue"
                    and getattr(nb, "get_parallel_chunksize")() == 0
                    and p <= 256
                ):
                    # Includes sort/cost/permutation/probe/index temporaries and native workspace.
                    extra_bytes = (
                        n * 64
                        + p * 48
                        + sum(a.nbytes for a in inputs + outputs)
                        + getattr(bound.get("segment_pos_workspace"), "nbytes", 0)
                    )
                    activity_arrays = [
                        v
                        for k, v in bound.items()
                        if k.startswith(("pos_bits_", "neg_bits_"))
                        or k in ("left_trade_t", "right_trade_t")
                    ]
                    extra_bytes += sum(a.shape[0] * 8 for a in activity_arrays)
                    reason = "memory_budget"
                    if n <= 2_000_000 and extra_bytes + state.cache_bytes <= (
                        policy.cost_permutation_max_bytes
                    ):
                        reason = "enabled"
    if reason != "enabled":
        state.record(name, reason)
        kernel(*args)
        return
    before = time.perf_counter()
    shape = (n, p)
    try:
        matches = True
        if shape not in state.verified_shapes:
            actual = native_owners(n)
            offset = 0
            matches = True
            for j in range(p):
                size = n // p + (1 if j < n % p else 0)
                matches = matches and bool(np.all(actual[offset : offset + size] == j))
                offset += size
            if matches and len(state.verified_shapes) < 64:
                state.verified_shapes.add(shape)
        cost = _cost(bound, n, state)
        perm = balanced_order(cost, p)
        estimated = time.perf_counter()
        copied = []
        outputs = []
        for key, value in bound.items():
            if key == "combo_idx_by_indicator":
                copied.append(np.ascontiguousarray(value[:, perm]))
            elif key in ("combo_left_idx", "combo_right_idx"):
                copied.append(np.ascontiguousarray(value[perm]))
            elif key == "segment_pos_workspace":
                copied.append(np.empty_like(value))
            elif key.startswith("out_") and value.size:
                temporary = np.empty_like(value)
                outputs.append((value, temporary))
                copied.append(temporary)
            else:
                copied.append(value)
    except MemoryError:
        state.costs.clear()
        state.record(name, "allocation_failure")
        kernel(*args)
        return
    if not matches:
        state.record(name, "schedule_mismatch")
        kernel(*args)
        return
    prepared = time.perf_counter()
    try:
        kernel(*copied)
    except BaseException:
        # No caller output was passed to the kernel; discard all partial buffers.
        state.costs.clear()
        state.record(name, "kernel_failure")
        raise
    computed = time.perf_counter()
    for destination, temporary in outputs:
        destination[perm] = temporary
    state.record(
        name,
        "enabled",
        estimate_s=estimated - before,
        copy_s=prepared - estimated,
        scoring_s=computed - prepared,
        restore_s=time.perf_counter() - computed,
        copied_bytes=float(
            sum(
                x.nbytes
                for x in copied
                if isinstance(x, np.ndarray) and not any(x is a for a in args)
            )
        ),
        bounded_extra_bytes=float(extra_bytes),
        rows=float(n),
    )
