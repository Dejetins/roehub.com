"""Independent prefix proof and exact unchanged-heap comparison, including payload ties."""

from itertools import product
from typing import Any

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.raw_exact import encode
from trading.contexts.backtest.application.services.v2 import local_top_k as local
from trading.contexts.backtest.application.services.v2 import no_risk_exact as nr
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy


def oracle(scores, trades, rows, multiplier, minimum, k):
    admitted = []
    predecessors = []
    for i, score in enumerate(scores):
        if trades[i] < minimum:
            continue
        key = (float(score) * multiplier, tuple(int(x) for x in rows[:, i]))
        if len(predecessors) < k or key > sorted(predecessors, reverse=True)[k - 1]:
            admitted.append(i)
        else:
            assert sum(previous >= key for previous in predecessors) >= k
        predecessors.append(key)
    return admitted


@pytest.mark.parametrize("arity", range(1, 8))
@pytest.mark.parametrize("k", [1, 50, 500])
@pytest.mark.parametrize("multiplier", [-1.0, 1.0])
@pytest.mark.parametrize("minimum", [0, 1, 3])
def test_compiled_prefix_proof(arity, k, multiplier, minimum):
    rng = np.random.default_rng(1703)
    scores = rng.integers(-4, 5, 101).astype(float)
    scores[:4] = [-np.inf, np.inf, -0.0, 0.0]
    trades = rng.integers(0, 5, len(scores))
    rows = rng.integers(-10, 11, (arity, len(scores)))
    result = local.select_indices(scores, trades, rows, multiplier, minimum, k)
    assert result is not None
    assert result.tolist() == oracle(scores, trades, rows, multiplier, minimum, k)


def batch(scores, arity=2, duplicates=False) -> dict[str, Any]:
    n = len(scores)
    ids = tuple(f"i{j}" for j in range(arity))
    selected = {name: np.arange(n, dtype=np.int32) for name in ids}
    rows = tuple(np.arange(n, dtype=np.int64) % (3 if duplicates else n + 1) for _ in ids)
    context = nr._TopKContext(ids, rows, tuple(tuple(range(n)) for _ in ids))
    buffers = nr._allocate_metric_buffers(n, full_metrics=True)
    for index, metric in enumerate(nr.NO_RISK_METRIC_NAMES):
        getattr(buffers, metric)[:] = np.arange(n) + index
    buffers.total_return_pct[:] = scores
    buffers.trade_count[:] = np.arange(n) % 4
    return dict(
        top_k_context=context,
        selected_rows_by_indicator=selected,
        buffers=buffers,
        confirm=np.arange(n),
        proxy=np.arange(n, dtype=float),
        min_closed_trades=1,
    )


@pytest.mark.parametrize("metric", nr.NO_RISK_METRIC_NAMES)
@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("arity", [1, 2, 3, 7])
@pytest.mark.parametrize("k", [1, 50, 500])
def test_all_metrics_unchanged_heap(metric, direction, arity, k):
    data = batch(np.cos(np.arange(121)), arity)
    heaps = []
    for enabled in (False, True):
        heap = []
        nr._update_heap_generic_ranking(
            heap=heap,
            **data,
            top_k=k,
            ranking=nr._RankingSpec(metric, direction),
            compute_policy=BacktestComputePolicy(
                local_top_k_max_bytes=64 * 1024 * 1024 if enabled else 0
            ),
        )
        heaps.append(encode(heap))
    assert heaps[0] == heaps[1]  # Includes topology, raw metrics, alignment and payload.


@pytest.mark.parametrize("nan_chunk", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 5, 50])
@pytest.mark.parametrize("duplicate", [False, True])
def test_cross_chunk_nan_duplicate_payload_exact(nan_chunk, k, duplicate):
    chunks = []
    for c in range(3):
        values = np.array([0.0, -0.0, np.inf, -np.inf, 3.0, 3.0, 1.0, 7.0, 8.0, 1.0, 2.0])
        if c == nan_chunk:
            values.view(np.uint64)[1] = 0xFFF8000000000007
        data = batch(values, duplicates=duplicate)
        data["buffers"].trade_count[:] = 2
        data["proxy"][3] = np.nan
        data["buffers"].sharpe_trades[6] = np.nan
        chunks.append(data)
    heaps = []
    for enabled in (False, True):
        heap = []
        for data in chunks:
            nr._update_heap_total_return_desc(
                heap=heap,
                **data,
                top_k=k,
                compute_policy=BacktestComputePolicy(
                    local_top_k_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
            )
        heaps.append(encode(heap))
    assert heaps[0] == heaps[1]


def test_exhaustive_ties_prefix_order():
    for values in product([-1.0, 0.0, 1.0], repeat=5):
        scores = np.array(values)
        rows = np.array([[0, 0, 1, 1, 0]], dtype=np.int64)
        trades = np.ones(5, dtype=np.int64)
        for k in (1, 2, 4):
            got = local.select_indices(scores, trades, rows, 1.0, 1, k)
            assert got is not None
            assert got.tolist() == oracle(scores, trades, rows, 1.0, 1, k)


def test_empty_nan_and_budget_fallback():
    empty = local.select_indices(
        np.array([]), np.array([], dtype=np.int64), np.empty((1, 0), dtype=np.int64), 1.0, 1, 1
    )
    assert empty is not None and empty.size == 0
    data = batch([1.0, np.nan, 2.0])
    args = dict(
        scores=data["buffers"].total_return_pct,
        trades=data["buffers"].trade_count,
        selected=tuple(data["selected_rows_by_indicator"].values()),
        row_ids=data["top_k_context"].row_ids_by_pos,
        multiplier=1.0,
        minimum=1,
        k=1,
        max_bytes=10000,
    )
    assert local.local_admission_indices(**args) is None
    args["scores"] = np.array([1.0, 2.0, 3.0])
    args["max_bytes"] = 0
    assert local.local_admission_indices(**args) is None


def test_allocation_failure_and_native_exception(monkeypatch):
    data = batch(np.arange(10.0))
    baseline = []
    nr._update_heap_total_return_desc(heap=baseline, **data, top_k=2)

    def fail(*args, **kwargs):
        raise MemoryError("forced")

    monkeypatch.setattr(local.np, "empty", fail)
    result = []
    nr._update_heap_total_return_desc(
        heap=result, **data, top_k=2, compute_policy=BacktestComputePolicy()
    )
    assert encode(result) == encode(baseline)
    monkeypatch.undo()

    def failed_native(*args):
        raise RuntimeError("forced native")

    monkeypatch.setattr(local, "select_indices", failed_native)
    result = []
    with pytest.raises(RuntimeError, match="forced native"):
        nr._update_heap_total_return_desc(
            heap=result, **data, top_k=2, compute_policy=BacktestComputePolicy()
        )
    assert not result


def test_default_resource_bounds():
    policy = BacktestComputePolicy()
    assert policy.threads.num_threads == 12
    assert policy.local_top_k_max_bytes == 64 * 1024 * 1024


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("initial", [0, 2, 5])
def test_randomized_preexisting_heap_and_missing_alignment(pending, initial):
    rng = np.random.default_rng(3917)
    for attempt in range(20):
        chunks = [
            batch(rng.integers(-2, 3, n).astype(float), duplicates=True) for n in [initial, 75, 31]
        ]
        for data in chunks:
            data["buffers"].trade_count[:] = 2
            data["buffers"].sharpe_trades[::7] = np.nan
            if pending:
                data["confirm"] = None
                data["proxy"] = None
            else:
                data["proxy"][::3] = np.nan
        heaps = []
        for enabled in (False, True):
            heap = []
            for data in chunks:
                nr._update_heap_total_return_desc(
                    heap=heap,
                    **data,
                    top_k=5,
                    compute_policy=BacktestComputePolicy(
                        local_top_k_max_bytes=64 * 1024 * 1024 if enabled else 0
                    ),
                )
            heaps.append(encode(heap))
        assert heaps[0] == heaps[1], attempt


def test_unbounded_python_minimum_keeps_original_admission():
    data = batch(np.arange(10.0))
    data["min_closed_trades"] = 10**100
    expected = []
    nr._update_heap_total_return_desc(heap=expected, **data, top_k=2)
    actual = []
    nr._update_heap_total_return_desc(
        heap=actual, **data, top_k=2, compute_policy=BacktestComputePolicy()
    )
    assert encode(actual) == encode(expected)
