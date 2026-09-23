"""Independent activity-set oracle and forced S5 dispatch/bounds proof."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_no_risk_exact_scoring_service as f,
)
from tests.unit.contexts.backtest.application.services.v2.raw_exact import encode
from trading.contexts.backtest.application.services.v2 import no_risk_exact as nr
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy
from trading.contexts.backtest.application.services.v2.matrix_backend import prefix_traversal as pt
from trading.contexts.backtest.application.services.v2.matrix_backend.bitsets import (
    PackedSignalBitsets,
    pack_signal_matrix,
)


def collect(packed, direction="long_only", minimum=1, **kwargs):
    return pt.collect_compiled_prefix_candidates(
        indicator_ids=tuple(map(str, range(len(packed)))),
        packed_by_indicator=packed,
        direction_mode=direction,
        min_closed_trades=minimum,
        **({"guard_max_bytes": 0} | kwargs),
    )


@pytest.mark.parametrize("arity", [6, 7])
@pytest.mark.parametrize("length", [0, 1, 5, 63, 64, 65, 129])
@pytest.mark.parametrize("direction", ["long_only", "short", "long_short_reversal"])
@pytest.mark.parametrize("shared", [False, True])
def test_independent_subset_order_and_threshold(arity, length, direction, shared):
    rng = np.random.default_rng(7123 + length)
    signals = [rng.integers(-1, 2, size=(2, length), dtype=np.int8) for _ in range(arity)]
    if shared and length:
        for s in signals:
            s[:, 0] = -1 if direction == "short" else 1
    packed = tuple(
        pack_signal_matrix(s)
        if length
        else PackedSignalBitsets(
            np.empty((2, 0), dtype=np.uint64), np.empty((2, 0), dtype=np.uint64), 0, 0
        )
        for s in signals
    )
    # Deliberately corrupt padding. Valid-tail masking must exclude these bits.
    if length % 64:
        for p in packed:
            p.pos_bits[:, -1] |= np.uint64(((1 << 64) - 1) ^ ((1 << (length % 64)) - 1))
            p.neg_bits[:, -1] |= np.uint64(((1 << 64) - 1) ^ ((1 << (length % 64)) - 1))
    bound, _ = pt._activity_lower_bound(packed, direction, 1 << 20)
    signs = [1] if direction == "long_only" else [-1] if direction == "short" else [1, -1]
    guaranteed = {sign: set(range(length)) for sign in signs}
    for s in signals:
        for row in s:
            for sign in signs:
                guaranteed[sign] &= set(np.flatnonzero(row == sign).tolist())
    assert bound == sum(map(len, guaranteed.values()))
    expected = []
    for combo in product(range(2), repeat=arity):
        active = {sign: set(range(length)) for sign in signs}
        for family, row in enumerate(combo):
            for sign in signs:
                active[sign] &= set(np.flatnonzero(signals[family][row] == sign).tolist())
                assert guaranteed[sign] <= active[sign]  # every selected prefix
        if sum(map(len, active.values())) >= 1:
            expected.append(combo)
    on, off = (
        collect(packed, direction, guard_max_bytes=64 * 1024 * 1024),
        collect(packed, direction),
    )
    actual = list(zip(*[x.tolist() for x in on.rows_by_indicator.values()]))
    assert actual == expected
    assert encode(on.rows_by_indicator) == encode(off.rows_by_indicator)
    if bound and bound >= 1:
        assert on.telemetry["compiled_loop_elapsed_s"] is None
        assert on.telemetry["traversal_status"] == "not_run_bound_proved_no_pruning"
    else:
        assert on.telemetry["traversal_status"] == "ran"


def test_every_row_and_direction_not_or_before_intersect():
    p = tuple(pack_signal_matrix(np.array([[1, -1], [-1, 1]], dtype=np.int8)) for _ in range(6))
    assert pt._activity_lower_bound(p, "long_short_reversal", 1024)[0] == 0
    assert (
        collect(p, "long_short_reversal", guard_max_bytes=64 * 1024 * 1024).telemetry[
            "prefix_guard_reason"
        ]
        == "bound_insufficient"
    )


@pytest.mark.parametrize("limit,reason", [(0, "proof_byte_limit"), (16, "enumeration_byte_limit")])
def test_allocation_limits_fall_back(limit, reason):
    p = tuple(pack_signal_matrix(np.ones((2, 65), dtype=np.int8)) for _ in range(6))
    if limit == 16:
        p = tuple(pack_signal_matrix(np.ones((2, 5), dtype=np.int8)) for _ in range(6))
    on = collect(p, guard_max_bytes=limit)
    assert on.telemetry["prefix_guard_reason"] == reason
    assert encode(on.rows_by_indicator) == encode(collect(p).rows_by_indicator)


@pytest.mark.parametrize("rows", [100, 2**32, 2**63])
def test_product_checked_before_ndarray_allocation(monkeypatch, rows):
    fake = SimpleNamespace(
        pos_bits=SimpleNamespace(shape=(rows, 1), ndim=2),
        neg_bits=SimpleNamespace(shape=(rows, 1)),
        signal_length=1,
        word_count=1,
    )
    monkeypatch.setattr(pt, "_bitset_stacks", lambda *a: pytest.fail("allocation before limit"))
    with pytest.raises(pt.PrefixMaterializationLimit):
        collect([fake] * 7, guard_max_bytes=64 * 1024 * 1024)


def test_empty_pool_does_not_allocate_large_companion_stack(monkeypatch):
    huge = SimpleNamespace(
        pos_bits=SimpleNamespace(shape=(2**63, 1), ndim=2),
        neg_bits=SimpleNamespace(shape=(2**63, 1)),
        signal_length=1,
        word_count=1,
    )
    empty = replace(
        pack_signal_matrix(np.ones((1, 1), dtype=np.int8)),
        pos_bits=np.empty((0, 1), dtype=np.uint64),
        neg_bits=np.empty((0, 1), dtype=np.uint64),
    )
    monkeypatch.setattr(pt, "_bitset_stacks", lambda *a: pytest.fail("stack allocation"))
    result = collect([huge] * 5 + [empty], guard_max_bytes=64 * 1024 * 1024)
    assert result.candidate_count == 0
    assert result.telemetry["compiled_loop_elapsed_s"] is None


@pytest.mark.parametrize("kind", ["proof", "enumeration"])
def test_memoryerror_uses_baseline(monkeypatch, kind):
    p = tuple(pack_signal_matrix(np.ones((2, 5), dtype=np.int8)) for _ in range(6))
    expected = collect(p)

    def fail(*args, **kwargs):
        raise MemoryError("forced")

    monkeypatch.setattr(pt.np, "full" if kind == "proof" else "arange", fail)
    result = collect(p, guard_max_bytes=64 * 1024 * 1024)
    assert result.telemetry["prefix_guard_reason"] == kind + "_allocation_failed"
    assert encode(result.rows_by_indicator) == encode(expected.rows_by_indicator)


@pytest.mark.parametrize("minimum", [-10, 0, 1, 2, 6])
def test_threshold_exactly_matches_baseline(minimum):
    p = tuple(pack_signal_matrix(np.ones((2, 5), dtype=np.int8)) for _ in range(6))
    assert encode(
        collect(p, minimum=minimum, guard_max_bytes=64 * 1024 * 1024).rows_by_indicator
    ) == encode(collect(p, minimum=minimum).rows_by_indicator)


@pytest.mark.parametrize("direction", ["long_only", "short", "long_short_reversal"])
@pytest.mark.parametrize("arity", [6, 7])
def test_service_forced_guard_scores_real_eligibility(direction, arity, monkeypatch):
    prepared = f._single_signal_prepared_result(
        signal_row=[1, 1, -1, -1, 0, 1, 1, -1],
        arity=arity,
        indicator_ids=tuple(f"indicator_{i}" for i in range(arity)),
    )
    request = f._normalized_request(direction_mode=direction, top_n=1)
    plan = f._combo_planning_result(
        prepared=prepared,
        backend_id=nr.COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND,
        direction_mode=direction,
    )
    baseline = nr.BacktestNoRiskExactScoringService(
        compute_policy=BacktestComputePolicy(prefix_guard_max_bytes=0)
    ).execute(prepared_result=prepared, combo_planning_result=plan, normalized_request=request)
    calls = []
    original = nr.evaluate_no_risk_exact_chunk

    def spy(**kwargs):
        result = original(**kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(nr, "evaluate_no_risk_exact_chunk", spy)
    result = nr.BacktestNoRiskExactScoringService(compute_policy=BacktestComputePolicy()).execute(
        prepared_result=prepared, combo_planning_result=plan, normalized_request=request
    )
    assert calls
    assert result.telemetry.prefix_traversal is not None
    assert encode(result.top_results) == encode(baseline.top_results)
    assert (
        result.telemetry.prefix_traversal["traversal_status"] == "not_run_bound_proved_no_pruning"
    )
    assert result.telemetry.prefix_traversal["compiled_loop_elapsed_s"] is None


def test_policy_default_and_bounds():
    assert BacktestComputePolicy().prefix_guard_max_bytes == 64 * 1024 * 1024
    with pytest.raises(ValueError):
        BacktestComputePolicy(prefix_guard_max_bytes=-1)


def test_partial_enumeration_released_before_baseline(monkeypatch):
    import gc
    import weakref

    p = tuple(pack_signal_matrix(np.ones((2, 5), dtype=np.int8)) for _ in range(6))
    real_asarray, real_stacks = pt.np.asarray, pt._bitset_stacks
    refs = []
    calls = 0

    def fail_second_row(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("second row")
        array = real_asarray(*args, **kwargs)
        refs.append(weakref.ref(array))
        return array

    def fallback(*args):
        gc.collect()
        assert refs and all(ref() is None for ref in refs)
        monkeypatch.setattr(pt.np, "asarray", real_asarray)
        return real_stacks(*args)

    monkeypatch.setattr(pt.np, "asarray", fail_second_row)
    monkeypatch.setattr(pt, "_bitset_stacks", fallback)
    result = collect(p, guard_max_bytes=64 * 1024 * 1024)
    assert result.candidate_count == 64
    assert result.telemetry["prefix_guard_reason"] == "enumeration_allocation_failed"


def test_empty_inconsistent_shape_rejected():
    valid = pack_signal_matrix(np.ones((1, 1), dtype=np.int8))
    bad = replace(valid, pos_bits=np.empty((0, 1), dtype=np.uint64))
    with pytest.raises(ValueError, match="aligned"):
        collect([valid] * 5 + [bad], guard_max_bytes=64 * 1024 * 1024)


@pytest.mark.parametrize("guard", [False, True])
def test_materialization_boundary_retains_rejection(guard):
    p = tuple(pack_signal_matrix(np.ones((2, 5), dtype=np.int8)) for _ in range(6))
    with pytest.raises(ValueError, match="materialization limit"):
        collect(p, guard_max_bytes=64 * 1024 * 1024 if guard else 0, max_materialized_candidates=63)
    assert (
        collect(
            p, guard_max_bytes=64 * 1024 * 1024 if guard else 0, max_materialized_candidates=64
        ).candidate_count
        == 64
    )


def test_unknown_dtype_proof_uses_baseline_cast():
    p = pack_signal_matrix(np.ones((2, 5), dtype=np.int8))
    p = replace(p, pos_bits=p.pos_bits.astype(np.int64), neg_bits=p.neg_bits.astype(np.int64))
    result = collect([p] * 6, guard_max_bytes=64 * 1024 * 1024)
    assert result.telemetry["prefix_guard_reason"] == "unsupported_shape"
    assert result.candidate_count == 64
