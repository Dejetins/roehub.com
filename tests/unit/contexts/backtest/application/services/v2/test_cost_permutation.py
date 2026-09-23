"""Independent raw-bit and actual-native coverage for S2 scheduling and restoration."""

from dataclasses import replace
from typing import Any, cast
from uuid import UUID

import numba as nb
import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.raw_exact import encode
from tests.unit.contexts.backtest.application.services.v2.score_capture import score_capture
from trading.contexts.backtest.application.services.v2 import cost_permutation as cp
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestNumbaThreadDecision,
)
from trading.contexts.backtest.application.services.v2.job_scratch import BacktestJobScratch


@pytest.fixture
def two_threads():
    previous = nb.get_num_threads()
    nb.set_num_threads(2)
    yield
    nb.set_num_threads(previous)


def policy(**kwargs):
    return BacktestComputePolicy(
        threads=BacktestNumbaThreadDecision(nb.get_num_threads(), "test"),
        cost_permutation_min_rows=1,
        **kwargs,
    )


@pytest.mark.parametrize("p", [1, 2, 3, 12])
@pytest.mark.parametrize("n", [0, 1, 2, 7, 13, 25, 101])
def test_installed_native_remainders_and_bijection(p, n):
    previous = nb.get_num_threads()
    try:
        nb.set_num_threads(p)
        owners = cp.native_owners(n)
        expected = np.repeat(np.arange(p), [n // p + (j < n % p) for j in range(p)])
        assert np.array_equal(owners, expected)
        costs = np.arange(n, dtype=np.int64) ** 2 + 1
        order = cp.balanced_order(costs, p)
        assert sorted(order.tolist()) == list(range(n))
    finally:
        nb.set_num_threads(previous)


@pytest.mark.parametrize(
    "backend,arity,direction,kernel",
    [
        ("event_segments_2_no_risk", 2, "long_short_reversal", "event_segments_2_no_risk"),
        ("streaming_2_no_risk", 2, "long_short_reversal", "streaming_2_no_risk"),
        ("event_segments_n_no_risk", 3, "long_short_reversal", "event_segments_n_no_risk"),
        ("matrix_bitset_no_risk_v1", 3, "long_only", "matrix_bitset_no_risk_long_only"),
        ("matrix_bitset_no_risk_v1", 3, "long_short_reversal", "matrix_bitset_no_risk"),
    ],
)
@pytest.mark.parametrize("ranking", ["total_return_pct", "sharpe_trades"])
def test_all_native_no_risk_variants(two_threads, backend, arity, direction, kernel, ranking):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_no_risk_exact_scoring_service as f,
    )

    prepared = f._prepared_result(indicator_ids=("alpha", "beta", "gamma")[:arity])
    planning = f._combo_planning_result(
        prepared=prepared, backend_id=backend, direction_mode=direction
    )
    request = f._normalized_request(top_n=100, direction_mode=direction)
    request["ranking"].update(
        primary_metric=ranking, requested_primary_metric=ranking, effective_primary_metric=ranking
    )
    outputs = []
    for enabled in (False, True):
        scratch = BacktestJobScratch(UUID(int=1))
        with score_capture() as audit:
            result = f.BacktestNoRiskExactScoringService(
                compute_policy=replace(
                    policy(), cost_permutation_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
                scratch=scratch,
            ).execute(
                prepared_result=prepared, combo_planning_result=planning, normalized_request=request
            )
        outputs.append(encode((audit, result.top_results)))
        if enabled:
            assert cp.state_for(scratch).telemetry[kernel + ":enabled"]["calls"] >= 1
        scratch.clear()
        assert scratch.retained_count == 0
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "backend,sizing,kernel",
    [
        ("event_segments_n_tp_sl_15m_grid", "all_in", "event_segments_n_tp_sl_15m_grid"),
        ("matrix_cell_tp_sl_v1", "all_in", "event_segments_n_tp_sl_15m_grid_cell_blocks"),
        (
            "event_segments_n_tp_sl_15m_grid",
            "fixed_quote",
            "event_segments_n_tp_sl_15m_grid_execution_sizing",
        ),
    ],
)
def test_all_native_tp_variants(two_threads, backend, sizing, kernel):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_tp_sl_exact_scoring_service as f,
    )

    prepared = f._prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, -1], [-1, -1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, -1]]
        },
    )
    planning = f._combo_planning_result(prepared=prepared, backend_id=backend)
    request = f._normalized_request(sizing={"mode": sizing, "quote_amount": 100.0})
    outputs = []
    for enabled in (False, True):
        scratch = BacktestJobScratch(UUID(int=1))
        with score_capture() as audit:
            result = f.BacktestTpSlExactScoringService(
                compute_policy=replace(
                    policy(), cost_permutation_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
                scratch=scratch,
            ).execute(
                prepared_result=prepared,
                combo_planning_result=planning,
                normalized_request=request,
                hit_times_result=f._no_hit_times_result(),
            )
        outputs.append(encode((audit, result.top_results)))
        if enabled:
            assert cp.state_for(scratch).telemetry[kernel + ":enabled"]["calls"] >= 1
        scratch.clear()
    assert outputs[0] == outputs[1]


class FakeKernel:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

        def streaming_2_no_risk(
            combo_left_idx, combo_right_idx, left_trade_t, right_trade_t, out_float, out_int
        ):
            pass

        self.py_func = streaming_2_no_risk

    def __call__(self, left, right, lt, rt, out_float, out_int):
        self.calls += 1
        bits = np.array(
            [0, 0x8000000000000000, 0x7FF8000000000001, 0xFFF8000000000007, 0x7FF0000000000000],
            dtype=np.uint64,
        )
        out_float.view(np.uint64)[:] = bits[left]
        out_int[:] = right
        if self.fail:
            raise RuntimeError("partial kernel")


def fake_args(n=5):
    return (
        np.arange(n, dtype=np.int32),
        np.arange(n, dtype=np.int32)[::-1].copy(),
        np.ones((5, 2), dtype=np.int8),
        np.ones((5, 2), dtype=np.int8),
        np.full(n, 17.0),
        np.full(n, 17, dtype=np.int32),
    )


def test_raw_ieee_restore_and_failure_discard(two_threads):
    baseline, candidate = fake_args(), fake_args()
    FakeKernel()(*baseline)
    scratch = BacktestJobScratch(UUID(int=1))
    cp.score_with_cost_permutation(FakeKernel(), policy(), scratch, *candidate)
    assert candidate[-2].tobytes() == baseline[-2].tobytes()
    assert candidate[-1].tobytes() == baseline[-1].tobytes()
    failed = fake_args()
    kernel = FakeKernel(fail=True)
    with pytest.raises(RuntimeError, match="partial kernel"):
        cp.score_with_cost_permutation(kernel, policy(), scratch, *failed)
    assert kernel.calls == 1
    assert np.all(failed[-2] == 17) and np.all(failed[-1] == 17)
    assert not cp.state_for(scratch).costs
    scratch.clear()


@pytest.mark.parametrize("n,budget", [(0, 100000), (1, 100000), (5, 0)])
def test_preallocation_fallback(two_threads, n, budget):
    args = fake_args(n)
    kernel = FakeKernel()
    scratch = BacktestJobScratch(UUID(int=1))
    cp.score_with_cost_permutation(
        kernel, policy(cost_permutation_max_bytes=budget), scratch, *args
    )
    assert kernel.calls == 1
    assert not cp.state_for(scratch).costs


def test_thread_mismatch_rejects_before_kernel(two_threads):
    kernel = FakeKernel()
    with pytest.raises(RuntimeError, match="thread budget"):
        cp.score_with_cost_permutation(
            kernel,
            BacktestComputePolicy(),
            BacktestJobScratch(UUID(int=1)),
            *fake_args(),
        )
    assert kernel.calls == 0


def test_single_thread_and_missing_scratch_fallback(two_threads):
    kernel = FakeKernel()
    cp.score_with_cost_permutation(kernel, policy(), None, *fake_args())
    nb.set_num_threads(1)
    scratch = BacktestJobScratch(UUID(int=1))
    cp.score_with_cost_permutation(kernel, policy(), scratch, *fake_args())
    assert kernel.calls == 2
    assert not cp.state_for(scratch).costs


def test_unsupported_shape_and_backend_fallback(two_threads, monkeypatch):
    for kind in ("shape", "backend"):
        args = list(fake_args())
        kernel = FakeKernel()
        if kind == "shape":
            args[0] = np.arange(10, dtype=np.int32)[::2] // 2
            args[0] = np.repeat(args[0], 2)[::2]
            assert not args[0].flags.c_contiguous
        else:
            monkeypatch.setattr(cp.nb, "threading_layer", lambda: "unsupported")
        scratch = BacktestJobScratch(UUID(int=1))
        cp.score_with_cost_permutation(kernel, policy(), scratch, *args)
        assert kernel.calls == 1
        assert not cp.state_for(scratch).costs


def test_allocation_failure_falls_back_once(two_threads, monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("bounded allocation")

    monkeypatch.setattr(cp.np, "empty_like", fail)
    scratch = BacktestJobScratch(UUID(int=1))
    kernel = FakeKernel()
    cp.score_with_cost_permutation(kernel, policy(), scratch, *fake_args())
    assert kernel.calls == 1
    assert not cp.state_for(scratch).costs
    assert "streaming_2_no_risk:allocation_failure" in cp.state_for(scratch).telemetry


def test_strong_cache_ownership_released_with_job(two_threads):
    import gc
    import weakref

    args = fake_args()
    owner = weakref.ref(args[2])
    scratch = BacktestJobScratch(UUID(int=1))
    cp.score_with_cost_permutation(FakeKernel(), policy(), scratch, *args)
    del args
    gc.collect()
    assert owner() is not None
    scratch.clear()
    gc.collect()
    assert owner() is None
    next_job = BacktestJobScratch(UUID(int=2))
    assert not cp.state_for(next_job).costs


def test_policy_rejects_invalid_limits_and_unintegrated_components():
    for kwargs in (
        {"cost_permutation_min_rows": 0},
        {"cost_permutation_max_bytes": -1},
        {"integer_tape_max_bytes": -1},
        {"local_top_k_max_bytes": -1},
        {"prefix_guard_max_bytes": -1},
    ):
        with pytest.raises(ValueError):
            BacktestComputePolicy(**cast(dict[str, Any], kwargs))


@pytest.mark.parametrize(
    "arity,backend",
    [
        (1, "event_segments_n_no_risk"),
        (4, "event_segments_n_no_risk"),
        (5, "event_segments_n_no_risk"),
        (6, "matrix_bitset_no_risk_v1"),
        (7, "compiled_prefix_product_traversal_v1"),
    ],
)
def test_remaining_arities_and_prefix_rows(two_threads, arity, backend):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_no_risk_exact_scoring_service as f,
    )

    ids = tuple(f"indicator_{j}" for j in range(arity))
    prepared = f._prepared_from_pools(
        indicator_ids=ids,
        pools=[
            f._pool(
                indicator_id=k,
                trade_rows=[[1, 1, 0, -1], [1, 1, 0, -1]],
                eval_rows=[[1, 1, 0], [1, 1, 0]],
            )
            for k in ids
        ],
        row_metadata_order_hash="f" * 64,
    )
    request = f._normalized_request(top_n=100)
    planning = f._combo_planning_result(prepared=prepared, backend_id=backend)
    results = []
    for enabled in (False, True):
        scratch = BacktestJobScratch(UUID(int=1))
        with score_capture() as audit:
            result = f.BacktestNoRiskExactScoringService(
                compute_policy=replace(
                    policy(), cost_permutation_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
                scratch=scratch,
            ).execute(
                prepared_result=prepared, combo_planning_result=planning, normalized_request=request
            )
        results.append(encode((audit, result.top_results)))
        if enabled:
            assert any(k.endswith(":enabled") for k in cp.state_for(scratch).telemetry)
        scratch.clear()
    assert results[0] == results[1]


def test_schedule_mismatch_and_unknown_kernel_fallback(two_threads, monkeypatch):
    monkeypatch.setattr(cp, "native_owners", lambda n: np.zeros(n, dtype=np.int32))
    kernel = FakeKernel()
    scratch = BacktestJobScratch(UUID(int=1))
    cp.score_with_cost_permutation(kernel, policy(), scratch, *fake_args())
    assert kernel.calls == 1
    assert "streaming_2_no_risk:schedule_mismatch" in cp.state_for(scratch).telemetry
    kernel.py_func.__name__ = "unknown_kernel"
    cp.score_with_cost_permutation(kernel, policy(), scratch, *fake_args())
    assert kernel.calls == 2
    assert "unknown_kernel:unsupported_kernel" in cp.state_for(scratch).telemetry


def test_kernel_memory_error_is_not_an_allocation_fallback(two_threads):
    class FailingNative(FakeKernel):
        def __call__(self, *args):
            super().__call__(*args)
            raise MemoryError("partial native outputs")

    kernel = FailingNative()
    args = fake_args()
    with pytest.raises(MemoryError, match="partial native"):
        cp.score_with_cost_permutation(kernel, policy(), BacktestJobScratch(UUID(int=1)), *args)
    assert kernel.calls == 1
    assert np.all(args[-2] == 17) and np.all(args[-1] == 17)


@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("min_trades", [0, 1, 99])
def test_duplicate_original_ids_and_full_ties(two_threads, direction, min_trades):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_no_risk_exact_scoring_service as f,
    )

    prepared = f._prepared_result(indicator_ids=("alpha", "beta"))
    prepared = replace(
        prepared,
        indicator_pools=tuple(
            replace(pool, row_ids=np.zeros_like(pool.row_ids)) for pool in prepared.indicator_pools
        ),
    )
    planning = f._combo_planning_result(prepared=prepared)
    request = f._normalized_request(top_n=100)
    request["ranking"]["direction"] = direction
    if min_trades:
        request["quality_constraints"] = {"min_closed_trades": min_trades}
    outputs = []
    for enabled in (False, True):
        scratch = BacktestJobScratch(UUID(int=1))
        with score_capture() as audit:
            result = f.BacktestNoRiskExactScoringService(
                compute_policy=replace(
                    policy(), cost_permutation_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
                scratch=scratch,
            ).execute(
                prepared_result=prepared, combo_planning_result=planning, normalized_request=request
            )
        outputs.append(encode((audit, result.top_results)))
        scratch.clear()
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "backend,sizing",
    [
        ("event_segments_n_tp_sl_15m_grid", "all_in"),
        ("matrix_cell_tp_sl_v1", "all_in"),
        ("event_segments_n_tp_sl_15m_grid", "fixed_quote"),
    ],
)
def test_forced_full_100_cell_grid(two_threads, backend, sizing):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_tp_sl_exact_scoring_service as f,
    )

    prepared = f._prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, -1], [-1, -1, 0, 1], [1, 0, 1, 1], [1, 1, 0, -1], [-1, 0, -1, -1]]
        },
    )
    planning = f._combo_planning_result(prepared=prepared, backend_id=backend)
    request = f._normalized_request(sizing={"mode": sizing, "quote_amount": 100.0})
    request["risk"]["tp"].update(start_pct=0.5, stop_pct=5.0, step_pct=0.5)
    request["risk"]["sl"].update(start_pct=0.5, stop_pct=5.0, step_pct=0.5)
    hits = f._hit_times_result(
        tp_values=[i * 0.005 for i in range(1, 11)],
        sl_values=[i * 0.005 for i in range(1, 11)],
        long_tp=[[4, 4, 2, 4]] * 10,
        long_sl=[[4, 4, 2, 4]] * 10,
        short_tp=[[4, 4, 2, 4]] * 10,
        short_sl=[[4, 4, 2, 4]] * 10,
    )
    outputs = []
    for enabled in (False, True):
        scratch = BacktestJobScratch(UUID(int=1))
        with score_capture() as audit:
            result = f.BacktestTpSlExactScoringService(
                compute_policy=replace(
                    policy(), cost_permutation_max_bytes=64 * 1024 * 1024 if enabled else 0
                ),
                scratch=scratch,
            ).execute(
                prepared_result=prepared,
                combo_planning_result=planning,
                normalized_request=request,
                hit_times_result=hits,
            )
        outputs.append(encode((audit, result.top_results)))
        if enabled:
            assert any(k.endswith(":enabled") for k in cp.state_for(scratch).telemetry)
        scratch.clear()
    assert outputs[0] == outputs[1]
