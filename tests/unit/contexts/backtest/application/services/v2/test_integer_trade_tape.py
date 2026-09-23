"""Independent interval oracle, every TP/SL cell, and bounded dependency failures."""

import dataclasses
import gc
import itertools
import weakref
from typing import Any, cast

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_tp_sl_exact_scoring_service as f,
)
from tests.unit.contexts.backtest.application.services.v2.raw_exact import encode
from trading.contexts.backtest.application.services.v2 import BacktestComboPlanningService
from trading.contexts.backtest.application.services.v2 import integer_trade_tape as tape
from trading.contexts.backtest.application.services.v2 import tp_sl_exact as tp
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy

MODES = ["long_only", "short", "long_short_reversal"]


def prepared(sequence):
    return f._prepared_result(indicator_ids=("alpha",), trade_rows_by_id={"alpha": [sequence]})


def oracle(sequence, mode, start, stop):
    # Model positions at successive executable boundaries, then coalesce intervals.
    positions = []
    current = 0
    for i, raw in enumerate(sequence):
        boundary = start + i + 1
        if boundary >= stop:
            break
        if mode == "long_only":
            current = int(raw == 1)
        elif mode == "short":
            current = -int(raw == -1)
        elif raw:
            current = raw
        positions.append((boundary, current))
    intervals = []
    opened = None
    for boundary, direction in positions + [(stop, 0)]:
        if opened and direction != opened[1]:
            intervals.append((opened[0], opened[1], boundary))
            opened = None
        if direction and opened is None:
            opened = (boundary, direction)
    return tuple(
        np.array([x[i] for x in intervals], dtype=dtype)
        for i, dtype in enumerate((np.int32, np.int8, np.int32))
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("length", range(1, 7))
def test_exhaustive_ternary_intervals(length, mode):
    builder = tape.IntegerTradeTapeBuilder(tp.build_trade_list_15m_for_indicator_rows_slow, 1)
    for sequence in itertools.product((-1, 0, 1), repeat=length):
        for start, stop in ((0, length + 1), (7, 7 + length), (7, 8)):
            p = dataclasses.replace(
                prepared(sequence), time_slice_start_15m=start, time_slice_stop_15m=stop
            )
            args: dict[str, Any] = dict(prepared_result=p, local_indices=(0,), direction_mode=mode)
            expected = oracle(sequence, mode, start, stop)
            assert encode(builder(**args)) == encode(expected)
            assert encode(tp.build_trade_list_15m_for_indicator_rows_slow(**args)) == encode(
                expected
            )


@pytest.mark.parametrize("arity", range(1, 11))
@pytest.mark.parametrize("mode", MODES)
def test_consensus_and_strided_rows(arity, mode):
    ids = tuple(str(i) for i in range(arity))
    rows = {name: [[1, 0, -1, 1, -1, 0], [1, 1, -1, 0, -1, 1]] for name in ids}
    p = f._prepared_result(indicator_ids=ids, trade_rows_by_id=rows)
    pools = tuple(
        dataclasses.replace(pool, trade_T=np.repeat(pool.trade_T, 2, axis=1)[:, ::2])
        for pool in p.indicator_pools
    )
    assert all(pool.trade_T.strides[1] == 2 for pool in pools)
    p = dataclasses.replace(p, time_slice_stop_15m=6, indicator_pools=pools)
    args: dict[str, Any] = dict(
        prepared_result=p, local_indices=tuple(i % 2 for i in range(arity)), direction_mode=mode
    )
    builder = tape.IntegerTradeTapeBuilder(tp.build_trade_list_15m_for_indicator_rows_slow, 1)
    assert encode(builder(**args)) == encode(
        tp.build_trade_list_15m_for_indicator_rows_slow(**args)
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "sizing",
    [
        {"mode": "all_in"},
        {"mode": "fixed_quote", "quote_amount": 100},
        {"mode": "fixed_equity_pct", "equity_pct": 50},
    ],
)
@pytest.mark.parametrize("lock,close", itertools.product((False, True), repeat=2))
@pytest.mark.parametrize("scenario", ["tp_first", "sl_first", "same_minute", "end"])
def test_every_cell_separately_and_jointly(mode, sizing, lock, close, scenario):
    p = prepared(
        [1, -1, 0, 1]
        if mode == "long_short_reversal"
        else ([1, 1, 0, 1] if mode == "long_only" else [-1, -1, 0, -1])
    )
    slow = tp.build_trade_list_15m_for_indicator_rows_slow
    builder = tape.IntegerTradeTapeBuilder(slow, 1)
    calls = []

    def observed(**kwargs):
        result = builder(**kwargs)
        assert encode(result) == encode(slow(**kwargs))
        calls.append(1)
        return result

    def evaluate(tps, sls):
        request = f._normalized_request(
            direction_mode=mode,
            market_type="futures",
            fee_rate=0.00075,
            sizing=sizing,
            profit_lock_enabled=lock,
            close_on_end=close,
        )
        request["execution"]["slippage_rate"] = 0.0002
        request["timeframe"] = "15m"
        for name, values in [("tp", tps), ("sl", sls)]:
            request["risk"][name].update(start_pct=values[0], stop_pct=values[-1], step_pct=0.5)
        ht = 2 if scenario in ("tp_first", "same_minute") else 4
        hs = 2 if scenario in ("sl_first", "same_minute") else 4
        hits = f._hit_times_result(
            tp_values=[x / 100 for x in tps],
            sl_values=[x / 100 for x in sls],
            long_tp=[[4, 4, ht, 4]] * len(tps),
            long_sl=[[4, 4, hs, 4]] * len(sls),
            short_tp=[[4, 4, ht, 4]] * len(tps),
            short_sl=[[4, 4, hs, 4]] * len(sls),
        )
        plan = BacktestComboPlanningService().execute(prepared_result=p, normalized_request=request)
        kwargs: dict[str, Any] = dict(
            prepared_result=p,
            combo_planning_result=plan,
            normalized_request=request,
            hit_times_result=hits,
        )
        baseline = tp.BacktestTpSlExactScoringService().execute(**kwargs)
        candidate = tp.BacktestTpSlExactScoringService(
            compute_policy=BacktestComputePolicy(integer_tape_min_bars=1),
            trade_tape_builder=observed,
        ).execute(**kwargs)
        assert encode(candidate.top_results) == encode(baseline.top_results)
        assert encode(candidate.telemetry.sample_metrics) == encode(
            baseline.telemetry.sample_metrics
        )

    levels = [i / 2 for i in range(1, 11)]
    for tp_value, sl_value in itertools.product(levels, repeat=2):
        evaluate([tp_value], [sl_value])
    evaluate(levels, levels)
    assert len(calls) == 101


@pytest.mark.parametrize("case", ["small", "budget", "timeframe", "bounds", "dtype"])
def test_fallback(case):
    p = prepared([1, -1, 0, 1])
    options = dict(min_bars=1)
    if case == "small":
        options["min_bars"] = 5
    elif case == "budget":
        options["max_bytes"] = 63
    elif case == "timeframe":
        p = dataclasses.replace(p, timeframe="1h")
    elif case == "bounds":
        p = dataclasses.replace(p, time_slice_stop_15m=2**31)
    elif case == "dtype":
        pool = dataclasses.replace(
            p.indicator_pools[0], trade_T=p.indicator_pools[0].trade_T.astype("i2")
        )
        p = dataclasses.replace(p, indicator_pools=(pool,))
    sentinel = (np.array([7]), np.array([7]), np.array([7]))
    calls = []

    def fallback(**kwargs):
        calls.append(kwargs)
        return sentinel

    builder = tape.IntegerTradeTapeBuilder(fallback, **options)
    assert builder(prepared_result=p, local_indices=(0,), direction_mode="long_only") is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_cancellation_discards_arrays(monkeypatch, error):
    refs = []

    def fail(signal, start, stop, reversal, entries, directions, exits):
        refs.extend(weakref.ref(x) for x in (signal, entries, directions, exits))
        raise error("injected scan failure")

    monkeypatch.setattr(tape, "scan_integer_intervals", fail)
    builder = tape.IntegerTradeTapeBuilder(tp.build_trade_list_15m_for_indicator_rows_slow, 1)
    with pytest.raises(error):
        builder(prepared_result=prepared([1, -1, 0, 1]), local_indices=(0,), direction_mode="short")
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_allocation_failure_and_success_lifetime(monkeypatch):
    p = prepared([1, -1, 0, 1])
    args: dict[str, Any] = dict(prepared_result=p, local_indices=(0,), direction_mode="short")
    baseline = tp.build_trade_list_15m_for_indicator_rows_slow(**args)
    builder = tape.IntegerTradeTapeBuilder(tp.build_trade_list_15m_for_indicator_rows_slow, 1)
    with monkeypatch.context() as m:

        def fail(*args, **kwargs):
            raise MemoryError("injected allocation failure")

        m.setattr(tape.np, "empty", fail)
        assert encode(builder(**args)) == encode(baseline)
    out = builder(**args)
    refs = [weakref.ref(x.base) for x in out]
    assert sum(cast(np.ndarray, x.base).nbytes for x in out) == 9 * 4
    del out
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_service_dependency_funding_fallback():
    funding = True

    def fail(**kwargs):
        raise AssertionError("must not invoke tape")

    p = prepared([1, 1, 0, 0])
    tp.BacktestTpSlExactScoringService(
        compute_policy=BacktestComputePolicy(),
        trade_tape_builder=fail,
    ).execute(
        prepared_result=p,
        combo_planning_result=f._combo_planning_result(prepared=p, direction_mode="long_only"),
        hit_times_result=f._hit_times_result(
            tp_values=(0.1,), sl_values=(0.05,), long_tp=[[4] * 4], long_sl=[[4] * 4]
        ),
        normalized_request=f._normalized_request(
            direction_mode="long_only",
            market_type="futures",
            funding_mode="include_when_futures" if funding else "off",
        ),
    )


def test_policy_defaults_bounds():
    assert BacktestComputePolicy().integer_tape_min_bars == 32
    for kwargs in (
        {"integer_tape_min_bars": 0},
        {"integer_tape_max_bytes": -1},
        {"prefix_guard_max_bytes": -1},
    ):
        with pytest.raises(ValueError):
            BacktestComputePolicy(**cast(dict[str, Any], kwargs))


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_service_native_failure_and_cancellation_cleanup(monkeypatch, error):
    refs = []

    def fail(signal, start, stop, reversal, entries, directions, exits):
        refs.extend(weakref.ref(x) for x in (signal, entries, directions, exits))
        raise error("injected second-pass failure")

    monkeypatch.setattr(tape, "scan_integer_intervals", fail)
    p = prepared([1, -1, 0, 1])
    service = tp.BacktestTpSlExactScoringService(
        compute_policy=BacktestComputePolicy(integer_tape_min_bars=1)
    )
    with pytest.raises(error):
        service.execute(
            prepared_result=p,
            combo_planning_result=f._combo_planning_result(prepared=p),
            hit_times_result=f._hit_times_result(
                tp_values=(0.1,),
                sl_values=(0.05,),
                long_tp=[[4] * 4],
                long_sl=[[4] * 4],
                short_tp=[[4] * 4],
                short_sl=[[4] * 4],
            ),
            normalized_request=f._normalized_request(),
        )
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
