from __future__ import annotations

from typing import Any

import pytest

from tests.unit.contexts.backtest.application.services.v2.test_backtest_preflight_service import (
    _service,
    _valid_request,
)
from tests.unit.contexts.backtest.application.services.v2.test_lazy_trades_detail_service import (
    _MemoryCache,
)
from tests.unit.contexts.backtest.application.services.v2.test_tp_sl_exact_scoring_service import (
    _combo_planning_result,
    _funding_arrays,
    _hit_times_result,
    _no_hit_times_result,
    _normalized_request,
    _prepared_result,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestComboPlanningService,
    BacktestNoRiskExactScoringService,
    BacktestTpSlExactScoringService,
)


@pytest.mark.parametrize("risk_mode", ["none", "tp_sl_grid"])
def test_standalone_short_closes_on_neutral(risk_mode: str) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[-1, -1, 0, 1]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(direction_mode="short", market_type="futures")
    request["risk"]["mode"] = risk_mode
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    if risk_mode == "none":
        result = BacktestNoRiskExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=planning,
            normalized_request=request,
        )
    else:
        result = BacktestTpSlExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=planning,
            normalized_request=request,
            hit_times_result=_no_hit_times_result(),
        )
    # Sell at 100, neutral signal closes at next open 120; never enter long.
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(-20, abs=1e-4)
    assert result.top_results[0].metrics["trade_count"] == 1


@pytest.mark.parametrize("direction, expected", [("asc", [-25, 20]), ("desc", [20, -25])])
@pytest.mark.parametrize("top_n", [1, 2])
def test_audit_tp_sl_ranking_counterexample(
    direction: str, expected: list[int], top_n: int
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0], [-1, -1, 1, 1]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(direction_mode="long_only", top_n=top_n)
    request["ranking"]["direction"] = direction
    from trading.contexts.backtest.application.dto import BacktestCoordinates

    request["ranking"] = _service()._normalize_ranking(
        payload=request,
        coordinates=BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT"),
        execution=request["execution"],
        risk=request["risk"],
    )
    request["risk"].update(
        tp={"start_pct": 50.0, "stop_pct": 50.0, "step_pct": 1.0},
        sl={"start_pct": 50.0, "stop_pct": 50.0, "step_pct": 1.0},
    )
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared, direction_mode="long_only"),
        normalized_request=request,
        hit_times_result=_no_hit_times_result(),
    )
    # Independently: 100 -> 120 = +20%; 120 -> 90 = -25%.
    assert [r.metrics["total_return_pct"] for r in result.top_results] == pytest.approx(
        expected[:top_n],
        abs=1e-4,
    )
    assert [r.indicator_rows["alpha"] for r in result.top_results] == (
        [1, 0] if direction == "asc" else [0, 1]
    )[:top_n]


@pytest.mark.parametrize(
    "metric",
    [
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    ],
)
@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("top_n", [1, 7, 12])
def test_all_ranking_metrics_before_multiblock_truncation(
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    direction: str,
    top_n: int,
) -> None:
    import trading.contexts.backtest.application.services.v2.tp_sl_exact as exact

    # Four identical winners, losers and flat rows cross block boundaries.
    # Each has zero/one trade, so Sharpe is zero; positive-only PF/return-DD are +inf.
    patterns = [[1, 1, 0, 0], [-1, -1, 1, 1], [0, 0, 0, 0]]
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [patterns[i % 3] for i in range(12)]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(direction_mode="long_only", top_n=top_n)
    request["ranking"] = {"primary_metric": metric, "direction": direction}
    expected_values = {
        "total_return_pct": [20.0, -25.0, 0.0],
        "max_drawdown_pct": [0.0, 25.0, 0.0],
        "return_over_max_drawdown": [float("inf"), -1.0, 0.0],
        "profit_factor": [float("inf"), 0.0, 0.0],
        "sharpe_trades": [0.0, 0.0, 0.0],
        "win_rate_pct": [100.0, 0.0, 0.0],
    }[metric]
    expected_rows = sorted(
        range(12),
        key=lambda i: (
            expected_values[i % 3] * (1 if direction == "asc" else -1),
            i,
        ),
    )[:top_n]
    for block_size in (3, 5, 4096):
        monkeypatch.setattr(exact, "COMBO_CHUNK_SIZE", block_size)
        result = BacktestTpSlExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=_combo_planning_result(
                prepared=prepared, direction_mode="long_only"
            ),
            normalized_request=request,
            hit_times_result=_no_hit_times_result(),
        )
        assert [r.indicator_rows["alpha"] for r in result.top_results] == expected_rows
        assert [r.rank for r in result.top_results] == list(range(1, top_n + 1))
        for row, index in zip(result.top_results, expected_rows, strict=True):
            assert row.score == pytest.approx(expected_values[index % 3], abs=1e-4)
            assert row.metrics[metric] == pytest.approx(expected_values[index % 3], abs=1e-4)
            assert (row.best_tp_idx, row.best_sl_idx) == (0, 0)


@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("metric", ["total_return_pct", "max_drawdown_pct", "profit_factor"])
def test_ranking_preserves_maximum_return_cell_contract(direction: str, metric: str) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 120],
    )
    hits = _hit_times_result(
        tp_values=(0.1, 0.5),
        sl_values=(0.5,),
        long_tp=[[4, 4, 2, 4], [4, 4, 4, 4]],
        long_sl=[[4, 4, 4, 4]],
    )
    request = _normalized_request(direction_mode="long_only", top_n=1)
    request["ranking"] = {"primary_metric": metric, "direction": direction}
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared, direction_mode="long_only"),
        normalized_request=request,
        hit_times_result=hits,
    )
    # Cell 0 hits +10%; cell 1 exits at +20%. The contract selects max return
    # within each combo, even when rows are ranked ascending/by another metric.
    assert result.top_results[0].best_tp_idx == 1
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(20, abs=1e-4)


@pytest.mark.parametrize(
    "backend, arity",
    [
        ("event_segments_n_no_risk", 1),
        ("event_segments_2_no_risk", 2),
        ("streaming_2_no_risk", 2),
        ("matrix_bitset_no_risk_v1", 6),
        ("compiled_prefix_product_traversal_v1", 7),
    ],
)
@pytest.mark.parametrize(
    "signals, expected_count, expected_return",
    [
        ([-1, -1, 0, 1], 1, -20.0),
        ([1, -1, -1, -1], 1, 100 * (1 - 90 / 110)),
        ([1, 1, 1, 1], 0, 0.0),
        ([-1, 1, -1, -1], 2, 12.5),
    ],
)
def test_short_no_risk_backends(
    backend: str,
    arity: int,
    signals: list[int],
    expected_count: int,
    expected_return: float,
) -> None:
    ids = tuple(f"i{i}" for i in range(arity))
    prepared = _prepared_result(
        indicator_ids=ids,
        trade_rows_by_id={key: [signals] for key in ids},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(direction_mode="short", market_type="futures")
    request["risk"] = {"mode": "none"}
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
        requested_backend_id=backend,
    )
    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
    )
    # Prefix traversal already prunes zero-activity combinations.
    if backend == "compiled_prefix_product_traversal_v1" and expected_count == 0:
        assert result.top_results == ()
        return
    assert result.top_results[0].metrics["trade_count"] == expected_count
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(
        expected_return,
        abs=1e-4,
    )


@pytest.mark.parametrize("risk_mode", ["none", "tp_sl_grid"])
@pytest.mark.parametrize("arity", [1, 6, 7])
def test_preflight_to_actual_short_orchestration(risk_mode: str, arity: int) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from trading.contexts.backtest.application.services.v2.job_orchestration import (
        BacktestRuntimeJobOrchestrationService,
    )

    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["time_range"] = {"start": "2020-01-01T00:00:00Z", "end": "2020-01-01T01:00:00Z"}
    request["quality_constraints"] = {"min_closed_trades": 1}
    risk: dict[str, Any] = {"mode": risk_mode}
    request["risk"] = risk
    if risk_mode == "tp_sl_grid":
        risk.update(
            tp={"start_pct": 50.0, "stop_pct": 50.0, "step_pct": 1.0},
            sl={"start_pct": 25.0, "stop_pct": 25.0, "step_pct": 1.0},
        )
    request["execution"].update(
        direction_mode="short",
        fee_rate=0.0,
        slippage_rate=0.0,
        sizing={"mode": "all_in"},
        funding={"mode": "off", "coverage_policy": "degraded_with_warning"},
    )
    ids = ("ma.dema", "ma.ema", "ma.hma", "ma.sma", "ma.tema", "ma.wma", "ma.zlema")[:arity]
    request["indicators"] = [
        {"indicator_id": key, "sources": ["close"], "window": {"start": 5, "stop": 5, "step": 1}}
        for key in ids
    ]
    preflight = _service().execute(request)
    prepared = _prepared_result(
        indicator_ids=ids,
        trade_rows_by_id={key: [[-1, -1, 0, 1]] for key in ids},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    service = BacktestRuntimeJobOrchestrationService(
        prepare_pools=_FixturePort(prepared),
        combo_planning=BacktestComboPlanningService(),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=_FixturePort(_no_hit_times_result()),
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=_FixturePort(None),
    )
    result = service.execute(
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        preflight=preflight,
        updated_at=datetime(2026, 9, 18, tzinfo=UTC),
    )
    assert len(result.top_variants) == 1
    if risk_mode == "none" and arity in (6, 7):
        expected_backend = (
            "matrix_bitset_no_risk_v1" if arity == 6 else "compiled_prefix_product_traversal_v1"
        )
        assert result.exact_diagnostics["telemetry"]["backend_id"] == expected_backend
    row = result.top_variants[0]
    assert row.summary_metrics_json["total_return_pct"] == pytest.approx(-20, abs=1e-4)
    assert row.summary_metrics_json["trade_count"] == 1


class _FixturePort:
    """Only substitute artifact I/O; planning, kernels, warmup and assembly are real."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def execute(self, **kwargs: Any) -> Any:
        return self.value

    def resolve_context(self, **kwargs: Any) -> None:
        return None


@pytest.mark.parametrize("risk_mode", ["none", "tp_sl_grid"])
@pytest.mark.parametrize("close_on_end", [False, True])
@pytest.mark.parametrize(
    "sizing, quote",
    [
        ({"mode": "all_in"}, 10000.0),
        ({"mode": "fixed_quote", "quote_amount": 100.0}, 100.0),
        ({"mode": "fixed_equity_pct", "equity_pct": 25.0}, 2500.0),
        ({"mode": "fixed_equity_pct_min_quote", "equity_pct": 1.0, "min_quote": 500.0}, 500.0),
        ({"mode": "fixed_equity_pct_max_quote", "equity_pct": 50.0, "max_quote": 100.0}, 100.0),
    ],
)
def test_short_sizing_fees_slippage_close_and_lazy_details(
    risk_mode: str,
    close_on_end: bool,
    sizing: dict,
    quote: float,
) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    import numpy as np

    from trading.contexts.backtest.application.services.v2 import BacktestLazyTradesDetailService

    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[-1, -1, -1, -1]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(
        direction_mode="short",
        market_type="futures",
        fee_rate=0.001,
        sizing=sizing,
        close_on_end=close_on_end,
    )
    request["execution"]["slippage_rate"] = 0.01
    request["risk"]["mode"] = risk_mode
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    hits = _no_hit_times_result()
    detail = BacktestLazyTradesDetailService(
        prepare_pools=cast(Any, _FixturePort(prepared)),
        tp_sl_hit_times=cast(Any, _FixturePort(hits)),
        cache=_MemoryCache(),
    )
    times = SimpleNamespace(
        open_time=np.arange(4) * 900_000, close_time=np.arange(4) * 900_000 + 899_999
    )
    arrays = SimpleNamespace(price_arrays_1m=times, price_arrays_15m=times)
    if risk_mode == "none":
        result = BacktestNoRiskExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=planning,
            normalized_request=request,
        )
        summary, trades, _ = detail._no_risk_detail(
            normalized_request=request,
            prepared=prepared,
            local_indices=(0,),
            runtime_arrays=arrays,
        )
        # Sell at 99, buy back at 90.9, charge fees on both actual notionals.
        expected_pnl = quote * ((99 - 90.9) / 99 - 0.001 * (1 + 90.9 / 99))
    else:
        result = BacktestTpSlExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=planning,
            normalized_request=request,
            hit_times_result=hits,
        )
        summary, trades, _ = detail._tp_sl_detail(
            normalized_request=request,
            prepared=prepared,
            local_indices=(0,),
            runtime_arrays=arrays,
            row=cast(Any, SimpleNamespace(best_tp_pct=50.0, best_sl_pct=50.0)),
            context=None,
        )
        # Preserve the existing TP/SL model: two fee factors, raw prices, no slippage.
        expected_pnl = quote * (1.1 * 0.999**2 - 1)
    expected = expected_pnl / 100 if close_on_end else 0.0
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(expected, abs=1e-4)
    assert summary["total_return_pct"] == pytest.approx(expected, abs=1e-4)
    assert len(trades) == int(close_on_end)
    if trades:
        assert trades[0]["side"] == "short"
        assert trades[0]["exit_reason"] == "close_on_end"
        assert trades[0]["net_pnl_quote"] == pytest.approx(expected_pnl, abs=1e-3)
        assert trades[0]["notional_quote"] == quote


@pytest.mark.parametrize("risk_mode", ["none", "tp_sl_grid"])
@pytest.mark.parametrize("funding_rate", [0.01, -0.01])
def test_short_funding_receives_positive_and_pays_negative(
    risk_mode: str, funding_rate: float
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[-1, -1, 0, 1]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(
        direction_mode="short",
        market_type="futures",
        funding_mode="include_when_futures",
    )
    request["risk"]["mode"] = risk_mode
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    funding = _funding_arrays(
        funding_time=(900_000, 1_800_000, 2_700_000, 3_600_000),
        funding_rate=(funding_rate,) * 4,
        mark_price=(100.0,) * 4,
    )
    kwargs: dict[str, Any] = dict(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        funding_arrays=funding,
    )
    if risk_mode == "none":
        result = BacktestNoRiskExactScoringService().execute(**kwargs)
    else:
        result = BacktestTpSlExactScoringService().execute(
            **kwargs,
            hit_times_result=_no_hit_times_result(),
        )
    metrics = result.top_results[0].metrics
    # Entry at 900000 excluded; exit at 2700000 included. 100 base units.
    assert metrics["funding_events_count"] == 2
    assert metrics["funding_pnl_quote"] == pytest.approx(2 * 100 * 100 * funding_rate)
    assert metrics["total_return_pct"] == pytest.approx(-20, abs=1e-4)
    assert metrics["total_return_pct_net_of_funding"] == pytest.approx(
        -20 + 200 * funding_rate,
        abs=1e-4,
    )


@pytest.mark.parametrize("tp_hit, sl_hit, expected", [(2, 4, 10.0), (4, 2, -5.0), (2, 2, -5.0)])
@pytest.mark.parametrize(
    "backend, sizing, fraction",
    [
        ("event_segments_n_tp_sl_15m_grid", {"mode": "all_in"}, 1.0),
        ("matrix_cell_tp_sl_v1", {"mode": "all_in"}, 1.0),
        ("event_segments_n_tp_sl_15m_grid", {"mode": "fixed_quote", "quote_amount": 100.0}, 0.01),
    ],
)
def test_short_tp_sl_hit_precedence_and_sizing(
    backend: str,
    tp_hit: int,
    sl_hit: int,
    expected: float,
    sizing: dict,
    fraction: float,
) -> None:
    from trading.contexts.backtest.application.dto import BacktestTpSlExactConfig

    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[-1, -1, 0, 1]]},
    )
    hits = _hit_times_result(short_tp=[[4, 4, tp_hit, 4]], short_sl=[[4, 4, sl_hit, 4]])
    request = _normalized_request(direction_mode="short", market_type="futures", sizing=sizing)
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
        requested_backend_id=backend,
    )
    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(run_self_check=True),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        hit_times_result=hits,
    )
    assert result.top_results[0].metrics["total_return_pct"] == pytest.approx(
        expected * fraction,
        abs=1e-4,
    )
    assert result.top_results[0].metrics["trade_count"] == 1
    assert result.self_check.status == "passed"


@pytest.mark.parametrize("direction, expected_row", [("asc", 0), ("desc", 1)])
def test_tp_sl_ranking_uses_finite_multitrade_sharpe(direction: str, expected_row: int) -> None:
    import math

    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, -1, 1, 0], [1, 0, -1, 0]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    request = _normalized_request(
        direction_mode="long_short_reversal", market_type="futures", top_n=1
    )
    request["ranking"] = {"primary_metric": "sharpe_trades", "direction": direction}
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=request,
        hit_times_result=_no_hit_times_result(),
    )
    # Three trades: long 100->110, short 110->120, long 120->90.
    # Two trades: long 100->120, short 120->90 (reversal ignores neutral).
    returns = ([0.1, 1 - 120 / 110, -0.25], [0.2, 0.25])[expected_row]
    mean = sum(returns) / len(returns)
    variance = sum(x * x for x in returns) / len(returns) - mean * mean
    expected = mean / math.sqrt(variance) * math.sqrt(len(returns) * 35040 / 4)
    top = result.top_results[0]
    assert top.indicator_rows["alpha"] == expected_row
    assert top.score == pytest.approx(expected, rel=1e-6)
    assert top.metrics["sharpe_trades"] == pytest.approx(expected, rel=1e-6)


def test_tp_sl_nonfinite_ratios_keep_existing_json_policy() -> None:
    import json
    from datetime import UTC, datetime
    from uuid import UUID

    from trading.contexts.backtest.application.services.v2 import BacktestTopResultAssemblyService

    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 120],
    )
    request = _normalized_request(direction_mode="long_only", top_n=1)
    request["ranking"] = {"primary_metric": "profit_factor", "direction": "desc"}
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared, direction_mode="long_only"),
        normalized_request=request,
        hit_times_result=_no_hit_times_result(),
    )
    assert result.top_results[0].score == float("inf")
    assembled = BacktestTopResultAssemblyService().assemble(
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        normalized_request=request,
        top_results=result.top_results,
        updated_at=datetime(2026, 9, 18, tzinfo=UTC),
    )
    row = assembled.top_variants[0]
    assert row.payload_json["source_top_result"]["metrics"]["profit_factor"] is None
    assert row.payload_json["source_top_result"]["metrics"]["return_over_max_drawdown"] is None
    json.dumps(dict(row.payload_json), allow_nan=False)


@pytest.mark.parametrize(
    "metric",
    [
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
    ],
)
@pytest.mark.parametrize("direction_mode", ["long_only", "short", "long_short_reversal"])
@pytest.mark.parametrize("close_on_end", [True, False])
@pytest.mark.parametrize("offset", [0, 3])
def test_compiled_cell_metrics_match_reference_with_sizing_and_profit_lock(
    metric: str,
    direction_mode: str,
    close_on_end: bool,
    offset: int,
) -> None:
    from dataclasses import replace

    import numpy as np

    from trading.contexts.backtest.application.dto import BacktestTpSlHitTimesSubset

    prepared = _prepared_result(
        indicator_ids=("alpha", "beta"),
        trade_rows_by_id={
            "alpha": [[1, -1, 1, 0], [-1, 0, -1, 1], [1, 1, -1, 0]],
            "beta": [[1, -1, 1, 0], [-1, 0, -1, 1], [1, 1, -1, 0]],
        },
        open_1m=[100, 100, 110, 120],
        close_1m=[100, 100, 110, 90],
    )
    prepared = replace(prepared, time_slice_start_15m=offset, time_slice_stop_15m=offset + 4)
    hits = _no_hit_times_result()
    table = np.full((1, offset + 4), offset + 4, dtype=np.uint32)
    hits = replace(
        hits,
        hit_times=BacktestTpSlHitTimesSubset(
            tp_values=hits.hit_times.tp_values,
            sl_values=hits.hit_times.sl_values,
            long_tp=table,
            long_sl=table,
            short_tp=table,
            short_sl=table,
            sentinel_index=offset + 4,
        ),
    )
    request = _normalized_request(
        direction_mode=direction_mode,
        market_type="futures",
        close_on_end=close_on_end,
        fee_rate=0.001,
        profit_lock_enabled=True,
        sizing={"mode": "fixed_equity_pct", "equity_pct": 25.0},
        top_n=9,
    )
    request["ranking"] = {"primary_metric": metric, "direction": "asc"}
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared, direction_mode=direction_mode
        ),
        normalized_request=request,
        hit_times_result=hits,
    )
    assert len(result.top_results) == 9
    # Metrics are rebuilt by the independent slow selected-cell detail path after
    # admission; score was computed by the compiled, pre-truncation path.
    for top in result.top_results:
        assert top.score == pytest.approx(top.metrics[metric], abs=1e-6)
    scores = [top.score for top in result.top_results]
    assert scores == sorted(scores)
