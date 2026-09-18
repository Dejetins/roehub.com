from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from apps.api.wiring.modules.strategy_execution_projection import project_fills


def facts(*orders) -> list[dict[str, Any]]:
    return [
        dict(
            fill_id=str(i),
            time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i),
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            reason=reason,
        )
        for i, (side, quantity, price, fee, reason) in enumerate(orders)
    ]


def test_partial_exit_and_scale_in_preserve_cycle_price_and_cost_basis():
    result = project_fills(
        facts(
            ("buy", 2, 100, 1, "signal"),
            ("sell", 1, 110, 1, "take_profit"),
            ("buy", 1, 200, 1, "manual"),
            ("sell", 2, 160, 1, "manual"),
        ),
        initial_cash=1000,
    )
    (trade,) = result.trades
    assert trade.entry == pytest.approx(400 / 3)
    assert trade.exit == pytest.approx(430 / 3)
    assert trade.gross_pnl == 30
    assert trade.net_pnl == 26
    assert trade.quantity == 3
    assert trade.remaining_quantity == 0
    assert trade.entry_reason == "signal"
    assert trade.exit_reason == "manual"
    assert result.equity[-1].value == 1026


def test_reversal_splits_fill_and_allocates_fee_without_losing_execution():
    result = project_fills(
        facts(
            ("buy", 1, 100, 1, "signal"),
            ("sell", 3, 90, 3, "stop_loss"),
            ("buy", 2, 80, 2, "manual"),
        ),
        initial_cash=1000,
    )
    short, long = result.trades
    assert (long.net_pnl, short.net_pnl) == (-12, 16)
    assert short.side == "short"
    fees = []
    for trade in result.trades:
        for fill in trade.fills:
            assert fill.fee is not None
            fees.append(fill.fee)
    assert sum(fees) == 6
    assert len({f.fill_id for t in result.trades for f in t.fills}) == 4
    assert result.drawdown[0].value == pytest.approx(-1.2)


def test_unknown_costs_never_become_zero_net_profit_or_equity():
    result = project_fills(
        facts(("buy", 1, 100, None, "manual"), ("sell", 1, 110, 1, "signal")), initial_cash=1000
    )
    assert result.trades[0].net_pnl is None
    assert result.trades[0].gross_pnl == 10
    assert result.equity == []


def test_unknown_initial_capital_does_not_invent_equity():
    result = project_fills(facts(("sell", 1, 100, 1, "signal"), ("buy", 1, 90, 1, "take_profit")))
    assert result.trades[0].net_pnl == 8
    assert result.equity == []


def test_unknown_funding_keeps_observed_fees_but_withholds_net():
    rows = facts(("buy", 1, 100, 1, "signal"), ("sell", 1, 110, 1, "manual"))
    for row in rows:
        row["costs_complete"] = False
    result = project_fills(rows, initial_cash=1000)
    assert result.trades[0].fees == 2
    assert result.trades[0].net_pnl is None
    assert result.equity == []


def test_demo_manual_duplicate_is_same_execution_and_stop_preserves_position(tmp_path):
    from tools.qa.strategy_execution_demo import demo_fills, demo_state, simulate_command

    strategy = "54f2d6f4-7cb7-4d7b-9469-437b2bb77362"
    path = tmp_path / "demo.json"
    before = len(demo_fills(strategy, demo_state(path)))
    first, status = simulate_command(strategy, "manual-exit", {"client_request_id": "same"}, path)
    duplicate, repeated_status = simulate_command(
        strategy, "manual-exit", {"client_request_id": "same"}, path
    )
    assert status == repeated_status == 200
    assert first["intent_id"] == duplicate["intent_id"]
    assert duplicate["duplicate"] is True
    assert len(demo_fills(strategy, demo_state(path))) == before + 1
    simulate_command(
        strategy, "manual-entry", {"client_request_id": "entry", "quote_notional": 1000}, path
    )
    simulate_command(strategy, "stop", {}, path)
    rows = demo_fills(strategy, demo_state(path))
    assert sum(row["quantity"] * (1 if row["side"] == "buy" else -1) for row in rows) > 0
    assert demo_state(path)["running"] is False


def test_reference_price_is_preserved_per_execution_and_missing_is_unknown():
    rows = facts(("buy", 1, 101, 0, "signal"), ("sell", 1, 99, 0, "manual"))
    rows[0]["reference_price"] = 100
    trade = project_fills(rows).trades[0]
    assert trade.fills[0].reference_price == 100
    assert trade.fills[0].price == 101
    assert trade.fills[1].reference_price is None


def test_oversized_history_does_not_publish_stale_trades_or_unseeded_pnl():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from apps.api.wiring.modules.strategy_operation_reads import StrategyOperationReads

    reader = Mock()
    reader.read.return_value = facts(*[("buy", 1, 100, 1, "manual")] * 5001)
    strategy = SimpleNamespace(
        strategy_id="strategy", spec=SimpleNamespace(
            instrument_id=SimpleNamespace(symbol="BTCUSDT"), market_type="spot"
        ),
    )
    operations, _ = StrategyOperationReads(reader).read(
        organization_id="org", user_id="owner", strategy=strategy,
        run=SimpleNamespace(run_id="run"), profile=SimpleNamespace(mode="paper"),
    )
    assert operations.state == "unavailable"
    assert operations.reason == "execution_history_limit_exceeded"
    assert operations.partial is True
    assert operations.trades == operations.equity == operations.drawdown == []
