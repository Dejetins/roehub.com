from __future__ import annotations

from scripts.rl_trading import stage08o_stage08k_dqn_forensic_decomposition as stage08o


def test_stage08o_reconstructs_closed_trades_from_balance_changes() -> None:
    rows = [
        _row(0, 100.0, "BTCUSDT", 1),
        _row(1, 100.0, "BTCUSDT", 0),
        _row(2, 112.5, "BTCUSDT", 3),
        _row(0, 112.5, "ETHUSDT", 2),
        _row(1, 107.0, "ETHUSDT", 3),
    ]

    trades = stage08o._reconstruct_trades(rows)  # noqa: SLF001

    assert [trade["side"] for trade in trades] == ["long", "short"]
    assert [trade["pnl_after_costs_quote"] for trade in trades] == [12.5, -5.5]
    assert sum(trade["pnl_after_costs_quote"] for trade in trades) == 7.0


def test_stage08o_dominance_uses_absolute_pnl_denominator() -> None:
    rows = [
        {"bucket": "high", "net_pnl_after_costs_quote": 12482.1319445},
        {"bucket": "low", "net_pnl_after_costs_quote": 307.01029846},
        {"bucket": "medium", "net_pnl_after_costs_quote": -286.48891271},
    ]

    dominance = stage08o._dominance(rows, label_key="bucket")  # noqa: SLF001

    assert dominance["dominant_group"] == "high"
    assert dominance["share"] == 0.954610281973835


def test_stage08o_positive_group_ratio_keeps_flat_groups_in_denominator() -> None:
    rows = [
        {"net_pnl_after_costs_quote": 10.0},
        {"net_pnl_after_costs_quote": 0.0},
        {"net_pnl_after_costs_quote": -1.0},
        {"net_pnl_after_costs_quote": 2.0},
    ]

    ratio = stage08o._positive_group_ratio(rows)  # noqa: SLF001

    assert ratio == {
        "positive_groups": 2,
        "flat_groups": 1,
        "non_positive_groups": 2,
        "total_groups": 4,
        "ratio": 0.5,
    }


def _row(
    step_idx: int,
    balance: float,
    symbol: str,
    effective_action_id: int,
) -> dict[str, object]:
    return {
        "effective_action_id": effective_action_id,
        "shared_balance_quote": balance,
        "signal_time_utc": "2026-01-01T00:00:00Z",
        "source_session_index": symbol,
        "step_idx": step_idx,
        "symbol": symbol,
    }
