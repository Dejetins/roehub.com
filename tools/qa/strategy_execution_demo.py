"""Opt-in local simulation. No runner, exchange, or trading provider is called."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from apps.api.wiring.modules.strategy_execution_projection import project_fills


def demo_state(path: Path | None = None) -> dict:
    if path and path.exists():
        return json.loads(path.read_text())
    state = dict(
        anchor=datetime.now(UTC).replace(second=0, microsecond=0).isoformat(),
        running=True,
        commands={},
        extra=[],
    )
    if path:
        path.write_text(json.dumps(state))
    return state


def demo_fills(strategy_id: str, state: dict) -> list:
    anchor = datetime.fromisoformat(state["anchor"])
    prices = [
        65000,
        65400,
        65300,
        65050,
        64800,
        65500,
        65600,
        65300,
        65200,
        66000,
        65800,
        66200,
        66000,
    ]
    reasons = [
        "signal",
        "take_profit",
        "manual",
        "stop_loss",
        "signal",
        "take_profit",
        "signal",
        "trailing_stop",
        "manual",
        "manual",
        "signal",
        "take_profit",
        "signal",
    ]
    rows: list[dict[str, Any]] = []
    for i, price in enumerate(prices):
        rows.append(
            dict(
                fill_id=str(uuid5(NAMESPACE_URL, f"{strategy_id}:demo-fill:{i}")),
                time=anchor - timedelta(minutes=30 * (len(prices) - i)),
                price=price,
                reference_price=price - 12 if i % 2 == 0 else price + 8,
                quantity=0.02,
                fee=price * 0.02 * 0.00075,
                side="buy" if i % 2 == 0 else "sell",
                reason=reasons[i],
            )
        )
    # One entry really consists of two fills; keep both in trade detail and chart.
    first = rows.pop(0)
    for i in range(2):
        rows.insert(
            i,
            dict(
                first,
                fill_id=first["fill_id"] + f"-{i}",
                quantity=0.01,
                fee=first["fee"] / 2,
                time=first["time"] + timedelta(seconds=i * 20),
            ),
        )
    rows.extend(dict(row, time=datetime.fromisoformat(row["time"])) for row in state["extra"])
    return rows


def apply_execution_demo(payload: dict, strategy_id: str, path: Path | None = None) -> dict:
    """Overlay only a successfully authorized, selected strategy dashboard."""
    if payload.get("selected_strategy", {}).get("strategy_id") != strategy_id:
        return payload
    from apps.api.dto.ui_strategies_dashboard import StrategyDashboardResponse

    state = demo_state(path)
    rows = demo_fills(strategy_id, state)
    now = datetime.now(UTC)
    operations = project_fills(rows, initial_cash=10000)
    operations.source = "synthetic_demo"
    operations.current_price = 66350
    operations.price_at = now
    operations.stop_loss = 65000
    operations.take_profit = 67900
    candles: list[dict[str, Any]] = []
    start = rows[0]["time"] - timedelta(minutes=60)
    for i in range(max(1, int((now-start).total_seconds() // 900))):
        ts = start + timedelta(minutes=15 * i)
        anchors = sorted(
            [(r["time"].timestamp(), r["price"]) for r in rows] + [(now.timestamp(), 66350)]
        )
        before = [a for a in anchors if a[0] <= ts.timestamp()]
        after = [a for a in anchors if a[0] > ts.timestamp()]
        left = before[-1] if before else anchors[0]
        right = after[0] if after else anchors[-1]
        fraction = (ts.timestamp() - left[0]) / (right[0] - left[0]) if left[0] != right[0] else 0
        close = left[1] + (right[1] - left[1]) * max(0, min(1, fraction)) + math.sin(i * 1.7) * 80
        opening = candles[-1]["close"] if candles else close - 80
        candles.append(
            dict(
                timestamp=ts.isoformat(),
                open=opening,
                close=close,
                high=max(opening, close) + 65,
                low=min(opening, close) - 70,
            )
        )
    active = next((t for t in operations.trades if t.remaining_quantity), None)
    quantity = active.remaining_quantity if active else 0
    realized = sum(t.gross_pnl or 0 for t in operations.trades)
    fees = sum(t.fees or 0 for t in operations.trades)
    unrealized = (66350 - active.entry) * quantity if active else 0
    payload["operations"] = operations.model_dump(mode="json")
    payload["chart"].update(source="synthetic_demo", state="ready", candles=candles, markers=[])
    payload["paper_accounting"].update(
        source="synthetic_demo",
        state="ready",
        reserved_budget="10000",
        position_quantity=str(quantity),
        average_entry_price=str(active.entry) if active else None,
        equity=str(10000 + realized + unrealized - fees),
        realized_pnl=str(realized),
        unrealized_pnl=str(unrealized),
        fee_total=str(fees),
        funding_total="0",
        fee_model="synthetic_demo_0.075pct",
        funding_model="spot_zero",
        pnl_complete=True,
        completeness_reason="synthetic_demo",
        updated_at=now.isoformat(),
        degradation_reason=None,
    )
    run_id = str(uuid5(NAMESPACE_URL, f"demo:{strategy_id}"))
    payload["signal_journal"].update(
        source="synthetic_demo", state="ready", items=[], degradation_reason=None
    )
    payload["runtime_status"].update(
        source="synthetic_demo",
        state="ready",
        environment="paper",
        producer_status="running" if state["running"] else "stopped",
        producer_reason="synthetic_demo",
        mainnet_available=False,
        run_id=run_id,
        run_state="running" if state["running"] else "stopped",
        run_started_at=rows[0]["time"].isoformat(),
        run_updated_at=now.isoformat(),
    )
    payload["selected_strategy"]["actions"].update(
        can_run=not state["running"],
        can_stop=state["running"],
        can_delete=not state["running"] and not quantity,
    )
    payload["generated_at"] = now.isoformat()
    return StrategyDashboardResponse.model_validate(payload).model_dump(mode="json")


def simulate_command(strategy_id: str, command: str, body: dict, path: Path) -> tuple[dict, int]:
    """Called only after real authentication/ownership check in disposable QA app."""
    state = demo_state(path)
    if body.get("expected_run_id") not in {None, str(uuid5(NAMESPACE_URL, f"demo:{strategy_id}"))}:
        return {"detail": "Strategy run changed"}, 409
    now = datetime.now(UTC).isoformat()
    reply: dict[str, Any]
    if command in {"run", "stop", "restart"}:
        state["running"] = command != "stop"
        reply = dict(
            run_id=str(uuid5(NAMESPACE_URL, f"demo:{strategy_id}")),
            state="running" if state["running"] else "stopped",
        )
    else:
        key = body.get("client_request_id")
        if not isinstance(key, str) or len(key) > 80:
            return {"detail": "Invalid command identity"}, 422
        if key in state["commands"]:
            return dict(state["commands"][key], duplicate=True), 200
        qty = sum(
            r["quantity"] * (1 if r["side"] == "buy" else -1)
            for r in demo_fills(strategy_id, state)
        )
        if (
            not state["running"]
            or (command == "manual-entry" and abs(qty) > 1e-8)
            or (command == "manual-exit" and abs(qty) < 1e-8)
        ):
            return {"detail": "Demo state does not permit this command"}, 409
        amount = float(body.get("quote_notional", 1000))
        if not math.isfinite(amount) or amount <= 0:
            return {"detail": "Invalid amount"}, 422
        quantity = amount / 66350 if command == "manual-entry" else abs(qty)
        state["extra"].append(
            dict(
                fill_id=str(uuid5(NAMESPACE_URL, key)),
                time=now,
                price=66350,
                quantity=quantity,
                fee=quantity * 66350 * 0.00075,
                side="buy" if command == "manual-entry" else "sell",
                reason="manual",
            )
        )
        reply = dict(
            status="accepted",
            duplicate=False,
            intent_id=str(uuid5(NAMESPACE_URL, f"intent:{key}")),
            outcome_reason="synthetic_demo",
            paper_order_state="filled",
        )
        state["commands"][key] = reply
    path.write_text(json.dumps(state))
    return reply, 200
