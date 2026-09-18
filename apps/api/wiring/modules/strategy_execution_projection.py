"""Pure fill-to-position-cycle projection. No signals are treated as executions."""

from decimal import Decimal

from apps.api.dto.strategy_operations import (
    OperationFill,
    OperationPoint,
    OperationTrade,
    StrategyOperationsResponse,
)

D = Decimal


def project_fills(rows, *, initial_cash=None, partial=False):
    result = StrategyOperationsResponse(
        state="ready" if rows else "empty", partial=partial, initial_cash=initial_cash
    )
    active = None
    net = D(0)
    average = D(0)
    gross = D(0)
    fees = D(0)
    entry_value = D(0)
    entry_quantity = D(0)
    exit_value = D(0)
    exit_quantity = D(0)
    known_fees = True
    complete_costs = True
    equity = D(str(initial_cash)) if initial_cash is not None else None
    if equity is not None and equity <= 0:
        raise ValueError("Initial capital must be positive")
    peak = equity
    for row in rows:
        qty = D(str(row["quantity"]))
        price = D(str(row["price"]))
        if qty <= 0 or price <= 0:
            raise ValueError("Invalid execution quantity/price")
        signed = qty if row["side"] == "buy" else -qty
        original = qty
        fee = D(str(row["fee"])) if row.get("fee") is not None else None
        piece_index = 0
        while qty > 0:
            entering = net == 0 or (net > 0) == (signed > 0)
            piece = qty if entering else min(abs(net), qty)
            allocated = fee * piece / original if fee is not None else None
            reason = row.get("reason") or "unknown_reason"
            fill = OperationFill(
                fill_id=f'{row["fill_id"]}:{piece_index}',
                time=row["time"],
                price=float(price),
                reference_price=row.get("reference_price"),
                quantity=float(piece),
                fee=float(allocated) if allocated is not None else None,
                action="entry" if entering else "exit",
                reason=reason,
                side=row["side"],
            )
            if net == 0:
                active = OperationTrade(
                    trade_id=str(row["fill_id"]),
                    side="long" if signed > 0 else "short",
                    entry_time=row["time"],
                    entry=float(price),
                    quantity=0,
                    remaining_quantity=0,
                    entry_reason=reason,
                    fills=[],
                )
                result.trades.append(active)
                average, gross, fees, exit_value, exit_quantity = D(0), D(0), D(0), D(0), D(0)
                entry_value, entry_quantity = D(0), D(0)
                complete_costs = True
                known_fees = True
            assert active is not None
            active.fills.append(fill)
            if row.get("costs_complete") is False:
                complete_costs = False
            if allocated is None:
                known_fees = False
                complete_costs = False
            else:
                fees += allocated
            if entering:
                average = (abs(net) * average + piece * price) / (abs(net) + piece)
                entry_value += piece * price
                entry_quantity += piece
                active.quantity = float(entry_quantity)
                active.entry = float(entry_value / entry_quantity)
            else:
                gross += (price - average) * piece * (1 if net > 0 else -1)
                exit_value += price * piece
                exit_quantity += piece
                active.exit = float(exit_value / exit_quantity)
                active.exit_reason = reason
            net += piece * (1 if signed > 0 else -1)
            active.remaining_quantity = float(abs(net))
            active.gross_pnl = float(gross)
            active.fees = float(fees) if known_fees else None
            active.net_pnl = float(gross - fees) if complete_costs else None
            if net == 0:
                active.exit_time = row["time"]
                if equity is not None and complete_costs:
                    equity += gross - fees
                    assert peak is not None
                    peak = max(peak, equity)
                    result.equity.append(OperationPoint(timestamp=row["time"], value=float(equity)))
                    result.drawdown.append(
                        OperationPoint(
                            timestamp=row["time"], value=float((equity / peak - 1) * 100)
                        )
                    )
                else:
                    equity = None
            qty -= piece
            piece_index += 1
    result.trades.reverse()
    return result
