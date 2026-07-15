from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import PaperAccountingRepository
from trading.contexts.live_execution.domain import (
    CapitalReservation,
    PaperFill,
    PaperOrder,
    StrategyPaperAccountingSnapshot,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresPaperAccountingRepository(PaperAccountingRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresPaperAccountingRepository requires gateway")
        self._gateway = gateway

    def record_reservation(self, *, reservation: CapitalReservation) -> CapitalReservation:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO strategy_capital_reservations
            (
                reservation_id, organization_id, owner_user_id, exchange_connection_id, strategy_id,
                live_profile_id, strategy_run_id, asset, requested_amount,
                reserved_amount, state, source_account_snapshot_id, acquired_at,
                released_at, reason, fee_model, funding_model, pnl_complete
            )
            VALUES
            (
                %(reservation_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s,
                %(strategy_id)s, %(live_profile_id)s, %(strategy_run_id)s,
                %(asset)s, %(requested_amount)s, %(reserved_amount)s, %(state)s,
                %(source_account_snapshot_id)s, %(acquired_at)s, %(released_at)s,
                %(reason)s, %(fee_model)s, %(funding_model)s, %(pnl_complete)s
            )
            ON CONFLICT (reservation_id) DO NOTHING
            RETURNING *
            """,
            parameters=_reservation_params(reservation),
        )
        if row is None:
            row = self._gateway.fetch_one(
                query="""
                SELECT *
                FROM strategy_capital_reservations
                WHERE organization_id = %(organization_id)s
                  AND reservation_id = %(reservation_id)s
                """,
                parameters={
                    "organization_id": str(reservation.organization_id),
                    "reservation_id": str(reservation.reservation_id),
                },
            )
        if row is None:
            raise ValueError("reservation insert returned no row")
        return _map_reservation(row)

    def release_reservation_for_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        changed_at: datetime,
        reason: str,
    ) -> CapitalReservation | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE strategy_capital_reservations
            SET state = 'released',
                released_at = %(changed_at)s,
                reason = %(reason)s
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND strategy_run_id = %(strategy_run_id)s
              AND state = 'reserved'
            RETURNING *
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "strategy_run_id": str(strategy_run_id),
                "changed_at": changed_at,
                "reason": reason,
            },
        )
        return _map_reservation(row) if row is not None else None

    def get_active_reservation_for_run(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, strategy_run_id: UUID
    ) -> CapitalReservation | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM strategy_capital_reservations
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND strategy_run_id = %(strategy_run_id)s
              AND state = 'reserved'
            ORDER BY acquired_at DESC, reservation_id DESC
            LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "strategy_run_id": str(strategy_run_id),
            },
        )
        return _map_reservation(row) if row is not None else None

    def sum_active_reserved(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        asset: str,
    ) -> Decimal:
        row = self._gateway.fetch_one(
            query="""
            SELECT COALESCE(SUM(reserved_amount), 0) AS amount
            FROM strategy_capital_reservations
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND exchange_connection_id = %(exchange_connection_id)s
              AND asset = %(asset)s
              AND state = 'reserved'
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "exchange_connection_id": str(exchange_connection_id),
                "asset": asset,
            },
        )
        return Decimal(str(row["amount"])) if row is not None else Decimal("0")

    def record_paper_execution(
        self,
        *,
        order: PaperOrder,
        fill: PaperFill,
        accounting: StrategyPaperAccountingSnapshot,
    ) -> StrategyPaperAccountingSnapshot:
        self._gateway.execute(
            query="""
            INSERT INTO paper_orders
            (
                paper_order_id, organization_id, owner_user_id, strategy_id, strategy_run_id,
                reservation_id, source_signal_id, source_event_id, instrument_key,
                market_type, side, order_type, quantity, quote_notional,
                reference_price, status, reason, created_at
            )
            VALUES
            (
                %(paper_order_id)s, %(organization_id)s, %(owner_user_id)s, %(strategy_id)s,
                %(strategy_run_id)s, %(reservation_id)s, %(source_signal_id)s,
                %(source_event_id)s, %(instrument_key)s, %(market_type)s,
                %(side)s, %(order_type)s, %(quantity)s, %(quote_notional)s,
                %(reference_price)s, %(status)s, %(reason)s, %(created_at)s
            )
            ON CONFLICT (organization_id, source_signal_id) DO NOTHING
            """,
            parameters=_order_params(order),
        )
        self._gateway.execute(
            query="""
            INSERT INTO paper_fills
            (
                paper_fill_id, paper_order_id, organization_id, owner_user_id, strategy_id,
                strategy_run_id, instrument_key, side, quantity, fill_price,
                quote_notional, fee_amount, fee_asset, funding_amount,
                funding_asset, filled_at
            )
            VALUES
            (
                %(paper_fill_id)s, %(paper_order_id)s, %(organization_id)s, %(owner_user_id)s,
                %(strategy_id)s, %(strategy_run_id)s, %(instrument_key)s,
                %(side)s, %(quantity)s, %(fill_price)s, %(quote_notional)s,
                %(fee_amount)s, %(fee_asset)s, %(funding_amount)s,
                %(funding_asset)s, %(filled_at)s
            )
            ON CONFLICT (organization_id, paper_fill_id) DO NOTHING
            """,
            parameters=_fill_params(fill),
        )
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO strategy_paper_accounting
            (
                accounting_id, organization_id, owner_user_id, strategy_id, strategy_run_id,
                reservation_id, paper_fill_id, instrument_key, market_type,
                position_quantity, average_entry_price, reserved_budget,
                cash_balance, equity, realized_pnl, unrealized_pnl, fee_total,
                funding_total, fee_model, funding_model, pnl_complete,
                completeness_reason, created_at
            )
            VALUES
            (
                %(accounting_id)s, %(organization_id)s, %(owner_user_id)s, %(strategy_id)s,
                %(strategy_run_id)s, %(reservation_id)s, %(paper_fill_id)s,
                %(instrument_key)s, %(market_type)s, %(position_quantity)s,
                %(average_entry_price)s, %(reserved_budget)s, %(cash_balance)s,
                %(equity)s, %(realized_pnl)s, %(unrealized_pnl)s, %(fee_total)s,
                %(funding_total)s, %(fee_model)s, %(funding_model)s,
                %(pnl_complete)s, %(completeness_reason)s, %(created_at)s
            )
            ON CONFLICT (organization_id, paper_fill_id) DO NOTHING
            RETURNING *
            """,
            parameters=_accounting_params(accounting),
        )
        if row is None:
            row = self._gateway.fetch_one(
                query="""
                SELECT *
                FROM strategy_paper_accounting
                WHERE organization_id = %(organization_id)s
                  AND paper_fill_id = %(paper_fill_id)s
                """,
                parameters={
                    "organization_id": str(accounting.organization_id),
                    "paper_fill_id": str(accounting.paper_fill_id),
                },
            )
        if row is None:
            raise ValueError("paper accounting insert returned no row")
        return _map_accounting(row)

    def get_latest_accounting_for_strategy(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, strategy_id: UUID
    ) -> StrategyPaperAccountingSnapshot | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM strategy_paper_accounting
            WHERE organization_id = %(organization_id)s
              AND owner_user_id = %(owner_user_id)s
              AND strategy_id = %(strategy_id)s
            ORDER BY created_at DESC, accounting_id DESC
            LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return _map_accounting(row) if row is not None else None


def _reservation_params(item: CapitalReservation) -> dict[str, object]:
    return {
        "reservation_id": str(item.reservation_id),
        "organization_id": str(item.organization_id),
        "owner_user_id": str(item.owner_user_id),
        "exchange_connection_id": str(item.exchange_connection_id),
        "strategy_id": str(item.strategy_id),
        "live_profile_id": str(item.live_profile_id) if item.live_profile_id else None,
        "strategy_run_id": str(item.strategy_run_id),
        "asset": item.asset,
        "requested_amount": item.requested_amount,
        "reserved_amount": item.reserved_amount,
        "state": item.state,
        "source_account_snapshot_id": (
            str(item.source_account_snapshot_id) if item.source_account_snapshot_id else None
        ),
        "acquired_at": item.acquired_at,
        "released_at": item.released_at,
        "reason": item.reason,
        "fee_model": item.fee_model,
        "funding_model": item.funding_model,
        "pnl_complete": item.pnl_complete,
    }


def _order_params(item: PaperOrder) -> dict[str, object]:
    return {
        "paper_order_id": str(item.paper_order_id),
        "organization_id": str(item.organization_id),
        "owner_user_id": str(item.owner_user_id),
        "strategy_id": str(item.strategy_id),
        "strategy_run_id": str(item.strategy_run_id),
        "reservation_id": str(item.reservation_id),
        "source_signal_id": str(item.source_signal_id),
        "source_event_id": str(item.source_event_id) if item.source_event_id else None,
        "instrument_key": item.instrument_key,
        "market_type": item.market_type,
        "side": item.side,
        "order_type": item.order_type,
        "quantity": item.quantity,
        "quote_notional": item.quote_notional,
        "reference_price": item.reference_price,
        "status": item.status,
        "reason": item.reason,
        "created_at": item.created_at,
    }


def _fill_params(item: PaperFill) -> dict[str, object]:
    return {
        "paper_fill_id": str(item.paper_fill_id),
        "paper_order_id": str(item.paper_order_id),
        "organization_id": str(item.organization_id),
        "owner_user_id": str(item.owner_user_id),
        "strategy_id": str(item.strategy_id),
        "strategy_run_id": str(item.strategy_run_id),
        "instrument_key": item.instrument_key,
        "side": item.side,
        "quantity": item.quantity,
        "fill_price": item.fill_price,
        "quote_notional": item.quote_notional,
        "fee_amount": item.fee_amount,
        "fee_asset": item.fee_asset,
        "funding_amount": item.funding_amount,
        "funding_asset": item.funding_asset,
        "filled_at": item.filled_at,
    }


def _accounting_params(item: StrategyPaperAccountingSnapshot) -> dict[str, object]:
    return {
        "accounting_id": str(item.accounting_id),
        "organization_id": str(item.organization_id),
        "owner_user_id": str(item.owner_user_id),
        "strategy_id": str(item.strategy_id),
        "strategy_run_id": str(item.strategy_run_id),
        "reservation_id": str(item.reservation_id),
        "paper_fill_id": str(item.paper_fill_id),
        "instrument_key": item.instrument_key,
        "market_type": item.market_type,
        "position_quantity": item.position_quantity,
        "average_entry_price": item.average_entry_price,
        "reserved_budget": item.reserved_budget,
        "cash_balance": item.cash_balance,
        "equity": item.equity,
        "realized_pnl": item.realized_pnl,
        "unrealized_pnl": item.unrealized_pnl,
        "fee_total": item.fee_total,
        "funding_total": item.funding_total,
        "fee_model": item.fee_model,
        "funding_model": item.funding_model,
        "pnl_complete": item.pnl_complete,
        "completeness_reason": item.completeness_reason,
        "created_at": item.created_at,
    }


def _map_reservation(row: Mapping[str, Any]) -> CapitalReservation:
    return CapitalReservation(
        reservation_id=UUID(str(row["reservation_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        live_profile_id=UUID(str(row["live_profile_id"])) if row["live_profile_id"] else None,
        strategy_run_id=UUID(str(row["strategy_run_id"])),
        asset=str(row["asset"]),
        requested_amount=Decimal(str(row["requested_amount"])),
        reserved_amount=Decimal(str(row["reserved_amount"])),
        state=str(row["state"]),  # type: ignore[arg-type]
        source_account_snapshot_id=(
            UUID(str(row["source_account_snapshot_id"]))
            if row["source_account_snapshot_id"]
            else None
        ),
        acquired_at=_utc(row["acquired_at"]),
        released_at=_utc(row["released_at"]) if row["released_at"] else None,
        reason=str(row["reason"]),
        fee_model=str(row["fee_model"]),
        funding_model=str(row["funding_model"]),
        pnl_complete=bool(row["pnl_complete"]),
    )


def _map_accounting(row: Mapping[str, Any]) -> StrategyPaperAccountingSnapshot:
    return StrategyPaperAccountingSnapshot(
        accounting_id=UUID(str(row["accounting_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        strategy_id=UUID(str(row["strategy_id"])),
        strategy_run_id=UUID(str(row["strategy_run_id"])),
        reservation_id=UUID(str(row["reservation_id"])),
        paper_fill_id=UUID(str(row["paper_fill_id"])),
        instrument_key=str(row["instrument_key"]),
        market_type=str(row["market_type"]),
        position_quantity=Decimal(str(row["position_quantity"])),
        average_entry_price=(
            Decimal(str(row["average_entry_price"])) if row["average_entry_price"] else None
        ),
        reserved_budget=Decimal(str(row["reserved_budget"])),
        cash_balance=Decimal(str(row["cash_balance"])),
        equity=Decimal(str(row["equity"])),
        realized_pnl=Decimal(str(row["realized_pnl"])),
        unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
        fee_total=Decimal(str(row["fee_total"])),
        funding_total=Decimal(str(row["funding_total"])),
        fee_model=str(row["fee_model"]),
        funding_model=str(row["funding_model"]),
        pnl_complete=bool(row["pnl_complete"]),
        completeness_reason=str(row["completeness_reason"]),
        created_at=_utc(row["created_at"]),
    )


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("expected datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
