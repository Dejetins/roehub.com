from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from trading.contexts.live_execution.application.ports import PaperAccountingRepository
from trading.contexts.live_execution.domain import (
    CapitalReservation,
    PaperFill,
    PaperOrder,
    StrategyPaperAccountingSnapshot,
)
from trading.shared_kernel.primitives import UserId


class InMemoryPaperAccountingRepository(PaperAccountingRepository):
    def __init__(self) -> None:
        self.reservations: list[CapitalReservation] = []
        self.orders: list[PaperOrder] = []
        self.fills: list[PaperFill] = []
        self.accounting: list[StrategyPaperAccountingSnapshot] = []

    def record_reservation(self, *, reservation: CapitalReservation) -> CapitalReservation:
        self.reservations.append(reservation)
        return reservation

    def release_reservation_for_run(
        self,
        *,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        changed_at: datetime,
        reason: str,
    ) -> CapitalReservation | None:
        for index, item in enumerate(self.reservations):
            if (
                item.owner_user_id == owner_user_id
                and item.strategy_run_id == strategy_run_id
                and item.state == "reserved"
            ):
                updated = CapitalReservation(
                    reservation_id=item.reservation_id,
                    owner_user_id=item.owner_user_id,
                    exchange_connection_id=item.exchange_connection_id,
                    strategy_id=item.strategy_id,
                    live_profile_id=item.live_profile_id,
                    strategy_run_id=item.strategy_run_id,
                    asset=item.asset,
                    requested_amount=item.requested_amount,
                    reserved_amount=item.reserved_amount,
                    state="released",
                    source_account_snapshot_id=item.source_account_snapshot_id,
                    acquired_at=item.acquired_at,
                    released_at=changed_at,
                    reason=reason,
                    fee_model=item.fee_model,
                    funding_model=item.funding_model,
                    pnl_complete=item.pnl_complete,
                )
                self.reservations[index] = updated
                return updated
        return None

    def get_active_reservation_for_run(
        self, *, owner_user_id: UserId, strategy_run_id: UUID
    ) -> CapitalReservation | None:
        matches = [
            item
            for item in self.reservations
            if item.owner_user_id == owner_user_id
            and item.strategy_run_id == strategy_run_id
            and item.state == "reserved"
        ]
        return matches[-1] if matches else None

    def sum_active_reserved(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID, asset: str
    ) -> Decimal:
        return sum(
            (
                item.reserved_amount
                for item in self.reservations
                if item.owner_user_id == owner_user_id
                and item.exchange_connection_id == exchange_connection_id
                and item.asset == asset
                and item.state == "reserved"
            ),
            Decimal("0"),
        )

    def record_paper_execution(
        self,
        *,
        order: PaperOrder,
        fill: PaperFill,
        accounting: StrategyPaperAccountingSnapshot,
    ) -> StrategyPaperAccountingSnapshot:
        existing = next(
            (
                item
                for item in self.accounting
                if item.paper_fill_id == accounting.paper_fill_id
            ),
            None,
        )
        if existing is not None:
            return existing
        existing_order = next(
            (
                item
                for item in self.orders
                if item.source_event_id is not None
                and item.source_event_id == order.source_event_id
            ),
            None,
        )
        if existing_order is not None:
            existing_accounting = next(
                (
                    item
                    for item in self.accounting
                    if item.paper_fill_id == accounting.paper_fill_id
                ),
                None,
            )
            if existing_accounting is not None:
                return existing_accounting
        self.orders.append(order)
        self.fills.append(fill)
        self.accounting.append(accounting)
        return accounting

    def get_latest_accounting_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> StrategyPaperAccountingSnapshot | None:
        matches = [
            item
            for item in self.accounting
            if item.owner_user_id == owner_user_id and item.strategy_id == strategy_id
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.created_at, str(item.accounting_id)))
