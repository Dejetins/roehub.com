from __future__ import annotations

from datetime import datetime
from typing import Callable
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import (
    StrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.domain import (
    StrategyPositionOwnership,
    StrategyPositionOwnershipConflictError,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class StrategyPositionOwnershipService:
    def __init__(
        self,
        *,
        repository: StrategyPositionOwnershipRepository,
        on_transition: Callable[[str, str], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyPositionOwnershipService requires repository")
        self._repository = repository
        self._on_transition = on_transition

    def reserve_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        market_type: str,
        instrument_key: str,
        position_mode: str,
        now: datetime,
        reason: str = "run_started",
    ) -> StrategyPositionOwnership:
        ownership = StrategyPositionOwnership(
            ownership_id=uuid4(),
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            strategy_id=strategy_id,
            live_profile_id=live_profile_id,
            strategy_run_id=strategy_run_id,
            market_type=market_type,
            instrument_key=instrument_key,
            position_mode=position_mode,
            state="reserved",
            acquired_at=now,
            released_at=None,
            expires_at=None,
            reason=reason,
        )
        try:
            reserved = self._repository.reserve(ownership=ownership)
        except StrategyPositionOwnershipConflictError:
            self._record(result="conflict", reason="position_ownership_conflict")
            raise
        self._record(result="reserved", reason=reason)
        return reserved

    def activate_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
    ) -> StrategyPositionOwnership | None:
        updated = self._repository.update_state(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            state="active",
            reason="run_started",
            changed_at=now,
        )
        if updated is not None:
            self._record(result="active", reason="run_started")
        return updated

    def mark_releasing_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str = "run_stopping",
    ) -> StrategyPositionOwnership | None:
        updated = self._repository.update_state(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            state="releasing",
            reason=reason,
            changed_at=now,
        )
        if updated is not None:
            self._record(result="releasing", reason=reason)
        return updated

    def release_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str = "run_stopped",
    ) -> StrategyPositionOwnership | None:
        updated = self._repository.update_state(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            state="released",
            reason=reason,
            changed_at=now,
        )
        if updated is not None:
            self._record(result="released", reason=reason)
        return updated

    def mark_stale_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str = "manual_repair_required",
    ) -> StrategyPositionOwnership | None:
        updated = self._repository.update_state(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            state="stale_requires_repair",
            reason=reason,
            changed_at=now,
        )
        if updated is not None:
            self._record(result="stale_requires_repair", reason=reason)
        return updated

    def _record(self, *, result: str, reason: str) -> None:
        if self._on_transition is not None:
            self._on_transition(result, reason)
