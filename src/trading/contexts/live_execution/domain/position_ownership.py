from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from trading.shared_kernel.primitives import UserId

StrategyPositionOwnershipState = Literal[
    "reserved",
    "active",
    "releasing",
    "released",
    "stale_requires_repair",
]

BLOCKING_POSITION_OWNERSHIP_STATES: frozenset[str] = frozenset(
    {"reserved", "active", "releasing", "stale_requires_repair"}
)


class StrategyPositionOwnershipConflictError(ValueError):
    def __init__(self, *, existing: StrategyPositionOwnership) -> None:
        super().__init__("position ownership is already held")
        self.existing = existing


class StrategyPositionOwnershipStorageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StrategyPositionOwnership:
    ownership_id: UUID
    owner_user_id: UserId
    exchange_connection_id: UUID
    strategy_id: UUID
    live_profile_id: UUID | None
    strategy_run_id: UUID
    market_type: str
    instrument_key: str
    position_mode: str
    state: StrategyPositionOwnershipState
    acquired_at: datetime
    released_at: datetime | None
    expires_at: datetime | None
    reason: str

    @property
    def blocks_scope(self) -> bool:
        return self.state in BLOCKING_POSITION_OWNERSHIP_STATES

    def with_state(
        self,
        *,
        state: StrategyPositionOwnershipState,
        reason: str,
        changed_at: datetime,
    ) -> StrategyPositionOwnership:
        return StrategyPositionOwnership(
            ownership_id=self.ownership_id,
            owner_user_id=self.owner_user_id,
            exchange_connection_id=self.exchange_connection_id,
            strategy_id=self.strategy_id,
            live_profile_id=self.live_profile_id,
            strategy_run_id=self.strategy_run_id,
            market_type=self.market_type,
            instrument_key=self.instrument_key,
            position_mode=self.position_mode,
            state=state,
            acquired_at=self.acquired_at,
            released_at=changed_at if state == "released" else self.released_at,
            expires_at=self.expires_at,
            reason=reason,
        )
