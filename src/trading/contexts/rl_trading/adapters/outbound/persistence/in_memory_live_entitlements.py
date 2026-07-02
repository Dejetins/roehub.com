from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from trading.contexts.rl_trading.domain.live_entitlements import (
    RlLiveTickerEntitlementSnapshot,
    RlLiveTickerIdentity,
    RlLiveTickerMode,
    evaluate_rl_live_ticker_entitlement,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class _ActiveProfileTicker:
    strategy_id: UUID
    live_profile_id: UUID
    ticker: RlLiveTickerIdentity
    activated_at: datetime
    updated_at: datetime


class InMemoryRlLiveTickerEntitlementRepository:
    def __init__(self) -> None:
        self._active_by_profile: dict[tuple[str, UUID], _ActiveProfileTicker] = {}
        self._overrides: dict[str, int] = {}

    def set_override(self, *, owner_user_id: UserId, live_slots_allowed: int) -> None:
        if live_slots_allowed < 0:
            raise ValueError("live_slots_allowed must be >= 0")
        self._overrides[str(owner_user_id)] = live_slots_allowed

    def snapshot(
        self,
        *,
        owner_user_id: UserId,
        paid_level: str,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None = None,
    ) -> RlLiveTickerEntitlementSnapshot:
        return evaluate_rl_live_ticker_entitlement(
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
            active_tickers=self._active_tickers(owner_user_id=owner_user_id),
            override_live_slots_allowed=self._overrides.get(str(owner_user_id)),
        )

    def sync_profile(
        self,
        *,
        owner_user_id: UserId,
        paid_level: str,
        strategy_id: UUID,
        live_profile_id: UUID,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None,
        profile_ready: bool,
        observed_at: datetime,
    ) -> RlLiveTickerEntitlementSnapshot:
        profile_key = (str(owner_user_id), strategy_id)
        if mode != "live" or not profile_ready or requested_ticker is None:
            self._active_by_profile.pop(profile_key, None)
            return self.snapshot(
                owner_user_id=owner_user_id,
                paid_level=paid_level,
                mode=mode,
                requested_ticker=requested_ticker,
            )

        previous = self._active_by_profile.pop(profile_key, None)
        snapshot = self.snapshot(
            owner_user_id=owner_user_id,
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
        )
        if not snapshot.eligible:
            if previous is not None:
                self._active_by_profile[profile_key] = previous
            return snapshot

        activated_at = previous.activated_at if previous is not None else observed_at
        self._active_by_profile[profile_key] = _ActiveProfileTicker(
            strategy_id=strategy_id,
            live_profile_id=live_profile_id,
            ticker=requested_ticker,
            activated_at=activated_at,
            updated_at=observed_at,
        )
        return self.snapshot(
            owner_user_id=owner_user_id,
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
        )

    def _active_tickers(self, *, owner_user_id: UserId) -> tuple[RlLiveTickerIdentity, ...]:
        owner = str(owner_user_id)
        return tuple(
            item.ticker
            for key, item in self._active_by_profile.items()
            if key[0] == owner
        )
