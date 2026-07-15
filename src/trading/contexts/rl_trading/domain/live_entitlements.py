from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

RlLiveTickerMode = Literal["monitor_only", "paper", "testnet", "live"]
RlLiveEntitlementSource = Literal["paid_level", "override", "fail_closed"]

RL_LIVE_TICKER_READY = "rl_live_ticker_entitlement_ready"
RL_LIVE_TICKER_NOT_COUNTED = "rl_live_ticker_not_counted_for_mode"
RL_LIVE_TICKER_BASE_FAIL_CLOSED = "rl_live_ticker_paid_level_base_fail_closed"
RL_LIVE_TICKER_UNKNOWN_FAIL_CLOSED = "rl_live_ticker_paid_level_unknown_fail_closed"
RL_LIVE_TICKER_QUOTA_EXCEEDED = "rl_live_ticker_quota_exceeded"

_PAID_LEVEL_LIMITS = {
    "free": ("Free", 1),
    "pro": ("Pro", 5),
    "ultra": ("Premium", 20),
    "base": ("internal/base", 0),
}


@dataclass(frozen=True, slots=True)
class RlLiveTickerIdentity:
    organization_id: OrganizationId
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        exchange_name = self.exchange_name.strip().lower()
        market_type = self.market_type.strip().lower()
        symbol = self.symbol.strip().upper()
        if not exchange_name:
            raise ValueError("RlLiveTickerIdentity.exchange_name must be non-empty")
        if market_type not in {"spot", "futures"}:
            raise ValueError("RlLiveTickerIdentity.market_type must be spot or futures")
        if not symbol:
            raise ValueError("RlLiveTickerIdentity.symbol must be non-empty")
        object.__setattr__(self, "exchange_name", exchange_name)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)

    @property
    def distinct_key(self) -> tuple[str, str, str, str, str]:
        return (
            str(self.organization_id),
            str(self.owner_user_id),
            self.exchange_name,
            self.market_type,
            self.symbol,
        )


@dataclass(frozen=True, slots=True)
class RlLiveTickerLimit:
    paid_level: str
    product_label: str
    live_slots_allowed: int
    entitlement_source: RlLiveEntitlementSource
    fail_closed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RlLiveTickerEntitlementSnapshot:
    paid_level: str
    product_label: str
    entitlement_source: RlLiveEntitlementSource
    live_slots_allowed: int
    live_slots_used: int
    mode: RlLiveTickerMode
    eligible: bool
    readiness_reason: str
    requested_ticker: RlLiveTickerIdentity | None
    active_tickers: tuple[RlLiveTickerIdentity, ...]


class RlLiveTickerEntitlementRepository(Protocol):
    def snapshot(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        paid_level: str,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None = None,
    ) -> RlLiveTickerEntitlementSnapshot: ...

    def sync_profile(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        paid_level: str,
        strategy_id: UUID,
        live_profile_id: UUID,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None,
        profile_ready: bool,
        observed_at: datetime,
    ) -> RlLiveTickerEntitlementSnapshot: ...


class RlLiveTickerEntitlementService:
    def __init__(self, *, repository: RlLiveTickerEntitlementRepository) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RlLiveTickerEntitlementService requires repository")
        self._repository = repository

    def snapshot(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        paid_level: str,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None = None,
    ) -> RlLiveTickerEntitlementSnapshot:
        return self._repository.snapshot(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            paid_level=paid_level,
            mode=mode,
            requested_ticker=requested_ticker,
        )

    def sync_profile(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        paid_level: str,
        strategy_id: UUID,
        live_profile_id: UUID,
        mode: RlLiveTickerMode,
        requested_ticker: RlLiveTickerIdentity | None,
        profile_ready: bool,
        observed_at: datetime,
    ) -> RlLiveTickerEntitlementSnapshot:
        return self._repository.sync_profile(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            paid_level=paid_level,
            strategy_id=strategy_id,
            live_profile_id=live_profile_id,
            mode=mode,
            requested_ticker=requested_ticker,
            profile_ready=profile_ready,
            observed_at=observed_at,
        )


def resolve_rl_live_ticker_limit(
    *,
    paid_level: str,
    override_live_slots_allowed: int | None = None,
) -> RlLiveTickerLimit:
    normalized_paid_level = paid_level.strip().lower()
    if override_live_slots_allowed is not None:
        if override_live_slots_allowed < 0:
            raise ValueError("override_live_slots_allowed must be >= 0")
        return RlLiveTickerLimit(
            paid_level=normalized_paid_level,
            product_label="Enterprise",
            live_slots_allowed=override_live_slots_allowed,
            entitlement_source="override",
            fail_closed_reason=None,
        )

    if normalized_paid_level == "base":
        return RlLiveTickerLimit(
            paid_level=normalized_paid_level,
            product_label="internal/base",
            live_slots_allowed=0,
            entitlement_source="fail_closed",
            fail_closed_reason=RL_LIVE_TICKER_BASE_FAIL_CLOSED,
        )

    configured = _PAID_LEVEL_LIMITS.get(normalized_paid_level)
    if configured is None:
        return RlLiveTickerLimit(
            paid_level=normalized_paid_level or "unknown",
            product_label="unknown",
            live_slots_allowed=0,
            entitlement_source="fail_closed",
            fail_closed_reason=RL_LIVE_TICKER_UNKNOWN_FAIL_CLOSED,
        )
    product_label, live_slots_allowed = configured
    return RlLiveTickerLimit(
        paid_level=normalized_paid_level,
        product_label=product_label,
        live_slots_allowed=live_slots_allowed,
        entitlement_source="paid_level",
        fail_closed_reason=None,
    )


def evaluate_rl_live_ticker_entitlement(
    *,
    paid_level: str,
    mode: RlLiveTickerMode,
    active_tickers: tuple[RlLiveTickerIdentity, ...],
    requested_ticker: RlLiveTickerIdentity | None = None,
    override_live_slots_allowed: int | None = None,
) -> RlLiveTickerEntitlementSnapshot:
    limit = resolve_rl_live_ticker_limit(
        paid_level=paid_level,
        override_live_slots_allowed=override_live_slots_allowed,
    )
    distinct_active = _distinct_active_tickers(active_tickers=active_tickers)
    live_slots_used = len(distinct_active)

    if mode != "live":
        return _snapshot(
            limit=limit,
            mode=mode,
            eligible=True,
            readiness_reason=RL_LIVE_TICKER_NOT_COUNTED,
            requested_ticker=requested_ticker,
            active_tickers=distinct_active,
        )

    if requested_ticker is not None and requested_ticker.distinct_key in {
        item.distinct_key for item in distinct_active
    }:
        return _snapshot(
            limit=limit,
            mode=mode,
            eligible=True,
            readiness_reason=RL_LIVE_TICKER_READY,
            requested_ticker=requested_ticker,
            active_tickers=distinct_active,
        )

    if limit.live_slots_allowed <= 0:
        return _snapshot(
            limit=limit,
            mode=mode,
            eligible=False,
            readiness_reason=limit.fail_closed_reason or RL_LIVE_TICKER_QUOTA_EXCEEDED,
            requested_ticker=requested_ticker,
            active_tickers=distinct_active,
        )

    if live_slots_used >= limit.live_slots_allowed:
        return _snapshot(
            limit=limit,
            mode=mode,
            eligible=False,
            readiness_reason=RL_LIVE_TICKER_QUOTA_EXCEEDED,
            requested_ticker=requested_ticker,
            active_tickers=distinct_active,
        )

    return _snapshot(
        limit=limit,
        mode=mode,
        eligible=True,
        readiness_reason=RL_LIVE_TICKER_READY,
        requested_ticker=requested_ticker,
        active_tickers=distinct_active,
    )


def _snapshot(
    *,
    limit: RlLiveTickerLimit,
    mode: RlLiveTickerMode,
    eligible: bool,
    readiness_reason: str,
    requested_ticker: RlLiveTickerIdentity | None,
    active_tickers: tuple[RlLiveTickerIdentity, ...],
) -> RlLiveTickerEntitlementSnapshot:
    return RlLiveTickerEntitlementSnapshot(
        paid_level=limit.paid_level,
        product_label=limit.product_label,
        entitlement_source=limit.entitlement_source,
        live_slots_allowed=limit.live_slots_allowed,
        live_slots_used=len(active_tickers),
        mode=mode,
        eligible=eligible,
        readiness_reason=readiness_reason,
        requested_ticker=requested_ticker,
        active_tickers=active_tickers,
    )


def _distinct_active_tickers(
    *, active_tickers: tuple[RlLiveTickerIdentity, ...]
) -> tuple[RlLiveTickerIdentity, ...]:
    by_key = {item.distinct_key: item for item in active_tickers}
    return tuple(by_key[key] for key in sorted(by_key))
