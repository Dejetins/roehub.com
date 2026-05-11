from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trading.shared_kernel.primitives import PaidLevel

from .dto import BacktestAiAdmissionDecision, BacktestAiQuotaSnapshot

DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS = 90
DEFAULT_AI_WEEKLY_QUOTA_RETRY_AFTER_SECONDS = 24 * 60 * 60
DEFAULT_AI_FIVE_HOUR_QUOTA_RETRY_AFTER_SECONDS = 5 * 60 * 60


@dataclass(frozen=True, slots=True)
class BacktestAiTierQuota:
    requests_per_5h: int
    requests_per_week: int
    max_queued_per_user: int
    max_active_user_jobs: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("requests_per_5h", self.requests_per_5h),
            ("requests_per_week", self.requests_per_week),
            ("max_queued_per_user", self.max_queued_per_user),
            ("max_active_user_jobs", self.max_active_user_jobs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


DEFAULT_AI_TIER_QUOTAS: Mapping[str, BacktestAiTierQuota] = MappingProxyType(
    {
        "free": BacktestAiTierQuota(
            requests_per_5h=3,
            requests_per_week=10,
            max_queued_per_user=1,
            max_active_user_jobs=1,
        ),
        "base": BacktestAiTierQuota(
            requests_per_5h=6,
            requests_per_week=25,
            max_queued_per_user=2,
            max_active_user_jobs=1,
        ),
        "pro": BacktestAiTierQuota(
            requests_per_5h=15,
            requests_per_week=75,
            max_queued_per_user=3,
            max_active_user_jobs=1,
        ),
        "ultra": BacktestAiTierQuota(
            requests_per_5h=40,
            requests_per_week=200,
            max_queued_per_user=5,
            max_active_user_jobs=1,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class BacktestAiQuotaConfig:
    tier_quotas: Mapping[str, BacktestAiTierQuota] = field(
        default_factory=lambda: DEFAULT_AI_TIER_QUOTAS
    )
    max_queue_size: int = 50
    estimated_wait_seconds: int = 8

    def __post_init__(self) -> None:
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if self.estimated_wait_seconds <= 0:
            raise ValueError("estimated_wait_seconds must be > 0")
        normalized: dict[str, BacktestAiTierQuota] = {}
        for raw_tier, quota in self.tier_quotas.items():
            tier = str(raw_tier).strip().lower()
            if not tier:
                raise ValueError("tier quota keys must be non-empty")
            normalized[tier] = quota
        for required in ("free", "base", "pro", "ultra"):
            if required not in normalized:
                raise ValueError(f"missing AI quota tier {required!r}")
        object.__setattr__(self, "tier_quotas", MappingProxyType(normalized))

    def quota_for(self, *, paid_level: PaidLevel) -> BacktestAiTierQuota:
        return self.tier_quotas[str(paid_level)]


@dataclass(frozen=True, slots=True)
class BacktestAiQuotaService:
    config: BacktestAiQuotaConfig = BacktestAiQuotaConfig()

    def evaluate(
        self,
        *,
        paid_level: PaidLevel,
        snapshot: BacktestAiQuotaSnapshot,
    ) -> BacktestAiAdmissionDecision:
        quota = self.config.quota_for(paid_level=paid_level)
        tier = str(paid_level)
        if snapshot.requests_5h >= quota.requests_per_5h:
            return BacktestAiAdmissionDecision(
                accepted=False,
                status="quota_exceeded",
                reason="requests_per_5h_exceeded",
                message=(
                    "AI configurator quota for the current 5 hour window is exhausted."
                ),
                retry_after_seconds=DEFAULT_AI_FIVE_HOUR_QUOTA_RETRY_AFTER_SECONDS,
                details={
                    "tier": tier,
                    "limit": quota.requests_per_5h,
                    "window": "5h",
                    "used": snapshot.requests_5h,
                },
            )
        if snapshot.requests_week >= quota.requests_per_week:
            return BacktestAiAdmissionDecision(
                accepted=False,
                status="quota_exceeded",
                reason="requests_per_week_exceeded",
                message="AI configurator weekly quota is exhausted.",
                retry_after_seconds=DEFAULT_AI_WEEKLY_QUOTA_RETRY_AFTER_SECONDS,
                details={
                    "tier": tier,
                    "limit": quota.requests_per_week,
                    "window": "week",
                    "used": snapshot.requests_week,
                },
            )
        if snapshot.queued_jobs_for_user >= quota.max_queued_per_user:
            return BacktestAiAdmissionDecision(
                accepted=False,
                status="capacity_delayed",
                reason="max_queued_per_user_exceeded",
                message=(
                    "AI configurator already has queued work for this account. "
                    "Try again after the current request starts."
                ),
                retry_after_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                estimated_wait_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                details={
                    "tier": tier,
                    "limit": quota.max_queued_per_user,
                    "used": snapshot.queued_jobs_for_user,
                },
            )
        if snapshot.active_jobs_for_user >= quota.max_active_user_jobs:
            return BacktestAiAdmissionDecision(
                accepted=False,
                status="capacity_delayed",
                reason="max_active_user_jobs_exceeded",
                message=(
                    "AI configurator is already processing a request for this account."
                ),
                retry_after_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                estimated_wait_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                details={
                    "tier": tier,
                    "limit": quota.max_active_user_jobs,
                    "used": snapshot.active_jobs_for_user,
                },
            )
        if snapshot.active_jobs_global >= self.config.max_queue_size:
            return BacktestAiAdmissionDecision(
                accepted=False,
                status="capacity_delayed",
                reason="global_queue_saturated",
                message=(
                    "AI configurator is under high load. Try again in about 1-2 minutes."
                ),
                retry_after_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                estimated_wait_seconds=DEFAULT_AI_QUOTA_RETRY_AFTER_SECONDS,
                details={
                    "limit": self.config.max_queue_size,
                    "used": snapshot.active_jobs_global,
                },
            )
        return BacktestAiAdmissionDecision(
            accepted=True,
            status="accepted",
            reason="accepted",
            message="AI configurator request was accepted.",
            estimated_wait_seconds=self.config.estimated_wait_seconds,
            details={"tier": tier},
        )


__all__ = [
    "BacktestAiQuotaConfig",
    "BacktestAiQuotaService",
    "BacktestAiTierQuota",
    "DEFAULT_AI_TIER_QUOTAS",
]
