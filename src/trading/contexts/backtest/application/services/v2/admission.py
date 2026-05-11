from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import ceil
from types import MappingProxyType
from typing import Any, Mapping

from trading.contexts.backtest.application.dto import (
    BacktestPreflightResult,
    BacktestRuntimeGuardrails,
    BacktestValidationIssue,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import PaidLevel

BACKTEST_ERROR_QUEUE_SATURATED = "backtest.queue_saturated"
BACKTEST_ERROR_RATE_LIMITED = "backtest.rate_limited"
BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE = "backtest.request_too_expensive"

DEFAULT_BACKTEST_QUOTA_RETRY_AFTER_SECONDS = 60
DEFAULT_BACKTEST_CREATE_WINDOW_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class BacktestTierAdmissionPolicy:
    max_active_full_jobs: int
    max_running_full_jobs: int
    max_full_job_creates_per_hour: int
    max_top_n: int
    max_indicator_arity: int
    max_range_days: int | None
    max_active_lazy_detail_tasks: int
    max_lazy_detail_creates_per_hour: int
    min_autorefresh_seconds: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_active_full_jobs", self.max_active_full_jobs),
            ("max_running_full_jobs", self.max_running_full_jobs),
            ("max_full_job_creates_per_hour", self.max_full_job_creates_per_hour),
            ("max_top_n", self.max_top_n),
            ("max_indicator_arity", self.max_indicator_arity),
            ("max_active_lazy_detail_tasks", self.max_active_lazy_detail_tasks),
            (
                "max_lazy_detail_creates_per_hour",
                self.max_lazy_detail_creates_per_hour,
            ),
            ("min_autorefresh_seconds", self.min_autorefresh_seconds),
        ):
            _require_positive_int(field_name=field_name, value=value)
        if self.max_range_days is not None:
            _require_positive_int(field_name="max_range_days", value=self.max_range_days)


def _require_positive_int(*, field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


DEFAULT_BACKTEST_TIER_ADMISSION_POLICIES: Mapping[str, BacktestTierAdmissionPolicy] = (
    MappingProxyType(
        {
            "free": BacktestTierAdmissionPolicy(
                max_active_full_jobs=2,
                max_running_full_jobs=1,
                max_full_job_creates_per_hour=5,
                max_top_n=20,
                max_indicator_arity=2,
                max_range_days=365,
                max_active_lazy_detail_tasks=2,
                max_lazy_detail_creates_per_hour=10,
                min_autorefresh_seconds=60,
            ),
            "base": BacktestTierAdmissionPolicy(
                max_active_full_jobs=5,
                max_running_full_jobs=1,
                max_full_job_creates_per_hour=15,
                max_top_n=50,
                max_indicator_arity=3,
                max_range_days=730,
                max_active_lazy_detail_tasks=5,
                max_lazy_detail_creates_per_hour=30,
                min_autorefresh_seconds=30,
            ),
            "pro": BacktestTierAdmissionPolicy(
                max_active_full_jobs=20,
                max_running_full_jobs=1,
                max_full_job_creates_per_hour=60,
                max_top_n=100,
                max_indicator_arity=7,
                max_range_days=None,
                max_active_lazy_detail_tasks=20,
                max_lazy_detail_creates_per_hour=120,
                min_autorefresh_seconds=15,
            ),
            "ultra": BacktestTierAdmissionPolicy(
                max_active_full_jobs=1_000_000_000_000,
                max_running_full_jobs=1_000_000_000_000,
                max_full_job_creates_per_hour=1_000_000_000_000,
                max_top_n=1_000_000_000,
                max_indicator_arity=1_000_000_000_000,
                max_range_days=None,
                max_active_lazy_detail_tasks=1_000_000_000_000,
                max_lazy_detail_creates_per_hour=1_000_000_000_000,
                min_autorefresh_seconds=1,
            ),
        }
    )
)


@dataclass(frozen=True, slots=True)
class BacktestAdmissionConfig:
    tier_policies: Mapping[str, BacktestTierAdmissionPolicy] = (
        DEFAULT_BACKTEST_TIER_ADMISSION_POLICIES
    )
    max_active_full_jobs_global: int = 200
    max_active_lazy_detail_tasks_global: int = 500
    retry_after_seconds: int = DEFAULT_BACKTEST_QUOTA_RETRY_AFTER_SECONDS

    def __post_init__(self) -> None:
        _require_positive_int(
            field_name="max_active_full_jobs_global",
            value=self.max_active_full_jobs_global,
        )
        _require_positive_int(
            field_name="max_active_lazy_detail_tasks_global",
            value=self.max_active_lazy_detail_tasks_global,
        )
        _require_positive_int(
            field_name="retry_after_seconds",
            value=self.retry_after_seconds,
        )
        normalized: dict[str, BacktestTierAdmissionPolicy] = {}
        for raw_tier, policy in self.tier_policies.items():
            tier = str(raw_tier).strip().lower()
            if not tier:
                raise ValueError("backtest admission tier key must be non-empty")
            normalized[tier] = policy
        for required_tier in ("free", "base", "pro", "ultra"):
            if required_tier not in normalized:
                raise ValueError(f"missing backtest admission tier {required_tier!r}")
        object.__setattr__(self, "tier_policies", MappingProxyType(normalized))

    def policy_for(self, *, paid_level: PaidLevel) -> BacktestTierAdmissionPolicy:
        return self.tier_policies[str(paid_level)]


@dataclass(frozen=True, slots=True)
class BacktestFullJobQuotaSnapshot:
    active_full_jobs_for_user: int
    full_job_creates_in_window: int
    active_full_jobs_global: int


@dataclass(frozen=True, slots=True)
class BacktestLazyDetailQuotaSnapshot:
    active_lazy_detail_tasks_for_user: int
    lazy_detail_creates_in_window: int
    active_lazy_detail_tasks_global: int


@dataclass(frozen=True, slots=True)
class BacktestAdmissionService:
    config: BacktestAdmissionConfig = BacktestAdmissionConfig()

    def preflight_validation_guardrails(
        self,
        *,
        base_guardrails: BacktestRuntimeGuardrails,
    ) -> BacktestRuntimeGuardrails:
        return replace(
            base_guardrails,
            max_top_n=max(policy.max_top_n for policy in self.config.tier_policies.values()),
            max_indicator_arity=max(
                policy.max_indicator_arity
                for policy in self.config.tier_policies.values()
            ),
        )

    def ensure_full_job_request_allowed(
        self,
        *,
        paid_level: PaidLevel,
        preflight: BacktestPreflightResult,
    ) -> None:
        tier = str(paid_level)
        policy = self.config.policy_for(paid_level=paid_level)
        top_n = int(preflight.normalized_request.get("top_n") or 0)
        if top_n > policy.max_top_n:
            _raise_request_too_expensive(
                paid_level=tier,
                limit_scope="full_jobs.top_n",
                issue=BacktestValidationIssue(
                    path="top_n",
                    code="max_top_n",
                    message=f"top_n must be <= {policy.max_top_n} for paid_level {tier}",
                ),
                limit=policy.max_top_n,
                requested=top_n,
            )

        indicators = preflight.normalized_request.get("indicators")
        indicator_arity = len(indicators) if isinstance(indicators, list) else 0
        if indicator_arity > policy.max_indicator_arity:
            _raise_request_too_expensive(
                paid_level=tier,
                limit_scope="full_jobs.indicator_arity",
                issue=BacktestValidationIssue(
                    path="indicators",
                    code="max_indicator_arity",
                    message=(
                        "indicator arity must be <= "
                        f"{policy.max_indicator_arity} for paid_level {tier}"
                    ),
                ),
                limit=policy.max_indicator_arity,
                requested=indicator_arity,
            )

        requested_days = _request_range_days(preflight=preflight)
        if policy.max_range_days is not None and requested_days > policy.max_range_days:
            _raise_request_too_expensive(
                paid_level=tier,
                limit_scope="full_jobs.range_days",
                issue=BacktestValidationIssue(
                    path="time_range",
                    code="max_range_days",
                    message=(
                        "time_range days must be <= "
                        f"{policy.max_range_days} for paid_level {tier}"
                    ),
                ),
                limit=policy.max_range_days,
                requested=requested_days,
            )

    def ensure_full_job_quota_allowed(
        self,
        *,
        paid_level: PaidLevel,
        snapshot: BacktestFullJobQuotaSnapshot,
    ) -> None:
        tier = str(paid_level)
        policy = self.config.policy_for(paid_level=paid_level)
        if snapshot.active_full_jobs_global >= self.config.max_active_full_jobs_global:
            _raise_queue_saturated(
                limit_scope="global.full_jobs.active",
                limit=self.config.max_active_full_jobs_global,
                used=snapshot.active_full_jobs_global,
                retry_after_seconds=self.config.retry_after_seconds,
            )
        if snapshot.active_full_jobs_for_user >= policy.max_active_full_jobs:
            _raise_rate_limited(
                paid_level=tier,
                limit_scope="full_jobs.active",
                limit=policy.max_active_full_jobs,
                used=snapshot.active_full_jobs_for_user,
                retry_after_seconds=self.config.retry_after_seconds,
            )
        if snapshot.full_job_creates_in_window >= policy.max_full_job_creates_per_hour:
            _raise_rate_limited(
                paid_level=tier,
                limit_scope="full_jobs.creates_per_hour",
                limit=policy.max_full_job_creates_per_hour,
                used=snapshot.full_job_creates_in_window,
                retry_after_seconds=DEFAULT_BACKTEST_CREATE_WINDOW_SECONDS,
            )

    def ensure_lazy_detail_quota_allowed(
        self,
        *,
        paid_level: PaidLevel,
        snapshot: BacktestLazyDetailQuotaSnapshot,
    ) -> None:
        tier = str(paid_level)
        policy = self.config.policy_for(paid_level=paid_level)
        if (
            snapshot.active_lazy_detail_tasks_global
            >= self.config.max_active_lazy_detail_tasks_global
        ):
            _raise_queue_saturated(
                limit_scope="global.lazy_detail.active",
                limit=self.config.max_active_lazy_detail_tasks_global,
                used=snapshot.active_lazy_detail_tasks_global,
                retry_after_seconds=self.config.retry_after_seconds,
            )
        if (
            snapshot.active_lazy_detail_tasks_for_user
            >= policy.max_active_lazy_detail_tasks
        ):
            _raise_rate_limited(
                paid_level=tier,
                limit_scope="lazy_detail.active",
                limit=policy.max_active_lazy_detail_tasks,
                used=snapshot.active_lazy_detail_tasks_for_user,
                retry_after_seconds=self.config.retry_after_seconds,
            )
        if (
            snapshot.lazy_detail_creates_in_window
            >= policy.max_lazy_detail_creates_per_hour
        ):
            _raise_rate_limited(
                paid_level=tier,
                limit_scope="lazy_detail.creates_per_hour",
                limit=policy.max_lazy_detail_creates_per_hour,
                used=snapshot.lazy_detail_creates_in_window,
                retry_after_seconds=DEFAULT_BACKTEST_CREATE_WINDOW_SECONDS,
            )


def _request_range_days(*, preflight: BacktestPreflightResult) -> int:
    time_range = preflight.normalized_request.get("time_range")
    if not isinstance(time_range, Mapping):
        return 0
    start = _parse_utc(value=time_range.get("start"))
    end = _parse_utc(value=time_range.get("end"))
    if start is None or end is None or end <= start:
        return 0
    return max(1, ceil((end - start).total_seconds() / 86_400))


def _parse_utc(*, value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _raise_request_too_expensive(
    *,
    paid_level: str,
    limit_scope: str,
    issue: BacktestValidationIssue,
    limit: int,
    requested: int,
) -> None:
    raise RoehubError(
        code=BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE,
        message="Backtest request exceeds tier admission policy",
        details={
            "errors": [issue.as_mapping()],
            "paid_level": paid_level,
            "limit_scope": limit_scope,
            "limit": limit,
            "requested": requested,
            "retryable": False,
        },
    )


def _raise_rate_limited(
    *,
    paid_level: str,
    limit_scope: str,
    limit: int,
    used: int,
    retry_after_seconds: int,
) -> None:
    raise RoehubError(
        code=BACKTEST_ERROR_RATE_LIMITED,
        message="Backtest quota was reached for the current tier",
        details={
            "paid_level": paid_level,
            "limit_scope": limit_scope,
            "limit": limit,
            "used": used,
            "retry_after_seconds": retry_after_seconds,
            "retryable": True,
        },
    )


def _raise_queue_saturated(
    *,
    limit_scope: str,
    limit: int,
    used: int,
    retry_after_seconds: int,
) -> None:
    raise RoehubError(
        code=BACKTEST_ERROR_QUEUE_SATURATED,
        message="Backtest queue is saturated",
        details={
            "limit_scope": limit_scope,
            "limit": limit,
            "used": used,
            "retry_after_seconds": retry_after_seconds,
            "retryable": True,
        },
    )

__all__ = [
    "BACKTEST_ERROR_QUEUE_SATURATED",
    "BACKTEST_ERROR_RATE_LIMITED",
    "BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE",
    "BacktestAdmissionConfig",
    "BacktestAdmissionService",
    "BacktestFullJobQuotaSnapshot",
    "BacktestLazyDetailQuotaSnapshot",
    "BacktestTierAdmissionPolicy",
    "DEFAULT_BACKTEST_TIER_ADMISSION_POLICIES",
]
