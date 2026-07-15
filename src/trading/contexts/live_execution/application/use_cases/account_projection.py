from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import (
    ExchangeAccountProjectionRepository,
    ExchangeAccountStateReader,
    LiveExecutionClock,
)
from trading.contexts.live_execution.domain import (
    AccountConfigGuardResult,
    AccountProjectionReadiness,
    ExchangeAccountProjection,
    ExpectedInstrumentConfig,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class ExchangeAccountProjectionService:
    def __init__(
        self,
        *,
        repository: ExchangeAccountProjectionRepository | None,
        clock: LiveExecutionClock,
        max_projection_age: timedelta = timedelta(minutes=2),
    ) -> None:
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeAccountProjectionService requires clock")
        if max_projection_age.total_seconds() <= 0:
            raise ValueError("max_projection_age must be positive")
        self._repository = repository
        self._clock = clock
        self._max_projection_age = max_projection_age

    def sync_connection(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        reader: ExchangeAccountStateReader,
        requirements: tuple[ExpectedInstrumentConfig, ...] = (),
    ) -> ExchangeAccountProjection:
        projection = reader.read_account_projection(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
        )
        if projection.organization_id != organization_id:
            raise ValueError("account projection organization does not match request")
        if projection.owner_user_id != owner_user_id:
            raise ValueError("account projection owner does not match request owner")
        if projection.exchange_connection_id != exchange_connection_id:
            raise ValueError("account projection connection does not match request connection")
        if self._repository is not None:
            projection = self._repository.record_projection(projection=projection)
            for requirement in requirements:
                self._repository.record_config_guard_result(
                    result=self.verify_config(
                        organization_id=organization_id,
                        owner_user_id=owner_user_id,
                        exchange_connection_id=exchange_connection_id,
                        requirement=requirement,
                        projection=projection,
                    )
                )
        return projection

    def verify_config(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        requirement: ExpectedInstrumentConfig,
        projection: ExchangeAccountProjection | None = None,
    ) -> AccountConfigGuardResult:
        checked_at = _utc(self._clock.now())
        if projection is None and self._repository is not None:
            projection = self._repository.get_latest_projection(
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
            )
        if projection is None:
            return AccountConfigGuardResult(
                config_guard_result_id=uuid4(),
                account_snapshot_id=None,
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                instrument_key=requirement.instrument_key,
                market_type=requirement.market_type,
                status="degraded",
                reason_codes=("account_projection_missing",),
                checked_at=checked_at,
                requirement=requirement,
            )
        reasons = _config_mismatch_reasons(projection=projection, requirement=requirement)
        return AccountConfigGuardResult(
            config_guard_result_id=uuid4(),
            account_snapshot_id=projection.account_snapshot_id,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            instrument_key=requirement.instrument_key,
            market_type=requirement.market_type,
            status="mismatch" if reasons else "verified",
            reason_codes=reasons or ("verify_only_config_ok",),
            checked_at=checked_at,
            requirement=requirement,
        )

    def get_readiness(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID | None,
        requirement: ExpectedInstrumentConfig | None,
    ) -> AccountProjectionReadiness:
        checked_at = _utc(self._clock.now())
        if exchange_connection_id is None:
            return _readiness(
                status="degraded",
                reason_codes=("exchange_connection_required",),
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=None,
            )
        if requirement is None:
            return _readiness(
                status="degraded",
                reason_codes=("instrument_requirement_required",),
                checked_at=checked_at,
                requirement=None,
                exchange_connection_id=exchange_connection_id,
            )
        if self._repository is None:
            return _readiness(
                status="degraded",
                reason_codes=("account_projection_repository_unavailable",),
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
            )
        projection = self._repository.get_latest_projection(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
        )
        if projection is None:
            return _readiness(
                status="degraded",
                reason_codes=("account_projection_missing",),
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
            )
        guard = self._repository.get_latest_config_guard_result(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            instrument_key=requirement.instrument_key,
            market_type=requirement.market_type,
        )
        if guard is None:
            guard = self.verify_config(
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                requirement=requirement,
                projection=projection,
            )
            guard = self._repository.record_config_guard_result(result=guard)
        age_seconds = projection.age_seconds(now=checked_at)
        if projection.sync_status == "degraded":
            return _readiness(
                status="degraded",
                reason_codes=(projection.sync_reason,),
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
                projection=projection,
                guard=guard,
                age_seconds=age_seconds,
            )
        if age_seconds > int(self._max_projection_age.total_seconds()):
            return _readiness(
                status="stale",
                reason_codes=("account_projection_stale",),
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
                projection=projection,
                guard=guard,
                age_seconds=age_seconds,
            )
        if guard.status == "mismatch":
            return _readiness(
                status="config_mismatch",
                reason_codes=guard.reason_codes,
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
                projection=projection,
                guard=guard,
                age_seconds=age_seconds,
            )
        if guard.status == "degraded":
            return _readiness(
                status="degraded",
                reason_codes=guard.reason_codes,
                checked_at=checked_at,
                requirement=requirement,
                exchange_connection_id=exchange_connection_id,
                projection=projection,
                guard=guard,
                age_seconds=age_seconds,
            )
        return _readiness(
            status="fresh",
            reason_codes=("account_projection_fresh",),
            checked_at=checked_at,
            requirement=requirement,
            exchange_connection_id=exchange_connection_id,
            projection=projection,
            guard=guard,
            age_seconds=age_seconds,
        )


def _config_mismatch_reasons(
    *, projection: ExchangeAccountProjection, requirement: ExpectedInstrumentConfig
) -> tuple[str, ...]:
    reasons: list[str] = []
    filters = next(
        (
            item
            for item in projection.instrument_filters
            if item.instrument_key == requirement.instrument_key
        ),
        None,
    )
    if filters is None:
        reasons.append("instrument_filters_missing")
        if requirement.order_notional is not None:
            reasons.append("min_notional_issue")
    else:
        if requirement.min_notional is not None and _lt(
            filters.min_notional,
            requirement.min_notional,
        ):
            reasons.append("min_notional_below_requirement")
        if requirement.tick_size is not None and filters.tick_size != requirement.tick_size:
            reasons.append("tick_size_mismatch")
        if requirement.step_size is not None and filters.step_size != requirement.step_size:
            reasons.append("step_size_mismatch")
        if requirement.order_notional is not None:
            if filters.min_notional is None or requirement.order_notional < filters.min_notional:
                reasons.append("min_notional_issue")
    if requirement.required_balance_asset is not None:
        balance = next(
            (
                item
                for item in projection.balances
                if item.asset == requirement.required_balance_asset.upper()
            ),
            None,
        )
        if balance is None:
            reasons.append("missing_balance")
        elif requirement.order_notional is not None and balance.free < requirement.order_notional:
            reasons.append("insufficient_balance")
    position = next(
        (
            item
            for item in projection.positions
            if item.instrument_key == requirement.instrument_key
        ),
        None,
    )
    if (
        requirement.market_type == "futures"
        and requirement.side == "short"
        and (
            requirement.expected_margin_mode is not None
            or requirement.required_leverage is not None
        )
        and position is None
    ):
        reasons.append("unsafe_futures_short")
    if position is not None:
        if requirement.expected_margin_mode and _normalize_margin_mode(
            position.margin_mode
        ) != _normalize_margin_mode(requirement.expected_margin_mode):
            reasons.append("margin_mode_mismatch")
        if requirement.expected_position_mode and _normalize_position_mode(
            position.position_mode
        ) != _normalize_position_mode(requirement.expected_position_mode):
            reasons.append("position_mode_mismatch")
        if (
            requirement.required_leverage is not None
            and position.leverage != requirement.required_leverage
        ):
            reasons.append("leverage_mismatch")
    return tuple(dict.fromkeys(reasons))


def _lt(left: Decimal | None, right: Decimal) -> bool:
    return left is None or left < right


def _normalize_margin_mode(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().casefold()
    if raw in {"isolated", "isolate", "1"}:
        return "isolated"
    if raw in {"cross", "crossed", "0"}:
        return "cross"
    return raw or None


def _normalize_position_mode(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().casefold()
    if raw in {"one_way", "one-way", "merged_single", "0"}:
        return "one_way"
    if raw in {"hedge", "both_sides", "1", "2"}:
        return "hedge"
    return raw or None


def _readiness(
    *,
    status: str,
    reason_codes: tuple[str, ...],
    checked_at: datetime,
    requirement: ExpectedInstrumentConfig | None,
    exchange_connection_id: UUID | None,
    projection: ExchangeAccountProjection | None = None,
    guard: AccountConfigGuardResult | None = None,
    age_seconds: int | None = None,
) -> AccountProjectionReadiness:
    return AccountProjectionReadiness(
        status=status,  # type: ignore[arg-type]
        reason_codes=reason_codes,
        exchange_connection_id=exchange_connection_id,
        instrument_key=requirement.instrument_key if requirement is not None else None,
        market_type=requirement.market_type if requirement is not None else None,
        account_snapshot_id=projection.account_snapshot_id if projection is not None else None,
        config_guard_result_id=(
            guard.config_guard_result_id if guard is not None else None
        ),
        age_seconds=age_seconds,
        source_hash=projection.source_hash if projection is not None else None,
        checked_at=checked_at,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
