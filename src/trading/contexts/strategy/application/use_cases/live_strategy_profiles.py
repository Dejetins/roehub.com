from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.exchange_connection_readiness import (
    ExchangeConnectionReadinessChecker,
)
from trading.contexts.strategy.application.ports.repositories import (
    LiveStrategyProfileRepository,
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
    require_owned_strategy,
)
from trading.contexts.strategy.application.use_cases.errors import map_strategy_exception
from trading.contexts.strategy.domain.entities.live_strategy_profile import (
    LiveStrategyProfile,
    LiveStrategyProfileMode,
    LiveStrategyProfileSizingMethod,
)
from trading.platform.errors import RoehubError


@dataclass(frozen=True, slots=True)
class LiveStrategyProfileConfig:
    mode: LiveStrategyProfileMode = "monitor_only"
    exchange_connection_id: UUID | None = None
    sizing_method: LiveStrategyProfileSizingMethod = "fixed_quote"
    sizing_value: Decimal = Decimal("0")
    max_position_notional: Decimal | None = None
    max_orders_per_run: int = 0
    max_notional_per_run: Decimal = Decimal("0")


class LiveStrategyProfileService:
    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository,
        profile_repository: LiveStrategyProfileRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
        exchange_connection_checker: ExchangeConnectionReadinessChecker | None = None,
    ) -> None:
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveStrategyProfileService requires strategy_repository")
        if profile_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveStrategyProfileService requires profile_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveStrategyProfileService requires clock")
        self._strategy_repository = strategy_repository
        self._profile_repository = profile_repository
        self._clock = clock
        self._event_repository = event_repository
        self._exchange_connection_checker = exchange_connection_checker

    def get_or_create_default(
        self, *, strategy_id: UUID, current_user: CurrentUser
    ) -> LiveStrategyProfile:
        strategy = require_owned_strategy(
            repository=self._strategy_repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )
        existing = self._profile_repository.get_for_strategy(
            owner_user_id=current_user.user_id,
            strategy_id=strategy.strategy_id,
        )
        if existing is not None:
            return existing
        now = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        profile = _default_profile(
            owner_user_id=current_user.user_id,
            strategy_id=strategy.strategy_id,
            now=now,
        )
        created = self._profile_repository.create(profile=profile)
        if created is None:
            loaded = self._profile_repository.get_for_strategy(
                owner_user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
            )
            if loaded is not None:
                return loaded
            raise RoehubError(
                code="live_strategy_profile_conflict",
                message="Live strategy profile already exists.",
                details={"strategy_id": str(strategy.strategy_id)},
            )
        append_strategy_event(
            repository=self._event_repository,
            strategy_id=strategy.strategy_id,
            current_user=current_user,
            event_type="live_strategy_profile_created",
            ts=now,
            payload_json=_event_payload(profile=created),
        )
        return created

    def update_profile(
        self,
        *,
        strategy_id: UUID,
        current_user: CurrentUser,
        config: LiveStrategyProfileConfig,
        recent_auth_confirmed: bool,
    ) -> LiveStrategyProfile:
        existing = self.get_or_create_default(
            strategy_id=strategy_id,
            current_user=current_user,
        )
        now = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        configured = LiveStrategyProfile(
            profile_id=existing.profile_id,
            owner_user_id=existing.owner_user_id,
            strategy_id=existing.strategy_id,
            mode=config.mode,
            exchange_connection_id=config.exchange_connection_id,
            sizing_method=config.sizing_method,
            sizing_value=config.sizing_value,
            max_position_notional=config.max_position_notional,
            max_orders_per_run=config.max_orders_per_run,
            max_notional_per_run=config.max_notional_per_run,
            readiness_status=existing.readiness_status,
            readiness_reason=existing.readiness_reason,
            created_at=existing.created_at,
            updated_at=now,
        )
        evaluated = self._evaluate_readiness(
            profile=configured,
            recent_auth_confirmed=recent_auth_confirmed,
            now=now,
        )
        persisted = self._profile_repository.update(profile=evaluated)
        append_strategy_event(
            repository=self._event_repository,
            strategy_id=persisted.strategy_id,
            current_user=current_user,
            event_type="live_strategy_profile_updated",
            ts=now,
            payload_json=_event_payload(profile=persisted),
        )
        return persisted

    def refresh_readiness(
        self,
        *,
        strategy_id: UUID,
        current_user: CurrentUser,
        recent_auth_confirmed: bool,
    ) -> LiveStrategyProfile:
        existing = self.get_or_create_default(
            strategy_id=strategy_id,
            current_user=current_user,
        )
        now = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
        evaluated = self._evaluate_readiness(
            profile=existing,
            recent_auth_confirmed=recent_auth_confirmed,
            now=now,
        )
        if (
            evaluated.readiness_status == existing.readiness_status
            and evaluated.readiness_reason == existing.readiness_reason
        ):
            return existing
        return self._profile_repository.update(profile=evaluated)

    def _evaluate_readiness(
        self,
        *,
        profile: LiveStrategyProfile,
        recent_auth_confirmed: bool,
        now,
    ) -> LiveStrategyProfile:
        if profile.mode == "monitor_only":
            return profile.with_readiness(
                readiness_status="ready",
                readiness_reason="monitor_only_no_exchange_submit",
                updated_at=now,
            )
        if profile.mode == "paper":
            return profile.with_readiness(
                readiness_status="ready",
                readiness_reason="paper_no_exchange_submit",
                updated_at=now,
            )
        if not recent_auth_confirmed:
            return profile.with_readiness(
                readiness_status="blocked",
                readiness_reason="recent_auth_required",
                updated_at=now,
            )
        if profile.exchange_connection_id is None:
            return profile.with_readiness(
                readiness_status="blocked",
                readiness_reason="exchange_connection_required",
                updated_at=now,
            )
        if self._exchange_connection_checker is None:
            return profile.with_readiness(
                readiness_status="blocked",
                readiness_reason="exchange_connection_checker_unavailable",
                updated_at=now,
            )
        try:
            readiness = self._exchange_connection_checker.check_trading_ready(
                owner_user_id=profile.owner_user_id,
                exchange_connection_id=profile.exchange_connection_id,
            )
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error
        if not readiness.eligible:
            return profile.with_readiness(
                readiness_status="blocked",
                readiness_reason=readiness.reason or "exchange_connection_not_ready",
                updated_at=now,
            )
        return profile.with_readiness(
            readiness_status="ready",
            readiness_reason="live_ready_recent_auth_and_connection",
            updated_at=now,
        )


def _default_profile(*, owner_user_id, strategy_id: UUID, now) -> LiveStrategyProfile:
    return LiveStrategyProfile(
        profile_id=uuid4(),
        owner_user_id=owner_user_id,
        strategy_id=strategy_id,
        mode="monitor_only",
        exchange_connection_id=None,
        sizing_method="fixed_quote",
        sizing_value=Decimal("0"),
        max_position_notional=None,
        max_orders_per_run=0,
        max_notional_per_run=Decimal("0"),
        readiness_status="ready",
        readiness_reason="monitor_only_no_exchange_submit",
        created_at=now,
        updated_at=now,
    )


def _event_payload(*, profile: LiveStrategyProfile) -> dict[str, object]:
    return {
        "profile_id": str(profile.profile_id),
        "strategy_id": str(profile.strategy_id),
        "mode": profile.mode,
        "exchange_connection_id": (
            str(profile.exchange_connection_id)
            if profile.exchange_connection_id is not None
            else None
        ),
        "readiness_status": profile.readiness_status,
        "readiness_reason": profile.readiness_reason,
    }
