from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.backtest_variant_launch_reader import (
    BacktestVariantLaunchSnapshot,
)
from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.market_data_readiness import (
    MarketDataReadinessReader,
    MarketDataReadinessSnapshot,
)
from trading.contexts.strategy.application.ports.repositories import (
    StrategyCompatibilityReadinessRepository,
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
    require_owned_strategy,
)
from trading.contexts.strategy.application.use_cases.create_strategy_from_backtest_variant import (
    strategy_spec_from_backtest_variant_snapshot,
)
from trading.contexts.strategy.application.use_cases.errors import map_strategy_exception
from trading.contexts.strategy.domain.entities import Strategy, StrategySpecV1
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId

CompatibilityState = Literal["launchable", "not_launchable", "degraded"]


@dataclass(frozen=True, slots=True)
class StrategyCompatibilityReadinessReport:
    compatibility_check_id: UUID
    market_data_requirement_id: UUID
    organization_id: OrganizationId
    owner_user_id: Any
    strategy_id: UUID | None
    source_job_id: UUID | None
    source_variant_key: str | None
    strategy_spec_hash: str
    instrument_key: str
    market_type: str
    timeframe: str
    compatibility_state: CompatibilityState
    compatibility_reason_codes: tuple[str, ...]
    market_data_state: Literal["ready", "missing", "stale", "pending"]
    market_data_reason_codes: tuple[str, ...]
    market_data_stream_name: str
    market_data_stream_length: int | None
    market_data_last_message_id: str | None
    market_data_last_observed_at: datetime | None
    market_data_age_seconds: int | None
    checked_at: datetime

    @property
    def launch_blocked(self) -> bool:
        return self.compatibility_state == "not_launchable" or self.market_data_state != "ready"

    @property
    def launch_blocked_reason(self) -> str:
        if self.compatibility_state == "not_launchable":
            return self.compatibility_reason_codes[0]
        if self.market_data_state != "ready":
            return self.market_data_reason_codes[0]
        if self.compatibility_state == "degraded":
            return self.compatibility_reason_codes[0]
        return "ready"


class StrategyCompatibilityReadinessService:
    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository | None,
        compatibility_repository: StrategyCompatibilityReadinessRepository | None,
        clock: StrategyClock,
        market_data_reader: MarketDataReadinessReader | None = None,
        event_repository: StrategyEventRepository | None = None,
    ) -> None:
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyCompatibilityReadinessService requires clock")
        self._strategy_repository = strategy_repository
        self._compatibility_repository = compatibility_repository
        self._clock = clock
        self._market_data_reader = market_data_reader
        self._event_repository = event_repository

    def check_strategy(
        self, *, strategy_id: UUID, current_user: CurrentUser
    ) -> StrategyCompatibilityReadinessReport:
        if self._strategy_repository is None:
            raise RoehubError(
                code="strategy_compatibility.unavailable",
                message="Strategy repository is not configured",
                details={"reason": "strategy_repository_unavailable"},
            )
        strategy = require_owned_strategy(
            repository=self._strategy_repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )
        return self._record_report(
            strategy=strategy,
            current_user=current_user,
            source_job_id=None,
            source_variant_key=None,
        )

    def check_backtest_variant(
        self,
        *,
        current_user: CurrentUser,
        snapshot: BacktestVariantLaunchSnapshot,
    ) -> StrategyCompatibilityReadinessReport:
        if snapshot.owner_user_id != current_user.user_id:
            raise RoehubError(
                code="strategy_compatibility.forbidden",
                message="Backtest variant does not belong to current user",
                details={"reason": "forbidden", "job_id": str(snapshot.job_id)},
            )
        spec = strategy_spec_from_backtest_variant_snapshot(snapshot=snapshot)
        synthetic_strategy = Strategy.create(
            organization_id=current_user.organization_id,
            user_id=current_user.user_id,
            spec=spec,
            created_at=ensure_utc_datetime(value=self._clock.now(), field_name="clock.now"),
        )
        return self._record_report(
            strategy=synthetic_strategy,
            current_user=current_user,
            source_job_id=snapshot.job_id,
            source_variant_key=snapshot.variant_key,
            persist_strategy_id=False,
        )

    def _record_report(
        self,
        *,
        strategy: Strategy,
        current_user: CurrentUser,
        source_job_id: UUID | None,
        source_variant_key: str | None,
        persist_strategy_id: bool = True,
    ) -> StrategyCompatibilityReadinessReport:
        try:
            checked_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            compatibility_state, compatibility_reasons = _compatibility_for_spec(
                spec=strategy.spec
            )
            readiness = self._check_market_data(spec=strategy.spec, observed_at=checked_at)
            report = StrategyCompatibilityReadinessReport(
                compatibility_check_id=uuid4(),
                market_data_requirement_id=uuid4(),
                organization_id=current_user.organization_id,
                owner_user_id=current_user.user_id,
                strategy_id=strategy.strategy_id if persist_strategy_id else None,
                source_job_id=source_job_id,
                source_variant_key=source_variant_key,
                strategy_spec_hash=_strategy_spec_hash(spec=strategy.spec),
                instrument_key=strategy.spec.instrument_key,
                market_type=strategy.spec.market_type,
                timeframe=strategy.spec.timeframe.code,
                compatibility_state=compatibility_state,
                compatibility_reason_codes=compatibility_reasons,
                market_data_state=readiness.state,
                market_data_reason_codes=(readiness.reason_code,),
                market_data_stream_name=readiness.stream_name,
                market_data_stream_length=readiness.stream_length,
                market_data_last_message_id=readiness.last_message_id,
                market_data_last_observed_at=readiness.last_observed_at,
                market_data_age_seconds=readiness.age_seconds,
                checked_at=checked_at,
            )
            if self._compatibility_repository is not None:
                report = self._compatibility_repository.record(report=report)
            if persist_strategy_id:
                append_strategy_event(
                    repository=self._event_repository,
                    strategy_id=strategy.strategy_id,
                    current_user=current_user,
                    event_type="strategy_compatibility_readiness_checked",
                    ts=checked_at,
                    payload_json=_event_payload(report=report),
                )
            return report
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error

    def _check_market_data(
        self, *, spec: StrategySpecV1, observed_at: datetime
    ) -> MarketDataReadinessSnapshot:
        if self._market_data_reader is None:
            return MarketDataReadinessSnapshot(
                state="pending",
                reason_code="market_data_readiness_reader_unavailable",
                stream_name=f"md.candles.1m.{spec.instrument_key}",
                stream_length=None,
                last_message_id=None,
                last_observed_at=None,
                age_seconds=None,
            )
        return self._market_data_reader.check(
            instrument_key=spec.instrument_key,
            timeframe=spec.timeframe.code,
            observed_at=observed_at,
        )


def _compatibility_for_spec(
    *, spec: StrategySpecV1
) -> tuple[CompatibilityState, tuple[str, ...]]:
    if spec.schema_version != 1:
        return "not_launchable", ("unsupported_strategy_schema_version",)
    if spec.spec_kind != "roehub.strategy.v1":
        return "not_launchable", ("unsupported_strategy_spec_kind",)
    if not _is_supported_ma_cross(spec=spec):
        return "not_launchable", ("unsupported_live_evaluator",)
    if spec.timeframe.code != "1m":
        return "degraded", ("timeframe_rollup_required",)
    if _estimate_strategy_warmup_bars(spec=spec) > 500:
        return "degraded", ("large_warmup_live_start_slow",)
    return "launchable", ("supported_live_evaluator",)


def _is_supported_ma_cross(*, spec: StrategySpecV1) -> bool:
    if len(spec.indicators) != 1:
        return False
    indicator = spec.indicators[0]
    name = str(indicator.get("name") or indicator.get("kind") or indicator.get("id") or "")
    if name.strip().upper() != "MA":
        return False
    params = indicator.get("params")
    if not isinstance(params, Mapping):
        return False
    try:
        fast = int(params["fast"])
        slow = int(params["slow"])
    except (KeyError, TypeError, ValueError):
        return False
    if fast <= 0 or slow <= 0 or fast >= slow:
        return False
    expected_template = f"MA({fast},{slow})"
    return spec.signal_template.strip().upper().replace(" ", "") == expected_template


def _estimate_strategy_warmup_bars(*, spec: StrategySpecV1) -> int:
    candidates: list[int] = []
    for indicator in spec.indicators:
        params = indicator.get("params", {})
        if isinstance(params, Mapping):
            candidates.extend(_collect_warmup_candidates(value=params))
    return max(candidates) if candidates else 1


def _collect_warmup_candidates(*, value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, float):
        if value <= 0 or math.isnan(value) or math.isinf(value):
            return []
        return [int(math.ceil(value))]
    if isinstance(value, Mapping):
        candidates: list[int] = []
        for _, item_value in sorted(value.items(), key=lambda item: str(item[0])):
            candidates.extend(_collect_warmup_candidates(value=item_value))
        return candidates
    if isinstance(value, (list, tuple)):
        candidates = []
        for item in value:
            candidates.extend(_collect_warmup_candidates(value=item))
        return candidates
    return []


def _strategy_spec_hash(*, spec: StrategySpecV1) -> str:
    return hashlib.sha256(spec.canonical_json().encode("utf-8")).hexdigest()


def _event_payload(*, report: StrategyCompatibilityReadinessReport) -> dict[str, object]:
    return {
        "compatibility_state": report.compatibility_state,
        "compatibility_reason_codes": list(report.compatibility_reason_codes),
        "market_data_state": report.market_data_state,
        "market_data_reason_codes": list(report.market_data_reason_codes),
        "instrument_key": report.instrument_key,
        "timeframe": report.timeframe,
        "strategy_spec_hash": report.strategy_spec_hash,
    }


def report_to_json(*, report: StrategyCompatibilityReadinessReport) -> dict[str, Any]:
    return {
        "compatibility_check_id": str(report.compatibility_check_id),
        "market_data_requirement_id": str(report.market_data_requirement_id),
        "strategy_id": str(report.strategy_id) if report.strategy_id is not None else None,
        "source_job_id": str(report.source_job_id) if report.source_job_id is not None else None,
        "source_variant_key": report.source_variant_key,
        "strategy_spec_hash": report.strategy_spec_hash,
        "instrument_key": report.instrument_key,
        "market_type": report.market_type,
        "timeframe": report.timeframe,
        "compatibility_state": report.compatibility_state,
        "compatibility_reason_codes": list(report.compatibility_reason_codes),
        "market_data_state": report.market_data_state,
        "market_data_reason_codes": list(report.market_data_reason_codes),
        "market_data_stream_name": report.market_data_stream_name,
        "market_data_stream_length": report.market_data_stream_length,
        "market_data_last_message_id": report.market_data_last_message_id,
        "market_data_last_observed_at": (
            report.market_data_last_observed_at.astimezone(UTC).isoformat()
            if report.market_data_last_observed_at is not None
            else None
        ),
        "market_data_age_seconds": report.market_data_age_seconds,
        "launch_blocked": report.launch_blocked,
        "launch_blocked_reason": report.launch_blocked_reason,
        "checked_at": report.checked_at.astimezone(UTC).isoformat(),
    }
