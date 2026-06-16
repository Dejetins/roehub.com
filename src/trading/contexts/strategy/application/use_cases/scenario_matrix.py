from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import product
from typing import Any, Literal, Mapping, get_args
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.backtest_variant_launch_reader import (
    BacktestVariantLaunchSnapshot,
)
from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyVariantScenarioMatrixRepository,
)
from trading.contexts.strategy.application.use_cases._shared import ensure_utc_datetime
from trading.contexts.strategy.application.use_cases.compatibility_readiness import (
    StrategyCompatibilityReadinessReport,
    StrategyCompatibilityReadinessService,
)
from trading.contexts.strategy.application.use_cases.create_strategy_from_backtest_variant import (
    strategy_spec_from_backtest_variant_snapshot,
)
from trading.contexts.strategy.application.use_cases.errors import map_strategy_exception
from trading.contexts.strategy.domain.entities import LiveStrategyProfileSizingMethod
from trading.platform.errors import RoehubError

SCENARIO_MATRIX_SCHEMA_V1 = "strategy_variant_scenario_matrix_v1"
SCENARIO_MATRIX_SYMBOL_SCOPE_V1 = "BTCUSDT"
SCENARIO_MATRIX_MODES_V1: tuple[Literal["paper", "testnet"], ...] = ("paper", "testnet")
SCENARIO_MATRIX_LAUNCH_RISK_MODES_V1: tuple[str, ...] = ("single_position_cap",)
SCENARIO_MATRIX_MIN_NOTIONAL_USD_V1 = Decimal("10")

ScenarioMatrixState = Literal["launchable", "degraded", "blocked"]
ScenarioOrderCapability = Literal["paper_only", "real_order_capable", "unsupported"]


@dataclass(frozen=True, slots=True)
class StrategyVariantScenarioMatrixRow:
    scenario_matrix_row_id: UUID
    owner_user_id: Any
    source_job_id: UUID
    source_variant_key: str
    variant_hash: str
    strategy_spec_hash: str
    scenario_key: str
    mode: str
    market_type: str
    symbol: str
    entry_sizing: str
    risk_mode: str
    direction: str
    backtest_risk_mode: str
    backtest_direction_mode: str
    scenario_state: ScenarioMatrixState
    scenario_reason_codes: tuple[str, ...]
    order_capability: ScenarioOrderCapability
    order_capability_reason_codes: tuple[str, ...]
    compatibility_check_id: UUID
    market_data_requirement_id: UUID
    compatibility_state: str
    compatibility_reason_codes: tuple[str, ...]
    market_data_state: str
    market_data_reason_codes: tuple[str, ...]
    checked_at: datetime

    @property
    def launch_blocked(self) -> bool:
        return self.scenario_state == "blocked"

    @property
    def launch_blocked_reason(self) -> str:
        if self.scenario_reason_codes:
            return self.scenario_reason_codes[0]
        return "ready"


@dataclass(frozen=True, slots=True)
class StrategyVariantScenarioMatrixReport:
    owner_user_id: Any
    source_job_id: UUID
    source_variant_key: str
    variant_hash: str
    source_market_type: str
    symbol: str
    strategy_spec_hash: str
    backtest_risk_mode: str
    backtest_direction_mode: str
    checked_at: datetime
    rows: tuple[StrategyVariantScenarioMatrixRow, ...]


class StrategyVariantScenarioMatrixService:
    def __init__(
        self,
        *,
        compatibility_readiness_service: StrategyCompatibilityReadinessService,
        clock: StrategyClock,
        repository: StrategyVariantScenarioMatrixRepository | None = None,
    ) -> None:
        if compatibility_readiness_service is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "StrategyVariantScenarioMatrixService requires compatibility_readiness_service"
            )
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyVariantScenarioMatrixService requires clock")
        self._compatibility_readiness_service = compatibility_readiness_service
        self._clock = clock
        self._repository = repository

    def build_for_backtest_variant(
        self,
        *,
        current_user: CurrentUser,
        snapshot: BacktestVariantLaunchSnapshot,
    ) -> StrategyVariantScenarioMatrixReport:
        if snapshot.owner_user_id != current_user.user_id:
            raise RoehubError(
                code="strategy_scenario_matrix.forbidden",
                message="Backtest variant does not belong to current user",
                details={"reason": "forbidden", "job_id": str(snapshot.job_id)},
            )
        try:
            checked_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            compatibility = self._compatibility_readiness_service.check_backtest_variant(
                current_user=current_user,
                snapshot=snapshot,
            )
            spec = strategy_spec_from_backtest_variant_snapshot(snapshot=snapshot)
            strategy_spec_hash = hashlib.sha256(spec.canonical_json().encode("utf-8")).hexdigest()
            backtest_risk_mode = _backtest_risk_mode(snapshot=snapshot)
            backtest_direction_mode = _backtest_direction_mode(snapshot=snapshot)
            report = StrategyVariantScenarioMatrixReport(
                owner_user_id=current_user.user_id,
                source_job_id=snapshot.job_id,
                source_variant_key=snapshot.variant_key,
                variant_hash=snapshot.variant_hash,
                source_market_type=snapshot.market_type.strip().casefold(),
                symbol=snapshot.symbol.strip().upper(),
                strategy_spec_hash=strategy_spec_hash,
                backtest_risk_mode=backtest_risk_mode,
                backtest_direction_mode=backtest_direction_mode,
                checked_at=checked_at,
                rows=tuple(
                    _scenario_row(
                        current_user=current_user,
                        snapshot=snapshot,
                        strategy_spec_hash=strategy_spec_hash,
                        backtest_risk_mode=backtest_risk_mode,
                        backtest_direction_mode=backtest_direction_mode,
                        compatibility=compatibility,
                        checked_at=checked_at,
                        mode=mode,
                        entry_sizing=entry_sizing,
                        risk_mode=risk_mode,
                        direction=direction,
                    )
                    for mode, entry_sizing, risk_mode, direction in product(
                        SCENARIO_MATRIX_MODES_V1,
                        _entry_sizing_modes(),
                        SCENARIO_MATRIX_LAUNCH_RISK_MODES_V1,
                        _directions_from_backtest_mode(backtest_direction_mode),
                    )
                ),
            )
            if self._repository is not None:
                report = self._repository.record(report=report)
            return report
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error


def _scenario_row(
    *,
    current_user: CurrentUser,
    snapshot: BacktestVariantLaunchSnapshot,
    strategy_spec_hash: str,
    backtest_risk_mode: str,
    backtest_direction_mode: str,
    compatibility: StrategyCompatibilityReadinessReport,
    checked_at: datetime,
    mode: str,
    entry_sizing: str,
    risk_mode: str,
    direction: str,
) -> StrategyVariantScenarioMatrixRow:
    market_type = snapshot.market_type.strip().casefold()
    symbol = snapshot.symbol.strip().upper()
    scenario_state, scenario_reasons = _scenario_state(
        compatibility=compatibility,
        mode=mode,
        market_type=market_type,
        symbol=symbol,
        direction=direction,
    )
    order_capability, order_reasons = _order_capability(
        mode=mode,
        market_type=market_type,
        direction=direction,
    )
    scenario_key = _scenario_key(
        {
            "schema": SCENARIO_MATRIX_SCHEMA_V1,
            "source_job_id": str(snapshot.job_id),
            "source_variant_key": snapshot.variant_key,
            "variant_hash": snapshot.variant_hash,
            "strategy_spec_hash": strategy_spec_hash,
            "mode": mode,
            "market_type": market_type,
            "symbol": symbol,
            "entry_sizing": entry_sizing,
            "risk_mode": risk_mode,
            "direction": direction,
            "backtest_risk_mode": backtest_risk_mode,
            "backtest_direction_mode": backtest_direction_mode,
        }
    )
    return StrategyVariantScenarioMatrixRow(
        scenario_matrix_row_id=uuid4(),
        owner_user_id=current_user.user_id,
        source_job_id=snapshot.job_id,
        source_variant_key=snapshot.variant_key,
        variant_hash=snapshot.variant_hash,
        strategy_spec_hash=strategy_spec_hash,
        scenario_key=scenario_key,
        mode=mode,
        market_type=market_type,
        symbol=symbol,
        entry_sizing=entry_sizing,
        risk_mode=risk_mode,
        direction=direction,
        backtest_risk_mode=backtest_risk_mode,
        backtest_direction_mode=backtest_direction_mode,
        scenario_state=scenario_state,
        scenario_reason_codes=scenario_reasons,
        order_capability=order_capability,
        order_capability_reason_codes=order_reasons,
        compatibility_check_id=compatibility.compatibility_check_id,
        market_data_requirement_id=compatibility.market_data_requirement_id,
        compatibility_state=compatibility.compatibility_state,
        compatibility_reason_codes=compatibility.compatibility_reason_codes,
        market_data_state=compatibility.market_data_state,
        market_data_reason_codes=compatibility.market_data_reason_codes,
        checked_at=checked_at,
    )


def _scenario_state(
    *,
    compatibility: StrategyCompatibilityReadinessReport,
    mode: str,
    market_type: str,
    symbol: str,
    direction: str,
) -> tuple[ScenarioMatrixState, tuple[str, ...]]:
    if symbol != SCENARIO_MATRIX_SYMBOL_SCOPE_V1:
        return "blocked", ("unsupported_symbol",)
    if market_type not in {"spot", "futures"}:
        return "blocked", ("invalid_market_type",)
    if mode == "testnet" and market_type == "spot" and direction == "short":
        return "blocked", ("spot_short_not_supported",)
    if compatibility.launch_blocked:
        return "blocked", (compatibility.launch_blocked_reason,)
    if mode == "testnet":
        return "blocked", ("exchange_connection_required",)
    if compatibility.compatibility_state == "degraded":
        return "degraded", compatibility.compatibility_reason_codes
    return "launchable", ("paper_no_exchange_submit",)


def _order_capability(
    *, mode: str, market_type: str, direction: str
) -> tuple[ScenarioOrderCapability, tuple[str, ...]]:
    if mode == "paper":
        if market_type == "spot" and direction == "short":
            return "paper_only", ("spot_short_not_real_order_capable",)
        return "paper_only", ("paper_no_exchange_submit",)
    if market_type == "spot" and direction == "short":
        return "unsupported", ("spot_short_not_supported",)
    if market_type == "futures" and direction == "short":
        return "real_order_capable", ("futures_short_requires_isolated_1x_guard",)
    return "real_order_capable", ("testnet_order_path_supported_when_exchange_ready",)


def _entry_sizing_modes() -> tuple[str, ...]:
    return tuple(str(value) for value in get_args(LiveStrategyProfileSizingMethod))


def _backtest_risk_mode(*, snapshot: BacktestVariantLaunchSnapshot) -> str:
    canonical = _mapping(snapshot.canonical_variant_params)
    risk = _mapping(canonical.get("risk"))
    return str(risk.get("mode", "none")).strip().casefold()


def _backtest_direction_mode(*, snapshot: BacktestVariantLaunchSnapshot) -> str:
    canonical = _mapping(snapshot.canonical_variant_params)
    execution = _mapping(canonical.get("execution"))
    return str(execution.get("direction_mode", "long_short_reversal")).strip().casefold()


def _directions_from_backtest_mode(direction_mode: str) -> tuple[str, ...]:
    if direction_mode == "long_only":
        return ("long",)
    if direction_mode == "long_short_reversal":
        return ("long", "short")
    return ("long",)


def _mapping(value: Any) -> Mapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _scenario_key(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def scenario_matrix_row_to_json(*, row: StrategyVariantScenarioMatrixRow) -> dict[str, Any]:
    return {
        "scenario_matrix_row_id": str(row.scenario_matrix_row_id),
        "scenario_key": row.scenario_key,
        "mode": row.mode,
        "market_type": row.market_type,
        "symbol": row.symbol,
        "entry_sizing": row.entry_sizing,
        "risk_mode": row.risk_mode,
        "direction": row.direction,
        "backtest_risk_mode": row.backtest_risk_mode,
        "backtest_direction_mode": row.backtest_direction_mode,
        "scenario_state": row.scenario_state,
        "scenario_reason_codes": list(row.scenario_reason_codes),
        "order_capability": row.order_capability,
        "order_capability_reason_codes": list(row.order_capability_reason_codes),
        "compatibility_check_id": str(row.compatibility_check_id),
        "market_data_requirement_id": str(row.market_data_requirement_id),
        "compatibility_state": row.compatibility_state,
        "compatibility_reason_codes": list(row.compatibility_reason_codes),
        "market_data_state": row.market_data_state,
        "market_data_reason_codes": list(row.market_data_reason_codes),
        "launch_blocked": row.launch_blocked,
        "launch_blocked_reason": row.launch_blocked_reason,
        "checked_at": row.checked_at.astimezone(UTC).isoformat(),
    }


__all__ = [
    "SCENARIO_MATRIX_LAUNCH_RISK_MODES_V1",
    "SCENARIO_MATRIX_MIN_NOTIONAL_USD_V1",
    "SCENARIO_MATRIX_MODES_V1",
    "SCENARIO_MATRIX_SCHEMA_V1",
    "SCENARIO_MATRIX_SYMBOL_SCOPE_V1",
    "StrategyVariantScenarioMatrixReport",
    "StrategyVariantScenarioMatrixRow",
    "StrategyVariantScenarioMatrixService",
    "scenario_matrix_row_to_json",
]
