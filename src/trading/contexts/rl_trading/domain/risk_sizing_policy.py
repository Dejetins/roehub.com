from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

STAGE14_SCHEMA_VERSION_V1 = 1
STAGE14_POLICY_KIND_V1 = "rl_risk_sizing_policy_v1"
STAGE14_SYNTHETIC_EXIT_KIND_V1 = "rl_synthetic_platform_exit_v1"

RL_RISK_POLICY_READY = "rl_risk_policy_ready"
RL_RISK_POLICY_NOT_CONFIGURED = "rl_risk_policy_not_configured"
RL_RISK_POLICY_INACTIVE = "rl_risk_policy_inactive"
RL_RISK_POLICY_MONITOR_ONLY_NO_INTENT = "monitor_only_no_intent"

RlRiskSizingMethod = Literal["fixed_quote", "fixed_equity_pct"]
RlRiskPolicyStatus = Literal["ready", "blocked"]
RlMlMode = Literal["monitor_only", "paper", "testnet", "live"]
RlActionName = Literal["hold", "open_long", "open_short", "close"]
RlSyntheticExitRuleType = Literal["take_profit", "stop_loss", "trailing_stop"]
RlPolicyIntentPreviewStatus = Literal["no_intent", "blocked", "would_create_open_intent"]

_VALID_MARKET_TYPES = frozenset({"spot", "futures"})
_VALID_SIZING_METHODS = frozenset({"fixed_quote", "fixed_equity_pct"})
_MAX_RATIO = Decimal("1")
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class RlRiskSizingPolicyKey:
    organization_id: OrganizationId
    owner_user_id: UserId
    strategy_id: UUID
    exchange_name: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        exchange_name = self.exchange_name.strip().lower()
        market_type = self.market_type.strip().lower()
        symbol = self.symbol.strip().upper()
        if not exchange_name:
            raise ValueError("RlRiskSizingPolicyKey.exchange_name must be non-empty")
        if market_type not in _VALID_MARKET_TYPES:
            raise ValueError("RlRiskSizingPolicyKey.market_type must be spot or futures")
        if not symbol:
            raise ValueError("RlRiskSizingPolicyKey.symbol must be non-empty")
        object.__setattr__(self, "exchange_name", exchange_name)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)

    @property
    def persistence_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            str(self.organization_id),
            str(self.owner_user_id),
            str(self.strategy_id),
            self.exchange_name,
            self.market_type,
            self.symbol,
        )


@dataclass(frozen=True, slots=True)
class RlSyntheticExitRule:
    rule_type: RlSyntheticExitRuleType
    trigger_pct: Decimal
    platform_side: bool = True
    creates_intent_action: Literal["close"] = "close"

    def as_payload(self) -> dict[str, str | bool]:
        return {
            "kind": STAGE14_SYNTHETIC_EXIT_KIND_V1,
            "rule_type": self.rule_type,
            "trigger_pct": str(self.trigger_pct),
            "platform_side": self.platform_side,
            "creates_intent_action": self.creates_intent_action,
        }


@dataclass(frozen=True, slots=True)
class RlRiskSizingPolicyConfig:
    sizing_method: RlRiskSizingMethod = "fixed_quote"
    base_quote_notional: Decimal = Decimal("0")
    max_position_notional: Decimal = Decimal("0")
    max_daily_loss_notional: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    max_turnover_notional: Decimal = Decimal("0")
    max_exposure_notional: Decimal = Decimal("0")
    min_expected_pnl_pct: Decimal = Decimal("0")
    min_confidence: Decimal | None = None
    take_profit_pct: Decimal | None = None
    stop_loss_pct: Decimal | None = None
    trailing_stop_pct: Decimal | None = None
    active: bool = True


@dataclass(frozen=True, slots=True)
class RlRiskSizingPolicyValidation:
    status: RlRiskPolicyStatus
    reasons: tuple[str, ...]
    synthetic_exit_rules: tuple[RlSyntheticExitRule, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True, slots=True)
class RlRiskSizingPolicyRecord:
    policy_id: UUID | None
    key: RlRiskSizingPolicyKey
    config: RlRiskSizingPolicyConfig
    validation: RlRiskSizingPolicyValidation
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class RlPolicyDecisionInput:
    mode: RlMlMode
    action_name: RlActionName
    confidence: Decimal | None = None
    expected_pnl_pct: Decimal | None = None


@dataclass(frozen=True, slots=True)
class RlSizedOrderIntentPreview:
    status: RlPolicyIntentPreviewStatus
    reason: str
    order_type: Literal["market"] | None
    side: Literal["buy", "sell"] | None
    quote_notional: Decimal | None
    synthetic_exit_rules: tuple[RlSyntheticExitRule, ...]
    advanced_order_flags: dict[str, object]


class RlRiskSizingPolicyRepository(Protocol):
    def get_policy(self, *, key: RlRiskSizingPolicyKey) -> RlRiskSizingPolicyRecord | None: ...

    def upsert_policy(
        self,
        *,
        key: RlRiskSizingPolicyKey,
        config: RlRiskSizingPolicyConfig,
        observed_at: datetime,
    ) -> RlRiskSizingPolicyRecord: ...


class RlRiskSizingPolicyService:
    def __init__(self, *, repository: RlRiskSizingPolicyRepository) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RlRiskSizingPolicyService requires repository")
        self._repository = repository

    def get_policy(self, *, key: RlRiskSizingPolicyKey) -> RlRiskSizingPolicyRecord:
        record = self._repository.get_policy(key=key)
        if record is not None:
            return record
        return default_blocked_rl_risk_sizing_policy(key=key)

    def upsert_policy(
        self,
        *,
        key: RlRiskSizingPolicyKey,
        config: RlRiskSizingPolicyConfig,
        observed_at: datetime,
    ) -> RlRiskSizingPolicyRecord:
        return self._repository.upsert_policy(
            key=key,
            config=config,
            observed_at=observed_at,
        )


def default_blocked_rl_risk_sizing_policy(
    *, key: RlRiskSizingPolicyKey
) -> RlRiskSizingPolicyRecord:
    config = RlRiskSizingPolicyConfig(active=False)
    return RlRiskSizingPolicyRecord(
        policy_id=None,
        key=key,
        config=config,
        validation=RlRiskSizingPolicyValidation(
            status="blocked",
            reasons=(RL_RISK_POLICY_NOT_CONFIGURED,),
            synthetic_exit_rules=(),
        ),
        created_at=None,
        updated_at=None,
    )


def validate_rl_risk_sizing_policy(
    *, config: RlRiskSizingPolicyConfig
) -> RlRiskSizingPolicyValidation:
    reasons: list[str] = []
    if not config.active:
        reasons.append(RL_RISK_POLICY_INACTIVE)
    if config.sizing_method not in _VALID_SIZING_METHODS:
        reasons.append("rl_risk_policy_unsupported_sizing_method")
    _require_positive(
        value=config.base_quote_notional,
        reason="rl_risk_policy_base_quote_notional_required",
        reasons=reasons,
    )
    _require_positive(
        value=config.max_position_notional,
        reason="rl_risk_policy_max_position_notional_required",
        reasons=reasons,
    )
    _require_positive(
        value=config.max_daily_loss_notional,
        reason="rl_risk_policy_max_daily_loss_required",
        reasons=reasons,
    )
    _require_positive_ratio(
        value=config.max_drawdown_pct,
        reason="rl_risk_policy_max_drawdown_pct_required",
        reasons=reasons,
    )
    _require_positive(
        value=config.max_turnover_notional,
        reason="rl_risk_policy_max_turnover_required",
        reasons=reasons,
    )
    _require_positive(
        value=config.max_exposure_notional,
        reason="rl_risk_policy_max_exposure_required",
        reasons=reasons,
    )
    _require_ratio(
        value=config.min_expected_pnl_pct,
        reason="rl_risk_policy_min_expected_pnl_pct_invalid",
        reasons=reasons,
    )
    if config.min_confidence is not None:
        _require_ratio(
            value=config.min_confidence,
            reason="rl_risk_policy_min_confidence_invalid",
            reasons=reasons,
        )
    if (
        config.base_quote_notional > _ZERO
        and config.max_position_notional > _ZERO
        and config.base_quote_notional > config.max_position_notional
    ):
        reasons.append("rl_risk_policy_base_quote_exceeds_position_cap")
    if (
        config.max_position_notional > _ZERO
        and config.max_exposure_notional > _ZERO
        and config.max_position_notional > config.max_exposure_notional
    ):
        reasons.append("rl_risk_policy_position_cap_exceeds_exposure_cap")
    if (
        config.max_position_notional > _ZERO
        and config.max_turnover_notional > _ZERO
        and config.max_position_notional > config.max_turnover_notional
    ):
        reasons.append("rl_risk_policy_position_cap_exceeds_turnover_cap")

    synthetic_exit_rules = _synthetic_exit_rules(config=config, reasons=reasons)
    if config.stop_loss_pct is None:
        reasons.append("rl_risk_policy_stop_loss_required")

    return RlRiskSizingPolicyValidation(
        status="ready" if not reasons else "blocked",
        reasons=(RL_RISK_POLICY_READY,) if not reasons else tuple(dict.fromkeys(reasons)),
        synthetic_exit_rules=synthetic_exit_rules if not reasons else (),
    )


def evaluate_rl_decision_policy(
    *,
    config: RlRiskSizingPolicyConfig,
    decision: RlPolicyDecisionInput,
) -> RlSizedOrderIntentPreview:
    validation = validate_rl_risk_sizing_policy(config=config)
    if decision.mode == "monitor_only":
        return _preview(
            status="no_intent",
            reason=RL_RISK_POLICY_MONITOR_ONLY_NO_INTENT,
            synthetic_exit_rules=validation.synthetic_exit_rules,
        )
    if not validation.ready:
        return _preview(
            status="blocked",
            reason=validation.reasons[0],
            synthetic_exit_rules=(),
        )
    if decision.action_name == "hold":
        return _preview(
            status="no_intent",
            reason="rl_action_hold_no_intent",
            synthetic_exit_rules=validation.synthetic_exit_rules,
        )
    if decision.action_name == "close":
        return _preview(
            status="no_intent",
            reason="rl_close_requires_existing_strategy_position",
            synthetic_exit_rules=validation.synthetic_exit_rules,
        )
    if config.min_confidence is not None and (
        decision.confidence is None or decision.confidence < config.min_confidence
    ):
        return _preview(
            status="no_intent",
            reason="rl_decision_confidence_below_policy_threshold",
            synthetic_exit_rules=validation.synthetic_exit_rules,
        )
    if decision.expected_pnl_pct is not None and (
        decision.expected_pnl_pct < config.min_expected_pnl_pct
    ):
        return _preview(
            status="no_intent",
            reason="rl_decision_expected_pnl_below_policy_threshold",
            synthetic_exit_rules=validation.synthetic_exit_rules,
        )
    quote_notional = min(
        config.base_quote_notional,
        config.max_position_notional,
        config.max_exposure_notional,
        config.max_turnover_notional,
    )
    if quote_notional <= _ZERO:
        return _preview(
            status="blocked",
            reason="rl_risk_policy_effective_quote_notional_invalid",
            synthetic_exit_rules=(),
        )
    side: Literal["buy", "sell"] = "buy" if decision.action_name == "open_long" else "sell"
    return RlSizedOrderIntentPreview(
        status="would_create_open_intent",
        reason="rl_policy_would_create_open_intent",
        order_type="market",
        side=side,
        quote_notional=quote_notional,
        synthetic_exit_rules=validation.synthetic_exit_rules,
        advanced_order_flags={},
    )


def _synthetic_exit_rules(
    *,
    config: RlRiskSizingPolicyConfig,
    reasons: list[str],
) -> tuple[RlSyntheticExitRule, ...]:
    rules: list[RlSyntheticExitRule] = []
    configured_rules: tuple[tuple[RlSyntheticExitRuleType, Decimal | None], ...] = (
        ("take_profit", config.take_profit_pct),
        ("stop_loss", config.stop_loss_pct),
        ("trailing_stop", config.trailing_stop_pct),
    )
    for rule_type, value in configured_rules:
        if value is None:
            continue
        if value <= _ZERO or value > _MAX_RATIO:
            reasons.append(f"rl_risk_policy_{rule_type}_pct_invalid")
            continue
        rules.append(RlSyntheticExitRule(rule_type=rule_type, trigger_pct=value))
    return tuple(rules)


def _require_positive(*, value: Decimal, reason: str, reasons: list[str]) -> None:
    if value <= _ZERO:
        reasons.append(reason)


def _require_positive_ratio(*, value: Decimal, reason: str, reasons: list[str]) -> None:
    if value <= _ZERO or value > _MAX_RATIO:
        reasons.append(reason)


def _require_ratio(*, value: Decimal, reason: str, reasons: list[str]) -> None:
    if value < _ZERO or value > _MAX_RATIO:
        reasons.append(reason)


def _preview(
    *,
    status: Literal["no_intent", "blocked"],
    reason: str,
    synthetic_exit_rules: tuple[RlSyntheticExitRule, ...],
) -> RlSizedOrderIntentPreview:
    return RlSizedOrderIntentPreview(
        status=status,
        reason=reason,
        order_type=None,
        side=None,
        quote_notional=None,
        synthetic_exit_rules=synthetic_exit_rules,
        advanced_order_flags={},
    )
