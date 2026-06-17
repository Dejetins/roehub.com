from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Sequence
from uuid import UUID

from trading.shared_kernel.primitives import UserId

ACTION_STATE_REWARD_CONTRACT_ID_V1 = "rl_trading.action_state_reward.roehub_v1"
ACTION_STATE_REWARD_CONTRACT_VERSION_V1 = 1
ACTION_NAMES_BY_ID_V1 = {
    0: "hold",
    1: "open_long",
    2: "open_short",
    3: "close",
}
_ACTION_MEANINGS_BY_ID_V1 = {
    0: "hold/no order intent",
    1: "open long for this RL strategy run",
    2: "open short for this RL strategy run",
    3: "close only this RL strategy run position",
}
RL_ACTION_COUNT_V1 = len(ACTION_NAMES_BY_ID_V1)
STATE_EXTRA_NAMES_V1: tuple[str, ...] = (
    "position",
    "unrealized",
    "time_elapsed",
    "time_remaining",
)

RlActionName = Literal["hold", "open_long", "open_short", "close"]
RlIntentKind = Literal["no_intent", "open_long", "open_short", "close"]
RlOrderSide = Literal["buy", "sell"]
RlPositionSide = Literal["long", "short"]


class RlActionContractViolation(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RlStrategyScope:
    owner_user_id: UserId
    strategy_run_id: UUID
    exchange: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        if not isinstance(self.owner_user_id, UserId):
            raise RlActionContractViolation(
                reason="invalid_owner_user_id", field="owner_user_id"
            )
        if not isinstance(self.strategy_run_id, UUID):
            raise RlActionContractViolation(
                reason="invalid_strategy_run_id", field="strategy_run_id"
            )
        exchange = _required_text(value=self.exchange, field="exchange").lower()
        market_type = _required_text(value=self.market_type, field="market_type").lower()
        symbol = _required_text(value=self.symbol, field="symbol").upper()
        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)

    @classmethod
    def from_instrument_key(
        cls,
        *,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        instrument_key: str,
    ) -> RlStrategyScope:
        parts = instrument_key.strip().split(":")
        if len(parts) != 3:
            raise RlActionContractViolation(
                reason="invalid_instrument_key", field="instrument_key"
            )
        exchange, market_type, symbol = parts
        return cls(
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
        )

    @property
    def instrument_key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"

    def identity_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            str(self.owner_user_id),
            str(self.strategy_run_id),
            self.exchange,
            self.market_type,
            self.symbol,
        )

    def matches(self, other: RlStrategyScope) -> bool:
        return self.identity_tuple() == other.identity_tuple()


@dataclass(frozen=True, slots=True)
class RlStrategyPosition:
    scope: RlStrategyScope
    side: RlPositionSide
    entry_price: float | None = None
    quantity: float | None = None

    def __post_init__(self) -> None:
        if self.side not in {"long", "short"}:
            raise RlActionContractViolation(reason="invalid_position_side", field="side")
        if self.entry_price is not None:
            _positive_float(value=self.entry_price, field="entry_price")
        if self.quantity is not None:
            _positive_float(value=self.quantity, field="quantity")


@dataclass(frozen=True, slots=True)
class RlActionResolution:
    requested_action_id: int
    action_name: RlActionName
    scope: RlStrategyScope
    intent_kind: RlIntentKind
    audit_reason: str
    order_side: RlOrderSide | None
    position_side: RlPositionSide | None


@dataclass(frozen=True, slots=True)
class RlTrainingState:
    balance: float
    position_side: RlPositionSide | None = None
    entry_price: float | None = None
    realized_pnl: float = 0.0
    closed_trades: int = 0
    profitable_trades: int = 0

    def __post_init__(self) -> None:
        _finite_float(value=self.balance, field="balance")
        _finite_float(value=self.realized_pnl, field="realized_pnl")
        if self.position_side is None:
            if self.entry_price is not None:
                raise RlActionContractViolation(
                    reason="flat_state_entry_price_must_be_none", field="entry_price"
                )
        elif self.position_side in {"long", "short"}:
            if self.entry_price is None:
                raise RlActionContractViolation(
                    reason="open_state_entry_price_required", field="entry_price"
                )
            _positive_float(value=self.entry_price, field="entry_price")
        else:
            raise RlActionContractViolation(
                reason="invalid_position_side", field="position_side"
            )
        if self.closed_trades < 0:
            raise RlActionContractViolation(
                reason="negative_closed_trades", field="closed_trades"
            )
        if self.profitable_trades < 0 or self.profitable_trades > self.closed_trades:
            raise RlActionContractViolation(
                reason="invalid_profitable_trades", field="profitable_trades"
            )


@dataclass(frozen=True, slots=True)
class RlTrainingStepResult:
    state: RlTrainingState
    reward: float
    pnl_change: float
    effective_action_id: int
    effective_action_name: RlActionName
    audit_reason: str
    inaction_penalty: float
    closed_position: bool


def action_state_reward_contract_payload_v1() -> dict[str, object]:
    return {
        "action_count": RL_ACTION_COUNT_V1,
        "actions": [
            {
                "action_id": action_id,
                "name": name,
                "roehub_meaning": _ACTION_MEANINGS_BY_ID_V1[action_id],
            }
            for action_id, name in ACTION_NAMES_BY_ID_V1.items()
        ],
        "backtest_live_distinction": {
            "live_outcome_source_of_truth": "execution_order_fill_reconciliation_ledgers",
            "paper_testnet_live_outcome_usage_v1": "monitoring_drift_and_evaluation_ledgers",
            "training_reward_source_of_truth": "offline_training_environment",
            "user_specific_live_outcomes_enter_platform_retraining_v1": False,
        },
        "contract_id": ACTION_STATE_REWARD_CONTRACT_ID_V1,
        "contract_version": ACTION_STATE_REWARD_CONTRACT_VERSION_V1,
        "execution_boundary": {
            "exchange_submission_owner": "live_execution_and_exchange_execution",
            "rl_output_source_type": "ml_agent_decision",
            "runtime_artifact_root": "/opt/roehub/state/rl_trading/",
            "secret_custody_owner": "exchange_control_live_execution_boundary",
        },
        "no_pyramiding": {
            "opposite_side_open_before_close": "no_intent_strategy_position_already_open",
            "same_side_open_while_open": "no_intent_strategy_position_already_open",
        },
        "ownership_scope": {
            "close_scope": [
                "owner_user_id",
                "strategy_run_id",
                "exchange",
                "market_type",
                "symbol",
            ],
            "multiple_strategies_same_ticker_allowed": True,
            "no_cross_strategy_close": True,
        },
        "reward_v1": {
            "formula": "pnl_change / initial_balance - flat_hold_inaction_penalty",
            "hold_flat": "inaction_penalty",
            "hold_open": "no_mark_to_market_reward",
            "last_step_flat_open": "coerce_to_hold",
            "last_step_open_position": "force_close",
            "open_action": "entry_fee_is_negative_pnl_change",
            "risk_score_rewrite": False,
        },
        "state_extras": {
            "action_history": "one_hot_last_n_actions",
            "order": list(STATE_EXTRA_NAMES_V1),
            "position_values": {"flat": 0.0, "long": 1.0, "short": -1.0},
            "unrealized": "(current_price - entry_price) * position / entry_price",
        },
    }


def action_state_reward_contract_canonical_json_v1() -> str:
    return json.dumps(
        action_state_reward_contract_payload_v1(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def action_state_reward_contract_hash_v1() -> str:
    return hashlib.sha256(
        action_state_reward_contract_canonical_json_v1().encode("utf-8")
    ).hexdigest()


ACTION_STATE_REWARD_CONTRACT_HASH_V1 = action_state_reward_contract_hash_v1()


def normalize_rl_action_id_v1(action_id: int) -> int:
    if isinstance(action_id, bool) or action_id not in ACTION_NAMES_BY_ID_V1:
        raise RlActionContractViolation(reason="unsupported_action_id", field="action_id")
    return action_id


def rl_action_name_v1(action_id: int) -> RlActionName:
    normalized = normalize_rl_action_id_v1(action_id)
    return ACTION_NAMES_BY_ID_V1[normalized]  # type: ignore[return-value]


def find_strategy_position_for_scope_v1(
    *,
    scope: RlStrategyScope,
    positions: Sequence[RlStrategyPosition],
) -> RlStrategyPosition | None:
    matches = [position for position in positions if position.scope.matches(scope)]
    if len(matches) > 1:
        raise RlActionContractViolation(
            reason="duplicate_strategy_position", field="positions"
        )
    return matches[0] if matches else None


def resolve_roehub_action_v1(
    *,
    action_id: int,
    scope: RlStrategyScope,
    positions: Sequence[RlStrategyPosition] = (),
) -> RlActionResolution:
    normalized_action_id = normalize_rl_action_id_v1(action_id)
    action_name = rl_action_name_v1(normalized_action_id)
    owned_position = find_strategy_position_for_scope_v1(scope=scope, positions=positions)

    if normalized_action_id == 0:
        return RlActionResolution(
            requested_action_id=normalized_action_id,
            action_name=action_name,
            scope=scope,
            intent_kind="no_intent",
            audit_reason="hold_no_order_intent",
            order_side=None,
            position_side=owned_position.side if owned_position is not None else None,
        )
    if normalized_action_id in {1, 2}:
        if owned_position is not None:
            return RlActionResolution(
                requested_action_id=normalized_action_id,
                action_name=action_name,
                scope=scope,
                intent_kind="no_intent",
                audit_reason="strategy_position_already_open",
                order_side=None,
                position_side=owned_position.side,
            )
        return RlActionResolution(
            requested_action_id=normalized_action_id,
            action_name=action_name,
            scope=scope,
            intent_kind=action_name,  # type: ignore[arg-type]
            audit_reason="intent_allowed",
            order_side="buy" if normalized_action_id == 1 else "sell",
            position_side="long" if normalized_action_id == 1 else "short",
        )

    if owned_position is None:
        return RlActionResolution(
            requested_action_id=normalized_action_id,
            action_name=action_name,
            scope=scope,
            intent_kind="no_intent",
            audit_reason="no_strategy_position",
            order_side=None,
            position_side=None,
        )
    return RlActionResolution(
        requested_action_id=normalized_action_id,
        action_name=action_name,
        scope=scope,
        intent_kind="close",
        audit_reason="intent_allowed",
        order_side="sell" if owned_position.side == "long" else "buy",
        position_side=owned_position.side,
    )


def build_state_extras_v1(
    *,
    position_side: RlPositionSide | None,
    entry_price: float | None,
    current_price: float,
    step_idx: int,
    session_len: int,
) -> tuple[float, float, float, float]:
    position_value = _position_value(position_side=position_side)
    current = _positive_float(value=current_price, field="current_price")
    if session_len <= 0:
        raise RlActionContractViolation(reason="invalid_session_len", field="session_len")
    if step_idx < 0 or step_idx > session_len:
        raise RlActionContractViolation(reason="invalid_step_idx", field="step_idx")
    if position_side is None:
        unrealized = 0.0
    else:
        entry = _positive_float(value=entry_price, field="entry_price")
        unrealized = ((current - entry) * position_value) / entry
    return (
        position_value,
        unrealized,
        float(step_idx) / float(session_len),
        float(session_len - step_idx) / float(session_len),
    )


def encode_action_history_v1(actions: Sequence[int | None]) -> tuple[float, ...]:
    encoded = [0.0] * (len(actions) * RL_ACTION_COUNT_V1)
    for idx, action_id in enumerate(actions):
        if action_id is None:
            continue
        normalized = normalize_rl_action_id_v1(action_id)
        encoded[idx * RL_ACTION_COUNT_V1 + normalized] = 1.0
    return tuple(encoded)


def apply_training_reward_step_v1(
    *,
    state: RlTrainingState,
    action_id: int,
    price: float,
    initial_balance: float,
    slippage: float,
    transaction_fee: float,
    inaction_penalty_ratio: float,
    is_last_step: bool = False,
) -> RlTrainingStepResult:
    requested_action_id = normalize_rl_action_id_v1(action_id)
    effective_action_id, audit_reason = coerce_last_step_action_v1(
        action_id=requested_action_id,
        position_side=state.position_side,
        is_last_step=is_last_step,
    )
    current_price = _positive_float(value=price, field="price")
    initial = _positive_float(value=initial_balance, field="initial_balance")
    slip = _non_negative_float(value=slippage, field="slippage")
    fee_rate = _non_negative_float(value=transaction_fee, field="transaction_fee")
    flat_hold_penalty = _non_negative_float(
        value=inaction_penalty_ratio,
        field="inaction_penalty_ratio",
    )

    prev_position_side = state.position_side
    balance = state.balance
    entry_price = state.entry_price
    realized_pnl = state.realized_pnl
    closed_trades = state.closed_trades
    profitable_trades = state.profitable_trades
    pnl_change = 0.0
    closed_position = False

    if effective_action_id == 1 and prev_position_side is None:
        exec_price = current_price * (1.0 + slip)
        volume = balance / exec_price
        pnl_change -= exec_price * volume * fee_rate
        balance += pnl_change
        position_side: RlPositionSide | None = "long"
        entry_price = exec_price
    elif effective_action_id == 2 and prev_position_side is None:
        exec_price = current_price * (1.0 - slip)
        volume = balance / exec_price
        pnl_change -= exec_price * volume * fee_rate
        balance += pnl_change
        position_side = "short"
        entry_price = exec_price
    elif effective_action_id == 3 and prev_position_side is not None:
        if entry_price is None:
            raise RlActionContractViolation(
                reason="open_state_entry_price_required", field="entry_price"
            )
        volume = balance / entry_price
        if prev_position_side == "long":
            exec_price = current_price * (1.0 - slip)
            trade_pnl = (exec_price - entry_price) * volume
        else:
            exec_price = current_price * (1.0 + slip)
            trade_pnl = (entry_price - exec_price) * volume
        pnl_change += trade_pnl - exec_price * volume * fee_rate
        balance += pnl_change
        realized_pnl += pnl_change
        closed_trades += 1
        if trade_pnl > 0.0:
            profitable_trades += 1
        closed_position = True
        position_side = None
        entry_price = None
    else:
        position_side = prev_position_side

    inaction_penalty = (
        flat_hold_penalty
        if effective_action_id == 0 and prev_position_side is None
        else 0.0
    )
    reward = (pnl_change / initial) - inaction_penalty

    return RlTrainingStepResult(
        state=RlTrainingState(
            balance=balance,
            position_side=position_side,
            entry_price=entry_price,
            realized_pnl=realized_pnl,
            closed_trades=closed_trades,
            profitable_trades=profitable_trades,
        ),
        reward=reward,
        pnl_change=pnl_change,
        effective_action_id=effective_action_id,
        effective_action_name=rl_action_name_v1(effective_action_id),
        audit_reason=audit_reason,
        inaction_penalty=inaction_penalty,
        closed_position=closed_position,
    )


def coerce_last_step_action_v1(
    *,
    action_id: int,
    position_side: RlPositionSide | None,
    is_last_step: bool,
) -> tuple[int, str]:
    normalized = normalize_rl_action_id_v1(action_id)
    if not is_last_step:
        return normalized, "requested_action"
    if position_side is None and normalized in {1, 2}:
        return 0, "last_step_open_blocked_as_hold"
    if position_side is not None and normalized != 3:
        return 3, "last_step_forced_close"
    return normalized, "requested_action"


def _position_value(*, position_side: RlPositionSide | None) -> float:
    if position_side is None:
        return 0.0
    if position_side == "long":
        return 1.0
    if position_side == "short":
        return -1.0
    raise RlActionContractViolation(reason="invalid_position_side", field="position_side")


def _required_text(*, value: str, field: str) -> str:
    if not isinstance(value, str):
        raise RlActionContractViolation(reason="invalid_text_field", field=field)
    stripped = value.strip()
    if not stripped:
        raise RlActionContractViolation(reason="required_text_field", field=field)
    return stripped


def _finite_float(*, value: float | None, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise RlActionContractViolation(reason="invalid_numeric_field", field=field)
    out = float(value)
    if not math.isfinite(out):
        raise RlActionContractViolation(reason="non_finite_numeric_field", field=field)
    return out


def _positive_float(*, value: float | None, field: str) -> float:
    out = _finite_float(value=value, field=field)
    if out <= 0.0:
        raise RlActionContractViolation(reason="non_positive_numeric_field", field=field)
    return out


def _non_negative_float(*, value: float, field: str) -> float:
    out = _finite_float(value=value, field=field)
    if out < 0.0:
        raise RlActionContractViolation(reason="negative_numeric_field", field=field)
    return out
