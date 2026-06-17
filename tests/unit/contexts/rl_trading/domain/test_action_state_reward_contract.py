from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest

from trading.contexts.rl_trading.domain import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
    RlStrategyPosition,
    RlStrategyScope,
    RlTrainingState,
    action_state_reward_contract_payload_v1,
    apply_training_reward_step_v1,
    build_state_extras_v1,
    encode_action_history_v1,
    resolve_roehub_action_v1,
)
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000002001")
_RUN_ID = UUID("00000000-0000-0000-0000-000000002101")
_OTHER_RUN_ID = UUID("00000000-0000-0000-0000-000000002102")


def test_action_state_reward_contract_hash_and_required_literals_are_stable() -> None:
    payload = cast(dict[str, Any], action_state_reward_contract_payload_v1())

    assert ACTION_NAMES_BY_ID_V1 == {
        0: "hold",
        1: "open_long",
        2: "open_short",
        3: "close",
    }
    assert ACTION_STATE_REWARD_CONTRACT_HASH_V1 == (
        "255d765b9474620671167412465fc55a058c0233d5da242a276143fb6816b557"
    )
    assert payload["execution_boundary"]["rl_output_source_type"] == "ml_agent_decision"
    assert payload["execution_boundary"]["runtime_artifact_root"] == (
        "/opt/roehub/state/rl_trading/"
    )
    assert payload["backtest_live_distinction"][
        "user_specific_live_outcomes_enter_platform_retraining_v1"
    ] is False


def test_roehub_action_semantics_hold_open_close_and_no_pyramiding() -> None:
    scope = _scope()

    hold = resolve_roehub_action_v1(action_id=0, scope=scope)
    open_long = resolve_roehub_action_v1(action_id=1, scope=scope)
    owned_long = RlStrategyPosition(scope=scope, side="long")
    repeated_open = resolve_roehub_action_v1(
        action_id=1,
        scope=scope,
        positions=(owned_long,),
    )
    opposite_open = resolve_roehub_action_v1(
        action_id=2,
        scope=scope,
        positions=(owned_long,),
    )
    close = resolve_roehub_action_v1(action_id=3, scope=scope, positions=(owned_long,))
    close_flat = resolve_roehub_action_v1(action_id=3, scope=scope)

    assert hold.intent_kind == "no_intent"
    assert hold.audit_reason == "hold_no_order_intent"
    assert open_long.intent_kind == "open_long"
    assert open_long.order_side == "buy"
    assert open_long.position_side == "long"
    assert repeated_open.intent_kind == "no_intent"
    assert repeated_open.audit_reason == "strategy_position_already_open"
    assert opposite_open.intent_kind == "no_intent"
    assert opposite_open.audit_reason == "strategy_position_already_open"
    assert close.intent_kind == "close"
    assert close.order_side == "sell"
    assert close.position_side == "long"
    assert close_flat.intent_kind == "no_intent"
    assert close_flat.audit_reason == "no_strategy_position"


def test_close_is_scoped_to_owner_strategy_exchange_market_and_symbol() -> None:
    scope = _scope()
    other_run_same_ticker = RlStrategyPosition(scope=_scope(run_id=_OTHER_RUN_ID), side="long")
    same_run_other_market = RlStrategyPosition(scope=_scope(market_type="spot"), side="long")
    same_run_other_symbol = RlStrategyPosition(scope=_scope(symbol="ETHUSDT"), side="long")
    positions = (other_run_same_ticker, same_run_other_market, same_run_other_symbol)

    close = resolve_roehub_action_v1(action_id=3, scope=scope, positions=positions)
    open_short = resolve_roehub_action_v1(action_id=2, scope=scope, positions=positions)

    assert close.intent_kind == "no_intent"
    assert close.audit_reason == "no_strategy_position"
    assert open_short.intent_kind == "open_short"
    assert open_short.order_side == "sell"
    assert open_short.position_side == "short"


def test_state_extras_and_action_history_match_external_environment_shape() -> None:
    flat = build_state_extras_v1(
        position_side=None,
        entry_price=None,
        current_price=100.0,
        step_idx=3,
        session_len=10,
    )
    long = build_state_extras_v1(
        position_side="long",
        entry_price=100.0,
        current_price=110.0,
        step_idx=3,
        session_len=10,
    )
    short = build_state_extras_v1(
        position_side="short",
        entry_price=100.0,
        current_price=90.0,
        step_idx=3,
        session_len=10,
    )

    assert flat == pytest.approx((0.0, 0.0, 0.3, 0.7))
    assert long == pytest.approx((1.0, 0.1, 0.3, 0.7))
    assert short == pytest.approx((-1.0, 0.1, 0.3, 0.7))
    assert encode_action_history_v1([None, 0, 3]) == (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_training_reward_open_fee_and_flat_hold_penalty_match_v1() -> None:
    opened = apply_training_reward_step_v1(
        state=RlTrainingState(balance=100.0),
        action_id=1,
        price=10.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=0.01,
    )
    held_flat = apply_training_reward_step_v1(
        state=RlTrainingState(balance=100.0),
        action_id=0,
        price=10.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=0.01,
    )

    assert opened.effective_action_name == "open_long"
    assert opened.pnl_change == pytest.approx(-0.1)
    assert opened.reward == pytest.approx(-0.001)
    assert opened.state.balance == pytest.approx(99.9)
    assert opened.state.position_side == "long"
    assert opened.state.entry_price == pytest.approx(10.0)
    assert held_flat.reward == pytest.approx(-0.01)
    assert held_flat.inaction_penalty == pytest.approx(0.01)
    assert held_flat.state.position_side is None


def test_hold_while_open_has_no_mark_to_market_reward() -> None:
    held_open = apply_training_reward_step_v1(
        state=RlTrainingState(balance=99.9, position_side="long", entry_price=100.0),
        action_id=0,
        price=150.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=0.01,
    )

    assert held_open.reward == pytest.approx(0.0)
    assert held_open.pnl_change == pytest.approx(0.0)
    assert held_open.inaction_penalty == pytest.approx(0.0)
    assert held_open.state.balance == pytest.approx(99.9)
    assert held_open.state.position_side == "long"


def test_close_realizes_trade_pnl_minus_fee_as_training_reward() -> None:
    opened = apply_training_reward_step_v1(
        state=RlTrainingState(balance=100.0),
        action_id=1,
        price=10.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=0.01,
    )

    closed = apply_training_reward_step_v1(
        state=opened.state,
        action_id=3,
        price=11.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.001,
        inaction_penalty_ratio=0.01,
    )

    assert closed.closed_position is True
    assert closed.pnl_change == pytest.approx(9.88011)
    assert closed.reward == pytest.approx(0.0988011)
    assert closed.state.balance == pytest.approx(109.78011)
    assert closed.state.realized_pnl == pytest.approx(9.88011)
    assert closed.state.closed_trades == 1
    assert closed.state.profitable_trades == 1
    assert closed.state.position_side is None


def test_last_step_blocks_new_open_and_forces_close_existing_position() -> None:
    blocked_open = apply_training_reward_step_v1(
        state=RlTrainingState(balance=100.0),
        action_id=1,
        price=10.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.0,
        inaction_penalty_ratio=0.01,
        is_last_step=True,
    )
    forced_close = apply_training_reward_step_v1(
        state=RlTrainingState(balance=100.0, position_side="long", entry_price=10.0),
        action_id=0,
        price=11.0,
        initial_balance=100.0,
        slippage=0.0,
        transaction_fee=0.0,
        inaction_penalty_ratio=0.01,
        is_last_step=True,
    )

    assert blocked_open.effective_action_id == 0
    assert blocked_open.audit_reason == "last_step_open_blocked_as_hold"
    assert blocked_open.reward == pytest.approx(-0.01)
    assert blocked_open.state.position_side is None
    assert forced_close.effective_action_id == 3
    assert forced_close.audit_reason == "last_step_forced_close"
    assert forced_close.closed_position is True
    assert forced_close.reward == pytest.approx(0.1)
    assert forced_close.state.position_side is None


def _scope(
    *,
    run_id: UUID = _RUN_ID,
    exchange: str = "binance",
    market_type: str = "futures",
    symbol: str = "BTCUSDT",
) -> RlStrategyScope:
    return RlStrategyScope(
        owner_user_id=_USER_ID,
        strategy_run_id=run_id,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
    )
