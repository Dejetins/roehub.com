from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from trading.contexts.strategy.domain import (
    Strategy,
    StrategyActiveRunConflictError,
    StrategyEvent,
    StrategyRun,
    StrategyRunTransitionError,
    StrategySpecV1,
    StrategySpecValidationError,
    ensure_single_active_run,
    generate_strategy_name,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe, UserId


def test_strategy_name_is_deterministic_for_same_user_and_spec() -> None:
    """
    Verify deterministic strategy name generation from `user_id + spec_json`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Name format must stay `BTCUSDT 1m [MA(20,50)] #<8HEX>` for same input payload.
    Raises:
        AssertionError: If determinism or format contract is broken.
    Side Effects:
        None.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000123")
    spec = _build_spec()

    generated_name_a = generate_strategy_name(user_id=user_id, spec=spec)
    generated_name_b = generate_strategy_name(user_id=user_id, spec=spec)

    assert generated_name_a == generated_name_b
    assert generated_name_a.startswith("BTCUSDT 1m [MA(20,50)] #")
    suffix = generated_name_a.split("#", maxsplit=1)[1]
    assert len(suffix) == 8
    assert suffix == suffix.upper()


def test_strategy_create_enforces_immutable_deterministic_name_contract() -> None:
    """
    Verify strategy aggregate validates generated name and rejects manual mismatch.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Strategy aggregate must remain immutable by spec and deterministic name generation.
    Raises:
        AssertionError: If aggregate accepts invalid name.
    Side Effects:
        None.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000123")
    spec = _build_spec()
    created_at = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)

    strategy = Strategy.create(
        user_id=user_id,
        spec=spec,
        created_at=created_at,
        strategy_id=UUID("00000000-0000-0000-0000-00000000A001"),
    )
    assert strategy.name.startswith("BTCUSDT 1m [MA(20,50)] #")

    with pytest.raises(StrategySpecValidationError):
        Strategy(
            strategy_id=strategy.strategy_id,
            user_id=user_id,
            name="manual mutable name",
            spec=spec,
            created_at=created_at,
            is_deleted=False,
        )


def test_strategy_spec_requires_schema_version_1_and_allows_timeframe_1m() -> None:
    """
    Verify `schema_version=1` invariant and explicit allowance for timeframe `1m`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Strategy v1 spec contract requires immutable schema version 1.
    Raises:
        AssertionError: If invariant validation is broken.
    Side Effects:
        None.
    """
    payload = _build_spec().to_json()
    payload["schema_version"] = 2

    with pytest.raises(StrategySpecValidationError):
        StrategySpecV1.from_json(payload=payload)

    valid = _build_spec()
    assert valid.timeframe.code == "1m"


def test_strategy_run_state_machine_allows_only_documented_transitions() -> None:
    """
    Verify run state machine transitions for v1 path and invalid transition rejection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Supported path is `starting -> warming_up -> running -> stopping -> stopped|failed`.
    Raises:
        AssertionError: If state machine contract is violated.
    Side Effects:
        None.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000124")
    started_at = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)
    run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000B001"),
        user_id=user_id,
        strategy_id=UUID("00000000-0000-0000-0000-00000000A001"),
        started_at=started_at,
    )

    warming = run.transition_to(
        next_state="warming_up",
        changed_at=datetime(2026, 2, 15, 11, 1, tzinfo=timezone.utc),
    )
    running = warming.transition_to(
        next_state="running",
        changed_at=datetime(2026, 2, 15, 11, 2, tzinfo=timezone.utc),
    )
    stopping = running.transition_to(
        next_state="stopping",
        changed_at=datetime(2026, 2, 15, 11, 3, tzinfo=timezone.utc),
    )
    stopped = stopping.transition_to(
        next_state="stopped",
        changed_at=datetime(2026, 2, 15, 11, 4, tzinfo=timezone.utc),
    )

    assert run.state == "starting"
    assert warming.state == "warming_up"
    assert running.state == "running"
    assert stopping.state == "stopping"
    assert stopped.state == "stopped"
    assert stopped.stopped_at == datetime(2026, 2, 15, 11, 4, tzinfo=timezone.utc)

    with pytest.raises(StrategyRunTransitionError):
        run.transition_to(
            next_state="running",
            changed_at=datetime(2026, 2, 15, 11, 1, tzinfo=timezone.utc),
        )

    with pytest.raises(StrategyRunTransitionError):
        StrategyRun(
            run_id=UUID("00000000-0000-0000-0000-00000000B002"),
            user_id=user_id,
            strategy_id=UUID("00000000-0000-0000-0000-00000000A001"),
            state="paused",  # type: ignore[arg-type]
            started_at=started_at,
            stopped_at=None,
            checkpoint_ts_open=None,
            last_error=None,
            updated_at=started_at,
        )


def test_single_active_run_invariant_rejects_second_active_run() -> None:
    """
    Verify invariant helper rejects second active run for same strategy.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Exactly one active run is allowed by Strategy v1 contract.
    Raises:
        AssertionError: If conflict check does not raise.
    Side Effects:
        None.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000124")
    base_time = datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc)

    active_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000B010"),
        user_id=user_id,
        strategy_id=UUID("00000000-0000-0000-0000-00000000A010"),
        started_at=base_time,
    )
    another_active_run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000B011"),
        user_id=user_id,
        strategy_id=UUID("00000000-0000-0000-0000-00000000A010"),
        started_at=datetime(2026, 2, 15, 12, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(StrategyActiveRunConflictError):
        ensure_single_active_run(existing_runs=[active_run], candidate_run=another_active_run)


def test_strategy_events_allow_nullable_run_id_and_are_json_serializable() -> None:
    """
    Verify append-only event supports strategy-level events with `run_id=None`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `run_id` is nullable by v1 storage and domain contract.
    Raises:
        AssertionError: If event creation or invariants fail.
    Side Effects:
        None.
    """
    event = StrategyEvent.create(
        event_id=UUID("00000000-0000-0000-0000-00000000E001"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000129"),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A020"),
        run_id=None,
        ts=datetime(2026, 2, 15, 12, 30, tzinfo=timezone.utc),
        event_type="strategy_created",
        payload_json={"schema_version": 1},
    )

    assert event.run_id is None
    assert event.payload_json["schema_version"] == 1



def _build_spec() -> StrategySpecV1:
    """
    Build deterministic StrategySpecV1 fixture for domain tests.

    Args:
        None.
    Returns:
        StrategySpecV1: Valid immutable spec fixture.
    Assumptions:
        Fixture uses canonical BTCUSDT+1m+MA(20,50) example from architecture docs.
    Raises:
        StrategySpecValidationError: If fixture values violate domain invariants.
    Side Effects:
        None.
    """
    return StrategySpecV1(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        timeframe=Timeframe("1m"),
        indicators=(
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            },
        ),
        signal_template="MA(20,50)",
        schema_version=1,
    )
