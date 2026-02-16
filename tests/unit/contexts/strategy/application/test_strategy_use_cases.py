from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import pytest

from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
)
from trading.contexts.strategy.application import (
    CloneStrategyUseCase,
    CreateStrategyUseCase,
    CurrentUser,
    GetMyStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
    estimate_strategy_warmup_bars,
)
from trading.contexts.strategy.domain.entities import StrategySpecV1
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


class _SequenceClock:
    """
    Deterministic UTC clock stub returning preconfigured timestamps in FIFO order.
    """

    def __init__(self, *, values: tuple[datetime, ...]) -> None:
        """
        Initialize deterministic timestamp queue.

        Args:
            values: Ordered UTC datetimes to return on each `now()` call.
        Returns:
            None.
        Assumptions:
            Values are timezone-aware UTC datetimes.
        Raises:
            ValueError: If no values are provided.
        Side Effects:
            Stores mutable internal queue state.
        """
        if not values:
            raise ValueError("_SequenceClock requires at least one value")
        self._values = list(values)

    def now(self) -> datetime:
        """
        Return next configured UTC datetime value.

        Args:
            None.
        Returns:
            datetime: Next queued timestamp.
        Assumptions:
            Tests provide enough values for all expected calls.
        Raises:
            ValueError: If queue is exhausted.
        Side Effects:
            Pops one timestamp from internal queue.
        """
        if not self._values:
            raise ValueError("_SequenceClock exhausted")
        return self._values.pop(0)



def test_get_my_strategy_use_case_rejects_non_owner_access() -> None:
    """
    Verify explicit ownership rule denies strategy access to non-owner user.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case must enforce owner checks explicitly and not rely on SQL-only scoping.
    Raises:
        AssertionError: If non-owner access does not return forbidden RoehubError.
    Side Effects:
        None.
    """
    strategy_repository = InMemoryStrategyRepository()
    event_repository = InMemoryStrategyEventRepository()
    create_use_case = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=_SequenceClock(values=(datetime(2026, 2, 16, 10, 0, tzinfo=timezone.utc),)),
    )
    get_use_case = GetMyStrategyUseCase(repository=strategy_repository)

    owner = CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000101"))
    another_user = CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000202"))

    created_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=owner,
    )

    with pytest.raises(RoehubError) as error_info:
        get_use_case.execute(
            strategy_id=created_strategy.strategy_id,
            current_user=another_user,
        )

    assert error_info.value.code == "forbidden"
    assert error_info.value.details == {"strategy_id": str(created_strategy.strategy_id)}



def test_clone_strategy_applies_whitelisted_overrides_and_rejects_unknown_fields() -> None:
    """
    Verify clone use-case applies instrument/timeframe overrides and rejects unknown override keys.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Clone request supports only explicit whitelist overrides in Strategy API v1.
    Raises:
        AssertionError: If override semantics or validation contract is broken.
    Side Effects:
        None.
    """
    strategy_repository = InMemoryStrategyRepository()
    event_repository = InMemoryStrategyEventRepository()
    clock = _SequenceClock(
        values=(
            datetime(2026, 2, 16, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 10, 1, tzinfo=timezone.utc),
        )
    )

    create_use_case = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    )
    clone_use_case = CloneStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    )

    current_user = CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000303"))
    source_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=current_user,
    )

    cloned_strategy = clone_use_case.execute(
        current_user=current_user,
        source_strategy_id=source_strategy.strategy_id,
        template_spec_payload=None,
        overrides={
            "instrument_id": {
                "market_id": 2,
                "symbol": "ETHUSDT",
            },
            "timeframe": "5m",
        },
    )

    assert cloned_strategy.strategy_id != source_strategy.strategy_id
    assert cloned_strategy.spec.timeframe.code == "5m"
    assert cloned_strategy.spec.instrument_id.market_id.value == 2
    assert str(cloned_strategy.spec.instrument_id.symbol) == "ETHUSDT"
    assert cloned_strategy.spec.instrument_key == "binance:spot:ETHUSDT"

    with pytest.raises(RoehubError) as error_info:
        clone_use_case.execute(
            current_user=current_user,
            source_strategy_id=source_strategy.strategy_id,
            template_spec_payload=None,
            overrides={"market_type": "futures"},
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body.overrides.market_type",
                "code": "unsupported_override",
                "message": "Override key is not allowed",
            }
        ]
    }



def test_run_stop_use_cases_allow_second_run_and_enforce_single_active_run() -> None:
    """
    Verify run/stop lifecycle allows second run after stop and blocks concurrent active runs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        One-active-run invariant applies to states starting|warming_up|running|stopping.
    Raises:
        AssertionError: If lifecycle transitions violate Strategy API v1 run-control contract.
    Side Effects:
        None.
    """
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    clock = _SequenceClock(
        values=(
            datetime(2026, 2, 16, 11, 0, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 11, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 11, 2, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 11, 3, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 11, 4, tzinfo=timezone.utc),
            datetime(2026, 2, 16, 11, 5, tzinfo=timezone.utc),
        )
    )

    create_use_case = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    )
    run_use_case = RunStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
    )
    stop_use_case = StopStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
    )

    current_user = CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000404"))
    created_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=current_user,
    )

    running = run_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )
    assert running.state == "running"
    assert running.metadata_json == {
        "warmup": {
            "algorithm": "numeric_max_param_v1",
            "bars": 50,
        }
    }

    with pytest.raises(RoehubError) as conflict_error:
        run_use_case.execute(strategy_id=created_strategy.strategy_id, current_user=current_user)
    assert conflict_error.value.code == "conflict"

    stopped = stop_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )
    assert stopped.state == "stopped"

    second_run = run_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )
    assert second_run.state == "running"
    assert second_run.run_id != running.run_id



def test_warmup_estimator_is_deterministic_for_equal_strategy_specs() -> None:
    """
    Verify deterministic warmup estimator returns same result for identical strategy specs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Warmup estimator depends only on indicator params payload values.
    Raises:
        AssertionError: If estimator output is non-deterministic.
    Side Effects:
        None.
    """
    spec_a = StrategySpecV1.from_json(payload=_build_spec_payload())
    spec_b = StrategySpecV1.from_json(payload=_build_spec_payload())

    warmup_a = estimate_strategy_warmup_bars(spec=spec_a)
    warmup_b = estimate_strategy_warmup_bars(spec=spec_b)

    assert warmup_a == 50
    assert warmup_b == 50
    assert warmup_a == warmup_b



def _build_spec_payload() -> Mapping[str, Any]:
    """
    Build deterministic StrategySpecV1 payload fixture for strategy use-case tests.

    Args:
        None.
    Returns:
        Mapping[str, Any]: Valid StrategySpecV1-compatible payload.
    Assumptions:
        Payload follows immutable Strategy API v1 contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": "BTCUSDT",
        },
        "instrument_key": "binance:spot:BTCUSDT",
        "market_type": "spot",
        "timeframe": "1m",
        "indicators": [
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            }
        ],
        "signal_template": "MA(20,50)",
    }
