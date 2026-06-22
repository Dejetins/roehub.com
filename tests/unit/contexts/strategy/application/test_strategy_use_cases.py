from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

import pytest

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.application import StrategyPositionOwnershipService
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryLiveStrategyProfileRepository,
    InMemoryStrategyBacktestVariantProvenanceRepository,
    InMemoryStrategyCompatibilityReadinessRepository,
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
    InMemoryStrategyVariantScenarioMatrixRepository,
)
from trading.contexts.strategy.application import (
    BacktestVariantLaunchSnapshot,
    CloneStrategyUseCase,
    CreateStrategyFromBacktestVariantUseCase,
    CreateStrategyUseCase,
    CurrentUser,
    GetMyStrategyUseCase,
    RestartStrategyUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
    StrategyCompatibilityReadinessService,
    StrategyVariantScenarioMatrixService,
    estimate_strategy_warmup_bars,
)
from trading.contexts.strategy.application.ports.market_data_readiness import (
    MarketDataReadinessSnapshot,
)
from trading.contexts.strategy.domain.entities import StrategySpecV1
from trading.contexts.strategy.domain.entities.live_strategy_profile import LiveStrategyProfile
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


def test_create_strategy_from_backtest_variant_persists_provenance_and_replays() -> None:
    strategy_repository = InMemoryStrategyRepository()
    event_repository = InMemoryStrategyEventRepository()
    provenance_repository = InMemoryStrategyBacktestVariantProvenanceRepository(
        strategy_repository=strategy_repository,
    )
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000123")
    )
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(snapshot=_launch_snapshot(current_user)),
        strategy_repository=strategy_repository,
        provenance_repository=provenance_repository,
        event_repository=event_repository,
        clock=_SequenceClock(
            values=(
                datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
            )
        ),
    )

    created = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-1",
    )
    replay = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-1",
    )

    assert created.duplicate is False
    assert replay.duplicate is True
    assert replay.duplicate_reason == "idempotent_replay"
    assert replay.strategy.strategy_id == created.strategy.strategy_id
    assert created.provenance.source_job_id == UUID("00000000-0000-0000-0000-00000000b001")
    assert created.provenance.source_variant_key == "job_demo__dema_close_w5__vh_aaaaaaaa"
    assert created.provenance.source_variant_hash == "a" * 64
    assert created.provenance.idempotency_key_hash != "launch-1"
    assert created.strategy.spec.instrument_key == "binance:spot:BTCUSDT"
    assert created.strategy.spec.indicators[0] == {
        "id": "ma.dema",
        "params": {"row_id": 7, "source": "close", "window": 5},
    }
    events = event_repository.list_for_strategy(
        user_id=current_user.user_id,
        strategy_id=created.strategy.strategy_id,
    )
    assert events[0].event_type == "strategy_created_from_backtest_variant"


def test_create_strategy_from_backtest_variant_returns_duplicate_for_same_source_variant() -> None:
    strategy_repository = InMemoryStrategyRepository()
    provenance_repository = InMemoryStrategyBacktestVariantProvenanceRepository(
        strategy_repository=strategy_repository,
    )
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000124")
    )
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(snapshot=_launch_snapshot(current_user)),
        strategy_repository=strategy_repository,
        provenance_repository=provenance_repository,
        clock=_SequenceClock(
            values=(
                datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
            )
        ),
    )

    created = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-1",
    )
    duplicate = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-2",
    )

    assert duplicate.duplicate is True
    assert duplicate.duplicate_reason == "source_variant_exists"
    assert duplicate.strategy.strategy_id == created.strategy.strategy_id


def test_create_strategy_from_backtest_variant_allows_distinct_launch_configs() -> None:
    strategy_repository = InMemoryStrategyRepository()
    provenance_repository = InMemoryStrategyBacktestVariantProvenanceRepository(
        strategy_repository=strategy_repository,
    )
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000127")
    )
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(snapshot=_launch_snapshot(current_user)),
        strategy_repository=strategy_repository,
        provenance_repository=provenance_repository,
        clock=_SequenceClock(
            values=(
                datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 5, 30, 10, 1, tzinfo=timezone.utc),
            )
        ),
    )

    first = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-1",
        launch_config={
            "mode": "paper",
            "entry_sizing": "fixed_quote",
            "direction": "long",
        },
    )
    second = use_case.execute(
        current_user=current_user,
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        idempotency_key="launch-2",
        launch_config={
            "mode": "paper",
            "entry_sizing": "fixed_equity_pct",
            "direction": "long",
        },
    )

    assert first.duplicate is False
    assert second.duplicate is False
    assert second.strategy.strategy_id != first.strategy.strategy_id
    assert second.provenance.launch_request_hash != first.provenance.launch_request_hash


def test_create_strategy_from_backtest_variant_rejects_direction_override() -> None:
    strategy_repository = InMemoryStrategyRepository()
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000128")
    )
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(
            snapshot=_launch_snapshot(
                current_user,
                market_type="futures",
                direction_mode="long_short_reversal",
            )
        ),
        strategy_repository=strategy_repository,
        provenance_repository=InMemoryStrategyBacktestVariantProvenanceRepository(
            strategy_repository=strategy_repository,
        ),
        clock=_SequenceClock(values=(datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),)),
    )

    with pytest.raises(RoehubError) as exc_info:
        use_case.execute(
            current_user=current_user,
            job_id=UUID("00000000-0000-0000-0000-00000000b001"),
            variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
            idempotency_key="launch-direction-mismatch",
            launch_config={
                "mode": "paper",
                "market_type": "futures",
                "symbol": "BTCUSDT",
                "entry_sizing": "fixed_quote",
                "direction": "long",
            },
        )

    assert exc_info.value.code == "strategy_launch.invalid_config"
    assert exc_info.value.details is not None
    assert exc_info.value.details["reason"] == "direction_mismatch"
    assert not strategy_repository.list_for_user(user_id=current_user.user_id)


def test_create_strategy_from_backtest_variant_rejects_spot_short_like_snapshot() -> None:
    strategy_repository = InMemoryStrategyRepository()
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000129")
    )
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(
            snapshot=_launch_snapshot(
                current_user,
                market_type="spot",
                direction_mode="long_short_reversal",
            )
        ),
        strategy_repository=strategy_repository,
        provenance_repository=InMemoryStrategyBacktestVariantProvenanceRepository(
            strategy_repository=strategy_repository,
        ),
        clock=_SequenceClock(values=(datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),)),
    )

    with pytest.raises(RoehubError) as exc_info:
        use_case.execute(
            current_user=current_user,
            job_id=UUID("00000000-0000-0000-0000-00000000b001"),
            variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
            idempotency_key="launch-spot-short-like",
        )

    assert exc_info.value.code == "strategy_launch.invalid_config"
    assert exc_info.value.details is not None
    assert (
        exc_info.value.details["reason"]
        == "short_direction_requires_futures_market"
    )
    assert exc_info.value.details["field"] == "market_type"
    assert not strategy_repository.list_for_user(user_id=current_user.user_id)


def test_create_strategy_from_backtest_variant_fails_closed_for_not_launchable_job() -> None:
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000125")
    )
    strategy_repository = InMemoryStrategyRepository()
    use_case = CreateStrategyFromBacktestVariantUseCase(
        variant_reader=_StaticBacktestVariantReader(
            snapshot=_launch_snapshot(current_user, job_state="running")
        ),
        strategy_repository=strategy_repository,
        provenance_repository=InMemoryStrategyBacktestVariantProvenanceRepository(
            strategy_repository=strategy_repository,
        ),
        clock=_SequenceClock(
            values=(datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc),)
        ),
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            current_user=current_user,
            job_id=UUID("00000000-0000-0000-0000-00000000b001"),
            variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
            idempotency_key="launch-1",
        )

    assert error_info.value.code == "strategy_variant_launch.not_launchable"
    assert error_info.value.details is not None
    assert error_info.value.details["reason"] == "not_launchable"



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
    assert running.state == "starting"
    assert running.metadata_json == {}

    with pytest.raises(RoehubError) as conflict_error:
        run_use_case.execute(strategy_id=created_strategy.strategy_id, current_user=current_user)
    assert conflict_error.value.code == "conflict"

    stopped = stop_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )
    assert stopped.state == "stopping"

    stopped_terminal = stopped.transition_to(
        next_state="stopped",
        changed_at=datetime(2026, 2, 16, 11, 5, tzinfo=timezone.utc),
        checkpoint_ts_open=stopped.checkpoint_ts_open,
        last_error=None,
    )
    run_repository.update(run=stopped_terminal)

    second_run = run_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )
    assert second_run.state == "starting"
    assert second_run.run_id != running.run_id


def test_position_ownership_blocks_second_strategy_on_same_connection_instrument() -> None:
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    profile_repository = InMemoryLiveStrategyProfileRepository()
    ownership_repository = InMemoryStrategyPositionOwnershipRepository()
    ownership_service = StrategyPositionOwnershipService(repository=ownership_repository)
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000408")
    )
    connection_id = UUID("00000000-0000-0000-0000-00000000c408")
    clock = _SequenceClock(
        values=(
            datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 10, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 10, 2, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 10, 3, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 10, 4, tzinfo=timezone.utc),
        )
    )
    create_use_case = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    )
    first_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=current_user,
    )
    second_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=current_user,
    )
    profile_repository.create(
        profile=_profile(
            current_user=current_user,
            strategy_id=first_strategy.strategy_id,
            exchange_connection_id=connection_id,
            now=datetime(2026, 5, 31, 10, 2, tzinfo=timezone.utc),
        )
    )
    profile_repository.create(
        profile=_profile(
            current_user=current_user,
            strategy_id=second_strategy.strategy_id,
            exchange_connection_id=connection_id,
            now=datetime(2026, 5, 31, 10, 2, tzinfo=timezone.utc),
        )
    )
    run_use_case = RunStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
        live_profile_repository=profile_repository,
        position_ownership_coordinator=ownership_service,
    )
    stop_use_case = StopStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
        position_ownership_coordinator=ownership_service,
    )

    first_run = run_use_case.execute(
        strategy_id=first_strategy.strategy_id,
        current_user=current_user,
    )
    first_ownership = ownership_repository.get_for_run(
        owner_user_id=current_user.user_id,
        strategy_run_id=first_run.run_id,
    )
    assert first_ownership is not None
    assert first_ownership.state == "active"

    with pytest.raises(RoehubError) as conflict_error:
        run_use_case.execute(
            strategy_id=second_strategy.strategy_id,
            current_user=current_user,
        )
    assert conflict_error.value.code == "position_ownership_conflict"
    assert conflict_error.value.details is not None
    assert conflict_error.value.details["existing_strategy_run_id"] == str(first_run.run_id)
    assert conflict_error.value.details["reason"] == "position_ownership_conflict"

    stopping = stop_use_case.execute(
        strategy_id=first_strategy.strategy_id,
        current_user=current_user,
    )
    releasing = ownership_repository.get_for_run(
        owner_user_id=current_user.user_id,
        strategy_run_id=stopping.run_id,
    )
    assert releasing is not None
    assert releasing.state == "releasing"


def test_run_strategy_blocks_when_market_data_readiness_is_missing() -> None:
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    compatibility_repository = InMemoryStrategyCompatibilityReadinessRepository()
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000406")
    )
    clock = _SequenceClock(
        values=(
            datetime(2026, 5, 31, 9, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 9, 2, tzinfo=timezone.utc),
        )
    )
    created_strategy = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    ).execute(spec_payload=_build_spec_payload(), current_user=current_user)
    readiness_service = StrategyCompatibilityReadinessService(
        strategy_repository=strategy_repository,
        compatibility_repository=compatibility_repository,
        market_data_reader=_StaticMarketDataReader(state="missing"),
        event_repository=event_repository,
        clock=clock,
    )
    run_use_case = RunStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
        compatibility_readiness_checker=readiness_service,
    )

    with pytest.raises(RoehubError) as error_info:
        run_use_case.execute(strategy_id=created_strategy.strategy_id, current_user=current_user)

    assert error_info.value.code == "strategy_run.readiness_blocked"
    assert error_info.value.details is not None
    assert error_info.value.details["market_data_state"] == "missing"
    assert compatibility_repository.compatibility_reports[0].compatibility_state == "launchable"
    assert compatibility_repository.compatibility_reports[0].market_data_state == "missing"


def test_compatibility_readiness_reports_degraded_and_ready_feed_for_rollup() -> None:
    strategy_repository = InMemoryStrategyRepository()
    compatibility_repository = InMemoryStrategyCompatibilityReadinessRepository()
    event_repository = InMemoryStrategyEventRepository()
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000407")
    )
    clock = _SequenceClock(
        values=(
            datetime(2026, 5, 31, 9, 10, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 9, 11, tzinfo=timezone.utc),
        )
    )
    strategy = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=event_repository,
        clock=clock,
    ).execute(
        spec_payload={**_build_spec_payload(), "timeframe": "15m"},
        current_user=current_user,
    )
    readiness_service = StrategyCompatibilityReadinessService(
        strategy_repository=strategy_repository,
        compatibility_repository=compatibility_repository,
        market_data_reader=_StaticMarketDataReader(state="ready"),
        event_repository=event_repository,
        clock=clock,
    )

    report = readiness_service.check_strategy(
        strategy_id=strategy.strategy_id,
        current_user=current_user,
    )

    assert report.compatibility_state == "degraded"
    assert report.compatibility_reason_codes == ("timeframe_rollup_required",)
    assert report.market_data_state == "ready"
    assert report.launch_blocked is False


def test_scenario_matrix_derives_spot_rows_and_blocks_short_like_spot() -> None:
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000408")
    )
    compatibility_repository = InMemoryStrategyCompatibilityReadinessRepository()
    matrix_repository = InMemoryStrategyVariantScenarioMatrixRepository()
    clock = _SequenceClock(
        values=(
            datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 17, 10, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 17, 10, 2, tzinfo=timezone.utc),
        )
    )
    compatibility_service = StrategyCompatibilityReadinessService(
        strategy_repository=None,
        compatibility_repository=compatibility_repository,
        market_data_reader=_StaticMarketDataReader(state="ready"),
        clock=clock,
    )
    matrix_service = StrategyVariantScenarioMatrixService(
        compatibility_readiness_service=compatibility_service,
        repository=matrix_repository,
        clock=clock,
    )

    report = matrix_service.build_for_backtest_variant(
        current_user=current_user,
        snapshot=_launch_snapshot(
            current_user,
            market_type="spot",
            timeframe="1m",
            direction_mode="long_short_reversal",
            live_compatible=True,
        ),
    )

    assert len(report.rows) == 8
    assert matrix_repository.reports == [report]
    paper_long = _find_matrix_row(
        report=report,
        mode="paper",
        market_type="spot",
        entry_sizing="fixed_quote",
        direction="long",
    )
    assert paper_long.scenario_state == "launchable"
    assert paper_long.scenario_reason_codes == ("paper_no_exchange_submit",)

    paper_short = _find_matrix_row(
        report=report,
        mode="paper",
        market_type="spot",
        entry_sizing="fixed_quote",
        direction="short",
    )
    assert paper_short.scenario_state == "blocked"
    assert paper_short.launch_blocked_reason == "short_direction_requires_futures_market"
    assert paper_short.order_capability == "unsupported"
    assert paper_short.order_capability_reason_codes == (
        "short_direction_requires_futures_market",
    )

    testnet_short = _find_matrix_row(
        report=report,
        mode="testnet",
        market_type="spot",
        entry_sizing="fixed_quote",
        direction="short",
    )
    assert testnet_short.scenario_state == "blocked"
    assert testnet_short.launch_blocked_reason == "short_direction_requires_futures_market"
    assert testnet_short.order_capability == "unsupported"
    assert testnet_short.order_capability_reason_codes == (
        "short_direction_requires_futures_market",
    )


def test_scenario_matrix_marks_futures_short_real_order_capable_but_not_bound() -> None:
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000409")
    )
    clock = _SequenceClock(
        values=(
            datetime(2026, 6, 17, 11, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 17, 11, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 17, 11, 2, tzinfo=timezone.utc),
        )
    )
    compatibility_service = StrategyCompatibilityReadinessService(
        strategy_repository=None,
        compatibility_repository=InMemoryStrategyCompatibilityReadinessRepository(),
        market_data_reader=_StaticMarketDataReader(state="ready"),
        clock=clock,
    )
    matrix_service = StrategyVariantScenarioMatrixService(
        compatibility_readiness_service=compatibility_service,
        clock=clock,
    )

    report = matrix_service.build_for_backtest_variant(
        current_user=current_user,
        snapshot=_launch_snapshot(
            current_user,
            market_type="futures",
            timeframe="1m",
            direction_mode="long_short_reversal",
            live_compatible=True,
        ),
    )

    testnet_short = _find_matrix_row(
        report=report,
        mode="testnet",
        market_type="futures",
        entry_sizing="fixed_quote",
        direction="short",
    )
    assert testnet_short.scenario_state == "blocked"
    assert testnet_short.launch_blocked_reason == "exchange_connection_required"
    assert testnet_short.order_capability == "real_order_capable"
    assert testnet_short.order_capability_reason_codes == (
        "futures_short_requires_isolated_1x_guard",
    )


def test_restart_use_case_records_durable_pending_operation() -> None:
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    clock = _SequenceClock(
        values=(
            datetime(2026, 5, 31, 8, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 8, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 8, 2, tzinfo=timezone.utc),
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
    restart_use_case = RestartStrategyUseCase(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        event_repository=event_repository,
        clock=clock,
    )
    current_user = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000405")
    )
    created_strategy = create_use_case.execute(
        spec_payload=_build_spec_payload(),
        current_user=current_user,
    )
    running = run_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )

    restarting = restart_use_case.execute(
        strategy_id=created_strategy.strategy_id,
        current_user=current_user,
    )

    assert restarting.run_id == running.run_id
    assert restarting.state == "stopping"
    restart = restarting.metadata_json["restart"]
    assert restart["state"] == "pending_start"
    assert restart["requested_at"] == "2026-05-31T08:02:00Z"
    assert restart["operation_id"] != ""
    events = event_repository.list_for_strategy(
        user_id=current_user.user_id,
        strategy_id=created_strategy.strategy_id,
    )
    assert events[-1].event_type == "run_restart_requested"
    assert events[-1].payload_json["restart_operation_id"] == restart["operation_id"]

    with pytest.raises(RoehubError) as error_info:
        restart_use_case.execute(
            strategy_id=created_strategy.strategy_id,
            current_user=current_user,
        )
    assert error_info.value.code == "conflict"
    assert error_info.value.message == "Strategy restart is already pending"



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



class _StaticBacktestVariantReader:
    def __init__(self, *, snapshot: BacktestVariantLaunchSnapshot) -> None:
        self._snapshot = snapshot

    def get(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestVariantLaunchSnapshot:
        _ = user_id, job_id, variant_key
        return self._snapshot


class _StaticMarketDataReader:
    def __init__(self, *, state: str) -> None:
        self._state = state

    def check(self, *, instrument_key: str, timeframe: str, observed_at: datetime):
        return MarketDataReadinessSnapshot(
            state=self._state,  # type: ignore[arg-type]
            reason_code=f"market_data_stream_{self._state}",
            stream_name=f"md.candles.1m.{instrument_key}",
            stream_length=1 if self._state == "ready" else 0,
            last_message_id="1790000000000-0" if self._state == "ready" else None,
            last_observed_at=observed_at if self._state == "ready" else None,
            age_seconds=0 if self._state == "ready" else None,
        )


def _launch_snapshot(
    current_user: CurrentUser,
    *,
    job_state: str = "succeeded",
    market_type: str = "spot",
    timeframe: str = "15m",
    direction_mode: str = "long_only",
    live_compatible: bool = False,
) -> BacktestVariantLaunchSnapshot:
    indicator_payload = (
        {
            "indicator_id": "MA",
            "row_id": 7,
            "fast": 20,
            "slow": 50,
        }
        if live_compatible
        else {
            "indicator_id": "ma.dema",
            "row_id": 7,
            "source": "close",
            "window": 5,
        }
    )
    return BacktestVariantLaunchSnapshot(
        job_id=UUID("00000000-0000-0000-0000-00000000b001"),
        owner_user_id=current_user.user_id,
        job_state=job_state,
        request_hash="d" * 64,
        result_config_hash="e" * 64,
        market_id=1,
        exchange="binance",
        market_type=market_type,
        symbol="BTCUSDT",
        timeframe=timeframe,
        variant_key="job_demo__dema_close_w5__vh_aaaaaaaa",
        variant_hash="a" * 64,
        indicator_variant_hash="b" * 64,
        rank=1,
        summary_metrics={"total_return_pct": 12.5, "trade_count": 2},
        canonical_variant_params={
            "schema_version": 1,
            "indicators": [indicator_payload],
            "risk": {"mode": "none"},
            "execution": {"direction_mode": direction_mode},
            "ranking": {"primary_metric": "total_return_pct"},
            **({"signal_template": "MA(20,50)"} if live_compatible else {}),
        },
        readable_params={"slug": "dema_close_w5"},
    )


def _find_matrix_row(
    *,
    report,
    mode: str,
    market_type: str,
    entry_sizing: str,
    direction: str,
):
    return next(
        row
        for row in report.rows
        if row.mode == mode
        and row.market_type == market_type
        and row.entry_sizing == entry_sizing
        and row.direction == direction
    )


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


def _profile(
    *,
    current_user: CurrentUser,
    strategy_id: UUID,
    exchange_connection_id: UUID,
    now: datetime,
) -> LiveStrategyProfile:
    return LiveStrategyProfile(
        profile_id=uuid4(),
        owner_user_id=current_user.user_id,
        strategy_id=strategy_id,
        mode="paper",
        exchange_connection_id=exchange_connection_id,
        sizing_method="fixed_quote",
        sizing_value=Decimal("100"),
        max_position_notional=None,
        max_orders_per_run=1,
        max_notional_per_run=Decimal("100"),
        readiness_status="ready",
        readiness_reason="paper_no_exchange_submit",
        created_at=now,
        updated_at=now,
    )
