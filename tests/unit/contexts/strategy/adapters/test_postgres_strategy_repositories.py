from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

import pytest

from trading.contexts.strategy.adapters.outbound.persistence.postgres import (
    PostgresStrategyEventRepository,
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
)
from trading.contexts.strategy.domain import StrategyRun, StrategySpecV1
from trading.contexts.strategy.domain.errors import StrategyActiveRunConflictError
from trading.contexts.strategy.domain.services import generate_strategy_name
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, Timeframe, UserId


class _FakeGateway:
    """
    Deterministic fake SQL gateway for strategy Postgres repository unit tests.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/gateway.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
    """

    def __init__(
        self,
        *,
        fetch_one_results: list[Mapping[str, Any] | None | Exception] | None = None,
        fetch_all_results: list[tuple[Mapping[str, Any], ...]] | None = None,
    ) -> None:
        """
        Initialize fake gateway with deterministic queued responses.

        Args:
            fetch_one_results: Sequence of `fetch_one` responses or exceptions.
            fetch_all_results: Sequence of `fetch_all` responses.
        Returns:
            None.
        Assumptions:
            Test controls call order and supplies enough queued responses.
        Raises:
            None.
        Side Effects:
            Stores mutable queues and query logs.
        """
        self._fetch_one_results = list(fetch_one_results or [])
        self._fetch_all_results = list(fetch_all_results or [])
        self.fetch_one_queries: list[str] = []
        self.fetch_all_queries: list[str] = []
        self.execute_queries: list[str] = []

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Return next queued `fetch_one` response while recording SQL query text.

        Args:
            query: SQL query text.
            parameters: SQL parameters mapping.
        Returns:
            Mapping[str, Any] | None: Queued response value.
        Assumptions:
            Queue item type is either mapping, None, or exception to raise.
        Raises:
            Exception: Propagates queued exception item.
        Side Effects:
            Appends query text to call log.
        """
        self.fetch_one_queries.append(query)
        _ = parameters
        if not self._fetch_one_results:
            return None
        result = self._fetch_one_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Return next queued `fetch_all` response while recording SQL query text.

        Args:
            query: SQL query text.
            parameters: SQL parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Queued response tuple.
        Assumptions:
            Queue item is tuple of mapping rows.
        Raises:
            None.
        Side Effects:
            Appends query text to call log.
        """
        self.fetch_all_queries.append(query)
        _ = parameters
        if not self._fetch_all_results:
            return tuple()
        return self._fetch_all_results.pop(0)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Record side-effect SQL statement call.

        Args:
            query: SQL query text.
            parameters: SQL parameters mapping.
        Returns:
            None.
        Assumptions:
            Tests assert side effects by checking query log.
        Raises:
            None.
        Side Effects:
            Appends query text to execute log.
        """
        self.execute_queries.append(query)
        _ = parameters


class _UniqueViolation(Exception):
    """
    Fake unique violation exception with PostgreSQL SQLSTATE used by tests.
    """

    sqlstate = "23505"


def test_strategy_repository_soft_delete_updates_only_is_deleted_field() -> None:
    """
    Verify strategy soft-delete SQL updates only `is_deleted` and not immutable `spec_json`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository contract allows only soft-delete updates for strategy table.
    Raises:
        AssertionError: If SQL mutation contains forbidden spec updates.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(
        fetch_one_results=[{"strategy_id": "00000000-0000-0000-0000-00000000A999"}]
    )
    repository = PostgresStrategyRepository(gateway=gateway)

    changed = repository.soft_delete(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000999"),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A999"),
    )

    assert changed is True
    assert "SET is_deleted = TRUE" in gateway.fetch_one_queries[0]
    assert "spec_json" not in gateway.fetch_one_queries[0]


def test_strategy_repository_list_for_user_uses_deterministic_ordering() -> None:
    """
    Verify strategy list query includes deterministic `ORDER BY created_at ASC, strategy_id ASC`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Deterministic ordering is mandatory for stable reads.
    Raises:
        AssertionError: If ordering clause is missing.
    Side Effects:
        None.
    """
    strategy_row = _build_strategy_row()
    gateway = _FakeGateway(fetch_all_results=[(strategy_row,)])
    repository = PostgresStrategyRepository(gateway=gateway)

    strategies = repository.list_for_user(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000001011"),
    )

    assert len(strategies) == 1
    assert "ORDER BY created_at ASC, strategy_id ASC" in gateway.fetch_all_queries[0]


def test_run_repository_find_active_includes_deterministic_order_clause() -> None:
    """
    Verify active-run query uses deterministic ordering for stable row selection.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Active run lookup must include deterministic descending order.
    Raises:
        AssertionError: If ordering clause is missing.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[_build_run_row(state="running")])
    repository = PostgresStrategyRunRepository(gateway=gateway)

    run = repository.find_active_for_strategy(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000001011"),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A111"),
    )

    assert run is not None
    assert run.state == "running"
    assert "ORDER BY started_at DESC, run_id DESC" in gateway.fetch_one_queries[0]


def test_run_repository_create_translates_unique_violation_to_domain_conflict() -> None:
    """
    Verify active-run unique DB conflict is mapped to deterministic domain conflict error.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository pre-check plus DB unique index both protect one-active-run invariant.
    Raises:
        AssertionError: If unique violation is not translated.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[None, _UniqueViolation("duplicate")])
    repository = PostgresStrategyRunRepository(gateway=gateway)

    run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-00000000B888"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000001012"),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A112"),
        started_at=datetime(2026, 2, 15, 13, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(StrategyActiveRunConflictError):
        repository.create(run=run)


def test_event_repository_list_for_strategy_uses_deterministic_ordering() -> None:
    """
    Verify strategy event stream query includes deterministic ordering by timestamp and event id.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Event stream contract requires stable ordering for replay/UI.
    Raises:
        AssertionError: If ordering clause is missing.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_all_results=[(_build_event_row(),)])
    repository = PostgresStrategyEventRepository(gateway=gateway)

    events = repository.list_for_strategy(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000001013"),
        strategy_id=UUID("00000000-0000-0000-0000-00000000A113"),
    )

    assert len(events) == 1
    assert events[0].run_id is None
    assert "ORDER BY ts ASC, event_id ASC" in gateway.fetch_all_queries[0]



def _build_spec() -> StrategySpecV1:
    """
    Build deterministic StrategySpecV1 fixture for adapter mapping tests.

    Args:
        None.
    Returns:
        StrategySpecV1: Valid immutable spec instance.
    Assumptions:
        Fixture mirrors canonical example with BTCUSDT, 1m, and MA(20,50).
    Raises:
        ValueError: If fixture values violate domain invariants.
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



def _build_strategy_row() -> Mapping[str, Any]:
    """
    Build deterministic strategy SQL row mapping fixture.

    Args:
        None.
    Returns:
        Mapping[str, Any]: Row payload matching `strategy_strategies` schema.
    Assumptions:
        `spec_json` includes required `schema_version` and `spec_kind` literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    spec = _build_spec()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001011")
    return {
        "strategy_id": "00000000-0000-0000-0000-00000000A111",
        "user_id": str(user_id),
        "name": generate_strategy_name(user_id=user_id, spec=spec),
        "instrument_id": spec.instrument_id.as_dict(),
        "instrument_key": spec.instrument_key,
        "market_type": spec.market_type,
        "symbol": str(spec.instrument_id.symbol),
        "timeframe": spec.timeframe.code,
        "indicators_json": list(spec.indicators),
        "spec_json": spec.to_json(),
        "created_at": datetime(2026, 2, 15, 13, 10, tzinfo=timezone.utc),
        "is_deleted": False,
    }



def _build_run_row(*, state: str) -> Mapping[str, Any]:
    """
    Build deterministic run SQL row mapping fixture.

    Args:
        state: Run state literal to embed in fixture.
    Returns:
        Mapping[str, Any]: Row payload matching `strategy_runs` schema.
    Assumptions:
        Active states include `running` in Strategy v1 state machine.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "run_id": "00000000-0000-0000-0000-00000000B111",
        "user_id": "00000000-0000-0000-0000-000000001011",
        "strategy_id": "00000000-0000-0000-0000-00000000A111",
        "state": state,
        "started_at": datetime(2026, 2, 15, 13, 20, tzinfo=timezone.utc),
        "stopped_at": None,
        "checkpoint_ts_open": datetime(2026, 2, 15, 13, 19, tzinfo=timezone.utc),
        "last_error": None,
        "updated_at": datetime(2026, 2, 15, 13, 20, tzinfo=timezone.utc),
    }



def _build_event_row() -> Mapping[str, Any]:
    """
    Build deterministic event SQL row mapping fixture.

    Args:
        None.
    Returns:
        Mapping[str, Any]: Row payload matching `strategy_events` schema.
    Assumptions:
        `run_id` is nullable for strategy-level events.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "event_id": "00000000-0000-0000-0000-00000000E111",
        "user_id": "00000000-0000-0000-0000-000000001011",
        "strategy_id": "00000000-0000-0000-0000-00000000A111",
        "run_id": None,
        "ts": datetime(2026, 2, 15, 13, 30, tzinfo=timezone.utc),
        "event_type": "strategy_created",
        "payload_json": json.loads('{"schema_version":1}'),
    }
