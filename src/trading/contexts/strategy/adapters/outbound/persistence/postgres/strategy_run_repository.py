from __future__ import annotations

from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.contexts.strategy.application.ports.repositories import StrategyRunRepository
from trading.contexts.strategy.domain.entities import StrategyRun, StrategyRunState
from trading.contexts.strategy.domain.errors import (
    StrategyActiveRunConflictError,
    StrategyStorageError,
)
from trading.shared_kernel.primitives import UserId

_ACTIVE_STATES_SQL_LITERAL = "'starting','warming_up','running','stopping'"


class PostgresStrategyRunRepository(StrategyRunRepository):
    """
    PostgresStrategyRunRepository — explicit SQL adapter for Strategy v1 run storage.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_run_repository.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    def __init__(
        self,
        *,
        gateway: StrategyPostgresGateway,
        runs_table: str = "strategy_runs",
    ) -> None:
        """
        Initialize run repository with SQL gateway and target table name.

        Args:
            gateway: SQL gateway abstraction.
            runs_table: Run table name.
        Returns:
            None.
        Assumptions:
            Table schema follows Strategy v1 migration contract.
        Raises:
            ValueError: If dependencies are invalid.
        Side Effects:
            None.
        """
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresStrategyRunRepository requires gateway")
        normalized_table = runs_table.strip()
        if not normalized_table:
            raise ValueError("PostgresStrategyRunRepository requires non-empty runs_table")
        self._gateway = gateway
        self._runs_table = normalized_table

    def create(self, *, run: StrategyRun) -> StrategyRun:
        """
        Insert run row and enforce one-active-run invariant.

        Args:
            run: Run snapshot to persist.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Partial unique index guards race conditions for active runs.
        Raises:
            StrategyActiveRunConflictError: On second active run attempt.
            StrategyStorageError: If insert returns no row or mapping fails.
        Side Effects:
            Executes one SQL select and one SQL insert statement.
        """
        if run.is_active():
            active_run = self.find_active_for_strategy(
                user_id=run.user_id,
                strategy_id=run.strategy_id,
            )
            if active_run is not None and active_run.run_id != run.run_id:
                raise StrategyActiveRunConflictError(
                    "Strategy v1 allows exactly one active run per strategy"
                )

        query = f"""
        INSERT INTO {self._runs_table}
        (
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        )
        VALUES
        (
            %(run_id)s,
            %(user_id)s,
            %(strategy_id)s,
            %(state)s,
            %(started_at)s,
            %(stopped_at)s,
            %(checkpoint_ts_open)s,
            %(last_error)s,
            %(updated_at)s
        )
        RETURNING
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        """
        try:
            row = self._gateway.fetch_one(
                query=query,
                parameters={
                    "run_id": str(run.run_id),
                    "user_id": str(run.user_id),
                    "strategy_id": str(run.strategy_id),
                    "state": run.state,
                    "started_at": run.started_at,
                    "stopped_at": run.stopped_at,
                    "checkpoint_ts_open": run.checkpoint_ts_open,
                    "last_error": run.last_error,
                    "updated_at": run.updated_at,
                },
            )
        except Exception as error:  # noqa: BLE001
            if _is_active_run_unique_violation(error=error):
                raise StrategyActiveRunConflictError(
                    "Strategy v1 allows exactly one active run per strategy"
                ) from error
            raise StrategyStorageError("PostgresStrategyRunRepository.create failed") from error

        if row is None:
            raise StrategyStorageError("PostgresStrategyRunRepository.create returned no row")
        return _map_run_row(row=row)

    def update(self, *, run: StrategyRun) -> StrategyRun:
        """
        Update run row with immutable transition snapshot values.

        Args:
            run: Run snapshot to persist.
        Returns:
            StrategyRun: Persisted run snapshot.
        Assumptions:
            Transition has already been validated by domain model.
        Raises:
            StrategyStorageError: If row is missing or cannot be mapped.
        Side Effects:
            Executes one SQL update statement.
        """
        query = f"""
        UPDATE {self._runs_table}
        SET
            state = %(state)s,
            stopped_at = %(stopped_at)s,
            checkpoint_ts_open = %(checkpoint_ts_open)s,
            last_error = %(last_error)s,
            updated_at = %(updated_at)s
        WHERE user_id = %(user_id)s
          AND run_id = %(run_id)s
        RETURNING
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "run_id": str(run.run_id),
                "user_id": str(run.user_id),
                "state": run.state,
                "stopped_at": run.stopped_at,
                "checkpoint_ts_open": run.checkpoint_ts_open,
                "last_error": run.last_error,
                "updated_at": run.updated_at,
            },
        )
        if row is None:
            raise StrategyStorageError("PostgresStrategyRunRepository.update cannot find run row")
        return _map_run_row(row=row)

    def find_by_run_id(self, *, user_id: UserId, run_id: UUID) -> StrategyRun | None:
        """
        Find run snapshot by owner and run identifier.

        Args:
            user_id: Strategy owner identifier.
            run_id: Run identifier.
        Returns:
            StrategyRun | None: Mapped run snapshot or `None`.
        Assumptions:
            Run id lookup is deterministic and owner-scoped.
        Raises:
            StrategyStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        FROM {self._runs_table}
        WHERE user_id = %(user_id)s
          AND run_id = %(run_id)s
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "run_id": str(run_id),
            },
        )
        if row is None:
            return None
        return _map_run_row(row=row)

    def find_active_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> StrategyRun | None:
        """
        Find current active run for strategy using deterministic ordering.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            StrategyRun | None: Active run or `None`.
        Assumptions:
            At most one row should satisfy active condition by storage constraints.
        Raises:
            StrategyStorageError: If row mapping fails.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        FROM {self._runs_table}
        WHERE user_id = %(user_id)s
          AND strategy_id = %(strategy_id)s
          AND state IN ({_ACTIVE_STATES_SQL_LITERAL})
        ORDER BY started_at DESC, run_id DESC
        LIMIT 1
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                "user_id": str(user_id),
                "strategy_id": str(strategy_id),
            },
        )
        if row is None:
            return None
        return _map_run_row(row=row)

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyRun, ...]:
        """
        List strategy runs ordered deterministically by start time and run id.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Strategy identifier.
        Returns:
            tuple[StrategyRun, ...]: Ordered run snapshots.
        Assumptions:
            SQL ordering is deterministic for same dataset.
        Raises:
            StrategyStorageError: If one of rows cannot be mapped.
        Side Effects:
            Executes one SQL select statement.
        """
        query = f"""
        SELECT
            run_id,
            user_id,
            strategy_id,
            state,
            started_at,
            stopped_at,
            checkpoint_ts_open,
            last_error,
            updated_at
        FROM {self._runs_table}
        WHERE user_id = %(user_id)s
          AND strategy_id = %(strategy_id)s
        ORDER BY started_at ASC, run_id ASC
        """
        rows = self._gateway.fetch_all(
            query=query,
            parameters={
                "user_id": str(user_id),
                "strategy_id": str(strategy_id),
            },
        )
        return tuple(_map_run_row(row=row) for row in rows)



def _map_run_row(*, row: Mapping[str, Any]) -> StrategyRun:
    """
    Map SQL row into immutable StrategyRun domain object.

    Args:
        row: SQL row mapping.
    Returns:
        StrategyRun: Mapped run snapshot.
    Assumptions:
        Row schema follows Strategy v1 run table contract.
    Raises:
        StrategyStorageError: If mapping fails.
    Side Effects:
        None.
    """
    try:
        state = _parse_run_state(value=row["state"])
        return StrategyRun(
            run_id=UUID(str(row["run_id"])),
            user_id=UserId.from_string(str(row["user_id"])),
            strategy_id=UUID(str(row["strategy_id"])),
            state=state,
            started_at=row["started_at"],
            stopped_at=row["stopped_at"],
            checkpoint_ts_open=row["checkpoint_ts_open"],
            last_error=row["last_error"],
            updated_at=row["updated_at"],
        )
    except Exception as error:  # noqa: BLE001
        raise StrategyStorageError("PostgresStrategyRunRepository cannot map run row") from error



def _is_active_run_unique_violation(*, error: Exception) -> bool:
    """
    Detect database unique-constraint conflict for active run invariant.

    Args:
        error: Caught database exception.
    Returns:
        bool: `True` when exception is active-run unique violation.
    Assumptions:
        Postgres unique violation SQLSTATE is `23505`.
    Raises:
        None.
    Side Effects:
        None.
    """
    sql_state = getattr(error, "sqlstate", None)
    if sql_state == "23505":
        return True

    message = str(error).lower()
    return "strategy_runs_one_active" in message


def _parse_run_state(*, value: Any) -> StrategyRunState:
    """
    Parse and validate run-state string from SQL row into StrategyRunState literal type.

    Args:
        value: Raw SQL row state value.
    Returns:
        StrategyRunState: Typed state literal for StrategyRun constructor.
    Assumptions:
        Storage state values follow Strategy v1 state-machine literals.
    Raises:
        StrategyStorageError: If state value is unknown.
    Side Effects:
        None.
    """
    normalized = str(value)
    allowed_states = {"starting", "warming_up", "running", "stopping", "stopped", "failed"}
    if normalized not in allowed_states:
        raise StrategyStorageError(f"Unexpected run state value from storage: {normalized!r}")
    return cast(StrategyRunState, normalized)
