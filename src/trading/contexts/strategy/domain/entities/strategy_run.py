from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID

from trading.contexts.strategy.domain.errors import StrategyRunTransitionError
from trading.shared_kernel.primitives import UserId

StrategyRunState = Literal["starting", "warming_up", "running", "stopping", "stopped", "failed"]

_ACTIVE_RUN_STATES: frozenset[str] = frozenset({"starting", "warming_up", "running", "stopping"})
_TERMINAL_RUN_STATES: frozenset[str] = frozenset({"stopped", "failed"})
_ALLOWED_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "starting": frozenset({"warming_up", "stopping", "failed"}),
    "warming_up": frozenset({"running", "stopping", "failed"}),
    "running": frozenset({"stopping", "failed"}),
    "stopping": frozenset({"stopped", "failed"}),
    "stopped": frozenset(),
    "failed": frozenset(),
}


@dataclass(frozen=True, slots=True)
class StrategyRun:
    """
    StrategyRun — immutable execution run snapshot with deterministic v1 state machine.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/services/run_invariants.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_run_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    run_id: UUID
    user_id: UserId
    strategy_id: UUID
    state: StrategyRunState
    started_at: datetime
    stopped_at: datetime | None
    checkpoint_ts_open: datetime | None
    last_error: str | None
    updated_at: datetime
    metadata_json: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate run snapshot invariants for timestamps and state-specific fields.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `paused` state is forbidden in Strategy v1.
        Raises:
            StrategyRunTransitionError: If state/timestamp invariants are violated.
        Side Effects:
            None.
        """
        if self.state not in _ALLOWED_STATE_TRANSITIONS:
            raise StrategyRunTransitionError(f"StrategyRun.state is unsupported: {self.state!r}")

        _ensure_utc_datetime(name="started_at", value=self.started_at)
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        if self.updated_at < self.started_at:
            raise StrategyRunTransitionError("StrategyRun.updated_at cannot be before started_at")

        if self.stopped_at is not None:
            _ensure_utc_datetime(name="stopped_at", value=self.stopped_at)
            if self.stopped_at < self.started_at:
                raise StrategyRunTransitionError(
                    "StrategyRun.stopped_at cannot be before started_at"
                )
            if self.updated_at < self.stopped_at:
                raise StrategyRunTransitionError(
                    "StrategyRun.updated_at cannot be before stopped_at"
                )

        if self.checkpoint_ts_open is not None:
            _ensure_utc_datetime(name="checkpoint_ts_open", value=self.checkpoint_ts_open)

        if self.state in _ACTIVE_RUN_STATES and self.stopped_at is not None:
            raise StrategyRunTransitionError(
                "StrategyRun.stopped_at must be None while state is active"
            )
        if self.state in _TERMINAL_RUN_STATES and self.stopped_at is None:
            raise StrategyRunTransitionError(
                "StrategyRun.stopped_at must be set while state is terminal"
            )

        if self.state == "failed":
            if self.last_error is None or not self.last_error.strip():
                raise StrategyRunTransitionError(
                    "StrategyRun.last_error must be set for failed state"
                )
        elif self.last_error is not None and not self.last_error.strip():
            raise StrategyRunTransitionError("StrategyRun.last_error cannot be blank")

        normalized_metadata = _normalize_metadata_json(metadata_json=self.metadata_json)
        object.__setattr__(self, "metadata_json", normalized_metadata)

    @classmethod
    def start(
        cls,
        *,
        run_id: UUID,
        user_id: UserId,
        strategy_id: UUID,
        started_at: datetime,
        metadata_json: Mapping[str, Any] | None = None,
    ) -> StrategyRun:
        """
        Create new run in initial `starting` state.

        Args:
            run_id: Stable run identifier.
            user_id: Strategy owner identifier.
            strategy_id: Target strategy identifier.
            started_at: Run start timestamp in UTC.
            metadata_json: Optional deterministic metadata payload for run traceability.
        Returns:
            StrategyRun: Initial run snapshot.
        Assumptions:
            Caller guarantees no other active run exists for the strategy.
        Raises:
            StrategyRunTransitionError: If created snapshot violates run invariants.
        Side Effects:
            None.
        """
        return cls(
            run_id=run_id,
            user_id=user_id,
            strategy_id=strategy_id,
            state="starting",
            started_at=started_at,
            stopped_at=None,
            checkpoint_ts_open=None,
            last_error=None,
            updated_at=started_at,
            metadata_json={} if metadata_json is None else dict(metadata_json),
        )

    def is_active(self) -> bool:
        """
        Check whether run belongs to active v1 states.

        Args:
            None.
        Returns:
            bool: `True` for `starting|warming_up|running|stopping`.
        Assumptions:
            Active state set is fixed by v1 domain contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.state in _ACTIVE_RUN_STATES

    def transition_to(
        self,
        *,
        next_state: StrategyRunState,
        changed_at: datetime,
        checkpoint_ts_open: datetime | None = None,
        last_error: str | None = None,
    ) -> StrategyRun:
        """
        Transition immutable run snapshot to next v1 state.

        Args:
            next_state: Target state.
            changed_at: Transition timestamp in UTC.
            checkpoint_ts_open: Optional deterministic checkpoint timestamp.
            last_error: Optional run error text (required for `failed`).
        Returns:
            StrategyRun: New run snapshot with updated state and timestamps.
        Assumptions:
            Transition graph is deterministic and does not include `paused` state.
        Raises:
            StrategyRunTransitionError: If transition or invariants are invalid.
        Side Effects:
            None.
        """
        if next_state not in _ALLOWED_STATE_TRANSITIONS:
            raise StrategyRunTransitionError(
                f"StrategyRun next_state is unsupported: {next_state!r}"
            )

        allowed_next_states = _ALLOWED_STATE_TRANSITIONS[self.state]
        if next_state not in allowed_next_states:
            raise StrategyRunTransitionError(
                f"StrategyRun invalid transition {self.state!r} -> {next_state!r}"
            )

        _ensure_utc_datetime(name="changed_at", value=changed_at)
        if changed_at < self.updated_at:
            raise StrategyRunTransitionError(
                "StrategyRun.transition_to changed_at cannot be before current updated_at"
            )

        if checkpoint_ts_open is not None:
            _ensure_utc_datetime(name="checkpoint_ts_open", value=checkpoint_ts_open)

        if next_state == "failed":
            if last_error is None or not last_error.strip():
                raise StrategyRunTransitionError(
                    "StrategyRun.transition_to failed requires last_error"
                )
        elif last_error is not None:
            if not last_error.strip():
                raise StrategyRunTransitionError(
                    "StrategyRun.transition_to last_error cannot be blank"
                )

        next_stopped_at = changed_at if next_state in _TERMINAL_RUN_STATES else None
        return StrategyRun(
            run_id=self.run_id,
            user_id=self.user_id,
            strategy_id=self.strategy_id,
            state=next_state,
            started_at=self.started_at,
            stopped_at=next_stopped_at,
            checkpoint_ts_open=checkpoint_ts_open,
            last_error=last_error,
            updated_at=changed_at,
            metadata_json=self.metadata_json,
        )



def is_strategy_run_state_active(*, state: StrategyRunState) -> bool:
    """
    Check whether run state is active in Strategy v1 lifecycle.

    Args:
        state: Run state value.
    Returns:
        bool: `True` when state is one of active states.
    Assumptions:
        Active state list is fixed and deterministic for v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    return state in _ACTIVE_RUN_STATES



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field for run invariants.

    Args:
        name: Field name used in deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        Run timestamps are always stored as UTC.
    Raises:
        StrategyRunTransitionError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise StrategyRunTransitionError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise StrategyRunTransitionError(f"{name} must be UTC datetime")


def _normalize_metadata_json(*, metadata_json: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Normalize run metadata mapping into deterministic JSON-compatible payload.

    Args:
        metadata_json: Raw metadata mapping.
    Returns:
        Mapping[str, Any]: Deterministic normalized mapping with sorted keys.
    Assumptions:
        Metadata payload is JSON-serializable and used for run traceability.
    Raises:
        StrategyRunTransitionError: If metadata cannot be serialized to JSON.
    Side Effects:
        None.
    """
    normalized = _normalize_json_value(value=dict(metadata_json))
    if not isinstance(normalized, Mapping):
        raise StrategyRunTransitionError("StrategyRun.metadata_json must be JSON object")

    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as error:
        raise StrategyRunTransitionError(
            "StrategyRun.metadata_json must be JSON-serializable"
        ) from error
    return normalized


def _normalize_json_value(*, value: Any) -> Any:
    """
    Normalize arbitrary JSON-like value into deterministic plain-Python structure.

    Args:
        value: Raw value from metadata payload.
    Returns:
        Any: Deterministic mapping/list/scalar value.
    Assumptions:
        Unknown non-JSON objects are stringified for safe trace payloads.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        for raw_key, raw_value in sorted_items:
            normalized_mapping[str(raw_key)] = _normalize_json_value(value=raw_value)
        return normalized_mapping

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)
