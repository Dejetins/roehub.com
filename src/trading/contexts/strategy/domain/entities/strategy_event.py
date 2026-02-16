from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.strategy.domain.errors import StrategySpecValidationError
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class StrategyEvent:
    """
    StrategyEvent — immutable append-only event snapshot for strategy/run streams.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_event_repository.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/
        strategy_event_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    event_id: UUID
    user_id: UserId
    strategy_id: UUID
    run_id: UUID | None
    ts: datetime
    event_type: str
    payload_json: Mapping[str, Any]

    def __post_init__(self) -> None:
        """
        Validate append-only event invariants for payload and UTC timestamp.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `payload_json` must be JSON-serializable deterministic mapping.
        Raises:
            StrategySpecValidationError: If timestamp, type, or payload invariants are violated.
        Side Effects:
            None.
        """
        _ensure_utc_datetime(name="ts", value=self.ts)
        if not self.event_type.strip():
            raise StrategySpecValidationError("StrategyEvent.event_type must be non-empty")

        try:
            json.dumps(self.payload_json, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except TypeError as error:
            raise StrategySpecValidationError(
                "StrategyEvent.payload_json must be JSON-serializable"
            ) from error

    @classmethod
    def create(
        cls,
        *,
        user_id: UserId,
        strategy_id: UUID,
        run_id: UUID | None,
        ts: datetime,
        event_type: str,
        payload_json: Mapping[str, Any],
        event_id: UUID | None = None,
    ) -> StrategyEvent:
        """
        Build new append-only strategy event record.

        Args:
            user_id: Strategy owner identifier.
            strategy_id: Target strategy identifier.
            run_id: Optional run identifier (nullable for strategy-level events).
            ts: Event timestamp in UTC.
            event_type: Deterministic event type literal.
            payload_json: JSON payload mapping.
            event_id: Optional explicit event id for deterministic tests.
        Returns:
            StrategyEvent: Immutable event snapshot.
        Assumptions:
            When `event_id` is not provided a random UUIDv4 is generated.
        Raises:
            StrategySpecValidationError: If created event violates invariants.
        Side Effects:
            None.
        """
        effective_event_id = event_id if event_id is not None else uuid4()
        return cls(
            event_id=effective_event_id,
            user_id=user_id,
            strategy_id=strategy_id,
            run_id=run_id,
            ts=ts,
            event_type=event_type.strip(),
            payload_json=dict(payload_json),
        )



def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    """
    Validate timezone-aware UTC datetime field.

    Args:
        name: Field name used in deterministic error messages.
        value: Datetime value to validate.
    Returns:
        None.
    Assumptions:
        Event timestamps are stored in UTC.
    Raises:
        StrategySpecValidationError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise StrategySpecValidationError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise StrategySpecValidationError(f"{name} must be UTC datetime")
