from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from trading.contexts.strategy.domain.entities.strategy_spec_v1 import StrategySpecV1
from trading.contexts.strategy.domain.errors import StrategySpecValidationError
from trading.contexts.strategy.domain.services.strategy_name import generate_strategy_name
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class Strategy:
    """
    Strategy — immutable strategy aggregate root snapshot with soft-delete flag.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
      - alembic/versions/20260215_0001_strategy_storage_v1.py
    """

    strategy_id: UUID
    user_id: UserId
    name: str
    spec: StrategySpecV1
    created_at: datetime
    is_deleted: bool = False

    def __post_init__(self) -> None:
        """
        Validate strategy identity, immutable name contract, and UTC timestamp invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Strategy name must match deterministic generation from `(user_id + spec_json)`.
        Raises:
            StrategySpecValidationError: If name or timestamp invariants are violated.
        Side Effects:
            None.
        """
        normalized_name = self.name.strip()
        if not normalized_name:
            raise StrategySpecValidationError("Strategy.name must be non-empty")
        expected_name = generate_strategy_name(user_id=self.user_id, spec=self.spec)
        if normalized_name != expected_name:
            raise StrategySpecValidationError(
                "Strategy.name must be deterministic result from user_id + spec_json"
            )
        _ensure_utc_datetime(name="created_at", value=self.created_at)

    @classmethod
    def create(
        cls,
        *,
        user_id: UserId,
        spec: StrategySpecV1,
        created_at: datetime,
        strategy_id: UUID | None = None,
    ) -> Strategy:
        """
        Build new immutable Strategy aggregate with deterministic generated name.

        Args:
            user_id: Strategy owner identifier.
            spec: Immutable StrategySpecV1 payload.
            created_at: Strategy creation UTC timestamp.
            strategy_id: Optional explicit strategy id for deterministic tests.
        Returns:
            Strategy: New immutable strategy instance.
        Assumptions:
            When `strategy_id` is not provided a random UUIDv4 is generated.
        Raises:
            StrategySpecValidationError: If generated aggregate violates invariants.
        Side Effects:
            None.
        """
        effective_strategy_id = strategy_id if strategy_id is not None else uuid4()
        generated_name = generate_strategy_name(user_id=user_id, spec=spec)
        return cls(
            strategy_id=effective_strategy_id,
            user_id=user_id,
            name=generated_name,
            spec=spec,
            created_at=created_at,
            is_deleted=False,
        )

    def soft_deleted(self) -> Strategy:
        """
        Return immutable copy marked as soft-deleted (`is_deleted=True`).

        Args:
            None.
        Returns:
            Strategy: New strategy snapshot with unchanged immutable spec and `is_deleted=True`.
        Assumptions:
            Strategy spec and name are immutable and preserved during soft-delete operation.
        Raises:
            StrategySpecValidationError: If resulting object unexpectedly violates invariants.
        Side Effects:
            None.
        """
        return Strategy(
            strategy_id=self.strategy_id,
            user_id=self.user_id,
            name=self.name,
            spec=self.spec,
            created_at=self.created_at,
            is_deleted=True,
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
        All strategy timestamps are stored in UTC.
    Raises:
        StrategySpecValidationError: If value is naive or has non-zero timezone offset.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise StrategySpecValidationError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise StrategySpecValidationError(f"{name} must be UTC datetime")
