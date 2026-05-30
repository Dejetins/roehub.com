from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
)
from trading.shared_kernel.primitives import UserId


class StrategyBacktestVariantProvenanceRepository(Protocol):
    """
    Storage port for atomic strategy creation with backtest-variant provenance.
    """

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        """
        Load a previous create-from-variant attempt by owner-local idempotency hash.
        """
        ...

    def find_by_source_variant(
        self,
        *,
        user_id: UserId,
        source_job_id: UUID,
        source_variant_key: str,
        strategy_spec_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        """
        Load an existing strategy provenance for the same owner/source variant/spec identity.
        """
        ...

    def create_with_strategy(
        self,
        *,
        strategy: Strategy,
        provenance: StrategyBacktestVariantProvenance,
    ) -> StrategyBacktestVariantProvenance:
        """
        Persist the immutable Strategy row and its provenance in one storage operation.
        """
        ...
