from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import (
    StrategyBacktestVariantProvenanceRepository,
)
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
)
from trading.contexts.strategy.domain.errors import StrategyStorageError
from trading.shared_kernel.primitives import UserId

from .strategy_repository import InMemoryStrategyRepository


class InMemoryStrategyBacktestVariantProvenanceRepository(
    StrategyBacktestVariantProvenanceRepository
):
    """
    In-memory provenance repository sharing the same strategy repository instance.
    """

    def __init__(self, *, strategy_repository: InMemoryStrategyRepository) -> None:
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "InMemoryStrategyBacktestVariantProvenanceRepository requires strategy_repository"
            )
        self._strategy_repository = strategy_repository
        self._by_strategy_id: dict[UUID, StrategyBacktestVariantProvenance] = {}

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        for provenance in self._by_strategy_id.values():
            if (
                provenance.user_id == user_id
                and provenance.idempotency_key_hash == idempotency_key_hash
            ):
                return provenance
        return None

    def find_by_source_variant(
        self,
        *,
        user_id: UserId,
        source_job_id: UUID,
        source_variant_key: str,
        strategy_spec_hash: str,
        launch_request_hash: str,
    ) -> StrategyBacktestVariantProvenance | None:
        for provenance in self._by_strategy_id.values():
            if (
                provenance.user_id == user_id
                and provenance.source_job_id == source_job_id
                and provenance.source_variant_key == source_variant_key
                and provenance.strategy_spec_hash == strategy_spec_hash
                and provenance.launch_request_hash == launch_request_hash
            ):
                return provenance
        return None

    def create_with_strategy(
        self,
        *,
        strategy: Strategy,
        provenance: StrategyBacktestVariantProvenance,
    ) -> StrategyBacktestVariantProvenance:
        if strategy.strategy_id in self._by_strategy_id:
            raise StrategyStorageError("duplicate strategy backtest variant provenance")
        if self.find_by_idempotency_key(
            user_id=provenance.user_id,
            idempotency_key_hash=provenance.idempotency_key_hash,
        ):
            raise StrategyStorageError("duplicate strategy variant idempotency hash")
        if self.find_by_source_variant(
            user_id=provenance.user_id,
            source_job_id=provenance.source_job_id,
            source_variant_key=provenance.source_variant_key,
            strategy_spec_hash=provenance.strategy_spec_hash,
            launch_request_hash=provenance.launch_request_hash,
        ):
            raise StrategyStorageError("duplicate strategy source variant provenance")
        self._strategy_repository.create(strategy=strategy)
        self._by_strategy_id[strategy.strategy_id] = provenance
        return provenance
