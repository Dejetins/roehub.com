from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.repositories import (
    StrategyExchangeBindingRepository,
    StrategyRepository,
)
from trading.contexts.strategy.domain.entities import StrategyExchangeBinding
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class StrategyExchangeBindingView:
    binding_id: UUID
    owner_user_id: UserId
    strategy_id: UUID
    exchange_connection_id: UUID
    usage_mode: Literal["trading"]
    binding_status: Literal["active", "paused", "disabled", "archived"]
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None
    archived_at: datetime | None


@dataclass(frozen=True, slots=True)
class StrategyExchangeBindingService:
    strategy_repository: StrategyRepository
    binding_repository: StrategyExchangeBindingRepository

    def list_bindings(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> tuple[StrategyExchangeBindingView, ...]:
        self._require_strategy(owner_user_id=owner_user_id, strategy_id=strategy_id)
        return tuple(
            _to_view(binding=binding)
            for binding in self.binding_repository.list_for_strategy(
                owner_user_id=owner_user_id,
                strategy_id=strategy_id,
            )
        )

    def create_binding(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        exchange_connection_id: UUID,
        usage_mode: str,
        now: datetime,
    ) -> StrategyExchangeBindingView:
        self._require_strategy(owner_user_id=owner_user_id, strategy_id=strategy_id)
        if usage_mode != "trading":
            raise RoehubError(
                code="validation_error",
                message="Validation failed",
                details={
                    "errors": [
                        {
                            "path": "usage_mode",
                            "code": "unsupported_usage_mode",
                            "message": "Only trading usage mode is supported.",
                        }
                    ]
                },
            )
        binding = StrategyExchangeBinding(
            binding_id=uuid4(),
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            exchange_connection_id=exchange_connection_id,
            usage_mode="trading",
            binding_status="active",
            created_at=now,
            updated_at=now,
        )
        created = self.binding_repository.create(binding=binding)
        if created is None:
            raise RoehubError(
                code="strategy_exchange_binding_already_active",
                message="Strategy already has an active binding to this exchange connection.",
                details={
                    "strategy_id": str(strategy_id),
                    "exchange_connection_id": str(exchange_connection_id),
                    "usage_mode": "trading",
                },
            )
        return _to_view(binding=created)

    def disable_binding(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        binding_id: UUID,
        now: datetime,
    ) -> StrategyExchangeBindingView:
        self._require_strategy(owner_user_id=owner_user_id, strategy_id=strategy_id)
        existing = self.binding_repository.get(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            binding_id=binding_id,
        )
        if existing is None:
            raise _binding_not_found(binding_id=binding_id)
        if existing.binding_status != "active":
            return _to_view(binding=existing)
        disabled = self.binding_repository.disable(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            binding_id=binding_id,
            disabled_at=now,
        )
        if disabled is None:
            raise _binding_not_found(binding_id=binding_id)
        return _to_view(binding=disabled)

    def _require_strategy(self, *, owner_user_id: UserId, strategy_id: UUID) -> None:
        strategy = self.strategy_repository.find_by_strategy_id(
            user_id=owner_user_id,
            strategy_id=strategy_id,
        )
        if strategy is None or strategy.is_deleted:
            raise RoehubError(
                code="not_found",
                message="Strategy was not found",
                details={"strategy_id": str(strategy_id)},
            )


def _to_view(*, binding: StrategyExchangeBinding) -> StrategyExchangeBindingView:
    return StrategyExchangeBindingView(
        binding_id=binding.binding_id,
        owner_user_id=binding.owner_user_id,
        strategy_id=binding.strategy_id,
        exchange_connection_id=binding.exchange_connection_id,
        usage_mode=binding.usage_mode,
        binding_status=binding.binding_status,
        created_at=binding.created_at,
        updated_at=binding.updated_at,
        disabled_at=binding.disabled_at,
        archived_at=binding.archived_at,
    )


def _binding_not_found(*, binding_id: UUID) -> RoehubError:
    return RoehubError(
        code="strategy_exchange_binding_not_found",
        message="Strategy exchange binding was not found",
        details={"binding_id": str(binding_id)},
    )
