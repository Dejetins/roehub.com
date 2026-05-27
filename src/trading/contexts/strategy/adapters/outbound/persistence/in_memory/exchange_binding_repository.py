from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import (
    StrategyExchangeBindingRepository,
)
from trading.contexts.strategy.domain.entities import StrategyExchangeBinding
from trading.shared_kernel.primitives import UserId


class InMemoryStrategyExchangeBindingRepository(StrategyExchangeBindingRepository):
    def __init__(self) -> None:
        self._bindings: dict[UUID, StrategyExchangeBinding] = {}

    def create(
        self, *, binding: StrategyExchangeBinding
    ) -> StrategyExchangeBinding | None:
        for existing in self._bindings.values():
            if existing.owner_user_id != binding.owner_user_id:
                continue
            if existing.strategy_id != binding.strategy_id:
                continue
            if existing.exchange_connection_id != binding.exchange_connection_id:
                continue
            if existing.usage_mode != binding.usage_mode:
                continue
            if existing.binding_status == "active":
                return None
        self._bindings[binding.binding_id] = binding
        return binding

    def get(
        self, *, owner_user_id: UserId, strategy_id: UUID, binding_id: UUID
    ) -> StrategyExchangeBinding | None:
        binding = self._bindings.get(binding_id)
        if binding is None:
            return None
        if binding.owner_user_id != owner_user_id or binding.strategy_id != strategy_id:
            return None
        return binding

    def list_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> tuple[StrategyExchangeBinding, ...]:
        rows = [
            binding
            for binding in self._bindings.values()
            if binding.owner_user_id == owner_user_id and binding.strategy_id == strategy_id
        ]
        rows.sort(key=lambda item: (item.created_at, str(item.binding_id)))
        return tuple(rows)

    def disable(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        binding_id: UUID,
        disabled_at: datetime,
    ) -> StrategyExchangeBinding | None:
        binding = self.get(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            binding_id=binding_id,
        )
        if binding is None:
            return None
        if binding.binding_status != "active":
            return binding
        disabled = replace(
            binding,
            binding_status="disabled",
            updated_at=disabled_at,
            disabled_at=disabled_at,
        )
        self._bindings[binding_id] = disabled
        return disabled
