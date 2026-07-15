from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import ExecutionRiskContext
from trading.shared_kernel.primitives import OrganizationId, UserId


class ExecutionRiskContextResolutionError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionRiskContextQuery:
    organization_id: OrganizationId
    owner_user_id: UserId
    source_event_id: UUID
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str


class ExecutionRiskContextResolver(Protocol):
    def resolve(self, *, query: ExecutionRiskContextQuery) -> ExecutionRiskContext: ...


class FailClosedExecutionRiskContextResolver(ExecutionRiskContextResolver):
    def resolve(self, *, query: ExecutionRiskContextQuery) -> ExecutionRiskContext:
        _ = query
        raise ExecutionRiskContextResolutionError(reason="risk_state_unavailable")
