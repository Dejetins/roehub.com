from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.shared_kernel.primitives import OrganizationId, UserId


@dataclass(frozen=True, slots=True)
class ResearchOrganizationScope:
    """Server-resolved organization scope for one authenticated research actor."""

    organization_id: OrganizationId
    user_id: UserId


class ResearchOrganizationScopeResolver(Protocol):
    """Resolve exactly one active organization without trusting request payload scope."""

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        """Return the only active organization membership or fail closed."""
        ...


__all__ = ["ResearchOrganizationScope", "ResearchOrganizationScopeResolver"]
