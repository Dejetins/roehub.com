from __future__ import annotations

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports.research_organization_scope import (
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresResearchOrganizationScopeResolver(ResearchOrganizationScopeResolver):
    """Resolve the sole active organization membership for one research request."""

    def __init__(self, *, gateway: BacktestPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "PostgresResearchOrganizationScopeResolver requires gateway"
            )
        self._gateway = gateway

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        rows = self._gateway.fetch_all(
            query="""
                SELECT memberships.organization_id
                FROM identity_memberships AS memberships
                JOIN identity_organizations AS organizations
                  ON organizations.organization_id = memberships.organization_id
                WHERE memberships.user_id = %(user_id)s
                  AND memberships.status = 'active'
                  AND organizations.status = 'active'
                ORDER BY memberships.organization_id
                LIMIT 2
            """,
            parameters={"user_id": str(user_id)},
        )
        if not rows:
            raise RoehubError(
                code="research.organization_scope_forbidden",
                message="No active organization is available for research operations",
                details={"reason": "no_active_membership"},
            )
        if len(rows) != 1:
            raise RoehubError(
                code="research.organization_scope_ambiguous",
                message="An active organization must be selected before research operations",
                details={"reason": "multiple_active_memberships"},
            )
        organization_id = OrganizationId.from_string(str(rows[0]["organization_id"]))
        return ResearchOrganizationScope(
            organization_id=organization_id,
            user_id=user_id,
        )


__all__ = ["PostgresResearchOrganizationScopeResolver"]
