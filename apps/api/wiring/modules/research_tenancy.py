from __future__ import annotations

from typing import Mapping
from uuid import UUID

from trading.contexts.backtest.adapters.outbound import (
    PostgresResearchOrganizationScopeResolver,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports import (
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_DEVELOPMENT_ORGANIZATION_ID = OrganizationId(
    UUID("00000000-0000-4000-8000-000000000010")
)


class DevelopmentOrganizationScopeResolver(ResearchOrganizationScopeResolver):
    """Deterministic non-production scope used only without a configured database."""

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=_DEVELOPMENT_ORGANIZATION_ID,
            user_id=user_id,
        )


def build_research_organization_scope_resolver(
    *, environ: Mapping[str, str]
) -> ResearchOrganizationScopeResolver | None:
    """Build the shared fail-closed research scope resolver from PostgreSQL."""
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return None
    return PostgresResearchOrganizationScopeResolver(
        gateway=PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    )


def build_required_organization_scope_resolver(
    *, environ: Mapping[str, str]
) -> ResearchOrganizationScopeResolver:
    resolver = build_research_organization_scope_resolver(environ=environ)
    if resolver is not None:
        return resolver
    if environ.get("ROEHUB_ENV", "dev").strip().lower() == "prod":
        raise ValueError("PostgreSQL organization scope resolver is required in prod")
    return DevelopmentOrganizationScopeResolver()


__all__ = [
    "DevelopmentOrganizationScopeResolver",
    "build_research_organization_scope_resolver",
    "build_required_organization_scope_resolver",
]
