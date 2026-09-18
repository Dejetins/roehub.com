"""Scope regression checks for the additive persisted research-origin read."""
from unittest.mock import Mock
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyRepository,
)
from trading.contexts.strategy.adapters.outbound.persistence.in_memory.strategy_backtest_variant_provenance_repository import (  # noqa: E501
    InMemoryStrategyBacktestVariantProvenanceRepository,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.strategy_backtest_variant_provenance_repository import (  # noqa: E501
    PostgresStrategyBacktestVariantProvenanceRepository,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


def test_postgres_origin_query_constrains_all_three_identities() -> None:
    gateway = Mock()
    gateway.fetch_one.return_value = None
    repository = PostgresStrategyBacktestVariantProvenanceRepository(gateway=gateway)
    organization = OrganizationId.from_string("00000000-0000-4000-8000-000000000001")
    user = UserId.from_string("00000000-0000-4000-8000-000000000002")
    strategy = UUID("00000000-0000-4000-8000-000000000003")
    assert repository.find_by_strategy_id(
        organization_id=organization, user_id=user, strategy_id=strategy,
    ) is None
    args = gateway.fetch_one.call_args.kwargs
    assert args["parameters"] == {
        "organization_id": str(organization), "user_id": str(user), "strategy_id": str(strategy),
    }
    for field in ("organization_id", "user_id", "strategy_id"):
        assert f"{field} = %({field})s" in args["query"]


def test_memory_origin_rejects_other_owner_or_organization() -> None:
    repository = InMemoryStrategyBacktestVariantProvenanceRepository(
        strategy_repository=InMemoryStrategyRepository(),
    )
    record = Mock()
    record.organization_id = OrganizationId.from_string("00000000-0000-4000-8000-000000000001")
    record.user_id = UserId.from_string("00000000-0000-4000-8000-000000000002")
    strategy = UUID("00000000-0000-4000-8000-000000000003")
    repository._by_strategy_id[strategy] = record
    assert repository.find_by_strategy_id(
        organization_id=record.organization_id, user_id=record.user_id, strategy_id=strategy,
    ) is record
    assert repository.find_by_strategy_id(
        organization_id=OrganizationId.from_string("00000000-0000-4000-8000-000000000004"),
        user_id=record.user_id, strategy_id=strategy,
    ) is None
    assert repository.find_by_strategy_id(
        organization_id=record.organization_id,
        user_id=UserId.from_string("00000000-0000-4000-8000-000000000004"), strategy_id=strategy,
    ) is None
