from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

import pytest

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    research_organization_scope,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


class _CapturingGateway:
    def __init__(self, rows: tuple[Mapping[str, Any], ...]) -> None:
        self.rows = rows
        self.query = ""
        self.parameters: Mapping[str, Any] = {}

    def fetch_one(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        raise AssertionError("fetch_one is not expected")

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        self.query = query
        self.parameters = parameters
        return self.rows

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        raise AssertionError("execute is not expected")


def test_resolver_derives_the_only_active_organization_server_side() -> None:
    user_id = UserId(uuid4())
    organization_id = uuid4()
    gateway = _CapturingGateway(({"organization_id": organization_id},))

    scope = research_organization_scope.PostgresResearchOrganizationScopeResolver(
        gateway=gateway
    ).resolve(user_id=user_id)

    assert scope.organization_id.value == organization_id
    assert scope.user_id == user_id
    assert gateway.parameters == {"user_id": str(user_id)}
    assert "memberships.status = 'active'" in gateway.query
    assert "organizations.status = 'active'" in gateway.query
    assert "LIMIT 2" in gateway.query


@pytest.mark.parametrize(
    ("rows", "expected_code"),
    [
        ((), "research.organization_scope_forbidden"),
        (
            (
                {"organization_id": uuid4()},
                {"organization_id": uuid4()},
            ),
            "research.organization_scope_ambiguous",
        ),
    ],
)
def test_resolver_fails_closed_without_exactly_one_active_organization(
    rows: tuple[Mapping[str, Any], ...],
    expected_code: str,
) -> None:
    gateway = _CapturingGateway(rows)

    with pytest.raises(RoehubError) as error_info:
        research_organization_scope.PostgresResearchOrganizationScopeResolver(
            gateway=gateway
        ).resolve(user_id=UserId(uuid4()))

    assert error_info.value.code == expected_code
