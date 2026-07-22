from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg
import pytest

from trading.contexts.identity.adapters.outbound.persistence.in_memory.delegation_repository import (  # noqa: E501
    InMemoryDelegationRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.delegation_repository import (  # noqa: E501
    PostgresDelegationRepository,
)
from trading.contexts.identity.application.authorization.models import CapabilityId
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityGrant,
    DelegationResourceScope,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
SCOPE = DelegationResourceScope.ORGANIZATION


def test_postgres_repository_matches_in_memory_lifecycle_and_concurrent_idempotency() -> None:
    dsn = os.environ.get("ROEHUB_TEST_DELEGATION_PG_DSN")
    if dsn is None:
        pytest.skip("ROEHUB_TEST_DELEGATION_PG_DSN is not configured")

    installation_id = InstallationId(uuid4())
    organization_id = OrganizationId(uuid4())
    owner_user_id = UserId(uuid4())
    grantee_user_id = UserId(uuid4())
    _seed_delegation_database(
        dsn=dsn,
        installation_id=installation_id,
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        grantee_user_id=grantee_user_id,
    )
    assert _resource_scope_constraint(dsn=dsn) == "CHECK ((resource_scope = 'organization'::text))"
    postgres = PostgresDelegationRepository(dsn=dsn)
    in_memory = InMemoryDelegationRepository()
    request = {
        "organization_id": organization_id,
        "grantee_user_id": grantee_user_id,
        "capability": CapabilityId.MODELS_MANAGE,
        "resource_scope": SCOPE,
        "granted_by_owner_user_id": owner_user_id,
        "granted_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        postgres_results = list(
            executor.map(lambda _: postgres.create_or_get_active_grant(**request), range(16))
        )
    in_memory_grant, in_memory_created = in_memory.create_or_get_active_grant(**request)
    postgres_grant = postgres_results[0][0]
    postgres_active = postgres.find_active_grant(
        organization_id=organization_id,
        grantee_user_id=grantee_user_id,
        capability=CapabilityId.MODELS_MANAGE,
        resource_scope=SCOPE,
        at=NOW,
    )
    in_memory_active = in_memory.find_active_grant(
        organization_id=organization_id,
        grantee_user_id=grantee_user_id,
        capability=CapabilityId.MODELS_MANAGE,
        resource_scope=SCOPE,
        at=NOW,
    )

    assert {result[0].delegation_id for result in postgres_results} == {
        postgres_grant.delegation_id
    }
    assert sum(created for _, created in postgres_results) == 1
    assert in_memory_created is True
    assert postgres_active is not None
    assert in_memory_active is not None
    assert _grant_shape(postgres_active) == _grant_shape(in_memory_active)

    postgres_revoked, postgres_changed = postgres.revoke_grant(
        delegation_id=postgres_grant.delegation_id,
        organization_id=organization_id,
        revoked_by_owner_user_id=owner_user_id,
        revoked_at=NOW + timedelta(minutes=1),
    )
    in_memory_revoked, in_memory_changed = in_memory.revoke_grant(
        delegation_id=in_memory_grant.delegation_id,
        organization_id=organization_id,
        revoked_by_owner_user_id=owner_user_id,
        revoked_at=NOW + timedelta(minutes=1),
    )
    postgres_revoked_again, postgres_changed_again = postgres.revoke_grant(
        delegation_id=postgres_grant.delegation_id,
        organization_id=organization_id,
        revoked_by_owner_user_id=owner_user_id,
        revoked_at=NOW + timedelta(minutes=2),
    )

    assert postgres_changed is True
    assert in_memory_changed is True
    assert postgres_revoked is not None
    assert in_memory_revoked is not None
    assert _grant_shape(postgres_revoked) == _grant_shape(in_memory_revoked)
    assert postgres_revoked_again == postgres_revoked
    assert postgres_changed_again is False
    assert (
        postgres.find_active_grant(
            organization_id=organization_id,
            grantee_user_id=grantee_user_id,
            capability=CapabilityId.MODELS_MANAGE,
            resource_scope=SCOPE,
            at=NOW,
        )
        is None
    )
    assert [
        event.action for event in postgres.list_audit_events(organization_id=organization_id)
    ] == [
        "delegation.revoked",
        "delegation.granted",
    ]
    assert [
        event.metadata for event in postgres.list_audit_events(organization_id=organization_id)
    ] == [
        {
            "capability_id": "models.manage",
            "grantee_user_id": str(grantee_user_id),
            "resource_scope": str(SCOPE),
        },
        {
            "capability_id": "models.manage",
            "grantee_user_id": str(grantee_user_id),
            "resource_scope": str(SCOPE),
        },
    ]


def _seed_delegation_database(
    *,
    dsn: str,
    installation_id: InstallationId,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    grantee_user_id: UserId,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            TRUNCATE identity_administrative_audit_events, identity_delegated_capabilities,
                     identity_memberships, identity_organizations, identity_installations,
                     identity_users CASCADE
            """
        )
        cursor.executemany(
            "INSERT INTO identity_users (user_id, created_at) VALUES (%s, %s)",
            [(str(owner_user_id), NOW), (str(grantee_user_id), NOW)],
        )
        cursor.execute(
            """
            INSERT INTO identity_installations (installation_id, display_name, created_at)
            VALUES (%s, %s, %s)
            """,
            (str(installation_id), "Delegation Test Installation", NOW),
        )
        cursor.execute(
            """
            INSERT INTO identity_organizations (
                organization_id, installation_id, slug, display_name, created_at
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                str(organization_id),
                str(installation_id),
                "delegation-test",
                "Delegation Test Organization",
                NOW,
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_memberships (
                organization_id, user_id, role, status, created_at, updated_at
            )
            VALUES (%s, %s, %s, 'active', %s, %s)
            """,
            [
                (str(organization_id), str(owner_user_id), "owner", NOW, NOW),
                (str(organization_id), str(grantee_user_id), "admin", NOW, NOW),
            ],
        )


def _grant_shape(grant: DelegatedCapabilityGrant) -> tuple[object, ...]:
    return (
        grant.organization_id,
        grant.grantee_user_id,
        grant.capability,
        grant.resource_scope,
        grant.granted_by_owner_user_id,
        grant.granted_at,
        grant.expires_at,
        grant.revoked_at,
        grant.revoked_by_owner_user_id,
    )


def _resource_scope_constraint(*, dsn: str) -> str:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT pg_get_constraintdef(constraint_id)
            FROM (
                SELECT oid AS constraint_id
                FROM pg_constraint
                WHERE conrelid = 'identity_delegated_capabilities'::regclass
                  AND conname = 'identity_delegated_capabilities_scope_chk'
            ) AS constraints
            """
        )
        row = cursor.fetchone()
    assert row is not None
    return str(row[0])
