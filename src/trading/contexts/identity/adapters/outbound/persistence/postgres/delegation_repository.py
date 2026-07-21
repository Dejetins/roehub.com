from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from trading.contexts.identity.application.authorization.models import CapabilityId
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityGrant,
    DelegationAuditEvent,
    DelegationResourceScope,
)
from trading.contexts.identity.application.ports.delegation_repository import (
    DelegationRepository,
    DelegationRepositoryConflictError,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class PostgresDelegationRepository(DelegationRepository):
    """Transactional PostgreSQL adapter for exact delegated-capability records and audit events."""

    def __init__(self, *, dsn: str) -> None:
        normalized = dsn.strip()
        if not normalized:
            raise ValueError("PostgresDelegationRepository requires non-empty dsn")
        self._dsn = normalized

    def create_or_get_active_grant(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        granted_by_owner_user_id: UserId,
        granted_at: datetime,
        expires_at: datetime,
    ) -> tuple[DelegatedCapabilityGrant, bool]:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                existing = self._select_unrevoked(
                    cursor=cursor,
                    organization_id=organization_id,
                    grantee_user_id=grantee_user_id,
                    capability=capability,
                    resource_scope=resource_scope,
                    for_update=True,
                )
                if existing is not None:
                    if self._same_grant_request(
                        grant=existing,
                        granted_by_owner_user_id=granted_by_owner_user_id,
                        granted_at=granted_at,
                        expires_at=expires_at,
                    ):
                        return existing, False
                    raise DelegationRepositoryConflictError(code="active_delegation_conflict")
                grant = DelegatedCapabilityGrant(
                    delegation_id=uuid4(),
                    organization_id=organization_id,
                    grantee_user_id=grantee_user_id,
                    capability=capability,
                    resource_scope=resource_scope,
                    granted_by_owner_user_id=granted_by_owner_user_id,
                    granted_at=granted_at,
                    expires_at=expires_at,
                )
                cursor.execute(
                    """
                    INSERT INTO identity_delegated_capabilities (
                        delegation_id, organization_id, grantee_user_id, capability_id,
                        resource_scope, granted_by_owner_user_id, granted_at, expires_at,
                        revoked_at, revoked_by_owner_user_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL)
                    """,
                    (
                        str(grant.delegation_id),
                        str(grant.organization_id),
                        str(grant.grantee_user_id),
                        str(grant.capability),
                        grant.resource_scope,
                        str(grant.granted_by_owner_user_id),
                        grant.granted_at,
                        grant.expires_at,
                    ),
                )
                self._insert_audit(
                    cursor=cursor,
                    grant=grant,
                    actor_user_id=granted_by_owner_user_id,
                    action="delegation.granted",
                    created_at=granted_at,
                )
                return grant, True
        except psycopg.errors.UniqueViolation:
            existing = self._get_unrevoked_after_conflict(
                organization_id=organization_id,
                grantee_user_id=grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
            )
            if existing is not None and self._same_grant_request(
                grant=existing,
                granted_by_owner_user_id=granted_by_owner_user_id,
                granted_at=granted_at,
                expires_at=expires_at,
            ):
                return existing, False
            raise DelegationRepositoryConflictError(code="active_delegation_conflict") from None

    def get_grant(self, *, delegation_id: UUID) -> DelegatedCapabilityGrant | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT delegation_id, organization_id, grantee_user_id, capability_id,
                       resource_scope, granted_by_owner_user_id, granted_at, expires_at,
                       revoked_at, revoked_by_owner_user_id
                FROM identity_delegated_capabilities
                WHERE delegation_id = %s
                """,
                (str(delegation_id),),
            )
            row = cursor.fetchone()
        return None if row is None else _grant(row)

    def revoke_grant(
        self,
        *,
        delegation_id: UUID,
        organization_id: OrganizationId,
        revoked_by_owner_user_id: UserId,
        revoked_at: datetime,
    ) -> tuple[DelegatedCapabilityGrant | None, bool]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT delegation_id, organization_id, grantee_user_id, capability_id,
                       resource_scope, granted_by_owner_user_id, granted_at, expires_at,
                       revoked_at, revoked_by_owner_user_id
                FROM identity_delegated_capabilities
                WHERE delegation_id = %s AND organization_id = %s
                FOR UPDATE
                """,
                (str(delegation_id), str(organization_id)),
            )
            row = cursor.fetchone()
            if row is None:
                return None, False
            grant = _grant(row)
            if grant.revoked_at is not None:
                return grant, False
            cursor.execute(
                """
                UPDATE identity_delegated_capabilities
                SET revoked_at = %s, revoked_by_owner_user_id = %s
                WHERE delegation_id = %s
                """,
                (revoked_at, str(revoked_by_owner_user_id), str(delegation_id)),
            )
            revoked = DelegatedCapabilityGrant(
                delegation_id=grant.delegation_id,
                organization_id=grant.organization_id,
                grantee_user_id=grant.grantee_user_id,
                capability=grant.capability,
                resource_scope=grant.resource_scope,
                granted_by_owner_user_id=grant.granted_by_owner_user_id,
                granted_at=grant.granted_at,
                expires_at=grant.expires_at,
                revoked_at=revoked_at,
                revoked_by_owner_user_id=revoked_by_owner_user_id,
            )
            self._insert_audit(
                cursor=cursor,
                grant=revoked,
                actor_user_id=revoked_by_owner_user_id,
                action="delegation.revoked",
                created_at=revoked_at,
            )
            return revoked, True

    def find_active_grant(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        at: datetime,
    ) -> DelegatedCapabilityGrant | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT delegation_id, organization_id, grantee_user_id, capability_id,
                       resource_scope, granted_by_owner_user_id, granted_at, expires_at,
                       revoked_at, revoked_by_owner_user_id
                FROM identity_delegated_capabilities
                WHERE organization_id = %s
                  AND grantee_user_id = %s
                  AND capability_id = %s
                  AND resource_scope = %s
                  AND revoked_at IS NULL
                  AND expires_at > %s
                """,
                (
                    str(organization_id),
                    str(grantee_user_id),
                    str(capability),
                    resource_scope,
                    at,
                ),
            )
            row = cursor.fetchone()
        return None if row is None else _grant(row)

    def list_audit_events(
        self, *, organization_id: OrganizationId
    ) -> tuple[DelegationAuditEvent, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_id, organization_id, actor_user_id, action, target_id,
                       metadata_json, created_at
                FROM identity_administrative_audit_events
                WHERE organization_id = %s
                  AND target_type = 'delegated_capability'
                ORDER BY created_at DESC, event_id DESC
                """,
                (str(organization_id),),
            )
            rows = cursor.fetchall()
        return tuple(_audit_event(row) for row in rows)

    def _get_unrevoked_after_conflict(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
    ) -> DelegatedCapabilityGrant | None:
        with self._connect() as connection, connection.cursor() as cursor:
            return self._select_unrevoked(
                cursor=cursor,
                organization_id=organization_id,
                grantee_user_id=grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
                for_update=False,
            )

    @staticmethod
    def _select_unrevoked(
        *,
        cursor: Any,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        for_update: bool,
    ) -> DelegatedCapabilityGrant | None:
        cursor.execute(
            """
            SELECT delegation_id, organization_id, grantee_user_id, capability_id,
                   resource_scope, granted_by_owner_user_id, granted_at, expires_at,
                   revoked_at, revoked_by_owner_user_id
            FROM identity_delegated_capabilities
            WHERE organization_id = %s
              AND grantee_user_id = %s
              AND capability_id = %s
              AND resource_scope = %s
              AND revoked_at IS NULL
            """
            + (" FOR UPDATE" if for_update else ""),
            (str(organization_id), str(grantee_user_id), str(capability), resource_scope),
        )
        row = cursor.fetchone()
        return None if row is None else _grant(row)

    @staticmethod
    def _same_grant_request(
        *,
        grant: DelegatedCapabilityGrant,
        granted_by_owner_user_id: UserId,
        granted_at: datetime,
        expires_at: datetime,
    ) -> bool:
        return (
            grant.granted_by_owner_user_id == granted_by_owner_user_id
            and grant.granted_at == granted_at
            and grant.expires_at == expires_at
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    def _insert_audit(
        self,
        *,
        cursor: Any,
        grant: DelegatedCapabilityGrant,
        actor_user_id: UserId,
        action: str,
        created_at: datetime,
    ) -> None:
        cursor.execute(
            """
            SELECT installation_id
            FROM identity_organizations
            WHERE organization_id = %s
            """,
            (str(grant.organization_id),),
        )
        row = cursor.fetchone()
        if row is None:
            raise DelegationRepositoryConflictError(code="organization_not_found")
        cursor.execute(
            """
            INSERT INTO identity_administrative_audit_events (
                event_id, installation_id, organization_id, actor_user_id,
                action, target_type, target_id, outcome, metadata_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, 'delegated_capability', %s, 'succeeded', %s::jsonb, %s)
            """,
            (
                str(uuid4()),
                str(InstallationId(UUID(str(row["installation_id"])))),
                str(grant.organization_id),
                str(actor_user_id),
                action,
                str(grant.delegation_id),
                json.dumps(
                    {
                        "capability_id": str(grant.capability),
                        "grantee_user_id": str(grant.grantee_user_id),
                        "resource_scope": grant.resource_scope,
                    },
                    sort_keys=True,
                ),
                created_at,
            ),
        )


def _grant(row: Mapping[str, Any]) -> DelegatedCapabilityGrant:
    return DelegatedCapabilityGrant(
        delegation_id=UUID(str(row["delegation_id"])),
        organization_id=OrganizationId(UUID(str(row["organization_id"]))),
        grantee_user_id=UserId(UUID(str(row["grantee_user_id"]))),
        capability=CapabilityId(str(row["capability_id"])),
        resource_scope=DelegationResourceScope(str(row["resource_scope"])),
        granted_by_owner_user_id=UserId(UUID(str(row["granted_by_owner_user_id"]))),
        granted_at=_datetime(row["granted_at"]),
        expires_at=_datetime(row["expires_at"]),
        revoked_at=None if row["revoked_at"] is None else _datetime(row["revoked_at"]),
        revoked_by_owner_user_id=(
            None
            if row["revoked_by_owner_user_id"] is None
            else UserId(UUID(str(row["revoked_by_owner_user_id"])))
        ),
    )


def _audit_event(row: Mapping[str, Any]) -> DelegationAuditEvent:
    metadata = row["metadata_json"]
    if not isinstance(metadata, dict):
        raise ValueError("Delegation audit metadata must be an object")
    return DelegationAuditEvent(
        event_id=UUID(str(row["event_id"])),
        organization_id=OrganizationId(UUID(str(row["organization_id"]))),
        actor_user_id=UserId(UUID(str(row["actor_user_id"]))),
        action=str(row["action"]),
        target_id=str(row["target_id"]),
        metadata={str(key): str(value) for key, value in metadata.items()},
        created_at=_datetime(row["created_at"]),
    )


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("PostgreSQL delegation datetime must be timezone-aware")
    return value.astimezone(timezone.utc)
