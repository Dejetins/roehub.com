from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from trading.contexts.identity.application.ports.organization_repository import (
    OrganizationRepository,
    OrganizationRepositoryInvariantError,
)
from trading.contexts.identity.domain.entities import (
    AdministrativeAuditEvent,
    Installation,
    Organization,
    OrganizationAccess,
    OrganizationInvitation,
    OrganizationMembership,
    OrganizationRole,
    PluginPermission,
    PluginPermissionGrant,
    SupportAccessGrant,
    permissions_for_role,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class PostgresOrganizationRepository(OrganizationRepository):
    """Transactional PostgreSQL adapter for installation, organization and RBAC state."""

    def __init__(self, *, dsn: str) -> None:
        normalized = dsn.strip()
        if not normalized:
            raise ValueError("PostgresOrganizationRepository requires non-empty dsn")
        self._dsn = normalized

    def get_installation(self) -> Installation | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT installation_id, display_name, created_at
                FROM identity_installations
                WHERE singleton_key = TRUE
                """
            )
            row = cursor.fetchone()
        return None if row is None else _installation(row)

    def is_installation_owner(self, *, user_id: UserId) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM identity_installation_owners WHERE user_id = %s
                ) AS allowed
                """,
                (str(user_id),),
            )
            row = cursor.fetchone()
        return bool(row and row["allowed"])

    def bootstrap_installation(
        self,
        *,
        owner_user_id: UserId,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        created_at: datetime,
    ) -> tuple[Installation, Organization]:
        installation_id = InstallationId(uuid4())
        organization_id = OrganizationId(uuid4())
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO identity_installations (
                        installation_id, singleton_key, display_name, created_at
                    ) VALUES (%s, TRUE, %s, %s)
                    """,
                    (str(installation_id), installation_name, created_at),
                )
                cursor.execute(
                    """
                    INSERT INTO identity_installation_owners (
                        installation_id, user_id, granted_by_user_id, granted_at
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        str(installation_id),
                        str(owner_user_id),
                        str(owner_user_id),
                        created_at,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO identity_organizations (
                        organization_id, installation_id, slug, display_name,
                        status, created_at
                    ) VALUES (%s, %s, %s, %s, 'active', %s)
                    """,
                    (
                        str(organization_id),
                        str(installation_id),
                        organization_slug,
                        organization_name,
                        created_at,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO identity_memberships (
                        organization_id, user_id, role, status, created_at, updated_at
                    ) VALUES (%s, %s, 'owner', 'active', %s, %s)
                    """,
                    (str(organization_id), str(owner_user_id), created_at, created_at),
                )
                self._insert_audit(
                    cursor=cursor,
                    installation_id=installation_id,
                    organization_id=organization_id,
                    actor_user_id=owner_user_id,
                    action="installation.bootstrap",
                    target_type="installation",
                    target_id=str(installation_id),
                    metadata={"organization_id": str(organization_id)},
                    created_at=created_at,
                )
        except psycopg.errors.UniqueViolation as error:
            raise OrganizationRepositoryInvariantError(
                code="installation_already_initialized"
            ) from error
        except psycopg.errors.ForeignKeyViolation as error:
            raise OrganizationRepositoryInvariantError(code="user_not_found") from error
        return (
            Installation(
                installation_id=installation_id,
                display_name=installation_name,
                created_at=created_at,
            ),
            Organization(
                organization_id=organization_id,
                installation_id=installation_id,
                slug=organization_slug,
                display_name=organization_name,
                created_at=created_at,
            ),
        )

    def create_organization(
        self,
        *,
        actor_user_id: UserId,
        slug: str,
        display_name: str,
        created_at: datetime,
    ) -> Organization:
        organization_id = OrganizationId(uuid4())
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                installation_id = self._installation_id(cursor=cursor)
                cursor.execute(
                    """
                    INSERT INTO identity_organizations (
                        organization_id, installation_id, slug, display_name,
                        status, created_at
                    ) VALUES (%s, %s, %s, %s, 'active', %s)
                    """,
                    (
                        str(organization_id),
                        str(installation_id),
                        slug,
                        display_name,
                        created_at,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO identity_memberships (
                        organization_id, user_id, role, status, created_at, updated_at
                    ) VALUES (%s, %s, 'owner', 'active', %s, %s)
                    """,
                    (str(organization_id), str(actor_user_id), created_at, created_at),
                )
                self._insert_audit(
                    cursor=cursor,
                    installation_id=installation_id,
                    organization_id=organization_id,
                    actor_user_id=actor_user_id,
                    action="organization.created",
                    target_type="organization",
                    target_id=str(organization_id),
                    metadata={"slug": slug},
                    created_at=created_at,
                )
        except psycopg.errors.UniqueViolation as error:
            raise OrganizationRepositoryInvariantError(code="organization_slug_conflict") from error
        return Organization(
            organization_id=organization_id,
            installation_id=installation_id,
            slug=slug,
            display_name=display_name,
            created_at=created_at,
        )

    def list_accesses_for_user(self, *, user_id: UserId) -> tuple[OrganizationAccess, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    organization.organization_id,
                    organization.installation_id,
                    organization.slug,
                    organization.display_name,
                    organization.status,
                    organization.created_at,
                    membership.role
                FROM identity_organizations AS organization
                JOIN identity_memberships AS membership
                  ON membership.organization_id = organization.organization_id
                 AND membership.user_id = %s
                 AND membership.status = 'active'
                WHERE organization.status = 'active'
                ORDER BY organization.slug, organization.organization_id
                """,
                (str(user_id),),
            )
            rows = cursor.fetchall()
        result: list[OrganizationAccess] = []
        for row in rows:
            role = cast(OrganizationRole, str(row["role"]))
            result.append(
                OrganizationAccess(
                    organization=_organization(row),
                    role=role,
                    permissions=permissions_for_role(role),
                )
            )
        return tuple(result)

    def get_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
    ) -> OrganizationMembership | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT organization_id, user_id, role, status, created_at
                FROM identity_memberships
                WHERE organization_id = %s AND user_id = %s
                """,
                (str(organization_id), str(user_id)),
            )
            row = cursor.fetchone()
        return None if row is None else _membership(row)

    def list_memberships(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[OrganizationMembership, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT organization_id, user_id, role, status, created_at
                FROM identity_memberships
                WHERE organization_id = %s
                ORDER BY role, user_id
                """,
                (str(organization_id),),
            )
            rows = cursor.fetchall()
        return tuple(_membership(row) for row in rows)

    def add_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        created_at: datetime,
    ) -> OrganizationMembership:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO identity_memberships (
                        organization_id, user_id, role, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, 'active', %s, %s)
                    """,
                    (str(organization_id), str(user_id), role, created_at, created_at),
                )
                self._audit_for_organization(
                    cursor=cursor,
                    organization_id=organization_id,
                    actor_user_id=actor_user_id,
                    action="membership.created",
                    target_type="membership",
                    target_id=str(user_id),
                    metadata={"role": role},
                    created_at=created_at,
                )
        except psycopg.errors.UniqueViolation as error:
            raise OrganizationRepositoryInvariantError(code="membership_conflict") from error
        except psycopg.errors.ForeignKeyViolation as error:
            raise OrganizationRepositoryInvariantError(code="user_not_found") from error
        return OrganizationMembership(
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            status="active",
            created_at=created_at,
        )

    def set_membership_role(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        changed_at: datetime,
    ) -> OrganizationMembership:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT organization_id, user_id, role, status, created_at
                FROM identity_memberships
                WHERE organization_id = %s AND user_id = %s
                FOR UPDATE
                """,
                (str(organization_id), str(user_id)),
            )
            row = cursor.fetchone()
            if row is None:
                raise OrganizationRepositoryInvariantError(code="membership_not_found")
            current = _membership(row)
            if current.role == "owner" and role != "owner":
                self._assert_not_last_owner(cursor=cursor, organization_id=organization_id)
            cursor.execute(
                """
                UPDATE identity_memberships
                SET role = %s, updated_at = %s
                WHERE organization_id = %s AND user_id = %s
                """,
                (role, changed_at, str(organization_id), str(user_id)),
            )
            self._audit_for_organization(
                cursor=cursor,
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="membership.role_changed",
                target_type="membership",
                target_id=str(user_id),
                metadata={"from_role": current.role, "to_role": role},
                created_at=changed_at,
            )
        return OrganizationMembership(
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            status=current.status,
            created_at=current.created_at,
        )

    def remove_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        actor_user_id: UserId,
        removed_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT organization_id, user_id, role, status, created_at
                FROM identity_memberships
                WHERE organization_id = %s AND user_id = %s
                FOR UPDATE
                """,
                (str(organization_id), str(user_id)),
            )
            row = cursor.fetchone()
            if row is None:
                raise OrganizationRepositoryInvariantError(code="membership_not_found")
            current = _membership(row)
            if current.role == "owner" and current.status == "active":
                self._assert_not_last_owner(cursor=cursor, organization_id=organization_id)
            cursor.execute(
                "DELETE FROM identity_memberships WHERE organization_id = %s AND user_id = %s",
                (str(organization_id), str(user_id)),
            )
            self._audit_for_organization(
                cursor=cursor,
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="membership.removed",
                target_type="membership",
                target_id=str(user_id),
                metadata={"previous_role": current.role},
                created_at=removed_at,
            )

    def create_invitation(
        self,
        *,
        organization_id: OrganizationId,
        recipient_email_sha256: str,
        role: OrganizationRole,
        actor_user_id: UserId,
        expires_at: datetime,
        created_at: datetime,
    ) -> OrganizationInvitation:
        invitation_id = uuid4()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO identity_invitations (
                        invitation_id, organization_id, recipient_email_sha256, role,
                        status, created_by_user_id, expires_at, created_at
                    ) VALUES (%s, %s, %s, %s, 'pending', %s, %s, %s)
                    """,
                    (
                        str(invitation_id),
                        str(organization_id),
                        recipient_email_sha256,
                        role,
                        str(actor_user_id),
                        expires_at,
                        created_at,
                    ),
                )
                self._audit_for_organization(
                    cursor=cursor,
                    organization_id=organization_id,
                    actor_user_id=actor_user_id,
                    action="invitation.created",
                    target_type="invitation",
                    target_id=str(invitation_id),
                    metadata={"role": role, "recipient_hash": recipient_email_sha256},
                    created_at=created_at,
                )
        except psycopg.errors.UniqueViolation as error:
            raise OrganizationRepositoryInvariantError(code="invitation_conflict") from error
        return OrganizationInvitation(
            invitation_id=invitation_id,
            organization_id=organization_id,
            role=role,
            expires_at=expires_at,
            created_at=created_at,
        )

    def set_plugin_permission(
        self,
        *,
        organization_id: OrganizationId,
        plugin_id: str,
        user_id: UserId,
        permission: PluginPermission,
        actor_user_id: UserId,
        updated_at: datetime,
    ) -> PluginPermissionGrant:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO identity_plugin_permissions (
                        organization_id, plugin_id, user_id, permission,
                        granted_by_user_id, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (organization_id, plugin_id, user_id)
                    DO UPDATE SET
                        permission = EXCLUDED.permission,
                        granted_by_user_id = EXCLUDED.granted_by_user_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        str(organization_id),
                        plugin_id,
                        str(user_id),
                        permission,
                        str(actor_user_id),
                        updated_at,
                        updated_at,
                    ),
                )
                self._audit_for_organization(
                    cursor=cursor,
                    organization_id=organization_id,
                    actor_user_id=actor_user_id,
                    action="plugin.permission_set",
                    target_type="plugin_permission",
                    target_id=f"{plugin_id}:{user_id}",
                    metadata={"permission": permission},
                    created_at=updated_at,
                )
        except psycopg.errors.ForeignKeyViolation as error:
            raise OrganizationRepositoryInvariantError(code="membership_not_found") from error
        return PluginPermissionGrant(
            organization_id=organization_id,
            plugin_id=plugin_id,
            user_id=user_id,
            permission=permission,
            updated_at=updated_at,
        )

    def grant_support_access(
        self,
        *,
        support_user_id: UserId,
        actor_user_id: UserId,
        reason: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> SupportAccessGrant:
        grant_id = uuid4()
        with self._connect() as connection, connection.cursor() as cursor:
            installation_id = self._installation_id(cursor=cursor)
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"support-access:{installation_id}:{support_user_id}",),
            )
            cursor.execute(
                """
                UPDATE identity_support_access_grants
                SET revoked_at = expires_at
                WHERE installation_id = %s
                  AND support_user_id = %s
                  AND revoked_at IS NULL
                  AND expires_at <= %s
                RETURNING grant_id
                """,
                (str(installation_id), str(support_user_id), created_at),
            )
            expired_grants = cursor.fetchall()
            for expired_grant in expired_grants:
                self._insert_audit(
                    cursor=cursor,
                    installation_id=installation_id,
                    organization_id=None,
                    actor_user_id=actor_user_id,
                    action="support_access.expired",
                    target_type="support_access",
                    target_id=str(expired_grant["grant_id"]),
                    metadata={},
                    created_at=created_at,
                )
            try:
                cursor.execute(
                    """
                    INSERT INTO identity_support_access_grants (
                        grant_id, installation_id, support_user_id, granted_by_user_id,
                        reason, expires_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(grant_id),
                        str(installation_id),
                        str(support_user_id),
                        str(actor_user_id),
                        reason,
                        expires_at,
                        created_at,
                    ),
                )
            except psycopg.errors.UniqueViolation as error:
                raise OrganizationRepositoryInvariantError(
                    code="support_access_conflict"
                ) from error
            self._insert_audit(
                cursor=cursor,
                installation_id=installation_id,
                organization_id=None,
                actor_user_id=actor_user_id,
                action="support_access.granted",
                target_type="support_access",
                target_id=str(grant_id),
                metadata={
                    "support_user_id": str(support_user_id),
                    "expires_at": expires_at.isoformat(),
                },
                created_at=created_at,
            )
        return SupportAccessGrant(
            grant_id=grant_id,
            installation_id=installation_id,
            support_user_id=support_user_id,
            expires_at=expires_at,
            created_at=created_at,
        )

    def list_audit_events(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[AdministrativeAuditEvent, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_id, installation_id, organization_id, actor_user_id,
                       action, target_type, target_id, outcome, metadata_json, created_at
                FROM identity_administrative_audit_events
                WHERE organization_id = %s
                ORDER BY created_at DESC, event_id DESC
                LIMIT %s
                """,
                (str(organization_id), limit),
            )
            rows = cursor.fetchall()
        return tuple(_audit_event(row) for row in rows)

    def record_rejected_event(
        self,
        *,
        organization_id: OrganizationId | None,
        actor_user_id: UserId,
        action: str,
        target_type: str,
        target_id: str,
        reason_code: str,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            installation_id = self._installation_id(cursor=cursor)
            effective_organization_id = organization_id
            if organization_id is not None:
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM identity_organizations "
                    "WHERE organization_id = %s) AS exists",
                    (str(organization_id),),
                )
                row = cursor.fetchone()
                if row is None or not row["exists"]:
                    effective_organization_id = None
            self._insert_audit(
                cursor=cursor,
                installation_id=installation_id,
                organization_id=effective_organization_id,
                actor_user_id=actor_user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                metadata={"reason_code": reason_code},
                created_at=created_at,
                outcome="rejected",
            )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    @staticmethod
    def _installation_id(*, cursor: Any) -> InstallationId:
        cursor.execute(
            "SELECT installation_id FROM identity_installations WHERE singleton_key = TRUE"
        )
        row = cursor.fetchone()
        if row is None:
            raise OrganizationRepositoryInvariantError(code="installation_not_initialized")
        return InstallationId(UUID(str(row["installation_id"])))

    def _audit_for_organization(
        self,
        *,
        cursor: Any,
        organization_id: OrganizationId,
        actor_user_id: UserId,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, str],
        created_at: datetime,
    ) -> None:
        cursor.execute(
            "SELECT installation_id FROM identity_organizations WHERE organization_id = %s",
            (str(organization_id),),
        )
        row = cursor.fetchone()
        if row is None:
            raise OrganizationRepositoryInvariantError(code="organization_not_found")
        self._insert_audit(
            cursor=cursor,
            installation_id=InstallationId(UUID(str(row["installation_id"]))),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata=metadata,
            created_at=created_at,
        )

    @staticmethod
    def _insert_audit(
        *,
        cursor: Any,
        installation_id: InstallationId,
        organization_id: OrganizationId | None,
        actor_user_id: UserId,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, str],
        created_at: datetime,
        outcome: str = "succeeded",
    ) -> None:
        cursor.execute(
            """
            INSERT INTO identity_administrative_audit_events (
                event_id, installation_id, organization_id, actor_user_id,
                action, target_type, target_id, outcome, metadata_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (
                str(uuid4()),
                str(installation_id),
                None if organization_id is None else str(organization_id),
                str(actor_user_id),
                action,
                target_type,
                target_id,
                outcome,
                json.dumps(metadata, sort_keys=True),
                created_at,
            ),
        )

    @staticmethod
    def _assert_not_last_owner(*, cursor: Any, organization_id: OrganizationId) -> None:
        cursor.execute(
            """
            SELECT user_id
            FROM identity_memberships
            WHERE organization_id = %s AND role = 'owner' AND status = 'active'
            FOR UPDATE
            """,
            (str(organization_id),),
        )
        rows = cursor.fetchall()
        if len(rows) <= 1:
            raise OrganizationRepositoryInvariantError(code="last_owner")


def _installation(row: Mapping[str, Any]) -> Installation:
    return Installation(
        installation_id=InstallationId(UUID(str(row["installation_id"]))),
        display_name=str(row["display_name"]),
        created_at=_datetime(row["created_at"]),
    )


def _organization(row: Mapping[str, Any]) -> Organization:
    return Organization(
        organization_id=OrganizationId(UUID(str(row["organization_id"]))),
        installation_id=InstallationId(UUID(str(row["installation_id"]))),
        slug=str(row["slug"]),
        display_name=str(row["display_name"]),
        status=cast(Any, str(row["status"])),
        created_at=_datetime(row["created_at"]),
    )


def _membership(row: Mapping[str, Any]) -> OrganizationMembership:
    return OrganizationMembership(
        organization_id=OrganizationId(UUID(str(row["organization_id"]))),
        user_id=UserId(UUID(str(row["user_id"]))),
        role=cast(OrganizationRole, str(row["role"])),
        status=cast(Any, str(row["status"])),
        created_at=_datetime(row["created_at"]),
    )


def _audit_event(row: Mapping[str, Any]) -> AdministrativeAuditEvent:
    raw_metadata = row["metadata_json"]
    if not isinstance(raw_metadata, dict):
        raise ValueError("Administrative audit metadata must be an object")
    return AdministrativeAuditEvent(
        event_id=UUID(str(row["event_id"])),
        installation_id=InstallationId(UUID(str(row["installation_id"]))),
        organization_id=(
            None
            if row["organization_id"] is None
            else OrganizationId(UUID(str(row["organization_id"])))
        ),
        actor_user_id=UserId(UUID(str(row["actor_user_id"]))),
        action=str(row["action"]),
        target_type=str(row["target_type"]),
        target_id=str(row["target_id"]),
        outcome=cast(Any, str(row["outcome"])),
        metadata={str(key): str(value) for key, value in raw_metadata.items()},
        created_at=_datetime(row["created_at"]),
    )


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("PostgreSQL datetime must be timezone-aware")
    return value.astimezone(timezone.utc)
