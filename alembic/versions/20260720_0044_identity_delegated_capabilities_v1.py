"""Add exact, expiring organization capability delegations.

Revision ID: 20260720_0044
Revises: 20260711_0043
"""

from __future__ import annotations

from alembic import op

revision = "20260720_0044"
down_revision = "20260711_0043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the minimal reversible storage for exact delegated capabilities."""
    op.execute(
        """
        CREATE TABLE identity_delegated_capabilities (
            delegation_id UUID PRIMARY KEY,
            organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id)
                ON DELETE CASCADE,
            grantee_user_id UUID NOT NULL,
            capability_id TEXT NOT NULL,
            resource_scope TEXT NOT NULL,
            granted_by_owner_user_id UUID NOT NULL,
            granted_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ NULL,
            revoked_by_owner_user_id UUID NULL,
            CONSTRAINT identity_delegated_capabilities_capability_chk
                CHECK (char_length(trim(capability_id)) > 0),
            CONSTRAINT identity_delegated_capabilities_scope_chk
                CHECK (resource_scope = 'organization'),
            CONSTRAINT identity_delegated_capabilities_expiry_chk
                CHECK (expires_at > granted_at),
            CONSTRAINT identity_delegated_capabilities_revoke_time_chk
                CHECK (revoked_at IS NULL OR revoked_at >= granted_at),
            CONSTRAINT identity_delegated_capabilities_revoke_actor_chk
                CHECK (
                    (revoked_at IS NULL AND revoked_by_owner_user_id IS NULL)
                    OR (revoked_at IS NOT NULL AND revoked_by_owner_user_id IS NOT NULL)
                ),
            FOREIGN KEY (organization_id, grantee_user_id)
                REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY (organization_id, granted_by_owner_user_id)
                REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
            FOREIGN KEY (organization_id, revoked_by_owner_user_id)
                REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX idx_identity_delegated_capabilities_one_unrevoked
            ON identity_delegated_capabilities (
                organization_id,
                grantee_user_id,
                capability_id,
                resource_scope
            )
            WHERE revoked_at IS NULL
        """
    )
    op.execute(
        """
        CREATE INDEX idx_identity_delegated_capabilities_evaluation
            ON identity_delegated_capabilities (
                organization_id,
                grantee_user_id,
                capability_id,
                resource_scope,
                expires_at
            )
            WHERE revoked_at IS NULL
        """
    )


def downgrade() -> None:
    """Remove delegated-capability storage without changing existing identity records."""
    op.execute("DROP TABLE identity_delegated_capabilities")
