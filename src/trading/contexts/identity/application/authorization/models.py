from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import OrganizationId, UserId


class CapabilityId(StrEnum):
    """Stable, server-owned capability identifiers; never derive these from client input."""

    AUTH_SESSION_MANAGE = "auth.session.manage"
    SETUP_BOOTSTRAP = "setup.bootstrap"
    PREFERENCES_MANAGE_PERSONAL = "preferences.manage_personal"
    DASHBOARD_READ = "dashboard.read"
    DATA_CATALOG_READ = "data.catalog.read"
    DATA_SELECTION_MANAGE = "data.selection.manage"
    STRATEGIES_READ = "strategies.read"
    STRATEGIES_MANAGE = "strategies.manage"
    STRATEGIES_RUN = "strategies.run"
    STRATEGIES_SAFE_STOP = "strategies.safe_stop"
    STRATEGIES_MANUAL_TRADE = "strategies.manual_trade"
    MODELS_READ = "models.read"
    MODELS_MANAGE = "models.manage"
    MODELS_PROMOTE_OR_ROLLBACK = "models.promote_or_rollback"
    BACKTESTS_READ = "backtests.read"
    BACKTESTS_MANAGE_OWN = "backtests.manage_own"
    BACKTESTS_QUEUE_OPERATE = "backtests.queue_operate"
    BACKTESTS_PROMOTE_OWN = "backtests.promote_own"
    LIVE_READ = "live.read"
    LIVE_RECONCILE = "live.reconcile"
    MONITORING_READ = "monitoring.read"
    MONITORING_SAFE_ACTION = "monitoring.safe_action"
    CONNECTIONS_STATUS_READ = "connections.status.read"
    CONNECTIONS_BIND_OWN = "connections.bind_own"
    CONNECTIONS_MANAGE = "connections.manage"
    CONNECTIONS_SAFE_OPERATE = "connections.safe_operate"
    CONNECTIONS_SECRET_REVEAL = "connections.secret.reveal"
    SETTINGS_PERSONAL_MANAGE = "settings.personal.manage"
    SETTINGS_ORGANIZATION_MANAGE = "settings.organization.manage"
    INSTALLATION_TRUST_MANAGE = "installation.trust.manage"
    INSTALLATION_RESOURCES_MANAGE = "installation.resources.manage"
    ADMIN_OVERVIEW_READ = "admin.overview.read"
    ADMIN_MEMBERS_MANAGE = "admin.members.manage"
    ADMIN_PLUGINS_MANAGE = "admin.plugins.manage"
    ADMIN_OPERATIONS_EXECUTE = "admin.operations.execute"
    INSTALLATION_RECOVERY_EXECUTE = "installation.recovery.execute"
    AUDIT_ORGANIZATION_READ = "audit.organization.read"
    DOCS_READ = "docs.read"
    DOCS_OPERATOR_READ = "docs.operator.read"
    QA_PLUGIN_QUERY = "qa.plugin.query"


class AuthorizationScope(StrEnum):
    PERSONAL = "personal"
    ORGANIZATION = "organization"
    SERVER_FILTERED_READ = "server_filtered_read"
    OWN = "own"
    OPERATIONAL_SAFE_SUBSET = "operational_safe_subset"
    VISIBILITY_SCOPED = "visibility_scoped"
    INSTALLATION = "installation"


class AuthorizationDenyReason(StrEnum):
    CLIENT_ROLE_SUPPLIED = "client_role_supplied"
    UNKNOWN_CAPABILITY = "unknown_capability"
    ORGANIZATION_CONTEXT_REQUIRED = "organization_context_required"
    INACTIVE_OR_MISSING_MEMBERSHIP = "inactive_or_missing_membership"
    RESOURCE_ORGANIZATION_MISMATCH = "resource_organization_mismatch"
    RESOURCE_CONTEXT_REQUIRED = "resource_context_required"
    OWNERSHIP_REQUIRED = "ownership_required"
    INSTALLATION_OWNER_REQUIRED = "installation_owner_required"
    STORED_SECRET_REVEAL_FORBIDDEN = "stored_secret_reveal_forbidden"
    ROLE_CAPABILITY_DENIED = "role_capability_denied"


@dataclass(frozen=True, slots=True)
class AuthorizationResource:
    """Server-resolved resource identity used to prevent cross-organization access."""

    organization_id: OrganizationId
    owner_user_id: UserId | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    """All inputs accepted at the application authorization boundary."""

    actor: CurrentUserPrincipal
    capability: CapabilityId | str
    selected_organization_id: OrganizationId | None = None
    resource: AuthorizationResource | None = None
    client_supplied_role: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Deterministic authorization result, suitable for a future transport adapter."""

    allowed: bool
    capability: CapabilityId | None
    scope: AuthorizationScope | None = None
    deny_reason: AuthorizationDenyReason | None = None

    @classmethod
    def allow(cls, *, capability: CapabilityId, scope: AuthorizationScope) -> AuthorizationDecision:
        return cls(allowed=True, capability=capability, scope=scope)

    @classmethod
    def deny(
        cls,
        *,
        capability: CapabilityId | None,
        reason: AuthorizationDenyReason,
    ) -> AuthorizationDecision:
        return cls(allowed=False, capability=capability, deny_reason=reason)
