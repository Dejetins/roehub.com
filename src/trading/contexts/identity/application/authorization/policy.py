from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping, cast

from trading.contexts.identity.application.authorization.models import (
    AuthorizationScope,
    CapabilityId,
)
from trading.contexts.identity.domain.entities import OrganizationRole


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    requires_organization_context: bool
    role_scopes: Mapping[OrganizationRole, AuthorizationScope]
    installation_owner_required: bool = False


def _policy(
    *,
    requires_organization_context: bool,
    installation_owner_required: bool = False,
    **role_scopes: AuthorizationScope,
) -> CapabilityPolicy:
    return CapabilityPolicy(
        requires_organization_context=requires_organization_context,
        installation_owner_required=installation_owner_required,
        role_scopes=MappingProxyType(cast(dict[OrganizationRole, AuthorizationScope], role_scopes)),
    )


_ALL_ROLES: Final[dict[OrganizationRole, AuthorizationScope]] = {
    "owner": AuthorizationScope.SERVER_FILTERED_READ,
    "admin": AuthorizationScope.SERVER_FILTERED_READ,
    "operator": AuthorizationScope.SERVER_FILTERED_READ,
    "trader": AuthorizationScope.SERVER_FILTERED_READ,
    "viewer": AuthorizationScope.SERVER_FILTERED_READ,
}

_NON_OWNER_READ_ROLES: Final[dict[OrganizationRole, AuthorizationScope]] = {
    "admin": AuthorizationScope.SERVER_FILTERED_READ,
    "operator": AuthorizationScope.SERVER_FILTERED_READ,
    "trader": AuthorizationScope.SERVER_FILTERED_READ,
    "viewer": AuthorizationScope.SERVER_FILTERED_READ,
}


CAPABILITY_POLICIES: Final[Mapping[CapabilityId, CapabilityPolicy]] = MappingProxyType(
    {
        CapabilityId.AUTH_SESSION_MANAGE: _policy(
            requires_organization_context=False,
        ),
        CapabilityId.SETUP_BOOTSTRAP: _policy(
            requires_organization_context=False,
            installation_owner_required=True,
        ),
        CapabilityId.PREFERENCES_MANAGE_PERSONAL: _policy(
            requires_organization_context=False,
        ),
        CapabilityId.DASHBOARD_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            **_NON_OWNER_READ_ROLES,
        ),
        CapabilityId.DATA_CATALOG_READ: _policy(
            requires_organization_context=True,
            **_ALL_ROLES,
        ),
        CapabilityId.DATA_SELECTION_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
        CapabilityId.STRATEGIES_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.SERVER_FILTERED_READ,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
            trader=AuthorizationScope.OWN,
            viewer=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.STRATEGIES_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.STRATEGIES_RUN: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.STRATEGIES_SAFE_STOP: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.STRATEGIES_MANUAL_TRADE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.MODELS_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.SERVER_FILTERED_READ,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
            trader=AuthorizationScope.OWN,
            viewer=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.MODELS_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.MODELS_PROMOTE_OR_ROLLBACK: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.BACKTESTS_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.SERVER_FILTERED_READ,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
            trader=AuthorizationScope.OWN,
            viewer=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.BACKTESTS_MANAGE_OWN: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.BACKTESTS_QUEUE_OPERATE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
        ),
        CapabilityId.BACKTESTS_PROMOTE_OWN: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.LIVE_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.SERVER_FILTERED_READ,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
            trader=AuthorizationScope.OWN,
            viewer=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.LIVE_RECONCILE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.MONITORING_READ: _policy(
            requires_organization_context=True,
            **_ALL_ROLES,
        ),
        CapabilityId.MONITORING_SAFE_ACTION: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
        ),
        CapabilityId.CONNECTIONS_STATUS_READ: _policy(
            requires_organization_context=True,
            **_ALL_ROLES,
        ),
        CapabilityId.CONNECTIONS_BIND_OWN: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            trader=AuthorizationScope.OWN,
        ),
        CapabilityId.CONNECTIONS_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
        CapabilityId.CONNECTIONS_SAFE_OPERATE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
        ),
        CapabilityId.CONNECTIONS_SECRET_REVEAL: _policy(
            requires_organization_context=False,
        ),
        CapabilityId.SETTINGS_PERSONAL_MANAGE: _policy(
            requires_organization_context=False,
        ),
        CapabilityId.SETTINGS_ORGANIZATION_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
        CapabilityId.INSTALLATION_TRUST_MANAGE: _policy(
            requires_organization_context=False,
            installation_owner_required=True,
        ),
        CapabilityId.INSTALLATION_RESOURCES_MANAGE: _policy(
            requires_organization_context=False,
            installation_owner_required=True,
        ),
        CapabilityId.ADMIN_OVERVIEW_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.ADMIN_MEMBERS_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
        CapabilityId.ADMIN_PLUGINS_MANAGE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
        CapabilityId.ADMIN_OPERATIONS_EXECUTE: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.OPERATIONAL_SAFE_SUBSET,
        ),
        CapabilityId.INSTALLATION_RECOVERY_EXECUTE: _policy(
            requires_organization_context=False,
            installation_owner_required=True,
        ),
        CapabilityId.AUDIT_ORGANIZATION_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
            operator=AuthorizationScope.SERVER_FILTERED_READ,
        ),
        CapabilityId.DOCS_READ: _policy(
            requires_organization_context=True,
            **_ALL_ROLES,
        ),
        CapabilityId.DOCS_OPERATOR_READ: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.VISIBILITY_SCOPED,
            admin=AuthorizationScope.VISIBILITY_SCOPED,
            operator=AuthorizationScope.VISIBILITY_SCOPED,
            trader=AuthorizationScope.VISIBILITY_SCOPED,
            viewer=AuthorizationScope.VISIBILITY_SCOPED,
        ),
        CapabilityId.QA_PLUGIN_QUERY: _policy(
            requires_organization_context=True,
            owner=AuthorizationScope.ORGANIZATION,
            admin=AuthorizationScope.ORGANIZATION,
        ),
    }
)
