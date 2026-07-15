from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId

PLUGIN_MANIFEST_API_VERSION = "roehub.io/v1alpha1"
PLUGIN_API_VERSION = "v1alpha1"
PLUGIN_RPC_VERSION = "roehub.plugin.rpc/v1alpha1"

PluginType = Literal["data-source", "panel", "app", "notification-provider"]
PluginOperationKind = Literal[
    "install", "update", "rollback", "configure", "enable", "disable", "health"
]
PluginOperationStatus = Literal["pending", "running", "succeeded", "failed", "unknown"]


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    version: str
    publisher: str
    plugin_type: PluginType
    plugin_api_version: str
    rpc_version: str
    image_reference: str
    image_digest: str
    architectures: tuple[str, ...]
    permissions: tuple[str, ...]
    config_schema: Mapping[str, Any]
    license_spdx: str
    package_digest: str
    publisher_key_id: str | None
    signed: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedPluginBundle:
    bundle_path: str
    manifest: PluginManifest
    artifact_digests: Mapping[str, str]
    publisher_public_key_b64: str | None
    publisher_key_fingerprint_sha256: str | None


@dataclass(frozen=True, slots=True)
class PluginPackage:
    package_id: UUID
    installation_id: InstallationId
    plugin_id: str
    version: str
    package_digest: str
    image_reference: str
    image_digest: str
    publisher_key_id: str | None
    publisher_public_key_b64: str | None
    publisher_key_fingerprint_sha256: str | None
    manifest: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PluginInstallation:
    plugin_installation_id: UUID
    installation_id: InstallationId
    organization_id: OrganizationId
    plugin_id: str
    package_id: UUID
    previous_package_id: UUID | None
    granted_permissions: tuple[str, ...]
    status: Literal["enabled", "disabled", "degraded"]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PluginInstance:
    instance_id: UUID
    plugin_installation_id: UUID
    installation_id: InstallationId
    organization_id: OrganizationId
    name: str
    config: Mapping[str, Any]
    config_revision: int
    status: Literal["enabled", "disabled", "degraded"]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PluginOperation:
    operation_id: UUID
    installation_id: InstallationId
    organization_id: OrganizationId
    actor_user_id: UserId
    kind: PluginOperationKind
    target_id: str
    idempotency_key: str
    request_hash: str
    request: Mapping[str, Any]
    status: PluginOperationStatus
    result: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PluginEvent:
    event_id: UUID
    installation_id: InstallationId
    organization_id: OrganizationId
    actor_user_id: UserId
    event_type: str
    target_type: str
    target_id: str
    outcome: Literal["succeeded", "rejected"]
    metadata: Mapping[str, str]
    created_at: datetime
