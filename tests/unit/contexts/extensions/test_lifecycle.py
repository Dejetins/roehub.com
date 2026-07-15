from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from trading.contexts.extensions.adapters import InMemoryPluginRepository
from trading.contexts.extensions.application import PluginLifecycleError, PluginLifecycleService
from trading.contexts.extensions.domain import (
    PluginManifest,
    PluginRuntimePolicy,
    ValidatedPluginBundle,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.shared_kernel.primitives import InstallationId, OrganizationId, PaidLevel, UserId


class _Authorization:
    def __init__(self, *, installation_id: InstallationId) -> None:
        self.installation_id = installation_id

    def require_manage(
        self, *, principal: CurrentUserPrincipal, organization_id: OrganizationId
    ) -> InstallationId:
        _ = principal, organization_id
        return self.installation_id

    def require_read(
        self, *, principal: CurrentUserPrincipal, organization_id: OrganizationId
    ) -> InstallationId:
        _ = principal, organization_id
        return self.installation_id


def _bundle(*, version: str, digest_character: str) -> ValidatedPluginBundle:
    raw: dict[str, Any] = {"apiVersion": "roehub.io/v1alpha1", "kind": "Plugin"}
    return ValidatedPluginBundle(
        bundle_path="/fixture",
        artifact_digests={},
        publisher_public_key_b64="fixture-public-key",
        publisher_key_fingerprint_sha256="f" * 64,
        manifest=PluginManifest(
            plugin_id="fixture.data",
            version=version,
            publisher="fixture.publisher",
            plugin_type="data-source",
            plugin_api_version="v1alpha1",
            rpc_version="roehub.plugin.rpc/v1alpha1",
            image_reference=f"fixture/plugin:{version}",
            image_digest="sha256:" + digest_character * 64,
            architectures=("linux/amd64",),
            permissions=("data.read", "panel.describe"),
            config_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"dataset": {"type": "string"}},
                "required": ["dataset"],
            },
            license_spdx="Apache-2.0",
            package_digest=digest_character * 64,
            publisher_key_id="fixture.publisher-key",
            signed=True,
            raw=raw,
        ),
    )


def _principal(*, now: datetime) -> CurrentUserPrincipal:
    return CurrentUserPrincipal(
        user_id=UserId(uuid4()),
        paid_level=PaidLevel("free"),
        session_created_at=now - timedelta(minutes=1),
    )


def test_lifecycle_is_idempotent_audited_and_rollback_restores_previous_package() -> None:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    organization_id = OrganizationId(uuid4())
    repository = InMemoryPluginRepository()
    service = PluginLifecycleService(
        repository=repository,
        authorization=_Authorization(installation_id=InstallationId(uuid4())),
        trusted_publisher_fingerprints={"fixture.publisher-key": "f" * 64},
    )
    principal = _principal(now=now)
    first_bundle = _bundle(version="0.1.0", digest_character="1")
    submitted = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=first_bundle,
        requested_permissions=("data.read",),
        instance_name="Primary",
        config={"dataset": "fixture"},
        idempotency_key="plugin-install-0001",
        now=now,
    )
    assert service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=first_bundle,
        requested_permissions=("data.read",),
        instance_name="Primary",
        config={"dataset": "fixture"},
        idempotency_key="plugin-install-0001",
        now=now,
    ) == submitted
    installed = service.execute_install_or_update(
        operation_id=submitted.operation_id,
        now=now,
    )
    assert installed.status == "succeeded"
    with pytest.raises(PluginLifecycleError) as duplicate_execution:
        service.execute_install_or_update(
            operation_id=submitted.operation_id,
            now=now + timedelta(seconds=1),
        )
    assert duplicate_execution.value.code == "plugin.operation_not_pending"

    second_bundle = _bundle(version="0.2.0", digest_character="2")
    update = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=second_bundle,
        requested_permissions=("data.read",),
        instance_name="Primary",
        config={"dataset": "fixture-v2"},
        idempotency_key="plugin-update-0002",
        now=now + timedelta(minutes=1),
    )
    updated = service.execute_install_or_update(
        operation_id=update.operation_id,
        now=now + timedelta(minutes=1),
    )
    rollback = service.submit_rollback(
        principal=principal,
        organization_id=organization_id,
        plugin_id="fixture.data",
        idempotency_key="plugin-rollback-0003",
        now=now + timedelta(minutes=2),
    )
    rolled_back = service.execute_rollback(
        operation_id=rollback.operation_id,
        now=now + timedelta(minutes=2),
    )

    assert updated.status == "succeeded"
    assert rolled_back.status == "succeeded"
    assert len(repository.list_events(organization_id=organization_id, limit=20)) >= 5


def test_permission_expansion_requires_recent_auth_and_records_rejection() -> None:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    organization_id = OrganizationId(uuid4())
    repository = InMemoryPluginRepository()
    service = PluginLifecycleService(
        repository=repository,
        authorization=_Authorization(installation_id=InstallationId(uuid4())),
        trusted_publisher_fingerprints={"fixture.publisher-key": "f" * 64},
    )
    stale = replace(_principal(now=now), session_created_at=now - timedelta(minutes=11))

    with pytest.raises(PluginLifecycleError) as error:
        service.submit_install_or_update(
            principal=stale,
            organization_id=organization_id,
            bundle=_bundle(version="0.1.0", digest_character="1"),
            requested_permissions=("data.read",),
            instance_name="Primary",
            config={"dataset": "fixture"},
            idempotency_key="plugin-install-0001",
            now=now,
        )

    assert error.value.code == "recent_auth_required"
    events = repository.list_events(organization_id=organization_id, limit=10)
    assert events[0].outcome == "rejected"
    assert events[0].metadata == {"reason_code": "recent_auth_required"}


def test_plugin_update_and_rollback_always_require_recent_auth() -> None:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    organization_id = OrganizationId(uuid4())
    repository = InMemoryPluginRepository()
    service = PluginLifecycleService(
        repository=repository,
        authorization=_Authorization(installation_id=InstallationId(uuid4())),
        trusted_publisher_fingerprints={"fixture.publisher-key": "f" * 64},
    )
    principal = _principal(now=now)
    first = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=_bundle(version="0.1.0", digest_character="1"),
        requested_permissions=("data.read",),
        instance_name="Primary",
        config={"dataset": "fixture"},
        idempotency_key="plugin-install-recent-0001",
        now=now,
    )
    service.execute_install_or_update(operation_id=first.operation_id, now=now)
    stale = replace(principal, session_created_at=now - timedelta(minutes=11))

    with pytest.raises(PluginLifecycleError) as update_error:
        service.submit_install_or_update(
            principal=stale,
            organization_id=organization_id,
            bundle=_bundle(version="0.2.0", digest_character="2"),
            requested_permissions=("data.read",),
            instance_name="Primary",
            config={"dataset": "fixture"},
            idempotency_key="plugin-update-stale-0001",
            now=now,
        )
    assert update_error.value.code == "recent_auth_required"

    with pytest.raises(PluginLifecycleError) as rollback_error:
        service.submit_rollback(
            principal=stale,
            organization_id=organization_id,
            plugin_id="fixture.data",
            idempotency_key="plugin-rollback-stale-0001",
            now=now,
        )
    assert rollback_error.value.code == "recent_auth_required"


def test_runtime_policy_has_no_mount_or_environment_escape() -> None:
    bundle = _bundle(version="0.1.0", digest_character="1")
    raw = dict(bundle.manifest.raw)
    raw["spec"] = {
        "runtime": {
            "nonRootUid": 10001,
            "readOnlyRootFilesystem": True,
            "noNewPrivileges": True,
            "resources": {"cpus": 0.5, "memoryMb": 128, "pids": 64},
            "egress": [],
        }
    }
    manifest = replace(bundle.manifest, raw=raw)

    spec = PluginRuntimePolicy.from_manifest(manifest).to_oci_spec(
        manifest=manifest,
        internal_network="stage12-plugin-internal",
    )

    assert spec.user == "10001:10001"
    assert spec.read_only_root_filesystem is True
    assert spec.cap_drop == ("ALL",)
    assert spec.mounts == ()
    assert spec.environment == ()
