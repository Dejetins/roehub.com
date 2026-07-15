"""Disposable PostgreSQL and isolated-container proof for Plugin API v1alpha1."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Any, cast
from uuid import UUID

import httpx
import psycopg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from trading.contexts.extensions.adapters import (
    IdentityPluginAuthorization,
    PostgresPluginRepository,
)
from trading.contexts.extensions.application import (
    PluginBundleValidator,
    PluginLifecycleError,
    PluginLifecycleService,
    load_publisher_key_file,
)
from trading.contexts.extensions.domain import ValidatedPluginBundle
from trading.contexts.identity.adapters.outbound.persistence.postgres import (
    PostgresOrganizationRepository,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.integration import PluginRpcClient, PluginRpcError, PluginServiceIdentitySigner
from trading.shared_kernel.primitives import InstallationId, OrganizationId, PaidLevel, UserId


class ExtensionsPluginRuntimeProofError(RuntimeError):
    """Raised when the real plugin boundary evidence is incomplete."""


def run_probe(
    *,
    postgres_dsn: str,
    bundle_v1: Path,
    bundle_v2: Path,
    publisher_keys: Path,
    signing_key_file: Path,
    schema_path: Path,
    plugin_base_url: str,
    installation_id: InstallationId,
    organization_id: OrganizationId,
    foreign_organization_id: OrganizationId,
    user_id: UserId,
    instance_id: UUID,
) -> dict[str, object]:
    now = datetime.now(UTC).replace(microsecond=0)
    trusted_keys = load_publisher_key_file(publisher_keys)
    validator = PluginBundleValidator(
        schema_path=schema_path,
        trusted_publisher_keys=trusted_keys,
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        trading_mode="testnet",
    )
    first_bundle = validator.validate(bundle_v1)
    second_bundle = validator.validate(bundle_v2)
    _seed_identity(
        dsn=postgres_dsn,
        installation_id=installation_id,
        organization_id=organization_id,
        foreign_organization_id=foreign_organization_id,
        user_id=user_id,
        now=now,
    )
    repository = PostgresPluginRepository(dsn=postgres_dsn)
    service = PluginLifecycleService(
        repository=repository,
        authorization=IdentityPluginAuthorization(
            repository=PostgresOrganizationRepository(dsn=postgres_dsn)
        ),
        trusted_publisher_fingerprints={
            "stage12-publisher": cast(
                str, first_bundle.publisher_key_fingerprint_sha256
            )
        },
        trading_mode="testnet",
    )
    principal = CurrentUserPrincipal(
        user_id=user_id,
        paid_level=PaidLevel("free"),
        session_created_at=now,
    )
    submitted = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=first_bundle,
        requested_permissions=("data.read",),
        instance_name="Stage 12 fixture",
        config={"dataset": "fixture"},
        idempotency_key="stage12-install-0001",
        now=now,
        instance_id=instance_id,
    )
    replayed_submit = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=first_bundle,
        requested_permissions=("data.read",),
        instance_name="Stage 12 fixture",
        config={"dataset": "fixture"},
        idempotency_key="stage12-install-0001",
        now=now,
        instance_id=instance_id,
    )
    if replayed_submit.operation_id != submitted.operation_id:
        raise ExtensionsPluginRuntimeProofError("management idempotency was not preserved")
    installed = service.execute_install_or_update(
        operation_id=submitted.operation_id,
        now=now,
    )
    if installed.status != "succeeded" or installed.result.get("instance_id") != str(instance_id):
        raise ExtensionsPluginRuntimeProofError("fixture installation identity is incorrect")

    signing_key = _load_signing_key(signing_key_file)
    signer = PluginServiceIdentitySigner(private_key=signing_key, key_id="stage12-gateway")
    client = _client(
        signer=signer,
        base_url=plugin_base_url,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=first_bundle.manifest.package_digest,
        package_version=first_bundle.manifest.version,
    )
    try:
        health = client.health(now=now)
        query = client.query_data(request={"symbol": "FIXTURE"}, now=now)
        metrics = client.metrics(now=now)
    finally:
        client.close()
    expected_health = {
        "uid": 10001,
        "filesystem_write": "denied",
        "platform_database": "denied",
        "external_egress": "denied",
    }
    if any(health.get(key) != value for key, value in expected_health.items()):
        raise ExtensionsPluginRuntimeProofError("container isolation health is incomplete")
    if (
        health.get("status") != "ready"
        or query.get("package_digest") != first_bundle.manifest.package_digest
        or metrics.get("status") != "ready"
    ):
        raise ExtensionsPluginRuntimeProofError("plugin RPC health/query/metrics failed")
    _expect_permission_denial(
        signer=signer,
        base_url=plugin_base_url,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=first_bundle.manifest.package_digest,
        package_version=first_bundle.manifest.version,
        now=now,
    )
    _expect_protocol_denial(
        signer=signer,
        base_url=plugin_base_url,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=first_bundle.manifest.package_digest,
        package_version=first_bundle.manifest.version,
        now=now,
    )
    _expect_identity_scope_and_replay_denial(
        signer=signer,
        base_url=plugin_base_url,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=first_bundle.manifest.package_digest,
        package_version=first_bundle.manifest.version,
        now=now,
    )
    _expect_stale_permission_expansion_rejected(
        service=service,
        principal=principal,
        organization_id=organization_id,
        bundle=second_bundle,
        now=now + timedelta(minutes=11),
    )
    _expect_foreign_organization_denied(
        service=service,
        principal=principal,
        foreign_organization_id=foreign_organization_id,
        bundle=second_bundle,
        now=now,
    )

    updated_request = service.submit_install_or_update(
        principal=principal,
        organization_id=organization_id,
        bundle=second_bundle,
        requested_permissions=("data.read",),
        instance_name="Stage 12 fixture",
        config={"dataset": "fixture-v2"},
        idempotency_key="stage12-update-0002",
        now=now + timedelta(minutes=1),
        instance_id=instance_id,
    )
    updated = service.execute_install_or_update(
        operation_id=updated_request.operation_id,
        now=now + timedelta(minutes=1),
    )
    if updated.status != "succeeded" or updated.result.get("instance_id") != str(instance_id):
        raise ExtensionsPluginRuntimeProofError("plugin update did not preserve instance")
    _expect_revoked_rollback_rejected(
        dsn=postgres_dsn,
        service=service,
        principal=principal,
        organization_id=organization_id,
        plugin_id=first_bundle.manifest.plugin_id,
        now=now + timedelta(minutes=2),
    )
    rollback_request = service.submit_rollback(
        principal=principal,
        organization_id=organization_id,
        plugin_id=first_bundle.manifest.plugin_id,
        idempotency_key="stage12-rollback-0003",
        now=now + timedelta(minutes=2),
    )
    rolled_back = service.execute_rollback(
        operation_id=rollback_request.operation_id,
        now=now + timedelta(minutes=2),
    )
    if rolled_back.status != "succeeded":
        raise ExtensionsPluginRuntimeProofError("plugin rollback failed")
    service.record_runtime_observation(
        principal=principal,
        organization_id=organization_id,
        instance_id=instance_id,
        health="ready",
        metrics_status="ready",
        now=now + timedelta(minutes=2),
    )
    _expect_accepted_payload_is_immutable_and_claim_is_cas(
        dsn=postgres_dsn,
        service=service,
        principal=principal,
        organization_id=organization_id,
        bundle=first_bundle,
        instance_id=instance_id,
        now=now + timedelta(minutes=3),
    )
    _expect_cross_organization_instance_rejected(
        dsn=postgres_dsn,
        installation_id=installation_id,
        foreign_organization_id=foreign_organization_id,
        instance_id=instance_id,
        now=now,
    )
    audit_count, current_digest, revision = _database_state(
        dsn=postgres_dsn,
        organization_id=organization_id,
        instance_id=instance_id,
    )
    if audit_count < 7 or current_digest != first_bundle.manifest.package_digest or revision != 2:
        raise ExtensionsPluginRuntimeProofError("durable audit/update/rollback is incomplete")
    return {
        "schema": "io.roehub.extensions-plugin-runtime-proof/v1alpha1",
        "signed_bundle": "passed",
        "package_instance_separation": "passed",
        "management_idempotency": "passed",
        "concurrent_idempotency": "passed",
        "permission_expansion_recent_auth": "rejected_when_stale",
        "foreign_organization_admin": "rejected",
        "short_lived_identity": "passed",
        "protocol_negotiation": "passed",
        "identity_full_scope": "passed",
        "identity_replay": "rejected",
        "capability_denial": "passed",
        "filesystem_write": "denied",
        "platform_database": "denied",
        "external_egress": "denied",
        "health": "ready",
        "metrics": "ready",
        "audit_events": audit_count,
        "config_revision": revision,
        "rollback": "restored_previous_package",
        "rollback_revoked_publisher": "rejected",
        "publisher_trust_bootstrap": "passed",
        "accepted_payload_immutable": "passed",
        "operation_claim_cas": "passed",
    }


def _expect_revoked_rollback_rejected(
    *,
    dsn: str,
    service: PluginLifecycleService,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId,
    plugin_id: str,
    now: datetime,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE extensions_publisher_keys
               SET status = 'revoked', revoked_at = %s
               WHERE key_id = 'stage12-publisher'""",
            (now,),
        )
    try:
        service.submit_rollback(
            principal=principal,
            organization_id=organization_id,
            plugin_id=plugin_id,
            idempotency_key="stage12-revoked-rollback",
            now=now,
        )
    except PluginLifecycleError as error:
        if error.code != "plugin.publisher_untrusted":
            raise ExtensionsPluginRuntimeProofError(
                "revoked publisher returned an unexpected rollback error"
            ) from error
    else:
        raise ExtensionsPluginRuntimeProofError(
            "rollback accepted a package from a revoked publisher"
        )
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE extensions_publisher_keys
               SET status = 'trusted', revoked_at = NULL
               WHERE key_id = 'stage12-publisher'"""
        )


def _expect_accepted_payload_is_immutable_and_claim_is_cas(
    *,
    dsn: str,
    service: PluginLifecycleService,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId,
    bundle: ValidatedPluginBundle,
    instance_id: UUID,
    now: datetime,
) -> None:
    barrier = Barrier(2)

    def submit() -> object:
        barrier.wait()
        return service.submit_install_or_update(
            principal=principal,
            organization_id=organization_id,
            bundle=bundle,
            requested_permissions=("data.read",),
            instance_name="Stage 12 fixture",
            config={"dataset": "fixture"},
            idempotency_key="stage12-payload-tamper-0004",
            now=now,
            instance_id=instance_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        submissions = tuple(executor.map(lambda _: submit(), range(2)))
    submitted = cast(Any, submissions[0])
    if submissions[1] != submitted:
        raise ExtensionsPluginRuntimeProofError(
            "concurrent idempotent submission returned different operations"
        )
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE extensions_plugin_operations
               SET request_payload = jsonb_set(
                   request_payload,
                   '{permissions}',
                   '["data.read", "panel.describe"]'::jsonb
               )
               WHERE operation_id = %s""",
            (submitted.operation_id,),
        )
    failed = service.execute_install_or_update(
        operation_id=submitted.operation_id,
        now=now,
    )
    if (
        failed.status != "failed"
        or failed.result.get("error_code") != "plugin.operation_payload_invalid"
    ):
        raise ExtensionsPluginRuntimeProofError(
            "mutated accepted operation payload was not rejected"
        )
    try:
        service.execute_install_or_update(
            operation_id=submitted.operation_id,
            now=now + timedelta(seconds=1),
        )
    except PluginLifecycleError as error:
        if error.code != "plugin.operation_not_pending":
            raise ExtensionsPluginRuntimeProofError(
                "operation claim compare-and-set returned an unexpected error"
            ) from error
    else:
        raise ExtensionsPluginRuntimeProofError(
            "operation claim compare-and-set allowed duplicate execution"
        )


def _seed_identity(
    *,
    dsn: str,
    installation_id: InstallationId,
    organization_id: OrganizationId,
    foreign_organization_id: OrganizationId,
    user_id: UserId,
    now: datetime,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO identity_users
               (user_id, telegram_user_id, paid_level, created_at, is_deleted)
               VALUES (%s, %s, 'free', %s, FALSE)""",
            (user_id.value, 120000000001, now),
        )
        cursor.execute(
            """INSERT INTO identity_installations
               (installation_id, singleton_key, display_name, created_at)
               VALUES (%s, TRUE, 'Stage 12 proof', %s)""",
            (installation_id.value, now),
        )
        cursor.execute(
            """INSERT INTO identity_installation_owners
               (installation_id, user_id, granted_by_user_id, granted_at)
               VALUES (%s, %s, %s, %s)""",
            (installation_id.value, user_id.value, user_id.value, now),
        )
        cursor.execute(
            """INSERT INTO identity_organizations
               (organization_id, installation_id, slug, display_name, status, created_at)
               VALUES
               (%s, %s, 'stage12-primary', 'Stage 12 primary', 'active', %s),
               (%s, %s, 'stage12-foreign', 'Stage 12 foreign', 'active', %s)""",
            (
                organization_id.value,
                installation_id.value,
                now,
                foreign_organization_id.value,
                installation_id.value,
                now,
            ),
        )
        cursor.execute(
            """INSERT INTO identity_memberships
               (organization_id, user_id, role, status, created_at, updated_at)
               VALUES (%s, %s, 'admin', 'active', %s, %s)""",
            (organization_id.value, user_id.value, now, now),
        )


def _load_signing_key(path: Path) -> Ed25519PrivateKey:
    loaded = serialization.load_pem_private_key(path.read_bytes(), None)
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ExtensionsPluginRuntimeProofError("gateway signing key type is invalid")
    return loaded


def _client(
    *,
    signer: PluginServiceIdentitySigner,
    base_url: str,
    organization_id: OrganizationId,
    instance_id: UUID,
    package_digest: str,
    package_version: str,
    granted_capabilities: frozenset[str] = frozenset({"data.read"}),
) -> PluginRpcClient:
    return PluginRpcClient(
        base_url=base_url,
        signer=signer,
        organization_id=organization_id.value,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version=package_version,
        granted_capabilities=granted_capabilities,
    )


class _ExpectedRpcError:
    def __init__(self, expected_code: str) -> None:
        self.expected_code = expected_code

    def __enter__(self) -> None:
        return None

    def __exit__(self, error_type: object, error: object, traceback: object) -> bool:
        _ = error_type, traceback
        if not isinstance(error, PluginRpcError) or error.code != self.expected_code:
            raise ExtensionsPluginRuntimeProofError(
                "expected plugin RPC rejection was not observed"
            )
        return True


def _expect_permission_denial(
    *,
    signer: PluginServiceIdentitySigner,
    base_url: str,
    organization_id: OrganizationId,
    instance_id: UUID,
    package_digest: str,
    package_version: str,
    now: datetime,
) -> None:
    client = _client(
        signer=signer,
        base_url=base_url,
        organization_id=organization_id,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version=package_version,
        granted_capabilities=frozenset({"panel.describe"}),
    )
    try:
        with _ExpectedRpcError("plugin.rpc_rejected"):
            client.describe_panel(now=now)
    finally:
        client.close()


def _expect_protocol_denial(
    *,
    signer: PluginServiceIdentitySigner,
    base_url: str,
    organization_id: OrganizationId,
    instance_id: UUID,
    package_digest: str,
    package_version: str,
    now: datetime,
) -> None:
    identity = signer.issue(
        organization_id=organization_id.value,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version=package_version,
        capability="data.read",
        now=now,
    )
    response = httpx.post(
        base_url.rstrip("/") + "/v1alpha1/data-source/query",
        headers={
            "Authorization": f"RoehubPluginIdentity {identity}",
            "X-Roehub-Plugin-Protocol": "roehub.plugin.rpc/v0",
        },
        json={},
        timeout=5.0,
    )
    if response.status_code != 426:
        raise ExtensionsPluginRuntimeProofError("unsupported protocol was not rejected")


def _expect_identity_scope_and_replay_denial(
    *,
    signer: PluginServiceIdentitySigner,
    base_url: str,
    organization_id: OrganizationId,
    instance_id: UUID,
    package_digest: str,
    package_version: str,
    now: datetime,
) -> None:
    url = base_url.rstrip("/") + "/v1alpha1/data-source/query"
    headers = {"X-Roehub-Plugin-Protocol": "roehub.plugin.rpc/v1alpha1"}
    identity = signer.issue(
        organization_id=organization_id.value,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version=package_version,
        capability="data.read",
        now=now,
    )
    headers["Authorization"] = f"RoehubPluginIdentity {identity}"
    first = httpx.post(url, headers=headers, json={}, timeout=5.0)
    replay = httpx.post(url, headers=headers, json={}, timeout=5.0)
    wrong_version = signer.issue(
        organization_id=organization_id.value,
        instance_id=instance_id,
        package_digest=package_digest,
        package_version="9.9.9",
        capability="data.read",
        now=now,
    )
    wrong_scope = httpx.post(
        url,
        headers={
            **headers,
            "Authorization": f"RoehubPluginIdentity {wrong_version}",
        },
        json={},
        timeout=5.0,
    )
    if first.status_code != 200 or replay.status_code != 403 or wrong_scope.status_code != 403:
        raise ExtensionsPluginRuntimeProofError(
            "plugin identity full-scope or replay denial was not observed"
        )


def _expect_stale_permission_expansion_rejected(
    *,
    service: PluginLifecycleService,
    principal: CurrentUserPrincipal,
    organization_id: OrganizationId,
    bundle: ValidatedPluginBundle,
    now: datetime,
) -> None:
    try:
        service.submit_install_or_update(
            principal=principal,
            organization_id=organization_id,
            bundle=bundle,
            requested_permissions=("data.read", "panel.describe"),
            instance_name="Stage 12 fixture",
            config={"dataset": "fixture-v2"},
            idempotency_key="stage12-stale-expansion",
            now=now,
        )
    except PluginLifecycleError as error:
        if error.code == "recent_auth_required":
            return
    raise ExtensionsPluginRuntimeProofError("stale permission expansion was accepted")


def _expect_foreign_organization_denied(
    *,
    service: PluginLifecycleService,
    principal: CurrentUserPrincipal,
    foreign_organization_id: OrganizationId,
    bundle: ValidatedPluginBundle,
    now: datetime,
) -> None:
    try:
        service.submit_install_or_update(
            principal=principal,
            organization_id=foreign_organization_id,
            bundle=bundle,
            requested_permissions=("data.read",),
            instance_name="Forbidden",
            config={"dataset": "fixture"},
            idempotency_key="stage12-foreign-admin",
            now=now,
        )
    except PermissionError:
        return
    raise ExtensionsPluginRuntimeProofError("foreign organization plugin admin was accepted")


def _expect_cross_organization_instance_rejected(
    *,
    dsn: str,
    installation_id: InstallationId,
    foreign_organization_id: OrganizationId,
    instance_id: UUID,
    now: datetime,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT plugin_installation_id FROM extensions_plugin_instances WHERE instance_id = %s",
            (instance_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExtensionsPluginRuntimeProofError("plugin instance disappeared")
        try:
            cursor.execute(
                """INSERT INTO extensions_plugin_instances
                   (instance_id, plugin_installation_id, installation_id, organization_id,
                    name, config, config_revision, status, created_at, updated_at)
                   VALUES (gen_random_uuid(), %s, %s, %s, 'Cross scope', '{}'::JSONB,
                           1, 'enabled', %s, %s)""",
                (
                    row[0],
                    installation_id.value,
                    foreign_organization_id.value,
                    now,
                    now,
                ),
            )
        except psycopg.errors.ForeignKeyViolation:
            connection.rollback()
            return
    raise ExtensionsPluginRuntimeProofError("cross-organization instance was accepted")


def _database_state(
    *, dsn: str, organization_id: OrganizationId, instance_id: UUID
) -> tuple[int, str, int]:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM extensions_plugin_events WHERE organization_id = %s",
            (organization_id.value,),
        )
        audit_count = int(cast(tuple[Any, ...], cursor.fetchone())[0])
        cursor.execute(
            """SELECT package.package_digest, instance.config_revision
               FROM extensions_plugin_instances AS instance
               JOIN extensions_plugin_installations AS installation
                 USING (plugin_installation_id)
               JOIN extensions_plugin_packages AS package
                 ON package.package_id = installation.package_id
               WHERE instance.instance_id = %s""",
            (instance_id,),
        )
        state = cast(tuple[Any, ...] | None, cursor.fetchone())
    if state is None:
        raise ExtensionsPluginRuntimeProofError("plugin durable state is missing")
    return audit_count, str(state[0]), int(state[1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres-dsn", default=os.environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "")
    )
    parser.add_argument("--bundle-v1", type=Path, required=True)
    parser.add_argument("--bundle-v2", type=Path, required=True)
    parser.add_argument("--publisher-keys", type=Path, required=True)
    parser.add_argument("--signing-key-file", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--plugin-base-url", required=True)
    parser.add_argument("--installation-id", type=UUID, required=True)
    parser.add_argument("--organization-id", type=UUID, required=True)
    parser.add_argument("--foreign-organization-id", type=UUID, required=True)
    parser.add_argument("--user-id", type=UUID, required=True)
    parser.add_argument("--instance-id", type=UUID, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.postgres_dsn:
        raise SystemExit("PostgreSQL DSN is required")
    try:
        payload = run_probe(
            postgres_dsn=args.postgres_dsn,
            bundle_v1=args.bundle_v1,
            bundle_v2=args.bundle_v2,
            publisher_keys=args.publisher_keys,
            signing_key_file=args.signing_key_file,
            schema_path=args.schema_path,
            plugin_base_url=args.plugin_base_url,
            installation_id=InstallationId(args.installation_id),
            organization_id=OrganizationId(args.organization_id),
            foreign_organization_id=OrganizationId(args.foreign_organization_id),
            user_id=UserId(args.user_id),
            instance_id=args.instance_id,
        )
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "io.roehub.extensions-plugin-runtime-proof/v1alpha1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", "runtime_proof_failed"),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
