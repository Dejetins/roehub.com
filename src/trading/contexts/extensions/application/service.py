from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Mapping, cast
from uuid import UUID, uuid4, uuid5

import jsonschema

from trading.contexts.extensions.application.ports import (
    PluginAuthorization,
    PluginRepository,
    PluginRepositoryInvariantError,
)
from trading.contexts.extensions.domain import (
    PluginEvent,
    PluginInstallation,
    PluginInstance,
    PluginManifest,
    PluginOperation,
    PluginPackage,
    ValidatedPluginBundle,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId

_RECENT_AUTH_WINDOW = timedelta(minutes=10)
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_SECRET_KEY_RE = re.compile(
    r"^(?:password|token|secret|credential|cookie|authorization|api[_-]?key)$",
    re.IGNORECASE,
)


class PluginLifecycleError(ValueError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PluginLifecycleService:
    """Typed asynchronous package lifecycle with permission-diff gates and audit."""

    def __init__(
        self,
        *,
        repository: PluginRepository,
        authorization: PluginAuthorization,
        trusted_publisher_fingerprints: Mapping[str, str] | None = None,
        allow_unsigned_development: bool = False,
        trading_mode: str = "paper",
    ) -> None:
        self._repository = repository
        self._authorization = authorization
        self._trusted_publisher_fingerprints = dict(
            trusted_publisher_fingerprints or {}
        )
        self._allow_unsigned_development = allow_unsigned_development
        self._trading_mode = trading_mode
        if trading_mode == "mainnet" and allow_unsigned_development:
            raise ValueError("unsigned development mode is unavailable to mainnet")

    def require_manage(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> InstallationId:
        return self._authorization.require_manage(
            principal=principal,
            organization_id=organization_id,
        )

    def list_inventory(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        limit: int = 100,
    ) -> tuple[
        tuple[PluginInstallation, ...],
        tuple[PluginInstance, ...],
        tuple[PluginOperation, ...],
        tuple[PluginEvent, ...],
    ]:
        self._authorization.require_read(
            principal=principal,
            organization_id=organization_id,
        )
        if not 1 <= limit <= 200:
            raise PluginLifecycleError(
                code="plugin.inventory_limit_invalid",
                message="Plugin inventory limit must be between 1 and 200",
            )
        return (
            self._repository.list_plugin_installations(
                organization_id=organization_id
            ),
            self._repository.list_instances_for_organization(
                organization_id=organization_id
            ),
            self._repository.list_operations(
                organization_id=organization_id,
                limit=limit,
            ),
            self._repository.list_events(
                organization_id=organization_id,
                limit=limit,
            ),
        )

    def submit_install_or_update(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        bundle: ValidatedPluginBundle,
        requested_permissions: tuple[str, ...],
        instance_name: str,
        config: Mapping[str, object],
        idempotency_key: str,
        now: datetime,
        instance_id: UUID | None = None,
    ) -> PluginOperation:
        normalized_now = _utc(now)
        installation_id = self._authorization.require_manage(
            principal=principal,
            organization_id=organization_id,
        )
        permissions = tuple(sorted(set(requested_permissions)))
        if set(permissions) - set(bundle.manifest.permissions):
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type="plugin.permission_expansion",
                target_id=bundle.manifest.plugin_id,
                code="plugin.permission_not_declared",
                message="Requested permission is not declared by the signed package",
                now=normalized_now,
            )
        try:
            jsonschema.Draft202012Validator(bundle.manifest.config_schema).validate(dict(config))
        except jsonschema.ValidationError as error:
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type="plugin.configuration_rejected",
                target_id=bundle.manifest.plugin_id,
                code="plugin.config_invalid",
                message="Plugin configuration does not match its signed schema",
                now=normalized_now,
            )
            raise AssertionError("unreachable") from error
        if _contains_secret_shaped_key(config):
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type="plugin.configuration_rejected",
                target_id=bundle.manifest.plugin_id,
                code="plugin.raw_secret_forbidden",
                message="Plugin configuration must use typed secret references",
                now=normalized_now,
            )
        normalized_name = instance_name.strip()
        if not 1 <= len(normalized_name) <= 120:
            raise PluginLifecycleError(
                code="plugin.instance_name_invalid", message="Plugin instance name is invalid"
            )
        _require_idempotency_key(idempotency_key)
        resolved_instance_id = instance_id or uuid5(
            organization_id.value,
            f"roehub-plugin-instance:{idempotency_key}",
        )
        accepted_request = _install_request_payload(
            bundle=bundle,
            permissions=permissions,
            instance_name=normalized_name,
            config=config,
            instance_id=resolved_instance_id,
        )
        request_hash = _request_hash(accepted_request)
        existing = self._repository.get_operation_by_idempotency(
            organization_id=organization_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise PluginLifecycleError(
                    code="plugin.idempotency_conflict",
                    message="Idempotency key is already bound to another request",
                )
            return existing
        current = self._repository.get_plugin_installation(
            organization_id=organization_id,
            plugin_id=bundle.manifest.plugin_id,
        )
        current_permissions = frozenset(current.granted_permissions if current else ())
        expansion = set(permissions) - current_permissions
        kind: Literal["install", "update"] = "update" if current is not None else "install"
        if not _is_recent(principal=principal, now=normalized_now):
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type=f"plugin.{kind}.rejected",
                target_id=bundle.manifest.plugin_id,
                code="recent_auth_required",
                message="Recent authentication is required for plugin lifecycle changes",
                now=normalized_now,
            )
        operation = PluginOperation(
            operation_id=uuid4(),
            installation_id=installation_id,
            organization_id=organization_id,
            actor_user_id=principal.user_id,
            kind=kind,
            target_id=bundle.manifest.plugin_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            request=accepted_request,
            status="pending",
            result={
                "contract": "PluginOperation/v1alpha1",
                "package_digest": bundle.manifest.package_digest,
            },
            created_at=normalized_now,
            updated_at=normalized_now,
        )
        try:
            created = self._repository.create_operation(operation=operation)
        except PluginRepositoryInvariantError as error:
            raise PluginLifecycleError(
                code=error.code, message="Plugin operation persistence rejected the request"
            ) from error
        if created.request_hash != request_hash:
            raise PluginLifecycleError(
                code="plugin.idempotency_conflict",
                message="Idempotency key is already bound to another request",
            )
        if created.operation_id != operation.operation_id:
            return created
        self._event(
            installation_id=installation_id,
            organization_id=organization_id,
            actor_user_id=principal.user_id,
            event_type=f"plugin.{kind}.requested",
            target_type="plugin",
            target_id=bundle.manifest.plugin_id,
            outcome="succeeded",
            metadata={
                "operation_id": str(created.operation_id),
                "permissions_added": ",".join(sorted(expansion)),
            },
            now=normalized_now,
        )
        return created

    def get_operation(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        operation_id: UUID,
    ) -> PluginOperation:
        self._authorization.require_manage(
            principal=principal,
            organization_id=organization_id,
        )
        operation = self._repository.get_operation(operation_id=operation_id)
        if operation is None or operation.organization_id != organization_id:
            raise PluginLifecycleError(
                code="plugin.operation_not_found", message="Plugin operation is not found"
            )
        return operation

    def record_runtime_observation(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        instance_id: UUID,
        health: str,
        metrics_status: str,
        now: datetime,
    ) -> None:
        installation_id = self.require_manage(
            principal=principal,
            organization_id=organization_id,
        )
        if health not in {"ready", "degraded"} or metrics_status not in {
            "ready",
            "unavailable",
        }:
            raise PluginLifecycleError(
                code="plugin.runtime_observation_invalid",
                message="Plugin runtime observation is invalid",
            )
        self._event(
            installation_id=installation_id,
            organization_id=organization_id,
            actor_user_id=principal.user_id,
            event_type="plugin.runtime.observed",
            target_type="plugin_instance",
            target_id=str(instance_id),
            outcome="succeeded",
            metadata={"health": health, "metrics": metrics_status},
            now=_utc(now),
        )

    def execute_install_or_update(
        self,
        *,
        operation_id: UUID,
        now: datetime,
    ) -> PluginOperation:
        normalized_now = _utc(now)
        operation = self._claim_pending_operation(
            operation_id=operation_id,
            now=normalized_now,
        )
        if operation.kind not in {"install", "update"}:
            return self._fail_operation(
                operation=operation,
                code="plugin.operation_kind_invalid",
                now=normalized_now,
            )
        try:
            bundle, requested_permissions, instance_name, config, instance_id = (
                _install_request_from_operation(operation)
            )
            self._require_bundle_activatable(bundle=bundle)
        except (PluginLifecycleError, TypeError, ValueError) as error:
            code = (
                error.code
                if isinstance(error, PluginLifecycleError)
                else "plugin.operation_payload_invalid"
            )
            return self._fail_operation(
                operation=operation,
                code=code,
                now=normalized_now,
            )
        package = PluginPackage(
            package_id=uuid4(),
            installation_id=operation.installation_id,
            plugin_id=bundle.manifest.plugin_id,
            version=bundle.manifest.version,
            package_digest=bundle.manifest.package_digest,
            image_reference=bundle.manifest.image_reference,
            image_digest=bundle.manifest.image_digest,
            publisher_key_id=bundle.manifest.publisher_key_id,
            publisher_public_key_b64=bundle.publisher_public_key_b64,
            publisher_key_fingerprint_sha256=(
                bundle.publisher_key_fingerprint_sha256
            ),
            manifest=bundle.manifest.raw,
            created_at=normalized_now,
        )
        try:
            registered = self._repository.register_package(
                package=package,
                actor_user_id=operation.actor_user_id,
            )
            current = self._repository.get_plugin_installation(
                organization_id=operation.organization_id,
                plugin_id=bundle.manifest.plugin_id,
            )
            plugin_installation = PluginInstallation(
                plugin_installation_id=(
                    current.plugin_installation_id if current is not None else uuid4()
                ),
                installation_id=operation.installation_id,
                organization_id=operation.organization_id,
                plugin_id=bundle.manifest.plugin_id,
                package_id=registered.package_id,
                previous_package_id=current.package_id if current is not None else None,
                granted_permissions=tuple(sorted(set(requested_permissions))),
                status="enabled",
                created_at=current.created_at if current is not None else normalized_now,
                updated_at=normalized_now,
            )
            instance = PluginInstance(
                instance_id=instance_id,
                plugin_installation_id=plugin_installation.plugin_installation_id,
                installation_id=operation.installation_id,
                organization_id=operation.organization_id,
                name=instance_name.strip(),
                config=dict(config),
                config_revision=1,
                status="enabled",
                created_at=normalized_now,
                updated_at=normalized_now,
            )
            installed, created_instance = self._repository.install_package(
                plugin_installation=plugin_installation,
                instance=instance,
            )
        except PluginRepositoryInvariantError as error:
            return self._fail_operation(
                operation=operation,
                code=error.code,
                now=normalized_now,
            )
        completed = self._repository.set_operation_status(
            operation_id=operation.operation_id,
            status="succeeded",
            result={
                "contract": "PluginOperation/v1alpha1",
                "plugin_installation_id": str(installed.plugin_installation_id),
                "instance_id": str(created_instance.instance_id),
                "package_id": str(registered.package_id),
                "package_digest": registered.package_digest,
            },
            updated_at=normalized_now,
        )
        self._event(
            installation_id=operation.installation_id,
            organization_id=operation.organization_id,
            actor_user_id=operation.actor_user_id,
            event_type=f"plugin.{operation.kind}.completed",
            target_type="plugin_installation",
            target_id=str(installed.plugin_installation_id),
            outcome="succeeded",
            metadata={"operation_id": str(operation.operation_id)},
            now=normalized_now,
        )
        return completed

    def submit_rollback(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        plugin_id: str,
        idempotency_key: str,
        now: datetime,
    ) -> PluginOperation:
        normalized_now = _utc(now)
        installation_id = self._authorization.require_manage(
            principal=principal,
            organization_id=organization_id,
        )
        if not _is_recent(principal=principal, now=normalized_now):
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type="plugin.rollback.rejected",
                target_id=plugin_id,
                code="recent_auth_required",
                message="Recent authentication is required for plugin lifecycle changes",
                now=normalized_now,
            )
        _require_idempotency_key(idempotency_key)
        current = self._repository.get_plugin_installation(
            organization_id=organization_id, plugin_id=plugin_id
        )
        if current is None or current.previous_package_id is None:
            raise PluginLifecycleError(
                code="plugin.rollback_unavailable", message="No previous package is available"
            )
        target_package = self._repository.get_package(
            package_id=current.previous_package_id
        )
        if target_package is None:
            raise PluginLifecycleError(
                code="plugin.rollback_unavailable",
                message="Previous plugin package is unavailable",
            )
        try:
            self._require_package_activatable(package=target_package)
        except PluginLifecycleError as error:
            self._reject(
                installation_id=installation_id,
                organization_id=organization_id,
                principal=principal,
                event_type="plugin.rollback.rejected",
                target_id=plugin_id,
                code=error.code,
                message=error.message,
                now=normalized_now,
            )
        accepted_request: Mapping[str, object] = {
            "contract": "PluginRollbackRequest/v1alpha1",
            "plugin_installation_id": str(current.plugin_installation_id),
            "expected_current_package_id": str(current.package_id),
            "target_package_id": str(current.previous_package_id),
        }
        request_hash = _request_hash(accepted_request)
        existing = self._repository.get_operation_by_idempotency(
            organization_id=organization_id, idempotency_key=idempotency_key
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise PluginLifecycleError(
                    code="plugin.idempotency_conflict",
                    message="Idempotency key is already bound to another request",
                )
            return existing
        operation = PluginOperation(
            operation_id=uuid4(),
            installation_id=installation_id,
            organization_id=organization_id,
            actor_user_id=principal.user_id,
            kind="rollback",
            target_id=str(current.plugin_installation_id),
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            request=accepted_request,
            status="pending",
            result={"contract": "PluginOperation/v1alpha1"},
            created_at=normalized_now,
            updated_at=normalized_now,
        )
        created = self._repository.create_operation(operation=operation)
        if created.request_hash != request_hash:
            raise PluginLifecycleError(
                code="plugin.idempotency_conflict",
                message="Idempotency key is already bound to another request",
            )
        return created

    def execute_rollback(self, *, operation_id: UUID, now: datetime) -> PluginOperation:
        normalized_now = _utc(now)
        operation = self._claim_pending_operation(
            operation_id=operation_id,
            now=normalized_now,
        )
        if operation.kind != "rollback":
            return self._fail_operation(
                operation=operation,
                code="plugin.operation_kind_invalid",
                now=normalized_now,
            )
        try:
            if _request_hash(operation.request) != operation.request_hash:
                raise PluginLifecycleError(
                    code="plugin.operation_payload_invalid",
                    message="Plugin operation request snapshot is invalid",
                )
            if operation.request.get("contract") != "PluginRollbackRequest/v1alpha1":
                raise PluginLifecycleError(
                    code="plugin.operation_payload_invalid",
                    message="Plugin rollback request snapshot is invalid",
                )
            plugin_installation_id = UUID(
                cast(str, operation.request["plugin_installation_id"])
            )
            if str(plugin_installation_id) != operation.target_id:
                raise PluginLifecycleError(
                    code="plugin.operation_payload_invalid",
                    message="Plugin rollback target does not match request snapshot",
                )
            expected_current_package_id = UUID(
                cast(str, operation.request["expected_current_package_id"])
            )
            target_package_id = UUID(cast(str, operation.request["target_package_id"]))
            target_package = self._repository.get_package(package_id=target_package_id)
            if target_package is None:
                raise PluginLifecycleError(
                    code="plugin.rollback_unavailable",
                    message="Previous plugin package is unavailable",
                )
            self._require_package_activatable(package=target_package)
            installation = self._repository.rollback_installation(
                plugin_installation_id=plugin_installation_id,
                expected_current_package_id=expected_current_package_id,
                target_package_id=target_package_id,
                updated_at=normalized_now,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            PluginLifecycleError,
            PluginRepositoryInvariantError,
        ) as error:
            code = (
                error.code
                if isinstance(error, (PluginLifecycleError, PluginRepositoryInvariantError))
                else "plugin.target_invalid"
            )
            return self._fail_operation(operation=operation, code=code, now=normalized_now)
        completed = self._repository.set_operation_status(
            operation_id=operation_id,
            status="succeeded",
            result={
                "contract": "PluginOperation/v1alpha1",
                "plugin_installation_id": str(installation.plugin_installation_id),
                "package_id": str(installation.package_id),
            },
            updated_at=normalized_now,
        )
        self._event(
            installation_id=operation.installation_id,
            organization_id=operation.organization_id,
            actor_user_id=operation.actor_user_id,
            event_type="plugin.rollback.completed",
            target_type="plugin_installation",
            target_id=str(installation.plugin_installation_id),
            outcome="succeeded",
            metadata={"operation_id": str(operation.operation_id)},
            now=normalized_now,
        )
        return completed

    def _claim_pending_operation(
        self,
        *,
        operation_id: UUID,
        now: datetime,
    ) -> PluginOperation:
        try:
            return self._repository.claim_pending_operation(
                operation_id=operation_id,
                updated_at=now,
            )
        except PluginRepositoryInvariantError as error:
            raise PluginLifecycleError(
                code=error.code,
                message="Plugin operation cannot be claimed",
            ) from error

    def _require_bundle_activatable(self, *, bundle: ValidatedPluginBundle) -> None:
        manifest = bundle.manifest
        if manifest.signed:
            key_id = manifest.publisher_key_id
            fingerprint = bundle.publisher_key_fingerprint_sha256
            if (
                key_id is None
                or fingerprint is None
                or bundle.publisher_public_key_b64 is None
                or self._trusted_publisher_fingerprints.get(key_id) != fingerprint
            ):
                raise PluginLifecycleError(
                    code="plugin.publisher_untrusted",
                    message="Plugin publisher key is not trusted for activation",
                )
            return
        development_mode = cast(Mapping[str, Any], manifest.raw.get("metadata", {})).get(
            "developmentMode"
        ) is True
        if (
            not development_mode
            or not self._allow_unsigned_development
            or self._trading_mode == "mainnet"
        ):
            raise PluginLifecycleError(
                code="plugin.signature_required",
                message="Unsigned plugin package cannot be activated",
            )

    def _require_package_activatable(self, *, package: PluginPackage) -> None:
        if package.publisher_key_id is None:
            development_mode = cast(
                Mapping[str, Any], package.manifest.get("metadata", {})
            ).get("developmentMode") is True
            if (
                not development_mode
                or not self._allow_unsigned_development
                or self._trading_mode == "mainnet"
            ):
                raise PluginLifecycleError(
                    code="plugin.signature_required",
                    message="Unsigned plugin package cannot be activated",
                )
            return
        fingerprint = package.publisher_key_fingerprint_sha256
        if (
            fingerprint is None
            or self._trusted_publisher_fingerprints.get(package.publisher_key_id)
            != fingerprint
            or not self._repository.is_publisher_key_trusted(
                installation_id=package.installation_id,
                key_id=package.publisher_key_id,
                fingerprint_sha256=fingerprint,
            )
        ):
            raise PluginLifecycleError(
                code="plugin.publisher_untrusted",
                message="Plugin publisher key is not currently trusted",
            )

    def _fail_operation(
        self, *, operation: PluginOperation, code: str, now: datetime
    ) -> PluginOperation:
        failed = self._repository.set_operation_status(
            operation_id=operation.operation_id,
            status="failed",
            result={"contract": "PluginOperation/v1alpha1", "error_code": code},
            updated_at=now,
        )
        self._event(
            installation_id=operation.installation_id,
            organization_id=operation.organization_id,
            actor_user_id=operation.actor_user_id,
            event_type=f"plugin.{operation.kind}.completed",
            target_type="plugin",
            target_id=operation.target_id,
            outcome="rejected",
            metadata={"operation_id": str(operation.operation_id), "reason_code": code},
            now=now,
        )
        return failed

    def _reject(
        self,
        *,
        installation_id: InstallationId,
        organization_id: OrganizationId,
        principal: CurrentUserPrincipal,
        event_type: str,
        target_id: str,
        code: str,
        message: str,
        now: datetime,
    ) -> None:
        self._event(
            installation_id=installation_id,
            organization_id=organization_id,
            actor_user_id=principal.user_id,
            event_type=event_type,
            target_type="plugin",
            target_id=target_id,
            outcome="rejected",
            metadata={"reason_code": code},
            now=now,
        )
        raise PluginLifecycleError(code=code, message=message)

    def _event(
        self,
        *,
        installation_id: InstallationId,
        organization_id: OrganizationId,
        actor_user_id: UserId,
        event_type: str,
        target_type: str,
        target_id: str,
        outcome: Literal["succeeded", "rejected"],
        metadata: Mapping[str, str],
        now: datetime,
    ) -> None:
        self._repository.record_event(
            event=PluginEvent(
                event_id=uuid4(),
                installation_id=installation_id,
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                event_type=event_type,
                target_type=target_type,
                target_id=target_id,
                outcome=outcome,
                metadata=dict(metadata),
                created_at=now,
            )
        )


def _request_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _install_request_payload(
    *,
    bundle: ValidatedPluginBundle,
    permissions: tuple[str, ...],
    instance_name: str,
    config: Mapping[str, object],
    instance_id: UUID,
) -> Mapping[str, object]:
    manifest = bundle.manifest
    return {
        "contract": "PluginInstallRequest/v1alpha1",
        "bundle": {
            "bundle_path": bundle.bundle_path,
            "artifact_digests": dict(bundle.artifact_digests),
            "publisher_public_key_b64": bundle.publisher_public_key_b64,
            "publisher_key_fingerprint_sha256": (
                bundle.publisher_key_fingerprint_sha256
            ),
            "manifest": {
                "plugin_id": manifest.plugin_id,
                "version": manifest.version,
                "publisher": manifest.publisher,
                "plugin_type": manifest.plugin_type,
                "plugin_api_version": manifest.plugin_api_version,
                "rpc_version": manifest.rpc_version,
                "image_reference": manifest.image_reference,
                "image_digest": manifest.image_digest,
                "architectures": list(manifest.architectures),
                "permissions": list(manifest.permissions),
                "config_schema": dict(manifest.config_schema),
                "license_spdx": manifest.license_spdx,
                "package_digest": manifest.package_digest,
                "publisher_key_id": manifest.publisher_key_id,
                "signed": manifest.signed,
                "raw": dict(manifest.raw),
            },
        },
        "permissions": list(permissions),
        "instance_name": instance_name,
        "config": dict(config),
        "instance_id": str(instance_id),
    }


def _install_request_from_operation(
    operation: PluginOperation,
) -> tuple[ValidatedPluginBundle, tuple[str, ...], str, Mapping[str, object], UUID]:
    request = operation.request
    if (
        request.get("contract") != "PluginInstallRequest/v1alpha1"
        or _request_hash(request) != operation.request_hash
    ):
        raise ValueError("plugin operation request hash mismatch")
    bundle_payload = cast(Mapping[str, Any], request["bundle"])
    manifest_payload = cast(Mapping[str, Any], bundle_payload["manifest"])
    manifest = PluginManifest(
        plugin_id=cast(str, manifest_payload["plugin_id"]),
        version=cast(str, manifest_payload["version"]),
        publisher=cast(str, manifest_payload["publisher"]),
        plugin_type=cast(Any, manifest_payload["plugin_type"]),
        plugin_api_version=cast(str, manifest_payload["plugin_api_version"]),
        rpc_version=cast(str, manifest_payload["rpc_version"]),
        image_reference=cast(str, manifest_payload["image_reference"]),
        image_digest=cast(str, manifest_payload["image_digest"]),
        architectures=tuple(cast(list[str], manifest_payload["architectures"])),
        permissions=tuple(cast(list[str], manifest_payload["permissions"])),
        config_schema=cast(Mapping[str, Any], manifest_payload["config_schema"]),
        license_spdx=cast(str, manifest_payload["license_spdx"]),
        package_digest=cast(str, manifest_payload["package_digest"]),
        publisher_key_id=cast(str | None, manifest_payload["publisher_key_id"]),
        signed=cast(bool, manifest_payload["signed"]),
        raw=cast(Mapping[str, Any], manifest_payload["raw"]),
    )
    if manifest.plugin_id != operation.target_id:
        raise ValueError("plugin operation target mismatch")
    permissions = tuple(cast(list[str], request["permissions"]))
    if set(permissions) - set(manifest.permissions):
        raise ValueError("plugin operation permissions mismatch")
    config = cast(Mapping[str, object], request["config"])
    if _contains_secret_shaped_key(config):
        raise ValueError("plugin operation contains raw secret-shaped configuration")
    bundle = ValidatedPluginBundle(
        bundle_path=cast(str, bundle_payload["bundle_path"]),
        artifact_digests=cast(
            Mapping[str, str], bundle_payload["artifact_digests"]
        ),
        publisher_public_key_b64=cast(
            str | None, bundle_payload["publisher_public_key_b64"]
        ),
        publisher_key_fingerprint_sha256=cast(
            str | None, bundle_payload["publisher_key_fingerprint_sha256"]
        ),
        manifest=manifest,
    )
    return (
        bundle,
        permissions,
        cast(str, request["instance_name"]),
        config,
        UUID(cast(str, request["instance_id"])),
    )


def _contains_secret_shaped_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SECRET_KEY_RE.fullmatch(str(key)) is not None
            or _contains_secret_shaped_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_shaped_key(item) for item in value)
    return False


def _require_idempotency_key(value: str) -> None:
    if _IDEMPOTENCY_RE.fullmatch(value) is None:
        raise PluginLifecycleError(
            code="plugin.idempotency_key_invalid", message="Idempotency key is invalid"
        )


def _is_recent(*, principal: CurrentUserPrincipal, now: datetime) -> bool:
    if principal.session_created_at is None:
        return False
    authenticated_at = _utc(principal.session_created_at)
    return authenticated_at <= now <= authenticated_at + _RECENT_AUTH_WINDOW


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise PluginLifecycleError(
            code="plugin.timestamp_invalid", message="Plugin timestamp must be timezone-aware"
        )
    return value.astimezone(UTC)
