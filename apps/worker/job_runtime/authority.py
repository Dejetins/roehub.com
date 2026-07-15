from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from trading.contexts.extensions.domain import PluginInstallation, PluginPackage
from trading.integration import JobCapability, JobEnvelope
from trading.integration.job_runtime_postgres import PostgresJobRuntimeCatalog

PluginTrustResolution = tuple[
    PluginPackage,
    PluginInstallation,
    Mapping[str, str],
]
PluginTrustResolver = Callable[[JobEnvelope], PluginTrustResolution]


class JobRuntimeAuthorityError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class TrustedRuntimeGrant:
    """Host-owned exact grant for one reviewed runtime entrypoint."""

    capability: JobCapability
    runtime_name: str
    runtime_version: str
    image_digest: str
    command_digest: str
    plugin_package_digest: str | None = None

    @classmethod
    def _from_envelope(cls, envelope: JobEnvelope) -> "TrustedRuntimeGrant":
        return cls(
            capability=envelope.capability,
            runtime_name=envelope.runtime.name,
            runtime_version=envelope.runtime.version,
            image_digest=envelope.image_digest,
            command_digest=envelope.command_digest,
            plugin_package_digest=envelope.runtime.plugin_package_digest,
        )

    @classmethod
    def for_builtin(cls, envelope: JobEnvelope) -> "TrustedRuntimeGrant":
        if envelope.capability == "custom_strategy":
            raise JobRuntimeAuthorityError(code="job.plugin_package_authority_required")
        if envelope.runtime.plugin_package_digest is not None:
            raise JobRuntimeAuthorityError(code="job.plugin_package_unexpected")
        return cls._from_envelope(envelope)

    @classmethod
    def for_signed_plugin(
        cls,
        *,
        envelope: JobEnvelope,
        package: PluginPackage,
        installation: PluginInstallation,
        trusted_publisher_fingerprints: Mapping[str, str],
    ) -> "TrustedRuntimeGrant":
        if envelope.capability != "custom_strategy":
            raise JobRuntimeAuthorityError(code="job.plugin_capability_invalid")
        if (
            package.publisher_key_id is None
            or package.publisher_public_key_b64 is None
            or package.publisher_key_fingerprint_sha256 is None
        ):
            raise JobRuntimeAuthorityError(code="job.plugin_package_unsigned")
        if (
            installation.organization_id.value != envelope.organization_id
            or installation.installation_id != package.installation_id
            or installation.package_id != package.package_id
            or installation.status != "enabled"
        ):
            raise JobRuntimeAuthorityError(code="job.plugin_installation_not_enabled")
        if (
            trusted_publisher_fingerprints.get(package.publisher_key_id)
            != package.publisher_key_fingerprint_sha256
        ):
            raise JobRuntimeAuthorityError(code="job.plugin_publisher_untrusted")
        if (
            package.package_digest != envelope.runtime.plugin_package_digest
            or package.image_digest != envelope.image_digest
            or package.version != envelope.runtime.version
        ):
            raise JobRuntimeAuthorityError(code="job.plugin_package_identity_mismatch")
        return cls._from_envelope(envelope)


class TrustedRuntimeAuthority:
    """Fail-closed registry supplied by the host composition root."""

    def __init__(
        self,
        *,
        grants: tuple[TrustedRuntimeGrant, ...],
        plugin_trust_resolver: PluginTrustResolver | None = None,
    ) -> None:
        if not grants:
            raise ValueError("trusted runtime authority requires at least one grant")
        self._grants = frozenset(grants)
        if len(self._grants) != len(grants):
            raise ValueError("trusted runtime grants must be unique")
        self._plugin_trust_resolver = plugin_trust_resolver

    def authorize(self, envelope: JobEnvelope) -> None:
        candidate = TrustedRuntimeGrant._from_envelope(envelope)
        if candidate not in self._grants:
            raise JobRuntimeAuthorityError(code="job.runtime_not_trusted")
        if envelope.capability != "custom_strategy":
            return
        if self._plugin_trust_resolver is None:
            raise JobRuntimeAuthorityError(code="job.plugin_trust_resolver_missing")
        try:
            package, installation, trust_root = self._plugin_trust_resolver(envelope)
        except (KeyError, LookupError, OSError, ValueError) as error:
            raise JobRuntimeAuthorityError(code="job.plugin_trust_unavailable") from error
        current = TrustedRuntimeGrant.for_signed_plugin(
            envelope=envelope,
            package=package,
            installation=installation,
            trusted_publisher_fingerprints=trust_root,
        )
        if current != candidate:
            raise JobRuntimeAuthorityError(code="job.runtime_not_trusted")


class JobSubmissionService:
    """Production-callable host boundary; untrusted envelopes never reach PostgreSQL."""

    def __init__(
        self,
        *,
        catalog: PostgresJobRuntimeCatalog,
        authority: TrustedRuntimeAuthority,
    ) -> None:
        self._catalog = catalog
        self._authority = authority

    def submit(self, *, envelope: JobEnvelope, created_at: datetime) -> UUID:
        self._authority.authorize(envelope)
        return self._catalog.submit(envelope=envelope, created_at=created_at)


__all__ = [
    "JobRuntimeAuthorityError",
    "JobSubmissionService",
    "PluginTrustResolution",
    "PluginTrustResolver",
    "TrustedRuntimeAuthority",
    "TrustedRuntimeGrant",
]
