from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from trading.integration import JobCapability, JobEnvelope
from trading.integration.job_runtime_postgres import PostgresJobRuntimeCatalog


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



class TrustedRuntimeAuthority:
    """Fail-closed registry supplied by the host composition root."""

    def __init__(
        self,
        *,
        grants: tuple[TrustedRuntimeGrant, ...],
    ) -> None:
        if not grants:
            raise ValueError("trusted runtime authority requires at least one grant")
        self._grants = frozenset(grants)
        if len(self._grants) != len(grants):
            raise ValueError("trusted runtime grants must be unique")

    def authorize(self, envelope: JobEnvelope) -> None:
        candidate = TrustedRuntimeGrant.for_builtin(envelope)
        if candidate not in self._grants:
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
    "TrustedRuntimeAuthority",
    "TrustedRuntimeGrant",
]
