"""Library-independent public contracts for ArtifactStore/v1."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

ARTIFACT_STORE_SCHEMA = "ArtifactStore/v1"
ARTIFACT_MANIFEST_SCHEMA = "ArtifactManifest/v1"
ARTIFACT_BACKUP_SCHEMA = "ArtifactBackup/v1"
MAX_ARTIFACT_BLOB_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BUNDLE_BYTES = 64 * 1024 * 1024

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PORTABLE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{2,127}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SECRET_KEY_RE = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|authorization|cookie|credential|password|secret|token)(?:$|[._-])",
    re.IGNORECASE,
)
_PORTABLE_PATH_SCHEMA_PATTERN = r"^(?!/)(?!.*(?:^|/)\.{1,2}(?:/|$))(?!.*\\\\)(?!.*//)(?!.*/$).+$"

PortableId = Annotated[str, StringConstraints(pattern=_PORTABLE_ID_RE.pattern)]
Sha256Digest = Annotated[str, StringConstraints(pattern=_DIGEST_RE.pattern)]
SemanticVersion = Annotated[str, StringConstraints(pattern=_SEMVER_RE.pattern)]
MetadataString = Annotated[str, StringConstraints(max_length=512)]
ArtifactMetadataValue = MetadataString | int | bool


def sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class ArtifactBlobDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    digest: Sha256Digest
    size_bytes: int = Field(ge=0, le=MAX_ARTIFACT_BLOB_BYTES)
    media_type: str = Field(min_length=1, max_length=127, pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")


class ArtifactManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        min_length=1,
        max_length=240,
        json_schema_extra={"pattern": _PORTABLE_PATH_SCHEMA_PATTERN},
    )
    blob: ArtifactBlobDescriptor
    executable: Literal[False] = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            value.startswith("/")
            or value.endswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != value
        ):
            raise ValueError("artifact entry path must be a normalized relative POSIX path")
        return value


class ArtifactBundleSignature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["Ed25519"] = "Ed25519"
    key_id: PortableId
    value_b64: str = Field(min_length=88, max_length=88, pattern=r"^[A-Za-z0-9+/]{86}==$")

    @field_validator("value_b64")
    @classmethod
    def validate_signature(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("artifact signature must be canonical base64") from error
        if len(decoded) != 64 or base64.b64encode(decoded).decode() != value:
            raise ValueError("artifact signature must be a 64-byte Ed25519 signature")
        return value


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["ArtifactManifest/v1"] = Field(alias="schema")
    bundle_id: PortableId
    name: str = Field(min_length=1, max_length=120)
    version: SemanticVersion
    created_at: datetime
    entries: tuple[ArtifactManifestEntry, ...] = Field(
        min_length=1,
        max_length=256,
        json_schema_extra={"x-roehub-unique-by": "path"},
    )
    metadata: dict[str, ArtifactMetadataValue] = Field(
        default_factory=dict,
        json_schema_extra={
            "maxProperties": 32,
            "propertyNames": {
                "pattern": _PORTABLE_ID_RE.pattern,
                "not": {"pattern": _SECRET_KEY_RE.pattern},
            },
        },
    )
    signature: ArtifactBundleSignature

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("artifact manifest created_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls, value: dict[str, ArtifactMetadataValue]
    ) -> dict[str, ArtifactMetadataValue]:
        if len(value) > 32:
            raise ValueError("artifact manifest metadata is too large")
        for key, item in value.items():
            if not _PORTABLE_ID_RE.fullmatch(key) or _SECRET_KEY_RE.search(key):
                raise ValueError("artifact manifest metadata key is unsafe")
            if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**53 - 1:
                raise ValueError("artifact manifest metadata integer is not portable")
        return value

    @model_validator(mode="after")
    def validate_entries(self) -> "ArtifactManifest":
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact manifest entry paths must be unique")
        return self

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("signature")
        return payload

    def signed_bytes(self) -> bytes:
        return json.dumps(
            self.unsigned_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()

    @property
    def manifest_digest(self) -> str:
        return sha256_digest(self.signed_bytes())


class ArtifactStoreCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    atomic_publish: Literal[True] = True
    backup_restore: Literal[True] = True
    digest_verification: Literal[True] = True
    garbage_collection: Literal[True] = True
    leases: Literal[True] = True
    local_materialization: Literal[True] = True
    pins: Literal[True] = True
    quotas: Literal[True] = True


class ArtifactStoreDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["ArtifactStore/v1"] = Field(alias="schema")
    backend: Literal["local_cas", "s3_compatible"]
    digest_algorithm: Literal["sha256"] = "sha256"
    immutable: Literal[True] = True
    capabilities: ArtifactStoreCapabilities = Field(default_factory=ArtifactStoreCapabilities)


class ArtifactBackupBlob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    digest: Sha256Digest
    size_bytes: int = Field(ge=0, le=MAX_ARTIFACT_BLOB_BYTES)


class ArtifactBackupCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["ArtifactBackup/v1"] = Field(alias="schema")
    source_organization_id: UUID
    quota_bytes: int = Field(ge=1, le=1_099_511_627_776)
    manifests: tuple[ArtifactManifest, ...] = Field(max_length=4096)
    pinned_digests: tuple[Sha256Digest, ...] = Field(max_length=65536)
    blobs: tuple[ArtifactBackupBlob, ...] = Field(max_length=65536)

    @model_validator(mode="after")
    def validate_catalog(self) -> "ArtifactBackupCatalog":
        manifest_digests = [manifest.manifest_digest for manifest in self.manifests]
        if len(manifest_digests) != len(set(manifest_digests)):
            raise ValueError("artifact backup manifests must be unique")
        blob_by_digest = {blob.digest: blob for blob in self.blobs}
        if len(blob_by_digest) != len(self.blobs):
            raise ValueError("artifact backup blobs must be unique")
        if len(self.pinned_digests) != len(set(self.pinned_digests)):
            raise ValueError("artifact backup pins must be unique")
        if not set(self.pinned_digests).issubset(blob_by_digest):
            raise ValueError("artifact backup pins must reference included blobs")
        for manifest in self.manifests:
            for entry in manifest.entries:
                backup_blob = blob_by_digest.get(entry.blob.digest)
                if backup_blob is None or backup_blob.size_bytes != entry.blob.size_bytes:
                    raise ValueError("artifact backup manifest blob is missing or mismatched")
        return self


__all__ = [
    "ARTIFACT_BACKUP_SCHEMA",
    "ARTIFACT_MANIFEST_SCHEMA",
    "ARTIFACT_STORE_SCHEMA",
    "MAX_ARTIFACT_BLOB_BYTES",
    "MAX_ARTIFACT_BUNDLE_BYTES",
    "ArtifactBackupBlob",
    "ArtifactBackupCatalog",
    "ArtifactBlobDescriptor",
    "ArtifactBundleSignature",
    "ArtifactManifest",
    "ArtifactManifestEntry",
    "ArtifactStoreCapabilities",
    "ArtifactStoreDescriptor",
    "sha256_digest",
]
