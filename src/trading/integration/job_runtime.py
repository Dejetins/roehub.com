"""Library-independent public contracts for isolated job runtimes."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

JOB_ENVELOPE_SCHEMA = "JobEnvelope/v1"
JOB_RESULT_SCHEMA = "JobResultManifest/v1"
MAX_JOB_CONFIG_BYTES = 256 * 1024
MAX_JOB_OUTPUT_BYTES = 64 * 1024 * 1024

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLUGIN_PACKAGE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{2,127}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SEMANTIC_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_SECRET_KEY_RE = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|authorization|cookie|credential|dsn|password|secret|token)(?:$|[._-])",
    re.IGNORECASE,
)

Sha256Digest = Annotated[str, StringConstraints(pattern=_DIGEST_RE.pattern)]
PluginPackageDigest = Annotated[str, StringConstraints(pattern=_PLUGIN_PACKAGE_DIGEST_RE.pattern)]
PortableId = Annotated[str, StringConstraints(pattern=_PORTABLE_ID_RE.pattern)]
SemanticVersion = Annotated[str, StringConstraints(pattern=_SEMVER_RE.pattern)]
SemanticJobKey = Annotated[str, StringConstraints(pattern=_SEMANTIC_KEY_RE.pattern)]
JobCapability = Literal[
    "backtest",
    "optimize",
    "history_import",
    "report",
    "artifact_transform",
    "ml_training",
    "ml_inference",
    "rl_training",
    "rl_inference",
    "custom_strategy",
]
JobOutcome = Literal[
    "succeeded",
    "failed",
    "crashed",
    "canceled",
    "timed_out",
    "resource_exhausted",
]


def sha256_job_payload(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _validate_json_value(value: JsonValue, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        if len(value) > 256:
            raise ValueError("job config object has too many properties")
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("job config key is invalid")
            if _SECRET_KEY_RE.search(key):
                raise ValueError("job config contains a secret-shaped key")
            _validate_json_value(item, path=(*path, key))
        return
    if isinstance(value, list):
        if len(value) > 4096:
            raise ValueError("job config array is too large")
        for index, item in enumerate(value):
            _validate_json_value(item, path=(*path, str(index)))
        return
    if isinstance(value, float):
        raise ValueError("job config floats are not canonical; use decimal strings")
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        raise ValueError("job config integer is not portable")
    if isinstance(value, str) and len(value) > 16_384:
        raise ValueError("job config string is too large")


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError("job output path must be a normalized relative POSIX path")
    return value


class JobResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_millis: int = Field(ge=50, le=64_000)
    memory_bytes: int = Field(ge=16 * 1024 * 1024, le=64 * 1024 * 1024 * 1024)
    pids: int = Field(ge=16, le=4096)
    wall_time_seconds: int = Field(ge=1, le=86_400)
    tmpfs_bytes: int = Field(ge=1024 * 1024, le=MAX_JOB_OUTPUT_BYTES)
    output_bytes: int = Field(ge=1, le=MAX_JOB_OUTPUT_BYTES)


class JobRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: PortableId
    version: SemanticVersion
    plugin_package_digest: PluginPackageDigest | None = None


class JobEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["JobEnvelope/v1"] = Field(alias="schema")
    job_id: UUID
    attempt_id: UUID
    attempt_number: int = Field(ge=1, le=10_000)
    organization_id: UUID
    semantic_job_key: SemanticJobKey
    capability: JobCapability
    image_digest: Sha256Digest
    runtime: JobRuntimeIdentity
    config_snapshot: dict[str, JsonValue]
    input_artifact_digests: tuple[Sha256Digest, ...] = Field(max_length=256)
    limits: JobResourceLimits
    deadline: datetime
    command: tuple[str, ...] = Field(min_length=1, max_length=64)
    network: Literal["none"] = "none"

    @field_validator("deadline")
    @classmethod
    def normalize_deadline(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("job deadline must include a timezone")
        return value.astimezone(UTC)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 4096 or "\x00" in item for item in value):
            raise ValueError("job command contains an invalid argument")
        if sum(len(item.encode()) for item in value) > 32 * 1024:
            raise ValueError("job command is too large")
        return value

    @field_validator("config_snapshot")
    @classmethod
    def validate_config_snapshot(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_json_value(value)
        if len(_canonical_json_bytes(value)) > MAX_JOB_CONFIG_BYTES:
            raise ValueError("job config snapshot is too large")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "JobEnvelope":
        if len(self.input_artifact_digests) != len(set(self.input_artifact_digests)):
            raise ValueError("job input artifact digests must be unique")
        if self.capability == "custom_strategy" and self.runtime.plugin_package_digest is None:
            raise ValueError("custom strategy requires a signed plugin package digest")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.model_dump(mode="json", by_alias=True))

    @property
    def envelope_digest(self) -> str:
        return sha256_job_payload(self.canonical_bytes())

    @property
    def semantic_spec_digest(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True)
        for key in ("job_id", "attempt_id", "attempt_number", "deadline"):
            payload.pop(key)
        return sha256_job_payload(_canonical_json_bytes(payload))

    @property
    def command_digest(self) -> str:
        return sha256_job_payload(_canonical_json_bytes(self.command))


class JobOutputDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=240)
    digest: Sha256Digest
    size_bytes: int = Field(ge=0, le=MAX_JOB_OUTPUT_BYTES)
    media_type: str = Field(
        min_length=1,
        max_length=127,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class StrategyRuntimeDecision(BaseModel):
    """Strict host-owned output: a strategy cannot return an exchange order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["signal", "intent"]
    instrument_id: PortableId
    side: Literal["buy", "sell", "flat"]
    strength_decimal: str = Field(pattern=r"^(0|1|0\.[0-9]{1,8})$")
    observed_at: datetime
    reason_code: PortableId

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("strategy decision observed_at must include a timezone")
        return value.astimezone(UTC)


class JobResultManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["JobResultManifest/v1"] = Field(alias="schema")
    job_id: UUID
    attempt_id: UUID
    organization_id: UUID
    outcome: JobOutcome
    envelope_digest: Sha256Digest
    output_artifact_manifest_digest: Sha256Digest | None = None
    outputs: tuple[JobOutputDescriptor, ...] = Field(max_length=256)
    strategy_decisions: tuple[StrategyRuntimeDecision, ...] = Field(max_length=4096)
    completed_at: datetime
    exit_code: int | None = Field(default=None, ge=0, le=255)
    error_code: PortableId | None = None

    @field_validator("completed_at")
    @classmethod
    def normalize_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("job result completed_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_result(self) -> "JobResultManifest":
        paths = [output.path for output in self.outputs]
        if len(paths) != len(set(paths)):
            raise ValueError("job output paths must be unique")
        if sum(output.size_bytes for output in self.outputs) > MAX_JOB_OUTPUT_BYTES:
            raise ValueError("job outputs exceed the aggregate byte limit")
        if self.outcome == "succeeded" and self.output_artifact_manifest_digest is None:
            raise ValueError("successful job result requires an artifact manifest")
        if self.outcome != "succeeded" and self.output_artifact_manifest_digest is not None:
            raise ValueError("failed job result cannot publish an artifact manifest")
        return self


__all__ = [
    "JOB_ENVELOPE_SCHEMA",
    "JOB_RESULT_SCHEMA",
    "MAX_JOB_CONFIG_BYTES",
    "MAX_JOB_OUTPUT_BYTES",
    "PluginPackageDigest",
    "JobCapability",
    "JobEnvelope",
    "JobOutcome",
    "JobOutputDescriptor",
    "JobResourceLimits",
    "JobResultManifest",
    "JobRuntimeIdentity",
    "StrategyRuntimeDecision",
    "sha256_job_payload",
]
