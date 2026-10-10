"""Versioned contracts crossing the control-agent trust boundary."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class OperationAction(StrEnum):
    INSPECT = "inspect"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    RECOVER = "recover"
    INSTALL = "install"
    UPDATE = "update"
    ROLLBACK = "rollback"
    PLUGIN_INSTALL = "plugin.install"
    PLUGIN_UPDATE = "plugin.update"
    PLUGIN_ROLLBACK = "plugin.rollback"
    PLUGIN_ENABLE = "plugin.enable"
    PLUGIN_DISABLE = "plugin.disable"
    BACKUP = "backup"
    RESTORE = "restore"
    BACKUP_CANCEL = "backup.cancel"
    RESTORE_CANCEL = "restore.cancel"
    DIAGNOSTICS = "diagnostics"


class OperationState(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class ControlOperationError(RuntimeError):
    """Stable failure crossing the operation boundary."""

    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class OperationRequest(BaseModel):
    """A closed, typed operation; arbitrary commands and runtime overrides are forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.control-operation/v1alpha1"] = Field(
        default="io.roehub.control-operation/v1alpha1", alias="schema"
    )
    operation_id: UUID
    action: OperationAction
    profile: Literal["base", "trading", "ml"] = "base"
    services: tuple[str, ...] = ()
    release_version: str | None = None
    subject_id: str | None = None
    package_digest: str | None = None

    @model_validator(mode="after")
    def _validate_semantics(self) -> OperationRequest:
        if len(self.services) != len(set(self.services)):
            raise ValueError("services must be unique")
        if any(_SAFE_IDENTIFIER.fullmatch(service) is None for service in self.services):
            raise ValueError("service name is invalid")
        if self.subject_id is not None and _SAFE_IDENTIFIER.fullmatch(self.subject_id) is None:
            raise ValueError("subject_id is invalid")
        if self.package_digest is not None and _DIGEST.fullmatch(self.package_digest) is None:
            raise ValueError("package_digest must be an exact sha256 digest")
        release_actions = {
            OperationAction.INSTALL,
            OperationAction.UPDATE,
            OperationAction.ROLLBACK,
        }
        plugin_actions = {
            OperationAction.PLUGIN_INSTALL,
            OperationAction.PLUGIN_UPDATE,
            OperationAction.PLUGIN_ROLLBACK,
            OperationAction.PLUGIN_ENABLE,
            OperationAction.PLUGIN_DISABLE,
        }
        subject_actions = plugin_actions | {
            OperationAction.BACKUP,
            OperationAction.RESTORE,
            OperationAction.BACKUP_CANCEL,
            OperationAction.RESTORE_CANCEL,
        }
        if self.action in release_actions and not self.release_version:
            raise ValueError("release_version is required for release lifecycle operations")
        if self.release_version is not None and self.action not in release_actions:
            raise ValueError("release_version is forbidden for this action")
        if self.action in subject_actions and not self.subject_id:
            raise ValueError("subject_id is required for this operation")
        if self.subject_id is not None and self.action not in subject_actions:
            raise ValueError("subject_id is forbidden for this action")
        digest_actions = {
            OperationAction.PLUGIN_INSTALL,
            OperationAction.PLUGIN_UPDATE,
        }
        if self.action in digest_actions and self.package_digest is None:
            raise ValueError("package_digest is required for plugin install or update")
        if self.package_digest is not None and self.action not in digest_actions:
            raise ValueError("package_digest is forbidden for this action")
        if self.services and self.action not in {
            OperationAction.INSPECT,
            OperationAction.START,
            OperationAction.STOP,
            OperationAction.RESTART,
            OperationAction.RECOVER,
            OperationAction.DIAGNOSTICS,
        }:
            raise ValueError("services are forbidden for this action")
        return self

    def canonical_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def request_digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class OperationResult(BaseModel):
    """Redacted operation result safe for the journal and API reconciliation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.control-operation-result/v1alpha1"] = Field(
        default="io.roehub.control-operation-result/v1alpha1", alias="schema"
    )
    operation_id: UUID
    action: OperationAction
    profile: Literal["base", "trading", "ml"]
    state: OperationState
    detail_code: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    active_services: tuple[str, ...] = ()
    journal_sequence: int | None = Field(default=None, ge=1)


class ControlAgentRequest(BaseModel):
    """Authenticated local transport envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.control-agent-request/v1alpha1"] = Field(
        default="io.roehub.control-agent-request/v1alpha1", alias="schema"
    )
    identity: Literal["api", "installation_owner"]
    credential: str = Field(min_length=32, max_length=512)
    method: Literal["submit", "get", "reconcile", "journal"]
    operation: OperationRequest | None = None
    operation_id: UUID | None = None
    after_sequence: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_method(self) -> ControlAgentRequest:
        if self.method == "submit" and self.operation is None:
            raise ValueError("operation is required for submit")
        if self.method in {"get", "reconcile"} and self.operation_id is None:
            raise ValueError("operation_id is required")
        if self.method != "submit" and self.operation is not None:
            raise ValueError("operation is forbidden for this method")
        if self.method not in {"get", "reconcile"} and self.operation_id is not None:
            raise ValueError("operation_id is forbidden for this method")
        if self.method != "journal" and self.after_sequence != 0:
            raise ValueError("after_sequence is allowed only for journal")
        return self

    @property
    def authorization_digest(self) -> str:
        """Hash every request field except the replaceable assertion itself."""

        payload = self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"credential"},
            exclude_none=True,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class ControlAgentResponse(BaseModel):
    """Secret-free response from the local control-agent transport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.control-agent-response/v1alpha1"] = Field(
        default="io.roehub.control-agent-response/v1alpha1", alias="schema"
    )
    status: Literal["ok", "error"]
    result: OperationResult | None = None
    journal_entries: tuple[dict[str, object], ...] = ()
    error_code: str | None = None


__all__ = [
    "ControlAgentRequest",
    "ControlAgentResponse",
    "ControlOperationError",
    "OperationAction",
    "OperationRequest",
    "OperationResult",
    "OperationState",
]
