from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
from uuid import UUID

PLUGIN_RPC_VERSION = "roehub.plugin.rpc/v1alpha1"
PluginCapability = Literal[
    "app.action",
    "data.read",
    "notification.send",
    "panel.describe",
]


@dataclass(frozen=True, slots=True)
class PluginContext:
    organization_id: UUID
    instance_id: UUID
    package_digest: str
    package_version: str
    capability: PluginCapability


@dataclass(frozen=True, slots=True)
class PluginResponse:
    status: str
    data: Mapping[str, Any]
    contract: str = "PluginResponse/v1alpha1"

    def as_json(self) -> dict[str, Any]:
        return {"contract": self.contract, "status": self.status, "data": dict(self.data)}


def require_idempotency_key(value: str | None) -> str:
    if value is None or not 8 <= len(value) <= 128:
        raise ValueError("Idempotency-Key must contain 8 to 128 characters")
    return value
