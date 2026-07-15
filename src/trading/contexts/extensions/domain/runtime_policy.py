from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, cast

from .models import PluginManifest


@dataclass(frozen=True, slots=True)
class PluginEgressRule:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class PluginOciRuntimeSpec:
    image_reference: str
    image_digest: str
    user: str
    read_only_root_filesystem: bool
    no_new_privileges: bool
    cap_drop: tuple[str, ...]
    cpus: float
    memory_mb: int
    pids: int
    tmpfs: tuple[str, ...]
    internal_network: str
    egress: tuple[PluginEgressRule, ...]
    mounts: tuple[object, ...] = ()
    environment: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class PluginRuntimePolicy:
    non_root_uid: int
    read_only_root_filesystem: bool
    no_new_privileges: bool
    cpus: float
    memory_mb: int
    pids: int
    egress: tuple[PluginEgressRule, ...]

    @classmethod
    def from_manifest(cls, manifest: PluginManifest) -> PluginRuntimePolicy:
        spec = cast(Mapping[str, Any], manifest.raw["spec"])
        runtime = cast(Mapping[str, Any], spec["runtime"])
        resources = cast(Mapping[str, Any], runtime["resources"])
        return cls(
            non_root_uid=int(runtime["nonRootUid"]),
            read_only_root_filesystem=runtime["readOnlyRootFilesystem"] is True,
            no_new_privileges=runtime["noNewPrivileges"] is True,
            cpus=float(resources["cpus"]),
            memory_mb=int(resources["memoryMb"]),
            pids=int(resources["pids"]),
            egress=tuple(
                PluginEgressRule(host=str(rule["host"]), port=int(rule["port"]))
                for rule in cast(list[Mapping[str, Any]], runtime["egress"])
            ),
        )

    def to_oci_spec(
        self,
        *,
        manifest: PluginManifest,
        internal_network: str,
        egress_gateway_configured: bool = False,
    ) -> PluginOciRuntimeSpec:
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{2,127}", internal_network) is None:
            raise ValueError("plugin internal network name is invalid")
        if self.egress and not egress_gateway_configured:
            raise ValueError("declared plugin egress requires a policy-enforcing gateway")
        if self.non_root_uid < 10000:
            raise ValueError("plugin runtime uid must be non-root")
        if not self.read_only_root_filesystem or not self.no_new_privileges:
            raise ValueError("plugin runtime hardening cannot be disabled")
        return PluginOciRuntimeSpec(
            image_reference=manifest.image_reference,
            image_digest=manifest.image_digest,
            user=f"{self.non_root_uid}:{self.non_root_uid}",
            read_only_root_filesystem=True,
            no_new_privileges=True,
            cap_drop=("ALL",),
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            pids=self.pids,
            tmpfs=("/tmp:rw,noexec,nosuid,size=16m",),
            internal_network=internal_network,
            egress=self.egress,
        )
