#!/usr/bin/env python3
"""Create and verify a signed Roehub offline release bundle.

The bundle is content-addressed and fail-closed: its detached SSHSIG-Ed25519
signature authenticates the canonical manifest, while the manifest
authenticates every payload byte. Image archives are inspected as OCI layouts
before inclusion so the online digest and both supported Linux platforms remain
one contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.release.runtime_license_audit import (
        RuntimeLicenseAuditError,
        verify_runtime_license_audit,
        write_runtime_license_audit,
    )
except ModuleNotFoundError:  # Standalone verification from an extracted bundle.
    from runtime_license_audit import (  # type: ignore[no-redef]
        RuntimeLicenseAuditError,
        verify_runtime_license_audit,
        write_runtime_license_audit,
    )

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_MANIFEST = ROOT / "tools" / "release" / "release-metadata.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "release" / "offline-release-manifest.schema.json"
MANIFEST_NAME = "offline-release-manifest.json"
SIGNATURE_NAME = "offline-release-manifest.signature.json"
SCHEMA_ID = "io.roehub.offline-release-bundle/v1alpha1"
SIGNATURE_SCHEMA_ID = "io.roehub.offline-release-signature/v1alpha1"
SUPPORTED_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})
SIGNATURE_NAMESPACE = "roehub.offline-release-manifest.v1"
SIGNER_IDENTITY = "roehub-release"
MAX_JSON_BYTES = 16 * 1024 * 1024
SOURCE_DATE_EPOCH = "1970-01-01T00:00:00Z"
OCI_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    }
)
OCI_MANIFEST_MEDIA_TYPES = frozenset(
    {
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
    }
)
OCI_CONFIG_MEDIA_TYPES = frozenset(
    {
        "application/vnd.docker.container.image.v1+json",
        "application/vnd.oci.image.config.v1+json",
    }
)

STATIC_FILES = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "configs/installation/roehub.yaml",
    "configs/installation/runtime-service-manifest.json",
    "infra/docker/Dockerfile.runtime",
    "infra/docker/openbao/Dockerfile",
    "infra/docker/openbao/buildkitd.toml",
    "infra/docker/openbao/openbao-2.5.4-licensed-qr.NOTICE",
    "infra/docker/openbao/openbao-2.5.4-licensed-qr.patch",
    "infra/docker/runtime-entrypoint.py",
    "infra/openbao/config/openbao.hcl",
    "tools/release/README.md",
    "tools/release/THIRD_PARTY_NOTICES.md",
    "tools/release/build_openbao_image.py",
    "tools/release/install-offline.sh",
    "tools/release/offline_bundle.py",
    "tools/release/runtime-license-policy.json",
    "tools/release/runtime_license_audit.py",
    "tools/release/release-metadata.json",
)
STATIC_TREES = (
    "configs/installation/generated",
    "docs/runbooks",
    "migrations",
    "schemas",
    "src/trading/resources/artifacts/demo_bundle",
)
REQUIRED_CORRESPONDING_SOURCES = frozenset({"grafana", "loki", "openbao"})


class OfflineBundleError(RuntimeError):
    """Raised when release material is incomplete, mutable, or unverifiable."""


@dataclass(frozen=True)
class OciArchiveDescriptor:
    digest: str
    media_type: str
    platforms: tuple[str, ...]
    child_digests: Mapping[str, str]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    payload = _read_regular_file(path, max_bytes=max_bytes)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise OfflineBundleError(f"JSON root must be an object: {path}")
    return value


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise OfflineBundleError(f"required file is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise OfflineBundleError(f"path must be a regular file: {path}")
    if metadata.st_size > max_bytes:
        raise OfflineBundleError(f"file exceeds size limit: {path}")
    with path.open("rb") as stream:
        return stream.read(max_bytes + 1)


def _validate_relative_path(raw: str) -> PurePosixPath:
    relative = PurePosixPath(raw)
    if raw != relative.as_posix() or relative.is_absolute() or not relative.parts:
        raise OfflineBundleError(f"bundle path is not canonical: {raw!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise OfflineBundleError(f"bundle path escapes root: {raw!r}")
    return relative


def _safe_output(root: Path, relative: str) -> Path:
    normalized = _validate_relative_path(relative)
    output = root.joinpath(*normalized.parts)
    resolved_parent = output.parent.resolve()
    if root.resolve() != resolved_parent and root.resolve() not in resolved_parent.parents:
        raise OfflineBundleError(f"bundle output escapes root: {relative}")
    return output


def _atomic_write(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o755 if executable else 0o644)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _copy_regular(source: Path, destination: Path, *, executable: bool | None = None) -> None:
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OfflineBundleError(f"release input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    make_executable = bool(metadata.st_mode & stat.S_IXUSR) if executable is None else executable
    destination.chmod(0o755 if make_executable else 0o644)


def _copy_tree_files(source_root: Path, bundle_root: Path, relative_root: str) -> None:
    if not source_root.is_dir():
        raise OfflineBundleError(f"release input tree is missing: {source_root}")
    for source in sorted(source_root.rglob("*")):
        metadata = source.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OfflineBundleError(f"release input symlink is forbidden: {source}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise OfflineBundleError(f"release input must be regular: {source}")
        relative = f"{relative_root}/{source.relative_to(source_root).as_posix()}"
        _copy_regular(source, _safe_output(bundle_root, relative))


def _tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = _validate_relative_path(member.name)
        name = path.as_posix()
        if name in members:
            raise OfflineBundleError(f"OCI archive contains duplicate path: {name}")
        if not (member.isfile() or member.isdir()):
            raise OfflineBundleError(f"OCI archive contains non-regular path: {name}")
        members[name] = member
    return members


def _tar_json(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    name: str,
) -> dict[str, Any]:
    member = members.get(name)
    if member is None or not member.isfile() or member.size > MAX_JSON_BYTES:
        raise OfflineBundleError(f"OCI JSON payload is missing or invalid: {name}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise OfflineBundleError(f"OCI JSON payload cannot be read: {name}")
    payload = extracted.read(MAX_JSON_BYTES + 1)
    if len(payload) != member.size:
        raise OfflineBundleError(f"OCI JSON payload is truncated: {name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise OfflineBundleError(f"OCI JSON payload root is not an object: {name}")
    return value


def _descriptor_contract(
    descriptor: Mapping[str, Any],
    *,
    label: str,
    media_types: frozenset[str] | None = None,
) -> tuple[str, int, str]:
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    media_type = descriptor.get("mediaType")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise OfflineBundleError(f"{label} has an unsupported digest")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise OfflineBundleError(f"{label} has an invalid size")
    if not isinstance(media_type, str) or not media_type:
        raise OfflineBundleError(f"{label} has no media type")
    if media_types is not None and media_type not in media_types:
        raise OfflineBundleError(f"{label} has an unsupported media type: {media_type}")
    return digest, size, media_type


def _tar_blob_member(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    *,
    descriptor: Mapping[str, Any],
    label: str,
    media_types: frozenset[str] | None = None,
) -> tuple[tarfile.TarInfo, str]:
    digest, size, media_type = _descriptor_contract(
        descriptor,
        label=label,
        media_types=media_types,
    )
    name = f"blobs/sha256/{digest.removeprefix('sha256:')}"
    member = members.get(name)
    if member is None or not member.isfile():
        raise OfflineBundleError(f"{label} blob is missing: {digest}")
    if member.size != size:
        raise OfflineBundleError(
            f"{label} descriptor size mismatch: expected={size}, actual={member.size}"
        )
    return member, media_type


def _verify_tar_blob(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    *,
    descriptor: Mapping[str, Any],
    label: str,
    media_types: frozenset[str] | None = None,
) -> str:
    member, media_type = _tar_blob_member(
        archive,
        members,
        descriptor=descriptor,
        label=label,
        media_types=media_types,
    )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise OfflineBundleError(f"{label} blob cannot be read")
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    expected_digest = str(descriptor["digest"])
    if size != member.size or f"sha256:{digest.hexdigest()}" != expected_digest:
        raise OfflineBundleError(f"{label} blob digest mismatch: {expected_digest}")
    return media_type


def _tar_json_blob(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    *,
    descriptor: Mapping[str, Any],
    label: str,
    media_types: frozenset[str],
) -> tuple[str, dict[str, Any]]:
    member, media_type = _tar_blob_member(
        archive,
        members,
        descriptor=descriptor,
        label=label,
        media_types=media_types,
    )
    if member.size > MAX_JSON_BYTES:
        raise OfflineBundleError(f"{label} JSON blob exceeds the size limit")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise OfflineBundleError(f"{label} JSON blob cannot be read")
    payload = extracted.read(MAX_JSON_BYTES + 1)
    expected_digest = str(descriptor["digest"])
    if len(payload) != member.size or f"sha256:{_sha256_bytes(payload)}" != expected_digest:
        raise OfflineBundleError(f"{label} blob digest mismatch: {expected_digest}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise OfflineBundleError(f"{label} JSON root is not an object")
    return media_type, value


def inspect_oci_archive(path: Path) -> OciArchiveDescriptor:
    """Inspect one multi-platform OCI archive without extracting it."""

    if not path.is_file() or path.is_symlink():
        raise OfflineBundleError(f"OCI archive is not a regular file: {path}")
    with tarfile.open(path, mode="r:*") as archive:
        members = _tar_members(archive)
        layout = _tar_json(archive, members, "oci-layout")
        if layout.get("imageLayoutVersion") != "1.0.0":
            raise OfflineBundleError(f"unsupported OCI layout version: {path}")
        index = _tar_json(archive, members, "index.json")
        descriptors = index.get("manifests")
        if not isinstance(descriptors, list) or len(descriptors) != 1:
            raise OfflineBundleError("OCI archive must contain exactly one top-level descriptor")
        top = descriptors[0]
        if not isinstance(top, dict):
            raise OfflineBundleError("OCI top-level descriptor must be an object")
        digest, _, _ = _descriptor_contract(
            top,
            label="OCI top-level descriptor",
            media_types=OCI_INDEX_MEDIA_TYPES,
        )
        media_type, manifest = _tar_json_blob(
            archive,
            members,
            descriptor=top,
            label="OCI image index",
            media_types=OCI_INDEX_MEDIA_TYPES,
        )
        if manifest.get("schemaVersion") != 2:
            raise OfflineBundleError("OCI image index schemaVersion must be 2")
        children = manifest.get("manifests")
        if not isinstance(children, list) or not children:
            raise OfflineBundleError("OCI archive does not contain a multi-platform image index")
        platforms: dict[str, str] = {}
        for child in children:
            if not isinstance(child, dict):
                raise OfflineBundleError("OCI child descriptor must be an object")
            platform = child.get("platform")
            if not isinstance(platform, dict):
                continue
            operating_system = platform.get("os")
            architecture = platform.get("architecture")
            if not isinstance(operating_system, str) or not isinstance(architecture, str):
                continue
            if operating_system == "unknown" or architecture == "unknown":
                continue
            key = f"{operating_system}/{architecture}"
            if key not in SUPPORTED_PLATFORMS:
                continue
            child_digest, _, _ = _descriptor_contract(
                child,
                label=f"OCI child descriptor ({key})",
                media_types=OCI_MANIFEST_MEDIA_TYPES,
            )
            _, child_manifest = _tar_json_blob(
                archive,
                members,
                descriptor=child,
                label=f"OCI child manifest ({key})",
                media_types=OCI_MANIFEST_MEDIA_TYPES,
            )
            if child_manifest.get("schemaVersion") != 2:
                raise OfflineBundleError(f"OCI child manifest schemaVersion is invalid: {key}")
            config = child_manifest.get("config")
            layers = child_manifest.get("layers")
            if not isinstance(config, dict) or not isinstance(layers, list):
                raise OfflineBundleError(f"OCI child manifest is incomplete: {key}")
            _tar_json_blob(
                archive,
                members,
                descriptor=config,
                label=f"OCI image config ({key})",
                media_types=OCI_CONFIG_MEDIA_TYPES,
            )
            for position, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    raise OfflineBundleError(f"OCI layer descriptor is invalid: {key}")
                layer_media_type = _verify_tar_blob(
                    archive,
                    members,
                    descriptor=layer,
                    label=f"OCI image layer ({key}, {position})",
                )
                if not (
                    layer_media_type.startswith("application/vnd.oci.image.layer.")
                    or layer_media_type.startswith(
                        "application/vnd.docker.image.rootfs.diff."
                    )
                ):
                    raise OfflineBundleError(
                        f"OCI image layer has an unsupported media type: {layer_media_type}"
                    )
            if key in platforms:
                raise OfflineBundleError(f"OCI index contains duplicate platform: {key}")
            platforms[key] = child_digest
    missing = sorted(SUPPORTED_PLATFORMS - set(platforms))
    if missing:
        raise OfflineBundleError(f"OCI archive lacks required platforms: {missing}")
    return OciArchiveDescriptor(
        digest=digest,
        media_type=media_type,
        platforms=tuple(sorted(platforms)),
        child_digests=dict(sorted(platforms.items())),
    )


def _parse_bindings(values: Iterable[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path or name in result:
            raise OfflineBundleError(f"invalid or duplicate {label} binding: {raw!r}")
        if not name.replace("_", "-").replace(".", "-").isalnum():
            raise OfflineBundleError(f"invalid {label} name: {name!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def _normalize_ssh_public_key(payload: bytes) -> bytes:
    try:
        fields = payload.decode("utf-8").strip().split()
    except UnicodeDecodeError as error:
        raise OfflineBundleError("release public key is not UTF-8") from error
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise OfflineBundleError("release public key must be OpenSSH Ed25519")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except ValueError as error:
        raise OfflineBundleError("release public key payload is invalid") from error
    key_type = b"ssh-ed25519"
    prefix = len(key_type).to_bytes(4, "big") + key_type
    if not blob.startswith(prefix) or len(blob) != len(prefix) + 4 + 32:
        raise OfflineBundleError("release public key blob is invalid")
    key_length = int.from_bytes(blob[len(prefix) : len(prefix) + 4], "big")
    if key_length != 32:
        raise OfflineBundleError("release public key length is invalid")
    return f"ssh-ed25519 {fields[1]}\n".encode()


def _public_key_from_private(path: Path, *, ssh_keygen: str = "ssh-keygen") -> bytes:
    _read_regular_file(path, max_bytes=64 * 1024)
    result = subprocess.run(
        [ssh_keygen, "-y", "-f", str(path)],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise OfflineBundleError(
            f"release signing key is not usable OpenSSH Ed25519: "
            f"{(result.stderr or result.stdout).decode(errors='replace').strip()}"
        )
    return _normalize_ssh_public_key(result.stdout)


def _load_public_key(path: Path) -> bytes:
    return _normalize_ssh_public_key(_read_regular_file(path, max_bytes=64 * 1024))


def _key_id(public_key: bytes) -> str:
    blob = base64.b64decode(public_key.decode().split()[1], validate=True)
    return f"sha256:{_sha256_bytes(blob)}"


def _validate_release_manifest(release: Mapping[str, Any]) -> None:
    if release.get("schema") != "io.roehub.release/v1alpha1":
        raise OfflineBundleError("unsupported embedded release manifest schema")
    if set(release.get("supported_architectures", [])) != SUPPORTED_PLATFORMS:
        raise OfflineBundleError("release manifest architecture set is incomplete")
    images = release.get("images")
    if not isinstance(images, dict) or not images:
        raise OfflineBundleError("release manifest has no image inventory")
    for name, raw in images.items():
        if not isinstance(raw, dict):
            raise OfflineBundleError(f"release image record is invalid: {name}")
        reference = raw.get("reference")
        if not isinstance(reference, str) or "@sha256:" not in reference:
            raise OfflineBundleError(f"release image is not digest-pinned: {name}")
        if ":latest" in reference.lower() or reference.endswith(":main"):
            raise OfflineBundleError(f"release image reference is mutable: {name}")
        if set(raw.get("platforms", [])) != SUPPORTED_PLATFORMS:
            raise OfflineBundleError(f"release image lacks a required platform: {name}")


def _normalize_sbom(
    path: Path,
    *,
    name: str,
    version: str,
    digest: str,
    platform: str,
) -> None:
    payload = _load_json(path, max_bytes=256 * 1024 * 1024)
    creation = payload.get("creationInfo")
    if not isinstance(creation, dict):
        raise OfflineBundleError(f"Syft output lacks SPDX creationInfo: {name}")
    creation["created"] = SOURCE_DATE_EPOCH
    platform_slug = platform.replace("/", "-")
    payload["name"] = f"roehub-{name}-{version}-{platform_slug}"
    payload["documentNamespace"] = (
        f"https://sbom.roehub.io/releases/{version}/{name}/{platform_slug}/"
        f"{digest.removeprefix('sha256:')}"
    )
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise OfflineBundleError(f"Syft output lacks SPDX packages: {name}")
    roots = [
        package
        for package in packages
        if isinstance(package, dict)
        and str(package.get("SPDXID", "")).startswith("SPDXRef-DocumentRoot-Image-")
        and package.get("primaryPackagePurpose") == "CONTAINER"
    ]
    if len(roots) != 1:
        raise OfflineBundleError(
            f"Syft output has an invalid container document root: {name} ({platform})"
        )
    root = roots[0]
    old_root_id = str(root["SPDXID"])
    new_root_id = (
        f"SPDXRef-DocumentRoot-Image-{name}-{platform_slug}-"
        f"{digest.removeprefix('sha256:')}"
    )
    root["SPDXID"] = new_root_id
    root["name"] = f"roehub/{name}:{version}-{platform_slug}"
    root["externalRefs"] = [
        {
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceLocator": (
                f"pkg:oci/roehub/{name}@{digest}?arch={platform.split('/', 1)[1]}"
            ),
            "referenceType": "purl",
        }
    ]
    relationships = payload.get("relationships")
    if not isinstance(relationships, list):
        raise OfflineBundleError(f"Syft output lacks SPDX relationships: {name}")
    for relationship in relationships:
        if not isinstance(relationship, dict):
            raise OfflineBundleError(f"Syft output has an invalid relationship: {name}")
        for key in ("spdxElementId", "relatedSpdxElement"):
            if relationship.get(key) == old_root_id:
                relationship[key] = new_root_id
    _atomic_write(path, _json_bytes(payload))


def _generate_sbom(
    *,
    syft: str,
    archive: Path,
    output: Path,
    name: str,
    version: str,
    digest: str,
    platform: str,
    skopeo: str = "skopeo",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "SOURCE_DATE_EPOCH": "0",
            "SYFT_CHECK_FOR_APP_UPDATE": "false",
        }
    )
    operating_system, architecture = platform.split("/", 1)
    with tempfile.TemporaryDirectory(prefix="roehub-sbom-") as temporary:
        platform_archive = Path(temporary) / f"{operating_system}-{architecture}.oci.tar"
        copy = subprocess.run(
            [
                skopeo,
                "copy",
                "--override-os",
                operating_system,
                "--override-arch",
                architecture,
                f"oci-archive:{archive}",
                f"oci-archive:{platform_archive}:{name}",
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        if copy.returncode != 0:
            raise OfflineBundleError(
                f"Skopeo platform extraction failed for {name} ({platform}): "
                f"{(copy.stderr or copy.stdout).strip()}"
            )
        result = subprocess.run(
            [syft, f"oci-archive:{platform_archive}", "-o", f"spdx-json={output}"],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        if result.returncode != 0:
            raise OfflineBundleError(
                f"Syft failed for {name} ({platform}): "
                f"{(result.stderr or result.stdout).strip()}"
            )
    _normalize_sbom(
        output,
        name=name,
        version=version,
        digest=digest,
        platform=platform,
    )


def _asset_record(path: Path, *, bundle_root: Path) -> dict[str, Any]:
    relative = path.relative_to(bundle_root).as_posix()
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OfflineBundleError(f"bundle payload is not regular: {relative}")
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    return {
        "executable": bool(metadata.st_mode & stat.S_IXUSR),
        "media_type": media_type,
        "path": relative,
        "sha256": _sha256_path(path),
        "size_bytes": metadata.st_size,
    }


def _all_payload_files(bundle_root: Path) -> list[Path]:
    excluded = {MANIFEST_NAME, SIGNATURE_NAME}
    result: list[Path] = []
    for path in sorted(bundle_root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OfflineBundleError(f"bundle symlink is forbidden: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise OfflineBundleError(f"bundle non-regular path is forbidden: {path}")
        if path.relative_to(bundle_root).as_posix() not in excluded:
            result.append(path)
    return result


def _archive_link_target_is_safe(*, member: tarfile.TarInfo, root: str) -> bool:
    """Return whether an archive link resolves inside its declared source root."""

    target = PurePosixPath(member.linkname)
    if target.is_absolute():
        return False
    candidate = PurePosixPath(member.name).parent / target if member.issym() else target
    normalized: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                return False
            normalized.pop()
            continue
        normalized.append(part)
    return bool(normalized) and normalized[0] == root


def _verify_corresponding_source_archive(
    *,
    path: Path,
    record: Mapping[str, Any],
    name: str,
) -> None:
    actual_sha256 = _sha256_path(path)
    if (
        record.get("sha256") != actual_sha256
        or record.get("expected_sha256") != actual_sha256
    ):
        raise OfflineBundleError(f"corresponding source digest mismatch: {name}")
    root = str(record.get("archive_root", ""))
    license_files = record.get("license_files")
    if not root or not isinstance(license_files, list) or not license_files:
        raise OfflineBundleError(f"corresponding source metadata is incomplete: {name}")
    required = {f"{root}/{relative}" for relative in license_files}
    found: set[str] = set()
    count = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            count += 1
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise OfflineBundleError(f"corresponding source path is unsafe: {name}")
            if not relative.parts or relative.parts[0] != root:
                raise OfflineBundleError(f"corresponding source root mismatch: {name}")
            if (member.issym() or member.islnk()) and not _archive_link_target_is_safe(
                member=member,
                root=root,
            ):
                raise OfflineBundleError(
                    f"corresponding source link target is unsafe: {name}"
                )
            if member.name in required and member.isfile() and member.size > 0:
                found.add(member.name)
    if count == 0 or found != required:
        raise OfflineBundleError(f"corresponding source licenses are incomplete: {name}")


def create_bundle(
    *,
    output: Path,
    release_manifest: Path,
    signing_key: Path,
    wheel: Path,
    image_inputs: Mapping[str, Path],
    source_inputs: Mapping[str, Path],
    source_metadata: Mapping[str, Mapping[str, Any]],
    provenance: Path,
    syft: str = "syft",
    repo_root: Path = ROOT,
    sbom_generator: Callable[..., None] = _generate_sbom,
) -> dict[str, Any]:
    """Create a complete signed bundle in an empty destination."""

    output = output.expanduser().resolve()
    if output.exists():
        if not output.is_dir() or output.is_symlink() or any(output.iterdir()):
            raise OfflineBundleError(f"bundle destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o755)

    release = _load_json(release_manifest)
    _validate_release_manifest(release)
    release_images = release["images"]
    if set(image_inputs) != set(release_images):
        missing = sorted(set(release_images) - set(image_inputs))
        extra = sorted(set(image_inputs) - set(release_images))
        raise OfflineBundleError(f"image input mismatch; missing={missing}, extra={extra}")
    if set(source_inputs) != REQUIRED_CORRESPONDING_SOURCES:
        missing = sorted(REQUIRED_CORRESPONDING_SOURCES - set(source_inputs))
        extra = sorted(set(source_inputs) - REQUIRED_CORRESPONDING_SOURCES)
        raise OfflineBundleError(
            f"corresponding source input mismatch; missing={missing}, extra={extra}"
        )
    if set(source_metadata) != REQUIRED_CORRESPONDING_SOURCES:
        missing = sorted(REQUIRED_CORRESPONDING_SOURCES - set(source_metadata))
        extra = sorted(set(source_metadata) - REQUIRED_CORRESPONDING_SOURCES)
        raise OfflineBundleError(
            f"corresponding source metadata mismatch; missing={missing}, extra={extra}"
        )

    for relative in STATIC_FILES:
        _copy_regular(repo_root / relative, _safe_output(output, relative))
    _copy_regular(
        release_manifest,
        _safe_output(output, "tools/release/release-metadata.json"),
        executable=False,
    )
    for relative in STATIC_TREES:
        _copy_tree_files(repo_root / relative, output, relative)
    _copy_regular(wheel, _safe_output(output, f"packages/{wheel.name}"))
    _copy_regular(provenance, _safe_output(output, "provenance/reproducibility.json"))

    public_key = _public_key_from_private(signing_key)
    _atomic_write(_safe_output(output, "trust/release-signing-key.pub"), public_key)

    image_records: dict[str, dict[str, Any]] = {}
    archives_by_digest: dict[str, str] = {}
    sboms_by_digest: dict[str, dict[str, str]] = {}
    for name in sorted(image_inputs):
        descriptor = inspect_oci_archive(image_inputs[name])
        expected_digest = release_images[name]["reference"].rsplit("@", 1)[1]
        if descriptor.digest != expected_digest:
            raise OfflineBundleError(
                f"image digest mismatch for {name}: expected={expected_digest}, "
                f"actual={descriptor.digest}"
            )
        if not SUPPORTED_PLATFORMS.issubset(descriptor.platforms):
            raise OfflineBundleError(f"image platform mismatch for {name}")
        archive_relative = archives_by_digest.get(descriptor.digest)
        sbom_relatives = sboms_by_digest.get(descriptor.digest)
        if archive_relative is None or sbom_relatives is None:
            stem = descriptor.digest.replace(":", "-")
            archive_relative = f"images/{stem}.oci.tar"
            copied_archive = _safe_output(output, archive_relative)
            _copy_regular(image_inputs[name], copied_archive, executable=False)
            sbom_relatives = {}
            for platform in sorted(SUPPORTED_PLATFORMS):
                platform_slug = platform.replace("/", "-")
                sbom_relative = f"sbom/images/{stem}.{platform_slug}.spdx.json"
                sbom_path = _safe_output(output, sbom_relative)
                sbom_generator(
                    syft=syft,
                    archive=copied_archive,
                    output=sbom_path,
                    name=name,
                    version=str(release["version"]),
                    digest=descriptor.digest,
                    platform=platform,
                )
                sbom_relatives[platform] = sbom_relative
            archives_by_digest[descriptor.digest] = archive_relative
            sboms_by_digest[descriptor.digest] = sbom_relatives
        image_records[name] = {
            "archive": archive_relative,
            "child_digests": dict(descriptor.child_digests),
            "index_digest": descriptor.digest,
            "media_type": descriptor.media_type,
            "online_reference": release_images[name]["reference"],
            "platforms": list(descriptor.platforms),
            "sboms": dict(sorted(sbom_relatives.items())),
        }

    license_audit_relative = "sbom/runtime-license-audit.json"
    try:
        license_audit = write_runtime_license_audit(
            output=_safe_output(output, license_audit_relative),
            bundle_root=output,
            image_records=image_records,
            policy_path=repo_root / "tools/release/runtime-license-policy.json",
        )
    except RuntimeLicenseAuditError as error:
        raise OfflineBundleError(f"runtime license audit failed: {error}") from error

    source_records: dict[str, dict[str, Any]] = {}
    for name, source in sorted(source_inputs.items()):
        metadata = source_metadata[name]
        required_fields = {
            "archive_root",
            "expected_sha256",
            "filename",
            "image",
            "license_files",
            "revision",
            "tag",
            "url",
        }
        if set(metadata) != required_fields:
            raise OfflineBundleError(f"corresponding source metadata is invalid: {name}")
        filename = metadata["filename"]
        if not isinstance(filename, str) or filename != source.name:
            raise OfflineBundleError(f"corresponding source filename mismatch: {name}")
        relative = f"sources/{source.name}"
        destination = _safe_output(output, relative)
        _copy_regular(source, destination, executable=False)
        actual_sha256 = _sha256_path(destination)
        if actual_sha256 != metadata["expected_sha256"]:
            raise OfflineBundleError(f"corresponding source digest mismatch: {name}")
        image_name = metadata["image"]
        if not isinstance(image_name, str) or image_name not in image_records:
            raise OfflineBundleError(f"corresponding source image binding is invalid: {name}")
        source_records[name] = {
            "archive_root": metadata["archive_root"],
            "expected_sha256": metadata["expected_sha256"],
            "image": image_name,
            "image_index_digest": image_records[image_name]["index_digest"],
            "license_files": metadata["license_files"],
            "path": relative,
            "revision": metadata["revision"],
            "sha256": actual_sha256,
            "size_bytes": destination.stat().st_size,
            "tag": metadata["tag"],
            "url": metadata["url"],
        }

    manifest: dict[str, Any] = {
        "assets": [],
        "compatibility": {
            "bundle_schema": SCHEMA_ID,
            "minimum_docker_api": "1.44",
            "release_manifest_schema": release["schema"],
        },
        "corresponding_sources": source_records,
        "images": image_records,
        "license_audit": {
            "path": license_audit_relative,
            "policy": license_audit["policy"],
            "policy_sha256": license_audit["policy_sha256"],
            "raw_noassertion_count": license_audit["raw_noassertion_count"],
            "schema": license_audit["schema"],
            "status": license_audit["status"],
            "unresolved_count": license_audit["unresolved_count"],
        },
        "network_policy": {
            "catalog_requests": "disabled",
            "default_egress": "denied",
            "telemetry": "disabled",
            "update_checks": "explicit_opt_in",
        },
        "product": "roehub",
        "release": {
            "manifest": "tools/release/release-metadata.json",
            "version": release["version"],
        },
        "schema": SCHEMA_ID,
        "signing": {
            "algorithm": "SSHSIG-Ed25519",
            "key_id": _key_id(public_key),
            "public_key": "trust/release-signing-key.pub",
        },
    }
    manifest["assets"] = [
        _asset_record(path, bundle_root=output) for path in _all_payload_files(output)
    ]
    manifest_payload = _json_bytes(manifest)
    _atomic_write(output / MANIFEST_NAME, manifest_payload)
    signature_path = output / f"{MANIFEST_NAME}.sig"
    signed = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(signing_key),
            "-n",
            SIGNATURE_NAMESPACE,
            str(output / MANIFEST_NAME),
        ],
        check=False,
        capture_output=True,
    )
    if signed.returncode != 0:
        raise OfflineBundleError(
            f"SSHSIG signing failed: "
            f"{(signed.stderr or signed.stdout).decode(errors='replace').strip()}"
        )
    signature = _read_regular_file(signature_path, max_bytes=1024 * 1024)
    signature_path.unlink()
    signature_record = {
        "algorithm": "SSHSIG-Ed25519",
        "key_id": _key_id(public_key),
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "schema": SIGNATURE_SCHEMA_ID,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    _atomic_write(output / SIGNATURE_NAME, _json_bytes(signature_record))
    verify_bundle(bundle=output, trusted_public_key=output / "trust/release-signing-key.pub")
    return manifest


def _schema_error(path: str, detail: str) -> OfflineBundleError:
    return OfflineBundleError(
        f"offline release manifest schema validation failed at {path}: {detail}"
    )


def _schema_reference(schema_root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise OfflineBundleError(f"unsupported JSON Schema reference: {reference}")
    current: Any = schema_root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise OfflineBundleError(f"unresolved JSON Schema reference: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise OfflineBundleError(f"JSON Schema reference is not an object: {reference}")
    return current


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    schema_root: Mapping[str, Any],
    path: str,
) -> None:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        _validate_schema_value(
            value,
            _schema_reference(schema_root, reference),
            schema_root=schema_root,
            path=path,
        )
        return
    for nested in schema.get("allOf", []):
        if not isinstance(nested, dict):
            raise OfflineBundleError("JSON Schema allOf entry must be an object")
        _validate_schema_value(value, nested, schema_root=schema_root, path=path)
    if "const" in schema and value != schema["const"]:
        raise _schema_error(path, f"expected constant {schema['const']!r}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise _schema_error(path, f"value is not in {enum!r}")

    expected_type = schema.get("type")
    type_matches = {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }
    if isinstance(expected_type, str) and not type_matches.get(expected_type, False):
        raise _schema_error(path, f"expected {expected_type}")

    if isinstance(value, dict):
        minimum_properties = schema.get("minProperties")
        if isinstance(minimum_properties, int) and len(value) < minimum_properties:
            raise _schema_error(path, f"requires at least {minimum_properties} properties")
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [item for item in required if item not in value]
            if missing:
                raise _schema_error(path, f"missing required properties {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise OfflineBundleError("JSON Schema properties must be an object")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child_path = f"{path}.{key}"
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate_schema_value(
                    item,
                    child_schema,
                    schema_root=schema_root,
                    path=child_path,
                )
            elif additional is False:
                raise _schema_error(child_path, "additional property is forbidden")
            elif isinstance(additional, dict):
                _validate_schema_value(
                    item,
                    additional,
                    schema_root=schema_root,
                    path=child_path,
                )

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            raise _schema_error(path, f"requires at least {minimum_items} items")
        if schema.get("uniqueItems") is True:
            normalized = [json.dumps(item, sort_keys=True) for item in value]
            if len(normalized) != len(set(normalized)):
                raise _schema_error(path, "items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    item_schema,
                    schema_root=schema_root,
                    path=f"{path}[{index}]",
                )
        contains = schema.get("contains")
        if isinstance(contains, dict):
            for index, item in enumerate(value):
                try:
                    _validate_schema_value(
                        item,
                        contains,
                        schema_root=schema_root,
                        path=f"{path}[{index}]",
                    )
                except OfflineBundleError:
                    continue
                break
            else:
                raise _schema_error(path, "does not contain a required item")

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise _schema_error(path, f"minimum length is {minimum_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise _schema_error(path, f"does not match pattern {pattern!r}")

    minimum = schema.get("minimum")
    if isinstance(minimum, int) and isinstance(value, int) and value < minimum:
        raise _schema_error(path, f"minimum value is {minimum}")


def _validate_manifest_schema(manifest: Mapping[str, Any], schema_path: Path) -> None:
    schema = _load_json(schema_path)
    _validate_schema_value(manifest, schema, schema_root=schema, path="$")


def verify_bundle(
    *,
    bundle: Path,
    trusted_public_key: Path,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Verify trust identity, detached signature, exact files, OCI digests, and policy."""

    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir() or bundle.is_symlink():
        raise OfflineBundleError(f"bundle root is not a directory: {bundle}")
    manifest_payload = _read_regular_file(bundle / MANIFEST_NAME, max_bytes=MAX_JSON_BYTES)
    signature_record = _load_json(bundle / SIGNATURE_NAME)
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict):
        raise OfflineBundleError("offline release manifest root must be an object")
    _validate_manifest_schema(manifest, schema_path)

    verifier = _load_public_key(trusted_public_key)
    expected_key_id = _key_id(verifier)
    if (
        signature_record.get("schema") != SIGNATURE_SCHEMA_ID
        or signature_record.get("algorithm") != "SSHSIG-Ed25519"
        or signature_record.get("key_id") != expected_key_id
        or manifest.get("signing", {}).get("key_id") != expected_key_id
        or signature_record.get("manifest_sha256") != _sha256_bytes(manifest_payload)
    ):
        raise OfflineBundleError("offline release signature identity mismatch")
    embedded_public_key = _load_public_key(
        _safe_output(bundle, str(manifest["signing"]["public_key"]))
    )
    if embedded_public_key != verifier:
        raise OfflineBundleError("embedded release key differs from the trusted public key")
    try:
        signature = base64.b64decode(signature_record["signature_base64"], validate=True)
    except (KeyError, ValueError) as error:
        raise OfflineBundleError("offline release signature is invalid") from error
    with tempfile.TemporaryDirectory(prefix="roehub-signature-verify-") as temporary:
        temporary_root = Path(temporary)
        allowed_signers = temporary_root / "allowed_signers"
        signature_path = temporary_root / "manifest.sig"
        allowed_signers.write_bytes(SIGNER_IDENTITY.encode() + b" " + verifier)
        signature_path.write_bytes(signature)
        verified = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                SIGNER_IDENTITY,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=manifest_payload,
            check=False,
            capture_output=True,
        )
        if verified.returncode != 0:
            raise OfflineBundleError(
                f"offline release signature is invalid: "
                f"{(verified.stderr or verified.stdout).decode(errors='replace').strip()}"
            )

    records = manifest.get("assets")
    if not isinstance(records, list):
        raise OfflineBundleError("offline release asset inventory is missing")
    expected_files = {MANIFEST_NAME, SIGNATURE_NAME}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise OfflineBundleError("offline release asset record is invalid")
        relative = str(record.get("path", ""))
        _validate_relative_path(relative)
        if relative in seen or relative in expected_files:
            raise OfflineBundleError(f"duplicate or reserved asset path: {relative}")
        seen.add(relative)
        expected_files.add(relative)
        path = _safe_output(bundle, relative)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OfflineBundleError(f"bundle asset is not a regular file: {relative}")
        if metadata.st_size != record.get("size_bytes"):
            raise OfflineBundleError(f"bundle asset size mismatch: {relative}")
        if _sha256_path(path) != record.get("sha256"):
            raise OfflineBundleError(f"bundle asset digest mismatch: {relative}")
        if bool(metadata.st_mode & stat.S_IXUSR) != record.get("executable"):
            raise OfflineBundleError(f"bundle asset mode mismatch: {relative}")

    actual_files: set[str] = set()
    for path in bundle.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OfflineBundleError(f"bundle symlink is forbidden: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise OfflineBundleError(f"bundle non-regular path is forbidden: {path}")
        actual_files.add(path.relative_to(bundle).as_posix())
    if actual_files != expected_files:
        raise OfflineBundleError(
            f"bundle file inventory mismatch; missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    release_path = _safe_output(bundle, str(manifest["release"]["manifest"]))
    release = _load_json(release_path)
    _validate_release_manifest(release)
    if release.get("version") != manifest["release"]["version"]:
        raise OfflineBundleError("embedded release version does not match bundle")
    for name, record in manifest["images"].items():
        descriptor = inspect_oci_archive(_safe_output(bundle, record["archive"]))
        expected = release["images"].get(name)
        if not isinstance(expected, dict):
            raise OfflineBundleError(f"bundle image is absent from release manifest: {name}")
        if (
            descriptor.digest != record["index_digest"]
            or descriptor.digest != expected["reference"].rsplit("@", 1)[1]
            or dict(descriptor.child_digests) != record["child_digests"]
            or not SUPPORTED_PLATFORMS.issubset(descriptor.platforms)
        ):
            raise OfflineBundleError(f"bundle OCI contract mismatch: {name}")
        if set(record["sboms"]) != SUPPORTED_PLATFORMS:
            raise OfflineBundleError(f"bundle image SBOM platform set is incomplete: {name}")
        for platform, relative in record["sboms"].items():
            sbom = _load_json(_safe_output(bundle, relative), max_bytes=256 * 1024 * 1024)
            if sbom.get("spdxVersion") != "SPDX-2.3" or not sbom.get("packages"):
                raise OfflineBundleError(
                    f"bundle image SBOM is missing or incomplete: {name} ({platform})"
                )

    license_audit_record = manifest.get("license_audit")
    if not isinstance(license_audit_record, dict):
        raise OfflineBundleError("runtime license audit manifest record is missing")
    try:
        license_audit = verify_runtime_license_audit(
            audit_path=_safe_output(bundle, str(license_audit_record.get("path", ""))),
            bundle_root=bundle,
            image_records=manifest["images"],
            policy_path=_safe_output(bundle, str(license_audit_record.get("policy", ""))),
        )
    except RuntimeLicenseAuditError as error:
        raise OfflineBundleError(f"runtime license audit failed: {error}") from error
    expected_license_audit_record = {
        "path": str(license_audit_record["path"]),
        "policy": license_audit["policy"],
        "policy_sha256": license_audit["policy_sha256"],
        "raw_noassertion_count": license_audit["raw_noassertion_count"],
        "schema": license_audit["schema"],
        "status": license_audit["status"],
        "unresolved_count": license_audit["unresolved_count"],
    }
    if license_audit_record != expected_license_audit_record:
        raise OfflineBundleError("runtime license audit manifest record does not match")

    if set(manifest["corresponding_sources"]) != REQUIRED_CORRESPONDING_SOURCES:
        raise OfflineBundleError("required corresponding source archives are incomplete")
    for name, source_record in manifest["corresponding_sources"].items():
        if not isinstance(source_record, dict):
            raise OfflineBundleError(f"corresponding source record is invalid: {name}")
        image_name = source_record.get("image")
        image_record = manifest["images"].get(image_name)
        if (
            not isinstance(image_record, dict)
            or source_record.get("image_index_digest")
            != image_record.get("index_digest")
        ):
            raise OfflineBundleError(f"corresponding source image binding is invalid: {name}")
        _verify_corresponding_source_archive(
            path=_safe_output(bundle, str(source_record["path"])),
            record=source_record,
            name=name,
        )
    network_policy = manifest.get("network_policy", {})
    if network_policy != {
        "catalog_requests": "disabled",
        "default_egress": "denied",
        "telemetry": "disabled",
        "update_checks": "explicit_opt_in",
    }:
        raise OfflineBundleError("offline release network policy is unsafe")
    installation = (bundle / "configs/installation/roehub.yaml").read_text(encoding="utf-8")
    if "update_checks:\n  enabled: false" not in installation:
        raise OfflineBundleError("default installation does not disable update checks")
    return {
        "asset_count": len(records),
        "bundle_version": manifest["release"]["version"],
        "image_count": len(manifest["images"]),
        "key_id": expected_key_id,
        "runtime_license_audit": {
            "raw_noassertion_count": license_audit["raw_noassertion_count"],
            "status": license_audit["status"],
            "unresolved_count": license_audit["unresolved_count"],
        },
        "signature_verified": True,
    }


def _run_command(command: Sequence[str], *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise OfflineBundleError(f"{label} failed: {detail}")
    return result


def _api_version_tuple(value: str, *, label: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value.strip())
    if match is None:
        raise OfflineBundleError(f"{label} is not a valid Docker API version: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _host_platform() -> tuple[str, str]:
    operating_system = sys.platform
    if operating_system.startswith("linux"):
        operating_system = "linux"
    elif operating_system == "darwin":
        operating_system = "linux"
    else:
        raise OfflineBundleError(f"unsupported Docker host operating system: {sys.platform}")
    machine = os.uname().machine.lower()
    architectures = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64"}
    architecture = architectures.get(machine)
    if architecture is None:
        raise OfflineBundleError(f"unsupported Docker host architecture: {machine}")
    return operating_system, architecture


def install_bundle(
    *,
    bundle: Path,
    trusted_public_key: Path,
    state_directory: Path,
    profile: str,
    skopeo: str = "skopeo",
    docker: str = "docker",
    runtime_smoke: bool = False,
) -> dict[str, Any]:
    """Verify, import one host platform, and emit an immutable Compose override."""

    verification = verify_bundle(bundle=bundle, trusted_public_key=trusted_public_key)
    bundle = bundle.expanduser().resolve()
    manifest = _load_json(bundle / MANIFEST_NAME)
    release = _load_json(bundle / manifest["release"]["manifest"])
    if profile not in {"base", "trading", "ml"}:
        raise OfflineBundleError(f"unsupported offline profile: {profile}")
    compose_path = bundle / f"configs/installation/generated/{profile}/compose.yaml"
    if not compose_path.is_file():
        raise OfflineBundleError(f"offline profile Compose is missing: {profile}")
    state_directory = state_directory.expanduser().resolve()
    if state_directory.exists():
        if not state_directory.is_dir() or state_directory.is_symlink():
            raise OfflineBundleError("offline installation state is not a directory")
    else:
        state_directory.mkdir(parents=True, mode=0o750)

    docker_api = _run_command(
        [docker, "version", "--format", "{{.Server.APIVersion}}"],
        label="Docker",
    ).stdout.strip()
    minimum_api = str(manifest["compatibility"]["minimum_docker_api"])
    if _api_version_tuple(docker_api, label="Docker Server API version") < _api_version_tuple(
        minimum_api,
        label="signed minimum Docker API version",
    ):
        raise OfflineBundleError(
            f"Docker Server API {docker_api} is below signed minimum {minimum_api}"
        )
    operating_system, architecture = _host_platform()
    platform = f"{operating_system}/{architecture}"
    imported_by_digest: dict[str, str] = {}
    imported_tags: list[str] = []
    image_ids: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="roehub-offline-import-") as temporary:
        temporary_root = Path(temporary)
        for name, record in sorted(manifest["images"].items()):
            digest = record["index_digest"]
            image_id = imported_by_digest.get(digest)
            if image_id is None:
                tag = (
                    f"roehub-offline/{name}:{release['version']}-"
                    f"{digest.removeprefix('sha256:')[:16]}"
                )
                archive = bundle / record["archive"]
                docker_archive = temporary_root / f"{name}.docker.tar"
                _run_command(
                    [
                        skopeo,
                        "copy",
                        "--override-os",
                        operating_system,
                        "--override-arch",
                        architecture,
                        f"oci-archive:{archive}",
                        f"docker-archive:{docker_archive}:{tag}",
                    ],
                    label=f"offline image conversion {name}",
                )
                _run_command(
                    [docker, "image", "load", "--input", str(docker_archive)],
                    label=f"offline image load {name}",
                )
                docker_archive.unlink(missing_ok=True)
                inspected = _run_command(
                    [docker, "image", "inspect", "--format", "{{.Id}}", tag],
                    label=f"offline image inspect {name}",
                )
                image_id = inspected.stdout.strip()
                if not image_id.startswith("sha256:") or len(image_id) != 71:
                    raise OfflineBundleError(
                        f"Docker returned invalid immutable image ID: {name}"
                    )
                imported_by_digest[digest] = image_id
                imported_tags.append(tag)
            image_ids[name] = image_id

    reference_to_id = {
        release["images"][name]["reference"]: image_id for name, image_id in image_ids.items()
    }
    rendered = _run_command(
        [docker, "compose", "-f", str(compose_path), "config", "--format", "json"],
        label="offline Compose rendering",
    )
    try:
        compose = json.loads(rendered.stdout)
    except json.JSONDecodeError as error:
        raise OfflineBundleError("Docker Compose returned invalid JSON") from error
    if not isinstance(compose, dict):
        raise OfflineBundleError("Docker Compose JSON root must be an object")
    override_services: dict[str, dict[str, str]] = {}
    for service_name, service in sorted(compose.get("services", {}).items()):
        if not isinstance(service, dict) or not isinstance(service.get("image"), str):
            raise OfflineBundleError(f"Compose service lacks a release image: {service_name}")
        image_id = reference_to_id.get(service["image"])
        if image_id is None:
            raise OfflineBundleError(
                f"Compose service image is absent from signed release: {service_name}"
            )
        override_services[service_name] = {"image": image_id, "pull_policy": "never"}
    override = {
        "name": f"roehub-offline-{profile}",
        "services": override_services,
    }
    override_path = state_directory / f"compose.{profile}.offline.yaml"
    _atomic_write(override_path, _json_bytes(override))
    lock = {
        "bundle_manifest_sha256": _sha256_path(bundle / MANIFEST_NAME),
        "images": dict(sorted(image_ids.items())),
        "platform": platform,
        "profile": profile,
        "server_docker_api": docker_api,
        "schema": "io.roehub.offline-image-lock/v1alpha1",
        "version": release["version"],
    }
    lock_path = state_directory / "offline-image-lock.json"
    _atomic_write(lock_path, _json_bytes(lock))
    _run_command(
        [
            docker,
            "compose",
            "-f",
            str(compose_path),
            "-f",
            str(override_path),
            "config",
            "--quiet",
        ],
        label="offline Compose validation",
    )
    if runtime_smoke:
        _run_command(
            [
                docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m",
                "--env",
                "NUMBA_CACHE_DIR=/tmp/roehub-numba-cache",
                "--entrypoint",
                "python",
                image_ids["runtime"],
                "-c",
                (
                    "import apps.roehubctl.main.main; "
                    "import trading.platform.config.installation; "
                    "print('roehub-offline-runtime-ok')"
                ),
            ],
            label="air-gapped Roehub runtime smoke",
        )
    return {
        **verification,
        "compose_override": str(override_path),
        "image_lock": str(lock_path),
        "imported_tags": imported_tags,
        "imported_unique_images": len(imported_by_digest),
        "platform": platform,
        "profile": profile,
        "runtime_smoke": "passed" if runtime_smoke else "not-requested",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="assemble and sign a complete bundle")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    create.add_argument("--signing-key", type=Path, required=True)
    create.add_argument("--wheel", type=Path, required=True)
    create.add_argument("--image", action="append", default=[], metavar="NAME=OCI_ARCHIVE")
    create.add_argument("--source", action="append", default=[], metavar="NAME=ARCHIVE")
    create.add_argument("--source-metadata", type=Path, required=True)
    create.add_argument("--provenance", type=Path, required=True)
    create.add_argument("--syft", default="syft")
    verify = subparsers.add_parser("verify", help="verify before any activation")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--trusted-public-key", type=Path, required=True)
    verify.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    install = subparsers.add_parser(
        "install",
        help="verify, import host-platform images, and emit an immutable Compose override",
    )
    install.add_argument("--bundle", type=Path, required=True)
    install.add_argument("--trusted-public-key", type=Path, required=True)
    install.add_argument("--state-directory", type=Path, required=True)
    install.add_argument("--profile", choices=("base", "trading", "ml"), default="base")
    install.add_argument("--skopeo", default="skopeo")
    install.add_argument("--docker", default="docker")
    install.add_argument("--runtime-smoke", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "create":
            source_metadata = _load_json(args.source_metadata)
            manifest = create_bundle(
                output=args.output,
                release_manifest=args.release_manifest,
                signing_key=args.signing_key,
                wheel=args.wheel,
                image_inputs=_parse_bindings(args.image, label="image"),
                source_inputs=_parse_bindings(args.source, label="source"),
                source_metadata=source_metadata,
                provenance=args.provenance,
                syft=args.syft,
            )
            result = {
                "bundle": str(args.output.expanduser().resolve()),
                "images": len(manifest["images"]),
                "schema": manifest["schema"],
                "status": "passed",
            }
        elif args.command == "verify":
            result = verify_bundle(
                bundle=args.bundle,
                trusted_public_key=args.trusted_public_key,
                schema_path=args.schema,
            )
            result["status"] = "passed"
        else:
            result = install_bundle(
                bundle=args.bundle,
                trusted_public_key=args.trusted_public_key,
                state_directory=args.state_directory,
                profile=args.profile,
                skopeo=args.skopeo,
                docker=args.docker,
                runtime_smoke=args.runtime_smoke,
            )
            result["status"] = "passed"
    except (OfflineBundleError, OSError, json.JSONDecodeError) as error:
        print(f"offline release bundle failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
