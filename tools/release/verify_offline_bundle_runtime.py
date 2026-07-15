#!/usr/bin/env python3
"""Prove the signed Roehub offline release at the Docker/runtime boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from tools.release.offline_bundle import (
    MANIFEST_NAME,
    OfflineBundleError,
    _archive_link_target_is_safe,
    create_bundle,
    inspect_oci_archive,
    verify_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = ROOT / "tools/release/release-metadata.json"
DEFAULT_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/22-signed-offline-release-bundle-runtime-proof.json"
)
CACHE_ROOT = Path.home() / ".cache/roehub/stage22-offline-release"
SOURCE_RECORDS: dict[str, dict[str, Any]] = {
    "grafana": {
        "archive_root": "grafana-12.0.2",
        "expected_sha256": "8524498289e7d1900626ea7c0763fd923cf7bd1effa48cda476e63b299acfe2d",
        "filename": "grafana-v12.0.2.tar.gz",
        "image": "grafana",
        "license_files": ["LICENSE", "LICENSING.md", "NOTICE.md"],
        "revision": "5bda17e7c1cb313eb96266f2fdda73a6b35c3977",
        "tag": "v12.0.2",
        "url": "https://github.com/grafana/grafana/archive/refs/tags/v12.0.2.tar.gz",
    },
    "loki": {
        "archive_root": "loki-3.5.1",
        "expected_sha256": "d360561de7ac97d05a6fc1dc0ca73d93c11a86234783dfd9ae92033300caabd7",
        "filename": "loki-v3.5.1.tar.gz",
        "image": "loki",
        "license_files": ["LICENSE", "LICENSING.md"],
        "revision": "d4e637cebb842a933b21f0753c028821b1ad5c26",
        "tag": "v3.5.1",
        "url": "https://github.com/grafana/loki/archive/refs/tags/v3.5.1.tar.gz",
    },
    "openbao": {
        "archive_root": "openbao-2.5.4-roehub-licensed-qr.1",
        "expected_sha256": "e1cc071b4666312de84e4bdf32e7e25be04f95738a7dbc5adff4c357c3a24f07",
        "filename": "openbao-v2.5.4-roehub-licensed-qr.1.tar.gz",
        "image": "openbao",
        "license_files": [
            "LICENSE",
            "roehub/openbao-2.5.4-licensed-qr.NOTICE",
            "roehub/openbao-2.5.4-licensed-qr.patch",
            "third_party/skip2-go-qrcode/LICENSE",
        ],
        "revision": "4f6d47246a053375271a5fd8af85c3b75695aa46+roehub-licensed-qr.1",
        "tag": "2.5.4-roehub-licensed-qr.1",
        "url": (
            "https://github.com/Dejetins/roehub.com/releases/download/v0.1.0/"
            "openbao-v2.5.4-roehub-licensed-qr.1.tar.gz"
        ),
    },
}
REGISTRY_INDEX_COPY_FORMATS = {
    "application/vnd.docker.distribution.manifest.list.v2+json": "v2s2",
    "application/vnd.oci.image.index.v1+json": "oci",
}


class RuntimeProofError(RuntimeError):
    """Raised when the local artifact cannot satisfy the Stage 22 runtime boundary."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeProofError(f"reproducibility tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeProofError(f"reproducibility tree contains a special file: {path}")
        hashes[path.relative_to(root).as_posix()] = _sha256(path)
    return hashes


def _directory_digest(hashes: dict[str, str]) -> str:
    payload = "".join(f"{path}\0{digest}\n" for path, digest in sorted(hashes.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _oci_json_blob(
    archive: tarfile.TarFile,
    digest: str,
    *,
    label: str,
) -> dict[str, Any]:
    if not digest.startswith("sha256:"):
        raise RuntimeProofError(f"{label} digest is invalid")
    member_name = f"blobs/sha256/{digest.removeprefix('sha256:')}"
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise RuntimeProofError(f"{label} blob is missing") from error
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeProofError(f"{label} blob is unreadable")
    payload = stream.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise RuntimeProofError(f"{label} blob is too large")
    if f"sha256:{hashlib.sha256(payload).hexdigest()}" != digest:
        raise RuntimeProofError(f"{label} blob digest mismatch")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeProofError(f"{label} blob is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeProofError(f"{label} blob root is not an object")
    return value


def _oci_platform_layers(
    archive: tarfile.TarFile,
    *,
    platform: str,
) -> list[str]:
    index_stream = archive.extractfile("index.json")
    if index_stream is None:
        raise RuntimeProofError("OCI layout index is unreadable")
    try:
        layout = json.load(index_stream)
    except json.JSONDecodeError as error:
        raise RuntimeProofError("OCI layout index is invalid") from error
    roots = layout.get("manifests") if isinstance(layout, dict) else None
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], dict):
        raise RuntimeProofError("OCI layout must contain one root descriptor")
    image_index = _oci_json_blob(
        archive,
        str(roots[0].get("digest", "")),
        label="OCI image index",
    )
    operating_system, architecture = platform.split("/", 1)
    matches = []
    for descriptor in image_index.get("manifests", []):
        if not isinstance(descriptor, dict):
            continue
        descriptor_platform = descriptor.get("platform")
        if (
            isinstance(descriptor_platform, dict)
            and descriptor_platform.get("os") == operating_system
            and descriptor_platform.get("architecture") == architecture
        ):
            matches.append(descriptor)
    if len(matches) != 1:
        raise RuntimeProofError(f"OCI platform descriptor is not unique: {platform}")
    image_manifest = _oci_json_blob(
        archive,
        str(matches[0].get("digest", "")),
        label=f"OCI image manifest {platform}",
    )
    layers = image_manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeProofError(f"OCI image layers are missing: {platform}")
    result: list[str] = []
    for layer in layers:
        digest = str(layer.get("digest", "")) if isinstance(layer, dict) else ""
        if not digest.startswith("sha256:"):
            raise RuntimeProofError(f"OCI layer digest is invalid: {platform}")
        result.append(digest)
    return result


def _layer_evidence_hashes(
    *,
    archive_path: Path,
    platform: str,
    expected: Mapping[str, str],
) -> dict[str, str]:
    targets = {PurePosixPath(path).as_posix().lstrip("/"): path for path in expected}
    if len(targets) != len(expected) or any(".." in PurePosixPath(path).parts for path in targets):
        raise RuntimeProofError("runtime license evidence path is unsafe")
    resolved: dict[str, str] = {}
    hidden: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as outer:
        layers = _oci_platform_layers(outer, platform=platform)
        for layer_digest in reversed(layers):
            if len(resolved) + len(hidden) == len(targets):
                break
            member_name = f"blobs/sha256/{layer_digest.removeprefix('sha256:')}"
            try:
                member = outer.getmember(member_name)
            except KeyError as error:
                raise RuntimeProofError(f"OCI layer blob is missing: {layer_digest}") from error
            layer_stream = outer.extractfile(member)
            if layer_stream is None:
                raise RuntimeProofError(f"OCI layer blob is unreadable: {layer_digest}")
            with tarfile.open(fileobj=layer_stream, mode="r|*") as layer:
                layer_files: dict[str, str] = {}
                layer_hidden: set[str] = set()
                for layer_member in layer:
                    normalized = PurePosixPath(layer_member.name.lstrip("./")).as_posix()
                    if not normalized or normalized.startswith("/") or ".." in PurePosixPath(
                        normalized
                    ).parts:
                        raise RuntimeProofError("OCI layer contains an unsafe path")
                    if (
                        normalized in targets
                        and normalized not in resolved
                        and layer_member.isfile()
                    ):
                        stream = layer.extractfile(layer_member)
                        if stream is None:
                            raise RuntimeProofError(
                                f"runtime license evidence is unreadable: {normalized}"
                            )
                        digest = hashlib.sha256()
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                        layer_files[normalized] = digest.hexdigest()
                    base = PurePosixPath(normalized).name
                    parent = PurePosixPath(normalized).parent
                    if base == ".wh..wh..opq":
                        prefix = parent.as_posix().rstrip("/") + "/"
                        layer_hidden.update(
                            path for path in targets if path.startswith(prefix)
                        )
                    elif base.startswith(".wh."):
                        target = (parent / base.removeprefix(".wh.")).as_posix()
                        if target in targets:
                            layer_hidden.add(target)
                for path, digest in layer_files.items():
                    resolved[path] = digest
                    layer_hidden.discard(path)
                hidden.update(path for path in layer_hidden if path not in resolved)
    missing = sorted(path for path in targets if path not in resolved)
    if missing:
        raise RuntimeProofError(
            f"runtime license evidence is missing from {platform}: {missing}"
        )
    actual = {targets[path]: digest for path, digest in resolved.items()}
    mismatched = sorted(
        path for path, digest in actual.items() if digest != expected[path]
    )
    if mismatched:
        details = {
            path: {"actual": actual[path], "expected": expected[path]}
            for path in mismatched
        }
        raise RuntimeProofError(
            f"runtime license evidence digest mismatch for {platform}: {details}"
        )
    return dict(sorted(actual.items()))


def _verify_runtime_license_evidence(
    *,
    audit_path: Path,
    image_archives: Mapping[str, Path],
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "io.roehub.runtime-license-audit/v1alpha1"
        or audit.get("status") != "passed"
        or audit.get("unresolved_count") != 0
    ):
        raise RuntimeProofError("runtime license audit is not accepted")
    records = audit.get("records")
    if not isinstance(records, list):
        raise RuntimeProofError("runtime license audit records are missing")
    expected_by_image: dict[str, dict[str, dict[str, str]]] = {
        "runtime": {"linux/amd64": {}, "linux/arm64": {}},
        "ml_runtime": {"linux/amd64": {}, "linux/arm64": {}},
    }
    for record in records:
        if not isinstance(record, dict) or record.get("classification") == "scanner-concluded":
            continue
        image = str(record.get("image", ""))
        platform = str(record.get("platform", ""))
        resolution = record.get("resolution")
        evidence = resolution.get("evidence") if isinstance(resolution, dict) else None
        if (
            image not in expected_by_image
            or platform not in expected_by_image[image]
            or not isinstance(evidence, dict)
        ):
            raise RuntimeProofError("runtime license evidence record is invalid")
        path = str(evidence.get("path", ""))
        digest = str(evidence.get("sha256", ""))
        previous = expected_by_image[image][platform].get(path)
        if not path.startswith("/") or len(digest) != 64 or previous not in {None, digest}:
            raise RuntimeProofError("runtime license evidence binding is invalid")
        expected_by_image[image][platform][path] = digest
    proof: dict[str, Any] = {}
    for image, expected_by_platform in sorted(expected_by_image.items()):
        archive = image_archives.get(image)
        if archive is None or any(not expected for expected in expected_by_platform.values()):
            raise RuntimeProofError(f"runtime license evidence set is incomplete: {image}")
        platforms: dict[str, Any] = {}
        for platform in ("linux/amd64", "linux/arm64"):
            expected = expected_by_platform[platform]
            actual = _layer_evidence_hashes(
                archive_path=archive,
                platform=platform,
                expected=expected,
            )
            platforms[platform] = {
                "evidence_count": len(actual),
                "evidence_set_sha256": hashlib.sha256(_json_bytes(actual)).hexdigest(),
                "status": "passed",
            }
        proof[image] = {"platforms": platforms, "status": "passed"}
    return {
        "classification_counts": audit.get("classification_counts"),
        "images": proof,
        "raw_noassertion_count": audit.get("raw_noassertion_count"),
        "status": "passed",
        "unresolved_count": 0,
    }


def _run(
    command: Sequence[str],
    *,
    label: str,
    timeout: int = 1800,
    retries: int = 1,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    last_detail = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result
        last_detail = (result.stderr or result.stdout).strip()
        if attempt < retries:
            time.sleep(min(attempt * 2, 5))
    raise RuntimeProofError(f"{label} failed: {last_detail}")


def _tool_version(command: Sequence[str]) -> str:
    result = _run(command, label=f"tool version {' '.join(command)}", timeout=60)
    return (result.stdout or result.stderr).strip()


def _digest_only_reference(reference: str) -> str:
    name, separator, digest = reference.partition("@")
    if not separator or not digest.startswith("sha256:"):
        raise RuntimeProofError(f"image reference is not digest-pinned: {reference}")
    last_slash = name.rfind("/")
    last_colon = name.rfind(":")
    if last_colon > last_slash:
        name = name[:last_colon]
    return f"{name}@{digest}"


def _registry_index_copy_format(*, raw_manifest: str, expected_digest: str) -> str:
    actual_digest = f"sha256:{hashlib.sha256(raw_manifest.encode()).hexdigest()}"
    if actual_digest != expected_digest:
        raise RuntimeProofError(
            f"registry manifest digest mismatch: {actual_digest} != {expected_digest}"
        )
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise RuntimeProofError("registry manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise RuntimeProofError("registry manifest root must be an object")
    media_type = manifest.get("mediaType")
    if not isinstance(media_type, str):
        raise RuntimeProofError("registry index lacks a media type")
    copy_format = REGISTRY_INDEX_COPY_FORMATS.get(media_type)
    if copy_format is None:
        raise RuntimeProofError(f"unsupported registry index media type: {media_type}")
    return copy_format


def _cached_oci_archive(*, name: str, reference: str) -> Path:
    digest = reference.rsplit("@", 1)[1]
    output = CACHE_ROOT / "images" / f"{digest.replace(':', '-')}.oci.tar"
    if output.is_file():
        try:
            if inspect_oci_archive(output).digest == digest:
                return output
        except (OfflineBundleError, OSError, tarfile.TarError, json.JSONDecodeError):
            output.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    source = _digest_only_reference(reference)
    try:
        raw_manifest = _run(
            [
                "skopeo",
                "inspect",
                "--retry-times",
                "5",
                "--raw",
                f"docker://{source}",
            ],
            label=f"registry index inspection {name}",
            timeout=600,
            retries=3,
        ).stdout
        copy_format = _registry_index_copy_format(
            raw_manifest=raw_manifest,
            expected_digest=digest,
        )
        _run(
            [
                "skopeo",
                "copy",
                "--retry-times",
                "3",
                "--all",
                "--format",
                copy_format,
                f"docker://{source}",
                f"oci-archive:{partial}:{name}",
            ],
            label=f"multi-architecture OCI copy {name}",
            timeout=3600,
            retries=3,
        )
        descriptor = inspect_oci_archive(partial)
        if descriptor.digest != digest:
            raise RuntimeProofError(
                f"registry/archive digest mismatch for {name}: {descriptor.digest} != {digest}"
            )
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def _validate_source_archive(*, name: str, path: Path, record: dict[str, Any]) -> None:
    if _sha256(path) != record["expected_sha256"]:
        raise RuntimeProofError(f"corresponding source digest mismatch: {name}")
    root = str(record["archive_root"])
    required = {f"{root}/{item}" for item in record["license_files"]}
    found: set[str] = set()
    with tarfile.open(path, mode="r:gz") as archive:
        count = 0
        for member in archive:
            count += 1
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeProofError(f"corresponding source archive is unsafe: {name}")
            if not relative.parts or relative.parts[0] != root:
                raise RuntimeProofError(f"corresponding source root mismatch: {name}")
            if (member.issym() or member.islnk()) and not _archive_link_target_is_safe(
                member=member,
                root=root,
            ):
                raise RuntimeProofError(
                    f"corresponding source link target is unsafe: {name}"
                )
            if member.name in required and member.isfile() and member.size > 0:
                found.add(member.name)
        if count == 0 or found != required:
            raise RuntimeProofError(
                f"corresponding source license inventory is incomplete: {name}"
            )


def _cached_source(*, name: str, record: dict[str, Any]) -> Path:
    output = CACHE_ROOT / "sources" / str(record["filename"])
    if output.is_file():
        try:
            _validate_source_archive(name=name, path=output, record=record)
            return output
        except (RuntimeProofError, OSError, tarfile.TarError):
            output.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    try:
        _run(
            [
                "curl",
                "-fL",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "20",
                "--output",
                str(partial),
                str(record["url"]),
            ],
            label=f"corresponding source download {name}",
            timeout=1800,
            retries=2,
        )
        _validate_source_archive(name=name, path=partial, record=record)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def _build_wheel(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        label="Roehub wheel build",
        timeout=600,
    )
    wheels = sorted(output.glob("roehub-*.whl"))
    if len(wheels) != 1:
        raise RuntimeProofError(f"expected exactly one Roehub wheel, found={len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        if not any(name.endswith(".dist-info/entry_points.txt") for name in names):
            raise RuntimeProofError("built wheel lacks console entry points")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
            raise RuntimeProofError("built wheel lacks the Apache license")
    return wheels[0]


def _write_key_pair(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    private_path = root / "release-signing-key"
    public_path = root / "release-signing-key.pub"
    _run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "roehub-release",
            "-f",
            str(private_path),
        ],
        label="ephemeral release signing key generation",
        timeout=60,
    )
    if not private_path.is_file() or not public_path.is_file():
        raise RuntimeProofError("ssh-keygen did not create the release key pair")
    return private_path, public_path


def _offline_tags() -> set[str]:
    output = _run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        label="Docker offline image inventory",
        timeout=60,
    ).stdout
    return {line for line in output.splitlines() if line.startswith("roehub-offline/")}


def _cleanup_new_tags(before: set[str]) -> list[str]:
    after = _offline_tags()
    owned = sorted(after - before)
    if owned:
        _run(
            ["docker", "image", "rm", "--force", *owned],
            label="offline proof image cleanup",
            timeout=600,
        )
    residual = _offline_tags() - before
    if residual:
        raise RuntimeProofError(f"offline proof left residual image tags: {sorted(residual)}")
    return owned


def _prepare_packet_observer(*, alpine_reference: str) -> tuple[str, str]:
    suffix = f"{os.getpid()}-{int(time.time())}"
    container = f"roehub-stage22-observer-build-{suffix}"
    tag = f"roehub-stage22/net-observer:{suffix}"
    try:
        _run(
            ["docker", "run", "-d", "--name", container, alpine_reference, "sleep", "300"],
            label="packet observer preparation container",
            timeout=300,
        )
        _run(
            [
                "docker",
                "exec",
                container,
                "apk",
                "add",
                "--no-cache",
                "libpcap=1.10.5-r1",
                "tcpdump=4.99.5-r1",
            ],
            label="packet observer tool installation",
            timeout=300,
        )
        _run(
            ["docker", "commit", container, tag],
            label="packet observer image commit",
            timeout=300,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            check=False,
            capture_output=True,
            text=True,
        )
    image_id = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        label="packet observer image inspection",
        timeout=60,
    ).stdout.strip()
    return tag, image_id


def _compose_airgap_proof(
    *,
    bundle: Path,
    state: Path,
    runtime_image_id: str,
    observer_image: str,
    root: Path,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o777)
    compose = bundle / "configs/installation/generated/base/compose.yaml"
    override = state / "compose.base.offline.yaml"
    project = f"roehub-stage22-airgap-{os.getpid()}"
    command = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(compose),
        "-f",
        str(override),
    ]
    capture_name = f"{project}-capture"
    capture_path = root / "network-boundary.pcap"
    capture_started = False
    network_name = ""
    external_packets: list[str] = []
    running: set[str] = set()
    expected_running: set[str] = set()
    stats = ""
    gateway = ""
    try:
        rendered = json.loads(
            _run(
                [*command, "config", "--format", "json"],
                label="retained candidate Compose rendering",
                timeout=120,
            ).stdout
        )
        if rendered.get("networks", {}).get("roehub", {}).get("internal") is not True:
            raise RuntimeProofError("retained candidate Compose does not deny default egress")
        services = rendered.get("services", {})
        if not isinstance(services, dict):
            raise RuntimeProofError("retained candidate Compose service map is invalid")
        expected_running = {
            name
            for name, service in services.items()
            if isinstance(service, dict)
            and not service.get("profiles")
            and service.get("restart") != "no"
        }
        grafana_environment = services.get("grafana", {}).get("environment", {})
        required_grafana = {
            "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",
            "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
            "GF_ANALYTICS_REPORTING_ENABLED": "false",
            "GF_PLUGINS_PREINSTALL_DISABLED": "true",
            "GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED": "true",
        }
        if not isinstance(grafana_environment, dict) or any(
            grafana_environment.get(key) != value for key, value in required_grafana.items()
        ):
            raise RuntimeProofError("Grafana phone-home defaults are not disabled")
        loki_config = (
            bundle
            / "configs/installation/generated/base/observability/loki.yml"
        ).read_text(encoding="utf-8")
        if "analytics:\n  reporting_enabled: false\n" not in loki_config:
            raise RuntimeProofError("Loki usage reporting is not disabled")

        _run([*command, "create"], label="retained candidate Compose create", timeout=300)
        network_names = _run(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--filter",
                "label=com.docker.compose.network=roehub",
                "--format",
                "{{.Name}}",
            ],
            label="Compose internal network discovery",
            timeout=60,
        ).stdout.splitlines()
        if len(network_names) != 1:
            raise RuntimeProofError(f"expected one Compose internal network: {network_names}")
        network_name = network_names[0]
        network = json.loads(
            _run(
                ["docker", "network", "inspect", network_name],
                label="Compose internal network inspection",
                timeout=60,
            ).stdout
        )[0]
        if network.get("Internal") is not True:
            raise RuntimeProofError("Docker did not enforce the Compose internal network")
        subnets: list[str] = []
        gateways: list[str] = []
        for item in network.get("IPAM", {}).get("Config", []):
            if not isinstance(item, dict):
                continue
            subnet = item.get("Subnet")
            if isinstance(subnet, str):
                subnets.append(subnet)
            item_gateway = item.get("Gateway")
            if isinstance(item_gateway, str):
                gateways.append(item_gateway)
        if len(subnets) != 1 or len(gateways) != 1:
            raise RuntimeProofError(
                "unexpected Compose network IPAM inventory: "
                f"subnets={subnets}, gateways={gateways}"
            )
        gateway = gateways[0]
        interface = f"br-{str(network['Id'])[:12]}"
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                capture_name,
                "--privileged",
                "--network",
                "host",
                "--mount",
                f"type=bind,src={root},dst=/capture",
                observer_image,
                "tcpdump",
                "-i",
                interface,
                "-nn",
                "-U",
                "-w",
                "/capture/network-boundary.pcap",
                "ip",
            ],
            label="Docker bridge packet observer start",
            timeout=120,
        )
        capture_started = True
        _run(
            [*command, "up", "-d", "--wait", "--wait-timeout", "300"],
            label="retained candidate base Compose activation",
            timeout=420,
        )
        running = set(
            _run(
                [*command, "ps", "--services", "--status", "running"],
                label="retained candidate running service inventory",
                timeout=60,
            ).stdout.splitlines()
        )
        if running != expected_running:
            raise RuntimeProofError(
                f"retained candidate service set mismatch; "
                f"missing={sorted(expected_running - running)}, "
                f"extra={sorted(running - expected_running)}"
            )
        for service in sorted(running):
            container_id = _run(
                [*command, "ps", "-q", service],
                label=f"Compose container lookup {service}",
                timeout=60,
            ).stdout.strip()
            attached = json.loads(
                _run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{json .NetworkSettings.Networks}}",
                        container_id,
                    ],
                    label=f"Compose network attachment {service}",
                    timeout=60,
                ).stdout
            )
            if set(attached) != {network_name}:
                raise RuntimeProofError(
                    f"service is attached outside the internal network: {service}"
                )
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                network_name,
                "--entrypoint",
                "python",
                runtime_image_id,
                "-c",
                (
                    "import socket; s=socket.socket(); s.settimeout(2); "
                    "r=s.connect_ex(('1.1.1.1',443)); "
                    "assert r != 0, 'external egress unexpectedly succeeded'; print(r)"
                ),
            ],
            label="Docker internal-network enforcement probe",
            timeout=60,
        )
        time.sleep(15)
        stats = _run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}|{{.MemUsage}}",
            ],
            label="Compose memory snapshot",
            timeout=60,
        ).stdout
        logs = _run(
            [*command, "logs", "--no-color"],
            label="retained candidate Compose logs",
            timeout=120,
        ).stdout.lower()
        forbidden_log_markers = (
            "stats.grafana.org",
            "grafana.com/api/plugins",
            "loki-usage-report",
        )
        observed_markers = [item for item in forbidden_log_markers if item in logs]
        if observed_markers:
            raise RuntimeProofError(
                f"phone-home destination appeared in service logs: {observed_markers}"
            )
        _run(
            ["docker", "stop", "--time", "5", capture_name],
            label="Docker bridge packet observer stop",
            timeout=30,
        )
        capture_started = False
        packet_output = _run(
            [
                "docker",
                "run",
                "--rm",
                "--privileged",
                "--network",
                "host",
                "--mount",
                f"type=bind,src={root},dst=/capture,readonly",
                observer_image,
                "tcpdump",
                "-nn",
                "-r",
                "/capture/network-boundary.pcap",
                "src",
                "net",
                subnets[0],
                "and",
                "not",
                "src",
                "host",
                gateway,
                "and",
                "not",
                "dst",
                "net",
                subnets[0],
            ],
            label="Docker bridge packet boundary analysis",
            timeout=120,
        ).stdout
        external_packets = [line for line in packet_output.splitlines() if line.strip()]
        if external_packets:
            raise RuntimeProofError(
                f"undeclared external packets observed: {external_packets[:5]}"
            )
    finally:
        if capture_started:
            subprocess.run(
                ["docker", "stop", "--time", "2", capture_name],
                check=False,
                capture_output=True,
                text=True,
            )
        subprocess.run(
            ["docker", "rm", "--force", capture_name],
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans", "--timeout", "10"],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    residual = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        label="Compose cleanup verification",
        timeout=60,
    ).stdout.splitlines()
    if residual:
        raise RuntimeProofError(f"retained candidate Compose cleanup is incomplete: {residual}")
    return {
        "boundary": "docker-internal-bridge",
        "bridge_gateway": gateway,
        "bridge_gateway_control_traffic_excluded": True,
        "capture_sha256": _sha256(capture_path),
        "external_packet_count": len(external_packets),
        "grafana_phone_home_disabled": True,
        "loki_usage_reporting_disabled": True,
        "network_internal": True,
        "permanent_services_observed": sorted(running),
        "service_count": len(running),
        "status": "passed",
        "memory_snapshot": sorted(line for line in stats.splitlines() if project in line),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-repeat-archive", type=Path, required=True)
    parser.add_argument("--ml-archive", type=Path, required=True)
    parser.add_argument("--ml-repeat-archive", type=Path, required=True)
    parser.add_argument("--openbao-archive", type=Path, required=True)
    parser.add_argument("--openbao-repeat-archive", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    before_tags: set[str] = set()
    tags_snapshot_taken = False
    cleaned_tags: list[str] = []
    observer_tag = ""
    observer_image_id = ""
    candidate_path: Path | None = None
    candidate_backup: Path | None = None
    trust_path: Path | None = None
    trust_backup: Path | None = None
    candidate_committed = False
    try:
        release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
        if not isinstance(release, dict):
            raise RuntimeProofError("release manifest root must be an object")
        runtime = inspect_oci_archive(args.runtime_archive)
        runtime_repeat = inspect_oci_archive(args.runtime_repeat_archive)
        ml = inspect_oci_archive(args.ml_archive)
        ml_repeat = inspect_oci_archive(args.ml_repeat_archive)
        openbao = inspect_oci_archive(args.openbao_archive)
        openbao_repeat = inspect_oci_archive(args.openbao_repeat_archive)
        if runtime != runtime_repeat or _sha256(args.runtime_archive) != _sha256(
            args.runtime_repeat_archive
        ):
            raise RuntimeProofError("runtime OCI rebuild is not byte-reproducible")
        if ml != ml_repeat or _sha256(args.ml_archive) != _sha256(args.ml_repeat_archive):
            raise RuntimeProofError("ML OCI rebuild is not byte-reproducible")
        if openbao != openbao_repeat or _sha256(args.openbao_archive) != _sha256(
            args.openbao_repeat_archive
        ):
            raise RuntimeProofError("OpenBao OCI rebuild is not byte-reproducible")
        expected_runtime = release["images"]["runtime"]["reference"].rsplit("@", 1)[1]
        expected_ml = release["images"]["ml_runtime"]["reference"].rsplit("@", 1)[1]
        expected_openbao = release["images"]["openbao"]["reference"].rsplit("@", 1)[1]
        if (
            runtime.digest != expected_runtime
            or ml.digest != expected_ml
            or openbao.digest != expected_openbao
        ):
            raise RuntimeProofError("first-party OCI digest does not match release manifest")

        image_inputs: dict[str, Path] = {}
        external_by_digest: dict[str, Path] = {}
        for name, record in sorted(release["images"].items()):
            if name == "runtime":
                image_inputs[name] = args.runtime_archive.resolve()
                continue
            if name == "ml_runtime":
                image_inputs[name] = args.ml_archive.resolve()
                continue
            if name == "openbao":
                image_inputs[name] = args.openbao_archive.resolve()
                continue
            reference = record["reference"]
            digest = reference.rsplit("@", 1)[1]
            archive = external_by_digest.get(digest)
            if archive is None:
                print(f"stage22: cache/verify OCI image {name}", flush=True)
                archive = _cached_oci_archive(name=name, reference=reference)
                external_by_digest[digest] = archive
            image_inputs[name] = archive

        sources = {
            name: _cached_source(name=name, record=record)
            for name, record in SOURCE_RECORDS.items()
        }
        print("stage22: prepare pinned packet observer before air-gapped activation", flush=True)
        observer_tag, observer_image_id = _prepare_packet_observer(
            alpine_reference=release["images"]["config_consumer"]["reference"]
        )
        with tempfile.TemporaryDirectory(prefix="roehub-stage22-", dir=CACHE_ROOT) as raw:
            work = Path(raw)
            wheel = _build_wheel(work / "wheel-first")
            wheel_repeat = _build_wheel(work / "wheel-repeat")
            wheel_sha256 = _sha256(wheel)
            wheel_repeat_sha256 = _sha256(wheel_repeat)
            if (
                wheel_sha256 != wheel_repeat_sha256
                or wheel.read_bytes() != wheel_repeat.read_bytes()
            ):
                raise RuntimeProofError("Roehub wheel rebuild is not byte-reproducible")
            private_key, public_key = _write_key_pair(work / "keys")
            provenance_path = work / "provenance.json"
            provenance = {
                "build": {
                    "ml": {
                        "archive_sha256": _sha256(args.ml_archive),
                        "byte_reproducible": True,
                        "index_digest": ml.digest,
                    },
                    "openbao": {
                        "archive_sha256": _sha256(args.openbao_archive),
                        "byte_reproducible": True,
                        "index_digest": openbao.digest,
                    },
                    "runtime": {
                        "archive_sha256": _sha256(args.runtime_archive),
                        "byte_reproducible": True,
                        "index_digest": runtime.digest,
                    },
                    "source_date_epoch": 0,
                    "wheel": {
                        "build_command": "uv build --wheel --out-dir <isolated-output>",
                        "byte_reproducible": True,
                        "filename": wheel.name,
                        "first_sha256": wheel_sha256,
                        "repeat_sha256": wheel_repeat_sha256,
                    },
                },
                "corresponding_sources": {
                    name: {
                        **SOURCE_RECORDS[name],
                        "sha256": _sha256(path),
                    }
                    for name, path in sorted(sources.items())
                },
                "schema": "io.roehub.release-provenance/v1alpha1",
                "tools": {
                    "cosign": _tool_version(["cosign", "version"]),
                    "docker": _tool_version(
                        ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"]
                    ),
                    "skopeo": _tool_version(["skopeo", "--version"]),
                    "syft": _tool_version(["syft", "version", "-o", "json"]),
                },
            }
            provenance_path.write_bytes(_json_bytes(provenance))
            bundle = work / "bundle"
            print("stage22: create signed bundle and image SBOMs", flush=True)
            manifest = create_bundle(
                output=bundle,
                release_manifest=args.release_manifest,
                signing_key=private_key,
                wheel=wheel_repeat,
                image_inputs=image_inputs,
                source_inputs=sources,
                source_metadata=SOURCE_RECORDS,
                provenance=provenance_path,
            )
            verification = verify_bundle(bundle=bundle, trusted_public_key=public_key)
            bundle_repeat = work / "bundle-repeat"
            print("stage22: recreate signed bundle for byte comparison", flush=True)
            manifest_repeat = create_bundle(
                output=bundle_repeat,
                release_manifest=args.release_manifest,
                signing_key=private_key,
                wheel=wheel,
                image_inputs=image_inputs,
                source_inputs=sources,
                source_metadata=SOURCE_RECORDS,
                provenance=provenance_path,
            )
            verification_repeat = verify_bundle(
                bundle=bundle_repeat,
                trusted_public_key=public_key,
            )
            first_hashes = _directory_hashes(bundle)
            repeat_hashes = _directory_hashes(bundle_repeat)
            if manifest != manifest_repeat or first_hashes != repeat_hashes:
                changed = sorted(
                    path
                    for path in set(first_hashes) | set(repeat_hashes)
                    if first_hashes.get(path) != repeat_hashes.get(path)
                )
                raise RuntimeProofError(
                    f"offline bundle rebuild is not byte-reproducible: {changed[:10]}"
                )
            if verification != verification_repeat:
                raise RuntimeProofError("offline bundle repeat verification differs")
            if any(path.name == ".git" for path in bundle.rglob(".git")):
                raise RuntimeProofError("offline bundle unexpectedly contains Git metadata")

            notice = bundle / "NOTICE"
            original_notice = notice.read_bytes()
            notice.write_bytes(original_notice + b"tamper")
            try:
                verify_bundle(bundle=bundle, trusted_public_key=public_key)
            except OfflineBundleError:
                tamper_rejected = True
            else:
                tamper_rejected = False
            finally:
                notice.write_bytes(original_notice)
                notice.chmod(0o644)
            if not tamper_rejected:
                raise RuntimeProofError("offline bundle tampering was not rejected")
            verify_bundle(bundle=bundle, trusted_public_key=public_key)
            print("stage22: verify runtime license evidence in both OCI platforms", flush=True)
            license_audit_proof = _verify_runtime_license_evidence(
                audit_path=bundle / str(manifest["license_audit"]["path"]),
                image_archives={
                    "ml_runtime": args.ml_archive,
                    "runtime": args.runtime_archive,
                },
            )

            version = str(release["version"])
            candidate_root = CACHE_ROOT / "candidates"
            trust_root = CACHE_ROOT / "trust"
            candidate_root.mkdir(parents=True, exist_ok=True)
            trust_root.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_root / f"roehub-{version}"
            candidate_backup = candidate_root / f".roehub-{version}.previous"
            trust_path = trust_root / f"roehub-{version}.pub"
            trust_backup = trust_root / f".roehub-{version}.pub.previous"
            if candidate_backup.exists() or trust_backup.exists():
                raise RuntimeProofError("stale candidate backup requires manual inspection")
            if candidate_path.exists():
                os.replace(candidate_path, candidate_backup)
            if trust_path.exists():
                os.replace(trust_path, trust_backup)
            os.replace(bundle, candidate_path)
            bundle = candidate_path
            trust_temporary = trust_path.with_suffix(".tmp")
            shutil.copyfile(public_key, trust_temporary)
            trust_temporary.chmod(0o644)
            os.replace(trust_temporary, trust_path)
            verification = verify_bundle(bundle=bundle, trusted_public_key=trust_path)

            before_tags = _offline_tags()
            tags_snapshot_taken = True
            state = work / "state"
            print("stage22: install verified bundle without registry access", flush=True)
            installed = _run(
                [
                    "env",
                    (
                        "PATH=/usr/bin:/opt/homebrew/bin:/usr/local/bin:/bin:"
                        "/usr/sbin:/sbin"
                    ),
                    str(bundle / "tools/release/install-offline.sh"),
                    "--trusted-public-key",
                    str(trust_path),
                    "--state-directory",
                    str(state),
                    "--profile",
                    "base",
                    "--runtime-smoke",
                ],
                label="one-command offline installer",
                timeout=1800,
            )
            installation = json.loads(installed.stdout)
            if not isinstance(installation, dict) or installation.get("status") != "passed":
                raise RuntimeProofError("one-command offline installer returned invalid output")
            lock = json.loads((state / "offline-image-lock.json").read_text(encoding="utf-8"))
            egress = _compose_airgap_proof(
                bundle=bundle,
                state=state,
                runtime_image_id=lock["images"]["runtime"],
                observer_image=observer_tag,
                root=work / "network-boundary",
            )
            cleaned_tags = _cleanup_new_tags(before_tags)
            cold_verification = verify_bundle(
                bundle=bundle,
                trusted_public_key=trust_path,
            )
            if cold_verification != verification:
                raise RuntimeProofError("retained candidate cold verification differs")
            if candidate_backup is not None and candidate_backup.exists():
                shutil.rmtree(candidate_backup)
            if trust_backup is not None and trust_backup.exists():
                trust_backup.unlink()
            candidate_committed = True

            payload = {
                "air_gapped_install": {
                    "git_metadata_present": False,
                    "image_lock": "verified-content-addressed",
                    "imported_unique_images": installation["imported_unique_images"],
                    "installer_entrypoint": "tools/release/install-offline.sh",
                    "installer_python": "/usr/bin/python3",
                    "profile": installation["profile"],
                    "runtime_smoke": installation["runtime_smoke"],
                },
                "bundle": {
                    "asset_count": verification["asset_count"],
                    "corresponding_sources": sorted(manifest["corresponding_sources"]),
                    "image_count": verification["image_count"],
                    "manifest_sha256": _sha256(bundle / MANIFEST_NAME),
                    "path": str(bundle),
                    "retained": True,
                    "signature_verified": verification["signature_verified"],
                    "tamper_rejected": tamper_rejected,
                    "total_size_bytes": _directory_size(bundle),
                    "tree_sha256": _directory_digest(first_hashes),
                    "trusted_public_key_path": str(trust_path),
                    "wheel": wheel.name,
                },
                "cleanup": {
                    "owned_image_tags_removed": len(cleaned_tags),
                    "residual_owned_tags": [],
                },
                "egress_observation": egress,
                "packet_observer": {
                    "image_id": observer_image_id,
                    "prepared_before_air_gapped_activation": True,
                    "runtime_dependency": False,
                },
                "images": {
                    name: {
                        "child_digests": record["child_digests"],
                        "index_digest": record["index_digest"],
                        "sbom": "verified",
                    }
                    for name, record in sorted(manifest["images"].items())
                },
                "license_audit": license_audit_proof,
                "reproducibility": {
                    "offline_bundle_byte_identical": True,
                    "offline_bundle_file_count": len(first_hashes),
                    "ml_archive_byte_identical": True,
                    "openbao_archive_byte_identical": True,
                    "runtime_archive_byte_identical": True,
                    "wheel_byte_identical": True,
                    "wheel_first_sha256": wheel_sha256,
                    "wheel_repeat_sha256": wheel_repeat_sha256,
                },
                "schema": "io.roehub.stage22-runtime-proof/v1alpha1",
                "status": "passed",
            }
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            args.evidence.write_bytes(_json_bytes(payload))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        KeyError,
        OSError,
        OfflineBundleError,
        RuntimeProofError,
        subprocess.SubprocessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Stage 22 runtime proof failed: {error}", file=sys.stderr)
        return 1
    finally:
        if tags_snapshot_taken:
            try:
                _cleanup_new_tags(before_tags)
            except (OfflineBundleError, RuntimeProofError, subprocess.SubprocessError):
                pass
        if observer_tag:
            subprocess.run(
                ["docker", "image", "rm", "--force", observer_tag],
                check=False,
                capture_output=True,
                text=True,
            )
        if candidate_path is not None and not candidate_committed:
            if candidate_path.exists():
                shutil.rmtree(candidate_path)
            if candidate_backup is not None and candidate_backup.exists():
                os.replace(candidate_backup, candidate_path)
            if trust_path is not None and trust_path.exists():
                trust_path.unlink()
            if (
                trust_path is not None
                and trust_backup is not None
                and trust_backup.exists()
            ):
                os.replace(trust_backup, trust_path)


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
