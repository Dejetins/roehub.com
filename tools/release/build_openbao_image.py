#!/usr/bin/env python3
"""Build the licensed Roehub OpenBao derivative without publishing it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.release.offline_bundle import inspect_oci_archive

ROOT = Path(__file__).resolve().parents[2]
OPENBAO_DOCKER = ROOT / "infra" / "docker" / "openbao"
DEFAULT_CACHE = Path.home() / ".cache" / "roehub" / "stage22-offline-release"
DEFAULT_CONTEXT = DEFAULT_CACHE / "openbao-build-context"
DEFAULT_OUTPUT = DEFAULT_CACHE / "openbao-build-proof"
OPENBAO_SOURCE_NAME = "openbao-v2.5.4.tar.gz"
OPENBAO_SOURCE_SHA256 = "f4ed87e3b6ec213e3aada0b4eba3ce1a9525620b14a0dde5be4de0c778be0291"
OPENBAO_SOURCE_ROOT = "openbao-2.5.4"
OPENBAO_UPSTREAM_IMAGE_DIGEST = (
    "sha256:436eaf9778cad75507ff70ea26ace30dcbe15606e619ac3823495663d7f7c115"
)
OPENBAO_IMAGE_VERSION = "2.5.4-roehub-licensed-qr.1"
OPENBAO_IMAGE_REFERENCE = f"ghcr.io/dejetins/roehub-openbao:{OPENBAO_IMAGE_VERSION}"
SOURCE_DATE_EPOCH = 1_779_292_428
SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")
PACKAGE_LAYERS = {
    "linux/amd64": "sha256:3e26a337cf92c8ca21797fbf622f27b9b492f2e74d9a5af980190330fa733b8f",
    "linux/arm64": "sha256:a537d976b5ef3dfae7ea89f86494c71a2ce077b6c22dbf077f6f2d4dc3586196",
}
SKIP2_SOURCE_SHA256 = "35460d655e1bef07615a38d47295bfa39d08310440d2b4a19cd5b0bedd2c8fd9"
SKIP2_SOURCE_ROOT = "go-qrcode-da1b6568686e89143e94f980a98bc2dbd5537f13"
SKIP2_SOURCE_NAME = f"skip2-{SKIP2_SOURCE_ROOT}.tar.gz"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
MAX_JSON_BYTES = 16 * 1024 * 1024


class OpenBaoBuildError(RuntimeError):
    """Raised when a source, image, or build proof is not exact."""


@dataclass(frozen=True)
class PlatformGraph:
    platform: str
    manifest_descriptor: dict[str, Any]
    blob_descriptors: tuple[dict[str, Any], ...]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256_path(path)
    if actual != expected:
        raise OpenBaoBuildError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _digest_parts(descriptor: Mapping[str, Any], *, label: str) -> tuple[str, int, str]:
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    media_type = descriptor.get("mediaType")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise OpenBaoBuildError(f"{label} has an invalid digest")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise OpenBaoBuildError(f"{label} has an invalid size")
    if not isinstance(media_type, str) or not media_type:
        raise OpenBaoBuildError(f"{label} has no media type")
    return digest, size, media_type


def _members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
            raise OpenBaoBuildError(f"unsafe OCI archive member: {member.name}")
        if member.name in result:
            raise OpenBaoBuildError(f"duplicate OCI archive member: {member.name}")
        result[member.name] = member
    return result


def _member_bytes(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    name: str,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile() or member.size > max_bytes:
        raise OpenBaoBuildError(f"missing or oversized archive member: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise OpenBaoBuildError(f"cannot read archive member: {name}")
    payload = stream.read(max_bytes + 1)
    if len(payload) != member.size:
        raise OpenBaoBuildError(f"truncated archive member: {name}")
    return payload


def _json_member(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    name: str,
) -> dict[str, Any]:
    value = json.loads(_member_bytes(archive, members, name))
    if not isinstance(value, dict):
        raise OpenBaoBuildError(f"JSON archive member is not an object: {name}")
    return value


def _verified_blob(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> tuple[tarfile.TarInfo, str]:
    digest, size, media_type = _digest_parts(descriptor, label=label)
    name = f"blobs/sha256/{digest.removeprefix('sha256:')}"
    member = members.get(name)
    if member is None or not member.isfile() or member.size != size:
        raise OpenBaoBuildError(f"{label} blob is missing or has the wrong size: {digest}")
    stream = archive.extractfile(member)
    if stream is None:
        raise OpenBaoBuildError(f"{label} blob cannot be read: {digest}")
    actual = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        actual.update(chunk)
    if f"sha256:{actual.hexdigest()}" != digest:
        raise OpenBaoBuildError(f"{label} blob digest mismatch: {digest}")
    return member, media_type


def _json_blob(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    member, _ = _verified_blob(archive, members, descriptor, label=label)
    if member.size > MAX_JSON_BYTES:
        raise OpenBaoBuildError(f"{label} JSON blob is too large")
    value = json.loads(_member_bytes(archive, members, member.name))
    if not isinstance(value, dict):
        raise OpenBaoBuildError(f"{label} JSON blob is not an object")
    return value


def _single_platform_graph(path: Path) -> PlatformGraph:
    with tarfile.open(path, "r") as archive:
        members = _members(archive)
        index = _json_member(archive, members, "index.json")
        descriptors = index.get("manifests")
        if not isinstance(descriptors, list) or len(descriptors) != 1:
            raise OpenBaoBuildError(f"single-platform OCI archive has {descriptors!r}")
        manifest_descriptor = descriptors[0]
        if not isinstance(manifest_descriptor, dict):
            raise OpenBaoBuildError("single-platform descriptor is not an object")
        digest, _, media_type = _digest_parts(
            manifest_descriptor,
            label="single-platform manifest",
        )
        if media_type != OCI_MANIFEST:
            raise OpenBaoBuildError(f"single-platform descriptor is not an OCI manifest: {digest}")
        platform_value = manifest_descriptor.get("platform")
        if not isinstance(platform_value, dict):
            raise OpenBaoBuildError("single-platform descriptor lacks a platform")
        operating_system = platform_value.get("os")
        architecture = platform_value.get("architecture")
        platform = f"{operating_system}/{architecture}"
        if platform not in SUPPORTED_PLATFORMS:
            raise OpenBaoBuildError(f"unsupported OpenBao platform: {platform}")
        manifest = _json_blob(
            archive,
            members,
            manifest_descriptor,
            label=f"{platform} manifest",
        )
        config = manifest.get("config")
        layers = manifest.get("layers")
        if not isinstance(config, dict) or not isinstance(layers, list):
            raise OpenBaoBuildError(f"{platform} manifest graph is incomplete")
        _, _, config_media_type = _digest_parts(config, label=f"{platform} config")
        if config_media_type != OCI_CONFIG:
            raise OpenBaoBuildError(f"{platform} config has the wrong media type")
        config_value = _json_blob(archive, members, config, label=f"{platform} config")
        if (
            config_value.get("os") != operating_system
            or config_value.get("architecture") != architecture
        ):
            raise OpenBaoBuildError(f"{platform} config platform mismatch")
        graph: list[dict[str, Any]] = [dict(manifest_descriptor), dict(config)]
        for position, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise OpenBaoBuildError(f"{platform} layer {position} is not an object")
            _verified_blob(
                archive,
                members,
                layer,
                label=f"{platform} layer {position}",
            )
            graph.append(dict(layer))
        return PlatformGraph(
            platform=platform,
            manifest_descriptor=dict(manifest_descriptor),
            blob_descriptors=tuple(graph),
        )


def _copy_verified_blob(
    *,
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    descriptor: Mapping[str, Any],
    destination: Path,
    label: str,
) -> None:
    member, _ = _verified_blob(archive, members, descriptor, label=label)
    digest = str(descriptor["digest"]).removeprefix("sha256:")
    target = destination / digest
    if target.is_file():
        _check_sha256(target, digest, label=f"deduplicated {label}")
        return
    stream = archive.extractfile(member)
    if stream is None:
        raise OpenBaoBuildError(f"cannot copy {label}")
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as output:
        shutil.copyfileobj(stream, output, length=1024 * 1024)
    _check_sha256(temporary, digest, label=label)
    os.replace(temporary, target)


def _write_oci_tar(layout: Path, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, "w") as archive:
        for path in sorted(item for item in layout.rglob("*") if item.is_file()):
            relative = path.relative_to(layout).as_posix()
            payload_size = path.stat().st_size
            info = tarfile.TarInfo(relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            info.mtime = SOURCE_DATE_EPOCH
            info.size = payload_size
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    os.replace(temporary, output)


def combine_platform_archives(
    inputs: Sequence[Path],
    output: Path,
    *,
    image_reference: str = OPENBAO_IMAGE_REFERENCE,
) -> str:
    """Combine two verified single-platform OCI archives into one local index."""

    if not image_reference or "\x00" in image_reference:
        raise OpenBaoBuildError("combined OCI image reference is invalid")
    graphs = [_single_platform_graph(path) for path in inputs]
    by_platform = {graph.platform: graph for graph in graphs}
    if set(by_platform) != set(SUPPORTED_PLATFORMS) or len(graphs) != len(by_platform):
        raise OpenBaoBuildError(
            f"OpenBao platform inputs must be exactly {SUPPORTED_PLATFORMS}: {sorted(by_platform)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roehub-openbao-oci-", dir=output.parent) as temporary:
        layout = Path(temporary)
        blobs = layout / "blobs" / "sha256"
        blobs.mkdir(parents=True)
        children: list[dict[str, Any]] = []
        platform_inputs = sorted(
            zip(inputs, graphs, strict=True),
            key=lambda item: item[1].platform,
        )
        for source, graph in platform_inputs:
            with tarfile.open(source, "r") as archive:
                members = _members(archive)
                for position, descriptor in enumerate(graph.blob_descriptors):
                    _copy_verified_blob(
                        archive=archive,
                        members=members,
                        descriptor=descriptor,
                        destination=blobs,
                        label=f"{graph.platform} graph blob {position}",
                    )
            operating_system, architecture = graph.platform.split("/", 1)
            digest, size, media_type = _digest_parts(
                graph.manifest_descriptor,
                label=f"{graph.platform} manifest",
            )
            children.append(
                {
                    "digest": digest,
                    "mediaType": media_type,
                    "platform": {"architecture": architecture, "os": operating_system},
                    "size": size,
                }
            )
        image_index = _json_bytes(
            {"manifests": children, "mediaType": OCI_INDEX, "schemaVersion": 2}
        )
        index_digest = hashlib.sha256(image_index).hexdigest()
        (blobs / index_digest).write_bytes(image_index)
        (layout / "oci-layout").write_bytes(_json_bytes({"imageLayoutVersion": "1.0.0"}))
        (layout / "index.json").write_bytes(
            _json_bytes(
                {
                    "manifests": [
                        {
                            "annotations": {
                                "org.opencontainers.image.ref.name": image_reference,
                            },
                            "digest": f"sha256:{index_digest}",
                            "mediaType": OCI_INDEX,
                            "size": len(image_index),
                        }
                    ],
                    "mediaType": OCI_INDEX,
                    "schemaVersion": 2,
                }
            )
        )
        _write_oci_tar(layout, output)
    descriptor = inspect_oci_archive(output)
    if set(descriptor.platforms) != set(SUPPORTED_PLATFORMS):
        raise OpenBaoBuildError(f"combined OCI platform mismatch: {descriptor.platforms}")
    return descriptor.digest


def _copy_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    output: Path,
) -> None:
    stream = archive.extractfile(member)
    if stream is None:
        raise OpenBaoBuildError(f"cannot extract archive member: {member.name}")
    with output.open("wb") as target:
        shutil.copyfileobj(stream, target, length=1024 * 1024)


def prepare_context(*, source: Path, upstream_image: Path, output: Path) -> None:
    """Prepare the small, digest-verified Docker build context."""

    _check_sha256(source, OPENBAO_SOURCE_SHA256, label="OpenBao source")
    if output.exists() and any(output.iterdir()):
        raise OpenBaoBuildError(f"build context is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output / OPENBAO_SOURCE_NAME)
    for name in ("openbao-2.5.4-licensed-qr.patch", "openbao-2.5.4-licensed-qr.NOTICE"):
        shutil.copyfile(OPENBAO_DOCKER / name, output / name)
    with tarfile.open(upstream_image, "r") as archive:
        members = _members(archive)
        index = _json_member(archive, members, "index.json")
        top = index.get("manifests")
        if not isinstance(top, list) or len(top) != 1 or not isinstance(top[0], dict):
            raise OpenBaoBuildError("upstream OpenBao OCI index is malformed")
        digest, _, _ = _digest_parts(top[0], label="upstream OpenBao index")
        if digest != OPENBAO_UPSTREAM_IMAGE_DIGEST:
            raise OpenBaoBuildError(f"unexpected upstream OpenBao image: {digest}")
        image_index = _json_blob(archive, members, top[0], label="upstream OpenBao index")
        manifests = image_index.get("manifests")
        if not isinstance(manifests, list):
            raise OpenBaoBuildError("upstream OpenBao image has no platform manifests")
        platform_manifests: dict[str, dict[str, Any]] = {}
        for descriptor in manifests:
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("platform"), dict):
                continue
            platform_value = descriptor["platform"]
            platform = f"{platform_value.get('os')}/{platform_value.get('architecture')}"
            if platform in PACKAGE_LAYERS:
                platform_manifests[platform] = descriptor
        for platform, package_digest in PACKAGE_LAYERS.items():
            descriptor = platform_manifests.get(platform)
            if descriptor is None:
                raise OpenBaoBuildError(f"upstream image lacks {platform}")
            manifest = _json_blob(archive, members, descriptor, label=f"upstream {platform}")
            layers = manifest.get("layers")
            if not isinstance(layers, list) or not any(
                isinstance(layer, dict) and layer.get("digest") == package_digest
                for layer in layers
            ):
                raise OpenBaoBuildError(f"upstream {platform} lacks package layer {package_digest}")
            member = members.get(f"blobs/sha256/{package_digest.removeprefix('sha256:')}")
            if member is None or not member.isfile():
                raise OpenBaoBuildError(f"package layer is missing: {package_digest}")
            architecture = platform.split("/", 1)[1]
            destination = output / f"openbao-runtime-packages-{architecture}.tar.gz"
            _copy_archive_member(archive, member, destination)
            _check_sha256(
                destination,
                package_digest.removeprefix("sha256:"),
                label=f"{platform} runtime package layer",
            )


def build_platform(
    *,
    platform: str,
    context: Path,
    output: Path,
    builder: str,
    no_cache: bool = False,
    no_cache_filters: Sequence[str] = (),
) -> None:
    """Build one platform locally; no tag, push, or registry write is performed."""

    if platform not in SUPPORTED_PLATFORMS:
        raise OpenBaoBuildError(f"unsupported build platform: {platform}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [
        "docker",
        "buildx",
        "build",
        "--builder",
        builder,
        "--platform",
        platform,
        "--file",
        str(OPENBAO_DOCKER / "Dockerfile"),
        "--build-arg",
        f"SOURCE_DATE_EPOCH={SOURCE_DATE_EPOCH}",
        "--provenance=false",
        "--sbom=false",
        "--progress=plain",
        "--output",
        (
            f"type=oci,dest={output},name={OPENBAO_IMAGE_REFERENCE},"
            "rewrite-timestamp=true"
        ),
    ]
    if no_cache:
        command.append("--no-cache")
    for stage in no_cache_filters:
        command.extend(("--no-cache-filter", stage))
    command.append(str(context))
    result = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if result.returncode != 0:
        raise OpenBaoBuildError(f"OpenBao {platform} build failed: {result.returncode}")
    graph = _single_platform_graph(output)
    if graph.platform != platform:
        raise OpenBaoBuildError(f"OpenBao build returned {graph.platform}, expected {platform}")


def _extract_source_archive(path: Path, destination: Path, *, expected_root: str) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path, "r:gz") as archive:
        roots: set[str] = set()
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise OpenBaoBuildError(f"unsafe source archive member: {member.name}")
            roots.add(relative.parts[0])
            if member.isdev() or member.isfifo():
                raise OpenBaoBuildError(f"special source archive member: {member.name}")
        if roots != {expected_root}:
            raise OpenBaoBuildError(f"source archive root mismatch: {sorted(roots)}")
        archive.extractall(destination, filter="data")
    return destination / expected_root


def _normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = SOURCE_DATE_EPOCH
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o755 if info.mode & stat.S_IXUSR else 0o644
    return info


def _write_source_tar(root: Path, output: Path, *, archive_root: str) -> None:
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=SOURCE_DATE_EPOCH) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                archive.add(
                    root,
                    arcname=archive_root,
                    recursive=True,
                    filter=_normalized_tarinfo,
                )
    os.replace(temporary, output)


def create_corresponding_source(
    *,
    openbao_source: Path,
    skip2_source: Path,
    output: Path,
) -> str:
    """Create complete, deterministic corresponding source for the derivative."""

    _check_sha256(openbao_source, OPENBAO_SOURCE_SHA256, label="OpenBao source")
    _check_sha256(skip2_source, SKIP2_SOURCE_SHA256, label="skip2/go-qrcode source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roehub-openbao-source-", dir=output.parent) as temp:
        temporary = Path(temp)
        openbao = _extract_source_archive(
            openbao_source,
            temporary / "openbao",
            expected_root=OPENBAO_SOURCE_ROOT,
        )
        patch_path = OPENBAO_DOCKER / "openbao-2.5.4-licensed-qr.patch"
        with patch_path.open("rb") as patch_stream:
            result = subprocess.run(
                ["patch", "--fuzz=0", "--batch", "--forward", "-d", str(openbao), "-p1"],
                stdin=patch_stream,
                check=False,
                capture_output=True,
            )
        if result.returncode != 0:
            raise OpenBaoBuildError(
                f"OpenBao corresponding-source patch failed: {result.stderr.decode().strip()}"
            )
        graph_files = (openbao / "go.mod", openbao / "go.sum")
        if any("github.com/yeqown" in path.read_text(encoding="utf-8") for path in graph_files):
            raise OpenBaoBuildError("corresponding source still references yeqown")
        skip2 = _extract_source_archive(
            skip2_source,
            temporary / "skip2",
            expected_root=SKIP2_SOURCE_ROOT,
        )
        third_party = openbao / "third_party" / "skip2-go-qrcode"
        shutil.copytree(skip2, third_party, symlinks=True)
        roehub = openbao / "roehub"
        roehub.mkdir()
        shutil.copyfile(patch_path, roehub / patch_path.name)
        notice = OPENBAO_DOCKER / "openbao-2.5.4-licensed-qr.NOTICE"
        shutil.copyfile(notice, roehub / notice.name)
        _write_source_tar(
            openbao,
            output,
            archive_root="openbao-2.5.4-roehub-licensed-qr.1",
        )
    return _sha256_path(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-context")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--upstream-image", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=DEFAULT_CONTEXT)

    build = subparsers.add_parser("build-platform")
    build.add_argument("--platform", choices=SUPPORTED_PLATFORMS, required=True)
    build.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--builder", default="roehub-openbao")
    build.add_argument("--no-cache", action="store_true")
    build.add_argument(
        "--no-cache-filter",
        action="append",
        choices=("go-build", "ui-build"),
        default=[],
    )

    combine = subparsers.add_parser("combine")
    combine.add_argument("--amd64", type=Path, required=True)
    combine.add_argument("--arm64", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--image-reference", default=OPENBAO_IMAGE_REFERENCE)

    source = subparsers.add_parser("corresponding-source")
    source.add_argument("--openbao-source", type=Path, required=True)
    source.add_argument("--skip2-source", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-context":
        prepare_context(source=args.source, upstream_image=args.upstream_image, output=args.output)
        value: object = {"context": str(args.output)}
    elif args.command == "build-platform":
        build_platform(
            platform=args.platform,
            context=args.context,
            output=args.output,
            builder=args.builder,
            no_cache=args.no_cache,
            no_cache_filters=args.no_cache_filter,
        )
        value = {"archive": str(args.output), "platform": args.platform}
    elif args.command == "combine":
        digest = combine_platform_archives(
            [args.amd64, args.arm64],
            args.output,
            image_reference=args.image_reference,
        )
        value = {"archive": str(args.output), "digest": digest}
    elif args.command == "corresponding-source":
        digest = create_corresponding_source(
            openbao_source=args.openbao_source,
            skip2_source=args.skip2_source,
            output=args.output,
        )
        value = {"archive": str(args.output), "sha256": digest}
    else:  # pragma: no cover - argparse enforces the command set.
        raise OpenBaoBuildError(f"unsupported command: {args.command}")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
