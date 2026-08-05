#!/usr/bin/env python3
"""Validate OSS distribution policy and generate deterministic release metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / "tools" / "release"
POLICY_PATH = RELEASE_DIR / "oss_policy.json"
SBOM_PATH = RELEASE_DIR / "preliminary-sbom.spdx.json"
NOTICES_PATH = RELEASE_DIR / "THIRD_PARTY_NOTICES.md"
METADATA_PATH = RELEASE_DIR / "release-metadata.json"

SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
EXACT_VERSION_RE = re.compile(r"==\s*([^,;\s]+)")
IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[a-f0-9]{64}$")
ASSET_SUFFIXES = {
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".otf",
    ".png",
    ".so",
    ".svg",
    ".ttf",
    ".wasm",
    ".webp",
    ".woff",
    ".woff2",
}
_NON_DISTRIBUTION_CONTAINER_SOURCE_PREFIXES = ("tests/fixtures/",)


class PolicyError(RuntimeError):
    """Raised when the current tree cannot form an accepted OSS inventory."""


@dataclass(frozen=True)
class Component:
    kind: str
    name: str
    version: str
    license_expression: str
    status: str
    source: str
    group: str = ""
    obligation: str = ""
    sha256: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME_RE.match(requirement)
    if not match:
        raise PolicyError(f"cannot parse requirement: {requirement!r}")
    return _normalize_name(match.group(1))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(item.decode() for item in result.stdout.split(b"\0") if item)


def _locked_versions() -> dict[str, str]:
    lock = _load_toml(ROOT / "uv.lock")
    return {
        _normalize_name(package["name"]): str(package["version"])
        for package in lock["package"]
    }


def _python_requirements(pyproject: dict[str, Any]) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for value in pyproject["project"].get("dependencies", []):
        requirements.append(("runtime", value))
    for group, values in sorted(pyproject["project"].get("optional-dependencies", {}).items()):
        for value in values:
            requirements.append((f"optional:{group}", value))
    for group, values in sorted(pyproject.get("dependency-groups", {}).items()):
        for value in values:
            requirements.append((f"development:{group}", value))
    for value in pyproject["build-system"].get("requires", []):
        requirements.append(("build-system", value))
    return requirements


def _python_components(policy: dict[str, Any], pyproject: dict[str, Any]) -> list[Component]:
    license_policy = {
        _normalize_name(name): value for name, value in policy["python_licenses"].items()
    }
    versions = _locked_versions()
    requirements = _python_requirements(pyproject)
    actual_names = {_requirement_name(value) for _, value in requirements}
    policy_names = set(license_policy)
    if actual_names != policy_names:
        missing = sorted(actual_names - policy_names)
        stale = sorted(policy_names - actual_names)
        raise PolicyError(f"python license policy mismatch; missing={missing}, stale={stale}")

    components: list[Component] = []
    for group, requirement in requirements:
        name = _requirement_name(requirement)
        exact = EXACT_VERSION_RE.search(requirement)
        version = versions.get(name) or (exact.group(1) if exact else "")
        if not version:
            raise PolicyError(f"no deterministic version for direct dependency {name}")
        record = license_policy[name]
        components.append(
            Component(
                kind="python",
                name=name,
                version=version,
                license_expression=record["license"],
                status=record["status"],
                source=f"https://pypi.org/project/{name}/{version}/",
                group=group,
                obligation=record.get("obligation", ""),
            )
        )
    return sorted(components, key=lambda item: (item.group, item.name))


def _discover_container_images(tracked: list[str]) -> set[str]:
    images: set[str] = set()
    for relative in tracked:
        if relative.startswith(_NON_DISTRIBUTION_CONTAINER_SOURCE_PREFIXES):
            continue
        path = Path(relative)
        name = path.name.lower()
        is_dockerfile = name.startswith("dockerfile")
        is_compose = (
            ("compose" in name or "docker-compose" in name)
            and path.suffix.lower() in {".yml", ".yaml"}
        )
        if not (is_dockerfile or is_compose):
            continue
        build_stages: set[str] = set()
        for raw_line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if is_dockerfile and line.upper().startswith("FROM "):
                fields = line.split()
                reference_position = 2 if fields[1].startswith("--platform=") else 1
                if len(fields) <= reference_position:
                    raise PolicyError(f"cannot parse Dockerfile FROM: {relative}: {line}")
                reference = fields[reference_position]
                if reference not in build_stages:
                    images.add(reference)
                if len(fields) >= 4 and fields[-2].upper() == "AS":
                    build_stages.add(fields[-1])
            elif is_compose and line.startswith("image:"):
                images.add(line.removeprefix("image:").strip())
    return images


def _container_components(policy: dict[str, Any], tracked: list[str]) -> list[Component]:
    actual = _discover_container_images(tracked)
    records = policy["container_images"]
    if actual != set(records):
        missing = sorted(actual - set(records))
        stale = sorted(set(records) - actual)
        raise PolicyError(f"container image policy mismatch; missing={missing}, stale={stale}")
    components = []
    for image in sorted(records):
        record = records[image]
        components.append(
            Component(
                kind="container",
                name=image,
                version=image.rsplit(":", 1)[-1] if ":" in image else "unversioned",
                license_expression=record["license"],
                status=record["status"],
                source=f"pkg:docker/{image}",
                group="distribution-container",
                obligation=record.get("obligation", ""),
            )
        )
    return components


def _release_image_components(policy: dict[str, Any]) -> list[Component]:
    components: list[Component] = []
    supported_architectures = set(policy["release_supported_architectures"])
    if supported_architectures != {"linux/amd64", "linux/arm64"}:
        raise PolicyError(
            "release_supported_architectures must be exactly linux/amd64 and linux/arm64"
        )
    for name, record in sorted(policy["release_images"].items()):
        reference = str(record["reference"])
        if not IMAGE_DIGEST_RE.fullmatch(reference):
            raise PolicyError(f"release image is not digest-pinned: {name}={reference}")
        if ":latest" in reference.lower():
            raise PolicyError(f"release image uses latest tag: {name}={reference}")
        platforms = set(record["platforms"])
        if platforms != supported_architectures:
            raise PolicyError(
                f"release image platform mismatch: {name}; "
                f"expected={sorted(supported_architectures)}, actual={sorted(platforms)}"
            )
        components.append(
            Component(
                kind="release-container",
                name=reference,
                version=reference.rsplit("@sha256:", 1)[1],
                license_expression=record["license"],
                status=record["status"],
                source=f"pkg:docker/{reference}",
                group=name,
                obligation=record.get("obligation", ""),
            )
        )
    return components


def _bundled_components(policy: dict[str, Any], tracked: list[str]) -> list[Component]:
    prefix = "apps/web/dist/vendor/"
    actual = {item for item in tracked if item.startswith(prefix)}
    records = policy["bundled_assets"]
    if actual != set(records):
        missing = sorted(actual - set(records))
        stale = sorted(set(records) - actual)
        raise PolicyError(f"bundled asset policy mismatch; missing={missing}, stale={stale}")

    components = []
    for relative in sorted(records):
        record = records[relative]
        actual_hash = _sha256_path(ROOT / relative)
        if actual_hash != record["sha256"]:
            raise PolicyError(
                f"bundled asset hash changed: {relative}; "
                f"expected={record['sha256']}, actual={actual_hash}"
            )
        notice = record.get("notice")
        if notice and notice not in records:
            raise PolicyError(f"untracked notice reference for {relative}: {notice}")
        attribution_file = record.get("attribution_file")
        attribution_pattern = record.get("attribution_pattern")
        if attribution_file and attribution_pattern:
            attribution_text = (ROOT / attribution_file).read_text(encoding="utf-8")
            if attribution_pattern not in attribution_text:
                raise PolicyError(
                    f"required user-visible attribution is missing: {attribution_file}"
                )
        components.append(
            Component(
                kind="bundled-asset",
                name=record["component"],
                version=record["version"],
                license_expression=record["license"],
                status=record["status"],
                source=record["source"],
                group=relative,
                obligation=record.get("obligation", ""),
                sha256=actual_hash,
            )
        )
    return components


def _validate_first_party_assets(policy: dict[str, Any], tracked: list[str]) -> None:
    excluded = tuple(policy["excluded_path_prefixes"])
    actual = {
        item
        for item in tracked
        if Path(item).suffix.lower() in ASSET_SUFFIXES
        and not item.startswith(excluded)
        and not item.startswith("apps/web/dist/vendor/")
    }
    records = policy["first_party_assets"]
    if actual != set(records):
        missing = sorted(actual - set(records))
        stale = sorted(set(records) - actual)
        raise PolicyError(f"first-party asset policy mismatch; missing={missing}, stale={stale}")
    for relative, record in records.items():
        actual_hash = _sha256_path(ROOT / relative)
        if actual_hash != record["sha256"]:
            raise PolicyError(f"first-party asset hash changed: {relative}")


def _validate_statuses(components: list[Component]) -> None:
    allowed = {"compatible", "conditional", "excluded"}
    for component in components:
        if component.status not in allowed:
            raise PolicyError(
                f"component {component.kind}:{component.name} has blocking status "
                f"{component.status!r}"
            )
        if component.status == "conditional" and not component.obligation:
            raise PolicyError(f"conditional component lacks obligation: {component.name}")
        if not component.license_expression or component.license_expression == "NOASSERTION":
            raise PolicyError(f"component lacks a reviewed license: {component.name}")


def _validate_project(policy: dict[str, Any], pyproject: dict[str, Any]) -> str:
    project = policy["project"]
    version = str(pyproject["project"]["version"])
    if not SEMVER_RE.fullmatch(version):
        raise PolicyError(f"project version is not SemVer: {version}")
    if version == "0.0.0":
        raise PolicyError("technical project version 0.0.0 is forbidden")
    if pyproject["project"].get("license") != project["license"]:
        raise PolicyError("pyproject project license differs from OSS policy")
    if _sha256_path(ROOT / "LICENSE") != project["license_sha256"]:
        raise PolicyError("LICENSE differs from the reviewed official Apache-2.0 text")

    lock = _load_toml(ROOT / "uv.lock")
    root_versions = [
        str(package["version"])
        for package in lock["package"]
        if package["name"] == pyproject["project"]["name"]
        and package.get("source", {}).get("editable") == "."
    ]
    if root_versions != [version]:
        raise PolicyError(f"uv.lock root version differs from pyproject: {root_versions}")
    return version


def _spdx_id(component: Component, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{component.kind}-{component.name}").strip("-")
    return f"SPDXRef-{slug}-{index}"


def _sbom_payload(
    policy: dict[str, Any], version: str, components: list[Component]
) -> dict[str, Any]:
    included = [component for component in components if component.status != "excluded"]
    fingerprint_payload = [component.__dict__ for component in included]
    fingerprint = _sha256_bytes(_json_bytes(fingerprint_payload))[:20]
    root_id = "SPDXRef-Package-Roehub"
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": root_id,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": policy["project"]["license"],
            "licenseDeclared": policy["project"]["license"],
            "name": "roehub",
            "supplier": "Organization: Roehub contributors",
            "versionInfo": version,
        }
    ]
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_id,
        }
    ]
    for index, component in enumerate(included, start=1):
        package_id = _spdx_id(component, index)
        package: dict[str, Any] = {
            "SPDXID": package_id,
            "comment": (
                f"kind={component.kind}; group={component.group}; "
                f"review_status={component.status}; obligation={component.obligation or 'none'}"
            ),
            "downloadLocation": component.source,
            "filesAnalyzed": False,
            "licenseConcluded": component.license_expression,
            "licenseDeclared": component.license_expression,
            "name": component.name,
            "versionInfo": component.version,
        }
        if component.sha256:
            package["checksums"] = [
                {"algorithm": "SHA256", "checksumValue": component.sha256}
            ]
        packages.append(package)
        relationships.append(
            {
                "spdxElementId": root_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package_id,
            }
        )
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": policy["project"]["document_created"],
            "creators": ["Tool: tools/release/oss_metadata.py"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": (
            "https://github.com/Dejetins/roehub.com/spdx/"
            f"roehub-{version}-preliminary-{fingerprint}"
        ),
        "name": f"roehub-{version}-preliminary",
        "packages": packages,
        "relationships": relationships,
        "spdxVersion": "SPDX-2.3",
    }


def _notice_table(title: str, components: list[Component]) -> list[str]:
    lines = [f"## {title}", "", "| Компонент | Версия | Лицензия | Статус |", "|---|---:|---|---|"]
    for component in components:
        name = component.name.replace("|", "\\|")
        license_expression = component.license_expression.replace("|", "\\|")
        lines.append(
            f"| `{name}` | `{component.version}` | `{license_expression}` | "
            f"`{component.status}` |"
        )
        if component.obligation:
            lines.append(f"| ↳ обязательство |  |  | {component.obligation} |")
    lines.append("")
    return lines


def _notices_bytes(policy: dict[str, Any], components: list[Component]) -> bytes:
    third_party_components = [
        item for item in components if not item.name.startswith("roehub/runtime")
    ]
    groups = {
        "Прямые зависимости Python": [
            item for item in third_party_components if item.kind == "python"
        ],
        "Контейнерные образы": [
            item for item in third_party_components if item.kind == "container"
        ],
        "Образы комплекта выпуска": [
            item for item in third_party_components if item.kind == "release-container"
        ],
        "Встроенные Web-ресурсы": [
            item for item in third_party_components if item.kind == "bundled-asset"
        ],
    }
    lines = [
        "# Реестр сторонних компонентов Roehub",
        "",
        "Файл сгенерирован `tools/release/oss_metadata.py`; ручные изменения будут отклонены.",
        "Статус `conditional` означает обязательства, которые должны быть выполнены для",
        "конкретного комплекта выпуска. Статус `excluded` означает, что компонент не входит",
        "в исходный или бинарный комплект Roehub.",
        "Собственные образы `roehub/runtime*` не являются сторонними компонентами и намеренно",
        "не включаются в этот файл: это исключает циклическую зависимость image digest от notice.",
        "",
    ]
    for title, values in groups.items():
        lines.extend(_notice_table(title, values))
    lines.extend(["## Известные риски транзитивных лицензий", ""])
    for risk in policy["known_transitive_risks"]:
        lines.append(f"- {risk}")
    lines.append("")
    return "\n".join(lines).encode()


def _release_metadata_bytes(
    policy: dict[str, Any], version: str, sbom_bytes: bytes, notices_bytes: bytes
) -> bytes:
    payload = {
        "artifacts": {
            "preliminary_sbom": {
                "path": str(SBOM_PATH.relative_to(ROOT)),
                "sha256": _sha256_bytes(sbom_bytes),
            },
            "third_party_notices": {
                "path": str(NOTICES_PATH.relative_to(ROOT)),
                "sha256": _sha256_bytes(notices_bytes),
            },
        },
        "compatibility": {
            "manifest_schema": "io.roehub.release/v1alpha1",
            "manifest_schema_rule": (
                "Unknown optional fields are ignored; removing or changing a required field "
                "requires a manifest schema version change."
            ),
            "pre_1_0_breaking_change": "minor",
            "stable_breaking_change": "major",
            "versioning": "SemVer 2.0.0",
        },
        "known_transitive_license_risks": policy["known_transitive_risks"],
        "images": {
            name: {
                "platforms": sorted(record["platforms"]),
                "reference": record["reference"],
            }
            for name, record in sorted(policy["release_images"].items())
        },
        "license": policy["project"]["license"],
        "schema": "io.roehub.release/v1alpha1",
        "supported_architectures": sorted(policy["release_supported_architectures"]),
        "version": version,
        "version_source": policy["project"]["version_source"],
    }
    return _json_bytes(payload)


def _expected_outputs() -> dict[Path, bytes]:
    policy = _load_json(POLICY_PATH)
    pyproject = _load_toml(ROOT / "pyproject.toml")
    tracked = _tracked_files()
    version = _validate_project(policy, pyproject)
    _validate_first_party_assets(policy, tracked)
    components = [
        *_python_components(policy, pyproject),
        *_container_components(policy, tracked),
        *_release_image_components(policy),
        *_bundled_components(policy, tracked),
    ]
    _validate_statuses(components)
    sbom_bytes = _json_bytes(_sbom_payload(policy, version, components))
    notices_bytes = _notices_bytes(policy, components)
    metadata_bytes = _release_metadata_bytes(policy, version, sbom_bytes, notices_bytes)
    return {
        SBOM_PATH: sbom_bytes,
        NOTICES_PATH: notices_bytes,
        METADATA_PATH: metadata_bytes,
    }


def _write_outputs(outputs: dict[Path, bytes]) -> None:
    for path, content in outputs.items():
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


def _check_outputs(outputs: dict[Path, bytes]) -> None:
    stale = [
        str(path.relative_to(ROOT))
        for path, content in outputs.items()
        if not path.exists() or path.read_bytes() != content
    ]
    if stale:
        raise PolicyError(f"generated release metadata is missing or stale: {stale}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write deterministic artifacts")
    mode.add_argument("--check", action="store_true", help="verify committed artifacts")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        outputs = _expected_outputs()
        if args.write:
            _write_outputs(outputs)
        else:
            _check_outputs(outputs)
    except (
        KeyError,
        OSError,
        PolicyError,
        subprocess.CalledProcessError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"OSS metadata validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "OSS metadata validation passed: "
        f"mode={'write' if args.write else 'check'}, artifacts={len(outputs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
