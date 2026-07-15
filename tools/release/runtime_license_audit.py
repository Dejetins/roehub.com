#!/usr/bin/env python3
"""Resolve raw runtime SPDX NOASSERTION records without rewriting scanner output."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "tools" / "release" / "runtime-license-policy.json"
SCHEMA_ID = "io.roehub.runtime-license-audit/v1alpha1"
MAX_SBOM_BYTES = 256 * 1024 * 1024
LICENSE_REF_RE = re.compile(r"LicenseRef-[A-Za-z0-9.-]+")
SOURCE_INFO_PATH_PREFIX = "acquired package info from the following paths: "


class RuntimeLicenseAuditError(RuntimeError):
    """Raised when a raw runtime SBOM record has no verified resolution."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    metadata = path.lstat()
    if not path.is_file() or path.is_symlink() or metadata.st_size > max_bytes:
        raise RuntimeLicenseAuditError(f"invalid JSON input: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeLicenseAuditError(f"JSON root must be an object: {path}")
    return value


def _safe_bundle_path(bundle_root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise RuntimeLicenseAuditError(f"unsafe bundle path: {relative}")
    path = bundle_root.joinpath(*parsed.parts)
    if path.is_symlink():
        raise RuntimeLicenseAuditError(f"bundle audit path must not be a symlink: {relative}")
    return path


def _evidence_record(
    record: Mapping[str, Any],
    *,
    platform: str,
) -> dict[str, str]:
    path = str(record.get("evidence_path", ""))
    digest = str(record.get("evidence_sha256", ""))
    platform_digests = record.get("evidence_sha256_by_platform")
    if platform_digests is not None:
        if (
            not isinstance(platform_digests, dict)
            or set(platform_digests) != {"linux/amd64", "linux/arm64"}
            or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in platform_digests.values()
            )
        ):
            raise RuntimeLicenseAuditError(
                "runtime license platform evidence digests are invalid"
            )
        digest = str(platform_digests.get(platform, ""))
    source = str(record.get("source", ""))
    license_expression = str(record.get("license", ""))
    parsed = PurePosixPath(path)
    if (
        not parsed.is_absolute()
        or ".." in parsed.parts
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not source.startswith("https://")
        or not license_expression
        or "NOASSERTION" in license_expression
    ):
        raise RuntimeLicenseAuditError("runtime license policy evidence is invalid")
    return {
        "license": license_expression,
        "path": path,
        "sha256": digest,
        "source": source,
    }


def _false_positive_index(policy: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    rules = policy.get("false_positive_records")
    if not isinstance(rules, list):
        raise RuntimeLicenseAuditError("runtime license false-positive policy is invalid")
    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            raise RuntimeLicenseAuditError("runtime license false-positive rule is invalid")
        for platform in ("linux/amd64", "linux/arm64"):
            _evidence_record(raw_rule, platform=platform)
        rule_id = str(raw_rule.get("id", ""))
        source_info_prefix = str(
            raw_rule.get("source_info_prefix", SOURCE_INFO_PATH_PREFIX)
        )
        records = raw_rule.get("records")
        if not rule_id or not source_info_prefix or not isinstance(records, list) or not records:
            raise RuntimeLicenseAuditError("runtime license false-positive rule is incomplete")
        for raw_record in records:
            if (
                not isinstance(raw_record, list)
                or len(raw_record) != 3
                or not all(isinstance(value, str) and value for value in raw_record)
            ):
                raise RuntimeLicenseAuditError(
                    f"runtime license false-positive record is invalid: {rule_id}"
                )
            name, version, path = raw_record
            key = (name, version, source_info_prefix + path)
            if key in result:
                raise RuntimeLicenseAuditError(
                    f"duplicate runtime license false-positive record: {key}"
                )
            result[key] = raw_rule
    return result


def _explicit_index(policy: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    records = policy.get("explicit_components")
    if not isinstance(records, list):
        raise RuntimeLicenseAuditError("runtime explicit license policy is invalid")
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise RuntimeLicenseAuditError("runtime explicit license record is invalid")
        for platform in ("linux/amd64", "linux/arm64"):
            _evidence_record(raw_record, platform=platform)
        name = str(raw_record.get("name", ""))
        version = str(raw_record.get("version", ""))
        prefix = str(raw_record.get("source_info_prefix", ""))
        if not name or not version or not prefix:
            raise RuntimeLicenseAuditError("runtime explicit license record is incomplete")
        key = (name, version)
        if key in result:
            raise RuntimeLicenseAuditError(f"duplicate runtime explicit license record: {key}")
        result[key] = raw_record
    return result


def _purl(package: Mapping[str, Any]) -> str:
    references = package.get("externalRefs", [])
    if not isinstance(references, list):
        return ""
    values = sorted(
        str(reference.get("referenceLocator", ""))
        for reference in references
        if isinstance(reference, dict)
        and reference.get("referenceType") == "purl"
        and str(reference.get("referenceLocator", "")).startswith("pkg:")
    )
    return values[0] if values else ""


def _scanner_resolution(
    package: Mapping[str, Any], extracted: Mapping[str, str]
) -> dict[str, Any] | None:
    concluded = str(package.get("licenseConcluded", ""))
    if not concluded or "NOASSERTION" in concluded or concluded == "NONE":
        return None
    refs = sorted(set(LICENSE_REF_RE.findall(concluded)))
    for reference in refs:
        text = extracted.get(reference, "").strip()
        if not text or text in {"NONE", "NOASSERTION"}:
            return None
    source = _purl(package) or str(package.get("sourceInfo", ""))
    if not source:
        raise RuntimeLicenseAuditError(
            f"SPDX concluded license lacks source identity: {package.get('name')}"
        )
    return {
        "classification": "scanner-concluded",
        "evidence": {
            "kind": "spdx-license-concluded",
            "license_refs": refs,
        },
        "resolved_license": concluded,
        "source": source,
    }


def _policy_resolution(
    *,
    package: Mapping[str, Any],
    image_name: str,
    platform: str,
    policy: Mapping[str, Any],
    explicit: Mapping[tuple[str, str], dict[str, Any]],
    false_positives: Mapping[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    spdx_id = str(package.get("SPDXID", ""))
    if spdx_id.startswith("SPDXRef-DocumentRoot-Image-"):
        evidence = _evidence_record(policy["first_party_image"], platform=platform)
        return {
            "classification": "first-party-image",
            "evidence": evidence,
            "resolved_license": evidence["license"],
            "source": evidence["source"],
        }

    name = str(package.get("name", ""))
    version = str(package.get("versionInfo", ""))
    source_info = str(package.get("sourceInfo", ""))
    false_positive = false_positives.get((name, version, source_info))
    if false_positive is not None:
        evidence = _evidence_record(false_positive, platform=platform)
        evidence["rule"] = str(false_positive["id"])
        return {
            "classification": "embedded-non-component",
            "evidence": evidence,
            "resolved_license": evidence["license"],
            "source": evidence["source"],
        }
    component = explicit.get((name, version))
    if component is None:
        return None
    if not source_info.startswith(str(component["source_info_prefix"])):
        raise RuntimeLicenseAuditError(
            f"runtime component source identity changed: {image_name}:{name}@{version}"
        )
    evidence = _evidence_record(component, platform=platform)
    return {
        "classification": "policy-license-file",
        "evidence": evidence,
        "resolved_license": evidence["license"],
        "source": evidence["source"],
    }


def build_runtime_license_audit(
    *,
    bundle_root: Path,
    image_records: Mapping[str, Mapping[str, Any]],
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Build a deterministic resolution overlay for first-party runtime SBOMs."""

    bundle_root = bundle_root.resolve()
    policy = _load_json(policy_path, max_bytes=4 * 1024 * 1024)
    if policy.get("schema") != "io.roehub.runtime-license-policy/v1alpha1":
        raise RuntimeLicenseAuditError("runtime license policy schema is unsupported")
    audited_images = policy.get("audited_images")
    if not isinstance(audited_images, list) or not audited_images:
        raise RuntimeLicenseAuditError("runtime license audited image list is invalid")
    if not all(isinstance(name, str) and name in image_records for name in audited_images):
        raise RuntimeLicenseAuditError("runtime license audit image set is incomplete")
    explicit = _explicit_index(policy)
    false_positives = _false_positive_index(policy)

    classifications: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for image_name in sorted(audited_images):
        image_record = image_records[image_name]
        index_digest = str(image_record.get("index_digest", ""))
        sboms = image_record.get("sboms")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", index_digest) or not isinstance(
            sboms, dict
        ):
            raise RuntimeLicenseAuditError(
                f"runtime license image record is invalid: {image_name}"
            )
        for platform, relative_value in sorted(sboms.items()):
            relative = str(relative_value)
            sbom_path = _safe_bundle_path(bundle_root, relative)
            sbom = _load_json(sbom_path, max_bytes=MAX_SBOM_BYTES)
            if sbom.get("spdxVersion") != "SPDX-2.3":
                raise RuntimeLicenseAuditError(f"runtime SBOM version is invalid: {relative}")
            packages = sbom.get("packages")
            if not isinstance(packages, list) or not packages:
                raise RuntimeLicenseAuditError(f"runtime SBOM packages are missing: {relative}")
            extracted_values = sbom.get("hasExtractedLicensingInfos", [])
            if not isinstance(extracted_values, list):
                raise RuntimeLicenseAuditError(
                    f"runtime SBOM extracted license list is invalid: {relative}"
                )
            extracted = {
                str(value.get("licenseId", "")): str(value.get("extractedText", ""))
                for value in extracted_values
                if isinstance(value, dict)
            }
            document_records = 0
            root_records = 0
            for package in packages:
                if not isinstance(package, dict):
                    raise RuntimeLicenseAuditError(
                        f"runtime SBOM package is invalid: {relative}"
                    )
                if str(package.get("SPDXID", "")).startswith(
                    "SPDXRef-DocumentRoot-Image-"
                ):
                    root_records += 1
                declared = package.get("licenseDeclared")
                if not isinstance(declared, str) or not declared:
                    raise RuntimeLicenseAuditError(
                        f"runtime SBOM package lacks licenseDeclared: {relative}"
                    )
                if declared != "NOASSERTION":
                    continue
                document_records += 1
                resolution = _scanner_resolution(package, extracted)
                if resolution is None:
                    resolution = _policy_resolution(
                        package=package,
                        image_name=image_name,
                        platform=str(platform),
                        policy=policy,
                        explicit=explicit,
                        false_positives=false_positives,
                    )
                if resolution is None:
                    identity = (
                        f"{image_name}:{platform}:{package.get('name')}@"
                        f"{package.get('versionInfo')}"
                    )
                    unresolved.append(identity)
                    continue
                classification = str(resolution["classification"])
                classifications[classification] += 1
                records.append(
                    {
                        "classification": classification,
                        "image": image_name,
                        "license_concluded": str(package.get("licenseConcluded", "")),
                        "license_declared": declared,
                        "name": str(package.get("name", "")),
                        "platform": str(platform),
                        "resolution": resolution,
                        "source_info": str(package.get("sourceInfo", "")),
                        "spdx_id": str(package.get("SPDXID", "")),
                        "version": str(package.get("versionInfo", "")),
                    }
                )
            if root_records != 1:
                raise RuntimeLicenseAuditError(
                    f"runtime SBOM must contain one document root: {relative}"
                )
            documents.append(
                {
                    "image": image_name,
                    "index_digest": index_digest,
                    "package_count": len(packages),
                    "platform": str(platform),
                    "raw_noassertion_count": document_records,
                    "sbom": relative,
                    "sbom_sha256": _sha256_path(sbom_path),
                }
            )
    if unresolved:
        raise RuntimeLicenseAuditError(
            f"unresolved runtime SPDX NOASSERTION records: {sorted(unresolved)[:20]}"
        )
    return {
        "classification_counts": dict(sorted(classifications.items())),
        "documents": documents,
        "policy": "tools/release/runtime-license-policy.json",
        "policy_sha256": _sha256_path(policy_path),
        "raw_noassertion_count": sum(
            document["raw_noassertion_count"] for document in documents
        ),
        "records": sorted(
            records,
            key=lambda record: (
                record["image"],
                record["platform"],
                record["name"],
                record["version"],
                record["source_info"],
            ),
        ),
        "schema": SCHEMA_ID,
        "status": "passed",
        "unresolved_count": 0,
    }


def write_runtime_license_audit(
    *,
    output: Path,
    bundle_root: Path,
    image_records: Mapping[str, Mapping[str, Any]],
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    payload = build_runtime_license_audit(
        bundle_root=bundle_root,
        image_records=image_records,
        policy_path=policy_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(payload))
    return payload


def verify_runtime_license_audit(
    *,
    audit_path: Path,
    bundle_root: Path,
    image_records: Mapping[str, Mapping[str, Any]],
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    actual = _load_json(audit_path, max_bytes=MAX_SBOM_BYTES)
    expected = build_runtime_license_audit(
        bundle_root=bundle_root,
        image_records=image_records,
        policy_path=policy_path,
    )
    if actual != expected:
        raise RuntimeLicenseAuditError("runtime license audit does not match raw SPDX SBOMs")
    return actual
