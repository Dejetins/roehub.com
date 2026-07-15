from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, cast

import jsonschema
import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from trading.contexts.extensions.domain import (
    PLUGIN_API_VERSION,
    PLUGIN_MANIFEST_API_VERSION,
    PLUGIN_RPC_VERSION,
    PluginManifest,
    ValidatedPluginBundle,
)

_MANIFEST_NAME = "roehub.plugin.yaml"
_SIGNATURE_CONTEXT = b"roehub-plugin-package-v1alpha1\0"
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PluginBundleValidationError(ValueError):
    """Stable offline validation failure without leaking bundle contents."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def canonical_package_digest(payload: Mapping[str, Any]) -> str:
    """Hash a manifest after removing only the detached signature value."""

    canonical = copy.deepcopy(dict(payload))
    raw_signature = canonical.get("signature")
    if isinstance(raw_signature, dict):
        raw_signature.pop("value", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sign_package_digest(*, private_key: Ed25519PrivateKey, package_digest: str) -> str:
    """Create a base64 Ed25519 signature for one canonical package digest."""

    signature = private_key.sign(_SIGNATURE_CONTEXT + package_digest.encode("ascii"))
    return base64.b64encode(signature).decode("ascii")


def load_publisher_key_file(path: Path) -> dict[str, Ed25519PublicKey]:
    """Load public Ed25519 publisher keys from the versioned non-secret file contract."""

    payload = _load_json_object(path, label="plugin publisher key file")
    if payload.get("contract") != "PluginPublisherKeys/v1alpha1":
        raise PluginBundleValidationError(
            code="plugin.publisher_keys_invalid",
            message="Plugin publisher key file contract is invalid",
        )
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, dict):
        raise PluginBundleValidationError(
            code="plugin.publisher_keys_invalid",
            message="Plugin publisher key file has no keys",
        )
    result: dict[str, Ed25519PublicKey] = {}
    for key_id, encoded in raw_keys.items():
        if not isinstance(key_id, str) or not isinstance(encoded, str):
            raise PluginBundleValidationError(
                code="plugin.publisher_keys_invalid",
                message="Plugin publisher key entry is invalid",
            )
        try:
            result[key_id] = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(encoded, validate=True)
            )
        except ValueError as error:
            raise PluginBundleValidationError(
                code="plugin.publisher_keys_invalid",
                message="Plugin publisher key entry is invalid",
            ) from error
    return result


class PluginBundleValidator:
    """Validate a signed immutable plugin bundle without importing plugin code."""

    def __init__(
        self,
        *,
        schema_path: Path,
        trusted_publisher_keys: Mapping[str, Ed25519PublicKey],
        roehub_version: str,
        supported_architectures: frozenset[str],
        allow_unsigned_development: bool = False,
        trading_mode: str = "paper",
    ) -> None:
        self._schema = _load_json_object(schema_path, label="plugin manifest schema")
        jsonschema.Draft202012Validator.check_schema(self._schema)
        self._trusted_publisher_keys = dict(trusted_publisher_keys)
        self._roehub_version = _parse_semver(roehub_version, field="Roehub version")
        self._supported_architectures = supported_architectures
        self._allow_unsigned_development = allow_unsigned_development
        self._trading_mode = trading_mode
        if not supported_architectures:
            raise ValueError("supported_architectures must not be empty")
        if trading_mode == "mainnet" and allow_unsigned_development:
            raise ValueError("unsigned development mode is unavailable to mainnet")

    def validate(self, bundle_path: Path) -> ValidatedPluginBundle:
        bundle_root = bundle_path.resolve()
        if not bundle_root.is_dir():
            raise PluginBundleValidationError(
                code="plugin.bundle_not_found", message="Plugin bundle directory is not found"
            )
        manifest_path = bundle_root / _MANIFEST_NAME
        payload = _load_yaml_object(manifest_path)
        errors = sorted(
            jsonschema.Draft202012Validator(self._schema).iter_errors(payload),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            path = ".".join(str(part) for part in errors[0].absolute_path) or "$"
            raise PluginBundleValidationError(
                code="plugin.manifest_invalid",
                message=f"Plugin manifest is invalid at {path}",
            )
        if payload["apiVersion"] != PLUGIN_MANIFEST_API_VERSION:
            raise PluginBundleValidationError(
                code="plugin.manifest_version_unsupported",
                message="Plugin manifest API version is unsupported",
            )
        spec = cast(dict[str, Any], payload["spec"])
        compatibility = cast(dict[str, Any], spec["compatibility"])
        if spec["pluginApi"] != PLUGIN_API_VERSION or spec["rpc"]["version"] != PLUGIN_RPC_VERSION:
            raise PluginBundleValidationError(
                code="plugin.protocol_version_unsupported",
                message="Plugin API or RPC version is unsupported",
            )
        _validate_compatibility(
            actual=self._roehub_version,
            minimum=compatibility["roehubMin"],
            maximum_exclusive=compatibility["roehubMaxExclusive"],
        )
        architectures = tuple(cast(list[str], spec["image"]["architectures"]))
        if not set(architectures).issubset(self._supported_architectures):
            raise PluginBundleValidationError(
                code="plugin.architecture_unsupported",
                message="Plugin requests an unsupported architecture",
            )
        if not _SHA256_RE.fullmatch(spec["image"]["digest"]):
            raise PluginBundleValidationError(
                code="plugin.image_digest_invalid", message="Plugin image digest is invalid"
            )

        artifact_digests: dict[str, str] = {}
        artifact_paths: dict[str, Path] = {}
        for label in ("configSchema", "license", "sbom"):
            artifact = cast(dict[str, Any], spec[label])
            artifact_path = _safe_bundle_file(bundle_root=bundle_root, relative=artifact["path"])
            actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise PluginBundleValidationError(
                    code="plugin.artifact_digest_mismatch",
                    message=f"Plugin {label} artifact digest does not match",
                )
            artifact_digests[label] = actual
            artifact_paths[label] = artifact_path
        config_schema_path = _safe_bundle_file(
            bundle_root=bundle_root,
            relative=spec["configSchema"]["path"],
        )
        config_schema = _load_json_object(config_schema_path, label="plugin config schema")
        try:
            jsonschema.Draft202012Validator.check_schema(config_schema)
        except jsonschema.SchemaError as error:
            raise PluginBundleValidationError(
                code="plugin.config_schema_invalid",
                message="Plugin configuration schema is invalid",
            ) from error
        spdx = spec["license"]["spdx"]
        if not _SPDX_RE.fullmatch(spdx):
            raise PluginBundleValidationError(
                code="plugin.license_invalid", message="Plugin SPDX license is invalid"
            )
        try:
            license_size = artifact_paths["license"].stat().st_size
        except OSError as error:
            raise PluginBundleValidationError(
                code="plugin.license_invalid", message="Plugin license cannot be inspected"
            ) from error
        if not 1 <= license_size <= 1_000_000:
            raise PluginBundleValidationError(
                code="plugin.license_invalid", message="Plugin license artifact is invalid"
            )
        sbom = _load_json_object(artifact_paths["sbom"], label="plugin SPDX SBOM")
        if (
            sbom.get("spdxVersion") != "SPDX-2.3"
            or sbom.get("SPDXID") != "SPDXRef-DOCUMENT"
            or sbom.get("dataLicense") != "CC0-1.0"
        ):
            raise PluginBundleValidationError(
                code="plugin.sbom_invalid", message="Plugin SPDX SBOM is invalid"
            )

        package_digest = canonical_package_digest(payload)
        signature = cast(dict[str, Any] | None, payload.get("signature"))
        signed, publisher_key_id, publisher_public_key = self._verify_signature(
            payload=payload,
            package_digest=package_digest,
            signature=signature,
        )
        metadata = cast(dict[str, Any], payload["metadata"])
        permissions = tuple(
            sorted(permission["capability"] for permission in spec["permissions"])
        )
        return ValidatedPluginBundle(
            bundle_path=str(bundle_root),
            manifest=PluginManifest(
                plugin_id=metadata["id"],
                version=metadata["version"],
                publisher=metadata["publisher"],
                plugin_type=spec["type"],
                plugin_api_version=spec["pluginApi"],
                rpc_version=spec["rpc"]["version"],
                image_reference=spec["image"]["reference"],
                image_digest=spec["image"]["digest"],
                architectures=architectures,
                permissions=permissions,
                config_schema=config_schema,
                license_spdx=spdx,
                package_digest=package_digest,
                publisher_key_id=publisher_key_id,
                signed=signed,
                raw=payload,
            ),
            artifact_digests=artifact_digests,
            publisher_public_key_b64=(
                base64.b64encode(
                    publisher_public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
                ).decode("ascii")
                if publisher_public_key is not None
                else None
            ),
            publisher_key_fingerprint_sha256=(
                hashlib.sha256(
                    publisher_public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
                ).hexdigest()
                if publisher_public_key is not None
                else None
            ),
        )

    def _verify_signature(
        self,
        *,
        payload: Mapping[str, Any],
        package_digest: str,
        signature: Mapping[str, Any] | None,
    ) -> tuple[bool, str | None, Ed25519PublicKey | None]:
        metadata = cast(Mapping[str, Any], payload["metadata"])
        development_mode = metadata.get("developmentMode") is True
        if signature is None:
            if (
                not development_mode
                or not self._allow_unsigned_development
                or self._trading_mode == "mainnet"
            ):
                raise PluginBundleValidationError(
                    code="plugin.signature_required",
                    message="Plugin signature is required",
                )
            return False, None, None
        key_id = cast(str, signature["keyId"])
        public_key = self._trusted_publisher_keys.get(key_id)
        if public_key is None:
            raise PluginBundleValidationError(
                code="plugin.publisher_untrusted",
                message="Plugin publisher key is not trusted",
            )
        try:
            decoded = base64.b64decode(signature["value"], validate=True)
            public_key.verify(
                decoded,
                _SIGNATURE_CONTEXT + package_digest.encode("ascii"),
            )
        except (InvalidSignature, ValueError) as error:
            raise PluginBundleValidationError(
                code="plugin.signature_invalid", message="Plugin signature is invalid"
            ) from error
        return True, key_id, public_key


def _load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PluginBundleValidationError(
            code="plugin.manifest_unreadable", message="Plugin manifest cannot be read"
        ) from error
    if not isinstance(payload, dict):
        raise PluginBundleValidationError(
            code="plugin.manifest_invalid", message="Plugin manifest root must be an object"
        )
    return cast(dict[str, Any], payload)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PluginBundleValidationError(
            code="plugin.artifact_unreadable", message=f"{label} cannot be read"
        ) from error
    if not isinstance(payload, dict):
        raise PluginBundleValidationError(
            code="plugin.artifact_invalid", message=f"{label} root must be an object"
        )
    return cast(dict[str, Any], payload)


def _safe_bundle_file(*, bundle_root: Path, relative: str) -> Path:
    candidate = (bundle_root / relative).resolve()
    if candidate == bundle_root or bundle_root not in candidate.parents or not candidate.is_file():
        raise PluginBundleValidationError(
            code="plugin.artifact_path_invalid",
            message="Plugin artifact path is missing or escapes the bundle",
        )
    return candidate


def _parse_semver(value: str, *, field: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{field} must be strict SemVer without prerelease metadata")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _validate_compatibility(
    *, actual: tuple[int, int, int], minimum: str, maximum_exclusive: str
) -> None:
    try:
        lower = _parse_semver(minimum, field="roehubMin")
        upper = _parse_semver(maximum_exclusive, field="roehubMaxExclusive")
    except ValueError as error:
        raise PluginBundleValidationError(
            code="plugin.compatibility_invalid",
            message="Plugin Roehub compatibility range is invalid",
        ) from error
    if not lower <= actual < upper:
        raise PluginBundleValidationError(
            code="plugin.compatibility_unsupported",
            message="Plugin is incompatible with this Roehub version",
        )
